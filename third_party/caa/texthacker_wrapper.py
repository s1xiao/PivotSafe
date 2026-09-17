# TextHackerWrapper.py - TextHacker攻击包装函数
# 参照HQA格式重构，统一接口和组件加载
# 修改版：攻击失败返回None
# 修改版2：支持传入query_budget参数控制攻击预算

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Any, Dict
from collections import defaultdict
import random
import time
import os

from .tool import get_component, calc_sim_interface, cleanup_memory, initialize_global_components, semantic_sim
from . import criteria_1


class Args:
    """TextHacker攻击参数配置类"""

    def __init__(self):
        # ========== 攻击基本参数 ==========
        self.query_budget = 1000  # 查询预算
        self.neighbor_delta = 5  # 邻域大小
        self.population_size = 4  # 种群大小
        self.local_search_num = 8  # 最大局部搜索次数
        self.synonym_num = 4  # 同义词数量

        # ========== 约束参数 ==========
        self.semantic_threshold = 0.9  # 语义相似度阈值
        self.perturb_ratio = 0.1  # 扰动比例阈值

        # ========== 遗传算法参数 ==========
        self.base_reward = 0.5  # 基础奖励值

        # ========== 其他参数 ==========
        self.batch_size = 32  # 批处理大小
        self.sim_score_window = 15  # 相似度计算窗口大小


def sigmoid(x):
    """Sigmoid激活函数"""
    y = 1 / (1 + np.exp(-x))
    return y


def get_pert_rate(original_text, modified_text):
    """计算扰动率"""
    # 处理原始文本
    if isinstance(original_text, (list, np.ndarray)):
        orig_words = list(original_text) if isinstance(original_text, np.ndarray) else original_text
    else:
        orig_words = original_text.split()

    # 处理修改后的文本
    if isinstance(modified_text, (list, np.ndarray)):
        mod_words = list(modified_text) if isinstance(modified_text, np.ndarray) else modified_text
    else:
        mod_words = modified_text.split()

    if len(orig_words) == 0:
        return 0.0

    min_len = min(len(orig_words), len(mod_words))
    changed_words = sum(1 for i in range(min_len) if orig_words[i] != mod_words[i])
    length_diff = abs(len(orig_words) - len(mod_words))
    total_changes = changed_words + length_diff

    return total_changes / len(orig_words)


def count_perturbation_rate(x_orig, x_new):
    """计算扰动率和扰动数量"""
    perturbation_num = 0
    l = len(x_orig)
    for i in range(l):
        if x_orig[i] != x_new[i]:
            perturbation_num += 1
    perturbation_rate = perturbation_num / l
    return perturbation_rate, perturbation_num


def generate_handle_list(x_orig, x_new):
    """生成操作列表，标记哪些位置的词被修改过"""
    handle_list = [1 if x_orig[i] != x_new[i] else 0 for i in range(len(x_orig))]
    return handle_list


class TextHackerWrapper:
    """TextHacker攻击包装类 - 修改版：攻击失败返回None，支持传入query_budget"""

    def __init__(self, predictor, word2idx, idx2word, cos_sim, sim_predictor,
                 stop_words_set,
                 embed_func=None, args=None):
        """
        初始化TextHacker攻击器

        Args:
            predictor: 模型预测器
            word2idx: 词到索引的映射
            idx2word: 索引到词的映射
            cos_sim: 余弦相似度矩阵
            args: 攻击参数配置
        """
        self.predictor = predictor
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.cos_sim = cos_sim

        # 设置参数
        if args is None:
            args = Args()
        self.args = args

        # 同义词字典缓存
        self.synonym_dict = {}

        print("TextHacker攻击器初始化完成")
        print(f"  默认查询预算: {self.args.query_budget}")
        print(f"  种群大小: {self.args.population_size}")
        print(f"  同义词数量: {self.args.synonym_num}")

    def query_model(self, text):
        """查询模型预测"""
        probs = self.predictor([text])[0].data.cpu()
        label = torch.argmax(probs, dim=-1).data.numpy()
        return probs, label

    def candidate_generate(self, word, sim_score):
        """
        生成候选同义词 - 完全匹配原版逻辑
        原版逻辑：
        1. 调用replace_with_synonym_embedding获取所有同义词（已按相似度排序）
        2. 取前synonym_num+1个
        3. 过滤掉相似度 <= sim_score的词
        """
        # 调用replace_with_synonym_embedding获取同义词
        synonym_words = []
        synonym_values = []

        if word in self.word2idx:
            word_idx = self.word2idx[word]
            # cos_sim[word_idx]返回的是排序好的(相似度, 索引)元组
            res = list(zip(*(self.cos_sim[word_idx])))

            # 提取同义词和相似度值
            for ii in res[1]:  # 索引
                synonym_words.append(self.idx2word[ii])
            for ii in res[0]:  # 相似度值
                synonym_values.append(ii)
        else:
            # 词不在词典中，返回原词
            synonym_words.append(word)
            synonym_values.append(1.0)

        # 取前synonym_num+1个（原版逻辑）
        candidate_word_list = np.array(synonym_words[:self.args.synonym_num + 1])
        candidate_value_list = np.array(synonym_values[:self.args.synonym_num + 1])

        # 过滤掉相似度 <= sim_score的词
        index = (candidate_value_list > sim_score)
        candidate_word_list = candidate_word_list[index].tolist()
        candidate_value_list = candidate_value_list[index].tolist()

        return candidate_word_list, candidate_value_list

    def init_weight_table(self, x_orig):
        """初始化权重表和候选集"""
        weight_table = []
        candidate_set = []
        handle_list = [-1] * len(x_orig)

        # 获取词性，只操作形容词、副词、动词和名词
        pos_ls = criteria_1.get_pos(x_orig)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]

        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos:
                    handle_list[i] = 0

        # 构建候选集和初始化权重
        for i in range(len(x_orig)):
            word = x_orig[i]

            # 查找同义词
            if word in self.synonym_dict:
                candidate_list = self.synonym_dict[word]
            else:
                candidate_list, _ = self.candidate_generate(word, 0.5)
                self.synonym_dict[word] = candidate_list

            candidate_set.append(candidate_list)

            # 初始化权重
            l = len(candidate_list)
            w = [0 for _ in range(l)]
            weight_table.append(w)

            # 只有一个同义词的词不可操作
            if len(candidate_list) <= 1:
                handle_list[i] = -1

        return weight_table, candidate_set, handle_list

    def replace_word_with_synonym_in_text(self, text, handle_list, candidate_set):
        """用同义词替换文本中的词"""
        index = [i for i in range(len(handle_list)) if handle_list[i] != -1]

        # 替换每个可操作位置的词
        for i in index:
            text[i] = random.choice(candidate_set[i])

        return text

    def update_weight_table_in_optim(self, weight_table, swap_index, fitness, base_reward):
        """更新权重表"""
        for index, fi, ti in swap_index:
            if fitness == 0:
                # 攻击失败，降低该替换的权重
                weight_table[index][fi] += 2 * base_reward
                weight_table[index][ti] -= base_reward
            else:
                # 攻击成功，增加该替换的权重
                weight_table[index][fi] -= 2 * base_reward
                weight_table[index][ti] += base_reward

        return weight_table

    def compute_fitness(self, x_orig, y_orig, x_new):
        """计算适应度"""
        _, label = self.query_model(list(x_new))

        if label == y_orig:
            # 攻击失败
            return 0
        else:
            # 攻击成功，返回1-扰动率作为适应度
            perturbation_rate, _ = count_perturbation_rate(x_orig, x_new)
            return 1 - perturbation_rate

    def local_search(self, x_orig, x_new, neighbor_delta, weight_table, candidate_set):
        """
        局部搜索
        """
        x_mutate = x_new.copy()

        # 生成操作列表
        handle_list = generate_handle_list(x_orig, x_mutate)
        index = [i for i in range(len(handle_list)) if handle_list[i] == 1]

        if len(index) == 0:
            return None, None

        # 计算采样概率
        prob = [np.sum(weight_table[i]) for i in range(len(handle_list)) if handle_list[i] == 1]
        prob = np.array(prob)
        prob = sigmoid(prob)
        prob = [1 - t + 0.0001 for t in prob]

        if np.sum(prob) == 0:
            prob = [p + 1 for p in prob]

        # 确定要改变的词数
        change_ratio, change_num = count_perturbation_rate(x_orig, x_new)
        prop = [i + 1 for i in range(min(neighbor_delta, change_num) - 1)]

        if not prop:
            prop.append(1)

        index_num = min(np.random.choice(prop), len(index))
        index = random.choices(index, prob, k=index_num)
        index = list(set(index))

        # 替换选中的词
        swap_index = []
        swap_prob = 0.5
        swap_select = np.random.choice([0, 1], p=np.array([1 - swap_prob, swap_prob]))

        for i in index:
            fi = 0
            for m, w in enumerate(candidate_set[i]):
                if w == x_mutate[i]:
                    fi = m
                    break

            if swap_select == 0:
                # 替换回原词
                x_mutate[i] = x_orig[i]
                swap_index.append([i, fi, 0])
            else:
                # 替换为候选词
                mutate_prob = np.array(weight_table[i])
                mutate_prob = sigmoid(mutate_prob) + 0.01
                ti = random.choices(list(range(len(candidate_set[i]))), mutate_prob)[0]
                swap_index.append([i, fi, ti])
                x_mutate[i] = candidate_set[i][ti]

        return x_mutate, swap_index

    def recombination(self, x_orig, x_new_1, x_new_2, weight_table, candidate_set):
        """
        重组操作
        """
        parents = [x_new_1, x_new_2]
        x_recombination = []
        l = len(x_orig)

        for i in range(l):
            c1, c2 = None, None

            # 获取两个父代在该位置的权重
            for m, w in enumerate(candidate_set[i]):
                if w == parents[0][i]:
                    c1 = weight_table[i][m]
                if w == parents[1][i]:
                    c2 = weight_table[i][m]

            # 根据权重选择
            if c1 is not None and c2 is not None:
                prob = [c1, c2]
                prob = np.array(prob)
                prob = sigmoid(prob) + 0.01
                prob = np.log(prob + 1)
                index = random.choices([0, 1], prob)[0]
            else:
                index = random.choice([0, 1])

            x_recombination.append(parents[index][i])

        return x_recombination

    def adversary_initialization(self, x, y, allowed_query_num, candidate_set, handle_list):
        """
        对抗样本初始化阶段

        Args:
            x: 原始文本词列表
            y: 真实标签
            allowed_query_num: 允许的查询次数（使用传入的预算）
            candidate_set: 候选词集合
            handle_list: 可操作位置列表
        """
        log = {}
        x_orig = x

        # 检查原始样本是否分类正确
        _, label = self.query_model(x_orig)

        if label != y:
            log['classification'] = False
            return log, None, None, None

        log['classification'] = True

        # 初始化对抗样本
        text = x_orig.copy()
        init_query_number = 1
        x_adv = x_orig.copy()

        # 随机替换所有词直到找到对抗样本
        while True:
            text = self.replace_word_with_synonym_in_text(text, handle_list, candidate_set)

            _, label = self.query_model(text)
            init_query_number += 1

            if label != y:
                # 找到对抗样本
                x_adv = text.copy()
                break

            # 超出查询预算（使用传入的预算）
            if init_query_number >= allowed_query_num:
                log['status'] = False
                log['init_query_number'] = init_query_number
                return log, text, x_orig, label

        # 记录初始化结果
        perturbation_rate, perturbation_num = count_perturbation_rate(x_orig, x_adv)

        log['init_adversarial_example'] = ' '.join(x_adv)
        log['init_perturbation_num'] = perturbation_num
        log['init_perturbation_rate'] = perturbation_rate
        log['init_query_number'] = init_query_number
        log['status'] = True

        return log, x_adv, x_orig, label

    def perturbation_optimization(self, x_adv, x_orig, y_orig, optim_allowed_query_num,
                                  population_size, neighbor_delta, local_search_num,
                                  top_k, weight_table, candidate_set, base_reward):
        """
        扰动优化阶段 - 使用混合遗传算法最小化扰动

        Args:
            x_adv: 初始对抗样本
            x_orig: 原始文本
            y_orig: 原始标签
            optim_allowed_query_num: 优化阶段允许的查询次数（使用传入的预算）
            population_size: 种群大小
            neighbor_delta: 邻域大小
            local_search_num: 局部搜索次数
            top_k: 选择的top-k个体
            weight_table: 权重表
            candidate_set: 候选词集合
            base_reward: 基础奖励值
        """
        log = {}
        best_text = x_adv.copy()
        fitness = self.compute_fitness(x_orig, y_orig, best_text)
        best_fitness = fitness
        optim_query_number = 1

        # 初始化种群
        population_lists = [x_adv]
        fitness_list = [fitness]
        x_adv_before = best_text.copy()

        # 通过局部搜索构建初始种群
        for m in range(population_size - 1):
            x_adv, swap_index = self.local_search(x_orig, x_adv_before, neighbor_delta,
                                                  weight_table, candidate_set)

            if x_adv is None:
                continue

            fitness = self.compute_fitness(x_orig, y_orig, x_adv)
            optim_query_number += 1

            # 更新权重表
            weight_table = self.update_weight_table_in_optim(weight_table.copy(), swap_index,
                                                             fitness, base_reward)

            population_lists.append(list(x_adv))
            fitness_list.append(fitness)

            if fitness == 0:
                continue

            x_adv_before = x_adv.copy()

            # 更新全局最优
            if fitness > best_fitness:
                best_text = x_adv.copy()
                best_fitness = fitness
                perturbation_rate, perturbation_num = count_perturbation_rate(x_orig, best_text)

                # 如果扰动数为1，直接返回
                if perturbation_num == 1:
                    _, label = self.query_model(best_text)
                    log['final_adversarial_example'] = ' '.join(best_text)
                    log['optim_perturbation_num'] = perturbation_num
                    log['optim_perturbation_rate'] = perturbation_rate
                    log['optim_query_number'] = optim_query_number
                    log['after_attack_label'] = label
                    log['status'] = True
                    return log

        # 混合遗传算法主循环
        while True:
            # 检查是否超出优化阶段的查询预算
            if optim_query_number >= optim_allowed_query_num:
                break

            # ===== 阶段1: 重组（Recombination） =====
            l = len(population_lists)
            index = list(range(l))
            np.random.shuffle(index)

            new_population_lists = []
            new_fitness_list = []

            # 两两配对进行重组，生成子代
            for t in range(len(index) // 2):
                # 重组生成子代
                x_recombination = self.recombination(x_orig, population_lists[index[2 * t]],
                                                     population_lists[index[2 * t + 1]],
                                                     weight_table, candidate_set)

                # 记录子代及其适应度
                fitness = self.compute_fitness(x_orig, y_orig, x_recombination)
                optim_query_number += 1
                new_population_lists.append(list(x_recombination))
                new_fitness_list.append(fitness)

                # 检查是否超出优化阶段的查询预算
                if optim_query_number >= optim_allowed_query_num:
                    break

                # 更新全局最优
                if fitness > best_fitness:
                    best_text = x_recombination.copy()
                    best_fitness = fitness
                    perturbation_rate, perturbation_num = count_perturbation_rate(x_orig, best_text)
                    if perturbation_num == 1:
                        break

            # 合并父代和子代
            population_lists = population_lists + new_population_lists
            fitness_list = fitness_list + new_fitness_list

            # ===== 阶段2: 局部搜索（Local Search） =====
            new_population_lists = []
            new_fitness_list = []
            l = len(population_lists)

            # 对种群中的每个个体进行多步局部搜索
            for j in range(l):
                x_adv_before = population_lists[j].copy()
                # 记录该个体的局部最优
                local_best_x = x_adv_before.copy()
                local_best_fitness = fitness_list[j]

                # 进行多步局部搜索
                for m in range(local_search_num):
                    # 从x_adv_before的邻域中搜索
                    x_adv, swap_index = self.local_search(x_orig, x_adv_before, neighbor_delta,
                                                          weight_table, candidate_set)
                    if x_adv is None:
                        break

                    fitness = self.compute_fitness(x_orig, y_orig, x_adv)
                    optim_query_number += 1

                    # 更新权重表
                    weight_table = self.update_weight_table_in_optim(weight_table.copy(),
                                                                     swap_index, fitness, base_reward)

                    # 检查是否超出优化阶段的查询预算
                    if optim_query_number >= optim_allowed_query_num:
                        break

                    # 如果不是对抗样本，继续在x_adv_before上搜索
                    if fitness == 0:
                        continue

                    x_adv_before = x_adv.copy()

                    # 更新局部最优
                    if fitness > local_best_fitness:
                        local_best_x = x_adv.copy()
                        local_best_fitness = fitness

                    # 更新全局最优
                    if fitness > best_fitness:
                        best_text = x_adv.copy()
                        best_fitness = fitness
                        perturbation_rate, perturbation_num = count_perturbation_rate(x_orig, best_text)
                        if perturbation_num == 1:
                            break

                new_population_lists.append(local_best_x.copy())
                new_fitness_list.append(local_best_fitness)

            # 更新种群为局部搜索后的结果
            population_lists = new_population_lists
            fitness_list = new_fitness_list

            # ===== 阶段3: 选择top-k进入下一代 =====
            index = np.argsort(fitness_list)[::-1][:top_k]
            population_lists = np.array(population_lists)[index].tolist()
            fitness_list = np.array(fitness_list)[index].tolist()

            # 检查是否达到最优
            perturbation_rate, perturbation_num = count_perturbation_rate(x_orig, best_text)
            if perturbation_num == 1:
                break

        # 记录最终结果
        perturbation_rate, perturbation_num = count_perturbation_rate(x_orig, best_text)
        _, label = self.query_model(best_text)

        confidence_adv, label_adv = self.query_model(best_text)
        confidence_orig, label_orig = self.query_model(x_orig)

        log['adv_confidence'] = confidence_adv.tolist()
        log['label_adv'] = int(label_adv)
        log['confidence_orig'] = confidence_orig.tolist()
        log['label_orig'] = int(label_orig)
        log['final_adversarial_example'] = ' '.join(best_text)
        log['optim_perturbation_num'] = perturbation_num
        log['optim_perturbation_rate'] = perturbation_rate
        log['optim_query_number'] = optim_query_number
        log['after_attack_label'] = int(label)

        if label == y_orig:
            log['status'] = False
        else:
            log['status'] = True

        return log

    def attack(self, text_ls, true_label, query_budget=None, **kwargs):
        """
        执行TextHacker攻击 - 修改版：攻击失败返回None，支持传入query_budget

        Args:
            text_ls: 文本词列表
            true_label: 真实标签
            query_budget: 查询预算（如果为None，则使用默认预算self.args.query_budget）
            **kwargs: 其他参数（用于兼容接口，如semantic_threshold, perturb_ratio等）

        Returns:
            result_text: 对抗样本（列表形式），攻击失败返回None
            attack_success: 是否攻击成功
            query_count: 查询次数
        """
        # 清理显存
        torch.cuda.empty_cache()

        # 确定使用的查询预算
        if query_budget is not None:
            effective_query_budget = query_budget
            print(f"TextHacker使用传入的查询预算: {effective_query_budget}")
        else:
            effective_query_budget = self.args.query_budget
            print(f"TextHacker使用默认查询预算: {effective_query_budget}")

        # 检查预算是否足够
        if effective_query_budget <= 0:
            print(f"TextHacker查询预算不足: {effective_query_budget}")
            return None, False, 0

        # 检查原始预测
        orig_probs = self.predictor([text_ls]).squeeze()
        orig_label = torch.argmax(orig_probs).item()

        if true_label != orig_label:
            return text_ls, False, 1

        # 初始化权重表和候选集
        weight_table, candidate_set, handle_list = self.init_weight_table(text_ls)

        # 阶段1: 对抗样本初始化（使用传入的预算）
        init_log, x_adv, x_orig, label = self.adversary_initialization(
            text_ls, true_label, effective_query_budget, candidate_set, handle_list
        )

        if not init_log['classification']:
            # 攻击失败返回None
            return None, False, init_log.get('init_query_number', 1)

        if not init_log['status']:
            # 攻击失败返回None
            return None, False, init_log.get('init_query_number', effective_query_budget)

        # 阶段2: 扰动优化（使用剩余预算）
        optim_allowed_query_num = effective_query_budget - init_log['init_query_number']

        # 检查优化阶段是否还有预算
        if optim_allowed_query_num <= 0:
            print(f"TextHacker优化阶段无剩余预算，直接返回初始化结果")
            # 初始化阶段成功但没有优化预算，直接返回初始化结果
            result_text = init_log['init_adversarial_example'].split()
            return result_text, True, init_log['init_query_number']

        optim_log = self.perturbation_optimization(
            x_adv, x_orig, true_label, optim_allowed_query_num,
            self.args.population_size, self.args.neighbor_delta,
            self.args.local_search_num, self.args.population_size,
            weight_table, candidate_set, self.args.base_reward
        )

        # 合并日志
        total_queries = init_log['init_query_number'] + optim_log['optim_query_number']
        attack_success = optim_log['status']

        if attack_success:
            result_text = optim_log['final_adversarial_example'].split()
        else:
            # 攻击失败返回None
            print("TextHacker攻击失败，返回None")
            result_text = None

        # 清理显存
        torch.cuda.empty_cache()

        return result_text, attack_success, total_queries


def create_texthacker(predictor, word2idx, idx2word, cos_sim, sim_predictor, stop_words_set, embed_func):
    """创建 TextHacker 实例的工厂函数（适配遗传算法）"""
    return TextHackerWrapper(
        predictor=predictor,
        word2idx=word2idx,
        idx2word=idx2word,
        cos_sim=cos_sim,
        sim_predictor=sim_predictor,
        stop_words_set=stop_words_set,
        embed_func=embed_func
    )


def run_texthacker_wrapper():
    """
    TextHacker包装版攻击主函数
    """
    print("=" * 80)
    print("TextHacker包装版攻击程序启动")
    print("=" * 80)

    main_start_time = time.time()

    # 1. 初始化参数
    args = Args()
    print("\n攻击参数配置:")
    print(f"  查询预算: {args.query_budget}")
    print(f"  种群大小: {args.population_size}")
    print(f"  邻域大小: {args.neighbor_delta}")
    print(f"  同义词数量: {args.synonym_num}")
    print(f"  语义阈值: {args.semantic_threshold}")
    print(f"  扰动阈值: {args.perturb_ratio}")
    print("参数解析完成")
    print("\n正在使用 tool.py 初始化全局组件...")

    # 初始化所有全局组件
    start_time = time.time()
    components = initialize_global_components(global_seed=2026)
    print(f"全局组件初始化完成，耗时: {time.time() - start_time:.2f}秒")

    # 2. 获取攻击所需组件
    print("\n正在获取攻击所需组件...")
    dataset = get_component('dataset')
    predictor = get_component('predictor')
    word2idx = get_component('word2idx')
    idx2word = get_component('idx2word')
    cos_sim = get_component('cos_sim')
    sim_predictor = get_component('sim_predictor')
    stop_words_set = get_component('stop_words_set')

    dataset_1 = dataset[:50]
    # 3. 创建TextHacker攻击器
    print("\n创建TextHacker攻击器...")
    texthacker = TextHackerWrapper(
        predictor=predictor,
        word2idx=word2idx,
        idx2word=idx2word,
        cos_sim=cos_sim,
        args=args,
        sim_predictor=sim_predictor,
        stop_words_set=stop_words_set,
    )

    # 4. 准备数据
    # dataset格式: [({'text': text_list}, label), ...]
    print("\n正在准备攻击数据...")
    data = []
    for item, label in dataset_1:
        text_list = item['text']  # 已经是词列表格式
        data.append((text_list, label))

    print(f"攻击数据准备完成，共 {len(data)} 个样本")
    if len(data) > 0:
        print(f"   示例文本: {' '.join(data[0][0])}")
        print(f"   示例标签: {data[0][1]}")

    # 约束阈值
    SEMANTIC_THRESHOLD = args.semantic_threshold
    PERTURB_RATIO = args.perturb_ratio
    QUERY_BUDGET =  args.query_budget

    # 5. 执行攻击
    print("\n" + "=" * 80)
    print("开始执行批量攻击")
    print("=" * 80)

    # 统计变量
    raw_successful_attacks = 0
    constraint_passed_attacks = 0
    skipped_samples = 0
    total_attack_time = 0
    sims = []
    pert_rates = []
    attacked_samples = 0
    total_queries = 0

    # 对每个样本执行攻击
    for idx, (text_ls, true_label) in enumerate(data):
        print(f"\n{'=' * 20} 样本 {idx + 1}/{len(data)} {'=' * 20}")
        sample_start_time = time.time()

        # 定期内存清理
        if idx % 5 == 0:
            cleanup_memory()

        print(f"原始文本: {' '.join(text_ls)}")
        print(f"真实标签: {true_label}")

        # 检测模型预测是否正确
        try:
            orig_probs = predictor([text_ls]).squeeze()
            orig_pred_label = torch.argmax(orig_probs).item()
            true_label_int = true_label.item() if isinstance(true_label, torch.Tensor) else true_label

            if orig_pred_label != true_label_int:
                print(f"⚠️  样本 {idx + 1} 模型原始预测错误（预测: {orig_pred_label}, "
                      f"真实: {true_label_int}），跳过该样本")
                skipped_samples += 1
                torch.cuda.empty_cache()
                continue

            print(f"✓ 样本 {idx + 1} 模型原始预测正确（预测: {orig_pred_label}, "
                  f"真实: {true_label_int}），开始攻击")

        except Exception as e:
            print(f"⚠️  样本 {idx + 1} 预测检测出错: {str(e)}")
            skipped_samples += 1
            torch.cuda.empty_cache()
            continue

        try:
            # 执行TextHacker攻击（使用默认预算）
            result_text, attack_success, query_count = texthacker.attack(
                text_ls=text_ls,
                true_label=true_label,
                query_budget=args.query_budget  # 显式传入预算
            )

            total_queries += query_count
            attacked_samples += 1

            if attack_success and result_text is not None:
                # 记录原始成功数
                raw_successful_attacks += 1

                # 计算相似度和扰动率
                original_text_str = ' '.join(text_ls)
                adversarial_text_str = ' '.join(result_text) if isinstance(result_text, list) else str(result_text)

                # 计算语义相似度
                try:
                    from .tool import semantic_sim
                    similarity_score = semantic_sim(original_text_str, adversarial_text_str)
                except Exception as e:
                    print(f"相似度计算失败: {e}")
                    similarity_score = 0.0

                # 计算扰动率
                perturbation_rate = get_pert_rate(original_text_str, adversarial_text_str)

                # 约束检查（添加查询次数检查）
                semantic_check = similarity_score >= SEMANTIC_THRESHOLD
                perturb_check = perturbation_rate <= PERTURB_RATIO
                query_check = query_count <= QUERY_BUDGET  # 新增查询检查

                print(f"\n样本 {idx + 1} 攻击成功 - 约束检查:")
                print(
                    f"   语义相似度: {similarity_score:.4f} (阈值: {SEMANTIC_THRESHOLD}) - {'✓ 通过' if semantic_check else '✗ 不通过'}")
                print(
                    f"   扰动比例: {perturbation_rate:.4f} (阈值: {PERTURB_RATIO}) - {'✓ 通过' if perturb_check else '✗ 不通过'}")
                print(
                    f"   查询次数: {query_count} (阈值: {QUERY_BUDGET}) - {'✓ 通过' if query_check else '✗ 不通过'}")  # 新增

                # 只有同时满足三个约束才计入成功统计
                if semantic_check and perturb_check and query_check:  # 添加 query_check
                    constraint_passed_attacks += 1
                    sims.append(similarity_score)
                    pert_rates.append(perturbation_rate)

                    print(f"   ✓ 约束检查全部通过，计入成功统计")
                    print(f"   原始文本: {original_text_str}")
                    print(f"   对抗文本: {adversarial_text_str}")
                else:
                    print(f"   ✗ 约束检查未通过，不计入成功统计")
                    if not semantic_check:
                        print(f"      原因: 语义相似度不足 ({similarity_score:.4f} < {SEMANTIC_THRESHOLD})")
                    if not perturb_check:
                        print(f"      原因: 扰动比例过大 ({perturbation_rate:.4f} > {PERTURB_RATIO})")
                    if not query_check:  # 新增
                        print(f"      原因: 查询次数超限 ({query_count} > {QUERY_BUDGET})")
            else:
                print(f"样本 {idx + 1} 攻击失败，返回None")
                print(f"   查询次数: {query_count}")

        except Exception as e:
            print(f"样本 {idx + 1} 攻击出错: {str(e)}")
            import traceback
            traceback.print_exc()
            attacked_samples += 1

        sample_time = time.time() - sample_start_time
        total_attack_time += sample_time
        print(f"样本 {idx + 1} 处理时间: {sample_time:.2f}秒")

        # 打印当前统计信息
        print(f"当前统计:")
        print(f"   已跳过样本: {skipped_samples}/{idx + 1}")
        print(f"   已攻击样本: {attacked_samples}/{idx + 1}")
        print(f"   原始攻击成功: {raw_successful_attacks}/{attacked_samples}")
        print(f"   约束筛选后成功: {constraint_passed_attacks}/{attacked_samples}")
        if constraint_passed_attacks > 0:
            print(f"   成功样本平均语义相似度: {np.mean(sims):.4f}")
            print(f"   成功样本平均扰动率: {np.mean(pert_rates):.4f}")
            print(f"   平均查询次数: {total_queries / attacked_samples:.1f}")

    # 计算和输出最终统计结果
    total_main_time = time.time() - main_start_time
    print(f"\n" + "=" * 80)
    print("TextHacker包装版攻击程序全部完成!")
    print("=" * 80)
    print(f"最终统计结果:")
    print(f"   总样本数: {len(data)}")
    print(f"   跳过样本数（模型预测错误）: {skipped_samples} ({skipped_samples / len(data) * 100:.1f}%)")
    print(f"   有效攻击样本数: {attacked_samples} ({attacked_samples / len(data) * 100:.1f}%)")
    print(f"   原始攻击成功: {raw_successful_attacks} "
          f"({raw_successful_attacks / attacked_samples * 100:.1f}%)" if attacked_samples > 0
          else "   原始攻击成功: 0 (N/A)")
    print(f"   约束筛选后成功: {constraint_passed_attacks} "
          f"({constraint_passed_attacks / attacked_samples * 100:.1f}%)" if attacked_samples > 0
          else "   约束筛选后成功: 0 (N/A)")
    print(f"   约束筛选通过率: {constraint_passed_attacks / raw_successful_attacks * 100:.1f}%"
          if raw_successful_attacks > 0 else "   约束筛选通过率: N/A")

    if attacked_samples > 0:
        print(f"   平均查询次数: {total_queries / attacked_samples:.1f}")

    if len(sims) > 0:
        print(f"   约束筛选后成功样本统计:")
        print(f"     平均语义相似度: {np.mean(sims):.4f} ± {np.std(sims):.4f}")
        print(f"     平均扰动率: {np.mean(pert_rates):.4f} ± {np.std(pert_rates):.4f}")
        print(f"     相似度范围: [{np.min(sims):.4f}, {np.max(sims):.4f}]")
        print(f"     扰动率范围: [{np.min(pert_rates):.4f}, {np.max(pert_rates):.4f}]")

    print(f"性能统计:")
    print(f"   总运行时间: {total_main_time:.2f}秒")
    if attacked_samples > 0:
        print(f"   平均每样本时间: {total_attack_time / attacked_samples:.2f}秒")
    print(f"   总查询次数: {total_queries}")

    # 最终清理所有资源
    print("执行最终资源清理...")
    cleanup_memory()
    print("=" * 80)

    # 返回详细的统计结果
    results = {
        'total_samples': len(data),
        'skipped_samples': skipped_samples,
        'valid_samples': attacked_samples,
        'raw_successful_attacks': raw_successful_attacks,
        'constraint_passed_attacks': constraint_passed_attacks,
        'raw_success_rate': raw_successful_attacks / attacked_samples if attacked_samples > 0 else 0,
        'final_success_rate': constraint_passed_attacks / attacked_samples if attacked_samples > 0 else 0,
        'constraint_pass_rate': constraint_passed_attacks / raw_successful_attacks if raw_successful_attacks > 0 else 0,
        'avg_similarity': np.mean(sims) if len(sims) > 0 else 0,
        'avg_perturbation_rate': np.mean(pert_rates) if len(pert_rates) > 0 else 0,
        'avg_queries': total_queries / attacked_samples if attacked_samples > 0 else 0,
        'total_time': total_main_time,
        'semantic_threshold': SEMANTIC_THRESHOLD,
        'perturb_threshold': PERTURB_RATIO
    }

    return results


def analyze_texthacker_attack_results(results: Dict[str, Any]):
    """分析TextHacker攻击结果并输出详细报告"""
    print("\n" + "=" * 60)
    print("TextHacker包装版攻击结果详细分析")
    print("=" * 60)

    print(f"攻击效果分析:")
    print(f"   总样本数: {results['total_samples']}")
    print(f"   有效样本数: {results['valid_samples']}")
    print(f"   原始成功攻击: {results['raw_successful_attacks']}")
    print(f"   约束筛选后成功: {results['constraint_passed_attacks']}")
    print(f"   最终攻击成功率: {results['final_success_rate']:.1%}")

    if results['constraint_passed_attacks'] > 0:
        print(f"\n质量指标:")
        print(f"   平均语义相似度: {results['avg_similarity']:.4f}")
        print(f"   平均扰动率: {results['avg_perturbation_rate']:.4f}")

    print(f"\n效率指标:")
    print(f"   平均查询次数: {results['avg_queries']:.1f}")
    print(f"   总运行时间: {results['total_time']:.2f}秒")
    if results['valid_samples'] > 0:
        print(f"   平均每样本时间: {results['total_time'] / results['valid_samples']:.2f}秒")

    print("=" * 60)


# 在文件底部添加直接运行的入口
if __name__ == "__main__":
    # 运行TextHacker包装版攻击
    try:
        results = run_texthacker_wrapper()
        if results:
            analyze_texthacker_attack_results(results)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()
