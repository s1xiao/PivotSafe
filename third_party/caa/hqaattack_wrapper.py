# HQAAttack 包装函数 - 修改版：攻击失败返回None
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

import gc


def fn(sim_ls):
    abs_sum = np.sum([np.abs(i) for i in sim_ls])
    for j in range(len(sim_ls)):
        sim_ls[j] = sim_ls[j] / abs_sum


def get_attack_result(new_text, predictor, orig_label, batch_size, hash_qrs):
    """测试攻击效果 - HQA版本（添加显存清理）"""
    if " ".join(new_text[0]) in hash_qrs:
        return hash_qrs[" ".join(new_text[0])], 1

    new_probs = predictor(new_text, batch_size=batch_size)

    torch.cuda.empty_cache()

    pr = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    hash_qrs[" ".join(new_text[0])] = pr
    return pr, 1

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

# def get_pert_rate(text, ade):
#     """计算扰动率"""
#     text_ls = text.split()
#     ade_ls = ade.split()
#     changed = 0
#     for i, j in zip(text_ls, ade_ls):
#         if i != j:
#             changed += 1
#     return changed / len(text_ls)


def get_current_perturbation_ratio(original_text, current_text):
    """计算当前扰动比例"""
    if len(original_text) != len(current_text):
        return 1.0

    changed_words = 0
    for orig_word, curr_word in zip(original_text, current_text):
        if orig_word != curr_word:
            changed_words += 1

    return changed_words / len(original_text)


def check_perturbation_constraint(original_text, candidate_text, max_perturb_ratio):
    """检查是否满足扰动比例约束"""
    current_ratio = get_current_perturbation_ratio(original_text, candidate_text)
    return current_ratio <= max_perturb_ratio


def softmax(x):
    x = np.array(x)
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def cos_sim_compute(x, y):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    if x.shape != y.shape:
        return 0.0

    return float(torch.cosine_similarity(x, y, dim=0).numpy())


def random_attack(top_k_words, text_ls, true_label,
                  predictor, word2idx, idx2word, cos_sim, sim_score_window=15,
                  batch_size=32):
    """
    随机攻击函数 - HQA版本（添加显存清理）

    Returns:
        (random_text, qrs, orig_label, random_success)
    """
    print(f"\n开始随机攻击...")

    torch.cuda.empty_cache()

    hash_qrs = {}
    start_time = time.time()

    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    torch.cuda.empty_cache()

    if true_label != orig_label:
        print(f"原始文本预测错误，无需攻击")
        return text_ls, 1, orig_label, False
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1

        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}

        words_perturb = []
        pos_ls = criteria_1.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]

        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]

        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, synonym_values = [], []

        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp = []
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp = []
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        qrs = 0
        num_changed = 0
        flag = 0
        th = 0

        # 第一阶段攻击
        while qrs < len(text_ls):
            random_text = text_ls[:]
            replaced_count = 0
            for i in range(len(synonyms_all)):
                idx = synonyms_all[i][0]
                syn = synonyms_all[i][1]
                random_text[idx] = random.choice(syn)
                replaced_count += 1
                if i >= th:
                    break

            pr, vqr = get_attack_result([random_text], predictor, orig_label, batch_size, hash_qrs=hash_qrs)

            if qrs % 10 == 0:
                torch.cuda.empty_cache()

            qrs += vqr
            th += 1
            if th > len_text:
                break
            if np.sum(pr) > 0:
                flag = 1
                break

        old_qrs = qrs

        if flag == 0:
            # 第二阶段攻击
            while qrs < old_qrs + 2500 and flag == 0:
                random_text = text_ls[:]
                replaced_count = 0
                for j in range(len(synonyms_all)):
                    idx = synonyms_all[j][0]
                    syn = synonyms_all[j][1]
                    random_text[idx] = random.choice(syn)
                    replaced_count += 1
                    if j >= len_text:
                        break

                pr, vqr = get_attack_result([random_text], predictor, orig_label, batch_size, hash_qrs=hash_qrs)

                if qrs % 50 == 0:
                    torch.cuda.empty_cache()

                qrs += vqr
                if np.sum(pr) > 0:
                    flag = 1
                    break

        total_time = time.time() - start_time
        if flag == 1:
            print(f"随机攻击成功完成!")
            return random_text, qrs, orig_label, True
        else:
            print(f"随机攻击失败")
            return text_ls, qrs, orig_label, False


def hqa_attack(fuzz_val, optim_step, orig_label, top_k_words, qrs, sample_index, text_ls, random_text_, true_label,
               predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
               import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
               batch_size=32, embed_func='', n_sample=5, k_sample=5, k_threshold=0.8, qrs_limits=1000,
               qrs_stopp=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], perturb_ratio=0.1, max_iterations=100):
    """
    HQA主攻击函数（添加显存清理）

    Returns:
        原有返回格式
    """
    print(f"\n开始HQA攻击...")

    torch.cuda.empty_cache()
    gc.collect()

    start_time = time.time()
    random_text = random_text_[:]

    word_idx_dict = {}
    with open(embed_func, 'r') as ifile:
        for index, line in enumerate(ifile):
            word = line.strip().split()[0]
            word_idx_dict[word] = index

    embed_file = open(embed_func)
    embed_content = embed_file.readlines()

    words_perturb = []
    pos_ls = criteria_1.get_pos(text_ls)
    pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
    for pos in pos_pref:
        for i in range(len(pos_ls)):
            if pos_ls[i] == pos and len(text_ls[i]) > 2:
                words_perturb.append((i, text_ls[i]))

    random.shuffle(words_perturb)
    words_perturb = words_perturb[:top_k_words]

    words_perturb_idx = []
    words_perturb_embed = []
    words_perturb_doc_idx = []

    for idx, word in words_perturb:
        if word in word_idx_dict:
            words_perturb_doc_idx.append(idx)
            words_perturb_idx.append(word2idx[word])
            words_perturb_embed.append(
                [float(num) for num in embed_content[word_idx_dict[word]].strip().split()[1:]])

    words_perturb_embed_matrix = np.asarray(words_perturb_embed)

    synonym_words, synonym_values = [], []
    for idx in words_perturb_idx:
        res = list(zip(*(cos_sim[idx])))
        temp = []
        for ii in res[1]:
            temp.append(idx2word[ii])
        synonym_words.append(temp)
        temp = []
        for ii in res[0]:
            temp.append(ii)
        synonym_values.append(temp)

    synonyms_all = []
    synonyms_dict = defaultdict(list)
    for idx, word in words_perturb:
        if word in word2idx:
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))
                synonyms_dict[word] = synonyms

    flag = 1
    best_qrs = 0
    hash_qrs = {}

    if flag == 1:
        changed = 0
        for i in range(len(text_ls)):
            if text_ls[i] != random_text[i]:
                changed += 1

        changed_indices = []
        num_changed = 0
        for i in range(len(text_ls)):
            if text_ls[i] != random_text[i]:
                changed_indices.append(i)
                num_changed += 1

        random_sim = calc_sim_interface(text_ls, [random_text], -1, sim_score_window)[0]

        best_sim = random_sim
        best_attack = random_text[:]
        best_qrs = qrs
        for kp in qrs_stopp:
            if qrs <= kp:
                optim_step[kp] = [best_sim, " ".join(best_attack), qrs, np.sum(
                    get_attack_result([best_attack], predictor, orig_label, batch_size, hash_qrs)[0]) > 0]

        x_t = random_text[:]

        # 特殊情况：只改变一个词
        if num_changed == 1:
            change_idx = 0
            for i in range(len(text_ls)):
                if text_ls[i] != x_t[i]:
                    change_idx = i
                    break

            idx = word2idx[text_ls[change_idx]]
            res = list(zip(*(cos_sim[idx])))
            for widx in res[1]:
                w = idx2word[widx]
                x_t_candidate = x_t[:]
                x_t_candidate[change_idx] = w

                pr, vqr = get_attack_result([x_t_candidate], predictor, orig_label, batch_size, hash_qrs)
                sim = calc_sim_interface(text_ls, [x_t_candidate], -1, sim_score_window)[0]
                qrs += vqr

                if np.sum(pr) > 0 and round(sim, 3) >= round(best_sim, 3):
                    best_attack = x_t_candidate[:]
                    best_sim = sim
                    best_qrs = qrs
                    for kp in qrs_stopp:
                        if qrs <= kp:
                            optim_step[kp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                get_attack_result([best_attack], predictor, orig_label, batch_size, hash_qrs)[0]) > 0]

            torch.cuda.empty_cache()

            total_time = time.time() - start_time
            result = (' '.join(best_attack), 1, 1, orig_label, 1, qrs, best_sim, random_sim, best_qrs)
            return result

        # 主要的HQA攻击流程
        stack = [random_text[:]]
        stack_str = []
        stack_over = []
        way_back_num = 3

        x_tilde = random_text[:]
        wbcount = 0

        for t in range(max_iterations):
            if t % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            x_t = x_tilde[:]
            x_t_str = " ".join(x_t)

            if " ".join(x_t) not in stack_over and " ".join(x_t) not in stack_str:
                pr, vqr = get_attack_result([x_t], predictor, orig_label, batch_size, hash_qrs)
                qrs += vqr
                if np.sum(pr) > 0:
                    stack.append(x_t[:])
                    stack_str.append(" ".join(x_t))

            num_changed = 0
            for i, j in zip(x_t, text_ls):
                if i != j:
                    num_changed += 1

            if wbcount > way_back_num:
                if len(stack) > 5:
                    popint = random.randint(3, 5)
                    for _ in range(popint):
                        x_t = stack.pop()
                        stack_str.pop()
                    stack_over.append(" ".join(x_t))
                else:
                    x_t = random_text[:]
                    way_back_num += 5
                wbcount = 0

            # 第一次迭代的特殊处理
            while True and t == 0:
                choices = []
                for i in range(len(text_ls)):
                    if x_t[i] != text_ls[i]:
                        new_text = x_t[:]
                        new_text[i] = text_ls[i]

                        semantic_sims = calc_sim_interface(text_ls, [new_text], -1, sim_score_window)

                        pr, vqr = get_attack_result([new_text], predictor, orig_label, batch_size, hash_qrs)
                        qrs += vqr
                        if np.sum(pr) > 0:
                            choices.append((i, semantic_sims[0]))

                torch.cuda.empty_cache()

                if len(choices) > 0:
                    choices.sort(key=lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = x_t[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]

                        pr, vqr = get_attack_result([new_text], predictor, orig_label, batch_size, hash_qrs)
                        qrs += vqr
                        if pr[0] == 0:
                            break
                        x_t[choices[i][0]] = text_ls[choices[i][0]]

                if len(choices) == 0:
                    break

            # 非第一次迭代的处理
            while True and t != 0:
                choices = []

                for i in range(len(text_ls)):
                    if x_t[i] != text_ls[i]:
                        new_text = x_t[:]
                        new_text[i] = text_ls[i]
                        semantic_sims = calc_sim_interface(text_ls, [new_text], -1, sim_score_window)
                        choices.append((i, semantic_sims[0]))

                choices.sort(key=lambda x: x[1])
                choices.reverse()
                flag = True
                for i in range(len(choices)):
                    new_text = x_t[:]
                    new_text[choices[i][0]] = text_ls[choices[i][0]]

                    pr, vqr = get_attack_result([new_text], predictor, orig_label, batch_size, hash_qrs)
                    qrs += vqr
                    if np.sum(pr) > 0:
                        x_t[choices[i][0]] = text_ls[choices[i][0]]
                        sims = calc_sim_interface(text_ls, [x_t], -1, sim_score_window)[0]
                        if round(sims, 3) >= round(best_sim, 3):
                            best_attack = x_t[:]
                            best_sim = sims
                            best_qrs = qrs
                            wbcount = 0
                            for ksp in qrs_stopp:
                                if qrs <= ksp:
                                    optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                        get_attack_result([best_attack], predictor, orig_label, batch_size, hash_qrs)[
                                            0]) > 0]
                        flag = False
                        break
                    if qrs >= qrs_limits:
                        result = (' '.join(best_attack), 1, len(changed_indices),
                                  orig_label, 1, qrs, best_sim, random_sim, best_qrs)
                        return result
                if flag:
                    break

            num_changed = 0
            for i in range(len(text_ls)):
                if text_ls[i] != x_t[i]:
                    num_changed += 1

            x_t_sim = calc_sim_interface(text_ls, [x_t], -1, sim_score_window)[0]

            # 单词优化处理
            if np.sum(get_attack_result([x_t], predictor, orig_label, batch_size, hash_qrs)[0]) > 0 and (
                    num_changed == 1):
                change_idx = 0
                for i in range(len(text_ls)):
                    if text_ls[i] != x_t[i]:
                        change_idx = i
                        break

                idx = word2idx[text_ls[change_idx]]
                res = list(zip(*(cos_sim[idx])))
                for widx in res[1]:
                    w = idx2word[widx]
                    x_t_candidate = x_t[:]
                    x_t_candidate[change_idx] = w

                    pr, vqr = get_attack_result([x_t_candidate], predictor, orig_label, batch_size, hash_qrs)
                    sim = calc_sim_interface(text_ls, [x_t_candidate], -1, sim_score_window)[0]
                    qrs += vqr

                    if np.sum(pr) > 0 and round(sim, 3) >= round(best_sim, 3):
                        best_sim = sim
                        best_attack = x_t_candidate[:]
                        best_qrs = qrs
                        wbcount = 0
                        for ksp in qrs_stopp:
                            if qrs <= ksp:
                                optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                    get_attack_result([best_attack], predictor, orig_label, batch_size, hash_qrs)[
                                        0]) > 0]
                    if qrs >= qrs_limits:
                        break

                torch.cuda.empty_cache()

                total_time = time.time() - start_time
                result = (
                ' '.join(best_attack), 1, len(changed_indices), orig_label, 1, qrs, best_sim, random_sim, best_qrs)
                return result

            # 更新最佳攻击
            if (np.sum(get_attack_result([x_t], predictor, orig_label, batch_size, hash_qrs)[0]) > 0 and
                    (round(x_t_sim, 3) > round(best_sim, 3))):
                best_attack = x_t[:]
                best_sim = x_t_sim
                best_qrs = qrs
                wbcount = 0
                for ksp in qrs_stopp:
                    if qrs <= ksp:
                        optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                            get_attack_result([best_attack], predictor, orig_label, batch_size, hash_qrs)[0]) > 0]
                wbcount = 0

            # 检查查询限制
            if qrs >= qrs_limits:
                break

            # 核心优化算法 - 基于词向量的同义词替换
            nonzero_ele = []
            perturb_word_idx_list = []
            ni = 0

            for idx in words_perturb_doc_idx:
                if text_ls[idx] != x_t[idx]:
                    nonzero_ele.append(ni)
                ni += 1

            l2s = []
            for j in range(len(nonzero_ele)):
                x_t_orig_word = x_t[synonyms_all[nonzero_ele[j]][0]]
                orig_word = text_ls[synonyms_all[nonzero_ele[j]][0]]
                v1 = np.array(
                    [float(num) for num in embed_content[word_idx_dict[x_t_orig_word]].strip().split()[1:]])
                v2 = np.array([float(num) for num in embed_content[word_idx_dict[orig_word]].strip().split()[1:]])
                sim_ = cos_sim_compute(v1, v2)
                distance = 1 + sim_
                l2s.append(distance)

            p = torch.softmax(torch.tensor(l2s), dim=0).numpy()
            p /= p.sum()

            perturb_word_idx_list = np.random.choice(nonzero_ele, len(nonzero_ele), replace=False, p=p)

            x_tilde = text_ls[:]

            for perturb_word_idx in perturb_word_idx_list:
                x_t_orig_word = x_t[synonyms_all[perturb_word_idx][0]]
                orig_word = text_ls[synonyms_all[perturb_word_idx][0]]
                ad_replacement = []
                n_samples = []

                while len(n_samples) < n_sample:
                    syn_idx = random.randint(0, 49)
                    if syn_idx not in n_samples:
                        n_samples.append(syn_idx)

                for _ in range(n_sample):
                    x_t_tmp = x_t[:]
                    syn_idx = n_samples[_]
                    replacement = synonyms_all[perturb_word_idx][1][syn_idx]
                    x_t_tmp[synonyms_all[perturb_word_idx][0]] = replacement

                    pr_v, vqr = get_attack_result([x_t_tmp], predictor, orig_label, batch_size, hash_qrs)
                    if np.sum(pr_v) > 0:
                        sim_tmp = calc_sim_interface(text_ls, [x_t_tmp], -1, sim_score_window)[0]
                        if round(sim_tmp, 3) >= round(best_sim, 3):
                            best_attack = x_t_tmp[:]
                            best_sim = sim_tmp
                            best_qrs = qrs
                            wbcount = 0
                            for ksp in qrs_stopp:
                                if qrs <= ksp:
                                    optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                        get_attack_result([best_attack], predictor, orig_label, batch_size,
                                                          hash_qrs)[
                                            0]) > 0]
                            wbcount = 0
                        ad_replacement.append((replacement, sim_tmp, syn_idx))

                    qrs += vqr
                    if qrs >= qrs_limits:
                        result = (' '.join(best_attack), 1, len(changed_indices),
                                  orig_label, 1, qrs, best_sim, random_sim, best_qrs)
                        return result

                if len(ad_replacement) != 0:
                    ad_replacement = sorted(ad_replacement, key=lambda x: x[1], reverse=True)

                    condi_replacement = ad_replacement[0][0]
                    condi_replacement_idx = word2idx[condi_replacement]
                    base_sim = ad_replacement[0][1]
                    x_t_base = x_t[:]
                    x_t_base[synonyms_all[perturb_word_idx][0]] = condi_replacement

                    res = list(zip(*(cos_sim[condi_replacement_idx])))
                    condi_replacements = []
                    for i, j in zip(res[0], res[1]):
                        if i > k_threshold and j != condi_replacement_idx:
                            condi_replacements.append((j, i, np.array(
                                [float(num) for num in
                                 embed_content[word_idx_dict[idx2word[j]]].strip().split()[1:]])))

                    best_rep = ad_replacement[0][0]

                    if len(condi_replacements) == 0:
                        cds = []
                    elif k_sample <= 0:
                        cds = []
                    elif len(condi_replacements) < k_sample:
                        cds = condi_replacements.copy()
                    else:
                        cds = random.sample(condi_replacements, k_sample)

                    vec_ls = []
                    condi_vec = np.array(
                        [float(num) for num in embed_content[condi_replacement_idx].strip().split()[1:]])
                    sim_ls = []
                    for ite in cds:
                        cd, similarity, vec = ite
                        cdw = idx2word[cd]
                        x_t_tmp = x_t[:]
                        x_t_tmp[synonyms_all[perturb_word_idx][0]] = cdw
                        sim_tmp = calc_sim_interface(text_ls, [x_t_tmp], -1, sim_score_window)[0]
                        inc = sim_tmp - base_sim
                        vec_ls.append(np.array(vec) - condi_vec)
                        sim_ls.append(inc)
                    sim_ls = np.array(sim_ls)
                    vec_ls = np.array(vec_ls)
                    fn(sim_ls)
                    estimate_vec = 0
                    for e_idx in range(len(sim_ls)):
                        estimate_vec += sim_ls[e_idx] * vec_ls[e_idx]

                    candi_2 = []

                    contextual_replacements = []
                    res = list(zip(*(cos_sim[word2idx[orig_word]])))
                    for i, j in zip(res[0], res[1]):
                        contextual_replacements.append((j, i, np.array(
                            [float(num) for num in embed_content[word_idx_dict[idx2word[j]]].strip().split()[1:]])))
                    for ite in contextual_replacements:
                        cd, similarity, vec = ite
                        candi_2.append((cd, cos_sim_compute(estimate_vec, np.array(vec) - condi_vec)))
                    candi_2 = sorted(candi_2, key=lambda x: x[-1], reverse=True)

                    for ite in candi_2:
                        cd, similarity = ite
                        x_t_tmp = x_t[:]
                        x_t_tmp[synonyms_all[perturb_word_idx][0]] = idx2word[cd]

                        pr, vqr = get_attack_result([x_t_tmp], predictor, orig_label, batch_size, hash_qrs)
                        qrs += vqr

                        if np.sum(pr) > 0:
                            sim_tmp = calc_sim_interface(text_ls, [x_t_tmp], -1, sim_score_window)[0]
                            if round(sim_tmp, 3) >= round(best_sim, 3):
                                best_attack = x_t_tmp[:]
                                best_sim = sim_tmp
                                best_qrs = qrs
                                wbcount = 0
                                for ksp in qrs_stopp:
                                    if qrs <= ksp:
                                        optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                            get_attack_result([best_attack], predictor, orig_label, batch_size,
                                                              hash_qrs)[0]) > 0]
                            best_rep = idx2word[cd]
                            break
                        if qrs >= qrs_limits:
                            if np.sum(pr) > 0:
                                sim_tmp = calc_sim_interface(text_ls, [x_t_tmp], -1, sim_score_window)[0]
                                if round(sim_tmp, 3) >= round(best_sim, 3):
                                    best_attack = x_t_tmp[:]
                                    best_sim = sim_tmp
                                    best_qrs = qrs
                                    wbcount = 0
                                    for ksp in qrs_stopp:
                                        if qrs <= ksp:
                                            optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                                get_attack_result([best_attack], predictor, orig_label, batch_size,
                                                                  hash_qrs)[0]) > 0]

                            result = (' '.join(best_attack), 1, len(changed_indices),
                                      orig_label, 1, qrs, best_sim, random_sim, best_qrs)
                            return result
                    x_tilde[synonyms_all[perturb_word_idx][0]] = best_rep
                else:
                    x_tilde[synonyms_all[perturb_word_idx][0]] = x_t_orig_word

                pr, vqr = get_attack_result([x_tilde], predictor, orig_label, batch_size, hash_qrs)
                qrs += vqr
                if np.sum(pr) > 0:
                    sim_new = calc_sim_interface(text_ls, [x_tilde], -1, sim_score_window)[0]
                    if (round(sim_new, 3) >= round(best_sim, 3)) and (
                            np.sum(
                                get_attack_result([x_tilde], predictor, orig_label, batch_size, hash_qrs)[0]) > 0):
                        best_attack = x_tilde[:]
                        best_sim = sim_new
                        best_qrs = qrs
                        wbcount = 0
                        for ksp in qrs_stopp:
                            if qrs <= ksp:
                                optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                    get_attack_result([best_attack], predictor, orig_label, batch_size, hash_qrs)[
                                        0]) > 0]
                if qrs >= qrs_limits:
                    result = (' '.join(best_attack), 1, len(changed_indices),
                              orig_label, 1, qrs, best_sim, random_sim, best_qrs)
                    return result
                if np.sum(pr) > 0:
                    break

            if np.sum(pr) > 0:
                sim_new = calc_sim_interface(text_ls, [x_tilde], -1, sim_score_window)[0]
                if (round(sim_new, 3) > round(best_sim, 3)) and (
                        np.sum(get_attack_result([x_tilde], predictor, orig_label, batch_size, hash_qrs)[0]) > 0):
                    best_attack = x_tilde[:]
                    best_sim = sim_new
                    best_qrs = qrs
                    wbcount = 0
                    for ksp in qrs_stopp:
                        if qrs <= ksp:
                            optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                get_attack_result([best_attack], predictor, orig_label, batch_size, hash_qrs)[
                                    0]) > 0]
            else:
                while True:
                    changed_tt = []
                    for i in range(len(x_t)):
                        if x_tilde[i] != x_t[i]:
                            x_tilde_base = x_tilde[:]
                            x_tilde_base[i] = x_t[i]
                            sim_tmp = calc_sim_interface(text_ls, [x_tilde_base], -1, sim_score_window)[0]
                            changed_tt.append((i, sim_tmp))
                    changed_tt = sorted(changed_tt, key=lambda x: x[-1], reverse=True)
                    x_tilde[changed_tt[0][0]] = x_t[changed_tt[0][0]]
                    if np.sum(get_attack_result([x_tilde], predictor, orig_label, batch_size, hash_qrs)[0]) > 0:
                        qrs += 1
                        sim_new = calc_sim_interface(text_ls, [x_tilde], -1, sim_score_window)[0]
                        if round(sim_new, 3) >= round(best_sim, 3):
                            best_sim = sim_new
                            best_attack = x_tilde[:]
                            wbcount = 0
                            for ksp in qrs_stopp:
                                if qrs <= ksp:
                                    optim_step[ksp] = [best_sim, " ".join(best_attack), best_qrs, np.sum(
                                        get_attack_result([best_attack], predictor, orig_label, batch_size,
                                                          hash_qrs)[
                                            0]) > 0]
                        if qrs >= qrs_limits:
                            result = (' '.join(best_attack), 1, len(changed_indices),
                                      orig_label, 1, qrs, best_sim, random_sim, best_qrs)
                            return result
                        break
                    qrs += 1

            if x_t_str == " ".join(x_tilde):
                wbcount += 1
            else:
                wbcount = 0

            torch.cuda.empty_cache()

        # 函数返回前
        sim = float(best_sim)

        if isinstance(best_attack, str):
            best_attack_list = best_attack.split()
        else:
            best_attack_list = best_attack

        max_changes = 0
        for i in range(len(text_ls)):
            if i < len(best_attack_list) and text_ls[i] != best_attack_list[i]:
                max_changes += 1

        total_time = time.time() - start_time
        print(f"\nHQA攻击完成!")

        torch.cuda.empty_cache()
        gc.collect()

        result = (' '.join(best_attack) if isinstance(best_attack, list) else best_attack,
                  max_changes, len(changed_indices), orig_label, 1, qrs, sim, random_sim, best_qrs)
        return result

    else:
        print("Not Found")
        result = ('', 0, 0, orig_label, orig_label, 0, 0, 0, 0)
        return result


class HQAAttackWrapper:
    def __init__(self,
                 predictor,
                 word2idx,
                 idx2word,
                 cos_sim,
                 sim_predictor,
                 stop_words_set,
                 embed_func=None,
                 ):
        """初始化 HQAAttack 包装器"""
        self.predictor = predictor
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.cos_sim = cos_sim
        self.sim_predictor = sim_predictor
        self.stop_words_set = stop_words_set
        self.embed_func = embed_func

    def attack(self, text_ls, true_label, query_budget=1000, semantic_threshold=0.8,
               top_k_words=1000000, max_iterations=100, sim_score_window=15,
               perturb_ratio=0.15, batch_size=16, embed_func=None):

        print(f"\n开始 HQAAttack 攻击...")

        torch.cuda.empty_cache()
        gc.collect()

        hash_qrs = {}
        orig_probs = self.predictor([text_ls]).squeeze()
        orig_label = torch.argmax(orig_probs)
        orig_confidence = orig_probs[orig_label].item()
        print(f"[原始预测] 标签: {orig_label}, 置信度: {orig_confidence:.4f}")

        try:
            perturb_ratio = min(perturb_ratio, 0.5)

            # 步骤1：随机攻击
            print("执行随机攻击阶段...")
            random_attack_result = random_attack(
                top_k_words=top_k_words,
                text_ls=text_ls,
                true_label=true_label,
                predictor=self.predictor,
                word2idx=self.word2idx,
                idx2word=self.idx2word,
                cos_sim=self.cos_sim,
                sim_score_window=sim_score_window,
                batch_size=batch_size
            )

            # 解包结果
            random_text, random_qrs, orig_label, random_success = random_attack_result

            torch.cuda.empty_cache()
            gc.collect()

            if random_qrs >= query_budget:
                print(f"随机攻击阶段用完查询预算")
                if random_success:
                    return random_text, True, random_qrs
                else:
                    # 攻击失败返回None
                    print("随机攻击失败，返回None")
                    return None, False, random_qrs

            if true_label != orig_label:
                print("原文本预测错误，无需攻击")
                return text_ls, False, random_qrs

            if not random_success:
                print("随机攻击失败，无法继续")
                # 攻击失败返回None
                return None, False, random_qrs

            # 步骤2：HQA攻击优化
            print("执行 HQA 攻击优化阶段...")
            remaining_budget = query_budget - random_qrs

            if remaining_budget <= 0:
                print("查询预算已用完，返回随机攻击结果")
                return random_text, random_success, random_qrs

            qrs_stopp = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
            optim_step = {}
            for kp in qrs_stopp:
                optim_step[kp] = []

            hqa_result = hqa_attack(
                fuzz_val=0,
                optim_step=optim_step,
                orig_label=orig_label,
                top_k_words=top_k_words,
                qrs=random_qrs,
                sample_index=0,
                text_ls=text_ls,
                random_text_=random_text,
                true_label=true_label,
                predictor=self.predictor,
                stop_words_set=self.stop_words_set,
                word2idx=self.word2idx,
                idx2word=self.idx2word,
                cos_sim=self.cos_sim,
                sim_predictor=self.sim_predictor,
                import_score_threshold=-1.0,
                sim_score_threshold=semantic_threshold,
                sim_score_window=sim_score_window,
                batch_size=batch_size,
                embed_func=self.embed_func,
                n_sample=5,
                k_sample=5,
                k_threshold=0.8,
                qrs_limits=query_budget,
                qrs_stopp=qrs_stopp,
                perturb_ratio=perturb_ratio,
                max_iterations=max_iterations
            )

            torch.cuda.empty_cache()
            gc.collect()

            # 处理HQA攻击返回结果
            if len(hqa_result) >= 9:
                final_text_str, num_changed, random_changed, orig_label_out, new_label, total_queries, db_sim, random_sim, best_qrs = hqa_result

                if isinstance(final_text_str, str):
                    final_text = final_text_str.split()
                else:
                    final_text = final_text_str

                if isinstance(final_text, list) and len(final_text) > 0:
                    text_changed = (final_text != text_ls)

                    final_attack_result, _ = get_attack_result(
                        [final_text],
                        self.predictor,
                        true_label,
                        batch_size,
                        {}
                    )
                    label_changed = bool(np.sum(final_attack_result) > 0)

                    attack_success = text_changed and label_changed

                else:
                    attack_success = False
                    final_text = None

                print(f"HQAAttack 完成:")
                print(f"   攻击成功: {attack_success}")
                print(f"   总查询次数: {total_queries}")
                print(f"   语义相似度: {db_sim:.4f}")

                if attack_success:
                    return final_text, True, total_queries
                else:
                    # 攻击失败返回None
                    print("HQA攻击失败，返回None")
                    return None, False, total_queries
            else:
                print("HQA attack 返回结果格式错误")
                # 攻击失败返回None
                return None, False, random_qrs

        except Exception as e:
            print(f"HQAAttack 执行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 异常时返回None
            return None, False, 0

        finally:
            torch.cuda.empty_cache()
            gc.collect()


def create_hqaattack(predictor, word2idx, idx2word, cos_sim, sim_predictor, stop_words_set, embed_func):
    """创建 HQAAttack 实例的工厂函数"""
    return HQAAttackWrapper(
        predictor=predictor,
        word2idx=word2idx,
        idx2word=idx2word,
        cos_sim=cos_sim,
        sim_predictor=sim_predictor,
        stop_words_set=stop_words_set,
        embed_func=embed_func
    )

def run_hqaattack_wrapper():
    """直接运行HQAAttack包装版攻击函数，无需命令行参数"""
    print("开始HQAAttack包装版文本对抗攻击程序...")
    print("=" * 80)

    # 创建模拟的命令行参数
    class Args:
        def __init__(self):
            # 攻击参数设置
            self.data_size = 10  # 处理的数据量
            self.query_budget = 1000  # 查询预算
            self.semantic_threshold = 0.9  # 语义相似度阈值
            self.perturb_ratio = 0.1  # 扰动强度
            self.top_k_words = 1000000  # 前K个词
            self.max_iterations = 100  # 最大迭代次数
            self.sim_score_window = 15  # 相似度计算窗口
            self.batch_size = 16  # 批处理大小

            # 输出设置
            self.output_dir = 'hqaattack_results'

    # 创建参数对象
    args = Args()
    print("HQAAttack攻击参数设置:")
    print(f"   处理样本数: {args.data_size}")
    print(f"   查询预算: {args.query_budget}")
    print(f"   语义相似度阈值: {args.semantic_threshold}")
    print(f"   扰动强度: {args.perturb_ratio}")
    print(f"   最大迭代次数: {args.max_iterations}")
    print(f"   批处理大小: {args.batch_size}")
    print("=" * 80)

    # 调用主函数
    main_hqaattack_wrapper(args)


def main_hqaattack_wrapper(args=None):
    """HQAAttack包装版主函数，使用tool.py组件和包装的HQAAttack"""
    main_start_time = time.time()

    print("参数解析完成")
    print("\n正在使用tool.py初始化全局组件...")

    # 初始化所有全局组件
    start_time = time.time()
    components = initialize_global_components(global_seed=2026)
    print(f"全局组件初始化完成，耗时: {time.time() - start_time:.2f}秒")

    # 获取各个组件
    print("\n正在获取攻击所需组件...")
    dataset = get_component('dataset')
    predictor = get_component('predictor')
    word2idx = get_component('word2idx')
    idx2word = get_component('idx2word')
    cos_sim = get_component('cos_sim')
    sim_predictor = get_component('sim_predictor')
    stop_words_set = get_component('stop_words_set')
    embed_func = get_component('embed_func')

    print("组件获取完成:")
    print(f"   数据集大小: {len(dataset)}")
    print(f"   词汇表大小: {len(word2idx)}")
    print(f"   停用词数量: {len(stop_words_set)}")

    # 转换数据格式为原始格式
    print("\n正在准备攻击数据...")
    data = []
    for item, label in dataset[:50]:  #=============================
        text_list = item['text']  # 已经是词列表格式
        data.append((text_list, label))

    print(f"攻击数据准备完成，共 {len(data)} 个样本")
    if len(data) > 0:
        print(f"   示例文本: {' '.join(data[0][0])}")
        print(f"   示例标签: {data[0][1]}")

    # 创建HQAAttack包装器实例
    print("\n正在创建HQAAttack包装器...")
    hqaattack = create_hqaattack(
        predictor=predictor,
        word2idx=word2idx,
        idx2word=idx2word,
        cos_sim=cos_sim,
        sim_predictor=sim_predictor,
        stop_words_set=stop_words_set,
        embed_func=embed_func
    )
    print("HQAAttack包装器创建完成")

    # 定义约束参数
    SEMANTIC_THRESHOLD = args.semantic_threshold  # 语义相似度阈值
    PERTURB_RATIO = args.perturb_ratio  # 扰动比例阈值
    QUERY_BUDGET =args.query_budget
    print(f"\n约束条件设置:")
    print(f"   语义相似度阈值: {SEMANTIC_THRESHOLD}")
    print(f"   扰动比例阈值: {PERTURB_RATIO}")

    print("\n" + "=" * 80)
    print("开始对抗攻击!")
    print("=" * 80)

    # 攻击统计变量
    sims = []
    pert_rates = []
    raw_successful_attacks = 0  # 原始攻击成功数（未筛选）
    constraint_passed_attacks = 0  # 通过约束筛选的成功数
    total_attack_time = 0
    skipped_samples = 0  # 跳过的样本数（模型预测错误的样本）
    attacked_samples = 0  # 实际攻击的样本数
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

        # 在攻击前检测模型预测是否正确
        try:
            # 获取模型对原始样本的预测
            orig_probs = predictor([text_ls]).squeeze()
            orig_pred_label = torch.argmax(orig_probs).item()

            # 转换真实标签为整数
            true_label_int = true_label.item() if isinstance(true_label, torch.Tensor) else true_label

            # 检查预测是否正确
            if orig_pred_label != true_label_int:
                print(
                    f"⚠️  样本 {idx + 1} 模型原始预测错误（预测: {orig_pred_label}, 真实: {true_label_int}），跳过该样本")
                skipped_samples += 1

                # 清理显存
                torch.cuda.empty_cache()
                continue  # 跳过该样本，不进行攻击

            print(f"✓ 样本 {idx + 1} 模型原始预测正确（预测: {orig_pred_label}, 真实: {true_label_int}），开始攻击")

        except Exception as e:
            print(f"⚠️  样本 {idx + 1} 预测检测出错: {str(e)}")
            skipped_samples += 1
            torch.cuda.empty_cache()
            continue

        try:
            # 执行HQAAttack攻击
            result_text, attack_success, query_count = hqaattack.attack(
                text_ls=text_ls,
                true_label=true_label,
                query_budget=args.query_budget,
                semantic_threshold=args.semantic_threshold,
                top_k_words=args.top_k_words,
                max_iterations=args.max_iterations,
                sim_score_window=args.sim_score_window,
                batch_size=args.batch_size,
                embed_func=embed_func,
                perturb_ratio=args.perturb_ratio,
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

        # 打印当前统计信息（仅针对有效样本）
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
    print("HQAAttack包装版攻击程序全部完成!")
    print("=" * 80)
    print(f"最终统计结果:")
    print(f"   总样本数: {len(data)}")
    print(f"   跳过样本数（模型预测错误）: {skipped_samples} ({skipped_samples / len(data) * 100:.1f}%)")
    print(f"   有效攻击样本数: {attacked_samples} ({attacked_samples / len(data) * 100:.1f}%)")
    print(
        f"   原始攻击成功: {raw_successful_attacks} ({raw_successful_attacks / attacked_samples * 100:.1f}%)" if attacked_samples > 0 else "   原始攻击成功: 0 (N/A)")
    print(
        f"   约束筛选后成功: {constraint_passed_attacks} ({constraint_passed_attacks / attacked_samples * 100:.1f}%)" if attacked_samples > 0 else "   约束筛选后成功: 0 (N/A)")
    print(
        f"   约束筛选通过率: {constraint_passed_attacks / raw_successful_attacks * 100:.1f}%" if raw_successful_attacks > 0 else "   约束筛选通过率: N/A")

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
def analyze_hqa_attack_results(results: Dict[str, Any]):
    """分析HQA攻击结果并输出详细报告"""
    print("\n" + "=" * 60)
    print("HQAAttack包装版攻击结果详细分析")
    print("=" * 60)

    print(f"攻击效果分析:")
    print(f"   总样本数: {results['total_samples']}")
    print(f"   成功攻击: {results['successful_attacks']}")
    print(f"   攻击成功率: {results['attack_success_rate']:.1%}")

    if results['successful_attacks'] > 0:
        print(f"\n质量指标:")
        print(f"   平均语义相似度: {results['avg_similarity']:.4f}")
        print(f"   平均扰动率: {results['avg_perturbation_rate']:.4f}")

        # 质量评估


    print(f"\n效率指标:")
    print(f"   平均查询次数: {results['avg_queries']:.1f}")
    print(f"   总运行时间: {results['total_time']:.2f}秒")
    print(f"   平均每样本时间: {results['total_time'] / results['total_samples']:.2f}秒")

    print("=" * 60)


# 在文件底部添加直接运行的入口
if __name__ == "__main__":
    # 运行HQAAttack包装版攻击
    try:
        results = run_hqaattack_wrapper()
        if results:
            analyze_hqa_attack_results(results)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()
