# tool.py - 独立的统一组件加载工具（完整版：已支持固定四个数据集自动切换/初始化）
# 四个数据集：MR / SST-2 / AG News / Yahoo Answers
# 你只需要改 GlobalComponentManager.config 里 target_model / target_model_path（或 ollama / lora 配置）
# 攻击脚本里按“每个数据集循环调用 initialize_global_components() / get_global_manager().use_dataset()”即可

import os
import random
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import tensorflow as tf
    import tensorflow_hub as hub  # noqa: F401
except ImportError:
    tf = None
    hub = None
from typing import List, Union, Optional, Dict, Any, Tuple
import tempfile
import shutil
import psutil
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 包内导入（副本在 cccc/third_party/caa，勿改 CAA 原仓库）
from . import dataloader_1
from . import criteria_1


def setup_unified_deterministic_environment(global_seed=12345, strict_mode=False):
    """
    统一的确定性环境设置函数
    供主函数和tool.py内部使用，确保环境设置完全一致

    Args:
        global_seed (int): 全局随机种子
        strict_mode (bool): 是否启用严格确定性模式
                           False: USE兼容模式（推荐）
                           True: 最大确定性模式（可能与USE不兼容）
    """
    print(f"=== 统一确定性环境设置 ===")
    print(f"全局种子: {global_seed}")
    print(f"模式: {'严格确定性' if strict_mode else 'USE兼容'}")

    # 1. 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(global_seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # 根据模式设置TensorFlow确定性
    if strict_mode:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        print("⚠️ 严格模式：可能与USE不兼容")
    else:
        os.environ['TF_DETERMINISTIC_OPS'] = '0'
        print("✓ USE兼容模式：平衡确定性和兼容性")

    # 2. 设置Python随机种子
    random.seed(global_seed)
    np.random.seed(global_seed)
    print(f"✓ Python随机种子设置: {global_seed}")

    # 3. 设置PyTorch
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.cuda.manual_seed_all(global_seed)

    try:
        if strict_mode:
            torch.use_deterministic_algorithms(True)
            print("✓ PyTorch严格确定性算法已启用")
        else:
            torch.use_deterministic_algorithms(False)
            print("✓ PyTorch兼容性模式（禁用严格确定性）")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✓ CUDNN确定性已启用")

    except RuntimeError as e:
        if "CUBLAS_WORKSPACE_CONFIG" in str(e):
            print("⚠️ 检测到CuBLAS确定性问题，降级处理...")
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("✓ 已降级到兼容性确定性模式")
        else:
            print(f"⚠️ PyTorch确定性设置警告: {e}")
    except Exception as e:
        print(f"⚠️ PyTorch设置异常: {e}")

    # 4–5. TensorFlow（可选；未安装则跳过，供 CCCC 仅注入 victim 时使用）
    if tf is not None:
        try:
            tf.random.set_seed(global_seed)
            print(f"✓ TensorFlow种子设置: {global_seed}")

            tf_version = tf.__version__
            print(f"检测到TensorFlow版本: {tf_version}")

            if strict_mode:
                if hasattr(tf.config.experimental, 'enable_op_determinism'):
                    try:
                        tf.config.experimental.enable_op_determinism()
                        print("✓ TensorFlow 2.8+严格确定性已启用")
                    except Exception as det_error:
                        print(f"⚠️ TF严格确定性失败: {det_error}")
                        print("⚠️ 降级到基础确定性设置")
                        tf.config.threading.set_inter_op_parallelism_threads(1)
                        tf.config.threading.set_intra_op_parallelism_threads(1)
                else:
                    print("⚠️ 旧版TensorFlow，使用基础严格设置")
                    tf.config.threading.set_inter_op_parallelism_threads(1)
                    tf.config.threading.set_intra_op_parallelism_threads(1)
            else:
                print("✓ TensorFlow USE兼容模式")
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)

        except Exception as e:
            print(f"⚠️ TensorFlow设置失败: {e}")
            print("⚠️ 将继续运行，但TensorFlow操作可能不完全确定性")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                with tf.device('/device:GPU:0'):
                    _ = tf.constant(0)
                print(f"✓ GPU设备配置完成: {len(gpus)} 个GPU")
                print(f"   主要GPU: {gpus[0]}")
            except RuntimeError as e:
                print(f"⚠️ GPU配置失败: {e}")
        else:
            print("ℹ️ TensorFlow 未检测到 GPU 设备（TF 视角）")
    else:
        print("ℹ️ 未安装 TensorFlow，跳过 TF 相关设置（CCCC 注入模式可正常使用）")

    print("=== 环境设置完成 ===")
    return True


def check_environment_consistency():
    """检查当前环境设置的一致性"""
    print("\n=== 环境一致性检查 ===")

    # 检查环境变量
    env_vars = [
        'PYTHONHASHSEED',
        'TF_DETERMINISTIC_OPS',
        'TF_CUDNN_DETERMINISTIC',
        'CUBLAS_WORKSPACE_CONFIG'
    ]

    print("环境变量设置:")
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        print(f"  {var}: {value}")

    # 检查随机状态
    print(f"\n当前随机状态:")
    print(f"  Python random state: {random.getstate()[1][0]}")
    print(f"  NumPy random state: {np.random.get_state()[1][0]}")

    # 检查PyTorch设置
    print(f"\nPyTorch设置:")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    print(f"  CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  CUDNN benchmark: {torch.backends.cudnn.benchmark}")

    # 检查TensorFlow设置
    print(f"\nTensorFlow设置:")
    if tf is None:
        print("  未安装 TensorFlow")
    else:
        try:
            print(f"  版本: {tf.__version__}")
            print(f"  GPU设备数: {len(tf.config.list_physical_devices('GPU'))}")
        except Exception as e:
            print(f"  TensorFlow检查失败: {e}")

    print("=== 检查完成 ===\n")


class MemoryManager:
    """内存和临时文件管理器"""

    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []

    def create_temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def create_temp_file(self, suffix="", dir=None):
        """创建临时文件"""
        temp_file = tempfile.mktemp(suffix=suffix, dir=dir)
        self.temp_files.append(temp_file)
        return temp_file

    def clear_tf_memory(self):
        """清理TensorFlow内存"""
        if tf is None:
            return
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception as e:
            print(f"TensorFlow内存清理警告: {e}")

    def clear_torch_memory(self):
        """清理PyTorch内存"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            print(f"PyTorch内存清理警告: {e}")

    def clear_temp_files(self):
        """清理所有临时文件"""
        for temp_file in self.temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                self.temp_files.remove(temp_file)
            except Exception as e:
                print(f"删除临时文件失败 {temp_file}: {e}")

        for temp_dir in self.temp_dirs[:]:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                self.temp_dirs.remove(temp_dir)
            except Exception as e:
                print(f"删除临时目录失败 {temp_dir}: {e}")

    def get_memory_usage(self):
        """获取当前内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }

    def clear_all_memory(self):
        """清理所有内存和临时文件"""
        self.clear_tf_memory()
        self.clear_torch_memory()
        self.clear_temp_files()
        _ = gc.collect()

    def __del__(self):
        """析构函数，确保清理资源"""
        self.clear_temp_files()


# 全局内存管理器实例
memory_manager = MemoryManager()


class USE(object):
    """Universal Sentence Encoder语义相似度计算模块 - 现已升级为Sentence Transformers"""

    def __init__(self, cache_path):
        super(USE, self).__init__()
        print("正在初始化语义相似度模块...")
        print(f"模型路径（已忽略，使用现代模型）: {cache_path}")
        start_time = time.time()

        from sentence_transformers import SentenceTransformer
        self.embed = SentenceTransformer("all-mpnet-base-v2")

        print(f"Sentence Transformers模型加载完成，耗时: {time.time() - start_time:.2f}秒")

        # 内存优化设置
        self.batch_size_limit = 32
        self.memory_threshold = 85

    def semantic_sim(self, sents1, sents2):
        """优化后的语义相似度计算，支持批处理和内存管理"""
        memory_info = memory_manager.get_memory_usage()
        if memory_info['percent'] > self.memory_threshold:
            print(f"内存使用率过高 ({memory_info['percent']:.1f}%)，执行内存清理...")
            memory_manager.clear_all_memory()

        total_pairs = len(sents2)

        if total_pairs <= self.batch_size_limit:
            result = self._compute_similarity_batch(sents1, sents2)
            return [result]

        all_similarities = []

        for i in range(0, total_pairs, self.batch_size_limit):
            batch_end = min(i + self.batch_size_limit, total_pairs)
            batch_sents2 = sents2[i:batch_end]

            batch_similarities = self._compute_similarity_batch(sents1, batch_sents2)
            all_similarities.extend(batch_similarities)

            if (i // self.batch_size_limit + 1) % 3 == 0:
                memory_manager.clear_torch_memory()

        return [np.array(all_similarities)]

    def _compute_similarity_batch(self, sents1, sents2):
        """计算单个批次的语义相似度 - 使用Sentence Transformers"""
        try:
            all_texts = sents1 + sents2
            embeddings = self.embed.encode(
                all_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            embed1 = embeddings[:len(sents1)]
            embed2 = embeddings[len(sents1):]

            embed1 = torch.nn.functional.normalize(embed1, p=2, dim=1)
            embed2 = torch.nn.functional.normalize(embed2, p=2, dim=1)

            cosine_similarities = torch.sum(embed1 * embed2, dim=1)
            sim_scores = torch.clamp(cosine_similarities, -1.0, 1.0)
            result = sim_scores.detach().cpu().numpy()

            del embeddings, embed1, embed2, cosine_similarities, sim_scores
            torch.cuda.empty_cache()

            return result

        except Exception as e:
            print(f"语义相似度计算错误: {e}")
            memory_manager.clear_torch_memory()
            raise e


class NLI_infer_BERT_Enhanced(nn.Module):
    """增强版BERT模型，使用transformers标准接口"""

    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32):
        super(NLI_infer_BERT_Enhanced, self).__init__()
        print(f"正在初始化增强版BERT模型...")
        print(f"   模型路径: {pretrained_dir}")
        print(f"   类别数量: {nclasses}")
        print(f"   最大序列长度: {max_seq_length}")

        start_time = time.time()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

        print(f"BERT模型加载完成，耗时: {time.time() - start_time:.2f}秒")

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        print("增强版BERT模型初始化完成!")

    def text_pred(self, text_data, batch_size=32):
        """使用transformers标准方式进行文本预测"""
        self.model.eval()

        texts = []
        for text in text_data:
            if isinstance(text, list):
                text_str = ' '.join(text)
            else:
                text_str = str(text)
            texts.append(text_str)

        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            if (i // batch_size + 1) % 5 == 0:
                torch.cuda.empty_cache()

            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )

            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.detach().cpu())

                del inputs, outputs, logits
                if (i // batch_size + 1) % 3 == 0:
                    torch.cuda.empty_cache()

        result = torch.cat(all_probs, dim=0).cuda()

        del all_probs, texts
        torch.cuda.empty_cache()

        return result


def smart_memory_cleanup(threshold=75):
    """智能内存清理函数，只在内存使用率超过阈值时清理"""
    memory_info = memory_manager.get_memory_usage()
    if memory_info['percent'] > threshold:
        print(f"内存使用率达到 {memory_info['percent']:.1f}%，执行清理...")
        memory_info_before = memory_info

        memory_manager.clear_all_memory()

        memory_info_after = memory_manager.get_memory_usage()
        memory_saved = memory_info_before['rss'] - memory_info_after['rss']

        if memory_saved > 0:
            print(f"释放内存: {memory_saved:.1f}MB")
        print(f"清理后内存使用: {memory_info_after['rss']:.1f}MB ({memory_info_after['percent']:.1f}%)")
        return True
    return False


def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):
    """优化后的相似度计算函数，添加内存管理"""
    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_range_min = 0
        text_range_max = len_text

    try:
        reference_text = ' '.join(text_ls[text_range_min:text_range_max])
        candidate_texts = []
        for new_text in new_texts:
            candidate_text = ' '.join(new_text[text_range_min:text_range_max])
            candidate_texts.append(candidate_text)

        semantic_sims = sim_predictor.semantic_sim([reference_text], candidate_texts)[0]

        if not isinstance(semantic_sims, (list, np.ndarray)):
            return [semantic_sims]
        else:
            return semantic_sims

    except Exception as e:
        print(f"calc_sim错误: {e}")
        memory_manager.clear_all_memory()
        raise e


# ===========================
# 统一 Prompt（你训练用的那套）
# ===========================
PROMPT_2CLS = (
    "You are a sentiment classifier.\n"
    "Task: determine whether the following sentence is negative or positive.\n"
    "Output: ONLY one character, 0 for negative or 1 for positive.\n"
    "Sentence: {text}\n"
    "Answer:"
)

PROMPT_YAHOO = (
    "You are a strict Yahoo Answers topic classifier. Return ONLY one digit 0-9 with NO other text.\n"
    "Mapping: 0=Society & Culture, 1=Science & Mathematics, 2=Health, 3=Education & Reference, 4=Computers & Internet, "
    "5=Sports, 6=Business & Finance, 7=Entertainment & Music, 8=Family & Relationships, 9=Politics & Government.\n"
    "Examples:\n"
    'Q: "Would my girlfriend break up with me if she found out Im atheist ?  Shes christian but not that religious if she loves you she woudlnt" -> 0\n'
    'Q: "Whats an easy way to visualize the 4th dimension ?  From what I remember in geometry class the 4th dimension was simply time .   And just like the other three length width and height it can be measured in various forms  seconds minutes hours etc .." -> 1\n'
    'Q: "when Im trying too lose 30lbs which diet should i go with ?  exercise is a must .  you cant drop that kind of weight quickly by diet without harming yourself . eat 6 very small meals a day .  this will get your metabolism up" -> 2\n'
    'Q: "how long does it take to get from new york to california ?  depends how you are going" -> 3\n'
    'Q: "how do i make an mp3 to an http mp3 ?  to put music on my site it needs to have http at beg abd mp3 at end .?" -> 4\n'
    'Q: "Who you think will win USA vs GHANA ?  Usa vs Ghana no offense but i predict a 20 win by usa or a 21 win by usa .  damn USA must win" -> 5\n'
    'Q: "how to make more money ?  Online  ?  Some easy money  ?" -> 6\n'
    'Q: "if chickens are comes from eggsand eggs lays by chickens then who came first in the world ?" -> 7\n'
    'Q: "is anybody here personal with each other ?  well yesmy fiancee is on here ." -> 8\n'
    'Q: "why china is usa bug boo  ?  were you high when you asked this question ?" -> 9\n'
    "Text: {text}\n"
    "Answer:"
)

PROMPT_AGNEWS = (
    "You are a strict AG News topic classifier. Return ONLY one digit 0-3 with NO other text.\n"
    "Mapping: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.\n"
    "Text: {text}\n"
    "Answer:"
)


def _infer_nclasses_from_dataset_path(dataset_path: str, default_nclasses: int) -> int:
    p = (dataset_path or "").lower()
    if "ag" in p:
        return 4
    if "yahoo" in p:
        return 10
    return int(default_nclasses)


def _select_prompt_by_dataset(dataset_path: str, nclasses: int) -> str:
    p = (dataset_path or "").lower()
    if "yahoo" in p or int(nclasses) == 10:
        return PROMPT_YAHOO
    if "ag" in p or int(nclasses) == 4:
        return PROMPT_AGNEWS
    return PROMPT_2CLS


def _to_text_str(x) -> str:
    if isinstance(x, (list, tuple)):
        return " ".join([str(t) for t in x])
    return str(x)


def _onehot_probs(labels: List[int], nclasses: int) -> np.ndarray:
    out = np.zeros((len(labels), int(nclasses)), dtype=float)
    for i, y in enumerate(labels):
        y = int(y)
        y = max(0, min(int(nclasses) - 1, y))
        out[i, y] = 1.0
    return out


# ===========================
# Loader 1：WordLSTM（自动定位 train_classifier.py）
# ===========================
class WordLSTMVictim:
    def __init__(self, word_embeddings_path: str, ckpt_path_piece: str, dataset_path: str, nclasses: int):
        import os, sys, types
        import torch

        # ===== 0) 让 Python 能找到 Limeattack/train_classifier.py =====
        here = os.path.dirname(os.path.abspath(__file__))
        lime_dir = os.path.join(here, "Limeattack")
        if lime_dir not in sys.path:
            sys.path.insert(0, lime_dir)

        # ===== 1) BERT shim：只为了解决 train_classifier.py 顶层 import BERT.* =====
        if "BERT" not in sys.modules:
            try:
                import BERT  # 如果你真有 BERT 包，就用你的
            except Exception:
                from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification
                bert_pkg = types.ModuleType("BERT")

                tokenization_mod = types.ModuleType("BERT.tokenization")
                tokenization_mod.BertTokenizer = BertTokenizer

                modeling_mod = types.ModuleType("BERT.modeling")
                modeling_mod.BertConfig = BertConfig
                modeling_mod.BertModel = BertModel
                modeling_mod.BertForSequenceClassification = BertForSequenceClassification

                sys.modules["BERT"] = bert_pkg
                sys.modules["BERT.tokenization"] = tokenization_mod
                sys.modules["BERT.modeling"] = modeling_mod

        # ===== 2) 完全按你原 loader 的路径逻辑拼 default_model_path =====
        default_model_path = "./model/wordLSTM"
        p = (dataset_path or "").lower()

        # 你原代码里这里一般会根据 dataset 决定 nclasses（我保留一致口径）
        if "ag" in p:
            nclasses = 4
            default_model_path = os.path.join(default_model_path, ckpt_path_piece)
        elif "yahoo" in p:
            nclasses = 10
            default_model_path = os.path.join(default_model_path, ckpt_path_piece)
        else:
            default_model_path = os.path.join(default_model_path, ckpt_path_piece)

        # ===== 3) 完全按你原 loader：Model + torch.load + load_state_dict =====
        from train_classifier import Model
        self.model = Model(word_embeddings_path, nclasses=nclasses).cuda()
        checkpoint = torch.load(default_model_path, map_location="cuda:0")
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.nclasses = int(nclasses)

    @torch.no_grad()
    def text_pred(self, text_data, batch_size=32):
        # 这里也保持你仓库口径：用 Model.text_pred
        return self.model.text_pred(text_data, batch_size=batch_size)

# ===========================
# Loader 1.5：WordCNN / TextCNN
# 约定：ckpt 放在 ./model/wordCNN/<piece>
# 使用 train_classifier.py 的 Model 类（与 WordLSTM 一致）
# ===========================
class WordCNNVictim:
    def __init__(self, word_embeddings_path: str, ckpt_path_piece: str, dataset_path: str, nclasses: int):
        import os, sys, types
        import torch

        # ===== 0) 让 Python 能找到 Limeattack/train_classifier.py =====
        here = os.path.dirname(os.path.abspath(__file__))
        lime_dir = os.path.join(here, "Limeattack")
        if lime_dir not in sys.path:
            sys.path.insert(0, lime_dir)

        # ===== 1) BERT shim：只为了解决 train_classifier.py 顶层 import BERT.* =====
        if "BERT" not in sys.modules:
            try:
                import BERT  # 如果你真有 BERT 包，就用你的
            except Exception:
                from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification
                bert_pkg = types.ModuleType("BERT")

                tokenization_mod = types.ModuleType("BERT.tokenization")
                tokenization_mod.BertTokenizer = BertTokenizer

                modeling_mod = types.ModuleType("BERT.modeling")
                modeling_mod.BertConfig = BertConfig
                modeling_mod.BertModel = BertModel
                modeling_mod.BertForSequenceClassification = BertForSequenceClassification

                sys.modules["BERT"] = bert_pkg
                sys.modules["BERT.tokenization"] = tokenization_mod
                sys.modules["BERT.modeling"] = modeling_mod

        # ===== 2) 完全按你原 loader 的路径逻辑拼 default_model_path =====
        default_model_path = "./model/wordCNN"
        p = (dataset_path or "").lower()

        # 你原代码里这里一般会根据 dataset 决定 nclasses（我保留一致口径）
        if "ag" in p:
            nclasses = 4
            default_model_path = os.path.join(default_model_path, ckpt_path_piece)
        elif "yahoo" in p:
            nclasses = 10
            default_model_path = os.path.join(default_model_path, ckpt_path_piece)
        else:
            default_model_path = os.path.join(default_model_path, ckpt_path_piece)

        # ===== 3) 完全按你原 loader：Model + torch.load + load_state_dict =====
        # checkpoint 是直接的 state_dict（不是嵌套字典）
        checkpoint = torch.load(default_model_path, map_location='cuda:0')
        
        # 先尝试 hidden_size=150，失败再尝试 100（完全按你参考代码）
        from train_classifier import Model
        try:
            self.model = Model(word_embeddings_path, nclasses=nclasses, hidden_size=150, cnn=True).cuda()
            self.model.load_state_dict(checkpoint)
        except:
            self.model = Model(word_embeddings_path, nclasses=nclasses, hidden_size=100, cnn=True).cuda()
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()

        self.nclasses = int(nclasses)

    @torch.no_grad()
    def text_pred(self, text_data, batch_size=32):
        # 这里也保持你仓库口径：用 Model.text_pred
        return self.model.text_pred(text_data, batch_size=batch_size)

# ===========================
# Loader 2：Ollama API victim（one-hot probs）
# ===========================
class OllamaVictim:
    def __init__(self, model: str, dataset_path: str, nclasses: int,
                 url: str = "http://localhost:11434/api/generate",
                 max_new_tokens: int = 2):
        import httpx
        self.httpx = httpx

        self.url = url
        self.model = model
        self.nclasses = int(nclasses)
        self.prompt_tmpl = _select_prompt_by_dataset(dataset_path, self.nclasses)
        self.max_new_tokens = int(max_new_tokens)

    def _call(self, prompt: str) -> str:
        r = self.httpx.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=600.0
        )
        r.raise_for_status()
        j = r.json()
        return str(j.get("response", ""))

    def predict(self, text_data) -> int:
        import re
        text = _to_text_str(text_data)
        prompt = self.prompt_tmpl.format(text=text)
        out = self._call(prompt)
        digits = re.findall(r"\d", out)
        if not digits:
            return 0
        y = int(digits[-1])
        return max(0, min(self.nclasses - 1, y))

    def text_pred(self, text_data, batch_size=32) -> torch.Tensor:
        xs = text_data
        labels = [self.predict(x) for x in xs]
        probs = _onehot_probs(labels, self.nclasses)
        return torch.tensor(probs, dtype=torch.float32)

    def lime_pred(self, texts) -> np.ndarray:
        xs = texts if isinstance(texts, (list, tuple)) else [texts]
        labels = [self.predict(x) for x in xs]
        return _onehot_probs(labels, self.nclasses)


# ===========================
# Loader 3：LoRA/QLoRA 本地 LLM（one-hot probs）
# ===========================
class LocalLoRALLMVictim:
    def __init__(self, base: str, adapter_dir: str, dataset_path: str, nclasses: int,
                 bnb_dtype: str = "fp16", max_new_tokens: int = 2):
        import re
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        self.re = re
        self.nclasses = int(nclasses)
        self.prompt_tmpl = _select_prompt_by_dataset(dataset_path, self.nclasses)
        self.max_new_tokens = int(max_new_tokens)

        if bnb_dtype == "bf16":
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            device_map="auto",
            quantization_config=bnb,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()

    @torch.inference_mode()
    def _gen(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        s = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return s

    def predict(self, text_data) -> int:
        text = _to_text_str(text_data)
        prompt = self.prompt_tmpl.format(text=text)
        s = self._gen(prompt)
        digits = self.re.findall(r"\d", s.split("Answer:")[-1])
        if not digits:
            return 0
        y = int(digits[-1])
        return max(0, min(self.nclasses - 1, y))

    def text_pred(self, text_data, batch_size=32) -> torch.Tensor:
        xs = text_data
        labels = [self.predict(x) for x in xs]
        probs = _onehot_probs(labels, self.nclasses)
        return torch.tensor(probs, dtype=torch.float32)

    def lime_pred(self, texts) -> np.ndarray:
        xs = texts if isinstance(texts, (list, tuple)) else [texts]
        labels = [self.predict(x) for x in xs]
        return _onehot_probs(labels, self.nclasses)


class GlobalComponentManager:
    """全局组件管理器 - 统一加载和管理所有攻击算法需要的组件"""

    def __init__(self, global_seed: int = 2025):
        self.global_seed = global_seed
        self.is_initialized = False

        self.components = {
            'model': None,
            'predictor': None,
            'word2idx': None,
            'idx2word': None,
            'cos_sim': None,
            'sim_predictor': None,
            'stop_words_set': None,
            'dataset': None,
            'embed_func': None,
            'lime_pred': None
        }

        # 配置信息（你只需要换这里的 target_model / target_model_path）
        self.config = {
            # 默认先给 MR
            'dataset_path': './data/sst2',
            'nclasses': 2,

            # distilbert / wordLSTM / ollama / lora_llm
            'target_model': 'distilbert',

            # 统一用 target_model_path 表示“当前 target 的主路径”
            # - distilbert: HF 序列分类模型目录
            # - wordLSTM:   checkpoint 文件名/相对片段（会拼到 ./model/wordLSTM/ 后面）
            'target_model_path': "./model/distilbert/sst2",  # ✅ 建议填 "mr" / "sst2" / "ag" / "yahoo" 对应你 ckpt 命名

            'word_embeddings_path': "./glove.6B.200d.txt",

            'counter_fitting_embeddings_path': "./counter-fitted-vectors.txt",
            'counter_fitting_cos_sim_path': './cos_sim_matrix.pkl',

            'USE_cache_path': "./all-mpnet-base-v2",
            'max_seq_length': 128,
            'batch_size': 16,
            'data_size': 100,

            # ---- ollama victim ----
            'ollama_url': "http://localhost:11434/api/generate",
            'ollama_model': "qwen2.5:1.5b-instruct-q4_K_M",
            # 'ollama_model': "qwen2.5:1.5b-instruct-q4_K_M",

            # ---- lora llm victim ----
            'llm_base': "Qwen/Qwen2.5-1.5B-Instruct",
            'llm_adapter_dir': "./adapters/mr",
            'bnb_dtype': "fp16",
            'llm_max_new_tokens': 2,
        }

        # ===========================
        # 固定四个数据集注册表（你要求的：MR/SST2/AG/Yahoo）
        # 你只需要保证 dataloader_1.read_corpus 认得这些路径
        # ===========================
        self.dataset_registry = {
            "mr": {
                "dataset_path": "./data/mr",
                "nclasses": 2,
            },
            "sst2": {
                "dataset_path": "./data/sst2",
                "nclasses": 2,
            },
            "agnews": {
                "dataset_path": "./data/ag",      # 如果你目录叫 ./data/agnews，就改成 "./data/agnews"
                "nclasses": 4,
            },
            "yahoo": {
                "dataset_path": "./data/yahoo",
                "nclasses": 10,
            },
            "yelp": {
                "dataset_path": "./data/yelp",
                "nclasses": 2,
            },
        }

        print(f"全局组件管理器初始化完成，种子: {self.global_seed}")

    def setup_deterministic_environment(self):
        print("使用统一环境设置函数...")
        return setup_unified_deterministic_environment(
            global_seed=self.global_seed,
            strict_mode=False
        )

    # ===========================
    # ✅ 新增：切换数据集（切换后强制失效缓存）
    # ===========================
    def use_dataset(self, dataset_name: str):
        name = str(dataset_name).lower()
        if name not in self.dataset_registry:
            raise ValueError(f"未知数据集: {dataset_name}，可选: {list(self.dataset_registry.keys())}")

        cfg = self.dataset_registry[name]
        self.update_config(**cfg)

        # 切数据集必须让组件失效，否则会沿用旧 dataset/model 的缓存
        self.is_initialized = False
        for k in self.components:
            self.components[k] = None

        print(f"\n✅ 已切换数据集: {name} -> {cfg['dataset_path']} (nclasses={cfg['nclasses']})")

    def run_on_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        ✅ 新增：按固定四个数据集依次初始化一套组件
        返回 dict[name] = components
        """
        out = {}
        for name in self.dataset_registry.keys():
            self.use_dataset(name)
            out[name] = self.initialize_all_components()
        return out

    def load_dataset(self) -> List[Tuple]:
        print("正在加载攻击数据集...")
        start_time = time.time()

        texts, labels = dataloader_1.read_corpus(
            self.config['dataset_path'],
            max_lines=10000,
            csvf=False
        )
        data = list(zip(texts, labels))
        data = data[:self.config['data_size']]

        formatted_data = [({'text': text}, label) for text, label in data]

        print(f"数据集加载完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"   样本数量: {len(data)}")
        if len(data) > 0:
            print(f"   示例文本: {' '.join(data[0][0])}")
            print(f"   示例标签: {data[0][1]}")

        return formatted_data

    def load_bert_model(self):
        print("正在构建BERT目标模型...")
        start_time = time.time()

        model = NLI_infer_BERT_Enhanced(
            self.config['target_model_path'],
            nclasses=self.config['nclasses'],
            max_seq_length=self.config['max_seq_length'],
            batch_size=self.config['batch_size']
        )

        def lime_pred_wrapper(texts):
            if isinstance(texts, str):
                texts = [texts]

            processed_texts = []
            for text in texts:
                if isinstance(text, str):
                    processed_texts.append(text.split())
                else:
                    processed_texts.append(text)

            probs = model.text_pred(processed_texts)
            return probs.detach().cpu().numpy()

        print(f"BERT模型构建完成，耗时: {time.time() - start_time:.2f}秒")
        return model, model.text_pred, lime_pred_wrapper

    def load_target_model(self):
        tm = str(self.config.get('target_model', 'distilbert')).lower()
        dataset_path = str(self.config.get('dataset_path', ''))
        nclasses = _infer_nclasses_from_dataset_path(dataset_path, int(self.config.get('nclasses', 2)))
        self.config['nclasses'] = int(nclasses)

        if tm in ['distilbert', 'bert', 'hf_bert', 'hf_cls']:
            return self.load_bert_model()

        if tm in ['wordlstm', 'word_lstm', 'lstm']:
            base_dir = "./model/wordLSTM"
            model_path_piece = str(self.config.get('target_model_path', '')).strip()
            ckpt_path = os.path.join(base_dir, model_path_piece) if model_path_piece else base_dir

            victim = WordLSTMVictim(
                word_embeddings_path=str(self.config.get('word_embeddings_path', './glove.6B.50d.txt')),
                ckpt_path_piece=str(self.config.get('target_model_path', 'mr')).strip(),  # 这里就是你 args.model_path
                dataset_path=str(self.config.get('dataset_path', './data/mr')),
                nclasses=int(self.config.get('nclasses', 2)),
            )

            def predictor(text_data, batch_size=32):
                return victim.text_pred(text_data, batch_size=batch_size)

            def lime_pred(texts):
                if isinstance(texts, str):
                    texts = [texts]
                processed = [t.split() if isinstance(t, str) else t for t in texts]
                probs = victim.text_pred(processed, batch_size=int(self.config.get('batch_size', 32)))
                return probs.detach().cpu().numpy()

            return victim.model, predictor, lime_pred

        if tm in ['cnn', 'textcnn', 'wordcnn', 'text_cnn']:
            victim = WordCNNVictim(
                word_embeddings_path=str(self.config.get('word_embeddings_path', './glove.6B.200d.txt')),
                ckpt_path_piece=str(self.config.get('target_model_path', 'mr')).strip(),
                dataset_path=dataset_path,
                nclasses=nclasses,
            )

            def predictor(text_data, batch_size=32):
                return victim.text_pred(text_data, batch_size=batch_size)

            def lime_pred(texts):
                if isinstance(texts, str):
                    texts = [texts]
                processed = [t.split() if isinstance(t, str) else t for t in texts]
                probs = victim.text_pred(processed, batch_size=int(self.config.get('batch_size', 32)))
                return probs.detach().cpu().numpy()

            return victim.model, predictor, lime_pred

        if tm in ['ollama', 'ollama_api']:
            victim = OllamaVictim(
                model=str(self.config.get('ollama_model', 'qwen2.5:1.5b')),
                dataset_path=dataset_path,
                nclasses=nclasses,
                url=str(self.config.get('ollama_url', "http://localhost:11434/api/generate")),
                max_new_tokens=int(self.config.get('llm_max_new_tokens', 2)),
            )

            def predictor(text_data, batch_size=32):
                return victim.text_pred(text_data, batch_size=batch_size)

            def lime_pred(texts):
                return victim.lime_pred(texts)

            return None, predictor, lime_pred

        if tm in ['lora_llm', 'peft', 'qlora', 'lora']:
            victim = LocalLoRALLMVictim(
                base=str(self.config.get('llm_base', "Qwen/Qwen2.5-1.5B-Instruct")),
                adapter_dir=str(self.config.get('llm_adapter_dir', "./adapters/mr")),
                dataset_path=dataset_path,
                nclasses=nclasses,
                bnb_dtype=str(self.config.get('bnb_dtype', "fp16")),
                max_new_tokens=int(self.config.get('llm_max_new_tokens', 2)),
            )

            def predictor(text_data, batch_size=32):
                return victim.text_pred(text_data, batch_size=batch_size)

            def lime_pred(texts):
                return victim.lime_pred(texts)

            return None, predictor, lime_pred

        raise ValueError(f"未知 target_model: {self.config.get('target_model')}")

    def load_vocabulary(self) -> Tuple[Dict, Dict]:
        print("正在构建词汇表...")
        start_time = time.time()

        idx2word = {}
        word2idx = {}

        with open(self.config['counter_fitting_embeddings_path'], 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in idx2word:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1

        print(f"词汇表构建完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"   词汇表大小: {len(word2idx)}")

        return word2idx, idx2word

    def load_cosine_similarity_matrix(self):
        print("正在加载余弦相似度矩阵...")
        start_time = time.time()

        if os.path.exists(self.config['counter_fitting_cos_sim_path']):
            with open(self.config['counter_fitting_cos_sim_path'], "rb") as fp:
                cos_sim = pickle.load(fp)

            print(f"余弦相似度矩阵加载完成，耗时: {time.time() - start_time:.2f}秒")

            if isinstance(cos_sim, dict):
                print(f"   数据格式: Top-K字典")
                print(f"   词汇数量: {len(cos_sim)}")
                if cos_sim:
                    sample_key = next(iter(cos_sim.keys()))
                    sample_value = cos_sim[sample_key]
                    print(f"   每词相似词数: {len(sample_value) if sample_value else 0}")
            elif isinstance(cos_sim, np.ndarray):
                print(f"   数据格式: 完整矩阵")
                print(f"   矩阵维度: {cos_sim.shape}")

            return cos_sim
        else:
            print("警告：未找到预计算的余弦相似度矩阵")
            return None

    def load_use_model(self):
        print("正在初始化USE语义相似度模块...")
        start_time = time.time()

        use = USE(self.config['USE_cache_path'])

        print(f"USE模块初始化完成，耗时: {time.time() - start_time:.2f}秒")
        return use

    def load_stopwords(self):
        print("正在加载停用词...")
        stop_words_set = criteria_1.get_stopwords()
        print(f"停用词加载完成，共 {len(stop_words_set)} 个")
        return stop_words_set

    def initialize_all_components(self):
        if self.is_initialized:
            print("组件已经初始化，跳过重复初始化")
            return self.components

        print("\n" + "=" * 60)
        print("开始初始化全局组件")
        print("=" * 60)

        self.setup_deterministic_environment()

        print("\n1. 加载数据集...")
        self.components['dataset'] = self.load_dataset()

        print("\n2. 加载目标/受害模型...")
        model, predictor, lime_pred = self.load_target_model()
        self.components['model'] = model
        self.components['predictor'] = predictor
        self.components['lime_pred'] = lime_pred

        print("\n3. 构建词汇表...")
        word2idx, idx2word = self.load_vocabulary()
        self.components['word2idx'] = word2idx
        self.components['idx2word'] = idx2word
        self.components['embed_func'] = self.config['counter_fitting_embeddings_path']

        print("\n4. 加载余弦相似度矩阵...")
        self.components['cos_sim'] = self.load_cosine_similarity_matrix()

        print("\n5. 初始化USE模块...")
        self.components['sim_predictor'] = self.load_use_model()
        # self.components['sim_predictor'] = None

        print("\n6. 加载停用词...")
        self.components['stop_words_set'] = self.load_stopwords()

        self.is_initialized = True

        print("\n" + "=" * 60)
        print("全局组件初始化完成！")
        print("=" * 60)

        return self.components

    def initialize_minimal_cccc(
        self,
        *,
        predictor,
        word2idx,
        idx2word,
        cos_sim,
        embed_func: str,
        sim_predictor=None,
        stop_words_set=None,
    ):
        """
        CCCC 专用：不加载 CAA 完整数据集与 config 中的 HF 受害模型，
        仅注入 HQAAttack / TextHacker 所需的 predictor、词表与 cos_sim。
        """
        self.setup_deterministic_environment()
        sw = stop_words_set if stop_words_set is not None else criteria_1.get_stopwords()
        self.components["predictor"] = predictor
        self.components["model"] = None
        self.components["lime_pred"] = None
        self.components["dataset"] = []
        self.components["word2idx"] = word2idx
        self.components["idx2word"] = idx2word
        self.components["cos_sim"] = cos_sim
        self.components["embed_func"] = embed_func
        self.components["sim_predictor"] = sim_predictor
        self.components["stop_words_set"] = sw
        self.is_initialized = True
        print("[CCCC] initialize_minimal_cccc: 已注入 predictor / vocab / cos_sim / sim_predictor")
        return self.components

    def get_component(self, component_name: str):
        if not self.is_initialized:
            self.initialize_all_components()

        if component_name in self.components:
            return self.components[component_name]
        else:
            raise ValueError(f"未知组件: {component_name}")

    def get_all_components(self) -> Dict[str, Any]:
        if not self.is_initialized:
            self.initialize_all_components()
        return self.components.copy()

    def calc_sim(self, text_ls, new_texts, idx, sim_score_window, sim_predictor):
        return calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor)

    def semantic_sim_single(self, text1: str, text2: str) -> float:
        if not self.is_initialized:
            self.initialize_all_components()

        sim_predictor = self.components['sim_predictor']
        if sim_predictor is None:
            return 0.8

        try:
            result = sim_predictor.semantic_sim([text1], [text2])

            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], (list, np.ndarray)) and len(result[0]) > 0:
                    return float(result[0][0])
                else:
                    return float(result[0])
            elif isinstance(result, (float, int)):
                return float(result)
            else:
                return 0.8

        except Exception as e:
            print(f"语义相似度计算失败: {e}")
            return 0.8

    def reset_random_seed(self, seed_offset: int = 0):
        seed = self.global_seed + seed_offset
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try:
            if tf is not None:
                tf.random.set_seed(seed)
        except Exception as e:
            print(f"TensorFlow种子重置失败: {e}")

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"配置更新: {key} = {value}")
            else:
                print(f"警告: 未知配置项 {key}")

    def memory_cleanup(self):
        try:
            memory_manager.clear_all_memory()
        except Exception as e:
            print(f"内存清理失败: {e}")

    def __del__(self):
        try:
            memory_manager.clear_all_memory()
        except:
            pass


# ===========================
# 全局单例实例 + 便捷接口
# ===========================
_global_manager = None


def reset_global_manager():
    """重置全局单例（切模型/想完全重来时调用一次）"""
    global _global_manager
    _global_manager = None


def get_global_manager(global_seed: int = 2025) -> GlobalComponentManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = GlobalComponentManager(global_seed)
    return _global_manager


def initialize_global_components(global_seed: int = 2025) -> Dict[str, Any]:
    manager = get_global_manager(global_seed)
    return manager.initialize_all_components()


def initialize_components_for_dataset(dataset_name: str, global_seed: int = 2025) -> Dict[str, Any]:
    """
    ✅ 新增：初始化“指定数据集”的组件（会切换数据集并初始化）
    """
    manager = get_global_manager(global_seed)
    manager.use_dataset(dataset_name)
    return manager.initialize_all_components()


def initialize_components_for_all_datasets(global_seed: int = 2025) -> Dict[str, Dict[str, Any]]:
    """
    ✅ 新增：一次性初始化四个数据集的组件
    返回 dict[name] = components
    """
    manager = get_global_manager(global_seed)
    return manager.run_on_all_datasets()


def get_component(component_name: str):
    manager = get_global_manager()
    return manager.get_component(component_name)


def initialize_cccc_injected_components(**kwargs):
    """重置全局管理器并以 CCCC 的 Baseline 检测器注入攻击所需组件。"""
    reset_global_manager()
    mgr = get_global_manager()
    return mgr.initialize_minimal_cccc(**kwargs)


def calc_sim_interface(text_ls: List[str],
                       new_texts: List[List[str]],
                       idx: int = -1,
                       sim_score_window: int = 15) -> List[float]:
    manager = get_global_manager()
    sim_predictor = manager.get_component('sim_predictor')
    return manager.calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor)


def semantic_sim(text1: str, text2: str) -> float:
    manager = get_global_manager()
    return manager.semantic_sim_single(text1, text2)


def reset_seed(seed_offset: int = 0):
    manager = get_global_manager()
    manager.reset_random_seed(seed_offset)


def cleanup_memory():
    manager = get_global_manager()
    manager.memory_cleanup()


def print_tool_info():
    print("=" * 60)
    print("GlobalComponentManager - 独立的统一组件加载工具")
    print("=" * 60)
    print("功能：")
    print("• 统一加载模型、数据集、词嵌入等组件")
    print("• 提供标准化的语义相似度计算接口")
    print("• 确保不同算法间的公平比较和可复现性")
    print("• 全局种子控制，保证结果一致性")
    print("• 已支持固定四个数据集：MR/SST2/AGNEWS/YAHOO")
    print("")
    print("主要接口：")
    print("• initialize_global_components(): 初始化当前 config 对应的一套组件")
    print("• initialize_components_for_dataset(name): 初始化指定数据集组件")
    print("• initialize_components_for_all_datasets(): 初始化四个数据集组件")
    print("• get_component(name): 获取指定组件")
    print("• calc_sim_interface(): 计算语义相似度")
    print("• semantic_sim(): 单对文本相似度")
    print("• reset_seed(): 重置随机种子")
    print("• cleanup_memory(): 清理内存")
    print("=" * 60)


def setup_environment(global_seed=2025, strict_mode=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return setup_unified_deterministic_environment(global_seed, strict_mode)


def reset_environment(global_seed=2025):
    print("重置环境设置...")
    return setup_unified_deterministic_environment(global_seed, strict_mode=False)
