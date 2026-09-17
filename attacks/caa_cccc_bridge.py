# -*- coding: utf-8 -*-
"""
将 third_party.caa 中的 HQAAttack / TextHacker 接到 CCCC 的 Baseline 文本检测器。

固定链 Step2/3 **始终**经本模块调 HQA / TextHacker（须传入 ``TextAIDetector``）；近义词默认走中文 MLM 候选建邻（无需 counter-fitted）。

环境变量：
  CCCC_CAA_SYNONYMS     chinese（默认）| legacy — legacy 需 counter-fitted + cos_sim pickle
  CCCC_CAA_EMBED / CCCC_CAA_COS — 仅 legacy
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import jieba
import torch

from attacks.greedy_mlm import GreedyMLMAttackResult
from detectors.text.detector import TextAIDetector

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EMBED = _ROOT / "third_party" / "caa" / "data" / "counter-fitted-vectors.txt"
_DEFAULT_COS = _ROOT / "third_party" / "caa" / "data" / "cos_sim_matrix.pkl"

_caa_legacy_ready = False
_chinese_temp_embed: Optional[str] = None


class DifflibWindowSim:
    """轻量窗口语义相似度，避免强制加载 sentence-transformers / TF。"""

    def semantic_sim(self, sents1, sents2):
        import numpy as np
        from difflib import SequenceMatcher

        ref = sents1[0] if sents1 else ""
        cands = sents2 if isinstance(sents2, (list, tuple)) else [sents2]
        out = []
        for c in cands:
            out.append(float(SequenceMatcher(None, ref, c).ratio()))
        return [np.array(out)]


def caa_synonym_mode() -> str:
    return os.environ.get("CCCC_CAA_SYNONYMS", "chinese").strip().lower()


def _embed_cos_paths() -> Tuple[str, str]:
    emb = os.environ.get("CCCC_CAA_EMBED", str(_DEFAULT_EMBED))
    cos = os.environ.get("CCCC_CAA_COS", str(_DEFAULT_COS))
    return emb, cos


def load_vocab_and_cos(embed_path: str, cos_path: str) -> Tuple[dict, dict, Any]:
    word2idx: dict = {}
    idx2word: dict = {}
    with open(embed_path, "r", encoding="utf-8", errors="ignore") as ifile:
        for line in ifile:
            parts = line.split()
            if not parts:
                continue
            w = parts[0]
            if w not in word2idx:
                idx2word[len(idx2word)] = w
                word2idx[w] = len(word2idx) - 1
    cos_sim = None
    if cos_path and os.path.isfile(cos_path):
        with open(cos_path, "rb") as fp:
            cos_sim = pickle.load(fp)
    return word2idx, idx2word, cos_sim


def build_cccc_predictor(detector: TextAIDetector):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predictor(text_data, batch_size=32):
        _ = batch_size
        texts: List[str] = []
        for t in text_data:
            if isinstance(t, (list, tuple)):
                texts.append("".join(str(x) for x in t))
            else:
                texts.append(str(t))
        out = detector.predict(texts)
        probs = torch.zeros(len(texts), 2, device=dev, dtype=torch.float32)
        for i, r in enumerate(out):
            ai = float(max(0.0, min(1.0, r.ai_score)))
            probs[i, 1] = ai
            probs[i, 0] = 1.0 - ai
        return probs

    return predictor


def _cleanup_chinese_embed_temp() -> None:
    global _chinese_temp_embed
    if _chinese_temp_embed and os.path.isfile(_chinese_temp_embed):
        try:
            os.unlink(_chinese_temp_embed)
        except OSError:
            pass
    _chinese_temp_embed = None


def ensure_caa_initialized_legacy(detector: TextAIDetector) -> None:
    global _caa_legacy_ready
    if _caa_legacy_ready:
        return
    emb, cos = _embed_cos_paths()
    if not os.path.isfile(emb):
        raise FileNotFoundError(
            f"缺少 counter-fitting 词向量: {emb}\n"
            "请设置 CCCC_CAA_EMBED 或改用 CCCC_CAA_SYNONYMS=chinese"
        )
    if not os.path.isfile(cos):
        raise FileNotFoundError(
            f"缺少 cos_sim pickle: {cos}\n"
            "请设置 CCCC_CAA_COS 或改用 CCCC_CAA_SYNONYMS=chinese"
        )
    w2i, i2w, cs = load_vocab_and_cos(emb, cos)
    if cs is None:
        raise FileNotFoundError("cos_sim 加载失败或为空")

    from third_party.caa.tool import initialize_cccc_injected_components

    initialize_cccc_injected_components(
        predictor=build_cccc_predictor(detector),
        word2idx=w2i,
        idx2word=i2w,
        cos_sim=cs,
        embed_func=emb,
        sim_predictor=DifflibWindowSim(),
        stop_words_set=None,
    )
    _caa_legacy_ready = True


def _inject_chinese_vocab(text: str, detector: TextAIDetector, pipe: Any) -> None:
    global _chinese_temp_embed
    from attacks.caa_chinese_synonyms import build_chinese_caa_vocab_for_text
    from third_party.caa.tool import initialize_cccc_injected_components

    _cleanup_chinese_embed_temp()
    try:
        from third_party.caa.tool import reset_global_manager

        reset_global_manager()
    except Exception:
        pass

    w2i, i2w, cos, path = build_chinese_caa_vocab_for_text(text.strip(), pipe)
    _chinese_temp_embed = path
    initialize_cccc_injected_components(
        predictor=build_cccc_predictor(detector),
        word2idx=w2i,
        idx2word=i2w,
        cos_sim=cos,
        embed_func=path,
        sim_predictor=DifflibWindowSim(),
        stop_words_set=None,
    )


def ensure_caa_for_text(
    text: str,
    detector: TextAIDetector,
    pipe: Optional[Any] = None,
) -> None:
    """每次攻击前调用：legacy 仅首次加载文件；chinese 按句重建词表与伪嵌入。"""
    if caa_synonym_mode() == "legacy":
        ensure_caa_initialized_legacy(detector)
        return
    from attacks.chinese_replace import ChineseReplacePipeline

    p = pipe or ChineseReplacePipeline()
    _inject_chinese_vocab(text, detector, p)


def ensure_caa_initialized(detector: TextAIDetector) -> None:
    """兼容旧调用：等价于 legacy 模式初始化。"""
    ensure_caa_initialized_legacy(detector)


def reset_caa_state() -> None:
    global _caa_legacy_ready
    _caa_legacy_ready = False
    _cleanup_chinese_embed_temp()
    try:
        from third_party.caa.tool import reset_global_manager

        reset_global_manager()
    except Exception:
        pass


def _true_label_int(orig_label: str) -> int:
    return 1 if str(orig_label).lower() == "ai" else 0


def _to_greedy_result(
    name: str,
    orig_text: str,
    adv_text: str,
    success: bool,
    orig_label: str,
    victim,
    queries: int,
    t0: float,
) -> GreedyMLMAttackResult:
    fs = victim.ai_score(adv_text)
    fl = victim.label(adv_text)
    oscore = victim.ai_score(orig_text)
    return GreedyMLMAttackResult(
        name=name,
        success=success,
        adversarial_text=adv_text,
        orig_label=orig_label,
        final_label=fl,
        orig_score=oscore,
        final_score=fs,
        perturbation_rate=_pert_rate_zh(orig_text, adv_text),
        query_count=queries,
        elapsed_sec=time.perf_counter() - t0,
        pivot_highlights=[],
        replaced_detail=[],
    )


def _pert_rate_zh(orig: str, adv: str) -> float:
    o = list(jieba.cut(orig))
    a = list(jieba.cut(adv))
    if not o:
        return 0.0
    changed = sum(1 for i in range(min(len(o), len(a))) if o[i] != a[i])
    changed += abs(len(o) - len(a))
    return min(1.0, changed / max(len(o), 1))


def run_hqa_as_greedy_result(
    text: str,
    victim,
    detector: TextAIDetector,
    orig_label: str,
    max_queries: int,
    pipe: Optional[Any] = None,
) -> GreedyMLMAttackResult:
    from third_party.caa.hqaattack_wrapper import create_hqaattack
    from third_party.caa.tool import get_component

    t0 = time.perf_counter()
    ensure_caa_for_text(text, detector, pipe=pipe)
    try:
        text_ls = list(jieba.cut(text.strip()))
        if len(text_ls) < 1:
            return _to_greedy_result("hqaattack", text, text, False, orig_label, victim, 0, t0)

        hqa = create_hqaattack(
            predictor=get_component("predictor"),
            word2idx=get_component("word2idx"),
            idx2word=get_component("idx2word"),
            cos_sim=get_component("cos_sim"),
            sim_predictor=get_component("sim_predictor"),
            stop_words_set=get_component("stop_words_set"),
            embed_func=get_component("embed_func"),
        )
        tl = _true_label_int(orig_label)
        result_text, attack_success, query_count = hqa.attack(
            text_ls=text_ls,
            true_label=tl,
            query_budget=max_queries,
            semantic_threshold=float(os.environ.get("CCCC_CAA_HQA_SEM", "0.85")),
            top_k_words=int(os.environ.get("CCCC_CAA_HQA_TOPK", "30")),
            max_iterations=int(os.environ.get("CCCC_CAA_HQA_ITER", "80")),
            sim_score_window=15,
            batch_size=int(os.environ.get("CCCC_CAA_BATCH", "8")),
            perturb_ratio=float(os.environ.get("CCCC_CAA_HQA_PERT", "0.2")),
        )
        if not attack_success or result_text is None:
            return _to_greedy_result("hqaattack", text, text, False, orig_label, victim, query_count, t0)
        adv = "".join(result_text) if isinstance(result_text, list) else str(result_text)
        ok = victim.is_attack_success(orig_label, adv)
        return _to_greedy_result("hqaattack", text, adv, ok, orig_label, victim, query_count, t0)
    finally:
        if caa_synonym_mode() != "legacy":
            _cleanup_chinese_embed_temp()
            try:
                from third_party.caa.tool import reset_global_manager

                reset_global_manager()
            except Exception:
                pass


def run_texthacker_as_greedy_result(
    text: str,
    victim,
    detector: TextAIDetector,
    orig_label: str,
    max_queries: int,
    pipe: Optional[Any] = None,
) -> GreedyMLMAttackResult:
    from third_party.caa.texthacker_wrapper import TextHackerWrapper, Args
    from third_party.caa.tool import get_component

    t0 = time.perf_counter()
    ensure_caa_for_text(text, detector, pipe=pipe)
    try:
        text_ls = list(jieba.cut(text.strip()))
        if len(text_ls) < 1:
            return _to_greedy_result("texthacker", text, text, False, orig_label, victim, 0, t0)

        args = Args()
        args.query_budget = max_queries
        args.batch_size = int(os.environ.get("CCCC_CAA_BATCH", "8"))
        th = TextHackerWrapper(
            predictor=get_component("predictor"),
            word2idx=get_component("word2idx"),
            idx2word=get_component("idx2word"),
            cos_sim=get_component("cos_sim"),
            sim_predictor=get_component("sim_predictor"),
            stop_words_set=get_component("stop_words_set"),
            args=args,
        )
        tl = _true_label_int(orig_label)
        result_text, attack_success, query_count = th.attack(text_ls=text_ls, true_label=tl, query_budget=max_queries)
        if not attack_success or result_text is None:
            return _to_greedy_result("texthacker", text, text, False, orig_label, victim, query_count, t0)
        adv = "".join(result_text) if isinstance(result_text, list) else str(result_text)
        ok = victim.is_attack_success(orig_label, adv)
        return _to_greedy_result("texthacker", text, adv, ok, orig_label, victim, query_count, t0)
    finally:
        if caa_synonym_mode() != "legacy":
            _cleanup_chinese_embed_temp()
            try:
                from third_party.caa.tool import reset_global_manager

                reset_global_manager()
            except Exception:
                pass


def use_caa_chain() -> bool:
    return os.environ.get("CCCC_USE_CAA_CHAIN", "").strip().lower() in ("1", "true", "yes")
