# -*- coding: utf-8 -*-
"""
AnchorAttack_classification 中 auto_beamsearch + issuccess 的语义，适配中文无空格拼接与 CCCC ScoreContext。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext, TokenSpan


def _join_zh(tokens: List[str]) -> str:
    return "".join(tokens)


@dataclass
class AnchorBeamResult:
    success: bool
    adversarial_text: str
    adversarial_tokens: List[str]
    query_count: int
    pivot_highlights: List[Dict[str, Any]]
    replaced_detail: List[Dict[str, Any]]


def _proba_batch(victim: ScoreContext, texts: List[str]) -> torch.Tensor:
    dev = torch.device("cpu")
    probs = torch.zeros(len(texts), 2, dtype=torch.float32, device=dev)
    for i, t in enumerate(texts):
        s = float(victim.ai_score(t))
        probs[i, 1] = max(0.0, min(1.0, s))
        probs[i, 0] = 1.0 - probs[i, 1]
    return probs


def _predictor_from_victim(victim: ScoreContext) -> Callable[..., torch.Tensor]:
    def predictor(all_text_ls: List[List[str]], batch_size: int = 32) -> torch.Tensor:
        _ = batch_size
        texts = [_join_zh(wl) for wl in all_text_ls]
        return _proba_batch(victim, texts)

    return predictor


def _span_to_index(spans_all: List[TokenSpan], sp: TokenSpan) -> int:
    for j, s in enumerate(spans_all):
        if s.start == sp.start and s.end == sp.end:
            return j
    return -1


def issuccess_zh(
    predictor: Callable[..., torch.Tensor],
    batch_size: int,
    all_text_ls: List[List[str]],
    text_ls: List[str],
    true_label: int,
    num_queries: int,
    k: int,
    query_budget: int,
    victim: ScoreContext,
    orig_joined: str,
) -> Tuple[List[Any], int]:
    _ = batch_size
    if not all_text_ls:
        return [0, [], []], num_queries

    new_probs = predictor(all_text_ls, batch_size=len(all_text_ls))
    _probs_argmax = torch.argmax(new_probs, dim=-1).cpu().numpy()
    np_true = np.array([true_label] * len(all_text_ls))
    flipped = np_true != _probs_argmax
    success_text_ls = [all_text_ls[i] for i in range(len(all_text_ls)) if flipped[i]]

    if len(success_text_ls) == 0:
        num_queries += len(all_text_ls)
        sims = []
        for wl in all_text_ls:
            tj = _join_zh(wl)
            sims.append(1.0 - abs(victim.ai_score(tj) - victim.ai_score(orig_joined)))
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        k_text_ls = [all_text_ls[order[i]] for i in range(min(k, len(order)))]
        return [0, k_text_ls, []], num_queries

    semantic = []
    for wl in success_text_ls:
        tj = _join_zh(wl)
        semantic.append(1.0 - abs(victim.ai_score(tj) - victim.ai_score(orig_joined)))
    num_queries += len(all_text_ls)
    best = success_text_ls[int(np.argmax(semantic))]
    if num_queries > query_budget:
        return [1, text_ls, num_queries, best], num_queries
    return [1, text_ls, num_queries, best], num_queries


def auto_beamsearch_zh(
    predictor: Callable[..., torch.Tensor],
    batch_size: int,
    text_ls: List[str],
    true_label: int,
    num_queries: int,
    k_text_ls: List[List[str]],
    words_perturb_idx: List[int],
    start: int,
    synonym_words: List[List[str]],
    k: int,
    query_budget: int,
    victim: ScoreContext,
    orig_joined: str,
) -> Tuple[List[Any], int]:
    curr: List[List[str]] = []
    tmp: List[List[str]] = []
    suc: List[Any] = [0, text_ls, num_queries, text_ls]
    for i in range(k):
        if i == 0:
            tmp = list(k_text_ls)
        else:
            tmp = curr
            curr = []
        for text in tmp:
            curr.append(list(text))
            if start + i >= len(synonym_words):
                continue
            for j in synonym_words[start + i]:
                aa = list(text)
                if 0 <= words_perturb_idx[start + i] < len(aa):
                    aa[words_perturb_idx[start + i]] = j
                curr.append(aa)
        suc, num_queries = issuccess_zh(
            predictor,
            batch_size,
            curr,
            text_ls,
            true_label,
            num_queries,
            k,
            query_budget,
            victim,
            orig_joined,
        )
        if num_queries >= query_budget + 10:
            return [1, text_ls, num_queries, text_ls], num_queries
        if suc[0] == 1:
            return suc, num_queries
        curr = suc[1]  # type: ignore[assignment]

    return suc, num_queries


def run_anchor_beam_pivot_attack(
    text: str,
    pipe: ChineseReplacePipeline,
    victim: ScoreContext,
    spans_all: List[TokenSpan],
    spans: List[TokenSpan],
    anchor_scores: List[float],
    *,
    orig_label: str,
    max_queries: int = 100,
    synonyms_num: int = 12,
    k_beam: int = 2,
    import_score_threshold: float = -1e6,
    batch_size: int = 8,
) -> AnchorBeamResult:
    """
    spans / anchor_scores 与 anchor_word_ranking 输出对齐（同长度）。
    """
    text_ls = [sp.word for sp in spans_all]
    orig_joined = text.strip()
    true_label = 1 if str(orig_label).lower() == "ai" else 0

    predictor = _predictor_from_victim(victim)
    probs0 = predictor([text_ls], batch_size=1)
    pred0 = int(torch.argmax(probs0, dim=-1).item())
    q = 1
    if pred0 != true_label:
        return AnchorBeamResult(
            success=False,
            adversarial_text=orig_joined,
            adversarial_tokens=text_ls,
            query_count=q,
            pivot_highlights=[],
            replaced_detail=[],
        )

    import_scores = [float(anchor_scores[i]) for i in range(len(spans))]
    words_perturb: List[Tuple[int, str]] = []
    for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
        if score <= import_score_threshold or idx >= len(spans):
            continue
        sp = spans[idx]
        ti = _span_to_index(spans_all, sp)
        if ti < 0:
            continue
        w = sp.word
        if w:
            words_perturb.append((ti, w))

    synonym_words: List[List[str]] = []
    word_perturb_text_idx: List[int] = []
    for ti, w in words_perturb:
        sp = spans_all[ti]
        cands = pipe.mlm_candidates_for_span(text, sp)
        good: List[str] = []
        for c in cands:
            if not c or c == w:
                continue
            if pipe.semantic_filter(w, c) >= pipe.cfg.min_sim:
                good.append(c)
        good = good[:synonyms_num]
        if not good:
            continue
        synonym_words.append(good)
        word_perturb_text_idx.append(ti)

    pivot_highlights: List[Dict[str, Any]] = []
    for ti, w in words_perturb[: min(8, len(words_perturb))]:
        pivot_highlights.append({"word": w, "index": ti})

    if len(word_perturb_text_idx) == 0 or len(synonym_words) == 0:
        return AnchorBeamResult(
            success=False,
            adversarial_text=orig_joined,
            adversarial_tokens=text_ls,
            query_count=q,
            pivot_highlights=pivot_highlights,
            replaced_detail=[],
        )

    num_queries = q
    k_use = min(k_beam, len(word_perturb_text_idx))
    if len(word_perturb_text_idx) < k_beam:
        suc, num_queries = auto_beamsearch_zh(
            predictor,
            batch_size,
            text_ls,
            true_label,
            num_queries,
            [text_ls],
            word_perturb_text_idx,
            0,
            synonym_words,
            k=len(word_perturb_text_idx),
            query_budget=max_queries,
            victim=victim,
            orig_joined=orig_joined,
        )
    else:
        suc, num_queries = auto_beamsearch_zh(
            predictor,
            batch_size,
            text_ls,
            true_label,
            num_queries,
            [text_ls],
            word_perturb_text_idx,
            0,
            synonym_words,
            k=k_use,
            query_budget=max_queries,
            victim=victim,
            orig_joined=orig_joined,
        )

    if suc[0] == 1 and len(suc) > 3:
        best_toks = suc[3]
        adv = _join_zh(best_toks)
        replaced_detail: List[Dict[str, Any]] = []
        for i, (a, b) in enumerate(zip(text_ls, best_toks)):
            if a != b:
                replaced_detail.append({"index": i, "from": a, "to": b})
        ok = victim.is_attack_success(orig_label, adv)
        return AnchorBeamResult(
            success=ok,
            adversarial_text=adv,
            adversarial_tokens=list(best_toks),
            query_count=num_queries,
            pivot_highlights=pivot_highlights,
            replaced_detail=replaced_detail,
        )

    return AnchorBeamResult(
        success=False,
        adversarial_text=orig_joined,
        adversarial_tokens=text_ls,
        query_count=num_queries,
        pivot_highlights=pivot_highlights,
        replaced_detail=[],
    )
