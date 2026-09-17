# -*- coding: utf-8 -*-
"""
中文 Pivot（语义锚点）攻击：基于重要度排序 + 统一中文替换管线。

Anchor 重要度入口（与 limeattack / AnchorAttack 排序同源）：动态导入
``third_party.anchor.anchor_word_ranking.anchor_word_ranking``（见 ``_compute_pivot_with_anchor``）。
开启 ``PIVOT_USE_ANCHOR=1`` 且 ``CCCC_PIVOT_BEAM=1`` 时，在排序后由 ``anchor_beam_zh`` 做束搜替换。

若 Anchor/Beam 不可用、关闭或未能改写文本：**不再**调用 MLM 贪心；仅通过 ``compute_pivot_ranking``
返回未成功（``adversarial_text`` 为原文）及 pivot 高亮，供前端展示。
高亮 pivot_spans：Anchor 成功时为最终锚点词；否则为遮挡/排序回退下的单条最高分词。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple  # Tuple kept for compute_pivot typing

import numpy as np

from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext, TokenSpan


def _remove_span(text: str, sp: TokenSpan) -> str:
    return (text[: sp.start] + text[sp.end :]).replace("  ", " ").strip()


def _pivot_use_anchor() -> bool:
    v = os.environ.get("PIVOT_USE_ANCHOR", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def _pivot_use_beam() -> bool:
    v = os.environ.get("CCCC_PIVOT_BEAM", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def _victim_proba_matrix(victim: ScoreContext, texts: List[str]) -> np.ndarray:
    """Anchor 需要 (N, C) 概率；二分类 human/ai 由 ai_score 构造。"""
    rows = []
    for t in texts:
        s = float(victim.ai_score(t))
        rows.append([1.0 - s, s])
    return np.asarray(rows, dtype=np.float64)


def _compute_pivot_with_anchor(
    text: str,
    pipe: ChineseReplacePipeline,
    victim: ScoreContext,
    spans_all: List[TokenSpan],
    spans: List[TokenSpan],
    max_queries: int,
) -> Optional[Tuple[List[Tuple[TokenSpan, float]], PivotHighlightResult, List[float]]]:
    """调用 ``third_party.anchor.anchor_word_ranking.anchor_word_ranking`` 得到词级重要度与最终锚点索引。"""
    if not spans:
        return None
    try:
        from third_party.anchor.anchor_word_ranking import anchor_word_ranking
    except Exception:
        return None

    text_tokens = [sp.word for sp in spans]

    def predictor(xs: List[str]) -> np.ndarray:
        return _victim_proba_matrix(victim, list(xs))

    try:
        scores, _pairs, q_anchor, final_idx = anchor_word_ranking(
            text_tokens,
            predictor,
            class_names=("human", "ai"),
            use_unk=True,
            threshold=0.85,
            beam_size=1,
            num_samples=200,
            onepass=False,
            batch_size=8,
            anchor_bonus=1.0,
            query_budget=max(16, max_queries),
        )
    except Exception:
        return None

    ranked = sorted(
        [(spans[i], float(scores[i])) for i in range(len(spans))],
        key=lambda x: -x[1],
    )
    pivot_spans: List[Dict[str, Any]] = []
    for i in final_idx:
        if 0 <= i < len(spans):
            sp = spans[i]
            pivot_spans.append(
                {
                    "word": sp.word,
                    "pos": sp.pos,
                    "start": sp.start,
                    "end": sp.end,
                    "importance": float(scores[i]),
                }
            )
    tokens_out: List[Dict[str, Any]] = []
    for sp in spans_all:
        tokens_out.append(
            {
                "word": sp.word,
                "pos": sp.pos,
                "start": sp.start,
                "end": sp.end,
            }
        )
    hl = PivotHighlightResult(
        text=text,
        tokens=tokens_out,
        pivot_spans=pivot_spans,
        query_count=int(q_anchor),
    )
    score_list = [float(scores[i]) for i in range(len(spans))]
    return ranked, hl, score_list


@dataclass
class PivotHighlightResult:
    text: str
    tokens: List[Dict[str, Any]]
    pivot_spans: List[Dict[str, Any]]
    query_count: int = 0


@dataclass
class PivotAttackResult:
    success: bool
    adversarial_text: str
    orig_label: str
    final_label: str
    orig_score: float
    final_score: float
    perturbation_rate: float
    query_count: int
    elapsed_sec: float
    pivot_highlights: List[Dict[str, Any]] = field(default_factory=list)
    replaced_detail: List[Dict[str, Any]] = field(default_factory=list)


def compute_pivot_ranking(
    text: str,
    pipe: ChineseReplacePipeline,
    victim: ScoreContext,
    max_queries: int = 64,
) -> Tuple[List[Tuple[TokenSpan, float]], PivotHighlightResult]:
    spans_all = pipe.tokenize_with_pos(text)
    spans = pipe.filter_pos(spans_all)

    if _pivot_use_anchor():
        anchor_out = _compute_pivot_with_anchor(
            text, pipe, victim, spans_all, spans, max_queries=max_queries
        )
        if anchor_out is not None:
            ranked, hl, _scores = anchor_out
            return ranked, hl

    base_score = victim.ai_score(text)
    q = 1
    scored: List[Tuple[TokenSpan, float]] = []
    for sp in spans:
        if q >= max_queries:
            break
        removed = _remove_span(text, sp)
        if len(removed) < 2:
            continue
        s2 = victim.ai_score(removed)
        q += 1
        importance = abs(base_score - s2)
        scored.append((sp, importance))
    scored.sort(key=lambda x: -x[1])
    tokens_out: List[Dict[str, Any]] = []
    for sp in spans_all:
        tokens_out.append(
            {
                "word": sp.word,
                "pos": sp.pos,
                "start": sp.start,
                "end": sp.end,
            }
        )
    # 回退路径：高亮仅保留最重要的一条，避免「多锚点列表」与论文语义不一致
    pivot_spans = (
        [
            {
                "word": scored[0][0].word,
                "pos": scored[0][0].pos,
                "start": scored[0][0].start,
                "end": scored[0][0].end,
                "importance": float(scored[0][1]),
            }
        ]
        if scored
        else []
    )
    return scored, PivotHighlightResult(
        text=text,
        tokens=tokens_out,
        pivot_spans=pivot_spans,
        query_count=q,
    )


def run_pivot_attack(
    text: str,
    pipe: ChineseReplacePipeline,
    victim: ScoreContext,
    orig_label: Optional[str] = None,
    max_replace: Optional[int] = None,
    max_queries: int = 1000,
) -> PivotAttackResult:
    import time

    spans_all = pipe.tokenize_with_pos(text)
    spans_f = pipe.filter_pos(spans_all)
    ol = orig_label if orig_label is not None else victim.label(text)
    t0 = time.perf_counter()

    def _no_rewrite_result() -> PivotAttackResult:
        _, hl = compute_pivot_ranking(text, pipe, victim, max_queries=max_queries)
        oscore = victim.ai_score(text)
        fl = victim.label(text)
        return PivotAttackResult(
            success=False,
            adversarial_text=text,
            orig_label=ol,
            final_label=fl,
            orig_score=oscore,
            final_score=oscore,
            perturbation_rate=0.0,
            query_count=hl.query_count,
            elapsed_sec=time.perf_counter() - t0,
            pivot_highlights=list(hl.pivot_spans),
            replaced_detail=[],
        )

    if _pivot_use_anchor() and _pivot_use_beam():
        anchor_out = _compute_pivot_with_anchor(
            text, pipe, victim, spans_all, spans_f, max_queries=max_queries
        )
        if anchor_out is not None:
            _ranked, _hl, anchor_scores = anchor_out
            try:
                from attacks.anchor_beam_zh import run_anchor_beam_pivot_attack

                beam = run_anchor_beam_pivot_attack(
                    text,
                    pipe,
                    victim,
                    spans_all,
                    spans_f,
                    anchor_scores,
                    orig_label=ol,
                    max_queries=max_queries,
                )
                adv = beam.adversarial_text
                fs = victim.ai_score(adv)
                fl = victim.label(adv)
                oscore = victim.ai_score(text)
                # 扰动率：按词级变化比例
                changed = sum(1 for a, b in zip(beam.adversarial_tokens, [sp.word for sp in spans_all]) if a != b)
                if len(spans_all) == 0:
                    prate = 0.0
                else:
                    prate = min(1.0, changed / max(len([sp.word for sp in spans_all]), 1))
                piv_hl = beam.pivot_highlights or []
                return PivotAttackResult(
                    success=beam.success,
                    adversarial_text=adv,
                    orig_label=ol,
                    final_label=fl,
                    orig_score=oscore,
                    final_score=fs,
                    perturbation_rate=prate,
                    query_count=beam.query_count,
                    elapsed_sec=time.perf_counter() - t0,
                    pivot_highlights=piv_hl,
                    replaced_detail=beam.replaced_detail,
                )
            except Exception:
                return _no_rewrite_result()

    return _no_rewrite_result()


