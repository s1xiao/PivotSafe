# -*- coding: utf-8 -*-
"""基于 Pivot 排序的 MLM 贪梦替换（Attacker2/3 与 Pivot 共用核心循环，仅超参不同）。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jieba

from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext
from attacks.pivot_attack import PivotHighlightResult, compute_pivot_ranking


@dataclass
class GreedyMLMAttackResult:
    name: str
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


def _pert_rate(orig: str, adv: str) -> float:
    o = list(jieba.cut(orig))
    a = list(jieba.cut(adv))
    if not o:
        return 0.0
    changed = sum(1 for i in range(min(len(o), len(a))) if o[i] != a[i])
    changed += abs(len(o) - len(a))
    return min(1.0, changed / max(len(o), 1))


def run_mlm_greedy_attack(
    name: str,
    text: str,
    pipe: ChineseReplacePipeline,
    victim: ScoreContext,
    *,
    min_sim: float,
    max_replace: int,
    max_queries: int = 100,
    orig_label: Optional[str] = None,
) -> GreedyMLMAttackResult:
    t0 = time.perf_counter()
    queries = 0
    orig_label = orig_label or victim.label(text)
    orig_score = victim.ai_score(text)
    queries += 1

    cur = text
    detail: List[Dict[str, Any]] = []
    hl_last: Optional[PivotHighlightResult] = None

    for _iter in range(max_replace):
        if queries >= max_queries:
            break
        ranked, hl = compute_pivot_ranking(cur, pipe, victim, max_queries=min(28, max_queries - queries))
        queries += hl.query_count
        hl_last = hl
        if not ranked:
            break
        sp, _ = ranked[0]
        cands = pipe.mlm_candidates_for_span(cur, sp)
        queries += 1
        best_new = None
        best_score = victim.ai_score(cur)
        queries += 1
        best_meta = None
        for c in cands:
            if queries >= max_queries:
                break
            new_text = cur[: sp.start] + c + cur[sp.end :]
            sim = pipe.semantic_filter(text, new_text)
            queries += 1
            if sim < min_sim:
                continue
            sc = victim.ai_score(new_text)
            queries += 1
            if sc < best_score:
                best_score = sc
                best_new = new_text
                best_meta = (c, sim)
        if best_new is None or best_meta is None:
            break
        c, sim = best_meta
        cur = best_new
        detail.append({"from": sp.word, "to": c, "pos": sp.pos, "sim": sim})
        if victim.is_attack_success(orig_label, cur):
            break

    if hl_last is None:
        _, hl_last = compute_pivot_ranking(text, pipe, victim, max_queries=min(28, max_queries))

    final_score = victim.ai_score(cur)
    queries += 1
    final_label = victim.label(cur)
    success = victim.is_attack_success(orig_label, cur)
    return GreedyMLMAttackResult(
        name=name,
        success=success,
        adversarial_text=cur,
        orig_label=orig_label,
        final_label=final_label,
        orig_score=orig_score,
        final_score=final_score,
        perturbation_rate=_pert_rate(text, cur),
        query_count=queries,
        elapsed_sec=time.perf_counter() - t0,
        pivot_highlights=hl_last.pivot_spans,
        replaced_detail=detail,
    )
