# -*- coding: utf-8 -*-
"""
固定长度 3 的攻击链：Pivot -> HQA -> TextHacker。
每一步从**同一原文**独立攻击 baseline；**前一步已成功翻转 baseline 时，后续步不再执行**（标记为未执行）。
Step2/3 **仅** HQA / TextHacker，需传入 `TextAIDetector`，无 MLM 回退。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext
from attacks.greedy_mlm import GreedyMLMAttackResult
from attacks.pivot_attack import PivotAttackResult, run_pivot_attack
from detectors.text.detector import TextAIDetector


@dataclass
class ChainStepResult:
    step_index: int
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
    """是否实际运行该步攻击器；False 表示因前步已成功而跳过。"""
    executed: bool = True


def _skipped_step(
    step_index: int,
    name: str,
    text: str,
    orig_label: str,
    orig_score: float,
) -> ChainStepResult:
    return ChainStepResult(
        step_index=step_index,
        name=name,
        success=False,
        adversarial_text=text,
        orig_label=orig_label,
        final_label=orig_label,
        orig_score=orig_score,
        final_score=orig_score,
        perturbation_rate=0.0,
        query_count=0,
        elapsed_sec=0.0,
        pivot_highlights=[],
        replaced_detail=[],
        executed=False,
    )


@dataclass
class FixedChainResult:
    original_text: str
    overall_success: bool
    winning_step: Optional[int]
    final_adversarial_text: Optional[str]
    steps: List[ChainStepResult]
    baseline_orig_label: str
    baseline_orig_score: float


def _to_step(idx: int, name: str, r: GreedyMLMAttackResult | PivotAttackResult) -> ChainStepResult:
    if isinstance(r, PivotAttackResult):
        return ChainStepResult(
            step_index=idx,
            name=name,
            success=r.success,
            adversarial_text=r.adversarial_text,
            orig_label=r.orig_label,
            final_label=r.final_label,
            orig_score=r.orig_score,
            final_score=r.final_score,
            perturbation_rate=r.perturbation_rate,
            query_count=r.query_count,
            elapsed_sec=r.elapsed_sec,
            pivot_highlights=r.pivot_highlights,
            replaced_detail=r.replaced_detail,
            executed=True,
        )
    return ChainStepResult(
        step_index=idx,
        name=name,
        success=r.success,
        adversarial_text=r.adversarial_text,
        orig_label=r.orig_label,
        final_label=r.final_label,
        orig_score=r.orig_score,
        final_score=r.final_score,
        perturbation_rate=r.perturbation_rate,
        query_count=r.query_count,
        elapsed_sec=r.elapsed_sec,
        pivot_highlights=r.pivot_highlights,
        replaced_detail=r.replaced_detail,
        executed=True,
    )


def _step_baseline_flipped(step: ChainStepResult) -> bool:
    """相对链起始原文，baseline 最终标签是否相对 orig_label 发生变化（攻击目标）。"""
    if not step.executed:
        return False
    return step.success


def _winning_perturbation(steps: List[ChainStepResult], winner: Optional[int]) -> Dict[str, Any]:
    if winner is None:
        return {"replacement_pairs": [], "pivot_highlights": [], "winning_adversarial_text": None}
    for s in steps:
        if s.step_index == winner and s.executed:
            return {
                "replacement_pairs": list(s.replaced_detail),
                "pivot_highlights": list(s.pivot_highlights),
                "winning_adversarial_text": s.adversarial_text,
                "perturbation_rate": s.perturbation_rate,
            }
    return {"replacement_pairs": [], "pivot_highlights": [], "winning_adversarial_text": None}


def run_fixed_chain(
    text: str,
    victim: ScoreContext,
    pipe: Optional[ChineseReplacePipeline] = None,
    detector: Optional[TextAIDetector] = None,
    *,
    max_queries_per_step: int = 1000,
) -> FixedChainResult:
    if detector is None:
        raise ValueError(
            "run_fixed_chain 需要传入 detector：Step2/3 固定为 HQA / TextHacker，不再支持无 detector 的 MLM 回退。"
        )
    from attacks.caa_cccc_bridge import run_hqa_as_greedy_result, run_texthacker_as_greedy_result

    pipe = pipe or ChineseReplacePipeline()
    orig_score = victim.ai_score(text)
    orig_label = victim.label(text)

    steps: List[ChainStepResult] = []
    winner: Optional[int] = None
    adv_text: Optional[str] = None

    p1 = run_pivot_attack(text, pipe, victim, orig_label=orig_label, max_queries=max_queries_per_step)
    steps.append(_to_step(1, "pivot", p1))
    if p1.success:
        winner = 1
        adv_text = p1.adversarial_text
        steps.append(_skipped_step(2, "attacker2_hqa", text, orig_label, orig_score))
        steps.append(_skipped_step(3, "attacker3_texthacker", text, orig_label, orig_score))
    else:
        r2: GreedyMLMAttackResult | PivotAttackResult
        r2 = run_hqa_as_greedy_result(
            text,
            victim,
            detector,
            orig_label,
            max_queries_per_step,
            pipe=pipe,
        )
        steps.append(_to_step(2, "attacker2", r2))
        if r2.success:
            winner = 2
            adv_text = r2.adversarial_text
            steps.append(_skipped_step(3, "attacker3_texthacker", text, orig_label, orig_score))
        else:
            r3 = run_texthacker_as_greedy_result(
                text,
                victim,
                detector,
                orig_label,
                max_queries_per_step,
                pipe=pipe,
            )
            steps.append(_to_step(3, "attacker3", r3))
            if r3.success:
                winner = 3
                adv_text = r3.adversarial_text

    return FixedChainResult(
        original_text=text,
        overall_success=winner is not None,
        winning_step=winner,
        final_adversarial_text=adv_text,
        steps=steps,
        baseline_orig_label=orig_label,
        baseline_orig_score=orig_score,
    )


def chain_result_to_json(r: FixedChainResult) -> Dict[str, Any]:
    pert = _winning_perturbation(r.steps, r.winning_step)
    steps_out: List[Dict[str, Any]] = []
    for s in r.steps:
        flipped = _step_baseline_flipped(s)
        steps_out.append(
            {
                "step_index": s.step_index,
                "name": s.name,
                "executed": s.executed,
                "baseline_prediction_flipped": flipped,
                "success": s.success,
                "adversarial_text": s.adversarial_text,
                "orig_label": s.orig_label,
                "final_label": s.final_label,
                "orig_score": s.orig_score,
                "final_score": s.final_score,
                "perturbation_rate": s.perturbation_rate,
                "query_count": s.query_count,
                "elapsed_sec": s.elapsed_sec,
                "pivot_highlights": s.pivot_highlights,
                "replaced_detail": s.replaced_detail,
            }
        )

    return {
        "original_text": r.original_text,
        "overall_success": r.overall_success,
        "baseline_bypassed": r.overall_success,
        "winning_step": r.winning_step,
        "final_adversarial_text": r.final_adversarial_text,
        "baseline_orig_label": r.baseline_orig_label,
        "baseline_orig_score": r.baseline_orig_score,
        "semantic_note": (
            "overall_success / baseline_bypassed：是否至少一步使 Baseline 相对原文标签翻转；"
            "executed=false 表示前步已成功，本步未运行；baseline_prediction_flipped 仅对已执行步有意义。"
        ),
        "perturbation_summary": pert,
        "steps": steps_out,
    }
