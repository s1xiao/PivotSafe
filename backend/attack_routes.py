# -*- coding: utf-8 -*-
"""Pivot 高亮与固定攻击链 API。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext
from attacks.fixed_chain import chain_result_to_json, run_fixed_chain
from attacks.pivot_attack import compute_pivot_ranking
from detectors.text.detector import TextAIDetector

_PIPE: ChineseReplacePipeline | None = None


def _get_pipe() -> ChineseReplacePipeline:
    global _PIPE
    if _PIPE is None:
        _PIPE = ChineseReplacePipeline()
    return _PIPE


class PivotHighlightRequest(BaseModel):
    text: str = Field(..., max_length=512)


class PivotHighlightResponse(BaseModel):
    text: str
    tokens: List[Dict[str, Any]]
    pivot_spans: List[Dict[str, Any]]
    query_count: int


class FixedChainRequest(BaseModel):
    text: str = Field(..., max_length=512)
    max_queries_per_step: int = Field(default=1000, ge=1, le=1000)


class FixedChainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    result: Dict[str, Any]


def register_attack_routes(
    app: FastAPI,
    baseline_ai_score: Callable[[str], float],
    text_detector: Optional[TextAIDetector] = None,
) -> None:
    """
    baseline_ai_score: 单条文本 -> AI 概率（与 TextAIDetector.score 一致）。
    text_detector: 供固定链 HQA/TextHacker 使用；为 None 时从 get_text_detector() 加载。固定链 Step2/3 仅 CAA，无 MLM。
    """

    def _victim() -> ScoreContext:
        return ScoreContext(score_fn=baseline_ai_score, threshold=0.5, positive_is_ai=True)

    r = APIRouter(tags=["attack"])

    @r.post("/api/attack/pivot-highlight", response_model=PivotHighlightResponse)
    async def pivot_highlight(payload: PivotHighlightRequest) -> PivotHighlightResponse:
        text = payload.text.strip()
        if len(text) < 2:
            raise HTTPException(status_code=400, detail="文本过短")
        pipe = _get_pipe()
        _, hl = compute_pivot_ranking(text, pipe, _victim(), max_queries=48)
        return PivotHighlightResponse(
            text=hl.text,
            tokens=hl.tokens,
            pivot_spans=hl.pivot_spans,
            query_count=hl.query_count,
        )

    @r.post("/api/attack/fixed-chain", response_model=FixedChainResponse)
    async def fixed_chain(payload: FixedChainRequest) -> FixedChainResponse:
        text = payload.text.strip()
        if len(text) < 2:
            raise HTTPException(status_code=400, detail="文本过短")
        pipe = _get_pipe()
        det = text_detector
        if det is None:
            from backend.main import get_text_detector

            det = get_text_detector()
        res = run_fixed_chain(
            text,
            _victim(),
            pipe,
            detector=det,
            max_queries_per_step=payload.max_queries_per_step,
        )
        return FixedChainResponse(result=chain_result_to_json(res))

    app.include_router(r)
