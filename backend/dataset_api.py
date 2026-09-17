# -*- coding: utf-8 -*-
"""
数据集样本池、批量评测产物、单条复盘、双模型对比 API。
路径可通过环境变量覆盖：CCCC_POOLS_DIR、CCCC_EVAL_DIR、CCCC_PREPARED_DIR
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from detectors.text.model_loader import (
    build_robust_text_detector,
    build_text_detector,
    robust_checkpoint_ready,
)

router = APIRouter(tags=["dataset"])


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def pools_dir() -> Path:
    p = os.environ.get("CCCC_POOLS_DIR")
    return Path(p) if p else _root() / "data" / "pools"


def eval_dir() -> Path:
    p = os.environ.get("CCCC_EVAL_DIR")
    return Path(p) if p else _root() / "data" / "eval"


def prepared_dir() -> Path:
    p = os.environ.get("CCCC_PREPARED_DIR")
    return Path(p) if p else _root() / "data" / "hf_prepared"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


_samples_cache: Dict[str, List[Dict[str, Any]]] = {}
_cases_index: Dict[str, Dict[str, Any]] | None = None

_eval_rebuild_lock = threading.Lock()


def _pool_path(kind: str) -> Path:
    return pools_dir() / {
        "clean": "sample_pool_clean.jsonl",
        "attackable": "sample_pool_attackable.jsonl",
        "adversarial": "sample_pool_adversarial.jsonl",
    }.get(kind, "sample_pool_clean.jsonl")


def _load_pool(kind: str) -> List[Dict[str, Any]]:
    path = _pool_path(kind)
    key = str(path.resolve())
    if key not in _samples_cache:
        _samples_cache[key] = _load_jsonl(path)
    return _samples_cache[key]


def _build_cases_index() -> Dict[str, Dict[str, Any]]:
    global _cases_index
    if _cases_index is not None:
        return _cases_index
    path = eval_dir() / "cases_review.jsonl"
    idx: Dict[str, Dict[str, Any]] = {}
    for row in _load_jsonl(path):
        cid = str(row.get("case_id", ""))
        if cid:
            idx[cid] = row
    _cases_index = idx
    return idx


def invalidate_caches() -> None:
    global _cases_index
    _samples_cache.clear()
    _cases_index = None


# --- Pydantic ---


class TextCompareRequest(BaseModel):
    text: str = Field(..., max_length=512)
    true_label: Optional[str] = Field(None, description="金标 human/ai，可选")


class ModelScoreBlock(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    label: str
    ai_score: float


class TextCompareResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    baseline: ModelScoreBlock
    robust: Optional[ModelScoreBlock]
    models_agree: bool
    gold_label: Optional[str]
    baseline_agrees_with_gold: Optional[bool]
    robust_agrees_with_gold: Optional[bool]


class DatasetSampleItem(BaseModel):
    id: str
    text: str
    label: str
    source: Optional[str] = None
    split: Optional[str] = None
    char_len: int = 0


class DatasetSamplesResponse(BaseModel):
    items: List[DatasetSampleItem]
    total: int
    page: int
    per_page: int
    pool: str


class DatasetSummaryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prepared_meta: Optional[Dict[str, Any]] = None
    pools_meta: Optional[Dict[str, Any]] = None
    dashboard_bundle: Optional[Dict[str, Any]] = None
    vuln_summary: Optional[Dict[str, Any]] = None
    pools_paths_exist: Dict[str, bool] = Field(default_factory=dict)
    eval_paths_exist: Dict[str, bool] = Field(default_factory=dict)


class RebuildEvalArtifactsRequest(BaseModel):
    limit: int = Field(
        default=200,
        ge=10,
        le=20_000,
        description="Clean test 与对抗池各自最多使用前 N 条（同时截断）",
    )


class RebuildEvalArtifactsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    ok: bool
    message: str
    stats: Dict[str, Any]


# --- Routes ---


@router.get("/api/dataset/summary", response_model=DatasetSummaryResponse)
async def dataset_summary() -> DatasetSummaryResponse:
    pm = _read_json(prepared_dir() / "meta.json")
    poolm = _read_json(pools_dir() / "pools_meta.json")
    bundle = _read_json(eval_dir() / "dashboard_bundle.json")
    vuln_path = _root() / "data" / "vuln" / "summary.json"
    vuln = _read_json(vuln_path) if vuln_path.is_file() else None

    pd = pools_dir()
    ed = eval_dir()
    pools_exist = {
        "clean": (pd / "sample_pool_clean.jsonl").is_file(),
        "attackable": (pd / "sample_pool_attackable.jsonl").is_file(),
        "adversarial": (pd / "sample_pool_adversarial.jsonl").is_file(),
    }
    eval_exist = {
        "dashboard_bundle": (ed / "dashboard_bundle.json").is_file(),
        "cases_review": (ed / "cases_review.jsonl").is_file(),
        "baseline_clean_metrics": (ed / "baseline_clean_metrics.json").is_file(),
    }
    return DatasetSummaryResponse(
        prepared_meta=pm,
        pools_meta=poolm,
        dashboard_bundle=bundle,
        vuln_summary=vuln,
        pools_paths_exist=pools_exist,
        eval_paths_exist=eval_exist,
    )


@router.get("/api/dataset/samples", response_model=DatasetSamplesResponse)
async def dataset_samples(
    pool: str = "clean",
    page: int = 1,
    per_page: int = 20,
    split: Optional[str] = None,
    label: Optional[str] = None,
    source_substr: Optional[str] = None,
    random_one: bool = False,
) -> DatasetSamplesResponse:
    if pool not in ("clean", "attackable", "adversarial"):
        raise HTTPException(status_code=400, detail="pool 必须是 clean|attackable|adversarial")
    rows = _load_pool(pool)
    if random_one:
        if not rows:
            return DatasetSamplesResponse(items=[], total=0, page=1, per_page=1, pool=pool)
        r = random.choice(rows)
        it = _to_sample_item(r, pool)
        return DatasetSamplesResponse(items=[it], total=1, page=1, per_page=1, pool=pool)

    filt: List[Dict[str, Any]] = []
    for r in rows:
        if split and str(r.get("split", "")) != split:
            continue
        if label and str(r.get("label", "")).lower() != label.lower():
            continue
        if source_substr and source_substr not in str(r.get("source", "")):
            continue
        filt.append(r)

    total = len(filt)
    start = max(0, (page - 1) * per_page)
    chunk = filt[start : start + per_page]
    items = [_to_sample_item(r, pool) for r in chunk]
    return DatasetSamplesResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        pool=pool,
    )


def _to_sample_item(r: Dict[str, Any], pool: str) -> DatasetSampleItem:
    if pool == "adversarial":
        ch = r.get("chain") or {}
        t = str(ch.get("original_text", ""))
        lab = str(r.get("true_label", r.get("label", "human")))
    else:
        t = str(r.get("text", ""))
        lab = str(r.get("label", "human"))
    clen = int(r.get("char_len", r.get("length", len(t))))
    return DatasetSampleItem(
        id=str(r.get("id", "")),
        text=t,
        label=lab,
        source=r.get("source"),
        split=r.get("split"),
        char_len=clen,
    )


@router.get("/api/cases/review/{case_id}")
async def case_review(case_id: str) -> Dict[str, Any]:
    idx = _build_cases_index()
    if case_id not in idx:
        raise HTTPException(
            status_code=404,
            detail="未找到案例，请先运行 scripts/build_eval_artifacts 生成 cases_review.jsonl",
        )
    return idx[case_id]


class CaseListItem(BaseModel):
    case_id: str
    text_preview: str
    true_label: str
    baseline_evaded: bool
    robust_recovered: Optional[bool] = None
    char_len: int = 0
    baseline_label_adv: Optional[str] = None
    robust_label_adv: Optional[str] = None
    winning_step: Optional[int] = None


class CaseListResponse(BaseModel):
    items: List[CaseListItem]
    total: int
    page: int
    per_page: int


def _load_all_cases_ordered() -> List[Dict[str, Any]]:
    path = eval_dir() / "cases_review.jsonl"
    return _load_jsonl(path)


@router.get("/api/cases/list", response_model=CaseListResponse)
async def cases_list(page: int = 1, per_page: int = 30) -> CaseListResponse:
    rows = _load_all_cases_ordered()
    total = len(rows)
    start = max(0, (page - 1) * per_page)
    chunk = rows[start : start + per_page]
    items: List[CaseListItem] = []
    for r in chunk:
        ch = r.get("attack_chain") or {}
        win = ch.get("winning_step")
        bl_adv = r.get("baseline_on_adversarial") or {}
        rb_adv = r.get("robust_on_adversarial")
        rb_lab = rb_adv.get("label") if isinstance(rb_adv, dict) else None
        bl_lab = bl_adv.get("label") if isinstance(bl_adv, dict) else None
        items.append(
            CaseListItem(
                case_id=str(r.get("case_id", "")),
                text_preview=str(r.get("text_preview", "")),
                true_label=str(r.get("true_label", "")),
                baseline_evaded=bool(r.get("baseline_evaded")),
                robust_recovered=r.get("robust_recovered_on_adversarial"),
                char_len=int(r.get("char_len", 0)),
                baseline_label_adv=str(bl_lab) if bl_lab is not None else None,
                robust_label_adv=str(rb_lab) if rb_lab is not None else None,
                winning_step=int(win) if isinstance(win, int) else None,
            )
        )
    return CaseListResponse(items=items, total=total, page=page, per_page=per_page)


@router.post("/api/text/compare", response_model=TextCompareResponse)
async def text_compare(payload: TextCompareRequest) -> TextCompareResponse:
    text = payload.text.strip()
    if len(text) < 2:
        raise HTTPException(status_code=400, detail="文本过短")

    base = build_text_detector()
    br = base.predict([text])[0]
    rb: Optional[Any] = None
    if robust_checkpoint_ready():
        rb = build_robust_text_detector().predict([text])[0]

    gold = payload.true_label.lower() if payload.true_label else None
    yt = 1 if gold == "ai" else 0 if gold == "human" else None

    def agrees(pred_label: str) -> Optional[bool]:
        if yt is None:
            return None
        return (1 if pred_label == "ai" else 0) == yt

    rob_block = None
    if rb:
        rob_block = ModelScoreBlock(label=rb.label_pred, ai_score=float(rb.ai_score))

    return TextCompareResponse(
        baseline=ModelScoreBlock(label=br.label_pred, ai_score=float(br.ai_score)),
        robust=rob_block,
        models_agree=rb is not None and br.label_pred == rb.label_pred,
        gold_label=gold,
        baseline_agrees_with_gold=agrees(br.label_pred),
        robust_agrees_with_gold=agrees(rb.label_pred) if rb else None,
    )


def _rebuild_eval_artifacts_sync(limit: int) -> Dict[str, Any]:
    from scripts.build_eval_artifacts import run_build_eval_artifacts

    with _eval_rebuild_lock:
        return run_build_eval_artifacts(
            test_path=prepared_dir() / "test.jsonl",
            pool_path=_root() / "data" / "vuln" / "adversarial_pool.jsonl",
            out_dir=eval_dir(),
            max_test=limit,
            max_pool=limit,
        )


@router.post("/api/dataset/rebuild-eval-artifacts", response_model=RebuildEvalArtifactsResponse)
async def rebuild_eval_artifacts_endpoint(
    payload: RebuildEvalArtifactsRequest,
) -> RebuildEvalArtifactsResponse:
    """
    一键生成评测产物：baseline_clean_metrics.json、cases_review.jsonl、
    dashboard_bundle.json 及 attack/robust 摘要，并刷新 data/vuln/summary.json。
    耗时与 limit 及模型推理次数成正比，大 limit 可能需数分钟。
    """
    try:
        stats = await asyncio.to_thread(_rebuild_eval_artifacts_sync, payload.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    invalidate_caches()
    return RebuildEvalArtifactsResponse(
        ok=True,
        message="已生成 dashboard_bundle、cases_review、baseline_clean_metrics 等，并更新 vuln/summary.json",
        stats=stats,
    )


def register_dataset_routes(app) -> None:
    app.include_router(router)
