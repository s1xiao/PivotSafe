import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from uuid import uuid4
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from detectors.text.model_loader import (
    build_robust_text_detector,
    build_text_detector,
    robust_checkpoint_ready,
)
from detectors.text.detector import TextDetectionResult
from backend.attack_routes import register_attack_routes
from backend.dataset_api import register_dataset_routes
from backend.batch_api import register_batch_routes
from detectors.image.detector import ImageAIDetector, ImageDetectionResult
from detectors.video.detector import VideoLightweightDetector, VideoDetectionResult


app = FastAPI(
    title="CCCC AI Content Detector",
    description="多模态 AI 生成内容检测与分析后端服务",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 后续可根据前端地址收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_TEXT_DETECTOR = None
_TEXT_DETECTOR_ROBUST = None


def get_text_detector():
    global _TEXT_DETECTOR
    if _TEXT_DETECTOR is None:
        _TEXT_DETECTOR = build_text_detector()
    return _TEXT_DETECTOR


def get_text_detector_for_variant(model_variant: Optional[str]) -> Any:
    """
    baseline：默认 / zh_v3 / baseline；robust：需 checkpoint 已存在。
    """
    v = (model_variant or "").strip().lower()
    if v == "robust":
        global _TEXT_DETECTOR_ROBUST
        if not robust_checkpoint_ready():
            raise HTTPException(
                status_code=503,
                detail="robust 模型未就绪：请先训练并保存到 configs/robust_text_detector.json 中 checkpoint_dir。",
            )
        if _TEXT_DETECTOR_ROBUST is None:
            _TEXT_DETECTOR_ROBUST = build_robust_text_detector()
        return _TEXT_DETECTOR_ROBUST
    return get_text_detector()


def _baseline_ai_score_single(text: str) -> float:
    d = get_text_detector()
    return float(d.score([text])[0])


_IMAGE_DETECTOR = None


def get_image_detector():
    global _IMAGE_DETECTOR
    if _IMAGE_DETECTOR is None:
        project_root = Path(__file__).resolve().parents[1]
        cfg_path = project_root / "configs" / "image_detector.json"
        cfg = {}
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8-sig") as f:
                cfg = json.load(f)

        checkpoint_value = cfg.get("checkpoint_path")
        checkpoint_path: Path | None = None
        if isinstance(checkpoint_value, str) and checkpoint_value.strip():
            checkpoint_path = Path(checkpoint_value.strip())
            if not checkpoint_path.is_absolute():
                checkpoint_path = (project_root / checkpoint_path).resolve()

        if checkpoint_path is None or not checkpoint_path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "图像检测模型未就绪：未找到可用 checkpoint。"
                    "请先运行 scripts/download_ssp_checkpoint.py 下载权重，"
                    "并检查 configs/image_detector.json 的 checkpoint_path。"
                ),
            )

        _IMAGE_DETECTOR = ImageAIDetector(
            checkpoint_path=checkpoint_path,
            device=cfg.get("device"),
            batch_size=cfg.get("batch_size", 8),
            threshold=cfg.get("threshold", 0.5),
            patch_size=cfg.get("patch_size", 32),
        )
    return _IMAGE_DETECTOR


class TextScoreRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    texts: List[str]
    # 预留字段，后续可用于切换中英模型或不同版本
    language: Optional[str] = "zh"
    model_variant: Optional[str] = "zh_v3"


class TextScoreItem(BaseModel):
    text: str
    ai_score: float
    label_pred: str


class TextScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    positive_label: str
    threshold: float
    results: List[TextScoreItem]


class WebCrawlJobRequest(BaseModel):
    """一键公开网页 真爬→检测→报告 的参数"""

    max_records: Optional[int] = None
    sites: Optional[str] = None
    download_images: Optional[bool] = True


class JobStatusResponse(BaseModel):
    status: str
    phase: str
    progress: int
    message: str


_job_state_lock = threading.Lock()
_job_state: dict[str, Any] = {
    "status": "idle",  # idle / running / success / error
    "phase": "",
    "progress": 0,
    "message": "",
}


def _set_job_state(*, status: str, phase: str, progress: int, message: str) -> None:
    with _job_state_lock:
        _job_state["status"] = status
        _job_state["phase"] = phase
        _job_state["progress"] = max(0, min(100, int(progress)))
        _job_state["message"] = message


def _get_job_state() -> dict[str, Any]:
    with _job_state_lock:
        return dict(_job_state)


def _get_job_python() -> str:
    """获取用于运行爬取任务的 Python 解释器路径，优先使用当前 conda 环境。"""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        py = Path(conda_prefix) / "bin" / "python"
        if py.exists():
            return str(py)
    return sys.executable


def _run_web_crawl_job(payload: WebCrawlJobRequest) -> None:
    """
    在后台线程中运行 scripts.run_simple_crawl_to_report，
    完成简单网页真爬→检测→数据报告，一路更新任务状态。
    """
    project_root = Path(__file__).resolve().parents[1]
    python_exe = _get_job_python()

    cmd: List[str] = [
        python_exe,
        "-m",
        "scripts.run_simple_crawl_to_report",
    ]
    if payload.max_records is not None:
        cmd.extend(["--max-records", str(payload.max_records)])
    if payload.sites:
        cmd.extend(["--sites", payload.sites])
    # run_simple_crawl_to_report 默认不下载图片；这里默认 True 时添加开关以启用图像检测
    if payload.download_images:
        cmd.append("--download-images")

    # 启动前先标记任务进入运行态
    _set_job_state(
        status="running",
        phase="init",
        progress=5,
        message="任务已启动，准备爬取公开网页…",
    )

    env = os.environ.copy()
    try:
        ret = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
        )
    except Exception as e:  # 极端情况下的错误保护
        _set_job_state(
            status="error",
            phase="error",
            progress=100,
            message=f"启动子进程失败: {e}",
        )
        return

    if ret.returncode != 0:
        stderr_tail = (ret.stderr or "").strip()
        stdout_tail = (ret.stdout or "").strip()
        # 优先展示 stderr（含 traceback），不足时补 stdout
        err = stderr_tail if stderr_tail else stdout_tail
        if len(err) > 600:
            err = err[:300] + "\n...\n" + err[-250:]
        msg = f"子进程退出码 {ret.returncode}。"
        if err:
            msg += f" {err}"
        _set_job_state(
            status="error",
            phase="error",
            progress=100,
            message=msg,
        )
        return

    # 成功时，尝试自动指向最新 basic_stats.json，便于 /api/summary 等读取
    out_dir = project_root / "data" / "pipeline_out_web"
    stats_path = out_dir / "basic_stats.json"
    if stats_path.exists():
        os.environ["STATS_JSON_PATH"] = str(stats_path.resolve())
        msg_suffix = f"统计与报告已生成于 {out_dir}。可设置 STATS_JSON_PATH 供 /api/summary 或 Next 前端读取。"
    else:
        msg_suffix = (
            "检测与报告已完成，但未找到 basic_stats.json，请检查输出目录或脚本运行日志。"
        )

    _set_job_state(
        status="success",
        phase="done",
        progress=100,
        message=msg_suffix,
    )


@app.get("/health")
async def health_check() -> dict:
    """健康检查接口，用于快速确认服务是否正常运行。"""
    return {"status": "ok"}


def _get_detection_results_path() -> Path | None:
    """返回 detection_results.jsonl 路径（与 basic_stats.json 同目录）。"""
    path = os.environ.get("STATS_JSON_PATH")
    if not path:
        return None
    p = Path(path).resolve().parent / "detection_results.jsonl"
    return p if p.exists() else None


@app.get("/api/summary")
async def summary() -> dict[str, Any]:
    """
    返回检测统计概览，供前端（如 Next `web/`）或工具读取。
    若环境变量 STATS_JSON_PATH 指向 basic_stats.json，则返回其内容；否则 stats 为 null。
    """
    out: dict[str, Any] = {"backend": "ok", "stats": None}
    path = os.environ.get("STATS_JSON_PATH")
    if path:
        p = Path(path)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    out["stats"] = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return out


@app.get("/api/detection/detail/{post_id}")
async def get_detection_detail(post_id: str) -> dict[str, Any]:
    """
    按 post_id 返回单条完整详情，供弹窗/抽屉展示。
    """
    results_path = _get_detection_results_path()
    if not results_path:
        raise HTTPException(status_code=404, detail="检测结果文件不存在")
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("post_id") == post_id:
                return rec
    raise HTTPException(status_code=404, detail=f"未找到 post_id={post_id}")


@app.get("/api/detection/score-distribution")
async def get_score_distribution() -> dict[str, Any]:
    """
    返回文本与图像分数分桶数据，供 ECharts 直方图使用。
    """
    results_path = _get_detection_results_path()
    if not results_path:
        return {"text_buckets": [], "image_buckets": []}
    text_scores: list[float] = []
    image_scores: list[float] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("text_ai_score")
            if ts is not None:
                try:
                    text_scores.append(float(ts))
                except (TypeError, ValueError):
                    pass
            imgs = rec.get("image_scores") or []
            if imgs:
                try:
                    image_scores.extend(float(x) for x in imgs)
                except (TypeError, ValueError):
                    pass
    bins = 20
    def bucket(scores: list[float], lo: float = 0.0, hi: float = 1.0) -> list[dict]:
        if not scores:
            return [{"bin_start": lo + i * (hi - lo) / bins, "bin_end": lo + (i + 1) * (hi - lo) / bins, "count": 0} for i in range(bins)]
        step = (hi - lo) / bins
        buckets = [{"bin_start": lo + i * step, "bin_end": lo + (i + 1) * step, "count": 0} for i in range(bins)]
        for s in scores:
            idx = min(max(0, int((s - lo) / step)), bins - 1)
            buckets[idx]["count"] += 1
        return buckets
    return {"text_buckets": bucket(text_scores), "image_buckets": bucket(image_scores) if image_scores else bucket([])}


@app.get("/api/serve/file")
async def serve_file(path: str) -> FileResponse:
    """
    安全代理本地文件，供前端预览 image_paths。仅允许 data/ 目录下的路径。
    """
    root = Path(__file__).resolve().parents[1]
    data_dir = (root / "data").resolve()
    p = Path(path).resolve()
    if not str(p).startswith(str(data_dir)):
        raise HTTPException(status_code=403, detail="仅允许访问 data/ 目录下的文件")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(p)


@app.get("/api/detection/results")
async def get_detection_results(
    page: int = 1,
    per_page: int = 20,
    platform: str | None = None,
) -> dict[str, Any]:
    """
    分页返回检测结果列表，供前端展示。支持按 platform 筛选。
    """
    results_path = _get_detection_results_path()
    if not results_path:
        return {"items": [], "total": 0, "page": page, "per_page": per_page}

    items: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if platform and rec.get("platform") != platform:
                continue
            items.append(rec)

    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]
    return {"items": page_items, "total": total, "page": page, "per_page": per_page}


@app.post("/api/jobs/web_crawl_pipeline/start")
async def start_web_crawl_pipeline(payload: WebCrawlJobRequest) -> dict[str, Any]:
    """
    启动“公开网页 真爬→检测→数据报告”一键流程。

    内部通过 scripts.run_simple_crawl_to_report 完成爬取与检测，在后台线程中执行。
    强制最少 1000 条：max_records 未指定时默认 1000，小于 1000 时返回 400。
    """
    max_records = payload.max_records if payload.max_records is not None else 1000
    if max_records < 1000:
        raise HTTPException(
            status_code=400,
            detail="抓取数量至少为 1000 条，请将 max_records 设置为 >= 1000。",
        )
    state = _get_job_state()
    if state.get("status") == "running":
        raise HTTPException(status_code=409, detail="已有网页爬取任务在运行，请稍后再试。")

    # 立即置为 running，后续由后台线程更新更细致状态
    _set_job_state(
        status="running",
        phase="init",
        progress=5,
        message="任务已启动，正在准备运行脚本…",
    )

    # 确保 payload 使用校验后的 max_records
    effective_payload = WebCrawlJobRequest(
        max_records=max_records,
        sites=payload.sites,
        download_images=payload.download_images,
    )
    t = threading.Thread(target=_run_web_crawl_job, args=(effective_payload,), daemon=True)
    t.start()

    return {"job_id": "web_crawl_pipeline"}


@app.get("/api/jobs/web_crawl_pipeline/status", response_model=JobStatusResponse)
async def web_crawl_pipeline_status() -> JobStatusResponse:
    """
    查询当前“公开网页 真爬→检测→报告”一键流程的任务状态。
    """
    state = _get_job_state()
    return JobStatusResponse(
        status=str(state.get("status", "idle")),
        phase=str(state.get("phase", "")),
        progress=int(state.get("progress", 0)),
        message=str(state.get("message", "")),
    )


def _ensure_sample_assets() -> dict[str, str]:
    """在项目 data/sample 下生成示例图片与视频（若不存在），返回其绝对路径。含 human/ai 双示例。"""
    base = Path(__file__).resolve().parents[1] / "data" / "sample"
    base.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    # 默认示例图（兼容旧逻辑）
    img_path = base / "sample.jpg"
    if not img_path.exists():
        try:
            from PIL import Image
            img = Image.new("RGB", (64, 64), color=(120, 120, 120))
            img.save(img_path)
        except Exception:
            pass
    if img_path.exists():
        out["image_path"] = str(img_path.resolve())
    # human/ai 示例图（人眼可区分：真人照片 vs AI 风格）
    for name, key in [("human.jpg", "human_image_path"), ("ai.jpg", "ai_image_path")]:
        p = base / name
        if p.exists():
            out[key] = str(p.resolve())
        elif out.get("image_path"):
            out[key] = out["image_path"]
    # 默认示例视频
    video_path = base / "sample.mp4"
    if not video_path.exists():
        try:
            import cv2
            import numpy as np
            w, h = 160, 120
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (w, h))
            for i in range(5):
                frame = np.full((h, w, 3), (i * 50) % 256, dtype=np.uint8)
                writer.write(frame)
            writer.release()
        except Exception:
            pass
    if video_path.exists():
        out["video_path"] = str(video_path.resolve())
    # human/ai 示例视频
    for name, key in [("human.mp4", "human_video_path"), ("ai.mp4", "ai_video_path")]:
        p = base / name
        if p.exists():
            out[key] = str(p.resolve())
        elif out.get("video_path"):
            out[key] = out["video_path"]
    return out


@app.get("/api/sample/paths")
async def sample_paths() -> dict[str, Any]:
    """返回可用于在线检测的示例文件路径（服务器本地），便于前端「示例」填充。含 human/ai 双示例。"""
    paths = _ensure_sample_assets()
    return {
        "image_path": paths.get("image_path", ""),
        "video_path": paths.get("video_path", ""),
        "human_image_path": paths.get("human_image_path", paths.get("image_path", "")),
        "ai_image_path": paths.get("ai_image_path", paths.get("image_path", "")),
        "human_video_path": paths.get("human_video_path", paths.get("video_path", "")),
        "ai_video_path": paths.get("ai_video_path", paths.get("video_path", "")),
    }


@app.get("/api/sample/image")
@app.get("/api/sample/image/{sample_type}")
async def sample_image(sample_type: str | None = None) -> FileResponse:
    """
    返回示例图片内容，供前端预览使用。sample_type 为 human 或 ai 时返回对应示例。
    """
    paths = _ensure_sample_assets()
    if sample_type == "human":
        img = paths.get("human_image_path")
    elif sample_type == "ai":
        img = paths.get("ai_image_path")
    else:
        img = paths.get("image_path")
    if not img or not Path(img).exists():
        raise HTTPException(status_code=404, detail="sample image not found")
    return FileResponse(img)


@app.get("/api/sample/video")
@app.get("/api/sample/video/{sample_type}")
async def sample_video(sample_type: str | None = None) -> FileResponse:
    """
    返回示例视频内容，供前端预览使用。sample_type 为 human 或 ai 时返回对应示例。
    """
    paths = _ensure_sample_assets()
    if sample_type == "human":
        vid = paths.get("human_video_path")
    elif sample_type == "ai":
        vid = paths.get("ai_video_path")
    else:
        vid = paths.get("video_path")
    if not vid:
        raise HTTPException(status_code=404, detail="sample video not found")
    # 让浏览器自己根据后缀处理 content-type
    return FileResponse(vid)


@app.get("/api/charts/distribution")
async def chart_distribution():
    """返回 basic_stats 同目录下的 score_distribution.png，供前端展示。需配置 STATS_JSON_PATH。"""
    path = os.environ.get("STATS_JSON_PATH")
    if not path:
        raise HTTPException(status_code=404, detail="STATS_JSON_PATH not set")
    p = Path(path).resolve().parent / "score_distribution.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="score_distribution.png not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/report")
async def get_report():
    """返回 basic_stats 同目录下的 report.html，供前端新窗口打开。需配置 STATS_JSON_PATH。"""
    path = os.environ.get("STATS_JSON_PATH")
    if not path:
        raise HTTPException(status_code=404, detail="STATS_JSON_PATH not set")
    p = Path(path).resolve().parent / "report.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="report.html not found")
    return FileResponse(p, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """避免浏览器请求 /favicon.ico 时返回 404。"""
    return Response(status_code=204)


@app.get("/")
async def root() -> dict:
    """API 根路径；比赛演示前端为独立 Next 应用（默认 http://localhost:3000）。"""
    return {
        "message": "CCCC AI 生成内容检测后端已启动",
        "docs": "/docs",
        "redoc": "/redoc",
        "web_ui": "本地请运行 web/: npm run dev → http://localhost:3000",
    }


@app.post("/api/text/score", response_model=TextScoreResponse)
async def text_score(payload: TextScoreRequest) -> TextScoreResponse:
    """
    文本 AI 生成内容检测接口。

    默认 baseline（configs/text_detector.json）。请求体 `model_variant` 为 `robust` 时使用 robust checkpoint。
    """
    detector = get_text_detector_for_variant(payload.model_variant)
    results: List[TextDetectionResult] = detector.predict(payload.texts)
    items = [
        TextScoreItem(
            text=r.text,
            ai_score=r.ai_score,
            label_pred=r.label_pred,
        )
        for r in results
    ]
    return TextScoreResponse(
        model_name=detector.model_name,
        positive_label=detector.positive_label,
        threshold=detector.threshold,
        results=items,
    )


_CLOSERS = set("”’\"'）)]】》」』")
_SENTENCE_ENDERS = set("。！？!?；;")


class ArticleScoreRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    text: str
    model_variant: Optional[str] = "zh_v3"
    min_sentence_len: int = 4


class ArticleSentenceItem(BaseModel):
    index: int
    text: str
    ai_score: float
    label_pred: str


class ArticleScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    positive_label: str
    threshold: float
    total_sentences: int
    ai_sentences: int
    ai_rate: float
    avg_ai_score: float
    max_ai_score: float
    sentences: List[ArticleSentenceItem]


class ArticleUploadParseResponse(BaseModel):
    filename: str
    chars: int
    text: str


def _split_article_sentences(text: str, *, min_sentence_len: int = 4) -> list[str]:
    """
    文章断句：
    1) 以中英文句末标点分句；
    2) 遇到右引号/右括号时并入当前句；
    3) 小数点（如 3.14）不切句；
    4) 对过短碎句进行合并，减少噪声。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    raw_sentences: list[str] = []
    for para in re.split(r"\n+", normalized):
        para = para.strip()
        if not para:
            continue
        buf: list[str] = []
        i = 0
        n = len(para)
        while i < n:
            ch = para[i]
            buf.append(ch)
            should_break = ch in _SENTENCE_ENDERS
            if ch == ".":
                prev_c = para[i - 1] if i > 0 else ""
                next_c = para[i + 1] if i + 1 < n else ""
                is_decimal_dot = prev_c.isdigit() and next_c.isdigit()
                if not is_decimal_dot and (next_c == "" or next_c.isspace() or next_c in _CLOSERS):
                    should_break = True

            if should_break:
                j = i + 1
                while j < n and para[j] in _CLOSERS:
                    buf.append(para[j])
                    j += 1
                sentence = "".join(buf).strip()
                if sentence:
                    raw_sentences.append(sentence)
                buf = []
                i = j
                continue
            i += 1

        tail = "".join(buf).strip()
        if tail:
            raw_sentences.append(tail)

    compact = [" ".join(s.split()) for s in raw_sentences if s.strip()]
    if not compact:
        return []

    min_len = max(1, int(min_sentence_len))

    def content_len(s: str) -> int:
        return len(re.sub(r"\s+", "", s))

    merged: list[str] = []
    i = 0
    while i < len(compact):
        cur = compact[i]
        cur_len = content_len(cur)
        if cur_len < min_len and i + 1 < len(compact):
            cur = cur + compact[i + 1]
            i += 1
        elif cur_len < min_len and merged:
            merged[-1] = merged[-1] + cur
            i += 1
            continue
        merged.append(cur)
        i += 1

    return [x for x in merged if x]



_ARTICLE_UPLOAD_SUFFIX_WHITELIST = {".pdf", ".doc", ".docx", ".txt"}


def _extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析依赖缺失（pypdf）：{e}") from e
    reader = PdfReader(str(path))
    texts: list[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()


def _extract_text_from_docx(path: Path) -> str:
    try:
        from docx import Document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word 解析依赖缺失（python-docx）：{e}") from e
    doc = Document(str(path))
    lines = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(lines).strip()


def _extract_text_from_doc(path: Path) -> str:
    try:
        cp = subprocess.run(
            ["antiword", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail="当前环境未安装 antiword，暂无法解析 .doc，请转为 .docx 后上传。",
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise HTTPException(status_code=400, detail=f".doc 解析失败：{stderr or 'unknown error'}") from e
    return (cp.stdout or "").strip()


def _extract_text_from_article_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(path)
    if suffix == ".docx":
        return _extract_text_from_docx(path)
    if suffix == ".doc":
        return _extract_text_from_doc(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    raise HTTPException(status_code=400, detail=f"unsupported article file type: {suffix}")

@app.post("/api/article/score", response_model=ArticleScoreResponse)
async def article_score(payload: ArticleScoreRequest) -> ArticleScoreResponse:
    """
    文章级检测：自动断句后逐句判断是否为 AIGC，并返回统计指标与逐句结果。
    """
    text = payload.text.strip()
    if len(text) < 2:
        raise HTTPException(status_code=400, detail="文章内容过短")

    sentences = _split_article_sentences(
        text,
        min_sentence_len=max(1, min(64, int(payload.min_sentence_len))),
    )
    if not sentences:
        raise HTTPException(status_code=400, detail="未能切分出有效句子")

    detector = get_text_detector_for_variant(payload.model_variant)
    results: List[TextDetectionResult] = detector.predict(sentences)
    positive_label = str(detector.positive_label).lower()

    items: list[ArticleSentenceItem] = []
    ai_sentences = 0
    scores: list[float] = []
    for idx, r in enumerate(results, start=1):
        score = float(r.ai_score)
        label = str(r.label_pred)
        scores.append(score)
        if label.lower() == positive_label:
            ai_sentences += 1
        items.append(
            ArticleSentenceItem(
                index=idx,
                text=r.text,
                ai_score=score,
                label_pred=label,
            )
        )

    total = len(items)
    avg_score = sum(scores) / total if total else 0.0
    max_score = max(scores) if scores else 0.0

    return ArticleScoreResponse(
        model_name=detector.model_name,
        positive_label=detector.positive_label,
        threshold=detector.threshold,
        total_sentences=total,
        ai_sentences=ai_sentences,
        ai_rate=round(ai_sentences / total, 4) if total else 0.0,
        avg_ai_score=round(avg_score, 4),
        max_ai_score=round(max_score, 4),
        sentences=items,
    )



@app.post("/api/article/upload-parse", response_model=ArticleUploadParseResponse)
async def article_upload_parse(file: UploadFile = File(...)) -> ArticleUploadParseResponse:
    """
    文章文档上传解析：支持 PDF / DOC / DOCX / TXT，返回提取后的纯文本供前端检测。
    """
    if file is None:
        raise HTTPException(status_code=400, detail="no file uploaded")
    tmp_dir = Path(tempfile.mkdtemp(prefix="cccc_article_upload_"))
    saved_path: Path | None = None
    try:
        saved_path = _save_upload_file(
            file,
            tmp_dir,
            allowed_suffixes=_ARTICLE_UPLOAD_SUFFIX_WHITELIST,
            fallback_suffix=".txt",
        )
        text = _extract_text_from_article_file(saved_path)
        if len(text.strip()) < 2:
            raise HTTPException(status_code=400, detail="未从文档中提取到有效正文")
        if len(text) > 120000:
            text = text[:120000]
        return ArticleUploadParseResponse(
            filename=saved_path.name,
            chars=len(text),
            text=text,
        )
    finally:
        try:
            await file.close()
        except Exception:
            pass
        if saved_path and saved_path.exists():
            saved_path.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

class ImageScoreRequest(BaseModel):
    image_paths: List[str]


class ImageScoreItem(BaseModel):
    path: str
    ai_score: float
    label_pred: str


class ImageScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    threshold: float
    results: List[ImageScoreItem]


_IMAGE_UPLOAD_SUFFIX_WHITELIST = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_UPLOAD_SUFFIX_WHITELIST = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _save_upload_file(
    upload: UploadFile,
    target_dir: Path,
    *,
    allowed_suffixes: set[str],
    fallback_suffix: str,
) -> Path:
    original_name = upload.filename or ""
    suffix = Path(original_name).suffix.lower()
    if not suffix:
        suffix = fallback_suffix
    if suffix not in allowed_suffixes:
        allow = ", ".join(sorted(allowed_suffixes))
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {original_name or 'unknown'} (allowed: {allow})",
        )
    name = Path(original_name).name if original_name else f"upload{suffix}"
    out_path = target_dir / name
    if out_path.exists():
        out_path = target_dir / f"{out_path.stem}_{uuid4().hex[:8]}{suffix}"
    with out_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return out_path


@app.post("/api/image/score", response_model=ImageScoreResponse)
async def image_score(payload: ImageScoreRequest) -> ImageScoreResponse:
    """
    图像 AI 生成内容检测接口。
    当前基于 SSP（bcmi/SSP-AI-Generated-Image-Detection），支持本地图片路径。
    """
    detector = get_image_detector()
    results: List[ImageDetectionResult] = detector.predict(payload.image_paths)
    items = [
        ImageScoreItem(path=r.path, ai_score=r.ai_score, label_pred=r.label_pred)
        for r in results
    ]
    return ImageScoreResponse(
        model_name="SSP",
        threshold=detector.threshold,
        results=items,
    )


@app.post("/api/image/upload-score", response_model=ImageScoreResponse)
async def image_upload_score(files: List[UploadFile] = File(...)) -> ImageScoreResponse:
    """
    图像上传检测接口：接收浏览器上传文件，临时落盘后调用 SSP 检测并清理文件。
    """
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")
    tmp_dir = Path(tempfile.mkdtemp(prefix="cccc_image_upload_"))
    saved_paths: list[Path] = []
    detector = get_image_detector()
    try:
        for upload in files:
            saved = _save_upload_file(
                upload,
                tmp_dir,
                allowed_suffixes=_IMAGE_UPLOAD_SUFFIX_WHITELIST,
                fallback_suffix=".jpg",
            )
            saved_paths.append(saved)
        results: List[ImageDetectionResult] = detector.predict(saved_paths)
        names = [Path(x.filename).name if x.filename else p.name for x, p in zip(files, saved_paths)]
        items = [
            ImageScoreItem(path=name, ai_score=r.ai_score, label_pred=r.label_pred)
            for name, r in zip(names, results)
        ]
        return ImageScoreResponse(
            model_name="SSP",
            threshold=detector.threshold,
            results=items,
        )
    finally:
        for upload in files:
            try:
                await upload.close()
            except Exception:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


_VIDEO_DETECTOR = None


def get_video_detector() -> VideoLightweightDetector:
    global _VIDEO_DETECTOR
    if _VIDEO_DETECTOR is None:
        _VIDEO_DETECTOR = VideoLightweightDetector(
            image_detector=get_image_detector(),
            num_frames=5,
            threshold=get_image_detector().threshold,
            aggregate="mean",
        )
    return _VIDEO_DETECTOR


class VideoScoreRequest(BaseModel):
    video_path: str


class VideoScoreResponse(BaseModel):
    video_path: str
    frame_scores: List[float]
    mean_score: float
    max_score: float
    label_pred: str
    num_frames: int


@app.post("/api/video/score", response_model=VideoScoreResponse)
async def video_score(payload: VideoScoreRequest) -> VideoScoreResponse:
    """
    轻量视频检测：抽帧后走图像检测，聚合得到整段视频的 AI 可疑度。
    请求体为本地视频路径 video_path。
    """
    path = Path(payload.video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {payload.video_path}")
    detector = get_video_detector()
    result: VideoDetectionResult = detector.predict(path)
    return VideoScoreResponse(
        video_path=result.video_path,
        frame_scores=result.frame_scores,
        mean_score=round(result.mean_score, 4),
        max_score=round(result.max_score, 4),
        label_pred=result.label_pred,
        num_frames=result.num_frames,
    )


@app.post("/api/video/upload-score", response_model=VideoScoreResponse)
async def video_upload_score(file: UploadFile = File(...)) -> VideoScoreResponse:
    """
    视频上传检测接口：接收浏览器上传文件，临时落盘后抽帧检测并清理文件。
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="cccc_video_upload_"))
    try:
        saved_path = _save_upload_file(
            file,
            tmp_dir,
            allowed_suffixes=_VIDEO_UPLOAD_SUFFIX_WHITELIST,
            fallback_suffix=".mp4",
        )
        detector = get_video_detector()
        result: VideoDetectionResult = detector.predict(saved_path)
        return VideoScoreResponse(
            video_path=Path(file.filename).name if file.filename else result.video_path,
            frame_scores=result.frame_scores,
            mean_score=round(result.mean_score, 4),
            max_score=round(result.max_score, 4),
            label_pred=result.label_pred,
            num_frames=result.num_frames,
        )
    finally:
        try:
            await file.close()
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


register_attack_routes(app, _baseline_ai_score_single)
register_dataset_routes(app)
register_batch_routes(
    app,
    get_text_detector_for_variant=get_text_detector_for_variant,
    get_image_detector=get_image_detector,
    get_video_detector=get_video_detector,
    split_article_sentences=_split_article_sentences,
    extract_text_from_article_file=_extract_text_from_article_file,
)


@app.get("/api/vuln/summary")
async def vuln_summary() -> dict[str, Any]:
    """
    返回脆弱性分析 JSON。默认读取环境变量 VULN_SUMMARY_JSON 或 data/vuln/summary.json。
    """
    root = Path(__file__).resolve().parents[1]
    p = os.environ.get("VULN_SUMMARY_JSON")
    path = Path(p) if p else root / "data" / "vuln" / "summary.json"
    if not path.is_file():
        return {
            "ready": False,
            "message": "尚未生成 summary.json，请运行 scripts/run_vulnerability_analysis.py",
            "path": str(path),
        }
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ready": True, "path": str(path.resolve()), "data": data}


# 历史 static 前端目录 frontend/ 保留在仓库中供参考，不再挂载到 /dashboard/；请使用 web/ Next 应用。

















