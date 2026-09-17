import csv
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict


_TEXT_BATCH_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".json", ".csv", ".tsv", ".pdf", ".doc", ".docx"}
_IMAGE_BATCH_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_BATCH_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_BATCH_ZIP_SUFFIXES = {".zip"}
_BATCH_MAX_FILES = max(1, int(os.environ.get("CCCC_BATCH_MAX_FILES", "400")))
_TEXT_DETAIL_MAX_SENTENCES = max(20, int(os.environ.get("CCCC_BATCH_TEXT_MAX_SENTENCES", "400")))


class BatchResultItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    file_name: str
    status: str
    label_pred: Optional[str] = None
    ai_score: Optional[float] = None
    message: Optional[str] = None
    detail: Optional[dict[str, Any]] = None


class BatchEngineStatus(BaseModel):
    status: str
    message: str
    path: Optional[str] = None


class BatchProcessResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    job_id: str
    modality: str
    total_files: int
    success_files: int
    failed_files: int
    ai_files: int
    human_files: int
    avg_ai_score: float
    results: list[BatchResultItem]
    artifacts: dict[str, str]
    engines: dict[str, BatchEngineStatus]


def _tail_text(s: str, max_len: int = 320) -> str:
    t = (s or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len // 2] + " ... " + t[-max_len // 2 :]



def _read_text_file_best_effort(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc).strip()
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _extract_failure_reason(stdout: str, stderr: str) -> str:
    text = "\n".join(x for x in [stderr or "", stdout or ""] if x)
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    if not lines:
        return "no output"
    keys = ("error", "exception", "traceback", "caused by", "failed")
    hits = [ln for ln in lines if any(k in ln.lower() for k in keys)]
    if hits:
        return _tail_text(" | ".join(hits[-8:]), 800)
    return _tail_text(" | ".join(lines[-10:]), 800)


def _local_postprocess_fallback(results_jsonl: Path, output_dir: Path, job_id: str, modality: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    success = 0
    ai_files = 0
    score_sum = 0.0

    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("status", "")) != "success":
                continue
            success += 1
            label = str(rec.get("label_pred", "")).lower()
            if label == "ai":
                ai_files += 1
            try:
                score_sum += float(rec.get("ai_score") or 0.0)
            except Exception:
                pass

    failed = max(0, total - success)
    human_files = max(0, success - ai_files)
    avg_ai = (score_sum / success) if success else 0.0

    summary = {
        "job_id": job_id,
        "modality": modality,
        "engine": "local_fallback",
        "total_files": int(total),
        "success_files": int(success),
        "failed_files": int(failed),
        "ai_files": int(ai_files),
        "human_files": int(human_files),
        "avg_ai_score": round(avg_ai, 4),
    }
    out = output_dir / "spark_summary.json"
    with out.open("w", encoding="utf-8") as wf:
        json.dump(summary, wf, ensure_ascii=False, indent=2)
    return out
def _allowed_suffixes(modality: str) -> set[str]:
    m = modality.strip().lower()
    if m == "text":
        return _TEXT_BATCH_SUFFIXES
    if m == "image":
        return _IMAGE_BATCH_SUFFIXES
    if m == "video":
        return _VIDEO_BATCH_SUFFIXES
    raise HTTPException(status_code=400, detail=f"unsupported modality: {modality}")


def _decode_zip_member_name(info: zipfile.ZipInfo) -> str:
    """
    兼容常见中文压缩包文件名编码。
    """
    name = info.filename or ""
    if not name:
        return ""

    if info.flag_bits & 0x800:
        fixed = name
    else:
        raw = name.encode("cp437", errors="replace")
        fixed = name
        for enc in ("utf-8", "gb18030", "gbk"):
            try:
                fixed = raw.decode(enc)
                break
            except Exception:
                continue

    fixed = fixed.replace("\\", "/").strip()
    return Path(fixed).name


def _extract_zip_selected_files(zip_path: Path, out_dir: Path, modality: str) -> list[Path]:
    allow = _allowed_suffixes(modality)
    selected: list[Path] = []
    seen_names: set[str] = set()
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = [x for x in zf.infolist() if not x.is_dir()]
            for info in infos:
                decoded_name = _decode_zip_member_name(info)
                suffix = Path(decoded_name).suffix.lower()
                if suffix not in allow:
                    continue
                safe_name = decoded_name
                if not safe_name:
                    continue
                if safe_name in seen_names:
                    safe_name = f"{Path(safe_name).stem}_{uuid4().hex[:6]}{suffix}"
                seen_names.add(safe_name)
                target = out_dir / safe_name
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                selected.append(target)
                if len(selected) > _BATCH_MAX_FILES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"压缩包内目标文件数量超过上限（{_BATCH_MAX_FILES}）",
                    )
    except zipfile.BadZipFile as e:
        raise HTTPException(status_code=400, detail=f"无效的 zip 文件：{e}") from e
    return selected


def _detect_text_batch(
    files: list[Path],
    *,
    model_variant: str,
    min_sentence_len: int,
    get_text_detector_for_variant: Callable[[Optional[str]], Any],
    split_article_sentences: Callable[..., list[str]],
    extract_text_from_article_file: Callable[[Path], str],
) -> list[BatchResultItem]:
    detector = get_text_detector_for_variant(model_variant)
    positive_label = str(detector.positive_label).lower()
    threshold = float(detector.threshold)
    items: list[BatchResultItem] = []

    for p in files:
        try:
            suffix = p.suffix.lower()
            if suffix in {".pdf", ".doc", ".docx", ".txt"}:
                text = extract_text_from_article_file(p)
            else:
                text = _read_text_file_best_effort(p)
            if len(text) < 2:
                raise ValueError("文件文本内容过短")

            sentences = split_article_sentences(
                text,
                min_sentence_len=max(1, min(64, int(min_sentence_len))),
            )
            if not sentences:
                raise ValueError("未切分出有效句子")

            results = detector.predict(sentences)
            scores = [float(r.ai_score) for r in results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores) if scores else 0.0
            ai_sentences = sum(1 for r in results if str(r.label_pred).lower() == positive_label)
            ai_rate = ai_sentences / len(results)

            sentence_items: list[dict[str, Any]] = []
            for idx, r in enumerate(results, start=1):
                s_label = str(r.label_pred)
                sentence_items.append(
                    {
                        "index": idx,
                        "text": str(r.text),
                        "ai_score": round(float(r.ai_score), 4),
                        "label_pred": s_label,
                        "is_ai": s_label.lower() == positive_label,
                    }
                )

            truncated = len(sentence_items) > _TEXT_DETAIL_MAX_SENTENCES
            shown_sentences = sentence_items[:_TEXT_DETAIL_MAX_SENTENCES]
            label_pred = str(detector.positive_label) if ai_rate >= 0.5 else "human"

            items.append(
                BatchResultItem(
                    file_name=p.name,
                    status="success",
                    label_pred=label_pred,
                    ai_score=round(avg_score, 4),
                    detail={
                        "total_sentences": len(results),
                        "ai_sentences": ai_sentences,
                        "ai_rate": round(ai_rate, 4),
                        "avg_ai_score": round(avg_score, 4),
                        "max_ai_score": round(max_score, 4),
                        "threshold": round(threshold, 4),
                        "sentences": shown_sentences,
                        "sentences_truncated": truncated,
                    },
                )
            )
        except Exception as e:
            items.append(
                BatchResultItem(
                    file_name=p.name,
                    status="error",
                    message=_tail_text(str(e)),
                )
            )
    return items


def _detect_image_batch(files: list[Path], *, get_image_detector: Callable[[], Any]) -> list[BatchResultItem]:
    detector = get_image_detector()
    items: list[BatchResultItem] = []
    for p in files:
        try:
            one = detector.predict([p])[0]
            items.append(
                BatchResultItem(
                    file_name=p.name,
                    status="success",
                    label_pred=str(one.label_pred),
                    ai_score=round(float(one.ai_score), 4),
                )
            )
        except Exception as e:
            items.append(BatchResultItem(file_name=p.name, status="error", message=_tail_text(str(e))))
    return items


def _detect_video_batch(files: list[Path], *, get_video_detector: Callable[[], Any]) -> list[BatchResultItem]:
    detector = get_video_detector()
    items: list[BatchResultItem] = []
    for p in files:
        try:
            r = detector.predict(p)
            items.append(
                BatchResultItem(
                    file_name=p.name,
                    status="success",
                    label_pred=str(r.label_pred),
                    ai_score=round(float(r.mean_score), 4),
                    detail={
                        "max_score": round(float(r.max_score), 4),
                        "num_frames": int(r.num_frames),
                    },
                )
            )
        except Exception as e:
            items.append(BatchResultItem(file_name=p.name, status="error", message=_tail_text(str(e))))
    return items


def _summarize(items: list[BatchResultItem]) -> dict[str, Any]:
    success = [x for x in items if x.status == "success"]
    failed = [x for x in items if x.status != "success"]
    ai_count = sum(1 for x in success if (x.label_pred or "").lower() == "ai")
    human_count = sum(1 for x in success if (x.label_pred or "").lower() != "ai")
    avg_ai = sum(float(x.ai_score or 0.0) for x in success) / len(success) if success else 0.0
    return {
        "total_files": len(items),
        "success_files": len(success),
        "failed_files": len(failed),
        "ai_files": ai_count,
        "human_files": human_count,
        "avg_ai_score": round(avg_ai, 4),
    }


def _write_local_artifacts(job_dir: Path, items: list[BatchResultItem], summary: dict[str, Any]) -> dict[str, str]:
    job_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = job_dir / "results.jsonl"
    csv_path = job_dir / "results.csv"
    summary_path = job_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

    # 使用 utf-8-sig，兼容 Windows/Excel 的中文显示
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "status", "label_pred", "ai_score", "message", "detail_json"])
        for item in items:
            w.writerow(
                [
                    item.file_name,
                    item.status,
                    item.label_pred or "",
                    "" if item.ai_score is None else item.ai_score,
                    item.message or "",
                    json.dumps(item.detail or {}, ensure_ascii=False),
                ]
            )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "results_jsonl": str(jsonl_path.resolve()),
        "results_csv": str(csv_path.resolve()),
        "summary_json": str(summary_path.resolve()),
    }


def _run_spark_postprocess(
    *,
    project_root: Path,
    job_id: str,
    modality: str,
    results_jsonl: Path,
    output_dir: Path,
    enable_spark: bool,
) -> BatchEngineStatus:
    if not enable_spark:
        return BatchEngineStatus(status="skipped", message="已关闭 Spark 后处理")

    spark_submit = shutil.which("spark-submit")
    fallback_enable = os.environ.get("CCCC_BATCH_SPARK_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "y"}

    if not spark_submit:
        if fallback_enable:
            summary_path = _local_postprocess_fallback(results_jsonl, output_dir, job_id, modality)
            return BatchEngineStatus(
                status="degraded",
                message="未找到 spark-submit，已回退为本地聚合后处理",
                path=str(summary_path.resolve()),
            )
        return BatchEngineStatus(status="unavailable", message="未找到 spark-submit")

    script = project_root / "scripts" / "batch_spark_postprocess.py"
    if not script.exists():
        if fallback_enable:
            summary_path = _local_postprocess_fallback(results_jsonl, output_dir, job_id, modality)
            return BatchEngineStatus(
                status="degraded",
                message=f"Spark 脚本缺失，已回退本地聚合: {script}",
                path=str(summary_path.resolve()),
            )
        return BatchEngineStatus(status="error", message=f"Spark 脚本不存在: {script}")

    configured_master = os.environ.get("CCCC_BATCH_SPARK_MASTER", "").strip()
    masters = [configured_master] if configured_master else ["yarn", "local[*]"]
    timeout_s = max(60, int(os.environ.get("CCCC_BATCH_SPARK_TIMEOUT", "900")))
    tried_msgs: list[str] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for master in masters:
        cmd = [
            spark_submit,
            "--master",
            master,
            str(script),
            "--input-jsonl",
            str(results_jsonl),
            "--output-dir",
            str(output_dir),
            "--job-id",
            job_id,
            "--modality",
            modality,
        ]
        try:
            cp = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except Exception as e:
            tried_msgs.append(f"{master}: {_tail_text(str(e))}")
            continue

        log_file = output_dir / f"spark_attempt_{master.replace('*', 'star').replace('[', '').replace(']', '').replace('/', '_')}.log"
        try:
            with log_file.open("w", encoding="utf-8") as lf:
                lf.write("# CMD\n")
                lf.write(" ".join(cmd) + "\n\n")
                lf.write("# STDOUT\n")
                lf.write(cp.stdout or "")
                lf.write("\n\n# STDERR\n")
                lf.write(cp.stderr or "")
        except Exception:
            pass

        if cp.returncode == 0:
            summary_path = output_dir / "spark_summary.json"
            if summary_path.exists():
                return BatchEngineStatus(
                    status="ok",
                    message=f"Spark 后处理完成（master={master}）",
                    path=str(summary_path.resolve()),
                )
            # 进程返回成功但未产出摘要，视为异常，继续尝试下一个 master
            tried_msgs.append(f"{master}: returncode=0 but spark_summary.json missing")
            continue

        fail_reason = _extract_failure_reason(cp.stdout, cp.stderr)
        tried_msgs.append(f"{master}: {fail_reason}")

    if fallback_enable:
        summary_path = _local_postprocess_fallback(results_jsonl, output_dir, job_id, modality)
        return BatchEngineStatus(
            status="degraded",
            message=f"Spark 执行失败，已回退本地聚合：{' | '.join(tried_msgs)}",
            path=str(summary_path.resolve()),
        )

    return BatchEngineStatus(status="error", message=f"Spark 任务失败：{' | '.join(tried_msgs)}")


def _sync_hdfs(job_id: str, local_paths: list[Path], *, enable_hdfs: bool) -> BatchEngineStatus:
    if not enable_hdfs:
        return BatchEngineStatus(status="skipped", message="已关闭 HDFS 同步")
    hdfs = shutil.which("hdfs")
    if not hdfs:
        return BatchEngineStatus(status="unavailable", message="未找到 hdfs 命令")

    remote_base = os.environ.get("CCCC_BATCH_HDFS_DIR", "/data/cccc/ads/batch_detect").rstrip("/")
    remote_dir = f"{remote_base}/{job_id}"

    try:
        mkdir_cp = subprocess.run([hdfs, "dfs", "-mkdir", "-p", remote_dir], capture_output=True, text=True)
        if mkdir_cp.returncode != 0:
            raise RuntimeError(mkdir_cp.stderr or mkdir_cp.stdout or "mkdir failed")

        for p in local_paths:
            put_cp = subprocess.run([hdfs, "dfs", "-put", "-f", str(p), remote_dir], capture_output=True, text=True)
            if put_cp.returncode != 0:
                raise RuntimeError(put_cp.stderr or put_cp.stdout or f"put failed: {p.name}")
    except Exception as e:
        return BatchEngineStatus(status="error", message=f"HDFS 同步失败：{_tail_text(str(e))}")

    return BatchEngineStatus(status="ok", message="已同步到 HDFS", path=remote_dir)


def _hive_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _sync_hive_summary(
    *,
    job_id: str,
    modality: str,
    summary: dict[str, Any],
    spark_status: str,
    hdfs_status: str,
    enable_hive: bool,
) -> BatchEngineStatus:
    if not enable_hive:
        return BatchEngineStatus(status="skipped", message="已关闭 Hive 写入")

    spark_sql = shutil.which("spark-sql")
    beeline = shutil.which("beeline")
    if not spark_sql and not beeline:
        return BatchEngineStatus(status="unavailable", message="未找到 spark-sql/beeline 命令")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_id_e = _hive_escape(job_id)
    modality_e = _hive_escape(modality)
    spark_e = _hive_escape(spark_status)
    hdfs_e = _hive_escape(hdfs_status)
    now_e = _hive_escape(now)
    total = int(summary.get("total_files", 0))
    success = int(summary.get("success_files", 0))
    failed = int(summary.get("failed_files", 0))
    ai_files = int(summary.get("ai_files", 0))
    human_files = int(summary.get("human_files", 0))
    avg_ai_score = float(summary.get("avg_ai_score", 0.0))

    sql = f"""
CREATE DATABASE IF NOT EXISTS cccc_dw;
CREATE TABLE IF NOT EXISTS cccc_dw.batch_detect_job_summary (
  job_id STRING,
  modality STRING,
  total_files INT,
  success_files INT,
  failed_files INT,
  ai_files INT,
  human_files INT,
  avg_ai_score DOUBLE,
  spark_status STRING,
  hdfs_status STRING,
  created_at STRING
)
STORED AS ORC;
INSERT INTO TABLE cccc_dw.batch_detect_job_summary VALUES (
  '{job_id_e}',
  '{modality_e}',
  {total},
  {success},
  {failed},
  {ai_files},
  {human_files},
  {avg_ai_score},
  '{spark_e}',
  '{hdfs_e}',
  '{now_e}'
);
"""

    fail_msgs: list[str] = []

    if spark_sql:
        spark_cmd = [spark_sql]
        spark_master = os.environ.get("CCCC_BATCH_SPARK_MASTER", "").strip()
        if spark_master:
            spark_cmd.extend(["--master", spark_master])
        spark_cmd.extend(["-e", sql])
        try:
            cp = subprocess.run(spark_cmd, capture_output=True, text=True)
            if cp.returncode == 0:
                return BatchEngineStatus(status="ok", message="Hive 摘要写入成功（via spark-sql）", path="cccc_dw.batch_detect_job_summary")
            fail_msgs.append(f"spark-sql: {_tail_text(cp.stderr or cp.stdout or f'exit code {cp.returncode}')}")
        except Exception as e:
            fail_msgs.append(f"spark-sql: {_tail_text(str(e))}")

    if beeline:
        beeline_url = os.environ.get("CCCC_BEELINE_URL", "jdbc:hive2://localhost:10000").strip()
        beeline_user = os.environ.get("CCCC_BEELINE_USER", os.environ.get("USER", "hive")).strip()
        beeline_cmd = [beeline, "-u", beeline_url, "-n", beeline_user, "-e", sql]
        try:
            cp = subprocess.run(beeline_cmd, capture_output=True, text=True)
            if cp.returncode == 0:
                return BatchEngineStatus(status="ok", message="Hive 摘要写入成功（via beeline）", path="cccc_dw.batch_detect_job_summary")
            fail_msgs.append(f"beeline: {_tail_text(cp.stderr or cp.stdout or f'exit code {cp.returncode}')}")
        except Exception as e:
            fail_msgs.append(f"beeline: {_tail_text(str(e))}")

    return BatchEngineStatus(status="error", message=f"Hive 写入失败：{' | '.join(fail_msgs)}")


def register_batch_routes(
    app: FastAPI,
    *,
    get_text_detector_for_variant: Callable[[Optional[str]], Any],
    get_image_detector: Callable[[], Any],
    get_video_detector: Callable[[], Any],
    split_article_sentences: Callable[..., list[str]],
    extract_text_from_article_file: Callable[[Path], str],
) -> None:
    project_root = Path(__file__).resolve().parents[1]

    @app.post("/api/batch/upload-process", response_model=BatchProcessResponse)
    async def batch_upload_process(
        file: UploadFile = File(...),
        modality: str = Form(...),
        model_variant: str = Form("zh_v3"),
        min_sentence_len: int = Form(4),
        enable_spark: bool = Form(True),
        enable_hdfs: bool = Form(True),
        enable_hive: bool = Form(True),
    ) -> BatchProcessResponse:
        m = (modality or "").strip().lower()
        if m not in {"text", "image", "video"}:
            raise HTTPException(status_code=400, detail="modality 仅支持 text / image / video")

        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in _BATCH_ZIP_SUFFIXES:
            raise HTTPException(status_code=400, detail="仅支持上传 .zip 压缩包")

        job_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8]
        job_dir = project_root / "data" / "batch_outputs" / job_id
        spark_dir = job_dir / "spark"

        tmp_dir = Path(tempfile.mkdtemp(prefix="cccc_batch_upload_"))
        zip_path = tmp_dir / "batch.zip"
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zip_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)

            selected_files = _extract_zip_selected_files(zip_path, extract_dir, m)
            if not selected_files:
                raise HTTPException(status_code=400, detail=f"压缩包中未找到可用于 {m} 检测的文件")

            if m == "text":
                results = _detect_text_batch(
                    selected_files,
                    model_variant=model_variant,
                    min_sentence_len=min_sentence_len,
                    get_text_detector_for_variant=get_text_detector_for_variant,
                    split_article_sentences=split_article_sentences,
                    extract_text_from_article_file=extract_text_from_article_file,
                )
            elif m == "image":
                results = _detect_image_batch(selected_files, get_image_detector=get_image_detector)
            else:
                results = _detect_video_batch(selected_files, get_video_detector=get_video_detector)

            summary = _summarize(results)
            artifacts = _write_local_artifacts(job_dir, results, summary)

            spark_status = _run_spark_postprocess(
                project_root=project_root,
                job_id=job_id,
                modality=m,
                results_jsonl=Path(artifacts["results_jsonl"]),
                output_dir=spark_dir,
                enable_spark=enable_spark,
            )

            hdfs_inputs = [Path(artifacts["results_jsonl"]), Path(artifacts["results_csv"]), Path(artifacts["summary_json"])]
            if spark_dir.exists():
                hdfs_inputs.append(spark_dir)
            hdfs_status = _sync_hdfs(job_id, hdfs_inputs, enable_hdfs=enable_hdfs)

            hive_status = _sync_hive_summary(
                job_id=job_id,
                modality=m,
                summary=summary,
                spark_status=spark_status.status,
                hdfs_status=hdfs_status.status,
                enable_hive=enable_hive,
            )

            artifacts.update(
                {
                    "download_json_url": f"/api/batch/results/{job_id}?fmt=jsonl",
                    "download_csv_url": f"/api/batch/results/{job_id}?fmt=csv",
                }
            )

            return BatchProcessResponse(
                job_id=job_id,
                modality=m,
                total_files=summary["total_files"],
                success_files=summary["success_files"],
                failed_files=summary["failed_files"],
                ai_files=summary["ai_files"],
                human_files=summary["human_files"],
                avg_ai_score=summary["avg_ai_score"],
                results=results,
                artifacts=artifacts,
                engines={
                    "spark": spark_status,
                    "hadoop": hdfs_status,
                    "hive": hive_status,
                },
            )
        finally:
            try:
                await file.close()
            except Exception:
                pass
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @app.get("/api/batch/results/{job_id}")
    async def batch_results_download(job_id: str, fmt: str = "jsonl") -> FileResponse:
        if not job_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in job_id):
            raise HTTPException(status_code=400, detail="invalid job_id")
        ext = "jsonl" if fmt.lower() != "csv" else "csv"
        p = project_root / "data" / "batch_outputs" / job_id / f"results.{ext}"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="结果文件不存在")
        media = "application/json" if ext == "jsonl" else "text/csv"
        return FileResponse(path=p, media_type=media, filename=f"batch_{job_id}.{ext}")




