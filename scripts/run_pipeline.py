# -*- coding: utf-8 -*-
"""
串联链路：统一样本 JSONL → 文本检测（+ 可选图像检测）→ 检测结果 JSONL → basic_stats → 报告 HTML。

统一样本可含可选字段 image_paths（本地图片路径列表）；有则调用后端图像检测并写入结果。
在项目根目录、cccc 环境中执行示例：
  python -m scripts.run_pipeline --samples data/unified_samples.jsonl --output-dir data/pipeline_out
  python -m scripts.run_pipeline --samples data/unified_samples.jsonl --no-image   # 仅文本
若未安装或未下载模型，首次运行可能较慢；可用 --max-samples 限制处理条数。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_unified_samples(path: Path, max_samples: int | None = None) -> list[dict]:
    """读取统一样本 JSONL（每行含 post_id, platform, url, text 等；可选 image_paths）。"""
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _get_client():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def run_text_detection(samples: list[dict], client, batch_size: int = 32) -> list[dict]:
    """通过后端 API 对样本正文做文本检测，返回与 samples 同序的检测结果列表。"""
    results: list[dict] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        texts = [s.get("text") or s.get("title") or "" for s in batch]
        r = client.post("/api/text/score", json={"texts": texts})
        r.raise_for_status()
        data = r.json()
        for i, item in enumerate(data["results"]):
            s = batch[i]
            text_full = (s.get("text") or s.get("title") or "")[:2000]
            image_urls = s.get("image_urls") or []
            if isinstance(image_urls, str):
                image_urls = [u.strip() for u in image_urls.split(",") if u.strip()]
            image_paths = s.get("image_paths") or []
            if isinstance(image_paths, str):
                image_paths = [p.strip() for p in image_paths.split(",") if p.strip()]
            rec = {
                "post_id": s.get("post_id", ""),
                "platform": s.get("platform", ""),
                "url": s.get("url", ""),
                "title": (s.get("title") or "")[:500],
                "text": text_full,
                "image_urls": image_urls[:10],
                "text_ai_score": item["ai_score"],
                "text_label": item["label_pred"],
                "image_scores": [],
                "image_labels": [],
            }
            if image_paths and not image_urls:
                rec["image_paths"] = image_paths[:10]
            results.append(rec)
    return results


def run_image_detection(samples: list[dict], records: list[dict], client) -> None:
    """对含 image_paths 的样本调用图像检测 API，就地更新 records 的 image_scores、image_labels。"""
    for i, s in enumerate(samples):
        paths = s.get("image_paths") or []
        if not paths or i >= len(records):
            continue
        paths = [p for p in paths if isinstance(p, str) and Path(p).exists()]
        if not paths:
            continue
        r = client.post("/api/image/score", json={"image_paths": paths})
        if r.status_code != 200:
            continue
        data = r.json()
        records[i]["image_scores"] = [x["ai_score"] for x in data["results"]]
        records[i]["image_labels"] = [x["label_pred"] for x in data["results"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="统一样本 → 检测 → 统计 → 报告 串联脚本")
    parser.add_argument("--samples", type=str, required=True, help="统一样本 JSONL 路径")
    parser.add_argument("--output-dir", type=str, default="data/pipeline_out", help="检测结果、统计与报告输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="最多处理样本数")
    parser.add_argument("--no-image", action="store_true", help="不运行图像检测，仅文本")
    parser.add_argument("--no-report", action="store_true", help="不生成报告 HTML")
    parser.add_argument("--no-plot", action="store_true", help="不生成分数分布图")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"统一样本文件不存在: {samples_path}")
        return 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_unified_samples(samples_path, args.max_samples)
    if not samples:
        print("未读取到任何样本")
        return 0
    print(f"已加载 {len(samples)} 条样本，开始文本检测…")
    client = _get_client()
    detection_records = run_text_detection(samples, client)
    if not args.no_image:
        run_image_detection(samples, detection_records, client)
        print("图像检测已合并（含 image_paths 的样本已打分）")
    results_path = out_dir / "detection_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in detection_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"检测结果已写入: {results_path}")

    from analysis.basic_stats import (
        load_detection_results,
        compute_stats,
        save_stats,
        plot_score_distribution,
    )
    records = load_detection_results(results_path)
    stats = compute_stats(records)
    stats_path = out_dir / "basic_stats.json"
    save_stats(stats, stats_path)
    print(f"统计已写入: {stats_path}")
    if not args.no_plot:
        plot_path = out_dir / "score_distribution.png"
        plot_score_distribution(records, plot_path)
        print(f"分布图已写入: {plot_path}")

    if not args.no_report:
        from report.generate_report import load_stats, generate_html
        html_path = out_dir / "report.html"
        stats_data = load_stats(stats_path)
        if stats_data:
            html_path.write_text(generate_html(stats_data), encoding="utf-8")
            print(f"报告已生成: {html_path}")
        else:
            print("跳过报告：无法读取统计文件")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
