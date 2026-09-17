# -*- coding: utf-8 -*-
"""
检测结果基础统计与简单可视化。

输入：统一检测结果 JSONL，每行一条记录，字段见下方 DETECTION_RESULT_FIELDS。
输出：汇总统计（JSON）、可选图表（PNG 等）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# 统一检测结果 JSONL 每行字段（与 backend 文本/图像打分输出合并后的格式）
# post_id, platform, url: 与 crawler 统一样本一致
# text_ai_score, text_label: 文本检测结果，无则 null
# image_scores, image_labels: 图像检测结果列表，无图则 []
# mean_image_score: 该条目多图时的平均分，可选
DETECTION_RESULT_FIELDS = (
    "post_id",
    "platform",
    "url",
    "text_ai_score",
    "text_label",
    "image_scores",
    "image_labels",
    "mean_image_score",
)


def load_detection_results(path: str | Path) -> list[dict[str, Any]]:
    """从 JSONL 文件加载检测结果列表。"""
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def compute_mean_image_score(record: dict[str, Any]) -> float | None:
    """单条记录的图像平均分；无图或无分返回 None。"""
    scores = record.get("image_scores")
    if not scores:
        return None
    try:
        return sum(float(x) for x in scores) / len(scores)
    except (TypeError, ValueError):
        return None


def ensure_mean_image_score(records: list[dict[str, Any]]) -> None:
    """就地为每条记录补充 mean_image_score（若缺失）。"""
    for r in records:
        if "mean_image_score" not in r or r["mean_image_score"] is None:
            r["mean_image_score"] = compute_mean_image_score(r)


def compute_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    对检测结果列表做基础统计。
    返回字典：总数、按平台计数、文本/图像分数均值、AI 判定比例等。
    """
    if not records:
        return {
            "total": 0,
            "by_platform": {},
            "text_score_mean": None,
            "text_score_count": 0,
            "image_score_mean": None,
            "image_score_count": 0,
            "text_ai_ratio": None,
            "image_ai_ratio": None,
        }

    ensure_mean_image_score(records)

    by_platform: dict[str, int] = {}
    text_scores: list[float] = []
    text_labels: list[str] = []

    image_scores_per_post: list[float] = []
    image_labels_per_post: list[str] = []

    for r in records:
        plat = r.get("platform") or "unknown"
        by_platform[plat] = by_platform.get(plat, 0) + 1

        ts = r.get("text_ai_score")
        if ts is not None:
            try:
                text_scores.append(float(ts))
                text_labels.append((r.get("text_label") or "").strip().lower())
            except (TypeError, ValueError):
                pass

        ms = r.get("mean_image_score")
        if ms is not None:
            try:
                image_scores_per_post.append(float(ms))
            except (TypeError, ValueError):
                pass
            labels = r.get("image_labels") or []
            ai_count = sum(1 for L in labels if str(L).strip().lower() == "ai")
            image_labels_per_post.append("ai" if ai_count > len(labels) / 2 else "human")

    text_score_mean = sum(text_scores) / len(text_scores) if text_scores else None
    image_score_mean = sum(image_scores_per_post) / len(image_scores_per_post) if image_scores_per_post else None
    text_ai_ratio = (sum(1 for L in text_labels if L == "ai") / len(text_labels)) if text_labels else None
    image_ai_ratio = (sum(1 for L in image_labels_per_post if L == "ai") / len(image_labels_per_post)) if image_labels_per_post else None

    return {
        "total": len(records),
        "by_platform": by_platform,
        "text_score_mean": round(text_score_mean, 4) if text_score_mean is not None else None,
        "text_score_count": len(text_scores),
        "image_score_mean": round(image_score_mean, 4) if image_score_mean is not None else None,
        "image_score_count": len(image_scores_per_post),
        "text_ai_ratio": round(text_ai_ratio, 4) if text_ai_ratio is not None else None,
        "image_ai_ratio": round(image_ai_ratio, 4) if image_ai_ratio is not None else None,
    }


def save_stats(stats: dict[str, Any], path: str | Path) -> None:
    """将统计结果写入 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def plot_score_distribution(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "AI Score Distribution",
) -> None:
    """
    绘制文本与图像分数分布直方图，保存到 output_path。
    依赖 matplotlib；若未安装则跳过绘图。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ensure_mean_image_score(records)
    text_scores = []
    image_scores = []
    for r in records:
        ts = r.get("text_ai_score")
        if ts is not None:
            try:
                text_scores.append(float(ts))
            except (TypeError, ValueError):
                pass
        ms = r.get("mean_image_score")
        if ms is not None:
            try:
                image_scores.append(float(ms))
            except (TypeError, ValueError):
                pass

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if text_scores:
        axes[0].hist(text_scores, bins=min(20, max(2, len(text_scores))), edgecolor="black", alpha=0.7)
    axes[0].set_title("Text AI Score")
    axes[0].set_xlabel("score")
    if image_scores:
        axes[1].hist(image_scores, bins=min(20, max(2, len(image_scores))), edgecolor="black", alpha=0.7)
    axes[1].set_title("Image AI Score (per-post mean)")
    axes[1].set_xlabel("score")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="检测结果基础统计与简单图表")
    parser.add_argument("input", type=str, help="检测结果 JSONL 文件路径")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="统计 JSON 与图表输出目录，默认与 input 同目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成分布图")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"文件不存在: {inp}")
        return 1
    out_dir = Path(args.output_dir) if args.output_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_detection_results(inp)
    if not records:
        print("未读取到任何记录")
        return 0

    stats = compute_stats(records)
    stats_path = out_dir / "basic_stats.json"
    save_stats(stats, stats_path)
    print(f"统计已写入: {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if not args.no_plot:
        plot_path = out_dir / "score_distribution.png"
        plot_score_distribution(records, plot_path)
        print(f"分布图已写入: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
