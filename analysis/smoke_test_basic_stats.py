# -*- coding: utf-8 -*-
"""
basic_stats 自测：用临时检测结果 JSONL 验证加载、统计与绘图。
在 cccc 环境中执行：
  python -m analysis.smoke_test_basic_stats
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from analysis.basic_stats import (
    compute_stats,
    load_detection_results,
    save_stats,
    plot_score_distribution,
)


def test_load_and_stats() -> None:
    records = [
        {
            "post_id": "p1",
            "platform": "xhs",
            "url": "https://xhs.com/1",
            "text_ai_score": 0.2,
            "text_label": "human",
            "image_scores": [0.1, 0.3],
            "image_labels": ["human", "human"],
        },
        {
            "post_id": "p2",
            "platform": "xhs",
            "url": "https://xhs.com/2",
            "text_ai_score": 0.9,
            "text_label": "ai",
            "image_scores": [0.85],
            "image_labels": ["ai"],
        },
        {
            "post_id": "p3",
            "platform": "tieba",
            "url": "https://tieba.baidu.com/p/3",
            "text_ai_score": 0.5,
            "text_label": "human",
            "image_scores": [],
            "image_labels": [],
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "detection_results.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        loaded = load_detection_results(path)
        assert len(loaded) == 3

        stats = compute_stats(loaded)
        assert stats["total"] == 3
        assert stats["by_platform"]["xhs"] == 2
        assert stats["by_platform"]["tieba"] == 1
        assert stats["text_score_mean"] is not None
        assert stats["text_score_count"] == 3
        assert stats["image_score_mean"] is not None
        assert stats["image_score_count"] == 2  # p3 无图，不贡献 mean_image_score

        save_stats(stats, Path(tmp) / "basic_stats.json")
        assert (Path(tmp) / "basic_stats.json").exists()


def test_plot() -> None:
    records = [
        {"post_id": "a", "platform": "xhs", "url": "u", "text_ai_score": 0.3, "text_label": "human", "image_scores": [0.2], "image_labels": ["human"]},
        {"post_id": "b", "platform": "xhs", "url": "u2", "text_ai_score": 0.8, "text_label": "ai", "image_scores": [0.9], "image_labels": ["ai"]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dist.png"
        plot_score_distribution(records, out)
        # 若安装了 matplotlib 则文件存在；未安装则静默跳过
        if out.exists():
            assert out.stat().st_size > 0


def main() -> None:
    test_load_and_stats()
    test_plot()
    print("analysis.basic_stats smoke test passed.")


if __name__ == "__main__":
    main()
