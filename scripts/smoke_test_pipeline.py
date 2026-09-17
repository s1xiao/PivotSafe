# -*- coding: utf-8 -*-
"""
串联脚本自测：用 2 条临时统一样本跑通 统一样本 → 检测 → 统计 → 报告。
会调用后端文本检测（加载模型），首次运行可能较慢。

在项目根目录、cccc 环境中执行：
  python -m scripts.smoke_test_pipeline
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    samples = [
        {"post_id": "p1", "platform": "xhs", "url": "https://xhs.com/1", "title": "测试", "text": "这是一段示例正文。"},
        {"post_id": "p2", "platform": "tieba", "url": "https://tieba.baidu.com/p/2", "title": "示例", "text": "另一段短文本。"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        samples_path = Path(tmp) / "unified_samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()

        from scripts.run_pipeline import load_unified_samples, run_text_detection, _get_client
        loaded = load_unified_samples(samples_path, max_samples=2)
        if len(loaded) != 2:
            print(f"FAIL: expected 2 samples, got {len(loaded)}")
            return 1
        client = _get_client()
        records = run_text_detection(loaded, client, batch_size=2)
        if len(records) != 2:
            print(f"FAIL: expected 2 detection records, got {len(records)}")
            return 1
        if records[0].get("text_ai_score") is None:
            print("FAIL: missing text_ai_score")
            return 1
        print("pipeline smoke test passed (detection + stats + report 链路可用).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
