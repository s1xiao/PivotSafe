# -*- coding: utf-8 -*-
"""
unified_io 自测：验证 write_unified_jsonl 写入与读取。
在 cccc 环境中执行：
  python -m crawler.smoke_test_unified_io
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from crawler.unified_io import write_unified_jsonl


def test_write_unified_jsonl() -> None:
    samples = [
        {"platform": "ithome", "post_id": "a", "url": "u1", "title": "T", "text": "t", "image_urls": []},
        {"platform": "solidot", "post_id": "b", "url": "u2", "title": "T2", "text": "t2", "image_urls": []},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "unified.jsonl"
        write_unified_jsonl(samples, out_path)
        assert out_path.exists()
        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        back = [json.loads(ln) for ln in lines]
        assert back[0]["post_id"] == "a" and back[1]["post_id"] == "b"


def main() -> None:
    test_write_unified_jsonl()
    print("crawler.unified_io smoke test passed.")


if __name__ == "__main__":
    main()
