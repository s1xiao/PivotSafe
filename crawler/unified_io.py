# -*- coding: utf-8 -*-
"""
统一样本 JSONL 读写：供爬虫输出与检测管线消费。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_unified_jsonl(samples: list[dict[str, Any]], path: str | Path) -> None:
    """将统一样本列表写入 JSONL 文件，便于后续检测与分析脚本消费。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
