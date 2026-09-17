# -*- coding: utf-8 -*-
"""
从 HuggingFace 下载中文 AI 检测相关数据集，清洗为 ≤200 字 JSONL（train/val/test）。

优先：AnxForever/chinese-ai-detection-dataset
可选合并：QiYuan-tech/LLM-Detector（字段名不一致时做启发式映射）

用法（在 cccc 根目录）:
  conda run -n cccc python -m scripts.prepare_hf_detection_data --out-dir data/hf_prepared --max-samples-per-split 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

_SEP = "[SEP]"


def truncate_chinese(text: str, max_chars: int = 200) -> str:
    text = (text or "").strip()
    text = text.replace("\r\n", "\n")
    if _SEP in text:
        text = text.split(_SEP)[0].strip()
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    for sep in ["。", "！", "？", "\n", "；", "，", " "]:
        idx = chunk.rfind(sep)
        if idx >= 16:
            return chunk[: idx + 1].strip()
    return chunk.strip()


def normalize_label_anx(row: dict[str, Any]) -> str | None:
    lab = row.get("label")
    if lab is None:
        return None
    try:
        v = int(lab)
    except (TypeError, ValueError):
        return None
    if v == 0:
        return "human"
    if v == 1:
        return "ai"
    return None


def row_ok(text: str) -> bool:
    if not text or len(text) < 4:
        return False
    if re.match(r"^[\W\d_]+$", text):
        return False
    return True


def stable_id(text: str, label: str, source: str) -> str:
    h = hashlib.sha256(f"{text}|{label}|{source}".encode("utf-8")).hexdigest()[:16]
    return h


def iter_anx_forever(max_per_split: int | None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("AnxForever/chinese-ai-detection-dataset")
    for split_name, hf_split in ds.items():
        if split_name not in ("train", "validation", "test"):
            continue
        out_split = "val" if split_name == "validation" else split_name
        seen: set[str] = set()
        n = 0
        for row in hf_split:
            if max_per_split is not None and n >= max_per_split:
                break
            label = normalize_label_anx(row)
            if label is None:
                continue
            raw = str(row.get("text", ""))
            short = truncate_chinese(raw)
            if not row_ok(short):
                continue
            src = f"AnxForever/chinese-ai-detection-dataset/{split_name}"
            dedupe = stable_id(short, label, src)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            n += 1
            clen = len(short)
            yield {
                "id": dedupe,
                "text": short,
                "label": label,
                "source": src,
                "length": clen,
                "char_len": clen,
                "split": out_split,
            }


def try_qiyuan(max_per_split: int | None) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return
    try:
        ds = load_dataset("QiYuan-tech/LLM-Detector", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("QiYuan-tech/LLM-Detector")
        except Exception:
            return

    def pick_text_label(row: dict[str, Any]) -> tuple[str, str] | None:
        keys = {k.lower(): k for k in row}
        text = None
        for cand in ("text", "content", "sentence", "input"):
            k = keys.get(cand)
            if k and row.get(k):
                text = str(row[k])
                break
        if not text:
            return None
        label_raw = None
        for cand in ("label", "labels", "target", "is_ai"):
            k = keys.get(cand)
            if k is not None and row[k] is not None:
                label_raw = row[k]
                break
        if label_raw is None:
            return None
        if isinstance(label_raw, str):
            s = label_raw.lower().strip()
            if s in ("human", "h", "0", "real", "human-written"):
                return text, "human"
            if s in ("ai", "machine", "1", "gpt", "llm", "fake"):
                return text, "ai"
        try:
            v = int(label_raw)
            return text, "ai" if v == 1 else "human"
        except (TypeError, ValueError):
            return None

    for split_name, hf_split in ds.items():
        if split_name not in ("train", "validation", "test", "val"):
            continue
        out_split = "val" if split_name in ("validation", "val") else split_name
        if out_split not in ("train", "val", "test"):
            continue
        seen: set[str] = set()
        n = 0
        for row in hf_split:
            if max_per_split is not None and n >= max_per_split:
                break
            picked = pick_text_label(row)
            if not picked:
                continue
            raw, label = picked
            short = truncate_chinese(raw)
            if not row_ok(short):
                continue
            src = f"QiYuan-tech/LLM-Detector/{split_name}"
            dedupe = stable_id(short, label, src)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            n += 1
            clen = len(short)
            yield {
                "id": dedupe,
                "text": short,
                "label": label,
                "source": src,
                "length": clen,
                "char_len": clen,
                "split": out_split,
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/hf_prepared"))
    ap.add_argument("--max-samples-per-split", type=int, default=None, help="每 split 最多条数，调试用小整数")
    ap.add_argument("--include-qiyuan", action="store_true", help="尝试合并 QiYuan 数据集")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for rec in iter_anx_forever(args.max_samples_per_split):
        sp = rec["split"]
        if sp in by_split:
            by_split[sp].append(rec)

    if args.include_qiyuan:
        for rec in try_qiyuan(args.max_samples_per_split):
            sp = rec["split"]
            if sp in by_split:
                by_split[sp].append(rec)

    for sp, rows in by_split.items():
        path = out_dir / f"{sp}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} -> {path}")

    meta = {
        "max_chars": 200,
        "datasets": ["AnxForever/chinese-ai-detection-dataset"]
        + (["QiYuan-tech/LLM-Detector"] if args.include_qiyuan else []),
        "counts": {k: len(v) for k, v in by_split.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("meta ->", out_dir / "meta.json")


if __name__ == "__main__":
    main()
