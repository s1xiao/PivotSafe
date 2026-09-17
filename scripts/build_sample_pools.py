# -*- coding: utf-8 -*-
"""
从 data/hf_prepared/*.jsonl 生成比赛用样本池文件（供 API 与前端批量展示）。

输出（默认 data/pools/）：
- sample_pool_clean.jsonl      全量去重样本（train+val+test）
- sample_pool_attackable.jsonl 倾向「可打」的 AI 标签样本（默认 test+val 中 label=ai）
- sample_pool_adversarial.jsonl 从对抗池复制（若存在 data/vuln/adversarial_pool.jsonl）

每条记录保证含：id, text, label, source, char_len, split（对抗池另有 chain 等字段）。

用法:
  python -m scripts.build_sample_pools --prepared-dir data/hf_prepared --out-dir data/pools
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def load_all_splits(prepared: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        p = prepared / name
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                cid = str(r.get("id", ""))
                if cid and cid in seen:
                    continue
                if cid:
                    seen.add(cid)
                t = str(r.get("text", ""))
                clen = int(r.get("char_len", r.get("length", len(t))))
                r = {
                    **r,
                    "char_len": clen,
                    "length": r.get("length", clen),
                }
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=Path("data/hf_prepared"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/pools"))
    ap.add_argument("--adversarial-src", type=Path, default=Path("data/vuln/adversarial_pool.jsonl"))
    args = ap.parse_args()

    all_rows = load_all_splits(args.prepared_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = args.out_dir / "sample_pool_clean.jsonl"
    with clean_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_rows)} -> {clean_path}")

    attackable = [
        r
        for r in all_rows
        if str(r.get("label", "")).lower() == "ai"
        and str(r.get("split", "")) in ("test", "val")
    ]
    atk_path = args.out_dir / "sample_pool_attackable.jsonl"
    with atk_path.open("w", encoding="utf-8") as f:
        for r in attackable:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(attackable)} -> {atk_path}")

    adv_dst = args.out_dir / "sample_pool_adversarial.jsonl"
    if args.adversarial_src.is_file():
        shutil.copy2(args.adversarial_src, adv_dst)
        print(f"copied adversarial pool -> {adv_dst}")
    else:
        adv_dst.write_text("", encoding="utf-8")
        print(f"no {args.adversarial_src}, touched empty {adv_dst}")

    meta = {
        "sample_pool_clean_count": len(all_rows),
        "sample_pool_attackable_count": len(attackable),
        "human_count": sum(1 for r in all_rows if str(r.get("label")).lower() == "human"),
        "ai_count": sum(1 for r in all_rows if str(r.get("label")).lower() == "ai"),
    }
    (args.out_dir / "pools_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("meta ->", args.out_dir / "pools_meta.json")


if __name__ == "__main__":
    main()
