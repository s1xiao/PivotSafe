# -*- coding: utf-8 -*-
"""
对 JSONL 中的文本逐条跑固定攻击链（每步从原文），输出对抗池 JSONL。

用法:
  conda run -n cccc python -m scripts.build_adversarial_pool \\
    --input data/hf_prepared/val.jsonl --limit 30 --output data/vuln/adversarial_pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext
from attacks.fixed_chain import chain_result_to_json, run_fixed_chain
from detectors.text.model_loader import build_text_detector


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("data/vuln/adversarial_pool.jsonl"))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-queries-per-step", type=int, default=1000)
    args = ap.parse_args()

    det = build_text_detector()

    def score_fn(t: str) -> float:
        return float(det.score([t])[0])

    victim = ScoreContext(score_fn=score_fn, threshold=det.threshold, positive_is_ai=True)
    pipe = ChineseReplacePipeline()

    rows = load_jsonl(args.input)[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fout:
        for row in rows:
            text = str(row.get("text", "")).strip()
            if len(text) < 4:
                continue
            chain = run_fixed_chain(
                text,
                victim,
                pipe,
                detector=det,
                max_queries_per_step=args.max_queries_per_step,
            )
            rec = {
                "id": row.get("id"),
                "source": row.get("source"),
                "split": row.get("split"),
                "true_label": row.get("label"),
                "chain": chain_result_to_json(chain),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("wrote", len(rows), "->", args.output)


if __name__ == "__main__":
    main()
