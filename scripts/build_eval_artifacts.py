# -*- coding: utf-8 -*-
"""
批量评测产物：供首页/脆弱性页直接读取，避免前端现算。

输出目录默认 data/eval/：
- baseline_clean_metrics.json   clean test 上 baseline 指标
- attack_eval_summary.json      对抗池攻击统计、三步贡献
- robust_eval_summary.json      robust 在对抗文本上的表现（若 checkpoint 存在）
- cases_review.jsonl            单条复盘（含 baseline/robust 对原文与对抗文）
- dashboard_bundle.json         首页/图表用聚合结构

前置：已存在 data/hf_prepared/test.jsonl 与 data/vuln/adversarial_pool.jsonl

用法:
  conda run -n cccc python -m scripts.build_eval_artifacts
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.text.model_loader import (
    build_robust_text_detector,
    build_text_detector,
    robust_checkpoint_ready,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def label_bin(s: str) -> int:
    return 1 if str(s).lower() == "ai" else 0


def pred_bin(label: str) -> int:
    return 1 if str(label).lower() == "ai" else 0


def metrics_classifier(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    try:
        from sklearn.metrics import accuracy_score, f1_score

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_ai": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        }
    except Exception:
        acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(len(y_true), 1)
        return {"accuracy": acc, "f1_macro": 0.0, "f1_ai": 0.0}


def _top_source_escape_rates(sb: Counter, top_n: int = 10) -> Tuple[List[str], List[float]]:
    names = [k[:-6] for k in sb if k.endswith("|total")]
    names.sort(key=lambda n: sb.get(f"{n}|total", 0), reverse=True)
    names = names[:top_n]
    cats: List[str] = []
    vals: List[float] = []
    for n in names:
        t = float(sb.get(f"{n}|total", 0))
        s = float(sb.get(f"{n}|success", 0))
        cats.append(n)
        vals.append((s / t) if t else 0.0)
    return cats, vals


def _pivot_replace_heatmap(
    hm: Counter[Tuple[str, str]], px: int = 8, rx: int = 8
) -> Optional[Dict[str, Any]]:
    if not hm:
        return None
    marg_p = Counter()
    marg_r = Counter()
    for (p, r), c in hm.items():
        marg_p[p] += c
        marg_r[r] += c
    top_p = [k for k, _ in marg_p.most_common(px)]
    top_r = [k for k, _ in marg_r.most_common(rx)]
    data: List[List[int]] = []
    for i, p in enumerate(top_p):
        for j, r in enumerate(top_r):
            v = int(hm.get((p, r), 0))
            if v:
                data.append([j, i, v])
    return {"pivot_pos": top_p, "replace_pos": top_r, "data": data}


def run_build_eval_artifacts(
    *,
    test_path: Path = Path("data/hf_prepared/test.jsonl"),
    pool_path: Path = Path("data/vuln/adversarial_pool.jsonl"),
    out_dir: Path = Path("data/eval"),
    max_test: int = 5000,
    max_pool: Optional[int] = None,
) -> Dict[str, Any]:
    """
    生成 baseline_clean_metrics、cases_review、dashboard_bundle 等（与命令行一致）。
    max_pool 为 None 时表示使用对抗池全部行；否则截取前 max_pool 条（仅含有效 chain 的仍会逐条推理）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    max_test = max(1, min(int(max_test), 50_000))
    if max_pool is not None:
        max_pool = max(1, min(int(max_pool), 50_000))

    baseline = build_text_detector()
    robust = build_robust_text_detector() if robust_checkpoint_ready() else None

    test_rows = load_jsonl(test_path)[:max_test]
    texts = [str(r["text"]) for r in test_rows]
    y_true = [label_bin(str(r["label"])) for r in test_rows]
    bp = baseline.predict(texts)
    y_b = [pred_bin(p.label_pred) for p in bp]
    clean_metrics = metrics_classifier(y_true, y_b)
    clean_metrics["n_evaluated"] = len(test_rows)
    (out_dir / "baseline_clean_metrics.json").write_text(
        json.dumps(clean_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    robust_clean_metrics: Optional[Dict[str, Any]] = None
    if robust is not None:
        rp = robust.predict(texts)
        y_r = [pred_bin(p.label_pred) for p in rp]
        robust_clean_metrics = metrics_classifier(y_true, y_r)
        robust_clean_metrics["n_evaluated"] = len(test_rows)
        (out_dir / "robust_clean_metrics.json").write_text(
            json.dumps(robust_clean_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pool_rows = load_jsonl(pool_path)
    if max_pool is not None:
        pool_rows = pool_rows[:max_pool]
    pivot_pos: Counter[str] = Counter()
    rep_pos: Counter[str] = Counter()
    step_counts = {1: 0, 2: 0, 3: 0}
    len_bucket = Counter()
    src_bucket = Counter()
    pair_pos_hm: Counter[Tuple[str, str]] = Counter()
    attacked = 0
    evaded = 0
    robust_ok_adv = 0
    robust_n = 0
    cases_lines: List[str] = []

    for pr in pool_rows:
        ch = pr.get("chain") or {}
        orig = str(ch.get("original_text", ""))
        if not orig:
            continue
        attacked += 1
        true_l = str(pr.get("true_label", "human")).lower()
        yt = label_bin(true_l)
        win = ch.get("winning_step")
        if ch.get("overall_success"):
            evaded += 1
            if isinstance(win, int) and win in step_counts:
                step_counts[win] += 1
        adv = ch.get("final_adversarial_text") or orig

        bl_o = baseline.predict([orig])[0]
        rb_o = robust.predict([orig])[0] if robust else None
        bl_a = baseline.predict([adv])[0]
        rb_a = robust.predict([adv])[0] if robust else None

        gold_agree_b_o = pred_bin(bl_o.label_pred) == yt
        gold_agree_r_o = (pred_bin(rb_o.label_pred) == yt) if rb_o else None
        gold_agree_b_a = pred_bin(bl_a.label_pred) == yt
        gold_agree_r_a = (pred_bin(rb_a.label_pred) == yt) if rb_a else None

        if ch.get("overall_success") and robust:
            robust_n += 1
            if gold_agree_r_a:
                robust_ok_adv += 1

        if ch.get("overall_success") and win is not None:
            for st in ch.get("steps") or []:
                if st.get("step_index") == win:
                    for h in st.get("pivot_highlights") or []:
                        pivot_pos[str(h.get("pos", "?"))] += 1
                    for d in st.get("replaced_detail") or []:
                        rep_pos[str(d.get("pos", "?"))] += 1
                    ph0 = (st.get("pivot_highlights") or [{}])[0]
                    rd0 = (st.get("replaced_detail") or [{}])[0]
                    pp = str(ph0.get("pos", "?"))
                    rp = str(rd0.get("pos", "?"))
                    pair_pos_hm[(pp, rp)] += 1
                    break

        clen = len(orig)
        if clen <= 50:
            bucket = "0-50"
        elif clen <= 100:
            bucket = "51-100"
        elif clen <= 150:
            bucket = "101-150"
        else:
            bucket = "151-200"
        if ch.get("overall_success"):
            len_bucket[f"{bucket}|success"] += 1
        len_bucket[f"{bucket}|total"] += 1

        src_raw = str(pr.get("source") or "unknown")
        src_key = src_raw if len(src_raw) <= 28 else src_raw[:25] + "…"
        if ch.get("overall_success"):
            src_bucket[f"{src_key}|success"] += 1
        src_bucket[f"{src_key}|total"] += 1

        cid = str(pr.get("id", pr.get("case_id", attacked)))
        preview = orig if len(orig) <= 48 else orig[:45] + "…"
        case = {
            "case_id": cid,
            "original_text": orig,
            "text_preview": preview,
            "true_label": true_l,
            "split": pr.get("split"),
            "source": pr.get("source"),
            "char_len": clen,
            "baseline_on_original": {
                "label": bl_o.label_pred,
                "score": round(float(bl_o.ai_score), 4),
                "agrees_with_gold": gold_agree_b_o,
            },
            "robust_on_original": (
                {
                    "label": rb_o.label_pred,
                    "score": round(float(rb_o.ai_score), 4),
                    "agrees_with_gold": gold_agree_r_o,
                }
                if rb_o
                else None
            ),
            "attack_chain": ch,
            "final_adversarial_text": adv,
            "baseline_on_adversarial": {
                "label": bl_a.label_pred,
                "score": round(float(bl_a.ai_score), 4),
                "agrees_with_gold": gold_agree_b_a,
            },
            "robust_on_adversarial": (
                {
                    "label": rb_a.label_pred,
                    "score": round(float(rb_a.ai_score), 4),
                    "agrees_with_gold": gold_agree_r_a,
                }
                if rb_a
                else None
            ),
            "attack_succeeded": bool(ch.get("overall_success")),
            "baseline_evaded": bool(ch.get("overall_success")),
            "robust_recovered_on_adversarial": (
                bool(gold_agree_r_a) if (robust and ch.get("overall_success")) else None
            ),
        }
        cases_lines.append(json.dumps(case, ensure_ascii=False))

    asr = evaded / attacked if attacked else 0.0
    attack_summary = {
        "attack_pool_size": attacked,
        "baseline_escape_rate": asr,
        "baseline_attack_success_count": evaded,
        "chain_step_new_success": step_counts,
        "step_labels": ["Pivot", "Attacker2", "Attacker3"],
        "pivot_pos_distribution": dict(pivot_pos.most_common(30)),
        "replaced_pos_distribution": dict(rep_pos.most_common(30)),
        "length_bucket_escape": {
            k: len_bucket.get(k, 0) for k in sorted(set(len_bucket.keys()))
        },
        "source_bucket_escape": {
            k: int(src_bucket[k]) for k in sorted(set(src_bucket.keys()))
        },
    }
    (out_dir / "attack_eval_summary.json").write_text(
        json.dumps(attack_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    robust_summary = {
        "robust_available": robust is not None,
        "adversarial_subset_size": robust_n,
        "robust_accuracy_on_adversarial": (robust_ok_adv / robust_n) if robust_n else None,
        "robust_recovered_count": robust_ok_adv,
        "escape_rate_baseline": asr,
        "note": "robust_accuracy_on_adversarial：在攻击成功子集上，robust 预测与金标一致的比例",
    }
    (out_dir / "robust_eval_summary.json").write_text(
        json.dumps(robust_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (out_dir / "cases_review.jsonl").write_text("\n".join(cases_lines) + ("\n" if cases_lines else ""), encoding="utf-8")

    escape_drop = None
    if evaded and robust and robust_n:
        recovered_frac = robust_ok_adv / max(evaded, 1)
        escape_drop = asr * (1 - recovered_frac)

    robust_recovery_rate = (robust_ok_adv / robust_n) if robust_n else None
    robust_escape_on_evaded = (
        ((robust_n - robust_ok_adv) / robust_n) if robust_n else None
    )

    clean_performance = {
        "n_evaluated": len(test_rows),
        "baseline": {
            "accuracy": clean_metrics.get("accuracy"),
            "f1_ai": clean_metrics.get("f1_ai"),
            "f1_macro": clean_metrics.get("f1_macro"),
        },
        "robust": (
            {
                "accuracy": robust_clean_metrics.get("accuracy"),
                "f1_ai": robust_clean_metrics.get("f1_ai"),
                "f1_macro": robust_clean_metrics.get("f1_macro"),
            }
            if robust_clean_metrics
            else None
        ),
    }

    robustness_under_attack = {
        "adversarial_pool_cases": attacked,
        "baseline_escape_rate": asr,
        "baseline_evaded_count": evaded,
        "robust_recovery_rate_on_evaded": robust_recovery_rate,
        "robust_recovery_count": robust_ok_adv,
        "robust_failed_recovery_on_evaded": (robust_n - robust_ok_adv) if robust_n else 0,
        "robust_escape_rate_on_evaded": robust_escape_on_evaded,
        "escape_rate_delta_visual": escape_drop,
        "note": (
            "baseline_escape_rate：对抗池中攻击链报告成功的比例；"
            "robust_recovery_rate_on_evaded：在 Baseline 已被绕过的子集上，Robust 仍与金标一致的比例；"
            "robust_escape_rate_on_evaded：同上子集上 Robust 仍被误导的比例（1 - recovery）。"
        ),
    }

    src_cats, src_rates = _top_source_escape_rates(src_bucket, 10)
    hm_chart = _pivot_replace_heatmap(pair_pos_hm)

    bundle = {
        "version": 2,
        "clean_performance": clean_performance,
        "robustness_under_attack": robustness_under_attack,
        "dataset_eval": {
            "clean_test_n": len(test_rows),
            "baseline_clean": clean_metrics,
        },
        "pools": {
            "adversarial_pool_n": attacked,
        },
        "comparison": {
            "baseline_clean_accuracy": clean_metrics.get("accuracy"),
            "baseline_clean_f1_ai": clean_metrics.get("f1_ai"),
            "baseline_attack_escape_rate": asr,
            "robust_on_adversarial_accuracy": robust_summary.get("robust_accuracy_on_adversarial"),
            "robust_recovered_count": robust_ok_adv,
            "escape_rate_delta_visual": escape_drop,
            "baseline_evaded_total": evaded,
        },
        "attack_chain": {
            "step_counts": step_counts,
            "step_names": ["pivot", "attacker2", "attacker3"],
        },
        "charts": {
            "clean_metrics": {
                "categories": ["Accuracy", "F1(ai)", "F1(macro)"],
                "values": [
                    clean_metrics.get("accuracy", 0),
                    clean_metrics.get("f1_ai", 0),
                    clean_metrics.get("f1_macro", 0),
                ],
            },
            "clean_performance_pair": {
                "categories": ["Accuracy", "F1(ai)", "F1(macro)"],
                "baseline": [
                    clean_metrics.get("accuracy", 0),
                    clean_metrics.get("f1_ai", 0),
                    clean_metrics.get("f1_macro", 0),
                ],
                "robust": (
                    [
                        robust_clean_metrics.get("accuracy", 0),
                        robust_clean_metrics.get("f1_ai", 0),
                        robust_clean_metrics.get("f1_macro", 0),
                    ]
                    if robust_clean_metrics
                    else None
                ),
            },
            "robustness_bars": {
                "categories": [
                    "Baseline 逃逸率",
                    "Robust 恢复率(已逃逸子集)",
                    "Robust 逃逸率(已逃逸子集)",
                ],
                "values": [
                    asr,
                    robust_recovery_rate or 0.0,
                    robust_escape_on_evaded or 0.0,
                ],
            },
            "escape_compare": {
                "categories": ["Baseline 逃逸率", "Robust 对抗子集恢复率"],
                "values": [
                    asr,
                    robust_summary.get("robust_accuracy_on_adversarial") or 0.0,
                ],
            },
            "step_contributions": {
                "categories": ["Pivot", "Attacker2", "Attacker3"],
                "values": [step_counts[1], step_counts[2], step_counts[3]],
            },
            "step_contributions_cumulative": {
                "categories": ["累计至 Pivot", "累计至 Step2", "累计逃逸"],
                "values": [
                    step_counts[1],
                    step_counts[1] + step_counts[2],
                    step_counts[1] + step_counts[2] + step_counts[3],
                ],
            },
            "source_escape_rates": {
                "categories": src_cats,
                "values": src_rates,
            },
            "pivot_replace_heatmap": hm_chart,
            "pivot_pos": {
                "names": list(dict(pivot_pos.most_common(12)).keys()),
                "values": list(dict(pivot_pos.most_common(12)).values()),
            },
            "replaced_pos": {
                "names": list(dict(rep_pos.most_common(12)).keys()),
                "values": list(dict(rep_pos.most_common(12)).values()),
            },
        },
        "robust_available": robust is not None,
    }
    (out_dir / "dashboard_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    vuln_compat = {
        "baseline_clean_test": clean_metrics,
        "attack_pool_size": attacked,
        "baseline_attack_success_rate": asr,
        "robust_on_adversarial_subset": {
            "n": robust_n,
            "accuracy_vs_true_label": robust_summary.get("robust_accuracy_on_adversarial"),
        },
        "escape_comparison": {
            "baseline_evaded_count": evaded,
            "robust_recovered_count": robust_ok_adv if robust else None,
        },
        "pivot_pos_distribution": dict(pivot_pos.most_common(40)),
        "replaced_pos_distribution": dict(rep_pos.most_common(40)),
        "chain_step_success_counts": step_counts,
        "length_bucket_escape": attack_summary.get("length_bucket_escape"),
        "source_bucket_escape": attack_summary.get("source_bucket_escape"),
        "source_escape_rates": {"categories": src_cats, "values": src_rates},
        "pivot_replace_heatmap": hm_chart,
        "clean_performance": clean_performance,
        "robustness_under_attack": robustness_under_attack,
        "robust_available": robust is not None,
        "dashboard_bundle_path": str((out_dir / "dashboard_bundle.json").resolve()),
    }
    (ROOT / "data" / "vuln" / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "vuln" / "summary.json").write_text(
        json.dumps(vuln_compat, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "out_dir": str(out_dir.resolve()),
        "clean_test_evaluated": len(test_rows),
        "adversarial_pool_rows_scanned": len(pool_rows),
        "attack_cases_written": attacked,
        "baseline_escape_rate": asr,
        "vuln_summary_updated": str((ROOT / "data" / "vuln" / "summary.json").resolve()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=Path("data/hf_prepared/test.jsonl"))
    ap.add_argument("--pool", type=Path, default=Path("data/vuln/adversarial_pool.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    ap.add_argument("--max-test", type=int, default=5000)
    ap.add_argument(
        "--max-pool",
        type=int,
        default=None,
        help="对抗池最多使用前 N 条（默认不截断）",
    )
    args = ap.parse_args()
    stats = run_build_eval_artifacts(
        test_path=args.test,
        pool_path=args.pool,
        out_dir=args.out_dir,
        max_test=args.max_test,
        max_pool=args.max_pool,
    )
    print("wrote", stats["out_dir"])
    print("updated data/vuln/summary.json", stats)


if __name__ == "__main__":
    main()
