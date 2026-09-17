# -*- coding: utf-8 -*-
"""
从 basic_stats.json 生成 HTML 实验报告。
用法（在 cccc 环境中）：
  python -m report.generate_report [--stats path/to/basic_stats.json] [--output path/to/report.html]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CCCC AI 检测 - 实验报告</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 0 auto; padding: 1.5rem; line-height: 1.6; }}
    h1 {{ font-size: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .meta {{ color: #666; font-size: 0.875rem; margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <h1>CCCC AI 生成内容检测 - 实验报告</h1>
  <p class="meta">由 report/generate_report.py 基于 basic_stats 生成。</p>
  <h2>1. 概览</h2>
  <table>
    <tr><th>指标</th><th>值</th></tr>
    <tr><td>总样本数</td><td>{total}</td></tr>
    <tr><td>文本检测条数</td><td>{text_score_count}</td></tr>
    <tr><td>图像检测条数（按帖）</td><td>{image_score_count}</td></tr>
    <tr><td>文本 AI 可疑度均值</td><td>{text_score_mean}</td></tr>
    <tr><td>图像 AI 可疑度均值</td><td>{image_score_mean}</td></tr>
    <tr><td>文本判为 AI 比例</td><td>{text_ai_ratio_pct}</td></tr>
    <tr><td>图像判为 AI 比例</td><td>{image_ai_ratio_pct}</td></tr>
  </table>
  <h2>2. 平台分布</h2>
  {by_platform_html}
</body>
</html>
"""


def _by_platform_to_html(by_platform: dict) -> str:
    """将 by_platform 转为 HTML 表格。"""
    if not by_platform:
        return "<p>无数据</p>"
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(by_platform.items())
    )
    return f'<table><tr><th>平台</th><th>样本数</th></tr>{rows}</table>'


def load_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_html(stats: dict) -> str:
    s = stats
    text_ratio = s.get("text_ai_ratio")
    image_ratio = s.get("image_ai_ratio")
    by_platform = s.get("by_platform") or {}
    return TEMPLATE.format(
        total=s.get("total", "-"),
        text_score_count=s.get("text_score_count", "-"),
        image_score_count=s.get("image_score_count", "-"),
        text_score_mean=s.get("text_score_mean") if s.get("text_score_mean") is not None else "-",
        image_score_mean=s.get("image_score_mean") if s.get("image_score_mean") is not None else "-",
        text_ai_ratio_pct=f"{text_ratio * 100:.1f}%" if text_ratio is not None else "-",
        image_ai_ratio_pct=f"{image_ratio * 100:.1f}%" if image_ratio is not None else "-",
        by_platform_html=_by_platform_to_html(by_platform),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="从 basic_stats.json 生成 HTML 报告")
    parser.add_argument("--stats", type=str, default=None, help="basic_stats.json 路径，默认 report/basic_stats.json")
    parser.add_argument("--output", type=str, default=None, help="输出 HTML 路径，默认 report/report.html")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    stats_path = Path(args.stats) if args.stats else base / "basic_stats.json"
    out_path = Path(args.output) if args.output else base / "report.html"

    stats = load_stats(stats_path)
    if not stats:
        print(f"未找到统计文件: {stats_path}")
        return 1

    html = generate_html(stats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"报告已生成: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
