#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键流程（无登录站点）：简单网页爬取 → 统一样本 → 检测管线 → 数据报告。

当前实现：从若干公开站点（如 IT 之家、Solidot）抓取文章，聚合为统一样本后跑现有 run_pipeline。
适合在无图形界面、无法登录平台的环境下快速获得真实数据做实验。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = __import__("argparse").ArgumentParser(
        description="简单网页爬取 → 检测 → 数据报告（无需登录）"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=1000,
        help="总共最多抓取文章数（多站点聚合）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pipeline_out_web",
        help="检测与报告输出目录",
    )
    parser.add_argument(
        "--output-samples",
        type=str,
        default=None,
        help="统一样本 JSONL 输出路径，默认 data/unified_web_corpus.jsonl",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default="ithome,solidot,v2ex,oschina,linuxcn,36kr",
        help="逗号分隔的站点列表，例如 ithome,solidot,v2ex,oschina,linuxcn,36kr",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="在检测前下载 image_urls 到本地并补充 image_paths，以启用图像检测",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="data/images",
        help="下载图片保存根目录，默认 data/images",
    )
    args = parser.parse_args()

    project_root = root

    # 1. 简单网页爬取（多站点聚合）
    from crawler.simple_web_crawler import crawl_multi_site_articles
    from crawler.unified_io import write_unified_jsonl

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    print(f"开始从站点 {sites} 抓取文章，总共最多 {args.max_records} 条…")
    samples = crawl_multi_site_articles(sites=sites, max_records=args.max_records)
    if not samples:
        print("未抓取到任何文章，请稍后重试或检查网络。")
        return 1
    if len(samples) < 1000:
        print("警告：未达到 1000 条（实际 %d 条），RSS 源单次可获取量有限。将继续处理现有样本。" % len(samples))

    samples_path = (
        Path(args.output_samples)
        if args.output_samples
        else project_root / "data" / "unified_web_corpus.jsonl"
    )
    write_unified_jsonl(samples, samples_path)
    print(f"统一样本已写入: {samples_path}（共 {len(samples)} 条）")

    # 若需要图像检测，则先将 image_urls 下载为本地 image_paths
    final_samples_path = samples_path
    if args.download_images:
        from crawler.image_downloader import download_images_from_unified_jsonl

        enriched_path = samples_path.with_name(samples_path.stem + "_with_images.jsonl")
        print("开始下载图片以启用图像检测…")
        download_images_from_unified_jsonl(
            input_path=samples_path,
            output_path=enriched_path,
            images_root=args.images_root,
        )
        final_samples_path = enriched_path
        print(f"已生成带 image_paths 的样本文件: {final_samples_path}")

    # 2. 运行检测与报告
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cmd = [
        sys.executable,
        "-m",
        "scripts.run_pipeline",
        "--samples",
        str(final_samples_path),
        "--output-dir",
        str(out_dir),
    ]
    if not args.download_images:
        pipeline_cmd.append("--no-image")
    print("运行检测与报告…")
    ret = subprocess.run(pipeline_cmd, cwd=str(project_root))
    if ret.returncode != 0:
        print("检测管线执行失败，退出码:", ret.returncode)
        return ret.returncode

    stats_path = out_dir / "basic_stats.json"
    plot_path = out_dir / "score_distribution.png"
    report_path = out_dir / "report.html"
    print("")
    print("数据报告已生成：")
    print(f"  统计: {stats_path}")
    print(f"  分布图: {plot_path}")
    print(f"  报告: {report_path}")
    print("将 STATS_JSON_PATH 指向上述 basic_stats.json 后，可通过 /api/summary 或 Next web/ 对接查看统计与图表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

