#!/usr/bin/env bash
# 一键全流程：爬取 1000 条 → 顶会检测 → 生成报告
# 用法：conda activate cccc && ./scripts/run_full_pipeline.sh

set -e
cd "$(dirname "$0")/.."

echo "=== 1. 爬取 1000 条 ==="
python -m scripts.run_simple_crawl_to_report \
  --sites ithome,solidot,v2ex,oschina,linuxcn,36kr \
  --max-records 1000 \
  --output-dir data/pipeline_out_web \
  --no-image

echo ""
echo "=== 完成 ==="
echo "报告: data/pipeline_out_web/report.html"
echo "统计: data/pipeline_out_web/basic_stats.json"
echo "将 STATS_JSON_PATH 指向 basic_stats.json 可在 Dashboard 中查看。"
