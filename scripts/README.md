# Scripts 目录

可复现的一键验证与串联脚本。

| 脚本 | 说明 |
|------|------|
| `run_smoke_tests` | 七项验证：crawler、analysis、report、后端、视频、检测 API（pivot / fixed-chain / vuln）、**dataset/cases/text.compare**。 |
| `test_caa_integration` | CAA + 桥接 + 固定链（Step2/3 固定 HQA/TextHacker）。慢测设 `CCCC_SKIP_CAA_SLOW=1`。 |
| `test_pivot_attack` | Pivot 高亮/排序：环境变量 `PIVOT_USE_ANCHOR`、遮挡回退、Anchor 路径、`run_pivot_attack` 烟测。 |
| `prepare_hf_detection_data` | 从 HuggingFace（AnxForever 等）下载并清洗为 ≤200 字 JSONL（train/val/test）。 |
| `build_sample_pools` | 生成 `data/pools/sample_pool_clean.jsonl`、`sample_pool_attackable.jsonl` 等及 `pools_meta.json`。 |
| `build_eval_artifacts` | 读 test + 对抗池，写 `dashboard_bundle`（含 clean/鲁棒性分列图表、按 source 逃逸率、Pivot×替换热力图、链累计曲线）、`cases_review.jsonl` 等，并覆盖 `data/vuln/summary.json`。 |
| `train_robust_detector` | 在 `hfl/chinese-roberta-wwm-ext` 上冻结前 8 层训练 robust，权重写入 `checkpoints/robust_roberta`。 |
| `build_adversarial_pool` | 对 JSONL 跑固定攻击链，输出 `data/vuln/adversarial_pool.jsonl`（记录含 `split`）。 |
| `run_vulnerability_analysis` | 汇总 baseline/攻击/robust 统计 → `data/vuln/summary.json`（可与 `build_eval_artifacts` 二选一或先后运行）。 |
| `gen_ai_qwen_optional` | Qwen 派生数据占位说明（主 MVP 不依赖）。 |
| `download_ai_sample` | 下载 AI 人像示例到 `data/sample/ai.jpg`（优先网络，失败则生成合成人像）。 |
| `gen_ai_portrait` | 本地生成 AI 风格人像（椭圆脸+五官）到 `data/sample/ai.jpg`，无需网络。 |
| `run_simple_crawl_to_report` | 公开网页真爬→检测→报告：从 IT之家 / Solidot / V2EX / 开源中国 / Linux 中国 / 36氪 抓取文章，写入统一样本并跑检测与报告；支持 1000+ 条。 |
| `run_pipeline` | 串联链路：统一样本 JSONL → 文本检测（+ 可选图像检测）→ 检测结果 JSONL → basic_stats → 报告 HTML。需传入 `--samples`；样本可含 `image_paths`（本地路径列表）以启用图像检测，加 `--no-image` 则仅文本。 |
| `smoke_test_pipeline` | 用 2 条临时样本跑通串联逻辑（含文本检测），会加载模型，首次较慢。 |

## 使用示例

```bash
conda activate cccc
cd /path/to/cccc

# 环境与链路验证
python -m scripts.run_smoke_tests

# 公开网页（无需登录）真爬 → 检测 → 报告（推荐，1000 条）
python -m scripts.run_simple_crawl_to_report --sites ithome,solidot,v2ex,oschina,linuxcn,36kr --max-records 1000
python -m scripts.run_simple_crawl_to_report --max-records 1000 --download-images --images-root data/images_web  # 启用图像检测

# 从统一样本跑满链路到报告（含图像检测：样本需有 image_paths）
python -m scripts.run_pipeline --samples data/unified_web_corpus.jsonl --output-dir data/pipeline_out_web
python -m scripts.run_pipeline --samples data/unified_web_corpus.jsonl --max-samples 10   # 限制 10 条
python -m scripts.run_pipeline --samples data/unified_web_corpus.jsonl --no-image         # 仅文本检测

# 串联自测（会加载文本模型）
python -m scripts.smoke_test_pipeline
```

## 爬取 → 检测 → 报告

`run_simple_crawl_to_report` 依次执行：从若干公开网站（IT之家、Solidot、V2EX、开源中国、Linux 中国、36氪）抓取文章 → 将结果聚合为统一样本并写入 `data/unified_web_corpus.jsonl` → （可选）下载正文配图以生成 `image_paths` → 调用 `run_pipeline` 做文本（+ 图像）检测与统计 → 生成 `data/pipeline_out_web` 下的 `basic_stats.json`、`score_distribution.png`、`report.html`。

- **前置**：无需登录，仅需保证当前环境可以访问目标网站。
- **参数**：`--sites` 指定站点列表（默认 `ithome,solidot,v2ex,oschina,linuxcn,36kr`）；`--max-records` 控制总样本数（默认 1000）；`--download-images` 启用配图下载并自动开启图像检测；`--images-root` 控制图片保存路径（默认 `data/images`）。
- 该脚本适合作为比赛数据入口；可视化请用 Next `web/` 或自行对接 `/api/summary`。

## 示例图（历史静态页 / 报告用）

`data/sample/` 下需有 `human.jpg`（真人示例）和 `ai.jpg`（AI 人像示例）。若 `ai.jpg` 为抽象渐变或缺失，可执行：

```bash
# 优先从网络下载真实 AI 人像（Pexels/Pixabay/thispersondoesnotexist）
python -m scripts.download_ai_sample

# 或本地生成合成人像（无需网络）
python scripts/gen_ai_portrait.py
```
