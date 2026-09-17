# CCCC · AI 生成内容检测平台

多模态（文本 / 图像 / 视频）AI 内容检测，支持爬取 → 检测 → 报告 全流程。

**CAA 代码副本（参考）**：`third_party/caa/` 仍为 HQA/TextHacker 等 vendored 代码。**固定链**：**Pivot（Anchor 排序 + 中文束搜，失败则仅高亮）→ HQA → TextHacker**；Step2/3 需 `TextAIDetector`，无 MLM 贪心回退。中文近义词默认按句用 MLM 候选建邻（`CCCC_CAA_SYNONYMS=chinese`），无需为固定链放入 `counter-fitted-vectors.txt`、`cos_sim_matrix.pkl`（`legacy` 模式除外）。

**Anchor / Pivot**：重要度排序入口为 `third_party.anchor.anchor_word_ranking.anchor_word_ranking`（由 `attacks/pivot_attack.py` 动态导入）；束搜在 `attacks/anchor_beam_zh.py`。`third_party/anchor/` 为从 **anchor-master** 复制的 Anchor-Text 核心；**不修改** `autodl-tmp` 下原仓库。`PIVOT_USE_ANCHOR=1` 且 `CCCC_PIVOT_BEAM=1` 为默认；关闭或失败时 Pivot **不改写文本**，仅返回 `compute_pivot_ranking` 高亮。`pivot-highlight` 的 `pivot_spans` 在 Anchor 成功时为 **最终锚点** 所含词（可多个）。

**文本攻防 MVP（比赛演示）**：Baseline 检测器见 `configs/text_detector.json`；Robust 见 `configs/robust_text_detector.json`（需先训练）。固定三元攻击链：`POST /api/attack/fixed-chain`；Pivot 高亮：`POST /api/attack/pivot-highlight`。首页「评测产物」也可在 **Next 总览** 上一键生成，或调用 `POST /api/dataset/rebuild-eval-artifacts`（JSON：`{"limit":200}`，同时截断 clean test 与对抗池条数，等价 `scripts.build_eval_artifacts.run_build_eval_artifacts`）。批量数据与对比：`GET /api/dataset/summary`（聚合 `dashboard_bundle` + 样本池元数据）、`GET /api/dataset/samples`（分页/随机/过滤）、`POST /api/text/compare`（同条 Baseline vs Robust）、`GET /api/cases/list` 与 `GET /api/cases/review/{id}`（读 `data/eval/cases_review.jsonl`）。脆弱性汇总：`GET /api/vuln/summary`（推荐用 `scripts/build_eval_artifacts` 同步写入 `data/vuln/summary.json`；亦可用 `VULN_SUMMARY_JSON` 指向其它文件）。

---

## 如何启动

```bash
# 1. 创建并激活环境（必须，否则一键流程会因缺包失败）
conda env create -f environment.yml
conda activate cccc
pip install -r requirements.txt   # 若 conda 未装全 pip 依赖

# 2. 验证（可选）
python -m scripts.run_smoke_tests
# Pivot（Anchor / 遮挡排序，无 MLM 改写回退）专项：`conda run -n cccc python -m scripts.test_pivot_attack`

# 3. 启动后端（务必在 cccc 环境下执行）
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. 启动 Next 演示前端（另开终端）
cd web && npm install && npm run dev
# 浏览器打开 http://localhost:3000 （API 默认 http://127.0.0.1:8000，可通过 web/.env.local 设置 NEXT_PUBLIC_API_URL）
```

演示与交互请使用 **Next 前端**：**http://localhost:3000**（`cd web && npm run dev`）。后端仅提供 API（`http://127.0.0.1:8000/docs`）。检测器在有 GPU 时会自动使用 CUDA。

**语义相似度（攻击管线）**：默认使用离线 `difflib`（无需下载 sentence-transformers）。若需句向量过滤，设置环境变量 `CHINESE_USE_ST=1`。

**固定链 / CAA 环境变量**：Step2/3 固定 HQA/TextHacker（须 detector）。`CCCC_CAA_SYNONYMS`=`chinese`（默认，按句用中文 MLM 候选建近邻）或 `legacy`（需 `third_party/caa/data` 下词向量与 cos pickle）。`CCCC_PIVOT_BEAM`=`1`（默认，Anchor 排序后走中文版束搜）或 `0`（Pivot 不做束搜改写，仅排序高亮）。每步查询预算 API/脚本默认 **1000**（`max_queries_per_step`，上限同）。

---

## 常用命令

| 用途 | 命令 |
|------|------|
| 启动服务 | `uvicorn backend.main:app --host 0.0.0.0 --port 8000` |
| HF 数据 → ≤200 字 JSONL | `python -m scripts.prepare_hf_detection_data --out-dir data/hf_prepared` |
| 样本池 JSONL | `python -m scripts.build_sample_pools` → `data/pools/sample_pool_*.jsonl` |
| 批量评测产物（首页/分析页） | `python -m scripts.build_eval_artifacts` → `data/eval/*.json` 并更新 `data/vuln/summary.json` |
| 训练 Robust | `python -m scripts.train_robust_detector --train data/hf_prepared/train.jsonl --validation data/hf_prepared/val.jsonl --max-steps 500` |
| 对抗池 +（旧）脆弱性脚本 | `python -m scripts.build_adversarial_pool --input data/hf_prepared/val.jsonl --limit 100`；统计可与 `build_eval_artifacts` 或 `run_vulnerability_analysis` 配合 |
| 一键爬取+检测+报告 | `python -m scripts.run_simple_crawl_to_report --max-records 1000` |
| 环境验证 | `python -m scripts.run_smoke_tests` |

---

## 常见问题

| 问题 | 解决 |
|------|------|
| 缺包 | `pip install -r requirements.txt` |
| 端口占用 | 换端口：`--port 8001` |
| 脆弱性页无数据 | 优先 `python -m scripts.build_eval_artifacts`（同步写入 `data/vuln/summary.json` 与扩展字段：按 source 逃逸率、Pivot×替换热力图等）；亦可 `run_vulnerability_analysis`；爬取统计见 `/api/summary`（需 `STATS_JSON_PATH`） |
| 命令报错 No module named 'torch' | 先执行 `conda activate cccc` 激活环境 |
| 端口 8000 已被占用 | `lsof -i :8000` 查 PID，`kill <PID>` 结束进程；或换端口启动 |

---

## 文档

- `scripts/README.md`、`crawler/README.md`：脚本与爬虫说明
- `docs/MCP_DEV_WORKFLOW.md`：基于 MCP 的开发修复流程
- `docs/竞赛项目评估.md`：竞赛提交自检与完成度评估
- `docs/大数据实践赛作品报告.md`：计设大数据实践赛作品报告（按大赛模板撰写）

## Next 演示站（`web/`，唯一推荐入口）

- **/** 比赛向总览大屏（读 `dashboard_bundle` + 池子规模）· **/detect** 样本池点选/随机 + `POST /api/text/compare` · **/redteam** 可攻击池选样 + 逐步链表格 + 结论卡 · **/vuln** 数据挖掘面板：Clean / 攻击下鲁棒性分区、链贡献与累计曲线、长度与数据源分层、Pivot×替换热力图（新产物）、案例表（含 Baseline/Robust 对抗预测与决胜步）+ 单条复盘（`?case=`）
- 仓库内 `frontend/` 为历史静态资源，**不再由后端挂载**；新开发以 `web/` 为准。
