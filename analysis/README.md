# Analysis 模块

对**统一检测结果**做统计分析与简单可视化。

## 统一检测结果格式（JSONL）

与 crawler 统一样本、backend 文本/图像打分对接后的**一条检测结果**为 JSONL 的一行，建议字段：

| 字段 | 说明 |
|------|------|
| `post_id` | 帖子/笔记 ID，与统一样本一致 |
| `platform` | 平台：`xhs` / `tieba` |
| `url` | 帖子链接 |
| `text_ai_score` | 文本 AI 可疑度 (0~1)，无则 `null` |
| `text_label` | 文本判定：`ai` / `human` |
| `image_scores` | 该帖各图 AI 分数列表 |
| `image_labels` | 与 `image_scores` 一一对应的标签列表 |
| `mean_image_score` | 可选，该帖图像平均分；缺失时由 basic_stats 自动计算 |

生成方式示例：对 crawler 统一样本逐条调用 `POST /api/text/score`（取正文 `text`）与 `POST /api/image/score`（取本地下载后的 `image_paths`），再按上述字段合并写出一行 JSON。

## basic_stats

- **脚本**：`analysis/basic_stats.py`
- **功能**：读取检测结果 JSONL，输出汇总统计（JSON）与可选分数分布图（PNG）。

### 用法

```bash
# 在 cccc 环境中
python -m analysis.basic_stats <检测结果.jsonl> [-o 输出目录] [--no-plot]
```

- 默认将 `basic_stats.json` 与 `score_distribution.png` 写入与输入文件同目录；使用 `-o` 指定输出目录。
- `--no-plot` 表示不生成分布图（仅统计 JSON）。

### 自测

```bash
python -m analysis.smoke_test_basic_stats
```

使用临时 JSONL 验证加载、统计与绘图逻辑，无需真实检测结果文件。
