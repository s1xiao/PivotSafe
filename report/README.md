# Report 模块

从 **basic_stats.json** 一键生成 HTML 实验报告，供计设提交或内部查阅。

## 用法

在 **cccc** 环境中，于项目根目录执行：

```bash
# 使用 report/basic_stats.json，输出 report/report.html
python -m report.generate_report

# 指定统计文件与输出路径
python -m report.generate_report --stats path/to/basic_stats.json --output path/to/report.html
```

可将 `analysis/basic_stats` 的输出目录设为 `--stats`（例如 `stats_out/basic_stats.json`），生成的 HTML 包含总样本数、文本/图像分数均值、AI 比例及平台分布（表格展示）等。

## 与链路衔接

1. 检测结果 JSONL → `python -m analysis.basic_stats <结果.jsonl> -o stats_out`
2. 得到 `stats_out/basic_stats.json`（及可选 `score_distribution.png`）
3. `python -m report.generate_report --stats stats_out/basic_stats.json --output report/report.html`
