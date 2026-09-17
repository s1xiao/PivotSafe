# Crawler 模块

本目录提供**数据采集与统一样本格式**：

- **公开站点**（`simple_web_crawler.py` + `rss_crawler.py`）：IT之家、Solidot、V2EX、开源中国、Linux 中国、36氪 等，RSS + API 聚合，无需登录，可扩展至 1000+ 条。

## 统一样本格式

`unified_io.write_unified_jsonl` 将爬取结果写入 JSONL，供后续检测与分析使用：

| 字段 | 说明 |
|------|------|
| `platform` | 平台：`ithome` / `solidot` / `v2ex` / `oschina` / `linuxcn` / `36kr` |
| `post_id` | 帖子 ID（URL 的 MD5） |
| `url` | 文章链接 |
| `title` | 标题 |
| `text` | 正文 |
| `image_urls` | 图片 URL 列表 |

## 使用示例

```python
from crawler.simple_web_crawler import crawl_multi_site_articles
from crawler.unified_io import write_unified_jsonl

samples = crawl_multi_site_articles(
    sites=["ithome", "solidot", "v2ex", "oschina", "linuxcn", "36kr"],
    max_records=1000,
)
write_unified_jsonl(samples, "data/unified_web_corpus.jsonl")
```

## 自测

```bash
python -m crawler.smoke_test_unified_io
```

## 目录说明

- `simple_web_crawler.py`：多站点聚合爬虫（RSS + 详情页抓取）
- `rss_crawler.py`：RSS 解析与统一样本转换
- `unified_io.py`：`write_unified_jsonl` 写出
- `image_downloader.py`：从 `image_urls` 下载到本地，生成 `image_paths`
