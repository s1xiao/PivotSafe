from __future__ import annotations

"""
简单网页爬虫：从公开网站抓取真实中文文本，用于跑检测→报告链路。

支持 RSS 解析 + 多站点聚合，可扩展至 1000+ 条。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List
import hashlib
import json
import time
from urllib.parse import urljoin, urlparse

import requests
from parsel import Selector

from .rss_crawler import parse_rss_feed, rss_items_to_samples


_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class Article:
    url: str
    title: str
    text: str
    image_urls: list[str] = field(default_factory=list)


def _fetch(url: str, timeout: int = 10) -> str:
    resp = requests.get(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def _extract_ithome_links(index_html: str, max_links: int) -> List[str]:
    sel = Selector(index_html)
    hrefs = sel.css("a::attr(href)").getall()
    links: list[str] = []
    for href in hrefs:
        if not href:
            continue
        href = href.strip()
        if href.startswith("//"):
            href = "https:" + href
        # 只取 it 之家文章页，粗略过滤
        if not href.startswith("https://www.ithome.com/"):
            continue
        if href.endswith("/") or href.endswith(".html") or href.endswith(".htm"):
            if href not in links:
                links.append(href)
        if len(links) >= max_links:
            break
    return links


def _extract_solidot_links(index_html: str, max_links: int) -> List[str]:
    """
    从 Solidot 首页提取若干文章链接。
    通过 href 中包含 'story?sid=' 来粗略筛选。
    """
    sel = Selector(index_html)
    hrefs = sel.css("a::attr(href)").getall()
    links: list[str] = []
    for href in hrefs:
        if not href:
            continue
        href = href.strip()
        if "story?sid=" not in href:
            continue
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://www.solidot.org" + href
        if not href.startswith("https://www.solidot.org/"):
            continue
        if href not in links:
            links.append(href)
        if len(links) >= max_links:
            break
    return links


def _extract_article(html: str, url: str) -> Article | None:
    sel = Selector(html)
    title = sel.css("title::text").get() or ""
    # 不同站点常在 title 中附带站点名，用第一个分隔段
    title = title.strip().split("_")[0].split("-")[0][:200]

    # 尝试常见正文容器
    paragraphs = sel.css("div#paragraph p::text, div.content p::text").getall()
    if not paragraphs:
        paragraphs = sel.css("article p::text").getall()
    if not paragraphs:
        paragraphs = sel.css("p::text").getall()
    text = "\n".join(p.strip() for p in paragraphs if p.strip())
    text = text[:5000]
    if not title and not text:
        return None
    if not title:
        title = text[:40]
    if not text:
        text = title

    # 尝试抽取代表性图片 URL
    img_srcs = sel.css("article img::attr(src), div.content img::attr(src), img::attr(src)").getall()
    base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(url))
    image_urls: list[str] = []
    for src in img_srcs:
        if not src:
            continue
        src = src.strip()
        full = urljoin(base, src)
        if full.lower().startswith("http") and full not in image_urls:
            image_urls.append(full)
        if len(image_urls) >= 3:
            break

    return Article(url=url, title=title, text=text, image_urls=image_urls)


def _article_to_sample(site: str, art: Article) -> dict:
    post_id = hashlib.md5(art.url.encode("utf-8")).hexdigest()
    return {
        "platform": site,
        "post_id": post_id,
        "url": art.url,
        "title": art.title,
        "text": art.text,
        "image_urls": art.image_urls,
    }


def _extract_article_for_rss(html: str, url: str) -> tuple[str, str, list] | None:
    """适配 _extract_article 供 RSS 详情页解析使用，返回 (title, text, image_urls)。"""
    art = _extract_article(html, url)
    if not art:
        return None
    return (art.title, art.text, art.image_urls)


def crawl_ithome_articles(max_records: int = 100, delay: float = 0.5) -> list[dict]:
    """
    抓取 IT 之家文章。从 RSS 取链接，再批量抓详情页取正文。
    """
    items = parse_rss_feed(
        "https://www.ithome.com/rss/",
        max_items=max_records,
        use_description_if_long=False,
        min_desc_len=200,
        fetch_detail=lambda u: _fetch(u),
        extract_article=_extract_article_for_rss,
        delay_after_fetch=delay,
    )
    return rss_items_to_samples("ithome", items)


def crawl_solidot_articles(max_records: int = 100, delay: float = 0.3) -> list[dict]:
    """
    抓取 Solidot 文章。RSS 的 description 含全文，直接使用无需抓详情。
    """
    items = parse_rss_feed(
        "https://www.solidot.org/index.rss",
        max_items=max_records,
        use_description_if_long=True,
        min_desc_len=100,
    )
    return rss_items_to_samples("solidot", items)


def crawl_oschina_articles(max_records: int = 100, delay: float = 0.3) -> list[dict]:
    """
    抓取开源中国 (OSChina) 新闻。RSS 的 description 有内容，直接使用。
    """
    items = parse_rss_feed(
        "https://www.oschina.net/news/rss",
        max_items=max_records,
        use_description_if_long=True,
        min_desc_len=80,
    )
    return rss_items_to_samples("oschina", items)


def crawl_linuxcn_articles(max_records: int = 100, delay: float = 0.3) -> list[dict]:
    """
    抓取 Linux 中国文章。RSS 解析。
    """
    items = parse_rss_feed(
        "https://linux.cn/rss.xml",
        max_items=max_records,
        use_description_if_long=True,
        min_desc_len=80,
    )
    return rss_items_to_samples("linuxcn", items)


def crawl_36kr_articles(max_records: int = 100, delay: float = 0.3) -> list[dict]:
    """
    抓取 36 氪文章。RSS 解析。
    """
    items = parse_rss_feed(
        "https://www.36kr.com/feed",
        max_items=max_records,
        use_description_if_long=True,
        min_desc_len=80,
    )
    return rss_items_to_samples("36kr", items)


def _fetch_v2ex_topics(api_path: str) -> list:
    """请求 V2EX API 返回主题列表。"""
    resp = requests.get(
        f"https://www.v2ex.com/api/topics/{api_path}",
        headers={"User-Agent": _UA},
        timeout=15,
    )
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    items = json.loads(resp.text)
    return items if isinstance(items, list) else []


def crawl_v2ex_articles(max_records: int = 120, delay: float = 0.3) -> list[dict]:
    """
    抓取 V2EX 主题。聚合 hot + latest 两路 API 以获取更多条数。
    参考：https://www.v2ex.com/api/topics/hot.json, topics/latest.json
    """
    all_items: list[dict] = []
    for api_path in ("hot.json", "latest.json"):
        try:
            items = _fetch_v2ex_topics(api_path)
            all_items.extend(items)
        except Exception:
            continue
        if delay > 0:
            time.sleep(delay)

    seen_urls: set[str] = set()
    samples: list[dict] = []
    for item in all_items:
        if len(samples) >= max_records:
            break
        url = item.get("url") or ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = (item.get("title") or "")[:500]
        content = (item.get("content") or "")[:5000]
        text = (title + "\n" + content).strip() or title or content
        if not text:
            continue
        post_id = hashlib.md5(url.encode("utf-8")).hexdigest()
        samples.append({
            "platform": "v2ex",
            "post_id": post_id,
            "url": url,
            "title": title,
            "text": text,
            "image_urls": [],
        })
    return samples


def crawl_multi_site_articles(
    sites: Iterable[str],
    max_records: int = 1000,
    delay: float = 1.0,
) -> list[dict]:
    """
    聚合多个公开站点的文章到统一样本列表。

    :param sites: 站点标识（如 'ithome', 'solidot'）
    :param max_records: 总样本上限
    :param delay: 单站点请求间隔
    """
    site_funcs = {
        "ithome": crawl_ithome_articles,
        "solidot": crawl_solidot_articles,
        "v2ex": crawl_v2ex_articles,
        "oschina": crawl_oschina_articles,
        "linuxcn": crawl_linuxcn_articles,
        "36kr": crawl_36kr_articles,
    }
    selected = [s for s in sites if s in site_funcs]
    if not selected:
        selected = ["ithome"]

    per_site = max(max_records // len(selected), 1)
    all_samples: list[dict] = []
    for s in selected:
        remaining = max_records - len(all_samples)
        if remaining <= 0:
            break
        n = min(per_site, remaining)
        try:
            site_samples = site_funcs[s](max_records=n, delay=delay)
        except Exception:
            site_samples = []
        all_samples.extend(site_samples)
    return all_samples


def main() -> None:
    out = crawl_multi_site_articles(
        sites=["ithome", "solidot", "v2ex", "oschina", "linuxcn"],
        max_records=100,
    )
    from crawler.unified_io import write_unified_jsonl

    out_path = Path("data/unified_web_debug.jsonl")
    write_unified_jsonl(out, out_path)
    print(f"wrote {len(out)} samples to {out_path}")


if __name__ == "__main__":
    main()

