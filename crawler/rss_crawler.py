# -*- coding: utf-8 -*-
"""
RSS 爬虫：解析 RSS 订阅源，提取文章链接与正文，用于扩展至 1000+ 条数据。
支持 feedparser 解析，对 description 足够长的项直接使用，否则抓取详情页。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from html import unescape
from typing import Callable, List, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# description 长度超过此值则直接使用，不抓详情页
_MIN_DESC_LEN = 150


@dataclass
class RSSItem:
    url: str
    title: str
    text: str
    image_urls: list[str]


def _strip_html(html: str) -> str:
    """粗略去除 HTML 标签，保留纯文本。"""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return unescape(text).strip()


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


def parse_rss_feed(
    feed_url: str,
    max_items: int = 100,
    use_description_if_long: bool = True,
    min_desc_len: int = _MIN_DESC_LEN,
    fetch_detail: Optional[Callable[[str], str]] = None,
    extract_article: Optional[Callable[[str, str], Optional[tuple[str, str, list]]]] = None,
    delay_after_fetch: float = 0.0,
) -> List[RSSItem]:
    """
    解析 RSS 订阅源，返回 (url, title, text, image_urls) 列表。

    :param feed_url: RSS 地址
    :param max_items: 最多取多少条
    :param use_description_if_long: 若 description 足够长则直接用，不抓详情
    :param min_desc_len: 视为「足够长」的最小字符数
    :param fetch_detail: 抓取详情页的函数，接收 url 返回 html
    :param extract_article: 从 html 提取 (title, text, image_urls)，若 None 则用默认逻辑
    """
    feed = feedparser.parse(
        feed_url,
        request_headers={"User-Agent": _UA},
        agent=_UA,
    )
    if feed.bozo and not feed.entries:
        return []

    items: List[RSSItem] = []
    seen_urls: set[str] = set()

    for entry in feed.entries[:max_items]:
        link = getattr(entry, "link", None) or ""
        if not link or link in seen_urls:
            continue
        seen_urls.add(link)

        title = _strip_html(getattr(entry, "title", "") or "")
        desc = _strip_html(getattr(entry, "description", "") or getattr(entry, "summary", "") or "")
        desc = desc[:5000]

        if use_description_if_long and len(desc) >= min_desc_len:
            text = desc
            image_urls: list[str] = []
        else:
            if fetch_detail and extract_article:
                try:
                    html = fetch_detail(link)
                    if delay_after_fetch > 0:
                        time.sleep(delay_after_fetch)
                    out = extract_article(html, link)
                    if out:
                        title, text, image_urls = out
                    else:
                        text = desc or title
                        image_urls = []
                except Exception:
                    text = desc or title
                    image_urls = []
            else:
                text = desc or title
                image_urls = []

        text = (text or title or "").strip()[:5000]
        if not title and not text:
            continue
        if not title:
            title = text[:80]
        if not text:
            text = title

        items.append(RSSItem(url=link, title=title, text=text, image_urls=image_urls or []))

    return items


def rss_items_to_samples(site: str, items: List[RSSItem]) -> list[dict]:
    """将 RSSItem 转为统一样本格式。"""
    samples: list[dict] = []
    for item in items:
        post_id = hashlib.md5(item.url.encode("utf-8")).hexdigest()
        samples.append({
            "platform": site,
            "post_id": post_id,
            "url": item.url,
            "title": item.title,
            "text": item.text,
            "image_urls": item.image_urls,
        })
    return samples
