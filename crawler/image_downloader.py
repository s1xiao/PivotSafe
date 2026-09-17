from __future__ import annotations

"""
从统一样本 JSONL 中的 image_urls 下载图片到本地，并补充 image_paths 字段。

输入：统一样本 JSONL（每行一个 dict，含 image_urls[list[str]]）
输出：新的 JSONL，除原字段外增加 image_paths[list[str]]，供图像检测使用。
"""

import json
from pathlib import Path
from typing import Iterable

import requests


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)[:80]


def download_images_from_unified_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    images_root: str | Path = "data/images",
    max_images_per_post: int = 3,
    max_total_images: int = 2000,
    timeout: int = 10,
) -> None:
    """
    读取统一样本 JSONL，按 image_urls 下载图片到本地，并写出带 image_paths 的新 JSONL。
    """
    inp = Path(input_path)
    outp = Path(output_path)
    images_root = Path(images_root)
    images_root.mkdir(parents=True, exist_ok=True)
    outp.parent.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    with inp.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_urls: Iterable[str] = rec.get("image_urls") or []
            image_paths: list[str] = []
            if image_urls and total_downloaded < max_total_images:
                plat = rec.get("platform") or "web"
                post_id = rec.get("post_id") or _safe_filename(rec.get("url", "")) or "noid"
                base_dir = images_root / str(plat) / _safe_filename(str(post_id))
                base_dir.mkdir(parents=True, exist_ok=True)

                for idx, url in enumerate(image_urls):
                    if idx >= max_images_per_post or total_downloaded >= max_total_images:
                        break
                    if not url:
                        continue
                    try:
                        resp = requests.get(url, timeout=timeout, stream=True)
                        resp.raise_for_status()
                    except Exception:
                        continue
                    ext = ".jpg"
                    ct = resp.headers.get("Content-Type", "").lower()
                    if "png" in ct:
                        ext = ".png"
                    elif "jpeg" in ct or "jpg" in ct:
                        ext = ".jpg"
                    fname = base_dir / f"{idx}{ext}"
                    try:
                        with fname.open("wb") as fimg:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if not chunk:
                                    continue
                                fimg.write(chunk)
                    except Exception:
                        if fname.exists():
                            fname.unlink(missing_ok=True)
                        continue
                    image_paths.append(str(fname))
                    total_downloaded += 1

            rec["image_paths"] = image_paths
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
