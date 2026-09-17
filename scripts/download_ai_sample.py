#!/usr/bin/env python3
"""下载 AI 人像示例图到 data/sample/ai.jpg"""
import urllib.request
import sys
from pathlib import Path

# AI 人像示例：Pexels AI 图库、Pixabay、thispersondoesnotexist
URLS = [
    "https://images.pexels.com/photos/18069158/pexels-photo-18069158.jpeg",
    "https://cdn.pixabay.com/photo/2023/09/19/11/30/ai-generated-8256974_960_720.jpg",
    "https://thispersondoesnotexist.com/image",
]

def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "sample" / "ai.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
            if len(data) < 2000:
                continue
            out.write_bytes(data)
            print(f"OK: {len(data)} bytes", file=sys.stderr)
            return 0
        except Exception as e:
            print(f"Skip {url[:50]}: {e}", file=sys.stderr)
            continue
    # 降级：生成类 AI 人像风格图（椭圆+肤色渐变，人眼可辨为合成）
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        w, h = 400, 500
        img = Image.new("RGB", (w, h), (240, 230, 220))
        draw = ImageDraw.Draw(img)
        # 椭圆脸部
        margin = 60
        draw.ellipse([margin, margin, w - margin, h - margin], fill=(255, 235, 220), outline=(220, 200, 190))
        # 眼睛
        draw.ellipse([w//2 - 50, h//2 - 30, w//2 - 20, h//2], fill=(80, 60, 50))
        draw.ellipse([w//2 + 20, h//2 - 30, w//2 + 50, h//2], fill=(80, 60, 50))
        # 嘴
        draw.arc([w//2 - 30, h//2 + 20, w//2 + 30, h//2 + 60], 0, 180, fill=(180, 120, 100), width=3)
        arr = np.array(img)
        from PIL import ImageFilter
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1))
        ext = out.suffix or ".jpg"
        img.save(out, "JPEG" if ext.lower() in (".jpg", ".jpeg") else "PNG", quality=90)
        print("Fallback: generated synthetic portrait", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Fallback failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
