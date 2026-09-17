#!/usr/bin/env python3
"""生成 AI 风格人像示例图（椭圆脸+五官，人眼可辨为合成）"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "data" / "sample" / "ai.jpg"
out.parent.mkdir(parents=True, exist_ok=True)

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

w, h = 400, 500
img = Image.new("RGB", (w, h), (245, 238, 230))
draw = ImageDraw.Draw(img)
# 椭圆脸部（肤色）
margin = 50
draw.ellipse([margin, margin, w - margin, h - margin], fill=(255, 240, 225), outline=(230, 210, 200))
# 眼睛
draw.ellipse([w//2 - 55, h//2 - 40, w//2 - 25, h//2 - 5], fill=(70, 55, 45))
draw.ellipse([w//2 + 25, h//2 - 40, w//2 + 55, h//2 - 5], fill=(70, 55, 45))
# 眉毛
draw.arc([w//2 - 55, h//2 - 65, w//2 - 25, h//2 - 35], 180, 360, fill=(100, 80, 60), width=4)
draw.arc([w//2 + 25, h//2 - 65, w//2 + 55, h//2 - 35], 180, 360, fill=(100, 80, 60), width=4)
# 嘴
draw.arc([w//2 - 35, h//2 + 25, w//2 + 35, h//2 + 70], 0, 180, fill=(200, 140, 120), width=4)
# 鼻子
draw.line([(w//2, h//2 - 20), (w//2 + 5, h//2 + 30)], fill=(220, 180, 160), width=3)
# 平滑处理
img = img.filter(ImageFilter.GaussianBlur(0.8))
img.save(out, "JPEG", quality=92)
print(f"Generated {out}", flush=True)
