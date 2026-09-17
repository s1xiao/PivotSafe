"""图像检测模块自测：用临时生成的图片调用 ImageAIDetector，验证接口与数值范围。"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from .detector import ImageAIDetector


def main() -> None:
    # 生成两张临时图供测试（无 checkpoint 时模型为随机初始化，仅验证流程）
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(2):
            p = Path(tmp) / f"test_{i}.png"
            img = Image.new("RGB", (64, 64), color=(i * 100, 128, 200))
            img.save(p)
            paths.append(str(p))

        detector = ImageAIDetector(checkpoint_path=None, batch_size=4, threshold=0.5)
        print("ImageAIDetector loaded (no checkpoint, random init)")
        print(f"Device: {detector.device}, threshold: {detector.threshold}")

        results = detector.predict(paths)
        for r in results:
            print(f"  {r.path} -> ai_score={r.ai_score:.4f} label={r.label_pred}")
        assert all(0 <= r.ai_score <= 1 for r in results), "ai_score should be in [0,1]"
    print("smoke_test_image passed.")


if __name__ == "__main__":
    main()
