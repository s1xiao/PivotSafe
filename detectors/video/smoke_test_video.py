# -*- coding: utf-8 -*-
"""
视频轻量检测自测：抽帧逻辑 + 可选整链路（抽帧+图像检测）。
不依赖真实视频文件：用 OpenCV 生成短测试视频后抽帧；整链路测试会加载图像模型，耗时较长。
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def _make_test_video(path: Path, num_frames: int = 5) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 5.0, (w, h))
    for i in range(num_frames):
        frame_val = (i * 40) % 256
        arr = np.full((h, w, 3), frame_val, dtype=np.uint8)
        out.write(arr)
    out.release()
    return path.exists()


def main() -> None:
    from detectors.video.frames import extract_frames
    from detectors.video.detector import VideoLightweightDetector, VideoDetectionResult

    # 1. 抽帧单元测试（不加载图像模型）
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  [SKIP] opencv not available")
        return
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        test_video = Path(f.name)
    try:
        if not _make_test_video(test_video, num_frames=5):
            print("  [SKIP] could not create test video")
            return
        paths = extract_frames(test_video, num_frames=5, output_dir=None)
        assert len(paths) >= 1, "expected at least 1 frame"
        assert all(Path(p).exists() for p in paths), "frame files should exist"
        # 使用临时目录时，抽帧目录在 paths[0] 的 parent，可在此清理
        if paths:
            import shutil
            d = Path(paths[0]).parent
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
    finally:
        test_video.unlink(missing_ok=True)

    # 2. 轻量检测整链路：同一段测试视频走 VideoLightweightDetector（会加载图像模型）
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        test_video2 = Path(f.name)
    try:
        if not _make_test_video(test_video2, num_frames=3):
            return
        detector = VideoLightweightDetector(num_frames=3, aggregate="mean")
        result = detector.predict(test_video2)
        assert isinstance(result, VideoDetectionResult)
        assert result.video_path == str(test_video2.resolve())
        assert result.num_frames in (1, 2, 3)
        assert len(result.frame_scores) == result.num_frames
        assert result.label_pred in ("ai", "human")
    finally:
        test_video2.unlink(missing_ok=True)

    print("  video smoke test passed (extract_frames + VideoLightweightDetector)")


if __name__ == "__main__":
    main()
