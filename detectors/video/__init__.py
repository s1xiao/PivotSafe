# -*- coding: utf-8 -*-
"""轻量视频检测：抽帧 + 图像检测聚合。"""

from .frames import extract_frames
from .detector import VideoLightweightDetector, VideoDetectionResult

__all__ = ["extract_frames", "VideoLightweightDetector", "VideoDetectionResult"]
