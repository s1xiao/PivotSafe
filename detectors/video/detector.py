# -*- coding: utf-8 -*-
"""
轻量视频检测：对视频抽帧后调用图像检测，聚合得到整段视频的 AI 可疑度。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from .frames import extract_frames


@dataclass
class VideoDetectionResult:
    """单段视频的检测结果。"""
    video_path: str
    frame_scores: List[float] = field(default_factory=list)
    mean_score: float = 0.0
    max_score: float = 0.0
    label_pred: str = "human"
    num_frames: int = 0


class VideoLightweightDetector:
    """
    轻量视频检测：抽帧 → 图像检测 → 聚合（均值/最大值）。
    不加载图像模型，由调用方传入已构建的 ImageAIDetector 或通过 set_image_detector 注入。
    """

    def __init__(
        self,
        image_detector=None,
        num_frames: int = 5,
        threshold: float = 0.5,
        aggregate: str = "mean",  # "mean" | "max"
    ) -> None:
        self._image_detector = image_detector
        self.num_frames = num_frames
        self.threshold = threshold
        self.aggregate = aggregate if aggregate in ("mean", "max") else "mean"

    def set_image_detector(self, detector) -> None:
        """注入图像检测器（如 backend 的 get_image_detector() 返回的实例）。"""
        self._image_detector = detector

    def _get_image_detector(self):
        if self._image_detector is not None:
            return self._image_detector
        # 延迟导入，避免 backend 未启动时强依赖
        import json
        from detectors.image.detector import ImageAIDetector

        project_root = Path(__file__).resolve().parents[2]
        cfg_path = project_root / "configs" / "image_detector.json"
        cfg = {}
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8-sig") as f:
                cfg = json.load(f)

        checkpoint_value = cfg.get("checkpoint_path")
        checkpoint_path: Path | None = None
        if isinstance(checkpoint_value, str) and checkpoint_value.strip():
            checkpoint_path = Path(checkpoint_value.strip())
            if not checkpoint_path.is_absolute():
                checkpoint_path = (project_root / checkpoint_path).resolve()

        self._image_detector = ImageAIDetector(
            checkpoint_path=checkpoint_path,
            device=cfg.get("device"),
            batch_size=cfg.get("batch_size", 8),
            threshold=cfg.get("threshold", 0.5),
            patch_size=cfg.get("patch_size", 32),
        )
        return self._image_detector

    def predict(self, video_path: Union[str, Path]) -> VideoDetectionResult:
        """
        对单个视频抽帧、做图像检测并聚合。
        :param video_path: 视频文件路径
        :return: VideoDetectionResult
        """
        video_path = str(Path(video_path).resolve())
        frame_paths = extract_frames(video_path, num_frames=self.num_frames, output_dir=None)
        result = VideoDetectionResult(video_path=video_path, num_frames=len(frame_paths))
        if not frame_paths:
            return result

        try:
            detector = self._get_image_detector()
            scores = detector.score(frame_paths)
        finally:
            # 清理抽帧临时目录（extract_frames 用 mkdtemp 时父目录在 paths[0] 的 parent）
            if frame_paths:
                try:
                    d = Path(frame_paths[0]).parent
                    if "cccc_video_frames_" in d.name:
                        shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass

        result.frame_scores = [float(s) for s in scores]
        if result.frame_scores:
            result.mean_score = sum(result.frame_scores) / len(result.frame_scores)
            result.max_score = max(result.frame_scores)
            agg = result.max_score if self.aggregate == "max" else result.mean_score
            result.label_pred = "ai" if agg >= self.threshold else "human"
        return result



