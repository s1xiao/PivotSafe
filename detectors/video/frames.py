# -*- coding: utf-8 -*-
"""
从视频中轻量抽帧：首帧 + 均匀间隔若干帧。
参考思路：OpenCV VideoCapture，与开源实现如 asmit404/Frame_Extraction 的用法一致；
本模块采用固定帧数、均匀采样以控制算力。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def extract_frames(
    video_path: str | Path,
    num_frames: int = 5,
    output_dir: Optional[str | Path] = None,
) -> List[str]:
    """
    从视频中抽取 num_frames 帧并保存为 JPEG，返回保存后的文件路径列表。
    策略：首帧 + 在时间轴上均匀取其余帧（总帧数不足时尽量取满）。

    :param video_path: 视频文件路径
    :param num_frames: 目标抽帧数（至少 1）
    :param output_dir: 保存目录，若为 None 则使用临时目录（调用方需自行清理）
    :return: 帧图片路径列表；若 OpenCV 不可用或视频无法打开则返回 []
    """
    if cv2 is None:
        return []
    video_path = Path(video_path)
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = max(1, num_frames)
    paths: List[str] = []
    delete_dir = None
    if output_dir is None:
        delete_dir = tempfile.mkdtemp(prefix="cccc_video_frames_")
        output_dir = delete_dir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        if total <= 0:
            # 无法取总帧数时只读第一帧
            ok, frame = cap.read()
            if ok:
                p = out / "frame_0.jpg"
                cv2.imwrite(str(p), frame)
                paths.append(str(p.resolve()))
        else:
            # 首帧 + 均匀取 num_frames-1 帧
            indices = [0]
            if num_frames > 1 and total > 1:
                step = (total - 1) / (num_frames - 1)
                for i in range(1, num_frames):
                    idx = min(int(round(i * step)), total - 1)
                    indices.append(idx)
            for i, frame_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                p = out / f"frame_{i}.jpg"
                cv2.imwrite(str(p), frame)
                paths.append(str(p.resolve()))
    finally:
        cap.release()
        # 不在此处删 delete_dir，由调用方或 detector 在推理后清理

    return paths
