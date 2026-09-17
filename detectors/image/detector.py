"""Image AI-generation detector wrapper based on SSP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch

from .preprocess import load_and_transform_batch
from .ssp_net.ssp import ssp


@dataclass
class ImageDetectionResult:
    path: str
    ai_score: float
    label_pred: str


class ImageAIDetector:
    """SSP image detector. Higher ai_score means more likely AI-generated."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        threshold: float = 0.5,
        patch_size: Optional[int] = 32,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self.batch_size = batch_size
        self.threshold = threshold
        self.patch_size = int(patch_size) if patch_size is not None else None
        if self.patch_size is not None and self.patch_size <= 0:
            self.patch_size = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = ssp(pretrain=False)
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"image checkpoint not found: {self.checkpoint_path}")
            state = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

    def score(self, image_paths: Sequence[Union[str, Path]]) -> List[float]:
        """Return AI-generation probability [0,1] for each image."""
        paths = [Path(p) for p in image_paths]
        if not paths:
            return []

        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                batch_paths = paths[i : i + self.batch_size]
                x = load_and_transform_batch(batch_paths, patch_size=self.patch_size).to(self.device)
                logits = self.model(x).ravel()
                # SSP uses label 1=real, 0=AI.
                prob_ai = (1.0 - torch.sigmoid(logits)).cpu().tolist()
                scores.extend(prob_ai)
        return scores

    def predict(self, image_paths: Sequence[Union[str, Path]]) -> List[ImageDetectionResult]:
        paths = [str(p) for p in image_paths]
        sc = self.score(paths)
        return [
            ImageDetectionResult(
                path=p,
                ai_score=float(s),
                label_pred="ai" if s >= self.threshold else "human",
            )
            for p, s in zip(paths, sc)
        ]
