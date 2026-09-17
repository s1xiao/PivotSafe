"""Image preprocess for SSP detector. Supports official patch mode (patch_size=32)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

_transform_full = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

_transform_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


def _patch_img(img: Image.Image, patch_size: int, image_size: int = IMAGE_SIZE, num_patch: int = 64) -> Image.Image:
    """Replicate SSP patch selection: sample random crops and keep the lowest-gradient patch."""
    if patch_size <= 0:
        return img

    img = img.resize((image_size, image_size), Image.BILINEAR)
    np_img = np.asarray(img)

    h, w = np_img.shape[:2]
    if patch_size >= h or patch_size >= w:
        return img

    grad_x = np.abs(np_img[:, 1:, :] - np_img[:, :-1, :]).sum(axis=2)
    grad_y = np.abs(np_img[1:, :, :] - np_img[:-1, :, :]).sum(axis=2)

    max_i = h - patch_size
    max_j = w - patch_size

    best_i = 0
    best_j = 0
    best_score = None

    for _ in range(max(1, int(num_patch))):
        i = np.random.randint(0, max_i + 1)
        j = np.random.randint(0, max_j + 1)

        gx = grad_x[i : i + patch_size, j : j + patch_size - 1].sum()
        gy = grad_y[i : i + patch_size - 1, j : j + patch_size].sum()
        score = float(gx + gy)

        if best_score is None or score < best_score:
            best_score = score
            best_i, best_j = i, j

    patch = np_img[best_i : best_i + patch_size, best_j : best_j + patch_size, :]
    return Image.fromarray(patch.astype(np.uint8), mode="RGB")


def load_and_transform(path: Union[str, Path], patch_size: Optional[int] = None) -> torch.Tensor:
    """Load one image and transform into input tensor (C, H, W)."""
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")

    if patch_size is not None and int(patch_size) > 0:
        img = _patch_img(img, patch_size=int(patch_size), image_size=IMAGE_SIZE)
        return _transform_patch(img)

    return _transform_full(img)


def load_and_transform_batch(paths: List[Union[str, Path]], patch_size: Optional[int] = None) -> torch.Tensor:
    """Multiple images to tensor batch (N, C, H, W)."""
    tensors = [load_and_transform(p, patch_size=patch_size) for p in paths]
    return torch.stack(tensors, dim=0)
