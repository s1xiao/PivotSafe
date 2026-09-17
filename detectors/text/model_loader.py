from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json

from .detector import TextAIDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "text_detector.json"
ROBUST_CONFIG_PATH = PROJECT_ROOT / "configs" / "robust_text_detector.json"


@dataclass
class TextDetectorConfig:
  hf_model_name: str
  device: Optional[str] = None
  max_length: int = 512
  batch_size: int = 8
  threshold: float = 0.5

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "TextDetectorConfig":
    return cls(
      hf_model_name=data.get("hf_model_name", "yuchuantian/AIGC_detector_zhv3"),
      device=data.get("device"),
      max_length=int(data.get("max_length", 512)),
      batch_size=int(data.get("batch_size", 8)),
      threshold=float(data.get("threshold", 0.5)),
    )


def load_text_detector_config(path: Path | str | None = None) -> TextDetectorConfig:
  """
  从 JSON 配置文件加载文本检测模型配置。

  如果未显式指定路径，则默认读取项目根目录下的 `configs/text_detector.json`。
  """
  config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
  with config_path.open("r", encoding="utf-8") as f:
    data = json.load(f)
  return TextDetectorConfig.from_dict(data)


def build_text_detector(config: Optional[TextDetectorConfig] = None) -> TextAIDetector:
  """
  根据配置构建一个 TextAIDetector 实例。
  """
  cfg = config or load_text_detector_config()
  return TextAIDetector(
    model_name=cfg.hf_model_name,
    device=cfg.device,
    max_length=cfg.max_length,
    batch_size=cfg.batch_size,
    threshold=cfg.threshold,
  )


def _resolve_robust_checkpoint_dir() -> Path:
  with ROBUST_CONFIG_PATH.open("r", encoding="utf-8") as f:
    raw = json.load(f)
  cdir = raw.get("checkpoint_dir", "checkpoints/robust_roberta")
  cp = Path(cdir)
  if not cp.is_absolute():
    cp = PROJECT_ROOT / cp
  return cp


def robust_checkpoint_ready() -> bool:
  cp = _resolve_robust_checkpoint_dir()
  return cp.is_dir() and (cp / "config.json").exists()


def load_robust_detector_config() -> TextDetectorConfig:
  """用于推理超参；模型权重路径为 checkpoint_dir。"""
  with ROBUST_CONFIG_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)
  cp = _resolve_robust_checkpoint_dir()
  return TextDetectorConfig(
    hf_model_name=str(cp.resolve()),
    device=data.get("device"),
    max_length=int(data.get("max_length", 256)),
    batch_size=int(data.get("batch_size", 8)),
    threshold=float(data.get("threshold", 0.5)),
  )


def build_robust_text_detector() -> TextAIDetector:
  """
  从 robust_text_detector.json 的 checkpoint_dir 加载微调后的序列分类模型。
  """
  cp = _resolve_robust_checkpoint_dir()
  if not cp.is_dir():
    raise FileNotFoundError(f"robust checkpoint 不存在: {cp}")
  cfg = load_robust_detector_config()
  return TextAIDetector(
    model_name=cfg.hf_model_name,
    device=cfg.device,
    max_length=cfg.max_length,
    batch_size=cfg.batch_size,
    threshold=cfg.threshold,
  )

