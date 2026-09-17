from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class TextDetectionResult:
  text: str
  ai_score: float
  label_pred: str


class TextAIDetector:
  """
  基于 HuggingFace Transformers 的文本 AI 生成内容检测封装。

  默认使用 YuchuanTian 在 HuggingFace 上发布的 `AIGC_detector_zhv3`，也可以通过配置切换到
  其它变体（如英文模型或 short 版本）。该类只负责推理，不负责训练。
  """

  def __init__(
    self,
    model_name: str,
    device: Optional[str] = None,
    max_length: int = 512,
    batch_size: int = 8,
    threshold: float = 0.5,
  ) -> None:
    self.model_name = model_name
    self.max_length = max_length
    self.batch_size = batch_size
    self.threshold = threshold

    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    self.model.to(self.device)
    self.model.eval()

    self._pos_label_id, self._pos_label_name = self._infer_positive_label()

  def _infer_positive_label(self) -> Tuple[int, str]:
    """
    根据模型 config 推断“AI 文本”对应的标签 id 及名称。

    经验规则：
    - 优先使用 config.id2label 中包含 'ai'/'aigc' 等关键词的标签；
    - 否则退化为假定 label id=1 为 AI 类。
    """
    id2label = getattr(self.model.config, "id2label", None)
    if isinstance(id2label, dict) and id2label:
      # 规范化 key 为 int
      normalized = {}
      for k, v in id2label.items():
        try:
          idx = int(k)
        except (TypeError, ValueError):
          continue
        normalized[idx] = str(v)

      candidates = []
      for idx, name in normalized.items():
        lower = name.lower()
        if "ai" in lower or "aigc" in lower or "machine" in lower:
          candidates.append((idx, name))

      if candidates:
        # 选择 id 最小的一个，以避免多重匹配歧义
        candidates.sort(key=lambda x: x[0])
        return candidates[0]

      # 没有明显的 AI 关键字，就选 label id=1（若存在），否则选最大 id
      if 1 in normalized:
        return 1, normalized[1]
      max_id = max(normalized.keys())
      return max_id, normalized[max_id]

    # 若 config 未提供 id2label，则默认使用 id=1
    return 1, "AI"

  @property
  def positive_label(self) -> str:
    return self._pos_label_name

  def score(self, texts: Sequence[str]) -> List[float]:
    """
    对一批文本进行打分，返回每条文本为“AI 生成”的概率（0-1）。
    """
    if not texts:
      return []

    scores: List[float] = []
    with torch.no_grad():
      for i in range(0, len(texts), self.batch_size):
        batch = list(texts[i : i + self.batch_size])
        encoded = self.tokenizer(
          batch,
          padding=True,
          truncation=True,
          max_length=self.max_length,
          return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        logits = outputs.logits  # shape: (batch, num_labels)
        probs = torch.softmax(logits, dim=-1)
        pos_probs = probs[:, self._pos_label_id]
        scores.extend(pos_probs.detach().cpu().tolist())
    return scores

  def predict(self, texts: Sequence[str]) -> List[TextDetectionResult]:
    """
    对一批文本进行检测，返回包含文本、AI 可疑度以及预测标签的结果列表。
    """
    scores = self.score(texts)
    results: List[TextDetectionResult] = []
    for text, score in zip(texts, scores):
      label = "ai" if score >= self.threshold else "human"
      results.append(
        TextDetectionResult(
          text=text,
          ai_score=float(score),
          label_pred=label,
        )
      )
    return results

