# -*- coding: utf-8 -*-
"""
中文替换管线：分词 + 词性过滤 + MLM 候选 + 语义相似度过滤 + 扰动率限制。
供 Pivot 与 Attacker2/3 共用。
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import jieba
import jieba.posseg as pseg
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 跳过高频虚词、标点类（词性）
DEFAULT_SKIP_POS_PREFIXES = ("u", "y", "e", "o", "w", "x", "m", "q", "t", "p")


@dataclass
class TokenSpan:
    word: str
    pos: str
    start: int
    end: int


@dataclass
class ChineseReplacePipelineConfig:
    mlm_model_name: str = "hfl/chinese-roberta-wwm-ext"
    sim_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_mlm_candidates: int = 12
    min_sim: float = 0.72
    max_replace_ratio: float = 0.35
    device: Optional[str] = None
    # 默认离线：字符级 difflib；设置环境变量 CHINESE_USE_ST=1 时尝试加载 sentence-transformers
    prefer_sentence_transformers: bool = False


@dataclass
class ScoreContext:
    """对 victim detector 的封装：返回 ai_score in [0,1]。"""

    score_fn: Callable[[str], float]
    threshold: float = 0.5
    positive_is_ai: bool = True

    def ai_score(self, text: str) -> float:
        return float(self.score_fn(text))

    def label(self, text: str) -> str:
        s = self.ai_score(text)
        return "ai" if s >= self.threshold else "human"

    def is_attack_success(self, orig_label: str, text: str) -> bool:
        """以 baseline 预测从 orig_label 翻转为目标视为成功（通常 orig 为 ai，目标 human）。"""
        new_l = self.label(text)
        return new_l != orig_label


class ChineseReplacePipeline:
    def __init__(self, cfg: Optional[ChineseReplacePipelineConfig] = None) -> None:
        self.cfg = cfg or ChineseReplacePipelineConfig()
        if os.environ.get("CHINESE_USE_ST", "").lower() in ("1", "true", "yes"):
            self.cfg.prefer_sentence_transformers = True
        dev = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self._tok = AutoTokenizer.from_pretrained(self.cfg.mlm_model_name)
        self._mlm = AutoModelForMaskedLM.from_pretrained(self.cfg.mlm_model_name)
        self._mlm.to(self.device)
        self._mlm.eval()
        self._sim_model = None
        self._sim_tok = None

    def _ensure_sim(self) -> None:
        if self._sim_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._sim_model = SentenceTransformer(self.cfg.sim_model_name, device=self.device)
        self._sim_tok = None

    def tokenize_with_pos(self, text: str) -> List[TokenSpan]:
        out: List[TokenSpan] = []
        for w, start, end in jieba.tokenize(text):
            if not w.strip():
                continue
            seg = list(pseg.cut(w))
            flag = seg[0].flag if seg else "x"
            out.append(TokenSpan(word=w, pos=flag, start=start, end=end))
        return out

    def filter_pos(self, spans: Sequence[TokenSpan]) -> List[TokenSpan]:
        good: List[TokenSpan] = []
        for sp in spans:
            fp = (sp.pos or "x").lower()
            if any(fp.startswith(p) for p in DEFAULT_SKIP_POS_PREFIXES):
                continue
            if len(sp.word) < 2 and not re.match(r"[\u4e00-\u9fff]", sp.word):
                continue
            if re.match(r"^[\d\s\W]+$", sp.word):
                continue
            good.append(sp)
        return good

    def mlm_candidates_for_span(self, text: str, sp: TokenSpan, top_k: Optional[int] = None) -> List[str]:
        k = top_k or self.cfg.max_mlm_candidates
        if sp.start < 0 or sp.end > len(text) or sp.start >= sp.end:
            return []
        masked = text[: sp.start] + self._tok.mask_token + text[sp.end :]
        enc = self._tok(
            masked,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        enc = {x: y.to(self.device) for x, y in enc.items()}
        mid = enc["input_ids"][0].tolist().index(self._tok.mask_token_id)
        with torch.no_grad():
            logits = self._mlm(**enc).logits[0, mid]
            probs, ids = torch.topk(F.log_softmax(logits, dim=-1), k=min(k + 5, logits.numel()))
        cands: List[str] = []
        for i in ids.tolist():
            w = self._tok.decode([i], skip_special_tokens=True).strip()
            if not w or w == sp.word:
                continue
            if len(w) > len(sp.word) + 4:
                continue
            cands.append(w)
            if len(cands) >= k:
                break
        return cands

    def semantic_filter(self, original: str, candidate: str) -> float:
        if not self.cfg.prefer_sentence_transformers:
            return float(difflib.SequenceMatcher(None, original, candidate).ratio())
        try:
            self._ensure_sim()
        except Exception:
            return float(difflib.SequenceMatcher(None, original, candidate).ratio())
        assert self._sim_model is not None
        emb = self._sim_model.encode([original, candidate], convert_to_tensor=True, show_progress_bar=False)
        a, b = emb[0], emb[1]
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    def max_replacements(self, n_tokens: int) -> int:
        return max(1, int(n_tokens * self.cfg.max_replace_ratio + 0.999))
