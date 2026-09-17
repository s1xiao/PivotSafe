# -*- coding: utf-8 -*-
"""
为 third_party CAA（HQA / TextHacker）提供**中文近义词行**，替代 counter-fitted + cos_sim pickle。

- 按当前句子构建临时 word2idx / idx2word
- cos_sim[i] 为 (sim, j) 元组列表，满足 list(zip(*cos_sim[i])) -> (sims, idxs)
- 生成临时「伪嵌入」文件，满足 hqa_attack 对 embed_func 的读文件逻辑
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Tuple

import jieba

from attacks.chinese_replace import ChineseReplacePipeline

# HQA 随机采样 syn_idx in 0..49
_MIN_SYN_ROWS = 50
_EMBED_DIM = 300


class ChineseSynonymCos:
    """可下标访问；行格式与 CAA pickle cos 一致。"""

    _cccc_chinese = True

    def __init__(self, rows: Dict[int, List[Tuple[float, int]]]) -> None:
        self._rows = rows

    def __getitem__(self, idx: int) -> List[Tuple[float, int]]:
        return self._rows.get(idx, [])


def _pad_syn_row(row: List[Tuple[float, int]], self_idx: int) -> List[Tuple[float, int]]:
    if not row:
        row = [(0.5, self_idx)]
    out = list(row)
    while len(out) < _MIN_SYN_ROWS:
        out.append(out[len(out) % len(row)])
    return out[: max(_MIN_SYN_ROWS, len(out))]


def build_chinese_caa_vocab_for_text(
    text: str,
    pipe: ChineseReplacePipeline,
    *,
    max_candidates_per_word: int = 24,
) -> Tuple[Dict[str, int], Dict[int, str], ChineseSynonymCos, str]:
    """
    返回 word2idx, idx2word, cos_adapter, embed_file_path（临时文件，调用方可删除）。
    """
    text = text.strip()
    spans_all = pipe.tokenize_with_pos(text)
    spans_f = pipe.filter_pos(spans_all)

    # 词 -> [(sim, cand_str), ...]
    syn_edges: Dict[str, List[Tuple[float, str]]] = {}
    for sp in spans_f:
        cands = pipe.mlm_candidates_for_span(text, sp, top_k=max_candidates_per_word + 6)
        scored: List[Tuple[float, str]] = []
        for c in cands:
            if not c or c == sp.word:
                continue
            sim = float(pipe.semantic_filter(sp.word, c))
            if sim < pipe.cfg.min_sim:
                continue
            scored.append((sim, c))
        scored.sort(key=lambda x: -x[0])
        syn_edges[sp.word] = scored[:max_candidates_per_word]

    # 全词表：句中所有 jieba 词 + 所有候选
    words_in_order: List[str] = []
    seen: set[str] = set()
    for w in jieba.cut(text):
        if w not in seen:
            seen.add(w)
            words_in_order.append(w)
    for w, pairs in syn_edges.items():
        for _s, c in pairs:
            if c not in seen:
                seen.add(c)
                words_in_order.append(c)

    word2idx: Dict[str, int] = {w: i for i, w in enumerate(words_in_order)}
    idx2word: Dict[int, str] = {i: w for w, i in word2idx.items()}
    n = len(word2idx)

    rows: Dict[int, List[Tuple[float, int]]] = {}
    for w, pairs in syn_edges.items():
        if w not in word2idx:
            continue
        i = word2idx[w]
        row: List[Tuple[float, int]] = []
        for sim, cand in pairs:
            if cand in word2idx:
                row.append((float(sim), word2idx[cand]))
        rows[i] = _pad_syn_row(row, i)

    # 为未出现在 syn_edges 的词补自环行，避免 cos 空行
    for w, i in word2idx.items():
        if i not in rows:
            rows[i] = _pad_syn_row([], i)

    cos = ChineseSynonymCos(rows)

    fd, embed_path = tempfile.mkstemp(suffix="_cccc_zh_embed.txt", text=True)
    os.close(fd)
    with open(embed_path, "w", encoding="utf-8") as ef:
        for idx in range(n):
            w = idx2word[idx]
            vec = " ".join(["0.0"] * _EMBED_DIM)
            ef.write(f"{w} {vec}\n")

    return word2idx, idx2word, cos, embed_path
