# -*- coding: utf-8 -*-
"""
CAA 副本与桥接（third_party/caa + caa_cccc_bridge）功能测试；固定链 Step2/3 固定 HQA/TextHacker（中文近义词）。

在项目根、conda 环境 cccc 下执行:
  python -m scripts.test_caa_integration

会加载 Baseline 文本检测器（较慢）；可用环境变量跳过重测试:
  CCCC_SKIP_CAA_SLOW=1  仅跑不加载 HF 模型的用例
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _jieba_tokens(*texts: str) -> list:
    import jieba

    s = set()
    for t in texts:
        s.update(jieba.cut(t.strip()))
    return sorted(s, key=len, reverse=True)


def _write_tiny_caa_data(tmp: Path, seed_texts: tuple[str, ...]) -> tuple[Path, Path]:
    """构造与 hqaattack 中 zip(*(cos_sim[idx])) 兼容的 cos 字典。"""
    words_set = set(_jieba_tokens(*seed_texts))
    for t in seed_texts:
        for ch in t.strip():
            if "\u4e00" <= ch <= "\u9fff":
                words_set.add(ch)
    words = sorted(words_set, key=len, reverse=True)
    while len(words) < 80:
        words.append(f"pad{len(words)}")
    n = len(words)
    dim = 8
    emb = tmp / "counter-fitted-vectors.txt"
    with emb.open("w", encoding="utf-8") as f:
        for i, w in enumerate(words):
            nums = " ".join(f"{(i + j) * 0.001:.6f}" for j in range(dim))
            f.write(f"{w} {nums}\n")
    cos: dict = {}
    for i in range(n):
        sims: list = []
        idxs: list = []
        for k in range(1, min(55, n)):
            j = (i + k) % n
            sims.append(max(0.1, 0.95 - k * 0.015))
            idxs.append(j)
        cos[i] = (sims, idxs)
    # 部分 CAA 路径会以 -1 为索引查 cos_sim，测试数据给退化邻接
    for k in range(-5, 0):
        cos[k] = ([0.92, 0.88], [0, min(1, n - 1)])
    cp = tmp / "cos_sim_matrix.pkl"
    with cp.open("wb") as f:
        pickle.dump(cos, f)
    return emb, cp


class TestCaaImports(unittest.TestCase):
    def test_third_party_caa_imports(self):
        from third_party.caa import criteria_1
        from third_party.caa.tool import (
            get_global_manager,
            initialize_cccc_injected_components,
            reset_global_manager,
        )

        self.assertTrue(hasattr(criteria_1, "get_pos"))
        reset_global_manager()
        mgr = get_global_manager()
        self.assertFalse(mgr.is_initialized)


class TestCriteriaZh(unittest.TestCase):
    def test_get_pos_chinese_uses_jieba(self):
        from third_party.caa.criteria_1 import get_pos

        words = list(__import__("jieba").cut("人工智能正在改变生活"))
        tags = get_pos(words)
        self.assertEqual(len(tags), len(words))
        for t in tags:
            self.assertIn(t, ("NOUN", "VERB", "ADJ", "ADV", "OTHER"))


class TestBridgeWithoutModel(unittest.TestCase):
    def test_paths_missing_raises(self):
        from attacks.caa_cccc_bridge import ensure_caa_initialized, reset_caa_state
        from detectors.text.model_loader import build_text_detector

        reset_caa_state()
        os.environ["CCCC_CAA_SYNONYMS"] = "legacy"
        os.environ["CCCC_CAA_EMBED"] = str(ROOT / "nonexistent_embed.txt")
        os.environ["CCCC_CAA_COS"] = str(ROOT / "nonexistent_cos.pkl")
        det = build_text_detector()
        with self.assertRaises(FileNotFoundError):
            ensure_caa_initialized(det)
        del os.environ["CCCC_CAA_SYNONYMS"]
        del os.environ["CCCC_CAA_EMBED"]
        del os.environ["CCCC_CAA_COS"]
        reset_caa_state()


class TestFixedChainRequiresDetector(unittest.TestCase):
    """无 detector 时固定链应直接报错（不再 MLM 回退）。"""

    def test_without_detector_raises(self):
        from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext
        from attacks.fixed_chain import run_fixed_chain

        pipe = ChineseReplacePipeline()
        victim = ScoreContext(score_fn=lambda t: 0.6, threshold=0.5, positive_is_ai=True)
        with self.assertRaises(ValueError):
            run_fixed_chain("人工智能正在改变生活。", victim, pipe, detector=None)


class TestChineseCaaVocab(unittest.TestCase):
    def test_build_vocab_and_cos_rows(self):
        if os.environ.get("CCCC_SKIP_CAA_SLOW", "").lower() in ("1", "true", "yes"):
            self.skipTest("CCCC_SKIP_CAA_SLOW")

        from attacks.caa_chinese_synonyms import build_chinese_caa_vocab_for_text
        from attacks.chinese_replace import ChineseReplacePipeline

        text = "深度学习在自然语言处理领域应用广泛。"
        pipe = ChineseReplacePipeline()
        w2i, i2w, cos, path = build_chinese_caa_vocab_for_text(text, pipe)
        try:
            self.assertGreater(len(w2i), 2)
            row0 = cos[0]
            self.assertIsInstance(row0, list)
            self.assertGreaterEqual(len(row0), 40)
        finally:
            import os as _os

            if path and _os.path.isfile(path):
                _os.unlink(path)


class TestFixedChainCaaSmoke(unittest.TestCase):
    """Step2/3 走 HQA+TextHacker（中文近义词），允许未翻转，但须跑完三步且无未捕获异常。"""

    def test_chain_runs_three_steps(self):
        if os.environ.get("CCCC_SKIP_CAA_SLOW", "").lower() in ("1", "true", "yes"):
            self.skipTest("CCCC_SKIP_CAA_SLOW")

        from attacks.caa_cccc_bridge import reset_caa_state
        from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext
        from attacks.fixed_chain import run_fixed_chain
        from detectors.text.model_loader import build_text_detector

        reset_caa_state()
        os.environ.pop("CCCC_CAA_SYNONYMS", None)

        det = build_text_detector()

        def score_fn(t: str) -> float:
            return float(det.score([t])[0])

        victim = ScoreContext(score_fn=score_fn, threshold=0.5, positive_is_ai=True)
        pipe = ChineseReplacePipeline()
        text = "人工智能正在改变生活方式。"
        r = run_fixed_chain(text, victim, pipe, detector=det, max_queries_per_step=20)
        self.assertEqual(len(r.steps), 3)
        self.assertEqual([s.step_index for s in r.steps], [1, 2, 3])

        reset_caa_state()


class TestToolMinimalInject(unittest.TestCase):
    def test_initialize_minimal_cccc(self):
        import torch
        from third_party.caa.tool import get_global_manager, reset_global_manager

        reset_global_manager()
        mgr = get_global_manager()
        from attacks.caa_cccc_bridge import load_vocab_and_cos

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            emb, cos = _write_tiny_caa_data(tmp, ("测试文本",))
            w2i, i2w, cs = load_vocab_and_cos(str(emb), str(cos))

        def fake_pred(batch, batch_size=32):
            n = len(batch)
            p = torch.zeros(n, 2)
            p[:, 0] = 0.6
            p[:, 1] = 0.4
            return p

        from attacks.caa_cccc_bridge import DifflibWindowSim

        mgr.initialize_minimal_cccc(
            predictor=fake_pred,
            word2idx=w2i,
            idx2word=i2w,
            cos_sim=cs,
            embed_func=str(emb),
            sim_predictor=DifflibWindowSim(),
            stop_words_set=set(),
        )
        self.assertTrue(mgr.is_initialized)
        p = mgr.get_component("predictor")
        out = p([["a", "b"], ["c", "d"]], batch_size=2)
        self.assertEqual(out.shape[0], 2)
        reset_global_manager()


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if r.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
