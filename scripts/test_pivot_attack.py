# -*- coding: utf-8 -*-
"""
Pivot 攻击 / 高亮功能测试：遮挡回退路径 + Anchor 路径、边界与 API 形状。

在项目根执行:
  conda run -n cccc python -m scripts.test_pivot_attack
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestPivotEnv(unittest.TestCase):
    def test_pivot_use_anchor_env(self):
        from attacks import pivot_attack as pa

        try:
            os.environ["PIVOT_USE_ANCHOR"] = "0"
            self.assertFalse(pa._pivot_use_anchor())
            os.environ["PIVOT_USE_ANCHOR"] = "1"
            self.assertTrue(pa._pivot_use_anchor())
            os.environ["PIVOT_USE_ANCHOR"] = "false"
            self.assertFalse(pa._pivot_use_anchor())
        finally:
            os.environ.pop("PIVOT_USE_ANCHOR", None)


class TestComputePivotOcclusion(unittest.TestCase):
    """PIVOT_USE_ANCHOR=0：遮挡重要度 + 单条 pivot_spans。"""

    @classmethod
    def setUpClass(cls):
        os.environ["PIVOT_USE_ANCHOR"] = "0"

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("PIVOT_USE_ANCHOR", None)

    def setUp(self):
        from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext

        self.pipe = ChineseReplacePipeline()

        def score_fn(t: str) -> float:
            # 略依赖长度，使遮挡不同词时分数有差异
            return min(0.95, 0.35 + 0.01 * min(len(t), 80))

        self.victim = ScoreContext(score_fn=score_fn, threshold=0.5, positive_is_ai=True)

    def test_chinese_sentence_shape_and_single_highlight(self):
        from attacks.pivot_attack import compute_pivot_ranking

        text = "人工智能正在改变我们的生活方式。"
        ranked, hl = compute_pivot_ranking(text, self.pipe, self.victim, max_queries=48)
        self.assertEqual(hl.text, text)
        self.assertTrue(len(hl.tokens) >= 1)
        self.assertLessEqual(len(hl.pivot_spans), 1, "回退路径应至多 1 条高亮")
        if hl.pivot_spans:
            p = hl.pivot_spans[0]
            self.assertIn("word", p)
            self.assertIn("start", p)
            self.assertIn("end", p)
            self.assertEqual(text[p["start"] : p["end"]], p["word"])
        self.assertGreaterEqual(hl.query_count, 1)
        self.assertEqual(len(ranked), len([x for x in ranked]), "ranked 为列表")

    def test_short_text_empty_pivots(self):
        from attacks.pivot_attack import compute_pivot_ranking

        text = "好"
        ranked, hl = compute_pivot_ranking(text, self.pipe, self.victim, max_queries=20)
        self.assertEqual(hl.text, text)
        self.assertEqual(hl.pivot_spans, [])


class TestComputePivotAnchor(unittest.TestCase):
    """PIVOT_USE_ANCHOR=1：Anchor beam + 最终锚点若干词。"""

    @classmethod
    def setUpClass(cls):
        os.environ["PIVOT_USE_ANCHOR"] = "1"

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("PIVOT_USE_ANCHOR", None)

    def setUp(self):
        from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext

        self.pipe = ChineseReplacePipeline()

        def score_fn(t: str) -> float:
            return min(0.92, 0.4 + 0.008 * min(len(t), 100))

        self.victim = ScoreContext(score_fn=score_fn, threshold=0.5, positive_is_ai=True)

    def test_anchor_path_runs_and_spans_in_text(self):
        from attacks.pivot_attack import _compute_pivot_with_anchor, compute_pivot_ranking
        from attacks.chinese_replace import TokenSpan

        text = "深度学习在自然语言处理领域应用广泛。"
        spans_all = self.pipe.tokenize_with_pos(text)
        spans = self.pipe.filter_pos(spans_all)
        if not spans:
            self.skipTest("分词后无可用 span")

        out = _compute_pivot_with_anchor(
            text, self.pipe, self.victim, spans_all, spans, max_queries=40
        )
        if out is None:
            self.skipTest("Anchor 未就绪（缺 spacy/sklearn 等）")

        ranked, hl, _sc = out
        self.assertIsInstance(ranked, list)
        self.assertEqual(hl.text, text)
        for p in hl.pivot_spans:
            w = p["word"]
            s, e = p["start"], p["end"]
            self.assertEqual(text[s:e], w)
            self.assertLessEqual(e, len(text))
            self.assertGreaterEqual(s, 0)

        ranked2, hl2 = compute_pivot_ranking(text, self.pipe, self.victim, max_queries=40)
        self.assertEqual(hl2.text, text)
        self.assertIsInstance(hl2.pivot_spans, list)
        self.assertGreater(hl2.query_count, 0, "Anchor 路径应产生若干次 victim 查询")
        # 最终锚点可为空（预算/阈值），若非空则与 ranked 中词一致
        if hl2.pivot_spans:
            words_in_spans = {sp.word for sp in spans}
            for p in hl2.pivot_spans:
                self.assertIn(p["word"], words_in_spans)


class TestVictimProbaMatrix(unittest.TestCase):
    def test_shape_and_argmax_consistency(self):
        from attacks.chinese_replace import ScoreContext
        from attacks.pivot_attack import _victim_proba_matrix

        v = ScoreContext(score_fn=lambda t: 0.7, threshold=0.5, positive_is_ai=True)
        m = _victim_proba_matrix(v, ["a b", "c d"])
        self.assertEqual(m.shape, (2, 2))
        self.assertAlmostEqual(m[0, 1], 0.7)
        self.assertAlmostEqual(m[0, 0], 0.3)


class TestRunPivotAttackSmoke(unittest.TestCase):
    def test_minimal_queries_returns_result(self):
        os.environ["PIVOT_USE_ANCHOR"] = "0"
        try:
            from attacks.chinese_replace import ChineseReplacePipeline, ScoreContext
            from attacks.pivot_attack import run_pivot_attack

            pipe = ChineseReplacePipeline()
            v = ScoreContext(score_fn=lambda t: 0.6, threshold=0.5, positive_is_ai=True)
            r = run_pivot_attack(
                "机器翻译质量不断提升。",
                pipe,
                v,
                max_queries=8,
                max_replace=1,
            )
            self.assertIsInstance(r.adversarial_text, str)
            self.assertEqual(r.adversarial_text, "机器翻译质量不断提升。")
            self.assertFalse(r.success)
            self.assertIn(r.orig_label, ("ai", "human"))
            self.assertGreaterEqual(r.query_count, 1)
            self.assertIsInstance(r.pivot_highlights, list)
        finally:
            os.environ.pop("PIVOT_USE_ANCHOR", None)


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if r.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
