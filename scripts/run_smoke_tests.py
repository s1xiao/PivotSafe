# -*- coding: utf-8 -*-
"""
一键运行各模块自测及后端接口校验，用于快速验证开发环境与主链路。

在项目根目录、cccc 环境中执行：
  python -m scripts.run_smoke_tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run(name: str, fn) -> bool:
    try:
        fn()
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ok = 0
    total = 0

    # 1. crawler unified_io 自测
    total += 1
    def _unified_io_test():
        import importlib
        mod = importlib.import_module("crawler.smoke_test_unified_io")
        mod.main()
    if run("crawler.unified_io", _unified_io_test):
        ok += 1

    # 2. analysis.basic_stats 自测
    total += 1
    if run("analysis.basic_stats", lambda: __import__("analysis.smoke_test_basic_stats").smoke_test_basic_stats.main()):
        ok += 1

    # 3. 报告生成（使用临时 basic_stats）
    total += 1
    def _report():
        import tempfile
        from report.generate_report import generate_html
        stats = {"total": 1, "by_platform": {"xhs": 1}, "text_score_mean": 0.5, "text_score_count": 1, "image_score_mean": None, "image_score_count": 0, "text_ai_ratio": 0.0, "image_ai_ratio": None}
        html = generate_html(stats)
        assert "总样本数" in html and "1" in html
    if run("report.generate_report", _report):
        ok += 1

    # 4. 后端 /health、/api/summary、根路径（不再挂载旧 /dashboard）
    total += 1
    def _backend():
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        r2 = client.get("/api/summary")
        assert r2.status_code == 200 and r2.json().get("backend") == "ok"
        r3 = client.get("/")
        assert r3.status_code == 200 and r3.json().get("docs") == "/docs"
    if run("backend /health, /api/summary, /", _backend):
        ok += 1

    # 5. 视频轻量检测（抽帧 + 图像检测聚合）
    total += 1
    def _video():
        import importlib
        mod = importlib.import_module("detectors.video.smoke_test_video")
        mod.main()
    if run("detectors.video", _video):
        ok += 1

    # 6. 检测 API 实测（/api/text/score、/api/image/score、/api/video/score）
    total += 1
    def _api_detection():
        import importlib
        mod = importlib.import_module("scripts.test_detection_api")
        mod.main()
    if run("检测 API 实测", _api_detection):
        ok += 1

    # 7. 数据集与对比 API（不依赖池文件是否存在，summary 应 200）
    total += 1
    def _dataset_api():
        from fastapi.testclient import TestClient
        from backend.main import app
        c = TestClient(app)
        s = c.get("/api/dataset/summary")
        assert s.status_code == 200
        body = s.json()
        assert "pools_paths_exist" in body and "eval_paths_exist" in body
        sp = c.get("/api/dataset/samples", params={"pool": "clean", "page": 1, "per_page": 2})
        assert sp.status_code == 200
        assert "items" in sp.json()
        cmp = c.post(
            "/api/text/compare",
            json={"text": "这是一条测试句子。"},
        )
        assert cmp.status_code == 200
        assert "baseline" in cmp.json()
        cl = c.get("/api/cases/list", params={"page": 1, "per_page": 5})
        assert cl.status_code == 200
        assert "items" in cl.json()
    if run("dataset / cases / text.compare API", _dataset_api):
        ok += 1

    print()
    if ok == total:
        print(f"全部通过 ({ok}/{total})。")
        return 0
    print(f"通过 {ok}/{total}，有 {total - ok} 项失败。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
