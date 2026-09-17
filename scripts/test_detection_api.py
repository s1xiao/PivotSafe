# -*- coding: utf-8 -*-
"""
实测三个检测 API：/api/text/score、/api/image/score、/api/video/score。
用真实请求体调用，确认返回 200 且结构正确。供 run_smoke_tests 或单独运行。
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def _make_test_image(path: Path) -> bool:
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(path)
        return path.exists()
    except Exception:
        return False


def _make_test_video(path: Path, num_frames: int = 5) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 5.0, (w, h))
    for i in range(num_frames):
        arr = np.full((h, w, 3), (i * 40) % 256, dtype=np.uint8)
        out.write(arr)
    out.release()
    return path.exists()


def main() -> None:
    import backend.attack_routes as attack_routes
    from fastapi.testclient import TestClient
    from backend.main import app

    attack_routes._PIPE = None
    client = TestClient(app)

    # 1. GET /health
    r = client.get("/health")
    assert r.status_code == 200, f"/health {r.status_code}"
    print("  GET /health OK")

    # 2. POST /api/text/score
    r = client.post("/api/text/score", json={"texts": ["这是一段测试文本。", "Hello world."]})
    assert r.status_code == 200, f"/api/text/score {r.status_code} {r.text}"
    data = r.json()
    assert "results" in data and len(data["results"]) == 2
    assert all("ai_score" in x and "label_pred" in x for x in data["results"])
    print("  POST /api/text/score OK")

    # 2b. POST /api/text/score robust（checkpoint 存在时）
    r = client.post(
        "/api/text/score",
        json={"texts": ["鲁棒检测探针。"], "model_variant": "robust"},
    )
    if r.status_code == 200:
        assert "results" in r.json()
        print("  POST /api/text/score (robust) OK")
    else:
        print(f"  POST /api/text/score (robust) SKIP ({r.status_code})")

    # 2c. 攻击与脆弱性 API
    r = client.post(
        "/api/attack/pivot-highlight",
        json={"text": "人工智能正在改变我们的生活。"},
    )
    assert r.status_code == 200, r.text
    assert "pivot_spans" in r.json()
    print("  POST /api/attack/pivot-highlight OK")

    r = client.post(
        "/api/attack/fixed-chain",
        json={"text": "人工智能正在改变我们的生活。", "max_queries_per_step": 24},
    )
    assert r.status_code == 200, r.text
    assert "result" in r.json() and "steps" in r.json()["result"]
    print("  POST /api/attack/fixed-chain OK")

    r = client.get("/api/vuln/summary")
    assert r.status_code == 200
    vs = r.json()
    assert "ready" in vs
    print("  GET /api/vuln/summary OK")

    # 3. POST /api/image/score（临时图片）
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = Path(f.name)
    try:
        if not _make_test_image(img_path):
            print("  POST /api/image/score SKIP (no PIL or image create failed)")
        else:
            r = client.post("/api/image/score", json={"image_paths": [str(img_path)]})
            assert r.status_code == 200, f"/api/image/score {r.status_code} {r.text}"
            data = r.json()
            assert "results" in data and len(data["results"]) == 1
            assert data["results"][0]["path"] == str(img_path) and "ai_score" in data["results"][0]
            print("  POST /api/image/score OK")
    finally:
        img_path.unlink(missing_ok=True)

    # 4. POST /api/video/score（临时视频）
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = Path(f.name)
    try:
        if not _make_test_video(video_path, 3):
            print("  POST /api/video/score SKIP (no opencv or video create failed)")
        else:
            r = client.post("/api/video/score", json={"video_path": str(video_path)})
            assert r.status_code == 200, f"/api/video/score {r.status_code} {r.text}"
            data = r.json()
            assert "mean_score" in data and "label_pred" in data and "num_frames" in data
            print("  POST /api/video/score OK")
    finally:
        video_path.unlink(missing_ok=True)

    print("  检测 API 实测通过。")


if __name__ == "__main__":
    main()
