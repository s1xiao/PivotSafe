from __future__ import annotations

from .model_loader import build_text_detector


SAMPLE_TEXTS = [
    "这是一个明显 AI 风格的测试文本，用来检查检测模型是否能给出较高的 AI 可疑度。",
    "今天下课之后和同学一起去吃了麻辣烫，顺便讨论了一下计设项目的进度。",
    "Large language models can generate fluent paragraphs, but we still want to detect whether a given text is machine-written.",
    "I woke up late, missed the first class, and had to run across campus in the rain.",
]


def main() -> None:
    detector = build_text_detector()
    print(f"Loaded text detector model: {detector.model_name}")
    print(f"Device: {detector.device}, threshold: {detector.threshold}")
    print(f"Positive label (AI class): {detector.positive_label}")
    print("-" * 80)

    results = detector.predict(SAMPLE_TEXTS)
    for i, r in enumerate(results, start=1):
        print(f"[{i}] ai_score={r.ai_score:.4f} label={r.label_pred}")
        print(f"    text: {r.text}")


if __name__ == "__main__":
    main()

