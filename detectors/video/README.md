# 轻量视频检测（detectors/video）

对视频不做逐帧重型建模，采用 **抽帧 + 图像检测聚合** 的轻量方案。

## 思路

- **抽帧**：使用 OpenCV `VideoCapture`，取首帧与时间轴上均匀分布的若干帧（可配置 `num_frames`）。抽帧逻辑参考开源实现 [asmit404/Frame_Extraction](https://github.com/asmit404/Frame_Extraction)（本项目为均匀采样、固定帧数）。
- **检测**：对每帧调用现有图像 AI 检测器（SSP），得到每帧的 `ai_score`。
- **聚合**：对帧分数做均值或最大值聚合，再按阈值得到整段视频的 `label_pred`（ai/human）。

## 模块说明

- `frames.py`：`extract_frames(video_path, num_frames=5, output_dir=None)`，返回帧图片路径列表；`output_dir` 为 None 时使用临时目录。
- `detector.py`：`VideoLightweightDetector`，可注入 `ImageAIDetector` 或从 `configs/image_detector.json` 延迟加载；`predict(video_path)` 返回 `VideoDetectionResult`（含 `frame_scores`、`mean_score`、`max_score`、`label_pred`）。

## 后端接口

- `POST /api/video/score`：请求体 `{"video_path": "/path/to/video.mp4"}`，返回 `frame_scores`、`mean_score`、`max_score`、`label_pred`、`num_frames`。

## 自测

```bash
conda activate cccc
python -m detectors.video.smoke_test_video
```

自测会生成合成短视频并跑通抽帧 + 整链路检测（会加载图像模型，耗时数秒）。一键全量自测中包含本项：`python -m scripts.run_smoke_tests`（5/5）。
