"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  fetchSamplePaths,
  postVideoScoreByFile,
  postVideoScoreByPath,
} from "@/lib/api";
import type { SamplePathsResponse, VideoScoreResponse } from "@/lib/api";
import { apiBase } from "@/lib/utils";

function formatErr(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export default function VideoDetectPage() {
  const [samplePaths, setSamplePaths] = useState<SamplePathsResponse | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPathInput, setVideoPathInput] = useState("");
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [remotePreviewUrl, setRemotePreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<VideoScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const ensureSamplePaths = useCallback(async () => {
    if (samplePaths) return samplePaths;
    const r = await fetchSamplePaths();
    setSamplePaths(r);
    return r;
  }, [samplePaths]);

  useEffect(() => {
    if (!videoFile) {
      setLocalPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setLocalPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  async function runDetect() {
    setLoading(true);
    setErr(null);
    setResult(null);
    try {
      let r: VideoScoreResponse;
      if (videoFile) {
        r = await postVideoScoreByFile(videoFile);
      } else {
        const path = videoPathInput.trim();
        if (!path) throw new Error("请先上传视频或填写服务器视频路径");
        r = await postVideoScoreByPath(path);
      }
      setResult(r);
    } catch (e) {
      setErr(formatErr(e));
    } finally {
      setLoading(false);
    }
  }

  async function pickSample(kind: "human" | "ai") {
    setErr(null);
    setResult(null);
    setVideoFile(null);
    try {
      const paths = await ensureSamplePaths();
      const p = kind === "human" ? paths.human_video_path || paths.video_path : paths.ai_video_path || paths.video_path;
      setVideoPathInput(p || "");
      setRemotePreviewUrl(`${apiBase}/api/sample/video/${kind}?t=${Date.now()}`);
    } catch (e) {
      setErr(formatErr(e));
    }
  }

  function previewByPath() {
    const p = videoPathInput.trim();
    if (!p) return;
    setVideoFile(null);
    setRemotePreviewUrl(`${apiBase}/api/serve/file?path=${encodeURIComponent(p)}`);
  }

  const previewSrc = localPreviewUrl ?? remotePreviewUrl;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900">视频检测</h1>
        <p className="mt-1 text-sm text-zinc-600">
          支持本地上传视频检测，也支持服务器路径检测（可用示例按钮快速体验）。
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>输入与检测</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <input
            type="file"
            accept="video/*"
            onChange={(e) => {
              setVideoFile(e.target.files?.[0] ?? null);
              setErr(null);
            }}
            className="block w-full text-xs text-zinc-600 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-white"
          />
          <input
            type="text"
            className="w-full rounded-lg border border-zinc-200 p-2"
            placeholder="服务器本地视频路径"
            value={videoPathInput}
            onChange={(e) => {
              setVideoPathInput(e.target.value);
              setErr(null);
            }}
          />
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => void runDetect()} disabled={loading}>
              {loading ? "检测中…" : "检测视频"}
            </Button>
            <Button type="button" variant="outline" onClick={() => void pickSample("human")}>
              人类示例
            </Button>
            <Button type="button" variant="outline" onClick={() => void pickSample("ai")}>
              AI 示例
            </Button>
            <Button type="button" variant="secondary" onClick={previewByPath}>
              路径预览
            </Button>
          </div>
          {err && <p className="text-red-600">{err}</p>}
        </CardContent>
      </Card>

      {previewSrc && (
        <Card>
          <CardHeader>
            <CardTitle>预览</CardTitle>
          </CardHeader>
          <CardContent>
            <video src={previewSrc} controls preload="metadata" className="max-h-96 w-full rounded-lg border border-zinc-200 bg-zinc-50" />
          </CardContent>
        </Card>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>检测结果</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-zinc-700">
            <p>文件：{result.video_path}</p>
            <p>抽帧数：{result.num_frames}</p>
            <p>均值：{result.mean_score.toFixed(4)} · 最大值：{result.max_score.toFixed(4)}</p>
            <p>
              标签：<strong>{result.label_pred}</strong>
            </p>
            {result.frame_scores?.length > 0 && (
              <p className="break-all text-xs">各帧分数：{result.frame_scores.map((x) => x.toFixed(4)).join(", ")}</p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
