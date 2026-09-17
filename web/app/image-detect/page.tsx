"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  fetchSamplePaths,
  postImageScoreByFiles,
  postImageScoreByPaths,
} from "@/lib/api";
import type { ImageScoreResponse, SamplePathsResponse } from "@/lib/api";
import { apiBase } from "@/lib/utils";

function formatErr(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function parseImagePaths(input: string): string[] {
  return input
    .split(/[\s,]+/)
    .map((x) => x.trim())
    .filter(Boolean);
}

export default function ImageDetectPage() {
  const [samplePaths, setSamplePaths] = useState<SamplePathsResponse | null>(null);
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imagePathInput, setImagePathInput] = useState("");
  const [localPreviewUrls, setLocalPreviewUrls] = useState<string[]>([]);
  const [remotePreviewUrl, setRemotePreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ImageScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const ensureSamplePaths = useCallback(async () => {
    if (samplePaths) return samplePaths;
    const r = await fetchSamplePaths();
    setSamplePaths(r);
    return r;
  }, [samplePaths]);

  useEffect(() => {
    if (imageFiles.length === 0) {
      setLocalPreviewUrls([]);
      return;
    }
    const urls = imageFiles.map((f) => URL.createObjectURL(f));
    setLocalPreviewUrls(urls);
    return () => {
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [imageFiles]);

  async function runDetect() {
    setLoading(true);
    setErr(null);
    setResult(null);
    try {
      let r: ImageScoreResponse;
      if (imageFiles.length > 0) {
        r = await postImageScoreByFiles(imageFiles);
      } else {
        const paths = parseImagePaths(imagePathInput);
        if (!paths.length) throw new Error("请先上传图片或填写服务器图片路径");
        r = await postImageScoreByPaths(paths);
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
    setImageFiles([]);
    try {
      const paths = await ensureSamplePaths();
      const p = kind === "human" ? paths.human_image_path || paths.image_path : paths.ai_image_path || paths.image_path;
      setImagePathInput(p || "");
      setRemotePreviewUrl(`${apiBase}/api/sample/image/${kind}?t=${Date.now()}`);
    } catch (e) {
      setErr(formatErr(e));
    }
  }

  function previewByPath() {
    const p = imagePathInput.trim();
    if (!p) return;
    setImageFiles([]);
    setRemotePreviewUrl(`${apiBase}/api/serve/file?path=${encodeURIComponent(p)}`);
  }

  const summary = useMemo(() => {
    if (!result?.results?.length) return null;
    const avg = result.results.reduce((a, b) => a + b.ai_score, 0) / result.results.length;
    const aiCount = result.results.filter((x) => x.label_pred === "ai").length;
    return { count: result.results.length, avg, aiCount };
  }, [result]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900">图片检测</h1>
        <p className="mt-1 text-sm text-zinc-600">
          支持本地上传图片检测，也支持服务器路径检测（可用示例按钮快速体验）。
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>输入与检测</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(e) => {
              setImageFiles(Array.from(e.target.files || []));
              setErr(null);
            }}
            className="block w-full text-xs text-zinc-600 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-white"
          />
          <input
            type="text"
            className="w-full rounded-lg border border-zinc-200 p-2"
            placeholder="服务器本地图片路径（多张可用逗号/空格分隔）"
            value={imagePathInput}
            onChange={(e) => {
              setImagePathInput(e.target.value);
              setErr(null);
            }}
          />
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => void runDetect()} disabled={loading}>
              {loading ? "检测中…" : "检测图片"}
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

      {localPreviewUrls.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>本地预览</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              {localPreviewUrls.map((src) => (
                <img key={src} src={src} alt="local preview" className="h-40 w-full rounded-lg border border-zinc-200 object-contain bg-zinc-50" />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {remotePreviewUrl && (
        <Card>
          <CardHeader>
            <CardTitle>路径/示例预览</CardTitle>
          </CardHeader>
          <CardContent>
            <img src={remotePreviewUrl} alt="remote preview" className="max-h-80 rounded-lg border border-zinc-200 object-contain bg-zinc-50" />
          </CardContent>
        </Card>
      )}

      {summary && (
        <Card>
          <CardContent className="pt-4 text-sm text-zinc-700">
            共 {summary.count} 张 · 平均 AI 分数 {summary.avg.toFixed(4)} · 判定 AI {summary.aiCount} 张
          </CardContent>
        </Card>
      )}

      {result?.results?.length ? (
        <Card>
          <CardHeader>
            <CardTitle>检测结果</CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full min-w-[420px] text-left text-sm">
              <thead className="bg-zinc-50 text-xs text-zinc-600">
                <tr>
                  <th className="p-2 font-medium">文件</th>
                  <th className="p-2 font-medium">AI 分数</th>
                  <th className="p-2 font-medium">标签</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((item) => (
                  <tr key={`${item.path}-${item.ai_score}`} className="border-t border-zinc-100">
                    <td className="p-2">{item.path}</td>
                    <td className="p-2 tabular-nums">{item.ai_score.toFixed(4)}</td>
                    <td className="p-2">{item.label_pred}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
