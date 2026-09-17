"use client";

import { Fragment, useMemo, useState } from "react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { apiBase, cn } from "@/lib/utils";
import { postBatchZipProcess } from "@/lib/api";
import type { BatchProcessResponse, BatchResultItem } from "@/lib/api";

type Modality = "text" | "image" | "video";
type FilterMode = "all" | "success" | "error" | "ai";

type TextSentenceDetail = {
  index: number;
  text: string;
  ai_score: number;
  label_pred: string;
  is_ai: boolean;
};

type TextFileDetail = {
  total_sentences?: number;
  ai_sentences?: number;
  ai_rate?: number;
  avg_ai_score?: number;
  max_ai_score?: number;
  threshold?: number;
  sentences?: TextSentenceDetail[];
  sentences_truncated?: boolean;
};

function formatErr(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function engineStyle(status: string): string {
  if (status === "ok") return "bg-emerald-100 text-emerald-800";
  if (status === "error") return "bg-red-100 text-red-700";
  if (status === "unavailable") return "bg-zinc-200 text-zinc-700";
  return "bg-amber-100 text-amber-800";
}

function asTextDetail(detail: Record<string, unknown> | null | undefined): TextFileDetail {
  if (!detail) return {};
  return {
    total_sentences: typeof detail.total_sentences === "number" ? detail.total_sentences : undefined,
    ai_sentences: typeof detail.ai_sentences === "number" ? detail.ai_sentences : undefined,
    ai_rate: typeof detail.ai_rate === "number" ? detail.ai_rate : undefined,
    avg_ai_score: typeof detail.avg_ai_score === "number" ? detail.avg_ai_score : undefined,
    max_ai_score: typeof detail.max_ai_score === "number" ? detail.max_ai_score : undefined,
    threshold: typeof detail.threshold === "number" ? detail.threshold : undefined,
    sentences: Array.isArray(detail.sentences) ? (detail.sentences as TextSentenceDetail[]) : undefined,
    sentences_truncated: Boolean(detail.sentences_truncated),
  };
}

export default function BatchPage() {
  const [modality, setModality] = useState<Modality>("text");
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [modelVariant, setModelVariant] = useState("zh_v3");
  const [minSentenceLen, setMinSentenceLen] = useState(4);
  const [enableSpark, setEnableSpark] = useState(true);
  const [enableHdfs, setEnableHdfs] = useState(true);
  const [enableHive, setEnableHive] = useState(true);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<BatchProcessResponse | null>(null);
  const [filter, setFilter] = useState<FilterMode>("all");
  const [expandedKeys, setExpandedKeys] = useState<Record<string, boolean>>({});

  async function runBatch() {
    if (!zipFile) {
      setErr("请先上传 zip 压缩包");
      return;
    }
    setLoading(true);
    setErr(null);
    setResult(null);
    setExpandedKeys({});
    try {
      const r = await postBatchZipProcess({
        zipFile,
        modality,
        modelVariant,
        minSentenceLen,
        enableSpark,
        enableHdfs,
        enableHive,
      });
      setResult(r);
    } catch (e) {
      setErr(formatErr(e));
    } finally {
      setLoading(false);
    }
  }

  function toggleExpand(key: string) {
    setExpandedKeys((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  const filteredResults = useMemo(() => {
    const items = result?.results || [];
    const isText = result?.modality === "text";
    if (filter === "all") return items;
    if (filter === "success") return items.filter((x) => x.status === "success");
    if (filter === "error") return items.filter((x) => x.status !== "success");
    if (isText) {
      return items.filter((x) => {
        if (x.status !== "success") return false;
        const detail = asTextDetail((x.detail || null) as Record<string, unknown> | null);
        return typeof detail.ai_rate === "number" && detail.ai_rate > 0;
      });
    }
    return items.filter((x) => x.status === "success" && (x.label_pred || "").toLowerCase() === "ai");
  }, [result, filter]);

  const jsonUrl = result?.artifacts?.download_json_url ? `${apiBase}${result.artifacts.download_json_url}` : "";
  const csvUrl = result?.artifacts?.download_csv_url ? `${apiBase}${result.artifacts.download_csv_url}` : "";
  const isTextResult = result?.modality === "text";

  return (
    <div className="space-y-6">
      <div className="relative overflow-hidden rounded-2xl border border-sky-200/70 bg-gradient-to-br from-sky-50 via-white to-zinc-50 px-6 py-8">
        <div className="relative z-10 max-w-3xl space-y-3">
          <p className="text-xs font-semibold uppercase tracking-widest text-sky-800/80">批量处理中心</p>
          <h1 className="text-3xl font-bold tracking-tight text-zinc-900">一键上传压缩包，完成文本/图片/视频批量检测</h1>
          <p className="text-sm leading-relaxed text-zinc-600">
            选择要处理的文件类型后，系统会从压缩包中自动筛选对应文件并输出逐文件结果；同时串联 Spark 后处理、HDFS 落盘和 Hive 批次摘要入库。
          </p>
        </div>
        <div className="pointer-events-none absolute -right-10 -top-10 h-44 w-44 rounded-full bg-sky-200/40 blur-2xl" />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>上传与配置</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div className="flex flex-wrap gap-2">
            {([
              ["text", "文本批处理"],
              ["image", "图片批处理"],
              ["video", "视频批处理"],
            ] as Array<[Modality, string]>).map(([k, label]) => (
              <button
                key={k}
                type="button"
                onClick={() => setModality(k)}
                className={cn(
                  "rounded-lg border px-3 py-1.5 text-sm transition",
                  modality === k ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-100"
                )}
              >
                {label}
              </button>
            ))}
          </div>

          <input
            type="file"
            accept=".zip,application/zip,application/x-zip-compressed"
            onChange={(e) => {
              setZipFile(e.target.files?.[0] ?? null);
              setErr(null);
            }}
            className="block w-full text-xs text-zinc-600 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-white"
          />

          {modality === "text" && (
            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-zinc-200 bg-zinc-50 p-3">
              <label className="flex items-center gap-2">
                <span className="text-zinc-500">模型</span>
                <select
                  className="rounded-md border border-zinc-200 bg-white px-2 py-1"
                  value={modelVariant}
                  onChange={(e) => setModelVariant(e.target.value)}
                >
                  <option value="zh_v3">Baseline</option>
                  <option value="robust">Robust</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                <span className="text-zinc-500">最短句长</span>
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={minSentenceLen}
                  onChange={(e) => setMinSentenceLen(Math.max(1, Math.min(64, Number(e.target.value) || 1)))}
                  className="w-20 rounded-md border border-zinc-200 bg-white px-2 py-1"
                />
              </label>
            </div>
          )}

          <div className="grid gap-2 sm:grid-cols-3">
            <label className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2">
              <input type="checkbox" checked={enableSpark} onChange={(e) => setEnableSpark(e.target.checked)} />
              <span>Spark 后处理</span>
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2">
              <input type="checkbox" checked={enableHdfs} onChange={(e) => setEnableHdfs(e.target.checked)} />
              <span>HDFS 同步</span>
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2">
              <input type="checkbox" checked={enableHive} onChange={(e) => setEnableHive(e.target.checked)} />
              <span>Hive 摘要入库</span>
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => void runBatch()} disabled={loading || !zipFile}>
              {loading ? "处理中…" : "开始批量处理"}
            </Button>
            {zipFile && <span className="text-xs text-zinc-500">当前文件：{zipFile.name}</span>}
          </div>
          {err && <p className="text-sm text-red-600">{err}</p>}
        </CardContent>
      </Card>

      {result && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-xl border border-zinc-200 bg-white p-4">
              <p className="text-xs text-zinc-500">处理文件总数</p>
              <p className="mt-1 text-2xl font-semibold">{result.total_files}</p>
            </div>
            <div className="rounded-xl border border-zinc-200 bg-white p-4">
              <p className="text-xs text-zinc-500">成功 / 失败</p>
              <p className="mt-1 text-2xl font-semibold">
                {result.success_files} / {result.failed_files}
              </p>
            </div>
            <div className="rounded-xl border border-zinc-200 bg-white p-4">
              <p className="text-xs text-zinc-500">AI / Human</p>
              <p className="mt-1 text-2xl font-semibold">
                {result.ai_files} / {result.human_files}
              </p>
            </div>
            <div className="rounded-xl border border-zinc-200 bg-white p-4">
              <p className="text-xs text-zinc-500">平均 AI 分数</p>
              <p className="mt-1 text-2xl font-semibold">{result.avg_ai_score.toFixed(4)}</p>
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>大数据组件运行状态</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {(["spark", "hadoop", "hive"] as const).map((k) => {
                const st = result.engines[k];
                return (
                  <div key={k} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-zinc-200 px-3 py-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium uppercase text-zinc-700">{k}</span>
                      <span className={cn("rounded px-2 py-0.5 text-xs", engineStyle(st.status))}>{st.status}</span>
                    </div>
                    <span className="text-xs text-zinc-500">{st.message}</span>
                  </div>
                );
              })}
              <div className="flex flex-wrap gap-2 pt-1">
                {jsonUrl && (
                  <a href={jsonUrl} className={cn(buttonVariants({ variant: "secondary", size: "sm" }))} target="_blank" rel="noreferrer">
                    下载 JSONL
                  </a>
                )}
                {csvUrl && (
                  <a href={csvUrl} className={cn(buttonVariants({ variant: "secondary", size: "sm" }))} target="_blank" rel="noreferrer">
                    下载 CSV
                  </a>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>逐文件结果</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-2 text-xs">
                {([
                  ["all", "全部"],
                  ["success", "成功"],
                  ["error", "失败"],
                  ["ai", "仅 AI"],
                ] as Array<[FilterMode, string]>).map(([k, label]) => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => setFilter(k)}
                    className={cn(
                      "rounded-md border px-2.5 py-1",
                      filter === k ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-200 bg-white text-zinc-600"
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="overflow-x-auto">
                {isTextResult ? (
                  <table className="w-full min-w-[920px] text-left text-sm">
                    <thead className="bg-zinc-50 text-xs text-zinc-600">
                      <tr>
                        <th className="p-2 font-medium">文件名</th>
                        <th className="p-2 font-medium">状态</th>
                        <th className="p-2 font-medium">AI率</th>
                        <th className="p-2 font-medium">AI句/总句</th>
                        <th className="p-2 font-medium">平均分</th>
                        <th className="p-2 font-medium">操作</th>
                        <th className="p-2 font-medium">信息</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredResults.map((row, idx) => {
                        const key = `${row.file_name}-${idx}`;
                        const detail = asTextDetail((row.detail || null) as Record<string, unknown> | null);
                        const aiRateText = typeof detail.ai_rate === "number" ? `${(detail.ai_rate * 100).toFixed(1)}%` : "-";
                        const ratioText =
                          typeof detail.ai_sentences === "number" && typeof detail.total_sentences === "number"
                            ? `${detail.ai_sentences}/${detail.total_sentences}`
                            : "-";
                        const avgText =
                          typeof detail.avg_ai_score === "number"
                            ? detail.avg_ai_score.toFixed(4)
                            : row.ai_score == null
                              ? "-"
                              : row.ai_score.toFixed(4);
                        const canExpand = row.status === "success" && Array.isArray(detail.sentences) && detail.sentences.length > 0;
                        const isExpanded = Boolean(expandedKeys[key]);

                        return (
                          <Fragment key={key}>
                            <tr key={key} className="border-t border-zinc-100">
                              <td className="p-2">{row.file_name}</td>
                              <td className="p-2">
                                <span
                                  className={cn(
                                    "rounded px-2 py-0.5 text-xs",
                                    row.status === "success" ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"
                                  )}
                                >
                                  {row.status}
                                </span>
                              </td>
                              <td className="p-2 tabular-nums">{aiRateText}</td>
                              <td className="p-2 tabular-nums">{ratioText}</td>
                              <td className="p-2 tabular-nums">{avgText}</td>
                              <td className="p-2">
                                {canExpand ? (
                                  <button
                                    type="button"
                                    onClick={() => toggleExpand(key)}
                                    className={cn(buttonVariants({ variant: "outline", size: "sm" }))}
                                  >
                                    {isExpanded ? "收起句子明细" : "查看句子明细"}
                                  </button>
                                ) : (
                                  <span className="text-xs text-zinc-400">-</span>
                                )}
                              </td>
                              <td className="p-2 text-xs text-zinc-500">{row.message || "-"}</td>
                            </tr>
                            {canExpand && isExpanded && (
                              <tr className="border-t border-zinc-100 bg-zinc-50/40">
                                <td colSpan={7} className="p-3">
                                  <div className="space-y-2">
                                    {detail.sentences?.map((s) => (
                                      <div
                                        key={`${key}-s-${s.index}`}
                                        className={cn(
                                          "rounded-lg border p-3",
                                          s.is_ai ? "border-red-300 bg-red-50" : "border-zinc-200 bg-white"
                                        )}
                                      >
                                        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                                          <span className="font-medium text-zinc-500">第 {s.index} 句</span>
                                          <span className={cn("rounded px-2 py-0.5", s.is_ai ? "bg-red-100 text-red-700" : "bg-zinc-100 text-zinc-600")}>
                                            {s.label_pred} · {Number(s.ai_score).toFixed(4)}
                                          </span>
                                        </div>
                                        <p className={cn("mt-2 leading-7", s.is_ai ? "text-red-800" : "text-zinc-800")}>{s.text}</p>
                                      </div>
                                    ))}
                                    {detail.sentences_truncated && (
                                      <p className="text-xs text-zinc-500">
                                        句子较多，仅展示前 {detail.sentences?.length ?? 0} 句。完整结果请下载 JSONL/CSV。
                                      </p>
                                    )}
                                  </div>
                                </td>
                              </tr>
                            )}
                          </Fragment>
                        );
                      })}
                    </tbody>
                  </table>
                ) : (
                  <table className="w-full min-w-[720px] text-left text-sm">
                    <thead className="bg-zinc-50 text-xs text-zinc-600">
                      <tr>
                        <th className="p-2 font-medium">文件名</th>
                        <th className="p-2 font-medium">状态</th>
                        <th className="p-2 font-medium">标签</th>
                        <th className="p-2 font-medium">AI 分数</th>
                        <th className="p-2 font-medium">信息</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredResults.map((row: BatchResultItem) => (
                        <tr key={`${row.file_name}-${row.status}-${row.ai_score ?? "na"}`} className="border-t border-zinc-100">
                          <td className="p-2">{row.file_name}</td>
                          <td className="p-2">
                            <span
                              className={cn(
                                "rounded px-2 py-0.5 text-xs",
                                row.status === "success" ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"
                              )}
                            >
                              {row.status}
                            </span>
                          </td>
                          <td className="p-2">{row.label_pred || "-"}</td>
                          <td className="p-2 tabular-nums">{row.ai_score == null ? "-" : row.ai_score.toFixed(4)}</td>
                          <td className="p-2 text-xs text-zinc-500">{row.message || "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}



