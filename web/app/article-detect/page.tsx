"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { postArticleParseByFile, postArticleScore } from "@/lib/api";
import type { ArticleScoreResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const SAMPLE_ARTICLE = `今早我在地铁上听到两位乘客讨论新发布的手机系统升级。整体体验有提升，但偶尔仍会出现应用切换卡顿。回到办公室后，我整理了团队周报，重点记录了项目进度、风险项和下周计划。综上所述，本文系统性地阐释了该技术方案在多场景下的落地路径，并从性能、稳定性与可扩展性三个维度给出了可复用的方法论框架。`;

function formatErr(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export default function ArticleDetectPage() {
  const [text, setText] = useState(SAMPLE_ARTICLE);
  const [modelVariant, setModelVariant] = useState("zh_v3");
  const [minSentenceLen, setMinSentenceLen] = useState(4);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadMeta, setUploadMeta] = useState<{ filename: string; chars: number } | null>(null);
  const [result, setResult] = useState<ArticleScoreResponse | null>(null);

  const aiLabelLower = useMemo(() => result?.positive_label?.toLowerCase() ?? "ai", [result]);

  async function runDetect() {
    const article = text.trim();
    if (!article) {
      setErr("请先输入文章内容");
      return;
    }
    setLoading(true);
    setErr(null);
    setResult(null);
    try {
      const r = await postArticleScore({
        text: article,
        modelVariant,
        minSentenceLen,
      });
      setResult(r);
    } catch (e) {
      setErr(formatErr(e));
    } finally {
      setLoading(false);
    }
  }

  async function parseUploadedFile() {
    if (!uploadFile) {
      setErr("请先选择文件");
      return;
    }
    setUploading(true);
    setErr(null);
    try {
      const parsed = await postArticleParseByFile(uploadFile);
      setText(parsed.text);
      setUploadMeta({ filename: parsed.filename, chars: parsed.chars });
      setResult(null);
    } catch (e) {
      setErr(formatErr(e));
    } finally {
      setUploading(false);
    }
  }

  const aiRatePct = result ? (result.ai_rate * 100).toFixed(1) : "0.0";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900">文章检测</h1>
        <p className="mt-1 text-sm text-zinc-600">
          支持上传 PDF / Word 文档自动解析正文，再进行逐句检测与高风险句高亮。
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>输入文章</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="rounded-lg border border-zinc-200 bg-zinc-50/50 p-3">
              <p className="mb-2 text-xs text-zinc-600">上传文档（支持 .pdf / .doc / .docx / .txt）</p>
              <div className="flex flex-wrap items-center gap-2">
                <input
                  type="file"
                  accept=".pdf,.doc,.docx,.txt,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
                  onChange={(e) => {
                    setUploadFile(e.target.files?.[0] ?? null);
                    setErr(null);
                  }}
                  className="block max-w-full text-xs text-zinc-600 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-white"
                />
                <Button type="button" variant="secondary" onClick={() => void parseUploadedFile()} disabled={uploading || !uploadFile}>
                  {uploading ? "解析中…" : "上传并解析"}
                </Button>
              </div>
              {uploadMeta && (
                <p className="mt-2 text-xs text-emerald-700">
                  已解析：{uploadMeta.filename}（{uploadMeta.chars} 字符）
                </p>
              )}
            </div>

            <textarea
              className="min-h-[260px] w-full rounded-lg border border-zinc-200 p-3 text-sm leading-7"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="粘贴、输入或上传文档后自动填入正文"
            />
            <div className="flex flex-wrap items-center gap-2 text-sm">
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
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => void runDetect()} disabled={loading}>
                {loading ? "检测中…" : "断句并检测"}
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={() => {
                  setText(SAMPLE_ARTICLE);
                  setUploadMeta(null);
                  setErr(null);
                }}
              >
                填入示例文章
              </Button>
            </div>
            {err && <p className="text-sm text-red-600">{err}</p>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>统计结果</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {result ? (
              <>
                <p>模型：{result.model_name}</p>
                <p>总句数：{result.total_sentences}</p>
                <p>AI 句数：{result.ai_sentences}</p>
                <p>
                  AI 率：<strong className="text-red-700">{aiRatePct}%</strong>
                </p>
                <p>平均分：{result.avg_ai_score.toFixed(4)}</p>
                <p>最高分：{result.max_ai_score.toFixed(4)}</p>
                <p>阈值：{result.threshold.toFixed(4)}</p>
                <div className="mt-2">
                  <div className="mb-1 flex items-center justify-between text-xs text-zinc-500">
                    <span>AI 率条</span>
                    <span>{aiRatePct}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-zinc-100">
                    <div
                      className="h-full bg-red-500 transition-all"
                      style={{ width: `${Math.max(0, Math.min(100, Number(aiRatePct)))}%` }}
                    />
                  </div>
                </div>
              </>
            ) : (
              <p className="text-zinc-500">提交文章后显示统计数据。</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>逐句结果</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {!result ? (
            <p className="text-sm text-zinc-500">检测后显示每句结果，判定为 AI 的句子会标红。</p>
          ) : result.sentences.length === 0 ? (
            <p className="text-sm text-zinc-500">未切分出可用句子。</p>
          ) : (
            result.sentences.map((s) => {
              const isAi = s.label_pred.toLowerCase() === aiLabelLower;
              return (
                <div
                  key={s.index}
                  className={cn(
                    "rounded-lg border p-3",
                    isAi ? "border-red-300 bg-red-50" : "border-zinc-200 bg-white"
                  )}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                    <span className="font-medium text-zinc-500">第 {s.index} 句</span>
                    <span className={cn("rounded px-2 py-0.5", isAi ? "bg-red-100 text-red-700" : "bg-zinc-100 text-zinc-600")}>
                      {s.label_pred} · {s.ai_score.toFixed(4)}
                    </span>
                  </div>
                  <p className={cn("mt-2 leading-7", isAi ? "text-red-800" : "text-zinc-800")}>
                    {s.text}
                  </p>
                </div>
              );
            })
          )}
        </CardContent>
      </Card>
    </div>
  );
}
