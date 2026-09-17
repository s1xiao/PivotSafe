"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchDatasetSamples, postTextCompare } from "@/lib/api";
import type { DatasetSample, TextCompareResponse } from "@/lib/api";
import { apiBase, cn } from "@/lib/utils";

export default function DetectPage() {
  const [text, setText] = useState("人工智能正在改变我们的生活与工作方式。");
  const [pool, setPool] = useState<"clean" | "attackable">("clean");
  const [page, setPage] = useState(1);
  const [rows, setRows] = useState<DatasetSample[]>([]);
  const [total, setTotal] = useState(0);
  const [sel, setSel] = useState<DatasetSample | null>(null);
  const [cmp, setCmp] = useState<TextCompareResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [tableLoading, setTableLoading] = useState(false);

  const perPage = 8;

  const loadTable = useCallback(async () => {
    setTableLoading(true);
    setErr(null);
    try {
      const r = await fetchDatasetSamples({ pool, page, per_page: perPage });
      setRows(r.items);
      setTotal(r.total);
    } catch (e) {
      setErr(String(e));
      setRows([]);
    } finally {
      setTableLoading(false);
    }
  }, [pool, page]);

  useEffect(() => {
    void loadTable();
  }, [loadTable]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const q = new URLSearchParams(window.location.search);
    const t = q.get("text");
    if (t) setText(t);
  }, []);

  async function runCompare() {
    setLoading(true);
    setErr(null);
    try {
      const gold = sel?.label || undefined;
      const j = await postTextCompare(text, gold);
      setCmp(j);
    } catch (e) {
      setErr(String(e));
      setCmp(null);
    } finally {
      setLoading(false);
    }
  }

  async function randomSample() {
    setErr(null);
    try {
      const r = await fetchDatasetSamples({ pool, random_one: true });
      const it = r.items[0];
      if (it) {
        setSel(it);
        setText(it.text);
        const j = await postTextCompare(it.text, it.label);
        setCmp(j);
      }
    } catch (e) {
      setErr(String(e));
    }
  }

  const redteamHref = `/redteam?text=${encodeURIComponent(text)}`;
  const reviewHref = sel?.id ? `/vuln?case=${encodeURIComponent(sel.id)}` : "/vuln";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900">文本检测</h1>
        <p className="mt-1 text-sm text-zinc-600">
          手动输入、样本池点选或随机抽样；单次请求对比 Baseline / Robust（<code className="text-xs">POST /api/text/compare</code>）。
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>输入与操作</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <textarea
              className="w-full min-h-[120px] rounded-lg border border-zinc-200 p-3 text-sm leading-relaxed"
              value={text}
              maxLength={200}
              onChange={(e) => setText(e.target.value)}
            />
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => void runCompare()} disabled={loading}>
                {loading ? "检测中…" : "检测（对比）"}
              </Button>
              <Button type="button" variant="secondary" onClick={() => void randomSample()}>
                随机换一条
              </Button>
              <Link href={redteamHref} className={cn(buttonVariants({ variant: "outline" }))}>
                固定链完整复盘
              </Link>
              <Link href={reviewHref} className={cn(buttonVariants({ variant: "outline" }))}>
                脆弱性案例表
              </Link>
            </div>
            {sel && (
              <p className="text-xs text-zinc-500">
                已选样本 id：<span className="font-mono">{sel.id}</span> · 金标{" "}
                <strong>{sel.label}</strong>
                {sel.split && <> · split {sel.split}</>}
              </p>
            )}
            {err && <p className="text-sm text-red-600">{err}</p>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">当前对比摘要</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {cmp ? (
              <>
                <p>
                  双模型一致：<strong>{cmp.models_agree ? "是" : "否"}</strong>
                </p>
                {cmp.gold_label != null && (
                  <p className="text-zinc-600">
                    Baseline 与金标一致：
                    {cmp.baseline_agrees_with_gold == null ? "—" : cmp.baseline_agrees_with_gold ? "是" : "否"}
                    <br />
                    Robust 与金标一致：
                    {cmp.robust_agrees_with_gold == null ? "—" : cmp.robust_agrees_with_gold ? "是" : "否"}
                  </p>
                )}
              </>
            ) : (
              <p className="text-zinc-500">点击检测后显示</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Baseline</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {cmp ? (
              <div className="space-y-1">
                <p>
                  标签：<strong>{cmp.baseline.label}</strong>
                </p>
                <p className="tabular-nums">AI 分数：{cmp.baseline.ai_score.toFixed(4)}</p>
              </div>
            ) : (
              <p className="text-zinc-500">暂无</p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Robust</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {cmp?.robust ? (
              <div className="space-y-1">
                <p>
                  标签：<strong>{cmp.robust.label}</strong>
                </p>
                <p className="tabular-nums">AI 分数：{cmp.robust.ai_score.toFixed(4)}</p>
              </div>
            ) : (
              <p className="text-zinc-500">未就绪或未训练 checkpoint</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle>样本池</CardTitle>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-zinc-500">池子</span>
            <select
              className="rounded-md border border-zinc-200 bg-white px-2 py-1"
              value={pool}
              onChange={(e) => {
                setPage(1);
                setPool(e.target.value as "clean" | "attackable");
              }}
            >
              <option value="clean">clean（全标签）</option>
              <option value="attackable">attackable（AI，可攻）</option>
            </select>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-zinc-500">
            共 {total} 条 · 第 {page} 页 · API {apiBase}/api/dataset/samples
          </p>
          <div className="overflow-x-auto rounded-lg border border-zinc-200">
            <table className="w-full min-w-[520px] text-left text-sm">
              <thead className="bg-zinc-50 text-xs text-zinc-600">
                <tr>
                  <th className="p-2 font-medium">id</th>
                  <th className="p-2 font-medium">label</th>
                  <th className="p-2 font-medium">text</th>
                  <th className="p-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                {tableLoading ? (
                  <tr>
                    <td colSpan={4} className="p-4 text-center text-zinc-500">
                      加载中…
                    </td>
                  </tr>
                ) : rows.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="p-4 text-center text-zinc-500">
                      无数据。请运行 scripts/build_sample_pools.py
                    </td>
                  </tr>
                ) : (
                  rows.map((r) => (
                    <tr key={r.id} className="border-t border-zinc-100 hover:bg-amber-50/40">
                      <td className="p-2 font-mono text-xs align-top">{r.id}</td>
                      <td className="p-2 align-top">{r.label}</td>
                      <td className="p-2 align-top text-zinc-700 line-clamp-2 max-w-md">{r.text}</td>
                      <td className="p-2 align-top">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-8 text-xs"
                          onClick={() => {
                            setSel(r);
                            setText(r.text);
                            void postTextCompare(r.text, r.label).then(setCmp).catch((e) => setErr(String(e)));
                          }}
                        >
                          选用
                        </Button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              上一页
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={page * perPage >= total}
              onClick={() => setPage((p) => p + 1)}
            >
              下一页
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
