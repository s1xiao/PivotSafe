"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchDatasetSamples } from "@/lib/api";
import type { DatasetSample } from "@/lib/api";
import { apiBase, cn } from "@/lib/utils";

type StepRow = {
  step_index: number;
  name: string;
  executed?: boolean;
  baseline_prediction_flipped?: boolean;
  success: boolean;
  orig_score?: number;
  final_score?: number;
  perturbation_rate?: number;
  adversarial_text?: string;
  query_count?: number;
  elapsed_sec?: number;
  replaced_detail?: { from?: string; to?: string; pos?: string; sim?: number }[];
};

export default function RedteamPage() {
  const [text, setText] = useState("人工智能正在改变我们的生活与工作方式。");
  const [trueLabel, setTrueLabel] = useState<string | null>(null);
  const [poolRows, setPoolRows] = useState<DatasetSample[]>([]);
  const [hl, setHl] = useState<{
    tokens: { word: string; start: number; end: number; pos: string }[];
    pivot_spans: { word: string; start: number; end: number; importance: number; pos?: string }[];
  } | null>(null);
  const [chain, setChain] = useState<Record<string, unknown> | null>(null);
  const [bOrig, setBOrig] = useState<{ label: string; score: number } | null>(null);
  const [bAdv, setBAdv] = useState<{ label: string; score: number } | null>(null);
  const [rAdv, setRAdv] = useState<{ label: string; score: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadAttackable = useCallback(async () => {
    try {
      const r = await fetchDatasetSamples({ pool: "attackable", page: 1, per_page: 30 });
      setPoolRows(r.items);
    } catch {
      setPoolRows([]);
    }
  }, []);

  useEffect(() => {
    void loadAttackable();
  }, [loadAttackable]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const q = new URLSearchParams(window.location.search);
    const t = q.get("text");
    if (t) setText(t);
  }, []);

  const rendered = useMemo(() => {
    if (!hl?.tokens?.length) return null;
    const pivots = new Set(hl.pivot_spans.map((p) => `${p.start}-${p.end}`));
    const parts: ReactNode[] = [];
    let i = 0;
    const s = text;
    for (const tk of hl.tokens) {
      if (tk.start > i) {
        parts.push(<span key={`g-${i}`}>{s.slice(i, tk.start)}</span>);
      }
      const key = `${tk.start}-${tk.end}`;
      const isPv = pivots.has(key);
      parts.push(
        <span
          key={key}
          className={isPv ? "rounded bg-amber-200/90 px-0.5 text-zinc-900" : "text-zinc-800"}
        >
          {tk.word}
        </span>
      );
      i = tk.end;
    }
    if (i < s.length) parts.push(<span key={`t-${i}`}>{s.slice(i)}</span>);
    return parts;
  }, [hl, text]);

  async function runAll() {
    setLoading(true);
    setErr(null);
    try {
      const bo = await fetch(`${apiBase}/api/text/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: [text], model_variant: "zh_v3" }),
      });
      if (bo.ok) {
        const j = await bo.json();
        setBOrig({
          label: j.results[0].label_pred,
          score: j.results[0].ai_score,
        });
      } else setBOrig(null);

      const ph = await fetch(`${apiBase}/api/attack/pivot-highlight`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!ph.ok) throw new Error(await ph.text());
      setHl(await ph.json());

      const ch = await fetch(`${apiBase}/api/attack/fixed-chain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, max_queries_per_step: 1000 }),
      });
      if (!ch.ok) throw new Error(await ch.text());
      const chj = await ch.json();
      const res = chj.result as Record<string, unknown>;
      setChain(res);

      const adv = (res.final_adversarial_text as string) || (res.original_text as string);
      const [rb, rr] = await Promise.all([
        fetch(`${apiBase}/api/text/score`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: [adv], model_variant: "zh_v3" }),
        }),
        fetch(`${apiBase}/api/text/score`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: [adv], model_variant: "robust" }),
        }),
      ]);
      if (rb.ok) {
        const j = await rb.json();
        setBAdv({ label: j.results[0].label_pred, score: j.results[0].ai_score });
      } else setBAdv(null);
      if (rr.ok) {
        const j = await rr.json();
        setRAdv({ label: j.results[0].label_pred, score: j.results[0].ai_score });
      } else setRAdv(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function pickRandomAttackable() {
    setErr(null);
    try {
      const r = await fetchDatasetSamples({ pool: "attackable", random_one: true });
      const it = r.items[0];
      if (it) {
        setText(it.text);
        setTrueLabel(it.label);
      }
    } catch (e) {
      setErr(String(e));
    }
  }

  const steps = (chain?.steps as StepRow[]) || [];
  const bypassed = Boolean(chain?.baseline_bypassed ?? chain?.overall_success);
  const goldAi = trueLabel?.toLowerCase() === "ai";
  const baselineEvaded = bypassed && goldAi && bAdv?.label?.toLowerCase() !== "ai";
  const robustHolds =
    goldAi && rAdv != null ? rAdv.label?.toLowerCase() === "ai" : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900">红队攻击</h1>
        <p className="mt-1 text-sm text-zinc-600">
          从可攻击池选样或手输；Pivot 高亮 →{" "}
          <strong>顺序链</strong>：仅当前一步未翻转 Baseline 时才执行下一步（后续步标记「未执行」）→
          对抗文本上 Baseline / Robust 复测。
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>样本来源</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Button type="button" variant="secondary" onClick={() => void pickRandomAttackable()}>
            随机可攻击样本
          </Button>
          <select
            className="max-w-full flex-1 min-w-[200px] rounded-md border border-zinc-200 bg-white px-2 py-2 text-sm"
            onChange={(e) => {
              const id = e.target.value;
              const it = poolRows.find((x) => x.id === id);
              if (it) {
                setText(it.text);
                setTrueLabel(it.label);
              }
            }}
            defaultValue=""
          >
            <option value="" disabled>
              从 attackable 池选择…
            </option>
            {poolRows.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id} · {r.text.slice(0, 24)}…
              </option>
            ))}
          </select>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>原始文本</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <textarea
            className="w-full min-h-[88px] rounded-lg border border-zinc-200 p-3 text-sm"
            value={text}
            maxLength={200}
            onChange={(e) => {
              setText(e.target.value);
              setTrueLabel(null);
            }}
          />
          {trueLabel && (
            <p className="text-xs text-zinc-500">
              数据集金标：<strong>{trueLabel}</strong>
            </p>
          )}
          <Button onClick={() => void runAll()} disabled={loading}>
            {loading ? "运行中…" : "高亮 + 攻击链 + 复测"}
          </Button>
          {err && <p className="text-sm text-red-600">{err}</p>}
        </CardContent>
      </Card>

      {bOrig && (
        <Card className="border-zinc-300">
          <CardHeader>
            <CardTitle className="text-base">Baseline · 原文判断</CardTitle>
          </CardHeader>
          <CardContent className="text-sm tabular-nums">
            {bOrig.label} / {bOrig.score.toFixed(4)}
          </CardContent>
        </Card>
      )}

      {hl && (
        <Card>
          <CardHeader>
            <CardTitle>Pivot 高亮（最终锚点 / span）</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm leading-relaxed">
            <div>{rendered}</div>
            {hl.pivot_spans.length > 0 && (
              <div className="rounded-md border border-amber-100 bg-amber-50/50 p-2 text-xs text-zinc-700">
                <p className="font-medium text-amber-900/90">锚点列表（当前为最终锚点，非 top-k）</p>
                <ul className="mt-1 list-inside list-disc">
                  {hl.pivot_spans.map((p, i) => (
                    <li key={i}>
                      「{p.word}」({p.start}–{p.end}) · pos {p.pos ?? "—"}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {chain && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>固定攻击链 · 逐步结果</CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-left text-sm">
                <thead className="border-b border-zinc-200 text-xs text-zinc-600">
                  <tr>
                    <th className="py-2 pr-2">步</th>
                    <th className="py-2 pr-2">攻击器</th>
                    <th className="py-2 pr-2">是否执行</th>
                    <th className="py-2 pr-2">翻转 Baseline</th>
                    <th className="py-2 pr-2">分数变化</th>
                    <th className="py-2 pr-2">扰动率</th>
                    <th className="py-2 pr-2">查询</th>
                    <th className="py-2">候选文本</th>
                  </tr>
                </thead>
                <tbody>
                  {steps.map((st) => {
                    const ex = st.executed !== false;
                    const flip = Boolean(st.baseline_prediction_flipped ?? (ex && st.success));
                    return (
                      <tr key={st.step_index} className="border-b border-zinc-100 align-top">
                        <td className="py-2 pr-2">{st.step_index}</td>
                        <td className="py-2 pr-2">{st.name}</td>
                        <td className="py-2 pr-2">{ex ? "已执行" : <span className="text-zinc-400">未执行</span>}</td>
                        <td className="py-2 pr-2">{ex ? (flip ? "是" : "否") : "—"}</td>
                        <td className="py-2 pr-2 tabular-nums text-xs">
                          {st.orig_score != null && st.final_score != null
                            ? `${Number(st.orig_score).toFixed(3)} → ${Number(st.final_score).toFixed(3)}`
                            : "—"}
                        </td>
                        <td className="py-2 pr-2 tabular-nums">
                          {st.perturbation_rate != null ? st.perturbation_rate.toFixed(3) : "—"}
                        </td>
                        <td className="py-2 pr-2">{st.query_count ?? "—"}</td>
                        <td className="py-2 max-w-xs break-words text-xs text-zinc-700">
                          {(st.adversarial_text || "").slice(0, 120)}
                          {(st.adversarial_text || "").length > 120 ? "…" : ""}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>原文 vs 对抗文</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-zinc-800">
              <div>
                <p className="text-xs font-medium text-zinc-500">原文</p>
                <p className="mt-1 rounded-md bg-zinc-50 p-2 leading-relaxed">{String(chain.original_text || text)}</p>
              </div>
              <div>
                <p className="text-xs font-medium text-zinc-500">对抗文（链输出）</p>
                <p className="mt-1 rounded-md border border-amber-200/80 bg-amber-50/40 p-2 leading-relaxed">
                  {String(chain.final_adversarial_text || chain.original_text || text)}
                </p>
              </div>
            </CardContent>
          </Card>

          {(() => {
            const ps = (chain.perturbation_summary as Record<string, unknown>) || {};
            const pairs = (ps.replacement_pairs as { from?: string; to?: string; pos?: string }[]) || [];
            if (!pairs.length) return null;
            return (
              <Card>
                <CardHeader>
                  <CardTitle>替换明细（决胜步）</CardTitle>
                </CardHeader>
                <CardContent className="overflow-x-auto text-sm">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b text-zinc-500">
                        <th className="py-1 pr-2">原词</th>
                        <th className="py-1 pr-2">→</th>
                        <th className="py-1 pr-2">新词</th>
                        <th className="py-1">词性</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pairs.map((p, i) => (
                        <tr key={i} className="border-b border-zinc-100">
                          <td className="py-1 pr-2 font-mono text-rose-800">{p.from ?? "—"}</td>
                          <td className="py-1 pr-2 text-zinc-400">→</td>
                          <td className="py-1 pr-2 font-mono text-emerald-800">{p.to ?? "—"}</td>
                          <td className="py-1 text-zinc-600">{p.pos ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            );
          })()}

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">对抗文本 · Baseline</CardTitle>
              </CardHeader>
              <CardContent className="text-sm tabular-nums">
                {bAdv ? `${bAdv.label} / ${bAdv.score.toFixed(4)}` : "—"}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">对抗文本 · Robust</CardTitle>
              </CardHeader>
              <CardContent className="text-sm tabular-nums">
                {rAdv ? `${rAdv.label} / ${rAdv.score.toFixed(4)}` : "不可用"}
              </CardContent>
            </Card>
          </div>

          <Card className={cnCardTone(bypassed, baselineEvaded, robustHolds)}>
            <CardHeader>
              <CardTitle>攻防结论</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p className="text-base font-semibold text-zinc-900">
                {bypassed
                  ? goldAi
                    ? baselineEvaded
                      ? "Baseline 已被绕过（相对金标 AI）"
                      : "链报告翻转，但对抗文仍被判为 AI（请核对金标与阈值）"
                    : "Baseline 判断已被链翻转"
                  : "攻击未翻转 Baseline（三步内）"}
              </p>
              <p>
                是否绕过 Baseline（链语义）：<strong>{bypassed ? "是" : "否"}</strong>
                {chain.winning_step != null && <> · 决胜步：{String(chain.winning_step)}</>}
              </p>
              {!trueLabel && (
                <p className="text-zinc-600">未绑定数据集金标：以下「逃逸/恢复」仅在选手动标注为 AI 的样本时启用。</p>
              )}
              {goldAi && (
                <>
                  <p>
                    Baseline 是否被绕过（金标 AI 且对抗后不再判 AI）：
                    <strong className="ml-1">{baselineEvaded ? "是" : "否"}</strong>
                  </p>
                  <p>
                    Robust 是否仍判 AI：
                    <strong className="ml-1">
                      {robustHolds == null ? "Robust 未就绪" : robustHolds ? "是（更稳）" : "否"}
                    </strong>
                  </p>
                </>
              )}
              <Link
                href="/vuln"
                className={cn(buttonVariants({ variant: "outline", size: "sm" }))}
              >
                查看批量统计与案例表
              </Link>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function cnCardTone(bypassed: boolean, evaded: boolean, robustHolds: boolean | null): string {
  if (bypassed && evaded && robustHolds) return "border-emerald-300 bg-emerald-50/60";
  if (bypassed && evaded) return "border-amber-300 bg-amber-50/50";
  return "border-zinc-200";
}
