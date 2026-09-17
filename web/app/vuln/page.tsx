"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchCaseReview, fetchCasesList, fetchDatasetSummary } from "@/lib/api";
import type { CaseListItem } from "@/lib/api";
import { apiBase, cn } from "@/lib/utils";

function formatPct(x: unknown): string {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function JudgeScoreCard({ title, block }: { title: string; block: unknown }) {
  if (block == null || typeof block !== "object") {
    return (
      <div className="rounded-xl border border-zinc-200/80 bg-white/90 p-3 text-xs text-zinc-500 shadow-sm">
        <div className="font-semibold text-zinc-800">{title}</div>
        <p className="mt-1">无数据</p>
      </div>
    );
  }
  const o = block as Record<string, unknown>;
  const score = o.score;
  const scoreStr = typeof score === "number" ? score.toFixed(4) : String(score ?? "—");
  return (
    <div className="rounded-xl border border-zinc-200/80 bg-white p-3 text-xs shadow-sm">
      <div className="font-semibold text-zinc-800">{title}</div>
      <p className="mt-2 text-zinc-700">
        标签 <span className="font-mono text-violet-700">{String(o.label ?? "—")}</span>
      </p>
      <p className="text-zinc-600">AI 分数 {scoreStr}</p>
      <p className="mt-1 text-zinc-500">
        与金标一致：{o.agrees_with_gold == null ? "—" : o.agrees_with_gold ? "是" : "否"}
      </p>
    </div>
  );
}

function stepLabel(n: number | null | undefined): string {
  if (n === 1) return "Pivot";
  if (n === 2) return "Attacker2";
  if (n === 3) return "Attacker3";
  return "—";
}

export default function VulnPage() {
  const [vulnData, setVulnData] = useState<Record<string, unknown> | null>(null);
  const [bundle, setBundle] = useState<Record<string, unknown> | null>(null);
  const [vulnErr, setVulnErr] = useState<string | null>(null);
  const [cases, setCases] = useState<CaseListItem[]>([]);
  const [caseTotal, setCaseTotal] = useState(0);
  const [casePage, setCasePage] = useState(1);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [detailErr, setDetailErr] = useState<string | null>(null);
  const [jsonOpen, setJsonOpen] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/api/vuln/summary`)
      .then((r) => r.json())
      .then((j) => {
        if (!j.ready) setVulnErr(j.message || "未就绪");
        else setVulnData(j.data as Record<string, unknown>);
      })
      .catch((e) => setVulnErr(String(e)));
  }, []);

  useEffect(() => {
    fetchDatasetSummary()
      .then((s) => setBundle((s.dashboard_bundle as Record<string, unknown>) || null))
      .catch(() => setBundle(null));
  }, []);

  const loadCases = useCallback(async (page: number) => {
    try {
      const r = await fetchCasesList(page, 20);
      setCases(r.items);
      setCaseTotal(r.total);
      setCasePage(r.page);
    } catch {
      setCases([]);
      setCaseTotal(0);
    }
  }, []);

  useEffect(() => {
    void loadCases(1);
  }, [loadCases]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const id = new URLSearchParams(window.location.search).get("case");
    if (!id) return;
    setDetailErr(null);
    fetchCaseReview(id)
      .then(setDetail)
      .catch((e) => setDetailErr(String(e)));
  }, []);

  const openCase = (id: string) => {
    window.history.replaceState(null, "", `/vuln?case=${encodeURIComponent(id)}`);
    setDetailErr(null);
    fetchCaseReview(id)
      .then(setDetail)
      .catch((e) => setDetailErr(String(e)));
  };

  const randomNextCase = async () => {
    if (cases.length === 0) return;
    const pick = cases[Math.floor(Math.random() * cases.length)];
    openCase(pick.case_id);
  };

  const cp =
    (bundle?.clean_performance as Record<string, unknown>) ||
    (vulnData?.clean_performance as Record<string, unknown>) ||
    {};
  const ru =
    (bundle?.robustness_under_attack as Record<string, unknown>) ||
    (vulnData?.robustness_under_attack as Record<string, unknown>) ||
    {};
  const blClean = (cp.baseline as Record<string, unknown>) || {};
  const rbClean = (cp.robust as Record<string, unknown> | null) || null;

  const chartCleanPair = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const pair = charts.clean_performance_pair as
      | { categories?: string[]; baseline?: number[]; robust?: number[] | null }
      | undefined;
    if (pair?.robust && pair.baseline?.length) {
      return {
        title: { text: "Clean · Baseline vs Robust", left: "center", textStyle: { fontSize: 13 } },
        tooltip: { trigger: "axis" },
        legend: { data: ["Baseline", "Robust"], bottom: 0 },
        xAxis: { type: "category", data: pair.categories || [] },
        yAxis: { type: "value", max: 1 },
        series: [
          { name: "Baseline", type: "bar", data: pair.baseline, itemStyle: { color: "#27272a" } },
          { name: "Robust", type: "bar", data: pair.robust, itemStyle: { color: "#15803d" } },
        ],
      };
    }
    const cm = (charts.clean_metrics as Record<string, unknown>) || {};
    const cats = (cm.categories as string[]) || ["Accuracy", "F1(ai)", "F1(macro)"];
    const vals = (cm.values as number[]) || [0, 0, 0];
    return {
      title: { text: "Baseline · Clean（Robust 未训练时仅 Baseline）", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: cats },
      yAxis: { type: "value", max: 1 },
      series: [{ type: "bar", data: vals, itemStyle: { color: "#27272a" } }],
    };
  }, [bundle]);

  const chartRobustnessBars = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const rb = charts.robustness_bars as { categories?: string[]; values?: number[] } | undefined;
    if (rb?.categories?.length && rb.values?.length) {
      return {
        title: { text: "攻击下鲁棒性（与 Clean 指标分列）", left: "center", textStyle: { fontSize: 12 } },
        tooltip: { trigger: "axis", valueFormatter: (v: number) => (v * 100).toFixed(1) + "%" },
        xAxis: { type: "category", data: rb.categories, axisLabel: { interval: 0, rotate: 22 } },
        yAxis: { type: "value", max: 1 },
        series: [
          {
            type: "bar",
            data: rb.values,
            itemStyle: {
              color: (p: { dataIndex: number }) =>
                p.dataIndex === 0 ? "#dc2626" : p.dataIndex === 1 ? "#15803d" : "#a16207",
            },
          },
        ],
      };
    }
    const ec = (charts.escape_compare as Record<string, unknown>) || {};
    const cats = (ec.categories as string[]) || ["Baseline 逃逸率", "Robust 对抗子集恢复率"];
    const vals = (ec.values as number[]) || [0, 0];
    return {
      title: { text: "逃逸 vs Robust（兼容旧产物）", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: cats },
      yAxis: { type: "value", max: 1 },
      series: [
        {
          type: "bar",
          data: vals,
          itemStyle: {
            color: (p: { dataIndex: number }) => (p.dataIndex === 0 ? "#dc2626" : "#15803d"),
          },
        },
      ],
    };
  }, [bundle]);

  const chartSteps = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const sc = (charts.step_contributions as Record<string, unknown>) || {};
    const cats = (sc.categories as string[]) || ["Pivot", "Attacker2", "Attacker3"];
    const vals = (sc.values as number[]) || [0, 0, 0];
    return {
      title: { text: "链上各步「决胜」次数（互斥，之和为逃逸总数）", left: "center", textStyle: { fontSize: 11 } },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: cats },
      yAxis: { type: "value", minInterval: 1 },
      series: [{ type: "bar", data: vals, itemStyle: { color: "#b45309" } }],
    };
  }, [bundle]);

  const chartCumulative = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const sc = (charts.step_contributions_cumulative as Record<string, unknown>) || {};
    const cats = (sc.categories as string[]) || ["累计至 Pivot", "累计至 Step2", "累计逃逸"];
    const vals = (sc.values as number[]) || [0, 0, 0];
    return {
      title: { text: "累计逃逸（瀑布语义：单调上升至总逃逸）", left: "center", textStyle: { fontSize: 11 } },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: cats },
      yAxis: { type: "value", minInterval: 1 },
      series: [
        {
          type: "line",
          smooth: true,
          areaStyle: { opacity: 0.12 },
          data: vals,
          itemStyle: { color: "#7c3aed" },
        },
      ],
    };
  }, [bundle]);

  const chartSource = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const se = (charts.source_escape_rates as { categories?: string[]; values?: number[] }) || {};
    const cats = se.categories || [];
    const vals = se.values || [];
    return {
      title: { text: "数据来源 vs 逃逸率（样本量 Top）", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => (v * 100).toFixed(1) + "%" },
      xAxis: { type: "category", data: cats, axisLabel: { rotate: 26, interval: 0 } },
      yAxis: { type: "value", max: 1 },
      series: [{ type: "bar", data: vals, itemStyle: { color: "#6366f1" } }],
    };
  }, [bundle]);

  const heatmapOption = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const hm = charts.pivot_replace_heatmap as
      | { pivot_pos?: string[]; replace_pos?: string[]; data?: number[][] }
      | undefined;
    if (!hm?.data?.length || !hm.pivot_pos?.length || !hm.replace_pos?.length) return null;
    const maxV = Math.max(...hm.data.map((d) => d[2]));
    return {
      title: { text: "Pivot 词性 × 替换词词性（决胜步首条）", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { position: "top" as const },
      grid: { top: 52, bottom: 72, left: 72, right: 20 },
      xAxis: { type: "category" as const, data: hm.replace_pos, splitArea: { show: true } },
      yAxis: { type: "category" as const, data: hm.pivot_pos, splitArea: { show: true } },
      visualMap: {
        min: 0,
        max: maxV || 1,
        calculable: true,
        orient: "horizontal" as const,
        left: "center",
        bottom: 8,
        inRange: { color: ["#f5f3ff", "#6d28d9"] },
      },
      series: [
        {
          name: "计数",
          type: "heatmap" as const,
          data: hm.data,
          label: { show: true },
          emphasis: {
            itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.25)" },
          },
        },
      ],
    };
  }, [bundle]);

  const pivotPie = useMemo(() => {
    const pd = (vulnData?.pivot_pos_distribution as Record<string, number>) || {};
    const keys = Object.keys(pd).slice(0, 14);
    return {
      title: { text: "Pivot 词性分布", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "item" },
      series: [
        {
          type: "pie",
          radius: ["32%", "62%"],
          data: keys.map((k) => ({ name: k, value: pd[k] })),
        },
      ],
    };
  }, [vulnData]);

  const repPie = useMemo(() => {
    const pd = (vulnData?.replaced_pos_distribution as Record<string, number>) || {};
    const keys = Object.keys(pd).slice(0, 14);
    return {
      title: { text: "替换词词性分布", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "item" },
      series: [
        {
          type: "pie",
          radius: ["32%", "62%"],
          data: keys.map((k) => ({ name: k, value: pd[k] })),
        },
      ],
    };
  }, [vulnData]);

  const lenEsc = useMemo(() => {
    const raw = (vulnData?.length_bucket_escape as Record<string, number>) || {};
    const buckets = ["0-50", "51-100", "101-150", "151-200"];
    const rates = buckets.map((b) => {
      const s = raw[`${b}|success`] || 0;
      const t = raw[`${b}|total`] || 0;
      return t ? s / t : 0;
    });
    return {
      title: { text: "文本长度区间 vs 逃逸率", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => (v * 100).toFixed(1) + "%" },
      xAxis: { type: "category", data: buckets.map((b) => `${b} 字`) },
      yAxis: { type: "value", max: 1 },
      series: [{ type: "line", data: rates, smooth: true, itemStyle: { color: "#0d9488" } }],
    };
  }, [vulnData]);

  const hasSourceChart = ((bundle?.charts as Record<string, unknown>)?.source_escape_rates as { categories?: string[] })
    ?.categories?.length;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900">脆弱性 · 数据挖掘面板</h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-zinc-600">
            上方为「Clean 检测能力」与「攻击下鲁棒性」两组口径；中部为链贡献、词性与分层；底部为可答辩翻页的案例表。原始 JSON
            仅作调试，默认折叠。
          </p>
        </div>
        <Button variant="secondary" size="sm" type="button" onClick={() => void randomNextCase()}>
          随机下一条案例
        </Button>
      </div>

      {vulnErr && !vulnData && (
        <Card className="border-amber-200 bg-amber-50/40">
          <CardContent className="pt-4 text-sm text-amber-900">{vulnErr}</CardContent>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="border-emerald-200/80 bg-gradient-to-br from-emerald-50/90 via-white to-white shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-emerald-950">Clean Performance</CardTitle>
            <p className="text-xs text-emerald-900/70">回答：无扰动时能否区分人机文本（与逃逸指标分列）。</p>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div>
              <div className="text-xs font-medium text-zinc-500">Baseline Acc</div>
              <div className="text-xl font-semibold tabular-nums text-zinc-900">{formatPct(blClean.accuracy)}</div>
            </div>
            <div>
              <div className="text-xs font-medium text-zinc-500">Baseline F1(ai)</div>
              <div className="text-xl font-semibold tabular-nums text-zinc-900">{formatPct(blClean.f1_ai)}</div>
            </div>
            <div>
              <div className="text-xs font-medium text-zinc-500">Robust Acc</div>
              <div className="text-xl font-semibold tabular-nums text-emerald-800">
                {rbClean ? formatPct(rbClean.accuracy) : "—"}
              </div>
            </div>
            <div>
              <div className="text-xs font-medium text-zinc-500">Robust F1(ai)</div>
              <div className="text-xl font-semibold tabular-nums text-emerald-800">
                {rbClean ? formatPct(rbClean.f1_ai) : "—"}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-rose-200/80 bg-gradient-to-br from-rose-50/80 via-white to-white shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-rose-950">Robustness under Attack</CardTitle>
            <p className="text-xs text-rose-900/70">回答：对抗样本下 Baseline 是否被绕过，Robust 是否恢复。</p>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div>
              <div className="text-xs font-medium text-zinc-500">Baseline 逃逸率</div>
              <div className="text-xl font-semibold tabular-nums text-rose-700">{formatPct(ru.baseline_escape_rate)}</div>
            </div>
            <div>
              <div className="text-xs font-medium text-zinc-500">逃逸条数</div>
              <div className="text-xl font-semibold tabular-nums text-zinc-900">
                {typeof ru.baseline_evaded_count === "number" ? ru.baseline_evaded_count : "—"}
              </div>
            </div>
            <div>
              <div className="text-xs font-medium text-zinc-500">恢复率(已逃逸子集)</div>
              <div className="text-xl font-semibold tabular-nums text-emerald-700">
                {formatPct(ru.robust_recovery_rate_on_evaded)}
              </div>
            </div>
            <div>
              <div className="text-xs font-medium text-zinc-500">恢复条数</div>
              <div className="text-xl font-semibold tabular-nums text-zinc-900">
                {typeof ru.robust_recovery_count === "number" ? ru.robust_recovery_count : "—"}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={chartCleanPair} style={{ height: 260 }} notMerge />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={chartRobustnessBars} style={{ height: 260 }} notMerge />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={chartSteps} style={{ height: 260 }} notMerge />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={chartCumulative} style={{ height: 260 }} notMerge />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={pivotPie} style={{ height: 280 }} notMerge />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={repPie} style={{ height: 280 }} notMerge />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={lenEsc} style={{ height: 260 }} notMerge />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            {hasSourceChart ? (
              <ReactECharts option={chartSource} style={{ height: 260 }} notMerge />
            ) : (
              <div className="flex h-[260px] items-center justify-center text-sm text-zinc-400">
                运行 <code className="mx-1 rounded bg-zinc-100 px-1 text-xs">build_eval_artifacts</code> 生成数据源分层后显示
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {heatmapOption && (
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={heatmapOption} style={{ height: 340 }} notMerge />
          </CardContent>
        </Card>
      )}

      {vulnData && (
        <Card className="border-zinc-200">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-zinc-600">调试：原始摘要 JSON</CardTitle>
            <Button variant="ghost" size="sm" type="button" onClick={() => setJsonOpen((o) => !o)}>
              {jsonOpen ? "收起" : "展开"}
            </Button>
          </CardHeader>
          {jsonOpen && (
            <CardContent>
              <pre className="max-h-64 overflow-auto rounded-lg bg-zinc-900 p-3 text-xs text-zinc-100">
                {JSON.stringify(vulnData, null, 2)}
              </pre>
            </CardContent>
          )}
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>案例表</CardTitle>
          <p className="text-xs font-normal text-zinc-500">
            Baseline 对抗列表示攻击后在对抗文本上的预测；决胜步表示固定链中哪一步翻转了 Baseline。
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-zinc-500">共 {caseTotal} 条 · 当前页 {casePage}</p>
          <div className="overflow-x-auto rounded-lg border border-zinc-200">
            <table className="w-full min-w-[960px] text-left text-sm">
              <thead className="bg-zinc-50 text-xs text-zinc-600">
                <tr>
                  <th className="p-2">摘要</th>
                  <th className="p-2">金标</th>
                  <th className="p-2">Baseline 对抗</th>
                  <th className="p-2">Robust 对抗</th>
                  <th className="p-2">绕过</th>
                  <th className="p-2">恢复</th>
                  <th className="p-2">决胜步</th>
                  <th className="p-2 w-24" />
                </tr>
              </thead>
              <tbody>
                {cases.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="p-4 text-center text-zinc-500">
                      无案例数据。请运行 scripts/build_eval_artifacts.py
                    </td>
                  </tr>
                ) : (
                  cases.map((c) => (
                    <tr key={c.case_id} className="border-t border-zinc-100 hover:bg-violet-50/40">
                      <td className="max-w-[200px] p-2 text-zinc-700">{c.text_preview}</td>
                      <td className="p-2 font-mono text-xs">{c.true_label}</td>
                      <td className="p-2 font-mono text-xs">{c.baseline_label_adv ?? "—"}</td>
                      <td className="p-2 font-mono text-xs">{c.robust_label_adv ?? "—"}</td>
                      <td className="p-2">{c.baseline_evaded ? "是" : "否"}</td>
                      <td className="p-2">
                        {c.robust_recovered == null ? "—" : c.robust_recovered ? "是" : "否"}
                      </td>
                      <td className="p-2 text-xs">{stepLabel(c.winning_step)}</td>
                      <td className="p-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-8 text-xs"
                          type="button"
                          onClick={() => openCase(c.case_id)}
                        >
                          复盘
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
              disabled={casePage <= 1}
              type="button"
              onClick={() => void loadCases(casePage - 1)}
            >
              上一页
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={casePage * 20 >= caseTotal}
              type="button"
              onClick={() => void loadCases(casePage + 1)}
            >
              下一页
            </Button>
          </div>
        </CardContent>
      </Card>

      {(detail || detailErr) && (
        <Card className="border-violet-200 bg-gradient-to-br from-violet-50/50 to-white shadow-sm">
          <CardHeader>
            <CardTitle>单条复盘</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            {detailErr && <p className="text-red-600">{detailErr}</p>}
            {detail && (
              <>
                <p className="font-mono text-xs text-zinc-500">case_id: {String(detail.case_id)}</p>
                <div>
                  <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">原文</div>
                  <p className="mt-1 rounded-lg border border-zinc-200/80 bg-white p-3 text-zinc-800 leading-relaxed">
                    {String(detail.original_text || "")}
                  </p>
                </div>
                {detail.final_adversarial_text ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">最终对抗文本</div>
                    <p className="mt-1 rounded-lg border border-rose-200/60 bg-rose-50/30 p-3 font-mono text-xs text-zinc-800 leading-relaxed">
                      {String(detail.final_adversarial_text)}
                    </p>
                  </div>
                ) : null}
                <div className="grid gap-2 sm:grid-cols-2">
                  <JudgeScoreCard title="Baseline · 原文" block={detail.baseline_on_original} />
                  <JudgeScoreCard title="Robust · 原文" block={detail.robust_on_original} />
                  <JudgeScoreCard title="Baseline · 对抗文" block={detail.baseline_on_adversarial} />
                  <JudgeScoreCard title="Robust · 对抗文" block={detail.robust_on_adversarial} />
                </div>
                <div className="flex flex-wrap gap-2">
                  <Link
                    href={`/redteam?text=${encodeURIComponent(String(detail.original_text || ""))}`}
                    className={cn(buttonVariants({ size: "sm", variant: "secondary" }))}
                  >
                    在红队页重放
                  </Link>
                  <Button size="sm" variant="outline" type="button" onClick={() => void randomNextCase()}>
                    随机下一条
                  </Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
