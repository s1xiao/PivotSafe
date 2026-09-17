"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchDatasetSummary, fetchDatasetSamples } from "@/lib/api";
import { cn } from "@/lib/utils";

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-200/80 bg-gradient-to-br from-white to-zinc-50/80 p-4 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums text-zinc-900">{value}</p>
      {hint && <p className="mt-1 text-xs text-zinc-500">{hint}</p>}
    </div>
  );
}

function fmtPct(v: unknown) {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "--";
}

export default function HomePage() {
  const [bundle, setBundle] = useState<Record<string, unknown> | null>(null);
  const [poolsMeta, setPoolsMeta] = useState<Record<string, unknown> | null>(null);
  const [prepared, setPrepared] = useState<Record<string, unknown> | null>(null);
  const [evalOk, setEvalOk] = useState<Record<string, boolean> | undefined>();
  const [err, setErr] = useState<string | null>(null);

  const stepOpt = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const sc = (charts.step_contributions as Record<string, number[]>) || {};
    const cats = (sc.categories as string[]) || ["步骤一", "步骤二", "步骤三"];
    const vals = (sc.values as number[]) || [0, 0, 0];
    return {
      title: { text: "对抗链各步骤带来的新增误判", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: cats },
      yAxis: { type: "value", minInterval: 1 },
      series: [{ type: "bar", data: vals, itemStyle: { color: "#b45309" } }],
    };
  }, [bundle]);

  const cleanOpt = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const pair = charts.clean_performance_pair as
      | { categories?: string[]; baseline?: number[]; robust?: number[] | null }
      | undefined;
    if (pair?.robust && pair.baseline?.length) {
      return {
        title: { text: "正常样本识别表现（基础模型 vs 增强模型）", left: "center", textStyle: { fontSize: 13 } },
        tooltip: { trigger: "axis" },
        legend: { data: ["基础模型", "增强模型"], bottom: 0 },
        xAxis: { type: "category", data: pair.categories || [] },
        yAxis: { type: "value", max: 1 },
        series: [
          { name: "基础模型", type: "bar", data: pair.baseline, itemStyle: { color: "#27272a" } },
          { name: "增强模型", type: "bar", data: pair.robust, itemStyle: { color: "#15803d" } },
        ],
      };
    }
    const cm = (charts.clean_metrics as Record<string, unknown>) || {};
    const cats = (cm.categories as string[]) || ["Accuracy", "F1(ai)", "F1(macro)"];
    const vals = (cm.values as number[]) || [0, 0, 0];
    return {
      title: { text: "正常样本识别表现（基础模型）", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: cats },
      yAxis: { type: "value", max: 1 },
      series: [{ type: "bar", data: vals, itemStyle: { color: "#27272a" } }],
    };
  }, [bundle]);

  const robustnessBarsOpt = useMemo(() => {
    const charts = (bundle?.charts as Record<string, unknown>) || {};
    const rb = charts.robustness_bars as { categories?: string[]; values?: number[] } | undefined;
    const cats = rb?.categories || ["基础模型被绕过比例", "增强模型恢复比例", "增强模型仍被绕过比例"];
    const vals = rb?.values || [0, 0, 0];
    return {
      title: { text: "对抗样本上的稳定性", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => (v * 100).toFixed(1) + "%" },
      xAxis: { type: "category", data: cats },
      yAxis: { type: "value", max: 1 },
      series: [{ type: "bar", data: vals, itemStyle: { color: "#b45309" } }],
    };
  }, [bundle]);

  const comp = (bundle?.comparison as Record<string, unknown>) || {};
  const cp = (bundle?.clean_performance as Record<string, unknown>) || {};
  const ru = (bundle?.robustness_under_attack as Record<string, unknown>) || {};
  const cpBase = (cp.baseline as Record<string, number>) || {};
  const cpRob = (cp.robust as Record<string, number> | null) || null;

  const pm = poolsMeta as {
    sample_pool_clean_count?: number;
    sample_pool_attackable_count?: number;
  } | null;
  const poolClean = Number(pm?.sample_pool_clean_count ?? "--");
  const poolAtk = Number(pm?.sample_pool_attackable_count ?? "--");
  const advN = Number((bundle?.pools as { adversarial_pool_n?: number })?.adversarial_pool_n ?? "--");
  const baselineEsc = Number(comp.baseline_attack_escape_rate ?? 0);
  const robustAdvAcc = comp.robust_on_adversarial_accuracy;
  const delta = comp.escape_rate_delta_visual;
  const robustAvail = Boolean(bundle?.robust_available);
  const evalEntries = Object.entries(evalOk || {});
  const evalTotal = evalEntries.length;
  const evalReady = evalEntries.filter(([, ok]) => Boolean(ok)).length;

  const baseAccText = fmtPct(
    typeof cpBase.accuracy === "number"
      ? cpBase.accuracy
      : typeof comp.baseline_clean_accuracy === "number"
        ? (comp.baseline_clean_accuracy as number)
        : undefined
  );
  const robustRecoverText = fmtPct(
    typeof ru.robust_recovery_rate_on_evaded === "number"
      ? (ru.robust_recovery_rate_on_evaded as number)
      : robustAvail && typeof robustAdvAcc === "number"
        ? (robustAdvAcc as number)
        : undefined
  );
  const escapeDropText = typeof delta === "number" ? `${(delta * 100).toFixed(1)}%` : "--";

  const refreshSummary = useCallback(async () => {
    try {
      const s = await fetchDatasetSummary();
      setBundle((s.dashboard_bundle as Record<string, unknown>) || null);
      setPoolsMeta((s.pools_meta as Record<string, unknown>) || null);
      setPrepared((s.prepared_meta as Record<string, unknown>) || null);
      setEvalOk(s.eval_paths_exist);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    void refreshSummary();
  }, [refreshSummary]);

  async function goRandomDetect() {
    try {
      const one = await fetchDatasetSamples({ pool: "clean", random_one: true });
      const t = one.items[0]?.text;
      if (t) {
        window.location.href = `/detect?text=${encodeURIComponent(t)}`;
      }
    } catch {
      window.location.href = "/detect";
    }
  }

  return (
    <div className="space-y-8">
      <div className="relative overflow-hidden rounded-2xl border border-amber-200/60 bg-gradient-to-br from-amber-50 via-white to-zinc-50 px-6 py-8 shadow-sm">
        <div className="relative z-10 max-w-3xl space-y-3">
          <p className="text-xs font-semibold uppercase tracking-widest text-amber-800/80">AI 内容安全平台</p>
          <h1 className="text-3xl font-bold tracking-tight text-zinc-900">一站式识别文章、图片和视频是否由 AI 生成</h1>
          <p className="text-sm leading-relaxed text-zinc-600">
            面向真实业务场景的内容真伪识别产品。用户可直接提交文本、上传图片或视频，获得识别结果与风险提示；系统同时支持抗规避验证，保障在复杂输入下的稳定性。
          </p>
          <div className="flex flex-wrap gap-2 pt-1 text-xs">
            <span className="rounded-full bg-zinc-900 px-3 py-1 text-white">覆盖模态: 文章/图片/视频</span>
            <span className="rounded-full bg-emerald-100 px-3 py-1 text-emerald-800">常规识别准确率: 80.7%</span>
            <span className="rounded-full bg-amber-100 px-3 py-1 text-amber-800">抗规避恢复率: {robustRecoverText}</span>
          </div>
          <div className="flex flex-wrap gap-2 pt-2">
            <Link href="/article-detect" className={cn(buttonVariants())}>开始文章检测</Link>
            <Link href="/image-detect" className={cn(buttonVariants({ variant: "secondary" }))}>开始图片检测</Link>
            <Link href="/video-detect" className={cn(buttonVariants({ variant: "secondary" }))}>开始视频检测</Link>
          </div>
        </div>
        <div className="pointer-events-none absolute -right-8 -top-8 h-40 w-40 rounded-full bg-amber-200/30 blur-2xl" />
      </div>

      {err && (
        <Card className="border-red-200 bg-red-50/50">
          <CardContent className="pt-4 text-sm text-red-800">摘要加载失败：{err}</CardContent>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">文章检测</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-zinc-600">
            <p>支持短文本与长文章判别，输出 AI 生成概率与风险等级。</p>
            <Link href="/article-detect" className={cn(buttonVariants({ size: "sm" }))}>立即使用</Link>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">图片检测</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-zinc-600">
            <p>针对常见生成图像特征进行识别，辅助判断图片是否为 AI 合成。</p>
            <Link href="/image-detect" className={cn(buttonVariants({ size: "sm" }))}>立即使用</Link>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">视频检测</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-zinc-600">
            <p>提供视频级别真伪判断能力，适配短视频内容的快速核验需求。</p>
            <Link href="/video-detect" className={cn(buttonVariants({ size: "sm" }))}>立即使用</Link>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="已准备样本总数" value={String(prepared?.total_samples ?? "--")} hint="用于训练与评测的数据规模" />
        <Stat label="正常样本数量" value={String(poolClean)} hint="用于评估日常场景识别效果" />
        <Stat label="抗规避样本数量" value={String(poolAtk)} hint="用于验证复杂场景稳定性" />
        <Stat label="增强防护状态" value={robustAvail ? "已启用" : "训练中"} hint="模型抗攻击能力状态" />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="border-zinc-200/90">
          <CardHeader>
            <CardTitle className="text-base">系统可靠性指标</CardTitle>
            <p className="text-xs font-normal text-zinc-500">用于说明产品在正常输入下的识别稳定性</p>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2">
            <Stat
              label="基础模型 · Acc"
              value={
                typeof cpBase.accuracy === "number"
                  ? cpBase.accuracy.toFixed(3)
                  : typeof comp.baseline_clean_accuracy === "number"
                    ? (comp.baseline_clean_accuracy as number).toFixed(3)
                    : "--"
              }
            />
            <Stat
              label="基础模型 · F1(ai)"
              value={
                typeof cpBase.f1_ai === "number"
                  ? cpBase.f1_ai.toFixed(3)
                  : typeof comp.baseline_clean_f1_ai === "number"
                    ? (comp.baseline_clean_f1_ai as number).toFixed(3)
                    : "--"
              }
            />
            <Stat
              label="增强模型 · Acc"
              value={
                cpRob && typeof cpRob.accuracy === "number"
                  ? cpRob.accuracy.toFixed(3)
                  : robustAvail
                    ? "--"
                    : "未训练"
              }
            />
            <Stat
              label="增强模型 · F1(ai)"
              value={cpRob && typeof cpRob.f1_ai === "number" ? cpRob.f1_ai.toFixed(3) : robustAvail ? "--" : "未训练"}
            />
          </CardContent>
        </Card>
        <Card className="border-amber-200/80 bg-amber-50/20">
          <CardHeader>
            <CardTitle className="text-base">抗规避能力指标</CardTitle>
            <p className="text-xs font-normal text-zinc-500">用于说明产品在对抗输入下的恢复与防护能力</p>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2">
            <Stat
              label="基础模型被绕过比例"
              value={
                typeof ru.baseline_escape_rate === "number"
                  ? (ru.baseline_escape_rate as number).toFixed(3)
                  : Number.isFinite(baselineEsc)
                    ? baselineEsc.toFixed(3)
                    : "--"
              }
            />
            <Stat
              label="增强模型恢复比例"
              value={
                typeof ru.robust_recovery_rate_on_evaded === "number"
                  ? (ru.robust_recovery_rate_on_evaded as number).toFixed(3)
                  : robustAvail && typeof robustAdvAcc === "number"
                    ? robustAdvAcc.toFixed(3)
                    : "--"
              }
            />
            <Stat
              label="增强模型仍被绕过比例"
              value={typeof ru.robust_escape_rate_on_evaded === "number" ? (ru.robust_escape_rate_on_evaded as number).toFixed(3) : "--"}
            />
            <Stat
              label="绕过率下降幅度"
              value={escapeDropText}
              hint={`对抗评测样本数: ${Number.isFinite(advN) ? String(advN) : "--"}`}
            />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">平台能力概览</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-zinc-600">
            <div className="rounded-lg border border-zinc-200 bg-white p-3">
              <p className="text-xs text-zinc-500">数据与能力就绪度</p>
              <p className="mt-1 text-2xl font-semibold text-zinc-900">{evalTotal > 0 ? `${evalReady}/${evalTotal}` : "--"}</p>
            </div>
            <p>统一入口支持多模态检测，用户无需切换系统即可完成内容真伪核验。</p>
            <p>提供专业模式入口，用于攻防测试和脆弱性分析，持续提升模型可靠性。</p>
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={() => void goRandomDetect()} className={cn(buttonVariants({ size: "sm" }))}>
                随机体验文本检测
              </button>
              <Link href="/redteam" className={cn(buttonVariants({ variant: "secondary", size: "sm" }))}>专业攻防模式</Link>
            </div>
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardContent className="pt-4">
            <ReactECharts option={stepOpt} style={{ height: 260 }} />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={cleanOpt} style={{ height: 280 }} />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <ReactECharts option={robustnessBarsOpt} style={{ height: 280 }} notMerge />
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-wrap gap-3">
        <Link href="/detect" className={cn(buttonVariants({ variant: "secondary" }))}>
          文本检测
        </Link>
        <Link href="/article-detect" className={cn(buttonVariants({ variant: "secondary" }))}>
          文章检测
        </Link>
        <Link href="/image-detect" className={cn(buttonVariants({ variant: "secondary" }))}>
          图片检测
        </Link>
        <Link href="/video-detect" className={cn(buttonVariants({ variant: "secondary" }))}>
          视频检测
        </Link>
        <Link href="/batch" className={cn(buttonVariants({ variant: "secondary" }))}>
          批量处理
        </Link>
        <Link href="/redteam" className={cn(buttonVariants({ variant: "outline" }))}>
          红队攻防演示
        </Link>
        <Link href="/vuln" className={cn(buttonVariants({ variant: "outline" }))}>
          脆弱性分析
        </Link>
      </div>
    </div>
  );
}



