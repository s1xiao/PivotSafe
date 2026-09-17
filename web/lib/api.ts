import { apiBase } from "@/lib/utils";

export type DatasetSummary = {
  prepared_meta?: Record<string, unknown> | null;
  pools_meta?: Record<string, unknown> | null;
  dashboard_bundle?: Record<string, unknown> | null;
  vuln_summary?: Record<string, unknown> | null;
  pools_paths_exist?: Record<string, boolean>;
  eval_paths_exist?: Record<string, boolean>;
};

export async function fetchDatasetSummary(): Promise<DatasetSummary> {
  const r = await fetch(`${apiBase}/api/dataset/summary`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type RebuildEvalArtifactsResponse = {
  ok: boolean;
  message: string;
  stats: Record<string, unknown>;
};

/** 一键生成 dashboard_bundle / cases_review / baseline_clean_metrics 等（可能耗时数分钟） */
export async function postRebuildEvalArtifacts(limit: number): Promise<RebuildEvalArtifactsResponse> {
  const r = await fetch(`${apiBase}/api/dataset/rebuild-eval-artifacts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type DatasetSample = {
  id: string;
  text: string;
  label: string;
  source?: string | null;
  split?: string | null;
  char_len: number;
};

export type DatasetSamplesResponse = {
  items: DatasetSample[];
  total: number;
  page: number;
  per_page: number;
  pool: string;
};

export async function fetchDatasetSamples(params: {
  pool?: "clean" | "attackable" | "adversarial";
  page?: number;
  per_page?: number;
  split?: string;
  label?: string;
  source_substr?: string;
  random_one?: boolean;
}): Promise<DatasetSamplesResponse> {
  const q = new URLSearchParams();
  if (params.pool) q.set("pool", params.pool);
  if (params.page != null) q.set("page", String(params.page));
  if (params.per_page != null) q.set("per_page", String(params.per_page));
  if (params.split) q.set("split", params.split);
  if (params.label) q.set("label", params.label);
  if (params.source_substr) q.set("source_substr", params.source_substr);
  if (params.random_one) q.set("random_one", "true");
  const r = await fetch(`${apiBase}/api/dataset/samples?${q}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type TextCompareResponse = {
  baseline: { label: string; ai_score: number };
  robust?: { label: string; ai_score: number } | null;
  models_agree: boolean;
  gold_label?: string | null;
  baseline_agrees_with_gold?: boolean | null;
  robust_agrees_with_gold?: boolean | null;
};

export async function postTextCompare(
  text: string,
  trueLabel?: string | null
): Promise<TextCompareResponse> {
  const r = await fetch(`${apiBase}/api/text/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      true_label: trueLabel || undefined,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchCaseReview(caseId: string): Promise<Record<string, unknown>> {
  const r = await fetch(`${apiBase}/api/cases/review/${encodeURIComponent(caseId)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type CaseListItem = {
  case_id: string;
  text_preview: string;
  true_label: string;
  baseline_evaded: boolean;
  robust_recovered: boolean | null;
  char_len: number;
  baseline_label_adv?: string | null;
  robust_label_adv?: string | null;
  winning_step?: number | null;
};

export type CaseListResponse = {
  items: CaseListItem[];
  total: number;
  page: number;
  per_page: number;
};

export async function fetchCasesList(page = 1, perPage = 25): Promise<CaseListResponse> {
  const q = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
  });
  const r = await fetch(`${apiBase}/api/cases/list?${q}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type ImageScoreItem = {
  path: string;
  ai_score: number;
  label_pred: string;
};

export type ImageScoreResponse = {
  model_name: string;
  threshold: number;
  results: ImageScoreItem[];
};

export type VideoScoreResponse = {
  video_path: string;
  frame_scores: number[];
  mean_score: number;
  max_score: number;
  label_pred: string;
  num_frames: number;
};

export type SamplePathsResponse = {
  image_path: string;
  video_path: string;
  human_image_path: string;
  ai_image_path: string;
  human_video_path: string;
  ai_video_path: string;
};

export async function fetchSamplePaths(): Promise<SamplePathsResponse> {
  const r = await fetch(`${apiBase}/api/sample/paths`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function postImageScoreByPaths(imagePaths: string[]): Promise<ImageScoreResponse> {
  const r = await fetch(`${apiBase}/api/image/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_paths: imagePaths }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function postImageScoreByFiles(files: File[]): Promise<ImageScoreResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const r = await fetch(`${apiBase}/api/image/upload-score`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function postVideoScoreByPath(videoPath: string): Promise<VideoScoreResponse> {
  const r = await fetch(`${apiBase}/api/video/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path: videoPath }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function postVideoScoreByFile(file: File): Promise<VideoScoreResponse> {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${apiBase}/api/video/upload-score`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}


export type ArticleSentenceItem = {
  index: number;
  text: string;
  ai_score: number;
  label_pred: string;
};

export type ArticleScoreResponse = {
  model_name: string;
  positive_label: string;
  threshold: number;
  total_sentences: number;
  ai_sentences: number;
  ai_rate: number;
  avg_ai_score: number;
  max_ai_score: number;
  sentences: ArticleSentenceItem[];
};

export async function postArticleScore(params: {
  text: string;
  modelVariant?: string;
  minSentenceLen?: number;
}): Promise<ArticleScoreResponse> {
  const r = await fetch(`${apiBase}/api/article/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: params.text,
      model_variant: params.modelVariant ?? "zh_v3",
      min_sentence_len: params.minSentenceLen ?? 4,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type ArticleUploadParseResponse = {
  filename: string;
  chars: number;
  text: string;
};

export async function postArticleParseByFile(file: File): Promise<ArticleUploadParseResponse> {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${apiBase}/api/article/upload-parse`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type BatchResultItem = {
  file_name: string;
  status: string;
  label_pred?: string | null;
  ai_score?: number | null;
  message?: string | null;
  detail?: Record<string, unknown> | null;
};

export type BatchEngineStatus = {
  status: string;
  message: string;
  path?: string | null;
};

export type BatchProcessResponse = {
  job_id: string;
  modality: "text" | "image" | "video";
  total_files: number;
  success_files: number;
  failed_files: number;
  ai_files: number;
  human_files: number;
  avg_ai_score: number;
  results: BatchResultItem[];
  artifacts: Record<string, string>;
  engines: {
    spark: BatchEngineStatus;
    hadoop: BatchEngineStatus;
    hive: BatchEngineStatus;
  };
};

export async function postBatchZipProcess(params: {
  zipFile: File;
  modality: "text" | "image" | "video";
  modelVariant?: string;
  minSentenceLen?: number;
  enableSpark?: boolean;
  enableHdfs?: boolean;
  enableHive?: boolean;
}): Promise<BatchProcessResponse> {
  const form = new FormData();
  form.append("file", params.zipFile);
  form.append("modality", params.modality);
  form.append("model_variant", params.modelVariant ?? "zh_v3");
  form.append("min_sentence_len", String(params.minSentenceLen ?? 4));
  form.append("enable_spark", String(params.enableSpark ?? true));
  form.append("enable_hdfs", String(params.enableHdfs ?? true));
  form.append("enable_hive", String(params.enableHive ?? true));

  const r = await fetch(`${apiBase}/api/batch/upload-process`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
