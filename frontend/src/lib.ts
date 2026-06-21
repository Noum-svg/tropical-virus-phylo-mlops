// API client, shared types, and small formatting helpers.

// Development calls FastAPI directly; production uses the nginx same-origin proxy.
export const API = import.meta.env.DEV ? "http://localhost:8000" : "";

export interface Metrics {
  n: number;
  n_quadruplets: number;
  n_sampled: number;
  mean_violation: number;
  median_violation: number;
  max_violation: number;
  l2_loss: number;
  percent_exact: number;
}

export interface Edge {
  parent: string;
  child: string;
  branch_length: number;
}

export interface PipelineResult {
  dataset: {
    file_name: string;
    type: string;
    n_total: number;
    n_valid: number;
    min_length: number;
    max_length: number;
    mean_length: number;
    gc_content: number;
    preview: { virus_name: string; rna_sequence: string }[];
    total_rows: number;
  };
  pairs: number;
  labels: string[];
  sequence_lengths: number[];
  distance_matrix: number[][];
  corrected_matrix: number[][];
  omega: number[][];
  metrics_before: Metrics;
  metrics_after: Metrics;
  relative_improvement: number;
  history: { epoch: number; loss: number; grad_tr_norm: number; eta: number }[];
  tree: {
    newick: string;
    edges: Edge[];
    dot: string;
    n_leaves: number;
    n_internal: number;
    total_branch_length: number;
  };
  params: Record<string, string | number>;
  online?: { added: number; previous_taxa: number; total_taxa: number };
}

export interface RunParams {
  file: File | null;
  epochs: number;
  gamma: number;
  lambda_reg: number;
  quadruplet_sample_size: number;
  min_seq_length: number;
  alpha: number;
  online?: boolean;
}

async function asError(r: Response): Promise<never> {
  let detail = r.statusText;
  try {
    detail = (await r.json()).detail ?? detail;
  } catch {
    /* ignore */
  }
  throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
}

export async function health(): Promise<{ status: string; model_loaded: boolean }> {
  const r = await fetch(`${API}/health`);
  if (!r.ok) return asError(r);
  return r.json();
}

export async function runPipeline(p: RunParams): Promise<PipelineResult> {
  const fd = new FormData();
  if (p.file) fd.append("file", p.file);
  else fd.append("use_demo", "true");
  fd.append("epochs", String(p.epochs));
  fd.append("gamma", String(p.gamma));
  fd.append("lambda_reg", String(p.lambda_reg));
  fd.append("quadruplet_sample_size", String(p.quadruplet_sample_size));
  fd.append("min_seq_length", String(p.min_seq_length));
  fd.append("alpha", String(p.alpha));
  // When online learning is enabled, each run updates the persistent model.
  const endpoint = p.online ? "/online-learn" : "/pipeline";
  const r = await fetch(`${API}${endpoint}`, { method: "POST", body: fd });
  if (!r.ok) return asError(r);
  return r.json();
}

export interface TreeOpts {
  matrix: number[][];
  labels: string[];
  layout: string;
  show_labels: boolean;
  label_size: number;
  use_branch_length: boolean;
}

export async function treeImage(opts: TreeOpts): Promise<string> {
  const r = await fetch(`${API}/tree-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  if (!r.ok) return asError(r);
  return URL.createObjectURL(await r.blob());
}

export const fmt = (v: number, d = 3) =>
  Number.isFinite(v) ? v.toFixed(d) : String(v);

export function matrixToCsv(mat: number[][], labels: string[]): string {
  let out = "," + labels.join(",") + "\n";
  mat.forEach((row, i) => {
    out += labels[i] + "," + row.map((v) => v.toFixed(6)).join(",") + "\n";
  });
  return out;
}

export function download(name: string, text: string) {
  const type = name.endsWith(".json")
    ? "application/json"
    : name.endsWith(".csv")
      ? "text/csv"
      : "text/plain";
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function downloadObjectUrl(name: string, url: string) {
  if (!url) return;
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
}
