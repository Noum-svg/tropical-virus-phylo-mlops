import { useEffect, useMemo, useState } from "react";
import {
  BarCompare,
  Card,
  Heatmap,
  Histogram,
  LossChart,
  Metric,
  Stepper,
  Tabs,
} from "./ui";
import {
  API,
  download,
  downloadObjectUrl,
  fmt,
  health,
  matrixToCsv,
  Metrics,
  PipelineResult,
  RunParams,
  runPipeline,
  treeImage,
} from "./lib";

type PageName =
  | "Overview"
  | "Dataset"
  | "Preprocessing"
  | "Distance Matrix"
  | "Tropical Correction"
  | "Phylogenetic Tree"
  | "Metrics"
  | "Downloads"
  | "API Docs"
  | "About";

type Theme = "light" | "dark";

interface NavItem {
  name: PageName;
  description: string;
}

const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: "Workspace",
    items: [
      { name: "Overview", description: "Run summary and health" },
      { name: "Dataset", description: "Upload and configuration" },
      { name: "Preprocessing", description: "Sequence quality control" },
    ],
  },
  {
    label: "Scientific analysis",
    items: [
      { name: "Distance Matrix", description: "Pairwise sequence distances" },
      { name: "Tropical Correction", description: "Optimization diagnostics" },
      { name: "Phylogenetic Tree", description: "Neighbor-Joining result" },
      { name: "Metrics", description: "Before and after comparison" },
    ],
  },
  {
    label: "Delivery",
    items: [
      { name: "Downloads", description: "Export analysis artifacts" },
      { name: "API Docs", description: "Integration endpoints" },
      { name: "About", description: "Methods and disclaimer" },
    ],
  },
];

const PAGE_META = Object.fromEntries(
  NAV_GROUPS.flatMap((group) => group.items.map((item) => [item.name, item])),
) as Record<PageName, NavItem>;

const ICONS: Record<
  PageName | "Menu" | "Close" | "Sun" | "Moon" | "Play" | "Check" | "Upload" | "Arrow",
  string
> = {
  Overview: "M4 13h6V4H4v9Zm10 7h6V9h-6v11ZM4 20h6v-3H4v3Zm10-15h6V4h-6v1Z",
  Dataset: "M5 5.5C5 4.12 8.13 3 12 3s7 1.12 7 2.5S15.87 8 12 8 5 6.88 5 5.5Zm0 0V12c0 1.38 3.13 2.5 7 2.5s7-1.12 7-2.5V5.5M5 12v6.5C5 19.88 8.13 21 12 21s7-1.12 7-2.5V12",
  Preprocessing: "M4 6h16M7 12h10M10 18h4M6 3v6m12-6v6M9 9v6m6-6v6m-3 6v-6",
  "Distance Matrix": "M4 4h6v6H4V4Zm10 0h6v6h-6V4ZM4 14h6v6H4v-6Zm10 0h6v6h-6v-6Z",
  "Tropical Correction": "m4 18 5-7 4 4 7-9M4 6h4m8 12h4",
  "Phylogenetic Tree": "M12 3v6m0 0-6 6m6-6 6 6M6 15v6m12-6v6m-9-3h6",
  Metrics: "M4 20V10m5 10V4m6 16v-7m5 7V7M2 20h20",
  Downloads: "M12 3v12m-5-5 5 5 5-5M5 21h14",
  "API Docs": "m8 4-4 8 4 8m8-16 4 8-4 8m-5-17-4 18",
  About: "M12 8h.01M11 12h1v4h1M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  Menu: "M4 7h16M4 12h16M4 17h16",
  Close: "m6 6 12 12M18 6 6 18",
  Sun: "M12 4V2m0 20v-2m8-8h2M2 12h2m13.66-5.66 1.42-1.42M4.92 19.08l1.42-1.42m0-11.32L4.92 4.92m14.16 14.16-1.42-1.42M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z",
  Moon: "M20 15.5A8.5 8.5 0 0 1 8.5 4 8.5 8.5 0 1 0 20 15.5Z",
  Play: "m9 7 8 5-8 5V7Z",
  Check: "m5 12 4 4L19 6",
  Upload: "M12 16V4m-5 5 5-5 5 5M5 20h14",
  Arrow: "M5 12h14m-6-6 6 6-6 6",
};

const STEPS = ["Upload", "Clean", "Distances", "Correction", "Tree", "Complete"];
const MKEYS: [keyof Metrics, string][] = [
  ["mean_violation", "Mean violation"],
  ["median_violation", "Median violation"],
  ["max_violation", "Maximum violation"],
  ["percent_exact", "Exact quadruplets"],
  ["l2_loss", "L2 loss"],
];

function Icon({
  name,
  className = "h-[18px] w-[18px]",
}: {
  name: keyof typeof ICONS;
  className?: string;
}) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={1.8}
      viewBox="0 0 24 24"
    >
      <path d={ICONS[name]} />
    </svg>
  );
}

function DnaMark({ small = false }: { small?: boolean }) {
  return (
    <div className={`dna-mark ${small ? "h-9 w-9" : "h-11 w-11"}`} aria-hidden="true">
      <svg fill="none" viewBox="0 0 48 48">
        <path
          d="M14 7c14 7 6 27 20 34M34 7c-14 7-6 27-20 34"
          stroke="currentColor"
          strokeLinecap="round"
          strokeWidth="3.2"
        />
        <path
          d="m17 12 14 2M14 20l20 4M14 29l18 4M17 38l14-2"
          opacity=".7"
          stroke="currentColor"
          strokeLinecap="round"
          strokeWidth="2.2"
        />
      </svg>
    </div>
  );
}

function MetricsTable({ before, after }: { before: Metrics; after: Metrics }) {
  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Before</th>
            <th>After</th>
            <th>Change</th>
          </tr>
        </thead>
        <tbody>
          {MKEYS.map(([key, label]) => {
            const beforeValue = before[key] as number;
            const afterValue = after[key] as number;
            const isExact = key === "percent_exact";
            const change = isExact
              ? afterValue - beforeValue
              : beforeValue
                ? ((beforeValue - afterValue) / beforeValue) * 100
                : 0;
            return (
              <tr key={key}>
                <td className="font-medium text-slate-800 dark:text-slate-100">{label}</td>
                <td>{fmt(beforeValue)}</td>
                <td>{fmt(afterValue)}</td>
                <td>
                  <span className={`change-pill ${change >= 0 ? "change-positive" : "change-negative"}`}>
                    {isExact
                      ? `${change >= 0 ? "+" : ""}${change.toFixed(2)} pp`
                      : `${change.toFixed(2)}%`}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function KeyValueList({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="divide-y divide-slate-100 dark:divide-slate-800">
      {rows.map(([key, value]) => (
        <div
          className="flex items-center justify-between gap-5 py-3 first:pt-0 last:pb-0"
          key={key}
        >
          <dt className="text-sm text-slate-500 dark:text-slate-400">{key}</dt>
          <dd className="min-w-0 text-right text-sm font-semibold text-slate-900 dark:text-white">
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function StatusDot({ status }: { status: string }) {
  const healthy = status === "ok";
  const checking = status === "checking";
  return (
    <span
      className={`status-pill ${
        healthy ? "status-online" : checking ? "status-checking" : "status-offline"
      }`}
    >
      <span className="status-dot" />
      {healthy ? "API online" : checking ? "Checking API" : "API unavailable"}
    </span>
  );
}

function LoadingOverlay() {
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-panel">
        <div className="loading-orbit">
          <span />
        </div>
        <p className="mt-6 text-lg font-semibold text-slate-950 dark:text-white">
          Running scientific pipeline
        </p>
        <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500 dark:text-slate-400">
          Cleaning sequences, computing distances, optimizing the tropical correction,
          and reconstructing the tree.
        </p>
        <div className="mt-5 flex items-center justify-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-teal-600 dark:text-teal-400">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
          Processing
        </div>
      </div>
    </div>
  );
}

function ErrorBanner({ message, onClose }: { message: string; onClose: () => void }) {
  return (
    <div className="mb-5 flex items-start gap-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-rose-800 shadow-sm dark:border-rose-900/60 dark:bg-rose-950/30 dark:text-rose-200">
      <span className="mt-0.5 grid h-6 w-6 flex-none place-items-center rounded-full bg-rose-100 text-sm font-bold dark:bg-rose-900">
        !
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold">The pipeline could not complete</p>
        <p className="mt-0.5 break-words text-sm opacity-80">{message}</p>
      </div>
      <button aria-label="Dismiss error" className="icon-button h-8 w-8" onClick={onClose}>
        <Icon name="Close" className="h-4 w-4" />
      </button>
    </div>
  );
}

function EmptyState({
  onDataset,
  onRun,
  loading,
}: {
  onDataset: () => void;
  onRun: () => void;
  loading: boolean;
}) {
  return (
    <div className="empty-state">
      <div className="empty-visual">
        <div className="empty-ring empty-ring-one" />
        <div className="empty-ring empty-ring-two" />
        <DnaMark />
      </div>
      <span className="eyebrow">New analysis</span>
      <h2 className="mt-3 text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">
        Turn viral sequences into a phylogenetic tree
      </h2>
      <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-slate-500 dark:text-slate-400">
        Upload a CSV containing <code>virus_name</code> and <code>rna_sequence</code>,
        or explore the workflow with the clearly labelled synthetic demonstration dataset.
      </p>
      <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
        <button className="btn btn-primary" onClick={onDataset}>
          <Icon name="Upload" /> Upload dataset
        </button>
        <button className="btn" disabled={loading} onClick={onRun}>
          <Icon name="Play" /> Run synthetic demo
        </button>
      </div>
      <div className="mt-8 grid w-full max-w-3xl gap-3 sm:grid-cols-3">
        {[
          ["01", "Validate", "Clean and filter sequence data"],
          ["02", "Optimize", "Reduce four-point violations"],
          ["03", "Reconstruct", "Generate the Neighbor-Joining tree"],
        ].map(([number, title, detail]) => (
          <div
            className="rounded-lg border border-slate-200/80 bg-white/70 p-4 text-left dark:border-slate-800 dark:bg-slate-900/60"
            key={number}
          >
            <span className="text-xs font-bold tracking-[0.18em] text-teal-600 dark:text-teal-400">
              {number}
            </span>
            <p className="mt-2 text-sm font-semibold text-slate-900 dark:text-white">
              {title}
            </p>
            <p className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">
              {detail}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function Toggle({
  checked,
  label,
  onChange,
}: {
  checked: boolean;
  label: string;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4 text-sm font-medium text-slate-700 dark:text-slate-200">
      {label}
      <span className={`toggle ${checked ? "toggle-on" : ""}`}>
        <input
          checked={checked}
          className="sr-only"
          type="checkbox"
          onChange={(event) => onChange(event.target.checked)}
        />
        <span />
      </span>
    </label>
  );
}

function CodePanel({ content }: { content: string }) {
  return (
    <pre className="max-h-[560px] overflow-auto whitespace-pre-wrap break-all rounded-lg border border-slate-200 bg-slate-950 p-5 font-mono text-xs leading-6 text-slate-200 dark:border-slate-800">
      {content}
    </pre>
  );
}

function TreePage({ data }: { data: PipelineResult }) {
  const [layout, setLayout] = useState("circular");
  const [labels, setLabels] = useState(true);
  const [branch, setBranch] = useState(true);
  const [size, setSize] = useState(9);
  const [imageUrl, setImageUrl] = useState("");
  const [imageError, setImageError] = useState("");

  useEffect(() => {
    let activeUrl = "";
    setImageError("");
    setImageUrl("");
    treeImage({
      matrix: data.corrected_matrix,
      labels: data.labels,
      layout,
      show_labels: labels,
      label_size: size,
      use_branch_length: branch,
    })
      .then((url) => {
        activeUrl = url;
        setImageUrl(url);
      })
      .catch((caught: Error) => setImageError(caught.message));
    return () => {
      if (activeUrl) URL.revokeObjectURL(activeUrl);
    };
  }, [data, layout, labels, branch, size]);

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
      <Card className="min-w-0 overflow-hidden" noPadding>
        <Tabs tabs={["Tree viewer", "Newick", "Edge list", "DOT"]}>
          {(activeTab) => (
            <div className="p-4 sm:p-6">
              {activeTab === "Tree viewer" ? (
                imageError ? (
                  <div className="grid min-h-[420px] place-items-center rounded-lg border border-dashed border-rose-200 bg-rose-50 p-8 text-center text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/20 dark:text-rose-300">
                    {imageError}
                  </div>
                ) : imageUrl ? (
                  <div className="tree-canvas">
                    <img
                      alt={`Phylogenetic tree in ${layout} layout`}
                      className="mx-auto max-h-[720px] w-full object-contain"
                      src={imageUrl}
                    />
                  </div>
                ) : (
                  <div className="grid min-h-[420px] place-items-center rounded-lg bg-slate-50 dark:bg-slate-950/50">
                    <div className="text-center text-sm text-slate-400">
                      <div className="loading-orbit mx-auto h-10 w-10">
                        <span />
                      </div>
                      <p className="mt-4">Rendering tree</p>
                    </div>
                  </div>
                )
              ) : activeTab === "Newick" ? (
                <CodePanel content={data.tree.newick} />
              ) : activeTab === "DOT" ? (
                <CodePanel content={data.tree.dot} />
              ) : (
                <div className="table-shell max-h-[560px] overflow-auto">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Parent</th>
                        <th>Child</th>
                        <th>Branch length</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.tree.edges.map((edge, index) => (
                        <tr key={`${edge.parent}-${edge.child}-${index}`}>
                          <td className="font-mono text-xs">{edge.parent}</td>
                          <td className="font-medium">{edge.child}</td>
                          <td>{Number(edge.branch_length).toFixed(5)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </Tabs>
      </Card>

      <div className="space-y-5 xl:sticky xl:top-24">
        <Card eyebrow="Visualization" title="Tree controls">
          <div className="space-y-5">
            <div>
              <label className="label" htmlFor="tree-layout">
                Layout
              </label>
              <select
                className="input"
                id="tree-layout"
                value={layout}
                onChange={(event) => setLayout(event.target.value)}
              >
                <option value="circular">Circular</option>
                <option value="rectangular">Rectangular</option>
              </select>
            </div>
            <Toggle checked={branch} label="Use branch lengths" onChange={setBranch} />
            <Toggle checked={labels} label="Show taxon labels" onChange={setLabels} />
            <div>
              <div className="mb-2 flex items-center justify-between">
                <label className="label !mb-0" htmlFor="label-size">
                  Label size
                </label>
                <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                  {size}px
                </span>
              </div>
              <input
                className="range"
                id="label-size"
                max={16}
                min={4}
                type="range"
                value={size}
                onChange={(event) => setSize(Number(event.target.value))}
              />
            </div>
          </div>
        </Card>
        <Card eyebrow="Result" title="Tree summary">
          <KeyValueList
            rows={[
              ["Leaves", data.tree.n_leaves],
              ["Internal nodes", data.tree.n_internal],
              ["Branch length", data.tree.total_branch_length],
              ["Method", "Neighbor-Joining"],
            ]}
          />
        </Card>
        <div className="grid grid-cols-3 gap-2">
          <button
            className="btn px-2"
            onClick={() => download("tree.newick", data.tree.newick)}
          >
            Newick
          </button>
          <button
            className="btn px-2"
            disabled={!imageUrl}
            onClick={() => downloadObjectUrl("tree.png", imageUrl)}
          >
            PNG
          </button>
          <button className="btn px-2" onClick={() => download("tree.dot", data.tree.dot)}>
            DOT
          </button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem("tropical-theme");
    if (stored === "dark" || stored === "light") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const [page, setPage] = useState<PageName>("Overview");
  const [data, setData] = useState<PipelineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [error, setError] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [params, setParams] = useState<RunParams>({
    file: null,
    epochs: 200,
    gamma: 0.05,
    lambda_reg: 0.01,
    quadruplet_sample_size: 500,
    min_seq_length: 1,
    alpha: 0.9,
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("tropical-theme", theme);
  }, [theme]);

  useEffect(() => {
    health().then((response) => setApiStatus(response.status)).catch(() => setApiStatus("down"));
  }, []);

  const navigate = (nextPage: PageName) => {
    setPage(nextPage);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const run = async () => {
    setError("");
    setLoading(true);
    try {
      const result = await runPipeline(params);
      setData(result);
      navigate("Overview");
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const numberHandler =
    (key: keyof RunParams) => (event: React.ChangeEvent<HTMLInputElement>) => {
      setParams((current) => ({ ...current, [key]: Number(event.target.value) }));
    };

  const pageMeta = PAGE_META[page];

  return (
    <div className="min-h-screen">
      {loading && <LoadingOverlay />}

      {mobileMenuOpen && (
        <button
          aria-label="Close navigation"
          className="fixed inset-0 z-40 bg-slate-950/55 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      <aside className={`sidebar ${mobileMenuOpen ? "sidebar-open" : ""}`}>
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between px-5 pb-4 pt-5">
            <div className="flex min-w-0 items-center gap-3">
              <DnaMark small />
              <div className="min-w-0">
                <p className="truncate text-sm font-bold tracking-tight text-white">
                  Tropical Phylo
                </p>
                <p className="truncate text-[11px] text-slate-400">Research workspace</p>
              </div>
            </div>
            <button
              aria-label="Close menu"
              className="icon-button border-white/10 text-slate-300 lg:hidden"
              onClick={() => setMobileMenuOpen(false)}
            >
              <Icon name="Close" />
            </button>
          </div>

          <div className="mx-4 rounded-lg border border-white/10 bg-white/[0.045] p-3.5">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                  Workspace
                </p>
                <p className="mt-1 truncate text-xs font-semibold text-slate-200">
                  {data?.dataset.file_name ?? "No analysis loaded"}
                </p>
              </div>
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  data ? "bg-teal-400 shadow-[0_0_14px_#2dd4bf]" : "bg-slate-600"
                }`}
              />
            </div>
          </div>

          <nav className="mt-3 flex-1 overflow-y-auto px-3 pb-4">
            {NAV_GROUPS.map((group) => (
              <div className="mb-4" key={group.label}>
                <p className="mb-1.5 px-3 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-600">
                  {group.label}
                </p>
                {group.items.map((item) => (
                  <button
                    className={`nav-item ${page === item.name ? "nav-item-active" : ""}`}
                    key={item.name}
                    onClick={() => navigate(item.name)}
                  >
                    <Icon name={item.name} />
                    <span className="min-w-0 flex-1 truncate text-left">{item.name}</span>
                    {page === item.name && (
                      <span className="h-1.5 w-1.5 rounded-full bg-teal-300" />
                    )}
                  </button>
                ))}
              </div>
            ))}
          </nav>

          <div className="border-t border-white/[0.07] p-4">
            <button className="sidebar-run" disabled={loading} onClick={run}>
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-white/15">
                <Icon name="Play" />
              </span>
              <span className="min-w-0 text-left">
                <span className="block text-sm font-semibold">Run pipeline</span>
                <span className="block truncate text-[10px] font-medium text-teal-100/70">
                  {params.file ? params.file.name : "Synthetic demo"}
                </span>
              </span>
              <Icon name="Arrow" className="ml-auto h-4 w-4" />
            </button>
            <div className="mt-3 flex items-center justify-between">
              <span className="text-[10px] font-medium text-slate-600">
                v2.0 · Research only
              </span>
              <div className="flex rounded-xl border border-white/[0.07] bg-white/[0.04] p-1">
                <button
                  aria-label="Use light theme"
                  className={`theme-button ${
                    theme === "light" ? "theme-button-active" : ""
                  }`}
                  onClick={() => setTheme("light")}
                >
                  <Icon name="Sun" className="h-4 w-4" />
                </button>
                <button
                  aria-label="Use dark theme"
                  className={`theme-button ${
                    theme === "dark" ? "theme-button-active" : ""
                  }`}
                  onClick={() => setTheme("dark")}
                >
                  <Icon name="Moon" className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </aside>

      <div className="lg:pl-[272px]">
        <header className="app-header">
          <div className="flex min-w-0 items-center gap-3">
            <button
              aria-label="Open navigation"
              className="icon-button lg:hidden"
              onClick={() => setMobileMenuOpen(true)}
            >
              <Icon name="Menu" />
            </button>
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                <span>Workspace</span>
                <span>/</span>
                <span className="truncate text-teal-600 dark:text-teal-400">{page}</span>
              </div>
              <h1 className="mt-1 truncate text-xl font-semibold tracking-tight text-slate-950 dark:text-white sm:text-2xl">
                {page}
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            {data && (
              <div className="hidden items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs shadow-sm dark:border-slate-800 dark:bg-slate-900 md:flex">
                <span className="grid h-6 w-6 place-items-center rounded-lg bg-teal-50 text-teal-700 dark:bg-teal-950 dark:text-teal-300">
                  <Icon name="Check" className="h-4 w-4" />
                </span>
                <span>
                  <b>{data.dataset.n_valid}</b> sequences ready
                </span>
              </div>
            )}
            <StatusDot status={apiStatus} />
          </div>
        </header>

        <main className="app-content">
          <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <p className="max-w-2xl text-sm leading-6 text-slate-500 dark:text-slate-400">
              {pageMeta.description}
            </p>
            {data && page !== "Dataset" && (
              <button className="text-button" onClick={() => navigate("Dataset")}>
                Configure analysis <Icon name="Arrow" className="h-4 w-4" />
              </button>
            )}
          </div>

          {error && <ErrorBanner message={error} onClose={() => setError("")} />}

          {page === "Overview" &&
            (data ? (
              <Overview data={data} navigate={navigate} />
            ) : (
              <EmptyState
                loading={loading}
                onDataset={() => navigate("Dataset")}
                onRun={run}
              />
            ))}
          {page === "Dataset" && (
            <DatasetPage
              data={data}
              loading={loading}
              numberHandler={numberHandler}
              params={params}
              run={run}
              setParams={setParams}
            />
          )}
          {page === "Preprocessing" &&
            (data ? (
              <Preprocessing data={data} />
            ) : (
              <EmptyState loading={loading} onDataset={() => navigate("Dataset")} onRun={run} />
            ))}
          {page === "Distance Matrix" &&
            (data ? (
              <DistancePage data={data} />
            ) : (
              <EmptyState loading={loading} onDataset={() => navigate("Dataset")} onRun={run} />
            ))}
          {page === "Tropical Correction" &&
            (data ? (
              <CorrectionPage data={data} />
            ) : (
              <EmptyState loading={loading} onDataset={() => navigate("Dataset")} onRun={run} />
            ))}
          {page === "Phylogenetic Tree" &&
            (data ? (
              <TreePage data={data} />
            ) : (
              <EmptyState loading={loading} onDataset={() => navigate("Dataset")} onRun={run} />
            ))}
          {page === "Metrics" &&
            (data ? (
              <MetricsPage data={data} />
            ) : (
              <EmptyState loading={loading} onDataset={() => navigate("Dataset")} onRun={run} />
            ))}
          {page === "Downloads" &&
            (data ? (
              <Downloads data={data} />
            ) : (
              <EmptyState loading={loading} onDataset={() => navigate("Dataset")} onRun={run} />
            ))}
          {page === "API Docs" && <ApiDocs />}
          {page === "About" && <About />}
        </main>
      </div>
    </div>
  );
}

function Overview({
  data,
  navigate,
}: {
  data: PipelineResult;
  navigate: (page: PageName) => void;
}) {
  const dataset = data.dataset;
  const improvement = Math.max(0, data.relative_improvement * 100);
  return (
    <div className="space-y-5">
      <section className="result-hero">
        <div className="relative z-10 max-w-2xl">
          <span className="hero-badge">
            <Icon name="Check" className="h-4 w-4" /> Analysis complete
          </span>
          <h2 className="mt-5 text-3xl font-semibold tracking-[-0.035em] text-white sm:text-4xl">
            Your phylogenetic tree is ready.
          </h2>
          <p className="mt-3 max-w-xl text-sm leading-6 text-slate-300">
            Tropical correction reduced the four-point L2 loss by{" "}
            <b className="text-white">{improvement.toFixed(2)}%</b> across{" "}
            {data.metrics_after.n_sampled.toLocaleString()} evaluated quadruplets.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <button
              className="btn btn-light"
              onClick={() => navigate("Phylogenetic Tree")}
            >
              Explore tree <Icon name="Arrow" />
            </button>
            <button
              className="btn btn-ghost-light"
              onClick={() => navigate("Downloads")}
            >
              Export results
            </button>
          </div>
        </div>
        <div className="hero-score">
          <div
            className="hero-score-ring"
            style={
              { "--score": `${Math.min(improvement, 100) * 3.6}deg` } as React.CSSProperties
            }
          >
            <div>
              <strong>{improvement.toFixed(1)}%</strong>
              <span>loss reduction</span>
            </div>
          </div>
        </div>
      </section>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Metric
          icon={<Icon name="Dataset" />}
          label="Valid sequences"
          tone="teal"
          value={dataset.n_valid}
        />
        <Metric
          icon={<Icon name="Distance Matrix" />}
          label="Pairwise distances"
          tone="blue"
          value={data.pairs.toLocaleString()}
        />
        <Metric
          icon={<Icon name="Tropical Correction" />}
          label="Mean violation"
          tone="amber"
          value={fmt(data.metrics_after.mean_violation)}
        />
        <Metric
          icon={<Icon name="Phylogenetic Tree" />}
          label="Tree leaves"
          tone="violet"
          value={data.tree.n_leaves}
        />
      </div>

      <Card
        eyebrow="Workflow"
        right={
          <span className="success-label">
            <Icon name="Check" className="h-3.5 w-3.5" /> Complete
          </span>
        }
        title="Pipeline progress"
      >
        <Stepper steps={STEPS} />
      </Card>

      <div className="grid gap-5 xl:grid-cols-[0.9fr_1.35fr_0.9fr]">
        <Card eyebrow="Input" title="Dataset profile">
          <KeyValueList
            rows={[
              [
                "File",
                <span
                  className="block max-w-[180px] truncate"
                  key="filename"
                  title={dataset.file_name}
                >
                  {dataset.file_name}
                </span>,
              ],
              ["Molecule type", dataset.type],
              ["Sequence length", `${dataset.min_length}–${dataset.max_length} nt`],
              ["Mean length", `${dataset.mean_length} nt`],
              ["GC content", `${dataset.gc_content}%`],
            ]}
          />
        </Card>
        <Card eyebrow="Optimization" title="Four-point compatibility">
          <MetricsTable after={data.metrics_after} before={data.metrics_before} />
        </Card>
        <Card eyebrow="Output" title="Tree profile">
          <KeyValueList
            rows={[
              ["Method", "Neighbor-Joining"],
              ["Leaves", data.tree.n_leaves],
              ["Internal nodes", data.tree.n_internal],
              ["Total branch length", data.tree.total_branch_length],
              ["Relative improvement", fmt(data.relative_improvement, 4)],
            ]}
          />
        </Card>
      </div>
    </div>
  );
}

function DatasetPage({
  data,
  params,
  setParams,
  numberHandler,
  run,
  loading,
}: {
  data: PipelineResult | null;
  params: RunParams;
  setParams: React.Dispatch<React.SetStateAction<RunParams>>;
  numberHandler: (
    key: keyof RunParams,
  ) => (event: React.ChangeEvent<HTMLInputElement>) => void;
  run: () => void;
  loading: boolean;
}) {
  const fields: [keyof RunParams, string, string, number][] = [
    ["epochs", "Epochs", "Maximum optimization iterations", 1],
    ["gamma", "Learning rate γ", "Normalized tropical step scale", 0.001],
    ["lambda_reg", "Regularization λ", "Correction magnitude penalty", 0.001],
    ["quadruplet_sample_size", "Quadruplet sample", "0 evaluates every quadruplet", 1],
    ["min_seq_length", "Minimum length", "Reject shorter cleaned sequences", 1],
    ["alpha", "Distance weight α", "Hamming contribution in [0, 1]", 0.01],
  ];

  return (
    <div className="space-y-5">
      <div className="grid gap-5 xl:grid-cols-[1.15fr_0.85fr]">
        <Card eyebrow="Step 1" title="Choose sequence data">
          <label className="upload-zone">
            <input
              accept=".csv"
              className="sr-only"
              type="file"
              onChange={(event) =>
                setParams((current) => ({
                  ...current,
                  file: event.target.files?.[0] ?? null,
                }))
              }
            />
            <span className="upload-icon">
              <Icon name="Upload" className="h-6 w-6" />
            </span>
            <span className="mt-4 text-base font-semibold text-slate-900 dark:text-white">
              {params.file ? params.file.name : "Drop a viral sequence CSV here"}
            </span>
            <span className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              {params.file
                ? `${(params.file.size / 1024).toFixed(1)} KB · Ready to analyze`
                : "or click to browse your computer"}
            </span>
            <span className="mt-4 rounded-lg bg-slate-100 px-3 py-1.5 font-mono text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
              virus_name, rna_sequence
            </span>
          </label>
          <div className="mt-4 flex flex-col gap-3 sm:flex-row">
            <a className="btn flex-1" href={`${API}/example-csv`}>
              <Icon name="Downloads" /> Download example
            </a>
            <button
              className="btn flex-1"
              onClick={() => setParams((current) => ({ ...current, file: null }))}
            >
              Use synthetic demo
            </button>
          </div>
          <div className="notice mt-4">
            <span className="notice-icon">i</span>
            <p>
              The bundled example is synthetic demonstration data only. It must not
              be interpreted as a biological result.
            </p>
          </div>
        </Card>

        <Card eyebrow="Step 2" title="Configure analysis">
          <div className="grid gap-4 sm:grid-cols-2">
            {fields.map(([key, label, hint, step]) => (
              <div key={key}>
                <label className="label" htmlFor={`param-${key}`}>
                  {label}
                </label>
                <input
                  className="input"
                  id={`param-${key}`}
                  max={key === "alpha" ? 1 : undefined}
                  min={key === "alpha" ? 0 : undefined}
                  step={step}
                  type="number"
                  value={params[key] as number}
                  onChange={numberHandler(key)}
                />
                <p className="mt-1.5 text-[11px] leading-4 text-slate-400">{hint}</p>
              </div>
            ))}
          </div>
          <button
            className="btn btn-primary mt-6 w-full py-3"
            disabled={loading}
            onClick={run}
          >
            <Icon name="Play" /> Run complete pipeline
          </button>
        </Card>
      </div>

      <Card
        eyebrow="Data review"
        right={data && <span className="neutral-label">{data.dataset.total_rows} rows</span>}
        title="Dataset preview"
      >
        {data ? (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Virus name</th>
                  <th>Sequence preview</th>
                </tr>
              </thead>
              <tbody>
                {data.dataset.preview.map((row, index) => (
                  <tr key={`${row.virus_name}-${index}`}>
                    <td className="font-medium text-slate-900 dark:text-white">
                      {row.virus_name}
                    </td>
                    <td className="max-w-xl truncate font-mono text-xs text-slate-500">
                      {row.rna_sequence}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="grid min-h-44 place-items-center rounded-lg border border-dashed border-slate-200 bg-slate-50/60 text-center dark:border-slate-800 dark:bg-slate-950/30">
            <div>
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                No dataset loaded
              </p>
              <p className="mt-1 text-xs text-slate-400">
                A sequence preview will appear after the first pipeline run.
              </p>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

function Preprocessing({ data }: { data: PipelineResult }) {
  const dataset = data.dataset;
  const retention = dataset.n_total ? (dataset.n_valid / dataset.n_total) * 100 : 0;
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-3">
        <Metric
          icon={<Icon name="Dataset" />}
          label="Input rows"
          tone="blue"
          value={dataset.n_total}
        />
        <Metric
          icon={<Icon name="Check" />}
          label="Valid sequences"
          tone="teal"
          value={dataset.n_valid}
        />
        <Metric
          icon={<Icon name="Preprocessing" />}
          label="Retention rate"
          tone="violet"
          value={`${retention.toFixed(1)}%`}
        />
      </div>
      <div className="grid gap-5 xl:grid-cols-[0.8fr_1.2fr]">
        <Card eyebrow="Quality control" title="Cleaning protocol">
          <div className="space-y-3">
            {[
              ["01", "Normalize case", "Convert all sequence symbols to uppercase."],
              ["02", "Convert RNA", "Replace uracil (U) with thymine (T)."],
              ["03", "Filter alphabet", "Keep canonical A, C, G, and T symbols only."],
              ["04", "Validate records", "Reject short, invalid, empty, and duplicate rows."],
            ].map(([number, title, detail]) => (
              <div className="process-row" key={number}>
                <span>{number}</span>
                <div>
                  <p>{title}</p>
                  <small>{detail}</small>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5 border-t border-slate-100 pt-5 dark:border-slate-800">
            <KeyValueList
              rows={[
                ["Length range", `${dataset.min_length}–${dataset.max_length} nt`],
                ["Mean length", `${dataset.mean_length} nt`],
                ["GC content", `${dataset.gc_content}%`],
              ]}
            />
          </div>
        </Card>
        <Card eyebrow="Distribution" title="Cleaned sequence lengths">
          <Histogram values={data.sequence_lengths} />
        </Card>
      </div>
    </div>
  );
}

function offDiagonal(matrix: number[][]): number[] {
  const values: number[] = [];
  matrix.forEach((row, rowIndex) =>
    row.forEach((value, columnIndex) => {
      if (columnIndex > rowIndex) values.push(value);
    }),
  );
  return values;
}

function DistancePage({ data }: { data: PipelineResult }) {
  const distances = useMemo(
    () => offDiagonal(data.distance_matrix),
    [data.distance_matrix],
  );
  const mean =
    distances.reduce((sum, value) => sum + value, 0) / (distances.length || 1);
  const min = Math.min(...distances);
  const max = Math.max(...distances);
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-3">
        <Metric
          icon={<Icon name="Distance Matrix" />}
          label="Sequence pairs"
          tone="blue"
          value={distances.length.toLocaleString()}
        />
        <Metric
          icon={<Icon name="Metrics" />}
          label="Mean distance"
          tone="teal"
          value={fmt(mean, 4)}
        />
        <Metric
          icon={<Icon name="Arrow" />}
          label="Observed range"
          tone="amber"
          value={`${fmt(min, 3)}–${fmt(max, 3)}`}
        />
      </div>
      <Card
        eyebrow="Distance geometry"
        noPadding
        right={
          <span className="success-label">
            <Icon name="Check" className="h-3.5 w-3.5" /> Matrix valid
          </span>
        }
        title="Pairwise distance matrix"
      >
        <Tabs tabs={["Heatmap", "Statistics", "Distribution"]}>
          {(activeTab) => (
            <div className="p-4 sm:p-6">
              {activeTab === "Heatmap" ? (
                <Heatmap labels={data.labels} matrix={data.distance_matrix} />
              ) : activeTab === "Statistics" ? (
                <div className="mx-auto max-w-2xl">
                  <KeyValueList
                    rows={[
                      ["Matrix shape", `${data.labels.length} × ${data.labels.length}`],
                      ["Unique pairs", distances.length],
                      ["Minimum distance", fmt(min, 6)],
                      ["Mean distance", fmt(mean, 6)],
                      ["Maximum distance", fmt(max, 6)],
                      [
                        "Matrix invariants",
                        "Symmetric · zero diagonal · non-negative",
                      ],
                    ]}
                  />
                </div>
              ) : (
                <Histogram color="#0f766e" values={distances} />
              )}
            </div>
          )}
        </Tabs>
      </Card>
    </div>
  );
}

function CorrectionPage({ data }: { data: PipelineResult }) {
  const improvement = data.relative_improvement * 100;
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-3">
        <Metric
          icon={<Icon name="Tropical Correction" />}
          label="Relative improvement"
          tone="teal"
          value={`${improvement.toFixed(2)}%`}
        />
        <Metric
          icon={<Icon name="Metrics" />}
          label="L2 loss before"
          tone="amber"
          value={fmt(data.metrics_before.l2_loss, 4)}
        />
        <Metric
          icon={<Icon name="Check" />}
          label="L2 loss after"
          tone="blue"
          value={fmt(data.metrics_after.l2_loss, 4)}
        />
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
        <Card eyebrow="Convergence" title="Optimization history">
          <LossChart data={data.history} />
        </Card>
        <Card eyebrow="Configuration" title="Training parameters">
          <KeyValueList
            rows={Object.entries(data.params).map(([key, value]) => [
              key.replace(/_/g, " "),
              String(value),
            ])}
          />
          <div className="mt-5 rounded-lg bg-teal-50 p-4 dark:bg-teal-950/30">
            <p className="text-xs font-bold uppercase tracking-[0.15em] text-teal-700 dark:text-teal-300">
              Best iterate retained
            </p>
            <p className="mt-2 text-sm leading-6 text-teal-900/75 dark:text-teal-100/70">
              The optimizer returns the lowest-loss iterate, ensuring the corrected
              matrix is not worse than the input matrix.
            </p>
          </div>
        </Card>
      </div>
      <Card eyebrow="Matrix inspection" noPadding title="Correction result">
        <Tabs tabs={["Corrected matrix X", "Correction ω"]}>
          {(activeTab) => (
            <div className="p-4 sm:p-6">
              <Heatmap
                labels={data.labels}
                matrix={
                  activeTab === "Correction ω" ? data.omega : data.corrected_matrix
                }
              />
            </div>
          )}
        </Tabs>
      </Card>
    </div>
  );
}

function MetricsPage({ data }: { data: PipelineResult }) {
  const chartKeys = [
    "mean_violation",
    "max_violation",
    "l2_loss",
  ] as (keyof Metrics)[];
  return (
    <div className="space-y-5">
      <section className="metric-summary">
        <div>
          <span className="eyebrow">Scientific result</span>
          <h2 className="mt-3 text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">
            Four-point violations decreased after correction.
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500 dark:text-slate-400">
            This indicates improved mathematical compatibility with an additive tree
            metric. It does not by itself prove biological correctness.
          </p>
        </div>
        <div className="metric-summary-score">
          <strong>{(data.relative_improvement * 100).toFixed(1)}%</strong>
          <span>relative L2 improvement</span>
        </div>
      </section>
      <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
        <Card eyebrow="Detailed comparison" title="Before and after">
          <MetricsTable after={data.metrics_after} before={data.metrics_before} />
        </Card>
        <Card eyebrow="Visual comparison" title="Violation metrics">
          <BarCompare
            after={chartKeys.map((key) => data.metrics_after[key] as number)}
            before={chartKeys.map((key) => data.metrics_before[key] as number)}
            keys={["Mean", "Maximum", "L2 loss"]}
          />
        </Card>
      </div>
    </div>
  );
}

function Downloads({ data }: { data: PipelineResult }) {
  const files: {
    name: string;
    type: string;
    description: string;
    content: () => string;
  }[] = [
    {
      name: "distance_matrix.csv",
      type: "CSV",
      description: "Initial pairwise distance matrix D",
      content: () => matrixToCsv(data.distance_matrix, data.labels),
    },
    {
      name: "corrected_matrix.csv",
      type: "CSV",
      description: "Optimized distance matrix X",
      content: () => matrixToCsv(data.corrected_matrix, data.labels),
    },
    {
      name: "omega.csv",
      type: "CSV",
      description: "Learned tropical correction ω",
      content: () => matrixToCsv(data.omega, data.labels),
    },
    {
      name: "metrics.json",
      type: "JSON",
      description: "Before/after metrics and improvement",
      content: () =>
        JSON.stringify(
          {
            before: data.metrics_before,
            after: data.metrics_after,
            relative_improvement: data.relative_improvement,
          },
          null,
          2,
        ),
    },
    {
      name: "history.csv",
      type: "CSV",
      description: "Optimizer convergence history",
      content: () =>
        `epoch,loss,grad_tr_norm,eta\n${data.history
          .map((row) => `${row.epoch},${row.loss},${row.grad_tr_norm},${row.eta}`)
          .join("\n")}`,
    },
    {
      name: "tree.newick",
      type: "NEWICK",
      description: "Phylogenetic tree in Newick format",
      content: () => data.tree.newick,
    },
    {
      name: "tree_edges.csv",
      type: "CSV",
      description: "Parent/child tree edge list",
      content: () =>
        `parent,child,branch_length\n${data.tree.edges
          .map((edge) => `${edge.parent},${edge.child},${edge.branch_length}`)
          .join("\n")}`,
    },
    {
      name: "tree.dot",
      type: "DOT",
      description: "Graphviz tree representation",
      content: () => data.tree.dot,
    },
  ];
  return (
    <div className="space-y-5">
      <section className="download-hero">
        <div>
          <span className="eyebrow !text-teal-200">Analysis package</span>
          <h2 className="mt-3 text-2xl font-semibold text-white">
            Export reproducible scientific artifacts
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Download matrices, optimizer history, metrics, and interoperable tree
            formats from this analysis.
          </p>
        </div>
        <div className="rounded-lg border border-white/10 bg-white/5 px-5 py-4 text-right">
          <strong className="block text-2xl text-white">{files.length}</strong>
          <span className="text-xs text-slate-400">files available</span>
        </div>
      </section>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {files.map((file) => (
          <button
            className="download-card"
            key={file.name}
            onClick={() => download(file.name, file.content())}
          >
            <span className="file-type">{file.type}</span>
            <span className="min-w-0 flex-1 text-left">
              <span className="block truncate text-sm font-semibold text-slate-900 dark:text-white">
                {file.name}
              </span>
              <span className="mt-1 block text-xs leading-5 text-slate-500 dark:text-slate-400">
                {file.description}
              </span>
            </span>
            <span className="download-arrow">
              <Icon name="Downloads" className="h-4 w-4" />
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function ApiDocs() {
  const endpoints: [string, string, string][] = [
    ["GET", "/health", "Service liveness and model artifact state"],
    ["POST", "/pipeline", "Complete CSV-to-tree dashboard workflow"],
    ["POST", "/distance-matrix", "Clean sequences and construct D"],
    ["POST", "/four-point-score", "Evaluate tropical four-point metrics"],
    ["POST", "/correct-distance-matrix", "Learn ω and return corrected X"],
    ["POST", "/tropical-correction", "Compatibility alias for correction"],
    ["POST", "/predict-from-sequences", "Run the complete scientific pipeline"],
    ["POST", "/phylogenetic-tree", "Reconstruct a Neighbor-Joining tree"],
    ["POST", "/tree-image", "Render a PNG tree visualization"],
  ];
  return (
    <div className="space-y-5">
      <section className="api-hero">
        <div>
          <span className="eyebrow">Developer platform</span>
          <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-950 dark:text-white">
            Integrate the phylogenetic pipeline through REST.
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500 dark:text-slate-400">
            FastAPI validates requests with Pydantic and converts NumPy outputs at the
            JSON boundary.
          </p>
        </div>
        <a
          className="btn btn-primary"
          href={`${API}/docs`}
          rel="noreferrer"
          target="_blank"
        >
          Open Swagger UI <Icon name="Arrow" />
        </a>
      </section>
      <Card eyebrow="Connection" title="API base URL">
        <div className="flex flex-col gap-3 sm:flex-row">
          <input className="input font-mono" readOnly value={location.origin} />
          <button
            className="btn"
            onClick={() => navigator.clipboard.writeText(location.origin)}
          >
            Copy URL
          </button>
        </div>
      </Card>
      <Card eyebrow="Reference" title="Available endpoints">
        <div className="divide-y divide-slate-100 dark:divide-slate-800">
          {endpoints.map(([method, path, description]) => (
            <div
              className="flex flex-col gap-2 py-4 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:gap-4"
              key={`${method}-${path}`}
            >
              <span
                className={`method-badge ${
                  method === "GET" ? "method-get" : "method-post"
                }`}
              >
                {method}
              </span>
              <code className="min-w-[220px] text-sm font-semibold text-slate-800 dark:text-slate-100">
                {path}
              </code>
              <span className="text-sm text-slate-500 dark:text-slate-400">
                {description}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function About() {
  const technologies = [
    "Python",
    "FastAPI",
    "React",
    "Vite",
    "TypeScript",
    "Tailwind CSS",
    "NumPy",
    "pandas",
    "Biopython",
    "Docker",
  ];
  return (
    <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
      <Card className="overflow-hidden" noPadding>
        <div className="about-banner">
          <DnaMark />
          <span className="hero-badge mt-6">Computational biology</span>
          <h2 className="mt-4 max-w-xl text-3xl font-semibold tracking-tight text-white">
            Tropical Virus PhyloTree MLOps
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-300">
            A reproducible research application that learns a tropical correction X =
            D + ω, reduces four-point violations, and reconstructs a Neighbor-Joining
            phylogenetic tree.
          </p>
        </div>
        <div className="p-5 sm:p-7">
          <h3 className="text-base font-semibold text-slate-950 dark:text-white">
            Scientific boundary
          </h3>
          <p className="mt-2 text-sm leading-7 text-slate-500 dark:text-slate-400">
            Reducing tropical four-point violations improves mathematical compatibility
            with an additive tree metric. It does not establish that the inferred
            topology is the biologically correct evolutionary history.
          </p>
          <div className="notice mt-5">
            <span className="notice-icon">!</span>
            <p>
              This software is a non-diagnostic research and educational tool. It must
              not be used to diagnose infection or disease.
            </p>
          </div>
        </div>
      </Card>
      <div className="space-y-5">
        <Card eyebrow="Architecture" title="Scientific core">
          <div className="space-y-3 text-sm leading-6 text-slate-500 dark:text-slate-400">
            <p>
              All distance, four-point, optimization, and Neighbor-Joining logic lives
              in the pure Python <code>src/</code> package.
            </p>
            <p>
              FastAPI, React, Streamlit, and batch commands remain orchestration layers.
            </p>
          </div>
        </Card>
        <Card eyebrow="Technology" title="Application stack">
          <div className="flex flex-wrap gap-2">
            {technologies.map((technology) => (
              <span className="tech-pill" key={technology}>
                {technology}
              </span>
            ))}
          </div>
        </Card>
        <Card eyebrow="Data policy" title="Real data by default">
          <p className="text-sm leading-6 text-slate-500 dark:text-slate-400">
            The committed sample is explicitly synthetic demonstration data. Scientific
            results require sourced, real viral sequences with appropriate provenance.
          </p>
        </Card>
      </div>
    </div>
  );
}
