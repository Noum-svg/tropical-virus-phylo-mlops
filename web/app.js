"use strict";

const PAGES = [
  ["Overview", "M3 12h18M3 6h18M3 18h18"],
  ["Dataset", "M4 7h16M4 12h16M4 17h10"],
  ["Preprocessing", "M3 6h18v4H3zM3 14h18v4H3z"],
  ["Distance Matrix", "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z"],
  ["Tropical Correction", "M4 18l5-7 4 4 7-9"],
  ["Phylogenetic Tree", "M12 3v6M12 9l-6 6M12 9l6 6M6 21v-3M18 21v-3"],
  ["Metrics", "M4 20V8M10 20V4M16 20v-9M22 20H2"],
  ["Downloads", "M12 3v12M7 10l5 5 5-5M5 21h14"],
  ["API Docs", "M8 4l-4 8 4 8M16 4l4 8-4 8"],
  ["About", "M12 8h.01M11 12h1v4h1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"],
];
const STEPS = ["Upload", "Preprocess", "Distances", "Correction", "Tree", "Completed"];

let STATE = { data: null, file: null };
const $ = (s) => document.querySelector(s);
const el = (id) => document.getElementById(id);

function api(path, opts) {
  return fetch(path, opts).then((r) => {
    if (!r.ok) return r.json().then((e) => Promise.reject(e.detail || r.statusText), () => Promise.reject(r.statusText));
    return r;
  });
}
function toast(msg) {
  const t = el("toast"); t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2600);
}
function themed(layout) {
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  return Object.assign({ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: dark ? "#cbd5e1" : "#334155", size: 11 },
    margin: { t: 18, r: 12, b: 40, l: 48 } }, layout);
}
const PCONF = { responsive: true, displayModeBar: false };

/* ---- navigation ---- */
function buildNav() {
  el("nav").innerHTML = PAGES.map(([name, d]) =>
    `<div class="nav-item" data-nav="${name}">
       <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg>
       ${name}</div>`).join("");
  document.querySelectorAll("[data-nav]").forEach((n) =>
    n.addEventListener("click", () => show(n.dataset.nav)));
}
function show(name) {
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.dataset.page === name));
  document.querySelectorAll("[data-nav]").forEach((n) => n.classList.toggle("active", n.dataset.nav === name));
  el("pageTitle").textContent = name;
  if (name === "Phylogenetic Tree" && STATE.data) updateTree();
}

/* ---- tabs ---- */
function wireTabs() {
  document.querySelectorAll(".tabs").forEach((tabs) => {
    tabs.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => {
        tabs.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        const card = tabs.closest(".card");
        card.querySelectorAll(".tab-pane").forEach((p) =>
          p.classList.toggle("active", p.dataset.pane === b.dataset.tab));
        // re-flow plotly when a hidden chart becomes visible
        card.querySelectorAll(".tab-pane.active [id]").forEach((d) => window.Plotly && Plotly.Plots.resize(d));
      }));
  });
}

/* ---- theme ---- */
function wireTheme() {
  document.querySelectorAll("[data-theme-btn]").forEach((b) =>
    b.addEventListener("click", () => {
      document.documentElement.setAttribute("data-theme", b.dataset.themeBtn);
      document.querySelectorAll("[data-theme-btn]").forEach((x) => x.classList.toggle("active", x === b));
      if (STATE.data) renderAll(STATE.data);
    }));
}

/* ---- health ---- */
function checkHealth() {
  api("/health").then((r) => r.json()).then((h) => {
    el("apiBadge").textContent = "API: " + h.status + (h.model_loaded ? " · model" : "");
  }).catch(() => { el("apiBadge").textContent = "API: down"; el("apiBadge").style.background = "#fee2e2"; el("apiBadge").style.color = "#dc2626"; });
}

/* ---- upload + run ---- */
function wireUpload() {
  const dz = el("dropzone"), fi = el("fileInput");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
  fi.addEventListener("change", () => fi.files[0] && setFile(fi.files[0]));
  el("useDemo").addEventListener("click", () => { STATE.file = null; el("fileChip").innerHTML = `<div class="file-chip"><span>📄 synthetic demo dataset</span><span class="badge">demo</span></div>`; toast("Demo selected — click Run Pipeline"); });
  el("runMain").addEventListener("click", run);
  el("runTop").addEventListener("click", run);
}
function setFile(f) {
  STATE.file = f;
  el("fileChip").innerHTML = `<div class="file-chip"><span>📄 ${f.name}</span><span class="badge">${(f.size / 1024).toFixed(0)} KB</span></div>`;
}
function run() {
  const fd = new FormData();
  if (STATE.file) fd.append("file", STATE.file); else fd.append("use_demo", "true");
  fd.append("epochs", el("p_epochs").value);
  fd.append("gamma", el("p_gamma").value);
  fd.append("lambda_reg", el("p_lambda").value);
  fd.append("quadruplet_sample_size", el("p_sample").value);
  fd.append("min_seq_length", el("p_minlen").value);
  fd.append("alpha", el("p_alpha").value);
  const btns = [el("runMain"), el("runTop")];
  btns.forEach((b) => { b.disabled = true; b.innerHTML = '<span class="spin"></span> Running…'; });
  api("/pipeline", { method: "POST", body: fd }).then((r) => r.json()).then((data) => {
    STATE.data = data; renderAll(data); show("Overview"); toast("Pipeline complete ✓");
  }).catch((e) => toast("Error: " + e)).finally(() =>
    btns.forEach((b, i) => { b.disabled = false; b.textContent = "Run Pipeline"; }));
}

/* ---- rendering ---- */
function matCsv(mat, labels) {
  let out = "," + labels.join(",") + "\n";
  mat.forEach((row, i) => { out += labels[i] + "," + row.map((v) => v.toFixed(6)).join(",") + "\n"; });
  return out;
}
function heat(divId, mat, labels, scale) {
  const showTicks = labels.length <= 35;
  Plotly.newPlot(el(divId), [{
    z: mat, x: labels, y: labels, type: "heatmap", colorscale: scale || "RdBu",
    reversescale: !!(scale === undefined), zsmooth: false, colorbar: { thickness: 12 },
  }], themed({ xaxis: { showticklabels: showTicks, tickfont: { size: 8 } }, yaxis: { showticklabels: showTicks, tickfont: { size: 8 }, autorange: "reversed" } }), PCONF);
}
function metricRows(b, a) {
  const keys = [["mean_violation", "Mean Violation"], ["median_violation", "Median Violation"], ["max_violation", "Max Violation"], ["percent_exact", "% Exact Quadruplets"], ["l2_loss", "L2 Loss"]];
  return keys.map(([k, lbl]) => {
    const bv = b[k], av = a[k];
    let imp;
    if (k === "percent_exact") imp = `<span class="up">+${(av - bv).toFixed(2)} pp ↑</span>`;
    else { const pct = bv ? ((bv - av) / bv * 100) : 0; imp = `<span class="${pct >= 0 ? "down" : "neg"}">${pct.toFixed(2)}% ${pct >= 0 ? "↓" : "↑"}</span>`; }
    return `<tr><td>${lbl}</td><td>${bv.toFixed(3)}</td><td>${av.toFixed(3)}</td><td>${imp}</td></tr>`;
  }).join("");
}
function renderAll(d) {
  el("ovEmpty").style.display = "none"; el("ovBody").style.display = "block";
  const ds = d.dataset;
  // metric cards
  el("metricCards").innerHTML = [
    ["", "📊", ds.n_total, "Total sequences"],
    ["green", "✔", ds.n_valid, "Valid sequences"],
    ["purple", "▦", d.pairs, "Pairs n(n-1)/2"],
    ["amber", "△", (d.metrics_after.mean_violation).toFixed(3), "Mean Violation (after)"],
  ].map(([c, ic, v, l]) => `<div class="card metric ${c}"><div class="ic">${ic}</div><div><div class="val">${v}</div><div class="lbl">${l}</div></div></div>`).join("");
  // stepper
  el("stepper").innerHTML = STEPS.map((s) => `<div class="step done"><div class="dot">✓</div><div class="name">${s}</div></div>`).join("");
  // summaries
  el("dsSummary").innerHTML = `
    <tr><td>File name</td><td>${ds.file_name}</td></tr>
    <tr><td>Type</td><td>${ds.type}</td></tr>
    <tr><td>Min / Max length</td><td>${ds.min_length} / ${ds.max_length}</td></tr>
    <tr><td>Mean length</td><td>${ds.mean_length}</td></tr>
    <tr><td>GC content</td><td>${ds.gc_content}%</td></tr>`;
  el("ovMetrics").innerHTML = `<tr><th>Metric</th><th>Before</th><th>After</th><th>Improvement</th></tr>` + metricRows(d.metrics_before, d.metrics_after);
  el("treeSummary").innerHTML = `
    <tr><td>Method</td><td>Neighbor-Joining</td></tr>
    <tr><td>Leaves</td><td>${d.tree.n_leaves}</td></tr>
    <tr><td>Internal nodes</td><td>${d.tree.n_internal}</td></tr>
    <tr><td>Total branch length</td><td>${d.tree.total_branch_length}</td></tr>
    <tr><td>Relative improvement</td><td>${d.relative_improvement.toFixed(4)}</td></tr>`;
  // dataset preview
  el("rowCount").textContent = `· ${ds.total_rows} rows`;
  el("previewTbl").innerHTML = `<thead><tr><th>virus_name</th><th>rna_sequence</th></tr></thead><tbody>` +
    ds.preview.map((p) => `<tr><td>${p.virus_name}</td><td class="mono" style="border:0;padding:4px 10px;background:none;">${p.rna_sequence}</td></tr>`).join("") + `</tbody>`;
  // preprocessing
  el("prepSummary").innerHTML = `
    <tr><td>Total rows</td><td>${ds.n_total}</td></tr>
    <tr><td>Valid after cleaning</td><td>${ds.n_valid}</td></tr>
    <tr><td>Min / Mean / Max length</td><td>${ds.min_length} / ${ds.mean_length} / ${ds.max_length}</td></tr>
    <tr><td>GC content</td><td>${ds.gc_content}%</td></tr>`;
  Plotly.newPlot(el("lenDist"), [{ x: d.sequence_lengths, type: "histogram", marker: { color: "#2563eb" } }], themed({ xaxis: { title: "length" }, yaxis: { title: "count" } }), PCONF);
  // distance matrix
  el("dmBadge").textContent = "Computed";
  heat("dmHeat", d.distance_matrix, d.labels);
  const off = []; d.distance_matrix.forEach((r, i) => r.forEach((v, j) => { if (j > i) off.push(v); }));
  const mean = off.reduce((a, b) => a + b, 0) / (off.length || 1);
  el("dmStats").innerHTML = `<tr><td>Pairs</td><td>${off.length}</td></tr><tr><td>Min</td><td>${Math.min(...off).toFixed(4)}</td></tr><tr><td>Mean</td><td>${mean.toFixed(4)}</td></tr><tr><td>Max</td><td>${Math.max(...off).toFixed(4)}</td></tr>`;
  Plotly.newPlot(el("dmDist"), [{ x: off, type: "histogram", marker: { color: "#0891b2" } }], themed({ xaxis: { title: "distance" }, yaxis: { title: "count" } }), PCONF);
  // correction
  el("tcBadge").textContent = "Completed";
  el("riVal").innerHTML = `Relative improvement: <span class="${d.relative_improvement >= 0 ? "up" : "neg"}">${d.relative_improvement.toFixed(4)}</span>`;
  el("trainParams").innerHTML = Object.entries(d.params).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  Plotly.newPlot(el("lossChart"), [{ x: d.history.map((h) => h.epoch), y: d.history.map((h) => h.loss), type: "scatter", mode: "lines", line: { color: "#2563eb" }, name: "Loss" }], themed({ xaxis: { title: "Iteration" }, yaxis: { title: "Loss", type: "log" } }), PCONF);
  heat("xHeat", d.corrected_matrix, d.labels);
  heat("omegaHeat", d.omega, d.labels, "RdBu");
  // metrics page
  el("ptBadge").textContent = "Completed";
  el("metricsFull").innerHTML = `<tr><th>Metric</th><th>Before</th><th>After</th><th>Improvement</th></tr>` + metricRows(d.metrics_before, d.metrics_after);
  const mk = ["mean_violation", "max_violation", "l2_loss"];
  Plotly.newPlot(el("metricsChart"), [
    { x: mk, y: mk.map((k) => d.metrics_before[k]), type: "bar", name: "before", marker: { color: "#dc2626" } },
    { x: mk, y: mk.map((k) => d.metrics_after[k]), type: "bar", name: "after", marker: { color: "#16a34a" } },
  ], themed({ barmode: "group" }), PCONF);
  // tree text + downloads
  el("newickBox").textContent = d.tree.newick;
  el("dotBox").textContent = d.tree.dot;
  el("edgesTbl").innerHTML = `<thead><tr><th>parent</th><th>child</th><th>branch_length</th></tr></thead><tbody>` +
    d.tree.edges.map((e) => `<tr><td>${e.parent}</td><td>${e.child}</td><td>${(+e.branch_length).toFixed(4)}</td></tr>`).join("") + `</tbody>`;
  buildDownloads(d);
  updateTree();
}

/* ---- tree image ---- */
function updateTree() {
  if (!STATE.data) return;
  const body = JSON.stringify({
    matrix: STATE.data.corrected_matrix, labels: STATE.data.labels,
    layout: el("treeLayout").value, show_labels: el("treeLabels").checked,
    label_size: +el("treeLabelSize").value, use_branch_length: el("treeBranch").checked,
  });
  api("/tree-image", { method: "POST", headers: { "Content-Type": "application/json" }, body })
    .then((r) => r.blob()).then((b) => { el("treeImg").src = URL.createObjectURL(b); })
    .catch((e) => toast("Tree error: " + e));
}
function wireTreeControls() {
  ["treeLayout", "treeBranch", "treeLabels"].forEach((id) => el(id).addEventListener("change", updateTree));
  el("treeLabelSize").addEventListener("input", () => { el("lsVal").textContent = el("treeLabelSize").value; });
  el("treeLabelSize").addEventListener("change", updateTree);
  el("dlNewick").addEventListener("click", () => dl("tree.newick", STATE.data.tree.newick));
  el("dlDot").addEventListener("click", () => dl("tree.dot", STATE.data.tree.dot));
  el("dlPng").addEventListener("click", () => { const a = document.createElement("a"); a.href = el("treeImg").src; a.download = "tree.png"; a.click(); });
}

/* ---- downloads ---- */
function dl(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: type || "text/plain" }));
  a.download = name; a.click();
}
function buildDownloads(d) {
  const items = [
    ["distance_matrix.csv", () => matCsv(d.distance_matrix, d.labels)],
    ["corrected_matrix.csv", () => matCsv(d.corrected_matrix, d.labels)],
    ["omega.csv", () => matCsv(d.omega, d.labels)],
    ["metrics.json", () => JSON.stringify({ before: d.metrics_before, after: d.metrics_after, relative_improvement: d.relative_improvement }, null, 2)],
    ["history.csv", () => "epoch,loss,grad_tr_norm,eta\n" + d.history.map((h) => `${h.epoch},${h.loss},${h.grad_tr_norm},${h.eta}`).join("\n")],
    ["tree.newick", () => d.tree.newick],
    ["tree_edges.csv", () => "parent,child,branch_length\n" + d.tree.edges.map((e) => `${e.parent},${e.child},${e.branch_length}`).join("\n")],
    ["tree.dot", () => d.tree.dot],
  ];
  el("dlGrid").innerHTML = items.map(([n], i) => `<button class="btn" data-dl="${i}">⬇ ${n}</button>`).join("");
  el("dlGrid").querySelectorAll("[data-dl]").forEach((b) => b.addEventListener("click", () => { const [n, fn] = items[+b.dataset.dl]; dl(n, fn()); }));
}

/* ---- api docs ---- */
function buildApiDocs() {
  el("baseUrl").value = location.origin;
  el("docsLink").href = location.origin + "/docs";
  const eps = [
    ["GET", "/health", "Health check"],
    ["POST", "/pipeline", "Full pipeline from a CSV upload"],
    ["POST", "/distance-matrix", "Sequences → D"],
    ["POST", "/four-point-score", "Matrix → tropical metrics"],
    ["POST", "/correct-distance-matrix", "Matrix → ω, X, metrics"],
    ["POST", "/phylogenetic-tree", "Matrix → tree"],
    ["POST", "/predict-from-sequences", "Sequences → full result"],
    ["POST", "/tree-image", "Matrix → tree PNG"],
  ];
  el("endpoints").innerHTML = eps.map(([m, p, desc]) =>
    `<div class="endpoint"><span class="method ${m.toLowerCase()}">${m}</span><b class="mono" style="background:none;border:0;padding:0;">${p}</b><span class="muted">${desc}</span></div>`).join("");
}

/* ---- init ---- */
buildNav(); wireTabs(); wireTheme(); wireUpload(); wireTreeControls(); buildApiDocs(); checkHealth();
