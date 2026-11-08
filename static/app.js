const $ = (s, r = document) => r.querySelector(s);
const state = { features: [], samples: [], sampleIdx: 0, metrics: null, task: "behaviour" };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(typeof e.detail === "string" ? e.detail : r.statusText);
  }
  return r.json();
}
const fmtNum = (x) => (x === null || x === undefined ? "" : Number(x.toPrecision(6)).toString());
const pct = (x) => (x * 100).toFixed(1) + "%";

/* ---------- form ---------- */
function buildForm() {
  const groups = {};
  state.features.forEach((f) => (groups[f.group] ||= []).push(f));
  $("#fields").innerHTML = Object.entries(groups).map(([g, fs]) => `
    <fieldset><legend>${g}</legend><div class="grid">
      ${fs.map((f) => `
        <label>${f.name}${f.unit ? ` <span class="unit">(${f.unit})</span>` : ""}
          <input type="number" step="any" name="${f.name}" inputmode="decimal"
                 placeholder="${fmtNum(f.default)}"
                 title="${f.min !== null ? `Training range: ${fmtNum(f.min)} to ${fmtNum(f.max)}` : ""}">
        </label>`).join("")}
    </div></fieldset>`).join("");
}

function setValues(values) {
  document.querySelectorAll("#fields input").forEach((i) => {
    i.value = values && values[i.name] !== undefined ? fmtNum(values[i.name]) : "";
    i.classList.remove("invalid");
  });
}

$("#btn-clear").onclick = () => { setValues(null); $("#form-msg").textContent = ""; };
$("#btn-example").onclick = () => {
  if (!state.samples.length) { $("#form-msg").textContent = "No examples found. Run prepare_deploy.py first."; return; }
  const s = state.samples[state.sampleIdx++ % state.samples.length];
  setValues(s.values);
  $("#form-msg").textContent = `Example loaded (true labels: ${s.behaviour} / ${s.circuit}).`;
};

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const values = {};
  let bad = false;
  document.querySelectorAll("#fields input").forEach((i) => {
    i.classList.remove("invalid");
    if (i.value.trim() === "") return;
    const n = Number(i.value);
    if (!Number.isFinite(n)) { i.classList.add("invalid"); bad = true; } else values[i.name] = n;
  });
  const msg = $("#form-msg");
  if (bad) { msg.textContent = "Fix the highlighted fields."; return; }
  msg.textContent = "";
  const btn = $("#btn-predict");
  btn.disabled = true; btn.textContent = "Classifying…";
  try {
    const out = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    showResult(out);
  } catch (err) {
    msg.textContent = "Prediction failed: " + err.message;
  } finally {
    btn.disabled = false; btn.textContent = "Classify load";
  }
});

function probBars(probs, label) {
  if (!probs) return "";
  return Object.entries(probs).sort((a, b) => b[1] - a[1]).map(([c, p]) => `
    <div class="bar-row ${c === label ? "top-class" : ""}"><span>${c}</span>
      <div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div><span>${pct(p)}</span></div>`).join("");
}
function showResult(r) {
  const card = (kind, o) => `
    <div class="result-card"><div class="kind">${kind}</div>
      <div class="label">${o.label}</div>${probBars(o.probabilities, o.label)}</div>`;
  const used = state.features.length - r.defaulted_fields.length;
  $("#result").innerHTML = `
    <div class="results">${card("Behaviour", r.behaviour)}${card("Circuit", r.circuit)}</div>
    <p class="note">${used} of ${state.features.length} fields entered by you; the other ${r.defaulted_fields.length} used the training-set median.</p>`;
  const panel = $("#result-panel");
  panel.hidden = false;
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- metrics ---------- */
function classStats(cm) {
  return cm.map((row, i) => {
    const tp = row[i];
    const support = row.reduce((a, b) => a + b, 0);
    const predicted = cm.reduce((a, r) => a + r[i], 0);
    const p = predicted ? tp / predicted : 0, r = support ? tp / support : 0;
    return { p, r, f1: p + r ? (2 * p * r) / (p + r) : 0, support };
  });
}
function renderMetrics() {
  const m = state.metrics, t = m[state.task], cfg = m.configuration;
  const cls = t.classes, cm = t.confusion_matrix, stats = classStats(cm);
  const heat = cm.map((row) => {
    const s = row.reduce((a, b) => a + b, 0) || 1;
    return `<tr><th>${cls[cm.indexOf(row)]}</th>${row.map((v, j) => {
      const a = (0.08 + 0.85 * (v / s)).toFixed(2);
      const diag = cm.indexOf(row) === j;
      return `<td class="${diag ? "diag" : ""}" style="--a:${a}">${v}</td>`;
    }).join("")}</tr>`;
  }).join("");
  $("#metrics").innerHTML = `
    <div class="kpis">
      <div class="kpi"><b>${pct(t.accuracy)}</b><span>Accuracy</span></div>
      <div class="kpi"><b>${pct(t.macro_f1)}</b><span>Macro F1</span></div>
      <div class="kpi"><b>${t.training_time_seconds.toFixed(1)} s</b><span>Training time</span></div>
      <div class="kpi"><b>${stats.reduce((a, s) => a + s.support, 0)}</b><span>Test samples</span></div>
    </div>
    <div class="two-col">
      <div><h3>Confusion matrix</h3>
        <div class="scroll"><table class="cm"><thead><tr><th></th>${cls.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${heat}</tbody></table></div>
        <p class="axis">Rows: true class. Columns: predicted class. Shading is the share of each true class.</p></div>
      <div><h3>Per-class results</h3>
        <div class="scroll"><table class="cls"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>
          ${stats.map((s, i) => `<tr><th>${cls[i]}</th><td>${pct(s.p)}</td><td>${pct(s.r)}</td><td>${pct(s.f1)}</td><td>${s.support}</td></tr>`).join("")}
        </tbody></table></div></div>
    </div>
    <p class="cfg">Forest: ${cfg.n_trees} trees, max depth ${cfg.max_depth}, min samples to split ${cfg.min_samples_split}, features per split ${cfg.n_feats ?? "default"}, seed ${cfg.seed}.</p>`;
}
document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  state.task = b.dataset.task;
  if (state.metrics) renderMetrics();
}));

/* ---------- init ---------- */
(async function init() {
  try {
    state.features = await api("/api/features");
    buildForm();
    state.samples = await api("/api/samples").catch(() => []);
  } catch (e) { $("#fields").innerHTML = `<p class="muted">Could not load form: ${e.message}</p>`; }
  try { state.metrics = await api("/api/metrics"); renderMetrics(); }
  catch (e) { $("#metrics").innerHTML = `<p class="muted">Metrics unavailable: ${e.message}</p>`; }
})();
