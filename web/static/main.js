"use strict";

// --- State ---
const state = { datasetId: null, hasKeys: false, running: false, turns: 0, memoryTurns: 0 };

// --- Elements ---
const $ = (id) => document.getElementById(id);
const keyPill = $("key-pill");
const fileInput = $("file-input");
const drop = $("drop");
const datasetInfo = $("dataset-info");
const modelSelect = $("model-select");
const stepsRange = $("steps-range");
const stepsValue = $("steps-value");
const questionEl = $("question");
const askBtn = $("ask-btn");
const askNote = $("ask-note");
const emptyState = $("empty-state");
const conversationEl = $("conversation");
const convoHead = $("convo-head");
const convoCount = $("convo-count");
const newConvoBtn = $("new-convo");

// --- Init ---
init();

async function init() {
  bindEvents();
  try {
    const cfg = await fetch("/api/config").then((r) => r.json());
    state.hasKeys = cfg.has_keys;
    modelSelect.innerHTML = cfg.models
      .map((m) => `<option value="${escapeAttr(m)}">${escapeHtml(m)}</option>`)
      .join("");
    stepsRange.value = cfg.max_steps;
    stepsValue.textContent = cfg.max_steps;
    if (cfg.has_keys) {
      keyPill.textContent = `${cfg.n_keys} key${cfg.n_keys > 1 ? "s" : ""} loaded`;
      keyPill.className = "pill pill-ok";
    } else {
      keyPill.textContent = "no API key";
      keyPill.className = "pill pill-err";
    }
  } catch {
    keyPill.textContent = "server error";
    keyPill.className = "pill pill-err";
  }
  refreshAsk();
}

function bindEvents() {
  $("browse-btn").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) uploadCsv(fileInput.files[0]);
  });
  $("sample-btn").addEventListener("click", loadSample);
  ["dragover", "dragenter"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("dragover"); })
  );
  drop.addEventListener("drop", (ev) => {
    const f = ev.dataTransfer.files[0];
    if (f) uploadCsv(f);
  });
  stepsRange.addEventListener("input", () => (stepsValue.textContent = stepsRange.value));
  questionEl.addEventListener("input", refreshAsk);
  questionEl.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !askBtn.disabled) ask();
  });
  askBtn.addEventListener("click", ask);
  newConvoBtn.addEventListener("click", newConversation);
}

function refreshAsk() {
  const ready = state.datasetId && state.hasKeys && questionEl.value.trim() && !state.running;
  askBtn.disabled = !ready;
  if (!state.hasKeys) askNote.textContent = "Add GEMINI_API_KEY to .env to enable.";
  else if (!state.datasetId) askNote.textContent = "Load a dataset first.";
  else if (state.running) askNote.textContent = "Working…";
  else if (state.memoryTurns > 0) askNote.textContent = "Follow-up? ⌘/Ctrl + Enter";
  else askNote.textContent = "⌘/Ctrl + Enter to ask";
}

function updateConvoCount() {
  const q = state.turns === 1 ? "question" : "questions";
  const mem = state.memoryTurns > 0 ? ` · remembers ${state.memoryTurns}` : "";
  convoCount.textContent = `${state.turns} ${q}${mem}`;
}

// --- Dataset ---
async function loadSample() {
  await withBusy($("sample-btn"), async () => {
    const meta = await fetch("/api/sample", { method: "POST" }).then(jsonOrThrow);
    onDataset(meta);
  });
}

async function uploadCsv(file) {
  const form = new FormData();
  form.append("file", file);
  drop.classList.add("dragover");
  try {
    const meta = await fetch("/api/upload", { method: "POST", body: form }).then(jsonOrThrow);
    onDataset(meta);
  } catch (err) {
    alert(err.message || "Upload failed.");
  } finally {
    drop.classList.remove("dragover");
  }
}

function onDataset(meta) {
  state.datasetId = meta.id;
  $("ds-name").textContent = meta.name;
  $("ds-dims").textContent = `${meta.rows} × ${meta.cols}`;
  $("ds-columns").innerHTML = meta.columns
    .map((c) => `<span class="col-chip"><b>${escapeHtml(c.name)}</b> ${escapeHtml(c.dtype)}</span>`)
    .join("");
  renderPreviewTable(meta.preview);
  datasetInfo.classList.remove("hidden");
  // A new dataset starts a fresh conversation.
  resetConversationView();
  refreshAsk();
}

function renderPreviewTable(preview) {
  const thead = `<tr>${preview.columns.map((c) => `<th>${escapeHtml(String(c))}</th>`).join("")}</tr>`;
  const rows = preview.data
    .map((row) => `<tr>${row.map((v) => `<td>${escapeHtml(fmt(v))}</td>`).join("")}</tr>`)
    .join("");
  $("ds-preview").innerHTML = thead + rows;
}

// --- Conversation control ---
function resetConversationView() {
  conversationEl.innerHTML = "";
  state.turns = 0;
  state.memoryTurns = 0;
  convoHead.classList.add("hidden");
  emptyState.classList.remove("hidden");
  refreshAsk();
}

async function newConversation() {
  if (state.datasetId) {
    try {
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: state.datasetId }),
      });
    } catch { /* best effort */ }
  }
  resetConversationView();
}

function createTurn(question, isFollowup) {
  const turn = document.createElement("article");
  turn.className = "turn";
  const fu = isFollowup ? `<span class="pill turn-followup">↳ follow-up</span>` : "";
  turn.innerHTML = `
    <div class="turn-q">
      <span class="turn-label">You asked</span>${fu}
      <span class="q-text">${escapeHtml(question)}</span>
    </div>
    <div class="trace turn-trace"></div>
    <div class="turn-answer"></div>`;
  conversationEl.appendChild(turn);
  return {
    trace: turn.querySelector(".turn-trace"),
    answer: turn.querySelector(".turn-answer"),
    el: turn,
  };
}

// --- Ask (streaming, multi-turn) ---
async function ask() {
  const question = questionEl.value.trim();
  if (!question || !state.datasetId || state.running) return;

  state.running = true;
  emptyState.classList.add("hidden");
  convoHead.classList.remove("hidden");
  _lastFigure = null;

  const isFollowup = state.memoryTurns > 0;
  const ctx = createTurn(question, isFollowup);
  state.turns += 1;
  updateConvoCount();
  questionEl.value = "";
  refreshAsk();

  const thinking = addThinking(ctx.trace);
  ctx.el.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: state.datasetId,
        question,
        model: modelSelect.value,
        max_steps: Number(stepsRange.value),
      }),
    });
    if (!res.ok) {
      const detail = await res.json().then((d) => d.detail).catch(() => res.statusText);
      thinking.remove();
      showRunError(ctx.answer, { message: detail });
      return;
    }
    await readStream(res, thinking, ctx);
  } catch (err) {
    thinking.remove();
    showRunError(ctx.answer, { message: err.message || "Network error." });
  } finally {
    state.running = false;
    refreshAsk();
  }
}

async function readStream(res, thinking, ctx) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) handleEvent(JSON.parse(line), thinking, ctx);
    }
  }
}

function handleEvent(evt, thinking, ctx) {
  if (evt.type === "step") {
    ctx.trace.insertBefore(renderStep(evt.data), thinking);
    thinking.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } else if (evt.type === "done") {
    thinking.remove();
    if (typeof evt.data.memory_turns === "number") state.memoryTurns = evt.data.memory_turns;
    renderAnswerInto(ctx.answer, evt.data);
    updateConvoCount();
    refreshAsk();
  } else if (evt.type === "error") {
    thinking.remove();
    showRunError(ctx.answer, evt.data);
  }
}

// --- Rendering ---
function addThinking(container) {
  const el = document.createElement("div");
  el.className = "thinking";
  el.innerHTML = `<span class="dots"><span></span><span></span><span></span></span> The agent is thinking…`;
  container.appendChild(el);
  return el;
}

function renderStep(s) {
  const el = document.createElement("div");
  el.className = "step";

  let badge, label;
  if (s.action === "plan") { badge = "badge-plan"; label = "plan"; }
  else if (s.action === "final_answer") { badge = "badge-answer"; label = "final_answer"; }
  else { badge = "badge-code"; label = s.action; }
  const fix = s.is_correction
    ? `<span class="step-badge badge-fix">🔧 self-correcting</span>` : "";

  const parts = [
    `<div class="step-head">
       <span class="step-badge ${badge}">${escapeHtml(label)}</span>${fix}
       <span class="step-num">Step ${s.index}</span>
     </div>`,
    `<div class="step-body">`,
  ];

  if (s.thought && s.action !== "plan") parts.push(`<div class="thought">${mdInline(s.thought)}</div>`);

  if (s.plan && s.plan.length) {
    parts.push(`<ol class="plan-list">${s.plan.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ol>`);
  }

  if (s.code) parts.push(`<pre class="code">${highlightPy(s.code)}</pre>`);

  if (s.observation) parts.push(renderObservation(s.observation));

  if (s.final_answer && s.action === "final_answer") {
    parts.push(`<div class="answer-text">${mdInline(s.final_answer)}</div>`);
  }

  parts.push(`</div>`);
  el.innerHTML = parts.join("");
  return el;
}

function renderObservation(obs) {
  if (obs.timed_out) {
    return `<pre class="output err">⏱ ${escapeHtml(obs.error_traceback || "Execution timed out.")}</pre>`;
  }
  if (!obs.ok) {
    return `<div><div class="output-label">error</div><pre class="output err">${escapeHtml(obs.error_traceback || "Execution error.")}</pre></div>`;
  }
  const blocks = [];
  if (obs.stdout && obs.stdout.trim())
    blocks.push(`<div><div class="output-label">stdout</div><pre class="output">${escapeHtml(obs.stdout)}</pre></div>`);
  if (obs.dataframe_preview)
    blocks.push(`<div><div class="output-label">result</div><pre class="output">${escapeHtml(obs.dataframe_preview)}</pre></div>`);
  else if (obs.result_repr && !obs.figure)
    blocks.push(`<div><div class="output-label">result</div><pre class="output">${escapeHtml(obs.result_repr)}</pre></div>`);
  if (obs.figure) blocks.push(`<img class="chart" src="${obs.figure}" alt="generated chart" />`);
  if (obs.execution_time_s != null)
    blocks.push(`<div class="meta-line">ran in ${obs.execution_time_s}s · ${escapeHtml(obs.result_kind)}</div>`);
  return blocks.join("") || `<div class="meta-line">ran with no explicit result</div>`;
}

function renderAnswerInto(el, run) {
  const status = {
    answered: ["pill-ok", "Answered"],
    cap_reached: ["pill-warn", "Step limit reached"],
    failed: ["pill-err", "Could not finish"],
    error: ["pill-err", "Error"],
  }[run.status] || ["pill-muted", run.status];

  const planPill = run.planned
    ? `<span class="pill pill-soft">📋 planned ${run.plan.length}</span>` : "";

  let note = "";
  if (run.n_errors && run.recovered)
    note = `<div class="note note-ok">🔧 Self-corrected: hit ${run.n_errors} error(s) and recovered.</div>`;
  else if (run.n_errors)
    note = `<div class="note note-warn">Hit ${run.n_errors} error(s) and could not fully recover.</div>`;

  const chart = run.figure && !traceHasFigure(run.figure)
    ? `<img class="chart" src="${run.figure}" alt="final chart" />` : "";

  const tokens = (run.usage && run.usage.total_tokens) || "?";
  el.innerHTML = `
    <div class="answer card">
      <div class="answer-head">
        <h2>Answer</h2>
        <span class="pill ${status[0]}">${escapeHtml(status[1])}</span>
        ${planPill}
      </div>
      <div class="answer-body">
        <div class="answer-text">${mdInline(run.final_answer || "No answer produced.")}</div>
        ${chart}
        ${note}
        <div class="answer-meta">
          <span>steps: ${run.n_steps}</span>
          <span>errors: ${run.n_errors}</span>
          <span>tokens: ${escapeHtml(String(tokens))}</span>
          <span>model: ${escapeHtml(modelSelect.value)}</span>
        </div>
      </div>
    </div>`;
}

// Avoid duplicating the final chart when it already appears in the last trace step.
let _lastFigure = null;
function traceHasFigure(fig) {
  const dup = _lastFigure === fig;
  _lastFigure = fig;
  return dup;
}

function showRunError(el, data) {
  const rate = data.rate_limited;
  const msg = rate
    ? "All configured keys hit the free-tier limit. Wait for the daily reset, switch the model, add a backup key, or enable billing."
    : (data.message || "Something went wrong.");
  el.innerHTML = `
    <div class="answer card">
      <div class="answer-head"><h2>Answer</h2><span class="pill pill-err">Error</span></div>
      <div class="answer-body"><div class="note note-err">${escapeHtml(msg)}</div></div>
    </div>`;
}

// --- Helpers ---
function jsonOrThrow(r) {
  if (r.ok) return r.json();
  return r.json().then((d) => { throw new Error(d.detail || r.statusText); });
}
async function withBusy(btn, fn) {
  const t = btn.textContent;
  btn.disabled = true;
  try { await fn(); } catch (e) { alert(e.message || "Failed."); } finally { btn.disabled = false; btn.textContent = t; }
}
function fmt(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Math.round(v * 1e4) / 1e4);
  return String(v);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}
// Minimal, safe inline markdown: escape first, then **bold**, `code`, newlines.
function mdInline(s) {
  let t = escapeHtml(s);
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>");
  return `<p>${t}</p>`;
}
// Lightweight Python highlighter (operates on already-escaped text, so it's safe).
const PY_RE = /(#[^\n]*)|('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|(\b(?:import|from|as|def|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|lambda|with|print|class)\b)|(\b\d+\.?\d*\b)/g;
function highlightPy(code) {
  return escapeHtml(code).replace(PY_RE, (m, cmt, str, kw, num) => {
    if (cmt) return `<span class="cmt">${cmt}</span>`;
    if (str) return `<span class="str">${str}</span>`;
    if (kw) return `<span class="kw">${kw}</span>`;
    if (num) return `<span class="num">${num}</span>`;
    return m;
  });
}
