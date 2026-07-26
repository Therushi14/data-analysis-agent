# Data-Analysis Agent

An LLM agent that answers plain-English questions about a dataset by **writing
Python, executing it in a sandbox, reading its own errors, correcting itself,
and returning a grounded answer** with the table or chart it produced.

The thing that makes this an *agent* (not a single LLM call) is the loop:
**reason → write code → run → observe → self-correct → repeat**, capped by a
max-iteration guard. The LLM is **Google Gemini** (via `google-genai`), behind a
provider-agnostic interface so it can be swapped.

> Status: **Levels 1–3 + a custom web UI** implemented — the reason → act →
> observe → self-correct loop runs end to end, with up-front **planning** for
> multi-part questions, a repair budget, and a hand-built FastAPI web app that
> streams the reasoning trace live. See [plan.md](plan.md) for the full roadmap
> (memory, evals).

---

## Why the code runs in *our* sandbox

Gemini ships a server-side `code_execution` tool that runs Python on Google's
machines. We deliberately **do not** use it. The self-correcting loop and the
"how do you stop it running dangerous code" guardrails are the point of this
project, and both require code running in a sandbox **we** control. So the model
calls a `run_python` **function**; we execute it out of process (see §Safety).

## Architecture

```
Question + CSV
      │
      ▼
Orchestrator (hand-rolled loop)  ◄──► LLMClient ──► Gemini adapter
      │        ▲                          (function calling: plan / run_python / final_answer)
      ▼        │ observation
   Sandbox (subprocess) ── worker.py execs code, captures result/traceback/figure
```

- `agent/orchestrator.py` — the loop controller (the piece we hand-roll).
- `agent/llm/` — `base.py` (provider-agnostic `LLMClient`) + `gemini.py` (adapter).
- `agent/tools/` — `sandbox.py` (host: spawn + timeout + parse) and `worker.py`
  (runs inside the subprocess).
- `agent/types.py` — the contracts (`ExecutionResult`, `Step`, `AgentRun`).
- `config.py` — typed settings from `.env`.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure your key
cp .env.example .env         # then set GEMINI_API_KEY=...

# 3. Ask a question about the bundled sample dataset
python main.py -q "Which region grew fastest from January to April by revenue?"

# ...or your own CSV
python main.py -d path/to/data.csv -q "What is the total revenue?"
```

The CLI prints the full reasoning trace (each step's code + observation), then
the final answer. Add `--verbose` for step logging.

**Model note.** The default is `gemini-3.6-flash` — the latest Flash, which works
on a free-tier key. Gemini **Pro** models require paid quota (they 429 on free
tier), and older `gemini-2.5-*` IDs may 404 for new keys. The adapter handles
Gemini 3.x thought-signatures, so 3.x models work out of the box. Change the
model via `GEMINI_MODEL` in `.env`.

**Free-tier quota.** The free tier allows ~**20 requests/day _per model_** (and a
low per-minute rate). Each question costs a few requests, so you get roughly
5–8 questions/day on one model. If you hit `RESOURCE_EXHAUSTED (429)`: wait for
the daily reset, switch `GEMINI_MODEL` to a different flash model (each model has
its own daily quota), use a key from a different project, or enable billing. The
CLI caps each request at 60s and shows a clear message instead of hanging.

## Web app (primary UI)

A hand-built, clean-and-minimal web UI (FastAPI backend + a static frontend in
[web/static/](web/static)) that reuses the **same** Orchestrator/Sandbox/Gemini
client as the CLI, so behavior can't diverge.

```bash
python run_web.py                 # http://127.0.0.1:8000
# or, with auto-reload for development:
uvicorn web.server:app --reload
```

It lets you:

- drag-and-drop a CSV (or load the bundled sample);
- pick the **model** and **max steps** (handy for dodging the per-model daily quota);
- ask a question and watch the **reasoning trace stream live** — the plan, each
  step's syntax-highlighted code, and its result — as newline-delimited JSON;
- see the final answer with any **chart rendered inline** (embedded as a data URI,
  so no artifact paths are exposed), plus a status/plan/recovery summary.

Keys are read from `.env` (`GEMINI_API_KEY` / `GEMINI_API_KEY_BACKUP`); the same
automatic failover applies.

### Alternate UI (Streamlit)

A simpler Streamlit version is also available:

```bash
streamlit run app.py
```

It offers the same core flow and, when deployed, reads keys from Streamlit
**secrets**.

## Self-correction (the centerpiece)

When generated code raises, the traceback is fed back as the next observation and
the agent rewrites its code — you'll see steps tagged **🔧 self-correcting** in
both the CLI and the UI. Three guards keep it honest and cheap:

- **Repair budget** (`MAX_CONSECUTIVE_FAILURES`, default 3): after N *consecutive*
  failed executions the agent stops with `status="failed"` and a message citing
  the last error, instead of burning every step (and API call). The counter
  resets on any success.
- **Repeat guard**: if the model re-emits byte-identical code that already failed,
  the observation tells it to change approach rather than loop.
- **Recovery tracking**: `AgentRun` exposes `n_errors` and `recovered`, so a run
  that hit an error and still answered is reported as a recovery.

## Planning (multi-step questions)

For anything beyond a one-liner, the agent **plans before it computes**. It calls
a `plan` tool with an ordered list of sub-tasks, then works through them — so a
multi-part question ("total km, *and* average price by doors, *and* plot the
distribution") gets fully answered instead of stopping after the first part.

- The plan is shown in the trace as a 📋 step (CLI and UI) and recorded on
  `AgentRun.plan`.
- While a plan is active, every observation carries a short reminder to finish
  **all** planned parts before the final answer.
- Planning is **optional**: a simple question skips it and answers in a single
  step, so easy questions cost no extra API call. The step cap is 8 (`MAX_STEPS`),
  leaving a planned run room to finish.

## Safety (the guardrails story)

Model-written Python never runs in the app process. `agent/tools/sandbox.py`
executes it in a **separate Python subprocess** with:

- a **timeout** that kills runaway code,
- an **import policy** that blocks `os`, `sys`, `subprocess`, `socket`,
  `requests`, and other filesystem/network/process modules,
- a **scrubbed environment** (secrets like `GEMINI_API_KEY` are stripped before
  the subprocess starts),
- **headless matplotlib** (figures render to PNG, never a display).

This is defense-in-depth, not a jail. A real container sandbox (`--network
none`, read-only rootfs, cgroups, non-root) is the documented hardening path in
[plan.md §7](plan.md).

## Tests

```bash
pytest            # sandbox (real subprocess) + orchestrator (mocked LLM) + adapter translation
```

- `test_sandbox.py` — success / dataframe / error traceback / blocked import / timeout / figure.
- `test_orchestrator.py` — happy path, self-correction (error fed back), step cap, plain-text answer, planning (plan → execute → answer, plan fed back, unplanned path unchanged).
- `test_gemini_adapter.py` — history↔Gemini translation, response parsing, and array-param (plan) schema mapping (no network).

## Roadmap

| Level | Focus | Status |
|---|---|---|
| 1 | Core loop (reason → act → observe → answer) | ✅ implemented |
| 2 | Self-correction: repair budget, repeat-guard, correction tagging | ✅ implemented |
| 3 | Planning / multi-step decomposition + charts | ✅ implemented |
| 4 | Web UI (custom FastAPI app + Streamlit) · session memory · deploy | UI ✅ (custom + Streamlit); memory/deploy planned |
| 5 | Eval harness + before/after accuracy | planned |

See [plan.md](plan.md) for the full plan, interface contracts, and milestone
acceptance criteria.
