# Data-Analysis Agent

An LLM agent that answers plain-English questions about a dataset by **writing
Python, executing it in a sandbox, reading its own errors, correcting itself,
and returning a grounded answer** with the table or chart it produced.

The thing that makes this an *agent* (not a single LLM call) is the loop:
**reason → write code → run → observe → self-correct → repeat**, capped by a
max-iteration guard. The LLM sits behind a **provider-agnostic interface**: the
default is **Groq** (`llama-3.3-70b-versatile`, via the `groq` SDK) for its fast
inference and generous free-tier limits; set `LLM_PROVIDER=gemini` to switch to
**Google Gemini** — no other code changes.

> Status: **Levels 1–5 + a custom web UI** implemented — the reason → act →
> observe → self-correct loop runs end to end, with up-front **planning** for
> multi-part questions, a repair budget, **session memory** for follow-up
> questions, a hand-built FastAPI web app that streams the reasoning trace live,
> and an **offline eval harness** that scores accuracy against verified ground
> truth. See [plan.md](plan.md) for the full roadmap (deploy remains).

---

## Why the code runs in *our* sandbox

Some providers ship a server-side code-execution tool that runs Python on their
machines. We deliberately **do not** use it. The self-correcting loop and the
"how do you stop it running dangerous code" guardrails are the point of this
project, and both require code running in a sandbox **we** control. So the model
calls a `run_python` **function**; we execute it out of process (see §Safety).

## Architecture

```
Question + CSV
      │
      ▼
Orchestrator (hand-rolled loop)  ◄──► LLMClient ──► Groq / Gemini adapter
      │        ▲                          (function calling: plan / run_python / final_answer)
      ▼        │ observation
   Sandbox (subprocess) ── worker.py execs code, captures result/traceback/figure
```

- `agent/orchestrator.py` — the loop controller (the piece we hand-roll).
- `agent/llm/` — `base.py` (provider-agnostic `LLMClient`), `groq_client.py` and
  `gemini.py` (adapters), `factory.py` (picks the provider from `LLM_PROVIDER`).
- `agent/tools/` — `sandbox.py` (host: spawn + timeout + parse) and `worker.py`
  (runs inside the subprocess).
- `agent/types.py` — the contracts (`ExecutionResult`, `Step`, `AgentRun`).
- `config.py` — typed settings from `.env`.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure your key
cp .env.example .env         # then set GROQ_API_KEY=... (get one at console.groq.com)

# 3. Ask a question about the bundled sample dataset
python main.py -q "Which region grew fastest from January to April by revenue?"

# ...or your own CSV
python main.py -d path/to/data.csv -q "What is the total revenue?"
```

The CLI prints the full reasoning trace (each step's code + observation), then
the final answer. Add `--verbose` for step logging.

**Providers.** Set `LLM_PROVIDER` in `.env`:

- **`groq` (default)** — `llama-3.3-70b-versatile` via the `groq` SDK. Fast
  inference and much more generous free-tier request limits than Gemini's flash
  tier, which is why it's the default. `GROQ_API_KEY` may be a **comma-separated
  list** for automatic failover. The adapter retries Groq's occasional
  `tool_use_failed` (a Llama tool-formatting hiccup) transparently. Change the
  model via `GROQ_MODEL`.
- **`gemini`** — `gemini-3.6-flash` (the latest Flash) via `google-genai`. Pro
  models need paid quota; older `gemini-2.5-*` IDs may 404 for new keys. The
  adapter handles Gemini 3.x thought-signatures. Change the model via `GEMINI_MODEL`.

Both go through the same `LLMClient` interface, so switching is a one-line `.env`
change. If all keys hit their rate limit, the CLI/UI show a clear message (each
request is capped at 60s) instead of hanging.

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
  so no artifact paths are exposed), plus a status/plan/recovery summary;
- ask **follow-up questions** — the agent remembers the conversation, and a
  *New conversation* button resets it.

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

## Session memory (follow-up questions)

Ask a question, then follow up — *"now just the BMWs"*, *"plot that"*, *"the same
but by month"* — and the agent uses the conversation so far to resolve what you
mean.

- Each session keeps a bounded list of prior **question → answer** turns
  (`agent/memory.py`), injected as a compact "earlier in this session" block into
  the next question's context.
- The web app shows a **multi-turn conversation thread** (follow-ups are tagged
  *↳ follow-up*) with a **New conversation** button to start fresh; the terminal
  has an interactive mode: `python main.py --chat`.
- Because the sandbox is stateless by design (each `run_python` is a fresh
  subprocess with only `df`), memory is **conversational, not computational**: the
  agent recomputes from `df` using the remembered context. Persisting the actual
  computed variables would need a long-lived kernel — a documented future step.

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

## Evaluation (does it actually work?)

Most "chat with your data" demos never prove they're right. This one ships an
**offline eval harness** ([evals/](evals/)) that scores the agent against **20
tiered questions** with **verified ground-truth answers**, computed from a
deliberately messy dataset (`evals/sales_data.csv` — dirty region casing, missing
segments, and *no* pre-computed revenue/profit, so the agent has to clean and
derive). Easy questions check the basic loop; medium ones check grouping and the
region-cleaning; the hard ones (biggest MoM drop, fastest-growing region) are
genuine multi-step reasoning — where planning and self-correction earn their keep.

```bash
# Prove the harness end to end without spending any quota:
python -m evals.run_evals --mock

# Run for real, a slice at a time (free tier ≈ 20 requests/day):
python -m evals.run_evals --difficulty easy
python -m evals.run_evals --ids q11,q12,q13

# Score whatever has accumulated and (re)write evals/report.md:
python -m evals.run_evals --report-only
```

- **Quota-aware by design.** Each answered question is cached the moment it
  completes, so a run interrupted by the daily limit just resumes later — you
  build up to 20/20 across a few days, and the report always scores what's cached.
- **Deterministic checkers** ([evals/checkers.py](evals/checkers.py)) compare the
  agent's prose to ground truth — numbers within a tolerance (handling `$`,
  commas, `%`, month names), categorical by exact word, dict by every value
  present — so scoring itself spends **no** quota.
- **Before/after story.** Keep two cached runs side by side with `--tag baseline`
  vs `--tag improved` and quote the delta. The generated `evals/report.md` is the
  scorecard (overall %, by-difficulty, tokens, and how many answers self-corrected).

## Tests

```bash
pytest            # sandbox (real subprocess) + orchestrator (mocked LLM) + adapter translation + evals
```

- `test_sandbox.py` — success / dataframe / error traceback / blocked import / timeout / figure.
- `test_orchestrator.py` — happy path, self-correction (error fed back), step cap, plain-text answer, planning (plan → execute → answer, plan fed back, unplanned path unchanged).
- `test_gemini_adapter.py` — history↔Gemini translation, response parsing, and array-param (plan) schema mapping (no network).
- `test_memory.py` / `test_web.py` — session memory + the FastAPI layer (mocked agent).
- `test_evals.py` — the eval checkers, question selection, and a mock end-to-end run (no quota).

## Roadmap

| Level | Focus | Status |
|---|---|---|
| 1 | Core loop (reason → act → observe → answer) | ✅ implemented |
| 2 | Self-correction: repair budget, repeat-guard, correction tagging | ✅ implemented |
| 3 | Planning / multi-step decomposition + charts | ✅ implemented |
| 4 | Web UI (custom FastAPI + Streamlit) · session memory · deploy | UI ✅; memory ✅; deploy planned |
| 5 | Eval harness (verified ground truth, quota-aware, scorecard) | ✅ implemented |

See [plan.md](plan.md) for the full plan, interface contracts, and milestone
acceptance criteria.
