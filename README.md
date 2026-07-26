# Data-Analysis Agent

An LLM agent that answers plain-English questions about a dataset by **writing
Python, executing it in a sandbox, reading its own errors, correcting itself,
and returning a grounded answer** with the table or chart it produced.

The thing that makes this an *agent* (not a single LLM call) is the loop:
**reason → write code → run → observe → self-correct → repeat**, capped by a
max-iteration guard. The LLM is **Google Gemini** (via `google-genai`), behind a
provider-agnostic interface so it can be swapped.

> Status: **Level 1 (core loop)** implemented — reason → act → observe → answer,
> end to end, over a hand-rolled loop. See [plan.md](plan.md) for the full
> roadmap (self-correction, planning, UI, evals).

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
      │        ▲                          (function calling: run_python / final_answer)
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

## Web UI (Streamlit)

Prefer clicking to typing? Run the UI:

```bash
streamlit run app.py
```

It opens in your browser and lets you:

- upload a CSV (or use the bundled sample);
- pick the **model** and **max steps** in the sidebar (handy for dodging the
  per-model daily quota);
- ask a question and watch the **reasoning trace stream live** (each step's code
  and result);
- see the final answer with any **chart rendered inline** and tables shown;
- keep previous questions (and their charts) visible for the session.

Keys are read from `.env` locally, or from Streamlit **secrets** when deployed
(`GEMINI_API_KEY` / `GEMINI_API_KEY_BACKUP`). The same failover applies.

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
- `test_orchestrator.py` — happy path, self-correction (error fed back), step cap, plain-text answer.
- `test_gemini_adapter.py` — history↔Gemini translation and response parsing (no network).

## Roadmap

| Level | Focus | Status |
|---|---|---|
| 1 | Core loop (reason → act → observe → answer) | ✅ implemented |
| 2 | Self-correction (traceback fed back, retry cap) | wired; deepen next |
| 3 | Planning / multi-step + charts | partial (charts land now) |
| 4 | Streamlit UI (live trace + inline charts) · session memory · deploy | UI ✅; memory/deploy planned |
| 5 | Eval harness + before/after accuracy | planned |

See [plan.md](plan.md) for the full plan, interface contracts, and milestone
acceptance criteria.
