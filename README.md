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
| 4 | Streamlit UI, session memory, deploy | planned |
| 5 | Eval harness + before/after accuracy | planned |

See [plan.md](plan.md) for the full plan, interface contracts, and milestone
acceptance criteria.
