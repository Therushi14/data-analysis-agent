"""Eval runner CLI.

    # Score whatever's already cached, write the report:
    python -m evals.run_evals --report-only

    # Run a subset live (spends quota), cache each result, then score:
    python -m evals.run_evals --difficulty easy
    python -m evals.run_evals --ids q11,q12,q13
    python -m evals.run_evals --limit 5

    # Prove the harness end-to-end without spending quota:
    python -m evals.run_evals --mock

Results cache per (tag, model, question), so a run interrupted by the daily
rate limit resumes later and the report always scores what has accumulated.
Use --tag to keep a "baseline" cache beside an "improved" one for before/after.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import get_settings
from evals.cache import Cache
from evals.checkers import check_item
from evals.eval_questions import DATA_PATH, EVAL_SET, get_questions
from evals.report import console_summary, render_markdown, summarize


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_rate_limit(msg: str) -> bool:
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


# --- Agent runners --------------------------------------------------------------

def _default_agent_fn(model: str, settings):
    """A callable(question) -> record backed by the real agent. None if no keys."""
    keys = settings.api_keys
    if not keys:
        return None

    import time as _time

    import pandas as pd

    from agent.llm.gemini import GeminiClient
    from agent.orchestrator import Orchestrator
    from agent.tools.sandbox import Sandbox

    df = pd.read_csv(DATA_PATH)
    work_dir = Path("artifacts") / "evals" / f"run-{int(_time.time())}"
    sandbox = Sandbox(
        work_dir=work_dir,
        timeout_s=settings.sandbox_timeout_s,
        stdout_char_cap=settings.stdout_char_cap,
    )
    llm = GeminiClient(
        api_keys=keys,
        model=model,
        temperature=settings.temperature,
        request_timeout_s=settings.request_timeout_s,
    )
    orch = Orchestrator(
        llm=llm,
        sandbox=sandbox,
        max_steps=settings.max_steps,
        max_consecutive_failures=settings.max_consecutive_failures,
    )

    def run_one(question: str) -> dict:
        start = _time.perf_counter()
        run = orch.run(question, df)  # no memory: eval questions are independent
        return {
            "answer": run.final_answer or "",
            "status": run.status,
            "steps": run.n_steps,
            "errors": run.n_errors,
            "recovered": run.recovered,
            "tokens": (run.usage or {}).get("total_tokens", 0),
            "elapsed_s": round(_time.perf_counter() - start, 2),
        }

    return run_one


def _format_expected(expected) -> str:
    if isinstance(expected, dict):
        return "; ".join(f"{k}: {v}" for k, v in expected.items())
    return str(expected)


def make_mock_agent_fn(correct: bool = True):
    """A fake agent that echoes the ground truth (or a wrong number). No quota."""
    by_q = {q["question"]: q for q in EVAL_SET}

    def run_one(question: str) -> dict:
        item = by_q.get(question)
        if item is None:
            answer = ""
        elif correct:
            answer = f"The answer is {_format_expected(item['expected'])}."
        elif item["kind"] == "numeric":
            answer = f"The answer is {float(item['expected']) * 1.5:g}."  # deliberately wrong
        else:
            answer = "I could not determine a reliable answer."
        return {
            "answer": answer, "status": "answered", "steps": 2, "errors": 0,
            "recovered": False, "tokens": 1234, "elapsed_s": 0.0,
        }

    return run_one


# --- Rows -----------------------------------------------------------------------

def _pending_row(item: dict) -> dict:
    return {
        "id": item["id"], "difficulty": item["difficulty"], "question": item["question"],
        "kind": item["kind"], "expected": item["expected"],
        "verdict": "pending", "detail": "not run yet", "answer": None,
    }


def _scored_row(item: dict, rec: dict, chk) -> dict:
    return {
        "id": item["id"], "difficulty": item["difficulty"], "question": item["question"],
        "kind": item["kind"], "expected": item["expected"],
        "verdict": chk.verdict, "detail": chk.detail, "answer": rec.get("answer"),
        "status": rec.get("status"), "steps": rec.get("steps"), "errors": rec.get("errors"),
        "recovered": rec.get("recovered"), "tokens": rec.get("tokens"),
        "elapsed_s": rec.get("elapsed_s"),
    }


# --- Core -----------------------------------------------------------------------

def run(
    ids: list[str] | None = None,
    difficulty: str | None = None,
    limit: int | None = None,
    model: str | None = None,
    tag: str = "default",
    tol_pct: float = 0.5,
    refresh: bool = False,
    use_cache: bool = True,
    report_only: bool = False,
    agent_fn=None,
    cache_root=None,
    out=None,
    verbose: bool = True,
) -> dict:
    settings = get_settings()
    model = model or settings.gemini_model
    cache = Cache(tag=tag, model=model, root=cache_root)
    stopped = None

    if not report_only:
        to_run = get_questions(ids, difficulty, limit)
        if agent_fn is None:
            agent_fn = _default_agent_fn(model, settings)
        if agent_fn is None and verbose:
            print(">> No GEMINI_API_KEY set — cannot run live. Scoring cache only.")
        for item in to_run:
            qid = item["id"]
            if use_cache and not refresh and cache.load(qid) is not None:
                continue
            if agent_fn is None:
                continue
            if verbose:
                print(f"running {qid} [{item['difficulty']}]: {item['question']}")
            try:
                rec = agent_fn(item["question"])
            except Exception as e:  # noqa: BLE001 — any failure just leaves it pending
                msg = str(e)
                if _is_rate_limit(msg):
                    stopped = "rate_limit"
                    if verbose:
                        print(">> free-tier limit hit — stopping. Cached results kept; resume later.")
                    break
                if verbose:
                    print(f">> error on {qid}: {msg[:200]}")
                continue
            rec["question"] = item["question"]
            rec["ts"] = _now()
            cache.save(qid, rec)
            if verbose:
                print(f"   -> {rec['status']} · {(rec['answer'] or '')[:80]}")

    # Score everything available in the cache against the full EVAL_SET.
    rows = []
    for item in EVAL_SET:
        rec = cache.load(item["id"])
        rows.append(_pending_row(item) if rec is None
                    else _scored_row(item, rec, check_item(item, rec.get("answer", ""), tol_pct)))

    meta = {"model": model, "tag": tag, "tol_pct": tol_pct, "stopped": stopped}
    out_path = Path(out) if out else (Path(__file__).parent / "report.md")
    out_path.write_text(render_markdown(rows, meta), encoding="utf-8")
    if verbose:
        print(console_summary(rows, meta))
        print(f"\nReport written to {out_path}")
    return {"rows": rows, "meta": meta, "summary": summarize(rows), "report_path": str(out_path)}


def main() -> None:
    # The scorecard uses ✅/🟡/❌ symbols; the Windows console defaults to cp1252
    # and would raise UnicodeEncodeError, so switch stdout/stderr to UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(description="Score the data-analysis agent against the eval set.")
    p.add_argument("--ids", help="Comma-separated question ids to run (e.g. q11,q12).")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Run only this tier.")
    p.add_argument("--limit", type=int, help="Cap how many questions to run (quota-friendly).")
    p.add_argument("--model", help="Override the Gemini model.")
    p.add_argument("--tag", default="default", help="Cache/report namespace (for before/after).")
    p.add_argument("--tolerance", type=float, default=0.5, help="Numeric match tolerance %% (default 0.5).")
    p.add_argument("--refresh", action="store_true", help="Re-run selected questions, ignoring cache.")
    p.add_argument("--no-cache", action="store_true", help="Do not read the cache when selecting.")
    p.add_argument("--report-only", action="store_true", help="Just score the cache; run nothing.")
    p.add_argument("--mock", action="store_true", help="Use a fake agent (no quota) to prove the harness.")
    p.add_argument("--mock-wrong", action="store_true", help="Mock agent that fails numeric questions.")
    p.add_argument("--out", help="Report output path (default evals/report.md).")
    args = p.parse_args()

    agent_fn = None
    if args.mock or args.mock_wrong:
        agent_fn = make_mock_agent_fn(correct=not args.mock_wrong)

    run(
        ids=[s.strip() for s in args.ids.split(",")] if args.ids else None,
        difficulty=args.difficulty,
        limit=args.limit,
        model=args.model,
        tag=args.tag,
        tol_pct=args.tolerance,
        refresh=args.refresh,
        use_cache=not args.no_cache,
        report_only=args.report_only,
        agent_fn=agent_fn,
    )


if __name__ == "__main__":
    main()
