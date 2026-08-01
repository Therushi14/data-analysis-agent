"""CLI entry point: answer one question about a CSV, end to end.

    python main.py --question "Which region grew fastest in Q4?"
    python main.py --data path/to/data.csv --question "..."

Prints the full reasoning trace (the hand-rolled loop in action), then the
final answer. Requires GEMINI_API_KEY in your environment or .env.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from agent.memory import SessionMemory
from agent.orchestrator import Orchestrator, result_summary
from agent.tools.sandbox import Sandbox
from agent.types import AgentRun, Step
from config import get_settings

DEFAULT_DATA = Path(__file__).parent / "data" / "sample_sales.csv"


def print_header(question: str) -> None:
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70, flush=True)


def print_step(step: Step) -> None:
    """Print one step as soon as it completes (live progress)."""
    marker = "  🔧 self-correcting" if step.is_correction else ""
    print(f"\n--- Step {step.index}: {step.action}{marker} ---")
    if step.thought:
        print(f"thought: {step.thought.strip()}")
    if step.plan:
        print("📋 plan:")
        for i, task in enumerate(step.plan, 1):
            print(f"    {i}. {task}")
    if step.code:
        print("code:")
        for line in step.code.strip().splitlines():
            print(f"    {line}")
    if step.observation:
        obs = step.observation
        status = "OK" if obs.ok else ("TIMEOUT" if obs.timed_out else "ERROR")
        print(f"observation [{status}] ({obs.execution_time_s}s): {obs.short_summary()}")
        if not obs.ok and obs.error_traceback:
            tail = obs.error_traceback.strip().splitlines()[-1]
            print(f"    {tail}")
    if step.final_answer is not None:
        print(f"final_answer: {step.final_answer}")
    sys.stdout.flush()


def print_footer(run: AgentRun) -> None:
    print("\n" + "=" * 70)
    print(f"STATUS: {run.status}")
    print(f"ANSWER: {run.final_answer}")
    if run.figure_path:
        print(f"FIGURE: {run.figure_path}")
    if run.n_errors:
        note = "recovered ✓" if run.recovered else "did not recover"
        print(f"ERRORS: {run.n_errors} ({note} via self-correction)")
    if run.usage:
        print(f"USAGE: {run.usage}")
    print("=" * 70 + "\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-Analysis Agent (CLI)")
    parser.add_argument("--question", "-q", help="Question to ask about the data (omit with --chat).")
    parser.add_argument("--data", "-d", default=str(DEFAULT_DATA), help="Path to a CSV file.")
    parser.add_argument("--model", default=None, help="Override the Gemini model ID.")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive multi-turn chat with session memory (ask follow-ups).")
    parser.add_argument("--verbose", action="store_true", help="Enable info logging.")
    args = parser.parse_args()
    if not args.chat and not args.question:
        parser.error("--question is required unless you use --chat")

    # Model answers / tracebacks / logs can contain non-ASCII (arrows, emoji);
    # the Windows console defaults to cp1252 and would raise UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    # Progress logging on by default so the loop shows movement between API
    # calls (e.g. "step 1: asking the model..."); --verbose adds DEBUG detail.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    settings = get_settings()
    model = args.model or settings.active_model

    from agent.llm.base import LLMError, LLMRateLimitError
    from agent.llm.factory import build_llm_client

    df = pd.read_csv(args.data)

    work_dir = Path("artifacts") / f"run-{int(time.time())}"
    sandbox = Sandbox(
        work_dir=work_dir,
        timeout_s=settings.sandbox_timeout_s,
        stdout_char_cap=settings.stdout_char_cap,
    )
    llm = build_llm_client(settings, model=model)
    orchestrator = Orchestrator(
        llm=llm,
        sandbox=sandbox,
        max_steps=settings.max_steps,
        max_consecutive_failures=settings.max_consecutive_failures,
    )

    print(
        f"(provider: {settings.llm_provider} | model: {model} | "
        f"keys: {len(settings.llm_keys)} | max_steps: {settings.max_steps} | "
        f"timeout: {settings.request_timeout_s}s)"
    )

    def answer(question: str, memory: SessionMemory | None = None) -> AgentRun | None:
        print_header(question)
        try:
            run = orchestrator.run(question, df, on_step=print_step, memory=memory)
        except LLMRateLimitError as e:
            print(f"\n[rate limit] {str(e)[:300]}")
            print(
                "\n>> All configured keys hit their rate limit. Wait a moment, add "
                "another key to GROQ_API_KEY (comma-separated), or switch the model."
            )
            return None
        except LLMError as e:
            print(f"\n[LLM error] {str(e)[:300]}")
            return None
        print_footer(run)
        return run

    if args.chat:
        memory = SessionMemory()
        print("\nInteractive chat — ask follow-ups (it remembers). Blank line or 'exit' to quit.")
        while True:
            try:
                question = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            run = answer(question, memory)
            if run is not None and run.final_answer:
                memory.add(question, run.final_answer, result_summary(run))
    else:
        answer(args.question)


if __name__ == "__main__":
    main()
