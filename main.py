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

from agent.orchestrator import Orchestrator
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
    print(f"\n--- Step {step.index}: {step.action} ---")
    if step.thought:
        print(f"thought: {step.thought.strip()}")
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
    if run.usage:
        print(f"USAGE: {run.usage}")
    print("=" * 70 + "\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-Analysis Agent (CLI)")
    parser.add_argument("--question", "-q", required=True, help="Question to ask about the data.")
    parser.add_argument("--data", "-d", default=str(DEFAULT_DATA), help="Path to a CSV file.")
    parser.add_argument("--model", default=None, help="Override the Gemini model ID.")
    parser.add_argument("--verbose", action="store_true", help="Enable info logging.")
    args = parser.parse_args()

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
    model = args.model or settings.gemini_model

    # Imported here so the CLI errors clearly if google-genai is missing.
    from google.genai import errors as genai_errors

    from agent.llm.gemini import GeminiClient

    df = pd.read_csv(args.data)

    work_dir = Path("artifacts") / f"run-{int(time.time())}"
    sandbox = Sandbox(
        work_dir=work_dir,
        timeout_s=settings.sandbox_timeout_s,
        stdout_char_cap=settings.stdout_char_cap,
    )
    llm = GeminiClient(
        api_key=settings.gemini_api_key,
        model=model,
        temperature=settings.temperature,
        request_timeout_s=settings.request_timeout_s,
    )
    orchestrator = Orchestrator(llm=llm, sandbox=sandbox, max_steps=settings.max_steps)

    print(f"(model: {model} | max_steps: {settings.max_steps} | timeout: {settings.request_timeout_s}s)")
    print_header(args.question)
    try:
        run = orchestrator.run(args.question, df, on_step=print_step)
    except genai_errors.APIError as e:
        msg = str(e)
        print(f"\n[Gemini API error] {msg[:300]}")
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            print(
                "\n>> This is the free-tier rate limit, not a hang. Wait ~30-60s "
                "and retry. (Pro models need paid quota; don't switch unless you "
                "have it.)"
            )
        return
    print_footer(run)


if __name__ == "__main__":
    main()
