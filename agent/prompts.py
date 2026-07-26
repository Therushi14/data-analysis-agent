"""Prompt construction: the system prompt, the turn-0 schema block, and the
mapping from an ExecutionResult into the payload fed back to the model.

All functions are pure (no I/O) so they can be snapshot-tested.
"""

from __future__ import annotations

import pandas as pd

from agent.types import ExecutionResult

SYSTEM_PROMPT = """\
You are a meticulous data analyst. You answer questions about a dataset by
writing and running Python, then explaining what the results mean.

You have a pandas DataFrame already loaded as `df`. You can use `pd` (pandas),
`np` (numpy), and `plt` (matplotlib.pyplot).

How to work:
- Plan first for anything non-trivial. If the question has several parts or needs
  multiple analysis steps (inspect -> compute -> visualize), call `plan` once at
  the start with an ordered list of short sub-tasks. Skip planning for a simple
  question you can answer in a single run_python call.
- Then work through the plan with `run_python`: write code that computes one part,
  and assign the value that answers it to a variable named `result` so it is
  captured and shown back to you. Address every part of a multi-part question
  before you finish — do not stop after the first part.
- Inspect the observation you get back (stdout, any error traceback, a preview
  of `result`) and let it drive your next step.
- If your code raises an error, read the traceback and rewrite the code to fix
  the actual cause — do not guess around it, and never re-run the exact same
  failing code. If after a couple of honest attempts you still cannot compute the
  answer, call final_answer and explain what went wrong instead of retrying.
- To make a chart, use matplotlib via `plt`; the figure is captured for you.
- Work efficiently: you have a limited number of steps, so combine related
  computations into one run_python call rather than spreading them thin.
- Never invent numbers. Every figure you state must come from code you ran.

When, and only when, every part of the question is answered from real results you
computed, call `final_answer` with a clear natural-language answer that covers all
parts. Reference the concrete numbers you found.
"""


def describe_dataframe(df: pd.DataFrame, sample_rows: int = 5) -> str:
    """A compact schema block: columns + dtypes, row count, and a few samples."""
    lines = ["Columns and dtypes:"]
    for col, dtype in df.dtypes.items():
        lines.append(f"  - {col} ({dtype})")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Sample (first {sample_rows} rows):")
    lines.append(df.head(sample_rows).to_string())
    return "\n".join(lines)


def build_initial_user_message(question: str, df: pd.DataFrame) -> str:
    return (
        f"Question: {question}\n\n"
        f"Here is the dataset you are working with (available as `df`):\n"
        f"{describe_dataframe(df)}"
    )


def plan_ack_payload(plan: list[str]) -> dict:
    """The tool-result returned when the model calls `plan`.

    Echoes the plan back and nudges the model to execute it, so the sub-tasks
    stay salient across the multi-step run.
    """
    return {
        "ok": True,
        "plan": plan,
        "steps_planned": len(plan),
        "instruction": (
            "Plan recorded. Now carry out each step with run_python. Only call "
            "final_answer once every step is done and grounded in real results."
        ),
    }


def observation_payload(
    result: ExecutionResult, repeated: bool = False, plan: list[str] | None = None
) -> dict:
    """Convert an ExecutionResult into the structured tool-result the model sees.

    `repeated` is True when this exact code already failed earlier — we then tell
    the model to change approach instead of retrying the same thing. `plan`, when
    set, keeps the outstanding sub-tasks salient so the model finishes every part
    before it answers.
    """
    if result.ok:
        payload: dict = {"ok": True, "result_kind": result.result_kind}
        if result.stdout.strip():
            payload["stdout"] = result.stdout
        if result.result_repr:
            payload["result"] = result.result_repr
        if result.dataframe_preview:
            payload["preview"] = result.dataframe_preview
        if result.figure_path:
            payload["figure"] = "A figure was produced and shown to the user."
        _add_plan_reminder(payload, plan)
        return payload

    # Failure (uncaught exception or timeout).
    payload = {"ok": False}
    if result.timed_out:
        payload["error"] = result.error_traceback
        payload["hint"] = "The code was too slow (possibly an infinite loop). Simplify it."
    else:
        payload["error_traceback"] = result.error_traceback
        if result.stdout.strip():
            payload["stdout"] = result.stdout
    if repeated:
        payload["repeat_warning"] = (
            "You already ran this exact code and it failed the same way. Do NOT "
            "repeat it — change your approach (different column, method, or logic)."
        )
    _add_plan_reminder(payload, plan)
    return payload


def _add_plan_reminder(payload: dict, plan: list[str] | None) -> None:
    """Attach a compact reminder to keep a multi-step plan on track."""
    if plan:
        payload["plan_reminder"] = (
            "Keep working through your plan; only call final_answer once every "
            "planned step is addressed with real results."
        )
