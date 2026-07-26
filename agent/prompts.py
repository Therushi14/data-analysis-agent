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
- Think about what needs to be computed, then call `run_python` with code that
  computes it. Assign the value that answers the question to a variable named
  `result` so it is captured and shown back to you.
- Inspect the observation you get back (stdout, any error traceback, a preview
  of `result`). Take more than one step when a question needs it
  (e.g. inspect -> compute -> visualize).
- If your code raises an error, read the traceback and rewrite the code. Do not
  guess around a failure; fix the actual cause.
- To make a chart, use matplotlib via `plt`; the figure is captured for you.
- Never invent numbers. Every figure you state must come from code you ran.

When, and only when, the question is fully answered from real results you
computed, call `final_answer` with a clear natural-language answer. Reference the
concrete numbers you found.
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


def observation_payload(result: ExecutionResult) -> dict:
    """Convert an ExecutionResult into the structured tool-result the model sees."""
    if result.timed_out:
        return {"ok": False, "error": result.error_traceback}
    if not result.ok:
        return {"ok": False, "error_traceback": result.error_traceback, "stdout": result.stdout}

    payload: dict = {"ok": True, "result_kind": result.result_kind}
    if result.stdout.strip():
        payload["stdout"] = result.stdout
    if result.result_repr:
        payload["result"] = result.result_repr
    if result.dataframe_preview:
        payload["preview"] = result.dataframe_preview
    if result.figure_path:
        payload["figure"] = "A figure was produced and shown to the user."
    return payload
