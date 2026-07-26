"""Sandbox worker — runs INSIDE an isolated subprocess. Not imported by the app.

Responsibilities:
  * load the session DataFrame as `df`
  * exec the model-written code with a restricted importer (deny-list)
  * capture stdout, exceptions, the `result` variable, and any matplotlib figure
  * write a single JSON envelope (mapping 1:1 to ExecutionResult) to --out

This file must stay dependency-light and self-contained (no project imports) so
it starts fast and stays isolated. Security posture is defense-in-depth, not a
jail — see plan.md §7 (container isolation is the documented hardening path).
"""

from __future__ import annotations

import argparse
import builtins as _builtins
import contextlib
import io
import json
import time
import traceback

import matplotlib

matplotlib.use("Agg")  # headless: render figures to file, never a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULT_REPR_CAP = 2000
PREVIEW_ROWS = 10

# Modules the model's code may NOT import. Blocks filesystem escape, process
# spawning, and all network access at import time. (pandas/numpy/matplotlib were
# already imported above by trusted code, so their internals are unaffected.)
DENY = {
    "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib",
    "http", "ftplib", "ctypes", "cffi", "importlib", "pickle", "marshal",
    "tempfile", "glob", "pathlib", "threading", "multiprocessing", "asyncio",
    "signal", "resource", "webbrowser", "smtplib", "ssl", "builtins",
}

_real_import = _builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = (name or "").split(".")[0]
    if top in DENY:
        raise ImportError(f"import of '{top}' is blocked by the sandbox policy")
    return _real_import(name, globals, locals, fromlist, level)


def _load_df(data_path: str) -> pd.DataFrame:
    if data_path.endswith(".parquet"):
        return pd.read_parquet(data_path)
    return pd.read_pickle(data_path)


def _describe_result(ns: dict) -> tuple[str, str | None, str | None]:
    """Return (result_kind, result_repr, dataframe_preview) from the exec namespace."""
    result = ns.get("result", None)
    if isinstance(result, pd.DataFrame):
        preview = result.head(PREVIEW_ROWS).to_string()
        return "dataframe", str(result)[:RESULT_REPR_CAP], preview
    if isinstance(result, pd.Series):
        preview = result.head(PREVIEW_ROWS).to_string()
        return "series", str(result)[:RESULT_REPR_CAP], preview
    if result is not None:
        return "scalar", repr(result)[:RESULT_REPR_CAP], None
    return "none", None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--stdout-cap", type=int, default=8000)
    args = parser.parse_args()

    envelope: dict = {
        "ok": False,
        "stdout": "",
        "error_traceback": None,
        "result_kind": "none",
        "result_repr": None,
        "dataframe_preview": None,
        "figure_path": None,
        "execution_time_s": 0.0,
    }

    try:
        with open(args.code, encoding="utf-8") as f:
            code = f.read()
        df = _load_df(args.data)
    except Exception:
        envelope["error_traceback"] = traceback.format_exc()
        _write(args.out, envelope)
        return

    safe_builtins = dict(vars(_builtins))
    safe_builtins["__import__"] = _guarded_import
    safe_globals = {"__builtins__": safe_builtins, "pd": pd, "np": np, "plt": plt, "df": df}

    buf = io.StringIO()
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<agent_code>", "exec"), safe_globals)  # noqa: S102
        envelope["ok"] = True
    except Exception:
        envelope["error_traceback"] = traceback.format_exc()
    envelope["execution_time_s"] = round(time.perf_counter() - start, 4)
    envelope["stdout"] = buf.getvalue()[: args.stdout_cap]

    if envelope["ok"]:
        kind, repr_, preview = _describe_result(safe_globals)
        envelope["result_kind"] = kind
        envelope["result_repr"] = repr_
        envelope["dataframe_preview"] = preview

    # Save a figure if the code created one (best effort, even on error).
    try:
        if plt.get_fignums():
            plt.savefig(args.figure, bbox_inches="tight", dpi=110)
            envelope["figure_path"] = args.figure
            if envelope["result_kind"] == "none" and envelope["ok"]:
                envelope["result_kind"] = "figure"
    except Exception:
        pass

    _write(args.out, envelope)


def _write(path: str, envelope: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f)


if __name__ == "__main__":
    main()
