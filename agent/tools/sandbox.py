"""Host side of the sandbox: spawn the worker subprocess, enforce a timeout,
scrub the environment, and parse the worker's JSON envelope into an
ExecutionResult. The app process never execs model code itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd

from agent.types import ExecutionResult

_WORKER = Path(__file__).parent / "worker.py"

# Env var name fragments whose values must never reach sandboxed code.
_SENSITIVE = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "API")


def _scrubbed_env() -> dict[str, str]:
    """A copy of the environment with obvious secrets removed and Agg forced."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(frag in k.upper() for frag in _SENSITIVE)
    }
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class Sandbox:
    """Executes model-written Python out of process against a prepared DataFrame."""

    def __init__(
        self,
        work_dir: str | Path,
        timeout_s: int = 15,
        stdout_char_cap: int = 8000,
        python_executable: str | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.stdout_char_cap = stdout_char_cap
        self.python = python_executable or sys.executable
        self.data_path: Path | None = None

    def prepare_data(self, df: pd.DataFrame) -> None:
        """Serialize the DataFrame once per session (parquet, pickle fallback)."""
        parquet = self.work_dir / "data.parquet"
        try:
            df.to_parquet(parquet)
            self.data_path = parquet
        except Exception:
            pkl = self.work_dir / "data.pkl"
            df.to_pickle(pkl)
            self.data_path = pkl

    def run(self, code: str) -> ExecutionResult:
        if self.data_path is None:
            raise RuntimeError("Sandbox.prepare_data() must be called before run().")

        run_id = uuid.uuid4().hex[:8]
        code_path = self.work_dir / f"cell_{run_id}.py"
        out_path = self.work_dir / f"envelope_{run_id}.json"
        fig_path = self.work_dir / f"figure_{run_id}.png"
        code_path.write_text(code, encoding="utf-8")

        cmd = [
            self.python,
            str(_WORKER),
            "--data", str(self.data_path),
            "--code", str(code_path),
            "--out", str(out_path),
            "--figure", str(fig_path),
            "--stdout-cap", str(self.stdout_char_cap),
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.work_dir),
                env=_scrubbed_env(),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                ok=False,
                timed_out=True,
                error_traceback=f"Execution timed out after {self.timeout_s}s.",
                execution_time_s=float(self.timeout_s),
            )

        if not out_path.exists():
            # Worker crashed before writing an envelope (e.g. segfault, OOM kill).
            return ExecutionResult(
                ok=False,
                error_traceback=(
                    "Sandbox worker produced no result.\n"
                    f"exit code: {proc.returncode}\nstderr:\n{proc.stderr}"
                ),
            )

        envelope = json.loads(out_path.read_text(encoding="utf-8"))
        figure_path = envelope.get("figure_path")
        if figure_path and not Path(figure_path).exists():
            figure_path = None

        return ExecutionResult(
            ok=envelope["ok"],
            stdout=envelope.get("stdout", ""),
            error_traceback=envelope.get("error_traceback"),
            result_kind=envelope.get("result_kind", "none"),
            result_repr=envelope.get("result_repr"),
            dataframe_preview=envelope.get("dataframe_preview"),
            figure_path=figure_path,
            execution_time_s=envelope.get("execution_time_s", 0.0),
        )
