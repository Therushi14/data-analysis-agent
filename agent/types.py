"""Core data contracts shared by every component.

These dataclasses are the interfaces between the sandbox, orchestrator, UI, and
eval harness. Fix them before changing behavior; change them deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    """Structured result of running one block of model-written Python."""

    ok: bool
    stdout: str = ""
    error_traceback: str | None = None
    result_kind: str = "none"  # "none" | "scalar" | "series" | "dataframe" | "figure"
    result_repr: str | None = None
    dataframe_preview: str | None = None
    figure_path: str | None = None
    execution_time_s: float = 0.0
    timed_out: bool = False

    def short_summary(self) -> str:
        """One-line human summary, handy for logs and best-effort answers."""
        if self.timed_out:
            return "execution timed out"
        if not self.ok:
            first = (self.error_traceback or "error").strip().splitlines()
            return f"error: {first[-1] if first else 'unknown'}"
        if self.result_kind == "figure":
            return "produced a figure"
        if self.result_repr:
            return f"result: {self.result_repr[:120]}"
        if self.stdout.strip():
            return f"stdout: {self.stdout.strip()[:120]}"
        return "ran with no explicit result"


@dataclass
class Step:
    """One iteration of the reason -> act -> observe loop."""

    index: int
    action: str  # "run_python" | "final_answer"
    thought: str | None = None
    code: str | None = None
    observation: ExecutionResult | None = None
    final_answer: str | None = None
    is_correction: bool = False  # this run_python follows a failed one (self-correction)


@dataclass
class AgentRun:
    """The full record of answering one question. Rendered by the UI, scored by evals."""

    question: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    status: str = "error"  # "answered" | "cap_reached" | "failed" | "error"
    figure_path: str | None = None
    final_table_md: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_errors(self) -> int:
        """How many code executions failed across the run."""
        return sum(
            1 for s in self.steps if s.observation is not None and not s.observation.ok
        )

    @property
    def recovered(self) -> bool:
        """Errored at least once but still reached a grounded answer (self-correction won)."""
        return self.n_errors > 0 and self.status == "answered"
