"""Session memory — prior question/answer turns, for follow-up questions.

The sandbox is stateless per run (each ``run_python`` gets a fresh subprocess
with only ``df``), so memory here is *conversational*, not computational: we
carry the prior turns' questions, answers, and a short note of what was computed,
and feed them into the next question's context. That lets the model resolve
references like "plot that" or "the same but for BMWs" — it recomputes from
``df`` using the remembered context. (Persisting the actual computed variables
would need a long-lived kernel; see plan.md as a future extension.)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """One answered question in a session."""

    question: str
    answer: str
    result_summary: str | None = None  # compact note of what was computed


@dataclass
class SessionMemory:
    """A bounded, ordered record of prior turns in one conversation."""

    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 6  # only the last N turns are surfaced, to bound context

    def add(self, question: str, answer: str | None, result_summary: str | None = None) -> None:
        q = (question or "").strip()
        if not q:
            return
        self.turns.append(
            Turn(question=q, answer=(answer or "").strip(), result_summary=result_summary)
        )

    def recent(self) -> list[Turn]:
        """The most recent turns (oldest first), capped at ``max_turns``."""
        return self.turns[-self.max_turns:]

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

    def __bool__(self) -> bool:
        return bool(self.turns)
