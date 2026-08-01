"""Generate tailored starter questions for a freshly loaded dataset.

One cheap, tool-less LLM call: we hand the model the schema + a sample and ask
for a handful of specific, answerable analytical questions grounded in the actual
columns. This kills the "blank page" problem — the UI turns them into clickable
chips. Kept resilient: a parse failure or LLM error yields an empty list, never
an exception, since suggestions are a nicety, not core to answering.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from agent.llm.base import LLMClient, user_turn
from agent.prompts import describe_dataframe

_SYSTEM = """\
You are a data analyst helping a user explore a new dataset. Given its schema and
a sample, propose {n} specific, insightful questions the user could ask about it.

Rules:
- Ground every question in the actual columns shown — never invent columns.
- Make them diverse: mix aggregation, grouping/breakdowns, comparisons, extremes,
  and (if a date/time column exists) trends over time.
- Keep each question short, natural, and answerable from this data alone.
- Return ONLY a JSON array of {n} question strings. No prose, no numbering.
"""


def suggest_questions(df: pd.DataFrame, llm: LLMClient, n: int = 4) -> list[str]:
    """Return up to `n` suggested questions for `df` (empty list on any failure)."""
    try:
        system = _SYSTEM.format(n=n)
        user = "Dataset:\n" + describe_dataframe(df)
        response = llm.generate(system, [user_turn(user)], [])
        return parse_questions(response.text or "", n)
    except Exception:  # noqa: BLE001 — suggestions must never break dataset loading
        return []


def parse_questions(text: str, n: int) -> list[str]:
    """Pull a clean list of questions out of the model's reply (JSON or lines)."""
    text = (text or "").strip()

    # Preferred: a JSON array somewhere in the reply.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            arr = json.loads(text[start : end + 1])
            items = [_clean(str(q)) for q in arr]
            picked = _dedupe([q for q in items if q])
            if picked:
                return picked[:n]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: one question per line (strip bullets / numbering / quotes).
    lines = [_clean(line) for line in text.splitlines()]
    return _dedupe([q for q in lines if len(q) >= 8 and "?" in q])[:n]


def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^[\s\-*•\d.)\]("]+', "", s)  # leading bullets / numbering / quotes
    s = s.rstrip('",')
    return s.strip().strip('"').strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in items:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out
