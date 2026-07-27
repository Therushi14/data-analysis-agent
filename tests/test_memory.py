"""Unit tests for session memory and its rendering into the prompt."""

from __future__ import annotations

import pandas as pd

from agent.memory import SessionMemory
from agent.prompts import build_initial_user_message, conversation_context

DF = pd.DataFrame({"x": [1, 2, 3]})


def test_add_and_len():
    m = SessionMemory()
    assert not m and len(m) == 0
    m.add("Q1?", "A1.")
    assert m and len(m) == 1
    assert m.turns[0].question == "Q1?"
    assert m.turns[0].answer == "A1."


def test_blank_question_is_ignored():
    m = SessionMemory()
    m.add("   ", "answer")
    assert len(m) == 0


def test_recent_is_bounded_by_max_turns():
    m = SessionMemory(max_turns=3)
    for i in range(6):
        m.add(f"Q{i}", f"A{i}")
    recent = m.recent()
    assert len(recent) == 3
    assert [t.question for t in recent] == ["Q3", "Q4", "Q5"]  # last 3, oldest first


def test_clear():
    m = SessionMemory()
    m.add("Q", "A")
    m.clear()
    assert len(m) == 0


def test_conversation_context_none_when_empty():
    assert conversation_context(None) is None
    assert conversation_context(SessionMemory()) is None


def test_conversation_context_lists_prior_turns():
    m = SessionMemory()
    m.add("Which region grew fastest?", "South, at +137%.")
    block = conversation_context(m)
    assert "Which region grew fastest?" in block
    assert "South, at +137%." in block


def test_conversation_context_truncates_long_answers():
    m = SessionMemory()
    m.add("Q", "y" * 900)
    block = conversation_context(m)
    assert "…" in block
    assert len(block) < 900


def test_initial_message_includes_memory_when_present():
    m = SessionMemory()
    m.add("Which region grew fastest?", "South.")
    msg = build_initial_user_message("Now plot that.", DF, m)
    assert "Which region grew fastest?" in msg   # prior question is carried in
    assert "follow-up" in msg.lower()
    assert "Now plot that." in msg


def test_initial_message_has_no_context_without_memory():
    msg = build_initial_user_message("Total revenue?", DF)
    assert "Earlier in this session" not in msg
    assert "Total revenue?" in msg
