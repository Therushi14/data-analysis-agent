"""Tests for agent-suggested starter questions (parsing + endpoint, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.llm.base import LLMResponse
from agent.suggest import parse_questions, suggest_questions

DF = pd.DataFrame({"region": ["N", "S"], "revenue": [10, 20]})


class _FakeLLM:
    def __init__(self, text=None, raise_exc=None):
        self.text = text
        self.raise_exc = raise_exc
        self.calls = 0

    def generate(self, system_prompt, history, tools):
        self.calls += 1
        assert tools == []  # suggestions use a tool-less completion
        if self.raise_exc:
            raise self.raise_exc
        return LLMResponse(text=self.text, tool_call=None)


# --- parsing --------------------------------------------------------------------

def test_parse_json_array():
    qs = parse_questions('["What is total revenue?", "Revenue by region?"]', 4)
    assert qs == ["What is total revenue?", "Revenue by region?"]


def test_parse_json_embedded_in_prose():
    txt = 'Sure! Here you go:\n["Q one?", "Q two?"]\nHope that helps.'
    assert parse_questions(txt, 4) == ["Q one?", "Q two?"]


def test_parse_numbered_lines_fallback():
    txt = "1. What is total revenue?\n2. Which region leads?\n- Any trend over time?"
    qs = parse_questions(txt, 4)
    assert qs == ["What is total revenue?", "Which region leads?", "Any trend over time?"]


def test_parse_dedupe_and_limit():
    assert parse_questions('["A?", "A?", "B?", "C?", "D?"]', 3) == ["A?", "B?", "C?"]


def test_parse_prose_without_questions_is_empty():
    assert parse_questions("I cannot help with that.", 4) == []


# --- suggest_questions ----------------------------------------------------------

def test_suggest_returns_parsed_list():
    llm = _FakeLLM(text='["Total revenue?", "By region?"]')
    assert suggest_questions(DF, llm, n=4) == ["Total revenue?", "By region?"]


def test_suggest_never_raises_on_llm_error():
    llm = _FakeLLM(raise_exc=RuntimeError("boom"))
    assert suggest_questions(DF, llm) == []


# --- web endpoint ---------------------------------------------------------------

def test_suggest_endpoint_generates_then_caches(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from web import server

    class _Stub:
        llm_keys = ["k1"]

    fake = _FakeLLM(text='["Q1?", "Q2?"]')
    monkeypatch.setattr(server, "build_llm_client", lambda *a, **k: fake)
    monkeypatch.setattr(server, "get_settings", lambda: _Stub())

    client = TestClient(server.app)
    ds_id = server._store_dataset(DF, "t.csv")

    r1 = client.post("/api/suggest", json={"dataset_id": ds_id})
    assert r1.status_code == 200
    assert r1.json()["questions"] == ["Q1?", "Q2?"]
    assert fake.calls == 1

    r2 = client.post("/api/suggest", json={"dataset_id": ds_id})  # served from cache
    assert r2.json()["questions"] == ["Q1?", "Q2?"]
    assert fake.calls == 1  # not regenerated


def test_suggest_endpoint_unknown_dataset_404():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from web import server

    r = TestClient(server.app).post("/api/suggest", json={"dataset_id": "nope"})
    assert r.status_code == 404
