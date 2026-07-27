"""Tests for the FastAPI web layer (no network, no API key).

Covers trace serialization, the dataset endpoints, and the streamed /api/ask
endpoint with a fake orchestrator patched in — so the streaming plumbing is
exercised without spending any Gemini quota.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from agent.types import AgentRun, ExecutionResult, Step  # noqa: E402
from web import server  # noqa: E402

client = TestClient(server.app)


# --- Serialization --------------------------------------------------------------

def test_serialize_step_includes_observation_fields():
    step = Step(
        index=2,
        action="run_python",
        code="result = df['x'].sum()",
        observation=ExecutionResult(ok=True, result_kind="scalar", result_repr="6"),
        is_correction=True,
    )
    d = server.serialize_step(step)
    assert d["index"] == 2
    assert d["action"] == "run_python"
    assert d["is_correction"] is True
    assert d["observation"]["ok"] is True
    assert d["observation"]["result_repr"] == "6"
    assert d["observation"]["figure"] is None


def test_serialize_run_exposes_summary():
    run = AgentRun(question="q", status="answered", final_answer="6", plan=["a", "b"])
    d = server.serialize_run(run)
    assert d["status"] == "answered"
    assert d["planned"] is True
    assert d["plan"] == ["a", "b"]
    assert d["final_answer"] == "6"


# --- Dataset endpoints ----------------------------------------------------------

def test_config_endpoint_shape():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and body["models"]
    assert "has_keys" in body


def test_sample_then_upload_roundtrip():
    r = client.post("/api/sample")
    assert r.status_code == 200
    meta = r.json()
    assert meta["rows"] > 0 and meta["cols"] > 0
    assert meta["id"] in server._DATASETS

    csv = b"a,b\n1,x\n2,y\n"
    up = client.post("/api/upload", files={"file": ("t.csv", csv, "text/csv")})
    assert up.status_code == 200
    assert up.json()["columns"][0]["name"] == "a"


def test_upload_rejects_non_csv():
    r = client.post("/api/upload", files={"file": ("t.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_ask_unknown_dataset_404():
    r = client.post("/api/ask", json={"dataset_id": "nope", "question": "hi"})
    assert r.status_code == 404


# --- Streamed run with a fake orchestrator --------------------------------------

class _FakeOrchestrator:
    def run(self, question, df, on_step=None, memory=None):
        s1 = Step(index=1, action="plan", plan=["Compute sum"])
        s2 = Step(
            index=2, action="run_python", code="result = df['a'].sum()",
            observation=ExecutionResult(ok=True, result_kind="scalar", result_repr="3"),
        )
        if on_step:
            on_step(s1)
            on_step(s2)
        return AgentRun(
            question=question, status="answered", final_answer="The sum is 3.",
            plan=["Compute sum"], steps=[s1, s2],
        )


def test_ask_streams_steps_then_done(monkeypatch):
    monkeypatch.setattr(server, "_make_orchestrator", lambda *a, **k: _FakeOrchestrator())
    # A dataset must exist and a key must appear present.
    ds_id = server._store_dataset(_df(), "t.csv")
    monkeypatch.setattr(
        server, "get_settings",
        lambda: _StubSettings(),
    )

    with client.stream("POST", "/api/ask",
                       json={"dataset_id": ds_id, "question": "sum of a?"}) as r:
        assert r.status_code == 200
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]

    types = [e["type"] for e in events]
    assert types == ["step", "step", "done"]
    assert events[0]["data"]["action"] == "plan"
    assert events[-1]["data"]["status"] == "answered"
    assert events[-1]["data"]["final_answer"] == "The sum is 3."


def test_ask_without_keys_is_400(monkeypatch):
    ds_id = server._store_dataset(_df(), "t.csv")
    monkeypatch.setattr(server, "get_settings", lambda: _StubSettings(keys=[]))
    r = client.post("/api/ask", json={"dataset_id": ds_id, "question": "hi"})
    assert r.status_code == 400


# --- Session memory across turns ------------------------------------------------

def _drain(ds_id, question):
    with client.stream("POST", "/api/ask",
                       json={"dataset_id": ds_id, "question": question}) as r:
        return [json.loads(line) for line in r.iter_lines() if line.strip()]


def test_memory_persists_and_seeds_followups(monkeypatch):
    seen_lengths = []

    class _Fake:
        def run(self, question, df, on_step=None, memory=None):
            seen_lengths.append(len(memory) if memory is not None else -1)
            return AgentRun(question=question, status="answered", final_answer=f"ans:{question}")

    monkeypatch.setattr(server, "_make_orchestrator", lambda *a, **k: _Fake())
    monkeypatch.setattr(server, "get_settings", lambda: _StubSettings())
    ds_id = server._store_dataset(_df(), "t.csv")

    _drain(ds_id, "first?")
    ev2 = _drain(ds_id, "second?")

    assert seen_lengths == [0, 1]                       # 2nd run saw 1 remembered turn
    assert len(server._DATASETS[ds_id]["memory"]) == 2  # both turns recorded
    assert ev2[-1]["data"]["memory_turns"] == 2


def test_reset_clears_memory(monkeypatch):
    class _Fake:
        def run(self, question, df, on_step=None, memory=None):
            return AgentRun(question=question, status="answered", final_answer="a")

    monkeypatch.setattr(server, "_make_orchestrator", lambda *a, **k: _Fake())
    monkeypatch.setattr(server, "get_settings", lambda: _StubSettings())
    ds_id = server._store_dataset(_df(), "t.csv")

    _drain(ds_id, "q?")
    assert len(server._DATASETS[ds_id]["memory"]) == 1

    r = client.post("/api/reset", json={"dataset_id": ds_id})
    assert r.status_code == 200
    assert len(server._DATASETS[ds_id]["memory"]) == 0


# --- Small stubs ----------------------------------------------------------------

def _df():
    import pandas as pd
    return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})


class _StubSettings:
    def __init__(self, keys=("k1",)):
        self.api_keys = list(keys)
        self.gemini_model = "gemini-3.6-flash"
        self.max_steps = 8
        self.temperature = 0.1
        self.request_timeout_s = 60
        self.max_consecutive_failures = 3
        self.sandbox_timeout_s = 15
        self.stdout_char_cap = 8000
