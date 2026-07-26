"""Unit tests for the loop controller with a scripted fake LLM and sandbox.

No network, no subprocess: these assert the loop's control flow — happy path,
self-correction (error fed back), the step cap, and a plain-text final answer.
"""

from __future__ import annotations

import pandas as pd

from agent.llm.base import LLMResponse, LLMToolCall
from agent.orchestrator import Orchestrator
from agent.types import ExecutionResult

DF = pd.DataFrame({"x": [1, 2, 3]})


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []  # snapshot of history passed to each generate()

    def generate(self, system_prompt, history, tools):
        self.calls.append(list(history))
        return self.responses.pop(0)


class FakeSandbox:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.runs = []
        self.prepared = False

    def prepare_data(self, df):
        self.prepared = True

    def run(self, code):
        self.runs.append(code)
        if self.results:
            return self.results.pop(0)
        return ExecutionResult(ok=True, result_kind="scalar", result_repr="42")


def _call(name, args):
    return LLMResponse(text=None, tool_call=LLMToolCall(name, args))


def test_happy_path():
    llm = FakeLLM([
        _call("run_python", {"code": "result = df['x'].sum()"}),
        _call("final_answer", {"answer": "The sum is 6."}),
    ])
    sb = FakeSandbox([ExecutionResult(ok=True, result_kind="scalar", result_repr="6")])
    run = Orchestrator(llm, sb, max_steps=6).run("What is the sum?", DF)

    assert run.status == "answered"
    assert run.final_answer == "The sum is 6."
    assert run.n_steps == 2
    assert sb.prepared
    assert sb.runs == ["result = df['x'].sum()"]


def test_self_correction_feeds_error_back():
    llm = FakeLLM([
        _call("run_python", {"code": "result = df['bad']"}),
        _call("run_python", {"code": "result = df['x'].sum()"}),
        _call("final_answer", {"answer": "6"}),
    ])
    sb = FakeSandbox([
        ExecutionResult(ok=False, error_traceback="KeyError: 'bad'"),
        ExecutionResult(ok=True, result_kind="scalar", result_repr="6"),
    ])
    run = Orchestrator(llm, sb, max_steps=6).run("sum?", DF)

    assert run.status == "answered"
    assert run.n_steps == 3
    # The error observation must appear in the history for the 2nd LLM call.
    second_history = llm.calls[1]
    assert any(t.get("role") == "tool" for t in second_history)


def test_cap_reached_returns_best_effort():
    llm = FakeLLM([_call("run_python", {"code": "x = 1"}) for _ in range(10)])
    sb = FakeSandbox()  # default: ok scalar every run
    run = Orchestrator(llm, sb, max_steps=3).run("loop forever", DF)

    assert run.status == "cap_reached"
    assert run.n_steps == 3
    assert run.final_answer  # graceful best-effort answer present


def test_plain_text_final_answer():
    llm = FakeLLM([LLMResponse(text="It is 6.", tool_call=None)])
    run = Orchestrator(llm, FakeSandbox(), max_steps=6).run("sum?", DF)

    assert run.status == "answered"
    assert run.final_answer == "It is 6."
    assert run.n_steps == 1
