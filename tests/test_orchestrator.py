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


# --- Level 2: self-correction ---------------------------------------------------

def test_gives_up_after_consecutive_failures():
    # Model keeps trying, sandbox keeps failing: stop at the repair budget,
    # NOT at max_steps (so we don't burn every API call on a hopeless question).
    llm = FakeLLM([_call("run_python", {"code": f"broken_{i}"}) for i in range(10)])
    sb = FakeSandbox([
        ExecutionResult(ok=False, error_traceback="KeyError: 'x'") for _ in range(10)
    ])
    run = Orchestrator(llm, sb, max_steps=8, max_consecutive_failures=3).run("q", DF)

    assert run.status == "failed"
    assert run.n_steps == 3            # stopped at the failure cap
    assert len(sb.runs) == 3
    assert run.n_errors == 3
    assert run.recovered is False
    assert "last error" in (run.final_answer or "").lower()


def test_recovery_resets_the_failure_counter():
    # Two failures then a success must reset the counter, so the run continues
    # and answers rather than giving up.
    llm = FakeLLM([
        _call("run_python", {"code": "bad1"}),
        _call("run_python", {"code": "bad2"}),
        _call("run_python", {"code": "good"}),
        _call("final_answer", {"answer": "6"}),
    ])
    sb = FakeSandbox([
        ExecutionResult(ok=False, error_traceback="E1"),
        ExecutionResult(ok=False, error_traceback="E2"),
        ExecutionResult(ok=True, result_kind="scalar", result_repr="6"),
    ])
    run = Orchestrator(llm, sb, max_steps=8, max_consecutive_failures=3).run("q", DF)

    assert run.status == "answered"
    assert run.n_errors == 2
    assert run.recovered is True


def test_is_correction_flag_marks_steps_after_an_error():
    llm = FakeLLM([
        _call("run_python", {"code": "bad"}),
        _call("run_python", {"code": "good"}),
        _call("final_answer", {"answer": "ok"}),
    ])
    sb = FakeSandbox([
        ExecutionResult(ok=False, error_traceback="E"),
        ExecutionResult(ok=True, result_kind="scalar", result_repr="1"),
    ])
    run = Orchestrator(llm, sb, max_steps=6, max_consecutive_failures=3).run("q", DF)

    assert run.steps[0].is_correction is False  # first attempt, nothing to correct
    assert run.steps[1].is_correction is True   # follows a failure


# --- Level 3: planning / multi-step ---------------------------------------------

def test_plan_then_execute_records_and_uses_plan():
    llm = FakeLLM([
        _call("plan", {"steps": ["Compute the sum", "Report it"]}),
        _call("run_python", {"code": "result = df['x'].sum()"}),
        _call("final_answer", {"answer": "The sum is 6."}),
    ])
    sb = FakeSandbox([ExecutionResult(ok=True, result_kind="scalar", result_repr="6")])
    run = Orchestrator(llm, sb, max_steps=8).run("sum and report", DF)

    assert run.status == "answered"
    assert run.planned is True
    assert run.plan == ["Compute the sum", "Report it"]
    assert run.n_steps == 3
    # The plan is its own step, recorded before any code executes.
    assert run.steps[0].action == "plan"
    assert run.steps[0].plan == ["Compute the sum", "Report it"]
    assert run.steps[1].is_correction is False   # a plan is not a failure to recover from
    assert sb.runs == ["result = df['x'].sum()"]  # planning runs no code


def test_plan_is_fed_back_into_history():
    llm = FakeLLM([
        _call("plan", {"steps": ["A", "B"]}),
        _call("run_python", {"code": "result = 1"}),
        _call("final_answer", {"answer": "done"}),
    ])
    sb = FakeSandbox([ExecutionResult(ok=True, result_kind="scalar", result_repr="1")])
    Orchestrator(llm, sb, max_steps=8).run("q", DF)

    # The model's 2nd call must see the plan acknowledgement as a tool result.
    second_history = llm.calls[1]
    plan_turns = [
        t for t in second_history
        if t.get("role") == "tool" and t.get("tool_name") == "plan"
    ]
    assert plan_turns and plan_turns[0]["response"]["plan"] == ["A", "B"]


def test_messy_plan_steps_are_cleaned():
    # Whitespace-only / empty entries dropped; values coerced to stripped strings.
    llm = FakeLLM([
        _call("plan", {"steps": ["  Step one  ", "", "   ", "Step two"]}),
        _call("final_answer", {"answer": "ok"}),
    ])
    run = Orchestrator(llm, FakeSandbox(), max_steps=8).run("q", DF)
    assert run.plan == ["Step one", "Step two"]


def test_no_plan_leaves_run_unplanned():
    # A simple question that skips planning must still behave exactly as before.
    llm = FakeLLM([
        _call("run_python", {"code": "result = df['x'].sum()"}),
        _call("final_answer", {"answer": "6"}),
    ])
    sb = FakeSandbox([ExecutionResult(ok=True, result_kind="scalar", result_repr="6")])
    run = Orchestrator(llm, sb, max_steps=8).run("sum?", DF)

    assert run.status == "answered"
    assert run.planned is False
    assert run.plan == []


def test_repeated_failing_code_gets_a_warning():
    # Identical failing code twice -> the 2nd observation carries a repeat warning.
    llm = FakeLLM([
        _call("run_python", {"code": "df['bad']"}),
        _call("run_python", {"code": "df['bad']"}),  # byte-identical
        _call("final_answer", {"answer": "x"}),
    ])
    sb = FakeSandbox([
        ExecutionResult(ok=False, error_traceback="KeyError"),
        ExecutionResult(ok=False, error_traceback="KeyError"),
    ])
    run = Orchestrator(llm, sb, max_steps=6, max_consecutive_failures=5).run("q", DF)

    third_history = llm.calls[2]  # history sent to the 3rd model call
    tool_turns = [t for t in third_history if t.get("role") == "tool"]
    assert any("repeat_warning" in t.get("response", {}) for t in tool_turns)
    assert run.status == "answered"
