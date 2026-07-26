"""Unit tests for prompt construction and the observation payload."""

from __future__ import annotations

import pandas as pd

from agent.prompts import describe_dataframe, observation_payload, plan_ack_payload
from agent.types import ExecutionResult


def test_success_payload_carries_result():
    r = ExecutionResult(ok=True, result_kind="scalar", result_repr="42")
    p = observation_payload(r)
    assert p["ok"] is True
    assert p["result"] == "42"
    assert "error_traceback" not in p


def test_error_payload_has_traceback_and_no_repeat_by_default():
    r = ExecutionResult(ok=False, error_traceback="KeyError: 'x'")
    p = observation_payload(r)
    assert p["ok"] is False
    assert "KeyError" in p["error_traceback"]
    assert "repeat_warning" not in p


def test_repeat_warning_added_when_flagged():
    r = ExecutionResult(ok=False, error_traceback="KeyError: 'x'")
    p = observation_payload(r, repeated=True)
    assert "repeat_warning" in p


def test_timeout_payload_has_hint():
    r = ExecutionResult(
        ok=False, timed_out=True, error_traceback="Execution timed out after 15s."
    )
    p = observation_payload(r)
    assert p["ok"] is False
    assert "hint" in p


def test_plan_reminder_added_only_when_plan_active():
    r = ExecutionResult(ok=True, result_kind="scalar", result_repr="6")
    assert "plan_reminder" in observation_payload(r, plan=["a", "b"])
    assert "plan_reminder" not in observation_payload(r)


def test_plan_reminder_added_on_failure_too():
    r = ExecutionResult(ok=False, error_traceback="KeyError: 'x'")
    assert "plan_reminder" in observation_payload(r, plan=["a"])


def test_plan_ack_payload_echoes_steps():
    p = plan_ack_payload(["one", "two"])
    assert p["ok"] is True
    assert p["plan"] == ["one", "two"]
    assert p["steps_planned"] == 2


def test_describe_dataframe_lists_columns_and_rows():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    s = describe_dataframe(df)
    assert "a (" in s
    assert "b (" in s
    assert "Rows: 2" in s
