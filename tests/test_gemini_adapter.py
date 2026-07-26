"""Pure translation tests for the Gemini adapter (no network).

Skipped if google-genai is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.genai")

from agent.llm.base import (  # noqa: E402
    TOOL_SPECS,
    LLMToolCall,
    model_tool_call_turn,
    tool_result_turn,
    user_turn,
)
from agent.llm.gemini import _build_tool, parse_response, to_contents  # noqa: E402


def test_to_contents_maps_roles_and_parts():
    history = [
        user_turn("hi"),
        model_tool_call_turn(LLMToolCall("run_python", {"code": "x = 1"})),
        tool_result_turn("run_python", {"ok": True}),
    ]
    contents = to_contents(history)

    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "run_python"
    assert contents[2].parts[0].function_response.name == "run_python"


def test_build_tool_declares_both_functions():
    tool = _build_tool(TOOL_SPECS)
    names = [d.name for d in tool.function_declarations]
    assert names == ["run_python", "final_answer"]


class _FakeFunctionCall:
    name = "final_answer"
    args = {"answer": "done"}


class _FakePart:
    text = None
    function_call = _FakeFunctionCall()


class _FakeContent:
    parts = [_FakePart()]


class _FakeCandidate:
    content = _FakeContent()


class _FakeResponse:
    candidates = [_FakeCandidate()]
    usage_metadata = None


def test_parse_response_extracts_tool_call():
    result = parse_response(_FakeResponse())
    assert result.tool_call is not None
    assert result.tool_call.name == "final_answer"
    assert result.tool_call.args == {"answer": "done"}
