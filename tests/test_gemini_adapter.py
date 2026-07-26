"""Pure translation tests for the Gemini adapter (no network).

Skipped if google-genai is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.genai")

from google.genai import types  # noqa: E402

from agent.llm.base import (  # noqa: E402
    TOOL_SPECS,
    LLMToolCall,
    model_tool_call_turn,
    tool_result_turn,
    user_turn,
)
from agent.llm.gemini import (  # noqa: E402
    GeminiClient,
    _build_tool,
    parse_response,
    to_contents,
)


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


def test_build_tool_declares_all_functions():
    tool = _build_tool(TOOL_SPECS)
    names = [d.name for d in tool.function_declarations]
    assert names == ["plan", "run_python", "final_answer"]


def test_build_tool_supports_array_param():
    # The `plan` tool's `steps` is an array-of-strings — the adapter must map it
    # to a Gemini ARRAY schema with a STRING item type.
    tool = _build_tool(TOOL_SPECS)
    plan_decl = next(d for d in tool.function_declarations if d.name == "plan")
    steps = plan_decl.parameters.properties["steps"]
    assert steps.type == types.Type.ARRAY
    assert steps.items.type == types.Type.STRING


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


class _EmptyResponse:
    candidates = []
    usage_metadata = None


def test_failover_to_backup_on_rate_limit(monkeypatch):
    gc = GeminiClient(api_keys=["key1", "key2"], model="m", temperature=0.0)
    seen: list[str] = []

    def gen_fail(model, contents, config):
        seen.append("k1")
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    def gen_ok(model, contents, config):
        seen.append("k2")
        return _EmptyResponse()

    monkeypatch.setattr(gc._clients[0].models, "generate_content", gen_fail)
    monkeypatch.setattr(gc._clients[1].models, "generate_content", gen_ok)

    gc.generate("sys", [], TOOL_SPECS)
    assert seen == ["k1", "k2"]  # tried primary, failed over to backup
    assert gc._idx == 1  # sticks with the backup key

    seen.clear()
    gc.generate("sys", [], TOOL_SPECS)
    assert seen == ["k2"]  # subsequent calls go straight to the backup


def test_non_rate_limit_error_is_not_failed_over(monkeypatch):
    gc = GeminiClient(api_keys=["key1", "key2"], model="m", temperature=0.0)

    def boom(model, contents, config):
        raise ValueError("a real bug, not a rate limit")

    monkeypatch.setattr(gc._clients[0].models, "generate_content", boom)
    with pytest.raises(ValueError):
        gc.generate("sys", [], TOOL_SPECS)


def test_no_keys_raises():
    with pytest.raises(ValueError):
        GeminiClient(api_keys=[])
