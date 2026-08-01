"""Pure translation tests for the Groq adapter (no network).

Skipped if the groq SDK is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("groq")

from agent.llm.base import (  # noqa: E402
    TOOL_SPECS,
    LLMRateLimitError,
    LLMToolCall,
    model_tool_call_turn,
    tool_result_turn,
    user_turn,
)
from agent.llm.groq_client import (  # noqa: E402
    GroqClient,
    build_tools,
    parse_response,
    to_messages,
)


# --- Tool + history translation -------------------------------------------------

def test_build_tools_openai_shape_with_array_param():
    tools = build_tools(TOOL_SPECS)
    names = [t["function"]["name"] for t in tools]
    assert names == ["plan", "run_python", "final_answer"]
    assert tools[0]["type"] == "function"
    steps = tools[0]["function"]["parameters"]["properties"]["steps"]
    assert steps["type"] == "array"
    assert steps["items"]["type"] == "string"


def test_to_messages_links_tool_result_to_call_id():
    history = [
        user_turn("hi"),
        model_tool_call_turn(LLMToolCall("run_python", {"code": "x = 1"})),
        tool_result_turn("run_python", {"ok": True}),
    ]
    msgs = to_messages("SYS", history)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "run_python"
    call_id = msgs[2]["tool_calls"][0]["id"]
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == call_id  # result references the call


# --- Response parsing -----------------------------------------------------------

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.id = "call_1"
        self.function = _Fn(name, arguments)


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class _Resp:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]
        self.usage = _Usage()


def test_parse_response_extracts_tool_call_from_json_arguments():
    resp = _Resp(_Message(tool_calls=[_ToolCall("run_python", '{"code": "x = 1"}')]))
    r = parse_response(resp)
    assert r.tool_call is not None
    assert r.tool_call.name == "run_python"
    assert r.tool_call.args == {"code": "x = 1"}
    assert r.usage["total_tokens"] == 15


def test_parse_response_plain_text():
    r = parse_response(_Resp(_Message(content="It is 6.")))
    assert r.text == "It is 6."
    assert r.tool_call is None


def test_parse_response_tolerates_bad_arguments():
    r = parse_response(_Resp(_Message(tool_calls=[_ToolCall("run_python", "not json")])))
    assert r.tool_call.args == {}


# --- Failover + errors ----------------------------------------------------------

def test_failover_to_backup_on_rate_limit(monkeypatch):
    gc = GroqClient(api_keys=["k1", "k2"], model="m", temperature=0.0)
    seen: list[str] = []

    def fail(**kwargs):
        seen.append("k1")
        raise RuntimeError("Error code: 429 - rate limit exceeded")

    def ok(**kwargs):
        seen.append("k2")
        return _Resp(_Message(content="hi"))

    monkeypatch.setattr(gc._clients[0].chat.completions, "create", fail)
    monkeypatch.setattr(gc._clients[1].chat.completions, "create", ok)

    gc.generate("sys", [], TOOL_SPECS)
    assert seen == ["k1", "k2"]
    assert gc._idx == 1  # sticks with the working key


def test_rate_limit_all_keys_raises_neutral_error(monkeypatch):
    gc = GroqClient(api_keys=["only"], model="m")

    def fail(**kwargs):
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(gc._clients[0].chat.completions, "create", fail)
    with pytest.raises(LLMRateLimitError):
        gc.generate("sys", [], TOOL_SPECS)


def test_non_rate_limit_error_propagates(monkeypatch):
    gc = GroqClient(api_keys=["k"], model="m")

    def boom(**kwargs):
        raise ValueError("a genuine bug in the code")

    monkeypatch.setattr(gc._clients[0].chat.completions, "create", boom)
    with pytest.raises(ValueError):
        gc.generate("sys", [], TOOL_SPECS)


def test_no_keys_raises():
    with pytest.raises(ValueError):
        GroqClient(api_keys=[])
