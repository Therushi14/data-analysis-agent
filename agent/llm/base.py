"""Provider-agnostic LLM abstraction.

The orchestrator talks only to `LLMClient`; it never imports a vendor SDK.
Swapping providers = writing one new adapter that satisfies this Protocol.

Conversation history is passed as a list of provider-neutral "turn" dicts:

    {"role": "user",  "text": "..."}                       # a user/text turn
    {"role": "model", "text": "...", "tool_call": {...}}   # assistant turn (text and/or a call)
    {"role": "tool",  "tool_name": "run_python", "response": {...}}  # a tool result

Helper builders below construct these so callers don't hand-write dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMToolCall:
    name: str  # "run_python" | "final_answer"
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str | None
    tool_call: LLMToolCall | None
    raw: Any = None
    usage: dict[str, Any] | None = None


class LLMClient(Protocol):
    """A stateless chat client that supports function/tool calling."""

    def generate(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        ...


# --- Neutral tool specification -------------------------------------------------
# The two tools the agent exposes. Adapters translate these into their SDK's
# function-declaration format. Keeping the spec here (not in the adapter) keeps
# the tool surface provider-independent.

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "run_python",
        "description": (
            "Execute Python against the loaded pandas DataFrame `df`. "
            "`pd` (pandas), `np` (numpy), and `plt` (matplotlib.pyplot) are available. "
            "Assign the value that answers the question to a variable named `result`. "
            "To make a chart, use matplotlib (`plt`); the figure is captured automatically. "
            "Returns stdout, any error traceback, and a preview of `result`."
        ),
        "parameters": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            }
        },
        "required": ["code"],
    },
    {
        "name": "final_answer",
        "description": (
            "Provide the final, natural-language answer to the user's question. "
            "Call this only once the data question is fully answered and grounded in "
            "results you actually computed with run_python."
        ),
        "parameters": {
            "answer": {
                "type": "string",
                "description": "The final answer in clear natural language.",
            }
        },
        "required": ["answer"],
    },
]


# --- Neutral history turn builders ---------------------------------------------

def user_turn(text: str) -> dict[str, Any]:
    return {"role": "user", "text": text}


def model_tool_call_turn(tool_call: LLMToolCall, text: str | None = None) -> dict[str, Any]:
    return {
        "role": "model",
        "text": text,
        "tool_call": {"name": tool_call.name, "args": tool_call.args},
    }


def tool_result_turn(tool_name: str, response: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_name": tool_name, "response": response}
