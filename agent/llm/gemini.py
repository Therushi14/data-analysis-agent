"""Gemini adapter — the only module that imports the vendor SDK.

Implements the LLMClient Protocol via Gemini function calling. We keep the loop
hand-rolled in the orchestrator and deliberately do NOT enable the SDK's
automatic function calling or Gemini's server-side code_execution tool: the
model's code must run in *our* sandbox (see plan.md §2, §7).

NOTE: Gemini model IDs rotate, and access depends on your billing tier. Pro
models require paid quota; the latest Flash (e.g. `gemini-3.6-flash`) works on
the free tier. Gemini 3.x requires echoing the model's thought_signature back on
function-call turns (handled in to_contents). Verify IDs against Google AI docs
if a model 404s or 429s.
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from agent.llm.base import LLMResponse, LLMToolCall

_TYPE_MAP = {
    "string": types.Type.STRING,
    "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER,
    "boolean": types.Type.BOOLEAN,
}


def _build_tool(tool_specs: list[dict[str, Any]]) -> types.Tool:
    declarations = []
    for spec in tool_specs:
        properties = {
            name: types.Schema(
                type=_TYPE_MAP.get(info["type"], types.Type.STRING),
                description=info.get("description"),
            )
            for name, info in spec["parameters"].items()
        }
        declarations.append(
            types.FunctionDeclaration(
                name=spec["name"],
                description=spec["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=spec.get("required", []),
                ),
            )
        )
    return types.Tool(function_declarations=declarations)


def to_contents(history: list[dict[str, Any]]) -> list[types.Content]:
    """Translate the provider-neutral history into Gemini `Content` objects."""
    contents: list[types.Content] = []
    for turn in history:
        role = turn["role"]
        if role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=turn["text"])])
            )
        elif role == "model":
            parts: list[types.Part] = []
            if turn.get("text"):
                parts.append(types.Part(text=turn["text"]))
            call = turn.get("tool_call")
            if call:
                part_kwargs: dict[str, Any] = {
                    "function_call": types.FunctionCall(
                        name=call["name"], args=call["args"]
                    )
                }
                # Gemini 3.x requires the model's thought_signature to be echoed
                # back on the functionCall part, or the next turn 400s.
                if call.get("signature") is not None:
                    part_kwargs["thought_signature"] = call["signature"]
                parts.append(types.Part(**part_kwargs))
            contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=turn["tool_name"], response=turn["response"]
                            )
                        )
                    ],
                )
            )
    return contents


def parse_response(response: Any) -> LLMResponse:
    """Extract text and the first tool call from a Gemini response."""
    tool_call: LLMToolCall | None = None
    text_parts: list[str] = []

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None and tool_call is None:
                args = dict(fc.args) if getattr(fc, "args", None) else {}
                signature = getattr(part, "thought_signature", None)
                tool_call = LLMToolCall(name=fc.name, args=args, signature=signature)

    usage = _usage(response)
    text = "\n".join(text_parts) if text_parts else None
    return LLMResponse(text=text, tool_call=tool_call, raw=response, usage=usage)


def _usage(response: Any) -> dict[str, Any]:
    um = getattr(response, "usage_metadata", None)
    if not um:
        return {}
    return {
        "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
        "candidate_tokens": getattr(um, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(um, "total_token_count", 0) or 0,
    }


class GeminiClient:
    """LLMClient implementation backed by the Gemini API."""

    def __init__(
        self,
        api_key: str | None,
        model: str = "gemini-3.6-flash",
        temperature: float = 0.1,
        request_timeout_s: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env or environment."
            )
        # Bound the SDK's retry/backoff and cap each request, so a rate-limited
        # (429) free-tier key fails fast with a clear error instead of hanging.
        http_options = types.HttpOptions(
            timeout=request_timeout_s * 1000,  # milliseconds
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=15.0,
                http_status_codes=[429, 503],
            ),
        )
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            tools=[_build_tool(tools)],
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=to_contents(history),
            config=config,
        )
        return parse_response(response)
