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

import logging
from typing import Any

from google import genai
from google.genai import types

from agent.llm.base import LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)


def _is_rate_limit(err: Exception) -> bool:
    """True for a quota / rate-limit (429 RESOURCE_EXHAUSTED) error."""
    return getattr(err, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(err)

_TYPE_MAP = {
    "string": types.Type.STRING,
    "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER,
    "boolean": types.Type.BOOLEAN,
}


def _schema(info: dict[str, Any]) -> types.Schema:
    """Translate one neutral parameter spec into a Gemini Schema (arrays included)."""
    kind = info.get("type", "string")
    if kind == "array":
        item_info = info.get("items", {"type": "string"})
        return types.Schema(
            type=types.Type.ARRAY,
            items=_schema(item_info),
            description=info.get("description"),
        )
    return types.Schema(
        type=_TYPE_MAP.get(kind, types.Type.STRING),
        description=info.get("description"),
    )


def _build_tool(tool_specs: list[dict[str, Any]]) -> types.Tool:
    declarations = []
    for spec in tool_specs:
        properties = {
            name: _schema(info) for name, info in spec["parameters"].items()
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
        api_keys: list[str] | str,
        model: str = "gemini-3.6-flash",
        temperature: float = 0.1,
        request_timeout_s: int = 60,
    ) -> None:
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        keys = [k for k in api_keys if k]
        if not keys:
            raise ValueError(
                "No GEMINI_API_KEY set. Add it to your .env or environment."
            )
        # Bound the SDK's retry/backoff and cap each request, so a rate-limited
        # (429) key fails fast (then we fail over to the next key) instead of hanging.
        http_options = types.HttpOptions(
            timeout=request_timeout_s * 1000,  # milliseconds
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=15.0,
                http_status_codes=[429, 503],
            ),
        )
        self._clients = [genai.Client(api_key=k, http_options=http_options) for k in keys]
        self._idx = 0  # index of the key currently in use
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
        contents = to_contents(history)

        # Try each key in turn, starting from the one currently in use. On a
        # quota/rate-limit error, advance to the next key and stick with it.
        n = len(self._clients)
        for attempt in range(n):
            try:
                response = self._clients[self._idx].models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                return parse_response(response)
            except Exception as e:  # noqa: BLE001 - re-raised unless it's a rate limit
                if _is_rate_limit(e) and attempt < n - 1:
                    nxt = (self._idx + 1) % n
                    logger.warning(
                        "API key #%d hit its quota; failing over to key #%d",
                        self._idx + 1, nxt + 1,
                    )
                    self._idx = nxt
                    continue
                raise
        raise RuntimeError("unreachable")  # loop always returns or raises
