"""Groq adapter — implements LLMClient via Groq's OpenAI-compatible chat API.

Groq serves open models (e.g. `llama-3.3-70b-versatile`) with fast inference and
much more generous free-tier *request* limits than Gemini's flash tier, which is
why it's the default provider. We keep the ReAct loop hand-rolled in the
orchestrator and use Groq's tool-calling to expose run_python / plan / final_answer.

Differences from the Gemini adapter handled here:
  * tools use OpenAI JSON-schema shape; arguments come back as a JSON *string*.
  * each tool result must reference the assistant tool-call's `id` via
    `tool_call_id`, so we mint ids while translating the neutral history.
  * no thought-signatures (that's a Gemini 3.x concept), so `signature` stays None.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from groq import APIError as GroqAPIError
from groq import Groq

from agent.llm.base import LLMError, LLMRateLimitError, LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)


def _is_rate_limit(err: Exception) -> bool:
    """True for a quota / rate-limit (HTTP 429) error from Groq."""
    if getattr(err, "status_code", None) == 429:
        return True
    msg = str(err).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def _is_tool_use_failed(err: Exception) -> bool:
    """True when Groq rejected the model's tool call as unparseable (400).

    Llama models occasionally emit a malformed function call; Groq returns
    `tool_use_failed`. Retrying (often at a higher temperature) usually fixes it.
    """
    if getattr(err, "code", None) == "tool_use_failed":
        return True
    msg = str(err)
    return "tool_use_failed" in msg or "tool call validation failed" in msg


def build_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the neutral tool specs into OpenAI/Groq function definitions."""
    tools: list[dict[str, Any]] = []
    for spec in tool_specs:
        properties: dict[str, Any] = {}
        for name, info in spec["parameters"].items():
            prop: dict[str, Any] = {"type": info.get("type", "string")}
            if info.get("description"):
                prop["description"] = info["description"]
            if info.get("type") == "array":
                item = info.get("items", {"type": "string"})
                prop["items"] = {"type": item.get("type", "string")}
            properties[name] = prop
        tools.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": spec.get("required", []),
                },
            },
        })
    return tools


def to_messages(system_prompt: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the provider-neutral history into OpenAI/Groq chat messages."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    call_counter = 0
    last_tool_call_id = "call_0"
    for turn in history:
        role = turn["role"]
        if role == "user":
            messages.append({"role": "user", "content": turn["text"]})
        elif role == "model":
            call = turn.get("tool_call")
            text = turn.get("text") or ""
            if call:
                call_counter += 1
                last_tool_call_id = f"call_{call_counter}"
                messages.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [{
                        "id": last_tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"]),
                        },
                    }],
                })
            else:
                messages.append({"role": "assistant", "content": text})
        elif role == "tool":
            # A tool result must point back at the preceding assistant tool-call id.
            messages.append({
                "role": "tool",
                "tool_call_id": last_tool_call_id,
                "content": json.dumps(turn["response"]),
            })
    return messages


def parse_response(response: Any) -> LLMResponse:
    """Extract text + the first tool call from a Groq chat completion."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return LLMResponse(text=None, tool_call=None, raw=response, usage=_usage(response))

    message = choices[0].message
    text = getattr(message, "content", None) or None

    tool_call: LLMToolCall | None = None
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        tc = tool_calls[0]
        raw_args = getattr(tc.function, "arguments", None) or "{}"
        try:
            args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        tool_call = LLMToolCall(name=tc.function.name, args=args, signature=None)

    return LLMResponse(text=text, tool_call=tool_call, raw=response, usage=_usage(response))


def _usage(response: Any) -> dict[str, Any]:
    u = getattr(response, "usage", None)
    if not u:
        return {}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        # map to the orchestrator's neutral key (Gemini calls these "candidates").
        "candidate_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }


class GroqClient:
    """LLMClient implementation backed by the Groq API."""

    def __init__(
        self,
        api_keys: list[str] | str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        request_timeout_s: int = 60,
    ) -> None:
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        keys = [k for k in api_keys if k]
        if not keys:
            raise ValueError("No GROQ_API_KEY set. Add it to your .env or environment.")
        # One client per key; the SDK handles a couple of transient retries itself.
        self._clients = [
            Groq(api_key=k, timeout=float(request_timeout_s), max_retries=2) for k in keys
        ]
        self._idx = 0  # index of the key currently in use
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        messages = to_messages(system_prompt, history)
        tool_defs = build_tools(tools)

        n = len(self._clients)
        for attempt in range(n):
            try:
                return self._complete(messages, tool_defs)
            except Exception as e:  # noqa: BLE001 — translated to neutral errors below
                if _is_rate_limit(e):
                    if attempt < n - 1:
                        nxt = (self._idx + 1) % n
                        logger.warning(
                            "Groq key #%d hit its rate limit; failing over to key #%d",
                            self._idx + 1, nxt + 1,
                        )
                        self._idx = nxt
                        continue
                    raise LLMRateLimitError(str(e)) from e
                if isinstance(e, GroqAPIError):
                    raise LLMError(str(e)) from e
                raise
        raise RuntimeError("unreachable")  # loop always returns or raises

    def _complete(self, messages: list[dict], tool_defs: list[dict]) -> LLMResponse:
        """One completion, retrying on Groq's `tool_use_failed` with a temperature
        bump — Llama sometimes emits a malformed tool call that a re-sample fixes."""
        temperatures = [self.temperature, min(self.temperature + 0.4, 1.0), 0.7]
        last_exc: Exception | None = None
        for temp in temperatures:
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temp,
                }
                if tool_defs:  # omit for plain text completions (e.g. suggestions)
                    kwargs["tools"] = tool_defs
                    kwargs["tool_choice"] = "auto"
                response = self._clients[self._idx].chat.completions.create(**kwargs)
                return parse_response(response)
            except GroqAPIError as e:
                if _is_tool_use_failed(e):
                    last_exc = e
                    logger.warning("Groq tool_use_failed; retrying at temperature %.1f", temp)
                    continue
                raise
        raise last_exc  # retries exhausted — let generate() translate it
