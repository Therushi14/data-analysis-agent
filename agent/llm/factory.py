"""Build the right LLM client for the configured provider.

This is the single place that decides Groq vs Gemini, so every entry point (CLI,
web app, Streamlit, evals) stays provider-agnostic — they ask for a client and
get one satisfying the `LLMClient` Protocol.
"""

from __future__ import annotations

from agent.llm.base import LLMClient

# Model options surfaced in the UIs, per provider. The first is the default.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.0-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
]


def provider_models(provider: str) -> list[str]:
    return GROQ_MODELS if provider == "groq" else GEMINI_MODELS


def build_llm_client(settings, model: str | None = None, api_keys: list[str] | None = None) -> LLMClient:
    """Construct the LLM client for `settings.llm_provider`.

    `api_keys` overrides the keys read from settings (used by the Streamlit app,
    which may pull keys from Streamlit secrets instead of .env).
    """
    provider = (settings.llm_provider or "groq").lower()

    if provider == "groq":
        from agent.llm.groq_client import GroqClient  # noqa: PLC0415 (defer heavy import)

        return GroqClient(
            api_keys=api_keys if api_keys is not None else settings.groq_keys,
            model=model or settings.groq_model,
            temperature=settings.temperature,
            request_timeout_s=settings.request_timeout_s,
        )

    if provider == "gemini":
        from agent.llm.gemini import GeminiClient  # noqa: PLC0415

        return GeminiClient(
            api_keys=api_keys if api_keys is not None else settings.api_keys,
            model=model or settings.gemini_model,
            temperature=settings.temperature,
            request_timeout_s=settings.request_timeout_s,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use 'groq' or 'gemini').")
