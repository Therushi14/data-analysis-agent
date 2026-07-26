"""Typed application configuration, loaded from environment / .env.

All tunables live here so the orchestrator, sandbox, and LLM adapter never read
os.environ directly. `gemini_api_key` is optional at import time so the modules
(and the test suite) can be imported without a key; the Gemini adapter raises a
clear error if it is actually needed and missing.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets
    gemini_api_key: str | None = None

    # LLM. Default to the latest Flash: Pro tiers require paid quota, and older
    # 2.5 models 404 for newly-created keys. Override GEMINI_MODEL in .env.
    gemini_model: str = "gemini-3.6-flash"
    temperature: float = 0.1
    request_timeout_s: int = 60  # hard per-request cap so a call can't hang forever

    # Agent loop
    max_steps: int = 6

    # Sandbox
    sandbox_timeout_s: int = 15
    stdout_char_cap: int = 8000


def get_settings() -> Settings:
    """Return a fresh Settings instance (reads .env / environment)."""
    return Settings()
