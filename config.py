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

    # Which LLM provider to use: "groq" (default) or "gemini".
    llm_provider: str = "groq"

    # --- Groq (default). Generous free-tier request limits. A key value may be a
    # comma-separated list for automatic failover; a backup key is also honored.
    groq_api_key: str | None = None
    groq_api_key_backup: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Gemini (alternative). A backup key (typically a different project) is
    # used automatically when the primary hits its quota (per-project-per-model).
    gemini_api_key: str | None = None
    gemini_api_key_backup: str | None = None
    gemini_model: str = "gemini-3.6-flash"

    temperature: float = 0.1
    request_timeout_s: int = 60  # hard per-request cap so a call can't hang forever

    @staticmethod
    def _split(*values: str | None) -> list[str]:
        """Flatten comma-separated key strings into a de-duplicated ordered list."""
        out: list[str] = []
        for value in values:
            if not value:
                continue
            for part in value.split(","):
                part = part.strip()
                if part and part not in out:
                    out.append(part)
        return out

    @property
    def api_keys(self) -> list[str]:
        """Gemini keys in priority order (primary first), empties dropped."""
        return self._split(self.gemini_api_key, self.gemini_api_key_backup)

    @property
    def groq_keys(self) -> list[str]:
        """Groq keys in priority order (comma-separated values are expanded)."""
        return self._split(self.groq_api_key, self.groq_api_key_backup)

    @property
    def llm_keys(self) -> list[str]:
        """Keys for the active provider."""
        return self.groq_keys if self.llm_provider == "groq" else self.api_keys

    @property
    def active_model(self) -> str:
        """Default model for the active provider."""
        return self.groq_model if self.llm_provider == "groq" else self.gemini_model

    # Agent loop. 8 gives a planned, multi-part question room to finish (one step
    # for the plan, plus a few compute/visualize steps) — within the 5-8 the
    # design doc suggests. Simple questions still finish well under the cap.
    max_steps: int = 8
    # Repair budget: give up after this many *consecutive* failed executions
    # (resets on any success). Stops a broken question from burning every step.
    max_consecutive_failures: int = 3

    # Sandbox
    sandbox_timeout_s: int = 15
    stdout_char_cap: int = 8000


def get_settings() -> Settings:
    """Return a fresh Settings instance (reads .env / environment)."""
    return Settings()
