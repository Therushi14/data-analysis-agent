"""Per-question result cache — the quota accumulator.

Each answered question is written to disk immediately, keyed by (tag, model,
question id). Re-runs skip cached questions, so a run interrupted by the daily
rate limit just resumes tomorrow, and the report scores whatever has piled up.
`tag` namespaces a cache so you can hold a "baseline" run beside an "improved"
one for a before/after comparison.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).parent / ".cache"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "default"


class Cache:
    def __init__(self, tag: str = "default", model: str = "model", root: Path | str | None = None):
        self.dir = Path(root or DEFAULT_ROOT) / _slug(tag) / _slug(model)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, qid: str) -> Path:
        return self.dir / f"{qid}.json"

    def load(self, qid: str) -> dict[str, Any] | None:
        p = self._path(qid)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, qid: str, record: dict[str, Any]) -> None:
        self._path(qid).write_text(json.dumps(record, indent=2), encoding="utf-8")

    def ids(self) -> set[str]:
        return {p.stem for p in self.dir.glob("*.json")}
