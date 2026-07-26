"""Convenience launcher for the web UI.

    python run_web.py            # http://127.0.0.1:8000

For auto-reload during development, prefer:
    uvicorn web.server:app --reload
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
