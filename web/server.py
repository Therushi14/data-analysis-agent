"""FastAPI web app for the Data-Analysis Agent — the primary UI.

A hand-built frontend (served from ``web/static``) that reuses the exact same
Orchestrator / Sandbox / GeminiClient as the CLI, so behaviour can't diverge.
While the agent works, its reasoning trace streams to the browser as
newline-delimited JSON (one event per step); charts are embedded inline as
base64 data URIs so no artifact paths are exposed.

Run:
    uvicorn web.server:app --reload      # dev (auto-reload)
    python run_web.py                    # convenience launcher
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.memory import SessionMemory
from agent.orchestrator import Orchestrator, result_summary
from agent.tools.sandbox import Sandbox
from agent.types import AgentRun, Step
from config import get_settings

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_DATA = BASE_DIR.parent / "data" / "sample_sales.csv"

# Flash models are free-tier accessible; quota is per-model, so switching models
# is a quick way to get fresh daily budget. (Pro models need paid quota.)
MODEL_OPTIONS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.0-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
]

app = FastAPI(title="Data-Analysis Agent")

# In-memory dataset store — this is a single-process demo app, not multi-tenant.
# Bounded so a long session can't grow memory without limit.
_DATASETS: dict[str, dict[str, Any]] = {}
_MAX_DATASETS = 24


# --- Dataset handling -----------------------------------------------------------

def _store_dataset(df: pd.DataFrame, name: str) -> str:
    ds_id = uuid.uuid4().hex[:12]
    # Each dataset session gets its own conversation memory for follow-ups.
    _DATASETS[ds_id] = {"df": df, "name": name, "memory": SessionMemory()}
    while len(_DATASETS) > _MAX_DATASETS:
        _DATASETS.pop(next(iter(_DATASETS)), None)
    return ds_id


def _dataset_meta(ds_id: str, df: pd.DataFrame, name: str) -> dict[str, Any]:
    """A JSON-safe summary the frontend renders (schema + a small preview)."""
    preview = json.loads(df.head(20).to_json(orient="split"))  # handles NaN/np types
    return {
        "id": ds_id,
        "name": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": [{"name": str(c), "dtype": str(t)} for c, t in df.dtypes.items()],
        "preview": {"columns": preview["columns"], "data": preview["data"]},
    }


# --- Trace serialization --------------------------------------------------------

def _figure_data_uri(path: str | None) -> str | None:
    if not path:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def serialize_step(step: Step) -> dict[str, Any]:
    obs = step.observation
    obs_dict = None
    if obs is not None:
        obs_dict = {
            "ok": obs.ok,
            "timed_out": obs.timed_out,
            "result_kind": obs.result_kind,
            "stdout": obs.stdout,
            "result_repr": obs.result_repr,
            "dataframe_preview": obs.dataframe_preview,
            "error_traceback": obs.error_traceback,
            "execution_time_s": obs.execution_time_s,
            "figure": _figure_data_uri(obs.figure_path),
        }
    return {
        "index": step.index,
        "action": step.action,
        "thought": step.thought,
        "code": step.code,
        "plan": step.plan,
        "final_answer": step.final_answer,
        "is_correction": step.is_correction,
        "observation": obs_dict,
    }


def serialize_run(run: AgentRun) -> dict[str, Any]:
    return {
        "question": run.question,
        "status": run.status,
        "final_answer": run.final_answer,
        "plan": run.plan,
        "planned": run.planned,
        "n_steps": run.n_steps,
        "n_errors": run.n_errors,
        "recovered": run.recovered,
        "figure": _figure_data_uri(run.figure_path),
        "usage": run.usage,
    }


# --- The agent run, streamed ----------------------------------------------------

def _make_orchestrator(model: str, max_steps: int, keys: list[str], sandbox: Sandbox):
    """Build a real Gemini-backed orchestrator. Isolated so tests can patch it."""
    from agent.llm.gemini import GeminiClient  # noqa: PLC0415 (defer heavy import)

    settings = get_settings()
    llm = GeminiClient(
        api_keys=keys,
        model=model,
        temperature=settings.temperature,
        request_timeout_s=settings.request_timeout_s,
    )
    return Orchestrator(
        llm=llm,
        sandbox=sandbox,
        max_steps=max_steps,
        max_consecutive_failures=settings.max_consecutive_failures,
    )


def _is_rate_limit(msg: str) -> bool:
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def stream_answer(
    question: str,
    df: pd.DataFrame,
    model: str,
    max_steps: int,
    keys: list[str],
    memory: SessionMemory | None = None,
):
    """Yield NDJSON events for a single agent run, streaming steps as they happen.

    The orchestrator runs on a worker thread and pushes each Step into a queue;
    this (sync) generator drains the queue so Starlette can stream it to the
    browser. Event shapes: {"type":"step","data":{...}}, {"type":"done",...},
    {"type":"error",...}. When ``memory`` is given, prior turns seed the run and
    this turn is appended once it answers.
    """
    settings = get_settings()
    events: queue.Queue = queue.Queue()
    work_dir = Path("artifacts") / "web" / f"run-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    sandbox = Sandbox(
        work_dir=work_dir,
        timeout_s=settings.sandbox_timeout_s,
        stdout_char_cap=settings.stdout_char_cap,
    )

    def on_step(step: Step) -> None:
        events.put(("step", serialize_step(step)))

    def worker() -> None:
        try:
            orch = _make_orchestrator(model, max_steps, keys, sandbox)
            run = orch.run(question, df, on_step=on_step, memory=memory)
            if memory is not None and run.final_answer:
                memory.add(question, run.final_answer, result_summary(run))
            payload = serialize_run(run)
            payload["memory_turns"] = len(memory) if memory is not None else 0
            events.put(("done", payload))
        except Exception as e:  # noqa: BLE001 — surface any failure to the client
            msg = str(e)
            events.put((
                "error",
                {"message": msg[:500] or type(e).__name__, "rate_limited": _is_rate_limit(msg)},
            ))
        finally:
            events.put((None, None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, payload = events.get()
        if kind is None:
            break
        yield json.dumps({"type": kind, "data": payload}) + "\n"


# --- Routes ---------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config() -> dict[str, Any]:
    settings = get_settings()
    keys = settings.api_keys
    default = settings.gemini_model
    models = [default] + [m for m in MODEL_OPTIONS if m != default]
    return {
        "models": models,
        "default_model": default,
        "max_steps": settings.max_steps,
        "has_keys": bool(keys),
        "n_keys": len(keys),
    }


@app.post("/api/sample")
def load_sample() -> dict[str, Any]:
    df = pd.read_csv(DEFAULT_DATA)
    name = "sample_sales.csv (bundled)"
    ds_id = _store_dataset(df, name)
    return _dataset_meta(ds_id, df, name)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse CSV: {e}") from e
    if df.empty:
        raise HTTPException(400, "That CSV has no rows.")
    ds_id = _store_dataset(df, file.filename)
    return _dataset_meta(ds_id, df, file.filename)


class AskRequest(BaseModel):
    dataset_id: str
    question: str
    model: str | None = None
    max_steps: int | None = None


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    entry = _DATASETS.get(req.dataset_id)
    if entry is None:
        raise HTTPException(404, "Dataset not found — load a CSV or the sample first.")
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "The question is empty.")

    settings = get_settings()
    keys = settings.api_keys
    if not keys:
        raise HTTPException(400, "No GEMINI_API_KEY configured. Add it to your .env.")

    model = req.model or settings.gemini_model
    max_steps = max(1, min(int(req.max_steps or settings.max_steps), 12))

    gen = stream_answer(question, entry["df"], model, max_steps, keys, entry.get("memory"))
    return StreamingResponse(gen, media_type="application/x-ndjson")


class ResetRequest(BaseModel):
    dataset_id: str


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    """Clear a session's conversation memory (start a fresh conversation)."""
    entry = _DATASETS.get(req.dataset_id)
    if entry is None:
        raise HTTPException(404, "Dataset not found.")
    mem = entry.get("memory")
    if mem is not None:
        mem.clear()
    return {"ok": True, "memory_turns": 0}


# Static assets (css/js). Mounted last so it can't shadow the API routes above.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
