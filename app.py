"""Streamlit UI for the Data-Analysis Agent.

    streamlit run app.py

Upload a CSV (or use the bundled sample), ask a question in plain English, and
watch the agent write Python, run it in the sandbox, and answer with any table
or chart it produced. Reuses the same Orchestrator/Sandbox/GeminiClient as the
CLI, so behavior can't diverge.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.orchestrator import Orchestrator
from agent.tools.sandbox import Sandbox
from agent.types import Step
from config import get_settings

DEFAULT_DATA = Path(__file__).parent / "data" / "sample_sales.csv"

# Flash models are free-tier accessible. Quota is per-model, so switching models
# is a quick way to get fresh daily budget. (Pro models need paid quota.)
MODEL_OPTIONS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.0-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
]

st.set_page_config(page_title="Data-Analysis Agent", page_icon="📊", layout="wide")


def resolve_keys(settings) -> list[str]:
    """Keys in priority order, from Streamlit secrets (deploy) or .env (local)."""

    def from_secrets(name: str):
        try:
            return st.secrets.get(name)
        except Exception:
            return None

    primary = from_secrets("GEMINI_API_KEY") or settings.gemini_api_key
    backup = from_secrets("GEMINI_API_KEY_BACKUP") or settings.gemini_api_key_backup
    return [k for k in (primary, backup) if k]


def render_step(step: Step) -> None:
    """Render one loop step as it completes (live trace)."""
    icon = {"run_python": "🐍", "final_answer": "✅"}.get(step.action, "•")
    with st.expander(f"{icon} Step {step.index} — {step.action}", expanded=True):
        if step.thought:
            st.markdown(step.thought)
        if step.code:
            st.code(step.code, language="python")

        obs = step.observation
        if obs is not None:
            if obs.timed_out:
                st.warning(f"⏱ Execution timed out after {obs.execution_time_s}s")
            elif obs.ok:
                if obs.stdout.strip():
                    st.code(obs.stdout, language="text")
                if obs.dataframe_preview:
                    st.code(obs.dataframe_preview, language="text")
                if obs.figure_path:
                    st.image(obs.figure_path)
                if not (obs.stdout.strip() or obs.dataframe_preview or obs.figure_path):
                    if obs.result_repr:
                        st.code(obs.result_repr, language="text")
            else:
                st.error("Execution error — the agent will read this and retry.")
                if obs.error_traceback:
                    st.code(obs.error_traceback, language="text")

        if step.final_answer is not None:
            st.markdown(step.final_answer)


def main() -> None:
    st.title("📊 Data-Analysis Agent")
    st.caption(
        "Ask a question in plain English. The agent writes Python, runs it in a "
        "sandbox, reads its own errors, and answers with the table or chart it made."
    )

    settings = get_settings()
    keys = resolve_keys(settings)

    with st.sidebar:
        st.header("⚙️ Settings")

        default_model = settings.gemini_model
        model_options = [default_model] + [m for m in MODEL_OPTIONS if m != default_model]
        model = st.selectbox(
            "Model",
            model_options,
            index=0,
            help="Free tier ≈ 20 requests/day per model. Switch models for fresh quota.",
        )
        max_steps = st.slider("Max steps", 1, 12, settings.max_steps)

        if keys:
            st.success(
                f"{len(keys)} API key(s) loaded" + (" · failover on" if len(keys) > 1 else "")
            )
        else:
            st.error("No GEMINI_API_KEY found. Add it to `.env` or Streamlit secrets.")

        st.divider()
        st.subheader("Dataset")
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        use_sample = st.checkbox("Use bundled sample", value=True)

    # Resolve the dataframe.
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source = uploaded.name
    elif use_sample:
        df = pd.read_csv(DEFAULT_DATA)
        source = "sample_sales.csv (bundled)"
    else:
        df = None
        source = None

    if df is None:
        st.info("⬅️ Upload a CSV or tick **Use bundled sample** in the sidebar to begin.")
        return

    with st.expander(f"📄 Dataset preview — {source} · {len(df)} rows × {len(df.columns)} cols"):
        st.dataframe(df.head(20), use_container_width=True)

    # Prior questions this session (keeps their answers + charts visible).
    history = st.session_state.setdefault("history", [])
    if history:
        st.subheader("🕑 Previous questions")
        for item in reversed(history):
            with st.expander(f"Q: {item['question']}"):
                st.markdown(item["answer"] or "_no answer_")
                if item.get("figure"):
                    st.image(item["figure"])

    st.subheader("Ask a question")
    question = st.text_area(
        "Question",
        placeholder="e.g. Which region grew fastest from January to April by revenue? Plot it.",
        label_visibility="collapsed",
    )
    ask = st.button("Ask", type="primary", disabled=not keys)

    if not (ask and question.strip()):
        return

    from google.genai import errors as genai_errors  # noqa: PLC0415

    from agent.llm.gemini import GeminiClient  # noqa: PLC0415

    work_dir = Path("artifacts") / "ui" / f"run-{int(time.time())}"
    sandbox = Sandbox(
        work_dir=work_dir,
        timeout_s=settings.sandbox_timeout_s,
        stdout_char_cap=settings.stdout_char_cap,
    )
    llm = GeminiClient(
        api_keys=keys,
        model=model,
        temperature=settings.temperature,
        request_timeout_s=settings.request_timeout_s,
    )
    orchestrator = Orchestrator(llm=llm, sandbox=sandbox, max_steps=max_steps)

    st.subheader("🧠 Reasoning trace")
    trace_area = st.container()

    def on_step(step: Step) -> None:
        with trace_area:
            render_step(step)

    with st.spinner(f"Running on {model} …"):
        try:
            run = orchestrator.run(question.strip(), df, on_step=on_step)
        except genai_errors.APIError as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                st.error(
                    "All configured keys hit the free-tier limit. Wait for the daily "
                    "reset, switch the model in the sidebar, add a backup key, or "
                    "enable billing."
                )
            else:
                st.error(f"Gemini API error: {msg[:400]}")
            return

    st.subheader("💬 Answer")
    if run.status == "answered":
        st.success(run.final_answer or "")
    elif run.status == "cap_reached":
        st.warning(run.final_answer or "Reached the step limit before a final answer.")
    else:
        st.error(run.final_answer or "No answer produced.")

    if run.figure_path:
        st.image(run.figure_path, caption="Generated chart")

    st.caption(
        f"status: {run.status} · steps: {run.n_steps} · "
        f"tokens: {run.usage.get('total_tokens', '?')} · model: {model}"
    )

    history.append(
        {
            "question": question.strip(),
            "answer": run.final_answer,
            "figure": run.figure_path,
        }
    )


if __name__ == "__main__":
    main()
