"""Streamlit UI for the Data-Analysis Agent.

    streamlit run app.py

Upload a CSV (or use the bundled sample), ask a question in plain English, and
watch the agent write Python, run it in the sandbox, and answer with any table
or chart it produced. Reuses the same Orchestrator/Sandbox/LLM client as the
CLI, so behavior can't diverge.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.llm.base import LLMError, LLMRateLimitError
from agent.llm.factory import build_llm_client, provider_models
from agent.memory import SessionMemory
from agent.orchestrator import Orchestrator, result_summary
from agent.tools.sandbox import Sandbox
from agent.types import Step
from config import get_settings

DEFAULT_DATA = Path(__file__).parent / "data" / "sample_sales.csv"

st.set_page_config(page_title="Data-Analysis Agent", page_icon="📊", layout="wide")


def resolve_keys(settings) -> list[str]:
    """Keys for the active provider, from Streamlit secrets (deploy) or .env (local)."""

    def from_secrets(name: str):
        try:
            return st.secrets.get(name)
        except Exception:
            return None

    if settings.llm_provider == "groq":
        names = ("GROQ_API_KEY", "GROQ_API_KEY_BACKUP")
    else:
        names = ("GEMINI_API_KEY", "GEMINI_API_KEY_BACKUP")

    keys: list[str] = []
    for name in names:
        value = from_secrets(name) or getattr(settings, name.lower(), None)
        if value:
            keys.extend(k.strip() for k in value.split(",") if k.strip())
    return keys


def render_step(step: Step) -> None:
    """Render one loop step as it completes (live trace)."""
    icon = {"plan": "📋", "run_python": "🐍", "final_answer": "✅"}.get(step.action, "•")
    label = f"{icon} Step {step.index} — {step.action}"
    if step.is_correction:
        label = f"🔧 Step {step.index} — {step.action} (self-correcting)"
    with st.expander(label, expanded=True):
        if step.is_correction:
            st.caption("↳ retrying after the previous step's error")
        if step.thought:
            st.markdown(step.thought)
        if step.plan:
            st.markdown("**Plan**")
            st.markdown("\n".join(f"{i}. {task}" for i, task in enumerate(step.plan, 1)))
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

        default_model = settings.active_model
        options = provider_models(settings.llm_provider)
        model_options = [default_model] + [m for m in options if m != default_model]
        model = st.selectbox(
            "Model",
            model_options,
            index=0,
            help=f"Provider: {settings.llm_provider}. Switch models for different speed/quality.",
        )
        max_steps = st.slider("Max steps", 1, 12, settings.max_steps)

        if st.button("🗑 New conversation", help="Forget prior questions in this session"):
            st.session_state["history"] = []
            st.session_state["memory"] = SessionMemory()
            st.rerun()

        if keys:
            st.success(
                f"{settings.llm_provider}: {len(keys)} API key(s) loaded"
                + (" · failover on" if len(keys) > 1 else "")
            )
        else:
            provider_key = "GROQ_API_KEY" if settings.llm_provider == "groq" else "GEMINI_API_KEY"
            st.error(f"No {provider_key} found. Add it to `.env` or Streamlit secrets.")

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

    # Session memory for follow-up questions. Switching datasets starts fresh.
    if source and st.session_state.get("_source") not in (None, source):
        st.session_state["history"] = []
        st.session_state["memory"] = SessionMemory()
    st.session_state["_source"] = source
    memory = st.session_state.setdefault("memory", SessionMemory())

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
    if memory:
        st.caption(
            f"🧠 Remembering {len(memory)} previous question(s) — ask a follow-up "
            "like “now just for the BMWs” or “plot that”."
        )
    question = st.text_area(
        "Question",
        placeholder="e.g. Which region grew fastest from January to April by revenue? Plot it.",
        label_visibility="collapsed",
    )
    ask = st.button("Ask", type="primary", disabled=not keys)

    if not (ask and question.strip()):
        return

    work_dir = Path("artifacts") / "ui" / f"run-{int(time.time())}"
    sandbox = Sandbox(
        work_dir=work_dir,
        timeout_s=settings.sandbox_timeout_s,
        stdout_char_cap=settings.stdout_char_cap,
    )
    llm = build_llm_client(settings, model=model, api_keys=keys)
    orchestrator = Orchestrator(
        llm=llm,
        sandbox=sandbox,
        max_steps=max_steps,
        max_consecutive_failures=settings.max_consecutive_failures,
    )

    st.subheader("🧠 Reasoning trace")
    trace_area = st.container()

    def on_step(step: Step) -> None:
        with trace_area:
            render_step(step)

    with st.spinner(f"Running on {model} …"):
        try:
            run = orchestrator.run(question.strip(), df, on_step=on_step, memory=memory)
        except LLMRateLimitError:
            st.error(
                "All configured keys hit their rate limit. Wait a moment, add another "
                "key (comma-separated), or switch the model in the sidebar."
            )
            return
        except LLMError as e:
            st.error(f"LLM error: {str(e)[:400]}")
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

    if run.n_errors:
        if run.recovered:
            st.info(f"🔧 Self-corrected: hit {run.n_errors} error(s) and recovered.")
        else:
            st.warning(f"Hit {run.n_errors} error(s) and could not recover.")

    plan_note = f"planned {len(run.plan)} steps · " if run.planned else ""
    st.caption(
        f"{plan_note}status: {run.status} · steps: {run.n_steps} · errors: {run.n_errors} · "
        f"tokens: {run.usage.get('total_tokens', '?')} · model: {model}"
    )

    memory.add(question.strip(), run.final_answer, result_summary(run))
    history.append(
        {
            "question": question.strip(),
            "answer": run.final_answer,
            "figure": run.figure_path,
        }
    )


if __name__ == "__main__":
    main()
