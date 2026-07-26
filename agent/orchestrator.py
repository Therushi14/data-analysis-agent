"""The loop controller — the piece we hand-roll.

    reason -> (write code -> execute -> observe) -> ... -> final answer

The two feedback edges from the design doc live here:
  * an error observation is fed back so the model can self-correct
  * an intermediate result is fed back so the model can take the next step

A max-iteration cap guards against infinite loops and runaway cost.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from agent.llm.base import (
    TOOL_SPECS,
    LLMClient,
    LLMToolCall,
    model_tool_call_turn,
    tool_result_turn,
    user_turn,
)
from agent.prompts import (
    SYSTEM_PROMPT,
    build_initial_user_message,
    observation_payload,
)
from agent.tools.sandbox import Sandbox
from agent.types import AgentRun, Step

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, llm: LLMClient, sandbox: Sandbox, max_steps: int = 6) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.max_steps = max_steps

    def run(
        self,
        question: str,
        df: pd.DataFrame,
        on_step: Callable[[Step], None] | None = None,
    ) -> AgentRun:
        self.sandbox.prepare_data(df)

        run = AgentRun(question=question)
        history = [user_turn(build_initial_user_message(question, df))]
        total_usage: dict = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}

        def emit(step: Step) -> None:
            run.steps.append(step)
            if on_step is not None:
                on_step(step)

        for step_index in range(1, self.max_steps + 1):
            logger.info("step %d: asking the model...", step_index)
            response = self.llm.generate(SYSTEM_PROMPT, history, TOOL_SPECS)
            _accumulate_usage(total_usage, response.usage)

            # No tool call -> the model answered directly in text.
            if response.tool_call is None:
                answer = (response.text or "").strip()
                emit(Step(index=step_index, action="final_answer",
                          thought=response.text, final_answer=answer))
                run.final_answer = answer
                run.status = "answered"
                break

            call: LLMToolCall = response.tool_call

            if call.name == "final_answer":
                answer = str(call.args.get("answer", "")).strip()
                emit(Step(index=step_index, action="final_answer",
                          thought=response.text, final_answer=answer))
                run.final_answer = answer
                run.status = "answered"
                break

            if call.name == "run_python":
                code = str(call.args.get("code", ""))
                history.append(model_tool_call_turn(call, text=response.text))

                result = self.sandbox.run(code)
                logger.debug("step %d run_python -> %s", step_index, result.short_summary())

                history.append(tool_result_turn("run_python", observation_payload(result)))
                emit(Step(index=step_index, action="run_python", thought=response.text,
                          code=code, observation=result))
                if result.figure_path:
                    run.figure_path = result.figure_path
                if result.dataframe_preview:
                    run.final_table_md = result.dataframe_preview
                continue

            # Unknown tool — tell the model and let it recover.
            history.append(model_tool_call_turn(call, text=response.text))
            history.append(
                tool_result_turn(call.name, {"ok": False, "error": f"unknown tool '{call.name}'"})
            )
            emit(Step(index=step_index, action=call.name, thought=response.text))
        else:
            # Loop fell through without a final answer: cap reached.
            run.status = "cap_reached"
            run.final_answer = _best_effort_answer(run)

        run.usage = total_usage
        return run


def _accumulate_usage(total: dict, usage: dict | None) -> None:
    if not usage:
        return
    total["prompt_tokens"] += usage.get("prompt_tokens", 0)
    total["candidate_tokens"] += usage.get("candidate_tokens", 0)
    total["total_tokens"] += usage.get("total_tokens", 0)


def _best_effort_answer(run: AgentRun) -> str:
    """A graceful fallback when the step cap is hit before a final answer."""
    for step in reversed(run.steps):
        if step.observation and step.observation.ok:
            return (
                "I reached the step limit before finishing. Based on the last "
                f"successful result: {step.observation.short_summary()}"
            )
    return "I reached the step limit without producing a grounded result."
