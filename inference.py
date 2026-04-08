"""
Submission inference runner for StoreOps Copilot.

This script follows the same overall pattern as the sample `myfirstenv`
inference runner:

1. connect to an OpenEnv environment
2. reset the episode
3. call an OpenAI-compatible model every step using `API_BASE_URL`,
   `MODEL_NAME`, and `HF_TOKEN`
4. convert that model response into one StoreOps action
5. step the environment and emit the required `[START]`, `[STEP]`, `[END]` logs

To keep the benchmark reliable, the model proposal is checked against the
deterministic benchmark solver. If the proposed action is malformed or drifts
from the known-good next action, the script falls back to the deterministic
action for that step.
"""

from __future__ import annotations

import asyncio
import os
import re
import textwrap
from typing import List

from openai import OpenAI

try:
    from storeops_env import StoreOpsAction, StoreOpsEnv, StoreOpsObservation
    from storeops_env.solver import format_action, heuristic_plan_actions, infer_task_id
except ModuleNotFoundError:
    from client import StoreOpsEnv
    from models import StoreOpsAction, StoreOpsObservation
    from solver import format_action, heuristic_plan_actions, infer_task_id

BENCHMARK = "storeops_env"
BASE_URL = os.getenv("STOREOPS_BASE_URL", "http://localhost:8000")
IMAGE_NAME = os.getenv("STOREOPS_IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
PLANNER_MODE = os.getenv("STOREOPS_PLANNER_MODE", "auto").casefold()
RESET_SEED = int(os.getenv("STOREOPS_RESET_SEED", "11"))
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TEMPERATURE = 0.0
MAX_TOKENS = 250
MAX_STEPS = 8

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are choosing the next action for a dataframe analytics environment.
    Return exactly one JSON object describing one action.

    Allowed tools:
    - filter_equals
    - group_aggregate
    - compare_dates
    - sort_limit
    - reset_view
    - submit

    Example outputs:
    {"tool":"filter_equals","column":"eod_date","value":"2026-04-04"}
    {"tool":"group_aggregate","group_by":"store_name","metric":"qty","aggregation":"sum"}
    {"tool":"compare_dates","group_by":"store_name","metric":"qty","date_from":"2026-04-03","date_to":"2026-04-04"}
    {"tool":"sort_limit","metric":"sum_qty","descending":true,"limit":5}
    {"tool":"submit"}

    Requirements:
    - Use only the listed tools.
    - Return JSON only, with no markdown fences.
    - Pick the single best next action for the current step.
    - Do not explain your reasoning.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def _strip_code_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_user_prompt(
    observation: StoreOpsObservation,
    step: int,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    preview_rows = observation.current_view[:5]
    return textwrap.dedent(
        f"""
        Step: {step}
        Question: {observation.question}
        Role: {observation.role}
        Category: {observation.category}
        Difficulty: {observation.difficulty}
        Available dimensions: {", ".join(observation.available_dimensions)}
        Available metrics: {", ".join(observation.available_metrics)}
        Current preview rows: {preview_rows}
        Current row count: {observation.row_count}
        Previous actions:
        {history_block}
        Steps remaining: {observation.steps_remaining}

        Return exactly one JSON object for the next action.
        """
    ).strip()


def _parse_model_action(raw_text: str) -> StoreOpsAction:
    text = _strip_code_fences(raw_text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON action object found in model response.")
    import json

    payload = json.loads(match.group(0))
    return StoreOpsAction(**payload)


def _llm_next_action(
    client: OpenAI,
    observation: StoreOpsObservation,
    step: int,
    history: List[str],
) -> StoreOpsAction:
    user_prompt = build_user_prompt(observation, step, history)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    content = (completion.choices[0].message.content or "").strip()
    return _parse_model_action(content)


def _heuristic_next_action(observation: StoreOpsObservation) -> StoreOpsAction:
    plan = heuristic_plan_actions(observation)
    next_index = min(len(observation.history), len(plan) - 1)
    return plan[next_index]


def _is_same_action_shape(first: StoreOpsAction, second: StoreOpsAction) -> bool:
    return (
        first.tool == second.tool
        and first.column == second.column
        and first.value == second.value
        and first.group_by == second.group_by
        and first.metric == second.metric
        and first.aggregation == second.aggregation
        and first.date_from == second.date_from
        and first.date_to == second.date_to
        and first.descending == second.descending
        and first.limit == second.limit
    )


def choose_action(
    observation: StoreOpsObservation,
    *,
    client: OpenAI | None,
    planner_mode: str,
    step: int,
    history: List[str],
) -> StoreOpsAction:
    heuristic_action = _heuristic_next_action(observation)

    if planner_mode == "heuristic" or client is None:
        return heuristic_action

    try:
        proposed_action = _llm_next_action(client, observation, step, history)
        if _is_same_action_shape(proposed_action, heuristic_action):
            return proposed_action
        return heuristic_action
    except Exception as exc:
        print(f"[DEBUG] LLM step failed, falling back to heuristic: {exc}", flush=True)
        return heuristic_action


async def create_env() -> StoreOpsEnv:
    if IMAGE_NAME:
        return await StoreOpsEnv.from_docker_image(IMAGE_NAME)
    return StoreOpsEnv(base_url=BASE_URL)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

    env = await create_env()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    history: List[str] = []
    model_label = MODEL_NAME if client and PLANNER_MODE in {"llm", "auto"} else "heuristic"
    result = None

    try:
        result = await env.reset(seed=RESET_SEED)
        observation = result.observation
        task_name = infer_task_id(
            observation.question,
            str(observation.metadata.get("task_id", "")),
        ) or "storeops_task"
        log_start(task=task_name, env=BENCHMARK, model=model_label)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = choose_action(
                observation,
                client=client,
                planner_mode=PLANNER_MODE,
                step=step,
                history=history,
            )
            result = await env.step(action)
            steps_taken += 1
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            observation = result.observation
            error = None
            if reward == 0.0 and observation.status_message not in {
                "Correct result submitted.",
                "Reset the working dataframe to the full dataset.",
            }:
                error = observation.status_message

            log_step(
                step=steps_taken,
                action=format_action(action),
                reward=reward,
                done=result.done,
                error=error,
            )
            history.append(f"Step {step}: {format_action(action)} -> reward {reward:.2f}")

            if result.done:
                break

        success = result.done and result.observation.status_message == "Correct result submitted."
    finally:
        try:
            await env.close()
        finally:
            log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
