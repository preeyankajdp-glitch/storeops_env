"""
Submission inference runner for StoreOps Copilot.

This script supports two planner modes:

- `heuristic` (default): deterministic task solver for reliable submission runs
- `llm`: asks an OpenAI-compatible model for an action plan, then falls back to
  the heuristic solver on any parsing or API error
- `auto`: use LLM planning only when API credentials are present

Environment variables:
- `STOREOPS_BASE_URL`      Base URL for a running StoreOps server
- `STOREOPS_IMAGE_NAME`    Docker image name for `from_docker_image(...)`
- `STOREOPS_PLANNER_MODE`  `heuristic`, `llm`, or `auto`
- `STOREOPS_RESET_SEED`    Deterministic seed for reproducible baseline runs
- `API_BASE_URL`           OpenAI-compatible base URL for optional LLM planning
- `MODEL_NAME`             Model name for optional LLM planning
- `HF_TOKEN` / `API_KEY`   API key for optional LLM planning

STDOUT format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
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
PLANNER_MODE = os.getenv("STOREOPS_PLANNER_MODE", "heuristic").casefold()
RESET_SEED = int(os.getenv("STOREOPS_RESET_SEED", "11"))
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TEMPERATURE = 0.0
MAX_TOKENS = 600

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are planning actions for a deterministic dataframe analytics environment.
    Return only JSON with this shape:
    {
      "actions": [
        {"tool": "filter_equals", "column": "eod_date", "value": "2026-04-04"},
        {"tool": "group_aggregate", "group_by": "store_name", "metric": "qty", "aggregation": "sum"},
        {"tool": "compare_dates", "group_by": "store_name", "metric": "qty", "date_from": "2026-04-03", "date_to": "2026-04-04"},
        {"tool": "sort_limit", "metric": "sum_qty", "descending": true, "limit": 5},
        {"tool": "submit"}
      ]
    }

    Allowed tools:
    - filter_equals
    - group_aggregate
    - compare_dates
    - sort_limit
    - reset_view
    - submit

    Requirements:
    - Use only the listed tools.
    - Do not explain your reasoning.
    - End with a submit action.
    - Prefer the shortest valid plan.
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


def _validate_action_plan(payload: object) -> list[StoreOpsAction]:
    if isinstance(payload, dict):
        candidate_actions = payload.get("actions", [])
    else:
        candidate_actions = payload

    if not isinstance(candidate_actions, list) or not candidate_actions:
        raise ValueError("LLM response did not include a non-empty actions list.")

    actions = [StoreOpsAction(**item) for item in candidate_actions]
    if actions[-1].tool != "submit":
        actions.append(StoreOpsAction(tool="submit"))
    return actions


def _llm_plan_actions(client: OpenAI, observation: StoreOpsObservation) -> list[StoreOpsAction]:
    user_prompt = textwrap.dedent(
        f"""
        Question: {observation.question}
        Role: {observation.role}
        Category: {observation.category}
        Task ID: {observation.metadata.get("task_id", "")}
        Available dimensions: {", ".join(observation.available_dimensions)}
        Available metrics: {", ".join(observation.available_metrics)}
        Current preview rows: {json.dumps(observation.current_view, ensure_ascii=True)}
        """
    ).strip()

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
    payload = json.loads(_strip_code_fences(content))
    return _validate_action_plan(payload)


def choose_plan(
    observation: StoreOpsObservation,
    *,
    planner_mode: str,
    client: OpenAI | None,
) -> tuple[list[StoreOpsAction], str]:
    if planner_mode == "heuristic":
        return heuristic_plan_actions(observation), "heuristic"

    if planner_mode in {"llm", "auto"} and client is not None:
        try:
            return _llm_plan_actions(client, observation), MODEL_NAME
        except Exception as exc:
            print(f"[DEBUG] LLM planner failed, falling back to heuristic: {exc}", flush=True)
            return heuristic_plan_actions(observation), f"{MODEL_NAME}-fallback"

    return heuristic_plan_actions(observation), "heuristic"


async def create_env() -> StoreOpsEnv:
    if IMAGE_NAME:
        return await StoreOpsEnv.from_docker_image(IMAGE_NAME)
    return StoreOpsEnv(base_url=BASE_URL)


async def main() -> None:
    client = None
    if PLANNER_MODE in {"llm", "auto"} and API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await create_env()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    model_label = "heuristic"

    try:
        result = await env.reset(seed=RESET_SEED)
        observation = result.observation
        task_name = infer_task_id(
            observation.question,
            str(observation.metadata.get("task_id", "")),
        ) or "storeops_task"
        plan, model_label = choose_plan(observation, planner_mode=PLANNER_MODE, client=client)
        log_start(task=task_name, env=BENCHMARK, model=model_label)

        for action in plan:
            result = await env.step(action)
            steps_taken += 1
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            error = None if reward >= 0 else result.observation.status_message

            log_step(
                step=steps_taken,
                action=format_action(action),
                reward=reward,
                done=result.done,
                error=error,
            )

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
