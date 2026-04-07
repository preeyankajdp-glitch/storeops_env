"""Client for the StoreOps analytics environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import StoreOpsAction, StoreOpsObservation


class StoreOpsEnv(
    EnvClient[StoreOpsAction, StoreOpsObservation, State]
):
    """Client for StoreOps multi-step analytics tasks over OpenEnv."""

    def _step_payload(self, action: StoreOpsAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[StoreOpsObservation]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward", obs_data.get("reward"))
        done = payload.get("done", obs_data.get("done", False))

        observation = StoreOpsObservation(
            question=obs_data.get("question", ""),
            role=obs_data.get("role", ""),
            category=obs_data.get("category", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            available_dimensions=obs_data.get("available_dimensions", []),
            available_metrics=obs_data.get("available_metrics", []),
            current_view=obs_data.get("current_view", []),
            row_count=obs_data.get("row_count", 0),
            history=obs_data.get("history", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            status_message=obs_data.get("status_message", ""),
            done=done,
            reward=reward,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
        )
