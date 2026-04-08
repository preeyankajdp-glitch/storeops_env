from pathlib import Path

from storeops_env import StoreOpsAction
from storeops_env.solver import heuristic_plan_actions
from storeops_env.server.storeops_environment import StoreOpsEnvironment


def _seed_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "synthetic_seed.csv"


def _build_env(monkeypatch) -> StoreOpsEnvironment:
    monkeypatch.setenv("STOREOPS_DATASET_PATH", str(_seed_path()))
    return StoreOpsEnvironment()


def test_reset_returns_question_and_preview(monkeypatch):
    env = _build_env(monkeypatch)

    observation = env.reset(seed=1)

    assert observation.question
    assert observation.current_view
    assert observation.row_count > 0
    assert observation.difficulty in {"easy", "medium", "hard"}
    assert 0.0 < observation.metadata["score"] < 1.0
    assert 0.0 < observation.metadata["progress_ratio"] < 1.0


def test_seeded_reset_is_reproducible(monkeypatch):
    env = _build_env(monkeypatch)

    first = env.reset(seed=7)
    second = env.reset(seed=7)

    assert first.metadata["task_id"] == second.metadata["task_id"]
    assert first.question == second.question
    assert first.metadata["score"] == second.metadata["score"]


def test_unseeded_reset_cycles_through_distinct_tasks(monkeypatch):
    env = _build_env(monkeypatch)

    first = env.reset()
    second = env.reset()
    third = env.reset()

    assert len({first.metadata["task_id"], second.metadata["task_id"], third.metadata["task_id"]}) == 3


def test_solver_can_complete_multiple_task_templates(monkeypatch):
    env = _build_env(monkeypatch)
    solved_task_ids: set[str] = set()
    total_tasks = len(env._tasks)

    for seed in range(total_tasks):
        observation = env.reset(seed=seed)
        task_id = observation.metadata["task_id"]
        if task_id in solved_task_ids:
            continue

        result = observation
        for action in heuristic_plan_actions(observation):
            result = env.step(action)
            if result.done:
                break

        assert result.done is True
        assert result.status_message == "Correct result submitted."
        solved_task_ids.add(task_id)

        if len(solved_task_ids) == 6:
            break

    assert solved_task_ids == {
        "store_item_qty_total",
        "store_top_variance_items",
        "central_top_stores_for_item",
        "central_city_breakdown",
        "central_top_variance_stores_for_item",
        "central_top_delta_stores_for_item",
    }


def test_wrong_submission_fails(monkeypatch):
    env = _build_env(monkeypatch)
    env.reset(seed=1)

    result = env.step(StoreOpsAction(tool="submit"))

    assert result.done is True
    assert result.reward == 0.0
    assert 0.0 < result.metadata["score"] < 1.0
    assert 0.0 < result.metadata["progress_ratio"] < 1.0


def test_successful_episode_rewards_stay_normalized(monkeypatch):
    env = _build_env(monkeypatch)
    observation = env.reset(seed=1)

    result = observation
    seen_difficulties = {observation.difficulty}
    rewards = []
    for action in heuristic_plan_actions(observation):
        result = env.step(action)
        rewards.append(result.reward)
        if result.done:
            break

    assert result.done is True
    assert result.status_message == "Correct result submitted."
    assert all(0.0 <= float(reward) <= 1.0 for reward in rewards)
    assert 0.0 < result.metadata["score"] < 1.0
    assert 0.0 < result.metadata["progress_ratio"] < 1.0
    assert seen_difficulties <= {"easy", "medium", "hard"}
