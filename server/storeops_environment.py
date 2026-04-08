"""OpenEnv environment for StoreOps analytics tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
from typing import Any
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import StoreOpsAction, StoreOpsObservation
    from .analytics_engine import StoreOpsAnalyticsEngine
except ImportError:
    from models import StoreOpsAction, StoreOpsObservation
    from server.analytics_engine import StoreOpsAnalyticsEngine


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    role: str
    category: str
    difficulty: str
    grader_score: float
    question: str
    target_view: pd.DataFrame


class StoreOpsEnvironment(Environment):
    """Multi-step analytics environment over D-1 style store ops data."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _MAX_STEPS = 8
    _ACTION_REWARD = {
        "filter_equals": 0.15,
        "group_aggregate": 0.20,
        "compare_dates": 0.25,
        "sort_limit": 0.10,
        "reset_view": 0.0,
    }
    _SUBMIT_REWARD = {
        "store_item_qty_total": 0.35,
        "store_top_variance_items": 0.40,
        "central_top_stores_for_item": 0.40,
        "central_city_breakdown": 0.40,
        "central_top_variance_stores_for_item": 0.40,
        "central_top_delta_stores_for_item": 0.50,
    }
    _TASK_SCORES = {
        "store_item_qty_total": 0.21,
        "store_top_variance_items": 0.43,
        "central_city_breakdown": 0.47,
        "central_top_stores_for_item": 0.67,
        "central_top_variance_stores_for_item": 0.73,
        "central_top_delta_stores_for_item": 0.89,
    }

    def __init__(self):
        self._rng = random.Random()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._history: list[str] = []
        self._status_message = ""
        self._terminated = False
        self._score = 0.0
        self._task: TaskDefinition | None = None
        self._task_cursor = 0
        self._task_rotation: list[TaskDefinition] = []
        self._engine = self._load_engine()
        self._tasks = self._build_tasks()
        self._task_rotation = self._build_task_rotation(self._tasks)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        difficulty: str | None = None,
        **_: Any,
    ) -> StoreOpsObservation:
        """Reset to a fresh analytics task."""
        if seed is not None:
            self._rng.seed(seed)

        self._engine = self._load_engine()
        self._tasks = self._build_tasks()
        self._task_rotation = self._build_task_rotation(self._tasks)
        self._history = []
        self._terminated = False
        self._score = 0.0
        self._status_message = "Task loaded. Use analytics actions to derive the answer."

        if task_id is not None:
            matching = [task for task in self._tasks if task.task_id == task_id]
            if not matching:
                raise ValueError(f"Unknown task_id: {task_id}")
            self._task = matching[0]
        elif difficulty is not None:
            matching = [task for task in self._task_rotation if task.difficulty == difficulty]
            if not matching:
                raise ValueError(f"Unknown difficulty: {difficulty}")
            task_index = self._task_cursor % len(matching)
            self._task = matching[task_index]
            self._task_cursor += 1
        elif seed is not None:
            task_index = seed % len(self._tasks)
            self._task = self._tasks[task_index]
        else:
            task_index = self._task_cursor % len(self._task_rotation)
            self._task = self._task_rotation[task_index]
            self._task_cursor += 1

        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
        )
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: StoreOpsAction) -> StoreOpsObservation:  # type: ignore[override]
        """Apply one analytics action to the current working dataframe."""
        if self._task is None:
            return self._build_observation(
                reward=0.0,
                done=True,
                status_message="Environment not initialized. Call reset() before step().",
            )

        if self._terminated:
            return self._build_observation(
                reward=0.0,
                done=True,
                status_message="Episode already terminated. Call reset() to start another task.",
            )

        self._state.step_count += 1
        reward = 0.0
        done = False

        if action.tool == "filter_equals":
            if not action.column or action.value is None:
                reward = 0.0
                self._status_message = "filter_equals requires both column and value."
            else:
                result = self._engine.filter_equals(action.column, action.value)
                reward = self._ACTION_REWARD["filter_equals"] if result.ok else 0.0
                self._status_message = result.message
        elif action.tool == "group_aggregate":
            if not action.group_by or not action.metric or not action.aggregation:
                reward = 0.0
                self._status_message = "group_aggregate requires group_by, metric, and aggregation."
            else:
                result = self._engine.group_aggregate(
                    action.group_by,
                    action.metric,
                    action.aggregation,
                )
                reward = self._ACTION_REWARD["group_aggregate"] if result.ok else 0.0
                self._status_message = result.message
        elif action.tool == "compare_dates":
            if not action.group_by or not action.metric or not action.date_from or not action.date_to:
                reward = 0.0
                self._status_message = (
                    "compare_dates requires group_by, metric, date_from, and date_to."
                )
            else:
                result = self._engine.compare_dates(
                    action.group_by,
                    action.metric,
                    action.date_from,
                    action.date_to,
                )
                reward = self._ACTION_REWARD["compare_dates"] if result.ok else 0.0
                self._status_message = result.message
        elif action.tool == "sort_limit":
            if not action.metric:
                reward = 0.0
                self._status_message = "sort_limit requires metric."
            else:
                result = self._engine.sort_limit(
                    action.metric,
                    descending=action.descending,
                    limit=action.limit,
                )
                reward = self._ACTION_REWARD["sort_limit"] if result.ok else 0.0
                self._status_message = result.message
        elif action.tool == "reset_view":
            self._engine.reset_view()
            reward = self._ACTION_REWARD["reset_view"]
            self._status_message = "Reset the working dataframe to the full dataset."
        elif action.tool == "submit":
            if self._matches_target():
                reward = self._SUBMIT_REWARD.get(self._task.task_id, 0.35)
                done = True
                self._terminated = True
                self._status_message = "Correct result submitted."
            else:
                reward = 0.0
                done = True
                self._terminated = True
                self._status_message = "Submitted result does not match the expected answer."
        else:
            reward = 0.0
            self._status_message = f"Unsupported tool: {action.tool}"

        self._history.append(self._format_action(action))
        self._score = min(1.0, round(self._score + reward, 4))

        if not done and self._state.step_count >= self._MAX_STEPS:
            reward = 0.0
            done = True
            self._terminated = True
            self._status_message = "Step budget exhausted before submitting the result."

        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    def _load_engine(self) -> StoreOpsAnalyticsEngine:
        data_dir = Path(__file__).resolve().parents[1] / "data"
        configured_path = os.getenv("STOREOPS_DATASET_PATH")
        candidate_paths = [
            Path(configured_path) if configured_path else None,
            data_dir / "synthetic_benchmark.csv",
            data_dir / "synthetic_generated.csv",
            data_dir / "synthetic_seed.csv",
        ]

        for candidate in candidate_paths:
            if candidate and candidate.exists():
                return StoreOpsAnalyticsEngine.from_csv(candidate)

        raise FileNotFoundError("No StoreOps dataset CSV could be found for the environment.")

    def _build_tasks(self) -> list[TaskDefinition]:
        base = self._engine.base_df
        tasks: list[TaskDefinition] = []
        triple_records = (
            base[["eod_date", "store_name", "inventory_name"]]
            .drop_duplicates()
            .sort_values(["eod_date", "store_name", "inventory_name"])
            .to_dict(orient="records")
        )
        store_date_records = (
            base[["eod_date", "store_name"]]
            .drop_duplicates()
            .sort_values(["eod_date", "store_name"])
            .to_dict(orient="records")
        )
        item_date_records = (
            base[["eod_date", "inventory_name"]]
            .drop_duplicates()
            .sort_values(["eod_date", "inventory_name"])
            .to_dict(orient="records")
        )
        date_values = sorted(base["eod_date"].astype(str).unique().tolist())

        for record in triple_records[:12]:
            date_value = str(record["eod_date"])
            store_value = str(record["store_name"])
            item_value = str(record["inventory_name"])
            filtered = base.loc[
                (base["eod_date"].astype(str) == date_value)
                & (base["store_name"].astype(str) == store_value)
                & (base["inventory_name"].astype(str) == item_value)
            ]
            if filtered.empty:
                continue

            target = (
                filtered.groupby("store_name", dropna=False)["qty"]
                .sum()
                .reset_index()
                .rename(columns={"qty": "sum_qty"})
            )
            tasks.append(
                TaskDefinition(
                    task_id="store_item_qty_total",
                    role="Store Manager",
                    category="Stock Availability",
                    difficulty="easy",
                    grader_score=self._TASK_SCORES["store_item_qty_total"],
                    question=(
                        f"What was the total D-1 quantity for {item_value} in {store_value} "
                        f"on {date_value}?"
                    ),
                    target_view=self._normalize_target(target),
                )
            )

        for record in store_date_records[:10]:
            date_value = str(record["eod_date"])
            store_value = str(record["store_name"])
            filtered = base.loc[
                (base["eod_date"].astype(str) == date_value)
                & (base["store_name"].astype(str) == store_value)
            ]
            grouped = (
                filtered.groupby("inventory_name", dropna=False)["variance_qty"]
                .sum()
                .reset_index()
                .rename(columns={"variance_qty": "sum_variance_qty"})
                .sort_values("sum_variance_qty", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            if grouped.empty:
                continue

            tasks.append(
                TaskDefinition(
                    task_id="store_top_variance_items",
                    role="Store Manager",
                    category="Variance",
                    difficulty="medium",
                    grader_score=self._TASK_SCORES["store_top_variance_items"],
                    question=(
                        f"Which 5 inventory items had the highest variance quantity in "
                        f"{store_value} on {date_value}?"
                    ),
                    target_view=self._normalize_target(grouped),
                )
            )

        for record in item_date_records[:12]:
            date_value = str(record["eod_date"])
            item_value = str(record["inventory_name"])
            filtered = base.loc[
                (base["eod_date"].astype(str) == date_value)
                & (base["inventory_name"].astype(str) == item_value)
            ]
            if filtered.empty:
                continue

            top_store_target = (
                filtered.groupby("store_name", dropna=False)["qty"]
                .sum()
                .reset_index()
                .rename(columns={"qty": "sum_qty"})
                .sort_values("sum_qty", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            if not top_store_target.empty:
                tasks.append(
                    TaskDefinition(
                        task_id="central_top_stores_for_item",
                        role="Central Team",
                        category="Stock Availability",
                        difficulty="hard",
                        grader_score=self._TASK_SCORES["central_top_stores_for_item"],
                        question=(
                            f"Which 5 stores had the highest D-1 quantity for {item_value} on "
                            f"{date_value}?"
                        ),
                        target_view=self._normalize_target(top_store_target),
                    )
                )

            city_target = (
                filtered.groupby("city", dropna=False)["qty"]
                .sum()
                .reset_index()
                .rename(columns={"qty": "sum_qty"})
                .sort_values("sum_qty", ascending=False)
                .reset_index(drop=True)
            )
            if not city_target.empty:
                tasks.append(
                    TaskDefinition(
                        task_id="central_city_breakdown",
                        role="Central Team",
                        category="Stock Availability",
                        difficulty="medium",
                        grader_score=self._TASK_SCORES["central_city_breakdown"],
                        question=f"Show city-wise D-1 quantity for {item_value} on {date_value}.",
                        target_view=self._normalize_target(city_target),
                    )
                )

            top_variance_store_target = (
                filtered.groupby("store_name", dropna=False)["variance_qty"]
                .sum()
                .reset_index()
                .rename(columns={"variance_qty": "sum_variance_qty"})
                .sort_values("sum_variance_qty", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            if not top_variance_store_target.empty:
                tasks.append(
                    TaskDefinition(
                        task_id="central_top_variance_stores_for_item",
                        role="Central Team",
                        category="Variance",
                        difficulty="hard",
                        grader_score=self._TASK_SCORES["central_top_variance_stores_for_item"],
                        question=(
                            f"Which 5 stores had the highest variance quantity for {item_value} "
                            f"on {date_value}?"
                        ),
                        target_view=self._normalize_target(top_variance_store_target),
                    )
                )

        if len(date_values) >= 2:
            date_from = date_values[-2]
            date_to = date_values[-1]
            inventory_values = (
                base["inventory_name"].drop_duplicates().astype(str).sort_values().tolist()[:12]
            )
            for item_value in inventory_values:
                filtered = base.loc[
                    (base["inventory_name"].astype(str) == item_value)
                    & (base["eod_date"].astype(str).isin([date_from, date_to]))
                ].copy()
                if filtered.empty:
                    continue

                grouped = (
                    filtered.groupby(["store_name", "eod_date"], dropna=False)["qty"]
                    .sum()
                    .reset_index()
                )
                pivoted = (
                    grouped.pivot(index="store_name", columns="eod_date", values="qty")
                    .fillna(0.0)
                    .reset_index()
                )
                if date_from not in pivoted.columns or date_to not in pivoted.columns:
                    continue

                target = (
                    pivoted.rename(columns={date_from: "from_qty", date_to: "to_qty"})
                    .assign(delta_qty=lambda frame: frame["to_qty"] - frame["from_qty"])
                    .sort_values("delta_qty", ascending=False)
                    .head(5)
                    .reset_index(drop=True)
                )
                if target.empty:
                    continue

                tasks.append(
                    TaskDefinition(
                        task_id="central_top_delta_stores_for_item",
                        role="Central Team",
                        category="Trend",
                        difficulty="hard",
                        grader_score=self._TASK_SCORES["central_top_delta_stores_for_item"],
                        question=(
                            f"Which 5 stores had the largest increase in D-1 quantity for "
                            f"{item_value} from {date_from} to {date_to}?"
                        ),
                        target_view=self._normalize_target(target),
                    )
                )

        if not tasks:
            raise ValueError("StoreOps environment could not generate any tasks from the dataset.")

        return tasks

    @staticmethod
    def _build_task_rotation(tasks: list[TaskDefinition]) -> list[TaskDefinition]:
        rotation: list[TaskDefinition] = []
        seen_task_ids: set[str] = set()
        for task in tasks:
            if task.task_id in seen_task_ids:
                continue
            rotation.append(task)
            seen_task_ids.add(task.task_id)
        return rotation or tasks

    def _normalize_target(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized = dataframe.copy()
        normalized.columns = [self._normalize_column(column) for column in normalized.columns]
        return normalized.sort_values(list(normalized.columns)).reset_index(drop=True)

    @staticmethod
    def _normalize_column(column: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in column).strip("_")

    def _matches_target(self) -> bool:
        if self._task is None:
            return False

        current = self._engine.normalized_result()
        target = self._task.target_view

        if list(current.columns) != list(target.columns):
            return False
        if len(current) != len(target):
            return False

        current = current.fillna("")
        target = target.fillna("")

        for column in current.columns:
            current_series = current[column]
            target_series = target[column]

            if pd.api.types.is_numeric_dtype(current_series.dtype) and pd.api.types.is_numeric_dtype(
                target_series.dtype
            ):
                if not current_series.round(4).equals(target_series.round(4)):
                    return False
            else:
                if not current_series.astype(str).equals(target_series.astype(str)):
                    return False
        return True

    def _format_action(self, action: StoreOpsAction) -> str:
        payload = action.model_dump(exclude={"metadata"}, exclude_none=True)
        return ", ".join(f"{key}={value}" for key, value in payload.items())

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        status_message: str | None = None,
    ) -> StoreOpsObservation:
        task = self._task
        return StoreOpsObservation(
            question=task.question if task else "",
            role=task.role if task else "",
            category=task.category if task else "",
            difficulty=task.difficulty if task else "easy",
            available_dimensions=self._engine.available_dimensions,
            available_metrics=self._engine.available_metrics,
            current_view=self._engine.preview(limit=10),
            row_count=self._engine.row_count(),
            history=self._history[:],
            steps_remaining=max(0, self._MAX_STEPS - self._state.step_count),
            status_message=status_message or self._status_message,
            done=done,
            reward=reward,
            metadata={
                "task_id": task.task_id if task else "",
                "difficulty": task.difficulty if task else "",
                "score": task.grader_score if task else 0.5,
                "progress_ratio": round(self._score, 4),
                "step_count": self._state.step_count,
            },
        )
