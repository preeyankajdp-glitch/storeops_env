"""Deterministic planner for StoreOps benchmark tasks."""

from __future__ import annotations

import re

from .models import StoreOpsAction, StoreOpsObservation


def infer_task_id(question: str, hinted_task_id: str | None = None) -> str:
    """Infer the canonical task template from the question text."""
    if hinted_task_id and hinted_task_id not in {"easy", "medium", "hard"}:
        return hinted_task_id

    patterns = [
        ("store_item_qty_total", r"^What was the total D-1 quantity for .+ in .+ on \d{4}-\d{2}-\d{2}\?$"),
        (
            "store_top_variance_items",
            r"^Which 5 inventory items had the highest variance quantity in .+ on \d{4}-\d{2}-\d{2}\?$",
        ),
        (
            "central_top_stores_for_item",
            r"^Which 5 stores had the highest D-1 quantity for .+ on \d{4}-\d{2}-\d{2}\?$",
        ),
        ("central_city_breakdown", r"^Show city-wise D-1 quantity for .+ on \d{4}-\d{2}-\d{2}\.$"),
        (
            "central_top_variance_stores_for_item",
            r"^Which 5 stores had the highest variance quantity for .+ on \d{4}-\d{2}-\d{2}\?$",
        ),
        (
            "central_top_delta_stores_for_item",
            r"^Which 5 stores had the largest increase in D-1 quantity for .+ from \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\?$",
        ),
    ]
    for task_id, pattern in patterns:
        if re.fullmatch(pattern, question):
            return task_id
    return hinted_task_id or ""


def heuristic_plan_actions(observation: StoreOpsObservation) -> list[StoreOpsAction]:
    """Build a deterministic action plan from the environment question."""
    task_id = infer_task_id(
        observation.question.strip(),
        str(
            observation.metadata.get(
                "underlying_task_id",
                observation.metadata.get("task_id", ""),
            )
        ),
    )
    question = observation.question.strip()

    if task_id == "store_item_qty_total":
        match = re.fullmatch(
            r"What was the total D-1 quantity for (?P<item>.+) in (?P<store>.+) on (?P<date>\d{4}-\d{2}-\d{2})\?",
            question,
        )
        if match:
            return [
                StoreOpsAction(tool="filter_equals", column="eod_date", value=match.group("date")),
                StoreOpsAction(tool="filter_equals", column="store_name", value=match.group("store")),
                StoreOpsAction(
                    tool="filter_equals",
                    column="inventory_name",
                    value=match.group("item"),
                ),
                StoreOpsAction(
                    tool="group_aggregate",
                    group_by="store_name",
                    metric="qty",
                    aggregation="sum",
                ),
                StoreOpsAction(tool="submit"),
            ]

    if task_id == "store_top_variance_items":
        match = re.fullmatch(
            r"Which 5 inventory items had the highest variance quantity in (?P<store>.+) on (?P<date>\d{4}-\d{2}-\d{2})\?",
            question,
        )
        if match:
            return [
                StoreOpsAction(tool="filter_equals", column="eod_date", value=match.group("date")),
                StoreOpsAction(tool="filter_equals", column="store_name", value=match.group("store")),
                StoreOpsAction(
                    tool="group_aggregate",
                    group_by="inventory_name",
                    metric="variance_qty",
                    aggregation="sum",
                ),
                StoreOpsAction(
                    tool="sort_limit",
                    metric="sum_variance_qty",
                    descending=True,
                    limit=5,
                ),
                StoreOpsAction(tool="submit"),
            ]

    if task_id == "central_top_stores_for_item":
        match = re.fullmatch(
            r"Which 5 stores had the highest D-1 quantity for (?P<item>.+) on (?P<date>\d{4}-\d{2}-\d{2})\?",
            question,
        )
        if match:
            return [
                StoreOpsAction(tool="filter_equals", column="eod_date", value=match.group("date")),
                StoreOpsAction(
                    tool="filter_equals",
                    column="inventory_name",
                    value=match.group("item"),
                ),
                StoreOpsAction(
                    tool="group_aggregate",
                    group_by="store_name",
                    metric="qty",
                    aggregation="sum",
                ),
                StoreOpsAction(tool="sort_limit", metric="sum_qty", descending=True, limit=5),
                StoreOpsAction(tool="submit"),
            ]

    if task_id == "central_city_breakdown":
        match = re.fullmatch(
            r"Show city-wise D-1 quantity for (?P<item>.+) on (?P<date>\d{4}-\d{2}-\d{2})\.",
            question,
        )
        if match:
            return [
                StoreOpsAction(tool="filter_equals", column="eod_date", value=match.group("date")),
                StoreOpsAction(
                    tool="filter_equals",
                    column="inventory_name",
                    value=match.group("item"),
                ),
                StoreOpsAction(
                    tool="group_aggregate",
                    group_by="city",
                    metric="qty",
                    aggregation="sum",
                ),
                StoreOpsAction(tool="sort_limit", metric="sum_qty", descending=True, limit=100),
                StoreOpsAction(tool="submit"),
            ]

    if task_id == "central_top_variance_stores_for_item":
        match = re.fullmatch(
            r"Which 5 stores had the highest variance quantity for (?P<item>.+) on (?P<date>\d{4}-\d{2}-\d{2})\?",
            question,
        )
        if match:
            return [
                StoreOpsAction(tool="filter_equals", column="eod_date", value=match.group("date")),
                StoreOpsAction(
                    tool="filter_equals",
                    column="inventory_name",
                    value=match.group("item"),
                ),
                StoreOpsAction(
                    tool="group_aggregate",
                    group_by="store_name",
                    metric="variance_qty",
                    aggregation="sum",
                ),
                StoreOpsAction(
                    tool="sort_limit",
                    metric="sum_variance_qty",
                    descending=True,
                    limit=5,
                ),
                StoreOpsAction(tool="submit"),
            ]

    if task_id == "central_top_delta_stores_for_item":
        match = re.fullmatch(
            (
                r"Which 5 stores had the largest increase in D-1 quantity for (?P<item>.+) "
                r"from (?P<date_from>\d{4}-\d{2}-\d{2}) to (?P<date_to>\d{4}-\d{2}-\d{2})\?"
            ),
            question,
        )
        if match:
            return [
                StoreOpsAction(
                    tool="filter_equals",
                    column="inventory_name",
                    value=match.group("item"),
                ),
                StoreOpsAction(
                    tool="compare_dates",
                    group_by="store_name",
                    metric="qty",
                    date_from=match.group("date_from"),
                    date_to=match.group("date_to"),
                ),
                StoreOpsAction(tool="sort_limit", metric="delta_qty", descending=True, limit=5),
                StoreOpsAction(tool="submit"),
            ]

    raise ValueError(
        f"No deterministic plan is available for task_id={task_id!r} question={question!r}"
    )


def format_action(action: StoreOpsAction) -> str:
    """Render one action in the compact logging style used by inference.py."""
    payload = action.model_dump(exclude={"metadata"}, exclude_none=True, exclude_defaults=True)
    tool = payload.pop("tool")
    if not payload:
        return tool
    arguments = ",".join(f"{key}={value}" for key, value in payload.items())
    return f"{tool}({arguments})"
