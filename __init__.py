"""StoreOps analytics environment."""

from dataclasses import dataclass

from .client import StoreOpsEnv
from .models import OfficeQueryRequest, OfficeQueryResponse, StoreOpsAction, StoreOpsObservation


@dataclass(frozen=True)
class StoreOpsTaskSpec:
    task_id: str
    difficulty: str
    grader_name: str
    objective: str


DEFAULT_TASK_ORDER = ["easy", "medium", "hard"]

_TASK_SPECS = {
    "easy": StoreOpsTaskSpec(
        task_id="easy",
        difficulty="easy",
        grader_name="easy_grader",
        objective="Answer a single-store D-1 quantity question correctly.",
    ),
    "medium": StoreOpsTaskSpec(
        task_id="medium",
        difficulty="medium",
        grader_name="medium_grader",
        objective="Answer a grouped store or city analytics question correctly.",
    ),
    "hard": StoreOpsTaskSpec(
        task_id="hard",
        difficulty="hard",
        grader_name="hard_grader",
        objective="Answer a ranking or cross-date comparison question correctly.",
    ),
}


def get_task_spec(task_id: str) -> StoreOpsTaskSpec:
    return _TASK_SPECS[task_id]

__all__ = [
    "DEFAULT_TASK_ORDER",
    "OfficeQueryRequest",
    "OfficeQueryResponse",
    "StoreOpsAction",
    "StoreOpsObservation",
    "StoreOpsEnv",
    "StoreOpsTaskSpec",
    "get_task_spec",
]
