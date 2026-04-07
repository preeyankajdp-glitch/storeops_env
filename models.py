"""Data models for the StoreOps analytics environment."""

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class StoreOpsAction(Action):
    """One analytics action applied to the current working dataframe."""

    tool: Literal[
        "filter_equals",
        "group_aggregate",
        "compare_dates",
        "sort_limit",
        "reset_view",
        "submit",
    ] = Field(..., description="Operation to apply to the current analytic view.")
    column: str | None = Field(default=None, description="Column to filter on.")
    value: str | None = Field(default=None, description="Value used by filter actions.")
    group_by: str | None = Field(default=None, description="Grouping column.")
    metric: str | None = Field(default=None, description="Metric column to aggregate or sort by.")
    aggregation: Literal["sum", "mean", "count", "max", "min"] | None = Field(
        default=None,
        description="Aggregation function for group_aggregate.",
    )
    date_from: str | None = Field(default=None, description="Earlier date used by compare_dates.")
    date_to: str | None = Field(default=None, description="Later date used by compare_dates.")
    descending: bool = Field(
        default=True,
        description="Whether sorting should be descending.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of rows to keep after sort_limit.",
    )


class StoreOpsObservation(Observation):
    """Observation returned after each StoreOps analytics step."""

    question: str = Field(default="", description="Current business question to answer.")
    role: str = Field(default="", description="The persona asking the question.")
    category: str = Field(default="", description="Question category such as stock or variance.")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Task difficulty tier used by the benchmark grader.",
    )
    available_dimensions: list[str] = Field(
        default_factory=list,
        description="Columns that can be used for filtering or grouping.",
    )
    available_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics available for aggregation and ranking.",
    )
    current_view: list[dict] = Field(
        default_factory=list,
        description="Preview rows of the current dataframe state.",
    )
    row_count: int = Field(default=0, ge=0, description="Row count of the current dataframe state.")
    history: list[str] = Field(
        default_factory=list,
        description="Human-readable list of operations executed so far.",
    )
    steps_remaining: int = Field(
        default=0,
        ge=0,
        description="How many steps remain before the episode times out.",
    )
    status_message: str = Field(default="", description="Transition summary or grader feedback.")


class OfficeQueryRequest(BaseModel):
    """Chat-style query request for the office demo API."""

    question: str = Field(..., min_length=3, description="Natural-language question to answer.")
    max_rows: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of result rows to return in the table.",
    )


class OfficeQueryResponse(BaseModel):
    """Structured response for the office demo API."""

    supported: bool = Field(
        ...,
        description="Whether the question can be answered from the current D-1 report schema.",
    )
    answer: str = Field(..., description="Human-readable answer or limitation message.")
    parsed_intent: str | None = Field(
        default=None,
        description="Best-effort deterministic intent label derived from the question.",
    )
    table: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Main result table returned by the analytics engine.",
    )
    evidence_rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="A small row sample from the filtered source data.",
    )
    applied_filters: dict[str, str] = Field(
        default_factory=dict,
        description="Filters extracted from the question and applied to the data.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Helpful caveats about dataset limits or parsing assumptions.",
    )
