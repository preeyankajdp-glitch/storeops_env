"""Deterministic analytics engine used by StoreOps tasks and demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import pandas as pd


def _normalize_column_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return slug.strip("_")


def _coerce_filter_value(series: pd.Series, value: str):
    if pd.api.types.is_numeric_dtype(series.dtype):
        try:
            return float(value)
        except ValueError:
            return value
    return value


@dataclass
class EngineStepResult:
    ok: bool
    message: str


class StoreOpsAnalyticsEngine:
    """Thin dataframe engine with deterministic business operations."""

    PREVIEW_COLUMNS = [
        "eod_date",
        "city",
        "store_name",
        "product_name",
        "inventory_name",
        "qty",
        "ideals",
        "variance_qty",
        "variance_pct",
    ]

    DEFAULT_DIMENSIONS = [
        "eod_date",
        "brand_name",
        "city",
        "store_code",
        "store_name",
        "product_name",
        "inventory_name",
        "state",
        "area_manager",
        "zonal_manager",
    ]

    DEFAULT_METRICS = [
        "qty",
        "ideals",
        "count",
        "prod_quantity",
        "item_price",
        "inventory_item_price",
        "inventory_item_basic_price",
        "ideal_wacc",
        "variance_qty",
        "variance_pct",
        "cost_delta",
    ]

    def __init__(self, dataframe: pd.DataFrame):
        normalized = dataframe.rename(columns=_normalize_column_name).copy()
        self.base_df = self._with_derived_metrics(normalized)
        self.current_df = self.base_df.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> "StoreOpsAnalyticsEngine":
        dataframe = pd.read_csv(path)
        return cls(dataframe)

    @staticmethod
    def _with_derived_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()

        numeric_columns = [
            "qty",
            "ideals",
            "count",
            "prod_quantity",
            "item_price",
            "inventory_item_price",
            "inventory_item_basic_price",
            "ideal_wacc",
            "conversion_value",
        ]

        for column in numeric_columns:
            if column in dataframe.columns:
                dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0)

        dataframe["variance_qty"] = dataframe.get("qty", 0.0) - dataframe.get("ideals", 0.0)
        ideals = dataframe.get("ideals", 0.0)
        dataframe["variance_pct"] = ideals.where(ideals != 0, other=float("nan"))
        dataframe["variance_pct"] = (
            100.0
            * dataframe["variance_qty"]
            / dataframe["variance_pct"]
        ).fillna(0.0)
        dataframe["cost_delta"] = dataframe.get("inventory_item_price", 0.0) - dataframe.get(
            "ideal_wacc", 0.0
        )
        return dataframe

    def reset_view(self) -> None:
        self.current_df = self.base_df.copy()

    @property
    def available_dimensions(self) -> list[str]:
        return [column for column in self.DEFAULT_DIMENSIONS if column in self.base_df.columns]

    @property
    def available_metrics(self) -> list[str]:
        return [column for column in self.DEFAULT_METRICS if column in self.base_df.columns]

    def filter_equals(self, column: str, value: str) -> EngineStepResult:
        if column not in self.current_df.columns:
            return EngineStepResult(False, f"Unknown filter column: {column}")

        series = self.current_df[column]
        coerced_value = _coerce_filter_value(series, value)
        if pd.api.types.is_numeric_dtype(series.dtype):
            mask = series == coerced_value
        else:
            mask = series.astype(str).str.casefold() == str(coerced_value).casefold()

        filtered = self.current_df.loc[mask].copy()
        if filtered.empty:
            return EngineStepResult(False, f"Filter returned no rows for {column}={value!r}")

        self.current_df = filtered
        return EngineStepResult(True, f"Filtered to {len(filtered)} rows where {column}={value!r}")

    def group_aggregate(self, group_by: str, metric: str, aggregation: str) -> EngineStepResult:
        if group_by not in self.current_df.columns:
            return EngineStepResult(False, f"Unknown group_by column: {group_by}")
        if metric not in self.current_df.columns:
            return EngineStepResult(False, f"Unknown metric column: {metric}")
        if aggregation not in {"sum", "mean", "count", "max", "min"}:
            return EngineStepResult(False, f"Unsupported aggregation: {aggregation}")

        grouped = (
            self.current_df.groupby(group_by, dropna=False)[metric]
            .agg(aggregation)
            .reset_index()
            .rename(columns={metric: f"{aggregation}_{metric}"})
        )
        self.current_df = grouped
        return EngineStepResult(
            True,
            f"Grouped by {group_by} with {aggregation} over {metric}.",
        )

    def sort_limit(self, metric: str, descending: bool = True, limit: int = 10) -> EngineStepResult:
        if metric not in self.current_df.columns:
            return EngineStepResult(False, f"Unknown sort column: {metric}")

        sorted_df = self.current_df.sort_values(metric, ascending=not descending).head(limit).copy()
        self.current_df = sorted_df.reset_index(drop=True)
        return EngineStepResult(
            True,
            f"Sorted by {metric} ({'desc' if descending else 'asc'}) and kept top {limit} rows.",
        )

    def preview(self, limit: int = 10) -> list[dict]:
        preview_df = self.current_df.head(limit).copy()
        visible_columns = [column for column in self.PREVIEW_COLUMNS if column in preview_df.columns]
        if visible_columns:
            extra_columns = [column for column in preview_df.columns if column not in visible_columns]
            preview_df = preview_df[visible_columns + extra_columns]
        preview_df = preview_df.where(pd.notnull(preview_df), None)

        records = preview_df.to_dict(orient="records")
        cleaned_records = []
        for record in records:
            cleaned = {}
            for key, value in record.items():
                if isinstance(value, float):
                    cleaned[key] = round(value, 4)
                elif value is None or (isinstance(value, float) and math.isnan(value)):
                    cleaned[key] = None
                else:
                    cleaned[key] = value
            cleaned_records.append(cleaned)
        return cleaned_records

    def row_count(self) -> int:
        return int(len(self.current_df))

    def normalized_result(self) -> pd.DataFrame:
        result = self.current_df.copy()
        result = result.reset_index(drop=True)
        result.columns = [_normalize_column_name(column) for column in result.columns]
        if not result.empty:
            result = result.sort_values(list(result.columns)).reset_index(drop=True)
        return result
