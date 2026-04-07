"""Deterministic query service for the office-facing StoreOps demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
from typing import Any

import pandas as pd

try:
    from ..models import OfficeQueryResponse
    from .analytics_engine import StoreOpsAnalyticsEngine
except ImportError:
    from models import OfficeQueryResponse
    from server.analytics_engine import StoreOpsAnalyticsEngine


UNSUPPORTED_PATTERNS: dict[str, str] = {
    "right now": "The D-1 report is historical, so it cannot answer live stock questions.",
    "current stock": "The D-1 report does not contain live stock-on-hand snapshots.",
    "stock level": "The D-1 report captures D-1 usage-style rows, not current stock-on-hand.",
    "out of stock": "Out-of-stock status needs live inventory snapshots, which are not present here.",
    "running low": "Low-stock alerts need threshold fields that are not present in the D-1 report.",
    "minimum stock": "Minimum stock levels are not present in the D-1 report schema.",
    "reorder": "Reorder logic needs threshold fields that are not present in the D-1 report.",
    "expiry": "Expiry tracking needs shelf-life data that is not present in the D-1 report.",
    "opening stock": "Opening stock is not present in the D-1 report schema.",
    "updated stock": "The D-1 report does not capture who updated stock.",
    "today's operations": "This dataset is D-1 historical data, not a live operational snapshot.",
    "today’s operations": "This dataset is D-1 historical data, not a live operational snapshot.",
    "overstocked": "Overstock detection needs target thresholds that are not present in the D-1 report.",
}

STOPWORDS = {
    "store",
    "city",
    "item",
    "inventory",
    "the",
    "and",
    "for",
    "with",
    "what",
    "which",
    "show",
    "total",
}


@dataclass(frozen=True)
class ParsedQuestion:
    """Structured, deterministic parse of a natural-language question."""

    intent: str | None
    date_value: str | None
    store_name: str | None
    inventory_name: str | None
    city: str | None
    notes: list[str] = field(default_factory=list)
    unsupported_reason: str | None = None


class StoreOpsQueryService:
    """Answer supported analytics questions with deterministic dataframe logic."""

    def __init__(self, dataset_path: str | Path | None = None):
        resolved_path = dataset_path or self._default_dataset_path()
        self.dataset_path = str(resolved_path)
        self.engine = StoreOpsAnalyticsEngine.from_csv(resolved_path)
        self.df = self.engine.base_df.copy()
        self.available_dates = sorted(self.df["eod_date"].astype(str).unique().tolist())

    def sample_questions(self) -> list[str]:
        """Return dataset-aware example questions that should work out of the box."""
        questions: list[str] = []
        if self.df.empty:
            return questions

        latest_date = self.available_dates[-1] if self.available_dates else None
        previous_date = self.available_dates[-2] if len(self.available_dates) >= 2 else None

        store_name = self._first_value("store_name")
        inventory_values = self.df["inventory_name"].dropna().astype(str).unique().tolist()
        inventory_name = inventory_values[0] if inventory_values else None
        second_inventory = inventory_values[1] if len(inventory_values) > 1 else inventory_name

        if store_name and inventory_name and latest_date:
            if len(self.available_dates) == 1:
                questions.append(
                    f"How much {inventory_name} was used in {store_name} yesterday?"
                )
            else:
                questions.append(
                    f"How much {inventory_name} was used in {store_name} on {latest_date}?"
                )

        if second_inventory and latest_date:
            questions.append(
                f"Which stores had the highest D-1 quantity for {second_inventory} on {latest_date}?"
            )
            questions.append(
                f"Show city-wise D-1 quantity for {second_inventory} on {latest_date}."
            )

        if store_name and latest_date:
            questions.append(f"Which items have the highest variance in {store_name}?")

        if store_name and inventory_name and latest_date:
            questions.append(f"What is the variance percentage for {inventory_name} in {store_name}?")

        if previous_date and store_name and inventory_name:
            questions.append(
                f"Has variance increased compared to yesterday for {inventory_name} in {store_name}?"
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for question in questions:
            if question not in seen:
                deduped.append(question)
                seen.add(question)
        return deduped[:6]

    @staticmethod
    def _default_dataset_path() -> Path:
        root = Path(__file__).resolve().parents[1] / "data"
        explicit = os.getenv("STOREOPS_OFFICE_DATASET_PATH")
        if explicit:
            return Path(explicit)
        report = root / "office_demo_sample.csv"
        if report.exists():
            return report
        return Path(os.getenv("STOREOPS_DATASET_PATH", root / "synthetic_seed.csv"))

    def answer(self, question: str, max_rows: int = 10) -> OfficeQueryResponse:
        parsed = self._parse_question(question)
        if parsed.unsupported_reason:
            return self._unsupported_response(parsed, parsed.unsupported_reason)

        if parsed.intent is None:
            return self._unsupported_response(
                parsed,
                (
                    "I could not map that question to the current D-1 analytics patterns yet. "
                    "Supported now: item quantity in store, top stores for an item, city-wise "
                    "usage, highest variance items in a store, item variance, and day-over-day variance."
                ),
            )

        handlers = {
            "store_item_qty_total": self._answer_store_item_qty_total,
            "store_top_variance_items": self._answer_store_top_variance_items,
            "item_variance": self._answer_item_variance,
            "item_variance_pct": self._answer_item_variance_pct,
            "top_stores_for_item": self._answer_top_stores_for_item,
            "city_breakdown": self._answer_city_breakdown,
            "day_over_day_variance": self._answer_day_over_day_variance,
        }

        handler = handlers.get(parsed.intent)
        if handler is None:
            return self._unsupported_response(parsed, f"Intent {parsed.intent!r} is not implemented yet.")

        try:
            return handler(parsed, max_rows=max_rows)
        except ValueError as exc:
            return self._unsupported_response(parsed, str(exc))

    def _parse_question(self, question: str) -> ParsedQuestion:
        lowered = question.casefold()

        for pattern, reason in UNSUPPORTED_PATTERNS.items():
            if pattern in lowered:
                return ParsedQuestion(
                    intent=None,
                    date_value=self._resolve_date(lowered),
                    store_name=self._resolve_entity("store_name", question),
                    inventory_name=self._resolve_entity("inventory_name", question),
                    city=self._resolve_entity("city", question),
                    notes=[],
                    unsupported_reason=reason,
                )

        notes: list[str] = []
        date_value = self._resolve_date(question)
        if date_value and not any(token in lowered for token in ("yesterday", "d-1", "d-2", "day before")):
            notes.append(f"No explicit date detected, so I defaulted to {date_value}.")

        store_name = self._resolve_entity("store_name", question)
        inventory_name = self._resolve_entity("inventory_name", question)
        city = self._resolve_entity("city", question)

        if "compared to yesterday" in lowered and "variance" in lowered:
            return ParsedQuestion(
                intent="day_over_day_variance",
                date_value=date_value,
                store_name=store_name,
                inventory_name=inventory_name,
                city=city,
                notes=notes,
            )

        if ("highest variance" in lowered or "top variance" in lowered) and (
            "item" in lowered or "items" in lowered
        ):
            return ParsedQuestion(
                intent="store_top_variance_items",
                date_value=date_value,
                store_name=store_name,
                inventory_name=inventory_name,
                city=city,
                notes=notes,
            )

        if "variance percentage" in lowered or "variance %" in lowered:
            return ParsedQuestion(
                intent="item_variance_pct",
                date_value=date_value,
                store_name=store_name,
                inventory_name=inventory_name,
                city=city,
                notes=notes,
            )

        if "variance" in lowered:
            return ParsedQuestion(
                intent="item_variance",
                date_value=date_value,
                store_name=store_name,
                inventory_name=inventory_name,
                city=city,
                notes=notes,
            )

        if ("city-wise" in lowered or "by city" in lowered) and inventory_name:
            return ParsedQuestion(
                intent="city_breakdown",
                date_value=date_value,
                store_name=store_name,
                inventory_name=inventory_name,
                city=city,
                notes=notes,
            )

        if (
            ("top stores" in lowered or "highest" in lowered)
            and "store" in lowered
            and inventory_name
            and any(token in lowered for token in ("quantity", "qty", "usage", "used"))
        ):
            return ParsedQuestion(
                intent="top_stores_for_item",
                date_value=date_value,
                store_name=store_name,
                inventory_name=inventory_name,
                city=city,
                notes=notes,
            )

        if inventory_name and store_name and any(
            token in lowered for token in ("quantity", "qty", "usage", "used", "how much")
        ):
            return ParsedQuestion(
                intent="store_item_qty_total",
                date_value=date_value,
                store_name=store_name,
                inventory_name=inventory_name,
                city=city,
                notes=notes,
            )

        return ParsedQuestion(
            intent=None,
            date_value=date_value,
            store_name=store_name,
            inventory_name=inventory_name,
            city=city,
            notes=notes,
        )

    def _resolve_date(self, question: str) -> str | None:
        if not self.available_dates:
            return None

        exact_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
        if exact_match:
            value = exact_match.group(0)
            if value in self.available_dates:
                return value

        lowered = question.casefold()
        if any(token in lowered for token in ("d-2", "day before", "previous day")) and len(self.available_dates) >= 2:
            return self.available_dates[-2]
        return self.available_dates[-1]

    def _resolve_entity(self, column: str, question: str) -> str | None:
        question_cf = question.casefold()
        candidates = sorted(
            self.df[column].dropna().astype(str).unique().tolist(),
            key=len,
            reverse=True,
        )

        for candidate in candidates:
            if candidate.casefold() in question_cf:
                return candidate

        question_tokens = {token for token in re.findall(r"[a-z0-9]+", question_cf) if token not in STOPWORDS}
        for candidate in candidates:
            candidate_tokens = {
                token
                for token in re.findall(r"[a-z0-9]+", candidate.casefold())
                if token not in STOPWORDS and len(token) >= 3
            }
            if candidate_tokens and candidate_tokens.issubset(question_tokens):
                return candidate

        return None

    def _first_value(self, column: str) -> str | None:
        values = self.df[column].dropna().astype(str).unique().tolist()
        return values[0] if values else None

    def _answer_store_item_qty_total(
        self,
        parsed: ParsedQuestion,
        *,
        max_rows: int,
    ) -> OfficeQueryResponse:
        if not parsed.store_name or not parsed.inventory_name or not parsed.date_value:
            raise ValueError("I need an explicit store name, inventory item, and date context for that question.")

        filtered = self._filter_frame(
            eod_date=parsed.date_value,
            store_name=parsed.store_name,
            inventory_name=parsed.inventory_name,
        )
        total_qty = float(filtered["qty"].sum())
        result = pd.DataFrame(
            [
                {
                    "eod_date": parsed.date_value,
                    "store_name": parsed.store_name,
                    "inventory_name": parsed.inventory_name,
                    "sum_qty": round(total_qty, 4),
                }
            ]
        )
        answer = (
            f"{parsed.inventory_name} had total D-1 quantity {round(total_qty, 2)} in "
            f"{parsed.store_name} on {parsed.date_value}."
        )
        return self._supported_response(parsed, answer, result, filtered, max_rows=max_rows)

    def _answer_store_top_variance_items(
        self,
        parsed: ParsedQuestion,
        *,
        max_rows: int,
    ) -> OfficeQueryResponse:
        if not parsed.store_name or not parsed.date_value:
            raise ValueError("I need an explicit store name for the highest-variance-items question.")

        filtered = self._filter_frame(
            eod_date=parsed.date_value,
            store_name=parsed.store_name,
        )
        grouped = self._variance_group(filtered, "inventory_name").head(max_rows)
        if grouped.empty:
            raise ValueError("No variance rows matched that store/date combination.")

        top_item = str(grouped.iloc[0]["inventory_name"])
        top_value = float(grouped.iloc[0]["sum_variance_qty"])
        answer = (
            f"{top_item} is the highest variance item in {parsed.store_name} on {parsed.date_value}, "
            f"with variance quantity {round(top_value, 2)}."
        )
        return self._supported_response(parsed, answer, grouped, filtered, max_rows=max_rows)

    def _answer_item_variance(
        self,
        parsed: ParsedQuestion,
        *,
        max_rows: int,
    ) -> OfficeQueryResponse:
        if not parsed.inventory_name or not parsed.date_value:
            raise ValueError("I need an explicit inventory item and date context for the variance question.")

        filtered = self._filter_frame(
            eod_date=parsed.date_value,
            store_name=parsed.store_name,
            inventory_name=parsed.inventory_name,
            city=parsed.city,
        )
        summary = self._aggregate_variance(filtered)
        scope = parsed.store_name or parsed.city or "all matching stores"
        answer = (
            f"{parsed.inventory_name} has variance quantity {summary['variance_qty']} and variance percentage "
            f"{summary['variance_pct']}% for {scope} on {parsed.date_value}."
        )
        table = pd.DataFrame([{**summary, "scope": scope, "inventory_name": parsed.inventory_name}])
        return self._supported_response(parsed, answer, table, filtered, max_rows=max_rows)

    def _answer_item_variance_pct(
        self,
        parsed: ParsedQuestion,
        *,
        max_rows: int,
    ) -> OfficeQueryResponse:
        response = self._answer_item_variance(parsed, max_rows=max_rows)
        if response.table:
            variance_pct = response.table[0]["variance_pct"]
            scope = response.table[0]["scope"]
            response.answer = (
                f"{parsed.inventory_name} has variance percentage {variance_pct}% for {scope} "
                f"on {parsed.date_value}."
            )
            response.parsed_intent = "item_variance_pct"
        return response

    def _answer_top_stores_for_item(
        self,
        parsed: ParsedQuestion,
        *,
        max_rows: int,
    ) -> OfficeQueryResponse:
        if not parsed.inventory_name or not parsed.date_value:
            raise ValueError("I need an explicit inventory item and date context to rank stores.")

        filtered = self._filter_frame(
            eod_date=parsed.date_value,
            inventory_name=parsed.inventory_name,
        )
        grouped = (
            filtered.groupby("store_name", dropna=False)["qty"]
            .sum()
            .reset_index()
            .rename(columns={"qty": "sum_qty"})
            .sort_values("sum_qty", ascending=False)
            .head(max_rows)
            .reset_index(drop=True)
        )
        if grouped.empty:
            raise ValueError("No rows matched that item/date combination.")

        leader = str(grouped.iloc[0]["store_name"])
        leader_qty = float(grouped.iloc[0]["sum_qty"])
        answer = (
            f"{leader} has the highest D-1 quantity for {parsed.inventory_name} on {parsed.date_value} "
            f"with {round(leader_qty, 2)}."
        )
        return self._supported_response(parsed, answer, grouped, filtered, max_rows=max_rows)

    def _answer_city_breakdown(
        self,
        parsed: ParsedQuestion,
        *,
        max_rows: int,
    ) -> OfficeQueryResponse:
        if not parsed.inventory_name or not parsed.date_value:
            raise ValueError("I need an explicit inventory item and date context for a city-wise breakdown.")

        filtered = self._filter_frame(
            eod_date=parsed.date_value,
            inventory_name=parsed.inventory_name,
        )
        grouped = (
            filtered.groupby("city", dropna=False)["qty"]
            .sum()
            .reset_index()
            .rename(columns={"qty": "sum_qty"})
            .sort_values("sum_qty", ascending=False)
            .head(max_rows)
            .reset_index(drop=True)
        )
        if grouped.empty:
            raise ValueError("No rows matched that item/date combination.")

        answer = f"Here is the city-wise D-1 quantity breakdown for {parsed.inventory_name} on {parsed.date_value}."
        return self._supported_response(parsed, answer, grouped, filtered, max_rows=max_rows)

    def _answer_day_over_day_variance(
        self,
        parsed: ParsedQuestion,
        *,
        max_rows: int,
    ) -> OfficeQueryResponse:
        if len(self.available_dates) < 2:
            raise ValueError("Day-over-day comparison needs at least two dates in the dataset.")
        if not parsed.inventory_name:
            raise ValueError("I need an explicit inventory item for a day-over-day variance comparison.")

        current_date = self.available_dates[-1]
        previous_date = self.available_dates[-2]

        current = self._filter_frame(
            eod_date=current_date,
            store_name=parsed.store_name,
            inventory_name=parsed.inventory_name,
            city=parsed.city,
        )
        previous = self._filter_frame(
            eod_date=previous_date,
            store_name=parsed.store_name,
            inventory_name=parsed.inventory_name,
            city=parsed.city,
        )

        current_summary = self._aggregate_variance(current)
        previous_summary = self._aggregate_variance(previous)
        comparison = pd.DataFrame(
            [
                {"eod_date": previous_date, **previous_summary},
                {"eod_date": current_date, **current_summary},
            ]
        )
        delta = round(current_summary["variance_qty"] - previous_summary["variance_qty"], 4)
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "stayed flat"
        scope = parsed.store_name or parsed.city or "all matching stores"
        answer = (
            f"Variance for {parsed.inventory_name} {direction} from {previous_date} to {current_date} "
            f"for {scope}. Delta: {delta}."
        )
        evidence = pd.concat([previous.head(max_rows), current.head(max_rows)], ignore_index=True)
        return self._supported_response(parsed, answer, comparison, evidence, max_rows=max_rows)

    def _filter_frame(
        self,
        *,
        eod_date: str | None = None,
        store_name: str | None = None,
        inventory_name: str | None = None,
        city: str | None = None,
    ) -> pd.DataFrame:
        filtered = self.df.copy()
        filters = {
            "eod_date": eod_date,
            "store_name": store_name,
            "inventory_name": inventory_name,
            "city": city,
        }
        for column, value in filters.items():
            if value:
                filtered = filtered.loc[filtered[column].astype(str) == str(value)].copy()
        if filtered.empty:
            scope = ", ".join(f"{k}={v}" for k, v in filters.items() if v)
            raise ValueError(f"No rows matched {scope}.")
        return filtered

    def _variance_group(self, filtered: pd.DataFrame, group_by: str) -> pd.DataFrame:
        grouped = (
            filtered.groupby(group_by, dropna=False)[["qty", "ideals"]]
            .sum()
            .reset_index()
        )
        grouped["sum_variance_qty"] = (grouped["qty"] - grouped["ideals"]).round(4)
        grouped["variance_pct"] = grouped.apply(
            lambda row: round((100.0 * (row["qty"] - row["ideals"]) / row["ideals"]), 4)
            if row["ideals"]
            else 0.0,
            axis=1,
        )
        return grouped.sort_values("sum_variance_qty", ascending=False).reset_index(drop=True)

    @staticmethod
    def _aggregate_variance(filtered: pd.DataFrame) -> dict[str, float]:
        qty = round(float(filtered["qty"].sum()), 4)
        ideals = round(float(filtered["ideals"].sum()), 4)
        variance_qty = round(qty - ideals, 4)
        variance_pct = round((100.0 * variance_qty / ideals), 4) if ideals else 0.0
        return {
            "total_qty": qty,
            "total_ideals": ideals,
            "variance_qty": variance_qty,
            "variance_pct": variance_pct,
        }

    def _supported_response(
        self,
        parsed: ParsedQuestion,
        answer: str,
        result_table: pd.DataFrame,
        evidence_df: pd.DataFrame,
        *,
        max_rows: int,
    ) -> OfficeQueryResponse:
        return OfficeQueryResponse(
            supported=True,
            answer=answer,
            parsed_intent=parsed.intent,
            table=self._to_records(result_table, limit=max_rows),
            evidence_rows=self._to_records(evidence_df, limit=min(5, max_rows)),
            applied_filters=self._filters_from_parse(parsed),
            notes=parsed.notes,
        )

    def _unsupported_response(
        self,
        parsed: ParsedQuestion,
        reason: str,
    ) -> OfficeQueryResponse:
        notes = list(parsed.notes)
        if self.available_dates:
            notes.append(
                f"Current demo data spans {self.available_dates[0]} to {self.available_dates[-1]}."
            )
        return OfficeQueryResponse(
            supported=False,
            answer=reason,
            parsed_intent=parsed.intent,
            table=[],
            evidence_rows=[],
            applied_filters=self._filters_from_parse(parsed),
            notes=notes,
        )

    @staticmethod
    def _filters_from_parse(parsed: ParsedQuestion) -> dict[str, str]:
        filters: dict[str, str] = {}
        if parsed.date_value:
            filters["eod_date"] = parsed.date_value
        if parsed.store_name:
            filters["store_name"] = parsed.store_name
        if parsed.inventory_name:
            filters["inventory_name"] = parsed.inventory_name
        if parsed.city:
            filters["city"] = parsed.city
        return filters

    @staticmethod
    def _to_records(dataframe: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
        preview = dataframe.head(limit).copy()
        preview = preview.where(pd.notnull(preview), None)
        records = preview.to_dict(orient="records")
        cleaned: list[dict[str, Any]] = []
        for record in records:
            formatted: dict[str, Any] = {}
            for key, value in record.items():
                if isinstance(value, float):
                    formatted[key] = round(value, 4)
                else:
                    formatted[key] = value
            cleaned.append(formatted)
        return cleaned
