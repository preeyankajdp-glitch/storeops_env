from pathlib import Path

from storeops_env.server.analytics_engine import StoreOpsAnalyticsEngine


def _seed_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "synthetic_seed.csv"


def test_engine_loads_and_exposes_dimensions_and_metrics():
    engine = StoreOpsAnalyticsEngine.from_csv(_seed_path())

    assert "store_name" in engine.available_dimensions
    assert "qty" in engine.available_metrics
    assert "variance_qty" in engine.available_metrics


def test_filter_group_and_sort_pipeline():
    engine = StoreOpsAnalyticsEngine.from_csv(_seed_path())

    assert engine.filter_equals("eod_date", "2026-04-04").ok
    assert engine.filter_equals("inventory_name", "Paper Napkin").ok
    assert engine.group_aggregate("store_name", "qty", "sum").ok
    assert engine.sort_limit("sum_qty", descending=True, limit=5).ok

    preview = engine.preview()
    assert preview
    assert preview[0]["sum_qty"] >= preview[-1]["sum_qty"]


def test_variance_metric_is_derived():
    engine = StoreOpsAnalyticsEngine.from_csv(_seed_path())

    assert engine.filter_equals("store_name", "Laxmi Nagar Store").ok
    assert engine.filter_equals("inventory_name", "Burger Bun").ok
    assert engine.group_aggregate("eod_date", "variance_qty", "sum").ok

    preview = engine.preview()
    assert len(preview) == 2
