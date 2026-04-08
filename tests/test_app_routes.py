from pathlib import Path
import os

from fastapi.testclient import TestClient

from storeops_env.server.app import app, get_query_service


def _seed_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "synthetic_seed.csv"


def test_office_query_route_returns_structured_answer():
    os.environ["STOREOPS_OFFICE_DATASET_PATH"] = str(_seed_path())
    get_query_service.cache_clear()
    client = TestClient(app)

    response = client.post(
        "/office/query",
        json={
            "question": "How much Paper Napkin was used in Laxmi Nagar Store yesterday?",
            "max_rows": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert payload["table"]


def test_office_capabilities_route_exposes_summary():
    os.environ["STOREOPS_OFFICE_DATASET_PATH"] = str(_seed_path())
    get_query_service.cache_clear()
    client = TestClient(app)

    response = client.get("/office/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert "supported_patterns" in payload
    assert payload["available_dates"]
    assert payload["sample_questions"]


def test_root_serves_ui_with_success_status():
    os.environ["STOREOPS_OFFICE_DATASET_PATH"] = str(_seed_path())
    get_query_service.cache_clear()
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "StoreOps Assistant" in response.text


def test_root_head_returns_success():
    os.environ["STOREOPS_OFFICE_DATASET_PATH"] = str(_seed_path())
    get_query_service.cache_clear()
    client = TestClient(app)

    response = client.head("/")

    assert response.status_code == 200


def test_ui_route_serves_html():
    os.environ["STOREOPS_OFFICE_DATASET_PATH"] = str(_seed_path())
    get_query_service.cache_clear()
    client = TestClient(app)

    response = client.get("/ui")

    assert response.status_code == 200
    assert "StoreOps Assistant" in response.text


def test_tasks_route_exposes_multiple_validator_tasks():
    client = TestClient(app)

    response = client.get("/tasks")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 3
    assert "store_item_qty_total" in payload


def test_grader_route_returns_strict_in_range_score():
    client = TestClient(app)

    response = client.post("/grader", json={"task_id": "store_item_qty_total"})

    assert response.status_code == 200
    payload = response.json()
    assert 0.0 < payload["score"] < 1.0


def test_validate_route_exposes_task_summary():
    client = TestClient(app)

    response = client.get("/validate")

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["task_count"] >= 3
