from pathlib import Path

from storeops_env.server.query_service import StoreOpsQueryService


def _seed_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "synthetic_seed.csv"


def test_query_service_answers_store_item_quantity_question():
    service = StoreOpsQueryService(dataset_path=_seed_path())

    response = service.answer(
        "How much Burger Bun was used in Laxmi Nagar Store yesterday?",
        max_rows=5,
    )

    assert response.supported is True
    assert response.parsed_intent == "store_item_qty_total"
    assert response.table
    assert response.table[0]["inventory_name"] == "Burger Bun"
    assert response.table[0]["store_name"] == "Laxmi Nagar Store"


def test_query_service_rejects_live_stock_question():
    service = StoreOpsQueryService(dataset_path=_seed_path())

    response = service.answer("What is the current stock level of Burger Bun in Laxmi Nagar Store?")

    assert response.supported is False
    assert "live stock" in response.answer.casefold() or "stock-on-hand" in response.answer.casefold()


def test_query_service_generates_dataset_aware_sample_questions():
    service = StoreOpsQueryService(dataset_path=_seed_path())

    questions = service.sample_questions()

    assert questions
    first_response = service.answer(questions[0])
    assert first_response.supported is True
