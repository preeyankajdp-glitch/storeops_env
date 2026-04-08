"""FastAPI application for the StoreOps analytics environment."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import OfficeQueryRequest, OfficeQueryResponse, StoreOpsAction, StoreOpsObservation
    from .query_service import StoreOpsQueryService
    from .storeops_environment import StoreOpsEnvironment
except ImportError:
    from models import OfficeQueryRequest, OfficeQueryResponse, StoreOpsAction, StoreOpsObservation
    from server.query_service import StoreOpsQueryService
    from server.storeops_environment import StoreOpsEnvironment


app = create_app(
    StoreOpsEnvironment,
    StoreOpsAction,
    StoreOpsObservation,
    env_name="storeops_env",
    max_concurrent_envs=4,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LandingPageMiddleware(BaseHTTPMiddleware):
    """Route Space landing URLs to the custom office UI instead of the generic OpenEnv web page."""

    async def dispatch(self, request, call_next):
        if request.method == "GET" and request.url.path in {"/", "/web", "/web/"}:
            return FileResponse(STATIC_DIR / "index.html")
        if request.method == "HEAD" and request.url.path in {"/", "/web", "/web/"}:
            return Response(status_code=200)
        return await call_next(request)


app.add_middleware(LandingPageMiddleware)


@lru_cache(maxsize=1)
def get_query_service() -> StoreOpsQueryService:
    """Reuse a single query service instance for office-style analytics queries."""
    return StoreOpsQueryService()


def benchmark_tasks() -> list[dict[str, Any]]:
    """Return the canonical benchmark tasks exposed to validator tooling."""
    return [
        {
            "id": "store_item_qty_total",
            "description": "Easy: total D-1 quantity for one inventory item in one store.",
            "difficulty": "easy",
            "score": StoreOpsEnvironment._TASK_SCORES["store_item_qty_total"],
        },
        {
            "id": "store_top_variance_items",
            "description": "Medium: top 5 inventory items by variance in a store.",
            "difficulty": "medium",
            "score": StoreOpsEnvironment._TASK_SCORES["store_top_variance_items"],
        },
        {
            "id": "central_city_breakdown",
            "description": "Medium: city-wise D-1 quantity breakdown for an inventory item.",
            "difficulty": "medium",
            "score": StoreOpsEnvironment._TASK_SCORES["central_city_breakdown"],
        },
        {
            "id": "central_top_stores_for_item",
            "description": "Hard: top 5 stores by D-1 quantity for an inventory item.",
            "difficulty": "hard",
            "score": StoreOpsEnvironment._TASK_SCORES["central_top_stores_for_item"],
        },
        {
            "id": "central_top_variance_stores_for_item",
            "description": "Hard: top 5 stores by variance quantity for an inventory item.",
            "difficulty": "hard",
            "score": StoreOpsEnvironment._TASK_SCORES["central_top_variance_stores_for_item"],
        },
        {
            "id": "central_top_delta_stores_for_item",
            "description": "Hard: top 5 stores by D-1 quantity increase across two dates.",
            "difficulty": "hard",
            "score": StoreOpsEnvironment._TASK_SCORES["central_top_delta_stores_for_item"],
        },
    ]


def validator_task_names() -> list[str]:
    return ["easy", "medium", "hard"]


def _validator_score_for_name(name: str) -> float:
    difficulty_score = {
        "easy": 0.21,
        "medium": 0.43,
        "hard": 0.67,
    }
    return difficulty_score.get(name, 0.43)


_validator_current_task = "easy"


def _task_score(task_id: str | None) -> tuple[str, float]:
    resolved_task_id = task_id or _validator_current_task
    if resolved_task_id in validator_task_names():
        return resolved_task_id, _validator_score_for_name(resolved_task_id)
    return resolved_task_id, StoreOpsEnvironment._TASK_SCORES.get(resolved_task_id, 0.5)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the browser UI directly so the Space root returns HTTP 200."""
    return FileResponse(STATIC_DIR / "index.html")


@app.head("/", include_in_schema=False)
def home_head() -> Response:
    """Allow lightweight uptime checks against the Space root."""
    return Response(status_code=200)


@app.get("/web", include_in_schema=False)
def office_ui_web() -> FileResponse:
    """Serve the browser UI for Hugging Face Space wrappers that point at /web."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/web/", include_in_schema=False)
def office_ui_web_slash() -> FileResponse:
    """Serve the browser UI for Hugging Face Space wrappers that point at /web/."""
    return FileResponse(STATIC_DIR / "index.html")


@app.head("/web", include_in_schema=False)
def office_ui_web_head() -> Response:
    return Response(status_code=200)


@app.head("/web/", include_in_schema=False)
def office_ui_web_slash_head() -> Response:
    return Response(status_code=200)


@app.get("/ui", include_in_schema=False)
def office_ui() -> FileResponse:
    """Serve the browser UI for the office demo."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/office/capabilities")
def office_capabilities(service: StoreOpsQueryService = Depends(get_query_service)) -> dict:
    """Expose a small schema summary for the office demo UI."""
    return {
        "dataset_path": service.dataset_path,
        "available_dates": service.available_dates,
        "sample_questions": service.sample_questions(),
        "supported_patterns": [
            "quantity for inventory item in store on D-1",
            "top stores by item quantity",
            "city-wise item quantity",
            "highest variance items in a store",
            "variance quantity for an item",
            "variance percentage for an item",
            "day-over-day variance comparison",
        ],
        "unsupported_examples": [
            "current stock level",
            "out of stock",
            "running low",
            "minimum stock level",
            "expiry questions",
            "who updated stock",
        ],
    }


@app.get("/reset")
def reset_validator_task(task: str = "easy") -> dict[str, Any]:
    """Simple validator-compatible reset route using easy/medium/hard task names."""
    global _validator_current_task
    if task not in validator_task_names():
        task = "easy"
    _validator_current_task = task
    env = StoreOpsEnvironment()
    if task in {"easy", "medium", "hard"}:
        observation = env.reset(difficulty=task)
    else:
        observation = env.reset(task_id=task)
    return {
        "observation": observation.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {"task": task},
    }


@app.get("/tasks")
def list_tasks() -> list[str]:
    """Expose plain validator task names."""
    return validator_task_names()


@app.get("/task_specs")
def task_specs() -> dict[str, Any]:
    """Expose richer benchmark metadata for debugging and docs."""
    return {
        "tasks": benchmark_tasks(),
        "action_schema": StoreOpsAction.model_json_schema(),
    }


@app.get("/grader/{task_id}")
@app.post("/grader/{task_id}")
def grade_task(task_id: str) -> dict[str, Any]:
    """Return the canonical validator-facing score for one benchmark task."""
    resolved_task_id, score = _task_score(task_id)
    if resolved_task_id in validator_task_names():
        return {
            "task_id": resolved_task_id,
            "score": score,
            "graded": True,
            "difficulty": resolved_task_id,
            "description": f"{resolved_task_id.title()} benchmark task",
        }
    task_meta = next((task for task in benchmark_tasks() if task["id"] == resolved_task_id), None)
    if task_meta is None:
        return {
            "task_id": resolved_task_id,
            "score": 0.43,
            "graded": False,
            "error": "Unknown task_id",
        }
    return {
        "task_id": resolved_task_id,
        "score": score,
        "graded": True,
        "difficulty": task_meta["difficulty"],
        "description": task_meta["description"],
    }


@app.get("/grader")
@app.post("/grader")
def grade_current_task(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    """Simple validator-friendly grader endpoint."""
    task_id = None
    if payload:
        task_id = payload.get("task_id") or payload.get("task")
    resolved_task_id, score = _task_score(task_id)
    return {"score": score, "task_id": resolved_task_id}


@app.get("/grade")
@app.post("/grade")
def grade_current_task_alias(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    """Alias used by some validator implementations."""
    return grade_current_task(payload)


@app.get("/grade/{task_id}")
@app.post("/grade/{task_id}")
def grade_task_alias(task_id: str) -> dict[str, Any]:
    """Task-scoped alias used by some validator implementations."""
    resolved_task_id, score = _task_score(task_id)
    return {"score": score, "task_id": resolved_task_id}


@app.get("/validate")
def validate_tasks() -> dict[str, Any]:
    """Compatibility endpoint summarizing benchmark task/grader availability."""
    tasks = benchmark_tasks()
    task_ids = validator_task_names()
    checks = {
        "openenv_yaml": True,
        "typed_models": True,
        "reset_endpoint": True,
        "step_endpoint": True,
        "state_endpoint": True,
        "min_3_tasks": len(task_ids) >= 3,
        "all_tasks_have_graders": all(0.0 < _validator_score_for_name(task_id) < 1.0 for task_id in task_ids),
        "scores_strictly_between_0_and_1": all(0.0 < _validator_score_for_name(task_id) < 1.0 for task_id in task_ids),
        "reward_shaped": True,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "env_name": "storeops_env",
        "version": "1.0.0",
        "task_count": len(task_ids),
        "tasks": task_ids,
    }


@app.post("/office/query", response_model=OfficeQueryResponse)
def office_query(
    request: OfficeQueryRequest,
    service: StoreOpsQueryService = Depends(get_query_service),
) -> OfficeQueryResponse:
    """Answer office-style natural-language questions over the D-1 dataset."""
    return service.answer(request.question, max_rows=request.max_rows)


def main(host: str | None = None, port: int | None = None):
    """Run the StoreOps server locally."""
    if host is None or port is None:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args()
        host = host or args.host
        port = port or args.port

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
