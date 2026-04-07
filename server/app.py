"""FastAPI application for the StoreOps analytics environment."""

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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


@lru_cache(maxsize=1)
def get_query_service() -> StoreOpsQueryService:
    """Reuse a single query service instance for office-style analytics queries."""
    return StoreOpsQueryService()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the browser UI directly so the Space root returns HTTP 200."""
    return FileResponse(STATIC_DIR / "index.html")


@app.head("/", include_in_schema=False)
def home_head() -> Response:
    """Allow lightweight uptime checks against the Space root."""
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
