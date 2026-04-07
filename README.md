---
title: StoreOps Copilot Env
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
short_description: OpenEnv benchmark for D-1 store analytics.
---

# StoreOps Copilot Environment

`storeops_env` is an OpenEnv environment for multi-step analytics over D-1 style
store inventory and variance data. It is designed to support both:

- a hackathon benchmark with structured tasks, graders, and rewards
- an office demo over a small sampled slice of real report data

This repository now supports two modes cleanly:

- `Hackathon mode`: synthetic benchmark data, richer task bank, `inference.py`,
  Docker/OpenEnv build path, and submission-facing docs
- `Office demo mode`: sampled real D-1 report, browser UI, and deterministic
  chat-plus-table answers for business questions

## Benchmark Tasks

The submission benchmark currently exposes 6 task templates with explicit
difficulty tiers:

- `easy`: `store_item_qty_total`
- `medium`: `store_top_variance_items`
- `medium`: `central_city_breakdown`
- `hard`: `central_top_stores_for_item`
- `hard`: `central_top_variance_stores_for_item`
- `hard`: `central_top_delta_stores_for_item`

Each task is graded by comparing the current dataframe view against a hidden
target dataframe for the task. The environment normalizes each successful run to
an episode score of `1.0`, keeps per-step rewards inside `[0.0, 1.0]`, and
includes the current cumulative `score` in observation metadata.

## What It Models

The environment exposes business questions such as:

- total quantity for an inventory item in a store on D-1
- top stores by item usage
- city-wise item usage
- highest variance items in a store
- highest variance stores for an item

The agent interacts by applying analytics actions to a working dataframe:

- `filter_equals`
- `group_aggregate`
- `compare_dates`
- `sort_limit`
- `reset_view`
- `submit`

Observations include:

- question, role, category, and difficulty
- available dimensions and metrics
- preview of the current dataframe
- steps remaining
- status message
- metadata containing `task_id`, `difficulty`, `step_count`, and cumulative `score`

## Data Strategy

- `data/synthetic_seed.csv` provides a tiny bundled dataset for tests
- `data/synthetic_benchmark.csv` is the richer default benchmark dataset for
  submission mode
- `scripts/build_real_sample.py` builds a small office demo sample from the S3 report
- `scripts/generate_synthetic_data.py` creates anonymized synthetic data with the same schema

## Quick Start

```bash
uv sync
uv run --with pytest pytest -q
uv run server --port 8000
```

Then open:

- `http://localhost:8000/` for the browser UI
- `http://localhost:8000/docs` for FastAPI docs

## Submission Mode

The benchmark environment defaults to `data/synthetic_benchmark.csv`. Run the
environment locally:

```bash
uv run server --port 8000
```

Run the submission agent against a local server:

```bash
uv run python inference.py
```

Run the submission agent against a Docker image:

```bash
STOREOPS_IMAGE_NAME=storeops_env:latest uv run python inference.py
```

Planner modes:

- `STOREOPS_PLANNER_MODE=heuristic` for deterministic benchmark runs
- `STOREOPS_PLANNER_MODE=auto` to try an LLM plan first and fall back to the heuristic
- `STOREOPS_PLANNER_MODE=llm` to force an LLM plan

Optional LLM planning variables:

- `API_BASE_URL`
- `MODEL_NAME` (recommended starting point: `gpt-4.1-mini`)
- `HF_TOKEN` or `API_KEY`

Reproducible baseline variables:

- `STOREOPS_RESET_SEED` selects the deterministic benchmark task for `inference.py`
- `STOREOPS_PLANNER_MODE=heuristic` is the recommended submission default

Build the Docker image with OpenEnv:

```bash
uv run openenv build . -c . -f Dockerfile -t storeops_env:latest
```

Push to Hugging Face Spaces:

```bash
uv run openenv push . -r <your-hf-username>/storeops-env
```

Validate before submission:

```bash
uv run openenv validate .
uv run --with pytest pytest -q
bash scripts/submission_smoke_test.sh
```

Root environment variables expected by the hackathon checker are listed in
`.env.example`.

Manual Hugging Face deployment helper:

```bash
uv run python scripts/deploy_hf_space.py --repo-id <your-hf-username>/storeops-env
```

## Office Demo API

The same server now exposes a simple office-facing query endpoint:

```bash
curl -X POST http://localhost:8000/office/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How much Inventory_001 was used in Store_001 yesterday?",
    "max_rows": 5
  }'
```

Capabilities summary:

```bash
curl http://localhost:8000/office/capabilities
```

To force the office demo to use the sampled real CSV:

```bash
STOREOPS_OFFICE_DATASET_PATH=data/office_demo_sample.csv uv run server --port 8000
```

Supported now:

- item quantity in a store on D-1
- top stores for an item
- city-wise item quantity
- highest variance items in a store
- item variance and variance percentage
- day-over-day variance comparison

Not supported yet from the D-1 report alone:

- current stock-on-hand
- out-of-stock / running-low alerts
- expiry questions
- minimum stock / reorder logic
- who updated stock

## Build An Office Demo Sample

```bash
python3 scripts/build_real_sample.py \
  --s3-path s3://dsa.faasos.io/analytics/reports/glob_ol_ws_ifc/ol_ws_ifc_reports/ol_ifc_data_India_2026-04-04.csv \
  --output data/office_demo_sample.csv \
  --rows 1500
```

## Generate Synthetic Data

```bash
python3 scripts/generate_synthetic_data.py \
  --input data/synthetic_seed.csv \
  --output data/synthetic_benchmark.csv \
  --rows 800
```
