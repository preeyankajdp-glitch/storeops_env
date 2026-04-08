"""Create or update a Hugging Face Docker Space from this repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_VARIABLES = {
    "API_BASE_URL": "https://api.openai.com/v1",
    "MODEL_NAME": "gpt-4.1-mini",
    "STOREOPS_PLANNER_MODE": "auto",
    "STOREOPS_RESET_SEED": "11",
}

IGNORE_PATTERNS = [
    ".git/*",
    ".venv/*",
    "__pycache__/*",
    ".pytest_cache/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "build/*",
    "dist/*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_SPACE_REPO_ID"),
        help="Hugging Face Space repo ID, e.g. username/storeops-env",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private.",
    )
    parser.add_argument(
        "--skip-secret",
        action="store_true",
        help="Do not push HF_TOKEN as a Space secret.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_id = args.repo_id
    if not repo_id:
        raise SystemExit("Missing --repo-id or HF_SPACE_REPO_ID.")

    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    variables = [
        {
            "key": key,
            "value": os.getenv(key, default),
            "description": f"StoreOps setting for {key}",
        }
        for key, default in DEFAULT_VARIABLES.items()
    ]

    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
        space_variables=variables,
        token=token,
    )

    for variable in variables:
        api.add_space_variable(
            repo_id=repo_id,
            key=variable["key"],
            value=variable["value"],
            description=variable["description"],
            token=token,
        )

    if token and not args.skip_secret:
        api.add_space_secret(
            repo_id=repo_id,
            key="HF_TOKEN",
            value=token,
            description="API key for optional LLM planner calls.",
            token=token,
        )

    root = Path(__file__).resolve().parents[1]
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=root,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy StoreOps Docker Space",
        token=token,
    )

    subdomain = repo_id.replace("_", "-").replace("/", "-")
    print(f"Space repo: {repo_url}")
    print(f"Upload commit: {commit_info.commit_url}")
    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print(f"Space app URL: https://{subdomain}.hf.space")


if __name__ == "__main__":
    main()
