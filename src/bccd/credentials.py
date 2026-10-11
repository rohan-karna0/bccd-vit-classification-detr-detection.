"""Load API keys from the project .env file. Nothing is hardcoded."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv


def load_env(project_root: Path) -> Path:
    env_path = Path(project_root) / ".env"
    load_dotenv(env_path)
    return env_path


def apply_kaggle_credentials() -> None:
    """Point the Kaggle client at the key stored in the environment.

    The current Kaggle client accepts ``KAGGLE_API_TOKEN``. The older
    ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` pair still works as a fallback.
    """
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
        return
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        destination = Path.home() / ".kaggle"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "kaggle.json"
        path.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return
    raise RuntimeError(
        "Kaggle credentials are missing. Copy .env.example to .env and set "
        "KAGGLE_API_TOKEN (from https://www.kaggle.com/settings/api). "
        "The older KAGGLE_USERNAME and KAGGLE_KEY pair also works."
    )


def huggingface_token() -> str | None:
    """Return HF_TOKEN when it is set. The public DETR checkpoint does not require it."""
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    if not token:
        return None
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token
