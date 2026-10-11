"""Read and write metric files produced by notebook training cells.

Numbers in the final report come only from these files. If a file is missing,
the report says the run has not been executed.
"""

from __future__ import annotations

import json
from pathlib import Path


def save_metrics(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metrics(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_row_or_missing(path: Path, model_name: str) -> dict:
    payload = load_metrics(path)
    if payload is None:
        return {
            "model": model_name,
            "training_configuration": "not yet run",
            "validation_metrics": "not yet run",
            "test_metrics": "not yet run",
            "training_duration": "not yet run",
            "inference_latency": "not yet run",
            "limitations": "Execute the training and evaluation cells on a GPU. This file is created only after that run.",
        }
    return {
        "model": payload.get("model", model_name),
        "training_configuration": payload.get("training_configuration", ""),
        "validation_metrics": payload.get("validation_metrics", ""),
        "test_metrics": payload.get("test_metrics", ""),
        "training_duration": payload.get("training_duration", ""),
        "inference_latency": payload.get("inference_latency", ""),
        "limitations": payload.get("limitations", ""),
    }
