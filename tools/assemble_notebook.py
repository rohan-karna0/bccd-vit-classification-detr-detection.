"""Build the interview notebook from the cell sources."""

import ast
import importlib.util
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = [ROOT / "tools" / "nb_part1.py", ROOT / "tools" / "nb_part2.py"]
OUTPUT = ROOT / "notebooks" / "BCCD_ViT_Classification_DETR_Detection.ipynb"


def load_cells(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CELLS


def as_lines(source):
    if not source.endswith("\n"):
        source += "\n"
    return source.splitlines(keepends=True)


def main():
    raw_cells = []
    for part in PARTS:
        raw_cells.extend(load_cells(part))
    notebook_cells = []
    for kind, source in raw_cells:
        if kind == "code":
            ast.parse(source)
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": as_lines(source),
                "id": uuid.uuid4().hex[:12],
            }
        else:
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": as_lines(source),
                "id": uuid.uuid4().hex[:12],
            }
        notebook_cells.append(cell)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": notebook_cells,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(notebook_cells)} cells")


if __name__ == "__main__":
    main()
