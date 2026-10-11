"""Validate the notebook structure and syntax. Does not train or download data."""

import ast
import importlib.util
import json
import py_compile
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "BCCD_ViT_Classification_DETR_Detection.ipynb"
SECTIONS = [f"## Section {index}:" for index in range(1, 15)]
SNIPPETS = [
    "class PatchEmbedding",
    "class MultiHeadSelfAttention",
    "class MLPBlock",
    "class TransformerEncoderBlock",
    "class ClassificationHead",
    "class VisionTransformer",
    "class HungarianMatcher",
    "class SmallCNN",
    "DetrForObjectDetection",
    "MeanAveragePrecision",
    "def predict_cell",
    "def predict_detections",
    "def draw_detections",
    "konstantinazov/bccd-dataset",
    "SMOKE_TEST = False",
    "CrossEntropyLoss",
    "AdamW",
    "linear_sum_assignment",
    "not yet run",
]


def load_splits():
    path = ROOT / "src" / "bccd" / "splits.py"
    spec = importlib.util.spec_from_file_location("bccd_splits_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_splits():
    splits = load_splits()
    records = [
        {"stem": f"img_{index}", "split_hint": None, "image_path": f"JPEGImages/img_{index}.jpg"}
        for index in range(10)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        assigned, description = splits.assign_splits(records, Path(tmp), seed=42)
    counts = splits.split_counts(assigned)
    if min(counts.values()) < 1:
        raise SystemExit(f"Random split produced an empty partition: {counts}")
    if sum(counts.values()) != 10:
        raise SystemExit(f"Random split lost images: {counts}")
    if "seeded random split" not in description:
        raise SystemExit(description)
    tiny = [{"stem": f"img_{index}", "split_hint": None} for index in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        assigned, _ = splits.assign_splits(tiny, Path(tmp), seed=1)
    if set(splits.split_counts(assigned).values()) != {1}:
        raise SystemExit(f"Three-image split was not 1/1/1: {splits.split_counts(assigned)}")


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise SystemExit("Notebook nbformat is not 4")
    cells = notebook.get("cells")
    if not cells:
        raise SystemExit("Notebook has no cells")
    markdown = []
    code = []
    for index, cell in enumerate(cells):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if cell["cell_type"] == "markdown":
            markdown.append(source)
        elif cell["cell_type"] == "code":
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise SystemExit(f"Syntax error in code cell {index}: {exc}") from exc
            if source.lstrip().startswith(("!", "%")):
                raise SystemExit(f"Code cell {index} uses a notebook magic; validation expects plain Python")
            code.append(source)
        else:
            raise SystemExit(f"Unexpected cell type at {index}: {cell['cell_type']}")
    markdown_text = "\n".join(markdown)
    positions = []
    for title in SECTIONS:
        found = markdown_text.find(title)
        if found < 0:
            raise SystemExit(f"Missing heading {title}")
        positions.append(found)
    if positions != sorted(positions):
        raise SystemExit("Section headings are out of order")
    combined = "\n".join(code + markdown)
    for snippet in SNIPPETS:
        if snippet not in combined:
            raise SystemExit(f"Missing required text: {snippet}")
    for path in (ROOT / "src").rglob("*.py"):
        py_compile.compile(str(path), doraise=True)
    check_splits()
    print(f"Validated {NOTEBOOK}")
    print(f"Cells: {len(cells)} ({len(code)} code, {len(markdown)} markdown)")
    print("Syntax, section order, required symbols, and split helper checks passed.")
    print("Training was not run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
