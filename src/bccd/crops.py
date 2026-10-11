"""Create classification crops from detection boxes.

The crop is saved only after its source image has been assigned to train, val,
or test. A crop is rejected when its shorter side is below ``min_side`` pixels
(default 16). Those boxes are too small to recognize after resizing to the
Vision Transformer input, and they are often partial or noisy platelet marks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def export_classification_crops(
    records: list[dict],
    output_dir: Path,
    class_to_idx: dict[str, int],
    min_side: int = 16,
    image_size: int = 128,
    stamp: dict | None = None,
) -> tuple[list[dict], dict]:
    """Write one JPEG crop per accepted box and a CSV manifest.

    Images are opened one at a time. The full dataset is not held in memory.
    If ``stamp`` matches the stamp already on disk, existing crops are reused.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "crops_manifest.csv"
    stamp_path = output_dir / "crops_stamp.json"
    stamp = stamp or {}
    if manifest_path.exists() and stamp_path.exists():
        previous = json.loads(stamp_path.read_text(encoding="utf-8"))
        if all(previous.get(key) == value for key, value in stamp.items()):
            rows = _read_manifest(manifest_path)
            return rows, {
                "reused": True,
                "n_crops": len(rows),
                "manifest_path": str(manifest_path),
                "rejected_small": previous.get("rejected_small", 0),
            }

    for split in ("train", "val", "test"):
        for name in class_to_idx:
            (output_dir / split / name).mkdir(parents=True, exist_ok=True)

    rows = []
    rejected_small = 0
    rejected_unknown = 0
    for record in records:
        image_path = Path(record["image_path"])
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            boxes = [obj["bbox_xyxy"] for obj in record["objects"]]
            names = [obj["name"] for obj in record["objects"]]
            if not boxes:
                continue
            coords = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            finite = np.isfinite(coords).all(axis=1)
            side_ok = finite & ((coords[:, 2] - coords[:, 0]) >= min_side) & ((coords[:, 3] - coords[:, 1]) >= min_side)
            for index, (name, box, ok) in enumerate(zip(names, coords, side_ok)):
                if not ok:
                    rejected_small += 1
                    continue
                if name not in class_to_idx:
                    rejected_unknown += 1
                    continue
                # VOC corners are inclusive. PIL crop treats the right and lower
                # edges as exclusive, so the saved crop adds one pixel.
                x1 = max(0, min(int(np.floor(float(box[0]))), width - 1))
                y1 = max(0, min(int(np.floor(float(box[1]))), height - 1))
                x2 = max(x1 + 1, min(int(np.ceil(float(box[2]))) + 1, width))
                y2 = max(y1 + 1, min(int(np.ceil(float(box[3]))) + 1, height))
                crop = rgb.crop((x1, y1, x2, y2)).resize((image_size, image_size), Image.BILINEAR)
                filename = f"{record['stem']}_{index:03d}.jpg"
                destination = output_dir / record["split"] / name / filename
                crop.save(destination, quality=95)
                rows.append(
                    {
                        "split": record["split"],
                        "crop_path": str(destination),
                        "source_image": str(image_path),
                        "source_stem": record["stem"],
                        "class_name": name,
                        "label": class_to_idx[name],
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                    }
                )

    _write_manifest(manifest_path, rows)
    stamp_out = dict(stamp)
    stamp_out["rejected_small"] = rejected_small
    stamp_out["n_crops"] = len(rows)
    stamp_path.write_text(json.dumps(stamp_out, indent=2), encoding="utf-8")
    summary = {
        "reused": False,
        "n_crops": len(rows),
        "rejected_small": rejected_small,
        "rejected_unknown": rejected_unknown,
        "manifest_path": str(manifest_path),
        "min_side": min_side,
        "image_size": image_size,
    }
    return rows, summary


def _write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "crop_path",
        "source_image",
        "source_stem",
        "class_name",
        "label",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        for key in ("xmin", "ymin", "xmax", "ymax"):
            row[key] = int(float(row[key]))
    return rows
