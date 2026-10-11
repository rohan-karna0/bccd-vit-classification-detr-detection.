"""Discover BCCD files and parse Pascal VOC XML annotations.

The verified Kaggle mirror ``konstantinazov/bccd-dataset`` contains:

    BCCD/JPEGImages/BloodImage_00XYZ.jpg
    BCCD/Annotations/BloodImage_00XYZ.xml
    BCCD/ImageSets/Main/{train,val,test}.txt

A Roboflow-style mirror such as ``quangnguynvnnn/bccd-dataset-pascal-voc``
stores images and XML together under ``train``, ``valid``, and ``test``.
This module walks the extracted directory instead of assuming one layout.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_FROM_DIR = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
CANONICAL_NAMES = {
    "rbc": "RBC",
    "red blood cell": "RBC",
    "redbloodcell": "RBC",
    "wbc": "WBC",
    "white blood cell": "WBC",
    "whitebloodcell": "WBC",
    "platelet": "Platelets",
    "platelets": "Platelets",
}
PREFERRED_CLASS_ORDER = ("RBC", "WBC", "Platelets")


def canonical_class_name(name: str) -> str:
    key = " ".join(name.strip().lower().replace("_", " ").split())
    compact = key.replace(" ", "")
    return CANONICAL_NAMES.get(key, CANONICAL_NAMES.get(compact, name.strip()))


def format_tree(root: Path, max_depth: int = 3, max_lines: int = 80) -> str:
    """Return a short directory listing so a download can be checked by eye."""
    root = Path(root)
    lines = [str(root)]
    count = 0

    def walk(directory: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_lines:
            return
        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            lines.append(f"{'  ' * depth}[unreadable: {exc}]")
            return
        for child in children:
            if count >= max_lines:
                lines.append(f"{'  ' * depth}...")
                return
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{'  ' * depth}{child.name}{suffix}")
            count += 1
            if child.is_dir() and child.name not in {".git", "__pycache__", "__MACOSX"}:
                walk(child, depth + 1)

    walk(root, 1)
    if count >= max_lines:
        lines.append("(listing truncated)")
    return "\n".join(lines)


def discover_files(root: Path) -> dict:
    """Recursively collect images, XML, JSON, CSV, and YAML files."""
    root = Path(root)
    grouped = {"images": [], "xml": [], "json": [], "csv": [], "yaml": []}
    if not root.exists():
        return grouped
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", "__MACOSX"} for part in path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            grouped["images"].append(path)
        elif suffix == ".xml":
            grouped["xml"].append(path)
        elif suffix == ".json":
            grouped["json"].append(path)
        elif suffix == ".csv":
            grouped["csv"].append(path)
        elif suffix in {".yaml", ".yml"}:
            grouped["yaml"].append(path)
    for key in grouped:
        grouped[key] = sorted(grouped[key])
    return grouped


def _split_hint(path: Path) -> str | None:
    for part in path.parts:
        mapped = SPLIT_FROM_DIR.get(part.lower())
        if mapped is not None:
            return mapped
    return None


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def parse_voc_xml(path: Path) -> dict:
    """Parse one Pascal VOC annotation file."""
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse XML annotation: {path}") from exc
    root = tree.getroot()
    if _local_name(root.tag) != "annotation":
        raise ValueError(f"XML root is '{root.tag}', expected 'annotation': {path}")

    size = root.find("size")
    width = height = None
    if size is not None:
        width_text = size.findtext("width")
        height_text = size.findtext("height")
        if width_text and height_text:
            width = int(float(width_text))
            height = int(float(height_text))

    objects = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not raw_name or box is None:
            continue
        try:
            coords = [
                float(box.findtext("xmin")),
                float(box.findtext("ymin")),
                float(box.findtext("xmax")),
                float(box.findtext("ymax")),
            ]
        except (TypeError, ValueError):
            continue
        objects.append(
            {
                "name": canonical_class_name(raw_name),
                "raw_name": raw_name,
                "bbox_xyxy": coords,
            }
        )
    return {
        "filename": (root.findtext("filename") or path.stem).strip(),
        "width": width,
        "height": height,
        "objects": objects,
    }


def _index_by_stem(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.stem, []).append(path)
    return grouped


def _choose_pair(stem: str, images: list[Path], xmls: list[Path]) -> tuple[Path, Path]:
    """Prefer the conventional JPEGImages/Annotations pairing when stems repeat."""
    if len(images) == 1 and len(xmls) == 1:
        return images[0], xmls[0]

    def score(image: Path, xml: Path) -> tuple[int, int]:
        image_ok = 1 if "JPEGImages" in image.parts else 0
        xml_ok = 1 if "Annotations" in xml.parts else 0
        same_parent = 1 if image.parent == xml.parent else 0
        return (image_ok + xml_ok + same_parent, -len(image.parts))

    return max(((image, xml) for image in images for xml in xmls), key=lambda pair: score(*pair))


def _read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def build_records(root: Path) -> tuple[list[dict], dict]:
    """Pair images with VOC XML and return lightweight annotation records.

    Pixel arrays are not stored. Invalid boxes are removed here.
    """
    discovered = discover_files(root)
    images = _index_by_stem(discovered["images"])
    xmls = _index_by_stem(discovered["xml"])
    stems = sorted(set(images) & set(xmls))
    records = []
    class_counts: Counter[str] = Counter()
    invalid_boxes = 0
    empty_images = 0
    parse_errors = []

    for stem in stems:
        image_path, xml_path = _choose_pair(stem, images[stem], xmls[stem])
        try:
            parsed = parse_voc_xml(xml_path)
        except ValueError as exc:
            parse_errors.append(str(exc))
            continue
        width = parsed["width"]
        height = parsed["height"]
        if not width or not height:
            width, height = _read_image_size(image_path)
        raw_boxes = [obj["bbox_xyxy"] for obj in parsed["objects"]]
        names = [obj["name"] for obj in parsed["objects"]]
        if raw_boxes:
            coords = np.asarray(raw_boxes, dtype=np.float32).reshape(-1, 4)
            finite = np.isfinite(coords).all(axis=1)
            clipped = coords.copy()
            limit_w = max(width - 1, 0)
            limit_h = max(height - 1, 0)
            clipped[:, 0] = np.clip(clipped[:, 0], 0, limit_w)
            clipped[:, 2] = np.clip(clipped[:, 2], 0, limit_w)
            clipped[:, 1] = np.clip(clipped[:, 1], 0, limit_h)
            clipped[:, 3] = np.clip(clipped[:, 3], 0, limit_h)
            wide = clipped[:, 2] - clipped[:, 0]
            high = clipped[:, 3] - clipped[:, 1]
            keep = finite & (wide >= 1.0) & (high >= 1.0)
            objects = [
                {"name": name, "bbox_xyxy": box.tolist()}
                for name, box, ok in zip(names, clipped, keep)
                if ok
            ]
            invalid_boxes += int((~keep).sum())
        else:
            objects = []
        if not objects:
            empty_images += 1
        for obj in objects:
            class_counts[obj["name"]] += 1
        records.append(
            {
                "stem": stem,
                "image_path": str(image_path),
                "xml_path": str(xml_path),
                "width": int(width),
                "height": int(height),
                "split_hint": _split_hint(image_path) or _split_hint(xml_path),
                "objects": objects,
            }
        )

    summary = {
        "root": str(root),
        "n_images": len(discovered["images"]),
        "n_xml": len(discovered["xml"]),
        "n_json": len(discovered["json"]),
        "n_csv": len(discovered["csv"]),
        "n_yaml": len(discovered["yaml"]),
        "n_paired": len(records),
        "n_unpaired_images": len(set(images) - set(xmls)),
        "n_unpaired_xml": len(set(xmls) - set(images)),
        "n_objects": int(sum(class_counts.values())),
        "n_empty_images": empty_images,
        "n_invalid_boxes": invalid_boxes,
        "class_counts": dict(class_counts),
        "parse_errors": parse_errors,
        "annotation_format": "pascal_voc_xml" if records else "not_found",
        "yaml_files": [str(path) for path in discovered["yaml"][:10]],
    }
    return records, summary


def class_to_index(records: list[dict]) -> dict[str, int]:
    """Map class names to ids, keeping RBC, WBC, Platelets first when present."""
    found = {obj["name"] for record in records for obj in record["objects"]}
    ordered = [name for name in PREFERRED_CLASS_ORDER if name in found]
    ordered.extend(sorted(found - set(ordered)))
    return {name: index for index, name in enumerate(ordered)}
