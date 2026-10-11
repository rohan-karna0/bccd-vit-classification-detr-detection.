"""Split source images before any cell crops are created.

Crops from the same microscopy field must stay in one split. Otherwise a
classifier can memorize a field that also appears in the test set.
"""

from __future__ import annotations

import random
from pathlib import Path


def _read_id_file(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(Path(line.split()[0]).stem)
    return ids


def find_imageset_splits(root: Path) -> dict[str, set[str]] | None:
    """Load ``ImageSets/Main/{train,val,test}.txt`` when the dataset provides them."""
    root = Path(root)
    mains = [path for path in root.rglob("Main") if path.is_dir() and path.parent.name == "ImageSets"]
    if not mains:
        return None
    names = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    splits: dict[str, set[str]] = {}
    for main in mains:
        for path in main.glob("*.txt"):
            key = names.get(path.stem.lower())
            if key is None:
                continue
            splits.setdefault(key, set()).update(_read_id_file(path))
    if "train" not in splits or "test" not in splits:
        return None
    return splits


def _covered(records: list[dict], id_map: dict[str, set[str]]) -> float:
    stems = {record["stem"] for record in records}
    if not stems:
        return 0.0
    known = set().union(*id_map.values()) if id_map else set()
    return len(stems & known) / len(stems)


def assign_splits(
    records: list[dict],
    dataset_root: Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[list[dict], str]:
    """Return records annotated with ``split`` and a short description of the rule used."""
    if not records:
        raise ValueError("No image records are available to split.")

    hinted = [record["split_hint"] for record in records]
    hint_fraction = sum(hint is not None for hint in hinted) / len(records)
    distinct_hints = {hint for hint in hinted if hint is not None}
    if hint_fraction >= 0.9 and len(distinct_hints) >= 2:
        assigned = []
        for record in records:
            if record["split_hint"] is None:
                raise ValueError(f"Image {record['stem']} has no train/val/test folder hint.")
            updated = dict(record)
            updated["split"] = record["split_hint"]
            assigned.append(updated)
        _assert_disjoint(assigned)
        return assigned, "folder names (train/val/test directories already present in the archive)"

    imagesets = find_imageset_splits(dataset_root)
    if imagesets is not None and _covered(records, imagesets) >= 0.9:
        names = list(imagesets)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = imagesets[left] & imagesets[right]
                if overlap:
                    example = next(iter(overlap))
                    raise ValueError(f"ImageSets lists {example} in both {left} and {right}.")
        assigned = []
        missing = []
        for record in records:
            split = next((name for name, ids in imagesets.items() if record["stem"] in ids), None)
            if split is None:
                missing.append(record["stem"])
                continue
            updated = dict(record)
            updated["split"] = split
            assigned.append(updated)
        if missing:
            raise ValueError(
                "ImageSets/Main did not list every paired image. Missing examples: "
                + ", ".join(missing[:8])
            )
        _assert_disjoint(assigned)
        return assigned, "official ImageSets/Main train, val, and test lists"

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    stems = sorted({record["stem"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    if n < 3:
        raise ValueError(f"Need at least 3 source images to build train/val/test, found {n}.")
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train -= 1 - n_test
        n_test = 1
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"Could not build three non-empty splits from {n} images.")
    split_of = {}
    for stem in stems[:n_train]:
        split_of[stem] = "train"
    for stem in stems[n_train : n_train + n_val]:
        split_of[stem] = "val"
    for stem in stems[n_train + n_val :]:
        split_of[stem] = "test"
    assigned = []
    for record in records:
        updated = dict(record)
        updated["split"] = split_of[record["stem"]]
        assigned.append(updated)
    _assert_disjoint(assigned)
    return assigned, f"seeded random split by source image (seed={seed}, ratios={ratios})"


def _assert_disjoint(records: list[dict]) -> None:
    grouped = {"train": set(), "val": set(), "test": set()}
    for record in records:
        grouped[record["split"]].add(record["stem"])
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        overlap = grouped[left] & grouped[right]
        if overlap:
            example = next(iter(overlap))
            raise ValueError(f"Source image {example} appears in both {left} and {right}.")


def split_counts(records: list[dict]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        counts[record["split"]] += 1
    return counts
