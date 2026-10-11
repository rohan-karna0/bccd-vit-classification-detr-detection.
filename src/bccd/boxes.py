"""Bounding-box conversions for Pascal VOC and DETR.

VOC boxes are absolute pixel coordinates in ``xyxy`` order:
``xmin, ymin, xmax, ymax``, with the origin at the top-left of the image.

DETR boxes are normalized ``cx, cy, w, h`` in ``[0, 1]``, relative to the
image that is passed to the image processor. Aspect-ratio-preserving resize
does not change those normalized coordinates.
"""

from __future__ import annotations

import numpy as np


def sanitize_xyxy(
    boxes: np.ndarray,
    width: int,
    height: int,
    min_side: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Clip boxes to the image and drop boxes that are empty or too small.

    Returns the kept boxes and the number of dropped boxes.
    """
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32), 0

    coords = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    finite = np.isfinite(coords).all(axis=1)
    coords = coords[finite]
    dropped = int((~finite).sum())
    if coords.size == 0:
        return np.zeros((0, 4), dtype=np.float32), dropped

    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
    coords[:, 3] = np.clip(coords[:, 3], 0, height - 1)

    widths = coords[:, 2] - coords[:, 0]
    heights = coords[:, 3] - coords[:, 1]
    keep = (widths >= min_side) & (heights >= min_side)
    dropped += int((~keep).sum())
    return coords[keep], dropped


def hflip_xyxy(boxes: np.ndarray, width: float) -> np.ndarray:
    """Mirror ``xyxy`` boxes when the image is flipped left-right."""
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    flipped = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    x1 = flipped[:, 0].copy()
    x2 = flipped[:, 2].copy()
    flipped[:, 0] = width - x2
    flipped[:, 2] = width - x1
    return flipped


def scale_xyxy(boxes: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Scale ``xyxy`` boxes after a resize that changes width and height."""
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    scaled = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    return scaled


def xyxy_to_cxcywh_norm(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    """Convert absolute ``xyxy`` pixels to normalized ``cxcywh``."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    coords = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx = (coords[:, 0] + coords[:, 2]) * 0.5 / width
    cy = (coords[:, 1] + coords[:, 3]) * 0.5 / height
    w = (coords[:, 2] - coords[:, 0]) / width
    h = (coords[:, 3] - coords[:, 1]) / height
    out = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def cxcywh_norm_to_xyxy(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    """Convert normalized ``cxcywh`` to absolute ``xyxy`` pixels."""
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    coords = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx = coords[:, 0] * width
    cy = coords[:, 1] * height
    w = coords[:, 2] * width
    h = coords[:, 3] * height
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
