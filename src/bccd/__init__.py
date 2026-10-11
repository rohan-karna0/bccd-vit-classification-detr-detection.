"""Helpers for the BCCD classification and detection notebook.

The Vision Transformer and the DETR training loop are defined in the notebook
so they can be read during an interview. This package covers file discovery,
Pascal VOC parsing, splits, crops, and metric files.
"""

from bccd.boxes import (
    hflip_xyxy,
    scale_xyxy,
    sanitize_xyxy,
    xyxy_to_cxcywh_norm,
    cxcywh_norm_to_xyxy,
)
from bccd.crops import export_classification_crops
from bccd.metrics_io import load_metrics, metrics_row_or_missing, save_metrics
from bccd.splits import assign_splits
from bccd.voc import build_records, discover_files, format_tree

__all__ = [
    "assign_splits",
    "build_records",
    "cxcywh_norm_to_xyxy",
    "discover_files",
    "export_classification_crops",
    "format_tree",
    "hflip_xyxy",
    "load_metrics",
    "metrics_row_or_missing",
    "sanitize_xyxy",
    "save_metrics",
    "scale_xyxy",
    "xyxy_to_cxcywh_norm",
]
