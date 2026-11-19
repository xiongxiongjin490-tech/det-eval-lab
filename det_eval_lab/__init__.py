"""Independent detection evaluation utilities. No detector model is bundled."""

__version__ = "0.1.0"

from .core import (
    Bbox,
    evaluate,
    iou,
    input_sha256,
    load_detections,
    load_ground_truth,
)

__all__ = [
    "Bbox",
    "evaluate",
    "iou",
    "input_sha256",
    "load_detections",
    "load_ground_truth",
]
