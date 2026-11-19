"""Synthetic ground-truth and detection captures for the demo.

Everything here is artificial and deterministic. It is not a trained detector's
output, not a real image annotation, and must not be presented as a model result
on a public benchmark.
"""

from __future__ import annotations

from .core import Bbox


def demo_ground_truth() -> list[Bbox]:
    return [
        Bbox("img1", "bolt", 10, 10, 50, 50),
        Bbox("img1", "bracket", 60, 10, 100, 50),
        Bbox("img2", "bolt", 10, 10, 50, 50),
        Bbox("img3", "bracket", 10, 20, 60, 70),
        Bbox("img4", "bolt", 20, 20, 70, 80),
    ]


def demo_detections() -> list[Bbox]:
    return [
        Bbox("img1", "bolt", 11, 11, 49, 49, score=0.92),   # TP
        Bbox("img1", "bracket", 61, 11, 99, 49, score=0.88),  # TP
        Bbox("img1", "bracket", 5, 60, 40, 90, score=0.74),  # FP: wrong location
        Bbox("img2", "bolt", 12, 12, 48, 48, score=0.81),     # TP
        # img3 bracket is missed -> FN
        Bbox("img4", "bolt", 22, 22, 68, 78, score=0.66),    # TP
    ]
