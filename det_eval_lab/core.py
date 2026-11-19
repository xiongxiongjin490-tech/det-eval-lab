"""Independent detection evaluation: IoU matching, PR curves and AP@threshold.

Implemented on the Python standard library. No detector, model weights, training
pipeline or dataset is bundled. Coordinates are plain pixel coordinates in the
``x1,y1,x2,y2`` convention; the tool does not know image resolution or class
semantics.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

GT_COLUMNS = {"image_id", "class", "x1", "y1", "x2", "y2"}
DET_COLUMNS = GT_COLUMNS | {"score"}


@dataclass(frozen=True)
class Bbox:
    image_id: str
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = None

    def __post_init__(self) -> None:
        _validate_box(self.image_id, self.cls, self.x1, self.y1,
                      self.x2, self.y2, self.score)


def _validate_box(image_id: str, cls: str, x1: float, y1: float,
                  x2: float, y2: float, score: float | None) -> None:
    if not image_id.strip() or not cls.strip():
        raise ValueError("image_id and class must be nonempty")
    for value in (x1, y1, x2, y2):
        if not math.isfinite(value):
            raise ValueError("box coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Require x2 > x1 and y2 > y1: {(x1, y1, x2, y2)}")
    if score is not None and (not math.isfinite(score) or not 0.0 <= score <= 1.0):
        raise ValueError("score must be a finite number in [0, 1]")


def iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-union for two axis-aligned boxes."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - intersection
    if union <= 0:
        raise ValueError("Box union area must be positive")
    return intersection / union


def _read_csv(path: Path, required: set[str], with_score: bool) -> list[Bbox]:
    boxes: list[Bbox] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            try:
                if None in row:
                    raise ValueError("Extra CSV fields")
                image_id = (row["image_id"] or "").strip()
                cls = (row["class"] or "").strip()
                x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                score = None
                if with_score:
                    raw_score = (row.get("score") or "").strip()
                    score = float(raw_score) if raw_score else None
                    if score is None:
                        raise ValueError("det score must be present")
                _validate_box(image_id, cls, x1, y1, x2, y2, score)
                boxes.append(Bbox(image_id, cls, x1, y1, x2, y2, score))
            except (ValueError, TypeError) as error:
                raise ValueError(f"CSV line {line}: {error}") from error
    if not boxes:
        raise ValueError("No boxes found")
    return boxes


def load_ground_truth(path: str | Path) -> list[Bbox]:
    return _read_csv(Path(path), GT_COLUMNS, with_score=False)


def load_detections(path: str | Path) -> list[Bbox]:
    return _read_csv(Path(path), DET_COLUMNS, with_score=True)


def _match_group(gt: list[Bbox], det: list[Bbox], threshold: float) -> tuple[list[bool], int, int]:
    """Greedy score-ordered matching for one (image, class) group.

    Returns per-detection TP flags, TP count and FP count. Unmatched ground truth
    are counted as FN by the caller.
    """
    ordered = sorted(det, key=lambda box: -(box.score or 0.0))
    matched: set[int] = set()
    tp_flags: list[bool] = []
    for box in ordered:
        best_index = -1
        best_iou = threshold
        for index, truth in enumerate(gt):
            if index in matched:
                continue
            overlap = iou(box, truth)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0:
            matched.add(best_index)
            tp_flags.append(True)
        else:
            tp_flags.append(False)
    return tp_flags, sum(tp_flags), len(tp_flags) - sum(tp_flags)


def _average_precision(scores: list[float], tp_flags: list[bool], n_gt: int) -> tuple[float, list[dict]]:
    """All-point interpolation AP from a score-ordered detection list."""
    if n_gt == 0:
        return 0.0, []
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    points: list[dict] = []
    for index in order:
        if tp_flags[index]:
            tp += 1
        else:
            fp += 1
        recall = tp / n_gt
        precision = tp / (tp + fp)
        points.append({"score": scores[index], "recall": recall, "precision": precision})
    if not points:
        return 0.0, points

    recalls = [point["recall"] for point in points]
    precisions = [point["precision"] for point in points]
    envelope: list[float] = [0.0] * len(points)
    envelope[-1] = precisions[-1]
    for index in range(len(points) - 2, -1, -1):
        envelope[index] = max(precisions[index], envelope[index + 1])

    ap = 0.0
    previous_recall = 0.0
    for recall, precision in zip(recalls, envelope):
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap, points


def evaluate(ground_truth: list[Bbox], detections: list[Bbox],
             iou_threshold: float = 0.5) -> dict:
    if not math.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    gt_groups: dict[tuple[str, str], list[Bbox]] = defaultdict(list)
    det_groups: dict[tuple[str, str], list[Bbox]] = defaultdict(list)
    for box in ground_truth:
        gt_groups[(box.image_id, box.cls)].append(box)
    for box in detections:
        det_groups[(box.image_id, box.cls)].append(box)

    classes = sorted({box.cls for box in ground_truth} | {box.cls for box in detections})
    per_class: list[dict] = []
    pr_curves: dict[str, list[dict]] = {}

    for cls in classes:
        class_gt = [box for box in ground_truth if box.cls == cls]
        class_det = [box for box in detections if box.cls == cls]
        n_gt = len(class_gt)
        tp = fp = fn = 0
        scores: list[float] = []
        tp_flags: list[bool] = []

        group_keys = set(gt_groups) | set(det_groups)
        for key in group_keys:
            if key[1] != cls:
                continue
            group_gt = gt_groups.get(key, [])
            group_det = det_groups.get(key, [])
            flags, group_tp, group_fp = _match_group(group_gt, group_det, iou_threshold)
            tp += group_tp
            fp += group_fp
            fn += len(group_gt) - group_tp
            for box, flag in zip(sorted(group_det, key=lambda b: -(b.score or 0.0)), flags):
                scores.append(box.score or 0.0)
                tp_flags.append(flag)

        ap, points = _average_precision(scores, tp_flags, n_gt)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_gt if n_gt else 0.0
        per_class.append({
            "class": cls, "ground_truth": n_gt, "detections": len(class_det),
            "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "ap": ap,
        })
        pr_curves[cls] = points

    map_at_threshold = (sum(row["ap"] for row in per_class) / len(per_class)) if per_class else 0.0
    total_tp = sum(row["tp"] for row in per_class)
    total_fp = sum(row["fp"] for row in per_class)
    total_fn = sum(row["fn"] for row in per_class)
    return {
        "iou_threshold": iou_threshold,
        "classes": per_class,
        "pr_curves": pr_curves,
        "mAP": map_at_threshold,
        "macro_precision": sum(row["precision"] for row in per_class) / len(per_class) if per_class else 0.0,
        "macro_recall": sum(row["recall"] for row in per_class) / len(per_class) if per_class else 0.0,
        "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }


def input_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
