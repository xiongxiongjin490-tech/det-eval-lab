import csv
import json

import pytest

from det_eval_lab import Bbox, evaluate, iou, load_detections, load_ground_truth
from det_eval_lab.__main__ import main


def box(image_id="img1", cls="a", x1=0, y1=0, x2=2, y2=2, score=None):
    return Bbox(image_id, cls, float(x1), float(y1), float(x2), float(y2), score)


def test_iou_full_overlap():
    assert iou(box(), box()) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou(box(x1=0, y1=0, x2=1, y2=1), box(x1=2, y1=2, x2=3, y2=3)) == 0.0


def test_iou_partial():
    # intersection 1x1=1, union 4+4-1=7
    assert iou(box(x1=0, y1=0, x2=2, y2=2), box(x1=1, y1=1, x2=3, y2=3)) == pytest.approx(1 / 7)


@pytest.mark.parametrize("x1,y1,x2,y2", [(0, 0, 0, 2), (0, 0, 2, 0), (0, 0, -1, 2)])
def test_invalid_box_rejected(x1, y1, x2, y2):
    with pytest.raises(ValueError):
        Bbox("img", "a", float(x1), float(y1), float(x2), float(y2), 0.9)


def test_score_out_of_range_rejected():
    with pytest.raises(ValueError, match="score"):
        Bbox("img", "a", 0, 0, 2, 2, 1.5)


def test_perfect_detection_ap_one():
    gt = [box(cls="a")]
    det = [box(cls="a", score=0.9)]
    result = evaluate(gt, det)
    row = result["classes"][0]
    assert row["tp"] == 1 and row["fp"] == 0 and row["fn"] == 0
    assert row["ap"] == pytest.approx(1.0)
    assert result["mAP"] == pytest.approx(1.0)


def test_missed_ground_truth_ap_zero():
    gt = [box(cls="a"), box(image_id="img2", cls="b")]
    det = [box(cls="a", score=0.9)]
    result = evaluate(gt, det)
    by_cls = {row["class"]: row for row in result["classes"]}
    assert by_cls["a"]["ap"] == pytest.approx(1.0)
    assert by_cls["b"]["ap"] == pytest.approx(0.0)
    assert by_cls["b"]["fn"] == 1


def test_low_iou_is_fp_and_fn():
    gt = [box(x1=0, y1=0, x2=2, y2=2)]
    det = [box(x1=3, y1=3, x2=5, y2=5, score=0.9)]
    result = evaluate(gt, det, iou_threshold=0.5)
    row = result["classes"][0]
    assert row["fp"] == 1 and row["fn"] == 1 and row["tp"] == 0


def test_demo_known_counts():
    from det_eval_lab.synthetic import demo_detections, demo_ground_truth

    result = evaluate(demo_ground_truth(), demo_detections())
    by_cls = {row["class"]: row for row in result["classes"]}
    assert by_cls["bolt"]["tp"] == 3 and by_cls["bolt"]["fp"] == 0
    assert by_cls["bracket"]["tp"] == 1 and by_cls["bracket"]["fp"] == 1
    assert by_cls["bracket"]["fn"] == 1
    assert 0.0 <= result["mAP"] <= 1.0


def _write_csv(path, rows, header):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_csv_load_and_missing_column(tmp_path):
    gt_path = tmp_path / "gt.csv"
    _write_csv(gt_path, [{"image_id": "i1", "class": "a", "x1": 0, "y1": 0, "x2": 2, "y2": 2}],
               ["image_id", "class", "x1", "y1", "x2", "y2"])
    assert len(load_ground_truth(gt_path)) == 1
    broken = tmp_path / "broken.csv"
    broken.write_text("image_id,class\ni1,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing CSV columns"):
        load_ground_truth(broken)


def test_csv_bad_score(tmp_path):
    det_path = tmp_path / "det.csv"
    _write_csv(det_path, [{"image_id": "i1", "class": "a", "x1": 0, "y1": 0, "x2": 2, "y2": 2, "score": "1.5"}],
               ["image_id", "class", "x1", "y1", "x2", "y2", "score"])
    with pytest.raises(ValueError):
        load_detections(det_path)


def test_cli_demo_end_to_end(tmp_path):
    output = tmp_path / "results"
    docs = tmp_path / "data.json"
    assert main(["demo", "--output", str(output), "--docs", str(docs)]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["data_kind"] == "synthetic-demo"
    assert report["report"]["totals"]["tp"] == 4
    assert (output / "per_class.csv").exists()
    assert (output / "pr_points.csv").exists()
    assert (output / "report.md").exists()
    assert json.loads(docs.read_text(encoding="utf-8"))["data_kind"] == "synthetic-demo"


def test_cli_eval_missing_file_returns_nonzero(tmp_path):
    assert main(["eval", "--gt", str(tmp_path / "gt.csv"), "--det", str(tmp_path / "det.csv"),
                 "--output", str(tmp_path / "out")]) == 2
