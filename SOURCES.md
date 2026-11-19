# Sources and provenance

1. Pascal VOC / COCO detection evaluation conventions: axis-aligned box IoU,
   score-ordered greedy matching at a fixed IoU threshold, per-class precision/recall
   and average precision. The matching and AP code here is an independent standard-library
   implementation; no evaluation toolkit source, model weight, or benchmark result is copied.
2. All-point interpolation for AP follows the envelope convention: precision at each
   recall level is the maximum precision achieved at any higher recall, and AP is the
   area under the resulting envelope.
3. Python standard library documentation: [csv](https://docs.python.org/3/library/csv.html),
   [math](https://docs.python.org/3/library/math.html), [hashlib](https://docs.python.org/3/library/hashlib.html)
   and [json](https://docs.python.org/3/library/json.html).

The demo generator in `det_eval_lab/synthetic.py` uses fixed hand-written boxes under
label `synthetic-demo`. Class names `bolt` and `bracket` are illustrative. The generated
boxes and scores do not come from any trained detector, image dataset, or public leaderboard.

Files in `results/` can be rebuilt with `python -m det_eval_lab demo`. The input SHA-256 for
external captures is recorded in `report.json`. No original researcher affiliation, authorship,
or award is attributed to this account.
