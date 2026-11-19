# Det Eval Lab · Object detection evaluation

[中文](README.md) · [Sources](SOURCES.md)

Turn annotation and detection CSVs into an auditable evaluation report: invalid boxes are
rejected first, then score-ordered greedy IoU matching runs per image and class, producing
per-class Precision / Recall, PR points, AP@0.5 and mAP@0.5. The project is implemented
independently and bundles no detector, weights, training pipeline, or dataset.

**The bundled demo uses programmatically generated synthetic boxes. They are not real model
outputs and do not represent any public benchmark score.** Per-class results and PR points
are in [results/](results/).

## Features

- Rejects missing/unknown columns, non-finite coordinates, invalid boxes
  (`x2 <= x1`, `y2 <= y1`) and out-of-range detection scores.
- Groups by `(image_id, class)`; detections are matched score-descending and greedily.
  Matches above the IoU threshold are TP, remaining detections FP, unmatched ground truth FN.
- Per-class ground-truth count, detection count, TP / FP / FN, Precision, Recall and AP;
  overall macro P / R and mAP.
- Exports a JSON report, per-class CSV, PR-point CSV and Markdown report, with input
  SHA-256 and runtime environment recorded.

## Quick start

Python 3.10+. Runtime depends on the standard library only; pytest is needed for tests.

```bash
python -m pip install -e ".[dev]"
python -m det_eval_lab demo
python -m pytest
```

Rebuild `results/` and `docs/data.json` by default. For an isolated CI run:

```bash
python -m det_eval_lab demo --quick --output results-quick --docs results-quick/data.json
```

## Evaluating your own results

```bash
python -m det_eval_lab eval --gt annotations.csv --det detections.csv --output my-report
```

Ground-truth columns: `image_id,class,x1,y1,x2,y2`; detections add `score` in [0, 1].
Coordinates use the `x1,y1,x2,y2` pixel convention; the tool is unaware of image resolution
or class semantics. The default IoU threshold is 0.5, adjustable with `--iou`. External input
defaults to `external-unverified`; the source label is a user declaration, not an
authentication. The CLI exits with code 2 on invalid input.

## Interpreting results

`report.md` is the entry point; `per_class.csv` keeps per-class denominators and metrics,
`pr_points.csv` keeps per-detection Recall / Precision. AP uses all-point interpolation under
the precision envelope. This tool performs offline result statistics only; it does not judge
annotation quality, class reasonableness, or model generalization. Results from different
datasets or resolution conventions must not be compared directly.

## Boundaries

This is an independent IoU / PR / AP statistics utility inspired by generic offline detection
evaluation. It does not reproduce any detector training pipeline and is not tied to a specific
network architecture. See [SOURCES.md](SOURCES.md). Code is MIT licensed; cited materials
retain their original rights.

## Reproducible environment

Python 3.12. The lock file pins development, build and cross-platform runtime dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
python -m det_eval_lab demo
```
