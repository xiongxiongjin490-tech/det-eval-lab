"""Command-line interface: ``python -m det_eval_lab ...``."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

from . import __version__
from .core import Bbox, evaluate, input_sha256, load_detections, load_ground_truth
from .synthetic import demo_detections, demo_ground_truth


def _write_per_class_csv(path: Path, classes: list[dict]) -> None:
    fields = ["class", "ground_truth", "detections", "tp", "fp", "fn",
              "precision", "recall", "ap"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in classes:
            writer.writerow({key: row.get(key) for key in fields})


def _write_pr_csv(path: Path, pr_curves: dict[str, list[dict]]) -> None:
    fields = ["class", "score", "recall", "precision"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cls, points in sorted(pr_curves.items()):
            for point in points:
                writer.writerow({"class": cls, "score": point["score"],
                                 "recall": point["recall"], "precision": point["precision"]})


def _write_boxes_csv(path: Path, boxes: list[Bbox], with_score: bool) -> None:
    fields = ["image_id", "class", "x1", "y1", "x2", "y2"] + (["score"] if with_score else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for box in boxes:
            row = {"image_id": box.image_id, "class": box.cls, "x1": box.x1,
                   "y1": box.y1, "x2": box.x2, "y2": box.y2}
            if with_score:
                row["score"] = box.score
            writer.writerow(row)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _render_markdown(report: dict, data_kind: str) -> str:
    totals = report["totals"]
    lines = [
        "# 目标检测评估报告",
        "",
        f"- 数据类型：`{data_kind}`（来源标签由使用者声明，工具不做真实性认证）",
        f"- IoU 阈值：{report['iou_threshold']}",
        f"- mAP@{report['iou_threshold']}：{_fmt(report['mAP'])}",
        f"- 宏平均 Precision：{_fmt(report['macro_precision'])}，宏平均 Recall：{_fmt(report['macro_recall'])}",
        f"- TP / FP / FN：{totals['tp']} / {totals['fp']} / {totals['fn']}",
        "",
        "## 逐类结果",
        "",
        "| 类别 | 标注 | 检测 | TP | FP | FN | Precision | Recall | AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["classes"]:
        lines.append(
            f"| {row['class']} | {row['ground_truth']} | {row['detections']} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
            f"{_fmt(row['ap'])} |"
        )
    lines += ["", "逐检测分数对应的 PR 点见 `pr_points.csv`。", ""]
    return "\n".join(lines)


def _persist(output_dir: Path, report: dict, data_kind: str, source: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "det-eval-lab",
        "version": __version__,
        "data_kind": data_kind,
        "source": source,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "report": report,
    }
    (output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_per_class_csv(output_dir / "per_class.csv", report["classes"])
    _write_pr_csv(output_dir / "pr_points.csv", report["pr_curves"])
    (output_dir / "report.md").write_text(_render_markdown(report, data_kind), encoding="utf-8")


def _cmd_demo(args: argparse.Namespace) -> int:
    gt = demo_ground_truth()
    det = demo_detections()
    report = evaluate(gt, det, args.iou)
    output = Path(args.output)
    _persist(output, report, "synthetic-demo", "fixed artificial capture")
    if not args.quick:
        _write_boxes_csv(output / "ground_truth.csv", gt, with_score=False)
        _write_boxes_csv(output / "detections.csv", det, with_score=True)
    if args.docs:
        docs = Path(args.docs)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(json.dumps({"data_kind": "synthetic-demo", "report": report},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    gt = load_ground_truth(args.gt)
    det = load_detections(args.det)
    report = evaluate(gt, det, args.iou)
    source = f"gt={input_sha256(args.gt)} det={input_sha256(args.det)}"
    _persist(Path(args.output), report, args.data_kind, source)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="det-eval", description="目标检测 IoU / PR / mAP 评估")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="生成固定合成标注与检测结果并评估")
    demo.add_argument("--output", default="results")
    demo.add_argument("--docs", help="额外写出供页面读取的 data.json")
    demo.add_argument("--iou", type=float, default=0.5)
    demo.add_argument("--quick", action="store_true", help="只写报告，不导出原始 CSV")
    demo.set_defaults(handler=_cmd_demo)

    eval_cmd = sub.add_parser("eval", help="评估一份外部标注与检测结果")
    eval_cmd.add_argument("--gt", required=True, help="标注 CSV：image_id,class,x1,y1,x2,y2")
    eval_cmd.add_argument("--det", required=True, help="检测 CSV：image_id,class,x1,y1,x2,y2,score")
    eval_cmd.add_argument("--output", default="report")
    eval_cmd.add_argument("--iou", type=float, default=0.5)
    eval_cmd.add_argument("--data-kind", default="external-unverified")
    eval_cmd.set_defaults(handler=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
