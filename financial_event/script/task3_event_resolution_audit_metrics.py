#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "task34_root_cause_repair" / "task3_event_resolution_audit_200.xlsx"
DEFAULT_JSON = ROOT / "reports" / "task3_event_resolution_audit_metrics.json"
DEFAULT_MD = ROOT / "docs" / "task3_event_resolution_audit_metrics.md"


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def norm_label(x: Any) -> str:
    return safe_str(x).lower()


def read_audit(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
            return pd.DataFrame(data["rows"])
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    df = frame.copy()
    if "machine_decision" not in df.columns:
        raise ValueError("Missing required column: machine_decision")
    if "human_merge_label" not in df.columns:
        raise ValueError("Missing required column: human_merge_label")
    if "human_error_type" not in df.columns:
        raise ValueError("Missing required column: human_error_type")

    df["machine_decision"] = df["machine_decision"].map(norm_label)
    df["human_merge_label"] = df["human_merge_label"].map(norm_label)
    df["human_error_type"] = df["human_error_type"].map(norm_label)

    labeled = df["human_merge_label"].isin({"merge", "no_merge"})
    same_pred = df["machine_decision"].eq("same_event")
    diff_pred = df["machine_decision"].eq("different_event")

    same_labeled = same_pred & labeled
    diff_labeled = diff_pred & labeled

    merge_tp = same_labeled & df["human_merge_label"].eq("merge")
    false_merge = same_labeled & (df["human_merge_label"].eq("no_merge") | df["human_error_type"].eq("false_merge"))
    split_error = diff_labeled & (
        df["human_merge_label"].eq("merge") | df["human_error_type"].isin({"split_error", "duplicate"})
    )
    duplicate = diff_labeled & df["human_error_type"].eq("duplicate")

    actual_merge = labeled & df["human_merge_label"].eq("merge")
    actual_no_merge = labeled & df["human_merge_label"].eq("no_merge")

    same_pred_count = int(same_pred.sum())
    diff_pred_count = int(diff_pred.sum())
    labeled_count = int(labeled.sum())

    merge_precision = float(merge_tp.sum() / same_labeled.sum()) if int(same_labeled.sum()) else None
    false_merge_rate = float(false_merge.sum() / same_labeled.sum()) if int(same_labeled.sum()) else None
    split_error_rate = float(split_error.sum() / diff_labeled.sum()) if int(diff_labeled.sum()) else None
    merge_recall = float(merge_tp.sum() / actual_merge.sum()) if int(actual_merge.sum()) else None
    no_merge_recall = float((diff_labeled & actual_no_merge).sum() / actual_no_merge.sum()) if int(actual_no_merge.sum()) else None

    inconsistent_rows = int(
        ((same_pred & df["human_error_type"].isin({"split_error", "duplicate"})) | (diff_pred & df["human_error_type"].eq("false_merge"))).sum()
    )

    by_bucket: dict[str, Any] = {}
    if "source_bucket" in df.columns:
        for bucket, sub in df.groupby("source_bucket", dropna=False):
            sub_labeled = sub["human_merge_label"].isin({"merge", "no_merge"})
            sub_same = sub["machine_decision"].eq("same_event") & sub_labeled
            sub_diff = sub["machine_decision"].eq("different_event") & sub_labeled
            sub_merge_tp = sub_same & sub["human_merge_label"].eq("merge")
            sub_false_merge = sub_same & (sub["human_merge_label"].eq("no_merge") | sub["human_error_type"].eq("false_merge"))
            sub_split_error = sub_diff & (sub["human_merge_label"].eq("merge") | sub["human_error_type"].isin({"split_error", "duplicate"}))
            by_bucket[safe_str(bucket)] = {
                "rows": int(len(sub)),
                "labeled_rows": int(sub_labeled.sum()),
                "same_event_rows": int(sub_same.sum()),
                "different_event_rows": int(sub_diff.sum()),
                "merge_precision": float(sub_merge_tp.sum() / sub_same.sum()) if int(sub_same.sum()) else None,
                "false_merge_rate": float(sub_false_merge.sum() / sub_same.sum()) if int(sub_same.sum()) else None,
                "split_error_rate": float(sub_split_error.sum() / sub_diff.sum()) if int(sub_diff.sum()) else None,
            }

    return {
        "status": "task3_event_resolution_audit_metrics_ready",
        "input_file": str(DEFAULT_INPUT if not str(frame.attrs.get("source_path", "")).strip() else frame.attrs["source_path"]),
        "total_rows": int(len(df)),
        "labeled_rows": labeled_count,
        "label_coverage": float(labeled_count / len(df)) if len(df) else None,
        "same_event_pred_rows": same_pred_count,
        "different_event_pred_rows": diff_pred_count,
        "same_event_labeled_rows": int(same_labeled.sum()),
        "different_event_labeled_rows": int(diff_labeled.sum()),
        "merge_precision": merge_precision,
        "false_merge_rate": false_merge_rate,
        "split_error_rate": split_error_rate,
        "merge_recall": merge_recall,
        "no_merge_recall": no_merge_recall,
        "merge_tp_rows": int(merge_tp.sum()),
        "false_merge_rows": int(false_merge.sum()),
        "split_error_rows": int(split_error.sum()),
        "duplicate_rows": int(duplicate.sum()),
        "actual_merge_rows": int(actual_merge.sum()),
        "actual_no_merge_rows": int(actual_no_merge.sum()),
        "inconsistent_rows": inconsistent_rows,
        "by_bucket": by_bucket,
    }


def write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Task3 Event Resolution Audit Metrics",
        "",
        f"- input_file: `{report['input_file']}`",
        f"- total_rows: {report['total_rows']}",
        f"- labeled_rows: {report['labeled_rows']}",
        f"- label_coverage: {report['label_coverage']}",
        f"- merge_precision: {report['merge_precision']}",
        f"- false_merge_rate: {report['false_merge_rate']}",
        f"- split_error_rate: {report['split_error_rate']}",
        f"- merge_recall: {report['merge_recall']}",
        f"- no_merge_recall: {report['no_merge_recall']}",
        f"- inconsistent_rows: {report['inconsistent_rows']}",
        "",
        "Bucket breakdown:",
    ]
    if report["by_bucket"]:
        for bucket, stats in report["by_bucket"].items():
            lines.append(
                f"- {bucket}: rows={stats['rows']}, labeled={stats['labeled_rows']}, merge_precision={stats['merge_precision']}, false_merge_rate={stats['false_merge_rate']}, split_error_rate={stats['split_error_rate']}"
            )
    else:
        lines.append("- not available")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Task3 event-resolution audit metrics from human labels.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the labeled audit pack.")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name when the input is xlsx/xls.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON, help="Path to write the metrics JSON.")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD, help="Path to write the markdown summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = read_audit(args.input, sheet_name=args.sheet)
    frame.attrs["source_path"] = str(args.input)
    report = compute_metrics(frame)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(args.output_md, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
