#!/usr/bin/env python3
"""Create a small human-review pilot queue from deterministic hard candidates.

The script does not infer labels. Pilot label fields remain empty for the
annotator to fill after reviewing the underlying documents.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from task3_common import ROOT, read_csv_rows, stable_hash


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--near-pairs", default="outputs/task3/near_duplicate_pairs.csv")
    parser.add_argument("--same-event-pairs", default="outputs/task3/same_event_pairs.csv")
    parser.add_argument("--near-output", default="outputs/task3/pilot_near_duplicate_pairs.csv")
    parser.add_argument("--same-output", default="outputs/task3/pilot_same_event_pairs.csv")
    parser.add_argument("--limit-near", type=int, default=30)
    parser.add_argument("--limit-same", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    near_rows = read_csv_rows(args.near_pairs)
    same_rows = read_csv_rows(args.same_event_pairs)
    near_priority = sorted(near_rows, key=lambda row: (not text(row.get("hard_negative")), text(row.get("candidate_reason")), stable_hash(args.seed, row.get("pair_id"))))
    same_priority = sorted(same_rows, key=lambda row: (not bool(row.get("same_document")), text(row.get("candidate_reason")), stable_hash(args.seed, row.get("pair_id"))))

    near_selected = near_priority[:args.limit_near]
    same_selected = same_priority[:args.limit_same]
    for row in near_selected:
        row["pilot_batch"] = "task3_pilot_2026_08"
        row["pilot_selection_reason"] = "hard_negative_or_deterministic_candidate"
        row["human_is_near_duplicate"] = ""
        row["annotator_note"] = ""
    for row in same_selected:
        row["pilot_batch"] = "task3_pilot_2026_08"
        row["pilot_selection_reason"] = "same_document_or_deterministic_candidate"
        row["human_same_event"] = ""
        row["human_relation"] = ""
        row["annotator_note"] = ""

    def write(path_value: str, rows: list[dict[str, Any]]) -> None:
        path = Path(path_value)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            if not rows:
                return
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write(args.near_output, near_selected)
    write(args.same_output, same_selected)
    summary = {
        "pilot_batch": "task3_pilot_2026_08",
        "near_duplicate_pilot_count": len(near_selected),
        "same_event_pilot_count": len(same_selected),
        "human_labels_generated": False,
        "selection": "deterministic hard-negative/same-document priority; labels intentionally blank",
    }
    report = ROOT / "reports/task3_manual_pilot_summary.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
