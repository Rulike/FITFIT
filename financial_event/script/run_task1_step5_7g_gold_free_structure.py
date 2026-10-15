#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gold-free inference wrapper for Task1 Step 5.7-G.

The wrapper intentionally projects each input row to the approved field
whitelist before calling the frozen Step 5.7-F adapter implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_7f_structure_training import predict_adapter  # noqa: E402


OUT_DIR = ROOT / "outputs/task1_step5_7g_gold_free_audit"
DEFAULT_INPUT = OUT_DIR / "gold_free_dev_input.jsonl"
DEFAULT_OUTPUT = OUT_DIR / "gold_free_dev_predictions.jsonl"
DEFAULT_MANIFEST = OUT_DIR / "gold_free_inference_manifest.json"

ALLOWED_FIELDS = ["event_mention_id", "article_id", "title", "content", "publish_time", "event_name"]
BANNED_PREFIXES = ("ai_", "human_", "reviewer_")
BANNED_EXACT = {
    "ai_fields",
    "reference_raw_text",
    "original_quality_grade",
    "selected_quality_grade",
    "needs_manual_correction",
    "validation_errors",
    "annotation_source",
    "annotation_status",
    "confirmation_note",
    "confirmed_at",
    "source_workbook",
    "trigger",
    "event_span",
    "subject_company",
    "event_time",
    "object_target",
    "direction",
    "arguments",
    "legacy_raw_structure",
    "multi_event_article",
    "source_set",
}


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def banned_fields(row: dict[str, Any]) -> list[str]:
    hits = []
    for key in row:
        if key in BANNED_EXACT or key.startswith(BANNED_PREFIXES):
            hits.append(key)
    return sorted(hits)


def project_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: str(row.get(field, "") or "") for field in ALLOWED_FIELDS}


def empty_output_count(predictions: list[dict[str, Any]]) -> int:
    count = 0
    for pred in predictions:
        spans_empty = all(not (pred.get(field) or {}).get("text") for field in ["trigger", "event_span", "subject_company", "event_time", "object_target"])
        if spans_empty and not pred.get("arguments"):
            count += 1
    return count


def run(input_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    raw_rows = read_jsonl(input_path)
    field_counter = Counter()
    banned_counter = Counter()
    for row in raw_rows:
        field_counter.update(row.keys())
        banned_counter.update(banned_fields(row))
    rows = [project_row(row) for row in raw_rows]
    predictions = predict_adapter(rows)
    for pred in predictions:
        pred["scheme"] = "rule_ai_adapter_baseline_gold_free_projection"
        pred["adapter_status"] = "label_dependent_adapter"
        pred["input_projection"] = "allowed_fields_only"
        pred["gold_read_by_inference"] = False
    write_jsonl(output_path, predictions)
    manifest = {
        "generated_at": now(),
        "status": "completed",
        "adapter_status": "label_dependent_adapter",
        "frozen_adapter": "Step 5.7-F rule_ai_adapter_baseline via predict_adapter/span_from_ai",
        "input": rel(input_path),
        "input_sha256": sha256_file(input_path),
        "output": rel(output_path),
        "output_sha256": sha256_file(output_path),
        "row_count": len(raw_rows),
        "allowed_fields": ALLOWED_FIELDS,
        "input_field_counts": dict(sorted(field_counter.items())),
        "ignored_banned_field_counts": dict(sorted(banned_counter.items())),
        "ignored_banned_field_hit_count": int(sum(banned_counter.values())),
        "gold_or_confirmed_workbook_read": False,
        "gold_test_labels_read": False,
        "local_32b_calls": 0,
        "external_llm_calls": 0,
        "empty_output_count": empty_output_count(predictions),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    result = run(input_path, output_path, manifest_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
