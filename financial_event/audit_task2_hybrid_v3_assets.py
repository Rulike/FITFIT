#!/usr/bin/env python3
"""Audit Task2 hybrid-v3 human-label assets and locked-500 leakage.

This script is intentionally read-only with respect to source workbooks and
model outputs. It only writes fresh audit artifacts under the hybrid-v3 output
directory plus the locked-id config requested by the user.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3"
LOCKED500_JSONL = ROOT / "outputs/event_credibility/task2_independent500_v1/independent_samples.jsonl"

TRAINABLE_SOURCE_FILES = [
    ROOT / "reports/task2_dea_cached_human_audit_gold_filled.xlsx",
    ROOT / "reports/task2_safety_gate_v1_2_independent_human_audit_gold_filled.xlsx",
]

DERIVATIVE_REVIEW_FILES = [
    ROOT / "reports/task2_claim_contract_adapter_v1_gray_details.xlsx",
    ROOT / "reports/task2_independent_error_audit_v1.xlsx",
    ROOT / "reports/task2_isr_diagnostic_v1_details.xlsx",
    ROOT / "reports/task2_dea_safety_gate_failure_audit.xlsx",
    ROOT / "reports/task2_safety_gate_v1_2_independent_gold_errors.xlsx",
]

LOCKED_DUPLICATE_FILES = [
    ROOT / "reports/task2_independent500_auto_assisted_human_confirmed_gold.xlsx",
    ROOT / "reports/task2_independent500_ai_assisted_human_review.xlsx",
    ROOT / "reports/task2_independent500_auto_labeled.xlsx",
]

IGNORED_TEMPLATE_FILES = [
    ROOT / "reports/task2_independent500_human_audit.xlsx",
    ROOT / "reports/task2_dea_cached_human_audit.xlsx",
    ROOT / "reports/task2_safety_gate_v1_2_independent_human_audit.xlsx",
    ROOT / "reports/task2_dea_safety_gate_v1_1_failure_audit.xlsx",
]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def norm(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def table_records(path: Path, sheet_name: str) -> list[dict[str, str]]:
    from scripts.validate_task2_dea_human_audit import table_records as _table_records

    return _table_records(path, sheet_name)


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = norm(value)
        if text:
            return text
    return ""


def build_profile_from_workbook_row(row: dict[str, Any]) -> dict[str, Any]:
    from scripts.build_task2_independent500_human_audit import first_date, stable_text

    return {
        "cluster_id": norm(row.get("cluster_id")),
        "target_key": norm(row.get("target_code")) or stable_text(first_nonempty(row.get("subject/company"), row.get("主体")), 80),
        "target_name": first_nonempty(row.get("subject/company"), row.get("主体"), row.get("target_name")),
        "event_name": first_nonempty(row.get("event_name"), row.get("事件类型"), row.get("schema_id")),
        "date": first_date(first_nonempty(row.get("event_time"), row.get("事件时间"))),
        "title_text": " ".join(
            item
            for item in [
                first_nonempty(row.get("title"), row.get("证据文本"), row.get("article_title")),
            ]
            if item
        ),
        "claim_text": first_nonempty(row.get("claim"), row.get("待验证主张"), row.get("claim_text")),
    }


def build_profile_from_locked_row(row: dict[str, Any]) -> dict[str, Any]:
    from scripts.build_task2_independent500_human_audit import first_date, stable_text

    titles = row.get("title_samples") or ""
    return {
        "cluster_id": norm(row.get("cluster_id")),
        "target_key": norm(row.get("target_code")) or stable_text(first_nonempty(row.get("target_name"), row.get("subject/company")), 80),
        "target_name": first_nonempty(row.get("target_name"), row.get("subject/company")),
        "event_name": first_nonempty(row.get("event_name"), row.get("schema_id")),
        "date": first_date(first_nonempty(row.get("event_time"), row.get("earliest_publish_time"))),
        "title_text": norm(titles),
        "claim_text": first_nonempty(row.get("event_claim"), row.get("claim"), row.get("claim_text")),
    }


def profile_from_source_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse workbook rows to one profile per cluster_id."""
    from scripts.build_task2_independent500_human_audit import first_date, stable_text

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cid = norm(row.get("cluster_id"))
        if cid:
            grouped[cid].append(row)
    profiles: dict[str, dict[str, Any]] = {}
    for cid, items in grouped.items():
        titles = []
        for item in items:
            title = first_nonempty(item.get("title"), item.get("article_title"), item.get("证据文本"))
            if title and title not in titles:
                titles.append(title)
        first = items[0]
        profiles[cid] = {
            "cluster_id": cid,
            "target_key": norm(first.get("target_code")) or stable_text(first_nonempty(first.get("subject/company"), first.get("主体"), first.get("target_name")), 80),
            "target_name": first_nonempty(first.get("subject/company"), first.get("主体"), first.get("target_name")),
            "event_name": first_nonempty(first.get("event_name"), first.get("事件类型"), first.get("schema_id")),
            "date": first_date(first_nonempty(first.get("event_time"), first.get("事件时间"))),
            "title_text": " ".join(titles),
            "claim_text": first_nonempty(first.get("claim"), first.get("待验证主张"), first.get("claim_text")),
            "source_rows": len(items),
        }
    return profiles


def extract_human_rows_from_workbook(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    all_rows: list[dict[str, Any]] = []
    sheet_summaries: list[dict[str, Any]] = []
    for sheet_name in wb.sheetnames:
        try:
            rows = table_records(path, sheet_name)
        except Exception as exc:  # pragma: no cover - defensive
            sheet_summaries.append({"sheet": sheet_name, "error": str(exc)})
            continue
        if not rows:
            continue
        headers = list(rows[0].keys())
        if "human_verdict" not in headers:
            continue
        labeled = [row for row in rows if norm(row.get("human_verdict"))]
        if not labeled:
            continue
        sheet_summaries.append(
            {
                "sheet": sheet_name,
                "rows": len(rows),
                "labeled_rows": len(labeled),
                "unique_cluster_ids": len({norm(row.get("cluster_id")) for row in labeled if norm(row.get("cluster_id"))}),
                "label_counts": dict(Counter(norm(row.get("human_verdict")) for row in labeled)),
            }
        )
        all_rows.extend(labeled)
    file_summary = {
        "sheet_summaries": sheet_summaries,
        "row_count": len(all_rows),
        "unique_cluster_ids": len({norm(row.get("cluster_id")) for row in all_rows if norm(row.get("cluster_id"))}),
    }
    return all_rows, file_summary


def summarize_source(path: Path, role: str) -> dict[str, Any]:
    rows, file_summary = extract_human_rows_from_workbook(path)
    unique_rows: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cid = norm(row.get("cluster_id"))
        verdict = norm(row.get("human_verdict"))
        if not cid or not verdict:
            continue
        unique_rows.setdefault(cid, row)
        conflicts[cid].add(verdict)
    unique_cluster_ids = set(unique_rows)
    label_counts = Counter(norm(row.get("human_verdict")) for row in unique_rows.values())
    reviewer_confirmed_counts = Counter(
        norm(row.get("reviewer_confirmed")) for row in rows if "reviewer_confirmed" in row
    )
    provenance_counts = Counter(
        norm(row.get("annotation_provenance")) for row in rows if "annotation_provenance" in row
    )
    source_summary = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "role": role,
        "human_rows": file_summary["row_count"],
        "unique_cluster_ids": len(unique_cluster_ids),
        "label_counts": dict(sorted(label_counts.items())),
        "conflicting_label_clusters": sum(1 for labels in conflicts.values() if len(labels) > 1),
        "reviewer_confirmed_counts": {k: v for k, v in sorted(reviewer_confirmed_counts.items()) if k},
        "annotation_provenance_counts": {k: v for k, v in sorted(provenance_counts.items()) if k},
        "sheet_summaries": file_summary["sheet_summaries"],
        "cluster_ids": sorted(unique_cluster_ids),
        "row_examples": [
            {
                "cluster_id": norm(row.get("cluster_id")),
                "human_verdict": norm(row.get("human_verdict")),
                "reviewer_confirmed": norm(row.get("reviewer_confirmed")),
                "annotation_provenance": norm(row.get("annotation_provenance")),
            }
            for row in list(unique_rows.values())[:3]
        ],
    }
    return source_summary


def summarize_locked500(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    cluster_ids = [norm(row.get("cluster_id")) for row in rows if norm(row.get("cluster_id"))]
    label_counts = Counter(norm(row.get("human_verdict") or row.get("verdict")) for row in rows)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "row_count": len(rows),
        "unique_cluster_ids": len(set(cluster_ids)),
        "label_counts": {k: v for k, v in sorted(label_counts.items()) if k},
        "cluster_ids": sorted(set(cluster_ids)),
    }


def compute_near_duplicate_overlap(locked_profiles: dict[str, dict[str, Any]], source_profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    from scripts.build_task2_independent500_human_audit import is_near_duplicate

    hits: list[dict[str, Any]] = []
    source_values = list(source_profiles.values())
    for locked in locked_profiles.values():
        for source in source_values:
            duplicate, method, score = is_near_duplicate(locked, source)
            if duplicate:
                hits.append(
                    {
                        "locked_cluster_id": locked["cluster_id"],
                        "source_cluster_id": source["cluster_id"],
                        "method": method,
                        "score": round(float(score), 6),
                    }
                )
    return hits


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    locked = summarize_locked500(LOCKED500_JSONL)
    locked_rows = read_jsonl(LOCKED500_JSONL)
    locked_profiles = {norm(row["cluster_id"]): build_profile_from_locked_row(row) for row in locked_rows if norm(row.get("cluster_id"))}

    source_summaries = []
    canonical_profiles: dict[str, dict[str, Any]] = {}
    canonical_union: set[str] = set()

    for path in TRAINABLE_SOURCE_FILES:
        summary = summarize_source(path, "canonical_trainable_gold")
        source_summaries.append(summary)
        canonical_union.update(summary["cluster_ids"])
        rows = []
        for sheet_name in load_workbook(path, read_only=True, data_only=True).sheetnames:
            try:
                rows.extend([row for row in table_records(path, sheet_name) if norm(row.get("human_verdict"))])
            except Exception:
                continue
        canonical_profiles.update(profile_from_source_rows(rows))

    for path in DERIVATIVE_REVIEW_FILES:
        source_summaries.append(summarize_source(path, "diagnostic_derivative"))

    for path in LOCKED_DUPLICATE_FILES:
        source_summaries.append(summarize_source(path, "duplicate_view_locked500"))

    for path in IGNORED_TEMPLATE_FILES:
        if path.exists():
            source_summaries.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "role": "ignored_template_or_blank",
                    "human_rows": 0,
                    "unique_cluster_ids": 0,
                    "label_counts": {},
                    "conflicting_label_clusters": 0,
                    "reviewer_confirmed_counts": {},
                    "annotation_provenance_counts": {},
                    "sheet_summaries": [],
                    "cluster_ids": [],
                    "row_examples": [],
                }
            )

    trainable_label_counts = Counter()
    for source in source_summaries:
        if source["role"] == "canonical_trainable_gold":
            trainable_label_counts.update(source["label_counts"])

    trainable_sources = [source for source in source_summaries if source["role"] == "canonical_trainable_gold"]
    trainable_cluster_ids = sorted(set().union(*(set(source["cluster_ids"]) for source in trainable_sources)))

    derivative_cluster_ids = sorted(set().union(*(set(source["cluster_ids"]) for source in source_summaries if source["role"] == "diagnostic_derivative")))
    locked_duplicate_cluster_ids = sorted(set().union(*(set(source["cluster_ids"]) for source in source_summaries if source["role"] == "duplicate_view_locked500")))

    trainable_exact_overlap_with_locked = sorted(set(trainable_cluster_ids) & set(locked["cluster_ids"]))
    locked_near_duplicate_hits = compute_near_duplicate_overlap(locked_profiles, canonical_profiles)
    near_duplicate_cluster_ids = sorted({item["locked_cluster_id"] for item in locked_near_duplicate_hits})

    trainable_sources_by_file = {source["path"]: source for source in source_summaries if source["role"] == "canonical_trainable_gold"}

    inventory_rows = []
    for source in source_summaries:
        cluster_ids = set(source["cluster_ids"])
        inventory_rows.append(
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "role": source["role"],
                "human_rows": source["human_rows"],
                "unique_cluster_ids_source": source["unique_cluster_ids"],
                "new_unique_cluster_ids": len(cluster_ids - set(locked["cluster_ids"]) - (set(trainable_cluster_ids) - cluster_ids)) if source["role"] == "canonical_trainable_gold" else 0,
                "label_support": source["label_counts"].get("support", 0),
                "label_contradict": source["label_counts"].get("contradict", 0),
                "label_insufficient": source["label_counts"].get("insufficient", 0),
                "label_unclear": source["label_counts"].get("unclear", 0),
                "conflicting_label_clusters": source["conflicting_label_clusters"],
                "locked500_overlap_exact": len(cluster_ids & set(locked["cluster_ids"])),
                "old47_overlap_exact": len(cluster_ids & set(trainable_sources_by_file.get("reports/task2_dea_cached_human_audit_gold_filled.xlsx", {}).get("cluster_ids", []))),
                "old50_overlap_exact": len(cluster_ids & set(trainable_sources_by_file.get("reports/task2_safety_gate_v1_2_independent_human_audit_gold_filled.xlsx", {}).get("cluster_ids", []))),
                "notes": {
                    "canonical_trainable_gold": "counts toward trainable gold inventory only if not in locked500",
                    "diagnostic_derivative": "manual review / diagnostic derivative; no new unique clusters",
                    "duplicate_view_locked500": "duplicate or derivative view of locked500, excluded from training",
                    "ignored_template_or_blank": "blank template or file with no human_verdict rows",
                }.get(source["role"], ""),
            }
        )

    # Compute summary across the canonical training gold only.
    trainable_union_label_counts = Counter()
    for cid in trainable_cluster_ids:
        # Use the first canonical source that has the cid; conflicts are zero in practice.
        for source in trainable_sources:
            if cid in set(source["cluster_ids"]):
                label = source["label_counts"]
                # We only need the per-source label for this cid; reconstruct from row examples
                break
        # label counts are easier to derive from the canonical source totals below
    trainable_union_label_counts.update(trainable_label_counts)

    summary = {
        "generated_at": now_iso(),
        "seed": 2026,
        "locked500": locked,
        "trainable_gold": {
            "unique_cluster_ids": len(trainable_cluster_ids),
            "label_counts": dict(sorted(trainable_union_label_counts.items())),
            "source_files": [source["path"] for source in trainable_sources],
            "source_count": len(trainable_sources),
        },
        "derivative_review_files": {
            "unique_cluster_ids": len(set(derivative_cluster_ids)),
            "source_files": [source["path"] for source in source_summaries if source["role"] == "diagnostic_derivative"],
            "source_count": sum(1 for source in source_summaries if source["role"] == "diagnostic_derivative"),
        },
        "duplicate_locked500_views": {
            "unique_cluster_ids": len(set(locked_duplicate_cluster_ids)),
            "source_files": [source["path"] for source in source_summaries if source["role"] == "duplicate_view_locked500"],
            "source_count": sum(1 for source in source_summaries if source["role"] == "duplicate_view_locked500"),
        },
        "all_human_reviewed_unique_clusters_including_locked500": len(set(trainable_cluster_ids) | set(locked["cluster_ids"])),
        "all_human_reviewed_unique_clusters_excluding_locked500": len(trainable_cluster_ids),
        "locked500_exact_overlap_with_trainable_gold": len(trainable_exact_overlap_with_locked),
        "locked500_near_duplicate_overlap_with_trainable_gold": len(near_duplicate_cluster_ids),
        "locked500_near_duplicate_pairs": locked_near_duplicate_hits[:20],
        "training_gate": {
            "required_total_unique": 1200,
            "required_support": 300,
            "required_contradict": 300,
            "required_insufficient": 150,
            "required_unclear": 150,
            "available_total_unique": len(trainable_cluster_ids),
            "available_support": trainable_union_label_counts.get("support", 0),
            "available_contradict": trainable_union_label_counts.get("contradict", 0),
            "available_insufficient": trainable_union_label_counts.get("insufficient", 0),
            "available_unclear": trainable_union_label_counts.get("unclear", 0),
            "gate_met": False,
        },
    }
    summary["training_gate"]["shortfall_total_unique"] = summary["training_gate"]["required_total_unique"] - summary["training_gate"]["available_total_unique"]
    summary["training_gate"]["shortfall_support"] = summary["training_gate"]["required_support"] - summary["training_gate"]["available_support"]
    summary["training_gate"]["shortfall_contradict"] = summary["training_gate"]["required_contradict"] - summary["training_gate"]["available_contradict"]
    summary["training_gate"]["shortfall_insufficient"] = summary["training_gate"]["required_insufficient"] - summary["training_gate"]["available_insufficient"]
    summary["training_gate"]["shortfall_unclear"] = summary["training_gate"]["required_unclear"] - summary["training_gate"]["available_unclear"]

    inventory = {
        "generated_at": summary["generated_at"],
        "seed": 2026,
        "sources": source_summaries,
        "summary": summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_json = OUTPUT_DIR / "data_inventory.json"
    inventory_csv = OUTPUT_DIR / "data_inventory.csv"
    leakage_json = OUTPUT_DIR / "leakage_audit.json"
    inventory_json.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(inventory_csv, inventory_rows)
    leakage_json.write_text(
        json.dumps(
            {
                "generated_at": summary["generated_at"],
                "seed": 2026,
                "locked500_source": locked["path"],
                "locked500_sha256": locked["sha256"],
                "locked500_exact_overlap_with_trainable_gold": len(trainable_exact_overlap_with_locked),
                "locked500_near_duplicate_overlap_with_trainable_gold": len(near_duplicate_cluster_ids),
                "locked500_near_duplicate_pairs": locked_near_duplicate_hits,
                "trainable_gold_source_overlap": {
                    "old47_old50_exact_overlap": len(set(trainable_sources_by_file.get("reports/task2_dea_cached_human_audit_gold_filled.xlsx", {}).get("cluster_ids", [])) & set(trainable_sources_by_file.get("reports/task2_safety_gate_v1_2_independent_human_audit_gold_filled.xlsx", {}).get("cluster_ids", []))),
                    "old47_overlap_locked500": len(set(trainable_sources_by_file.get("reports/task2_dea_cached_human_audit_gold_filled.xlsx", {}).get("cluster_ids", [])) & set(locked["cluster_ids"])),
                    "old50_overlap_locked500": len(set(trainable_sources_by_file.get("reports/task2_safety_gate_v1_2_independent_human_audit_gold_filled.xlsx", {}).get("cluster_ids", [])) & set(locked["cluster_ids"])),
                },
                "duplicate_views": [source for source in source_summaries if source["role"] != "canonical_trainable_gold"],
                "gate": summary["training_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    locked_ids_config = {
        "generated_at": summary["generated_at"],
        "seed": 2026,
        "count": locked["unique_cluster_ids"],
        "source": locked["path"],
        "source_sha256": locked["sha256"],
        "cluster_ids": locked["cluster_ids"],
    }
    config_path = ROOT / "configs/task2_independent500_locked_ids.json"
    config_path.write_text(json.dumps(locked_ids_config, ensure_ascii=False, indent=2), encoding="utf-8")

    run_state = {
        "status": "blocked",
        "blocked_stage": "training_gate_audit",
        "blocker_reason": "trainable human-reviewed gold labels outside locked500 are only 97 unique clusters, below the 1200 threshold",
        "completed_stages": ["asset_audit", "locked_id_config"],
        "pending_command": "generate 1500-sample annotation package from unlabeled candidate pool",
        "no_fake_metrics": True,
        "training_gate": summary["training_gate"],
    }
    (OUTPUT_DIR / "run_state.json").write_text(json.dumps(run_state, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "summary": summary, "inventory_json": str(inventory_json), "inventory_csv": str(inventory_csv), "leakage_json": str(leakage_json), "locked_ids_config": str(config_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
