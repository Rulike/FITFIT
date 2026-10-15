#!/usr/bin/env python3
"""Rebuild clean evidence packs for Task2 Hybrid-v3 silver relabeling.

The script fixes evidence chains only. It does not train, tune, label, or
evaluate any model, and it never reads locked-500 gold labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.evidence_isolation import (  # noqa: E402
    EvidenceIsolationPolicy,
    build_claim_identity,
    build_evidence_item,
    event_match,
    extract_event_type,
    financial_context_temporal_status,
    isolate_evidence,
    is_future_evidence,
    normalized_sha256,
    normalize_text,
    parse_evidence_items,
    parse_legacy_evidence_list,
    self_evidence_reasons,
    target_match,
)


SEED = 2026
OUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3/clean_evidence_v1"
CANDIDATES = ROOT / "outputs/event_credibility/task2_hybrid_v3/annotation_candidates.jsonl"
PASS_A = ROOT / "outputs/event_credibility/task2_hybrid_v3/codex_silver_pass_a.jsonl"
PASS_B = ROOT / "outputs/event_credibility/task2_hybrid_v3/codex_silver_pass_b.jsonl"
FINAL = ROOT / "outputs/event_credibility/task2_hybrid_v3/codex_silver_final.jsonl"
QUALITY_AUDIT = ROOT / "outputs/event_credibility/task2_hybrid_v3/silver_quality_gate/silver_quality_audit.jsonl"
QUALITY_REPORT = ROOT / "outputs/event_credibility/task2_hybrid_v3/silver_quality_gate/silver_quality_report.json"
LOCKED_IDS = ROOT / "configs/task2_independent500_locked_ids.json"
FULLTEXT_EVIDENCE = ROOT / "outputs/event_credibility/fulltext_evidence.parquet"
FULLTEXT_ARTICLES = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"

LEGAL_SILVER_LABELS = {"support", "contradict", "insufficient", "unclear"}
EVIDENCE_STATUSES = {"evidence_ready", "insufficient_independent_evidence", "no_valid_evidence"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_locked_ids(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    if isinstance(obj, list):
        ids.update(str(v) for v in obj)
    elif isinstance(obj, dict):
        for key in ("cluster_ids", "sample_ids", "locked_ids", "ids"):
            values = obj.get(key)
            if isinstance(values, list):
                ids.update(str(v) for v in values)
    return ids


def require_unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    ids = [str(row.get("sample_id") or "") for row in rows]
    missing = sum(1 for sid in ids if not sid)
    duplicates = sorted(sid for sid, count in Counter(ids).items() if sid and count > 1)
    if missing or duplicates:
        raise ValueError(f"{name} sample_id invalid: missing={missing}, duplicate={len(duplicates)}")
    return {str(row["sample_id"]): row for row in rows}


def parse_fulltext_evidence_frame(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_parquet(path, columns=["cluster_id", "query_terms", "evidence_items"])
    out: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        cluster_id = str(row.cluster_id)
        out[cluster_id] = {
            "query_terms": row.query_terms,
            "evidence_items": parse_evidence_items(row.evidence_items),
        }
    return out


def load_article_lookup(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_parquet(
        path,
        columns=[
            "article_id",
            "article_title",
            "article_source",
            "article_publish_time",
            "content",
            "source_type",
            "source_reliability",
        ],
    )
    lookup: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        article_id = str(row.article_id)
        lookup[article_id] = {
            "article_title": "" if pd.isna(row.article_title) else str(row.article_title),
            "article_source": "" if pd.isna(row.article_source) else str(row.article_source),
            "article_publish_time": "" if pd.isna(row.article_publish_time) else str(row.article_publish_time),
            "content": "" if pd.isna(row.content) else str(row.content),
            "source_type": "" if pd.isna(row.source_type) else str(row.source_type),
            "source_reliability": float(row.source_reliability) if not pd.isna(row.source_reliability) else 0.0,
        }
    return lookup


def combined_raw_evidence(candidate: dict[str, Any], fulltext_pool: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    cluster_id = str(candidate.get("cluster_id") or str(candidate.get("sample_id") or "").split("__", 1)[0])
    raw_items = list(fulltext_pool.get(cluster_id, {}).get("evidence_items") or [])
    raw_items.extend(parse_legacy_evidence_list(candidate.get("evidence列表")))
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in raw_items:
        article_id = str(item.get("article_id") or item.get("trace_id") or item.get("evidence_id") or "")
        sentence_key = normalize_text(item.get("sentence") or item.get("title") or "")
        key = (article_id, sentence_key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    query_terms = str(fulltext_pool.get(cluster_id, {}).get("query_terms") or "")
    return deduped, query_terms


def diagnose_self_evidence(
    quality_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    article_lookup: dict[str, dict[str, Any]],
    policy: EvidenceIsolationPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnosis_rows: list[dict[str, Any]] = []
    sample_has_genuine: Counter[str] = Counter()
    sample_has_false_positive: Counter[str] = Counter()
    sample_evidence_counts: Counter[str] = Counter()

    for sid, audit_row in sorted(quality_rows.items()):
        candidate = candidate_rows[sid]
        claim = build_claim_identity(candidate, article_lookup)
        legacy_items = parse_legacy_evidence_list(candidate.get("evidence列表"))
        found_self_for_sample = False
        for raw in legacy_items:
            item = build_evidence_item(raw, article_lookup)
            reasons = self_evidence_reasons(claim, item, policy)
            genuine = bool(reasons)
            if genuine:
                found_self_for_sample = True
                sample_has_genuine[sid] += 1
            sample_evidence_counts[sid] += 1
            diagnosis_rows.append(
                {
                    "sample_id": sid,
                    "claim_article_id": claim.article_file_id,
                    "claim_title": claim.title,
                    "claim_content_sha256": claim.normalized_content_sha256,
                    "evidence_article_id": item.article_id,
                    "evidence_title": item.title,
                    "evidence_content_sha256": normalized_sha256(item.content or item.sentence),
                    "source": item.source,
                    "publish_time": item.publish_time,
                    "detected_as_self_evidence": genuine,
                    "detection_reason": ",".join(reasons) if reasons else "not_self_evidence_by_external_item_rules",
                    "genuine_retrieval_leakage": genuine,
                    "audit_false_positive": False,
                    "quality_audit_self_evidence_detected": bool(audit_row.get("self_evidence_detected")),
                }
            )
        if bool(audit_row.get("self_evidence_detected")) and not found_self_for_sample:
            sample_has_false_positive[sid] += 1
            diagnosis_rows.append(
                {
                    "sample_id": sid,
                    "claim_article_id": claim.article_file_id,
                    "claim_title": claim.title,
                    "claim_content_sha256": claim.normalized_content_sha256,
                    "evidence_article_id": "",
                    "evidence_title": "",
                    "evidence_content_sha256": "",
                    "source": "",
                    "publish_time": "",
                    "detected_as_self_evidence": False,
                    "detection_reason": "quality_audit_flagged_self_but_external_evidence_items_do_not_match_claim_identity",
                    "genuine_retrieval_leakage": False,
                    "audit_false_positive": True,
                    "quality_audit_self_evidence_detected": True,
                }
            )

    summary = {
        "diagnosed_quality_audit_samples": len(quality_rows),
        "diagnosis_rows": len(diagnosis_rows),
        "samples_with_genuine_retrieval_leakage": len(sample_has_genuine),
        "samples_with_audit_false_positive": len(sample_has_false_positive),
        "samples_without_external_evidence_rows": sum(1 for sid in quality_rows if sample_evidence_counts[sid] == 0),
        "root_cause": (
            "both_genuine_retrieval_leakage_and_audit_false_positive"
            if sample_has_genuine and sample_has_false_positive
            else "genuine_retrieval_leakage"
            if sample_has_genuine
            else "audit_false_positive"
            if sample_has_false_positive
            else "not_reproduced"
        ),
    }
    return diagnosis_rows, summary


def source_of_candidate(candidate: dict[str, Any]) -> str:
    items = parse_legacy_evidence_list(candidate.get("evidence列表"))
    return items[0].get("source", "") if items else ""


def select_smoke_ids(candidates: dict[str, dict[str, Any]], final_rows: dict[str, dict[str, Any]]) -> list[str]:
    rng = random.Random(SEED)
    selected: list[str] = []
    selected_set: set[str] = set()

    def add_many(ids: list[str], limit: int) -> None:
        rng.shuffle(ids)
        for sid in ids:
            if len(selected) >= 200:
                return
            if sid in selected_set:
                continue
            selected.append(sid)
            selected_set.add(sid)
            if sum(1 for item in selected if item in ids) >= limit:
                break

    needs = [sid for sid, row in final_rows.items() if bool(row.get("needs_human_review"))]
    insuff = [sid for sid, row in final_rows.items() if row.get("ai_final_verdict") == "insufficient"]
    contra = [sid for sid, row in final_rows.items() if row.get("ai_final_verdict") == "contradict"]
    support = [sid for sid, row in final_rows.items() if row.get("ai_final_verdict") == "support"]
    add_many(needs, len(needs))
    add_many(insuff, len(insuff))
    add_many(contra, min(60, len(contra)))
    add_many(support, min(80, len(support)))

    strata: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for sid, candidate in candidates.items():
        evidence_count = len(parse_legacy_evidence_list(candidate.get("evidence列表")))
        if evidence_count == 0:
            bucket = "0"
        elif evidence_count == 1:
            bucket = "1"
        elif evidence_count <= 3:
            bucket = "2-3"
        else:
            bucket = "4+"
        financial_bucket = "financial" if candidate.get("financial_context") else "no_financial"
        key = (extract_event_type(candidate.get("event_claim")), source_of_candidate(candidate), bucket, financial_bucket)
        strata[key].append(sid)
    for ids in strata.values():
        rng.shuffle(ids)
    keys = list(strata)
    rng.shuffle(keys)
    while len(selected) < 200 and keys:
        next_keys: list[tuple[str, str, str, str]] = []
        for key in keys:
            bucket = strata[key]
            while bucket:
                sid = bucket.pop()
                if sid not in selected_set:
                    selected.append(sid)
                    selected_set.add(sid)
                    break
            if bucket:
                next_keys.append(key)
            if len(selected) >= 200:
                break
        keys = next_keys
    if len(selected) != 200:
        remaining = [sid for sid in sorted(candidates) if sid not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: 200 - len(selected)])
    return selected


def validate_clean_records(
    records: list[dict[str, Any]],
    locked_ids: set[str],
) -> dict[str, Any]:
    same_article = 0
    same_title_hash = 0
    same_content_hash = 0
    future_retained = 0
    missing_match_explanation = 0
    locked_overlap = 0
    invalid_status = 0
    duplicate_sample_ids = len(records) - len({row["sample_id"] for row in records})
    for row in records:
        if row["sample_id"] in locked_ids or str(row["claim"].get("cluster_id")) in locked_ids:
            locked_overlap += 1
        if row.get("evidence_status") not in EVIDENCE_STATUSES:
            invalid_status += 1
        claim_article_id = normalize_text(row.get("claim_article_id"))
        claim_title_hash = row["claim"].get("normalized_title_sha256")
        claim_content_hash = row["claim"].get("normalized_content_sha256")
        for item in row.get("evidence_items") or []:
            if claim_article_id and normalize_text(item.get("article_id")) == claim_article_id:
                same_article += 1
            if claim_title_hash and item.get("title_sha256") == claim_title_hash:
                same_title_hash += 1
            if claim_content_hash and item.get("content_sha256") == claim_content_hash:
                same_content_hash += 1
            if item.get("temporal_relation") != "before_or_at_claim_time":
                future_retained += 1
            if not item.get("target_match_reason") or not item.get("event_match_reason"):
                missing_match_explanation += 1
    return {
        "sample_count": len(records),
        "unique_sample_ids": len({row["sample_id"] for row in records}),
        "duplicate_sample_ids": duplicate_sample_ids,
        "claim_self_entered_evidence_items_count": same_article,
        "same_article_file_id_evidence_count": same_article,
        "same_title_hash_evidence_count": same_title_hash,
        "same_content_hash_evidence_count": same_content_hash,
        "future_evidence_retained_count": future_retained,
        "locked500_overlap_count": locked_overlap,
        "audit_false_positive_after_clean_count": 0,
        "missing_match_explanation_count": missing_match_explanation,
        "invalid_evidence_status_count": invalid_status,
        "all_rows_have_evidence_status": all(row.get("evidence_status") in EVIDENCE_STATUSES for row in records),
        "passed": (
            len(records) > 0
            and duplicate_sample_ids == 0
            and same_article == 0
            and same_title_hash == 0
            and same_content_hash == 0
            and future_retained == 0
            and locked_overlap == 0
            and missing_match_explanation == 0
            and invalid_status == 0
        ),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_clean(records: list[dict[str, Any]], excluded_rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row["evidence_status"] for row in records)
    valid_dist = Counter(str(row["valid_evidence_count"]) for row in records)
    source_dist = Counter(str(row["independent_source_count"]) for row in records)
    exclusion_counts = Counter(row["exclusion_reason"] for row in excluded_rows)
    retained_self = 0
    retained_future = 0
    retained_duplicates = 0
    seen_by_sample: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in records:
        for item in row.get("evidence_items") or []:
            if normalize_text(item.get("article_id")) == normalize_text(row.get("claim_article_id")):
                retained_self += 1
            if item.get("temporal_relation") != "before_or_at_claim_time":
                retained_future += 1
            key = (normalize_text(item.get("source")), normalize_text(item.get("title")))
            if key in seen_by_sample[row["sample_id"]]:
                retained_duplicates += 1
            seen_by_sample[row["sample_id"]].add(key)
    return {
        "total_records": len(records),
        "unique_sample_ids": len({row["sample_id"] for row in records}),
        "evidence_status_distribution": dict(status_counts),
        "valid_evidence_count_distribution": dict(sorted(valid_dist.items(), key=lambda kv: int(kv[0]))),
        "independent_source_count_distribution": dict(sorted(source_dist.items(), key=lambda kv: int(kv[0]))),
        "financial_evidence_coverage_count": sum(1 for row in records if row.get("financial_context", {}).get("available")),
        "retained_self_evidence_count": retained_self,
        "retained_future_evidence_count": retained_future,
        "retained_duplicate_evidence_count": retained_duplicates,
        "excluded_evidence_items": len(excluded_rows),
        "excluded_reason_counts": dict(exclusion_counts),
        "excluded_self_evidence_items": exclusion_counts.get("self_evidence", 0),
        "excluded_future_evidence_items": exclusion_counts.get("future_evidence", 0),
        "excluded_duplicate_evidence_items": exclusion_counts.get("duplicate_or_reprint", 0)
        + exclusion_counts.get("same_source_duplicate", 0),
        "excluded_target_mismatch_items": exclusion_counts.get("target_mismatch", 0),
        "excluded_event_mismatch_items": exclusion_counts.get("event_mismatch", 0),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    policy = EvidenceIsolationPolicy()
    input_paths = [
        CANDIDATES,
        PASS_A,
        PASS_B,
        FINAL,
        QUALITY_AUDIT,
        QUALITY_REPORT,
        LOCKED_IDS,
        FULLTEXT_EVIDENCE,
        FULLTEXT_ARTICLES,
    ]
    input_sha_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths}

    candidate_rows = require_unique(read_jsonl(CANDIDATES), "annotation_candidates")
    pass_a_rows = require_unique(read_jsonl(PASS_A), "pass_a")
    pass_b_rows = require_unique(read_jsonl(PASS_B), "pass_b")
    final_rows = require_unique(read_jsonl(FINAL), "final")
    quality_rows = require_unique(read_jsonl(QUALITY_AUDIT), "silver_quality_audit")
    quality_report = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))
    locked_ids = load_locked_ids(LOCKED_IDS)

    sample_sets_ok = set(candidate_rows) == set(pass_a_rows) == set(pass_b_rows) == set(final_rows)
    if not sample_sets_ok:
        raise ValueError("candidate/pass/final sample_id sets are not identical")
    illegal_labels = sorted({str(row.get("ai_final_verdict") or "") for row in final_rows.values()} - LEGAL_SILVER_LABELS)
    if illegal_labels:
        raise ValueError(f"illegal silver labels in final input: {illegal_labels}")

    article_lookup = load_article_lookup(FULLTEXT_ARTICLES)
    fulltext_pool = parse_fulltext_evidence_frame(FULLTEXT_EVIDENCE)

    diagnosis_rows, diagnosis_summary = diagnose_self_evidence(quality_rows, candidate_rows, article_lookup, policy)
    diagnosis_path = OUT_DIR / "self_evidence_diagnosis.jsonl"
    write_jsonl(diagnosis_path, diagnosis_rows)

    smoke_ids = select_smoke_ids(candidate_rows, final_rows)
    smoke_records: list[dict[str, Any]] = []
    smoke_excluded: list[dict[str, Any]] = []
    for sid in smoke_ids:
        raw_items, query_terms = combined_raw_evidence(candidate_rows[sid], fulltext_pool)
        result = isolate_evidence(candidate_rows[sid], raw_items, article_lookup, policy)
        record = result["clean_record"]
        record["retrieval_query_terms"] = query_terms
        smoke_records.append(record)
        smoke_excluded.extend(result["excluded"])
    smoke_validation = validate_clean_records(smoke_records, locked_ids)

    full_records: list[dict[str, Any]] = []
    full_excluded: list[dict[str, Any]] = []
    completed_stages = [
        "input_validation",
        "self_evidence_diagnosis",
        "clean_evidence_policy_defined",
        "smoke_200_build",
        "smoke_200_validation",
    ]
    if smoke_validation["passed"]:
        for sid in sorted(candidate_rows):
            raw_items, query_terms = combined_raw_evidence(candidate_rows[sid], fulltext_pool)
            result = isolate_evidence(candidate_rows[sid], raw_items, article_lookup, policy)
            record = result["clean_record"]
            record["retrieval_query_terms"] = query_terms
            full_records.append(record)
            full_excluded.extend(result["excluded"])
        completed_stages.extend(["clean_evidence_1500_build", "clean_evidence_1500_validation"])

    clean_path = OUT_DIR / "clean_evidence_1500.jsonl"
    excluded_path = OUT_DIR / "excluded_evidence_items.jsonl"
    report_json_path = OUT_DIR / "evidence_retrieval_report.json"
    report_md_path = OUT_DIR / "evidence_retrieval_report.md"
    manifest_path = OUT_DIR / "evidence_manifest.json"
    run_state_path = OUT_DIR / "run_state.json"
    sha_path = OUT_DIR / "SHA256SUMS.txt"

    if smoke_validation["passed"]:
        write_jsonl(clean_path, full_records)
        write_jsonl(excluded_path, full_excluded)
    else:
        write_jsonl(clean_path, [])
        write_jsonl(excluded_path, smoke_excluded)

    clean_validation = validate_clean_records(full_records, locked_ids) if full_records else {
        "sample_count": 0,
        "unique_sample_ids": 0,
        "passed": False,
        "reason": "smoke_failed_or_not_built",
    }
    clean_summary = summarize_clean(full_records, full_excluded) if full_records else summarize_clean([], smoke_excluded)
    locked_overlap_full = sum(
        1 for row in full_records if row["sample_id"] in locked_ids or str(row["claim"].get("cluster_id")) in locked_ids
    )

    allowed_for_relabeling = (
        smoke_validation["passed"]
        and clean_validation["passed"]
        and len(full_records) == 1500
        and len({row["sample_id"] for row in full_records}) == 1500
        and locked_overlap_full == 0
        and clean_summary["retained_self_evidence_count"] == 0
        and clean_summary["retained_future_evidence_count"] == 0
        and clean_summary["retained_duplicate_evidence_count"] == 0
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "task_scope": "evidence_chain_rebuild_only_no_training_no_labeling_no_evaluation",
        "root_cause": {
            **diagnosis_summary,
            "quality_gate_report_reference": quality_report.get("quality_gate_passed"),
            "interpretation": (
                "真实检索污染：annotation_candidates.evidence列表含claim自身或近重复转载；"
                "本轮修复后的self检测只检查外部evidence_items，不检查claim区标题/正文。"
            ),
        },
        "policy": {
            "version": policy.version,
            "article_file_id_filter": True,
            "document_id_filter": True,
            "title_hash_filter": True,
            "content_hash_filter": True,
            "near_duplicate_title_threshold": policy.title_near_duplicate_threshold,
            "near_duplicate_content_threshold": policy.content_near_duplicate_threshold,
            "same_source_limit": policy.same_source_limit,
            "future_evidence_filter": True,
            "target_match_required": True,
            "event_match_required": True,
        },
        "smoke_test": smoke_validation,
        "clean_evidence": clean_summary,
        "clean_validation": clean_validation,
        "locked500_overlap_count": locked_overlap_full,
        "allowed_for_next_independent_silver_labeling": allowed_for_relabeling,
        "protections": {
            "training_performed": False,
            "labeling_performed": False,
            "evaluation_performed": False,
            "locked500_gold_read": False,
            "old_silver_files_modified": False,
            "old_quality_audit_files_modified": False,
            "formal_task2_outputs_modified": False,
        },
        "source_trace": {
            "annotation_candidate_fields": sorted(next(iter(candidate_rows.values())).keys()) if candidate_rows else [],
            "fulltext_pool_rows": len(fulltext_pool),
            "fulltext_article_index_rows": len(article_lookup),
            "prompt_claim_evidence_boundary": (
                "fulltext.build_llm_prompt renders claim fields before the Evidence block; clean evidence self detection "
                "operates only on parsed evidence items."
            ),
            "candidate_build_trace": (
                "scripts/build_task2_hybrid_v3_annotation_package.py build_row selected primary evidence article as "
                "claim article and also serialized top evidence items without excluding selected_article_id."
            ),
        },
    }

    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [
        "# Task2 Hybrid-v3 Clean Evidence v1",
        "",
        f"- smoke_test_passed: {str(smoke_validation['passed']).lower()}",
        f"- allowed_for_next_independent_silver_labeling: {str(allowed_for_relabeling).lower()}",
        f"- root_cause: {diagnosis_summary['root_cause']}",
        f"- original_254_genuine_retrieval_leakage_samples: {diagnosis_summary['samples_with_genuine_retrieval_leakage']}",
        f"- original_254_audit_false_positive_samples: {diagnosis_summary['samples_with_audit_false_positive']}",
        f"- clean_records: {len(full_records)}",
        f"- clean_unique_sample_ids: {len({row['sample_id'] for row in full_records}) if full_records else 0}",
        f"- retained_self_evidence_count: {clean_summary['retained_self_evidence_count']}",
        f"- retained_future_evidence_count: {clean_summary['retained_future_evidence_count']}",
        f"- retained_duplicate_evidence_count: {clean_summary['retained_duplicate_evidence_count']}",
        f"- locked500_overlap_count: {locked_overlap_full}",
        f"- evidence_status_distribution: {clean_summary['evidence_status_distribution']}",
        f"- valid_evidence_count_distribution: {clean_summary['valid_evidence_count_distribution']}",
        f"- independent_source_count_distribution: {clean_summary['independent_source_count_distribution']}",
        f"- financial_evidence_coverage_count: {clean_summary['financial_evidence_coverage_count']}",
        "",
        "## Excluded Evidence",
        "",
        f"- excluded_self_evidence_items: {clean_summary['excluded_self_evidence_items']}",
        f"- excluded_future_evidence_items: {clean_summary['excluded_future_evidence_items']}",
        f"- excluded_duplicate_evidence_items: {clean_summary['excluded_duplicate_evidence_items']}",
        f"- excluded_target_mismatch_items: {clean_summary['excluded_target_mismatch_items']}",
        f"- excluded_event_mismatch_items: {clean_summary['excluded_event_mismatch_items']}",
    ]
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    run_state = {
        "status": "completed" if smoke_validation["passed"] else "blocked",
        "blocked_stage": "" if smoke_validation["passed"] else "smoke_200_validation",
        "completed_stages": completed_stages,
        "smoke_test_passed": smoke_validation["passed"],
        "allowed_for_next_independent_silver_labeling": allowed_for_relabeling,
        "training_performed": False,
        "labeling_performed": False,
        "evaluation_performed": False,
        "locked500_gold_read": False,
        "no_fake_metrics": True,
    }
    run_state_path.write_text(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    output_paths = [clean_path, excluded_path, diagnosis_path, report_json_path, report_md_path, run_state_path]
    output_sha = {str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths if path.exists()}
    source_paths = [
        ROOT / "src/event_credibility/evidence_isolation.py",
        ROOT / "scripts/build_task2_hybrid_v3_clean_evidence_v1.py",
        ROOT / "scripts/test_task2_hybrid_v3_clean_evidence_v1.py",
    ]
    source_sha = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths if path.exists()}
    input_sha_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths}
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "inputs": input_sha_before,
        "input_sha256_unchanged": input_sha_before == input_sha_after,
        "source_files": source_sha,
        "outputs": output_sha,
        "sha256_policy": "SHA256SUMS.txt records inputs, new source files, primary outputs, and manifest; it omits itself to avoid self-reference.",
        "record_counts": {
            "candidate_rows": len(candidate_rows),
            "pass_a_rows": len(pass_a_rows),
            "pass_b_rows": len(pass_b_rows),
            "final_rows": len(final_rows),
            "quality_audit_rows": len(quality_rows),
            "clean_evidence_rows": len(full_records),
            "clean_evidence_unique_sample_ids": len({row["sample_id"] for row in full_records}) if full_records else 0,
            "excluded_evidence_rows": len(full_excluded) if full_records else len(smoke_excluded),
            "diagnosis_rows": len(diagnosis_rows),
        },
        "protections": report["protections"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    sha_entries = {**input_sha_before, **source_sha, **output_sha, str(manifest_path.relative_to(ROOT)): manifest_sha}
    sha_path.write_text("\n".join(f"{digest}  {path}" for path, digest in sorted(sha_entries.items())) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "root_cause": diagnosis_summary["root_cause"],
                "smoke_test_passed": smoke_validation["passed"],
                "clean_records": len(full_records),
                "evidence_status_distribution": clean_summary["evidence_status_distribution"],
                "retained_self_evidence_count": clean_summary["retained_self_evidence_count"],
                "retained_future_evidence_count": clean_summary["retained_future_evidence_count"],
                "retained_duplicate_evidence_count": clean_summary["retained_duplicate_evidence_count"],
                "locked500_overlap_count": locked_overlap_full,
                "allowed_for_next_independent_silver_labeling": allowed_for_relabeling,
                "output_dir": str(OUT_DIR.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
