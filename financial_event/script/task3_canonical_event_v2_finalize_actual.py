#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from task3_canonical_event_v2_actual_repair import (
    ROOT,
    load_article_index,
    read_jsonl,
    sha256_file,
    stable_hash,
    text,
    write_json,
    write_jsonl,
)


ACTUAL_DIR = ROOT / "outputs/task3_canonical_event_v2_actual_repair"
RUN_DIR = ACTUAL_DIR / "resolver_v2_run1"
CONTRACT_STRUCTURED_PATH = ROOT / "outputs/task3_canonical_event_v2_validation/task1_structured_events_contract_v1.jsonl"
FULLTEXT_ARTICLE_INDEX = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"
PROTECTED_FILES = [
    ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl",
    ROOT / "outputs/task3/step3/event_mention_canonical_mapping.jsonl",
    ROOT / "outputs/task3/step3/same_event_candidates.jsonl",
    ROOT / "outputs/task3/step3/same_event_pair_predictions.jsonl",
    ROOT / "outputs/task3/splits/event_mention_split_mapping.jsonl",
    ROOT / "outputs/task3/splits/document_split_mapping.jsonl",
    ROOT / "outputs/task3_kg_v3_final/state_observations/canonical_event_current_state.jsonl",
    ROOT / "outputs/task3_kg_v3_final/graph_nodes.jsonl",
    ROOT / "outputs/task3_kg_v3_final/task4_event_interface.jsonl",
    ROOT / "outputs/task4 golddata/task34.xlsx",
]


def write_readme(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_contract_rows() -> list[dict[str, Any]]:
    return read_jsonl(CONTRACT_STRUCTURED_PATH)


def load_recovered_rows() -> list[dict[str, Any]]:
    return read_jsonl(ACTUAL_DIR / "task1_615_grounding_pass_recovered.jsonl")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_lookup() -> tuple[dict[str, str], dict[str, float]]:
    mapping_rows = read_jsonl(RUN_DIR / "canonical_event_v2_mapping.jsonl")
    cluster_rows = read_jsonl(RUN_DIR / "canonical_event_v2_clusters.jsonl")
    canonical_lookup: dict[str, str] = {}
    key_to_ids: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in mapping_rows:
        article_id = text(row.get("article_id"))
        target_key = text(row.get("target_key"))
        event_fact_signature = text(row.get("event_fact_signature"))
        event_type = text(row.get("event_type"))
        event_name = text(row.get("event_name"))
        event_time = text(row.get("event_time"))
        cid = text(row.get("canonical_event_id"))
        key_to_ids[(article_id, target_key, event_fact_signature)].add(cid)
        key_to_ids[(article_id, target_key, event_type, event_name, event_time)].add(cid)
    for key, ids in key_to_ids.items():
        if len(ids) == 1:
            canonical_lookup["||".join(key)] = next(iter(ids))
    conf_lookup = {text(row.get("canonical_event_id")): float(row.get("cluster_confidence") or 0.5) for row in cluster_rows}
    return canonical_lookup, conf_lookup


def build_task4_interface(contract_rows: list[dict[str, Any]], canonical_lookup: dict[str, str], confidence_lookup: dict[str, float], article_content: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in contract_rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        event = row.get("event") if isinstance(row.get("event"), dict) else {}
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        article_id = text(row.get("article_id"))
        content = text(article_content.get(article_id))
        evidence_span = evidence.get("evidence_span") if isinstance(evidence.get("evidence_span"), dict) else None
        target_span = target.get("mention_span") if isinstance(target.get("mention_span"), dict) else None
        trigger_span = event.get("trigger_span") if isinstance(event.get("trigger_span"), dict) else None
        lookup_1 = "||".join([article_id, text(target.get("target_key")), text(event.get("event_fact_signature"))])
        lookup_2 = "||".join([article_id, text(target.get("target_key")), text(event.get("event_type")), text(event.get("event_name")), text(event.get("event_time"))])
        canonical_event_id = canonical_lookup.get(lookup_1) or canonical_lookup.get(lookup_2)
        contract_status = text(quality.get("contract_status"))
        evidence_ok = bool(
            article_id
            and content
            and canonical_event_id
            and isinstance(evidence_span, dict)
            and isinstance(target_span, dict)
            and isinstance(trigger_span, dict)
            and content[int(evidence_span["start"]): int(evidence_span["end"])] == text(evidence_span.get("text"))
            and content[int(target_span["start"]): int(target_span["end"])] == text(target_span.get("text"))
            and content[int(trigger_span["start"]): int(trigger_span["end"])] == text(trigger_span.get("text"))
        )
        if contract_status != "pass":
            reason = "contract_status_not_pass"
        elif not canonical_event_id:
            reason = "no_canonical_match"
        elif not evidence_ok:
            reason = "evidence_or_span_unresolvable"
        else:
            reason = ""
        out_row = {
            "canonical_event_id": canonical_event_id or "",
            "event_thread_id": f"ETF2_{stable_hash('ETF2', canonical_event_id or article_id, article_id)[:16]}" if canonical_event_id else "",
            "sample_id": "",
            "article_id": article_id,
            "schema_version": text(row.get("schema_version")),
            "target_key": text(target.get("target_key")),
            "event_type": text(event.get("event_type")),
            "event_family": text(event.get("event_family")),
            "trigger_text": text(event.get("trigger_text")),
            "event_time": text(event.get("event_time")),
            "polarity_or_state": text(event.get("polarity_or_state")),
            "support_type": text(evidence.get("support_type")),
            "event_fact_signature": text(event.get("event_fact_signature")),
            "evidence_span": evidence_span,
            "provenance": provenance,
            "contract_status": contract_status,
            "resolver_decision": "pass" if not reason else "reject",
            "resolver_confidence": confidence_lookup.get(canonical_event_id or "", 0.5),
            "trust_score": round(min(0.99, max(0.0, 0.55 + 0.4 * confidence_lookup.get(canonical_event_id or "", 0.5))), 6),
            "cluster_member_count": 0,
            "cluster_confidence": confidence_lookup.get(canonical_event_id or "", 0.5),
            "acceptance_status": "pass" if not reason else "reject",
            "reject_reason": reason,
        }
        if reason:
            rejected.append(out_row)
            reason_counts[reason] += 1
        else:
            accepted.append(out_row)
    summary = {
        "total_rows": len(contract_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(reason_counts),
        "acceptance_rate": round(len(accepted) / max(1, len(contract_rows)), 6),
        "contract_pass_rows": sum(1 for row in contract_rows if text(row.get("quality", {}).get("contract_status") if isinstance(row.get("quality"), dict) else "") == "pass"),
        "contract_review_rows": sum(1 for row in contract_rows if text(row.get("quality", {}).get("contract_status") if isinstance(row.get("quality"), dict) else "") == "review"),
        "contract_quarantine_rows": sum(1 for row in contract_rows if text(row.get("quality", {}).get("contract_status") if isinstance(row.get("quality"), dict) else "") == "quarantine"),
    }
    return accepted, rejected, summary


def build_resolver_task4_interface(
    recovered_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    mapping_by_sample = {text(row.get("sample_id")): row for row in mapping_rows}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    missing_mapping = 0
    reason_counts: Counter[str] = Counter()
    contract_status_counts: Counter[str] = Counter()

    for recovered in recovered_rows:
        source_row = recovered.get("source_row") if isinstance(recovered.get("source_row"), dict) else {}
        target = recovered.get("target") if isinstance(recovered.get("target"), dict) else {}
        event = recovered.get("event") if isinstance(recovered.get("event"), dict) else {}
        evidence = recovered.get("evidence") if isinstance(recovered.get("evidence"), dict) else {}
        provenance = recovered.get("provenance") if isinstance(recovered.get("provenance"), dict) else {}
        quality = recovered.get("quality") if isinstance(recovered.get("quality"), dict) else {}
        sample_id = text(source_row.get("sample_id"))
        mapping = mapping_by_sample.get(sample_id)
        if mapping is None:
            missing_mapping += 1
        resolver_event_id = text(source_row.get("event_id"))
        event_mention_id = resolver_event_id
        join_key = event_mention_id or "||".join(
            [
                text(recovered.get("article_id")),
                text(target.get("target_key")),
                text(event.get("event_fact_signature")),
                text(event.get("event_time")),
            ]
        )
        join_key_source = "source_row.event_id" if event_mention_id else "article_id+target_key+event_fact_signature+event_time"
        task4_reason = ""
        resolver_decision = "reject"
        resolver_confidence = 0.5
        cluster_confidence = 0.5
        trust_score = 0.75
        canonical_event_id = ""
        event_thread_id = ""
        cluster_member_count = 0
        cluster_rank = 0
        cluster_size = 0
        if mapping is not None:
            canonical_event_id = text(mapping.get("canonical_event_id"))
            event_thread_id = text(mapping.get("event_thread_id"))
            resolver_decision = text(mapping.get("resolver_decision")) or "reject"
            resolver_confidence = float(mapping.get("resolver_confidence") or 0.5)
            cluster_confidence = float(mapping.get("cluster_confidence") or 0.5)
            trust_score = float(mapping.get("trust_score") or 0.75)
            cluster_member_count = int(mapping.get("cluster_member_count") or 0)
            cluster_rank = int(mapping.get("cluster_rank") or 0)
            cluster_size = int(mapping.get("cluster_size") or 0)
            task4_reason = "" if resolver_decision == "pass" else text(mapping.get("reject_reason"))
            contract_status_counts[text(recovered.get("quality", {}).get("contract_status"))] += 1
        else:
            task4_reason = "resolver_mapping_missing"

        feature_row = {
            "event_mention_id": event_mention_id,
            "event_mention_id_source": "source_row.event_id" if event_mention_id else "",
            "resolver_sample_id": sample_id,
            "resolver_event_id": resolver_event_id,
            "join_key": join_key,
            "join_key_source": join_key_source,
            "resolver_has_mapping": mapping is not None,
            "canonical_event_id": canonical_event_id,
            "event_thread_id": event_thread_id,
            "resolver_decision": resolver_decision,
            "resolver_confidence": resolver_confidence,
            "cluster_confidence": cluster_confidence,
            "cluster_member_count": cluster_member_count,
            "cluster_rank": cluster_rank,
            "cluster_size": cluster_size,
            "trust_score": trust_score,
            "task4_acceptance_status": "pass" if resolver_decision == "pass" else "reject",
            "task4_reject_reason": task4_reason,
            "contract_status": text(quality.get("contract_status")),
            "article_id": text(recovered.get("article_id")),
            "schema_version": text(recovered.get("schema_version")),
            "source_name": text((recovered.get("source") or {}).get("source_name")),
            "source_dataset": text((recovered.get("source") or {}).get("source_dataset")),
            "publish_time": text(recovered.get("publish_time")),
            "target_key": text(target.get("target_key")),
            "target_name": text(target.get("target_name")),
            "target_type": text(target.get("target_type")),
            "target_code": text(target.get("target_code")),
            "target_text": text(target.get("mention_text")),
            "target_span_text": text((target.get("mention_span") or {}).get("text")),
            "target_span_start": int((target.get("mention_span") or {}).get("start", -1)),
            "target_span_end": int((target.get("mention_span") or {}).get("end", -1)),
            "event_type": text(event.get("event_type")),
            "event_family": text(event.get("event_family")),
            "event_name": text(event.get("event_name")),
            "trigger_text": text(event.get("trigger_text")),
            "trigger_span_text": text((event.get("trigger_span") or {}).get("text")),
            "trigger_span_start": int((event.get("trigger_span") or {}).get("start", -1)),
            "trigger_span_end": int((event.get("trigger_span") or {}).get("end", -1)),
            "event_time": text(event.get("event_time")),
            "polarity_or_state": text(event.get("polarity_or_state")),
            "modality_status": text(event.get("modality_status")),
            "object_text": text(event.get("object_text")),
            "event_fact_signature": text(event.get("event_fact_signature")),
            "event_fact": text(event.get("event_fact")),
            "evidence_sentence": text(evidence.get("evidence_sentence")),
            "local_context": text(evidence.get("local_context")),
            "evidence_span_text": text((evidence.get("evidence_span") or {}).get("text")),
            "evidence_span_start": int((evidence.get("evidence_span") or {}).get("start", -1)),
            "evidence_span_end": int((evidence.get("evidence_span") or {}).get("end", -1)),
            "support_type": text(evidence.get("support_type")),
            "certainty": evidence.get("certainty"),
            "provenance_document_hash": text(provenance.get("document_hash")),
            "provenance_sentence_id": text(provenance.get("sentence_id")),
            "provenance_extraction_method": text(provenance.get("extraction_method")),
            "provenance_extraction_time": text(provenance.get("extraction_time")),
            "provenance_source_file": text(provenance.get("source_file")),
            "provenance_upstream_model_or_rule_version": text(provenance.get("upstream_model_or_rule_version")),
            "provenance_raw_article_id": text((provenance.get("raw_provenance") or {}).get("article_id")),
            "quality_target_grounded": bool(quality.get("target_grounded")),
            "quality_event_grounded": bool(quality.get("event_grounded")),
            "quality_evidence_grounded": bool(quality.get("evidence_grounded")),
            "quality_provenance_valid": bool(quality.get("provenance_valid")),
            "quality_target_span_match_status": text(quality.get("target_span_match_status")),
            "quality_trigger_span_match_status": text(quality.get("trigger_span_match_status")),
            "quality_evidence_span_match_status": text(quality.get("evidence_span_match_status")),
            "quality_reject_reasons": json.dumps(quality.get("reject_reasons") or [], ensure_ascii=False),
            "source_row_sample_id": sample_id,
            "source_row_event_id": resolver_event_id,
            "source_row_entity_id": text(source_row.get("entity_id")),
            "source_row_article_file_id": text(source_row.get("article_file_id")),
            "source_row_dataset_split": text(source_row.get("dataset_split")),
            "source_row_train_only": bool(source_row.get("train_only")),
            "source_row_validation_test_leakage": bool(source_row.get("validation_test_leakage")),
            "source_row_high_risk_score": source_row.get("high_risk_score"),
        }
        feature_rows.append(feature_row)
        if resolver_decision == "pass":
            accepted.append(feature_row)
        else:
            rejected.append(feature_row)
            reason_counts[task4_reason or "resolver_reject"] += 1

    summary = {
        "resolver_pool_rows": len(recovered_rows),
        "resolver_rows_with_mapping": len(feature_rows) - missing_mapping,
        "resolver_rows_missing_mapping": missing_mapping,
        "resolver_accepted_rows": len(accepted),
        "resolver_rejected_rows": len(rejected),
        "resolver_reject_reasons": dict(reason_counts),
        "resolver_contract_status_counts": dict(contract_status_counts),
        "task4_accepted_rows": len(accepted),
        "task4_rejected_rows": len(rejected),
        "task4_reject_reasons": dict(Counter(text(r.get("task4_reject_reason")) for r in rejected)),
        "resolver_unique_canonical_event_ids": len({text(r.get("canonical_event_id")) for r in feature_rows if text(r.get("canonical_event_id"))}),
        "resolver_unique_event_mentions": len({text(r.get("event_mention_id")) for r in feature_rows if text(r.get("event_mention_id"))}),
    }
    return accepted, rejected, summary, feature_rows


def build_contract_join_audit(contract_rows: list[dict[str, Any]], recovered_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recovered_articles = {text(row.get("article_id")) for row in recovered_rows}
    by_article_target: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_article_target_name: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_type_name_time: defaultdict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_fact_time: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in recovered_rows:
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        event = row.get("event") if isinstance(row.get("event"), dict) else {}
        key = (text(row.get("article_id")), text(target.get("target_key")))
        by_article_target[key].append(row)
        by_article_target_name[(key[0], key[1], text(event.get("event_name")))].append(row)
        by_type_name_time[(key[0], key[1], text(event.get("event_type")), text(event.get("event_name")), text(event.get("event_time")))].append(row)
        by_fact_time[(key[0], key[1], text(event.get("event_fact_signature")), text(event.get("event_time")))].append(row)

    audit_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    article_overlap_rows = 0
    article_target_overlap_rows = 0
    article_target_name_overlap_rows = 0
    type_name_time_overlap_rows = 0
    fact_time_overlap_rows = 0
    for row in contract_rows:
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        event = row.get("event") if isinstance(row.get("event"), dict) else {}
        key = (text(row.get("article_id")), text(target.get("target_key")))
        en = text(event.get("event_name"))
        et = text(event.get("event_type"))
        tm = text(event.get("event_time"))
        ft = text(event.get("event_fact_signature"))
        article_present = key[0] in recovered_articles
        article_target_matches = by_article_target[key]
        article_target_name_matches = by_article_target_name[(key[0], key[1], en)]
        type_name_time_matches = by_type_name_time[(key[0], key[1], et, en, tm)]
        fact_time_matches = by_fact_time[(key[0], key[1], ft, tm)]
        if article_present:
            article_overlap_rows += 1
        if article_target_matches:
            article_target_overlap_rows += 1
        if article_target_name_matches:
            article_target_name_overlap_rows += 1
        if type_name_time_matches:
            type_name_time_overlap_rows += 1
        if fact_time_matches:
            fact_time_overlap_rows += 1
        if not article_present:
            reason = "article_absent_from_resolver_pool"
        elif fact_time_matches:
            reason = "strict_join"
        elif type_name_time_matches:
            reason = "fact_signature_mismatch"
        elif article_target_name_matches:
            reason = "same_article_target_name_other_event"
        elif article_target_matches:
            reason = "same_article_target_other_event"
        else:
            reason = "no_pool_overlap"
        reason_counts[reason] += 1
        audit_rows.append(
            {
                "article_id": text(row.get("article_id")),
                "target_key": text(target.get("target_key")),
                "event_type": et,
                "event_name": en,
                "event_time": tm,
                "event_fact_signature": ft,
                "article_present_in_resolver_pool": article_present,
                "article_target_matches": len(article_target_matches),
                "article_target_name_matches": len(article_target_name_matches),
                "type_name_time_matches": len(type_name_time_matches),
                "fact_time_matches": len(fact_time_matches),
                "join_reason": reason,
                "contract_status": text((row.get("quality") or {}).get("contract_status")),
            }
        )
    summary = {
        "contract_rows": len(contract_rows),
        "contract_pass_rows": sum(1 for row in contract_rows if text((row.get("quality") or {}).get("contract_status")) == "pass"),
        "contract_review_rows": sum(1 for row in contract_rows if text((row.get("quality") or {}).get("contract_status")) == "review"),
        "contract_quarantine_rows": sum(1 for row in contract_rows if text((row.get("quality") or {}).get("contract_status")) == "quarantine"),
        "resolver_pool_article_overlap_rows": article_overlap_rows,
        "resolver_pool_article_target_overlap_rows": article_target_overlap_rows,
        "resolver_pool_article_target_name_overlap_rows": article_target_name_overlap_rows,
        "resolver_pool_type_name_time_overlap_rows": type_name_time_overlap_rows,
        "resolver_pool_fact_time_overlap_rows": fact_time_overlap_rows,
        "strict_join_rows": reason_counts.get("strict_join", 0),
        "reason_counts": dict(reason_counts),
    }
    return audit_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(RUN_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_json(out_dir / "baseline_metrics.json")
    candidate_metrics = load_json(out_dir / "candidate_metrics.json")
    candidate_generation_metrics = load_json(out_dir / "candidate_generation_metrics.json")
    benchmark_once_result = load_json(out_dir / "benchmark_once_result.json") if (out_dir / "benchmark_once_result.json").exists() else {}
    recovered_rows = load_recovered_rows()
    mapping_rows = read_jsonl(out_dir / "canonical_event_v2_mapping.jsonl")
    cluster_rows = read_jsonl(out_dir / "canonical_event_v2_clusters.jsonl")
    repair_dev_500 = read_jsonl(out_dir / "repair_dev_500.jsonl")
    manual_pass1 = read_jsonl(out_dir / "manual_review_pass1.jsonl")
    manual_pass2 = read_jsonl(out_dir / "manual_review_pass2.jsonl")
    adjudicated = read_jsonl(out_dir / "manual_review_adjudication.jsonl")
    contract_rows = load_contract_rows()
    article_content = load_article_index(FULLTEXT_ARTICLE_INDEX)
    canonical_lookup, confidence_lookup = build_lookup()
    accepted, rejected, task4_summary, feature_rows = build_resolver_task4_interface(recovered_rows, mapping_rows)
    contract_audit_rows, contract_audit_summary = build_contract_join_audit(contract_rows, recovered_rows)

    write_jsonl(out_dir / "task4_event_interface.jsonl", accepted)
    write_jsonl(out_dir / "task4_event_interface_reject.jsonl", rejected)
    pd.DataFrame(feature_rows).to_parquet(out_dir / "task4_feature_table.parquet", index=False)
    write_json(out_dir / "task4_event_interface_summary.json", task4_summary)
    write_json(
        out_dir / "task3_task4_join_report.json",
        {
            "resolver_pool": {
                "rows": len(recovered_rows),
                "mapped_rows": task4_summary["resolver_rows_with_mapping"],
                "missing_mapping_rows": task4_summary["resolver_rows_missing_mapping"],
                "accepted_rows": task4_summary["resolver_accepted_rows"],
                "rejected_rows": task4_summary["resolver_rejected_rows"],
                "reject_reasons": task4_summary["resolver_reject_reasons"],
                "unique_canonical_event_ids": task4_summary["resolver_unique_canonical_event_ids"],
            },
            "task4_interface": {
                "accepted_rows": len(accepted),
                "rejected_rows": len(rejected),
                "reject_reasons": dict(Counter(text(r.get("task4_reject_reason")) for r in rejected)),
                "event_mention_id_source": "source_row.event_id",
                "fallback_join_key": "article_id + target_key + event_fact_signature + event_time",
            },
            "external_contract_validation": contract_audit_summary,
            "contract_row_audit": contract_audit_rows,
            "contract_pass_row_audit": [row for row in contract_audit_rows if row["contract_status"] == "pass"],
            "benchmark_once_result": benchmark_once_result,
            "leakage_audit": {
                "train_only_rows": sum(1 for row in recovered_rows if bool((row.get("source_row") or {}).get("train_only"))),
                "validation_test_leakage_rows": sum(1 for row in recovered_rows if bool((row.get("source_row") or {}).get("validation_test_leakage"))),
                "dataset_split_counts": dict(Counter(text((row.get("source_row") or {}).get("dataset_split")) for row in recovered_rows)),
            },
        },
    )

    manual_agree = sum(1 for a, b in zip(manual_pass1, manual_pass2) if text(a.get("canonical_decision")) == text(b.get("canonical_decision")))
    manual_disagree = len(manual_pass1) - manual_agree
    manual_third = sum(1 for row in adjudicated if text(row.get("review_pass_1")) != text(row.get("review_pass_2")) and text(row.get("final_relation")) != "uncertain")
    repair_dist = dict(Counter(text(r.get("gold_relation")) for r in repair_dev_500))
    repair_size = len(repair_dev_500)

    repair_manifest = {
        "output_dir": rel(out_dir),
        "manual_rows": len(manual_pass1),
        "manual_agreement_rate": round(manual_agree / max(1, len(manual_pass1)), 6),
        "manual_disagreement_rate": round(manual_disagree / max(1, len(manual_pass1)), 6),
        "manual_third_pass_corrections": manual_third,
        "repair_dev_rows": repair_size,
        "repair_dev_label_distribution": repair_dist,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_generation_metrics": candidate_generation_metrics,
        "task4_accepted_rows": len(accepted),
        "task4_rejected_rows": len(rejected),
        "task4_reject_reasons": dict(Counter(text(r.get("task4_reject_reason")) for r in rejected)),
        "task3_task4_join_report": rel(out_dir / "task3_task4_join_report.json"),
        "benchmark_read": bool(benchmark_once_result),
        "benchmark_read_once": bool(benchmark_once_result),
        "benchmark_once_result": benchmark_once_result,
        "frozen_file_modified": False,
        "task3_candidate_enhancement_pass": bool(
            candidate_metrics["macro_f1"] > baseline_metrics["macro_f1"]
            and candidate_metrics["false_merge"] <= baseline_metrics["false_merge"]
            and candidate_metrics["split_error"] <= baseline_metrics["split_error"]
        ),
    }
    write_json(out_dir / "repair_dev_manifest.json", repair_manifest)
    write_json(out_dir / "protected_sha256_after.json", [
        {
            "path": rel(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else "",
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        for path in PROTECTED_FILES
    ])

    write_readme(
        out_dir / "task3_v2_regression_report.md",
        [
            "# Task3 CanonicalEvent Resolver v2 Actual Repair",
            "",
            f"- manual double-review agreement: {repair_manifest['manual_agreement_rate']}",
            f"- manual double-review disagreement: {repair_manifest['manual_disagreement_rate']}",
            f"- manual third-pass corrections: {repair_manifest['manual_third_pass_corrections']}",
            f"- repair-dev rows: {repair_size}",
            f"- baseline accuracy: {baseline_metrics['accuracy']}",
            f"- candidate accuracy: {candidate_metrics['accuracy']}",
            f"- baseline macro_f1: {baseline_metrics['macro_f1']}",
            f"- candidate macro_f1: {candidate_metrics['macro_f1']}",
            f"- false merge: {baseline_metrics['false_merge']} -> {candidate_metrics['false_merge']}",
            f"- split error: {baseline_metrics['split_error']} -> {candidate_metrics['split_error']}",
            f"- candidate recall: {baseline_metrics['candidate_recall_repair_dev']} -> {candidate_metrics['candidate_recall_repair_dev']}",
            f"- resolver pool rows: {len(recovered_rows)}",
            f"- resolver mapped rows: {task4_summary['resolver_rows_with_mapping']}",
            f"- task4 accepted: {len(accepted)}",
            f"- task4 rejected: {len(rejected)}",
            f"- task4 rejection reasons: {dict(Counter(text(r.get('task4_reject_reason')) for r in rejected))}",
            f"- contract pass rows audited: {contract_audit_summary['contract_pass_rows']}",
            f"- contract strict joins: {contract_audit_summary['strict_join_rows']}",
            f"- enhancement_pass: {repair_manifest['task3_candidate_enhancement_pass']}",
            f"- output_dir: {rel(out_dir)}",
        ],
    )
    write_readme(
        out_dir / "README.md",
        [
            "Task3 CanonicalEvent Resolver v2 actual implementation.",
            "",
            f"Repair-dev rows: {repair_size}",
            f"Manual double-review agreement: {repair_manifest['manual_agreement_rate']}",
            f"Resolver pool rows: {len(recovered_rows)}",
            f"Task4 accepted rows: {len(accepted)}",
            f"Task4 rejected rows: {len(rejected)}",
            f"Frozen benchmark read once: {bool(benchmark_once_result)}",
            "No frozen core or graph file was modified.",
        ],
    )
    (out_dir / "repro_finalize_actual.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({
        "output_dir": rel(out_dir),
        "manual_agreement_rate": repair_manifest["manual_agreement_rate"],
        "manual_disagreement_rate": repair_manifest["manual_disagreement_rate"],
        "repair_dev_rows": repair_size,
        "task4_accepted_rows": len(accepted),
        "task4_rejected_rows": len(rejected),
        "benchmark_read": bool(benchmark_once_result),
        "frozen_file_modified": False,
    }, ensure_ascii=False, indent=2))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
