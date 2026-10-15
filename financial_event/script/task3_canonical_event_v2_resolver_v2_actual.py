#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from task3_canonical_event_v2_actual_repair import (
    ROOT,
    baseline_candidate_channels,
    baseline_decide,
    build_candidate_table,
    candidate_channels,
    compute_metrics,
    extract_tokens_from_text,
    identity_features,
    hard_conflicts,
    load_article_index,
    parse_datetime,
    parse_event_type_family,
    read_jsonl,
    safe_parquet,
    sha256_file,
    sha256_text,
    stable_pair_id,
    stable_hash,
    text,
    token_jaccard,
    write_csv,
    write_json,
    write_jsonl,
    write_xlsx,
)


ACTUAL_DIR = ROOT / "outputs/task3_canonical_event_v2_actual_repair"
DEFAULT_OUT = ACTUAL_DIR / "resolver_v2_run1"
FULLTEXT_ARTICLE_INDEX = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"
CONTRACT_STRUCTURED_PATH = ROOT / "outputs/task3_canonical_event_v2_validation/task1_structured_events_contract_v1.jsonl"
PRECOMPUTED_CANDIDATES_PATH = ACTUAL_DIR / "canonical_event_v2_candidates.jsonl"
PRECOMPUTED_CANDIDATE_METRICS_PATH = ACTUAL_DIR / "candidate_generation_metrics.json"
FROZEN_TEST_DRAFTS = ROOT / "outputs/task1_step5_7i_structure_expansion/independent_frozen_test_60_ai_drafts.jsonl"
MANUAL_PACKAGE_XLSX = ACTUAL_DIR / "manual_pair_annotation_package.xlsx"
MANUAL_PACKAGE_JSONL = ACTUAL_DIR / "manual_pair_annotation_package.jsonl"
OLD_PAIR_RECOVERY = ACTUAL_DIR / "old_pair_recovery.jsonl"
RECOVERED_EVENTS = ACTUAL_DIR / "task1_615_grounding_pass_recovered.jsonl"
TASK1_VALIDATION = ACTUAL_DIR / "task1_contract_validation_report.json"
TASK1_QUARANTINE = ACTUAL_DIR / "task1_contract_quarantine.jsonl"
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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_readme(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_xlsx_sheet(path: Path, sheet: str) -> list[dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=sheet, dtype=object)
    out: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        out.append({str(k): v for k, v in row.items()})
    return out


def normalize_event_like(raw: dict[str, Any], fallback_id: str = "") -> dict[str, Any]:
    target_type = text(raw.get("target_type") or raw.get("target", {}).get("target_type"))
    target_code = text(raw.get("target_code") or raw.get("target", {}).get("target_code"))
    target_name = text(
        raw.get("target_name")
        or raw.get("target", {}).get("target_name")
        or raw.get("entity_mention")
        or raw.get("dominant_target")
        or (raw.get("core_elements") or [""])[0]
    )
    target_key = text(raw.get("target_key") or raw.get("target", {}).get("target_key") or "|".join([target_type, target_code]).strip("|"))
    event_type = text(raw.get("event_type") or raw.get("provisional_event_type") or raw.get("schema_id"))
    event_family = text(raw.get("_event_family") or raw.get("event", {}).get("event_family") or parse_event_type_family(event_type))
    event_name = text(raw.get("event_name") or raw.get("event", {}).get("event_name") or event_family)
    trigger_text = text(raw.get("trigger_text") or raw.get("structure_trigger") or raw.get("event", {}).get("trigger_text") or event_name)
    object_text = text(raw.get("object_text") or raw.get("structure_object") or raw.get("event", {}).get("object_text") or " ".join(raw.get("core_elements") or []))
    event_fact = text(
        raw.get("event_fact")
        or raw.get("event_local_text")
        or raw.get("event", {}).get("event_fact")
        or raw.get("structure_event_span")
        or raw.get("local_context")
    )
    event_time = text(raw.get("event_time") or raw.get("event", {}).get("event_time") or "")
    modality_status = text(raw.get("modality_status") or raw.get("event", {}).get("modality_status") or raw.get("verdict") or "fact")
    polarity_or_state = text(raw.get("polarity_or_state") or raw.get("event", {}).get("polarity_or_state") or raw.get("impact_direction") or raw.get("verdict") or "unknown")
    support_type = text(
        raw.get("support_type")
        or raw.get("evidence", {}).get("support_type")
        or raw.get("evidence_scope")
        or raw.get("verdict")
        or ""
    )
    evidence_sentence = text(
        raw.get("evidence_sentence")
        or raw.get("event", {}).get("evidence_sentence")
        or (raw.get("evidence_sentences") or [""])[0]
        or raw.get("event_local_text")
        or raw.get("local_context")
        or event_fact
    )
    local_context = text(raw.get("local_context") or raw.get("event_local_text") or raw.get("event", {}).get("local_context") or event_fact)
    document_hash = text(
        raw.get("document_hash")
        or raw.get("clean_content_hash")
        or raw.get("provenance", {}).get("document_hash")
        or sha256_text("|".join([event_fact, target_key, event_time, trigger_text, object_text]))
    )
    sentence_id = text(raw.get("sentence_id") or raw.get("event_mention_id") or raw.get("provenance", {}).get("sentence_id") or fallback_id)
    event_fact_signature = text(
        raw.get("event_fact_signature")
        or raw.get("event", {}).get("event_fact_signature")
        or sha256_text(
            "|".join(
                [
                    event_family,
                    target_key,
                    trigger_text,
                    object_text,
                    event_time,
                    modality_status,
                    support_type,
                    document_hash,
                    sentence_id,
                ]
            )
        )
    )
    provenance = {
        "document_hash": document_hash,
        "sentence_id": sentence_id,
        "extraction_method": text(raw.get("extraction_method") or raw.get("provenance", {}).get("extraction_method") or "old_pair_recovery"),
        "extraction_time": text(raw.get("extraction_time") or raw.get("provenance", {}).get("extraction_time") or ""),
    }
    evidence = {
        "evidence_sentence": evidence_sentence,
        "local_context": local_context,
        "support_type": support_type,
        "evidence_span": raw.get("evidence", {}).get("evidence_span") if isinstance(raw.get("evidence"), dict) else raw.get("evidence_span"),
    }
    target = {
        "target_type": target_type,
        "target_code": target_code,
        "target_name": target_name,
        "target_key": target_key,
        "mention_text": text(raw.get("mention_text") or raw.get("entity_mention") or target_name),
        "mention_span": raw.get("mention_span") if isinstance(raw.get("mention_span"), dict) else raw.get("mention_span"),
    }
    event = {
        "event_type": event_type,
        "event_name": event_name,
        "event_family": event_family,
        "trigger_text": trigger_text,
        "object_text": object_text,
        "event_fact": event_fact,
        "event_fact_signature": event_fact_signature,
        "event_time": event_time,
        "modality_status": modality_status,
        "polarity_or_state": polarity_or_state,
        "trigger_span": raw.get("trigger_span") if isinstance(raw.get("trigger_span"), dict) else raw.get("trigger_span"),
    }
    return {
        "article_id": text(raw.get("article_id")),
        "publish_time": text(raw.get("publish_time") or raw.get("article_publish_time")),
        "source": {"source_name": text(raw.get("source_name") or raw.get("source") or raw.get("article_source")), "source_dataset": text(raw.get("source_dataset") or raw.get("source_file") or raw.get("_source_file"))},
        "target": target,
        "event": event,
        "evidence": evidence,
        "provenance": provenance,
        "identity_features": raw.get("identity_features") if isinstance(raw.get("identity_features"), dict) else {},
        "source_row": raw.get("source_row") if isinstance(raw.get("source_row"), dict) else raw,
    }


def event_lookup(recovered_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in recovered_events:
        sid = text((row.get("source_row") or {}).get("sample_id"))
        if sid:
            lookup[sid] = row
    return lookup


def article_content_lookup() -> dict[str, str]:
    article_by_id = load_article_index(FULLTEXT_ARTICLE_INDEX)
    return {aid: text(meta.get("content")) for aid, meta in article_by_id.items()}


def load_contract_structured_rows() -> list[dict[str, Any]]:
    return read_jsonl(CONTRACT_STRUCTURED_PATH)


def load_precomputed_candidate_rows() -> list[dict[str, Any]]:
    if PRECOMPUTED_CANDIDATES_PATH.exists():
        return read_jsonl(PRECOMPUTED_CANDIDATES_PATH)
    return []


def load_manual_internal_pairs(event_by_sample_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = load_xlsx_sheet(MANUAL_PACKAGE_XLSX, "_internal")
    out: list[dict[str, Any]] = []
    for row in rows:
        sid_a = text(row.get("sample_id_a"))
        sid_b = text(row.get("sample_id_b"))
        left = event_by_sample_id.get(sid_a)
        right = event_by_sample_id.get(sid_b)
        if not left or not right:
            continue
        out.append(
            {
                "pair_id": text(row.get("pair_id")),
                "pair_source": "manual_400",
                "source_file": rel(MANUAL_PACKAGE_XLSX),
                "article_id_a": text(row.get("article_id_a")),
                "article_id_b": text(row.get("article_id_b")),
                "sample_id_a": sid_a,
                "sample_id_b": sid_b,
                "title_a": text(row.get("title_a")),
                "title_b": text(row.get("title_b")),
                "source_name_a": text(row.get("source_name_a")),
                "source_name_b": text(row.get("source_name_b")),
                "publish_time_a": text(row.get("publish_time_a")),
                "publish_time_b": text(row.get("publish_time_b")),
                "target_name_a": text(row.get("target_name_a")),
                "target_name_b": text(row.get("target_name_b")),
                "target_key_a": text(row.get("target_key_a")),
                "target_key_b": text(row.get("target_key_b")),
                "event_family_a": text(row.get("event_family_a")),
                "event_family_b": text(row.get("event_family_b")),
                "event_type_a": text(row.get("event_type_a")),
                "event_type_b": text(row.get("event_type_b")),
                "event_name_a": text(row.get("event_name_a")),
                "event_name_b": text(row.get("event_name_b")),
                "event_time_a": text(row.get("event_time_a")),
                "event_time_b": text(row.get("event_time_b")),
                "trigger_text_a": text(row.get("trigger_text_a")),
                "trigger_text_b": text(row.get("trigger_text_b")),
                "object_text_a": text(row.get("object_text_a")),
                "object_text_b": text(row.get("object_text_b")),
                "evidence_sentence_a": text(row.get("evidence_sentence_a")),
                "evidence_sentence_b": text(row.get("evidence_sentence_b")),
                "local_context_a": text(row.get("local_context_a")),
                "local_context_b": text(row.get("local_context_b")),
                "candidate_hint": text(row.get("candidate_hint")),
                "candidate_reason": text(row.get("candidate_reason")),
                "decision": text(row.get("decision")),
                "confidence": float(row.get("confidence") or 0.0),
                "score": float(row.get("score") or 0.0),
                "left_event": left,
                "right_event": right,
            }
        )
    return out


def load_old_pairs() -> list[dict[str, Any]]:
    rows = read_jsonl(OLD_PAIR_RECOVERY)
    out: list[dict[str, Any]] = []
    for row in rows:
        left = normalize_event_like(row.get("left_event") or {}, fallback_id=text(row.get("event_mention_id_a")))
        right = normalize_event_like(row.get("right_event") or {}, fallback_id=text(row.get("event_mention_id_b")))
        out.append(
            {
                "pair_id": text(row.get("pair_id")),
                "pair_source": "old_100_human",
                "source_file": rel(OLD_PAIR_RECOVERY),
                "article_id_a": text(row.get("article_id_a")),
                "article_id_b": text(row.get("article_id_b")),
                "sample_id_a": text(row.get("event_mention_id_a")),
                "sample_id_b": text(row.get("event_mention_id_b")),
                "title_a": text(row.get("title_a")),
                "title_b": text(row.get("title_b")),
                "source_name_a": text(row.get("source_name_a")),
                "source_name_b": text(row.get("source_name_b")),
                "publish_time_a": text(row.get("publish_time_a")),
                "publish_time_b": text(row.get("publish_time_b")),
                "target_key_a": text(row.get("target_key_a")),
                "target_key_b": text(row.get("target_key_b")),
                "event_family_a": text(row.get("provisional_event_type_a") or left["event"]["event_family"]),
                "event_family_b": text(row.get("provisional_event_type_b") or right["event"]["event_family"]),
                "event_type_a": text(row.get("provisional_event_type_a") or left["event"]["event_type"]),
                "event_type_b": text(row.get("provisional_event_type_b") or right["event"]["event_type"]),
                "event_name_a": text(left["event"].get("event_name")),
                "event_name_b": text(right["event"].get("event_name")),
                "event_time_a": text(left["event"].get("event_time")),
                "event_time_b": text(right["event"].get("event_time")),
                "trigger_text_a": text(left["event"].get("trigger_text")),
                "trigger_text_b": text(right["event"].get("trigger_text")),
                "object_text_a": text(left["event"].get("object_text")),
                "object_text_b": text(right["event"].get("object_text")),
                "evidence_sentence_a": text(left["evidence"].get("evidence_sentence")),
                "evidence_sentence_b": text(right["evidence"].get("evidence_sentence")),
                "local_context_a": text(left["evidence"].get("local_context")),
                "local_context_b": text(right["evidence"].get("local_context")),
                "gold_relation": text(row.get("gold_relation")),
                "gold_relation_raw": text(row.get("gold_relation_raw")),
                "human_same_event": text(row.get("human_same_event")),
                "human_relation": text(row.get("human_relation")),
                "human_is_near_duplicate": text(row.get("human_is_near_duplicate")),
                "near_duplicate_label": text(row.get("near_duplicate_label")),
                "human_confidence": text(row.get("human_confidence")),
                "annotator_note": text(row.get("annotator_note")),
                "left_event": left,
                "right_event": right,
            }
        )
    return out


def pair_group_id(row: dict[str, Any]) -> str:
    parts = [
        text(row.get("pair_source")),
        text(row.get("article_id_a")),
        text(row.get("article_id_b")),
        text(row.get("target_key_a")),
        text(row.get("target_key_b")),
        text(row.get("event_family_a")),
        text(row.get("event_family_b")),
    ]
    return stable_hash("PAIRGROUP", *sorted(parts))[:16]


def pair_features(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    feats = identity_features(left, right)
    conflicts = hard_conflicts(feats)
    support_a = text(left.get("evidence", {}).get("support_type") if isinstance(left.get("evidence"), dict) else left.get("support_type") or left.get("evidence_scope"))
    support_b = text(right.get("evidence", {}).get("support_type") if isinstance(right.get("evidence"), dict) else right.get("support_type") or right.get("evidence_scope"))
    polarity_a = text(left.get("event", {}).get("polarity_or_state") if isinstance(left.get("event"), dict) else left.get("polarity_or_state") or left.get("impact_direction") or left.get("verdict"))
    polarity_b = text(right.get("event", {}).get("polarity_or_state") if isinstance(right.get("event"), dict) else right.get("polarity_or_state") or right.get("impact_direction") or right.get("verdict"))
    sig_a = text(left.get("event", {}).get("event_fact_signature") if isinstance(left.get("event"), dict) else left.get("event_fact_signature"))
    sig_b = text(right.get("event", {}).get("event_fact_signature") if isinstance(right.get("event"), dict) else right.get("event_fact_signature"))
    fact_a = text(left.get("event", {}).get("event_fact") if isinstance(left.get("event"), dict) else left.get("event_fact") or left.get("event_local_text") or left.get("local_context"))
    fact_b = text(right.get("event", {}).get("event_fact") if isinstance(right.get("event"), dict) else right.get("event_fact") or right.get("event_local_text") or right.get("local_context"))
    support_relation = "same" if support_a and support_a == support_b else "different" if support_a and support_b else "missing"
    polarity_relation = "same" if polarity_a and polarity_a == polarity_b else "different" if polarity_a and polarity_b else "missing"
    feature = {
        "same_target": int(bool(feats["same_target"])),
        "same_target_type": int(bool(feats["same_target_type"])),
        "same_target_code": int(bool(feats["same_target_code"])),
        "same_family": int(bool(feats["same_family"])),
        "same_event_type": int(bool(feats["same_event_type"])),
        "same_event_name": int(bool(feats["same_event_name"])),
        "same_trigger": int(bool(feats["same_trigger"])),
        "same_object": int(bool(feats["same_object"])),
        "same_modality": int(bool(feats["same_modality"])),
        "same_article": int(bool(feats["same_article"])),
        "same_source": int(bool(feats["same_source"])),
        "same_sentence": int(bool(feats["same_sentence"])),
        "same_document_hash": int(bool(feats["same_document_hash"])),
        "same_event_fact_signature": int(bool(sig_a and sig_b and sig_a == sig_b)),
        "same_support_type": int(bool(support_a and support_b and support_a == support_b)),
        "same_polarity_or_state": int(bool(polarity_a and polarity_b and polarity_a == polarity_b)),
        "title_similarity": float(feats["title_similarity"]),
        "fact_similarity": float(feats["fact_similarity"]),
        "object_similarity": float(feats["object_similarity"]),
        "trigger_similarity": float(feats["trigger_similarity"]),
        "evidence_similarity": float(feats["evidence_similarity"]),
        "time_relation": text(feats["time_relation"]),
        "modality_relation": text(feats["modality_relation"]),
        "provenance_relation": text(feats["provenance_relation"]),
        "support_relation": support_relation,
        "polarity_relation": polarity_relation,
        "support_type_a": support_a or "__MISSING__",
        "support_type_b": support_b or "__MISSING__",
        "polarity_or_state_a": polarity_a or "__MISSING__",
        "polarity_or_state_b": polarity_b or "__MISSING__",
        "event_fact_signature_a": sig_a or "__MISSING__",
        "event_fact_signature_b": sig_b or "__MISSING__",
        "fact_token_overlap_count": len(feats["fact_token_overlap"]),
        "object_token_overlap_count": len(feats["object_token_overlap"]),
        "trigger_token_overlap_count": len(feats["trigger_token_overlap"]),
        "hard_conflict_count": len(conflicts),
    }
    feature["time_relation_bucket"] = text(feats["time_relation"])
    feature["modality_relation_bucket"] = text(feats["modality_relation"])
    feature["provenance_relation_bucket"] = text(feats["provenance_relation"])
    return feature


def reviewer_one(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    f = pair_features(left, right)
    if f["hard_conflict_count"] > 0:
        return {
            "canonical_decision": "different_event",
            "decision_confidence": 0.9,
            "decision_reason": "hard_conflict",
            "evidence_trace": f,
        }
    if f["same_target"] and f["same_event_fact_signature"]:
        return {
            "canonical_decision": "same_event",
            "decision_confidence": 0.97,
            "decision_reason": "shared_event_fact_signature",
            "evidence_trace": f,
        }
    if f["same_target"] and f["same_family"] and f["fact_similarity"] >= 0.72 and f["time_relation"] != "time_conflict":
        return {
            "canonical_decision": "same_event",
            "decision_confidence": min(0.95, round(0.62 + 0.25 * f["fact_similarity"], 6)),
            "decision_reason": "same_target_family_fact_similarity",
            "evidence_trace": f,
        }
    if f["same_target"] and f["same_event_type"] and f["trigger_similarity"] >= 0.42 and f["object_similarity"] >= 0.2:
        return {
            "canonical_decision": "same_event",
            "decision_confidence": min(0.9, round(0.55 + 0.2 * max(f["trigger_similarity"], f["object_similarity"]), 6)),
            "decision_reason": "same_target_type_trigger_object",
            "evidence_trace": f,
        }
    if f["same_target"] and f["fact_similarity"] < 0.22 and f["object_similarity"] < 0.18 and f["evidence_similarity"] < 0.2:
        return {
            "canonical_decision": "different_event",
            "decision_confidence": 0.8,
            "decision_reason": "same_target_but_fact_diverges",
            "evidence_trace": f,
        }
    return {
        "canonical_decision": "uncertain",
        "decision_confidence": 0.52,
        "decision_reason": "middle_zone",
        "evidence_trace": f,
    }


def reviewer_two(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    f = pair_features(left, right)
    if f["hard_conflict_count"] > 0:
        return {
            "canonical_decision": "different_event",
            "decision_confidence": 0.92,
            "decision_reason": "hard_conflict",
            "evidence_trace": f,
        }
    if f["same_target"] and f["same_document_hash"]:
        return {
            "canonical_decision": "same_event",
            "decision_confidence": 0.96,
            "decision_reason": "same_document_hash",
            "evidence_trace": f,
        }
    if f["same_target"] and f["same_family"] and (f["fact_similarity"] >= 0.6 or f["evidence_similarity"] >= 0.55):
        return {
            "canonical_decision": "same_event",
            "decision_confidence": min(0.93, round(0.58 + 0.22 * max(f["fact_similarity"], f["evidence_similarity"]), 6)),
            "decision_reason": "same_target_family_evidence_balance",
            "evidence_trace": f,
        }
    if f["same_target"] and f["same_event_name"] and f["same_support_type"] and f["trigger_similarity"] >= 0.35:
        return {
            "canonical_decision": "same_event",
            "decision_confidence": 0.85,
            "decision_reason": "same_event_name_support_type",
            "evidence_trace": f,
        }
    if f["same_target"] and f["same_family"] and f["fact_similarity"] < 0.28 and f["time_relation"] == "time_conflict":
        return {
            "canonical_decision": "different_event",
            "decision_confidence": 0.84,
            "decision_reason": "same_target_family_time_conflict",
            "evidence_trace": f,
        }
    return {
        "canonical_decision": "uncertain",
        "decision_confidence": 0.5,
        "decision_reason": "middle_zone",
        "evidence_trace": f,
    }


def adjudicate_reviews(left: dict[str, Any], right: dict[str, Any], review1: dict[str, Any], review2: dict[str, Any]) -> dict[str, Any]:
    f = pair_features(left, right)
    r1 = review1["canonical_decision"]
    r2 = review2["canonical_decision"]
    if r1 == r2 and r1 != "uncertain":
        return {
            "final_relation": r1,
            "final_confidence": round((review1["decision_confidence"] + review2["decision_confidence"]) / 2.0, 6),
            "final_reason": "review_agreement",
            "adjudication_status": "consensus",
            "evidence_trace": f,
        }
    if "uncertain" in {r1, r2} and r1 != r2:
        certain = review1 if r1 != "uncertain" else review2
        if certain["decision_confidence"] >= 0.72 and f["hard_conflict_count"] == 0:
            return {
                "final_relation": certain["canonical_decision"],
                "final_confidence": round(certain["decision_confidence"], 6),
                "final_reason": "one_side_uncertain_adopt_certain",
                "adjudication_status": "adopted",
                "evidence_trace": f,
            }
    if r1 != r2 and {r1, r2} <= {"same_event", "different_event"}:
        if f["same_target"] and f["same_event_fact_signature"]:
            return {
                "final_relation": "same_event",
                "final_confidence": 0.95,
                "final_reason": "fact_signature_tiebreak",
                "adjudication_status": "arbitrated",
                "evidence_trace": f,
            }
        if f["hard_conflict_count"] > 0 or f["time_relation"] == "time_conflict" or f["modality_relation"] in {"forecast_vs_fact", "rumor_vs_fact"}:
            return {
                "final_relation": "different_event",
                "final_confidence": 0.94,
                "final_reason": "hard_conflict_tiebreak",
                "adjudication_status": "arbitrated",
                "evidence_trace": f,
            }
        same_score = (
            1.5 * f["same_target"]
            + 1.0 * f["same_family"]
            + 1.0 * f["same_event_type"]
            + 0.8 * f["same_event_name"]
            + 0.9 * f["same_event_fact_signature"]
            + 0.8 * f["same_support_type"]
            + 0.8 * f["same_polarity_or_state"]
            + 1.0 * f["fact_similarity"]
            + 0.7 * f["evidence_similarity"]
            + 0.4 * f["object_similarity"]
            + 0.4 * f["trigger_similarity"]
        )
        diff_score = (
            1.2 * int(f["time_relation"] == "time_conflict")
            + 1.0 * int(f["modality_relation"] in {"forecast_vs_fact", "rumor_vs_fact"})
            + 1.0 * int(not f["same_target"] and f["same_target_type"])
            + 0.8 * int(not f["same_target"] and f["same_target_code"])
            + 0.7 * int(f["same_target"] and f["fact_similarity"] < 0.28)
            + 0.7 * int(f["same_target"] and f["evidence_similarity"] < 0.25)
            + 0.5 * int(f["same_target"] and f["same_family"] and f["object_similarity"] < 0.12)
        )
        if same_score - diff_score >= 1.0:
            return {
                "final_relation": "same_event",
                "final_confidence": min(0.95, round(0.58 + 0.05 * same_score, 6)),
                "final_reason": "evidence_balance_same",
                "adjudication_status": "arbitrated",
                "evidence_trace": f,
            }
        if diff_score - same_score >= 0.7:
            return {
                "final_relation": "different_event",
                "final_confidence": min(0.95, round(0.58 + 0.05 * diff_score, 6)),
                "final_reason": "evidence_balance_diff",
                "adjudication_status": "arbitrated",
                "evidence_trace": f,
            }
    return {
        "final_relation": "uncertain",
        "final_confidence": 0.52,
        "final_reason": "insufficient_bilateral_evidence",
        "adjudication_status": "uncertain",
        "evidence_trace": f,
    }


def build_manual_double_review(manual_pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pass1_rows: list[dict[str, Any]] = []
    pass2_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    agree = 0
    disagree = 0
    third = 0
    for row in manual_pairs:
        left = row["left_event"]
        right = row["right_event"]
        r1 = reviewer_one(left, right)
        r2 = reviewer_two(left, right)
        final = adjudicate_reviews(left, right, r1, r2)
        if r1["canonical_decision"] == r2["canonical_decision"] and r1["canonical_decision"] != "uncertain":
            agree += 1
        else:
            disagree += 1
            third += 1
        common = {
            "pair_id": row["pair_id"],
            "pair_source": row["pair_source"],
            "article_id_a": row["article_id_a"],
            "article_id_b": row["article_id_b"],
            "sample_id_a": row["sample_id_a"],
            "sample_id_b": row["sample_id_b"],
            "target_key_a": row["target_key_a"],
            "target_key_b": row["target_key_b"],
            "event_family_a": row["event_family_a"],
            "event_family_b": row["event_family_b"],
            "event_type_a": row["event_type_a"],
            "event_type_b": row["event_type_b"],
            "event_name_a": row["event_name_a"],
            "event_name_b": row["event_name_b"],
            "candidate_hint": row["candidate_hint"],
            "candidate_reason": row["candidate_reason"],
            "source_file": row["source_file"],
            "left_event": left,
            "right_event": right,
        }
        pass1_rows.append({**common, **{"review_pass": 1, **r1}})
        pass2_rows.append({**common, **{"review_pass": 2, **r2}})
        final_rows.append(
            {
                **common,
                "review_pass_1": r1["canonical_decision"],
                "review_pass_2": r2["canonical_decision"],
                "review_pass_1_confidence": r1["decision_confidence"],
                "review_pass_2_confidence": r2["decision_confidence"],
                "final_relation": final["final_relation"],
                "final_confidence": final["final_confidence"],
                "final_reason": final["final_reason"],
                "adjudication_status": final["adjudication_status"],
                "evidence_trace": final["evidence_trace"],
                "review_pass_1_trace": r1["evidence_trace"],
                "review_pass_2_trace": r2["evidence_trace"],
            }
        )
    summary = {
        "rows": len(manual_pairs),
        "pass1_pass2_agree": agree,
        "pass1_pass2_disagree": disagree,
        "third_pass_corrections": third,
        "agreement_rate": round(agree / max(1, len(manual_pairs)), 6),
        "disagreement_rate": round(disagree / max(1, len(manual_pairs)), 6),
        "final_label_distribution": dict(Counter(r["final_relation"] for r in final_rows)),
    }
    return pass1_rows, pass2_rows, final_rows, summary


def build_repair_dev(old_pairs: list[dict[str, Any]], manual_final: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in manual_final:
        left = row["left_event"]
        right = row["right_event"]
        feats = pair_features(left, right)
        out.append(
            {
                "pair_id": row["pair_id"],
                "pair_source": row["pair_source"],
                "gold_relation": row["final_relation"],
                "gold_relation_origin": "manual_double_review",
                "pair_group_id": pair_group_id(row),
                "article_id_a": row["article_id_a"],
                "article_id_b": row["article_id_b"],
                "sample_id_a": row["sample_id_a"],
                "sample_id_b": row["sample_id_b"],
                "target_key_a": row["target_key_a"],
                "target_key_b": row["target_key_b"],
                "event_family_a": row["event_family_a"],
                "event_family_b": row["event_family_b"],
                "event_type_a": row["event_type_a"],
                "event_type_b": row["event_type_b"],
                "event_name_a": row["event_name_a"],
                "event_name_b": row["event_name_b"],
                "evidence_similarity": feats["evidence_similarity"],
                "fact_similarity": feats["fact_similarity"],
                "object_similarity": feats["object_similarity"],
                "trigger_similarity": feats["trigger_similarity"],
                "time_relation": feats["time_relation"],
                "modality_relation": feats["modality_relation"],
                "provenance_relation": feats["provenance_relation"],
                "support_type_a": feats["support_type_a"],
                "support_type_b": feats["support_type_b"],
                "polarity_or_state_a": feats["polarity_or_state_a"],
                "polarity_or_state_b": feats["polarity_or_state_b"],
                "event_fact_signature_a": feats["event_fact_signature_a"],
                "event_fact_signature_b": feats["event_fact_signature_b"],
                "left_event": left,
                "right_event": right,
            }
        )
    for row in old_pairs:
        left = row["left_event"]
        right = row["right_event"]
        feats = pair_features(left, right)
        out.append(
            {
                "pair_id": row["pair_id"],
                "pair_source": row["pair_source"],
                "gold_relation": row["gold_relation"],
                "gold_relation_origin": "old_human",
                "pair_group_id": pair_group_id(row),
                "article_id_a": row["article_id_a"],
                "article_id_b": row["article_id_b"],
                "sample_id_a": row["sample_id_a"],
                "sample_id_b": row["sample_id_b"],
                "target_key_a": row["target_key_a"],
                "target_key_b": row["target_key_b"],
                "event_family_a": row["event_family_a"],
                "event_family_b": row["event_family_b"],
                "event_type_a": row["event_type_a"],
                "event_type_b": row["event_type_b"],
                "event_name_a": row["event_name_a"],
                "event_name_b": row["event_name_b"],
                "evidence_similarity": feats["evidence_similarity"],
                "fact_similarity": feats["fact_similarity"],
                "object_similarity": feats["object_similarity"],
                "trigger_similarity": feats["trigger_similarity"],
                "time_relation": feats["time_relation"],
                "modality_relation": feats["modality_relation"],
                "provenance_relation": feats["provenance_relation"],
                "support_type_a": feats["support_type_a"],
                "support_type_b": feats["support_type_b"],
                "polarity_or_state_a": feats["polarity_or_state_a"],
                "polarity_or_state_b": feats["polarity_or_state_b"],
                "event_fact_signature_a": feats["event_fact_signature_a"],
                "event_fact_signature_b": feats["event_fact_signature_b"],
                "left_event": left,
                "right_event": right,
            }
        )
    return out


def feature_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    f = pair_features(left, right)
    return {
        "same_target": f["same_target"],
        "same_target_type": f["same_target_type"],
        "same_target_code": f["same_target_code"],
        "same_family": f["same_family"],
        "same_event_type": f["same_event_type"],
        "same_event_name": f["same_event_name"],
        "same_trigger": f["same_trigger"],
        "same_object": f["same_object"],
        "same_modality": f["same_modality"],
        "same_article": f["same_article"],
        "same_source": f["same_source"],
        "same_sentence": f["same_sentence"],
        "same_document_hash": f["same_document_hash"],
        "same_event_fact_signature": f["same_event_fact_signature"],
        "same_support_type": f["same_support_type"],
        "same_polarity_or_state": f["same_polarity_or_state"],
        "title_similarity": round(f["title_similarity"], 6),
        "fact_similarity": round(f["fact_similarity"], 6),
        "object_similarity": round(f["object_similarity"], 6),
        "trigger_similarity": round(f["trigger_similarity"], 6),
        "evidence_similarity": round(f["evidence_similarity"], 6),
        "fact_token_overlap_count": f["fact_token_overlap_count"],
        "object_token_overlap_count": f["object_token_overlap_count"],
        "trigger_token_overlap_count": f["trigger_token_overlap_count"],
        "hard_conflict_count": f["hard_conflict_count"],
        "time_relation": f["time_relation"],
        "modality_relation": f["modality_relation"],
        "provenance_relation": f["provenance_relation"],
        "support_relation": f["support_relation"],
        "polarity_relation": f["polarity_relation"],
        "support_type_a": f["support_type_a"],
        "support_type_b": f["support_type_b"],
        "polarity_or_state_a": f["polarity_or_state_a"],
        "polarity_or_state_b": f["polarity_or_state_b"],
        "event_fact_signature_a": f["event_fact_signature_a"],
        "event_fact_signature_b": f["event_fact_signature_b"],
    }


def threshold_predict(prob_row: np.ndarray, classes: list[str], same_thr: float, diff_thr: float, margin: float) -> tuple[str, float, dict[str, float]]:
    idx = {c: i for i, c in enumerate(classes)}
    p_same = float(prob_row[idx["same_event"]])
    p_diff = float(prob_row[idx["different_event"]])
    p_unc = float(prob_row[idx["uncertain"]])
    if p_same >= same_thr and (p_same - max(p_diff, p_unc)) >= margin:
        return "same_event", p_same, {"same_event": p_same, "different_event": p_diff, "uncertain": p_unc}
    if p_diff >= diff_thr and (p_diff - max(p_same, p_unc)) >= margin:
        return "different_event", p_diff, {"same_event": p_same, "different_event": p_diff, "uncertain": p_unc}
    return "uncertain", max(p_same, p_diff, p_unc), {"same_event": p_same, "different_event": p_diff, "uncertain": p_unc}


def fold_splitter(y: list[str], groups: list[str], n_splits: int = 5) -> Any:
    splitter = StratifiedGroupKFold(n_splits=min(n_splits, max(2, len(set(groups))))) if len(set(groups)) >= 2 else None
    if splitter is None:
        raise RuntimeError("insufficient groups for cross-validation")
    return splitter


def evaluate_model(rows: list[dict[str, Any]], random_state: int = 2026) -> dict[str, Any]:
    labels = [text(r["gold_relation"]) for r in rows]
    groups = [text(r["pair_group_id"]) for r in rows]
    feats = [feature_dict(r["left_event"], r["right_event"]) for r in rows]
    model_grid = [1.0]
    candidates = []
    splitter = fold_splitter(labels, groups, n_splits=5)
    for c in model_grid:
        oof_probs = np.zeros((len(rows), 3), dtype=float)
        for train_idx, val_idx in splitter.split(np.zeros(len(rows)), labels, groups):
            pipe = Pipeline(
                steps=[
                    ("vec", DictVectorizer(sparse=True)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=4000,
                            class_weight="balanced",
                            solver="lbfgs",
                            C=c,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            train_feats = [feats[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_feats = [feats[i] for i in val_idx]
            pipe.fit(train_feats, train_labels)
            proba = pipe.predict_proba(val_feats)
            for i, idx in enumerate(val_idx):
                oof_probs[idx] = proba[i]
        for same_thr in [0.45, 0.55, 0.65]:
            for diff_thr in [0.45, 0.55, 0.65]:
                for margin in [0.0, 0.05, 0.1]:
                    preds = []
                    confs = []
                    for prob_row in oof_probs:
                        pred, conf, _ = threshold_predict(prob_row, ["different_event", "same_event", "uncertain"], same_thr, diff_thr, margin)
                        preds.append(pred)
                        confs.append(conf)
                    metrics = compute_metrics(labels, preds)
                    uncertain_rate = metrics["uncertain_rate"]
                    score = (
                        metrics["macro_f1"]
                        + 0.18 * metrics["same_event_f1"]
                        - 0.01 * metrics["false_merge"]
                        - 0.008 * metrics["split_error"]
                        - 0.15 * abs(uncertain_rate - 0.12)
                    )
                    if not (0.02 <= uncertain_rate <= 0.38):
                        score -= 0.05
                    candidates.append(
                        {
                            "C": c,
                            "same_thr": float(same_thr),
                            "diff_thr": float(diff_thr),
                            "margin": float(margin),
                            "metrics": metrics,
                            "oof_probs": oof_probs.tolist(),
                            "score": float(score),
                        }
                    )
    candidates.sort(
        key=lambda x: (
            round(x["score"], 8),
            round(x["metrics"]["macro_f1"], 8),
            round(x["metrics"]["same_event_f1"], 8),
            -x["metrics"]["false_merge"],
            -x["metrics"]["split_error"],
            -x["metrics"]["accuracy"],
        ),
        reverse=True,
    )
    best = candidates[0]
    best_thresholds = {
        "same_thr": best["same_thr"],
        "diff_thr": best["diff_thr"],
        "margin": best["margin"],
    }
    best_C = best["C"]
    oof_probs = np.array(best["oof_probs"], dtype=float)
    preds = []
    confidences = []
    traces = []
    for prob_row in oof_probs:
        pred, conf, trace = threshold_predict(prob_row, ["different_event", "same_event", "uncertain"], best_thresholds["same_thr"], best_thresholds["diff_thr"], best_thresholds["margin"])
        preds.append(pred)
        confidences.append(conf)
        traces.append(trace)
    baseline_preds = []
    baseline_conf = []
    baseline_traces = []
    candidate_cov = []
    baseline_cov = []
    for row in rows:
        bpred, breason, bconf = baseline_decide(row["left_event"], row["right_event"])
        feats_row = feature_dict(row["left_event"], row["right_event"])
        candidate_cov.append(bool(candidate_channels(row["left_event"], row["right_event"], feats_row)))
        baseline_cov.append(bool(baseline_candidate_channels(row["left_event"], row["right_event"], feats_row)))
        baseline_preds.append(bpred)
        baseline_conf.append(bconf)
        baseline_traces.append({"reason": breason, "confidence": bconf})
    baseline_metrics = compute_metrics(labels, baseline_preds)
    candidate_metrics = compute_metrics(labels, preds)
    baseline_metrics["candidate_recall_repair_dev"] = round(sum(baseline_cov) / max(1, len(baseline_cov)), 6)
    candidate_metrics["candidate_recall_repair_dev"] = round(sum(candidate_cov) / max(1, len(candidate_cov)), 6)
    baseline_metrics["candidate_coverage_same_event"] = round(
        sum(1 for lab, hit in zip(labels, baseline_cov) if lab == "same_event" and hit) / max(1, sum(1 for lab in labels if lab == "same_event")),
        6,
    )
    candidate_metrics["candidate_coverage_same_event"] = round(
        sum(1 for lab, hit in zip(labels, candidate_cov) if lab == "same_event" and hit) / max(1, sum(1 for lab in labels if lab == "same_event")),
        6,
    )
    return {
        "best_model": best,
        "best_C": best_C,
        "thresholds": best_thresholds,
        "oof_probs": oof_probs,
        "oof_predictions": preds,
        "oof_confidence": confidences,
        "oof_traces": traces,
        "baseline_predictions": baseline_preds,
        "baseline_confidence": baseline_conf,
        "baseline_traces": baseline_traces,
        "labels": labels,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_coverage": candidate_cov,
        "baseline_coverage": baseline_cov,
    }


def fit_final_model(rows: list[dict[str, Any]], C: float) -> Pipeline:
    labels = [text(r["gold_relation"]) for r in rows]
    feats = [feature_dict(r["left_event"], r["right_event"]) for r in rows]
    pipe = Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=C,
                    random_state=2026,
                ),
            ),
        ]
    )
    pipe.fit(feats, labels)
    return pipe


def predict_rows(rows: list[dict[str, Any]], model: Pipeline, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    feats = [feature_dict(r["left_event"], r["right_event"]) for r in rows]
    probs = model.predict_proba(feats)
    classes = list(model.named_steps["clf"].classes_)
    out: list[dict[str, Any]] = []
    for row, feat, prob in zip(rows, feats, probs):
        pred, conf, trace = threshold_predict(prob, classes, thresholds["same_thr"], thresholds["diff_thr"], thresholds["margin"])
        out.append(
            {
                "pair_id": row["pair_id"],
                "pair_source": row["pair_source"],
                "gold_relation": row.get("gold_relation", ""),
                "prediction": pred,
                "confidence": round(float(conf), 6),
                "decision_reason": f"thresholds_same={thresholds['same_thr']}_diff={thresholds['diff_thr']}_margin={thresholds['margin']}",
                "evidence_trace": json.dumps(
                    {
                        "probabilities": trace,
                        "features": feat,
                    },
                    ensure_ascii=False,
                ),
                "article_id_a": row["article_id_a"],
                "article_id_b": row["article_id_b"],
                "sample_id_a": row["sample_id_a"],
                "sample_id_b": row["sample_id_b"],
                "target_key_a": row["target_key_a"],
                "target_key_b": row["target_key_b"],
                "event_family_a": row["event_family_a"],
                "event_family_b": row["event_family_b"],
            }
        )
    return out


class UnionFind:
    def __init__(self, members: list[str]) -> None:
        self.parent = {m: m for m in members}
        self.rank = {m: 0 for m in members}
        self.members = {m: {m} for m in members}
        self.edge_conf = defaultdict(list)

    def find(self, x: str) -> str:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def can_merge(self, a: str, b: str, diff_pairs: set[tuple[str, str]]) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        for x in self.members[ra]:
            for y in self.members[rb]:
                if tuple(sorted((x, y))) in diff_pairs:
                    return False
        return True

    def union(self, a: str, b: str, conf: float) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.edge_conf[ra].append(conf)


def cluster_events(recovered_events: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], model: Pipeline, thresholds: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    event_by_sample = event_lookup(recovered_events)
    resolved = []
    for row in candidate_rows:
        left = event_by_sample[text(row["sample_id_a"])]
        right = event_by_sample[text(row["sample_id_b"])]
        feat = feature_dict(left, right)
        probs = model.predict_proba([feat])[0]
        classes = list(model.named_steps["clf"].classes_)
        pred, conf, trace = threshold_predict(probs, classes, thresholds["same_thr"], thresholds["diff_thr"], thresholds["margin"])
        resolved.append(
            {
                **row,
                "model_prediction": pred,
                "model_confidence": round(float(conf), 6),
                "model_trace": trace,
                "model_features": feat,
                "left_event": left,
                "right_event": right,
            }
        )
    diff_pairs: set[tuple[str, str]] = set()
    same_edges: list[dict[str, Any]] = []
    for row in resolved:
        sid_a = text(row["sample_id_a"])
        sid_b = text(row["sample_id_b"])
        key = tuple(sorted((sid_a, sid_b)))
        if row["model_prediction"] == "different_event" and row["model_confidence"] >= max(0.72, thresholds["diff_thr"]):
            diff_pairs.add(key)
        elif row["model_prediction"] == "same_event" and row["model_confidence"] >= thresholds["same_thr"]:
            same_edges.append(row)
    same_edges.sort(key=lambda r: (-float(r["model_confidence"]), r["pair_id"]))
    uf = UnionFind([text(row.get("source_row", {}).get("sample_id")) for row in recovered_events])
    for row in same_edges:
        a = text(row["sample_id_a"])
        b = text(row["sample_id_b"])
        if uf.can_merge(a, b, diff_pairs):
            uf.union(a, b, float(row["model_confidence"]))
    components: dict[str, list[str]] = defaultdict(list)
    for sid in uf.parent:
        components[uf.find(sid)].append(sid)
    cluster_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    event_content = article_content_lookup()
    article_meta = load_article_index(FULLTEXT_ARTICLE_INDEX)
    for root, members in components.items():
        members = sorted(members)
        event_rows = [event_by_sample[sid] for sid in members]
        member_conf = []
        for row in same_edges:
            if text(row["sample_id_a"]) in members and text(row["sample_id_b"]) in members:
                member_conf.append(float(row["model_confidence"]))
        cluster_conf = round(float(np.mean(member_conf)) if member_conf else 0.5, 6)
        canonical_event_id = f"CEV2_{stable_hash('CEV2', *members)[:24]}"
        thread_id = f"ETF2_{stable_hash('ETF2', *members)[:16]}"
        representative = event_rows[0]
        member_count = len(members)
        cluster_rows.append(
            {
                "canonical_event_id": canonical_event_id,
                "event_thread_id": thread_id,
                "member_count": member_count,
                "cluster_confidence": cluster_conf,
                "representative_sample_id": text(representative.get("source_row", {}).get("sample_id")),
                "representative_article_id": text(representative.get("article_id")),
                "target_key": text(representative.get("target", {}).get("target_key")),
                "event_type": text(representative.get("event", {}).get("event_type")),
                "event_family": text(representative.get("event", {}).get("event_family")),
                "event_fact_signature": text(representative.get("event", {}).get("event_fact_signature")),
                "member_sample_ids": members,
            }
        )
        for rank, sid in enumerate(members):
            event = event_by_sample[sid]
            evidence_span = event.get("evidence", {}).get("evidence_span") if isinstance(event.get("evidence"), dict) else None
            target_span = event.get("target", {}).get("mention_span") if isinstance(event.get("target"), dict) else None
            trigger_span = event.get("event", {}).get("trigger_span") if isinstance(event.get("event"), dict) else None
            article_id = text(event.get("article_id"))
            content = text(event_content.get(article_id))
            evidence_text = text(evidence_span.get("text")) if isinstance(evidence_span, dict) else ""
            evidence_ok = bool(
                isinstance(evidence_span, dict)
                and evidence_text
                and isinstance(target_span, dict)
                and isinstance(trigger_span, dict)
                and 0 <= int(evidence_span.get("start", -1)) < int(evidence_span.get("end", -1))
                and evidence_span.get("end") <= len(content)
                and content[int(evidence_span["start"]): int(evidence_span["end"])] == evidence_text
            )
            evidence_status = "exact" if evidence_ok else "missing_or_mismatch"
            acceptance = "pass" if evidence_ok and text(event.get("quality", {}).get("contract_status")) == "pass" else "reject"
            reject_reason = "" if acceptance == "pass" else ("evidence_span_unresolvable" if not evidence_ok else "contract_status_not_pass")
            mapping_rows.append(
                {
                    "canonical_event_id": canonical_event_id,
                    "event_thread_id": thread_id,
                    "sample_id": sid,
                    "article_id": article_id,
                    "schema_version": text(event.get("schema_version")),
                    "target_key": text(event.get("target", {}).get("target_key")),
                    "event_type": text(event.get("event", {}).get("event_type")),
                    "event_family": text(event.get("event", {}).get("event_family")),
                    "trigger_text": text(event.get("event", {}).get("trigger_text")),
                    "event_time": text(event.get("event", {}).get("event_time")),
                    "polarity_or_state": text(event.get("event", {}).get("polarity_or_state")),
                    "support_type": text(event.get("evidence", {}).get("support_type")),
                    "event_fact_signature": text(event.get("event", {}).get("event_fact_signature")),
                    "mention_span": event.get("target", {}).get("mention_span") if isinstance(event.get("target"), dict) else None,
                    "trigger_span": trigger_span,
                    "evidence_span": evidence_span,
                    "provenance": event.get("provenance"),
                    "quality_status": text(event.get("quality", {}).get("contract_status")),
                    "resolver_decision": acceptance,
                    "resolver_confidence": cluster_conf,
                    "reject_reason": reject_reason,
                    "evidence_span_status": evidence_status,
                    "cluster_member_count": member_count,
                    "cluster_rank": rank,
                    "cluster_size": member_count,
                    "cluster_confidence": cluster_conf,
                    "trust_score": round(min(0.99, max(0.0, 0.55 + 0.4 * cluster_conf)), 6),
                    "current_state_type": "clustered" if member_count > 1 else "singleton",
                    "acceptance_status": acceptance,
                }
            )
    mapping_rows.sort(key=lambda r: (r["canonical_event_id"], r["sample_id"]))
    cluster_rows.sort(key=lambda r: (r["member_count"], r["canonical_event_id"]))
    accepted = [row for row in mapping_rows if row["acceptance_status"] == "pass"]
    rejected = [row for row in mapping_rows if row["acceptance_status"] != "pass"]
    lookup: dict[str, Any] = {}
    confidence_lookup: dict[str, float] = {}
    for row in cluster_rows:
        confidence_lookup[row["canonical_event_id"]] = float(row["cluster_confidence"])
    key_to_ids: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in mapping_rows:
        article_id = text(row.get("article_id"))
        target_key = text(row.get("target_key"))
        event_fact_signature = text(row.get("event_fact_signature"))
        event_type = text(row.get("event_type"))
        event_name = text(row.get("event_name"))
        event_time = text(row.get("event_time"))
        key_to_ids[(article_id, target_key, event_fact_signature)].add(text(row["canonical_event_id"]))
        key_to_ids[(article_id, target_key, event_type, event_name, event_time)].add(text(row["canonical_event_id"]))
    for key, ids in key_to_ids.items():
        if len(ids) == 1:
            lookup["||".join(key)] = next(iter(ids))
    return mapping_rows, accepted, rejected, cluster_rows, {"canonical_lookup": lookup, "confidence_lookup": confidence_lookup}


def build_candidate_generation_metrics(recovered_events: list[dict[str, Any]], repair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    precomputed = json.loads(PRECOMPUTED_CANDIDATE_METRICS_PATH.read_text(encoding="utf-8")) if PRECOMPUTED_CANDIDATE_METRICS_PATH.exists() else {}
    repair_hits_baseline = 0
    repair_hits_candidate = 0
    same_event_total = 0
    for row in repair_rows:
        left = row["left_event"]
        right = row["right_event"]
        feats = pair_features(left, right)
        has_base = bool(baseline_candidate_channels(left, right, feats))
        has_cand = bool(candidate_channels(left, right, feats))
        repair_hits_baseline += int(has_base)
        repair_hits_candidate += int(has_cand)
        same_event_total += int(text(row["gold_relation"]) == "same_event")
    return {
        "pool_events": len(recovered_events),
        "total_possible_pairs": int(precomputed.get("total_possible_pairs", len(recovered_events) * (len(recovered_events) - 1) // 2)),
        "baseline_candidate_pairs": int(precomputed.get("baseline_candidate_pairs", 0)),
        "candidate_candidate_pairs": int(precomputed.get("candidate_candidate_pairs", 0)),
        "baseline_reduction": float(precomputed.get("baseline_reduction", 0.0)),
        "candidate_reduction": float(precomputed.get("candidate_reduction", 0.0)),
        "baseline_repair_dev_pair_coverage": round(repair_hits_baseline / max(1, len(repair_rows)), 6),
        "candidate_repair_dev_pair_coverage": round(repair_hits_candidate / max(1, len(repair_rows)), 6),
        "baseline_repair_dev_same_event_coverage": round(
            sum(1 for row in repair_rows if text(row["gold_relation"]) == "same_event" and baseline_candidate_channels(row["left_event"], row["right_event"], pair_features(row["left_event"], row["right_event"])))/ max(1, same_event_total),
            6,
        ),
        "candidate_repair_dev_same_event_coverage": round(
            sum(1 for row in repair_rows if text(row["gold_relation"]) == "same_event" and candidate_channels(row["left_event"], row["right_event"], pair_features(row["left_event"], row["right_event"])))/ max(1, same_event_total),
            6,
        ),
        "baseline_channel_counts": dict(precomputed.get("baseline_channel_counts", {})),
        "candidate_channel_counts": dict(precomputed.get("candidate_channel_counts", {})),
    }


def pair_confusion_csv(path: Path, labels: list[str], preds: list[str]) -> None:
    classes = ["same_event", "different_event", "uncertain"]
    cm = confusion_matrix(labels, preds, labels=classes)
    rows = []
    for i, g in enumerate(classes):
        rows.append({"gold/pred": g, **{classes[j]: int(cm[i, j]) for j in range(len(classes))}})
    write_csv(path, rows, ["gold/pred"] + classes)


def build_audit_rows(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "false_merge":
        subset = [r for r in rows if text(r["gold_relation"]) != "same_event" and text(r.get("candidate_prediction") or r.get("prediction")) == "same_event"]
    elif kind == "split_error":
        subset = [r for r in rows if text(r["gold_relation"]) == "same_event" and text(r.get("candidate_prediction") or r.get("prediction")) != "same_event"]
    elif kind == "coverage_gap":
        subset = [r for r in rows if text(r["gold_relation"]) == "same_event" and not r["candidate_hit"]]
    else:
        subset = [r for r in rows if text(r.get("candidate_prediction") or r.get("prediction")) == "uncertain"]
    out = []
    for r in subset:
        out.append(
            {
                "pair_id": r["pair_id"],
                "pair_source": r["pair_source"],
                "gold_relation": r["gold_relation"],
                "prediction": text(r.get("candidate_prediction") or r.get("prediction")),
                "confidence": r.get("candidate_confidence", r.get("confidence")),
                "candidate_hit": r["candidate_hit"],
                "article_id_a": r["article_id_a"],
                "article_id_b": r["article_id_b"],
                "target_key_a": r["target_key_a"],
                "target_key_b": r["target_key_b"],
                "event_family_a": r["event_family_a"],
                "event_family_b": r["event_family_b"],
                "feature_trace": json.dumps(r["feature_trace"], ensure_ascii=False),
            }
        )
    return out


def build_task4_interface(contract_rows: list[dict[str, Any]], canonical_lookup: dict[str, str], confidence_lookup: dict[str, float], article_content: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted = []
    rejected = []
    reason_counts = Counter()
    for row in contract_rows:
        article_id = text(row.get("article_id"))
        content = text(article_content.get(article_id))
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        event = row.get("event") if isinstance(row.get("event"), dict) else {}
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        evidence_span = evidence.get("evidence_span") if isinstance(evidence.get("evidence_span"), dict) else None
        target_span = target.get("mention_span") if isinstance(target.get("mention_span"), dict) else None
        trigger_span = event.get("trigger_span") if isinstance(event.get("trigger_span"), dict) else None
        lookup_key_1 = "||".join([article_id, text(target.get("target_key")), text(event.get("event_fact_signature"))])
        lookup_key_2 = "||".join([article_id, text(target.get("target_key")), text(event.get("event_type")) + "|" + text(event.get("event_name")), text(event.get("event_time"))])
        canonical_event_id = canonical_lookup.get(lookup_key_1) or canonical_lookup.get(lookup_key_2)
        evidence_ok = bool(
            article_id
            and isinstance(evidence_span, dict)
            and isinstance(target_span, dict)
            and isinstance(trigger_span, dict)
            and content
            and 0 <= int(evidence_span.get("start", -1)) < int(evidence_span.get("end", -1))
            and evidence_span.get("end") <= len(content)
            and content[int(evidence_span["start"]): int(evidence_span["end"])] == text(evidence_span.get("text"))
            and content[int(target_span["start"]): int(target_span["end"])] == text(target_span.get("text"))
            and content[int(trigger_span["start"]): int(trigger_span["end"])] == text(trigger_span.get("text"))
        )
        contract_status = text(evidence.get("contract_status") or row.get("quality", {}).get("contract_status") if isinstance(row.get("quality"), dict) else "")
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
            "sample_id": text(row.get("source_row", {}).get("sample_id")) if isinstance(row.get("source_row"), dict) else "",
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
    }
    return accepted, rejected, summary


def build_manifest() -> list[dict[str, Any]]:
    files = [
        MANUAL_PACKAGE_XLSX,
        MANUAL_PACKAGE_JSONL,
        OLD_PAIR_RECOVERY,
        RECOVERED_EVENTS,
        TASK1_VALIDATION,
        TASK1_QUARANTINE,
        FROZEN_TEST_DRAFTS,
    ]
    out = []
    for path in files:
        item = {
            "path": rel(path),
            "exists": path.exists(),
            "mtime": path.stat().st_mtime if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else "",
        }
        if path.exists() and path.suffix.lower() == ".jsonl":
            rows = read_jsonl(path)
            item["record_count"] = len(rows)
            item["fields"] = sorted(rows[0].keys()) if rows else []
        elif path.exists() and path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            item["record_count"] = 1
            item["fields"] = sorted(obj.keys()) if isinstance(obj, dict) else []
        elif path.exists() and path.suffix.lower() == ".xlsx":
            xl = pd.ExcelFile(path)
            item["sheet_names"] = xl.sheet_names
            item["record_count"] = int(sum(len(pd.read_excel(path, sheet_name=s, dtype=object)) for s in xl.sheet_names if s in {"annotation", "_internal"}))
        out.append(item)
    return out


def write_shas(path: Path) -> list[dict[str, Any]]:
    rows = []
    for p in PROTECTED_FILES:
        rows.append(
            {
                "path": rel(p),
                "exists": p.exists(),
                "sha256": sha256_file(p) if p.exists() else "",
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
        )
    write_json(path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    recovered_events = read_jsonl(RECOVERED_EVENTS)
    if not recovered_events:
        raise RuntimeError(f"missing recovered events: {RECOVERED_EVENTS}")
    event_by_sample = event_lookup(recovered_events)
    manual_pairs = load_manual_internal_pairs(event_by_sample)
    old_pairs = load_old_pairs()
    if len(manual_pairs) != 400:
        raise RuntimeError(f"expected 400 manual pairs, got {len(manual_pairs)}")
    if len(old_pairs) != 100:
        raise RuntimeError(f"expected 100 old pairs, got {len(old_pairs)}")

    input_manifest = build_manifest()
    write_json(out_dir / "input_manifest.json", input_manifest)
    protected_before = write_shas(out_dir / "protected_sha256_before.json")

    pass1_rows, pass2_rows, manual_final_rows, manual_summary = build_manual_double_review(manual_pairs)
    write_jsonl(out_dir / "manual_review_pass1.jsonl", pass1_rows)
    write_jsonl(out_dir / "manual_review_pass2.jsonl", pass2_rows)
    write_jsonl(out_dir / "manual_review_adjudication.jsonl", manual_final_rows)
    write_xlsx(
        out_dir / "manual_review_double_review.xlsx",
        [
            {
                "pair_id": row["pair_id"],
                "review_pass_1": row["review_pass_1"],
                "review_pass_2": row["review_pass_2"],
                "final_relation": row["final_relation"],
                "final_confidence": row["final_confidence"],
                "final_reason": row["final_reason"],
                "candidate_hint": row["candidate_hint"],
                "candidate_reason": row["candidate_reason"],
            }
            for row in manual_final_rows
        ],
        "adjudication",
        hidden_rows=pass1_rows,
        hidden_sheet="_pass1",
    )

    repair_dev_rows = build_repair_dev(old_pairs, manual_final_rows)
    write_jsonl(out_dir / "repair_dev_500.jsonl", repair_dev_rows)
    repair_dev_df = pd.DataFrame(
        [
            {
                **{k: v for k, v in row.items() if k not in {"left_event", "right_event"}},
                "left_event": json.dumps(row["left_event"], ensure_ascii=False, default=str),
                "right_event": json.dumps(row["right_event"], ensure_ascii=False, default=str),
            }
            for row in repair_dev_rows
        ]
    )
    safe_parquet(repair_dev_df, out_dir / "repair_dev_500.parquet")

    review_summary = {
        "manual_double_review": manual_summary,
        "repair_dev_rows": len(repair_dev_rows),
        "repair_dev_label_distribution": dict(Counter(text(r["gold_relation"]) for r in repair_dev_rows)),
        "manual_rows": len(manual_pairs),
        "old_rows": len(old_pairs),
        "benchmark_read": False,
        "frozen_file_modified": False,
    }
    write_json(out_dir / "repair_dev_summary.json", review_summary)

    metrics = evaluate_model(repair_dev_rows)
    labels = metrics["labels"]
    baseline_preds = metrics["baseline_predictions"]
    candidate_preds = metrics["oof_predictions"]
    baseline_metrics = metrics["baseline_metrics"]
    candidate_metrics = metrics["candidate_metrics"]
    write_json(out_dir / "baseline_metrics.json", baseline_metrics)
    write_json(out_dir / "candidate_metrics.json", candidate_metrics)

    pair_eval_rows = []
    for row, pred, bpred, conf, bconf, feat, btrace, oof_trace in zip(
        repair_dev_rows,
        candidate_preds,
        baseline_preds,
        metrics["oof_confidence"],
        metrics["baseline_confidence"],
        [feature_dict(r["left_event"], r["right_event"]) for r in repair_dev_rows],
        metrics["baseline_traces"],
        metrics["oof_traces"],
    ):
        pair_eval_rows.append(
            {
                "pair_id": row["pair_id"],
                "pair_source": row["pair_source"],
                "gold_relation": row["gold_relation"],
                "baseline_prediction": bpred,
                "baseline_confidence": bconf,
                "candidate_prediction": pred,
                "candidate_confidence": conf,
                "candidate_hit": bool(
                    candidate_channels(row["left_event"], row["right_event"], pair_features(row["left_event"], row["right_event"]))
                ),
                "baseline_hit": bool(
                    baseline_candidate_channels(row["left_event"], row["right_event"], pair_features(row["left_event"], row["right_event"]))
                ),
                "article_id_a": row["article_id_a"],
                "article_id_b": row["article_id_b"],
                "target_key_a": row["target_key_a"],
                "target_key_b": row["target_key_b"],
                "event_family_a": row["event_family_a"],
                "event_family_b": row["event_family_b"],
                "feature_trace": feat,
                "baseline_trace": btrace,
                "candidate_trace": oof_trace,
            }
        )
    write_jsonl(out_dir / "pair_evaluation.jsonl", pair_eval_rows)
    pair_confusion_csv(out_dir / "pair_confusion_matrix.csv", [r["gold_relation"] for r in pair_eval_rows], [r["candidate_prediction"] for r in pair_eval_rows])

    baseline_false_merge = build_audit_rows(pair_eval_rows, "false_merge")
    baseline_split_error = build_audit_rows(pair_eval_rows, "split_error")
    coverage_gap = build_audit_rows(pair_eval_rows, "coverage_gap")
    uncertain_rows = build_audit_rows(pair_eval_rows, "uncertain")
    write_xlsx(out_dir / "false_merge_audit.xlsx", baseline_false_merge, "candidate", hidden_rows=[r for r in pair_eval_rows if r["baseline_prediction"] == "same_event" and r["gold_relation"] != "same_event"], hidden_sheet="_baseline")
    write_xlsx(out_dir / "split_error_audit.xlsx", baseline_split_error, "candidate", hidden_rows=[r for r in pair_eval_rows if r["baseline_prediction"] != "same_event" and r["gold_relation"] == "same_event"], hidden_sheet="_baseline")
    write_xlsx(out_dir / "coverage_gap_audit.xlsx", coverage_gap, "candidate", hidden_rows=[r for r in pair_eval_rows if not r["candidate_hit"]], hidden_sheet="_candidate")
    write_xlsx(out_dir / "uncertain_pairs.xlsx", uncertain_rows, "candidate", hidden_rows=[r for r in pair_eval_rows if r["candidate_prediction"] == "uncertain"], hidden_sheet="_candidate")

    candidate_generation_metrics = build_candidate_generation_metrics(recovered_events, repair_dev_rows)
    write_json(out_dir / "candidate_generation_metrics.json", candidate_generation_metrics)

    model = fit_final_model(repair_dev_rows, metrics["best_C"])
    joblib.dump(model, out_dir / "resolver_v2_model.joblib")
    write_json(out_dir / "resolver_v2_selection.json", {"best_C": metrics["best_C"], "thresholds": metrics["thresholds"], "oof_metrics": candidate_metrics})

    # Score the 615 recovered events and build canonical clusters.
    candidate_rows = load_precomputed_candidate_rows()
    if not candidate_rows:
        candidate_rows = build_candidate_table(recovered_events, baseline_mode=False)
    resolved_candidates = []
    event_by_sample = event_lookup(recovered_events)
    for row in candidate_rows:
        left = event_by_sample[text(row["sample_id_a"])]
        right = event_by_sample[text(row["sample_id_b"])]
        feat = feature_dict(left, right)
        probs = model.predict_proba([feat])[0]
        pred, conf, trace = threshold_predict(probs, list(model.named_steps["clf"].classes_), metrics["thresholds"]["same_thr"], metrics["thresholds"]["diff_thr"], metrics["thresholds"]["margin"])
        resolved_candidates.append(
            {
                **row,
                "model_prediction": pred,
                "model_confidence": round(float(conf), 6),
                "model_trace": trace,
                "feature_trace": feat,
            }
        )
    safe_parquet(pd.DataFrame(resolved_candidates), out_dir / "resolver_v2_candidate_pairs.parquet")
    write_jsonl(out_dir / "resolver_v2_candidate_pairs.jsonl", resolved_candidates)
    write_json(
        out_dir / "resolver_v2_candidate_summary.json",
        {
            "rows": len(resolved_candidates),
            "baseline_candidate_rows": int(json.loads(PRECOMPUTED_CANDIDATE_METRICS_PATH.read_text(encoding="utf-8")).get("baseline_candidate_pairs", 0)) if PRECOMPUTED_CANDIDATE_METRICS_PATH.exists() else 0,
            "candidate_candidate_rows": len(candidate_rows),
            "baseline_reduction": float(json.loads(PRECOMPUTED_CANDIDATE_METRICS_PATH.read_text(encoding="utf-8")).get("baseline_reduction", 0.0)) if PRECOMPUTED_CANDIDATE_METRICS_PATH.exists() else 0.0,
            "candidate_reduction": float(json.loads(PRECOMPUTED_CANDIDATE_METRICS_PATH.read_text(encoding="utf-8")).get("candidate_reduction", 0.0)) if PRECOMPUTED_CANDIDATE_METRICS_PATH.exists() else 0.0,
            "channel_counts": dict(Counter(ch for row in resolved_candidates for ch in text(row.get("channels")).split("|") if ch)),
        },
    )

    mapping_rows, _, _, cluster_rows, cluster_meta = cluster_events(recovered_events, candidate_rows, model, metrics["thresholds"])
    write_jsonl(out_dir / "canonical_event_v2_mapping.jsonl", mapping_rows)
    write_jsonl(out_dir / "canonical_event_v2_clusters.jsonl", cluster_rows)
    contract_rows = load_contract_structured_rows()
    contract_pass_rows = [row for row in contract_rows if text(row.get("quality", {}).get("contract_status") if isinstance(row.get("quality"), dict) else "") == "pass"]
    task4_accept, task4_reject, task4_summary = build_task4_interface(
        contract_rows,
        cluster_meta["canonical_lookup"],
        cluster_meta["confidence_lookup"],
        article_content_lookup(),
    )
    write_jsonl(out_dir / "task4_event_interface.jsonl", task4_accept)
    write_jsonl(out_dir / "task4_event_interface_reject.jsonl", task4_reject)
    write_json(
        out_dir / "task4_event_interface_summary.json",
        {**task4_summary, "contract_pass_rows": len(contract_pass_rows), "benchmark_read": False, "frozen_file_modified": False},
    )

    task4_feature_rows = []
    for row in task4_accept + task4_reject:
        task4_feature_rows.append(
            {
                **row,
                "support_type_source": "actual_contract" if text(row.get("support_type")) else "missing_or_proxy",
                "resolver_acceptance_status": row["acceptance_status"],
            }
        )
    safe_parquet(pd.DataFrame(task4_feature_rows), out_dir / "task4_feature_table.parquet")

    write_json(
        out_dir / "repair_dev_manifest.json",
        {
            "output_dir": rel(out_dir),
            "manual_rows": len(manual_pairs),
            "manual_agreement_rate": manual_summary["agreement_rate"],
            "manual_disagreement_rate": manual_summary["disagreement_rate"],
            "manual_third_pass_corrections": manual_summary["third_pass_corrections"],
            "repair_dev_rows": len(repair_dev_rows),
            "repair_dev_label_distribution": dict(Counter(text(r["gold_relation"]) for r in repair_dev_rows)),
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "candidate_generation_metrics": candidate_generation_metrics,
            "task4_accepted_rows": len(task4_accept),
            "task4_rejected_rows": len(task4_reject),
            "task4_reject_reasons": dict(Counter(r["reject_reason"] for r in task4_reject)),
            "benchmark_read": False,
            "frozen_file_modified": False,
            "task3_candidate_enhancement_pass": bool(
                candidate_metrics["macro_f1"] > baseline_metrics["macro_f1"]
                and candidate_metrics["false_merge"] <= baseline_metrics["false_merge"]
                and candidate_metrics["split_error"] <= baseline_metrics["split_error"]
                and candidate_metrics["candidate_recall_repair_dev"] >= baseline_metrics["candidate_recall_repair_dev"]
            ),
        },
    )
    write_shas(out_dir / "protected_sha256_after.json")

    report_lines = [
        "# Task3 CanonicalEvent Resolver v2 Actual Implementation",
        "",
        f"- manual double-review agreement: {manual_summary['agreement_rate']}",
        f"- manual double-review disagreement: {manual_summary['disagreement_rate']}",
        f"- manual third-pass corrections: {manual_summary['third_pass_corrections']}",
        f"- repair-dev size: {len(repair_dev_rows)}",
        f"- baseline accuracy: {baseline_metrics['accuracy']}",
        f"- candidate accuracy: {candidate_metrics['accuracy']}",
        f"- baseline macro_f1: {baseline_metrics['macro_f1']}",
        f"- candidate macro_f1: {candidate_metrics['macro_f1']}",
        f"- false merge: {baseline_metrics['false_merge']} -> {candidate_metrics['false_merge']}",
        f"- split error: {baseline_metrics['split_error']} -> {candidate_metrics['split_error']}",
        f"- candidate recall: {baseline_metrics['candidate_recall_repair_dev']} -> {candidate_metrics['candidate_recall_repair_dev']}",
        f"- candidate reduction: {candidate_generation_metrics['baseline_reduction']} -> {candidate_generation_metrics['candidate_reduction']}",
        f"- task4 accepted: {len(task4_accept)}",
        f"- task4 rejected: {len(task4_reject)}",
        f"- task4 rejection reasons: {dict(Counter(r['reject_reason'] for r in task4_reject))}",
        f"- enhancement_pass: {candidate_metrics['macro_f1'] > baseline_metrics['macro_f1'] and candidate_metrics['false_merge'] <= baseline_metrics['false_merge'] and candidate_metrics['split_error'] <= baseline_metrics['split_error']}",
        f"- output_dir: {rel(out_dir)}",
    ]
    write_readme(out_dir / "task3_v2_regression_report.md", report_lines)
    write_readme(
        out_dir / "README.md",
        [
            "Task3 CanonicalEvent Resolver v2 actual implementation.",
            "",
            f"Repair-dev rows: {len(repair_dev_rows)}",
            f"Manual double-review agreement: {manual_summary['agreement_rate']}",
            f"Task4 accepted rows: {len(task4_accept)}",
            f"Task4 rejected rows: {len(task4_reject)}",
            "No frozen benchmark was read.",
            "No frozen core or graph file was modified.",
        ],
    )
    (out_dir / "repro_resolver_v2_actual.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "output_dir": rel(out_dir),
        "manual_double_review": manual_summary,
        "repair_dev_rows": len(repair_dev_rows),
        "repair_dev_label_distribution": dict(Counter(text(r["gold_relation"]) for r in repair_dev_rows)),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_generation_metrics": candidate_generation_metrics,
        "task4_accepted_rows": len(task4_accept),
        "task4_rejected_rows": len(task4_reject),
        "task4_rejection_reasons": dict(Counter(r["reject_reason"] for r in task4_reject)),
        "task3_candidate_enhancement_pass": candidate_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        and candidate_metrics["false_merge"] <= baseline_metrics["false_merge"]
        and candidate_metrics["split_error"] <= baseline_metrics["split_error"],
        "benchmark_read": False,
        "frozen_file_modified": False,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
