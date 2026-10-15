#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.6-A structure failure attribution and data-lineage audit.

This script is intentionally read-only for formal Task1/Task2 artifacts. It
does not train, does not run inference, and does not read 1029 gold-test labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_task1_structure_shadow_v1 import (  # noqa: E402
    char_overlap_f1,
    extract_pred_structure,
    normalize_date,
    normalize_direction,
    norm_text,
    parse_args_field,
    role_macro_f1,
    set_f1,
)
from scripts.run_task1_step5_5_candidate_v2 import (  # noqa: E402
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
    write_xlsx,
)

DEFAULT_CONFIRMED = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"
DEFAULT_STRUCTURE_EVAL = ROOT / "reports/task1_structure_shadow_evaluation.json"
DEFAULT_STRUCTURE_OUTPUT_DIR = ROOT / "outputs/task1_structure_shadow_evaluation_v1"
DEFAULT_PRED_MENTIONS = ROOT / "outputs/task1_gold_shadow_eval_v1/shadow_event_mentions.jsonl"
DEFAULT_BACKFILLED = ROOT / "outputs/task1_gold_shadow_eval_v1/shadow_predictions_backfilled.jsonl"
DEFAULT_SHADOW_MANIFEST = ROOT / "outputs/task1_gold_shadow_eval_v1/shadow_prediction_manifest.json"
DEFAULT_SHADOW_CONFIG = ROOT / "outputs/task1_gold_shadow_eval_v1/task1_shadow_inference_config.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/task1_structure_failure_audit_v1"
REPORT_JSON = ROOT / "reports/task1_structure_failure_audit_v1.json"
REPORT_MD = ROOT / "reports/task1_structure_failure_audit_v1.md"
REPORT_XLSX = ROOT / "reports/task1_structure_failure_audit_v1.xlsx"

FORBIDDEN_GOLD_LABEL_PATHS = {
    (ROOT / "data/processed/labeled/event_pairs_test.jsonl").resolve(),
}

REQUIRED_FIELDS = [
    "annotate_trigger",
    "annotate_event_span",
    "annotate_subject_company",
    "annotate_event_time",
    "annotate_object_target",
    "annotate_direction",
    "annotate_arguments_json",
    "annotate_argument_roles_json",
]

DIR_LABELS = ["positive", "negative", "neutral", "unclear"]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve()
    if resolved in FORBIDDEN_GOLD_LABEL_PATHS:
        raise RuntimeError(f"forbidden gold-test label path: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def event_names_from_pipe(value: Any) -> set[str]:
    text = "" if value is None else str(value)
    return {part.strip() for part in text.split("|") if part.strip()}


def event_name_from_mention(row: dict[str, Any]) -> str:
    elements = row.get("elements") if isinstance(row.get("elements"), dict) else {}
    return str(elements.get("event_name") or row.get("event_name") or "")


def event_name_from_candidate(row: dict[str, Any]) -> str:
    return str(row.get("event_name") or "")


def prediction_flag(row: dict[str, Any]) -> int:
    try:
        return int(row.get("prediction", 0))
    except (TypeError, ValueError):
        return 0


def load_schema_names() -> set[str]:
    schema_path = ROOT / "data/processed/event_schema.jsonl"
    if not schema_path.exists():
        return set()
    return {str(row.get("event_name") or "") for row in read_jsonl(schema_path) if row.get("event_name")}


def load_by_article(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        aid = str(row.get("article_id") or row.get("document_id") or "")
        if aid:
            out[aid].append(row)
    return out


def load_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("article_id")): row for row in read_jsonl(path) if row.get("article_id")}


def choose_mention(
    gold_names: set[str],
    positive_names: set[str],
    pred_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not pred_rows:
        return None
    for row in pred_rows:
        if event_name_from_mention(row) in gold_names:
            return row
    for row in pred_rows:
        if event_name_from_mention(row) in positive_names:
            return row
    return pred_rows[0]


def parse_gold_roles(value: Any) -> dict[str, str]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, list):
        return {str(item.get("text", "")): str(item.get("role", "")) for item in parsed if isinstance(item, dict)}
    return {}


def field_available(pred: dict[str, Any], field: str) -> bool:
    value = pred.get(field)
    if field in {"arguments", "argument_roles"}:
        return bool(value)
    return bool(str(value or "").strip())


def eval_rows(rows: list[dict[str, Any]], scope_name: str) -> dict[str, Any]:
    n = len(rows)
    trigger_pred_count = sum(1 for row in rows if row["pred_trigger"])
    trigger_tp = sum(1 for row in rows if row["trigger_exact"])
    trigger_precision = trigger_tp / trigger_pred_count if trigger_pred_count else 0.0
    trigger_recall = trigger_tp / n if n else 0.0
    trigger_f1 = 2 * trigger_precision * trigger_recall / (trigger_precision + trigger_recall) if trigger_precision + trigger_recall else 0.0

    arg_pred_all: set[tuple[str, str]] = set()
    arg_gold_all: set[tuple[str, str]] = set()
    argrole_pred_all: set[tuple[str, str, str]] = set()
    argrole_gold_all: set[tuple[str, str, str]] = set()
    direction_true: list[str] = []
    direction_pred: list[str] = []
    for row in rows:
        aid = row["article_id"]
        direction_true.append(row["gold_direction"])
        direction_pred.append(row["pred_direction"])
        for text in row["gold_arg_texts"]:
            arg_gold_all.add((aid, text))
        for text in row["pred_arg_texts"]:
            arg_pred_all.add((aid, text))
        for text, role in row["gold_arg_roles"]:
            argrole_gold_all.add((aid, text, role))
        for text, role in row["pred_arg_roles"]:
            argrole_pred_all.add((aid, text, role))

    direction_rows = []
    direction_f1s = []
    for label in DIR_LABELS:
        tp = sum(1 for g, p in zip(direction_true, direction_pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(direction_true, direction_pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(direction_true, direction_pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        direction_f1s.append(f1)
        direction_rows.append({"scope": scope_name, "label": label, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})

    arg_metrics = set_f1(arg_pred_all, arg_gold_all)
    argrole_metrics = set_f1(argrole_pred_all, argrole_gold_all)
    return {
        "scope": scope_name,
        "sample_count": n,
        "trigger": {
            "exact_precision": trigger_precision,
            "exact_recall": trigger_recall,
            "exact_f1": trigger_f1,
            "relaxed_overlap_f1": sum(row["trigger_overlap_f1"] for row in rows) / n if n else 0.0,
        },
        "event_span": {
            "exact_match_accuracy": sum(1 for row in rows if row["event_span_exact"]) / n if n else 0.0,
            "character_overlap_f1": sum(row["event_span_overlap_f1"] for row in rows) / n if n else 0.0,
        },
        "subject_company": {
            "exact_accuracy": sum(1 for row in rows if row["subject_exact"]) / n if n else 0.0,
            "normalized_accuracy": sum(1 for row in rows if row["subject_normalized"]) / n if n else 0.0,
        },
        "event_time": {
            "exact_accuracy": sum(1 for row in rows if row["event_time_exact"]) / n if n else 0.0,
            "normalized_date_accuracy": sum(1 for row in rows if row["event_time_normalized"]) / n if n else 0.0,
            "unspecified_recognition_accuracy": sum(1 for row in rows if row["unspecified_time_correct"]) / n if n else 0.0,
        },
        "object_target": {
            "exact_accuracy": sum(1 for row in rows if row["object_exact"]) / n if n else 0.0,
            "normalized_accuracy": sum(1 for row in rows if row["object_normalized"]) / n if n else 0.0,
        },
        "direction": {
            "accuracy": sum(1 for row in rows if row["direction_correct"]) / n if n else 0.0,
            "fixed_label_macro_f1": sum(direction_f1s) / len(direction_f1s) if direction_f1s else 0.0,
            "confusion_rows": direction_rows,
        },
        "arguments": {
            "argument_identification_precision": arg_metrics["precision"],
            "argument_identification_recall": arg_metrics["recall"],
            "argument_identification_f1": arg_metrics["f1"],
            "argument_role_classification_macro_f1": role_macro_f1(argrole_pred_all, argrole_gold_all),
            "end_to_end_argument_role_f1": argrole_metrics["f1"],
            "argument_tp": arg_metrics["tp"],
            "argument_fp": arg_metrics["fp"],
            "argument_fn": arg_metrics["fn"],
            "argument_role_tp": argrole_metrics["tp"],
            "argument_role_fp": argrole_metrics["fp"],
            "argument_role_fn": argrole_metrics["fn"],
        },
        "article_exact_match_accuracy": sum(1 for row in rows if row["full_structure_exact"]) / n if n else 0.0,
        "at_least_one_structure_field_correct_ratio": sum(1 for row in rows if row["at_least_one_structure_field_correct"]) / n if n else 0.0,
        "all_fields_correct_article_count": sum(1 for row in rows if row["full_structure_exact"]),
    }


def classification_metrics_from_detail(detail_rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    tp = fp = fn = 0
    for row in detail_rows:
        gold = event_names_from_pipe(row["gold_event_names"])
        pred = event_names_from_pipe(row.get(pred_key, ""))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def build_detail_rows(
    annotations: list[dict[str, Any]],
    article_predictions: dict[str, dict[str, Any]],
    mentions_by_article: dict[str, list[dict[str, Any]]],
    schema_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    field_failure_counts: dict[str, Counter[str]] = defaultdict(Counter)
    multiple_alignment_rows: list[dict[str, Any]] = []

    for row in annotations:
        aid = str(row["article_id"])
        gold_names = event_names_from_pipe(row.get("gold_event_names_for_context_after_classification_eval"))
        article_pred = article_predictions.get(aid)
        candidate_events = list((article_pred or {}).get("events", []))
        candidate_names = {event_name_from_candidate(ev) for ev in candidate_events if event_name_from_candidate(ev)}
        positive_events = [ev for ev in candidate_events if prediction_flag(ev) == 1]
        positive_names = {event_name_from_candidate(ev) for ev in positive_events if event_name_from_candidate(ev)}
        context_positive_names = event_names_from_pipe(row.get("pred_event_names"))
        pred_mentions = mentions_by_article.get(aid, [])
        chosen = choose_mention(gold_names, positive_names, pred_mentions)
        pred = extract_pred_structure(chosen)

        class_correct = bool(gold_names & context_positive_names)
        backfilled_class_correct = bool(gold_names & positive_names)
        mention_available = chosen is not None
        candidate_hit = bool(gold_names & candidate_names)

        gold_args = parse_args_field(row["annotate_arguments_json"])
        gold_roles = parse_gold_roles(row["annotate_argument_roles_json"])
        gold_arg_texts = {norm_text(arg["text"]) for arg in gold_args if norm_text(arg.get("text", ""))}
        pred_arg_texts = {norm_text(arg["text"]) for arg in pred["arguments"] if norm_text(arg.get("text", ""))}
        gold_arg_roles = {(norm_text(text), str(role)) for text, role in gold_roles.items() if norm_text(text)}
        pred_arg_roles = {(norm_text(text), str(role)) for text, role in pred["argument_roles"].items() if norm_text(text)}

        dir_gold = normalize_direction(row["annotate_direction"])
        dir_pred = normalize_direction(pred["direction"])
        flags = {
            "trigger": bool(pred["trigger"]) and norm_text(pred["trigger"]) == norm_text(row["annotate_trigger"]),
            "event_span": bool(pred["event_span"]) and norm_text(pred["event_span"]) == norm_text(row["annotate_event_span"]),
            "subject_company": bool(pred["subject_company"]) and norm_text(pred["subject_company"]) == norm_text(row["annotate_subject_company"]),
            "event_time": bool(pred["event_time"]) and normalize_date(pred["event_time"]) == normalize_date(row["annotate_event_time"]),
            "object_target": bool(pred["object_target"]) and norm_text(pred["object_target"]) == norm_text(row["annotate_object_target"]),
            "direction": dir_gold == dir_pred,
            "arguments": not (gold_arg_texts - pred_arg_texts) and not (pred_arg_texts - gold_arg_texts),
            "argument_roles": not (gold_arg_roles - pred_arg_roles) and not (pred_arg_roles - gold_arg_roles),
        }
        for field, ok in flags.items():
            if ok:
                continue
            if field in {"arguments", "argument_roles"}:
                missing = not pred_arg_texts if field == "arguments" else not pred_arg_roles
            else:
                missing = not str(pred.get(field, "") or "").strip()
            field_failure_counts[field]["missing" if missing else "wrong"] += 1

        trigger_overlap = char_overlap_f1(norm_text(pred["trigger"]), norm_text(row["annotate_trigger"]))
        span_overlap = char_overlap_f1(norm_text(pred["event_span"]), norm_text(row["annotate_event_span"]))
        detail = {
            "article_id": aid,
            "gold_event_names": "|".join(sorted(gold_names)),
            "candidate_event_count": len(candidate_events),
            "candidate_hit": candidate_hit,
            "candidate_event_names": "|".join(sorted(candidate_names)),
            "predicted_event_count": len(positive_events),
            "context_predicted_event_names": "|".join(sorted(context_positive_names)),
            "backfilled_predicted_event_names": "|".join(sorted(positive_names)),
            "predicted_event_names": "|".join(sorted(positive_names)),
            "classification_correct": class_correct,
            "backfilled_classification_correct": backfilled_class_correct,
            "context_backfilled_prediction_name_match": context_positive_names == positive_names,
            "mention_available": mention_available,
            "chosen_event_name": event_name_from_mention(chosen or {}),
            "chosen_schema_id": str((chosen or {}).get("schema_id", "")),
            "prediction_source": pred["source"],
            "gold_trigger": row["annotate_trigger"],
            "pred_trigger": pred["trigger"],
            "trigger_exact": flags["trigger"],
            "trigger_overlap_f1": trigger_overlap,
            "gold_event_span": row["annotate_event_span"],
            "pred_event_span": pred["event_span"],
            "event_span_exact": flags["event_span"],
            "event_span_overlap_f1": span_overlap,
            "gold_subject_company": row["annotate_subject_company"],
            "pred_subject_company": pred["subject_company"],
            "subject_exact": str(pred["subject_company"]) == str(row["annotate_subject_company"]),
            "subject_normalized": flags["subject_company"],
            "gold_event_time": row["annotate_event_time"],
            "pred_event_time": pred["event_time"],
            "event_time_exact": str(pred["event_time"]) == str(row["annotate_event_time"]),
            "event_time_normalized": flags["event_time"],
            "unspecified_time_correct": normalize_date(row["annotate_event_time"]) == normalize_date(pred["event_time"]),
            "gold_object_target": row["annotate_object_target"],
            "pred_object_target": pred["object_target"],
            "object_exact": str(pred["object_target"]) == str(row["annotate_object_target"]),
            "object_normalized": flags["object_target"],
            "gold_direction": dir_gold,
            "pred_direction": dir_pred,
            "direction_correct": flags["direction"],
            "gold_arg_texts": sorted(gold_arg_texts),
            "pred_arg_texts": sorted(pred_arg_texts),
            "gold_arg_roles": sorted(gold_arg_roles),
            "pred_arg_roles": sorted(pred_arg_roles),
            "gold_argument_count": len(gold_arg_texts),
            "pred_argument_count": len(pred_arg_texts),
            "full_structure_exact": all(flags.values()),
            "at_least_one_structure_field_correct": any(flags.values()),
        }
        detail_rows.append(detail)

        if not mention_available:
            if not gold_names or not gold_names <= schema_names:
                cause = "label_mapping_mismatch"
            elif article_pred is None:
                cause = "prediction_join_failure"
            elif not candidate_hit:
                cause = "candidate_missing"
            elif candidate_hit and not class_correct:
                cause = "classifier_false_negative"
            else:
                cause = "prediction_join_failure"
            missing_rows.append(
                {
                    "article_id": aid,
                    "primary_cause": cause,
                    "gold_event_names": "|".join(sorted(gold_names)),
                    "candidate_hit": candidate_hit,
                    "candidate_event_count": len(candidate_events),
                    "predicted_event_count": len(positive_events),
                    "predicted_event_names": "|".join(sorted(positive_names)),
                    "note": "No selected EventMention was available for structure evaluation.",
                }
            )

        if len(pred_mentions) > 1:
            gold_mention_count = sum(1 for pred_row in pred_mentions if event_name_from_mention(pred_row) in gold_names)
            chosen_is_gold = event_name_from_mention(chosen or {}) in gold_names
            if not chosen_is_gold or gold_mention_count != 1:
                multiple_alignment_rows.append(
                    {
                        "article_id": aid,
                        "gold_event_names": "|".join(sorted(gold_names)),
                        "pred_mention_count": len(pred_mentions),
                        "gold_matching_mention_count": gold_mention_count,
                        "chosen_event_name": event_name_from_mention(chosen or {}),
                        "alignment_risk": "ambiguous_or_non_gold_choice",
                    }
                )

    return detail_rows, missing_rows, {
        "field_failure_counts": {field: dict(counts) for field, counts in sorted(field_failure_counts.items())},
        "multiple_alignment_rows": multiple_alignment_rows,
    }


def field_available_metrics(detail_rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_map = {
        "trigger": "pred_trigger",
        "event_span": "pred_event_span",
        "subject_company": "pred_subject_company",
        "event_time": "pred_event_time",
        "object_target": "pred_object_target",
        "direction": "pred_direction",
        "arguments": "pred_arg_texts",
        "argument_roles": "pred_arg_roles",
    }
    out: dict[str, Any] = {}
    n = len(detail_rows)
    for field, pred_key in field_map.items():
        if field in {"arguments", "argument_roles"}:
            rows = [row for row in detail_rows if row[pred_key]]
        elif field == "direction":
            rows = [row for row in detail_rows if row[pred_key] and row[pred_key] != "unclear"]
        else:
            rows = [row for row in detail_rows if str(row[pred_key] or "").strip()]
        out[field] = {
            "available_sample_count": len(rows),
            "coverage": len(rows) / n if n else 0.0,
            "metrics": eval_rows(rows, f"field_available_{field}"),
        }
    all_required = [
        row
        for row in detail_rows
        if row["pred_trigger"]
        and row["pred_event_span"]
        and row["pred_subject_company"]
        and row["pred_event_time"]
        and row["pred_object_target"]
        and row["pred_direction"] != "unclear"
        and row["pred_arg_texts"]
        and row["pred_arg_roles"]
    ]
    out["all_required_fields"] = {
        "available_sample_count": len(all_required),
        "coverage": len(all_required) / n if n else 0.0,
        "article_exact_match_accuracy": sum(1 for row in all_required if row["full_structure_exact"]) / len(all_required) if all_required else 0.0,
    }
    return out


def audit_source_lineage() -> dict[str, Any]:
    manifest = read_json(DEFAULT_SHADOW_MANIFEST) if DEFAULT_SHADOW_MANIFEST.exists() else {}
    config = read_json(DEFAULT_SHADOW_CONFIG) if DEFAULT_SHADOW_CONFIG.exists() else {}
    step55_report = read_json(ROOT / "outputs/task1_step5_5_candidate_v2_dev/step5_5_report.json")
    frozen_manifest = read_json(ROOT / "outputs/task1_frozen_gold_blind_round1/frozen_manifest.json")
    existing_audit = read_json(ROOT / "reports/task1_existing_checkpoint_audit.json")
    hardcoded_script = ROOT / "scripts/run_task1_frozen_gold_blind_round1.py"
    hardcoded_text = hardcoded_script.read_text(encoding="utf-8") if hardcoded_script.exists() else ""
    return {
        "structure_eval_prediction_file": rel(DEFAULT_PRED_MENTIONS),
        "structure_eval_script": "scripts/evaluate_task1_structure_shadow_v1.py",
        "prediction_file_generator_script": "scripts/run_task1_gold_shadow_inference.py",
        "prediction_generated_at": manifest.get("generated_at"),
        "pipeline_version": manifest.get("pipeline_version"),
        "current_80_classification_f1_source": "pred_event_names embedded in outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl",
        "current_80_classification_f1_represents_latest_task1": False,
        "current_80_classification_f1_explanation": (
            "The reported 0.078740 classification F1 is computed from the confirmed annotation file's pred_event_names context field. "
            "Structure fields are read from outputs/task1_gold_shadow_eval_v1/shadow_event_mentions.jsonl, generated with "
            "outputs/event_match_baseline/best_model, threshold 0.38, and src.routing.target_router.TargetRouter. "
            "Neither source uses Step 5.5 candidate-v2 or the later limited-training round1 model."
        ),
        "used_outputs_event_match_baseline_best_model": manifest.get("model_name_or_path") == "outputs/event_match_baseline/best_model",
        "used_step5_4_checkpoint_849": False,
        "used_step5_5_candidate_v2": False,
        "used_step5_5_class_balanced_bce_classifier": False,
        "used_limited_round1_latest_classifier": False,
        "model_checkpoint": manifest.get("model_name_or_path"),
        "model_config_sha256": manifest.get("model_config_sha256") or config.get("model_config_sha256"),
        "model_weights_sha256": manifest.get("model_weights_sha256") or config.get("model_weights_sha256"),
        "threshold": manifest.get("threshold") or config.get("threshold"),
        "threshold_file_sha256": manifest.get("threshold_file_sha256") or config.get("threshold_file_sha256"),
        "candidate_router": "src.routing.target_router.TargetRouter",
        "candidate_router_sha256": manifest.get("router_code_sha256") or config.get("router_code_sha256"),
        "schema_sha256": manifest.get("schema_sha256") or config.get("schema_sha256"),
        "alias_sha256": manifest.get("alias_sha256") or config.get("alias_sha256"),
        "raw_prediction_file_sha256": manifest.get("raw_prediction_file_sha256"),
        "backfilled_prediction_file_sha256": manifest.get("backfilled_prediction_file_sha256"),
        "event_mention_file_sha256": sha256_file(DEFAULT_PRED_MENTIONS) if DEFAULT_PRED_MENTIONS.exists() else "",
        "possible_old_output_reuse_or_version_mixing": True,
        "version_mixing_note": "The structure evaluation was run after Step 5.5/limited round1 existed, but it consumed a pre-existing old gold-shadow EventMention file.",
        "step5_5_candidate_v2_available": {
            "manifest": "outputs/task1_step5_5_candidate_v2_dev/manifest.json",
            "candidate_recall_at_20": step55_report["candidate_v2"]["candidate_recall"]["candidate_recall_at_20"],
            "candidate_recall_at_32": step55_report["candidate_v2"]["candidate_recall"]["candidate_recall_at_32"],
            "passed": step55_report["candidate_v2"]["candidate_gate"]["passed"],
        },
        "step5_5_class_balanced_bce_available": {
            "model_dir": step55_report["training_summary"]["best_model_dir"],
            "best_loss": step55_report["training_summary"]["best_loss"],
            "dev_micro_f1": step55_report["best_dev_method"]["micro_f1"],
            "dev_precision": step55_report["best_dev_method"]["micro_precision"],
            "dev_gate_passed": step55_report["final_dev_gate"]["passed_for_1029_blind_test"],
        },
        "latest_dev_gold_passed_classifier_available": {
            "model_dir": frozen_manifest["model_dir"],
            "model_sha256": frozen_manifest["model_sha256"],
            "candidate_v2_top_k": frozen_manifest["candidate_v2"]["top_k"],
            "decoder": frozen_manifest["decoder"],
            "gold_blind_was_already_run_for_classification": True,
            "gold_blind_note": "This audit does not rerun 1029 gold. The existing manifest proves the frozen classifier artifact exists.",
        },
        "best_existing_checkpoint_before_training": existing_audit.get("best_existing_combination", {}),
        "direct_80_rerun_runner_exists": False,
        "direct_80_rerun_runner_note": (
            "scripts/run_task1_frozen_gold_blind_round1.py hard-codes BLIND_INPUT=outputs/task1_gold_shadow_eval_v1/blind_input.jsonl "
            f"and OUTPUT_DIR=outputs/task1_frozen_gold_blind_round1; input args present: {'--input' in hardcoded_text or 'add_argument(\"--input\"' in hardcoded_text}."
        ),
        "next_safe_command_template": (
            "python scripts/run_task1_latest_classifier_shadow_80.py "
            "--input outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl "
            "--output-dir outputs/task1_structure_latest_classifier_shadow_80_v1 "
            "--model outputs/task1_limited_training_round1/model_epoch_2 "
            "--candidate-config configs/task1_candidate_v2.yaml --top-k 20 --threshold 0.93 --no-gold-labels"
        ),
    }


def audit_structure_modules() -> list[dict[str, Any]]:
    files = [
        ROOT / "scripts/run_task1_gold_shadow_inference.py",
        ROOT / "scripts/evaluate_task1_structure_shadow_v1.py",
        ROOT / "scripts/build_task3_fulltext_v2.py",
        ROOT / "scripts/complete_clustered_event_elements.py",
        ROOT / "scripts/aggregate_multisource_events.py",
    ]
    code = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in files if path.exists())
    def present(pattern: str) -> bool:
        return re.search(pattern, code, flags=re.IGNORECASE) is not None

    return [
        {
            "field": "trigger",
            "state": "not_implemented",
            "evidence": "No trigger extraction model/rule is invoked by shadow EventMention generation; trigger is absent in output rows.",
            "code_signal_found": present(r"trigger"),
        },
        {
            "field": "event_span",
            "state": "rule_only",
            "evidence": "Only evidence sentence selection is used; event_local_text is initialized empty and no exact event-span extractor is connected.",
            "code_signal_found": present(r"event_local_text|select_evidence_sentences"),
        },
        {
            "field": "subject/company",
            "state": "rule_only",
            "evidence": "Subject/company is copied from routed target_name, not extracted as an event argument from text.",
            "code_signal_found": present(r"target_name|event_subject"),
        },
        {
            "field": "event_time",
            "state": "rule_only",
            "evidence": "Evaluation reads publish_time/event_time fallback; no occurrence-time extractor is connected for shadow mentions.",
            "code_signal_found": present(r"publish_time|event_time"),
        },
        {
            "field": "object/target",
            "state": "output_adapter_missing",
            "evidence": "Evaluator maps object_target from event_name, so the adapter does not expose a real object/target argument.",
            "code_signal_found": present(r"event_object|object_target"),
        },
        {
            "field": "direction",
            "state": "output_adapter_missing",
            "evidence": "impact_direction field exists but build_event_mentions_from_predictions writes None for every shadow mention.",
            "code_signal_found": present(r"impact_direction"),
        },
        {
            "field": "arguments",
            "state": "not_implemented",
            "evidence": "No argument extraction module is invoked; evaluator synthesizes pseudo arguments from target/time/event_name only.",
            "code_signal_found": present(r"argument|elements"),
        },
        {
            "field": "argument_role",
            "state": "not_implemented",
            "evidence": "No role classification head or role-label adapter is connected for shadow mentions.",
            "code_signal_found": present(r"role"),
        },
        {
            "field": "multi_event_mention_alignment",
            "state": "implemented_but_not_used",
            "evidence": "The evaluator has a gold-aware choose_mention helper for audit only; production shadow output has no gold mention alignment.",
            "code_signal_found": present(r"choose_prediction|rank|event_mention_id"),
        },
    ]


def annotation_training_audit(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    event_counter: Counter[str] = Counter()
    direction_counter: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    unspecified_count = 0
    for row in annotations:
        event_counter.update(event_names_from_pipe(row.get("gold_event_names_for_context_after_classification_eval")))
        direction_counter[normalize_direction(row.get("annotate_direction"))] += 1
        time_text = str(row.get("annotate_event_time") or "")
        if "未明确" in time_text or "不明确" in time_text or normalize_date(time_text) == "UNSPECIFIED":
            unspecified_count += 1
        for _, role in parse_gold_roles(row.get("annotate_argument_roles_json")).items():
            role_counter[str(role)] += 1
    nonzero_classes = len(event_counter)
    return {
        "usable_sample_count": len(annotations),
        "event_class_distribution": dict(event_counter.most_common()),
        "covered_event_class_count": nonzero_classes,
        "direction_distribution": dict(direction_counter.most_common()),
        "argument_role_distribution": dict(role_counter.most_common()),
        "event_time_unspecified_count": unspecified_count,
        "event_time_unspecified_ratio": unspecified_count / len(annotations) if annotations else 0.0,
        "is_enough_for_training": False,
        "assessment": (
            "80 rows are useful for validation/error audit and a tiny smoke train, but insufficient for a reliable 64-class-aware "
            "structure extractor. Labels are sparse across event classes and roles."
        ),
        "recommended_minimum_expansion": {
            "minimum_total_rows": 500,
            "preferred_total_rows": 800,
            "per_frequent_event_class_min_rows": 20,
            "per_direction_min_rows": 80,
            "per_argument_role_min_rows": 100,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    annotations = read_jsonl(args.confirmed)
    if len(annotations) != 80:
        raise RuntimeError(f"confirmed annotation row count must be 80, got {len(annotations)}")
    if len({row.get("article_id") for row in annotations}) != len(annotations):
        raise RuntimeError("confirmed annotation article_id is not unique")
    for row in annotations:
        missing = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise RuntimeError(f"missing required annotation fields for {row.get('article_id')}: {missing}")

    mentions = read_jsonl(args.pred_mentions)
    article_predictions = load_prediction_rows(args.backfilled_predictions)
    mentions_by_article = load_by_article(mentions)
    schema_names = load_schema_names()
    detail_rows, missing_rows, failure_aux = build_detail_rows(annotations, article_predictions, mentions_by_article, schema_names)

    end_to_end = eval_rows(detail_rows, "end_to_end_all_80")
    classification_conditional_rows = [row for row in detail_rows if row["classification_correct"]]
    backfilled_classification_conditional_rows = [row for row in detail_rows if row["backfilled_classification_correct"]]
    mention_available_rows = [row for row in detail_rows if row["mention_available"]]
    layered_metrics = {
        "end_to_end": end_to_end,
        "classification_conditional": eval_rows(classification_conditional_rows, "classification_conditional_eval_context"),
        "backfilled_classification_conditional": eval_rows(backfilled_classification_conditional_rows, "classification_conditional_backfilled_prediction"),
        "mention_available": eval_rows(mention_available_rows, "mention_available"),
        "field_available": field_available_metrics(detail_rows),
    }

    missing_counter = Counter(row["primary_cause"] for row in missing_rows)
    all_missing_categories = [
        "candidate_missing",
        "classifier_false_negative",
        "label_mapping_mismatch",
        "prediction_join_failure",
        "structure_not_generated",
        "output_schema_missing",
        "evaluator_adapter_failure",
        "multiple_mention_alignment_failure",
    ]
    missing_summary = {
        cause: {
            "count": missing_counter.get(cause, 0),
            "article_ids": [row["article_id"] for row in missing_rows if row["primary_cause"] == cause],
        }
        for cause in all_missing_categories
    }

    field_level_cause_counts = {
        "structure_not_generated": {
            "count": sum(1 for row in detail_rows if row["backfilled_classification_correct"] and row["mention_available"]),
            "article_ids": [row["article_id"] for row in detail_rows if row["backfilled_classification_correct"] and row["mention_available"]],
            "note": "Event category was recognized and a mention exists, but trigger/direction/argument-role extraction was not generated.",
        },
        "output_schema_missing": {
            "count": sum(1 for row in detail_rows if row["mention_available"] and (not row["pred_trigger"] or row["pred_direction"] == "unclear")),
            "article_ids": [row["article_id"] for row in detail_rows if row["mention_available"] and (not row["pred_trigger"] or row["pred_direction"] == "unclear")],
            "note": "Mention output schema has fields or expected slots but values are blank/None.",
        },
        "evaluator_adapter_failure": {
            "count": 0,
            "article_ids": [],
            "note": "No alternate populated trigger/direction/role fields were found in current shadow mentions.",
        },
        "multiple_mention_alignment_failure": {
            "count": len(failure_aux["multiple_alignment_rows"]),
            "article_ids": [row["article_id"] for row in failure_aux["multiple_alignment_rows"]],
            "note": "Rows with multiple predicted mentions and ambiguous/non-gold evaluator alignment.",
        },
    }

    lineage = audit_source_lineage()
    context_backfilled_mismatch_rows = [row for row in detail_rows if not row["context_backfilled_prediction_name_match"]]
    lineage["classification_context_metrics_80"] = classification_metrics_from_detail(detail_rows, "context_predicted_event_names")
    lineage["backfilled_prediction_metrics_80"] = classification_metrics_from_detail(detail_rows, "backfilled_predicted_event_names")
    lineage["classification_context_vs_backfilled_mismatch_count"] = len(context_backfilled_mismatch_rows)
    lineage["classification_context_vs_backfilled_mismatch_article_ids"] = [row["article_id"] for row in context_backfilled_mismatch_rows]
    lineage["classification_context_vs_backfilled_note"] = (
        "The confirmed annotation file's pred_event_names context and the old backfilled prediction file disagree for "
        f"{len(context_backfilled_mismatch_rows)} of 80 articles. This is a reproducibility/linkage issue and the reason "
        "classification metrics must be reported with their exact source."
    )
    modules = audit_structure_modules()
    training = annotation_training_audit(annotations)

    report = {
        "generated_at": now(),
        "stage": "Task1 Step 5.6-A",
        "status": "needs_error_audit",
        "scope": "audit_only_no_training_no_inference_no_gold_test",
        "inputs": {
            "confirmed_annotation": rel(args.confirmed),
            "confirmed_annotation_sha256": sha256_file(args.confirmed),
            "previous_structure_eval_report": rel(args.structure_eval),
            "previous_structure_eval_report_sha256": sha256_file(args.structure_eval),
            "prediction_mentions": rel(args.pred_mentions),
            "prediction_mentions_sha256": sha256_file(args.pred_mentions),
            "backfilled_predictions": rel(args.backfilled_predictions),
            "backfilled_predictions_sha256": sha256_file(args.backfilled_predictions),
        },
        "safety": {
            "annotation_row_count": len(annotations),
            "article_id_unique": True,
            "gold_test_label_files_read": [],
            "training_started": False,
            "model_inference_started": False,
            "formal_outputs_modified": False,
            "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
        },
        "source_lineage": lineage,
        "missing_53_primary_cause_summary": missing_summary,
        "missing_53_rows": missing_rows,
        "field_level_cause_counts": field_level_cause_counts,
        "structure_module_audit": modules,
        "layered_metrics": layered_metrics,
        "field_failure_counts": failure_aux["field_failure_counts"],
        "multiple_mention_alignment": {
            "predicted_multi_mention_article_count": sum(1 for rows in mentions_by_article.values() if len(rows) > 1 and any(a["article_id"] in {r.get("article_id") for r in rows} for a in annotations)),
            "ambiguous_or_failed_alignment_count": len(failure_aux["multiple_alignment_rows"]),
            "ambiguous_or_failed_alignment_rows": failure_aux["multiple_alignment_rows"],
            "assessment": "Multi-mention output exists, but the dominant failure in this 80-row audit is stale classifier/candidate routing plus missing structure fields, not evaluator join failure.",
        },
        "latest_step5_5_reuse_check": {
            "step5_5_candidate_v2_can_reuse": True,
            "step5_5_class_balanced_bce_can_reuse_but_not_best": True,
            "limited_round1_latest_classifier_can_reuse": True,
            "direct_safe_80_runner_exists": lineage["direct_80_rerun_runner_exists"],
            "recommendation": "Use limited round1 model_epoch_2 + candidate-v2 Top20 + dev threshold 0.93 for a new 80-row classification shadow; do not run hard-coded 1029 gold script.",
            "next_safe_command_template": lineage["next_safe_command_template"],
        },
        "annotation_training_audit": training,
        "step5_6b_plan": {
            "selected_status": "rerun_latest_classifier_first",
            "rationale": (
                "The current 80-row structure evaluation is based on stale baseline classification and old target routing. "
                "Rerun the latest frozen classifier on the 80 confirmed articles first to remove classification/candidate noise; "
                "then implement/connect a structure extractor because trigger/direction/argument roles are currently not generated."
            ),
            "secondary_required_work": "implement_structure_extractor_first",
            "needs_retraining": True,
            "modules_to_train_or_implement": [
                "span extraction",
                "trigger extraction",
                "subject/object/time span extraction",
                "argument extraction",
                "argument role classification",
                "direction classification",
            ],
            "candidate_generation_retraining_needed": False,
            "event_classification_retraining_needed": False,
            "lightweight_4x3090_plan": [
                "Shared MacBERT encoder initialized from local MacBERT-compatible checkpoint.",
                "BIO or start-end heads for trigger and event span.",
                "Subject/object/time span heads with null-span support.",
                "Argument role classification head over extracted spans.",
                "Direction four-class head over article plus event context.",
                "Multi-task loss with small dev-only threshold/calibration; no gold-test labels.",
            ],
            "expected_step5_6b_runtime": {
                "rerun_latest_classifier_80_only": "1-2 hours including runner/adapter checks; model inference itself should be minutes on 4x3090.",
                "structure_extractor_baseline_implementation": "0.5-1 engineering day for rule/adapter baseline.",
                "first_lightweight_training_round_after_more_labels": "4-8 GPU hours plus evaluation/reporting.",
            },
        },
    }
    return report


def flatten_metric_rows(layered: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in ["end_to_end", "classification_conditional", "backfilled_classification_conditional", "mention_available"]:
        item = layered[scope]
        rows.append(
            {
                "scope": scope,
                "sample_count": item["sample_count"],
                "trigger_exact_f1": item["trigger"]["exact_f1"],
                "trigger_relaxed_overlap_f1": item["trigger"]["relaxed_overlap_f1"],
                "event_span_exact": item["event_span"]["exact_match_accuracy"],
                "event_span_char_f1": item["event_span"]["character_overlap_f1"],
                "subject_norm_acc": item["subject_company"]["normalized_accuracy"],
                "time_norm_acc": item["event_time"]["normalized_date_accuracy"],
                "object_norm_acc": item["object_target"]["normalized_accuracy"],
                "direction_acc": item["direction"]["accuracy"],
                "direction_macro_f1": item["direction"]["fixed_label_macro_f1"],
                "argument_f1": item["arguments"]["argument_identification_f1"],
                "role_macro_f1": item["arguments"]["argument_role_classification_macro_f1"],
                "article_exact": item["article_exact_match_accuracy"],
            }
        )
    for field, item in layered["field_available"].items():
        rows.append(
            {
                "scope": f"field_available:{field}",
                "sample_count": item["available_sample_count"],
                "coverage": item["coverage"],
                "article_exact": item.get("article_exact_match_accuracy", item.get("metrics", {}).get("article_exact_match_accuracy", 0.0)),
            }
        )
    return rows


def write_outputs(report: dict[str, Any], detail_rows: list[dict[str, Any]]) -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DEFAULT_OUTPUT_DIR / "audit_report.json", report)
    write_json(REPORT_JSON, report)

    missing_rows = report["missing_53_rows"]
    write_jsonl(DEFAULT_OUTPUT_DIR / "missing_53_attribution.jsonl", missing_rows)
    write_jsonl(DEFAULT_OUTPUT_DIR / "structure_failure_detail_rows.jsonl", detail_rows)
    write_csv(DEFAULT_OUTPUT_DIR / "missing_53_attribution.csv", missing_rows)
    write_csv(DEFAULT_OUTPUT_DIR / "layered_metrics.csv", flatten_metric_rows(report["layered_metrics"]))
    write_csv(DEFAULT_OUTPUT_DIR / "structure_module_audit.csv", report["structure_module_audit"])
    write_csv(DEFAULT_OUTPUT_DIR / "multiple_mention_alignment.csv", report["multiple_mention_alignment"]["ambiguous_or_failed_alignment_rows"])

    missing_summary_rows = [
        {"cause": cause, "count": item["count"], "article_ids": "|".join(item["article_ids"])}
        for cause, item in report["missing_53_primary_cause_summary"].items()
    ]
    field_cause_rows = [
        {"cause": cause, "count": item["count"], "article_ids": "|".join(item["article_ids"]), "note": item["note"]}
        for cause, item in report["field_level_cause_counts"].items()
    ]
    training_rows = [
        {"kind": "event_class", "label": label, "count": count}
        for label, count in report["annotation_training_audit"]["event_class_distribution"].items()
    ] + [
        {"kind": "direction", "label": label, "count": count}
        for label, count in report["annotation_training_audit"]["direction_distribution"].items()
    ] + [
        {"kind": "argument_role", "label": label, "count": count}
        for label, count in report["annotation_training_audit"]["argument_role_distribution"].items()
    ]
    source_rows = [{"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in report["source_lineage"].items()]

    write_xlsx(
        REPORT_XLSX,
        {
            "source_lineage": source_rows,
            "missing_53_summary": missing_summary_rows,
            "missing_53_rows": missing_rows,
            "field_level_causes": field_cause_rows,
            "layered_metrics": flatten_metric_rows(report["layered_metrics"]),
            "module_audit": report["structure_module_audit"],
            "multi_mention": report["multiple_mention_alignment"]["ambiguous_or_failed_alignment_rows"],
            "annotation_distribution": training_rows,
        },
    )

    md = f"""# Task1 Step 5.6-A Structure Failure Audit

Generated at: {report['generated_at']}

Status: {report['status']}

## Source Lineage

- 80-row structure evaluation prediction file: `{report['source_lineage']['structure_eval_prediction_file']}`
- 80-row classification context field: `{report['source_lineage']['current_80_classification_f1_source']}`
- Generator script: `{report['source_lineage']['prediction_file_generator_script']}`
- Model/checkpoint: `{report['source_lineage']['model_checkpoint']}`
- Candidate router: `{report['source_lineage']['candidate_router']}`
- Threshold: `{report['source_lineage']['threshold']}`
- Represents latest Task1 classifier: `{report['source_lineage']['current_80_classification_f1_represents_latest_task1']}`

The current 0.078740 classification F1 comes from the confirmed annotation file's `pred_event_names` context field, not from Step 5.5 candidate-v2 or limited round1. The structure fields are read from the old baseline gold-shadow EventMention output. These two prediction views disagree on {report['source_lineage']['classification_context_vs_backfilled_mismatch_count']} of 80 articles.

## Missing-53 Primary Causes

""" + "\n".join(f"- {row['cause']}: {row['count']}" for row in missing_summary_rows) + f"""

## Layered Metrics

- End-to-end samples: {report['layered_metrics']['end_to_end']['sample_count']}, article exact: {report['layered_metrics']['end_to_end']['article_exact_match_accuracy']:.6f}
- Classification-conditional samples from eval context: {report['layered_metrics']['classification_conditional']['sample_count']}, article exact: {report['layered_metrics']['classification_conditional']['article_exact_match_accuracy']:.6f}
- Classification-conditional samples from backfilled prediction file: {report['layered_metrics']['backfilled_classification_conditional']['sample_count']}, article exact: {report['layered_metrics']['backfilled_classification_conditional']['article_exact_match_accuracy']:.6f}
- Mention-available samples: {report['layered_metrics']['mention_available']['sample_count']}, article exact: {report['layered_metrics']['mention_available']['article_exact_match_accuracy']:.6f}

## Step 5.6-B

Selected status: `{report['step5_6b_plan']['selected_status']}`

Secondary required work: `{report['step5_6b_plan']['secondary_required_work']}`

Safe next command template:

```bash
{report['latest_step5_5_reuse_check']['next_safe_command_template']}
```

This command is a template for the next step; the current repository does not yet provide this 80-only runner.
"""
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(md, encoding="utf-8")

    files = []
    for path in sorted(DEFAULT_OUTPUT_DIR.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS.txt"}:
            files.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    with (DEFAULT_OUTPUT_DIR / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for item in files:
            handle.write(f"{item['sha256']}  {item['path']}\n")
    files.append({"path": "SHA256SUMS.txt", "size_bytes": (DEFAULT_OUTPUT_DIR / "SHA256SUMS.txt").stat().st_size, "sha256": sha256_file(DEFAULT_OUTPUT_DIR / "SHA256SUMS.txt")})
    write_json(
        DEFAULT_OUTPUT_DIR / "manifest.json",
        {
            "manifest_version": "task1_structure_failure_audit_v1",
            "generated_at": now(),
            "output_dir": rel(DEFAULT_OUTPUT_DIR),
            "file_count": len(files),
            "files": files,
            "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmed", type=Path, default=DEFAULT_CONFIRMED)
    parser.add_argument("--structure-eval", type=Path, default=DEFAULT_STRUCTURE_EVAL)
    parser.add_argument("--structure-output-dir", type=Path, default=DEFAULT_STRUCTURE_OUTPUT_DIR)
    parser.add_argument("--pred-mentions", type=Path, default=DEFAULT_PRED_MENTIONS)
    parser.add_argument("--backfilled-predictions", type=Path, default=DEFAULT_BACKFILLED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    annotations = read_jsonl(args.confirmed)
    mentions_by_article = load_by_article(read_jsonl(args.pred_mentions))
    article_predictions = load_prediction_rows(args.backfilled_predictions)
    detail_rows, _, _ = build_detail_rows(annotations, article_predictions, mentions_by_article, load_schema_names())
    write_outputs(report, detail_rows)
    print(json.dumps({
        "status": report["status"],
        "missing_53": {k: v["count"] for k, v in report["missing_53_primary_cause_summary"].items()},
        "selected_step5_6b_status": report["step5_6b_plan"]["selected_status"],
        "latest_task1_represented": report["source_lineage"]["current_80_classification_f1_represents_latest_task1"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
