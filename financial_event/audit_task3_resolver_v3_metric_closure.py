#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/task3_resolver_v3_candidate_recall"
OUT = ROOT / "outputs/task3_resolver_v3_metric_closure"
RAW_REPAIR = ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/repair_dev_500.parquet"
LABELS_3 = ["same_event", "different_event", "uncertain"]
LABELS_2 = ["same_event", "different_event"]
SAME_THR = 0.98
DIFF_THR = 0.76
MARGIN = 0.0


CAUSE_OVERRIDES = {
    "DOCV2-dc26eb37c330b0fc": {
        "primary": "scorer_error",
        "secondary": ["argument_mismatch", "threshold_error"],
        "gold_ambiguous": False,
        "rationale": "Candidate was found, but same probability stayed at 0.732128 because the fact/evidence text diverged despite same target/object.",
    },
    "EMV2-3f30153b4e6573c1": {
        "primary": "candidate_recall_miss",
        "secondary": ["entity_identity_error", "event_family_boundary", "argument_mismatch"],
        "gold_ambiguous": False,
        "rationale": "Same article/event thread was split across analyst/source target and covered-company target; target and family mismatch blocked recall channels.",
    },
    "EMV2-4529b8bf917973a0": {
        "primary": "candidate_recall_miss",
        "secondary": ["entity_identity_error", "event_family_boundary", "argument_mismatch"],
        "gold_ambiguous": False,
        "rationale": "Institution participant and covered company were represented as different targets/families, so same-document evidence was not enough for recall.",
    },
    "EMV2-5deed9f672186973": {
        "primary": "candidate_recall_miss",
        "secondary": ["entity_identity_error", "event_family_boundary", "argument_mismatch"],
        "gold_ambiguous": False,
        "rationale": "Institution participant and covered company were represented as different targets/families, so same-source evidence was not recalled.",
    },
    "EMV2-847986ff33ecbd87": {
        "primary": "event_family_boundary",
        "secondary": ["scorer_error", "threshold_error"],
        "gold_ambiguous": False,
        "rationale": "Same BYD Dolphin G evidence was split between launch/pricing and overseas-launch families; scorer abstained after recall.",
    },
    "EMV2-a49df8bf5de3c264": {
        "primary": "candidate_recall_miss",
        "secondary": ["entity_identity_error", "event_family_boundary", "argument_mismatch"],
        "gold_ambiguous": False,
        "rationale": "Covered-company event and analyst/source event were normalized to different targets and families, blocking recall.",
    },
    "EMV2-c1be3db59f6a9acf": {
        "primary": "candidate_recall_miss",
        "secondary": ["entity_identity_error", "event_family_boundary", "argument_mismatch"],
        "gold_ambiguous": False,
        "rationale": "Covered-company event and analyst/source event were normalized to different targets and families, blocking recall.",
    },
    "EMV2-d091e3c82d2dd9ed": {
        "primary": "candidate_recall_miss",
        "secondary": ["entity_identity_error", "event_family_boundary", "argument_mismatch"],
        "gold_ambiguous": False,
        "rationale": "Institution participant and covered company were represented as different targets/families, blocking candidate generation.",
    },
    "EMV2-f29e77797fd142bd": {
        "primary": "candidate_recall_miss",
        "secondary": ["entity_identity_error", "event_family_boundary", "argument_mismatch"],
        "gold_ambiguous": False,
        "rationale": "Fund-holder organization and covered stock were normalized as different targets/families; lexical evidence was weak.",
    },
    "EMV2-fc74d7a6b614fa96": {
        "primary": "candidate_recall_miss",
        "secondary": ["entity_identity_error", "event_family_boundary", "argument_mismatch"],
        "gold_ambiguous": False,
        "rationale": "Institution participant and covered company were represented as different targets/families, blocking candidate generation.",
    },
    "MP-0b69ca4599739aab": {
        "primary": "scorer_error",
        "secondary": ["argument_mismatch", "gold_ambiguous"],
        "gold_ambiguous": True,
        "rationale": "Human label is different_event, but events share target, trigger and date; scorer over-weighted shared event key despite low fact/object similarity.",
    },
    "MP-45c3b32f5dec6bc2": {
        "primary": "gold_ambiguous",
        "secondary": ["argument_mismatch", "scorer_error"],
        "gold_ambiguous": True,
        "rationale": "Human label says same_event, but facts point to Micro LED/CPO layout versus Gree stake; evidence sentence is generic and insufficient.",
    },
    "MP-7aebfe650867475c": {
        "primary": "threshold_error",
        "secondary": ["argument_mismatch", "time_window_error", "gold_ambiguous"],
        "gold_ambiguous": True,
        "rationale": "Candidate was found and same probability was high but below 0.98; Beijing versus Guangdong subsidiary wording needs human boundary review.",
    },
    "MP-8734484caff85105": {
        "primary": "scorer_error",
        "secondary": ["argument_mismatch", "gold_ambiguous"],
        "gold_ambiguous": True,
        "rationale": "Human label is different_event, but events share target, trigger and date; scorer over-weighted shared event key despite low fact/object similarity.",
    },
    "MP-d8924f5b873f487f": {
        "primary": "scorer_error",
        "secondary": ["argument_mismatch", "gold_ambiguous"],
        "gold_ambiguous": True,
        "rationale": "Human label is different_event, but events share target, trigger and date; scorer over-weighted shared event key despite low fact/object similarity.",
    },
    "MP-f73ca65c3dfb5711": {
        "primary": "gold_ambiguous",
        "secondary": ["argument_mismatch", "scorer_error"],
        "gold_ambiguous": True,
        "rationale": "Human label says same_event, but facts point to Micro LED/CPO layout versus Gree stake; evidence sentence is generic and insufficient.",
    },
}


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def pair_key(a: Any, b: Any) -> tuple[str, str]:
    aa = text(a)
    bb = text(b)
    return tuple(sorted((aa, bb)))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def metadata_snapshot(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        if not path.exists():
            rows.append({"path": rel(path), "exists": False})
            continue
        stat = path.stat()
        rows.append(
            {
                "path": rel(path),
                "exists": True,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "metadata_sha256_no_file_content_read",
        "official1799_paths_scanned": 0,
        "final_holdout_rows_read": 0,
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "files": rows,
    }


def protected_paths(script_path: Path) -> list[Path]:
    return [
        SRC / "dev_pair_predictions.parquet",
        SRC / "r_channel_candidate_pairs_dev.parquet",
        SRC / "resolver_v3_constrained_clusters_dev.parquet",
        SRC / "r3_ann_dev_input.json",
        SRC / "resolver_v3_model/resolver_v3_pair_model.joblib",
        SRC / "resolver_v3_model/resolver_v3_pair_model_selection.json",
        SRC / "metric_denominator_audit.json",
        SRC / "metric_denominator_audit.md",
        script_path,
    ]


def read_parquet_filtered(path: Path, filter_expr: Any, columns: list[str] | None = None) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(filter=filter_expr, columns=columns)
    return table.to_pandas()


def read_repair_dev_pairs() -> pd.DataFrame:
    cols = [
        "pair_id",
        "sample_id_a",
        "sample_id_b",
        "gold_relation",
        "gold_relation_origin",
        "pair_split",
    ]
    df = read_parquet_filtered(
        SRC / "repair_pair_split_mapping.parquet",
        ds.field("pair_split") == "dev",
        columns=cols,
    )
    df["pair_key"] = df.apply(lambda row: pair_key(row["sample_id_a"], row["sample_id_b"]), axis=1)
    return df.drop_duplicates("pair_key", keep="first").copy()


def read_raw_repair_rows(pair_ids: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(RAW_REPAIR, format="parquet")
    table = dataset.to_table(filter=pc.field("pair_id").isin(pair_ids))
    return table.to_pandas().drop_duplicates("pair_id", keep="first")


def parse_event_json(value: Any) -> dict[str, Any]:
    raw = text(value)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def nested_get(row: dict[str, Any], *path: str) -> Any:
    cur: Any = row
    for part in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(part)
    return cur


def event_from_contract(raw: dict[str, Any]) -> dict[str, str]:
    return {
        "article_id": text(raw.get("article_id")),
        "source_name": text(nested_get(raw, "source", "source_name")),
        "target_key": text(nested_get(raw, "target", "target_key")),
        "target_name": text(nested_get(raw, "target", "target_name")),
        "event_type": text(nested_get(raw, "event", "event_type")),
        "event_family": text(nested_get(raw, "event", "event_family")),
        "event_name": text(nested_get(raw, "event", "event_name")),
        "trigger_text": text(nested_get(raw, "event", "trigger_text")),
        "object_text": text(nested_get(raw, "event", "object_text")),
        "event_time": text(nested_get(raw, "event", "event_time")),
        "publish_time": text(raw.get("publish_time")),
        "modality_status": text(nested_get(raw, "event", "modality_status")),
        "polarity_or_state": text(nested_get(raw, "event", "polarity_or_state")),
        "event_fact_signature": text(nested_get(raw, "event", "event_fact_signature")),
        "event_fact": text(nested_get(raw, "event", "event_fact")),
        "evidence_sentence": text(nested_get(raw, "evidence", "evidence_sentence")),
        "local_context": text(nested_get(raw, "evidence", "local_context")),
        "support_type": text(nested_get(raw, "evidence", "support_type")),
        "document_hash": text(nested_get(raw, "provenance", "document_hash")),
        "sentence_id": text(nested_get(raw, "provenance", "sentence_id")),
    }


def parse_r3_field(block: str, field: str, all_fields: list[str]) -> str:
    prefix = f"{field}: "
    start = block.find(prefix)
    if start < 0:
        return ""
    start += len(prefix)
    next_positions = []
    for other in all_fields:
        marker = f"\n{other}: "
        idx = block.find(marker, start)
        if idx >= 0:
            next_positions.append(idx)
    end = min(next_positions) if next_positions else len(block)
    return block[start:end].strip()


def read_dev_events() -> dict[str, dict[str, Any]]:
    event_cols = [
        "event_uid",
        "article_id",
        "target_key",
        "target_name",
        "event_family",
        "event_type",
        "event_name",
        "trigger_text",
        "object_text",
        "publish_time",
        "event_time",
        "event_fact_signature",
        "document_hash",
        "sentence_id",
    ]
    event_df = read_parquet_filtered(
        SRC / "event_split_mapping.parquet",
        ds.field("split") == "dev",
        columns=event_cols,
    )
    events = {text(row.event_uid): dict(row._asdict()) for row in event_df.itertuples(index=False)}
    r3 = json.loads((SRC / "r3_ann_dev_input.json").read_text(encoding="utf-8"))
    fields = [
        "target_key",
        "target_name",
        "event_family",
        "event_name",
        "trigger",
        "object",
        "event_time",
        "publish_time",
        "source",
        "event_fact",
        "evidence",
    ]
    for item in r3.get("events", []):
        uid = text(item.get("event_uid"))
        if uid not in events:
            continue
        block = text(item.get("text"))
        events[uid]["source_name"] = parse_r3_field(block, "source", fields)
        events[uid]["event_fact"] = parse_r3_field(block, "event_fact", fields)
        events[uid]["evidence_sentence"] = parse_r3_field(block, "evidence", fields)
    return events


def build_candidate_map() -> dict[tuple[str, str], dict[str, Any]]:
    df = pd.read_parquet(SRC / "r_channel_candidate_pairs_dev.parquet")
    df["pair_key"] = df.apply(lambda row: pair_key(row["sample_id_a"], row["sample_id_b"]), axis=1)
    return {row["pair_key"]: row.drop(labels=["pair_key"]).to_dict() for _, row in df.iterrows()}


def build_prediction_map() -> dict[tuple[str, str], dict[str, Any]]:
    df = pd.read_parquet(SRC / "dev_pair_predictions.parquet")
    df["pair_key"] = df.apply(lambda row: pair_key(row["sample_id_a"], row["sample_id_b"]), axis=1)
    return {row["pair_key"]: row.drop(labels=["pair_key"]).to_dict() for _, row in df.iterrows()}


def build_cluster_map() -> dict[str, str]:
    clusters = pd.read_parquet(SRC / "resolver_v3_constrained_clusters_dev.parquet")
    out: dict[str, str] = {}
    for row in clusters.itertuples(index=False):
        for member in getattr(row, "member_sample_ids"):
            out[text(member)] = text(getattr(row, "resolver_v3_cluster_id"))
    return out


def build_merge_audit_map() -> dict[tuple[str, str], dict[str, Any]]:
    path = SRC / "resolver_v3_cluster_merge_audit.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["pair_key"] = df.apply(lambda row: pair_key(row["sample_id_a"], row["sample_id_b"]), axis=1)
    return {row["pair_key"]: row.drop(labels=["pair_key"]).to_dict() for _, row in df.iterrows()}


def raw_feature_map(raw_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    keep = [
        "evidence_similarity",
        "fact_similarity",
        "object_similarity",
        "trigger_similarity",
        "time_relation",
        "modality_relation",
        "provenance_relation",
        "support_type_a",
        "support_type_b",
        "polarity_or_state_a",
        "polarity_or_state_b",
        "event_fact_signature_a",
        "event_fact_signature_b",
    ]
    for row in raw_rows.itertuples(index=False):
        out[text(getattr(row, "pair_id"))] = {col: getattr(row, col) for col in keep if hasattr(row, col)}
    return out


def event_side_columns(prefix: str, ev: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_article_id": ev.get("article_id", ""),
        f"{prefix}_source_name": ev.get("source_name", ""),
        f"{prefix}_target_key": ev.get("target_key", ""),
        f"{prefix}_target_name": ev.get("target_name", ""),
        f"{prefix}_event_family": ev.get("event_family", ""),
        f"{prefix}_event_type": ev.get("event_type", ""),
        f"{prefix}_event_name": ev.get("event_name", ""),
        f"{prefix}_trigger_text": ev.get("trigger_text", ""),
        f"{prefix}_object_text": ev.get("object_text", ""),
        f"{prefix}_event_time": ev.get("event_time", ""),
        f"{prefix}_publish_time": ev.get("publish_time", ""),
        f"{prefix}_modality_status": ev.get("modality_status", ""),
        f"{prefix}_support_type": ev.get("support_type", ""),
        f"{prefix}_event_fact_signature": ev.get("event_fact_signature", ""),
        f"{prefix}_event_fact": ev.get("event_fact", ""),
        f"{prefix}_evidence_sentence": ev.get("evidence_sentence", ""),
        f"{prefix}_document_hash": ev.get("document_hash", ""),
        f"{prefix}_sentence_id": ev.get("sentence_id", ""),
    }


def label_eval(gold: str, pred: str) -> str:
    if gold == "same_event" and pred == "same_event":
        return "TP"
    if gold == "different_event" and pred == "same_event":
        return "FP_false_merge"
    if gold == "same_event" and pred != "same_event":
        return "FN_split_error"
    if gold == "different_event" and pred != "same_event":
        return "TN"
    return "other"


def candidate_blockers(row: dict[str, Any], cand: dict[str, Any] | None) -> str:
    if cand:
        return ""
    blockers = ["R1/R2/R3/R4_union_miss"]
    if row["event_a_target_key"] != row["event_b_target_key"]:
        blockers.append("target_identity_mismatch_blocks_target_alias_channels")
    if row["event_a_event_family"] != row["event_b_event_family"]:
        blockers.append("event_family_mismatch_blocks_family_channels")
    if float(row.get("raw_fact_similarity") or 0.0) < 0.5:
        blockers.append("weak_fact_similarity")
    if float(row.get("raw_object_similarity") or 0.0) < 0.5:
        blockers.append("weak_argument_object_similarity")
    if float(row.get("raw_trigger_similarity") or 0.0) < 0.5:
        blockers.append("weak_trigger_similarity")
    blockers.append("R3_ANN_not_returned_for_pair")
    return "|".join(blockers)


def build_row_level() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    repair = read_repair_dev_pairs()
    raw_rows = read_raw_repair_rows(repair["pair_id"].astype(str).tolist())
    raw_by_pair = {text(row.pair_id): row for row in raw_rows.itertuples(index=False)}
    raw_features = raw_feature_map(raw_rows)
    predictions = build_prediction_map()
    candidates = build_candidate_map()
    events = read_dev_events()
    cluster_map = build_cluster_map()
    merge_audit = build_merge_audit_map()

    rows: list[dict[str, Any]] = []
    for item in repair.itertuples(index=False):
        key = pair_key(item.sample_id_a, item.sample_id_b)
        pred = predictions.get(key)
        if pred is None:
            raise RuntimeError(f"missing prediction for {key}")
        cand = candidates.get(key)
        raw = raw_by_pair.get(text(item.pair_id))
        left_event = event_from_contract(parse_event_json(getattr(raw, "left_event", ""))) if raw is not None else events.get(key[0], {})
        right_event = event_from_contract(parse_event_json(getattr(raw, "right_event", ""))) if raw is not None else events.get(key[1], {})
        # Fall back to dev event text if the raw pair event has empty fields.
        for ev, uid in [(left_event, text(item.sample_id_a)), (right_event, text(item.sample_id_b))]:
            fallback = events.get(uid, {})
            for col in ["source_name", "event_fact", "evidence_sentence"]:
                if not text(ev.get(col)):
                    ev[col] = text(fallback.get(col))
        gold = text(item.gold_relation)
        prediction = text(pred.get("prediction"))
        evaluation = label_eval(gold, prediction)
        cluster_a = cluster_map.get(text(item.sample_id_a), "")
        cluster_b = cluster_map.get(text(item.sample_id_b), "")
        cluster_relation = "same_event" if cluster_a and cluster_a == cluster_b else "different_event"
        merge_row = merge_audit.get(key, {})
        row = {
            "pair_id": text(item.pair_id),
            "sample_id_a": text(item.sample_id_a),
            "sample_id_b": text(item.sample_id_b),
            "pair_key": "||".join(key),
            "gold_label": gold,
            "gold_label_source": text(item.gold_relation_origin),
            "prediction": prediction,
            "confidence": float(pred.get("confidence", 0.0)),
            "prob_same_event": float(pred.get("prob_same_event", 0.0)),
            "prob_different_event": float(pred.get("prob_different_event", 0.0)),
            "prob_uncertain": float(pred.get("prob_uncertain", 0.0)),
            "candidate_hit": bool(pred.get("candidate_hit")),
            "decision_source": text(pred.get("decision_source")),
            "evaluation": evaluation,
            "exact_label_correct": gold == prediction,
            "binary_same_correct": (gold == "same_event") == (prediction == "same_event"),
            "is_abstention": prediction == "uncertain",
            "is_review_label": prediction in {"review", "unresolved", "abstain", "uncertain"},
            "cluster_id_a": cluster_a,
            "cluster_id_b": cluster_b,
            "cluster_relation_from_dev_clusters": cluster_relation,
            "pair_prediction_changed_by_cluster": prediction != cluster_relation,
            "cluster_relation_correct_vs_gold": cluster_relation == gold,
            "cluster_introduced_error": (prediction == gold) and (cluster_relation != gold),
            "cluster_corrected_pair_error": (prediction != gold) and (cluster_relation == gold),
            "cluster_merge_action": text(merge_row.get("merge_action")),
            "cluster_veto_reason": text(merge_row.get("veto_reason")),
            "R1": bool(cand.get("R1")) if cand else False,
            "R2": bool(cand.get("R2")) if cand else False,
            "R3": bool(cand.get("R3")) if cand else False,
            "R4": bool(cand.get("R4")) if cand else False,
            "r1_reasons": text(cand.get("r1_reasons")) if cand else "",
            "r2_reasons": text(cand.get("r2_reasons")) if cand else "",
            "r3_reasons": text(cand.get("r3_reasons")) if cand else "",
            "r4_reasons": text(cand.get("r4_reasons")) if cand else "",
            "candidate_time_relation": text(cand.get("time_relation")) if cand else "",
            "candidate_hard_conflicts": text(cand.get("hard_conflicts")) if cand else "",
            "candidate_fact_similarity": cand.get("fact_similarity") if cand else np.nan,
            "candidate_object_similarity": cand.get("object_similarity") if cand else np.nan,
            "candidate_trigger_similarity": cand.get("trigger_similarity") if cand else np.nan,
            "candidate_evidence_similarity": cand.get("evidence_similarity") if cand else np.nan,
            "same_target_candidate_feature": cand.get("same_target") if cand else False,
            "same_family_candidate_feature": cand.get("same_family") if cand else False,
        }
        for k, v in raw_features.get(text(item.pair_id), {}).items():
            row[f"raw_{k}"] = v
        row.update(event_side_columns("event_a", left_event))
        row.update(event_side_columns("event_b", right_event))
        row["candidate_generation_blockers"] = candidate_blockers(row, cand)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "manual_pair_count": int(len(df)),
        "manual_gold_counts": {k: int(v) for k, v in df["gold_label"].value_counts().to_dict().items()},
        "gold_label_source_counts": {k: int(v) for k, v in df["gold_label_source"].value_counts().to_dict().items()},
        "actual_prediction_label_set": sorted(df["prediction"].dropna().astype(str).unique().tolist()),
        "special_prediction_labels_present": {
            "review": bool((df["prediction"] == "review").any()),
            "unresolved": bool((df["prediction"] == "unresolved").any()),
            "abstain": bool((df["prediction"] == "abstain").any()),
            "uncertain": bool((df["prediction"] == "uncertain").any()),
        },
        "thresholds": {"same_thr": SAME_THR, "diff_thr": DIFF_THR, "margin": MARGIN},
    }
    io = {
        "repair_pair_split_mapping": rel(SRC / "repair_pair_split_mapping.parquet"),
        "raw_repair_pair_source": rel(RAW_REPAIR),
        "dev_pair_predictions": rel(SRC / "dev_pair_predictions.parquet"),
        "candidate_pairs": rel(SRC / "r_channel_candidate_pairs_dev.parquet"),
        "dev_event_text": rel(SRC / "r3_ann_dev_input.json"),
        "cluster_predictions": rel(SRC / "resolver_v3_constrained_clusters_dev.parquet"),
    }
    return df, summary, io


def metrics_for_dev109(df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["gold_label"].astype(str).tolist()
    y_pred = df["prediction"].astype(str).tolist()
    cm3 = confusion_matrix(y_true, y_pred, labels=LABELS_3)
    p3, r3, f3, s3 = precision_recall_fscore_support(y_true, y_pred, labels=LABELS_3, zero_division=0)
    p2, r2, f2, s2 = precision_recall_fscore_support(y_true, y_pred, labels=LABELS_2, zero_division=0)

    pred_closed = ["same_event" if p == "same_event" else "different_event" for p in y_pred]
    cm2_closed = confusion_matrix(y_true, pred_closed, labels=LABELS_2)
    p2c, r2c, f2c, s2c = precision_recall_fscore_support(y_true, pred_closed, labels=LABELS_2, zero_division=0)

    covered = df[df["prediction"].isin(LABELS_2)].copy()
    y_true_cov = covered["gold_label"].astype(str).tolist()
    y_pred_cov = covered["prediction"].astype(str).tolist()
    p_cov, r_cov, f_cov, s_cov = precision_recall_fscore_support(y_true_cov, y_pred_cov, labels=LABELS_2, zero_division=0)
    cm_cov = confusion_matrix(y_true_cov, y_pred_cov, labels=LABELS_2)

    tp = int(((df["gold_label"] == "same_event") & (df["prediction"] == "same_event")).sum())
    fp = int(((df["gold_label"] == "different_event") & (df["prediction"] == "same_event")).sum())
    fn = int(((df["gold_label"] == "same_event") & (df["prediction"] != "same_event")).sum())
    tn = int(((df["gold_label"] == "different_event") & (df["prediction"] != "same_event")).sum())

    exact_accuracy = float(accuracy_score(y_true, y_pred))
    exact_balanced = float(balanced_accuracy_score(y_true, y_pred))
    closed_accuracy = float(accuracy_score(y_true, pred_closed))
    closed_balanced = float(balanced_accuracy_score(y_true, pred_closed))
    covered_accuracy = float(accuracy_score(y_true_cov, y_pred_cov)) if y_true_cov else None
    covered_balanced = float(balanced_accuracy_score(y_true_cov, y_pred_cov)) if y_true_cov else None

    same_f1 = float(f2[0])
    different_f1 = float(f2[1])
    historical_macro = float(f1_score(y_true, y_pred, labels=LABELS_2, average="macro", zero_division=0))
    return {
        "formal_denominator": {
            "pair_count": int(len(df)),
            "gold_same_event": int((df["gold_label"] == "same_event").sum()),
            "gold_different_event": int((df["gold_label"] == "different_event").sum()),
            "unlabeled_pairs_included": 0,
        },
        "actual_prediction_label_set": sorted(df["prediction"].dropna().astype(str).unique().tolist()),
        "third_or_abstention_label_present": bool((df["prediction"] == "uncertain").any()),
        "special_label_counts": {k: int(v) for k, v in df["prediction"].value_counts().to_dict().items()},
        "same_vs_rest_binary_counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "three_label_confusion_matrix": {
            LABELS_3[i]: {LABELS_3[j]: int(cm3[i, j]) for j in range(len(LABELS_3))}
            for i in range(len(LABELS_3))
        },
        "closed_binary_confusion_matrix_abstain_as_different": {
            LABELS_2[i]: {LABELS_2[j]: int(cm2_closed[i, j]) for j in range(len(LABELS_2))}
            for i in range(len(LABELS_2))
        },
        "covered_only_confusion_matrix_excluding_uncertain_predictions": {
            LABELS_2[i]: {LABELS_2[j]: int(cm_cov[i, j]) for j in range(len(LABELS_2))}
            for i in range(len(LABELS_2))
        },
        "historical_sklearn_binary_labels_metric": {
            "name": "sklearn_macro_f1_labels_same_event_different_event_predictions_keep_uncertain",
            "sklearn_call": "f1_score(y_true, y_pred, labels=['same_event','different_event'], average='macro', zero_division=0)",
            "same_event_f1": round(same_f1, 6),
            "different_event_f1": round(different_f1, 6),
            "binary_macro_f1": round(historical_macro, 6),
            "explanation": "Uncertain predictions are not averaged as a class, but they still reduce recall for the true same_event or different_event class.",
        },
        "sklearn_three_label_macro": {
            "sklearn_call": "f1_score(y_true, y_pred, labels=['same_event','different_event','uncertain'], average='macro', zero_division=0)",
            "per_class_f1": {LABELS_3[i]: round(float(f3[i]), 6) for i in range(len(LABELS_3))},
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS_3, average="macro", zero_division=0)), 6),
        },
        "standard_closed_binary_metric": {
            "name": "closed_binary_same_vs_different_uncertain_mapped_to_different",
            "mapping": "prediction == same_event => same_event; all other predictions => different_event",
            "same_event_precision": round(float(p2c[0]), 6),
            "same_event_recall": round(float(r2c[0]), 6),
            "same_event_f1": round(float(f2c[0]), 6),
            "different_event_precision": round(float(p2c[1]), 6),
            "different_event_recall": round(float(r2c[1]), 6),
            "different_event_f1": round(float(f2c[1]), 6),
            "binary_macro_f1": round(float((f2c[0] + f2c[1]) / 2.0), 6),
            "accuracy": round(closed_accuracy, 6),
            "balanced_accuracy": round(closed_balanced, 6),
        },
        "covered_only_binary_metric": {
            "name": "binary_metric_excluding_uncertain_predictions",
            "covered_pair_count": int(len(covered)),
            "same_event_precision": round(float(p_cov[0]), 6),
            "same_event_recall": round(float(r_cov[0]), 6),
            "same_event_f1": round(float(f_cov[0]), 6),
            "different_event_precision": round(float(p_cov[1]), 6),
            "different_event_recall": round(float(r_cov[1]), 6),
            "different_event_f1": round(float(f_cov[1]), 6),
            "binary_macro_f1": round(float((f_cov[0] + f_cov[1]) / 2.0), 6),
            "accuracy": None if covered_accuracy is None else round(covered_accuracy, 6),
            "balanced_accuracy": None if covered_balanced is None else round(covered_balanced, 6),
        },
        "accuracy": {
            "exact_label_accuracy_keep_uncertain": round(exact_accuracy, 6),
            "exact_label_balanced_accuracy_keep_uncertain": round(exact_balanced, 6),
            "closed_binary_accuracy_abstain_as_different": round(closed_accuracy, 6),
            "closed_binary_balanced_accuracy_abstain_as_different": round(closed_balanced, 6),
        },
        "coverage": {
            "non_abstained_predictions": int(len(covered)),
            "total_pairs": int(len(df)),
            "coverage": round(float(len(covered) / max(1, len(df))), 6),
            "abstention_count": int((df["prediction"] == "uncertain").sum()),
            "abstention_rate": round(float((df["prediction"] == "uncertain").mean()), 6),
        },
        "false_merge": fp,
        "split_error": fn,
    }


def build_error16(df: pd.DataFrame) -> pd.DataFrame:
    errors = df[
        ((df["gold_label"] == "same_event") & (df["prediction"] != "same_event"))
        | ((df["gold_label"] == "different_event") & (df["prediction"] == "same_event"))
    ].copy()
    if len(errors) != 16:
        raise RuntimeError(f"expected 16 formal errors, got {len(errors)}")

    rows = []
    for row in errors.itertuples(index=False):
        pair_id = text(getattr(row, "pair_id"))
        cause = CAUSE_OVERRIDES.get(pair_id)
        if cause is None:
            raise RuntimeError(f"missing root cause override for {pair_id}")
        prediction = text(getattr(row, "prediction"))
        gold = text(getattr(row, "gold_label"))
        cluster_relation = text(getattr(row, "cluster_relation_from_dev_clusters"))
        if prediction == gold:
            cluster_effect = "no_pair_error"
        elif cluster_relation == gold:
            cluster_effect = "cluster_would_correct_pair_error"
        elif cluster_relation != prediction:
            cluster_effect = "cluster_changed_but_still_wrong"
        else:
            cluster_effect = "no_cluster_change"
        rows.append(
            {
                "pair_id": pair_id,
                "gold_label": gold,
                "gold_label_source": text(getattr(row, "gold_label_source")),
                "prediction": prediction,
                "probability": float(getattr(row, "confidence")),
                "prob_same_event": float(getattr(row, "prob_same_event")),
                "prob_different_event": float(getattr(row, "prob_different_event")),
                "prob_uncertain": float(getattr(row, "prob_uncertain")),
                "error_type": "split_error" if gold == "same_event" else "false_merge",
                "primary_root_cause": cause["primary"],
                "secondary_causes": "|".join(cause["secondary"]),
                "root_cause_rationale": cause["rationale"],
                "candidate_hit": bool(getattr(row, "candidate_hit")),
                "candidate_generation_miss": not bool(getattr(row, "candidate_hit")),
                "candidate_generation_blockers": text(getattr(row, "candidate_generation_blockers")),
                "R1": bool(getattr(row, "R1")),
                "R2": bool(getattr(row, "R2")),
                "R3": bool(getattr(row, "R3")),
                "R4": bool(getattr(row, "R4")),
                "r1_reasons": text(getattr(row, "r1_reasons")),
                "r2_reasons": text(getattr(row, "r2_reasons")),
                "r3_reasons": text(getattr(row, "r3_reasons")),
                "r4_reasons": text(getattr(row, "r4_reasons")),
                "same_thr": SAME_THR,
                "diff_thr": DIFF_THR,
                "margin": MARGIN,
                "scorer_applied": bool(getattr(row, "candidate_hit")),
                "threshold_decision_note": threshold_note(row),
                "key_feature_fact_similarity": value_or_raw(row, "candidate_fact_similarity", "raw_fact_similarity"),
                "key_feature_object_similarity": value_or_raw(row, "candidate_object_similarity", "raw_object_similarity"),
                "key_feature_trigger_similarity": value_or_raw(row, "candidate_trigger_similarity", "raw_trigger_similarity"),
                "key_feature_evidence_similarity": value_or_raw(row, "candidate_evidence_similarity", "raw_evidence_similarity"),
                "key_feature_time_relation": text(getattr(row, "candidate_time_relation")) or text(getattr(row, "raw_time_relation")),
                "key_feature_hard_conflicts": text(getattr(row, "candidate_hard_conflicts")),
                "pair_prediction_changed_by_cluster": bool(getattr(row, "pair_prediction_changed_by_cluster")),
                "cluster_relation_from_dev_clusters": cluster_relation,
                "cluster_change_effect": cluster_effect,
                "cluster_merge_action": text(getattr(row, "cluster_merge_action")),
                "cluster_veto_reason": text(getattr(row, "cluster_veto_reason")),
                "cluster_transitivity_error": bool(getattr(row, "cluster_introduced_error")),
                "gold_ambiguous_or_evidence_insufficient": bool(cause["gold_ambiguous"]),
                "event_a_target": f"{text(getattr(row, 'event_a_target_key'))} {text(getattr(row, 'event_a_target_name'))}",
                "event_b_target": f"{text(getattr(row, 'event_b_target_key'))} {text(getattr(row, 'event_b_target_name'))}",
                "event_a_type": text(getattr(row, "event_a_event_type")),
                "event_b_type": text(getattr(row, "event_b_event_type")),
                "event_a_family": text(getattr(row, "event_a_event_family")),
                "event_b_family": text(getattr(row, "event_b_event_family")),
                "event_a_time": text(getattr(row, "event_a_event_time")) or text(getattr(row, "event_a_publish_time")),
                "event_b_time": text(getattr(row, "event_b_event_time")) or text(getattr(row, "event_b_publish_time")),
                "event_a_arguments": text(getattr(row, "event_a_object_text")),
                "event_b_arguments": text(getattr(row, "event_b_object_text")),
                "event_a_fact": text(getattr(row, "event_a_event_fact")),
                "event_b_fact": text(getattr(row, "event_b_event_fact")),
                "event_a_evidence": text(getattr(row, "event_a_evidence_sentence")),
                "event_b_evidence": text(getattr(row, "event_b_evidence_sentence")),
            }
        )
    return pd.DataFrame(rows).sort_values(["error_type", "primary_root_cause", "pair_id"])


def value_or_raw(row: Any, cand_col: str, raw_col: str) -> Any:
    cand = getattr(row, cand_col, np.nan)
    if cand == cand:
        return cand
    return getattr(row, raw_col, np.nan)


def threshold_note(row: Any) -> str:
    if not bool(getattr(row, "candidate_hit")):
        return "scorer_not_applied_non_candidate_default_different"
    pred = text(getattr(row, "prediction"))
    p_same = float(getattr(row, "prob_same_event"))
    p_diff = float(getattr(row, "prob_different_event"))
    p_unc = float(getattr(row, "prob_uncertain"))
    if pred == "same_event":
        return f"p_same={p_same:.6f} >= same_thr={SAME_THR:.2f}"
    if pred == "different_event":
        return f"p_diff={p_diff:.6f} >= diff_thr={DIFF_THR:.2f}; p_same={p_same:.6f}"
    return f"abstained_uncertain; p_same={p_same:.6f}, p_diff={p_diff:.6f}, p_uncertain={p_unc:.6f}, same_thr={SAME_THR:.2f}, diff_thr={DIFF_THR:.2f}"


def candidate_oracle(df: pd.DataFrame, error16: pd.DataFrame) -> dict[str, Any]:
    same = df[df["gold_label"] == "same_event"]
    hit = int(same["candidate_hit"].sum())
    miss = int(len(same) - hit)
    split = error16[error16["error_type"] == "split_error"]
    return {
        "status": "formal_dev109_candidate_oracle_recall",
        "positive_denominator": "87 human same_event dev pairs only",
        "human_same_event_pair_count": int(len(same)),
        "candidate_hit_count": hit,
        "candidate_miss_count": miss,
        "candidate_oracle_recall": round(float(hit / max(1, len(same))), 6),
        "split_error_count": int(len(split)),
        "split_error_candidate_generation_miss_count": int((~split["candidate_hit"]).sum()),
        "split_error_candidate_hit_but_resolver_not_same_count": int(split["candidate_hit"].sum()),
        "legacy_candidate_hit_323_note": "323 is the old full-dev candidate-hit count over 7140 dev pairs; it is not a formal candidate recall denominator.",
    }


def build_pair_expansion_candidates(row_df: pd.DataFrame, error16: pd.DataFrame, max_uncertain: int = 139) -> pd.DataFrame:
    predictions = pd.read_parquet(SRC / "dev_pair_predictions.parquet")
    predictions["pair_key"] = predictions.apply(lambda row: pair_key(row["sample_id_a"], row["sample_id_b"]), axis=1)
    labeled_keys = set(row_df["pair_key"].str.split(r"\|\|").map(lambda parts: pair_key(parts[0], parts[1])))
    candidates = build_candidate_map()
    events = read_dev_events()

    selected_keys: dict[tuple[str, str], str] = {}
    for row in error16.itertuples(index=False):
        selected_keys[pair_key(getattr(row, "event_a_target", ""), getattr(row, "event_b_target", ""))] = ""
    # Use formal error sample ids from row_df for exact keys.
    for row in row_df[row_df["evaluation"].isin(["FP_false_merge", "FN_split_error"])].itertuples(index=False):
        selected_keys[pair_key(row.sample_id_a, row.sample_id_b)] = f"formal_error_{row.evaluation}"

    unlabeled = predictions[(~predictions["pair_key"].isin(labeled_keys)) & (predictions["candidate_hit"].astype(bool))].copy()
    same_unlabeled = unlabeled[unlabeled["prediction"] == "same_event"].copy()
    uncertain_unlabeled = unlabeled[unlabeled["prediction"] == "uncertain"].copy()
    uncertain_unlabeled = uncertain_unlabeled.sort_values(["prob_same_event", "prob_uncertain"], ascending=False).head(max_uncertain)
    diff_unlabeled = unlabeled[unlabeled["prediction"] == "different_event"].copy().sort_values("prob_different_event", ascending=False)

    for row in same_unlabeled.itertuples(index=False):
        selected_keys.setdefault(row.pair_key, "unlabeled_high_score_same_prediction_review")
    for row in uncertain_unlabeled.itertuples(index=False):
        selected_keys.setdefault(row.pair_key, "unlabeled_uncertain_boundary_review")
    for row in diff_unlabeled.head(40).itertuples(index=False):
        selected_keys.setdefault(row.pair_key, "unlabeled_hard_negative_or_boundary_review")

    rows = []
    pred_map = {row.pair_key: row._asdict() for row in predictions.itertuples(index=False)}
    for idx, (key, reason) in enumerate(sorted(selected_keys.items()), start=1):
        if not reason:
            continue
        pred = pred_map.get(key, {})
        cand = candidates.get(key, {})
        a, b = key
        eva = events.get(a, {})
        evb = events.get(b, {})
        rows.append(
            {
                "annotation_id": f"V3_1_PAIR_{idx:04d}",
                "selection_bucket": reason,
                "annotator_relation": "",
                "annotator_confidence": "",
                "conflict_type": "",
                "notes": "",
                "sample_id_a": a,
                "sample_id_b": b,
                "candidate_channels": "|".join([c for c in ["R1", "R2", "R3", "R4"] if bool(cand.get(c))]),
                "event_a_article_id": text(eva.get("article_id")),
                "event_a_target_key": text(eva.get("target_key")),
                "event_a_target_name": text(eva.get("target_name")),
                "event_a_event_family": text(eva.get("event_family")),
                "event_a_event_type": text(eva.get("event_type")),
                "event_a_trigger": text(eva.get("trigger_text")),
                "event_a_arguments": text(eva.get("object_text")),
                "event_a_event_time": text(eva.get("event_time")),
                "event_a_publish_time": text(eva.get("publish_time")),
                "event_a_fact": text(eva.get("event_fact")),
                "event_a_evidence": text(eva.get("evidence_sentence")),
                "event_b_article_id": text(evb.get("article_id")),
                "event_b_target_key": text(evb.get("target_key")),
                "event_b_target_name": text(evb.get("target_name")),
                "event_b_event_family": text(evb.get("event_family")),
                "event_b_event_type": text(evb.get("event_type")),
                "event_b_trigger": text(evb.get("trigger_text")),
                "event_b_arguments": text(evb.get("object_text")),
                "event_b_event_time": text(evb.get("event_time")),
                "event_b_publish_time": text(evb.get("publish_time")),
                "event_b_fact": text(evb.get("event_fact")),
                "event_b_evidence": text(evb.get("evidence_sentence")),
            }
        )
    return pd.DataFrame(rows)


def human_component_review_sets(row_df: pd.DataFrame) -> dict[str, str]:
    ids = sorted(set(row_df["sample_id_a"].astype(str)) | set(row_df["sample_id_b"].astype(str)))
    parent = {uid: uid for uid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for row in row_df[row_df["gold_label"] == "same_event"].itertuples(index=False):
        union(text(row.sample_id_a), text(row.sample_id_b))
    roots = {uid: find(uid) for uid in ids}
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for uid, root in roots.items():
        groups[root].append(uid)
    mapping = {}
    for idx, root in enumerate(sorted(groups), start=1):
        review_id = f"V3_1_CLUSTER_REVIEW_{idx:04d}"
        for uid in groups[root]:
            mapping[uid] = review_id
    return mapping


def build_cluster_member_candidates(row_df: pd.DataFrame) -> pd.DataFrame:
    events = read_dev_events()
    review_sets = human_component_review_sets(row_df)
    rows = []
    for uid, ev in sorted(events.items()):
        rows.append(
            {
                "mention_id": uid,
                "suggested_review_set_id": review_sets.get(uid, f"V3_1_CLUSTER_REVIEW_SINGLETON_{uid[-8:]}"),
                "canonical_event_gold_id": "",
                "annotator_notes": "",
                "article_id": text(ev.get("article_id")),
                "source_name": text(ev.get("source_name")),
                "target_key": text(ev.get("target_key")),
                "target_name": text(ev.get("target_name")),
                "event_family": text(ev.get("event_family")),
                "event_type": text(ev.get("event_type")),
                "event_name": text(ev.get("event_name")),
                "trigger_text": text(ev.get("trigger_text")),
                "arguments": text(ev.get("object_text")),
                "event_time": text(ev.get("event_time")),
                "publish_time": text(ev.get("publish_time")),
                "event_fact_signature": text(ev.get("event_fact_signature")),
                "event_fact": text(ev.get("event_fact")),
                "original_evidence": text(ev.get("evidence_sentence")),
            }
        )
    return pd.DataFrame(rows)


def write_workbooks(out: Path, row_df: pd.DataFrame, metrics: dict[str, Any], error16: pd.DataFrame, pair_pkg: pd.DataFrame, cluster_pkg: pd.DataFrame) -> None:
    summary_rows = flatten_metric_summary(metrics)
    with pd.ExcelWriter(out / "dev109_row_level_evaluation.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="metric_summary", index=False)
        row_df.to_excel(writer, sheet_name="row_level_109", index=False)
        confusion_rows(metrics).to_excel(writer, sheet_name="confusion_matrices", index=False)

    root_summary = []
    for error_type, sub in error16.groupby("error_type"):
        for cause, n in sub["primary_root_cause"].value_counts().sort_index().items():
            root_summary.append({"error_type": error_type, "primary_root_cause": cause, "count": int(n)})
    stage_summary = [
        {"item": "formal_error_count", "value": int(len(error16))},
        {"item": "split_error_count", "value": int((error16["error_type"] == "split_error").sum())},
        {"item": "false_merge_count", "value": int((error16["error_type"] == "false_merge").sum())},
        {"item": "candidate_generation_miss_count", "value": int((error16["candidate_generation_miss"]).sum())},
        {"item": "candidate_hit_error_count", "value": int((error16["candidate_hit"]).sum())},
        {"item": "cluster_transitivity_error_count", "value": int((error16["cluster_transitivity_error"]).sum())},
    ]
    with pd.ExcelWriter(out / "formal_error16_root_cause.xlsx", engine="openpyxl") as writer:
        error16.to_excel(writer, sheet_name="error16_root_cause", index=False)
        pd.DataFrame(root_summary).to_excel(writer, sheet_name="root_cause_summary", index=False)
        pd.DataFrame(stage_summary).to_excel(writer, sheet_name="stage_summary", index=False)

    pair_instructions = pd.DataFrame(
        [
            {"field": "annotator_relation", "instruction": "Fill same_event, different_event, or uncertain."},
            {"field": "conflict_type", "instruction": "Optional: time_boundary, family_boundary, argument_mismatch, entity_identity, evidence_insufficient, other."},
            {"field": "notes", "instruction": "Do not use model predictions as answers; inspect the two event facts and evidence."},
        ]
    )
    with pd.ExcelWriter(out / "resolver_v3_1_pair_gold_expansion_candidates.xlsx", engine="openpyxl") as writer:
        pair_instructions.to_excel(writer, sheet_name="instructions", index=False)
        pair_pkg.to_excel(writer, sheet_name="pair_candidates", index=False)

    cluster_instructions = pd.DataFrame(
        [
            {"field": "canonical_event_gold_id", "instruction": "Blank by design. Human annotators assign a non-official canonical event id."},
            {"field": "suggested_review_set_id", "instruction": "Review grouping aid from non-official dev human pair components, not a model answer."},
            {"field": "original_evidence", "instruction": "Use evidence and fact text to decide final cluster membership."},
        ]
    )
    with pd.ExcelWriter(out / "resolver_v3_1_cluster_member_gold_candidates.xlsx", engine="openpyxl") as writer:
        cluster_instructions.to_excel(writer, sheet_name="instructions", index=False)
        cluster_pkg.to_excel(writer, sheet_name="cluster_member_candidates", index=False)


def flatten_metric_summary(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section, value in metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                rows.append({"section": section, "item": k, "value": json.dumps(to_jsonable(v), ensure_ascii=False) if isinstance(v, (dict, list)) else v})
        else:
            rows.append({"section": "top_level", "item": section, "value": value})
    return rows


def confusion_rows(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name in [
        "three_label_confusion_matrix",
        "closed_binary_confusion_matrix_abstain_as_different",
        "covered_only_confusion_matrix_excluding_uncertain_predictions",
    ]:
        matrix = metrics[name]
        for gold, cols in matrix.items():
            row = {"matrix": name, "gold_label": gold}
            row.update(cols)
            rows.append(row)
    return pd.DataFrame(rows)


def build_markdown(metric_def: dict[str, Any], candidate_recall: dict[str, Any], error16: pd.DataFrame, pair_pkg: pd.DataFrame, cluster_pkg: pd.DataFrame) -> str:
    hist = metric_def["metrics"]["historical_sklearn_binary_labels_metric"]
    closed = metric_def["metrics"]["standard_closed_binary_metric"]
    coverage = metric_def["metrics"]["coverage"]
    split_dist = error16[error16["error_type"] == "split_error"]["primary_root_cause"].value_counts().sort_index().to_dict()
    fm_dist = error16[error16["error_type"] == "false_merge"]["primary_root_cause"].value_counts().sort_index().to_dict()
    return f"""# Task3 Resolver v3 Metric Closure

## Scope

- final_holdout read: 0
- official1799 filled Gold or prediction read: 0
- Resolver rerun/training/threshold tuning: 0
- Formal metric denominator: 109 human dev pairs only
- Legacy 7140 default-negative metrics are excluded from formal Dev performance

## Metric Closure

- Actual prediction labels: {metric_def['summary']['actual_prediction_label_set']}
- same-vs-rest counts: {metric_def['metrics']['same_vs_rest_binary_counts']}
- Historical macro-F1 name: `{hist['name']}`
- Historical macro-F1: {hist['binary_macro_f1']}
- same-event F1 / different-event F1 under historical sklearn labels: {hist['same_event_f1']} / {hist['different_event_f1']}
- Standard closed-binary macro-F1: {closed['binary_macro_f1']}
- Exact-label accuracy: {metric_def['metrics']['accuracy']['exact_label_accuracy_keep_uncertain']}
- Closed-binary accuracy: {closed['accuracy']}
- Closed-binary balanced accuracy: {closed['balanced_accuracy']}
- Coverage / abstention rate: {coverage['coverage']} / {coverage['abstention_rate']}

The historical 0.701220 is `(same_event_F1 0.902439 + different_event_F1 0.500000) / 2`.
It uses `f1_score(y_true, y_pred, labels=['same_event','different_event'], average='macro')` while keeping `uncertain` predictions in `y_pred`.
Those `uncertain` predictions lower recall for the true class but are not averaged as their own class.

## Candidate Oracle Recall

- Human same-event denominator: {candidate_recall['human_same_event_pair_count']}
- Candidate hit / miss: {candidate_recall['candidate_hit_count']} / {candidate_recall['candidate_miss_count']}
- Candidate oracle recall: {candidate_recall['candidate_oracle_recall']}
- Split errors from candidate miss / candidate-hit scorer stage: {candidate_recall['split_error_candidate_generation_miss_count']} / {candidate_recall['split_error_candidate_hit_but_resolver_not_same_count']}

## Error Root Causes

- Split error primary roots: {dict(split_dist)}
- False merge primary roots: {dict(fm_dist)}
- Candidate generation misses among 16 errors: {int(error16['candidate_generation_miss'].sum())}
- Candidate-hit errors among 16 errors: {int(error16['candidate_hit'].sum())}
- Primary scorer_error count: {int((error16['primary_root_cause'] == 'scorer_error').sum())}
- Cluster transitivity introduced errors: {int(error16['cluster_transitivity_error'].sum())}

## New Annotation Candidates

- Pair expansion candidates: {len(pair_pkg)}
- Cluster/member candidates: {len(cluster_pkg)}

Generated artifacts in this directory are audit-only. No frozen prediction/model files are modified.
"""


def write_readme(out: Path, metric_def: dict[str, Any], candidate_recall: dict[str, Any], error16: pd.DataFrame, pair_pkg: pd.DataFrame, cluster_pkg: pd.DataFrame) -> None:
    readme = build_markdown(metric_def, candidate_recall, error16, pair_pkg, cluster_pkg)
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "metric_definition_audit.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    global SRC, OUT
    SRC = args.source
    OUT = args.out

    OUT.mkdir(parents=True, exist_ok=True)
    before = metadata_snapshot(protected_paths(Path(__file__)))
    write_json(OUT / "protected_sha256_before.json", before)

    row_df, summary, io_manifest = build_row_level()
    metrics = metrics_for_dev109(row_df)
    error16 = build_error16(row_df)
    cand_recall = candidate_oracle(row_df, error16)
    pair_pkg = build_pair_expansion_candidates(row_df, error16)
    cluster_pkg = build_cluster_member_candidates(row_df)

    metric_def = {
        "status": "formal_dev109_metric_definition_audit_complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": io_manifest,
        "summary": summary,
        "metrics": metrics,
        "formal_policy": {
            "unlabeled_pairs_defaulted_to_different_event": False,
            "legacy_7140_statistics_used_as_formal_gold_performance": False,
            "cluster_member_gold_available": False,
            "cluster_member_gold_note": "No non-official complete cluster/member Gold was available; official/final-holdout Gold files were not read.",
        },
        "official1799_read": 0,
        "final_holdout_read": 0,
        "resolver_rerun": 0,
        "training_or_threshold_tuning": 0,
    }
    write_json(OUT / "metric_definition_audit.json", metric_def)
    write_json(OUT / "dev109_confusion_matrix.json", metrics)
    write_json(OUT / "candidate_oracle_recall.json", cand_recall)
    write_json(
        OUT / "formal_error16_root_cause.json",
        {
            "status": "formal_error16_root_cause_complete",
            "root_cause_distribution": {
                "split_error": {
                    k: int(v)
                    for k, v in error16[error16["error_type"] == "split_error"]["primary_root_cause"].value_counts().sort_index().to_dict().items()
                },
                "false_merge": {
                    k: int(v)
                    for k, v in error16[error16["error_type"] == "false_merge"]["primary_root_cause"].value_counts().sort_index().to_dict().items()
                },
            },
            "scorer_error_primary_count": int((error16["primary_root_cause"] == "scorer_error").sum()),
            "candidate_recall_miss_primary_count": int((error16["primary_root_cause"] == "candidate_recall_miss").sum()),
            "cluster_transitivity_error_count": int(error16["cluster_transitivity_error"].sum()),
            "rows": error16.to_dict("records"),
        },
    )
    write_workbooks(OUT, row_df, metrics, error16, pair_pkg, cluster_pkg)
    write_readme(OUT, metric_def, cand_recall, error16, pair_pkg, cluster_pkg)

    reproduce = OUT / "reproduce_metric_closure.sh"
    reproduce.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /opt/data/private/JinRong/kechuang/financial_event_project\n"
        "/root/miniconda3/envs/dianjin-cu128/bin/python scripts/audit_task3_resolver_v3_metric_closure.py\n",
        encoding="utf-8",
    )
    reproduce.chmod(0o755)

    after = metadata_snapshot(protected_paths(Path(__file__)))
    write_json(OUT / "protected_sha256_after.json", after)
    write_json(
        OUT / "run_manifest.json",
        {
            "status": "complete",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": rel(OUT),
            "required_outputs_present": {
                "metric_definition_audit.json": (OUT / "metric_definition_audit.json").exists(),
                "metric_definition_audit.md": (OUT / "metric_definition_audit.md").exists(),
                "dev109_row_level_evaluation.xlsx": (OUT / "dev109_row_level_evaluation.xlsx").exists(),
                "dev109_confusion_matrix.json": (OUT / "dev109_confusion_matrix.json").exists(),
                "formal_error16_root_cause.xlsx": (OUT / "formal_error16_root_cause.xlsx").exists(),
                "candidate_oracle_recall.json": (OUT / "candidate_oracle_recall.json").exists(),
                "resolver_v3_1_pair_gold_expansion_candidates.xlsx": (OUT / "resolver_v3_1_pair_gold_expansion_candidates.xlsx").exists(),
                "resolver_v3_1_cluster_member_gold_candidates.xlsx": (OUT / "resolver_v3_1_cluster_member_gold_candidates.xlsx").exists(),
                "protected_sha256_before.json": (OUT / "protected_sha256_before.json").exists(),
                "protected_sha256_after.json": (OUT / "protected_sha256_after.json").exists(),
                "README.md": (OUT / "README.md").exists(),
                "reproduce_metric_closure.sh": (OUT / "reproduce_metric_closure.sh").exists(),
            },
            "protected_metadata_equal_before_after": before["metadata_sha256"] == after["metadata_sha256"],
            "official1799_read": 0,
            "final_holdout_read": 0,
            "resolver_rerun": 0,
            "training_or_threshold_tuning": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(OUT),
                "historical_macro_f1": metrics["historical_sklearn_binary_labels_metric"]["binary_macro_f1"],
                "standard_closed_binary_macro_f1": metrics["standard_closed_binary_metric"]["binary_macro_f1"],
                "candidate_oracle_recall": cand_recall["candidate_oracle_recall"],
                "split_root_distribution": error16[error16["error_type"] == "split_error"]["primary_root_cause"].value_counts().sort_index().to_dict(),
                "false_merge_root_distribution": error16[error16["error_type"] == "false_merge"]["primary_root_cause"].value_counts().sort_index().to_dict(),
                "pair_gold_expansion_candidates": len(pair_pkg),
                "cluster_member_gold_candidates": len(cluster_pkg),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
