#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path("/opt/data/private/JinRong/kechuang/financial_event_project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.task4_trace_impact_modeling import clean_text, numeric_value  # noqa: E402


READINESS = ROOT / "outputs/task4_trace_impact_training_readiness_v1"
TRAIN_OUT = ROOT / "outputs/task4_trace_impact_v1_training"
OUT = ROOT / "outputs/task4_trace_impact_v1_final_acceptance"
MODEL_PATH = TRAIN_OUT / "models/task4_trace_impact_v1_model_bundle.joblib"
INFERENCE_PATH = TRAIN_OUT / "inference_pipeline.py"
UNIFIED_PATH = READINESS / "task4_unified_gold_silver.parquet"

SEED = 20260827
CREATED_AT = datetime.now(timezone(timedelta(hours=8))).isoformat()

RAW_ALLOWED_FIELDS = [
    "sample_id",
    "article_id",
    "content_hash",
    "canonical_event_id",
    "target",
    "target_type",
    "target_code",
    "event_name",
    "event_type",
    "evidence",
    "title",
    "local_context",
    "financial_facts",
    "publish_time",
]

ANSWER_OR_PROVENANCE_FIELDS = [
    "event_applicable",
    "target_effect_exists",
    "evidence_sufficient",
    "score_evaluable",
    "impact_directness",
    "impact_channel",
    "target_role",
    "direction_short",
    "direction_long",
    "event_shock_severity",
    "target_exposure",
    "target_impact_severity",
    "scope_axis",
    "urgency_axis",
    "novelty_axis",
    "evolution_axis",
    "annotation_confidence",
    "annotation_source",
    "label_tier",
    "task_membership",
    "eligible_for_direction_v2",
    "eligible_for_impact_ordinal",
    "eligible_for_applicability_gate",
    "gate_negative_reason",
    "legacy_risk_intensity",
    "legacy_severity_label",
    "source_dataset",
    "source_path",
    "source_sheet",
    "source_row_number",
    "review_status",
    "label_priority",
    "excluded_reason",
    "training_bucket",
]

MODEL_E5_INFERRED_FEATURE_FIELDS = [
    "title",
    "evidence",
    "target",
    "target_type",
    "target_code",
    "event_name",
    "event_type",
    "structured_event",
    "impact_channel",
    "impact_directness",
    "target_role",
    "target_effect_exists",
    "evidence_sufficient",
    "event_shock_severity",
    "target_exposure",
    "scope_axis",
    "urgency_axis",
    "novelty_axis",
    "evolution_axis",
    "legacy_aux_risk_bin",
    "legacy_aux_high_bin",
    "legacy_aux_expected_risk",
    "legacy_aux_high_prob",
    "credibility_level",
    "verification_status",
    "task3_current_state_type",
    "task3_verification_status",
    "title_len",
    "evidence_len",
    "target_len",
    "event_len",
    "event_shock_num",
    "target_exposure_num",
    "scope_num",
    "urgency_num",
    "novelty_num",
    "evolution_num",
    "credibility_score",
    "source_entropy",
    "member_count",
    "cluster_cohesion_score",
    "fulltext_credibility_score",
    "task3_thread_size",
    "task3_graph_density",
    "task3_trust_score",
    "task2_feature_available",
    "task3_feature_available",
]

DIRECTION_LABELS = ["negative", "neutral", "positive", "mixed"]
ORDINAL_LABELS = [1, 2, 3, 4, 5]
AXIS_FIELDS = [
    "event_shock_severity",
    "target_exposure",
    "scope_axis",
    "urgency_axis",
    "novelty_axis",
    "evolution_axis",
]
PROTECTED_FILES = {
    "task4_trace_v1_model_bundle": MODEL_PATH,
    "task4_trace_v1_inference_pipeline": INFERENCE_PATH,
    "task4_trace_v1_training_manifest": TRAIN_OUT / "training_manifest.json",
    "task4_trace_v1_frozen_manifest": TRAIN_OUT / "frozen_model_manifest.json",
    "readiness_unified": UNIFIED_PATH,
    "readiness_split_leakage_report": READINESS / "split_leakage_report.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, set):
        return [json_ready(v) for v in sorted(value)]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def protected_sha256() -> dict[str, Any]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": int(path.stat().st_size) if path.exists() else None,
        }
        for name, path in PROTECTED_FILES.items()
    }


def output_sha256() -> dict[str, Any]:
    rows = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows[str(path.relative_to(OUT))] = {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
    return rows


def value_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.fillna("").map(clean_text).value_counts(dropna=False).to_dict().items()}


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray | list[Any], labels: list[str]) -> dict[str, Any]:
    yt = np.asarray([clean_text(v) for v in y_true])
    yp = np.asarray([clean_text(v) for v in y_pred])
    present = [label for label in labels if label in set(yt) or label in set(yp)]
    if not present:
        present = labels
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=present, zero_division=0)
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)) if len(yt) else 0.0,
        "macro_f1": float(f1_score(yt, yp, labels=present, average="macro", zero_division=0)) if len(yt) else 0.0,
        "per_class": {
            label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, label in enumerate(present)
        },
        "confusion_matrix": {"labels": present, "matrix": confusion_matrix(yt, yp, labels=present).astype(int).tolist()},
    }


def ordinal_metrics(y_true: pd.Series, y_pred: np.ndarray | list[Any]) -> dict[str, Any]:
    values = [(clean_text(a), clean_text(b)) for a, b in zip(y_true, y_pred)]
    values = [(a, b) for a, b in values if a in {"1", "2", "3", "4", "5"} and b in {"1", "2", "3", "4", "5"}]
    if not values:
        return {"n": 0}
    yt = np.asarray([int(a) for a, _ in values], dtype=int)
    yp = np.asarray([int(b) for _, b in values], dtype=int)
    rho = spearmanr(yt, yp).statistic if len(set(yt)) > 1 and len(set(yp)) > 1 else np.nan
    high_true = yt >= 4
    high_pred = yp >= 4
    return {
        "n": int(len(yt)),
        "qwk": float(cohen_kappa_score(yt, yp, weights="quadratic", labels=ORDINAL_LABELS)),
        "mae": float(mean_absolute_error(yt, yp)),
        "spearman": float(rho) if rho is not None and math.isfinite(float(rho)) else None,
        "pm1_accuracy": float(np.mean(np.abs(yt - yp) <= 1)),
        "exact_accuracy": float(np.mean(yt == yp)),
        "high_impact_f1": float(f1_score(high_true, high_pred, zero_division=0)),
        "confusion_matrix": {"labels": ORDINAL_LABELS, "matrix": confusion_matrix(yt, yp, labels=ORDINAL_LABELS).astype(int).tolist()},
    }


def nonempty_set(frame: pd.DataFrame, field: str) -> set[str]:
    if field not in frame.columns:
        return set()
    return {clean_text(v) for v in frame[field] if clean_text(v)}


def select_nonofficial_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = df[df["training_bucket"].isin(["direction_v2_silver_a_train", "impact_ordinal_silver_a_train", "applicability_gate_train"])].copy()
    candidates = df[(df["label_tier"].eq("Silver-B")) & (df["training_bucket"].eq("excluded_or_review"))].copy()
    audit: dict[str, Any] = {
        "candidate_policy": "Silver-B rows excluded from main supervision; official1799 and final_holdout not read.",
        "train_rows": int(len(train)),
        "silver_b_candidates": int(len(candidates)),
        "key_overlap_removed": {},
    }
    mask = pd.Series(True, index=candidates.index)
    for field in ["sample_id", "article_id", "content_hash", "canonical_event_id", "dedup_key", "split_group_key"]:
        blocked = nonempty_set(train, field)
        overlap = candidates[field].map(lambda v: clean_text(v) in blocked if clean_text(v) else False) if field in candidates.columns else pd.Series(False, index=candidates.index)
        audit["key_overlap_removed"][field] = int(overlap.sum())
        mask &= ~overlap
    holdout = candidates[mask].copy()
    train_targets = nonempty_set(train, "target")
    target_overlap = holdout["target"].map(lambda v: clean_text(v) in train_targets if clean_text(v) else False) if "target" in holdout.columns else pd.Series(False, index=holdout.index)
    audit["holdout_rows_after_core_key_filter"] = int(len(holdout))
    audit["target_overlap_rows_after_core_key_filter"] = int(target_overlap.sum())
    audit["strict_target_isolated_rows"] = int((~target_overlap).sum())
    audit["strict_target_isolated_direction_rows"] = int(holdout[~target_overlap]["direction_short"].isin(DIRECTION_LABELS).sum()) if "direction_short" in holdout.columns else 0
    audit["strict_target_isolated_impact_rows"] = int(holdout[~target_overlap]["target_impact_severity"].isin(["1", "2", "3", "4", "5"]).sum()) if "target_impact_severity" in holdout.columns else 0
    audit["holdout_source_distribution"] = value_counts(holdout["source_dataset"]) if "source_dataset" in holdout.columns else {}
    return holdout, audit


def gate_gold(frame: pd.DataFrame) -> np.ndarray:
    yes = (
        frame["event_applicable"].map(clean_text).eq("yes")
        & frame["target_effect_exists"].map(clean_text).eq("yes")
        & frame["evidence_sufficient"].map(clean_text).eq("yes")
        & frame["score_evaluable"].map(clean_text).eq("yes")
    )
    return np.where(yes, "auto_score", "abstain")


def raw_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for field in RAW_ALLOWED_FIELDS:
        out[field] = frame[field] if field in frame.columns else ""
    for field in ANSWER_OR_PROVENANCE_FIELDS:
        out[field] = ""
    out["structured_event"] = (
        out["target"].fillna("").map(clean_text)
        + "|"
        + out["target_type"].fillna("").map(clean_text)
        + "|"
        + out["event_name"].fillna("").map(clean_text)
        + "|"
        + out["evidence"].fillna("").map(clean_text)
    )
    out["title_len"] = out["title"].fillna("").map(clean_text).str.len()
    out["evidence_len"] = out["evidence"].fillna("").map(clean_text).str.len()
    out["target_len"] = out["target"].fillna("").map(clean_text).str.len()
    out["event_len"] = out["event_name"].fillna("").map(clean_text).str.len()
    for field in [
        "event_shock_num",
        "target_exposure_num",
        "scope_num",
        "urgency_num",
        "novelty_num",
        "evolution_num",
        "credibility_score",
        "source_entropy",
        "member_count",
        "cluster_cohesion_score",
        "fulltext_credibility_score",
        "task3_thread_size",
        "task3_graph_density",
        "task3_trust_score",
        "task2_feature_available",
        "task3_feature_available",
        "legacy_aux_expected_risk",
        "legacy_aux_high_prob",
    ]:
        out[field] = 0.0
    for field in ["credibility_level", "verification_status", "task3_current_state_type", "task3_verification_status", "legacy_aux_risk_bin", "legacy_aux_high_bin"]:
        out[field] = ""
    return out


def predict_direct_unsafe(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["gate_prediction"] = bundle["applicability_gate_model"].predict(frame)
    out["direction_short_pred"] = bundle["direction_short_model"].predict(frame)
    out["direction_long_pred"] = bundle["direction_long_model"].predict(frame)
    for axis, model in bundle["impact_axis_models"].items():
        out[f"{axis}_pred"] = model.predict(frame)
    if "monotonic_impact_combiner" in bundle and bundle["monotonic_impact_combiner"] is not None:
        out["impact_combined_level"] = bundle["monotonic_impact_combiner"].predict(frame)
        out["impact_combined_score"] = bundle["monotonic_impact_combiner"].predict_score(frame)
    return out


def predict_sanitized_cascade(bundle: dict[str, Any], raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = raw_feature_frame(raw)
    out = pd.DataFrame(index=raw.index)
    out["sample_id"] = work["sample_id"].to_numpy()
    out["gate_prediction"] = bundle["applicability_gate_model"].predict(work)

    channel_model_available = "impact_channel_model" in bundle or "channel_model" in bundle
    directness_model_available = "impact_directness_model" in bundle or "directness_model" in bundle
    out["impact_channel_pred"] = "uncertain"
    out["impact_directness_pred"] = "uncertain"
    out["impact_channel_model_status"] = "missing_frozen_head"
    out["impact_directness_model_status"] = "missing_frozen_head"
    work["impact_channel"] = out["impact_channel_pred"]
    work["impact_directness"] = out["impact_directness_pred"]

    out["direction_short_pred"] = bundle["direction_short_model"].predict(work)
    out["direction_long_pred"] = bundle["direction_long_model"].predict(work)
    work["direction_positive"] = out["direction_short_pred"].eq("positive").astype(float)
    work["direction_negative"] = out["direction_short_pred"].eq("negative").astype(float)
    work["direction_mixed"] = out["direction_short_pred"].eq("mixed").astype(float)
    work["uncertainty_score"] = (
        out["direction_short_pred"].eq("uncertain").astype(float)
        + out["direction_long_pred"].eq("uncertain").astype(float)
        + out["gate_prediction"].ne("auto_score").astype(float)
    )

    for axis in AXIS_FIELDS:
        model = bundle["impact_axis_models"].get(axis)
        if model is None:
            out[f"{axis}_pred"] = ""
            work[f"{axis}_num"] = 0.0
            continue
        pred = model.predict(work)
        out[f"{axis}_pred"] = pred
        num_col = {
            "event_shock_severity": "event_shock_num",
            "target_exposure": "target_exposure_num",
            "scope_axis": "scope_num",
            "urgency_axis": "urgency_num",
            "novelty_axis": "novelty_num",
            "evolution_axis": "evolution_num",
        }[axis]
        work[num_col] = [numeric_value(v, 0.0) for v in pred]

    direct_model = bundle["impact_axis_models"].get("target_impact_severity")
    if direct_model is not None:
        out["target_impact_severity_direct_pred"] = direct_model.predict(work)
    else:
        out["target_impact_severity_direct_pred"] = ""
    out["target_impact_combined_level"] = bundle["monotonic_impact_combiner"].predict(work)
    out["target_impact_combined_score"] = bundle["monotonic_impact_combiner"].predict_score(work)
    out["final_decision"] = np.where(out["gate_prediction"].eq("auto_score"), "review", "abstain")
    out["review_or_abstain_reason"] = np.where(
        out["gate_prediction"].eq("auto_score"),
        "missing_frozen_channel_directness_heads",
        "applicability_gate_abstain",
    )
    diagnostics = {
        "channel_model_available": channel_model_available,
        "directness_model_available": directness_model_available,
        "downstream_gold_fields_used": False,
        "final_auto_score_coverage": 0.0,
    }
    return out, diagnostics


def permute_answer_fields(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = frame.copy()
    for field in ANSWER_OR_PROVENANCE_FIELDS:
        if field in out.columns and len(out):
            values = out[field].to_numpy(copy=True)
            rng.shuffle(values)
            out[field] = values
    return out


def changed_fraction(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    result = {}
    for col in columns:
        if col in a.columns and col in b.columns and len(a):
            result[col] = float((a[col].astype(str).to_numpy() != b[col].astype(str).to_numpy()).mean())
    return result


def source_only_baseline(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    source_col = "source_dataset"
    for task, label_col, labels in [
        ("direction_short", "direction_short", DIRECTION_LABELS),
        ("direction_long", "direction_long", DIRECTION_LABELS),
    ]:
        train_task = train[train[label_col].isin(labels)].copy()
        eval_task = holdout[holdout[label_col].isin(labels)].copy()
        if train_task.empty or eval_task.empty:
            report[task] = {"n": int(len(eval_task)), "status": "not_enough_data"}
            continue
        global_majority = Counter(train_task[label_col].map(clean_text)).most_common(1)[0][0]
        by_source = train_task.groupby(source_col)[label_col].agg(lambda s: Counter(s.map(clean_text)).most_common(1)[0][0]).to_dict()
        pred = [by_source.get(clean_text(src), global_majority) for src in eval_task[source_col]]
        report[task] = classification_metrics(eval_task[label_col], pred, labels)

    train_impact = train[train["target_impact_severity"].isin(["1", "2", "3", "4", "5"])].copy()
    eval_impact = holdout[holdout["target_impact_severity"].isin(["1", "2", "3", "4", "5"])].copy()
    if not train_impact.empty and not eval_impact.empty:
        global_majority = Counter(train_impact["target_impact_severity"].map(clean_text)).most_common(1)[0][0]
        by_source = train_impact.groupby(source_col)["target_impact_severity"].agg(lambda s: Counter(s.map(clean_text)).most_common(1)[0][0]).to_dict()
        pred = [by_source.get(clean_text(src), global_majority) for src in eval_impact[source_col]]
        report["target_impact_severity"] = ordinal_metrics(eval_impact["target_impact_severity"], pred)
    else:
        report["target_impact_severity"] = {"n": int(len(eval_impact)), "status": "not_enough_data"}
    return report


def acceptance_gate(leakage: dict[str, Any], holdout_audit: dict[str, Any], metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if holdout_audit.get("holdout_rows_after_core_key_filter", 0) <= 0:
        failures.append("no_nonofficial_holdout_after_core_key_filter")
    if leakage.get("feature_allowlist", {}).get("violating_model_fields"):
        failures.append("model_feature_schema_uses_answer_or_provenance_fields")
    if not leakage.get("pipeline_capability", {}).get("channel_model_available", False):
        failures.append("missing_frozen_impact_channel_head")
    if not leakage.get("pipeline_capability", {}).get("directness_model_available", False):
        failures.append("missing_frozen_impact_directness_head")
    if metrics.get("end_to_end", {}).get("final_auto_score_coverage", 0.0) <= 0:
        failures.append("no_final_auto_score_due_review_routing")
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and any(OUT.iterdir()) and not args.allow_existing:
        raise RuntimeError(f"output directory exists and is not empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protected_sha256_before.json", protected_sha256())

    bundle = joblib.load(MODEL_PATH)
    df = pd.read_parquet(UNIFIED_PATH).fillna("")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(clean_text)
    train = df[df["training_bucket"].isin(["direction_v2_silver_a_train", "impact_ordinal_silver_a_train", "applicability_gate_train"])].copy()
    holdout, holdout_audit = select_nonofficial_holdout(df)
    holdout = holdout.reset_index(drop=True)

    raw = raw_feature_frame(holdout)
    cascade_pred, cascade_diag = predict_sanitized_cascade(bundle, raw)
    direct_pred = predict_direct_unsafe(bundle, raw)
    unsafe_original = predict_direct_unsafe(bundle, holdout)
    unsafe_permuted = predict_direct_unsafe(bundle, permute_answer_fields(holdout))
    sanitized_original, _ = predict_sanitized_cascade(bundle, holdout)
    sanitized_permuted, _ = predict_sanitized_cascade(bundle, permute_answer_fields(holdout))

    output_cols = [
        "gate_prediction",
        "direction_short_pred",
        "direction_long_pred",
        "target_impact_severity_pred",
        "impact_combined_level",
    ]
    unsafe_shuffle = changed_fraction(unsafe_original, unsafe_permuted, [c for c in output_cols if c in unsafe_original.columns])
    sanitized_shuffle = changed_fraction(
        sanitized_original,
        sanitized_permuted,
        [
            "gate_prediction",
            "direction_short_pred",
            "direction_long_pred",
            "target_impact_combined_level",
            "final_decision",
        ],
    )

    direction_short_eval = holdout[holdout["direction_short"].isin(DIRECTION_LABELS)].copy()
    direction_long_eval = holdout[holdout["direction_long"].isin(DIRECTION_LABELS)].copy()
    impact_eval = holdout[holdout["target_impact_severity"].isin(["1", "2", "3", "4", "5"])].copy()
    gate_gold_values = gate_gold(holdout)
    gate_metrics = classification_metrics(pd.Series(gate_gold_values), cascade_pred["gate_prediction"], ["auto_score", "abstain"])
    gate_metrics["coverage"] = float(cascade_pred["gate_prediction"].eq("auto_score").mean()) if len(cascade_pred) else 0.0
    gate_metrics["abstain_rate"] = float(cascade_pred["gate_prediction"].eq("abstain").mean()) if len(cascade_pred) else 0.0

    metrics = {
        "status": "nonofficial_holdout_evaluated",
        "created_at": CREATED_AT,
        "holdout_policy": holdout_audit,
        "label_source_warning": "Silver-B is nonofficial non-human-Gold supervision/review data; it is used only as an independent acceptance stress holdout because no materialized final_holdout exists.",
        "independent_model": {
            "direction_short": classification_metrics(direction_short_eval["direction_short"], direct_pred.loc[direction_short_eval.index, "direction_short_pred"], DIRECTION_LABELS) if len(direction_short_eval) else {"n": 0},
            "direction_long": classification_metrics(direction_long_eval["direction_long"], direct_pred.loc[direction_long_eval.index, "direction_long_pred"], DIRECTION_LABELS) if len(direction_long_eval) else {"n": 0},
            "target_impact_severity": ordinal_metrics(impact_eval["target_impact_severity"], direct_pred.loc[impact_eval.index, "target_impact_severity_pred"]) if len(impact_eval) and "target_impact_severity_pred" in direct_pred else {"n": 0},
        },
        "end_to_end": {
            "direction_short": classification_metrics(direction_short_eval["direction_short"], cascade_pred.loc[direction_short_eval.index, "direction_short_pred"], DIRECTION_LABELS) if len(direction_short_eval) else {"n": 0},
            "direction_long": classification_metrics(direction_long_eval["direction_long"], cascade_pred.loc[direction_long_eval.index, "direction_long_pred"], DIRECTION_LABELS) if len(direction_long_eval) else {"n": 0},
            "target_impact_severity": ordinal_metrics(impact_eval["target_impact_severity"], cascade_pred.loc[impact_eval.index, "target_impact_combined_level"]) if len(impact_eval) else {"n": 0},
            "gate": gate_metrics,
            "final_auto_score_coverage": float(cascade_pred["final_decision"].eq("auto_score").mean()) if len(cascade_pred) else 0.0,
            "review_rate": float(cascade_pred["final_decision"].eq("review").mean()) if len(cascade_pred) else 0.0,
            "abstain_rate": float(cascade_pred["final_decision"].eq("abstain").mean()) if len(cascade_pred) else 0.0,
        },
        "source_only_baseline": source_only_baseline(train, holdout),
    }
    ind_impact = metrics["independent_model"]["target_impact_severity"]
    e2e_impact = metrics["end_to_end"]["target_impact_severity"]
    metrics["independent_to_cascade_delta"] = {
        "direction_short_macro_f1_delta": (
            metrics["independent_model"]["direction_short"].get("macro_f1", 0.0)
            - metrics["end_to_end"]["direction_short"].get("macro_f1", 0.0)
        ),
        "direction_long_macro_f1_delta": (
            metrics["independent_model"]["direction_long"].get("macro_f1", 0.0)
            - metrics["end_to_end"]["direction_long"].get("macro_f1", 0.0)
        ),
        "impact_qwk_delta": ind_impact.get("qwk", 0.0) - e2e_impact.get("qwk", 0.0),
        "impact_mae_delta": e2e_impact.get("mae", 0.0) - ind_impact.get("mae", 0.0),
    }

    feature_allowlist = {
        "allowed_raw_input_fields": RAW_ALLOWED_FIELDS,
        "forbidden_answer_or_provenance_fields": ANSWER_OR_PROVENANCE_FIELDS,
        "model_declared_feature_schema": bundle.get("feature_schema", {}),
        "model_inferred_e5_feature_fields": MODEL_E5_INFERRED_FEATURE_FIELDS,
        "violating_model_fields": sorted(set(MODEL_E5_INFERRED_FEATURE_FIELDS) & set(ANSWER_OR_PROVENANCE_FIELDS)),
        "sanitized_cascade_runtime_forbidden_fields_passed_as_values": False,
        "verdict": "fail" if sorted(set(MODEL_E5_INFERRED_FEATURE_FIELDS) & set(ANSWER_OR_PROVENANCE_FIELDS)) else "pass",
    }
    leakage = {
        "created_at": CREATED_AT,
        "official1799_raw_read_count": 0,
        "official1799_filled_gold_read_count": 0,
        "official1799_predictions_read_count": 0,
        "final_holdout_read_count": 0,
        "human_or_answer_fields_used_as_model_input_in_sanitized_cascade": False,
        "feature_allowlist": feature_allowlist,
        "label_shuffle_test": {
            "unsafe_direct_pipeline_changed_fraction": unsafe_shuffle,
            "sanitized_cascade_changed_fraction": sanitized_shuffle,
            "sanitized_invariant": all(v == 0.0 for v in sanitized_shuffle.values()),
        },
        "pipeline_capability": cascade_diag,
        "holdout_overlap_audit": holdout_audit,
    }
    passed, failures = acceptance_gate(leakage, holdout_audit, metrics)
    leakage["nonofficial_acceptance_passed"] = passed
    leakage["acceptance_failures"] = failures
    metrics["nonofficial_acceptance_passed"] = passed
    metrics["acceptance_failures"] = failures

    prediction_frame = pd.concat(
        [
            holdout[RAW_ALLOWED_FIELDS + [c for c in ANSWER_OR_PROVENANCE_FIELDS if c in holdout.columns]],
            cascade_pred.drop(columns=["sample_id"], errors="ignore"),
        ],
        axis=1,
    )
    prediction_frame.to_parquet(OUT / "nonofficial_holdout_predictions.parquet", index=False)

    write_json(OUT / "feature_allowlist.json", feature_allowlist)
    write_json(OUT / "leakage_audit.json", leakage)
    write_json(OUT / "nonofficial_e2e_metrics.json", metrics)

    trace_report = {
        "stage": "nonofficial_holdout_only",
        "official1799_trace_outputs": "not_executed_nonofficial_acceptance_failed",
        "nonofficial_rows": int(len(cascade_pred)),
        "gate_distribution": value_counts(cascade_pred["gate_prediction"]),
        "final_decision_distribution": value_counts(cascade_pred["final_decision"]),
        "direction_short_distribution": value_counts(cascade_pred["direction_short_pred"]),
        "direction_long_distribution": value_counts(cascade_pred["direction_long_pred"]),
        "impact_channel_distribution": value_counts(cascade_pred["impact_channel_pred"]),
        "impact_directness_distribution": value_counts(cascade_pred["impact_directness_pred"]),
        "axis_prediction_distributions": {f"{axis}_pred": value_counts(cascade_pred[f"{axis}_pred"]) for axis in AXIS_FIELDS if f"{axis}_pred" in cascade_pred.columns},
        "target_impact_combined_distribution": value_counts(cascade_pred["target_impact_combined_level"].astype(str)),
    }
    coverage_risk = {
        "stage": "nonofficial_holdout_only",
        "official1799_not_executed_reason": "nonofficial_acceptance_failed" if not passed else None,
        "gate_coverage": metrics["end_to_end"]["gate"]["coverage"],
        "gate_abstain_rate": metrics["end_to_end"]["gate"]["abstain_rate"],
        "final_auto_score_coverage": metrics["end_to_end"]["final_auto_score_coverage"],
        "final_review_rate": metrics["end_to_end"]["review_rate"],
        "final_abstain_rate": metrics["end_to_end"]["abstain_rate"],
    }
    write_json(OUT / "trace_unlabeled_output_report.json", trace_report)
    write_json(OUT / "coverage_risk_report.json", coverage_risk)

    official_placeholder = pd.DataFrame(
        [
            {
                "status": "not_executed",
                "reason": "nonofficial_acceptance_failed",
                "official1799_raw_read_count": 0,
                "official1799_filled_gold_read_count": 0,
            }
        ]
    )
    official_placeholder.to_parquet(OUT / "official1799_predictions_frozen.parquet", index=False)
    official_placeholder.to_excel(OUT / "official1799_predictions_frozen.xlsx", index=False)
    write_json(
        OUT / "official1799_prediction_sha256.json",
        {
            "status": "not_executed",
            "reason": "nonofficial_acceptance_failed" if not passed else None,
            "official1799_raw_read_count": 0,
            "official1799_filled_gold_read_count": 0,
            "files": {
                "official1799_predictions_frozen.parquet": sha256_file(OUT / "official1799_predictions_frozen.parquet"),
                "official1799_predictions_frozen.xlsx": sha256_file(OUT / "official1799_predictions_frozen.xlsx"),
            },
        },
    )
    write_json(
        OUT / "official1799_formal_metrics.json",
        {
            "status": "not_executed",
            "reason": "nonofficial_acceptance_failed",
            "official1799_raw_read_count": 0,
            "official1799_filled_gold_read_count": 0,
            "gold_read_after_prediction": False,
            "modification_after_gold_read": False,
            "actual_evaluable_gold_fields": [],
            "formal_metrics": {},
        },
    )

    report = f"""# Task4 TRACE-Impact v1 Final Acceptance

Generated at: {CREATED_AT}

## Decision

- Nonofficial acceptance passed: `{passed}`
- Failures: `{", ".join(failures) if failures else "none"}`
- Official1799 executed: `false`
- Official1799 raw read count: `0`
- Official1799 filled Gold read count: `0`

## Nonofficial Holdout

- Candidate source: Silver-B rows excluded from v1 main supervision.
- Rows after core key filter: `{holdout_audit.get("holdout_rows_after_core_key_filter", 0)}`
- Strict target-isolated rows: `{holdout_audit.get("strict_target_isolated_rows", 0)}`
- Target-overlap rows after core filter: `{holdout_audit.get("target_overlap_rows_after_core_key_filter", 0)}`

## Metrics

- Direction short E2E Macro-F1: `{metrics["end_to_end"]["direction_short"].get("macro_f1")}`
- Direction long E2E Macro-F1: `{metrics["end_to_end"]["direction_long"].get("macro_f1")}`
- Impact E2E QWK: `{metrics["end_to_end"]["target_impact_severity"].get("qwk")}`
- Impact E2E MAE: `{metrics["end_to_end"]["target_impact_severity"].get("mae")}`
- Gate F1(auto_score): `{metrics["end_to_end"]["gate"].get("per_class", {}).get("auto_score", {}).get("f1")}`
- Gate coverage: `{metrics["end_to_end"]["gate"].get("coverage")}`
- Gate abstain: `{metrics["end_to_end"]["gate"].get("abstain_rate")}`

## Leakage Audit

The frozen v1 model schema uses fields that are answer or intermediate Gold fields under the final acceptance contract:
`{", ".join(feature_allowlist["violating_model_fields"])}`

The sanitized cascade does not pass these values at runtime, but the frozen model itself is not an allowlisted end-to-end model. Channel and directness frozen heads are missing, so final auto-score coverage is zero and official1799 evaluation was not entered.
"""
    (OUT / "final_acceptance_report.md").write_text(report, encoding="utf-8")

    after = protected_sha256()
    before = json.loads((OUT / "protected_sha256_before.json").read_text(encoding="utf-8"))
    after["_comparison_to_before"] = {name: before.get(name, {}).get("sha256") == payload.get("sha256") for name, payload in after.items()}
    after["_output_sha256"] = output_sha256()
    write_json(OUT / "protected_sha256_after.json", after)
    shutil.copy2(Path(__file__), OUT / "run_task4_trace_impact_v1_final_acceptance.py")

    print(
        json.dumps(
            {
                "status": "complete",
                "nonofficial_acceptance_passed": passed,
                "failures": failures,
                "output_dir": str(OUT),
                "official1799_executed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
