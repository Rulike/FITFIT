#!/usr/bin/env python3
"""Blind frozen inference for Task2 independent500-v2.

This script is deliberately pre-gold only.  It never opens the adjudicated
workbook and only writes frozen predictions for systems that can be recreated
from already-existing frozen artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.csmar_adapter import load_csmar_financial_records_v2  # noqa: E402
from src.event_credibility.market_consistency_v21 import (  # noqa: E402
    VERSION as MARKET_V21_VERSION,
    MarketRuleConfig,
    aggregate_market_consistency,
    build_market_index,
)
from src.event_credibility.resset_market_adapter import load_market_cache  # noqa: E402


LABELS = ["support", "contradict", "insufficient", "unclear"]
VALID_LABELS = set(LABELS)
SEED = 2026
EXPECTED_INPUT_SHA256 = "0ea6679aeb3fc9c3d5d7dd311832d1e652ded2d92adb428e2481e39f6c11e18b"
DEFAULT_INPUT = ROOT / "outputs/event_credibility/task2_independent500_v2/independent_samples.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/event_credibility/task2_independent500_v2_evaluation_v1"
DEFAULT_OLD_DEV = ROOT / "outputs/event_credibility/task2_independent500_v1/independent_samples.jsonl"
DEFAULT_FUSION_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3/fusion_model_v1"
DEFAULT_MARKET_CACHE = ROOT / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v1/resset_market_daily_cache.parquet"
DEFAULT_CSMAR_CACHE = ROOT / "data/processed/event_credibility/csmar_financial_cache_v2"
DEFAULT_MODEL_URL = "http://127.0.0.1:8000/v1/models"
GOLD_LIKE_FIELDS = {
    "human_label",
    "human_verdict",
    "gold",
    "gold_label",
    "true_label",
    "adjudicated_label",
    "human_reason",
    "selected_evidence_ids",
}
PREDICTION_FILE_BY_SYSTEM = {
    "Baseline": "baseline_predictions_frozen_before_gold.jsonl",
    "DEA v2": "dea_v2_predictions_frozen_before_gold.jsonl",
    "CachedDEA": "cached_dea_predictions_frozen_before_gold.jsonl",
    "Financial-enhanced CachedDEA": "financial_enhanced_predictions_frozen_before_gold.jsonl",
    "Financial + market-rule Enhanced CachedDEA": "financial_market_predictions_frozen_before_gold.jsonl",
    "Hybrid-v3": "hybrid_v3_predictions_frozen_before_gold.jsonl",
}


def to_dense_array(value: Any) -> Any:
    """Compatibility shim for the frozen fusion_model_v1 pickle."""
    return value.toarray() if hasattr(value, "toarray") else value


def load_fusion_module() -> Any:
    path = ROOT / "scripts/train_task2_hybrid_v3_fusion_model_v1.py"
    spec = importlib.util.spec_from_file_location("task2_fusion_model_v1_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fusion module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fusion = load_fusion_module()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compact(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", norm(value).lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(obj)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(clean_json(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if math.isfinite(number) else default


def input_gold_like_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted({field for row in rows for field in row if field in GOLD_LIKE_FIELDS})
    nonempty: dict[str, int] = {}
    for field in present:
        count = sum(1 for row in rows if norm(row.get(field)))
        if count:
            nonempty[field] = count
    return {
        "gold_like_fields_present": present,
        "gold_like_fields_present_but_blank": bool(present) and not bool(nonempty),
        "gold_like_nonempty_counts": nonempty,
        "gold_like_nonempty_total": int(sum(nonempty.values())),
        "prediction_input_gold_like_fields_stripped": True,
    }


def strip_gold_like(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in GOLD_LIKE_FIELDS}


def title_hash(row: dict[str, Any]) -> str:
    return norm(row.get("normalized_title_sha256"))


def content_hash(row: dict[str, Any]) -> str:
    return norm(row.get("normalized_content_sha256"))


def near_duplicate_overlap(new_rows: list[dict[str, Any]], old_rows: list[dict[str, Any]], threshold: float = 0.92) -> list[dict[str, Any]]:
    old_profiles = []
    for row in old_rows:
        text = compact(" ".join([norm(row.get("claim_title")), norm(row.get("claim_text")), norm((row.get("claim") or {}).get("title"))]))
        if text:
            old_profiles.append((norm(row.get("sample_id")), norm(row.get("cluster_id")), text))
    overlaps: list[dict[str, Any]] = []
    for row in new_rows:
        text = compact(" ".join([norm(row.get("claim_title")), norm(row.get("claim_text"))]))
        if not text:
            continue
        for old_sample_id, old_cluster_id, old_text in old_profiles:
            max_possible = (2.0 * min(len(text), len(old_text))) / max(1, len(text) + len(old_text))
            if max_possible < threshold:
                continue
            score = SequenceMatcher(None, text, old_text).ratio()
            if score >= threshold:
                overlaps.append(
                    {
                        "sample_id": norm(row.get("sample_id")),
                        "cluster_id": norm(row.get("cluster_id")),
                        "old_sample_id": old_sample_id,
                        "old_cluster_id": old_cluster_id,
                        "similarity": round(float(score), 6),
                    }
                )
                break
    return overlaps


def old_development_overlap_audit(rows: list[dict[str, Any]], old_path: Path) -> dict[str, Any]:
    if not old_path.exists():
        return {"old_development_input_path": rel(old_path), "status": "skipped_missing_old_development_input"}
    old_rows = read_jsonl(old_path)
    new_ids = {norm(row.get("sample_id")) for row in rows if norm(row.get("sample_id"))}
    old_ids = {norm(row.get("sample_id")) for row in old_rows if norm(row.get("sample_id"))}
    new_clusters = {norm(row.get("cluster_id")) for row in rows if norm(row.get("cluster_id"))}
    old_clusters = {norm(row.get("cluster_id")) for row in old_rows if norm(row.get("cluster_id"))}
    new_article_ids = {norm(row.get("article_file_id")) for row in rows if norm(row.get("article_file_id"))}
    old_article_ids = {norm(row.get("article_file_id")) for row in old_rows if norm(row.get("article_file_id"))}
    title_overlap = {title_hash(row) for row in rows if title_hash(row)} & {title_hash(row) for row in old_rows if title_hash(row)}
    content_overlap = {content_hash(row) for row in rows if content_hash(row)} & {content_hash(row) for row in old_rows if content_hash(row)}
    near = near_duplicate_overlap(rows, old_rows)
    return {
        "old_development_input_path": rel(old_path),
        "old_development_input_sha256": sha256_file(old_path),
        "old_development_rows": len(old_rows),
        "sample_id_overlap_count": len(new_ids & old_ids),
        "cluster_id_overlap_count": len(new_clusters & old_clusters),
        "article_file_id_overlap_count": len(new_article_ids & old_article_ids),
        "title_sha256_overlap_count": len(title_overlap),
        "content_sha256_overlap_count": len(content_overlap),
        "near_duplicate_overlap_count": len(near),
        "near_duplicate_examples": near[:20],
        "exact_or_near_overlap_count": len(new_ids & old_ids)
        + len(new_clusters & old_clusters)
        + len(new_article_ids & old_article_ids)
        + len(title_overlap)
        + len(content_overlap)
        + len(near),
    }


def validate_input(path: Path, rows: list[dict[str, Any]], limit: int, old_dev_path: Path) -> dict[str, Any]:
    file_sha = sha256_file(path)
    if file_sha != EXPECTED_INPUT_SHA256 and limit == 0:
        raise ValueError(f"independent500-v2 SHA256 mismatch: {file_sha}")
    ids = [norm(row.get("sample_id")) for row in rows]
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if item and count > 1)
    if limit == 0 and len(rows) != 500:
        raise ValueError(f"independent500-v2 must contain 500 rows, got {len(rows)}")
    if len(set(ids)) != len(rows) or duplicate_ids or any(not item for item in ids):
        raise ValueError(f"sample_id audit failed: duplicate={duplicate_ids[:10]}")
    gold_like = input_gold_like_audit(rows)
    if gold_like["gold_like_nonempty_total"]:
        raise ValueError(f"input contains non-empty gold-like fields: {gold_like['gold_like_nonempty_counts']}")
    old_overlap = old_development_overlap_audit(rows, old_dev_path)
    if int(old_overlap.get("exact_or_near_overlap_count") or 0) > 0:
        raise ValueError(f"independent500-v2 overlaps retired human development input: {old_overlap}")
    return {
        "input_path": rel(path),
        "input_sha256": file_sha,
        "expected_input_sha256": EXPECTED_INPUT_SHA256,
        "input_sha256_ok": file_sha == EXPECTED_INPUT_SHA256,
        "source_rows": len(rows),
        "source_unique_sample_ids": len(set(ids)),
        "limit": limit,
        "selected_rows": len(rows[:limit] if limit and limit > 0 else rows),
        "selected_unique_sample_ids": len(set(ids[:limit] if limit and limit > 0 else ids)),
        "forbidden_label_value_audit": gold_like,
        "retired_human_development_overlap_audit": old_overlap,
    }


def service_check() -> dict[str, Any]:
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        proc = subprocess.run(
            ["curl", "-s", DEFAULT_MODEL_URL],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        body = (proc.stdout or proc.stderr or "")[:1000]
        available = proc.returncode == 0 and "DianJin-R1-32B" in body
        return {
            "checked_at": started,
            "command": f"curl -s {DEFAULT_MODEL_URL}",
            "returncode": int(proc.returncode),
            "available": bool(available),
            "response_preview": body,
        }
    except Exception as exc:
        return {
            "checked_at": started,
            "command": f"curl -s {DEFAULT_MODEL_URL}",
            "returncode": -1,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def to_clean_like(row: dict[str, Any]) -> dict[str, Any]:
    target_name = norm(row.get("target_name") or row.get("target"))
    target_code = norm(row.get("target_code"))
    claim_title = norm(row.get("claim_title") or row.get("title"))
    claim_text = norm(row.get("claim_text") or claim_title)
    publish_time = norm(row.get("publish_time") or row.get("event_time"))
    evidence_items: list[dict[str, Any]] = []
    for idx, item in enumerate(row.get("evidence_items") or [], 1):
        if not isinstance(item, dict):
            continue
        evidence_items.append(
            {
                **item,
                "evidence_id": norm(item.get("evidence_id") or f"E{idx}"),
                "article_id": norm(item.get("article_id") or item.get("document_id")),
                "document_id": norm(item.get("document_id") or item.get("article_id")),
                "publish_time": norm(item.get("publish_time")),
                "source": norm(item.get("source")),
                "source_type": norm(item.get("source_type")),
                "title": norm(item.get("title")),
                "sentence": norm(item.get("sentence") or item.get("evidence_text") or item.get("content")),
            }
        )
    return {
        "sample_id": norm(row.get("sample_id")),
        "cluster_id": norm(row.get("cluster_id") or row.get("sample_id")),
        "claim_article_id": norm(row.get("article_file_id") or row.get("claim_article_id")),
        "event_time": publish_time,
        "event_type": norm(row.get("event_name") or row.get("event_type")),
        "evidence_items": evidence_items,
        "evidence_status": "evidence_ready" if evidence_items else "insufficient_independent_evidence",
        "target_entity": target_name,
        "valid_evidence_count": len(evidence_items),
        "independent_source_count": int(row.get("distinct_evidence_source_count") or len({norm(item.get("source")) for item in evidence_items if norm(item.get("source"))})),
        "claim": {
            "article_file_id": norm(row.get("article_file_id") or row.get("claim_article_id")),
            "cluster_id": norm(row.get("cluster_id") or row.get("sample_id")),
            "document_id": norm(row.get("claim_article_id") or row.get("article_file_id")),
            "event_claim": claim_text,
            "normalized_content_sha256": norm(row.get("normalized_content_sha256")),
            "normalized_title_sha256": norm(row.get("normalized_title_sha256")),
            "publish_time": publish_time,
            "source": norm(row.get("source")),
            "target": f"{target_name} / {target_code}".strip(" /"),
            "title": claim_title or claim_text,
        },
        "source_type": norm(row.get("source_type")),
        "source": norm(row.get("source")),
        "raw_financial_context": norm(row.get("raw_financial_context")),
        "raw_market_context": norm(row.get("raw_market_context")),
    }


def build_records(rows: list[dict[str, Any]], market_cache: Path, csmar_cache: Path) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    market_index = build_market_index(load_market_cache(market_cache))
    csmar_records = load_csmar_financial_records_v2(csmar_cache)
    csmar_by_code = fusion.financial_index(csmar_records)
    cfg = MarketRuleConfig(version=MARKET_V21_VERSION)
    records: list[Any] = []
    feature_audit: list[dict[str, Any]] = []
    sanitation_totals = Counter()
    future_financial_count = 0
    future_market_count = 0
    for row in rows:
        clean_row = to_clean_like(strip_gold_like(row))
        market_row, accepted, rejected = aggregate_market_consistency(clean_row, market_index, cfg)
        features, financial_context, market_context, sanitation_counts = fusion.build_feature_row(clean_row, market_row, csmar_by_code)
        sanitation_totals.update(sanitation_counts)
        future_financial_count += int(bool(financial_context.get("future_financial_data_used")))
        future_market_count += int(bool(market_row.get("future_data_used")))
        sid = norm(clean_row.get("sample_id"))
        records.append(
            fusion.PreparedRecord(
                sample_id=sid,
                group_id=norm(clean_row.get("cluster_id")) or sid,
                label="insufficient",
                split="independent500_v2_blind",
                clean=clean_row,
                silver={},
                features=features,
                financial_context=financial_context,
                market_context=market_row,
            )
        )
        feature_audit.append(
            {
                "sample_id": sid,
                "cluster_id": norm(clean_row.get("cluster_id")),
                "evidence_ids_after_sanitation": [norm(item.get("evidence_id")) for item in fusion.sanitize_evidence(clean_row)[0]],
                "evidence_sanitation_removed_counts": sanitation_counts,
                "financial_consistency_verdict": financial_context.get("financial_consistency_verdict"),
                "financial_claim_count": financial_context.get("financial_claim_count"),
                "financial_context_available": bool((financial_context.get("data_availability") or {}).get("has_candidate_records")),
                "market_claim_type": market_row.get("market_claim_type"),
                "market_consistency_verdict": market_row.get("market_consistency_verdict"),
                "closed_track_verifiable": bool(market_row.get("closed_track_verifiable")),
                "market_context_available": bool(market_row.get("causal_market_available")),
                "accepted_market_claim_count": len(accepted),
                "rejected_market_claim_count": len(rejected),
                "future_market_data_used": bool(market_row.get("future_data_used")),
            }
        )
    summary = {
        "records": len(records),
        "market_cache": rel(market_cache),
        "market_cache_sha256": sha256_file(market_cache),
        "market_source_version": MARKET_V21_VERSION,
        "csmar_cache": rel(csmar_cache),
        "feature_builder": "scripts/train_task2_hybrid_v3_fusion_model_v1.py::build_feature_row",
        "evidence_sanitation_removed_counts": dict(sanitation_totals),
        "future_evidence_removed_before_prediction_count": int(sanitation_totals.get("future_evidence_removed", 0)),
        "future_financial_data_used_count": future_financial_count,
        "future_market_data_used_count": future_market_count,
        "future_evidence_used_in_closed_track": False,
        "post_event_market_features_used": False,
    }
    if future_financial_count or future_market_count:
        raise ValueError(f"future data audit failed: {summary}")
    return records, feature_audit, summary


def train_labels_from_fusion_train(path: Path) -> list[str]:
    rows = read_jsonl(path)
    labels = [norm(row.get("label")) for row in rows]
    labels = [label for label in labels if label in VALID_LABELS]
    if not labels:
        raise ValueError(f"no valid labels in frozen train artifact: {path}")
    return labels


def probability_payload_from_model(model: Any, frame: pd.DataFrame) -> list[dict[str, float]]:
    if not hasattr(model, "predict_proba"):
        predictions = list(model.predict(frame))
        return [{label: 1.0 if label == pred else 0.0 for label in LABELS} for pred in predictions]
    probabilities = model.predict_proba(frame)
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    classes = [str(item) for item in getattr(classifier, "classes_", [])]
    payload: list[dict[str, float]] = []
    for values in probabilities:
        row = {label: 0.0 for label in LABELS}
        for idx, label in enumerate(classes):
            if idx < len(values) and label in row:
                row[label] = round(float(values[idx]), 6)
        payload.append(row)
    return payload


def prediction_rows(
    records: list[Any],
    source_rows: list[dict[str, Any]],
    system_name: str,
    predictions: list[str],
    probabilities: list[dict[str, float]],
    feature_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        proba = probabilities[idx]
        predicted = predictions[idx] if predictions[idx] in VALID_LABELS else "unclear"
        confidence = max((safe_float(proba.get(label), 0.0) for label in LABELS), default=0.0)
        clean_row = record.clean
        kept_items, _counts = fusion.sanitize_evidence(clean_row)
        rows.append(
            {
                "sample_id": record.sample_id,
                "cluster_id": norm(clean_row.get("cluster_id")),
                "system_name": system_name,
                "predicted_label": predicted,
                "confidence": round(float(confidence), 6),
                "probabilities": {label: round(float(proba.get(label, 0.0)), 6) for label in LABELS},
                "evidence_ids": [norm(item.get("evidence_id")) for item in kept_items if norm(item.get("evidence_id"))],
                "failure_reason": "" if predictions[idx] in VALID_LABELS else "invalid_label_mapped_to_unclear",
                "source_type": norm(source_rows[idx].get("source_type")),
                "event_name": norm(source_rows[idx].get("event_name")),
                "target": norm(source_rows[idx].get("target")),
                "target_code": norm(source_rows[idx].get("target_code")),
                "target_name": norm(source_rows[idx].get("target_name")),
                "market_context_available": bool(feature_audit[idx].get("market_context_available")),
                "financial_context_available": bool(feature_audit[idx].get("financial_context_available")),
                "prediction_generated_without_gold": True,
            }
        )
    return rows


def build_predictions(
    records: list[Any],
    source_rows: list[dict[str, Any]],
    feature_audit: list[dict[str, Any]],
    fusion_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    outputs: dict[str, list[dict[str, Any]]] = {}
    train_labels = train_labels_from_fusion_train(fusion_dir / "train.jsonl")
    baseline_pred, baseline_proba = fusion.baseline_majority_predictions(records, train_labels)
    outputs[PREDICTION_FILE_BY_SYSTEM["Baseline"]] = prediction_rows(
        records,
        source_rows,
        "Baseline",
        baseline_pred,
        baseline_proba,
        feature_audit,
    )

    best_manifest = json.loads((fusion_dir / "best_model_manifest.json").read_text(encoding="utf-8"))
    feature_set = norm(best_manifest.get("feature_set") or "text_financial_market")
    model = joblib.load(fusion_dir / "best_model.pkl")
    frame = fusion.dataframe_for(records, feature_set)
    hybrid_pred = list(model.predict(frame))
    hybrid_proba = probability_payload_from_model(model, frame)
    outputs[PREDICTION_FILE_BY_SYSTEM["Hybrid-v3"]] = prediction_rows(
        records,
        source_rows,
        "Hybrid-v3",
        hybrid_pred,
        hybrid_proba,
        feature_audit,
    )
    details = {
        "baseline": {
            "frozen_artifact": rel(fusion_dir / "train.jsonl"),
            "train_label_distribution_used_for_prior": dict(Counter(train_labels)),
            "prediction_distribution": dict(Counter(baseline_pred)),
        },
        "hybrid_v3": {
            "model_path": rel(fusion_dir / "best_model.pkl"),
            "best_model_manifest": rel(fusion_dir / "best_model_manifest.json"),
            "best_model_sha256": sha256_file(fusion_dir / "best_model.pkl"),
            "feature_set": feature_set,
            "model_name": best_manifest.get("model_name"),
            "model_family": best_manifest.get("model_family"),
            "selected_on_dev_only": bool(best_manifest.get("selected_on_dev_only")),
            "locked500_gold_read_in_manifest": bool(best_manifest.get("locked500_gold_read")),
            "prediction_distribution": dict(Counter(hybrid_pred)),
        },
    }
    return outputs, details


def inventory_systems(service: dict[str, Any], fusion_dir: Path, included_files: dict[str, str]) -> dict[str, Any]:
    fusion_manifest = {}
    if (fusion_dir / "best_model_manifest.json").exists():
        fusion_manifest = json.loads((fusion_dir / "best_model_manifest.json").read_text(encoding="utf-8"))
    llm_blocker = "DianJin model service check did not confirm availability; no new LLM calls were made before gold"
    if service.get("response_preview") and "Unauthorized" in service.get("response_preview", ""):
        llm_blocker = "DianJin model service returned Unauthorized to the required pre-inference models probe"
    if service.get("returncode") not in (0, None):
        llm_blocker = f"DianJin model service probe failed with returncode={service.get('returncode')}"
    systems = {
        "Baseline": {
            "included": "baseline_predictions_frozen_before_gold.jsonl" in included_files,
            "frozen_before_new_gold": True,
            "frozen_asset": rel(fusion_dir / "train.jsonl"),
            "requires_gold": False,
            "requires_dianjin": False,
            "future_evidence_used": False,
            "leakage_detected": False,
            "exclusion_reason": "",
        },
        "Round1": {
            "included": False,
            "frozen_before_new_gold": True,
            "requires_gold": False,
            "requires_dianjin": True,
            "future_evidence_used": False,
            "leakage_detected": False,
            "exclusion_reason": "Round1 is an LLM intermediate result, not one of the required frozen output files for this official run; " + llm_blocker,
        },
        "DEA v2": {
            "included": False,
            "frozen_before_new_gold": True,
            "requires_gold": False,
            "requires_dianjin": True,
            "future_evidence_used": False,
            "leakage_detected": False,
            "exclusion_reason": llm_blocker,
        },
        "CachedDEA": {
            "included": False,
            "frozen_before_new_gold": True,
            "requires_gold": False,
            "requires_dianjin": True,
            "future_evidence_used": False,
            "leakage_detected": False,
            "exclusion_reason": llm_blocker,
        },
        "Financial-enhanced CachedDEA": {
            "included": False,
            "frozen_before_new_gold": True,
            "requires_gold": False,
            "requires_dianjin": True,
            "future_evidence_used": False,
            "leakage_detected": False,
            "exclusion_reason": "requires a valid CachedDEA base prediction for each I500V2 sample; " + llm_blocker,
        },
        "Financial + market-rule Enhanced CachedDEA": {
            "included": False,
            "frozen_before_new_gold": True,
            "requires_gold": False,
            "requires_dianjin": True,
            "future_evidence_used": False,
            "leakage_detected": False,
            "exclusion_reason": "requires a valid Financial-enhanced CachedDEA base prediction for each I500V2 sample; " + llm_blocker,
        },
        "Hybrid-v3": {
            "included": "hybrid_v3_predictions_frozen_before_gold.jsonl" in included_files,
            "frozen_before_new_gold": True,
            "frozen_asset": rel(fusion_dir / "best_model.pkl"),
            "requires_gold": False,
            "requires_dianjin": False,
            "future_evidence_used": False,
            "leakage_detected": False,
            "exclusion_reason": "" if "hybrid_v3_predictions_frozen_before_gold.jsonl" in included_files else "best_model.pkl unavailable",
            "best_model_manifest": fusion_manifest,
        },
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "new_gold_read": False,
        "old_gold_used_for_prediction": False,
        "model_service_check": service,
        "systems": systems,
    }


def write_prediction_freeze(
    output_dir: Path,
    input_audit: dict[str, Any],
    inventory: dict[str, Any],
    feature_summary: dict[str, Any],
    prediction_outputs: dict[str, list[dict[str, Any]]],
    prediction_details: dict[str, Any],
    code_paths: list[Path],
    limit: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_files: dict[str, dict[str, Any]] = {}
    for filename, rows in prediction_outputs.items():
        path = output_dir / filename
        write_jsonl(path, rows)
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        ids = [norm(row.get("sample_id")) for row in rows]
        forbidden_present = sorted({field for row in rows for field in row if field in GOLD_LIKE_FIELDS or field == "true_label"})
        if forbidden_present:
            raise ValueError(f"{filename} contains forbidden fields: {forbidden_present}")
        prediction_files[filename] = {
            "path": rel(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "unique_sample_ids": len(set(ids)),
            "failed_count": sum(1 for row in rows if norm(row.get("failure_reason"))),
            "read_only": not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)),
            "prediction_distribution": dict(Counter(norm(row.get("predicted_label")) for row in rows)),
        }
    manifest = {
        "status": "prediction_frozen_before_gold",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "freeze_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "limit": limit,
        "input": input_audit,
        "frozen_system_inventory_path": rel(output_dir / "frozen_system_inventory.json"),
        "feature_summary": feature_summary,
        "prediction_files": prediction_files,
        "prediction_details": prediction_details,
        "actual_model_name": "DianJin-R1-32B not called; frozen sklearn Hybrid-v3 model loaded from best_model.pkl",
        "code_and_config_sha256": {rel(path): sha256_file(path) for path in code_paths if path.exists()},
        "training_performed": False,
        "tuning_performed": False,
        "threshold_search_performed": False,
        "gold_read": False,
        "old_gold_used_for_prediction": False,
        "new_gold_used_for_prediction": False,
        "future_evidence_used": 0,
        "prediction_frozen": True,
    }
    write_json(output_dir / "prediction_freeze_manifest.json", manifest)
    lines = [f"{payload['sha256']}  {payload['path']}" for payload in prediction_files.values()]
    (output_dir / "prediction_sha256_before_gold.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_sha256s(output_dir: Path, extra_paths: Iterable[Path]) -> dict[str, str]:
    paths = [path for path in output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"]
    for path in extra_paths:
        if path.exists() and path.is_file():
            paths.append(path)
    results: dict[str, str] = {}
    lines: list[str] = []
    for path in sorted(set(paths)):
        digest = sha256_file(path)
        results[rel(path)] = digest
        lines.append(f"{digest}  {rel(path)}")
    sums_path = output_dir / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results[rel(sums_path)] = sha256_file(sums_path)
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    input_path = resolve(args.input)
    output_dir = resolve(args.output_dir)
    if (output_dir / "gold_validation.json").exists() or (output_dir / "final_run_state.json").exists():
        raise RuntimeError("gold has already been read for this output directory; refusing to rerun pre-gold inference")
    if (output_dir / "prediction_freeze_manifest.json").exists() and not args.resume:
        existing = json.loads((output_dir / "prediction_freeze_manifest.json").read_text(encoding="utf-8"))
        if existing.get("gold_read"):
            raise RuntimeError("prediction_freeze_manifest already has gold_read=true; refusing to rerun inference")
        raise FileExistsError(f"{output_dir}/prediction_freeze_manifest.json exists; pass --resume only to verify existing freeze")
    if args.resume and (output_dir / "prediction_freeze_manifest.json").exists():
        manifest = json.loads((output_dir / "prediction_freeze_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("gold_read"):
            raise RuntimeError("cannot resume pre-gold inference after gold_read=true")
        for payload in (manifest.get("prediction_files") or {}).values():
            path = resolve(payload["path"])
            if sha256_file(path) != payload["sha256"]:
                raise ValueError(f"prediction SHA mismatch on resume: {path}")
        return {"status": "existing_prediction_freeze_verified", "output_dir": rel(output_dir), "manifest": manifest}

    source_rows_all = read_jsonl(input_path)
    input_audit = validate_input(input_path, source_rows_all, int(args.limit or 0), resolve(args.old_development_input))
    selected_rows = source_rows_all[: int(args.limit)] if args.limit and args.limit > 0 else source_rows_all
    selected_rows = [strip_gold_like(row) for row in selected_rows]
    service = service_check()
    records, feature_audit, feature_summary = build_records(selected_rows, resolve(args.market_cache), resolve(args.csmar_cache))
    prediction_outputs, prediction_details = build_predictions(records, selected_rows, feature_audit, resolve(args.fusion_dir))
    included_files = {name: name for name in prediction_outputs}
    inventory = inventory_systems(service, resolve(args.fusion_dir), included_files)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "pre_gold_input_audit.json", input_audit)
    write_json(output_dir / "frozen_system_inventory.json", inventory)
    write_jsonl(output_dir / "blind_feature_audit.jsonl", feature_audit)
    manifest = write_prediction_freeze(
        output_dir,
        input_audit,
        inventory,
        feature_summary,
        prediction_outputs,
        prediction_details,
        [
            Path(__file__).resolve(),
            ROOT / "scripts/train_task2_hybrid_v3_fusion_model_v1.py",
            ROOT / "src/event_credibility/market_consistency_v21.py",
            ROOT / "src/event_credibility/market_consistency_v2.py",
            ROOT / "src/event_credibility/financial_consistency.py",
            ROOT / "src/event_credibility/resset_market_adapter.py",
            ROOT / "configs/task2_hybrid_v3_fusion_model_v1.yaml",
            resolve(args.fusion_dir) / "best_model_manifest.json",
            resolve(args.fusion_dir) / "best_model.pkl",
        ],
        int(args.limit or 0),
    )
    # Verify after chmod that prediction hashes are stable.
    post_hashes = {}
    for filename, payload in manifest["prediction_files"].items():
        path = resolve(payload["path"])
        digest = sha256_file(path)
        if digest != payload["sha256"]:
            raise ValueError(f"post-freeze SHA mismatch for {filename}")
        post_hashes[filename] = digest
    run_state = {
        "status": "prediction_frozen_before_gold",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "independent_test_rows": len(selected_rows),
        "independent_test_unique_ids": len({norm(row.get("sample_id")) for row in selected_rows}),
        "prediction_files": manifest["prediction_files"],
        "prediction_sha_match_after_chmod": True,
        "training_performed": False,
        "tuning_performed": False,
        "threshold_search_performed": False,
        "gold_read": False,
        "future_evidence_used": 0,
        "old_gold_used_for_prediction": False,
        "new_gold_used_for_prediction": False,
        "prediction_frozen": True,
        "included_systems": [name for name, payload in inventory["systems"].items() if payload.get("included")],
        "excluded_systems": {name: payload.get("exclusion_reason") for name, payload in inventory["systems"].items() if not payload.get("included")},
    }
    write_json(output_dir / "pre_gold_run_state.json", run_state)
    sha256s = write_sha256s(output_dir, [input_path])
    return {
        "status": "prediction_frozen_before_gold",
        "output_dir": rel(output_dir),
        "included_systems": run_state["included_systems"],
        "excluded_systems": run_state["excluded_systems"],
        "prediction_files": manifest["prediction_files"],
        "sha256s": sha256s,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-development-input", type=Path, default=DEFAULT_OLD_DEV)
    parser.add_argument("--fusion-dir", type=Path, default=DEFAULT_FUSION_DIR)
    parser.add_argument("--market-cache", type=Path, default=DEFAULT_MARKET_CACHE)
    parser.add_argument("--csmar-cache", type=Path, default=DEFAULT_CSMAR_CACHE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
