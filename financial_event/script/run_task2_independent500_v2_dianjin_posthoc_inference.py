#!/usr/bin/env python3
"""DianJin post-hoc supplemental inference for Task2 independent500-v2.

This script is pre-gold for this supplemental run. It validates the gold
workbook only with stat and SHA256, never opens Excel contents, then runs only
previously existing Task2 DEA/financial/market components on the unlabeled
Independent500-v2 input and freezes prediction files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_task2_final_cached_dea_v1 import (  # noqa: E402
    build_dea_config,
    load_config,
    set_runtime_env,
)
from scripts.run_task2_financial_enhanced_cached_dea_v2 import (  # noqa: E402
    analyze_rows as analyze_financial_rows,
)
from scripts.run_task2_market_rule_enhanced_cached_dea import (  # noqa: E402
    build_market_index,
    evaluate_market_rule,
    fuse_market_rule,
)
from src.event_credibility import dea, fulltext  # noqa: E402
from src.event_credibility.csmar_adapter import load_csmar_financial_records_v2  # noqa: E402
from src.event_credibility.resset_market_adapter import load_market_cache  # noqa: E402


LABELS = ["support", "contradict", "insufficient", "unclear"]
VALID_LABELS = set(LABELS)
SEED = 2026
EXPECTED_INPUT_SHA256 = "0ea6679aeb3fc9c3d5d7dd311832d1e652ded2d92adb428e2481e39f6c11e18b"
EXPECTED_GOLD_SHA256 = "699f2e9bedb3a9b925f33be752166db6cf02e984c9444c85d29eee4284b27052"
MODEL_NAME = "DianJin-R1-32B"
BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_INPUT = ROOT / "outputs/event_credibility/task2_independent500_v2/independent_samples.jsonl"
DEFAULT_GOLD = ROOT / "reports/task2_independent500_v2_double_blind_adjudicated.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/event_credibility/task2_independent500_v2_dianjin_posthoc_evaluation_v1"
DEFAULT_CONFIG = ROOT / "configs/task2_final_cached_dea_v1.yaml"
DEFAULT_MARKET_CACHE = ROOT / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v1/resset_market_daily_cache.parquet"
DEFAULT_CSMAR_CACHE = ROOT / "data/processed/event_credibility/csmar_financial_cache_v2"
DEFAULT_OLD_OFFICIAL_DIR = ROOT / "outputs/event_credibility/task2_independent500_v2_evaluation_v1"

GOLD_LIKE_FIELDS = {
    "human_label",
    "human_verdict",
    "adjudicated_label",
    "annotator_a_label",
    "annotator_b_label",
    "human_reason",
    "selected_evidence_ids",
    "needs_second_review",
    "gold",
    "gold_label",
    "true_label",
    "final_gold",
}

PREDICTION_FILES = {
    "Round1": "round1_predictions_frozen_before_gold.jsonl",
    "DEA v2": "dea_v2_predictions_frozen_before_gold.jsonl",
    "CachedDEA": "cached_dea_predictions_frozen_before_gold.jsonl",
    "Financial-enhanced CachedDEA": "financial_enhanced_cached_dea_predictions_frozen_before_gold.jsonl",
    "Financial + market-rule Enhanced CachedDEA": "financial_market_enhanced_cached_dea_predictions_frozen_before_gold.jsonl",
}


class CausalFulltextArticleIndex(dea.FulltextArticleIndex):
    """Existing DEA retrieval index with a closed-track publication cutoff."""

    @staticmethod
    def _cutoff(row: dict[str, Any]) -> datetime | None:
        return dea.parse_dt(row.get("event_time") or row.get("publish_time") or row.get("earliest_publish_time"))

    def extract_from_articles(self, rows: pd.DataFrame, row: dict[str, Any], source_label: str, limit: int = 12) -> list[dict[str, Any]]:
        cutoff = self._cutoff(row)
        if cutoff is not None and not rows.empty and "article_publish_time" in rows.columns:
            pub = pd.to_datetime(rows["article_publish_time"], errors="coerce")
            rows = rows[pub.notna() & (pub <= pd.Timestamp(cutoff))].copy()
        return super().extract_from_articles(rows, row, source_label, limit=limit)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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


def strip_gold_like(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in GOLD_LIKE_FIELDS}


def input_gold_like_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted({field for row in rows for field in row if field in GOLD_LIKE_FIELDS})
    nonempty: dict[str, int] = {}
    for field in present:
        count = sum(1 for row in rows if norm(row.get(field)))
        if count:
            nonempty[field] = count
    return {
        "gold_like_fields_present": present,
        "gold_like_nonempty_counts": nonempty,
        "gold_like_nonempty_total": int(sum(nonempty.values())),
        "prediction_input_gold_like_fields_stripped": True,
    }


def run_curl(args: list[str], timeout: int = 30) -> dict[str, Any]:
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    proc = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)
    return {
        "checked_at": started,
        "command": " ".join(args),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def dianjin_service_audit() -> dict[str, Any]:
    models_cmd = [
        "curl",
        "--noproxy",
        "*",
        "--connect-timeout",
        "5",
        "--max-time",
        "20",
        f"{BASE_URL}/models",
    ]
    models = run_curl(models_cmd, timeout=25)
    model_payload: dict[str, Any] = {}
    model_ids: list[str] = []
    try:
        model_payload = json.loads(models.get("stdout") or "{}")
        model_ids = [str(item.get("id")) for item in model_payload.get("data", []) if isinstance(item, dict)]
    except Exception as exc:
        models["json_error"] = f"{type(exc).__name__}: {exc}"
    chat_body = json.dumps(
        {
            "model": MODEL_NAME,
            "temperature": 0,
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": "你只输出严格JSON。"},
                {"role": "user", "content": "输出 {\"status\":\"ok\"}，不要输出其他文字。"},
            ],
        },
        ensure_ascii=False,
    )
    chat_cmd = [
        "curl",
        "--noproxy",
        "*",
        "--connect-timeout",
        "5",
        "--max-time",
        "60",
        "-sS",
        f"{BASE_URL}/chat/completions",
        "-H",
        "Content-Type: application/json",
        "-d",
        chat_body,
    ]
    chat = run_curl(chat_cmd, timeout=65)
    chat_payload: dict[str, Any] = {}
    content = ""
    try:
        chat_payload = json.loads(chat.get("stdout") or "{}")
        content = str((((chat_payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip()
    except Exception as exc:
        chat["json_error"] = f"{type(exc).__name__}: {exc}"
    ok = models["returncode"] == 0 and model_ids == [MODEL_NAME] and chat["returncode"] == 0 and bool(content)
    audit = {
        "service_ok": bool(ok),
        "base_url": BASE_URL,
        "required_model": MODEL_NAME,
        "models_probe": {
            **{key: value for key, value in models.items() if key not in {"stdout"}},
            "http_200_assumed_from_curl_success": models["returncode"] == 0,
            "model_ids": model_ids,
            "raw": model_payload,
        },
        "chat_smoke": {
            **{key: value for key, value in chat.items() if key not in {"stdout"}},
            "content_nonempty": bool(content),
            "content_preview": content[:300],
        },
    }
    if not ok:
        raise RuntimeError(f"DianJin service audit failed: {audit}")
    return audit


def gold_file_stat_only(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != EXPECTED_GOLD_SHA256:
        raise ValueError(f"gold workbook SHA256 mismatch without reading contents: {digest}")
    st = path.stat()
    return {
        "gold_path": rel(path),
        "gold_exists": True,
        "gold_sha256": digest,
        "gold_sha256_ok": digest == EXPECTED_GOLD_SHA256,
        "stat_size": int(st.st_size),
        "stat_mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
        "gold_content_read": False,
        "allowed_pre_gold_operations": ["test -f", "stat", "sha256sum"],
    }


def validate_input(path: Path, rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != EXPECTED_INPUT_SHA256:
        raise ValueError(f"Independent500-v2 input SHA256 mismatch: {digest}")
    ids = [norm(row.get("sample_id")) for row in rows]
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if item and count > 1)
    if len(rows) != 500 or len(set(ids)) != 500 or duplicate_ids or any(not item for item in ids):
        raise ValueError(f"Independent500-v2 row/id audit failed: rows={len(rows)} duplicate={duplicate_ids[:10]}")
    gold_like = input_gold_like_audit(rows)
    if gold_like["gold_like_nonempty_total"]:
        raise ValueError(f"input contains non-empty gold-like fields: {gold_like['gold_like_nonempty_counts']}")
    selected = rows[:limit] if limit and limit > 0 else rows
    return {
        "input_path": rel(path),
        "input_sha256": digest,
        "expected_input_sha256": EXPECTED_INPUT_SHA256,
        "input_sha256_ok": True,
        "source_rows": len(rows),
        "source_unique_sample_ids": len(set(ids)),
        "selected_rows": len(selected),
        "selected_unique_sample_ids": len({norm(row.get("sample_id")) for row in selected}),
        "limit": limit,
        "forbidden_label_value_audit": gold_like,
    }


def parse_dt(value: Any) -> datetime | None:
    return dea.parse_dt(value)


def evidence_is_causal(item: dict[str, Any], publish_dt: datetime | None) -> bool:
    item_dt = parse_dt(item.get("publish_time") or item.get("article_publish_time"))
    return bool(publish_dt and item_dt and item_dt <= publish_dt)


def to_dea_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    sample_id = norm(row.get("sample_id"))
    publish_time = norm(row.get("publish_time") or row.get("event_time"))
    publish_dt = parse_dt(publish_time)
    claim_article = norm(row.get("article_file_id") or row.get("claim_article_id"))
    counts = Counter()
    evidence_items: list[dict[str, Any]] = []
    for idx, item in enumerate(row.get("evidence_items") or [], 1):
        if not isinstance(item, dict):
            continue
        article_id = norm(item.get("article_id") or item.get("document_id"))
        if claim_article and article_id == claim_article:
            counts["self_evidence_removed"] += 1
            continue
        if not evidence_is_causal(item, publish_dt):
            counts["future_or_undated_evidence_removed"] += 1
            continue
        evidence_items.append(
            {
                "evidence_id": norm(item.get("evidence_id") or f"E{idx}"),
                "article_id": article_id,
                "trace_id": norm(item.get("trace_id") or article_id or item.get("evidence_id") or f"E{idx}"),
                "title": norm(item.get("title") or item.get("article_title")),
                "source": norm(item.get("source") or item.get("article_source")),
                "source_type": norm(item.get("source_type")),
                "publish_time": norm(item.get("publish_time") or item.get("article_publish_time")),
                "sentence": norm(item.get("sentence") or item.get("evidence_text") or item.get("content")),
                "sentence_score": item.get("sentence_score", 0),
                "evidence_source": norm(item.get("evidence_source") or "independent500_v2"),
            }
        )
    target_name = norm(row.get("target_name") or row.get("target")).split(" / ")[0]
    target_code = norm(row.get("target_code"))
    claim_title = norm(row.get("claim_title") or row.get("title"))
    claim_text = norm(row.get("claim_text") or claim_title)
    converted = {
        "sample_id": sample_id,
        "cluster_id": sample_id,
        "source_cluster_id": norm(row.get("cluster_id")),
        "target_code": target_code,
        "target_name": target_name,
        "schema_id": norm(row.get("schema_id")),
        "event_name": norm(row.get("event_name")),
        "event_claim": claim_text,
        "claim_text": claim_text,
        "title_samples": claim_title,
        "event_time": publish_time,
        "publish_time": publish_time,
        "earliest_publish_time": publish_time,
        "latest_publish_time": publish_time,
        "source_type": norm(row.get("source_type")),
        "source": norm(row.get("source")),
        "article_file_id": claim_article,
        "claim_article_id": norm(row.get("claim_article_id") or claim_article),
        "raw_financial_context": norm(row.get("raw_financial_context")),
        "raw_market_context": norm(row.get("raw_market_context")),
        "evidence_items": evidence_items,
    }
    counts["input_evidence_kept"] = len(evidence_items)
    return converted, dict(counts)


def build_dea_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    totals = Counter()
    for row in rows:
        clean, counts = to_dea_row(strip_gold_like(row))
        converted.append(clean)
        totals.update(counts)
    return converted, {
        "rows": len(converted),
        "unique_sample_ids": len({row["sample_id"] for row in converted}),
        "evidence_sanitation_counts": dict(totals),
        "future_evidence_removed_before_prediction_count": int(totals.get("future_or_undated_evidence_removed", 0)),
        "self_evidence_removed_count": int(totals.get("self_evidence_removed", 0)),
        "future_evidence_used_in_closed_track": False,
    }


def load_causal_article_index(config: dict[str, Any]) -> CausalFulltextArticleIndex:
    retrieval = config.get("retrieval") or {}
    index = CausalFulltextArticleIndex(
        resolve(retrieval.get("article_index", "data/processed/event_credibility/fulltext_article_index.parquet")),
        members_path=resolve(retrieval.get("members", "outputs/event_clustering/event_cluster_members.parquet")),
    )
    index.load()
    if not index.available:
        raise RuntimeError(f"retrieval index unavailable: {index.error}")
    return index


def configure_runtime(config: dict[str, Any], workers: int) -> dict[str, Any]:
    config = dict(config)
    config["api_base"] = BASE_URL
    config["model_name"] = MODEL_NAME
    config["temperature"] = float(config.get("temperature", 0))
    # Match the required `curl --noproxy "*"` local-service policy for urllib.
    # Without this, proxy environment variables can turn localhost calls into
    # proxy 502 failures even when vLLM itself is healthy.
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    return set_runtime_env(config, workers)


def run_dea_phase(
    rows: list[dict[str, Any]],
    output_path: Path,
    runtime_config: dict[str, Any],
    dea_config: dea.DEAConfig,
    workers: int,
    resume: bool,
    article_index: CausalFulltextArticleIndex,
    checkpoint_every: int,
) -> dict[str, Any]:
    return dea.run_dea_batch(
        rows,
        output_path,
        runtime_config,
        dea_config,
        mode="dea",
        workers=workers,
        resume=resume,
        checkpoint_every=checkpoint_every,
        article_index=article_index,
    )


def evidence_time_map(raw: dict[str, Any]) -> dict[str, str]:
    items = list(raw.get("initial_evidence") or []) + list(raw.get("additional_evidence") or []) + list(raw.get("final_evidence") or [])
    return {norm(item.get("evidence_id")): norm(item.get("publish_time") or item.get("article_publish_time")) for item in items if isinstance(item, dict)}


def future_evidence_used_count(raw: dict[str, Any], publish_time: str, evidence_ids: list[str]) -> int:
    cutoff = parse_dt(publish_time)
    times = evidence_time_map(raw)
    count = 0
    for eid in evidence_ids:
        item_dt = parse_dt(times.get(eid))
        if not cutoff or not item_dt or item_dt > cutoff:
            count += 1
    return count


def label_from(value: Any) -> str:
    label = norm(value)
    return label if label in VALID_LABELS else "unclear"


def prediction_common(
    sample: dict[str, Any],
    system_name: str,
    label: str,
    confidence: float,
    evidence_ids: list[str],
    evidence_times: list[str],
    failure_reason: str,
    model_name: str = MODEL_NAME,
) -> dict[str, Any]:
    return {
        "sample_id": norm(sample.get("sample_id")),
        "cluster_id": norm(sample.get("cluster_id")),
        "system_name": system_name,
        "predicted_label": label_from(label),
        "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
        "evidence_ids": [norm(item) for item in evidence_ids if norm(item)],
        "evidence_times": [norm(item) for item in evidence_times if norm(item)],
        "publish_time": norm(sample.get("publish_time") or sample.get("event_time")),
        "failure_reason": failure_reason,
        "model_name": model_name,
        "prompt_version": "src.event_credibility.fulltext.build_llm_prompt",
        "rule_version": "task2_final_cached_dea_v1.DEAConfig.trigger_policy=v2",
        "financial_layer_used": False,
        "market_layer_used": False,
        "future_evidence_used": 0,
        "request_attempts": 0,
        "source_type": norm(sample.get("source_type")),
        "event_name": norm(sample.get("event_name")),
        "target": f"{norm(sample.get('target_name'))} / {norm(sample.get('target_code'))}".strip(" /"),
        "target_code": norm(sample.get("target_code")),
        "target_name": norm(sample.get("target_name")),
        "financial_context_available": False,
        "market_context_available": False,
    }


def missing_prediction(sample: dict[str, Any], system_name: str, reason: str) -> dict[str, Any]:
    return prediction_common(sample, system_name, "unclear", 0.0, [], [], reason)


def round1_prediction(raw: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    round1 = raw.get("round1") or {}
    ids = [norm(item) for item in round1.get("evidence_ids") or [] if norm(item)]
    times = evidence_time_map(raw)
    failure = "" if round1.get("status") == "ok" and label_from(round1.get("verdict")) != "unclear" else norm(round1.get("error_type") or raw.get("error_type") or "round1_invalid_or_failed")
    row = prediction_common(sample, "Round1", round1.get("verdict"), safe_float(round1.get("confidence")), ids, [times.get(eid, "") for eid in ids], failure)
    row["request_attempts"] = int(round1.get("attempt_count") or 0)
    row["future_evidence_used"] = future_evidence_used_count(raw, row["publish_time"], ids)
    return row


def final_prediction(raw: dict[str, Any], sample: dict[str, Any], system_name: str, financial_layer: bool = False) -> dict[str, Any]:
    ids = [norm(item.get("evidence_id")) for item in raw.get("final_evidence") or [] if isinstance(item, dict) and norm(item.get("evidence_id"))]
    times = evidence_time_map(raw)
    failure = "" if raw.get("status") in {"success", "round2_failed"} and label_from(raw.get("final_verdict")) != "unclear" else norm(raw.get("error_type") or raw.get("error_message") or "final_invalid_or_failed")
    row = prediction_common(sample, system_name, raw.get("final_verdict"), safe_float(raw.get("final_confidence")), ids, [times.get(eid, "") for eid in ids], failure)
    row["request_attempts"] = int((raw.get("round1") or {}).get("attempt_count") or 0) + int((raw.get("round2") or {}).get("attempt_count") or 0)
    row["future_evidence_used"] = future_evidence_used_count(raw, row["publish_time"], ids)
    row["financial_layer_used"] = bool(financial_layer)
    row["financial_context_available"] = bool(financial_layer and raw.get("financial_context"))
    return row


def predictions_from_raw(raw_rows: list[dict[str, Any]], samples: list[dict[str, Any]], system_name: str, financial_layer: bool = False) -> list[dict[str, Any]]:
    raw_by_id = {norm(row.get("cluster_id")): row for row in raw_rows}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sid = norm(sample.get("sample_id"))
        raw = raw_by_id.get(sid)
        if not raw:
            rows.append(missing_prediction(sample, system_name, "missing_raw_prediction"))
        elif system_name == "Round1":
            rows.append(round1_prediction(raw, sample))
        else:
            rows.append(final_prediction(raw, sample, system_name, financial_layer=financial_layer))
    return rows


def attach_financial_context(rows: list[dict[str, Any]], financial_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {norm(row.get("cluster_id")): row for row in financial_rows}
    enriched: list[dict[str, Any]] = []
    future_count = 0
    available_count = 0
    for row in rows:
        audit = by_id.get(norm(row.get("sample_id"))) or {}
        ctx = audit.get("financial_context") or {}
        future_count += int(bool(ctx.get("future_financial_data_used")))
        available_count += int(bool((ctx.get("data_availability") or {}).get("has_candidate_records")))
        merged = dict(row)
        merged["financial_context"] = ctx
        enriched.append(merged)
    if future_count:
        raise ValueError(f"future financial data used in closed-track context: {future_count}")
    return enriched, {"financial_context_available_count": available_count, "future_financial_data_used_count": future_count}


def financial_market_predictions(
    financial_predictions: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    financial_audit_rows: list[dict[str, Any]],
    market_cache: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    market_index = build_market_index(load_market_cache(market_cache))
    audit_by_id = {norm(row.get("cluster_id")): row for row in financial_audit_rows}
    pred_by_id = {norm(row.get("sample_id")): row for row in financial_predictions}
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    future_count = 0
    effective_count = 0
    for sample in samples:
        sid = norm(sample.get("sample_id"))
        base = pred_by_id.get(sid) or missing_prediction(sample, "Financial-enhanced CachedDEA", "missing_financial_prediction")
        market_context = evaluate_market_rule(sample, audit_by_id.get(sid), market_index)
        fused = fuse_market_rule({"verdict": base.get("predicted_label"), "confidence": base.get("confidence"), "source_status": "financial_enhanced_cached_dea"}, market_context)
        row = dict(base)
        row.update(
            {
                "system_name": "Financial + market-rule Enhanced CachedDEA",
                "predicted_label": label_from(fused.get("verdict")),
                "confidence": safe_float(fused.get("confidence")),
                "market_layer_used": True,
                "market_context_available": bool(market_context.get("causal_market_available")),
                "market_claim_type": market_context.get("market_claim_type"),
                "market_consistency_verdict": market_context.get("market_consistency_verdict"),
                "market_evidence_ids": market_context.get("market_evidence_ids") or [],
                "market_evidence_date": market_context.get("market_evidence_date") or "",
                "market_rule_effective": bool(fused.get("market_rule_effective")),
                "rule_version": "task2_final_cached_dea_v1 + run_task2_market_rule_enhanced_cached_dea.RuleConfig",
            }
        )
        future_flag = bool(market_context.get("future_market_data_used_in_closed_track"))
        row["future_evidence_used"] = int(row.get("future_evidence_used") or 0) + int(future_flag)
        future_count += int(future_flag)
        effective_count += int(bool(fused.get("market_rule_effective")))
        output_rows.append(row)
        audit_rows.append({"sample_id": sid, **market_context, "market_rule_enhanced_result": fused})
    if future_count:
        raise ValueError(f"future market data used in closed-track: {future_count}")
    summary = {
        "market_cache": rel(market_cache),
        "market_cache_sha256": sha256_file(market_cache),
        "market_rows": len(audit_rows),
        "market_rule_effective_count": effective_count,
        "future_market_data_used_in_closed_track_count": future_count,
        "market_claim_type_distribution": dict(Counter(norm(row.get("market_claim_type")) for row in audit_rows)),
        "market_consistency_verdict_distribution": dict(Counter(norm(row.get("market_consistency_verdict")) for row in audit_rows)),
    }
    return output_rows, audit_rows, summary


def validate_prediction_rows(filename: str, rows: list[dict[str, Any]], expected_ids: set[str]) -> dict[str, Any]:
    ids = [norm(row.get("sample_id")) for row in rows]
    forbidden = sorted({field for row in rows for field in row if field in GOLD_LIKE_FIELDS})
    invalid = sorted(set(norm(row.get("predicted_label")) for row in rows) - VALID_LABELS)
    if forbidden:
        raise ValueError(f"{filename} contains forbidden gold-like fields: {forbidden}")
    if set(ids) != expected_ids or len(ids) != len(expected_ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{filename} sample_id validation failed")
    if invalid:
        raise ValueError(f"{filename} has invalid labels: {invalid}")
    return {
        "rows": len(rows),
        "unique_sample_ids": len(set(ids)),
        "failed_count": sum(1 for row in rows if norm(row.get("failure_reason"))),
        "illegal_label_count": len(invalid),
        "gold_field_leakage_count": len(forbidden),
        "future_evidence_used": sum(int(row.get("future_evidence_used") or 0) for row in rows),
        "prediction_distribution": dict(Counter(norm(row.get("predicted_label")) for row in rows)),
    }


def inventory_systems(service_audit: dict[str, Any], included: Iterable[str]) -> dict[str, Any]:
    included_set = set(included)
    systems: dict[str, Any] = {}
    for name in PREDICTION_FILES:
        systems[name] = {
            "historical_implementation_exists": True,
            "implementation_created_before_gold": True,
            "frozen_prompt_exists": True,
            "frozen_rules_exist": True,
            "frozen_thresholds_exist": True,
            "runnable_on_unlabeled_input": name in included_set,
            "requires_dianjin": name in {"Round1", "DEA v2", "CachedDEA", "Financial-enhanced CachedDEA"},
            "gold_required_for_inference": False,
            "future_evidence_possible": False,
            "eligible_for_posthoc_run": name in included_set,
            "included": name in included_set,
            "exclusion_reason": "" if name in included_set else "not run",
        }
    systems["DEA v2"]["implementation_note"] = "shares canonical run_dea_batch(mode='dea', trigger_policy='v2') output with CachedDEA in this post-hoc wrapper"
    systems["CachedDEA"]["implementation_note"] = "final CachedDEA v1 uses the same canonical run_dea_batch(mode='dea', trigger_policy='v2') output, with cached retrieval index"
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_type": "dianjin_posthoc_supplemental",
        "service_audit_ok": bool(service_audit.get("service_ok")),
        "gold_content_read": False,
        "old_gold_used_for_prediction": False,
        "systems": systems,
    }


def write_prediction_freeze(
    output_dir: Path,
    input_audit: dict[str, Any],
    gold_audit: dict[str, Any],
    service_audit: dict[str, Any],
    inventory: dict[str, Any],
    inference_manifest: dict[str, Any],
    prediction_outputs: dict[str, list[dict[str, Any]]],
    code_paths: list[Path],
    limit: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = set(inference_manifest["selected_sample_ids"])
    prediction_files: dict[str, dict[str, Any]] = {}
    for filename, rows in prediction_outputs.items():
        path = output_dir / filename
        write_jsonl(path, rows)
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        validation = validate_prediction_rows(filename, rows, expected_ids)
        prediction_files[filename] = {
            "path": rel(path),
            "sha256": sha256_file(path),
            "read_only": not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)),
            "system_name": rows[0].get("system_name") if rows else "",
            "actual_model_name": MODEL_NAME,
            **validation,
        }
    manifest = {
        "status": "prediction_frozen_before_this_run_gold_read",
        "evaluation_type": "dianjin_posthoc_supplemental",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "freeze_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "limit": limit,
        "input": input_audit,
        "gold_file_stat_only": gold_audit,
        "service_audit_path": rel(output_dir / "dianjin_service_audit.json"),
        "frozen_system_inventory_path": rel(output_dir / "frozen_system_inventory.json"),
        "inference_manifest_path": rel(output_dir / "inference_manifest.json"),
        "prediction_files": prediction_files,
        "code_and_config_sha256": {rel(path): sha256_file(path) for path in code_paths if path.exists()},
        "training_performed": False,
        "tuning_performed": False,
        "prompt_tuning_performed": False,
        "threshold_search_performed": False,
        "gold_content_read": False,
        "gold_read": False,
        "gold_was_historically_known_before_run": True,
        "old_gold_used_for_prediction": False,
        "new_gold_used_for_prediction": False,
        "future_evidence_used": sum(payload["future_evidence_used"] for payload in prediction_files.values()),
        "model_name_all_predictions": MODEL_NAME,
        "prediction_frozen": True,
        "prediction_frozen_before_this_run_gold_read": True,
        "official_independent_claim_allowed": False,
    }
    if manifest["future_evidence_used"]:
        raise ValueError(f"future evidence used in closed-track predictions: {manifest['future_evidence_used']}")
    write_json(output_dir / "prediction_freeze_manifest.json", manifest)
    lines = [f"{payload['sha256']}  {payload['path']}" for payload in prediction_files.values()]
    (output_dir / "prediction_sha256_before_gold.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for filename, payload in prediction_files.items():
        if sha256_file(resolve(payload["path"])) != payload["sha256"]:
            raise ValueError(f"post-freeze SHA mismatch for {filename}")
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
    gold_path = resolve(args.gold)
    output_dir = resolve(args.output_dir)
    if str(output_dir).rstrip("/") == str((ROOT / "outputs/event_credibility/task2_independent500_v2_evaluation_v1")).rstrip("/"):
        raise ValueError("refusing to write into the original official evaluation directory")
    if (output_dir / "gold_validation.json").exists() or (output_dir / "final_run_state.json").exists():
        raise RuntimeError("this post-hoc output directory already has gold/evaluation outputs; refusing to rerun inference")
    if (output_dir / "prediction_freeze_manifest.json").exists() and not args.resume:
        existing = read_json(output_dir / "prediction_freeze_manifest.json")
        if existing.get("gold_content_read") or existing.get("gold_read"):
            raise RuntimeError("prediction_freeze_manifest already has gold read; refusing pre-gold inference")
        raise FileExistsError(f"{output_dir}/prediction_freeze_manifest.json exists; pass --resume to verify or continue raw DEA checkpoints")
    if args.resume and (output_dir / "prediction_freeze_manifest.json").exists():
        manifest = read_json(output_dir / "prediction_freeze_manifest.json")
        if manifest.get("gold_content_read") or manifest.get("gold_read"):
            raise RuntimeError("cannot resume pre-gold inference after this run read gold")
        for payload in (manifest.get("prediction_files") or {}).values():
            if sha256_file(resolve(payload["path"])) != payload["sha256"]:
                raise ValueError(f"prediction SHA mismatch on resume: {payload}")
        return {"status": "existing_prediction_freeze_verified", "output_dir": rel(output_dir), "manifest": manifest}

    output_dir.mkdir(parents=True, exist_ok=True)
    service_audit = dianjin_service_audit()
    write_json(output_dir / "dianjin_service_audit.json", service_audit)
    gold_audit = gold_file_stat_only(gold_path)
    source_rows_all = read_jsonl(input_path)
    limit = int(args.limit or 0)
    input_audit = validate_input(input_path, source_rows_all, limit)
    selected_source = source_rows_all[:limit] if limit and limit > 0 else source_rows_all
    dea_rows, sanitation_summary = build_dea_rows(selected_source)
    selected_ids = [row["sample_id"] for row in dea_rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("duplicate selected sample IDs after conversion")

    config = load_config(resolve(args.config))
    workers = int(args.workers or min(4, int(config.get("workers", 4))))
    runtime_config = configure_runtime(config, workers)
    dea_config = build_dea_config(config)
    article_index = load_causal_article_index(config)
    checkpoint_every = int(config.get("checkpoint_interval") or 10)

    raw_cached_path = output_dir / "raw_cached_dea_results.jsonl"
    cached_summary = run_dea_phase(
        dea_rows,
        raw_cached_path,
        runtime_config,
        dea_config,
        workers,
        resume=bool(args.resume),
        article_index=article_index,
        checkpoint_every=checkpoint_every,
    )
    raw_cached = read_jsonl(raw_cached_path) if raw_cached_path.exists() else []

    csmar_records = load_csmar_financial_records_v2(resolve(args.csmar_cache))
    financial_audit_rows = analyze_financial_rows(dea_rows, csmar_records, limit=0)
    write_jsonl(output_dir / "financial_audit_500.jsonl", financial_audit_rows)
    financial_dea_rows, financial_summary = attach_financial_context(dea_rows, financial_audit_rows)
    raw_financial_path = output_dir / "raw_financial_cached_dea_results.jsonl"
    financial_cached_summary = run_dea_phase(
        financial_dea_rows,
        raw_financial_path,
        runtime_config,
        dea_config,
        workers,
        resume=bool(args.resume),
        article_index=article_index,
        checkpoint_every=checkpoint_every,
    )
    raw_financial = read_jsonl(raw_financial_path) if raw_financial_path.exists() else []

    prediction_outputs: dict[str, list[dict[str, Any]]] = {}
    prediction_outputs[PREDICTION_FILES["Round1"]] = predictions_from_raw(raw_cached, dea_rows, "Round1")
    prediction_outputs[PREDICTION_FILES["DEA v2"]] = predictions_from_raw(raw_cached, dea_rows, "DEA v2")
    prediction_outputs[PREDICTION_FILES["CachedDEA"]] = predictions_from_raw(raw_cached, dea_rows, "CachedDEA")
    prediction_outputs[PREDICTION_FILES["Financial-enhanced CachedDEA"]] = predictions_from_raw(
        raw_financial,
        dea_rows,
        "Financial-enhanced CachedDEA",
        financial_layer=True,
    )
    market_predictions, market_audit_rows, market_summary = financial_market_predictions(
        prediction_outputs[PREDICTION_FILES["Financial-enhanced CachedDEA"]],
        dea_rows,
        financial_audit_rows,
        resolve(args.market_cache),
    )
    prediction_outputs[PREDICTION_FILES["Financial + market-rule Enhanced CachedDEA"]] = market_predictions
    write_jsonl(output_dir / "market_rule_audit_500.jsonl", market_audit_rows)

    inventory = inventory_systems(service_audit, PREDICTION_FILES.keys())
    inference_manifest = {
        "evaluation_type": "dianjin_posthoc_supplemental",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_path": rel(input_path),
        "selected_sample_ids": selected_ids,
        "selected_rows": len(selected_ids),
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "workers": workers,
        "temperature": float(config.get("temperature", 0)),
        "seed": SEED,
        "cached_dea_raw_path": rel(raw_cached_path),
        "financial_cached_dea_raw_path": rel(raw_financial_path),
        "cached_dea_summary": cached_summary,
        "financial_cached_dea_summary": financial_cached_summary,
        "financial_summary": financial_summary,
        "market_summary": market_summary,
        "sanitation_summary": sanitation_summary,
        "gold_content_read": False,
        "training_performed": False,
        "tuning_performed": False,
        "prompt_tuning_performed": False,
        "threshold_search_performed": False,
        "future_evidence_used": 0,
    }
    write_json(output_dir / "pre_gold_input_audit.json", {"input": input_audit, "gold_file_stat_only": gold_audit, "sanitation": sanitation_summary})
    write_json(output_dir / "frozen_system_inventory.json", inventory)
    write_json(output_dir / "inference_manifest.json", inference_manifest)
    manifest = write_prediction_freeze(
        output_dir,
        input_audit,
        gold_audit,
        service_audit,
        inventory,
        inference_manifest,
        prediction_outputs,
        [
            Path(__file__).resolve(),
            ROOT / "scripts/evaluate_task2_independent500_v2_dianjin_posthoc_once.py",
            ROOT / "scripts/run_task2_final_cached_dea_v1.py",
            ROOT / "scripts/run_task2_financial_enhanced_cached_dea_v2.py",
            ROOT / "scripts/run_task2_market_rule_enhanced_cached_dea.py",
            ROOT / "src/event_credibility/dea.py",
            ROOT / "src/event_credibility/fulltext.py",
            ROOT / "src/event_credibility/financial_consistency.py",
            ROOT / "src/event_credibility/resset_market_adapter.py",
            resolve(args.config),
        ],
        limit,
    )
    run_state = {
        "status": "prediction_frozen_before_this_run_gold_read",
        "evaluation_type": "dianjin_posthoc_supplemental",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current_stage": "stage_a_prediction_frozen",
        "independent_test_rows": len(selected_ids),
        "independent_test_unique_ids": len(set(selected_ids)),
        "model": MODEL_NAME,
        "prediction_files": manifest["prediction_files"],
        "prediction_sha_match_after_chmod": True,
        "training_performed": False,
        "tuning_performed": False,
        "prompt_tuning_performed": False,
        "threshold_search_performed": False,
        "gold_content_read": False,
        "future_evidence_used": 0,
        "old_gold_used_for_prediction": False,
        "new_gold_used_for_prediction": False,
        "prediction_frozen": True,
        "included_systems": list(PREDICTION_FILES),
        "original_official_outputs_modified": False,
    }
    write_json(output_dir / "pre_gold_run_state.json", run_state)
    sha256s = write_sha256s(output_dir, [input_path, gold_path])
    return {
        "status": "prediction_frozen_before_this_run_gold_read",
        "output_dir": rel(output_dir),
        "included_systems": run_state["included_systems"],
        "prediction_files": manifest["prediction_files"],
        "sha256s": sha256s,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--market-cache", type=Path, default=DEFAULT_MARKET_CACHE)
    parser.add_argument("--csmar-cache", type=Path, default=DEFAULT_CSMAR_CACHE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
