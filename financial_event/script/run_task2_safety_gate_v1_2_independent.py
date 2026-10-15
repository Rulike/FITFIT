#!/usr/bin/env python3
"""Run the blind Safety Gate v1.2 independent validation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_task2_safety_gate_v1_2_independent_set as sample_builder  # noqa: E402
from src.event_credibility.dea import (  # noqa: E402
    DEAConfig,
    FulltextArticleIndex,
    check_vllm_models,
    run_dea_batch,
)
from src.event_credibility.dea_safety_gate_adapter import (  # noqa: E402
    build_safety_gate_claim,
    build_safety_gate_evidence_records,
    canonical_result_for_gate,
)
from src.event_credibility.dea_safety_gate_v1_2 import (  # noqa: E402
    apply_dea_safety_gate_v1_2,
    load_dea_safety_gate_v1_2_config,
)
from src.event_credibility.fulltext import (  # noqa: E402
    llm_runtime_config,
    read_jsonl_records,
    run_llm_review,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent"
DEFAULT_CONFIG = ROOT / "configs/task2_dea_safety_gate_v1_2.yaml"
DEFAULT_ARTICLE_INDEX = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"
DEFAULT_MEMBERS = ROOT / "outputs/event_clustering/event_cluster_members.parquet"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "DianJin-R1-32B"

EXPECTED_V1_2_SHA256 = {
    "src/event_credibility/dea_safety_gate_v1_2.py": "21316a077436a3cb9677249a764cc74a324ea8f0cf4ad3b54c68df5950fdfc33",
    "configs/task2_dea_safety_gate_v1_2.yaml": "1c97e3ff54c6d3d1bddf6ea1b351e13026311f88f54de89e2978aa31b62a7e5d",
}

FROZEN_FILES = [
    Path("src/event_credibility/dea_safety_gate.py"),
    Path("configs/task2_dea_safety_gate_v1.yaml"),
    Path("scripts/test_task2_dea_safety_gate.py"),
    Path("src/event_credibility/dea_safety_gate_v1_1.py"),
    Path("configs/task2_dea_safety_gate_v1_1.yaml"),
    Path("scripts/test_task2_dea_safety_gate_v1_1.py"),
    Path("src/event_credibility/dea_safety_gate_adapter.py"),
    Path("src/event_credibility/dea_safety_gate_v1_2.py"),
    Path("configs/task2_dea_safety_gate_v1_2.yaml"),
]

FORMAL_OUTPUTS = [
    Path("outputs/event_credibility/llm_review_results.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.parquet"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.xlsx"),
]

HUMAN_FIELD_PATTERNS = (
    "human_",
    "gold_",
    "audit_",
    "preferred_result",
    "claim_is_clear",
    "evidence_relevance",
    "evidence_stance",
    "evidence_independent",
)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths: list[Path]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in paths:
        resolved = resolve(path)
        output[str(path)] = sha256(resolved) if resolved.exists() else "missing"
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_records(path) if path.exists() else []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("input_index", 10**9)) if str(row.get("input_index", "")).strip() else 10**9,
            norm(row.get("cluster_id")),
        ),
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_known_outputs(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()


def validate_samples(rows: list[dict[str, Any]], expected_count: int = 50) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} independent samples, got {len(rows)}")
    cluster_ids = [norm(row.get("cluster_id")) for row in rows]
    if any(not cid for cid in cluster_ids):
        raise ValueError("independent samples contain empty cluster_id")
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("independent samples contain duplicate cluster_id")
    input_indices = [int(row.get("input_index")) for row in rows]
    if sorted(input_indices) != list(range(expected_count)):
        raise ValueError("input_index must be a complete 0..49 sequence")
    counts = Counter(norm(row.get("sample_group")) for row in rows)
    if counts.get("primary", 0) != 40 or counts.get("contradiction_stress", 0) != 10:
        raise ValueError(f"unexpected sample-group counts: {dict(counts)}")
    for row in rows:
        if not isinstance(row.get("evidence_items"), list) or not row["evidence_items"]:
            raise ValueError(f"{row['cluster_id']} has no evidence_items")
        for item in row["evidence_items"]:
            if not isinstance(item, dict) or not norm(item.get("evidence_id")):
                raise ValueError(f"{row['cluster_id']} has an untraceable evidence item")
        leaked = [
            key
            for key in row
            if any(pattern in str(key).lower() for pattern in HUMAN_FIELD_PATTERNS)
        ]
        if leaked:
            raise ValueError(f"human fields leaked into sample {row['cluster_id']}: {leaked}")
    return {
        "sample_count": len(rows),
        "primary_count": counts.get("primary", 0),
        "contradiction_stress_count": counts.get("contradiction_stress", 0),
        "cluster_id_count": len(set(cluster_ids)),
    }


def latest_by_cluster(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = norm(row.get("cluster_id"))
        if not cid:
            continue
        mapping[cid] = row
    return mapping


def evidence_rows_for_gate(sample: dict[str, Any], cached: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for container in (
        sample.get("evidence_items"),
        cached.get("initial_evidence"),
        cached.get("additional_evidence"),
        cached.get("final_evidence"),
    ):
        if not isinstance(container, list):
            continue
        values.extend(dict(item) for item in container if isinstance(item, dict))
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in values:
        evidence_id = norm(item.get("evidence_id") or item.get("id"))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        output.append(item)
    return output


def extract_round1_rows(samples: list[dict[str, Any]], cached_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_by_id = latest_by_cluster(samples, "samples")
    result: list[dict[str, Any]] = []
    for cached in cached_rows:
        cid = norm(cached.get("cluster_id"))
        nested = dict(cached.get("round1") or {})
        nested.update(
            {
                "cluster_id": cid,
                "input_index": cached.get("input_index", sample_by_id.get(cid, {}).get("input_index")),
                "sample_group": sample_by_id.get(cid, {}).get("sample_group", ""),
                "status": nested.get("status", "failed" if not nested.get("verdict") else "ok"),
                "llm_valid": bool(nested.get("llm_valid")),
            }
        )
        result.append(nested)
    return result


def build_safety_gate_rows(
    samples: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    cached_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_id = latest_by_cluster(baseline_rows, "baseline")
    cached_by_id = latest_by_cluster(cached_rows, "cached_dea")
    output: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda row: int(row["input_index"])):
        cid = norm(sample.get("cluster_id"))
        baseline = baseline_by_id.get(cid, {})
        cached = cached_by_id.get(cid, {})
        round1 = cached.get("round1") if isinstance(cached.get("round1"), dict) else {}
        evidence_rows = evidence_rows_for_gate(sample, cached)
        claim = build_safety_gate_claim(sample, sample)
        evidence_records = build_safety_gate_evidence_records(cid, evidence_rows, cached, round1)

        cached_for_gate = dict(cached)
        if not cached_for_gate.get("evidence_ids"):
            cached_for_gate["evidence_ids"] = [
                norm(item.get("evidence_id"))
                for item in cached_for_gate.get("final_evidence", [])
                if isinstance(item, dict) and norm(item.get("evidence_id"))
            ]
        canonical_round1 = canonical_result_for_gate(round1, evidence_records, role="round1")
        canonical_cached = canonical_result_for_gate(cached_for_gate, evidence_records, role="cached")
        gate = apply_dea_safety_gate_v1_2(
            claim,
            baseline,
            canonical_round1,
            canonical_cached,
            evidence_records,
            config,
        )
        output.append(
            {
                "cluster_id": cid,
                "input_index": sample.get("input_index"),
                "sample_group": sample.get("sample_group", ""),
                "subject": norm(claim.get("subject")),
                "event_type": norm(claim.get("event_type")),
                "event_time": norm(claim.get("event_time")),
                "claim_text": norm(claim.get("claim_text")),
                "baseline_verdict": norm(baseline.get("verdict")),
                "round1_verdict": norm(canonical_round1.get("verdict")),
                "cached_verdict": norm(canonical_cached.get("verdict")),
                "cached_confidence": canonical_cached.get("confidence"),
                "safety_gate_v1_2_verdict": norm(gate.get("gated_verdict")),
                "safety_gate_v1_2_selected_source": norm(gate.get("selected_result_source")),
                "gate_blocked": bool(gate.get("gate_blocked")),
                "gate_reasons": list(gate.get("gate_reasons") or []),
                "fallback_verdict": norm(gate.get("fallback_verdict")),
                "fallback_reason": norm(gate.get("fallback_reason")),
                "structurally_unverifiable": bool(gate.get("structurally_unverifiable")),
                "raw_claim_incomplete": bool(gate.get("raw_claim_incomplete")),
                "evidence_resolves_claim": bool(gate.get("evidence_resolves_claim")),
                "cited_evidence_ids": list(gate.get("cited_evidence_ids") or []),
                "valid_cited_evidence_ids": list(gate.get("valid_cited_evidence_ids") or []),
                "unmatched_cited_evidence_ids": list(gate.get("unmatched_cited_evidence_ids") or []),
                "cited_evidence_scope_size": int(gate.get("cited_evidence_scope_size") or 0),
                "cited_subject_match": bool(gate.get("cited_subject_match")),
                "cited_time_match": bool(gate.get("cited_time_match")),
                "cited_contains_concrete_fact": bool(gate.get("cited_contains_concrete_fact")),
                "cited_only_investor_questions": bool(gate.get("cited_only_investor_questions")),
                "cited_only_market_signals": bool(gate.get("cited_only_market_signals")),
                "cited_traceable": bool(gate.get("cited_traceable")),
                "direct_evidence_ids": list(gate.get("direct_evidence_ids") or []),
                "direct_evidence_diagnostic_only": bool(gate.get("direct_evidence_diagnostic_only")),
                "direct_evidence_scores": gate.get("direct_evidence_scores") or {},
                "safety_gate_version": norm(gate.get("safety_gate_version")),
                "evidence_records": evidence_records,
                "human_fields_used": False,
            }
        )
    return output


def build_blind_queue(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda row: int(row["input_index"])):
        queue.append(
            {
                "cluster_id": norm(sample.get("cluster_id")),
                "event_id": norm(sample.get("cluster_id")),
                "input_index": int(sample.get("input_index")),
                "sample_group": norm(sample.get("sample_group")),
                "target_name": norm(sample.get("target_name") or sample.get("subject")),
                "event_name": norm(sample.get("event_name") or sample.get("schema_id")),
                "schema_id": norm(sample.get("schema_id")),
                "event_claim": norm(sample.get("event_claim") or sample.get("claim_text")),
                "event_time": norm(sample.get("event_time")),
                "evidence_items": [
                    dict(item)
                    for item in sample.get("evidence_items", [])
                    if isinstance(item, dict)
                ],
            }
        )
    return queue


def check_no_human_fields(value: Any, path: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(pattern in lowered for pattern in HUMAN_FIELD_PATTERNS):
                errors.append(f"{path}.{key}".strip("."))
            errors.extend(check_no_human_fields(item, f"{path}.{key}".strip(".")))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            errors.extend(check_no_human_fields(item, f"{path}[{idx}]"))
    return errors


def numeric_stats(values: list[float]) -> dict[str, float]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return {"count": 0, "mean": 0.0, "p95": 0.0, "total": 0.0}
    ordered = sorted(cleaned)
    position = 0.95 * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    p95 = ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {
        "count": len(cleaned),
        "mean": round(sum(cleaned) / len(cleaned), 6),
        "p95": round(p95, 6),
        "total": round(sum(cleaned), 6),
    }


def model_error_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        text = json.dumps(record, ensure_ascii=False)
        if re.search(r"CUDA OOM|out of memory|\bOOM\b", text, re.I):
            counts["cuda_oom"] += 1
        if re.search(r"timeout|timed out", text, re.I):
            counts["timeout"] += 1
        if re.search(r"HTTP(?:Error)?[^0-9]*(?:500|502|503|504)|\b(?:500|502|503|504)\b", text, re.I):
            counts["http_5xx"] += 1
        if "unparseable" in text or "validation" in text and record.get("status") == "failed":
            counts["parse_failure"] += 1
    return dict(counts)


def summarize_request_records(
    baseline_rows: list[dict[str, Any]],
    cached_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_requests = sum(int(row.get("attempt_count") or 0) for row in baseline_rows)
    cached_requests = 0
    cached_retries = 0
    cached_latencies: list[float] = []
    cached_tokens: list[float] = []
    for row in cached_rows:
        for nested in (row.get("round1"), row.get("round2")):
            if not isinstance(nested, dict) or not nested:
                continue
            cached_requests += int(nested.get("attempt_count") or 0)
            cached_retries += max(0, int(nested.get("attempt_count") or 0) - 1)
            cached_latencies.append(float(nested.get("latency_seconds") or 0.0))
            cached_tokens.append(float(nested.get("effective_total_tokens") or 0.0))
    blind_requests = sum(int(row.get("attempt_count") or 0) for row in blind_rows)
    blind_retries = sum(max(0, int(row.get("attempt_count") or 0) - 1) for row in blind_rows)
    production_latencies = [float(row.get("latency_seconds") or 0.0) for row in baseline_rows]
    production_tokens = [float(row.get("effective_total_tokens") or 0.0) for row in baseline_rows]
    for row in cached_rows:
        for nested in (row.get("round1"), row.get("round2")):
            if isinstance(nested, dict) and nested:
                production_latencies.append(float(nested.get("latency_seconds") or 0.0))
                production_tokens.append(float(nested.get("effective_total_tokens") or 0.0))
    blind_latencies = [float(row.get("latency_seconds") or 0.0) for row in blind_rows]
    blind_tokens = [float(row.get("total_tokens") or 0.0) for row in blind_rows]
    all_records = baseline_rows + cached_rows + blind_rows
    return {
        "logical_model_calls": {
            "baseline": sum(int(row.get("llm_call_count") or 1) for row in baseline_rows),
            "cached_dea": sum(int(row.get("llm_call_count") or 0) for row in cached_rows),
            "blind_assistance": len(blind_rows),
        },
        "http_request_attempts": {
            "baseline": baseline_requests,
            "cached_dea": cached_requests,
            "blind_assistance": blind_requests,
            "total": baseline_requests + cached_requests + blind_requests,
        },
        "retries": {
            "cached_dea": cached_retries,
            "blind_assistance": blind_retries,
            "total": sum(max(0, int(row.get("attempt_count") or 0) - 1) for row in baseline_rows)
            + cached_retries
            + blind_retries,
        },
        "latency_seconds": {
            "production": numeric_stats(production_latencies),
            "blind_assistance": numeric_stats(blind_latencies),
            "all": numeric_stats(production_latencies + blind_latencies),
        },
        "total_tokens": {
            "production": numeric_stats(production_tokens),
            "blind_assistance": numeric_stats(blind_tokens),
            "all": numeric_stats(production_tokens + blind_tokens),
        },
        "error_counts": model_error_counts(all_records),
        "successful_baseline": sum(1 for row in baseline_rows if row.get("status") == "success"),
        "successful_cached_dea": sum(1 for row in cached_rows if row.get("status") in {"success", "round2_failed"}),
        "successful_blind_assistance": sum(1 for row in blind_rows if row.get("status") == "ok"),
        "failed_baseline": sum(1 for row in baseline_rows if row.get("status") != "success"),
        "failed_cached_dea": sum(1 for row in cached_rows if row.get("status") not in {"success", "round2_failed"}),
        "failed_blind_assistance": sum(1 for row in blind_rows if row.get("status") != "ok"),
    }


def write_phase_checkpoint(
    path: Path,
    stage: str,
    sample_summary: dict[str, Any],
    phase_summaries: dict[str, Any],
    model_calls: int = 0,
    status: str = "running",
) -> None:
    write_json(
        path,
        {
            "step": "4.4-H-A",
            "stage": stage,
            "status": status,
            "sample_summary": sample_summary,
            "phase_summaries": phase_summaries,
            "model_calls": model_calls,
            "resume_command": "python scripts/run_task2_safety_gate_v1_2_independent.py --resume",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def configure_runtime(args: argparse.Namespace) -> None:
    os.environ["TASK2_LLM_BASE_URL"] = args.base_url
    os.environ["TASK2_LLM_MODEL"] = args.model
    os.environ["TASK2_LLM_WIRE_API"] = args.wire_api
    os.environ["TASK2_LLM_ENDPOINT_STYLE"] = args.endpoint_style
    os.environ["TASK2_LLM_TIMEOUT"] = str(args.timeout)
    os.environ["TASK2_LLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["TASK2_LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["TASK2_LLM_MAX_RETRIES"] = str(args.max_retries)
    os.environ["TASK2_LLM_FORMAT_RETRIES"] = str(args.format_retries)
    os.environ["TASK2_LLM_WORKERS"] = str(args.workers)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        remove_known_outputs(output_dir)

    sample_path = output_dir / "independent_samples.jsonl"
    primary_path = output_dir / "primary_samples.jsonl"
    stress_path = output_dir / "contradiction_stress_samples.jsonl"
    if not sample_path.exists():
        sample_builder.build_and_write(
            resolve(args.scores),
            resolve(args.evidence),
            output_dir,
            primary_count=40,
            stress_count=10,
            seed=args.seed,
            overwrite=False,
        )
    samples = read_jsonl(sample_path)
    sample_summary = validate_samples(samples)

    frozen_before = hash_files(FROZEN_FILES)
    formal_before = hash_files(FORMAL_OUTPUTS)
    for rel_path, expected in EXPECTED_V1_2_SHA256.items():
        actual = frozen_before.get(rel_path)
        if actual != expected:
            raise RuntimeError(f"frozen v1.2 hash mismatch for {rel_path}: expected {expected}, got {actual}")

    configure_runtime(args)
    runtime = llm_runtime_config()
    model_check = check_vllm_models(
        args.base_url,
        args.endpoint_style,
        api_key=runtime.get("api_key", ""),
        timeout=min(args.timeout, 30.0),
    )
    if not model_check.get("ok"):
        raise RuntimeError(f"vLLM service is not reachable: {model_check.get('error')}")
    advertised_models = [
        item.get("id")
        for item in (model_check.get("data") or {}).get("data", [])
        if isinstance(item, dict)
    ]
    if args.model not in advertised_models:
        raise RuntimeError(f"expected model {args.model!r} is not advertised: {advertised_models}")

    checkpoint_path = output_dir / "run_checkpoint.json"
    phase_summaries: dict[str, Any] = {}
    write_phase_checkpoint(checkpoint_path, "starting", sample_summary, phase_summaries)

    dea_config = DEAConfig(
        initial_evidence_limit=args.initial_evidence_limit,
        additional_evidence_limit=args.additional_evidence_limit,
        confidence_threshold=args.confidence_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        max_rounds=2,
        trigger_policy=args.trigger_policy,
        compact_prompt=True,
    )

    baseline_path = output_dir / "baseline_results.jsonl"
    cached_path = output_dir / "cached_dea_results.jsonl"
    baseline_rows = read_jsonl(baseline_path)
    cached_rows = read_jsonl(cached_path)
    if len(baseline_rows) == len(samples) and len(cached_rows) == len(samples):
        baseline_summary = {"status": "reused", "completed": len(samples), "succeeded": len(samples), "failed": 0}
        cached_summary = {"status": "reused", "completed": len(samples), "succeeded": len(samples), "failed": 0}
    else:
        baseline_summary = run_dea_batch(
            samples,
            baseline_path,
            runtime,
            dea_config,
            mode="baseline",
            workers=args.workers,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            article_index=None,
        )
        baseline_rows = read_jsonl(baseline_path)
        write_jsonl(baseline_path, baseline_rows)
        write_phase_checkpoint(checkpoint_path, "baseline_complete", sample_summary, phase_summaries)
        if baseline_summary.get("safety_stop_reason"):
            write_phase_checkpoint(
                checkpoint_path,
                "stopped",
                sample_summary,
                phase_summaries,
                status="safety_stopped",
            )
            raise RuntimeError(f"baseline safety stopped: {baseline_summary['safety_stop_reason']}")

        article_index = FulltextArticleIndex(resolve(args.article_index), members_path=resolve(args.members))
        article_index.load()
        cached_summary = run_dea_batch(
            samples,
            cached_path,
            runtime,
            dea_config,
            mode="dea",
            workers=args.workers,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            article_index=article_index,
        )
        cached_rows = read_jsonl(cached_path)
        write_jsonl(cached_path, cached_rows)
        write_phase_checkpoint(checkpoint_path, "cached_dea_complete", sample_summary, phase_summaries)
        if cached_summary.get("safety_stop_reason"):
            write_phase_checkpoint(
                checkpoint_path,
                "stopped",
                sample_summary,
                phase_summaries,
                status="safety_stopped",
            )
            raise RuntimeError(f"cached DEA safety stopped: {cached_summary['safety_stop_reason']}")
    phase_summaries["baseline"] = baseline_summary
    phase_summaries["cached_dea"] = cached_summary

    if len(latest_by_cluster(baseline_rows, "baseline")) != len(samples):
        raise RuntimeError("baseline results are incomplete")
    if len(latest_by_cluster(cached_rows, "cached_dea")) != len(samples):
        raise RuntimeError("cached DEA results are incomplete")

    round1_rows = extract_round1_rows(samples, cached_rows)
    write_jsonl(output_dir / "round1_results.jsonl", round1_rows)

    config = load_dea_safety_gate_v1_2_config(resolve(args.config))
    gate_rows = build_safety_gate_rows(samples, baseline_rows, cached_rows, config)
    if len(gate_rows) != len(samples):
        raise RuntimeError("safety gate results are incomplete")
    write_jsonl(output_dir / "safety_gate_v1_2_results.jsonl", gate_rows)
    phase_summaries["safety_gate_v1_2"] = {
        "sample_count": len(gate_rows),
        "gate_blocked": sum(1 for row in gate_rows if row["gate_blocked"]),
        "verdict_distribution": dict(Counter(row["safety_gate_v1_2_verdict"] for row in gate_rows)),
        "safety_gate_version": sorted({row["safety_gate_version"] for row in gate_rows}),
    }
    write_phase_checkpoint(checkpoint_path, "safety_gate_complete", sample_summary, phase_summaries)

    blind_queue_path = output_dir / "blind_review_queue.jsonl"
    blind_queue = build_blind_queue(samples)
    write_jsonl(blind_queue_path, blind_queue)
    blind_output_path = output_dir / "model_request_log.jsonl"
    blind_summary = run_llm_review(
        blind_queue_path,
        blind_output_path,
        max_reviews=0,
        sleep_seconds=0.0,
        workers=args.workers,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        retry_failed=True,
    )
    blind_rows = read_jsonl(blind_output_path)
    write_jsonl(blind_output_path, blind_rows)
    phase_summaries["blind_assistance"] = blind_summary
    write_phase_checkpoint(checkpoint_path, "blind_assistance_complete", sample_summary, phase_summaries)
    errors = model_error_counts(baseline_rows + cached_rows + blind_rows)
    if errors.get("cuda_oom"):
        write_phase_checkpoint(
            checkpoint_path,
            "stopped",
            sample_summary,
            phase_summaries,
            status="cuda_oom",
        )
        raise RuntimeError("CUDA OOM detected; checkpoint saved and task stopped")

    request_summary = summarize_request_records(baseline_rows, cached_rows, blind_rows)
    formal_after = hash_files(FORMAL_OUTPUTS)
    frozen_after = hash_files(FROZEN_FILES)
    if formal_before != formal_after:
        raise RuntimeError("formal output SHA256 changed during independent validation")
    if frozen_before != frozen_after:
        raise RuntimeError("frozen file SHA256 changed during independent validation")

    builder_manifest_path = output_dir / "run_manifest.json"
    manifest = {}
    if builder_manifest_path.exists():
        manifest = json.loads(builder_manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "step": "4.4-H-A",
            "output_dir": str(output_dir),
            "sample_summary": sample_summary,
            "sample_files": {
                "independent_samples": str(sample_path),
                "primary_samples": str(primary_path),
                "contradiction_stress_samples": str(stress_path),
            },
            "frozen_sha256_before": frozen_before,
            "frozen_sha256_after": frozen_after,
            "frozen_files_unchanged": frozen_before == frozen_after,
            "formal_output_sha256_before": formal_before,
            "formal_output_sha256_after": formal_after,
            "formal_outputs_unchanged": formal_before == formal_after,
            "model": {
                "base_url": args.base_url,
                "model": args.model,
                "workers": args.workers,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "max_retries": args.max_retries,
                "format_retries": args.format_retries,
                "model_check": model_check,
            },
            "phase_summaries": phase_summaries,
            "request_summary": request_summary,
            "error_counts": errors,
            "blind_input_policy": {
                "human_fields_used_for_selection": False,
                "human_fields_passed_to_safety_gate": False,
                "human_fields_passed_to_blind_model": False,
                "hidden_reasoning_saved": False,
            },
            "model_calls": request_summary["http_request_attempts"]["total"],
            "status": "completed",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    write_json(builder_manifest_path, manifest)
    write_phase_checkpoint(
        checkpoint_path,
        "completed",
        sample_summary,
        phase_summaries,
        model_calls=request_summary["http_request_attempts"]["total"],
        status="completed",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scores", type=Path, default=ROOT / "outputs/event_credibility/event_credibility_scores.parquet")
    parser.add_argument("--evidence", type=Path, default=ROOT / "outputs/event_credibility/fulltext_evidence.parquet")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=sample_builder.SEED)
    parser.add_argument("--base-url", default=os.environ.get("TASK2_LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("TASK2_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--wire-api", default=os.environ.get("TASK2_LLM_WIRE_API", "chat_completions"))
    parser.add_argument("--endpoint-style", default=os.environ.get("TASK2_LLM_ENDPOINT_STYLE", "openai_v1"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("TASK2_LLM_TIMEOUT", "180")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("TASK2_LLM_MAX_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TASK2_LLM_TEMPERATURE", "0")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("TASK2_LLM_MAX_RETRIES", "2")))
    parser.add_argument("--format-retries", type=int, default=1)
    parser.add_argument("--initial-evidence-limit", type=int, default=4)
    parser.add_argument("--additional-evidence-limit", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--trigger-policy", choices=("v1", "v2"), default="v2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_pipeline(args)
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "sample_summary": manifest.get("sample_summary"),
                "model_calls": manifest.get("model_calls"),
                "formal_outputs_unchanged": manifest.get("formal_outputs_unchanged"),
                "frozen_files_unchanged": manifest.get("frozen_files_unchanged"),
                "output_dir": manifest.get("output_dir"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
