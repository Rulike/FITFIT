#!/usr/bin/env python3
"""Run the 20-sample Claim Contract Adapter v1 gray experiment."""

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

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility import dea  # noqa: E402
from src.event_credibility import fulltext  # noqa: E402
from src.event_credibility.claim_contract import (  # noqa: E402
    ADAPTER_VERSION,
    VALID_MODES,
    build_claim_contract,
)
from src.event_credibility.isr import (  # noqa: E402
    ISRConfig,
    invoke_json_round,
    norm,
    validate_final_object,
)


DEFAULT_OUTPUT = ROOT / "outputs/event_credibility/claim_contract_adapter_v1_gray"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "sample_manifest.json"
DEFAULT_SAMPLES = ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent/independent_samples.jsonl"
DEFAULT_CONFIG = ROOT / "configs/task2_claim_contract_adapter_v1.yaml"
DEFAULT_MODEL = "DianJin-R1-32B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

FROZEN_FILES = [
    Path("src/event_credibility/dea.py"),
    Path("configs/task2_dea_safety_gate_v1.yaml"),
    Path("src/event_credibility/retrieval_index.py"),
    Path("src/event_credibility/dea_safety_gate.py"),
    Path("configs/task2_dea_safety_gate_v1_1.yaml"),
    Path("src/event_credibility/dea_safety_gate_v1_1.py"),
    Path("configs/task2_dea_safety_gate_v1_2.yaml"),
    Path("src/event_credibility/dea_safety_gate_v1_2.py"),
    Path("src/event_credibility/dea_safety_gate_adapter.py"),
    Path("src/event_credibility/isr.py"),
    Path("configs/task2_isr_diagnostic_v1.yaml"),
]
FORMAL_OUTPUTS = [
    Path("outputs/event_credibility/llm_review_results.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.parquet"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.xlsx"),
    Path("reports/task2_safety_gate_v1_2_independent_human_audit_gold_filled.xlsx"),
]
INDEPENDENT_INPUTS = [
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/independent_samples.jsonl"),
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/baseline_results.jsonl"),
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/round1_results.jsonl"),
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/cached_dea_results.jsonl"),
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/safety_gate_v1_2_results.jsonl"),
]
FORBIDDEN_INPUT_KEYS = {
    "human_verdict",
    "reviewer_confirmed",
    "human_notes",
    "gold_verdict",
    "cached_verdict",
    "baseline_verdict",
    "round1_verdict",
    "safety_gate_v1_2_verdict",
    "error_type",
    "error_stage",
    "credibility_score",
    "AI辅助建议_verdict",
    "isr_result",
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(resolve(path)) if resolve(path).exists() else "missing" for path in paths}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: int(item.get("input_index", 10**9))):
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def recursive_forbidden_keys(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_INPUT_KEYS or key_text.lower().startswith("human_"):
                found.append(f"{path}.{key_text}".strip("."))
            found.extend(recursive_forbidden_keys(child, f"{path}.{key_text}".strip(".")))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(recursive_forbidden_keys(child, f"{path}[{index}]"))
    return found


def load_config(path: Path) -> tuple[ISRConfig, dea.DEAConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = payload.get("model") or {}
    dea_payload = payload.get("dea") or {}
    model_config = ISRConfig(
        base_url=str(model.get("base_url") or DEFAULT_BASE_URL),
        model=str(model.get("name") or DEFAULT_MODEL),
        temperature=float(model.get("temperature", 0.0)),
        max_tokens=int(model.get("max_tokens", 512)),
        timeout_seconds=float(model.get("timeout_seconds", 180)),
        format_retries=int(model.get("format_retries", 1)),
    )
    dea_config = dea.DEAConfig(
        initial_evidence_limit=int(dea_payload.get("initial_evidence_limit", 4)),
        additional_evidence_limit=int(dea_payload.get("additional_evidence_limit", 4)),
        confidence_threshold=float(dea_payload.get("confidence_threshold", 0.75)),
        low_confidence_threshold=float(dea_payload.get("low_confidence_threshold", 0.65)),
        max_rounds=int(dea_payload.get("max_rounds", 2)),
        trigger_policy=str(dea_payload.get("trigger_policy", "v2")),
        compact_prompt=bool(dea_payload.get("compact_prompt", True)),
    )
    if model_config.model != DEFAULT_MODEL:
        raise ValueError(f"unexpected model in config: {model_config.model}")
    return model_config, dea_config


def load_input_rows(manifest_path: Path, samples_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    if manifest.get("sample_count") != 20:
        raise ValueError("sample manifest must contain 20 samples")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("sample manifest rows must contain 20 items")
    ids = [norm(row.get("cluster_id")) for row in rows]
    if len(set(ids)) != 20 or any(not cid for cid in ids):
        raise ValueError("sample manifest contains missing or duplicate cluster_id")
    groups = Counter(norm(row.get("cohort")) for row in rows)
    if groups.get("target_difficult") != 10 or groups.get("correct_control") != 10:
        raise ValueError(f"unexpected sample cohorts: {dict(groups)}")

    independent = {
        norm(row.get("cluster_id")): row
        for row in read_jsonl(samples_path)
        if norm(row.get("cluster_id"))
    }
    output = []
    for row in sorted(rows, key=lambda item: int(item.get("input_index", 10**9))):
        cid = norm(row.get("cluster_id"))
        source = independent.get(cid)
        if not source:
            raise ValueError(f"{cid} not found in independent samples")
        evidence_items = source.get("evidence_items")
        if not isinstance(evidence_items, list) or not evidence_items:
            raise ValueError(f"{cid} has no evidence_items")
        evidence = []
        seen_ids: set[str] = set()
        for item in evidence_items:
            if not isinstance(item, dict):
                raise ValueError(f"{cid} contains malformed evidence")
            evidence_id = norm(item.get("evidence_id") or item.get("id"))
            if not re.fullmatch(r"E\d+", evidence_id) or evidence_id in seen_ids:
                raise ValueError(f"{cid} has invalid or duplicate evidence_id {evidence_id!r}")
            seen_ids.add(evidence_id)
            text = norm(item.get("sentence") or item.get("text") or item.get("content"))
            if not text or not norm(item.get("source") or item.get("article_source")):
                raise ValueError(f"{cid} evidence is not traceable")
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "article_id": norm(item.get("article_id") or item.get("trace_id")),
                    "title": norm(item.get("title") or item.get("article_title")),
                    "source": norm(item.get("source") or item.get("article_source")),
                    "source_type": norm(item.get("source_type")),
                    "publish_time": norm(item.get("publish_time") or item.get("article_publish_time")),
                    "sentence": text,
                    "evidence_source": norm(item.get("evidence_source") or "independent_cluster_pool"),
                }
            )
        contract = build_claim_contract(
            row.get("claim") or source.get("event_claim"),
            row.get("subject") or source.get("target_name"),
            row.get("schema_id") or source.get("schema_id"),
            row.get("event_name") or source.get("event_name"),
            row.get("event_time") or source.get("event_time"),
        )
        model_input = {
            "claim": contract,
            "evidence": evidence,
        }
        leaked = recursive_forbidden_keys(model_input)
        if leaked:
            raise ValueError(f"{cid} model input contains forbidden fields: {leaked}")
        output.append(
            {
                "input_index": int(row.get("input_index")),
                "cluster_id": cid,
                "cohort": norm(row.get("cohort")),
                "subgroup": norm(row.get("subgroup")),
                "contract": contract,
                "evidence": evidence,
            }
        )
    return output


def evidence_block(evidence: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{item['evidence_id']} | 来源:{item['source']} | 发布时间:{item['publish_time']} | "
        f"标题:{item['title'] or '(无)'}\n{item['sentence']}"
        for item in evidence
    )


def adapter_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    contract = sample["contract"]
    return [
        {
            "role": "system",
            "content": "你是金融事件核验员。只输出严格JSON，不输出思维过程。",
        },
        {
            "role": "user",
            "content": (
                "根据规范化主张和给定证据完成一次核验。规范化只定义核验对象，"
                "不能从证据外推新的事实。category_existence必须要求具体动作或事实，"
                "投资者提问、行情和弱关联内容不能单独support。"
                "\n\n主张契约："
                + json.dumps(contract, ensure_ascii=False)
                + "\n\n证据：\n"
                + evidence_block(sample["evidence"])
                + '\n\n严格输出：{"verdict":"support|contradict|insufficient|unclear",'
                '"confidence":0到1,"evidence_ids":[],"matched_facts":[],"reason":"简洁理由",'
                '"claim_mode":"specific_fact|category_existence|structurally_unverifiable"}'
            ),
        },
    ]


def validate_adapter_object(value: Any, allowed_ids: list[str]) -> tuple[dict[str, Any] | None, list[str]]:
    base, errors = validate_final_object(value, allowed_ids)
    if base is None:
        return None, errors
    matched = value.get("matched_facts")
    if not isinstance(matched, list) or not all(isinstance(item, str) for item in matched):
        errors.append("matched_facts must be a string list")
    mode = norm(value.get("claim_mode"))
    if mode not in VALID_MODES:
        errors.append(f"invalid claim_mode: {mode!r}")
    if errors:
        return None, errors
    return {
        **base,
        "matched_facts": [norm(item) for item in matched if norm(item)][:20],
        "claim_mode": mode,
    }, []


def adapter_result_row(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "input_index": sample["input_index"],
        "cluster_id": sample["cluster_id"],
        "cohort": sample["cohort"],
        "subgroup": sample["subgroup"],
        "claim_mode_expected": sample["contract"]["claim_mode"],
        "normalized_claim": sample["contract"]["normalized_claim"],
        "status": result.get("status", "failed"),
        "attempt_count": result.get("attempt_count", 0),
        "format_retry_count": result.get("format_retry_count", 0),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
    }
    if isinstance(result.get("parsed"), dict):
        row.update(result["parsed"])
    return row


def adapter_dea_row(sample: dict[str, Any]) -> dict[str, Any]:
    contract = sample["contract"]
    return {
        "cluster_id": sample["cluster_id"],
        "event_id": sample["cluster_id"],
        "target_name": contract["subject"],
        "target_code": "",
        "schema_id": sample["contract"].get("schema_id", ""),
        "event_name": contract["event_type"],
        "event_claim": contract["normalized_claim"],
        "event_time": contract["time_scope"],
        "evidence_items": sample["evidence"],
        "title_samples": "",
    }


def adapter_to_round1(adapter_row: dict[str, Any], initial_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {norm(item.get("evidence_id")) for item in initial_evidence}
    ids = [norm(item) for item in adapter_row.get("evidence_ids", []) if norm(item) in allowed]
    return {
        "status": "ok",
        "llm_valid": True,
        "verdict": norm(adapter_row.get("verdict")),
        "confidence": float(adapter_row.get("confidence") or 0.0),
        "reason": norm(adapter_row.get("reason")),
        "evidence_ids": ids,
        "attempt_count": int(adapter_row.get("attempt_count") or 1),
        "format_retry_count": int(adapter_row.get("format_retry_count") or 0),
        "error_type": "",
        "error_message": "",
    }


def summary_round(result: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evidence_ids": [norm(item.get("evidence_id")) for item in evidence],
        "verdict": norm(result.get("verdict")),
        "confidence": float(result.get("confidence") or 0.0),
        "reason": norm(result.get("reason")),
        "status": norm(result.get("status")),
        "llm_valid": bool(result.get("llm_valid")),
        "attempt_count": int(result.get("attempt_count") or 0),
        "format_retry_count": int(result.get("format_retry_count") or 0),
        "effective_prompt_tokens": int(result.get("effective_prompt_tokens") or result.get("prompt_tokens") or 0),
        "effective_total_tokens": int(result.get("effective_total_tokens") or result.get("total_tokens") or 0),
        "error_type": norm(result.get("error_type")),
        "error_message": norm(result.get("error_message")),
    }


def run_adapter_dea(
    sample: dict[str, Any],
    adapter_row: dict[str, Any],
    llm_config: dict[str, Any],
    dea_config: dea.DEAConfig,
    article_index: dea.FulltextArticleIndex,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    row = adapter_dea_row(sample)
    initial_evidence, rejected = dea.select_ranked_evidence(row, dea_config.initial_evidence_limit)
    round1 = adapter_to_round1(adapter_row, initial_evidence)
    trigger, reasons = dea.should_trigger_dea(
        round1,
        initial_evidence,
        dea_config.confidence_threshold,
        low_confidence_threshold=dea_config.low_confidence_threshold,
        trigger_policy=dea_config.trigger_policy,
        row=row,
    )
    acquisition = {
        "queries": [],
        "new_evidence": [],
        "new_evidence_ids": [],
        "new_source_count": 0,
        "retrieval_mode": "not_triggered",
        "retrieval_latency_seconds": 0.0,
        "global_hit_document_count": 0,
        "global_raw_hit_document_count": 0,
        "global_selected_evidence_count": 0,
        "unselected_reasons": [],
        "index_error": "",
    }
    round2 = None
    final_result = round1
    final_source = "adapter_round1"
    combined = list(initial_evidence)
    if trigger and dea_config.max_rounds >= 2:
        acquisition = dea.acquire_additional_evidence(
            row,
            initial_evidence,
            round1,
            article_index,
            dea_config.additional_evidence_limit,
        )
        combined = (initial_evidence + acquisition.get("new_evidence", []))[
            : dea_config.initial_evidence_limit + dea_config.additional_evidence_limit
        ]
        if acquisition.get("new_evidence"):
            round2 = dea.review_with_evidence(
                row,
                combined,
                sample["input_index"],
                llm_config,
                compact_prompt=dea_config.compact_prompt,
            )
            metrics.append(
                {
                    "input_index": sample["input_index"],
                    "cluster_id": sample["cluster_id"],
                    "phase": "dea_round2",
                    "logical_calls": 1,
                    "request_attempts": int(round2.get("attempt_count") or 0),
                    "format_retry_count": int(round2.get("format_retry_count") or 0),
                    "latency_seconds": float(round2.get("latency_seconds") or 0.0),
                    "prompt_tokens": int(round2.get("effective_prompt_tokens") or 0),
                    "total_tokens": int(round2.get("effective_total_tokens") or 0),
                    "error_type": norm(round2.get("error_type")),
                    "error_message": norm(round2.get("error_message")),
                }
            )
            if round2.get("status") == "ok" and round2.get("llm_valid"):
                final_result = round2
                final_source = "dea_round2"
            else:
                final_source = "adapter_round1_fallback_after_round2_failure"
        else:
            reasons.append("no_new_evidence_available")
            final_source = "adapter_round1_no_new_evidence"
    return {
        "input_index": sample["input_index"],
        "cluster_id": sample["cluster_id"],
        "cohort": sample["cohort"],
        "subgroup": sample["subgroup"],
        "normalized_claim": sample["contract"]["normalized_claim"],
        "claim_mode": sample["contract"]["claim_mode"],
        "dea_triggered": bool(trigger and dea_config.max_rounds >= 2),
        "trigger_reasons": list(dict.fromkeys(reasons)),
        "retrieval_mode": acquisition.get("retrieval_mode", "not_triggered"),
        "round1": summary_round(round1, initial_evidence),
        "acquisition": {
            "queries": acquisition.get("queries", []),
            "new_evidence_ids": acquisition.get("new_evidence_ids", []),
            "new_source_count": acquisition.get("new_source_count", 0),
            "retrieval_latency_seconds": acquisition.get("retrieval_latency_seconds", 0.0),
            "global_hit_document_count": acquisition.get("global_hit_document_count", 0),
            "global_raw_hit_document_count": acquisition.get("global_raw_hit_document_count", 0),
            "global_selected_evidence_count": acquisition.get("global_selected_evidence_count", 0),
            "unselected_reasons": acquisition.get("unselected_reasons", []),
            "index_error": acquisition.get("index_error", ""),
        },
        "round2": summary_round(round2, combined) if round2 else {},
        "round1_evidence_ids": summary_round(round1, initial_evidence)["evidence_ids"],
        "final_evidence_ids": [
            norm(item.get("evidence_id")) for item in (
                combined if final_source == "dea_round2" else initial_evidence
            )
        ],
        "final_verdict": norm(final_result.get("verdict")),
        "final_confidence": float(final_result.get("confidence") or 0.0),
        "final_reason": norm(final_result.get("reason")),
        "final_source": final_source,
        "status": "success" if final_result.get("status") == "ok" and final_result.get("llm_valid") else "failed",
        "initial_rejected": rejected[:20],
        "final_evidence": combined if final_source == "dea_round2" else initial_evidence,
        "llm_call_count": 1 + int(round2 is not None),
        "effective_total_tokens": int(round1.get("effective_total_tokens") or 0)
        + (int(round2.get("effective_total_tokens") or 0) if round2 else 0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_existing = {output_dir / "sample_manifest.json"}
    unexpected = [item for item in output_dir.iterdir() if item.is_file() and item not in allowed_existing]
    if unexpected and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing gray outputs: {unexpected}")

    model_config, dea_config = load_config(resolve(args.config))
    if model_config.base_url != args.base_url:
        model_config = ISRConfig(
            base_url=args.base_url,
            model=args.model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout_seconds=model_config.timeout_seconds,
            format_retries=model_config.format_retries,
        )
    if args.workers != 1:
        raise ValueError("workers is fixed at 1 for Step 4.5-A")
    if dea_config.max_rounds != 2 or dea_config.trigger_policy != "v2":
        raise ValueError("DEA trigger configuration is not the frozen v2 setting")

    manifest = read_json(resolve(args.sample_manifest))
    samples = load_input_rows(resolve(args.sample_manifest), resolve(args.samples))
    if len(samples) != 20:
        raise ValueError("exactly 20 gray samples are required")
    model_inputs = [
        {
            "claim": sample["contract"],
            "evidence": sample["evidence"],
        }
        for sample in samples
    ]
    if any(recursive_forbidden_keys(item) for item in model_inputs):
        raise ValueError("forbidden fields detected in model input")

    model_check = dea.check_vllm_models(
        model_config.base_url,
        "openai_v1",
        api_key="",
        timeout=min(model_config.timeout_seconds, 30),
    )
    if not model_check.get("ok"):
        raise RuntimeError(f"vLLM service unavailable: {model_check.get('error')}")

    protected_before = {
        "frozen": hash_paths(FROZEN_FILES),
        "formal": hash_paths(FORMAL_OUTPUTS),
        "independent": hash_paths(INDEPENDENT_INPUTS),
    }
    for section, values in protected_before.items():
        if any(value == "missing" for value in values.values()):
            raise FileNotFoundError(f"missing protected file in {section}")

    contract_path = output_dir / "claim_contracts.jsonl"
    adapter_path = output_dir / "adapter_round1_results.jsonl"
    dea_path = output_dir / "adapter_cached_dea_results.jsonl"
    metrics_path = output_dir / "request_metrics.jsonl"
    if not args.resume:
        for path in (contract_path, adapter_path, dea_path, metrics_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists")

    contracts = []
    for sample in samples:
        contracts.append(
            {
                "input_index": sample["input_index"],
                "cluster_id": sample["cluster_id"],
                "cohort": sample["cohort"],
                "subgroup": sample["subgroup"],
                "claim_contract": sample["contract"],
                "evidence": sample["evidence"],
            }
        )
    write_jsonl(contract_path, contracts)

    os.environ["TASK2_LLM_BASE_URL"] = model_config.base_url
    os.environ["TASK2_LLM_MODEL"] = model_config.model
    os.environ["TASK2_LLM_ENDPOINT_STYLE"] = "openai_v1"
    os.environ["TASK2_LLM_WIRE_API"] = "chat_completions"
    os.environ["TASK2_LLM_TIMEOUT"] = str(model_config.timeout_seconds)
    os.environ["TASK2_LLM_MAX_TOKENS"] = str(model_config.max_tokens)
    os.environ["TASK2_LLM_TEMPERATURE"] = str(model_config.temperature)
    os.environ["TASK2_LLM_MAX_RETRIES"] = "1"
    os.environ["TASK2_LLM_FORMAT_RETRIES"] = str(model_config.format_retries)
    os.environ["TASK2_LLM_WORKERS"] = "1"
    llm_config = fulltext.llm_runtime_config()

    article_index = dea.FulltextArticleIndex(
        resolve(args.article_index),
        members_path=resolve(args.members),
    )
    article_index.load()

    existing_adapter = {
        norm(row.get("cluster_id")): row
        for row in read_jsonl(adapter_path)
    } if args.resume and adapter_path.exists() else {}
    existing_dea = {
        norm(row.get("cluster_id")): row
        for row in read_jsonl(dea_path)
    } if args.resume and dea_path.exists() else {}
    metrics = read_jsonl(metrics_path) if args.resume and metrics_path.exists() else []
    adapter_rows: dict[str, dict[str, Any]] = dict(existing_adapter)
    dea_rows: dict[str, dict[str, Any]] = dict(existing_dea)

    def sink(metric: dict[str, Any]) -> None:
        metrics.append(
            {
                "input_index": next(
                    sample["input_index"] for sample in samples if sample["cluster_id"] == metric["cluster_id"]
                ),
                "cluster_id": metric["cluster_id"],
                "phase": metric["phase"],
                "logical_calls": 1,
                "request_attempts": 1,
                "format_retry_count": int(bool(metric.get("format_retry"))),
                "latency_seconds": float(metric.get("latency_seconds") or 0.0),
                "prompt_tokens": int(metric.get("prompt_tokens") or 0),
                "total_tokens": int(metric.get("total_tokens") or 0),
                "error_type": norm(metric.get("error_type")),
                "error_message": norm(metric.get("error_message")),
                "parse_errors": metric.get("parse_errors", []),
            }
        )
        write_jsonl(metrics_path, metrics)

    for sample in samples:
        cid = sample["cluster_id"]
        if cid not in adapter_rows or adapter_rows[cid].get("status") != "ok":
            result = invoke_json_round(
                config=model_config,
                cluster_id=cid,
                phase="adapter_round1",
                messages=adapter_messages(sample),
                allowed_ids=[item["evidence_id"] for item in sample["evidence"]],
                validator=validate_adapter_object,
                metrics_sink=sink,
            )
            row = adapter_result_row(sample, result)
            if row.get("claim_mode") != sample["contract"]["claim_mode"] and row.get("status") == "ok":
                row["status"] = "failed"
                row["error_type"] = "claim_mode_mismatch"
                row["error_message"] = (
                    f"expected {sample['contract']['claim_mode']}, got {row.get('claim_mode')}"
                )
            adapter_rows[cid] = row
            write_jsonl(adapter_path, list(adapter_rows.values()))
        if cid not in dea_rows or dea_rows[cid].get("status") != "success":
            if adapter_rows[cid].get("status") != "ok":
                raise RuntimeError(f"adapter round1 failed for {cid}")
            row = run_adapter_dea(
                sample,
                adapter_rows[cid],
                llm_config,
                dea_config,
                article_index,
                metrics,
            )
            dea_rows[cid] = row
            write_jsonl(dea_path, list(dea_rows.values()))
            write_jsonl(metrics_path, metrics)

    if len(adapter_rows) != 20 or len(dea_rows) != 20:
        raise RuntimeError("gray outputs are incomplete")
    if any(row.get("status") != "ok" for row in adapter_rows.values()):
        raise RuntimeError("adapter round1 contains failed records")
    if any(row.get("status") != "success" for row in dea_rows.values()):
        raise RuntimeError("adapter+DEA contains failed records")

    output_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (contract_path, adapter_path, dea_path, metrics_path)
    }
    protected_after = {
        "frozen": hash_paths(FROZEN_FILES),
        "formal": hash_paths(FORMAL_OUTPUTS),
        "independent": hash_paths(INDEPENDENT_INPUTS),
    }
    if protected_before != protected_after:
        raise RuntimeError("protected SHA256 changed during Step 4.5-A")

    manifest.update(
        {
            "status": "model_outputs_frozen",
            "model_input_policy": {
                "gold_fields_passed_to_model": False,
                "cached_or_baseline_verdict_passed_to_model": False,
                "diagnostic_fields_passed_to_model": False,
                "evidence_used_to_generate_claim": False,
                "claim_generation_model_call": False,
            },
            "model": {
                "base_url": model_config.base_url,
                "name": model_config.model,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens,
                "format_retries": model_config.format_retries,
                "workers": 1,
            },
            "dea": {
                "initial_evidence_limit": dea_config.initial_evidence_limit,
                "additional_evidence_limit": dea_config.additional_evidence_limit,
                "confidence_threshold": dea_config.confidence_threshold,
                "low_confidence_threshold": dea_config.low_confidence_threshold,
                "trigger_policy": dea_config.trigger_policy,
                "max_rounds": dea_config.max_rounds,
                "retrieval_index_unchanged": True,
            },
            "model_request_attempts": sum(int(row.get("request_attempts") or 0) for row in metrics),
            "logical_model_calls": sum(int(row.get("logical_calls") or 0) for row in metrics),
            "adapter_round1_success": sum(row.get("status") == "ok" for row in adapter_rows.values()),
            "adapter_dea_success": sum(row.get("status") == "success" for row in dea_rows.values()),
            "dea_trigger_count": sum(bool(row.get("dea_triggered")) for row in dea_rows.values()),
            "output_sha256": output_hashes,
            "protected_sha256_before": protected_before,
            "protected_sha256_after": protected_after,
            "protected_files_unchanged": protected_before == protected_after,
            "gold_loaded_for_model_inference": False,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    write_json(resolve(args.sample_manifest), manifest)
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--article-index", type=Path, default=ROOT / "data/processed/event_credibility/fulltext_article_index.parquet")
    parser.add_argument("--members", type=Path, default=ROOT / "outputs/event_clustering/event_cluster_members.parquet")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    manifest = run(build_parser().parse_args(argv))
    print(json.dumps(
        {
            "status": manifest.get("status"),
            "sample_count": manifest.get("sample_count"),
            "adapter_round1_success": manifest.get("adapter_round1_success"),
            "adapter_dea_success": manifest.get("adapter_dea_success"),
            "dea_trigger_count": manifest.get("dea_trigger_count"),
            "model_request_attempts": manifest.get("model_request_attempts"),
            "protected_files_unchanged": manifest.get("protected_files_unchanged"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
