#!/usr/bin/env python3
"""Run the six-sample offline ISR diagnostic experiment.

The model is called before the gold workbook is opened. The model-facing
payload is reconstructed from non-gold sample metadata and evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.isr import (  # noqa: E402
    ISRConfig,
    invoke_json_round,
    norm,
    validate_critique_object,
    validate_final_object,
    validate_final_revision_object,
)


EXPECTED_IDS = [
    "EC-000083",
    "EC-000158",
    "EC-000220",
    "EC-001656",
    "EC-001860",
    "EC-002025",
]
VALID_VERDICTS = {"support", "contradict", "insufficient", "unclear"}
DEFAULT_CANDIDATES = ROOT / "outputs/event_credibility/isr_candidate_audit_v1/isr_candidates.jsonl"
DEFAULT_SAMPLES = ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent/independent_samples.jsonl"
DEFAULT_CONFIG = ROOT / "configs/task2_isr_diagnostic_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/event_credibility/isr_diagnostic_v1"
DEFAULT_MODEL = "DianJin-R1-32B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

FROZEN_FILES = [
    Path("src/event_credibility/dea.py"),
    Path("src/event_credibility/dea_safety_gate.py"),
    Path("configs/task2_dea_safety_gate_v1.yaml"),
    Path("src/event_credibility/dea_safety_gate_v1_1.py"),
    Path("configs/task2_dea_safety_gate_v1_1.yaml"),
    Path("src/event_credibility/dea_safety_gate_v1_2.py"),
    Path("configs/task2_dea_safety_gate_v1_2.yaml"),
    Path("src/event_credibility/dea_safety_gate_adapter.py"),
]
FORMAL_OUTPUTS = [
    Path("outputs/event_credibility/llm_review_results.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.parquet"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.xlsx"),
]
INDEPENDENT_INPUTS = [
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/independent_samples.jsonl"),
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/baseline_results.jsonl"),
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/round1_results.jsonl"),
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/cached_dea_results.jsonl"),
    Path("outputs/event_credibility/dea_safety_gate_v1_2_independent/safety_gate_v1_2_results.jsonl"),
]
FORBIDDEN_MODEL_KEYS = {
    "human_verdict",
    "reviewer_confirmed",
    "human_notes",
    "gold_verdict",
    "baseline_verdict",
    "round1_verdict",
    "cached_verdict",
    "safety_gate_verdict",
    "safety_gate_v1_2_verdict",
    "AI辅助建议_verdict",
    "error_type",
    "error_stage",
    "isr_suitable",
    "isr_corrective_action",
    "diagnosis",
    "route",
    "route_label",
    "primary_error_type",
    "secondary_error_type",
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda item: int(item.get("input_index", 10**9)))
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def recursive_forbidden_keys(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_MODEL_KEYS or key_text.lower().startswith("human_"):
                found.append(f"{path}.{key_text}".strip("."))
            found.extend(recursive_forbidden_keys(child, f"{path}.{key_text}".strip(".")))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(recursive_forbidden_keys(child, f"{path}[{index}]"))
    return found


def validate_candidate_rows(rows: list[dict[str, Any]]) -> None:
    actual = [norm(row.get("cluster_id")) for row in rows]
    if actual != EXPECTED_IDS:
        raise ValueError(f"candidate cluster_id mismatch: {actual}")
    if len(set(actual)) != len(actual) or len(rows) != 6:
        raise ValueError("candidate rows must contain exactly six unique cluster_id values")
    for row in rows:
        evidence = row.get("cited_evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{row['cluster_id']} has no cited evidence")
        ids = []
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError(f"{row['cluster_id']} has malformed evidence")
            evidence_id = norm(item.get("evidence_id") or item.get("id"))
            if not re.fullmatch(r"E\d+", evidence_id):
                raise ValueError(f"{row['cluster_id']} has invalid evidence_id {evidence_id!r}")
            if evidence_id in ids:
                raise ValueError(f"{row['cluster_id']} has duplicate evidence_id {evidence_id}")
            ids.append(evidence_id)
            if not norm(item.get("source")) or not norm(item.get("publish_time")):
                raise ValueError(f"{row['cluster_id']} evidence lacks source/time")
            if not norm(item.get("text") or item.get("evidence_text") or item.get("sentence")):
                raise ValueError(f"{row['cluster_id']} evidence lacks text")
            if norm(item.get("cluster_id")) and norm(item.get("cluster_id")) != norm(row["cluster_id"]):
                raise ValueError(f"cross-cluster evidence in {row['cluster_id']}")


def load_config(path: Path) -> ISRConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = payload.get("model") or {}
    return ISRConfig(
        base_url=str(model.get("base_url") or DEFAULT_BASE_URL),
        model=str(model.get("name") or DEFAULT_MODEL),
        temperature=float(model.get("temperature", 0.0)),
        max_tokens=int(model.get("max_tokens", 512)),
        timeout_seconds=float(model.get("timeout_seconds", 180)),
        format_retries=int(model.get("format_retries", 1)),
    )


def check_model(config: ISRConfig) -> dict[str, Any]:
    request = urllib.request.Request(config.base_url.rstrip("/") + "/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=min(config.timeout_seconds, 30.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = [
            item.get("id")
            for item in (payload.get("data") or [])
            if isinstance(item, dict)
        ]
        if config.model not in ids:
            raise RuntimeError(f"model {config.model!r} not advertised: {ids}")
        return {"ok": True, "model": config.model, "advertised_models": ids}
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        raise RuntimeError(f"model service unavailable: {exc}") from exc


def metadata_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = norm(row.get("cluster_id"))
        if not cid or cid in mapping:
            raise ValueError(f"duplicate or empty independent sample cluster_id: {cid!r}")
        mapping[cid] = row
    return mapping


def build_blind_samples(
    candidates: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for index, candidate in enumerate(candidates):
        cid = norm(candidate["cluster_id"])
        meta = metadata.get(cid)
        if not meta:
            raise ValueError(f"{cid} missing from independent samples")
        evidence = []
        for item in candidate["cited_evidence"]:
            evidence.append(
                {
                    "evidence_id": norm(item.get("evidence_id") or item.get("id")),
                    "source": norm(item.get("source")),
                    "publish_time": norm(item.get("publish_time")),
                    "title": norm(item.get("title") or item.get("article_title")),
                    "evidence_text": norm(item.get("text") or item.get("evidence_text") or item.get("sentence")),
                }
            )
        sample = {
            "cluster_id": cid,
            "input_index": index,
            "claim": norm(meta.get("event_claim") or candidate.get("claim")),
            "subject": norm(meta.get("target_name")),
            "schema_id": norm(meta.get("schema_id")),
            "event_name": norm(meta.get("event_name")),
            "event_time": norm(meta.get("event_time")),
            "evidence": evidence,
        }
        if not sample["claim"] or not sample["subject"]:
            raise ValueError(f"{cid} lacks claim or subject")
        leaked = recursive_forbidden_keys(sample)
        if leaked:
            raise ValueError(f"model input leakage in {cid}: {leaked}")
        output.append(sample)
    return output


def evidence_text(sample: dict[str, Any]) -> str:
    lines = []
    for item in sample["evidence"]:
        lines.append(
            f"{item['evidence_id']} | 来源: {item['source']} | 发布时间: {item['publish_time']}"
            f" | 标题: {item['title'] or '(无)'}\n{item['evidence_text']}"
        )
    return "\n".join(lines)


def base_context(sample: dict[str, Any]) -> str:
    return (
        f"主体：{sample['subject']}\n"
        f"schema_id：{sample['schema_id'] or '(无)'}\n"
        f"事件类型/名称：{sample['event_name'] or '(无)'}\n"
        f"事件时间：{sample['event_time'] or '(无)'}\n"
        f"待验证主张：{sample['claim']}\n\n"
        f"候选证据：\n{evidence_text(sample)}"
    )


def round_a_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是金融事件证据审核员。只输出严格JSON，不输出思维过程。",
        },
        {
            "role": "user",
            "content": (
                "请独立判断下面的主张。只能使用给出的候选证据，不要补充外部事实。"
                "support或contradict必须引用直接相关证据；证据不足时选择insufficient，"
                "只有主张结构上无法验证时选择unclear。\n\n"
                + base_context(sample)
                + '\n\n输出格式：{"verdict":"support|contradict|insufficient|unclear",'
                '"confidence":0到1,"evidence_ids":["E1"],"reason":"简洁理由"}'
            ),
        },
    ]


def round_b_messages(sample: dict[str, Any], round_a: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是金融事件审核的批判复核员。只输出严格JSON，不输出思维过程。",
        },
        {
            "role": "user",
            "content": (
                "审查以下初始裁决是否合理。不要使用外部事实，不要引入任何人工标签。"
                "重点检查投资者提问、行情信号、主体和时间、直接支持、反证及证据冲突。"
                "如果证据不足，建议降级；如果初始裁决合理，也要明确说明无需修订。\n\n"
                + base_context(sample)
                + "\n\n初始裁决："
                + json.dumps(round_a, ensure_ascii=False)
                + '\n\n输出格式：{"problems":[],"evidence_reassessment":[],'
                '"recommended_verdict":"support|contradict|insufficient|unclear",'
                '"recommended_evidence_ids":[],"needs_revision":true}'
            ),
        },
    ]


def round_c_messages(
    sample: dict[str, Any],
    round_a: dict[str, Any],
    critique: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是金融事件最终裁决员。只输出严格JSON，不输出思维过程。",
        },
        {
            "role": "user",
            "content": (
                "根据主张、候选证据、初始裁决和批判意见作最终修订。"
                "不得使用外部事实或人工标签；引用的证据ID必须来自给定证据。"
                "不要为了改变而改变，批判意见不足以支持修改时可保持初始裁决。\n\n"
                + base_context(sample)
                + "\n\n初始裁决："
                + json.dumps(round_a, ensure_ascii=False)
                + "\n\n批判意见："
                + json.dumps(critique, ensure_ascii=False)
                + '\n\n输出格式：{"verdict":"support|contradict|insufficient|unclear",'
                '"confidence":0到1,"evidence_ids":[],"reason":"最终理由",'
                '"revision_summary":"相对首轮修改了什么","isr_rounds":3}'
            ),
        },
    ]


def load_phase(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        cid = norm(row.get("cluster_id"))
        if cid:
            result[cid] = row
    return result


def save_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def phase_row(sample: dict[str, Any], result: dict[str, Any], phase: str) -> dict[str, Any]:
    row = {
        "cluster_id": sample["cluster_id"],
        "input_index": sample["input_index"],
        "phase": phase,
        "claim": sample["claim"],
        "subject": sample["subject"],
        "schema_id": sample["schema_id"],
        "event_name": sample["event_name"],
        "event_time": sample["event_time"],
        "status": result.get("status", "failed"),
        "attempt_count": result.get("attempt_count", 0),
        "format_retry_count": result.get("format_retry_count", 0),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
    }
    parsed = result.get("parsed")
    if isinstance(parsed, dict):
        row.update(parsed)
    if phase == "round_b" and isinstance(parsed, dict):
        row["recommended_verdict"] = parsed.get("recommended_verdict", "")
    return row


def final_row(
    sample: dict[str, Any],
    final_result: dict[str, Any],
    round_a: dict[str, Any],
    round_b: dict[str, Any],
) -> dict[str, Any]:
    row = phase_row(sample, final_result, "final")
    row.update(
        {
            "isr_rounds": 3,
            "round_a_verdict": round_a.get("verdict", ""),
            "round_a_confidence": round_a.get("confidence"),
            "round_a_evidence_ids": round_a.get("evidence_ids", []),
            "round_b_recommended_verdict": round_b.get("recommended_verdict", ""),
            "round_b_recommended_evidence_ids": round_b.get("recommended_evidence_ids", []),
            "round_b_needs_revision": round_b.get("needs_revision"),
            "round_b_problems": round_b.get("problems", []),
            "round_b_evidence_reassessment": round_b.get("evidence_reassessment", []),
        }
    )
    return row


def checkpoint(
    path: Path,
    *,
    stage: str,
    phase_counts: dict[str, Any],
    metrics_count: int,
    status: str,
    final_sha256: str = "",
) -> None:
    write_json(
        path,
        {
            "step": "4.4-J",
            "stage": stage,
            "status": status,
            "workers": 1,
            "phase_counts": phase_counts,
            "model_request_attempts": metrics_count,
            "final_isr_results_sha256": final_sha256,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates_path = resolve(args.candidates)
    samples_path = resolve(args.samples)
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [item for item in output_dir.iterdir() if item.is_file()]
    if existing and not args.resume:
        raise FileExistsError(
            f"{output_dir} is non-empty; use --resume only for this diagnostic run"
        )

    config = load_config(config_path)
    if args.base_url:
        config = ISRConfig(
            base_url=args.base_url,
            model=args.model or config.model,
            temperature=args.temperature if args.temperature is not None else config.temperature,
            max_tokens=args.max_tokens if args.max_tokens is not None else config.max_tokens,
            timeout_seconds=args.timeout if args.timeout is not None else config.timeout_seconds,
            format_retries=config.format_retries,
        )
    check_model(config)

    candidate_rows = read_jsonl(candidates_path)
    validate_candidate_rows(candidate_rows)
    metadata = metadata_map(samples_path)
    samples = build_blind_samples(candidate_rows, metadata)
    if len(samples) != 6:
        raise ValueError("ISR diagnostic requires exactly six samples")

    protected_before = {
        "frozen": hash_paths(FROZEN_FILES),
        "formal": hash_paths(FORMAL_OUTPUTS),
        "independent_inputs": hash_paths(INDEPENDENT_INPUTS),
        "candidates": {str(candidates_path): sha256(candidates_path)},
    }
    phase_paths = {
        "round_a": output_dir / "round_a_results.jsonl",
        "round_b": output_dir / "round_b_critiques.jsonl",
        "final": output_dir / "final_isr_results.jsonl",
    }
    metrics_path = output_dir / "request_metrics.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    phase_rows = {phase: load_phase(path) for phase, path in phase_paths.items()}
    metrics_rows = read_jsonl(metrics_path) if metrics_path.exists() else []
    checkpoint(
        checkpoint_path,
        stage="starting",
        phase_counts={phase: len(rows) for phase, rows in phase_rows.items()},
        metrics_count=len(metrics_rows),
        status="running",
    )

    def metric_sink(metric: dict[str, Any]) -> None:
        metrics_rows.append(metric)
        save_metrics(metrics_path, metrics_rows)

    for sample in samples:
        cid = sample["cluster_id"]
        a_row = phase_rows["round_a"].get(cid)
        if not a_row or a_row.get("status") != "ok":
            result = invoke_json_round(
                config=config,
                cluster_id=cid,
                phase="round_a",
                messages=round_a_messages(sample),
                allowed_ids=[item["evidence_id"] for item in sample["evidence"]],
                validator=validate_final_object,
                metrics_sink=metric_sink,
            )
            a_row = phase_row(sample, result, "round_a")
            phase_rows["round_a"][cid] = a_row
            write_jsonl(phase_paths["round_a"], list(phase_rows["round_a"].values()))
        a_payload = {
            key: a_row.get(key)
            for key in ("verdict", "confidence", "evidence_ids", "reason")
            if key in a_row
        }

        b_row = phase_rows["round_b"].get(cid)
        if not b_row or b_row.get("status") != "ok":
            result = invoke_json_round(
                config=config,
                cluster_id=cid,
                phase="round_b",
                messages=round_b_messages(sample, a_payload),
                allowed_ids=[item["evidence_id"] for item in sample["evidence"]],
                validator=validate_critique_object,
                metrics_sink=metric_sink,
            )
            b_row = phase_row(sample, result, "round_b")
            phase_rows["round_b"][cid] = b_row
            write_jsonl(phase_paths["round_b"], list(phase_rows["round_b"].values()))
        b_payload = {
            key: b_row.get(key)
            for key in (
                "problems",
                "evidence_reassessment",
                "recommended_verdict",
                "recommended_evidence_ids",
                "needs_revision",
            )
            if key in b_row
        }

        f_row = phase_rows["final"].get(cid)
        if not f_row or f_row.get("status") != "ok":
            result = invoke_json_round(
                config=config,
                cluster_id=cid,
                phase="final",
                messages=round_c_messages(sample, a_payload, b_payload),
                allowed_ids=[item["evidence_id"] for item in sample["evidence"]],
                validator=validate_final_revision_object,
                metrics_sink=metric_sink,
            )
            f_row = final_row(sample, result, a_payload, b_payload)
            phase_rows["final"][cid] = f_row
            write_jsonl(phase_paths["final"], list(phase_rows["final"].values()))
        checkpoint(
            checkpoint_path,
            stage=f"completed_{cid}",
            phase_counts={phase: sum(1 for row in rows.values() if row.get("status") == "ok") for phase, rows in phase_rows.items()},
            metrics_count=len(metrics_rows),
            status="running",
        )

    final_rows = list(phase_rows["final"].values())
    successful = [row for row in final_rows if row.get("status") == "ok"]
    if len(final_rows) != 6 or len(successful) != 6:
        checkpoint(
            checkpoint_path,
            stage="failed",
            phase_counts={phase: len(rows) for phase, rows in phase_rows.items()},
            metrics_count=len(metrics_rows),
            status="failed",
        )
        raise RuntimeError("ISR final results are incomplete; resume may retry failed rows")

    final_path = phase_paths["final"]
    final_hash = sha256(final_path)
    protected_after = {
        "frozen": hash_paths(FROZEN_FILES),
        "formal": hash_paths(FORMAL_OUTPUTS),
        "independent_inputs": hash_paths(INDEPENDENT_INPUTS),
        "candidates": {str(candidates_path): sha256(candidates_path)},
    }
    if protected_before != protected_after:
        raise RuntimeError("protected file SHA256 changed during ISR diagnostic")

    manifest = {
        "step": "4.4-J",
        "status": "completed",
        "diagnostic_only": True,
        "gold_loaded_before_model_run": False,
        "candidate_cluster_ids": EXPECTED_IDS,
        "sample_count": 6,
        "model": {
            "base_url": config.base_url,
            "name": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout_seconds": config.timeout_seconds,
            "format_retries": config.format_retries,
            "workers": 1,
        },
        "model_request_attempts": len(metrics_rows),
        "logical_rounds": 18,
        "successful_final_count": 6,
        "final_isr_results": str(final_path),
        "final_isr_results_sha256": final_hash,
        "protected_sha256_before": protected_before,
        "protected_sha256_after": protected_after,
        "protected_files_unchanged": protected_before == protected_after,
        "model_input_policy": {
            "human_fields_used": False,
            "existing_system_verdicts_used": False,
            "gold_loaded": False,
            "hidden_reasoning_saved": False,
            "cross_cluster_evidence_allowed": False,
        },
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    checkpoint(
        checkpoint_path,
        stage="final_frozen",
        phase_counts={phase: sum(1 for row in rows.values() if row.get("status") == "ok") for phase, rows in phase_rows.items()},
        metrics_count=len(metrics_rows),
        status="completed",
        final_sha256=final_hash,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps({
        "status": manifest["status"],
        "sample_count": manifest["sample_count"],
        "successful_final_count": manifest["successful_final_count"],
        "model_request_attempts": manifest["model_request_attempts"],
        "final_isr_results_sha256": manifest["final_isr_results_sha256"],
        "protected_files_unchanged": manifest["protected_files_unchanged"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
