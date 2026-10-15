#!/usr/bin/env python3
"""Benchmark task 2 local LLM review concurrency on a fixed 24-item sample."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.fulltext import (  # noqa: E402
    ALLOWED_LLM_WORKERS,
    build_endpoint_url,
    llm_request_headers,
    llm_review_event_id,
    read_jsonl_records,
    run_llm_review,
)


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "DianJin-R1-32B"
DEFAULT_QUEUE = ROOT / "data/processed/event_credibility/llm_review_queue.jsonl"
DEFAULT_REPORT_JSON = ROOT / "reports/task2_llm_concurrency_benchmark.json"
DEFAULT_REPORT_MD = ROOT / "reports/task2_llm_concurrency_benchmark.md"
CONFLICT_TERMS = ("反转", "矛盾", "澄清", "辟谣", "不属实", "否认", "撤回", "更正")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--base-url", default=os.environ.get("TASK2_LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("TASK2_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--wire-api", default=os.environ.get("TASK2_LLM_WIRE_API", "chat_completions"))
    parser.add_argument("--endpoint-style", default=os.environ.get("TASK2_LLM_ENDPOINT_STYLE", "openai_v1"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("TASK2_LLM_TIMEOUT", "180")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("TASK2_LLM_MAX_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TASK2_LLM_TEMPERATURE", "0")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("TASK2_LLM_MAX_RETRIES", "1")))
    parser.add_argument("--format-retries", type=int, default=int(os.environ.get("TASK2_LLM_FORMAT_RETRIES", "1")))
    parser.add_argument("--checkpoint-every", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    return parser


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sample_rows(rows: list[dict[str, Any]], sample_count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_id = {llm_review_event_id(row): row for row in rows}
    rows = list(by_id.values())

    def is_conflict(row: dict[str, Any]) -> bool:
        text = " ".join(
            str(row.get(field, ""))
            for field in ("fulltext_evidence_status", "risk_reasons", "event_name", "evidence_items")
        )
        return row.get("fulltext_evidence_status") == "contradict" or any(term in text for term in CONFLICT_TERMS)

    def is_insufficient(row: dict[str, Any]) -> bool:
        return row.get("fulltext_evidence_status") == "insufficient"

    conflict = [row for row in rows if is_conflict(row)]
    insufficient = [row for row in rows if is_insufficient(row)]
    support = [row for row in rows if row.get("fulltext_evidence_status") == "support" and not is_conflict(row)]
    groups = [("support_like", support), ("conflict_like", conflict), ("insufficient_like", insufficient)]
    selected: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_group = max(1, sample_count // 3)
    for label, candidates in groups:
        candidates = [row for row in candidates if llm_review_event_id(row) not in selected_ids]
        rng.shuffle(candidates)
        for row in candidates[:per_group]:
            event_id = llm_review_event_id(row)
            selected.append(row)
            selected_ids.add(event_id)
            metadata.append({"cluster_id": event_id, "sample_bucket": label})
    if len(selected) < sample_count:
        remaining = [row for row in rows if llm_review_event_id(row) not in selected_ids]
        rng.shuffle(remaining)
        for row in remaining[: sample_count - len(selected)]:
            event_id = llm_review_event_id(row)
            selected.append(row)
            selected_ids.add(event_id)
            metadata.append({"cluster_id": event_id, "sample_bucket": "fallback"})
    selected = selected[:sample_count]
    selected_ids = {llm_review_event_id(row) for row in selected}
    metadata = [item for item in metadata if item["cluster_id"] in selected_ids]
    selected.sort(key=lambda row: llm_review_event_id(row))
    bucket_by_id = {item["cluster_id"]: item["sample_bucket"] for item in metadata}
    metadata = [{"cluster_id": llm_review_event_id(row), "sample_bucket": bucket_by_id.get(llm_review_event_id(row), "")} for row in selected]
    return selected, metadata


def check_models(base_url: str, api_key: str, endpoint_style: str, timeout: float) -> dict[str, Any]:
    url = build_endpoint_url(base_url, "models", endpoint_style=endpoint_style)
    req = urllib.request.Request(url, headers=llm_request_headers(api_key), method="GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 30)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
        return {"ok": True, "status": resp.status, "elapsed_seconds": round(time.perf_counter() - started, 3), "models": data}
    except Exception as exc:
        return {"ok": False, "status": None, "elapsed_seconds": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"[:800]}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return round(float(ordered[index]), 3)


def is_success(row: dict[str, Any]) -> bool:
    return row.get("status") == "ok" and bool(row.get("llm_valid", True))


def has_text(row: dict[str, Any], pattern: str) -> bool:
    text = " ".join(str(row.get(field, "")) for field in ("error", "error_type", "error_message", "message", "raw_response_preview"))
    return bool(re.search(pattern, text, flags=re.I))


def summarize_results(workers: int, output_path: Path, summary: dict[str, Any], sample_count: int, models_check: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl_records(output_path)
    successes = [row for row in rows if is_success(row)]
    failed = [row for row in rows if not is_success(row)]
    latencies = [float(row.get("latency_seconds") or row.get("elapsed_seconds") or 0.0) for row in rows]
    success_ids = [llm_review_event_id(row) for row in successes]
    duplicate_success_ids = sorted({event_id for event_id in success_ids if success_ids.count(event_id) > 1})
    sorted_rows = sorted(rows, key=lambda row: int(row.get("input_index", 0)))
    max_consecutive_failures = 0
    current = 0
    for row in sorted_rows:
        if is_success(row):
            current = 0
        else:
            current += 1
            max_consecutive_failures = max(max_consecutive_failures, current)
    return {
        "workers": workers,
        "sample_count": sample_count,
        "result_count": len(rows),
        "success_count": len(successes),
        "failed_count": len(failed),
        "success_rate": round(len(successes) / sample_count, 4) if sample_count else 0.0,
        "retry_count": sum(1 for row in rows if int(row.get("attempt_count") or row.get("attempts") or 0) > 1 or int(row.get("format_retry_count") or 0) > 0),
        "total_wall_seconds": round(float(summary.get("elapsed", 0.0)), 3),
        "throughput_items_per_second": round(float(summary.get("throughput_items_per_second", 0.0)), 6),
        "average_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p50_latency": percentile(latencies, 0.50),
        "p95_latency": percentile(latencies, 0.95),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "http_5xx_count": sum(1 for row in failed if has_text(row, r"HTTP.*5\\d\\d|\\b5\\d\\d\\b")),
        "timeout_count": sum(1 for row in failed if has_text(row, r"timeout|timed out|超时")),
        "parse_failure_count": sum(1 for row in failed if has_text(row, r"format_validation_failed|parse|JSON|unparseable")),
        "cuda_oom_count": sum(1 for row in failed if has_text(row, r"CUDA out of memory|out of memory|OOM")),
        "max_consecutive_failures": max_consecutive_failures,
        "duplicate_success_ids": duplicate_success_ids,
        "failed_sample_ids": [llm_review_event_id(row) for row in failed],
        "verdict_distribution": Counter(row.get("verdict", "") for row in successes).most_common(),
        "output_path": str(output_path),
        "checkpoint_path": summary.get("checkpoint_path", ""),
        "models_check": models_check,
    }


def result_map(output_path: Path) -> dict[str, dict[str, Any]]:
    return {llm_review_event_id(row): row for row in read_jsonl_records(output_path)}


def compare_to_serial(sample_ids: list[str], serial_path: Path, current_path: Path) -> dict[str, Any]:
    serial = result_map(serial_path)
    current = result_map(current_path)
    verdict_same = 0
    evidence_same = 0
    confidence_diffs: list[float] = []
    failed_ids: list[str] = []
    changed_ids: list[str] = []
    for event_id in sample_ids:
        base = serial.get(event_id)
        row = current.get(event_id)
        if not base or not row or not is_success(base) or not is_success(row):
            failed_ids.append(event_id)
            continue
        if base.get("verdict") == row.get("verdict"):
            verdict_same += 1
        else:
            changed_ids.append(event_id)
        if sorted(base.get("evidence_ids") or []) == sorted(row.get("evidence_ids") or []):
            evidence_same += 1
        confidence_diffs.append(abs(float(base.get("confidence") or 0.0) - float(row.get("confidence") or 0.0)))
    denominator = len(sample_ids) or 1
    return {
        "verdict_consistency_rate": round(verdict_same / denominator, 4),
        "confidence_mean_abs_diff": round(sum(confidence_diffs) / len(confidence_diffs), 4) if confidence_diffs else None,
        "evidence_ids_consistency_rate": round(evidence_same / denominator, 4),
        "failed_sample_ids": failed_ids,
        "verdict_changed_sample_ids": changed_ids,
    }


def should_stop(summary: dict[str, Any], serial_p95: float) -> tuple[bool, str]:
    if not summary["models_check"].get("ok"):
        return True, "vLLM service is not reachable"
    if summary["cuda_oom_count"] > 0:
        return True, "CUDA OOM detected"
    if summary["failed_count"] / max(1, summary["sample_count"]) >= 0.05:
        return True, "failure rate reached 5%"
    if summary["workers"] > 1 and serial_p95 > 0 and summary["p95_latency"] > serial_p95 * 3:
        return True, "p95 latency exceeded 3x serial p95"
    if summary["max_consecutive_failures"] >= 3:
        return True, "three consecutive requests failed"
    return False, ""


def recommend_workers(run_summaries: list[dict[str, Any]], comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stable: list[dict[str, Any]] = []
    for item in run_summaries:
        comp = comparisons.get(str(item["workers"]), {})
        confidence_diff = comp.get("confidence_mean_abs_diff")
        if confidence_diff is None:
            confidence_diff = 0.0
        if (
            item["success_rate"] >= 0.95
            and item["cuda_oom_count"] == 0
            and not item["duplicate_success_ids"]
            and comp.get("verdict_consistency_rate", 0.0) >= 0.95
            and confidence_diff <= 0.15
        ):
            stable.append(item)
    if not stable:
        return {"workers": 1, "reason": "no concurrency level met stability criteria; use serial fallback"}
    chosen = max(stable, key=lambda item: item["throughput_items_per_second"])
    by_workers = {item["workers"]: item for item in stable}
    if chosen["workers"] == 8 and 4 in by_workers:
        w4 = by_workers[4]
        gain = (chosen["throughput_items_per_second"] - w4["throughput_items_per_second"]) / max(w4["throughput_items_per_second"], 1e-9)
        p95_worse = chosen["p95_latency"] > w4["p95_latency"] * 1.2
        if gain < 0.15 or p95_worse:
            chosen = w4
            return {"workers": 4, "reason": "workers=8 added little throughput or worse p95; workers=4 is the stable higher-throughput choice"}
    return {
        "workers": chosen["workers"],
        "reason": "selected the highest-throughput level that met success, OOM, duplicate-output, verdict-consistency, and confidence-drift criteria",
    }


def write_markdown_report(report_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Task 2 LLM Concurrency Benchmark",
        "",
        f"- Run directory: `{payload['run_dir']}`",
        f"- Sample count: {payload['sample_count']}",
        f"- Recommended workers: {payload['recommendation']['workers']} ({payload['recommendation']['reason']})",
        "",
        "## Results",
        "",
        "| workers | wall(s) | throughput | avg latency | p95 | success | retries | 5xx | timeout | parse fail | CUDA OOM | verdicts |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in payload["runs"]:
        verdicts = ", ".join(f"{k}:{v}" for k, v in item["verdict_distribution"])
        lines.append(
            f"| {item['workers']} | {item['total_wall_seconds']} | {item['throughput_items_per_second']} | "
            f"{item['average_latency']} | {item['p95_latency']} | {item['success_rate']:.2%} | "
            f"{item['retry_count']} | {item['http_5xx_count']} | {item['timeout_count']} | "
            f"{item['parse_failure_count']} | {item['cuda_oom_count']} | {verdicts} |"
        )
    lines.extend(["", "## Consistency vs workers=1", ""])
    for workers, comp in payload["comparisons"].items():
        lines.append(
            f"- workers={workers}: verdict={comp['verdict_consistency_rate']:.2%}, "
            f"confidence MAD={comp['confidence_mean_abs_diff']}, evidence_ids={comp['evidence_ids_consistency_rate']:.2%}, "
            f"failed={comp['failed_sample_ids']}, changed={comp['verdict_changed_sample_ids']}"
        )
    if payload.get("stop_reason"):
        lines.extend(["", f"Stopped early: {payload['stop_reason']}"])
    lines.extend(
        [
            "",
            "Note: differences across concurrency levels can come from model generation variability; input_index and cluster_id are used to distinguish output corruption or row mismatch.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    workers_to_run = [int(part.strip()) for part in args.workers.split(",") if part.strip()]
    unsupported = [value for value in workers_to_run if value not in ALLOWED_LLM_WORKERS]
    if unsupported:
        raise ValueError(f"unsupported workers values: {unsupported}; allowed={sorted(ALLOWED_LLM_WORKERS)}")
    os.environ["TASK2_LLM_BASE_URL"] = args.base_url
    os.environ["TASK2_LLM_MODEL"] = args.model
    os.environ["TASK2_LLM_WIRE_API"] = args.wire_api
    os.environ["TASK2_LLM_ENDPOINT_STYLE"] = args.endpoint_style
    os.environ["TASK2_LLM_TIMEOUT"] = str(args.timeout)
    os.environ["TASK2_LLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["TASK2_LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["TASK2_LLM_MAX_RETRIES"] = str(args.max_retries)
    os.environ["TASK2_LLM_FORMAT_RETRIES"] = str(args.format_retries)

    queue_path = Path(args.queue)
    queue_path = queue_path if queue_path.is_absolute() else ROOT / queue_path
    rows = read_jsonl_records(queue_path)
    sampled, sample_metadata = sample_rows(rows, args.sample_count, args.seed)
    if len(sampled) != args.sample_count:
        raise ValueError(f"expected {args.sample_count} samples, got {len(sampled)}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "reports" / "task2_llm_concurrency_benchmark_runs" / timestamp
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    sample_queue = run_dir / "task2_llm_concurrency_sample_24.jsonl"
    sample_meta_path = run_dir / "task2_llm_concurrency_sample_24_metadata.json"
    write_jsonl(sample_queue, sampled)
    sample_meta_path.write_text(json.dumps(sample_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sample_ids = [llm_review_event_id(row) for row in sampled]

    run_summaries: list[dict[str, Any]] = []
    comparisons: dict[str, dict[str, Any]] = {}
    serial_output: Path | None = None
    serial_p95 = 0.0
    stop_reason = ""
    for workers in workers_to_run:
        models_check = check_models(args.base_url, os.environ.get("TASK2_LLM_API_KEY", ""), args.endpoint_style, args.timeout)
        if not models_check.get("ok"):
            stop_reason = models_check.get("error", "vLLM service is not reachable")
            break
        output_path = run_dir / f"llm_review_workers_{workers}.jsonl"
        print(json.dumps({"workers": workers, "sample_count": len(sampled), "output_path": str(output_path)}, ensure_ascii=False), flush=True)
        summary = run_llm_review(
            sample_queue,
            output_path,
            max_reviews=0,
            sleep_seconds=0,
            workers=workers,
            resume=False,
            checkpoint_every=args.checkpoint_every,
        )
        item = summarize_results(workers, output_path, summary, len(sampled), models_check)
        run_summaries.append(item)
        if workers == 1:
            serial_output = output_path
            serial_p95 = item["p95_latency"]
        if serial_output is not None:
            comparisons[str(workers)] = compare_to_serial(sample_ids, serial_output, output_path)
        stop, reason = should_stop(item, serial_p95)
        if stop:
            stop_reason = f"workers={workers}: {reason}"
            break

    payload = {
        "base_url": args.base_url,
        "model": args.model,
        "wire_api": args.wire_api,
        "endpoint_style": args.endpoint_style,
        "seed": args.seed,
        "sample_count": len(sampled),
        "sample_queue": str(sample_queue),
        "sample_metadata": str(sample_meta_path),
        "run_dir": str(run_dir),
        "runs": run_summaries,
        "comparisons": comparisons,
        "stop_reason": stop_reason,
        "recommendation": recommend_workers(run_summaries, comparisons),
    }
    report_json = Path(args.report_json)
    report_json = report_json if report_json.is_absolute() else ROOT / report_json
    report_md = Path(args.report_md)
    report_md = report_md if report_md.is_absolute() else ROOT / report_md
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md), "recommendation": payload["recommendation"]}, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
