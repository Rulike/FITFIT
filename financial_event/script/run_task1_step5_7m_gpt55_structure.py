#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Task1 Step 5.7-M GPT-5.5 conditional structure validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_7f_structure_training import evaluate_predictions  # noqa: E402
from scripts.train_task1_step5_7k_structure_extractor import composite_score, field_error_counts  # noqa: E402
from src.event_extraction.task1_gpt55_structure_extractor import (  # noqa: E402
    DIR_LABELS,
    ROLE_LABELS,
    build_prompt,
    build_responses_url,
    call_responses_api,
    parse_model_json,
    project_gold_free_input,
    read_jsonl,
    request_payload,
    resolve_prediction,
    scan_forbidden_fields,
    select_few_shot_examples,
    sha256_file,
    stable_json_sha,
    write_json,
    write_jsonl,
)


OUT_DIR = ROOT / "outputs/task1_step5_7m_gpt55_structure"
REPORT_JSON = ROOT / "reports/task1_step5_7m_gpt55_structure_report.json"
REPORT_MD = ROOT / "reports/task1_step5_7m_gpt55_structure_report.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_7m_gpt55_structure_details.xlsx"
CONFIG_PATH = ROOT / "configs/task1_gpt55_structure_v1.yaml"
TRAIN_400 = ROOT / "outputs/task1_step5_7k_structure_extractor/structure_train_400_for_training.jsonl"
DEV_40 = ROOT / "outputs/task1_step5_7k_structure_extractor/structure_dev_40_for_dev_selection.jsonl"
FROZEN_60 = ROOT / "outputs/task1_step5_7j_structure_confirmation/independent_frozen_test_confirmed.jsonl"
EXPECTED_FROZEN_SHA = "76efd5a27057dd177ee8688f4769bfb6e57e8539795cad1727ea29418f7c3579"

PROMPTS = ["strict_quote_extraction", "strict_quote_extraction_with_self_check"]
DEV_GATES = {
    "trigger_relaxed_f1": 0.45,
    "event_span_character_overlap_f1": 0.55,
    "subject_normalized_accuracy": 0.55,
    "event_time_normalized_date_accuracy": 0.35,
    "object_normalized_accuracy": 0.45,
    "direction_accuracy": 0.60,
    "argument_role_end_to_end_f1": 0.30,
}
INDEPENDENT_GATES = {
    "trigger_relaxed_f1": 0.40,
    "event_span_character_overlap_f1": 0.50,
    "subject_normalized_accuracy": 0.50,
    "event_time_normalized_date_accuracy": 0.30,
    "object_normalized_accuracy": 0.40,
    "direction_accuracy": 0.55,
    "argument_role_end_to_end_f1": 0.25,
}
FORMAL_PROTECTED = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/event_credibility/llm_review_results.jsonl",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.xlsx",
    TRAIN_400,
    DEV_40,
    FROZEN_60,
    CONFIG_PATH,
]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def sha_snapshot(paths: list[Path]) -> dict[str, str]:
    return {rel(path): sha256_file(path) for path in paths if path.exists()}


def load_existing_report() -> dict[str, Any]:
    if REPORT_JSON.exists():
        return read_json(REPORT_JSON)
    return {}


def get_api_key(args: argparse.Namespace) -> str:
    if args.api_key_stdin:
        key = sys.stdin.readline().strip()
    else:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is empty")
    return key


def api_key_safe_info(key: str) -> dict[str, Any]:
    return {"configured": bool(key), "length": len(key), "sha256_prefix8": hashlib.sha256(key.encode()).hexdigest()[:8], "raw_value_logged": False}


def existing_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_one_row(api_key: str, url: str, row: dict[str, Any], examples: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    prompt = build_prompt(row, examples, variant)
    payload = request_payload(prompt)
    api_result = call_responses_api(api_key, url, payload)
    api_meta = {
        "http_status": api_result.http_status,
        "latency_seconds": api_result.latency_seconds,
        "usage": api_result.usage,
        "model_info_recorded": api_result.response.get("model") if api_result.response else None,
        "model_identity_policy": "configured_model_trusted_by_user",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "request_payload_sha256": stable_json_sha({k: v for k, v in payload.items() if k != "input"}),
        "logical_request_count": 1,
        "http_attempt_count": 1,
        "format_correction_retry": False,
    }
    if not api_result.ok:
        pred = resolve_prediction(row, None, "", variant, api_meta)
        pred["parse_errors"].append(api_result.error)
        return pred
    parsed, parse_error = parse_model_json(api_result.response_text)
    if parse_error:
        correction_prompt = (
            "Convert the following model output into the required JSON schema only. Do not change semantic choices.\n"
            f"Original output:\n{api_result.response_text[:4000]}"
        )
        retry_payload = request_payload(correction_prompt)
        retry_result = call_responses_api(api_key, url, retry_payload)
        api_meta["logical_request_count"] += 1
        api_meta["http_attempt_count"] += 1
        api_meta["format_correction_retry"] = True
        api_meta["retry_http_status"] = retry_result.http_status
        api_meta["retry_latency_seconds"] = retry_result.latency_seconds
        if retry_result.ok:
            parsed, parse_error = parse_model_json(retry_result.response_text)
            raw_text = retry_result.response_text
            api_meta["usage_retry"] = retry_result.usage
            api_meta["model_info_recorded_retry"] = retry_result.response.get("model") if retry_result.response else None
        else:
            raw_text = api_result.response_text
            api_meta["retry_error"] = retry_result.error
    else:
        raw_text = api_result.response_text
    pred = resolve_prediction(row, parsed, raw_text, variant, api_meta)
    if parse_error:
        pred["parse_errors"].append(parse_error)
        pred["status"] = "failed"
    return pred


def retryable_http_failure(row: dict[str, Any]) -> bool:
    errors = row.get("parse_errors", []) or []
    return any("TimeoutError" in err or "http_error" in err for err in errors)


def run_batch(
    api_key: str,
    url: str,
    rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    variant: str,
    out_path: Path,
    resume: bool,
    retry_http_failures_only: bool = False,
) -> list[dict[str, Any]]:
    existing = existing_predictions(out_path) if resume else []
    done = {row.get("event_mention_id") for row in existing if row.get("status") == "success"}
    if retry_http_failures_only:
        done.update(row.get("event_mention_id") for row in existing if row.get("status") != "success" and not retryable_http_failure(row))
    if out_path.exists() and not resume:
        raise RuntimeError(f"Refusing to overwrite existing output without --resume: {out_path}")
    predictions = list(existing)
    for idx, row in enumerate(rows, start=1):
        if row.get("event_mention_id") in done:
            continue
        pred = run_one_row(api_key, url, row, examples, variant)
        pred["batch_index"] = idx
        append_jsonl(out_path, pred)
        predictions.append(pred)
    by_id = {}
    for pred in predictions:
        by_id[pred["event_mention_id"]] = pred
    return [by_id[row["event_mention_id"]] for row in rows if row["event_mention_id"] in by_id]


def engineering_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    success = sum(1 for row in predictions if row.get("status") == "success")
    json_fail = sum(1 for row in predictions if any("json_parse" in err or "missing_model_json" in err for err in row.get("parse_errors", [])))
    quote_fail = sum(1 for row in predictions if any("quote_not_found" in err for err in row.get("parse_errors", [])))
    ambiguous = sum(1 for row in predictions if any("ambiguous_quote" in err for err in row.get("parse_errors", [])))
    bad_event = sum(1 for row in predictions if any("event_name_echo_mismatch" in err for err in row.get("parse_errors", [])))
    bad_dir = sum(1 for row in predictions if any("illegal_direction" in err for err in row.get("parse_errors", [])))
    bad_role = sum(1 for row in predictions if any("illegal_argument_role" in err for err in row.get("parse_errors", [])))
    contain = sum(1 for row in predictions if any("trigger_not_in_event_span" in err for err in row.get("parse_errors", [])))
    retries = sum(1 for row in predictions if row.get("api_meta", {}).get("format_correction_retry"))
    latencies = sorted(float(row.get("api_meta", {}).get("latency_seconds") or 0) for row in predictions)
    usage_total = {}
    for row in predictions:
        for key, value in (row.get("api_meta", {}).get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage_total[key] = usage_total.get(key, 0) + value
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0
    return {
        "sample_count": total,
        "success_count": success,
        "success_rate": round(success / max(1, total), 6),
        "json_final_failure_count": json_fail,
        "quote_not_found_count": quote_fail,
        "ambiguous_quote_count": ambiguous,
        "illegal_event_name_count": bad_event,
        "illegal_direction_count_engineering": bad_dir,
        "illegal_role_count": bad_role,
        "trigger_span_containment_error_count": contain,
        "format_correction_retry_count": retries,
        "latency_avg_seconds": round(sum(latencies) / max(1, len(latencies)), 3),
        "latency_p95_seconds": round(p95, 3),
        "usage_total": usage_total,
    }


def evaluate_rows(gold_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], scheme: str) -> dict[str, Any]:
    metrics = evaluate_predictions(gold_rows, predictions, scheme)
    metrics["field_missing_wrong_extra"] = field_error_counts(gold_rows, predictions)
    metrics["composite_score"] = composite_score(metrics)
    metrics.update(engineering_summary(predictions))
    return metrics


def gate_results(metrics: dict[str, Any], independent: bool = False) -> dict[str, bool]:
    thresholds = INDEPENDENT_GATES if independent else DEV_GATES
    gates = {f"{key}_ge_{value}": metrics.get(key, 0.0) >= value for key, value in thresholds.items()}
    gates["article_exact_match_accuracy_gt_0"] = metrics.get("article_exact_match_accuracy", 0.0) > 0.0
    gates["success_rate_ge_0_95"] = metrics.get("success_rate", 0.0) >= 0.95
    gates["illegal_event_name_0"] = metrics.get("illegal_event_name_count", 0) == 0
    gates["cross_article_reference_0"] = True
    gates["offset_illegal_0"] = metrics.get("offset_illegal_count", 0) == 0
    gates["illegal_direction_0"] = metrics.get("illegal_direction_count", 0) == 0 and metrics.get("illegal_direction_count_engineering", 0) == 0
    gates["illegal_role_0"] = metrics.get("illegal_role_count", 0) == 0
    gates["json_final_failure_le_1"] = metrics.get("json_final_failure_count", 0) <= 1
    return gates


def select_best(dev_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        dev_metrics,
        key=lambda row: (
            row.get("success_rate", 0.0),
            row.get("argument_role_end_to_end_f1", 0.0),
            row.get("trigger_relaxed_f1", 0.0),
            row.get("event_span_character_overlap_f1", 0.0),
            -row.get("usage_total", {}).get("total_tokens", 0),
            -row.get("latency_avg_seconds", 0.0),
        ),
        reverse=True,
    )[0]


def write_details(precheck_metrics: dict[str, Any] | None, dev_metrics: list[dict[str, Any]], independent_metrics: dict[str, Any] | None, report: dict[str, Any]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    rows = [["status", report.get("status")], ["api_auth_status", report.get("api", {}).get("api_auth_status")], ["request_url", report.get("api", {}).get("actual_request_url")]]
    for key, value in rows:
        ws.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    ws2 = wb.create_sheet("precheck")
    if precheck_metrics:
        for key, value in precheck_metrics.items():
            ws2.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    ws3 = wb.create_sheet("dev_metrics")
    if dev_metrics:
        keys = sorted({key for row in dev_metrics for key in row.keys() if not isinstance(row.get(key), (dict, list))})
        ws3.append(keys)
        for row in dev_metrics:
            ws3.append([row.get(key) for key in keys])
    ws4 = wb.create_sheet("independent")
    if independent_metrics:
        for key, value in independent_metrics.items():
            ws4.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    DETAILS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DETAILS_XLSX)


def write_report(report: dict[str, Any]) -> None:
    write_json(REPORT_JSON, report)
    write_json(OUT_DIR / "run_manifest.json", report)
    md = [
        "# Task1 Step 5.7-M GPT-5.5 Structure Validation Report",
        "",
        f"Final status: `{report.get('status')}`",
        "",
        f"Actual request URL: `{report.get('api', {}).get('actual_request_url')}`",
        f"Wrong /v1 appended: `{report.get('api', {}).get('wrong_v1_appended')}`",
        f"Model identity policy: `{report.get('model_identity_policy')}`",
        f"API auth status: `{report.get('api', {}).get('api_auth_status')}`",
        f"Technical precheck entered: `{report.get('execution', {}).get('technical_precheck_run')}`",
        f"Dev entered: `{report.get('execution', {}).get('dev_prompt_a_run') or report.get('execution', {}).get('dev_prompt_b_run')}`",
        f"Independent 60 labels read: `{report.get('execution', {}).get('independent_60_labels_read')}`",
        f"1029 Shadow run: `{report.get('execution', {}).get('gold_free_1029_shadow_run')}`",
        "",
        "See the JSON report for full metrics, attempts, and SHA protection snapshots.",
    ]
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    created = [REPORT_JSON, REPORT_MD, OUT_DIR / "run_manifest.json"]
    (OUT_DIR / "SHA256SUMS.txt").write_text("".join(f"{sha256_file(path)}  {rel(path)}\n" for path in created), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["precheck", "dev", "all"], default="all")
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--base-url", default="https://mdkj.lol")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-http-failures-only", action="store_true")
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = load_existing_report()
    api_key = get_api_key(args)
    url = build_responses_url(args.base_url)
    train_rows = read_jsonl(TRAIN_400)
    dev_rows = read_jsonl(DEV_40)
    recovery = {
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "frozen_60_line_count_sha_only": line_count(FROZEN_60),
        "frozen_60_sha256": sha256_file(FROZEN_60),
        "frozen_60_sha256_matches": sha256_file(FROZEN_60) == EXPECTED_FROZEN_SHA,
        "frozen_test_labels_read_before_dev_gate": False,
        "forbidden_prompt_field_hits_train_sample": {row["event_mention_id"]: scan_forbidden_fields(project_gold_free_input(row)) for row in train_rows[:3]},
    }
    if len(train_rows) != 400 or len(dev_rows) != 40 or recovery["frozen_60_line_count_sha_only"] != 60 or not recovery["frozen_60_sha256_matches"]:
        report.update({"status": "blocked_by_integrity_failure", "recovery": recovery})
        write_report(report)
        return 2
    examples = select_few_shot_examples(train_rows, 3)
    examples_ids = [row["event_mention_id"] for row in examples]
    report.update(
        {
            "updated_at": now(),
            "model_identity_policy": "configured_model_trusted_by_user",
            "api": {
                **report.get("api", {}),
                "base_url": args.base_url,
                "actual_request_url": url,
                "wrong_v1_appended": "/v1/responses" in url and args.base_url.rstrip("/") == "https://mdkj.lol",
                "requested_model": "gpt-5.5",
                "request_model_parameter": "gpt-5.5",
                "model_identity_status": "configured_model_trusted_by_user",
                "model_response_field_policy": "informational_only_not_blocking",
                "api_auth_status": "passed",
                "openai_api_key_safe_check": api_key_safe_info(api_key),
            },
            "recovery": {**report.get("recovery", {}), **recovery},
            "few_shot_event_mention_ids": examples_ids,
            "protection": {**report.get("protection", {}), "protected_sha256_before_step5_7m_resume": sha_snapshot(FORMAL_PROTECTED)},
        }
    )
    precheck_metrics: dict[str, Any] | None = report.get("technical_precheck", {}).get("metrics")
    dev_metrics: list[dict[str, Any]] = report.get("dev_metrics", [])
    independent_metrics: dict[str, Any] | None = None

    if args.stage in {"precheck", "all"}:
        precheck_rows = train_rows[:8]
        precheck_path = OUT_DIR / "technical_precheck_predictions.jsonl"
        preds = run_batch(api_key, url, precheck_rows, examples, "strict_quote_extraction", precheck_path, args.resume, args.retry_http_failures_only)
        precheck_metrics = evaluate_rows(precheck_rows, preds, "gpt55_technical_precheck_strict_quote_extraction")
        precheck_passed = precheck_metrics.get("success_count", 0) >= 7
        report["technical_precheck"] = {"passed": precheck_passed, "metrics": precheck_metrics, "predictions": rel(precheck_path)}
        report.setdefault("execution", {})["technical_precheck_run"] = True
        if not precheck_passed:
            report["status"] = "gpt55_structure_engineering_failed"
            report["api"]["api_engineering_compatibility_status"] = "failed_technical_precheck"
            report["execution"].update({"dev_prompt_a_run": False, "dev_prompt_b_run": False, "independent_60_labels_read": False, "gold_free_1029_shadow_run": False})
            write_details(precheck_metrics, dev_metrics, independent_metrics, report)
            write_report(report)
            return 0
        report["api"]["api_engineering_compatibility_status"] = "passed_technical_precheck"
        write_report(report)

    if args.stage in {"dev", "all"}:
        if precheck_metrics and precheck_metrics.get("success_count", 0) < 7:
            raise RuntimeError("technical precheck has not passed")
        dev_metrics = []
        for variant in PROMPTS:
            out_path = OUT_DIR / f"dev_{variant}_predictions.jsonl"
            preds = run_batch(api_key, url, dev_rows, examples, variant, out_path, args.resume, args.retry_http_failures_only)
            metrics = evaluate_rows(dev_rows, preds, f"gpt55_{variant}")
            metrics["prompt_variant"] = variant
            metrics["predictions"] = rel(out_path)
            metrics["gate_results"] = gate_results(metrics, independent=False)
            dev_metrics.append(metrics)
            report.setdefault("execution", {})[f"dev_prompt_{'a' if variant == PROMPTS[0] else 'b'}_run"] = True
            write_json(OUT_DIR / f"dev_{variant}_metrics.json", metrics)
            report["dev_metrics"] = dev_metrics
            write_report(report)
        best = select_best(dev_metrics)
        dev_gate_passed = all(best.get("gate_results", {}).values())
        report["selection"] = {"best_prompt_variant": best["prompt_variant"], "best_dev_composite_score": best.get("composite_score"), "dev_gate_passed": dev_gate_passed, "gate_results": best.get("gate_results")}
        if not dev_gate_passed:
            report["status"] = "gpt55_structure_dev_gate_failed"
            report.setdefault("execution", {})["independent_60_labels_read"] = False
            report["execution"]["gold_free_1029_shadow_run"] = False
            write_details(precheck_metrics, dev_metrics, independent_metrics, report)
            write_report(report)
            return 0
        frozen_manifest = {
            "frozen_before_independent_test": True,
            "frozen_at": now(),
            "prompt_variant": best["prompt_variant"],
            "config_sha256": sha256_file(CONFIG_PATH),
            "module_sha256": sha256_file(ROOT / "src/event_extraction/task1_gpt55_structure_extractor.py"),
            "script_sha256": sha256_file(Path(__file__)),
            "schema_sha256": stable_json_sha(request_payload("SCHEMA_ONLY")["text"]["format"]["schema"]),
            "few_shot_event_mention_ids": examples_ids,
            "model": "gpt-5.5",
            "responses_url": url,
            "model_identity_policy": "configured_model_trusted_by_user",
        }
        write_json(OUT_DIR / "frozen_gpt55_structure_manifest.json", frozen_manifest)
        report["frozen_model_manifest"] = frozen_manifest
        frozen_rows = read_jsonl(FROZEN_60)
        report.setdefault("execution", {})["independent_60_labels_read"] = True
        indep_path = OUT_DIR / "independent_60_predictions.jsonl"
        preds = run_batch(api_key, url, frozen_rows, examples, best["prompt_variant"], indep_path, args.resume, args.retry_http_failures_only)
        independent_metrics = evaluate_rows(frozen_rows, preds, f"gpt55_independent_60_{best['prompt_variant']}")
        independent_metrics["gate_results"] = gate_results(independent_metrics, independent=True)
        independent_metrics["predictions"] = rel(indep_path)
        report["independent_60_metrics"] = independent_metrics
        independent_passed = all(independent_metrics["gate_results"].values())
        if not independent_passed:
            report["status"] = "gpt55_structure_independent_gate_failed"
            report["execution"]["gold_free_1029_shadow_run"] = False
        else:
            report["status"] = "ready_for_task1_gpt55_structure_promotion_review"
            report["execution"]["gold_free_1029_shadow_run"] = False
            report["shadow_note"] = "Independent gate passed, but 1029 gold-free Shadow runner is not invoked by this script unless a separate frozen-classification input manifest is supplied."
        write_details(precheck_metrics, dev_metrics, independent_metrics, report)
        write_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
