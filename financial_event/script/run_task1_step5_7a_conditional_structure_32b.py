#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Task1 Step 5.7-A conditional structure extraction with local 32B."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_5_candidate_v2 import protected_file_list, rel, sha256_file, write_json, write_jsonl, write_xlsx  # noqa: E402
from src.event_extraction.task1_conditional_structure_32b import (  # noqa: E402
    API_BASE,
    MODEL_NAME,
    acceptance_status,
    build_gold_free_input,
    check_model_service,
    evaluate_structure_predictions,
    prompt_selection_score,
    prompt_template,
    read_jsonl,
    request_structure_one,
    stratified_prompt_eval_split,
    summarize_request_metrics,
    validate_confirmed_annotations,
)

OUTPUT_DIR = ROOT / "outputs/task1_step5_7a_conditional_structure_32b"
REPORT_JSON = ROOT / "reports/task1_step5_7a_conditional_structure_32b_report.json"
REPORT_MD = ROOT / "reports/task1_step5_7a_conditional_structure_32b_report.md"
REPORT_XLSX = ROOT / "reports/task1_step5_7a_conditional_structure_32b_details.xlsx"
CONFIRMED = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"
EXPECTED_CONFIRMED_SHA = "414174fdef7b8aeb89edecdde4a85f43ac08d86093664cb4a6d4541732f1186c"
VLLM_LOG = ROOT / "reports/task1_step5_7a_vllm.log"


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def protected_and_frozen_files() -> list[Path]:
    files = list(protected_file_list())
    for path in [
        CONFIRMED,
        ROOT / "src/routing/task1_candidate_v2.py",
        ROOT / "configs/task1_candidate_v2.yaml",
        ROOT / "outputs/task1_step5_6f_classifier_consolidation/frozen_classifier_manifest.json",
        ROOT / "outputs/task1_step5_6f_classifier_consolidation/SHA256SUMS.txt",
        ROOT / "outputs/event_match_baseline/best_model",
        ROOT / "outputs/task1_limited_training_round1/model_epoch_2",
        ROOT / "outputs/task1_step5_6d_classifier_retraining/experiment_C/best_model",
    ]:
        files.extend(files_under(path))
    return sorted(set(files), key=lambda p: rel(p))


def sha_snapshot(path: Path, files: list[Path]) -> dict[str, str]:
    rows = {rel(item): sha256_file(item) for item in files if item.exists()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(rows.items())), encoding="utf-8")
    return rows


def process_check() -> dict[str, Any]:
    proc = subprocess.run(
        ["bash", "-lc", "ps -eo pid,ppid,stat,etime,cmd | rg -i 'vllm|DianJin-R1-32B|train_task1|run_task1|evaluate_task1' | rg -v 'rg -i' || true\nss -ltnp '( sport = :8000 )' || true"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {"raw": proc.stdout.splitlines(), "returncode": proc.returncode}


def create_backup_and_safety(args: argparse.Namespace, validation: dict[str, Any]) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7a_conditional_structure_32b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    files = protected_and_frozen_files()
    before = sha_snapshot(args.output_dir / "protected_and_frozen_sha256_before.txt", files)
    shutil.copy2(args.output_dir / "protected_and_frozen_sha256_before.txt", backup_dir / "protected_and_frozen_sha256_before.txt")
    safety = {
        "generated_at": now(),
        "backup_dir": str(backup_dir),
        "output_dir": rel(args.output_dir),
        "confirmed_annotation_sha256": sha256_file(CONFIRMED),
        "expected_confirmed_annotation_sha256": EXPECTED_CONFIRMED_SHA,
        "input_validation": validation,
        "before_sha_count": len(before),
        "process_check_before": process_check(),
        "gold_test_labels_read": False,
        "train_or_finetune_executed": False,
        "event_classifier_modified": False,
        "external_llm_calls": 0,
    }
    write_json(args.output_dir / "safety_manifest.json", safety)
    write_json(backup_dir / "safety_manifest.json", safety)
    return safety


def start_vllm_if_needed(args: argparse.Namespace) -> tuple[subprocess.Popen[str] | None, dict[str, Any]]:
    initial = check_model_service(args.api_base, args.model, timeout=5)
    if initial.get("available") and initial.get("model_found"):
        return None, {"service_preexisting": True, "initial_health": initial, "started_by_this_run": False}
    if initial.get("available") and not initial.get("model_found"):
        raise RuntimeError(f"8000 service is running but model {args.model} is not available: {initial}")
    if args.no_start_vllm:
        raise RuntimeError(f"model service unavailable and --no-start-vllm was set: {initial}")
    model_path = Path("/opt/data/private/JinRong/models/DianJin-R1-32B")
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    VLLM_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = VLLM_LOG.open("a", encoding="utf-8")
    cmd = [
        "/root/miniconda3/envs/dianjin-cu128/bin/vllm",
        "serve",
        str(model_path),
        "--served-model-name",
        args.model,
        "--tensor-parallel-size",
        "4",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.90",
        "--disable-custom-all-reduce",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env)
    deadline = time.time() + args.service_wait_seconds
    last_health: dict[str, Any] = {}
    while time.time() < deadline:
        if proc.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"vLLM exited during startup with code {proc.returncode}; see {rel(VLLM_LOG)}")
        last_health = check_model_service(args.api_base, args.model, timeout=5)
        if last_health.get("available") and last_health.get("model_found"):
            return proc, {
                "service_preexisting": False,
                "started_by_this_run": True,
                "pid": proc.pid,
                "command": "CUDA_VISIBLE_DEVICES=0,1,2,3 " + " ".join(cmd),
                "log": rel(VLLM_LOG),
                "health": last_health,
            }
        time.sleep(10)
    raise RuntimeError(f"vLLM startup health check failed after {args.service_wait_seconds}s; last_health={last_health}; see {rel(VLLM_LOG)}")


def stop_vllm(proc: subprocess.Popen[str] | None) -> dict[str, Any]:
    if proc is None:
        return {"stopped": False, "reason": "service_preexisting_or_not_started"}
    if proc.poll() is not None:
        return {"stopped": True, "already_exited": True, "returncode": proc.returncode}
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        return {"stopped": True, "killed_after_timeout": True, "returncode": proc.returncode}
    return {"stopped": True, "killed_after_timeout": False, "returncode": proc.returncode}


def run_requests(items: list[dict[str, Any]], prompt: str, prompt_version: str, split_name: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(request_structure_one, item, prompt, args, prompt_version): item for item in items}
        for future in concurrent.futures.as_completed(future_map):
            raw, pred, metric = future.result()
            raw_rows.append(raw)
            pred_rows.append(pred)
            metric_rows.append(metric)
    raw_rows.sort(key=lambda row: int(row["input_index"]))
    pred_rows.sort(key=lambda row: int(row["input_index"]))
    metric_rows.sort(key=lambda row: int(row["input_index"]))
    return raw_rows, pred_rows, metric_rows


def write_sha_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append((rel(path), sha256_file(path)))
    (output_dir / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in rows), encoding="utf-8")


def flatten(section: str, payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"section": section}
    for key, value in payload.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if not isinstance(sub_value, (dict, list)):
                    out[f"{key}.{sub_key}"] = sub_value
        elif not isinstance(value, list):
            out[key] = value
    return out


def write_reports(report: dict[str, Any], prompt_details: list[dict[str, Any]], frozen_details: list[dict[str, Any]], direction_rows: list[dict[str, Any]]) -> None:
    write_json(REPORT_JSON, report)
    md = [
        "# Task1 Step 5.7-A Conditional Structure 32B Baseline",
        "",
        f"- status: {report['status']}",
        f"- reference: {report['reference_annotation']}",
        f"- split prompt_dev/frozen_eval: {report['split']['prompt_dev_count']}/{report['split']['frozen_eval_count']}",
        f"- frozen prompt: {report['prompt_selection']['selected_prompt_version']} ({report['prompt_selection']['selected_prompt_sha256']})",
        f"- frozen_eval success: {report['engineering']['frozen_eval_request_summary']['success_count']}/{report['engineering']['frozen_eval_request_summary']['record_count']}",
        f"- trigger exact P/R/F1: {report['frozen_eval_metrics']['trigger']['exact_precision']:.6f}/{report['frozen_eval_metrics']['trigger']['exact_recall']:.6f}/{report['frozen_eval_metrics']['trigger']['exact_f1']:.6f}",
        f"- trigger relaxed F1: {report['frozen_eval_metrics']['trigger']['relaxed_overlap_f1']:.6f}",
        f"- event span exact/char-F1: {report['frozen_eval_metrics']['event_span']['exact_match_accuracy']:.6f}/{report['frozen_eval_metrics']['event_span']['character_overlap_f1']:.6f}",
        f"- subject normalized accuracy: {report['frozen_eval_metrics']['subject_company']['normalized_accuracy']:.6f}",
        f"- event time normalized accuracy: {report['frozen_eval_metrics']['event_time']['normalized_date_accuracy']:.6f}",
        f"- object normalized accuracy: {report['frozen_eval_metrics']['object_target']['normalized_accuracy']:.6f}",
        f"- direction accuracy/macro-F1: {report['frozen_eval_metrics']['direction']['accuracy']:.6f}/{report['frozen_eval_metrics']['direction']['fixed_label_macro_f1']:.6f}",
        f"- argument identification F1: {report['frozen_eval_metrics']['arguments']['argument_identification_f1']:.6f}",
        f"- argument+role F1: {report['frozen_eval_metrics']['arguments']['end_to_end_argument_role_f1']:.6f}",
        f"- article exact / at-least-one: {report['frozen_eval_metrics']['complete_structure']['article_exact_match_accuracy']:.6f}/{report['frozen_eval_metrics']['complete_structure']['at_least_one_structure_field_correct_ratio']:.6f}",
        f"- 32B chat attempts / external LLM calls: {report['llm_calls']['dianjin_r1_32b_chat_completion_attempts']}/{report['llm_calls']['external_llm_calls']}",
        f"- vLLM stopped: {report['service']['shutdown'].get('stopped')}",
    ]
    write_text(REPORT_MD, "\n".join(md) + "\n")
    write_xlsx(
        REPORT_XLSX,
        {
            "overall": [flatten("overall", report)],
            "prompt_comparison": report["prompt_selection"]["prompt_comparison_rows"],
            "frozen_eval_details": frozen_details,
            "prompt_dev_details": prompt_details,
            "direction_confusion": direction_rows,
            "field_errors": [
                {"field": field, **counts}
                for field, counts in report["frozen_eval_metrics"]["complete_structure"]["field_error_counts"].items()
            ],
            "request_summary": [
                {"scope": "prompt_dev", **report["engineering"]["prompt_dev_request_summary"]},
                {"scope": "frozen_eval", **report["engineering"]["frozen_eval_request_summary"]},
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=CONFIRMED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--service-wait-seconds", type=int, default=600)
    parser.add_argument("--no-start-vllm", action="store_true")
    args = parser.parse_args()

    if args.input != CONFIRMED:
        raise RuntimeError("Step 5.7-A is pinned to the confirmed 80-row annotation file")
    if not args.input.exists():
        raise FileNotFoundError(str(args.input))
    confirmed_sha = sha256_file(args.input)
    if confirmed_sha != EXPECTED_CONFIRMED_SHA:
        raise RuntimeError(f"confirmed annotation SHA mismatch: {confirmed_sha} != {EXPECTED_CONFIRMED_SHA}")
    rows = read_jsonl(args.input)
    validation = validate_confirmed_annotations(rows)
    if validation["status"] != "valid":
        raise RuntimeError(f"confirmed annotation validation failed: {validation}")
    safety = create_backup_and_safety(args, validation)

    split = stratified_prompt_eval_split(rows, seed=args.seed, prompt_size=20)
    write_json(args.output_dir / "split_manifest.json", split)
    by_id = {str(row["article_id"]): row for row in rows}
    prompt_inputs = [build_gold_free_input(by_id[aid], idx, "prompt_dev") for idx, aid in enumerate(split["prompt_dev_article_ids"])]
    frozen_inputs = [build_gold_free_input(by_id[aid], idx, "frozen_eval") for idx, aid in enumerate(split["frozen_eval_article_ids"])]
    write_jsonl(args.output_dir / "gold_free_prompt_dev_inputs.jsonl", prompt_inputs)
    write_jsonl(args.output_dir / "gold_free_frozen_eval_inputs.jsonl", frozen_inputs)

    service_proc: subprocess.Popen[str] | None = None
    service_info: dict[str, Any] = {}
    started_at = time.perf_counter()
    try:
        service_proc, service_info = start_vllm_if_needed(args)
        prompt_versions = ["strict", "strict_self_check"]
        prompt_result_rows = []
        prompt_metric_rows = []
        prompt_raw_rows = []
        prompt_comparison_rows = []
        prompt_detail_rows_all = []
        for version in prompt_versions:
            prompt = prompt_template(version)
            raw_rows, pred_rows, metric_rows = run_requests(prompt_inputs, prompt, version, "prompt_dev", args)
            metrics, details, _ = evaluate_structure_predictions([by_id[aid] for aid in split["prompt_dev_article_ids"]], pred_rows, f"prompt_dev_{version}")
            summary = summarize_request_metrics(metric_rows)
            score = prompt_selection_score(metrics)
            prompt_comparison_rows.append(
                {
                    "prompt_version": version,
                    "selection_score": score,
                    "success_count": summary["success_count"],
                    "trigger_relaxed_f1": metrics["trigger"]["relaxed_overlap_f1"],
                    "event_span_char_f1": metrics["event_span"]["character_overlap_f1"],
                    "subject_norm_acc": metrics["subject_company"]["normalized_accuracy"],
                    "event_time_norm_acc": metrics["event_time"]["normalized_date_accuracy"],
                    "direction_acc": metrics["direction"]["accuracy"],
                    "argument_role_f1": metrics["arguments"]["end_to_end_argument_role_f1"],
                    "at_least_one_ratio": metrics["complete_structure"]["at_least_one_structure_field_correct_ratio"],
                    "avg_latency_seconds": summary["avg_latency_seconds"],
                }
            )
            for row in pred_rows:
                prompt_result_rows.append(row)
            prompt_metric_rows.extend(metric_rows)
            prompt_raw_rows.extend(raw_rows)
            prompt_detail_rows_all.extend(details)
        prompt_comparison_rows.sort(key=lambda row: (-float(row["selection_score"]), -int(row["success_count"]), 0 if row["prompt_version"] == "strict" else 1))
        selected_prompt_version = str(prompt_comparison_rows[0]["prompt_version"])
        frozen_prompt = prompt_template(selected_prompt_version)
        frozen_prompt_path = args.output_dir / "frozen_prompt.txt"
        write_text(frozen_prompt_path, frozen_prompt)
        prompt_manifest = {
            "generated_at": now(),
            "selected_prompt_version": selected_prompt_version,
            "selected_prompt_sha256": sha256_file(frozen_prompt_path),
            "json_schema_version": "task1_conditional_structure_32b_schema_v1",
            "parser_version": "task1_conditional_structure_32b_parser_v1_offset_exact_with_unique_text_repair",
            "selection_rule": "maximize prompt_selection_score on prompt_dev only; tie prefers strict prompt",
            "prompt_dev_labels_used_for_prompt_selection": True,
            "frozen_eval_labels_used_before_prediction": False,
        }
        write_json(args.output_dir / "frozen_prompt_manifest.json", prompt_manifest)
        write_jsonl(args.output_dir / "prompt_dev_results.jsonl", prompt_result_rows)
        write_jsonl(args.output_dir / "prompt_dev_raw_responses.jsonl", prompt_raw_rows)

        raw_rows, frozen_pred_rows, frozen_metric_rows = run_requests(frozen_inputs, frozen_prompt, selected_prompt_version, "frozen_eval", args)
        write_jsonl(args.output_dir / "frozen_eval_raw_responses.jsonl", raw_rows)
        write_jsonl(args.output_dir / "frozen_eval_results.jsonl", frozen_pred_rows)
        all_metric_rows = prompt_metric_rows + frozen_metric_rows
        write_jsonl(args.output_dir / "request_metrics.jsonl", all_metric_rows)
        frozen_sha_before_eval = sha256_file(args.output_dir / "frozen_eval_results.jsonl")
        frozen_metrics, frozen_details, direction_rows = evaluate_structure_predictions([by_id[aid] for aid in split["frozen_eval_article_ids"]], frozen_pred_rows, "frozen_eval")
        frozen_request_summary = summarize_request_metrics(frozen_metric_rows)
        prompt_request_summary = summarize_request_metrics(prompt_metric_rows)
        status, gates = acceptance_status(frozen_metrics, frozen_request_summary)
        frozen_sha_after_eval = sha256_file(args.output_dir / "frozen_eval_results.jsonl")
        after = sha_snapshot(args.output_dir / "protected_and_frozen_sha256_after.txt", protected_and_frozen_files())
        before = {}
        for line in (args.output_dir / "protected_and_frozen_sha256_before.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                before[name] = digest
        sha_unchanged = before == after
        report = {
            "generated_at": now(),
            "status": status,
            "reference_annotation": "AI-assisted, human-confirmed 80-row diagnostic structure annotation; not an independent manual gold corpus.",
            "classification_context": {
                "step5_6f_status": "classification_baseline_frozen",
                "frozen_classifier_strategy": "I_score_fusion_0.20_0.40_0.40_min_0.40",
                "event_classifier_modified": False,
                "event_classifier_used_for_this_step": False,
                "conditional_on_correct_event_category": True,
            },
            "input_validation": validation,
            "split": split,
            "service": {**service_info, "shutdown": {}},
            "prompt_selection": {
                **prompt_manifest,
                "prompt_comparison_rows": prompt_comparison_rows,
            },
            "frozen_eval_metrics": frozen_metrics,
            "engineering": {
                "prompt_dev_request_summary": prompt_request_summary,
                "frozen_eval_request_summary": frozen_request_summary,
                "frozen_eval_prediction_sha256_before_gold_eval": frozen_sha_before_eval,
                "frozen_eval_prediction_sha256_after_gold_eval": frozen_sha_after_eval,
                "frozen_outputs_unchanged_after_eval": frozen_sha_before_eval == frozen_sha_after_eval,
                "runtime_seconds": round(time.perf_counter() - started_at, 3),
            },
            "acceptance_gates": gates,
            "recommendations": {
                "recommend_rule_plus_32b_hybrid": status == "hybrid_structure_baseline_recommended",
                "need_structure_annotation_expansion": status in {"structure_training_data_expansion_required", "hybrid_structure_baseline_recommended"},
                "allowed_next_step": "design_500_row_silver_structure_plan_only",
                "do_not_promote_formal_task1": True,
            },
            "protection": {
                "backup_dir": safety["backup_dir"],
                "formal_and_frozen_sha_unchanged": sha_unchanged,
                "confirmed_annotation_sha256": sha256_file(CONFIRMED),
                "confirmed_annotation_sha_unchanged": sha256_file(CONFIRMED) == EXPECTED_CONFIRMED_SHA,
                "gold_test_labels_read": False,
                "formal_outputs_modified": False,
            },
            "llm_calls": {
                "dianjin_r1_32b_logical_records": len(prompt_inputs) * len(prompt_versions) + len(frozen_inputs),
                "dianjin_r1_32b_chat_completion_attempts": prompt_request_summary["request_count"] + frozen_request_summary["request_count"],
                "external_llm_calls": 0,
            },
            "training_or_finetune_executed": False,
        }
        write_json(args.output_dir / "run_manifest.json", report)
        write_reports(report, prompt_detail_rows_all, frozen_details, direction_rows)
        write_sha_manifest(args.output_dir)
        return 0
    finally:
        shutdown = stop_vllm(service_proc)
        if (args.output_dir / "run_manifest.json").exists():
            manifest = read_json(args.output_dir / "run_manifest.json")
            manifest.setdefault("service", service_info)
            manifest["service"]["shutdown"] = shutdown
            write_json(args.output_dir / "run_manifest.json", manifest)
            write_json(REPORT_JSON, manifest)
            write_text(
                REPORT_MD,
                "\n".join(
                    [
                        "# Task1 Step 5.7-A Conditional Structure 32B Baseline",
                        "",
                        f"- status: {manifest.get('status')}",
                        f"- frozen_eval success: {manifest.get('engineering', {}).get('frozen_eval_request_summary', {}).get('success_count')}/{manifest.get('engineering', {}).get('frozen_eval_request_summary', {}).get('record_count')}",
                        f"- frozen prompt: {manifest.get('prompt_selection', {}).get('selected_prompt_version')} ({manifest.get('prompt_selection', {}).get('selected_prompt_sha256')})",
                        f"- vLLM stopped: {shutdown.get('stopped')}",
                        f"- 32B/external LLM calls: {manifest.get('llm_calls', {}).get('dianjin_r1_32b_chat_completion_attempts')}/{manifest.get('llm_calls', {}).get('external_llm_calls')}",
                    ]
                )
                + "\n",
            )
            write_sha_manifest(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
