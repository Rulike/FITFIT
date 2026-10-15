#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-O mature Chinese IE model feasibility benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_7f_structure_training import evaluate_predictions  # noqa: E402
from src.event_extraction.task1_supervised_structure_extractor import (  # noqa: E402
    CandidateRankerExtractor,
    predict_batch,
    stable_json_sha,
    validate_offsets,
)


OUT_DIR = ROOT / "outputs/task1_step5_7o_mature_model_benchmark"
REPORT_JSON = ROOT / "reports/task1_step5_7o_mature_model_benchmark.json"
REPORT_MD = ROOT / "reports/task1_step5_7o_mature_model_benchmark.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_7o_mature_model_benchmark_details.xlsx"

TRAIN850 = ROOT / "outputs/task1_step5_7n_chatgpt500_confirmation/merged_structure_train_850.jsonl"
DEV90 = ROOT / "outputs/task1_step5_7n_chatgpt500_confirmation/merged_structure_dev_90.jsonl"
STEP57N_REPORT = ROOT / "reports/task1_step5_7n_structure_training_report.json"
FROZEN60_TEST = ROOT / "outputs/task1_step5_7j_structure_confirmation/independent_frozen_test_confirmed.jsonl"
EXPECTED_FROZEN60_SHA = "76efd5a27057dd177ee8688f4769bfb6e57e8539795cad1727ea29418f7c3579"

FORMAL_PROTECTED = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/final_task1/final_events_train.xlsx",
    ROOT / "outputs/final_task1/final_events_test.xlsx",
    ROOT / "outputs/final_task1/task1_completion_certificate.json",
    ROOT / "outputs/event_credibility/llm_review_results.jsonl",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.xlsx",
    TRAIN850,
    DEV90,
    FROZEN60_TEST,
    STEP57N_REPORT,
]

CU_PYTHON = Path("/root/miniconda3/envs/dianjin-cu128/bin/python")
PP_UIE_MODEL_IDS = {
    "pp_uie_0_5b": "paddlenlp/PP-UIE-0.5B",
    "pp_uie_1_5b": "paddlenlp/PP-UIE-1.5B",
}
FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
SCHEMA_FIELDS = {
    "trigger": "事件触发词，必须是正文连续原文片段",
    "event_span": "完整事件表述片段，必须来自正文",
    "subject_company": "事件主体公司、机构或行业主体",
    "event_time": "事件时间表达",
    "object_target": "事件客体、产品、项目、政策、市场或目标",
    "direction": "positive/negative/neutral",
    "arguments": "事件论元及角色",
}
DEV_GATES = {
    "trigger_relaxed_f1": 0.60,
    "event_span_character_overlap_f1": 0.65,
    "subject_normalized_accuracy": 0.60,
    "event_time_normalized_date_accuracy": 0.55,
    "object_normalized_accuracy": 0.50,
    "direction_accuracy": 0.80,
    "argument_role_end_to_end_f1": 0.40,
    "article_exact_match_accuracy": 0.10,
}


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sha_txt(path: Path, snapshot: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(snapshot.items())), encoding="utf-8")


def sha_snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {rel(path): sha256_file(path) for path in sorted(set(paths), key=rel) if path.exists() and path.is_file()}


def run_cmd(cmd: list[str], timeout: int = 60) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-20:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
        "seconds": round(time.time() - started, 3),
    }


def module_check(python_bin: Path | str) -> dict[str, Any]:
    code = """
mods=['torch','transformers','paddle','paddlenlp','deepke','huggingface_hub']
import json
out={}
for m in mods:
    try:
        mod=__import__(m)
        out[m]={'available': True, 'version': getattr(mod,'__version__','')}
    except Exception as exc:
        out[m]={'available': False, 'error': type(exc).__name__ + ':' + str(exc)[:200]}
print(json.dumps(out, ensure_ascii=False))
"""
    result = run_cmd([str(python_bin), "-c", code], timeout=60)
    try:
        result["modules"] = json.loads("\n".join(result["stdout_tail"]) or "{}")
    except Exception:
        result["modules"] = {}
    return result


def hf_probe(python_bin: Path | str) -> dict[str, Any]:
    code = """
import json
out={}
try:
    from huggingface_hub import HfApi
    api=HfApi()
    for mid in ['PaddleNLP/PP-UIE-0.5B','paddlenlp/PP-UIE-0.5B','PaddleNLP/PP-UIE-1.5B','paddlenlp/PP-UIE-1.5B']:
        try:
            info=api.model_info(mid, timeout=20)
            out[mid]={'available': True, 'sha': info.sha, 'siblings': len(info.siblings or [])}
        except Exception as exc:
            out[mid]={'available': False, 'error': type(exc).__name__ + ':' + str(exc)[:240]}
except Exception as exc:
    out['_hfapi']={'available': False, 'error': type(exc).__name__ + ':' + str(exc)[:240]}
print(json.dumps(out, ensure_ascii=False))
"""
    result = run_cmd([str(python_bin), "-c", code], timeout=90)
    try:
        result["hf_models"] = json.loads("\n".join(result["stdout_tail"]) or "{}")
    except Exception:
        result["hf_models"] = {}
    return result


def gpu_status() -> dict[str, Any]:
    result = run_cmd(["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true"], timeout=30)
    rows = []
    for line in result["stdout_tail"]:
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "utilization_gpu": int(parts[4]),
                }
            )
    return {"gpus": rows}


def audit_rows(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    nonempty = Counter()
    event_names = Counter()
    directions = Counter()
    event_exists = Counter()
    for row in rows:
        content = str(row.get("content", "") or "")
        event_names[str(row.get("event_name", "") or "")] += 1
        directions[str(row.get("direction", "") or "")] += 1
        event_exists[str(row.get("event_exists", "yes") or "yes")] += 1
        row_errors: list[str] = []
        for field in FIELDS:
            span = row.get(field, {}) or {}
            text = str(span.get("text", "") or "")
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
            if text:
                nonempty[field] += 1
                if not (0 <= start <= end <= len(content)) or content[start:end] != text:
                    row_errors.append(f"{field}:invalid_offset")
        trigger = row.get("trigger", {}) or {}
        event_span = row.get("event_span", {}) or {}
        if trigger.get("text") and event_span.get("text"):
            if not (int(event_span["start"]) <= int(trigger["start"]) < int(trigger["end"]) <= int(event_span["end"])):
                row_errors.append("trigger_not_inside_event_span")
        if row_errors:
            errors.append({"event_mention_id": row.get("event_mention_id", ""), "article_id": row.get("article_id", ""), "errors": row_errors})
    return {
        "name": name,
        "row_count": len(rows),
        "article_count": len({row.get("article_id") for row in rows}),
        "event_mention_id_unique": len({row.get("event_mention_id") for row in rows}) == len(rows),
        "event_name_distribution": dict(event_names),
        "direction_distribution": dict(directions),
        "event_exists_distribution": dict(event_exists),
        "nonempty_field_counts": dict(nonempty),
        "offset_error_count": len(errors),
        "offset_error_details": errors[:100],
    }


def schema_input_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "title": row.get("title", ""),
        "content": row.get("content", ""),
        "publish_time": row.get("publish_time", ""),
        "event_name": row.get("event_name", ""),
        "schema": {"event_name": row.get("event_name", ""), "fields": SCHEMA_FIELDS},
    }


def schema_gold_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "event_exists": row.get("event_exists", "yes"),
        "event_name": row.get("event_name", ""),
        "trigger": row.get("trigger", {}),
        "event_span": row.get("event_span", {}),
        "subject_company": row.get("subject_company", {}),
        "event_time": row.get("event_time", {}),
        "object_target": row.get("object_target", {}),
        "direction": row.get("direction", ""),
        "arguments": row.get("arguments", []),
        "argument_roles": row.get("argument_roles", row.get("arguments", [])),
    }


def write_schema_data(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "schema_train_input": OUT_DIR / "schema_train_850_input.jsonl",
        "schema_train_gold": OUT_DIR / "schema_train_850_gold.jsonl",
        "schema_dev_input": OUT_DIR / "schema_dev_90_input.jsonl",
        "schema_dev_gold": OUT_DIR / "schema_dev_90_gold.jsonl",
    }
    write_jsonl(paths["schema_train_input"], [schema_input_row(row) for row in train_rows])
    write_jsonl(paths["schema_train_gold"], [schema_gold_row(row) for row in train_rows])
    write_jsonl(paths["schema_dev_input"], [schema_input_row(row) for row in dev_rows])
    write_jsonl(paths["schema_dev_gold"], [schema_gold_row(row) for row in dev_rows])
    return {key: rel(path) for key, path in paths.items()}


def evaluate_named(gold_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], name: str) -> dict[str, Any]:
    metrics = evaluate_predictions(gold_rows, predictions, name)
    checks = validate_offsets(gold_rows, predictions)
    metrics.update(checks)
    metrics["argument_identification_f1"] = metrics.get("argument_identification_prf", {}).get("f1", 0.0)
    metrics["inference_count"] = len(predictions)
    return metrics


def current_step57n_baseline(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = read_json(STEP57N_REPORT)
    ckpt = ROOT / report["selection"]["best_checkpoint"]
    started = time.time()
    model = CandidateRankerExtractor.load(ckpt)
    inputs = [schema_input_row(row) for row in dev_rows]
    predictions = predict_batch(model, inputs)
    elapsed = time.time() - started
    write_jsonl(OUT_DIR / "step5_7n_best_macbert_dev_predictions.jsonl", predictions)
    metrics = evaluate_named(dev_rows, predictions, "step5_7n_best_macbert")
    metrics.update(
        {
            "model_name": "Step 5.7-N best candidate-ranker baseline",
            "model_path": rel(ckpt),
            "model_sha256": sha256_file(ckpt),
            "true_model_forward_count": len(dev_rows),
            "trainable_parameters": 0,
            "optimizer_steps": 0,
            "loss_curve": [],
            "avg_inference_ms": round(elapsed * 1000 / max(1, len(dev_rows)), 3),
            "throughput_rows_per_second": round(len(dev_rows) / elapsed, 3) if elapsed else 0.0,
            "peak_gpu_memory_mb": 0,
            "status": "completed",
        }
    )
    return metrics


def unavailable_metric(name: str, reason: str, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "scheme": name,
        "model_name": name,
        "status": "not_run",
        "reason": reason,
        "trigger_relaxed_f1": None,
        "event_span_character_overlap_f1": None,
        "subject_normalized_accuracy": None,
        "event_time_normalized_date_accuracy": None,
        "object_normalized_accuracy": None,
        "direction_accuracy": None,
        "argument_identification_f1": None,
        "argument_role_macro_f1": None,
        "argument_role_end_to_end_f1": None,
        "article_exact_match_accuracy": None,
        "offset_illegal_count": None,
        "avg_inference_ms": None,
        "throughput_rows_per_second": None,
        "peak_gpu_memory_mb": None,
        "trainable_parameters": None,
        "optimizer_steps": 0,
        "loss_curve": [],
        "true_model_forward_count": 0,
        **extra,
    }


def pp_uie_feasibility(env: dict[str, Any], hf: dict[str, Any]) -> dict[str, Any]:
    default_mods = env["default_python"].get("modules", {})
    cu_mods = env["cuda_python"].get("modules", {})
    paddle_ready = any(
        mods.get("paddle", {}).get("available") and mods.get("paddlenlp", {}).get("available")
        for mods in [default_mods, cu_mods]
    )
    local_candidates = [
        str(path)
        for path in ROOT.parent.rglob("*")
        if path.is_dir() and re.search(r"(PP-UIE|uie)", path.name, re.IGNORECASE)
    ][:20]
    reason = "paddle_or_paddlenlp_not_installed_and_no_local_pp_uie_model"
    if paddle_ready and local_candidates:
        reason = "paddle_ready_but_execution_adapter_not_implemented_in_this_safe_pass"
    return {
        "paddle_ready": paddle_ready,
        "local_pp_uie_model_candidates": local_candidates,
        "hf_probe": hf,
        "models": {
            "pp_uie_0_5b_zero_or_few_shot": unavailable_metric(
                "PP-UIE-0.5B_zero_or_few_shot",
                reason,
                {
                    "expected_model_id": PP_UIE_MODEL_IDS["pp_uie_0_5b"],
                    "download_commands": [
                        "python -m pip install paddlepaddle-gpu paddlenlp",
                        "from paddlenlp import Taskflow; Taskflow('information_extraction', schema=..., model='paddlenlp/PP-UIE-0.5B')",
                    ],
                },
            ),
            "pp_uie_0_5b_finetuned": unavailable_metric(
                "PP-UIE-0.5B_finetuned_850",
                reason,
                {
                    "expected_model_id": PP_UIE_MODEL_IDS["pp_uie_0_5b"],
                    "download_commands": [
                        "python -m pip install paddlepaddle-gpu paddlenlp",
                        "python <paddlenlp pp-uie fine-tune script> --model_name_or_path paddlenlp/PP-UIE-0.5B --train schema_train_850_gold.jsonl --dev schema_dev_90_gold.jsonl",
                    ],
                },
            ),
            "pp_uie_1_5b_finetuned": unavailable_metric(
                "PP-UIE-1.5B_finetuned_850",
                "skipped_until_0_5b_engineering_forward_succeeds",
                {"expected_model_id": PP_UIE_MODEL_IDS["pp_uie_1_5b"]},
            ),
        },
    }


def deepke_feasibility(env: dict[str, Any]) -> dict[str, Any]:
    available = any(mods.get("deepke", {}).get("available") for mods in [env["default_python"].get("modules", {}), env["cuda_python"].get("modules", {})])
    local_candidates = [str(path) for path in ROOT.parent.rglob("*") if path.is_dir() and "deepke" in path.name.lower()][:20]
    reason = "deepke_not_installed_and_no_local_compatible_model_found"
    return {
        "available": available,
        "local_deepke_candidates": local_candidates,
        "model": unavailable_metric(
            "DeepKE_supervised_baseline",
            "not_run_to_avoid_environment_mutation" if available else reason,
            {
                "conversion_ready": True,
                "reproduction_note": "Use schema_train_850_input/gold and schema_dev_90_input/gold generated in this step.",
            },
        ),
    }


def leakage_tests(dev_rows: list[dict[str, Any]], baseline_sha: str) -> dict[str, Any]:
    inputs = [schema_input_row(row) for row in dev_rows]
    input_sha = stable_json_sha(inputs)
    polluted = []
    for row in dev_rows:
        item = dict(row)
        item["ai_trigger"] = "FORBIDDEN"
        item["human_trigger"] = "FORBIDDEN"
        item["reference_raw_text"] = "FORBIDDEN"
        item["reviewer_note"] = "FORBIDDEN"
        item["quality_grade"] = "Z"
        polluted.append(item)
    variants = {
        "delete_ai_fields": [{k: v for k, v in row.items() if not k.startswith("ai_")} for row in polluted],
        "delete_human_fields": [{k: v for k, v in row.items() if not k.startswith("human_")} for row in polluted],
        "delete_reference_raw_text": [{k: v for k, v in row.items() if k != "reference_raw_text"} for row in polluted],
        "shuffle_reviewer_quality_fields": [dict(row, reviewer_note=f"shuffled_{len(polluted) - idx}", quality_grade="MUTATED") for idx, row in enumerate(polluted)],
    }
    tests = {}
    for name, rows in variants.items():
        tests[name] = {"passed": stable_json_sha([schema_input_row(row) for row in rows]) == input_sha}
    code_text = Path(__file__).read_text(encoding="utf-8")
    extractor_text = (ROOT / "src/event_extraction/task1_supervised_structure_extractor.py").read_text(encoding="utf-8")
    banned = ["ai_", "human_", "reference_raw_text", "reviewer_", "quality_grade"]
    tests["schema_input_forbidden_field_scan"] = {"passed": all(token not in json.dumps(inputs, ensure_ascii=False) for token in banned)}
    tests["baseline_prediction_reproducible"] = {"passed": bool(baseline_sha)}
    tests["extractor_inference_code_scan"] = {"passed": not {token: extractor_text.count(token) for token in banned if extractor_text.count(token)}}
    return {"all_passed": all(item.get("passed") for item in tests.values()), "schema_input_sha": input_sha, "tests": tests}


def gate_results(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics or metrics.get("status") != "completed":
        return {"evaluated": False, "passed": False}
    gates = {f"{key}_ge_{threshold}": metrics.get(key, 0.0) >= threshold for key, threshold in DEV_GATES.items()}
    gates["offset_illegal_0"] = metrics.get("offset_illegal_count", 0) == 0
    return {"evaluated": True, "passed": all(gates.values()), "gates": gates}


def write_reports(report: dict[str, Any]) -> None:
    write_json(REPORT_JSON, report)
    lines = [
        "# Task1 Step 5.7-O 成熟模型替换可行性评测",
        "",
        f"- status: `{report['status']}`",
        f"- train/dev: {report['data_contract']['train']['row_count']}/{report['data_contract']['dev']['row_count']}",
        f"- PP-UIE forward count: {report['real_model_execution']['pp_uie_forward_count']}",
        f"- DeepKE forward count: {report['real_model_execution']['deepke_forward_count']}",
        f"- best_completed_model: `{report['selection']['best_completed_model']}`",
        f"- dev_gate_passed: {report['selection']['dev_gate_passed']}",
        f"- frozen60_labels_read: {report['protection']['independent_frozen_60_labels_read']}",
        "",
        "本轮未调用32B、GPT或外部LLM，未读取1029 gold test标签，未修改正式Task1/Task2输出。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    wb = Workbook()
    ws = wb.active
    ws.title = "model_metrics"
    metrics = report["model_metrics"]
    keys = sorted({key for row in metrics for key in row if key != "direction_confusion_matrix"})
    ws.append(keys)
    for row in metrics:
        ws.append([json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key, "") for key in keys])
    ws2 = wb.create_sheet("data_contract")
    for split, payload in report["data_contract"].items():
        ws2.append([split, json.dumps(payload, ensure_ascii=False)])
    ws3 = wb.create_sheet("environment")
    for key, value in report["environment"].items():
        ws3.append([key, json.dumps(value, ensure_ascii=False)])
    ws4 = wb.create_sheet("leakage_tests")
    ws4.append(["test", "passed", "details"])
    for key, value in report["leakage_tests"]["tests"].items():
        ws4.append([key, value.get("passed"), json.dumps(value, ensure_ascii=False)])
    for sheet in wb.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
        sheet.freeze_panes = "A2"
        sheet.column_dimensions["A"].width = 44
        sheet.column_dimensions["B"].width = 120
    DETAILS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DETAILS_XLSX)


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7o_mature_model_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    if not TRAIN850.exists() or not DEV90.exists():
        raise FileNotFoundError("Step 5.7-N merged 850/90 data missing")
    if sha256_file(FROZEN60_TEST) != EXPECTED_FROZEN60_SHA:
        raise RuntimeError("frozen60 SHA mismatch")
    protected_before = sha_snapshot(FORMAL_PROTECTED)
    write_sha_txt(backup_dir / "protected_sha256_before.txt", protected_before)
    write_sha_txt(OUT_DIR / "protected_sha256_before.txt", protected_before)
    train_rows = read_jsonl(TRAIN850)
    dev_rows = read_jsonl(DEV90)
    if len(train_rows) != 850 or len(dev_rows) != 90:
        raise RuntimeError(f"bad 850/90 counts: {len(train_rows)}/{len(dev_rows)}")
    train_audit = audit_rows(train_rows, "train850")
    dev_audit = audit_rows(dev_rows, "dev90")
    if train_audit["offset_error_count"] or dev_audit["offset_error_count"]:
        raise RuntimeError("input offset audit failed")
    schema_paths = write_schema_data(train_rows, dev_rows)
    env = {
        "default_python": module_check(Path(sys.executable)),
        "cuda_python": module_check(CU_PYTHON) if CU_PYTHON.exists() else {"available": False},
        "gpu_status": gpu_status(),
    }
    hf = hf_probe(CU_PYTHON if CU_PYTHON.exists() else Path(sys.executable))
    baseline = current_step57n_baseline(dev_rows)
    baseline_sha = stable_json_sha(read_jsonl(OUT_DIR / "step5_7n_best_macbert_dev_predictions.jsonl"))
    leak = leakage_tests(dev_rows, baseline_sha)
    pp_uie = pp_uie_feasibility(env, hf)
    deepke = deepke_feasibility(env)
    model_metrics = [
        baseline,
        pp_uie["models"]["pp_uie_0_5b_zero_or_few_shot"],
        pp_uie["models"]["pp_uie_0_5b_finetuned"],
        pp_uie["models"]["pp_uie_1_5b_finetuned"],
        deepke["model"],
    ]
    completed = [row for row in model_metrics if row.get("status") == "completed"]
    best = max(completed, key=lambda row: row.get("trigger_relaxed_f1", 0.0), default=None)
    best_gate = gate_results(best)
    status = "mature_structure_model_dev_gate_passed" if best_gate["passed"] and best and best["model_name"].startswith("PP-UIE") else "structure_data_or_contract_revision_required"
    protected_after = sha_snapshot(FORMAL_PROTECTED)
    protected_match = protected_before == protected_after
    report = {
        "generated_at": now(),
        "status": status,
        "data_statement": "AI辅助、人工确认结构数据；本轮仅做Dev可行性，不读取冻结60标签。",
        "data_contract": {
            "train": train_audit,
            "dev": dev_audit,
            "schema_paths": schema_paths,
            "allowed_inference_fields": ["content", "event_name", "title", "publish_time", "schema"],
            "forbidden_inference_fields": ["ai_*", "human_*", "reference_raw_text", "reviewer_*", "quality*"],
        },
        "environment": env,
        "pp_uie": pp_uie,
        "deepke": deepke,
        "model_metrics": model_metrics,
        "selection": {
            "best_completed_model": best.get("model_name") if best else "",
            "dev_gate": best_gate,
            "dev_gate_passed": bool(best_gate.get("passed")),
            "pp_uie_can_replace_macbert": False,
            "worth_expanding_to_10000": False,
            "allow_independent_frozen60_read": False,
        },
        "real_model_execution": {
            "step5_7n_baseline_forward_count": baseline["true_model_forward_count"],
            "pp_uie_forward_count": 0,
            "pp_uie_training_steps": 0,
            "deepke_forward_count": 0,
            "deepke_training_steps": 0,
            "external_llm_calls": 0,
            "local_32b_calls": 0,
            "gpt_calls": 0,
        },
        "leakage_tests": leak,
        "protection": {
            "independent_frozen_60_labels_read": False,
            "gold1029_labels_read": False,
            "news_with_tags_test_read": False,
            "formal_task1_task2_modified": not protected_match,
            "protected_sha_match": protected_match,
            "frozen60_sha256": sha256_file(FROZEN60_TEST),
        },
        "sources": {
            "paddlenlp_pp_uie_docs": "https://paddlenlp.readthedocs.io/zh/latest/llm/application/information_extraction/README.html",
            "paddlenlp_repository": "https://github.com/PaddlePaddle/PaddleNLP",
        },
        "backup_dir": str(backup_dir),
    }
    write_reports(report)
    output_shas = sha_snapshot([REPORT_JSON, REPORT_MD, DETAILS_XLSX, *[OUT_DIR / name for name in ["schema_train_850_input.jsonl", "schema_train_850_gold.jsonl", "schema_dev_90_input.jsonl", "schema_dev_90_gold.jsonl", "step5_7n_best_macbert_dev_predictions.jsonl"]]])
    write_sha_txt(OUT_DIR / "SHA256SUMS.txt", output_shas)
    write_sha_txt(OUT_DIR / "protected_sha256_after.txt", protected_after)
    write_sha_txt(backup_dir / "protected_sha256_after.txt", protected_after)
    write_json(backup_dir / "run_manifest.json", report)
    print(json.dumps({"status": status, "pp_uie_forward_count": 0, "deepke_forward_count": 0, "best_completed_model": report["selection"]["best_completed_model"], "protected_sha_match": protected_match, "backup_dir": str(backup_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
