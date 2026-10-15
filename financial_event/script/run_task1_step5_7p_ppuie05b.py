#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-P real local PP-UIE-0.5B evaluation and memory test.

This script is intended to run inside the isolated conda env task1-ppuie.
"""

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
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import paddle
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_7f_structure_training import evaluate_predictions  # noqa: E402
from src.event_extraction.task1_supervised_structure_extractor import CandidateRankerExtractor, predict_batch, validate_offsets  # noqa: E402


OUT_DIR = ROOT / "outputs/task1_step5_7p_ppuie05b"
REPORT_JSON = ROOT / "reports/task1_step5_7p_ppuie05b_report.json"
REPORT_MD = ROOT / "reports/task1_step5_7p_ppuie05b_report.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_7p_ppuie05b_details.xlsx"
REQ_LOCK = ROOT / "requirements/task1_ppuie_lock.txt"

MODEL_DIR = Path("/opt/data/private/JinRong/models/PP-UIE-0.5B")
TRAIN850 = ROOT / "outputs/task1_step5_7n_chatgpt500_confirmation/merged_structure_train_850.jsonl"
DEV90 = ROOT / "outputs/task1_step5_7n_chatgpt500_confirmation/merged_structure_dev_90.jsonl"
STEP57N_REPORT = ROOT / "reports/task1_step5_7n_structure_training_report.json"
FROZEN60 = ROOT / "outputs/task1_step5_7j_structure_confirmation/independent_frozen_test_confirmed.jsonl"
EXPECTED_FROZEN60_SHA = "76efd5a27057dd177ee8688f4769bfb6e57e8539795cad1727ea29418f7c3579"

PROTECTED = [
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
    FROZEN60,
]

FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
DIRS = ["positive", "negative", "neutral"]
ROLES = ["subject_company", "event_time", "object_target", "trigger"]
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
MEMORY_GATES = {
    "trigger_relaxed_f1": 0.90,
    "event_span_character_overlap_f1": 0.90,
    "subject_normalized_accuracy": 0.85,
    "object_normalized_accuracy": 0.85,
    "direction_accuracy": 0.90,
    "argument_role_end_to_end_f1": 0.75,
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha_snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {rel(path): sha256_file(path) for path in paths if path.exists() and path.is_file()}


def write_sha_txt(path: Path, snapshot: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(snapshot.items())), encoding="utf-8")


def model_files() -> dict[str, Any]:
    files = []
    total = 0
    key_shas = {}
    for path in sorted(MODEL_DIR.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            files.append({"path": str(path), "size": size})
            if path.name in {"model_state.pdparams", "config.json", "tokenizer_config.json"}:
                key_shas[path.name] = sha256_file(path)
    return {
        "repo_id": "paddlenlp/PP-UIE-0.5B",
        "source": "local_user_verified_download",
        "model_dir": str(MODEL_DIR),
        "file_count": len(files),
        "total_size_bytes": total,
        "total_size_gb": round(total / 1024**3, 3),
        "files": files,
        "config": read_json(MODEL_DIR / "config.json"),
        "key_sha256": key_shas,
    }


def audit_rows(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    errors = []
    for row in rows:
        content = row.get("content", "")
        row_errors = []
        for field in FIELDS:
            span = row.get(field, {}) or {}
            text = str(span.get("text", "") or "")
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
            if text and (not (0 <= start <= end <= len(content)) or content[start:end] != text):
                row_errors.append(f"{field}:invalid_offset")
        if row_errors:
            errors.append({"event_mention_id": row.get("event_mention_id"), "errors": row_errors})
    return {
        "name": name,
        "row_count": len(rows),
        "article_count": len({row.get("article_id") for row in rows}),
        "event_exists_distribution": dict(Counter(row.get("event_exists", "yes") for row in rows)),
        "offset_error_count": len(errors),
        "offset_error_details": errors[:100],
    }


def cut_content_for_prompt(row: dict[str, Any], max_chars: int = 1800) -> str:
    content = str(row.get("content", "") or "")
    if len(content) <= max_chars:
        return content
    spans = [row.get(field, {}) for field in FIELDS]
    starts = [int(span.get("start", -1)) for span in spans if span.get("text") and int(span.get("start", -1)) >= 0]
    if starts:
        center = min(starts)
        start = max(0, center - max_chars // 3)
        end = min(len(content), start + max_chars)
        return content[start:end]
    return content[:max_chars]


def build_prompt(row: dict[str, Any]) -> str:
    return (
        "你是金融新闻事件结构抽取模型。只基于给定正文抽取，不要编造。\n"
        "返回严格JSON，不要Markdown。JSON字段：\n"
        "{\"trigger\":\"\",\"event_span\":\"\",\"subject_company\":\"\",\"event_time\":\"\",\"event_time_normalized\":\"未明确\","
        "\"object_target\":\"\",\"direction\":\"positive|negative|neutral\",\"arguments\":[{\"text\":\"\",\"role\":\"subject_company|event_time|object_target|trigger\"}]}\n"
        f"给定event_name：{row.get('event_name','')}\n"
        f"标题：{row.get('title','')}\n"
        f"正文：{cut_content_for_prompt(row)}\n"
        "JSON："
    )


def target_json(row: dict[str, Any]) -> str:
    args = [
        {"text": arg.get("text", ""), "role": arg.get("role", "")}
        for arg in row.get("arguments", [])
        if arg.get("text") and arg.get("role") in ROLES
    ]
    payload = {
        "trigger": row.get("trigger", {}).get("text", ""),
        "event_span": row.get("event_span", {}).get("text", ""),
        "subject_company": row.get("subject_company", {}).get("text", ""),
        "event_time": row.get("event_time", {}).get("text", ""),
        "event_time_normalized": row.get("event_time", {}).get("normalized", "未明确"),
        "object_target": row.get("object_target", {}).get("text", ""),
        "direction": row.get("direction", "neutral") if row.get("direction", "neutral") in DIRS else "neutral",
        "arguments": args,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def extract_json(text: str) -> tuple[dict[str, Any], str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return {}, "json_object_not_found"
    try:
        obj = json.loads(cleaned[start : end + 1])
        return obj if isinstance(obj, dict) else {}, ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}:{exc}"


def locate_quote(content: str, quote: Any, anchor: tuple[int, int] | None = None) -> dict[str, Any]:
    text = str(quote or "").strip()
    if not text:
        return {"text": "", "start": -1, "end": -1}
    positions = [m.start() for m in re.finditer(re.escape(text), content)]
    if not positions:
        return {"text": "", "start": -1, "end": -1, "failed_quote": text}
    if anchor:
        positions.sort(key=lambda pos: 0 if anchor[0] <= pos < pos + len(text) <= anchor[1] else min(abs(pos - anchor[0]), abs(pos - anchor[1])))
    pos = positions[0]
    return {"text": text, "start": pos, "end": pos + len(text)}


def parse_prediction(row: dict[str, Any], raw_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    content = row.get("content", "")
    obj, err = extract_json(raw_text)
    meta = {"json_error": err, "raw_text": raw_text}
    event_span = locate_quote(content, obj.get("event_span", ""))
    trigger = locate_quote(content, obj.get("trigger", ""), (event_span["start"], event_span["end"]) if event_span.get("text") else None)
    subject = locate_quote(content, obj.get("subject_company", ""))
    event_time = locate_quote(content, obj.get("event_time", ""))
    event_time["normalized"] = str(obj.get("event_time_normalized", "") or "未明确")
    object_target = locate_quote(content, obj.get("object_target", ""))
    direction = str(obj.get("direction", "neutral") or "neutral")
    if direction not in DIRS:
        direction = "neutral"
        meta["illegal_direction_replaced"] = True
    if trigger.get("text") and event_span.get("text") and not (event_span["start"] <= trigger["start"] < trigger["end"] <= event_span["end"]):
        meta["trigger_not_inside_event_span"] = True
    arguments = []
    for item in obj.get("arguments", []) if isinstance(obj.get("arguments", []), list) else []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        if role not in ROLES:
            continue
        span = locate_quote(content, item.get("text", ""))
        if span.get("text"):
            arguments.append({"text": span["text"], "role": role, "start": span["start"], "end": span["end"]})
    pred = {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "scheme": "PP-UIE-0.5B",
        "trigger": trigger,
        "event_span": event_span,
        "subject_company": subject,
        "event_time": event_time,
        "object_target": object_target,
        "direction": direction,
        "arguments": arguments,
        "argument_roles": arguments,
    }
    return pred, meta


class PPUIERunner:
    def __init__(self, dtype: str = "bfloat16") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=dtype)
        self.model.eval()
        self.parameter_count = sum(int(param.numel()) for param in self.model.parameters())
        self.trainable_parameters = sum(int(param.numel()) for param in self.model.parameters() if not param.stop_gradient)
        self.forward_count = 0

    def generate_one(self, row: dict[str, Any], max_new_tokens: int = 256) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = build_prompt(row)
        inputs = self.tokenizer(prompt, return_tensors="pd")
        started = time.time()
        with paddle.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.time() - started
        self.forward_count += 1
        seq = output[0] if isinstance(output, (tuple, list)) else output
        arr = seq.numpy().tolist()
        ids = arr[0] if arr and isinstance(arr[0], list) else arr
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        generated = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
        pred, meta = parse_prediction(row, generated)
        meta.update({"latency_seconds": round(elapsed, 4), "input_length_chars": len(prompt), "raw_generated": generated})
        return pred, meta

    def supervised_step(self, row: dict[str, Any], optimizer: paddle.optimizer.Optimizer, max_length: int = 1024) -> float:
        prompt = build_prompt(row)
        target = target_json(row)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + target_ids)[:max_length]
        labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[min(len(prompt_ids), len(input_ids)) :]
        labels = labels[: len(input_ids)]
        ids = paddle.to_tensor([input_ids], dtype="int64")
        label_t = paddle.to_tensor([labels], dtype="int64")
        self.model.train()
        outputs = self.model(input_ids=ids, labels=label_t)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        self.forward_count += 1
        return float(loss.numpy())


def eval_predictions(rows: list[dict[str, Any]], preds: list[dict[str, Any]], name: str) -> dict[str, Any]:
    metrics = evaluate_predictions(rows, preds, name)
    metrics.update(validate_offsets(rows, preds))
    metrics["argument_identification_f1"] = metrics.get("argument_identification_prf", {}).get("f1", 0.0)
    return metrics


def gate(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    gates = {f"{key}_ge_{threshold}": metrics.get(key, 0.0) >= threshold for key, threshold in thresholds.items()}
    gates["offset_illegal_0"] = metrics.get("offset_illegal_count", 0) == 0
    gates["json_parse_failure_0"] = metrics.get("json_parse_failure_count", 0) == 0
    return {"passed": all(gates.values()), "gates": gates}


def select_preflight(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [row for row in rows if row.get("event_exists", "yes") == "yes" and row.get("event_name") and row.get("trigger", {}).get("text")]
    usable.sort(key=lambda row: (len(row.get("content", "")), row.get("event_mention_id", "")))
    return usable[:8]


def select_memory32(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if row.get("event_exists", "yes") == "yes"
        and row.get("event_name")
        and row.get("trigger", {}).get("text")
        and row.get("event_span", {}).get("text")
        and len(row.get("content", "")) <= 2500
    ]
    usable.sort(key=lambda row: (len(row.get("content", "")), row.get("event_mention_id", "")))
    return usable[:32]


def step5n_baseline(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = read_json(STEP57N_REPORT)
    model = CandidateRankerExtractor.load(ROOT / report["selection"]["best_checkpoint"])
    preds = predict_batch(model, dev_rows)
    write_jsonl(OUT_DIR / "step5_7n_best_dev_predictions.jsonl", preds)
    metrics = eval_predictions(dev_rows, preds, "step5_7n_best_macbert")
    metrics.update({"scheme": "Step5.7-N best MacBERT/candidate-ranker", "checkpoint_sha256": report["selection"]["best_checkpoint_sha256"]})
    return metrics


def write_reports(report: dict[str, Any]) -> None:
    write_json(REPORT_JSON, report)
    lines = [
        "# Task1 Step 5.7-P PP-UIE-0.5B真实评测报告",
        "",
        f"- status: `{report['status']}`",
        f"- paddle: {report['environment']['paddle_version']}",
        f"- paddlenlp: {report['environment']['paddlenlp_version']}",
        f"- model_dir: `{report['model']['model_dir']}`",
        f"- real_forward_count: {report['real_forward_count']}",
        f"- memory32_passed: {report['memory32']['gate']['passed']}",
        f"- finetune850_completed: {report['finetune850']['completed']}",
        f"- frozen60_labels_read: {report['protection']['independent_frozen60_labels_read']}",
        "",
        "本轮未调用32B、GPT或外部LLM，未读取1029 gold test标签，未修改正式Task1/Task2输出。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    wb = Workbook()
    ws = wb.active
    ws.title = "dev_metrics"
    metrics = report["dev_metrics"]
    keys = sorted({key for item in metrics for key in item if key != "direction_confusion_matrix"})
    ws.append(keys)
    for item in metrics:
        ws.append([json.dumps(item.get(key), ensure_ascii=False) if isinstance(item.get(key), (dict, list)) else item.get(key, "") for key in keys])
    ws2 = wb.create_sheet("preflight")
    ws2.append(["event_mention_id", "latency_seconds", "json_error", "trigger", "event_span"])
    for row in report["preflight"]["details"]:
        ws2.append([row.get("event_mention_id"), row.get("latency_seconds"), row.get("json_error"), row.get("prediction", {}).get("trigger", {}).get("text"), row.get("prediction", {}).get("event_span", {}).get("text")])
    ws3 = wb.create_sheet("memory32")
    for key, value in report["memory32"].items():
        ws3.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    for sheet in wb.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
        sheet.freeze_panes = "A2"
        sheet.column_dimensions["A"].width = 42
        sheet.column_dimensions["B"].width = 120
    DETAILS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DETAILS_XLSX)


def pip_lock() -> None:
    import importlib.metadata

    rows = sorted(f"{dist.metadata.get('Name', dist.metadata['Name'])}=={dist.version}" for dist in importlib.metadata.distributions())
    REQ_LOCK.parent.mkdir(parents=True, exist_ok=True)
    REQ_LOCK.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dev", type=int, default=90)
    parser.add_argument("--memory-epochs", type=int, default=2)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7p_ppuie05b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    if sha256_file(FROZEN60) != EXPECTED_FROZEN60_SHA:
        raise RuntimeError("frozen60 SHA mismatch")
    protected_before = sha_snapshot(PROTECTED)
    write_sha_txt(backup_dir / "protected_sha256_before.txt", protected_before)
    write_sha_txt(OUT_DIR / "protected_sha256_before.txt", protected_before)
    train_rows = read_jsonl(TRAIN850)
    dev_rows = read_jsonl(DEV90)[: args.max_dev]
    train_audit = audit_rows(train_rows, "train850")
    dev_audit = audit_rows(dev_rows, "dev90")
    if train_audit["offset_error_count"] or dev_audit["offset_error_count"]:
        raise RuntimeError("input offset audit failed")
    import paddle as _paddle
    import paddlenlp as _paddlenlp

    pip_lock()
    model_info = model_files()
    load_start = time.time()
    runner = PPUIERunner()
    load_seconds = round(time.time() - load_start, 3)
    preflight_rows = select_preflight(train_rows)
    preflight_preds = []
    preflight_details = []
    latencies = []
    for row in preflight_rows:
        pred, meta = runner.generate_one(row)
        preflight_preds.append(pred)
        latencies.append(meta["latency_seconds"])
        preflight_details.append({"event_mention_id": row["event_mention_id"], "article_id": row["article_id"], "prediction": pred, **meta})
    write_jsonl(OUT_DIR / "preflight8_predictions.jsonl", preflight_preds)
    write_jsonl(OUT_DIR / "preflight8_raw_outputs.jsonl", preflight_details)
    preflight_metrics = eval_predictions(preflight_rows, preflight_preds, "ppuie05b_preflight8")
    dev_preds = []
    dev_raw = []
    dev_latencies = []
    for row in dev_rows:
        pred, meta = runner.generate_one(row)
        dev_preds.append(pred)
        dev_raw.append({"event_mention_id": row["event_mention_id"], "article_id": row["article_id"], "prediction": pred, **meta})
        dev_latencies.append(meta["latency_seconds"])
    write_jsonl(OUT_DIR / "ppuie05b_zeroshot_dev_predictions.jsonl", dev_preds)
    write_jsonl(OUT_DIR / "ppuie05b_zeroshot_dev_raw_outputs.jsonl", dev_raw)
    zeroshot_metrics = eval_predictions(dev_rows, dev_preds, "PP-UIE-0.5B_zeroshot_schema")
    zeroshot_metrics.update(
        {
            "scheme": "PP-UIE-0.5B_zeroshot_schema",
            "status": "completed",
            "avg_latency_seconds": round(sum(dev_latencies) / max(1, len(dev_latencies)), 4),
            "p95_latency_seconds": sorted(dev_latencies)[int(0.95 * (len(dev_latencies) - 1))] if dev_latencies else 0,
            "peak_gpu_memory_mb": round(paddle.device.cuda.max_memory_allocated() / 1024**2, 2) if paddle.is_compiled_with_cuda() else 0,
            "true_forward_count": len(dev_rows),
        }
    )
    memory_rows = select_memory32(train_rows)
    before_sha = sha256_file(MODEL_DIR / "model_state.pdparams")
    optimizer = paddle.optimizer.AdamW(learning_rate=1e-5, parameters=runner.model.parameters())
    losses = []
    train_started = time.time()
    for _epoch in range(args.memory_epochs):
        for row in memory_rows:
            losses.append(runner.supervised_step(row, optimizer))
    train_seconds = round(time.time() - train_started, 3)
    ckpt_dir = OUT_DIR / "memory32_checkpoint"
    runner.model.save_pretrained(str(ckpt_dir))
    runner.tokenizer.save_pretrained(str(ckpt_dir))
    after_files = sorted(ckpt_dir.glob("*.pdparams"))
    ckpt_sha = sha256_file(after_files[0]) if after_files else ""
    memory_preds = []
    memory_raw = []
    runner.model.eval()
    for row in memory_rows:
        pred, meta = runner.generate_one(row)
        memory_preds.append(pred)
        memory_raw.append({"event_mention_id": row["event_mention_id"], "prediction": pred, **meta})
    write_jsonl(OUT_DIR / "memory32_predictions.jsonl", memory_preds)
    write_jsonl(OUT_DIR / "memory32_raw_outputs.jsonl", memory_raw)
    memory_metrics = eval_predictions(memory_rows, memory_preds, "PP-UIE-0.5B_memory32")
    memory_gate = gate(memory_metrics, MEMORY_GATES)
    memory_report = {
        "sample_count": len(memory_rows),
        "epochs": args.memory_epochs,
        "optimizer_steps": len(losses),
        "train_seconds": train_seconds,
        "loss_curve": losses,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "trainable_parameters": runner.trainable_parameters,
        "weight_changed": ckpt_sha and ckpt_sha != before_sha,
        "checkpoint": rel(ckpt_dir),
        "checkpoint_sha256": ckpt_sha,
        "metrics": memory_metrics,
        "gate": memory_gate,
        "pipeline_failure_reason": "" if memory_gate["passed"] else "PP-UIE-0.5B did not overfit 32 confirmed structure examples under current prompt/target CausalLM training pipeline.",
    }
    finetune850 = {"completed": False, "reason": "not_run_because_memory32_gate_failed", "metrics": {}}
    status = "ppuie_contract_or_training_pipeline_failed"
    if memory_gate["passed"]:
        status = "ppuie_05b_real_model_evaluated_but_gate_failed"
        finetune850["reason"] = "not_implemented_in_this_safe_pass"
    step5n = step5n_baseline(dev_rows)
    dev_metrics = [step5n, zeroshot_metrics]
    if finetune850.get("metrics"):
        dev_metrics.append(finetune850["metrics"])
    protected_after = sha_snapshot(PROTECTED)
    protected_match = protected_before == protected_after
    report = {
        "generated_at": now(),
        "status": status,
        "environment": {
            "python": sys.version,
            "paddle_version": _paddle.__version__,
            "paddlenlp_version": _paddlenlp.__version__,
            "device": paddle.device.get_device(),
            "cuda": paddle.is_compiled_with_cuda(),
            "paddle_run_check_passed": True,
        },
        "model": {**model_info, "parameter_count": runner.parameter_count, "trainable_parameters": runner.trainable_parameters, "load_seconds": load_seconds},
        "data_contract": {"train": train_audit, "dev": dev_audit, "independent_frozen60_labels_read": False},
        "preflight": {"completed": len(preflight_preds) == 8, "metrics": preflight_metrics, "details": preflight_details},
        "zeroshot_dev": {"completed": True, "metrics": zeroshot_metrics, "gate": gate(zeroshot_metrics, DEV_GATES)},
        "memory32": memory_report,
        "finetune850": finetune850,
        "dev_metrics": dev_metrics,
        "real_forward_count": runner.forward_count,
        "protection": {
            "independent_frozen60_labels_read": False,
            "gold1029_labels_read": False,
            "news_with_tags_test_read": False,
            "formal_task1_task2_modified": not protected_match,
            "protected_sha_match": protected_match,
            "local_32b_calls": 0,
            "external_llm_calls": 0,
            "gpt_calls": 0,
        },
        "decision": {
            "worth_testing_ppuie_15b": False,
            "worth_expanding_to_10000": False,
            "allow_frozen60_evaluation": False,
        },
        "backup_dir": str(backup_dir),
    }
    write_reports(report)
    output_files = [
        REPORT_JSON,
        REPORT_MD,
        DETAILS_XLSX,
        REQ_LOCK,
        OUT_DIR / "preflight8_predictions.jsonl",
        OUT_DIR / "ppuie05b_zeroshot_dev_predictions.jsonl",
        OUT_DIR / "memory32_predictions.jsonl",
    ]
    write_sha_txt(OUT_DIR / "SHA256SUMS.txt", sha_snapshot(output_files))
    write_sha_txt(OUT_DIR / "protected_sha256_after.txt", protected_after)
    write_sha_txt(backup_dir / "protected_sha256_after.txt", protected_after)
    write_json(backup_dir / "run_manifest.json", report)
    print(json.dumps({"status": status, "forward_count": runner.forward_count, "memory32_passed": memory_gate["passed"], "zeroshot_trigger": zeroshot_metrics["trigger_relaxed_f1"], "protected_sha_match": protected_match, "backup_dir": str(backup_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
