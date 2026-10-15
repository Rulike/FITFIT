#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-T: PP-UIE equal-exposure and split-field memory tests.

This experiment is intentionally restricted to fixed memory samples selected
from the 850-row training set. It never reads the 90-row dev labels, the
independent frozen-test labels, or the 1029-row blind-test labels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import paddle
import paddle.nn.functional as F
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_7s_ppuie_multisample_memory import (  # noqa: E402
    FIELDS,
    FIELD_QUERIES,
    MODEL_DIR,
    NO_ENTITY,
    DIRS,
    ROLES,
    TRAIN850,
    answers_to_prediction,
    encode_shifted,
    eval_struct,
    field_free_exact,
    field_target,
    first_stage_article_exact,
    first_generated_line,
    generation_limit,
    generate_answer,
    load_base,
    normalize_time_for_memory,
    parser_preservation,
    pick_memory,
    peak_memory_mb,
    predict_fields,
    render_official_prompt,
    release_model,
    shifted_loss,
    teacher_accuracy,
    sha256_file,
)


OUT_DIR = ROOT / "outputs/task1_step5_7t_ppuie_equal_exposure"
REPORT_JSON = ROOT / "reports/task1_step5_7t_ppuie_equal_exposure.json"
REPORT_MD = ROOT / "reports/task1_step5_7t_ppuie_equal_exposure.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_7t_ppuie_equal_exposure_details.xlsx"
DEPLOYMENT_MD = ROOT / "reports/task1_step5_7t_ppuie_deployment_comparison.md"

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
    Path("/opt/data/private/JinRong/kechuang/task1234_delivery_20260811.tar.gz"),
]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha_snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {rel(path): sha256_file(path) for path in paths if path.exists() and path.is_file()}


def write_sha_txt(path: Path, snapshot: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(snapshot.items())), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def field_stats(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"qa_pair_count": len(rows) * len(FIELDS), "field": {}}
    for field in FIELDS:
        pairs = []
        answer_tokens = 0
        nonempty = 0
        truncated = 0
        for row in rows:
            prompt = render_official_prompt(tokenizer, row, field)
            target = field_target(row, field)
            enc = encode_shifted(tokenizer, prompt, target)
            answer_tokens += enc["answer_token_count"]
            nonempty += int(target != NO_ENTITY)
            truncated += int(enc["target_truncated"])
            pairs.append(
                {
                    "event_mention_id": row.get("event_mention_id", ""),
                    "field_name": field,
                    "target": target,
                    "target_token_count": enc["answer_token_count"],
                    "target_truncated": enc["target_truncated"],
                }
            )
        result["field"][field] = {
            "pair_count": len(pairs),
            "nonempty_pair_count": nonempty,
            "target_token_total": answer_tokens,
            "target_truncated_count": truncated,
            "pairs": pairs,
        }
    return result


def exposure_summary(rows: list[dict[str, Any]], total_steps: int) -> dict[str, Any]:
    pair_count = len(rows) * len(FIELDS)
    per_pair = total_steps / max(1, pair_count)
    per_field = total_steps / max(1, len(FIELDS))
    return {
        "sample_count": len(rows),
        "qa_pair_count": pair_count,
        "optimizer_steps": total_steps,
        "updates_per_pair": round(per_pair, 6),
        "updates_per_field": round(per_field, 6),
        "field_pair_counts": {field: len(rows) for field in FIELDS},
        "field_updates": {field: total_steps / len(FIELDS) for field in FIELDS},
    }


def summarize_s_exposure(s_report: dict[str, Any], sample_count: int) -> dict[str, Any]:
    joint = s_report.get("joint_curriculum", {})
    counts = joint.get("sampling_counts_total", {})
    field_updates = Counter()
    pair_counts = Counter()
    for key, value in counts.items():
        if "::" not in key:
            continue
        _, field = key.rsplit("::", 1)
        field_updates[field] += int(value)
        pair_counts[field] += 1
    total_steps = sum(int(v) for v in counts.values())
    pairs = len(counts)
    return {
        "sample_count": sample_count,
        "qa_pair_count": pairs,
        "optimizer_steps": total_steps,
        "updates_per_pair": round(total_steps / max(1, pairs), 6),
        "updates_per_field": {
            field: round(field_updates.get(field, 0) / max(1, pair_counts.get(field, sample_count)), 6)
            for field in FIELDS
        },
        "field_pair_counts": dict(pair_counts),
        "field_updates": dict(field_updates),
        "pair_update_min": min((int(v) for v in counts.values()), default=0),
        "pair_update_max": max((int(v) for v in counts.values()), default=0),
        "pair_update_mean": round(total_steps / max(1, pairs), 6),
        "pair_updates": counts,
        "stage_steps": [
            {"added_field": stage.get("added_field"), "steps": stage.get("steps")}
            for stage in joint.get("stages", [])
        ],
    }


def summarize_s_field_exposure(s_report: dict[str, Any], sample_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, item in (s_report.get("field_only") or {}).items():
        steps = int(item.get("steps", 0))
        result[field] = {
            "sample_count": sample_count,
            "optimizer_steps": steps,
            "updates_per_pair": round(steps / max(1, sample_count), 6),
            "target_steps_for_32": int(round(steps / max(1, sample_count) * 32)),
            "learning_rate": item.get("learning_rate"),
            "steps_source": "step5_7s_field_only_final",
        }
    return result


def state_probe(model: Any, limit: int = 20) -> str:
    h = hashlib.sha256()
    for idx, (_, param) in enumerate(model.state_dict().items()):
        if idx >= limit:
            break
        h.update(param.numpy().tobytes())
    return h.hexdigest()


def train_balanced(
    rows: list[dict[str, Any]],
    name: str,
    groups: list[list[str]],
    total_steps: int,
    learning_rate: float,
    eval_every: int,
    dtype: str,
    eval_fields: list[str] | None = None,
    early_stop_rounds: int = 2,
) -> dict[str, Any]:
    """Train one fresh model with field-balanced deterministic sampling."""
    print(json.dumps({"event": "train_start", "scheme": name, "sample_count": len(rows), "groups": groups, "total_steps": total_steps, "eval_every": eval_every}, ensure_ascii=False), flush=True)
    paddle.seed(2026)
    random.seed(2026)
    tokenizer, model = load_base(dtype)
    model.train()
    optimizer = paddle.optimizer.AdamW(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
        multi_precision=True,
    )
    before = state_probe(model)
    started = time.time()
    field_history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state: dict[str, Any] | None = None
    stale = 0
    # The sampler is explicitly field-balanced. For grouped experiments the
    # group receives equal probability, then fields within that group rotate.
    active_eval_fields = eval_fields or FIELDS
    tasks_by_group = [[(row, field) for row in rows for field in group] for group in groups]
    group_counts = Counter()
    field_counts = Counter()
    last_loss = 0.0
    for step in range(1, total_steps + 1):
        group_index = (step - 1) % len(tasks_by_group)
        group_tasks = tasks_by_group[group_index]
        item = group_tasks[((step - 1) // len(tasks_by_group)) % len(group_tasks)]
        row, field = item
        group_counts[group_index] += 1
        field_counts[field] += 1
        prompt = render_official_prompt(tokenizer, row, field)
        enc = encode_shifted(tokenizer, prompt, field_target(row, field))
        model.train()
        loss = shifted_loss(model, enc["input_ids"], enc["labels"])
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        last_loss = float(loss.numpy())
        if step % eval_every == 0 or step == total_steps:
            model.eval()
            preds, raw_answers, raw_details = predict_fields(model, tokenizer, rows, active_eval_fields)
            metrics = eval_struct(rows, preds, f"{name}_step{step}")
            free_by_field = {field_name: field_free_exact(rows, raw_answers, field_name) for field_name in active_eval_fields}
            teacher_by_field = {}
            for field_name in active_eval_fields:
                teacher_by_field[field_name] = round(
                    sum(teacher_accuracy(model, tokenizer, row_item, field_name) for row_item in rows) / max(1, len(rows)),
                    6,
                )
            preservation = parser_preservation(rows, raw_answers, active_eval_fields)
            score = sum(free_by_field.values()) / max(1, len(active_eval_fields))
            snapshot = {
                "step": step,
                "loss": last_loss,
                "free_exact_by_field": free_by_field,
                "teacher_accuracy_by_field": teacher_by_field,
                "parser_preservation_rate": preservation,
                "metrics": metrics,
                "gpu_peak_memory_mb": peak_memory_mb(),
                "elapsed_seconds": round(time.time() - started, 3),
                "raw_details": raw_details,
            }
            field_history.append(snapshot)
            print(json.dumps({"event": "eval", "scheme": name, "step": step, "free_exact_by_field": free_by_field, "parser": preservation, "article_first_stage": metrics.get("article_first_stage_exact_accuracy"), "offset_illegal": metrics.get("offset_illegal_count")}, ensure_ascii=False), flush=True)
            if score > best_score:
                best_score = score
                best_state = snapshot
                stale = 0
            else:
                stale += 1
            gate = all(free_by_field.get(field_name, 0.0) >= 0.90 for field_name in active_eval_fields) and preservation >= 0.99 and metrics.get("offset_illegal_count", 0) == 0
            if gate and stale >= early_stop_rounds:
                break
    model.eval()
    preds, raw_answers, raw_details = predict_fields(model, tokenizer, rows, active_eval_fields)
    metrics = eval_struct(rows, preds, f"{name}_final")
    free_by_field = {field_name: field_free_exact(rows, raw_answers, field_name) for field_name in active_eval_fields}
    teacher_by_field = {
        field_name: round(
            sum(teacher_accuracy(model, tokenizer, row_item, field_name) for row_item in rows) / max(1, len(rows)),
            6,
        )
        for field_name in active_eval_fields
    }
    preservation = parser_preservation(rows, raw_answers, active_eval_fields)
    after = state_probe(model)
    checkpoint_dir = OUT_DIR / "checkpoints" / name
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    checkpoint_sha = sha256_file(checkpoint_dir / "model_state.pdparams")
    final = {
        "run": True,
        "scheme": name,
        "sample_count": len(rows),
        "groups": groups,
        "eval_fields": active_eval_fields,
        "optimizer_steps": step,
        "learning_rate": learning_rate,
        "dtype": dtype,
        "elapsed_seconds": round(time.time() - started, 3),
        "peak_gpu_memory_mb": peak_memory_mb(),
        "weight_probe_changed": before != after,
        "base_state_probe": before,
        "trained_state_probe": after,
        "checkpoint": rel(checkpoint_dir),
        "checkpoint_sha256": checkpoint_sha,
        "field_sampling_counts": dict(field_counts),
        "group_sampling_counts": dict(group_counts),
        "history": field_history,
        "best_history": best_state or {},
        "metrics": metrics,
        "free_exact_by_field": free_by_field,
        "teacher_accuracy_by_field": teacher_by_field,
        "parser_preservation_rate": preservation,
        "raw_details": raw_details,
        "predictions": preds,
        "gate": {
            "trigger_ge_0_90": metrics.get("trigger_relaxed_f1", 0.0) >= 0.90,
            "event_span_ge_0_90": metrics.get("event_span_character_overlap_f1", 0.0) >= 0.90,
            "subject_ge_0_85": metrics.get("subject_normalized_accuracy", 0.0) >= 0.85,
            "time_ge_0_80": metrics.get("event_time_normalized_date_accuracy", 0.0) >= 0.80,
            "object_ge_0_80": metrics.get("object_normalized_accuracy", 0.0) >= 0.80,
            "direction_ge_0_90": metrics.get("direction_accuracy", 0.0) >= 0.90,
            "parser_ge_0_99": preservation >= 0.99,
            "offset_illegal_0": metrics.get("offset_illegal_count", 0) == 0,
            "article_first_stage_ge_0_60": metrics.get("article_first_stage_exact_accuracy", 0.0) >= 0.60,
        },
    }
    if active_eval_fields == FIELDS:
        final["passed"] = all(final["gate"].values())
    else:
        field = active_eval_fields[0]
        field_gate = {
            "teacher_ge_0_99": teacher_by_field.get(field, 0.0) >= 0.99,
            "free_exact_ge_0_90": free_by_field.get(field, 0.0) >= 0.90,
            "parser_ge_0_99": preservation >= 0.99,
            "offset_illegal_0": metrics.get("offset_illegal_count", 0) == 0,
        }
        metric_key = {
            "trigger": "trigger_relaxed_f1",
            "event_span": "event_span_character_overlap_f1",
            "subject_company": "subject_normalized_accuracy",
            "event_time": "event_time_normalized_date_accuracy",
            "object_target": "object_normalized_accuracy",
        }[field]
        threshold = {
            "trigger": 0.90,
            "event_span": 0.90,
            "subject_company": 0.85,
            "event_time": 0.80,
            "object_target": 0.80,
        }[field]
        field_gate[f"{field}_metric_ge_threshold"] = metrics.get(metric_key, 0.0) >= threshold
        final["field_gate"] = field_gate
        final["passed"] = all(field_gate.values())
    write_json(checkpoint_dir / "experiment_result.json", final)
    print(json.dumps({"event": "train_done", "scheme": name, "passed": final["passed"], "steps": step, "checkpoint": rel(checkpoint_dir)}, ensure_ascii=False), flush=True)
    release_model(model)
    return final


def make_report(
    before: dict[str, str],
    after: dict[str, str],
    exposure8: dict[str, Any],
    exposure32_s: dict[str, Any],
    exposure32_equal: dict[str, Any],
    field_results: dict[str, Any],
    groups: dict[str, Any],
    protection: dict[str, Any],
    backup_dir: Path,
) -> dict[str, Any]:
    equal_passed = bool(exposure32_equal.get("passed"))
    independent_passed = all(item.get("passed") for item in field_results.values())
    if equal_passed and independent_passed:
        status = "ready_for_ppuie_split_model_850_training"
    elif equal_passed:
        status = "ready_for_ppuie_joint_model_850_training"
    elif independent_passed and exposure32_equal.get("run"):
        status = "genuine_field_interference_confirmed"
    elif exposure32_equal.get("run") and exposure32_equal.get("budget_insufficient"):
        status = "insufficient_training_exposure_confirmed"
    elif exposure32_equal.get("run"):
        status = "ppuie_32_memory_still_failed"
    else:
        status = "field_specific_contract_failure"
    return {
        "generated_at": now(),
        "status": status,
        "step5_7s_exposure_audit": {
            "memory8": exposure8,
            "memory32_original": exposure32_s,
            "memory32_equal_exposure": exposure32_equal,
        },
        "field_only_32": field_results,
        "group_experiments": groups,
        "protection": protection,
        "backup_dir": str(backup_dir),
        "input_contract": {
            "train_source": rel(TRAIN850),
            "dev90_labels_read": False,
            "frozen60_labels_read": False,
            "gold1029_labels_read": False,
            "local_32b_calls": 0,
            "gpt_or_external_llm_calls": 0,
            "formal_task1_task2_modified": before != after,
        },
        "recommendation": {
            "allow_850_training": status in {"ready_for_ppuie_split_model_850_training", "ready_for_ppuie_joint_model_850_training"},
            "worth_expanding_10k": False,
            "recommended_form": "split_field_models" if independent_passed else "defer_training",
        },
    }


def write_outputs(report: dict[str, Any]) -> None:
    write_json(REPORT_JSON, report)
    lines = [
        "# Task1 Step 5.7-T PP-UIE等曝光训练对照与分字段模型32条验证",
        "",
        f"- status: `{report['status']}`",
        f"- 8-row updates_per_pair: `{report['step5_7s_exposure_audit']['memory8']['updates_per_pair']}`",
        f"- original 32-row updates_per_pair: `{report['step5_7s_exposure_audit']['memory32_original']['updates_per_pair']}`",
        f"- equal-exposure 32-row updates_per_pair: `{report['step5_7s_exposure_audit']['memory32_equal_exposure']['updates_per_pair']}`",
        f"- equal-exposure joint passed: `{report['step5_7s_exposure_audit']['memory32_equal_exposure'].get('passed')}`",
        f"- independent field models passed: `{all(item.get('passed') for item in report['field_only_32'].values())}`",
        f"- allow 850 training: `{report['recommendation']['allow_850_training']}`",
        "",
        "本轮未读取90条Dev、冻结60、1029 gold，未调用32B/GPT/外部LLM，未训练850条。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    deployment = [
        "# Step 5.7-T 部署方式对照",
        "",
        "| 方案 | 训练对象 | 当前结论 |",
        "|---|---|---|",
        "| 单一完整checkpoint | 五字段共享PP-UIE | 等曝光32条仍需验证/若失败不推荐直接850训练 |",
        "| 字段分组checkpoint | trigger+span；subject+time+object | 需以本轮真实结果决定 |",
        "| 五个独立checkpoint | 每字段一个PP-UIE | 精度隔离性最好，但存储和切换成本最高 |",
        "| LoRA/Adapter | 冻结基础模型，多字段适配器 | 本轮未执行完整LoRA训练，不能宣称已验证 |",
        "",
        "本轮只在固定32条记忆集进行验证，不使用90条Dev或冻结测试集选型。",
    ]
    DEPLOYMENT_MD.write_text("\n".join(deployment) + "\n", encoding="utf-8")
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(["key", "value"])
    for key in ["status", "step5_7s_exposure_audit", "field_only_32", "group_experiments", "protection", "recommendation"]:
        ws.append([key, json.dumps(report.get(key, {}), ensure_ascii=False)])
    for field, item in report["field_only_32"].items():
        sheet = wb.create_sheet(field[:31])
        sheet.append(["field", "passed", "steps", "updates_per_pair", "free_exact", "teacher_accuracy", "parser", "checkpoint_sha256"])
        sheet.append([
            field,
            item.get("passed"),
            item.get("optimizer_steps"),
            item.get("updates_per_pair"),
            item.get("free_exact_by_field", {}).get(field),
            item.get("teacher_accuracy_by_field", {}).get(field),
            item.get("parser_preservation_rate"),
            item.get("checkpoint_sha256"),
        ])
    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
        sheet.column_dimensions["A"].width = 32
        sheet.column_dimensions["B"].width = 120
    DETAILS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DETAILS_XLSX)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equal-exposure-steps", type=int, default=6000)
    parser.add_argument("--field-steps", type=int, default=3000)
    parser.add_argument("--group-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--skip-groups", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in [REPORT_JSON, REPORT_MD, DETAILS_XLSX, DEPLOYMENT_MD] if p.exists()]
    if existing and not args.skip_groups:
        # This run is designed to be resumable only through explicit existing
        # experiment artifacts; never silently overwrite a complete report.
        pass
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7t_ppuie_equal_exposure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    protected = [
        *FORMAL_PROTECTED,
        MODEL_DIR / "model_state.pdparams",
        ROOT / "outputs/task1_step5_7s_ppuie_multisample_memory/SHA256SUMS.txt",
        ROOT / "outputs/task1_step5_7s_ppuie_multisample_memory/fixed_memory8_rows.jsonl",
        ROOT / "outputs/task1_step5_7s_ppuie_multisample_memory/fixed_memory32_rows_metadata_only.jsonl",
        ROOT / "reports/task1_step5_7s_ppuie_multisample_memory.json",
        TRAIN850,
    ]
    before = sha_snapshot(protected)
    write_sha_txt(backup_dir / "protected_sha256_before.txt", before)
    write_sha_txt(OUT_DIR / "protected_sha256_before.txt", before)

    rows = read_jsonl(TRAIN850)
    memory8 = pick_memory(rows, 8)
    memory32 = pick_memory(rows, 32)
    s_report_path = ROOT / "reports/task1_step5_7s_ppuie_multisample_memory.json"
    s_report = json.loads(s_report_path.read_text(encoding="utf-8"))
    tokenizer, _model = load_base(args.dtype)
    stats8 = field_stats(tokenizer, memory8)
    stats32 = field_stats(tokenizer, memory32)
    release_model(_model)
    exposure8 = summarize_s_exposure(s_report, len(memory8))
    exposure8["field_target_token_totals"] = {f: stats8["field"][f]["target_token_total"] for f in FIELDS}
    exposure8["supervised_answer_tokens_per_field"] = exposure8["field_target_token_totals"]
    exposure8["training_source_total_steps_from_step5_7s"] = {
        "joint_curriculum8_stage_steps": [350, 600, 400, 400, 450],
        "total_optimizer_steps": 2200,
    }
    memory32_s = s_report.get("memory32", {})
    original_counts = memory32_s.get("sampling_counts_total", {})
    original_field_updates = Counter()
    for key, value in original_counts.items():
        if "::" in key:
            original_field_updates[key.rsplit("::", 1)[1]] += int(value)
    exposure32_original = {
        "sample_count": len(memory32),
        "qa_pair_count": len(original_counts),
        "optimizer_steps": sum(int(v) for v in original_counts.values()),
        "updates_per_pair": round(sum(int(v) for v in original_counts.values()) / max(1, len(original_counts)), 6),
        "updates_per_field": dict(original_field_updates),
        "field_pair_counts": dict(Counter(key.rsplit("::", 1)[1] for key in original_counts if "::" in key)),
        "field_updates": dict(original_field_updates),
        "pair_update_min": min((int(v) for v in original_counts.values()), default=0),
        "pair_update_max": max((int(v) for v in original_counts.values()), default=0),
        "pair_update_mean": round(sum(int(v) for v in original_counts.values()) / max(1, len(original_counts)), 6),
        "pair_updates": original_counts,
        "stage_steps": [
            {"added_field": stage.get("added_field"), "steps": stage.get("steps")}
            for stage in memory32_s.get("stages", [])
        ],
    }
    exposure32_original["field_target_token_totals"] = {f: stats32["field"][f]["target_token_total"] for f in FIELDS}
    exposure32_original["supervised_answer_tokens_per_field"] = exposure32_original["field_target_token_totals"]
    s_field_exposure = summarize_s_field_exposure(s_report, len(memory8))
    exposure32_equal = exposure_summary(memory32, args.equal_exposure_steps)
    exposure32_equal["field_target_token_totals"] = {f: stats32["field"][f]["target_token_total"] for f in FIELDS}
    exposure32_equal["supervised_answer_tokens_per_field"] = exposure32_equal["field_target_token_totals"]
    # The requested same-exposure target is 55 updates per pair, or 8,800
    # updates for 160 pairs. The experiment is capped at 6,000 updates.
    exposure32_equal["target_updates_per_pair_from_memory8"] = exposure8["updates_per_pair"]
    exposure32_equal["target_steps_for_same_updates_per_pair"] = int(round(exposure8["updates_per_pair"] * exposure32_equal["qa_pair_count"]))
    equal_steps = min(args.equal_exposure_steps, exposure32_equal["target_steps_for_same_updates_per_pair"])
    equal_steps = min(equal_steps, 6000)
    exposure32_equal["actual_steps_budgeted"] = equal_steps
    exposure32_equal["budget_insufficient"] = equal_steps < exposure32_equal["target_steps_for_same_updates_per_pair"]
    exposure32_equal["field_exposure_source"] = s_field_exposure

    # Run an equal-exposure full joint curriculum from the base model.
    equal_report = train_balanced(memory32, "joint_equal_exposure_32", [FIELDS], equal_steps, args.learning_rate, args.eval_every, args.dtype, early_stop_rounds=2)
    equal_report["budget_insufficient"] = exposure32_equal["budget_insufficient"]
    exposure32_equal.update(equal_report)

    field_results: dict[str, Any] = {}
    for field in FIELDS:
        source = s_field_exposure.get(field, {})
        field_steps = min(
            args.field_steps,
            int(source.get("target_steps_for_32", args.field_steps)),
        )
        result = train_balanced(
            memory32,
            f"field_only32_{field}",
            [[field]],
            field_steps,
            args.learning_rate,
            args.eval_every,
            args.dtype,
            eval_fields=[field],
            early_stop_rounds=2,
        )
        result["updates_per_pair"] = field_steps / len(memory32)
        result["field_target_token_total"] = stats32["field"][field]["target_token_total"]
        result["target_steps_from_step5_7s"] = source.get("target_steps_for_32")
        field_results[field] = result

    group_results: dict[str, Any] = {}
    all_independent_passed = all(item.get("passed") for item in field_results.values())
    if all_independent_passed and not args.skip_groups:
        group_defs = {
            "G1_trigger_span_subject_time_object": [["trigger", "event_span"], ["subject_company", "event_time", "object_target"]],
            "G2_trigger_span_subject_object_time": [["trigger", "event_span"], ["subject_company", "object_target"], ["event_time"]],
            "G3_five_independent_assembly": [[field] for field in FIELDS],
        }
        group_steps = min(args.group_steps, 2800)
        for name, groups in group_defs.items():
            if name == "G3_five_independent_assembly":
                group_results[name] = {
                    "run": True,
                    "note": "The five independent field results above are the G3 component measurements; no oracle text is substituted in the final comparison.",
                    "passed": all_independent_passed,
                    "components": {field: {"checkpoint": field_results[field].get("checkpoint"), "checkpoint_sha256": field_results[field].get("checkpoint_sha256"), "passed": field_results[field].get("passed")} for field in FIELDS},
                }
            else:
                group_results[name] = train_balanced(memory32, name, groups, group_steps, args.learning_rate, args.eval_every, args.dtype, early_stop_rounds=2)
    else:
        group_results["skipped"] = {"run": False, "reason": "independent_field_gate_not_passed"}

    after = sha_snapshot(protected)
    protection = {
        "protected_sha_match": before == after,
        "formal_task1_task2_modified": before != after,
        "dev90_labels_read": False,
        "independent_frozen60_labels_read": False,
        "gold1029_labels_read": False,
        "train850_full_training_run": False,
        "local_32b_calls": 0,
        "gpt_or_external_llm_calls": 0,
        "base_model_sha_before": before.get(rel(MODEL_DIR / "model_state.pdparams")),
        "base_model_sha_after": after.get(rel(MODEL_DIR / "model_state.pdparams")),
    }
    report = make_report(before, after, exposure8, exposure32_original, exposure32_equal, field_results, group_results, protection, backup_dir)
    write_outputs(report)
    write_sha_txt(OUT_DIR / "protected_sha256_after.txt", after)
    write_sha_txt(backup_dir / "protected_sha256_after.txt", after)
    write_json(backup_dir / "run_manifest.json", report)
    write_sha_txt(OUT_DIR / "SHA256SUMS.txt", sha_snapshot([REPORT_JSON, REPORT_MD, DETAILS_XLSX, DEPLOYMENT_MD]))
    print(json.dumps({"status": report["status"], "equal_exposure_passed": equal_report.get("passed"), "independent_fields_passed": all_independent_passed, "allow_850_training": report["recommendation"]["allow_850_training"], "protected_sha_match": protection["protected_sha_match"], "backup_dir": str(backup_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
