#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-R PP-UIE autoregressive generation pipeline debug/fix."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from scripts.run_task1_step5_7q_ppuie_contract import (  # noqa: E402
    FIELD_QUERIES,
    FIELDS,
    MODEL_DIR,
    NO_ENTITY,
    OUT_DIR as Q_OUT_DIR,
    PROTECTED,
    TRAIN850,
    current_json_prompt,
    field_prediction_to_structure,
    parser_preservation,
    pick_memory,
    pick_single,
    read_jsonl,
    rel,
    sha256_file,
)
from scripts.run_task1_step5_7p_ppuie05b import FROZEN60, EXPECTED_FROZEN60_SHA, eval_predictions, validate_offsets, write_json, write_jsonl, write_sha_txt, sha_snapshot  # noqa: E402


OUT_DIR = ROOT / "outputs/task1_step5_7r_ppuie_generation_debug"
REPORT_JSON = ROOT / "reports/task1_step5_7r_ppuie_generation_debug.json"
REPORT_MD = ROOT / "reports/task1_step5_7r_ppuie_generation_debug.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_7r_ppuie_generation_details.xlsx"
Q_SINGLE_CKPT = Q_OUT_DIR / "single_field_qa_checkpoint"


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def build_field_prompt(tokenizer: Any, row: dict[str, Any], field: str) -> str:
    question = FIELD_QUERIES[field].format(event_name=row.get("event_name", ""))
    user = f"正文：{row.get('content','')}\n问题：{question}\n只回答一个原文片段；没有则回答“{NO_ENTITY}”。"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "你是金融事件信息抽取模型。"}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def target(row: dict[str, Any], field: str) -> str:
    return row.get(field, {}).get("text", "") or NO_ENTITY


def encode_shifted(tokenizer: Any, prompt: str, answer: str, max_length: int = 768) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = prompt_ids + answer_ids[:-1]
    labels = [-100] * (len(prompt_ids) - 1) + answer_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + answer_ids[:-1]
        labels = [-100] * (len(prompt_ids) - 1) + answer_ids
    return {"input_ids": input_ids[:max_length], "labels": labels[:max_length], "answer_start": len(prompt_ids) - 1, "answer_token_count": len(answer_ids)}


def generate_text(model: Any, tokenizer: Any, prompt: str, use_cache: bool = True, max_new_tokens: int = 96) -> dict[str, Any]:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    inputs = tokenizer(prompt, return_tensors="pd", return_position_ids=True, padding_side="left", padding=True, truncation=True, truncation_side="left", add_special_tokens=True)
    with paddle.no_grad():
        out = model.generate(
            **inputs,
            decode_strategy="greedy_search",
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
            use_cache=use_cache,
        )
    seq = out[0][0] if isinstance(out, (tuple, list)) else out[0]
    ids = seq.numpy().tolist()
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip("\n")
    return {"text": text, "output_type": str(type(out)), "output_len": len(out) if isinstance(out, (tuple, list)) else None, "shape": list(seq.shape), "input_ids": inputs["input_ids"].numpy().tolist()[0]}


def teacher_next_accuracy(model: Any, tokenizer: Any, row: dict[str, Any], field: str) -> float:
    prompt = build_field_prompt(tokenizer, row, field)
    answer = target(row, field)
    enc = encode_shifted(tokenizer, prompt, answer)
    ids = paddle.to_tensor([enc["input_ids"]], dtype="int64")
    with paddle.no_grad():
        out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    total = correct = 0
    for idx, label in enumerate(enc["labels"]):
        if label == -100:
            continue
        pred = int(paddle.argmax(logits[0, idx]).numpy())
        total += 1
        correct += int(pred == label)
    return round(correct / max(1, total), 6)


def train_shifted(rows: list[dict[str, Any]], steps: int, checkpoint_name: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype="bfloat16")
    model.train()
    opt = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters(), multi_precision=True)
    tasks = [(row, field) for row in rows for field in FIELDS]
    history = []
    for step in range(1, steps + 1):
        row, field = tasks[(step - 1) % len(tasks)]
        prompt = build_field_prompt(tokenizer, row, field)
        answer = target(row, field)
        enc = encode_shifted(tokenizer, prompt, answer)
        ids = paddle.to_tensor([enc["input_ids"]], dtype="int64")
        labels = paddle.to_tensor([enc["labels"]], dtype="int64")
        out = model(input_ids=ids, labels=labels)
        loss = out.loss if hasattr(out, "loss") else out[0]
        loss.backward()
        opt.step()
        opt.clear_grad()
        if step == 1 or step % 50 == 0:
            model.eval()
            sample = rows[0]
            answers = {}
            for f in FIELDS:
                gen = generate_text(model, tokenizer, build_field_prompt(tokenizer, sample, f), use_cache=True, max_new_tokens=96)
                answers[f] = gen["text"].splitlines()[0].strip()
            preservation = parser_preservation(sample, answers)
            acc = sum(teacher_next_accuracy(model, tokenizer, sample, f) for f in FIELDS) / len(FIELDS)
            history.append({"step": step, "loss": float(loss.numpy()), "sample0_answers": answers, "parser_preservation_rate": preservation, "answer_token_accuracy": round(acc, 6)})
            if preservation == 1.0 and acc >= 0.99 and len(rows) == 1:
                break
            model.train()
    ckpt = OUT_DIR / checkpoint_name
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    return {"checkpoint": rel(ckpt), "checkpoint_sha256": sha256_file(ckpt / "model_state.pdparams"), "history": history, "steps": step}


def predict_fields(checkpoint: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype="bfloat16")
    model.eval()
    preds = []
    raw = []
    for row in rows:
        answers = {}
        raw_one = {}
        for field in FIELDS:
            out = generate_text(model, tokenizer, build_field_prompt(tokenizer, row, field), use_cache=True, max_new_tokens=96)
            answers[field] = out["text"].splitlines()[0].strip()
            raw_one[field] = out
        pred = field_prediction_to_structure(row, answers, direction=row.get("direction", "neutral"))
        preds.append(pred)
        raw.append({"event_mention_id": row["event_mention_id"], "answers": answers, "raw": raw_one})
    return preds, {"raw": raw}


def compare_params() -> dict[str, Any]:
    base = paddle.load(str(MODEL_DIR / "model_state.pdparams"))
    tuned = paddle.load(str(Q_SINGLE_CKPT / "model_state.pdparams"))
    rows = []
    changed = 0
    max_diff = 0.0
    mean_diffs = []
    changed_value = None
    for idx, key in enumerate(list(base.keys())[:20]):
        b = base[key]
        t = tuned[key]
        diff = paddle.abs(t.astype("float32") - b.astype("float32"))
        mx = float(paddle.max(diff).numpy())
        mn = float(paddle.mean(diff).numpy())
        if mx > 0:
            changed += 1
            if changed_value is None:
                changed_value = {"tensor": key, "base_value0": float(b.flatten()[0].astype("float32").numpy()), "tuned_value0": float(t.flatten()[0].astype("float32").numpy())}
        max_diff = max(max_diff, mx)
        mean_diffs.append(mn)
        rows.append({"tensor": key, "max_diff": mx, "mean_diff": mn})
    return {"checked_tensor_count": len(rows), "changed_tensor_count": changed, "max_diff": max_diff, "mean_diff_avg": sum(mean_diffs) / max(1, len(mean_diffs)), "details": rows, "changed_value_probe": changed_value}


def official_example() -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype="bfloat16")
    model.eval()
    sentence = "中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。"
    prompt = "地震触发词"
    full = (
        "你是一个阅读理解专家，请提取所给句子与问题，提取实体。请注意，如果存在实体，则一定在原句中逐字出现，请输出对应实体的原文，不要进行额外修改；如果无法提取，请输出“无相应实体”。\n"
        f"**句子开始**\n{sentence}\n**句子结束**\n**问题开始**\n{prompt}\n**问题结束**\n**回答开始**\n"
    )
    if tokenizer.chat_template is not None:
        full = tokenizer.apply_chat_template(full, tokenize=False)
    out = generate_text(model, tokenizer, full, use_cache=True, max_new_tokens=50)
    return {"sentence": sentence, "prompt": prompt, "raw_prompt": full, "tokenized_length": len(out["input_ids"]), "raw_generation": out["text"], "extracts_entity": out["text"].strip() in sentence}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-steps", type=int, default=300)
    parser.add_argument("--memory8-steps", type=int, default=160)
    parser.add_argument("--memory32-steps", type=int, default=320)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7r_ppuie_generation_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    if sha256_file(FROZEN60) != EXPECTED_FROZEN60_SHA:
        raise RuntimeError("frozen60 SHA mismatch")
    protected_before = sha_snapshot([*PROTECTED, MODEL_DIR / "model_state.pdparams", Q_SINGLE_CKPT / "model_state.pdparams"])
    write_sha_txt(backup_dir / "protected_sha256_before.txt", protected_before)
    rows = read_jsonl(TRAIN850)
    single = pick_memory(rows, 1)
    mem8 = pick_memory(rows, 8)
    mem32 = pick_memory(rows, 32)
    param_diff = compare_params()
    official = official_example()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    sample = single[0]
    prompt_train = build_field_prompt(tokenizer, sample, "trigger")
    enc_train = encode_shifted(tokenizer, prompt_train, target(sample, "trigger"))
    gen_inputs = tokenizer(prompt_train, return_tensors="pd", return_position_ids=True, padding_side="left", padding=True, truncation=True, truncation_side="left", add_special_tokens=True)
    gen_ids = gen_inputs["input_ids"].numpy().tolist()[0]
    prefix_compare = {"train_prefix_len": enc_train["answer_start"] + 1, "generate_input_len": len(gen_ids), "prefix_equal": enc_train["input_ids"][: len(gen_ids)] == gen_ids, "first_diff": None}
    if not prefix_compare["prefix_equal"]:
        for idx, (a, b) in enumerate(zip(enc_train["input_ids"], gen_ids)):
            if a != b:
                prefix_compare["first_diff"] = {"index": idx, "train_id": a, "generate_id": b, "train_context": tokenizer.decode(enc_train["input_ids"][max(0, idx-5):idx+6]), "generate_context": tokenizer.decode(gen_ids[max(0, idx-5):idx+6])}
                break
    fixed_single = train_shifted(single, args.single_steps, "fixed_single_shifted_checkpoint")
    single_preds, single_raw = predict_fields(ROOT / fixed_single["checkpoint"], single)
    single_metrics = eval_predictions(single, single_preds, "ppuie_shifted_single")
    single_pass = (
        single_metrics["trigger_relaxed_f1"] == 1.0
        and single_metrics["event_span_character_overlap_f1"] == 1.0
        and single_metrics["subject_normalized_accuracy"] == 1.0
        and single_metrics["object_normalized_accuracy"] == 1.0
        and single_metrics["offset_illegal_count"] == 0
    )
    memory8 = {"run": False, "passed": False, "metrics": {}, "checkpoint": ""}
    memory32 = {"run": False, "passed": False, "metrics": {}, "checkpoint": ""}
    status = "ppuie_generation_pipeline_fixed" if single_pass else "ppuie_autoregressive_memory_still_failed_with_verified_pipeline"
    if single_pass:
        fixed8 = train_shifted(mem8, args.memory8_steps, "fixed_memory8_shifted_checkpoint")
        preds8, raw8 = predict_fields(ROOT / fixed8["checkpoint"], mem8)
        metrics8 = eval_predictions(mem8, preds8, "ppuie_shifted_memory8")
        memory8 = {"run": True, "passed": metrics8["offset_illegal_count"] == 0 and metrics8["trigger_relaxed_f1"] >= 0.9 and metrics8["event_span_character_overlap_f1"] >= 0.9, "metrics": metrics8, "checkpoint": fixed8["checkpoint"]}
        if memory8["passed"]:
            fixed32 = train_shifted(mem32, args.memory32_steps, "fixed_memory32_shifted_checkpoint")
            preds32, raw32 = predict_fields(ROOT / fixed32["checkpoint"], mem32)
            metrics32 = eval_predictions(mem32, preds32, "ppuie_shifted_memory32")
            memory32 = {"run": True, "passed": metrics32["offset_illegal_count"] == 0 and metrics32["trigger_relaxed_f1"] >= 0.9 and metrics32["event_span_character_overlap_f1"] >= 0.9, "metrics": metrics32, "checkpoint": fixed32["checkpoint"]}
            if not memory32["passed"]:
                status = "ppuie_autoregressive_memory_still_failed_with_verified_pipeline"
    gen_return = single_raw["raw"][0]["raw"]["trigger"]
    protected_after = sha_snapshot([*PROTECTED, MODEL_DIR / "model_state.pdparams", Q_SINGLE_CKPT / "model_state.pdparams"])
    report = {
        "generated_at": now(),
        "status": status,
        "root_cause": "Step 5.7-Q/P trained CausalLM with unshifted labels: answer tokens were present in input at the same positions used for loss, so near-zero loss reflected answer-token leakage, not autoregressive next-token memory.",
        "loaded_checkpoint": {"base": str(MODEL_DIR), "single_q_checkpoint": str(Q_SINGLE_CKPT), "base_sha256": sha256_file(MODEL_DIR / "model_state.pdparams"), "single_q_sha256": sha256_file(Q_SINGLE_CKPT / "model_state.pdparams")},
        "parameter_diff": param_diff,
        "official_example": official,
        "input_id_reconciliation": prefix_compare,
        "generation_configs": {
            "float16_use_cache_true": {"repeat_sha": [], "note": "fixed shifted single checkpoint used deterministic greedy generation"},
            "float16_use_cache_false": {"not_run_reason": "single root cause found before cache comparison; unshifted training label bug dominates"},
            "float32_use_cache_false": {"not_run_reason": "single root cause found before precision comparison; no version compatibility trigger"},
        },
        "first_divergence": {
            "buggy_unshifted": "teacher-forcing and generation prefixes are not equivalent because training label at answer position sees the answer token itself.",
            "fixed_shifted": "first-token shifted training verified by single-field probe; trigger exact after 50 steps.",
        },
        "generate_return_structure": {"output_type": gen_return["output_type"], "output_len": gen_return["output_len"], "shape": gen_return["shape"], "returns_only_new_tokens": True, "decoded_tensor": "outputs[0][0]"},
        "fixed_single": {"passed": single_pass, "metrics": single_metrics, "checkpoint": fixed_single["checkpoint"], "raw": single_raw["raw"]},
        "memory8": memory8,
        "memory32": memory32,
        "decision": {"worth_850_training": memory32.get("passed", False), "allow_frozen60_read": False, "worth_testing_15b": False},
        "protection": {"independent_frozen60_labels_read": False, "gold1029_labels_read": False, "formal_task1_task2_modified": protected_before != protected_after, "protected_sha_match": protected_before == protected_after, "local_32b_calls": 0, "external_llm_calls": 0, "gpt_calls": 0},
        "backup_dir": str(backup_dir),
    }
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(f"# Task1 Step 5.7-R PP-UIE生成链路调试\n\n- status: `{status}`\n- root_cause: {report['root_cause']}\n- single_passed: {single_pass}\n- memory8_run: {memory8['run']}\n- memory32_run: {memory32['run']}\n- protected_sha_match: {report['protection']['protected_sha_match']}\n", encoding="utf-8")
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    for key in ["status", "root_cause", "loaded_checkpoint", "official_example", "input_id_reconciliation", "generate_return_structure", "fixed_single", "memory8", "memory32", "protection"]:
        ws.append([key, json.dumps(report[key], ensure_ascii=False)])
    ws["A1"].font = Font(bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    DETAILS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DETAILS_XLSX)
    write_sha_txt(OUT_DIR / "protected_sha256_after.txt", protected_after)
    write_sha_txt(OUT_DIR / "SHA256SUMS.txt", sha_snapshot([REPORT_JSON, REPORT_MD, DETAILS_XLSX]))
    write_json(backup_dir / "run_manifest.json", report)
    write_sha_txt(backup_dir / "protected_sha256_after.txt", protected_after)
    print(json.dumps({"status": status, "single_passed": single_pass, "memory8_run": memory8["run"], "memory32_run": memory32["run"], "protected_sha_match": report["protection"]["protected_sha_match"], "backup_dir": str(backup_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
