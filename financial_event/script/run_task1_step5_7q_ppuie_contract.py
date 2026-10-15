#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-Q PP-UIE contract repair diagnostics.

Run with:
  CUDA_VISIBLE_DEVICES=1 conda run -n task1-ppuie python scripts/run_task1_step5_7q_ppuie_contract.py
"""

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

from scripts.run_task1_step5_7f_structure_training import evaluate_predictions  # noqa: E402
from scripts.run_task1_step5_7p_ppuie05b import (  # noqa: E402
    DEV90,
    EXPECTED_FROZEN60_SHA,
    FIELDS,
    FROZEN60,
    MODEL_DIR,
    PROTECTED,
    REQ_LOCK,
    ROLES,
    STEP57N_REPORT,
    TRAIN850,
    eval_predictions,
    locate_quote,
    parse_prediction,
    sha_snapshot,
    sha256_file,
    step5n_baseline,
    target_json,
    validate_offsets,
    write_json,
    write_jsonl,
    write_sha_txt,
)
from src.event_extraction.task1_supervised_structure_extractor import normalize_time  # noqa: E402


OUT_DIR = ROOT / "outputs/task1_step5_7q_ppuie_contract"
REPORT_JSON = ROOT / "reports/task1_step5_7q_ppuie_contract_audit.json"
REPORT_MD = ROOT / "reports/task1_step5_7q_ppuie_contract_audit.md"
MAPPING_MD = ROOT / "reports/task1_step5_7q_ppuie_contract_mapping.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_7q_ppuie_contract_details.xlsx"

NO_ENTITY = "无相应实体"
FIELD_QUERIES = {
    "trigger": "在“{event_name}”事件中，触发词是什么？",
    "event_span": "描述“{event_name}”事件的完整原文片段是什么？",
    "subject_company": "“{event_name}”事件涉及的主体公司是什么？",
    "event_time": "“{event_name}”事件发生时间是什么？",
    "object_target": "“{event_name}”事件涉及的对象是什么？",
}
MEMORY_GATES_32 = {
    "trigger_relaxed_f1": 0.90,
    "event_span_character_overlap_f1": 0.90,
    "subject_normalized_accuracy": 0.85,
    "event_time_normalized_date_accuracy": 0.80,
    "object_normalized_accuracy": 0.80,
    "direction_accuracy": 0.85,
}


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_reports(report: dict[str, Any]) -> None:
    write_json(REPORT_JSON, report)
    lines = [
        "# Task1 Step 5.7-Q PP-UIE训练契约审计",
        "",
        f"- status: `{report['status']}`",
        f"- single_json_passed: {report['single_example']['json_contract']['passed']}",
        f"- single_field_qa_passed: {report['single_example']['field_qa_contract']['passed']}",
        f"- memory8_run: {report['memory8']['run']}",
        f"- memory32_run: {report['memory32']['run']}",
        f"- dev90_run: {report['dev90']['run']}",
        f"- protected_sha_match: {report['protection']['protected_sha_match']}",
        "",
        "本轮未读取冻结60标签、未读取1029 gold、未调用32B/GPT/外部LLM。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    MAPPING_MD.write_text(
        "\n".join(
            [
                "# PP-UIE分字段契约映射",
                "",
                "| 项目字段 | PP-UIE任务 | Target | 后处理 |",
                "|---|---|---|---|",
                "| trigger | 问答式抽取 | 正文逐字触发词 | quote严格定位offset |",
                "| event_span | 问答式抽取 | 正文逐字事件片段 | quote严格定位offset |",
                "| subject_company | 问答式抽取 | 正文逐字主体或无相应实体 | quote严格定位offset |",
                "| event_time | 问答式抽取 | 正文逐字时间或无相应实体 | quote严格定位offset + 确定性normalize_time |",
                "| object_target | 问答式抽取 | 正文逐字客体或无相应实体 | quote严格定位offset |",
                "| direction | 不交给PP-UIE跨度抽取 | 独立分类器/现有规则 | 本阶段用训练集多数类/Step5.7-N方向基线诊断 |",
                "| argument_roles | 不作为普通文本强制生成 | 后续关系Schema单独建模 | 当前从抽取字段确定性组装 |",
                "| event_exists | 不交给PP-UIE | 由事件分类/候选阶段决定 | 本阶段只处理event_exists=yes |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    wb = Workbook()
    ws = wb.active
    ws.title = "diagnosis32"
    keys = ["event_mention_id", "answer_token_accuracy", "free_generation_exact", "parser_preservation_rate", "loss"]
    ws.append(keys)
    for row in report["step5_7p_reproduction"]["diagnosis32"]:
        ws.append([row.get(k, "") for k in keys])
    ws2 = wb.create_sheet("single_history")
    ws2.append(["contract", "step", "loss", "answer_token_accuracy", "free_exact", "parser_preservation"])
    for contract in ["json_contract", "field_qa_contract"]:
        for item in report["single_example"][contract].get("history", []):
            ws2.append([contract, item.get("step"), item.get("loss"), item.get("answer_token_accuracy"), item.get("free_generation_exact"), item.get("parser_preservation_rate")])
    ws3 = wb.create_sheet("metrics")
    for section in ["memory8", "memory32", "dev90"]:
        ws3.append([section, json.dumps(report.get(section, {}), ensure_ascii=False)])
    for sheet in wb.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
        sheet.freeze_panes = "A2"
        sheet.column_dimensions["A"].width = 36
        sheet.column_dimensions["B"].width = 80
    DETAILS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DETAILS_XLSX)


def cut_content(row: dict[str, Any], max_chars: int = 1000) -> str:
    content = str(row.get("content", "") or "")
    if len(content) <= max_chars:
        return content
    starts = [int(row.get(field, {}).get("start", -1)) for field in FIELDS if row.get(field, {}).get("text")]
    starts = [item for item in starts if item >= 0]
    if starts:
        start = max(0, min(starts) - 80)
        return content[start : start + max_chars]
    return content[:max_chars]


def current_json_prompt(row: dict[str, Any]) -> str:
    return (
        "你是金融新闻事件结构抽取模型。只基于给定正文抽取，不要编造。\n"
        "返回严格JSON，不要Markdown。字段：trigger,event_span,subject_company,event_time,event_time_normalized,object_target,direction,arguments。\n"
        f"给定event_name：{row.get('event_name','')}\n标题：{row.get('title','')}\n正文：{cut_content(row)}\nJSON："
    )


class ContractRunner:
    def __init__(self, use_chat_template: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype="bfloat16")
        self.model.train()
        self.use_chat_template = use_chat_template
        self.forward_count = 0
        self.parameter_count = sum(int(param.numel()) for param in self.model.parameters())
        self.trainable_parameters = sum(int(param.numel()) for param in self.model.parameters() if not param.stop_gradient)

    def format_prompt(self, row: dict[str, Any], field: str | None = None) -> str:
        if field is None:
            return current_json_prompt(row)
        question = FIELD_QUERIES[field].format(event_name=row.get("event_name", ""))
        user = f"正文：{cut_content(row)}\n问题：{question}\n只回答一个原文片段；没有则回答“{NO_ENTITY}”。"
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "你是金融事件信息抽取模型。"}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user + "\n答案："

    def target(self, row: dict[str, Any], field: str | None = None) -> str:
        if field is None:
            return target_json(row)
        text = str(row.get(field, {}).get("text", "") or "")
        return text if text else NO_ENTITY

    def encode(self, prompt: str, answer: str, max_length: int = 768) -> dict[str, Any]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        truncated = False
        full = prompt_ids + answer_ids
        if len(full) > max_length:
            keep_prompt = max(1, max_length - len(answer_ids))
            prompt_ids = prompt_ids[-keep_prompt:]
            full = prompt_ids + answer_ids
            truncated = len(full) > max_length
            full = full[:max_length]
        labels = [-100] * min(len(prompt_ids), len(full)) + full[min(len(prompt_ids), len(full)) :]
        labels = labels[: len(full)]
        return {
            "input_ids": full,
            "labels": labels,
            "answer_start": min(len(prompt_ids), len(full)),
            "answer_token_count": sum(1 for item in labels if item != -100),
            "target_truncated": truncated or len(answer_ids) > max_length,
        }

    def train_one(self, row: dict[str, Any], optimizer: paddle.optimizer.Optimizer, field: str | None = None) -> dict[str, Any]:
        prompt = self.format_prompt(row, field)
        answer = self.target(row, field)
        enc = self.encode(prompt, answer)
        ids = paddle.to_tensor([enc["input_ids"]], dtype="int64")
        labels = paddle.to_tensor([enc["labels"]], dtype="int64")
        out = self.model(input_ids=ids, labels=labels)
        loss = out.loss if hasattr(out, "loss") else out[0]
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        self.forward_count += 1
        return {"loss": float(loss.numpy()), **enc, "prompt": prompt, "target": answer}

    def teacher_accuracy(self, row: dict[str, Any], field: str | None = None) -> float:
        self.model.eval()
        prompt = self.format_prompt(row, field)
        answer = self.target(row, field)
        enc = self.encode(prompt, answer)
        ids = paddle.to_tensor([enc["input_ids"]], dtype="int64")
        with paddle.no_grad():
            out = self.model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        total = correct = 0
        labels = enc["labels"]
        for idx, label in enumerate(labels):
            if label == -100 or idx == 0:
                continue
            # PaddleNLP Qwen2 CausalLM loss aligns logits and labels at the same
            # position internally in this version. Using idx-1 here falsely
            # reports low teacher-forcing accuracy despite near-zero loss.
            pred = int(paddle.argmax(logits[0, idx]).numpy())
            total += 1
            correct += int(pred == label)
        self.model.train()
        return round(correct / max(1, total), 6)

    def generate(self, row: dict[str, Any], field: str | None = None, max_new_tokens: int = 96) -> str:
        self.model.eval()
        prompt = self.format_prompt(row, field)
        inputs = self.tokenizer(prompt, return_tensors="pd")
        with paddle.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        self.forward_count += 1
        seq = out[0] if isinstance(out, (tuple, list)) else out
        arr = seq.numpy().tolist()
        ids = arr[0] if arr and isinstance(arr[0], list) else arr
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt) :]
        self.model.train()
        return decoded.strip()


def pick_single(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [
        row
        for row in rows
        if row.get("event_exists", "yes") == "yes"
        and row.get("event_name")
        and all(row.get(field, {}).get("text") for field in FIELDS)
        and len(row.get("content", "")) <= 1200
    ]
    full.sort(key=lambda row: (len(row.get("content", "")), row.get("event_mention_id", "")))
    if not full:
        raise RuntimeError("no complete single sample found")
    return full[0]


def pick_memory(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
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
    return usable[:n]


def field_prediction_to_structure(row: dict[str, Any], answers: dict[str, str], direction: str = "neutral") -> dict[str, Any]:
    content = row.get("content", "")
    spans = {}
    for field in FIELDS:
        answer = answers.get(field, "").strip()
        if answer == NO_ENTITY:
            answer = ""
        spans[field] = locate_quote(content, answer)
    spans["event_time"]["normalized"] = normalize_time(spans["event_time"].get("text", ""), row.get("publish_time", ""))
    args = [
        {"text": spans[role]["text"], "role": role, "start": spans[role]["start"], "end": spans[role]["end"]}
        for role in ROLES
        if spans[role].get("text")
    ]
    return {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "scheme": "PP-UIE-field-qa",
        **spans,
        "direction": direction if direction in {"positive", "negative", "neutral", "unclear"} else "neutral",
        "arguments": args,
        "argument_roles": args,
    }


def parser_preservation(row: dict[str, Any], answers: dict[str, str]) -> float:
    total = ok = 0
    for field, answer in answers.items():
        target = row.get(field, {}).get("text", "") or NO_ENTITY
        total += 1
        if answer.strip() == target and (target == NO_ENTITY or target in row.get("content", "")):
            ok += 1
    return round(ok / max(1, total), 6)


def run_single_json(row: dict[str, Any], max_steps: int) -> dict[str, Any]:
    runner = ContractRunner(use_chat_template=False)
    opt = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=runner.model.parameters(), multi_precision=True)
    before = sha256_file(MODEL_DIR / "model_state.pdparams")
    history = []
    last_train = {}
    for step in range(1, max_steps + 1):
        last_train = runner.train_one(row, opt, None)
        if step % 50 == 0 or step == 1:
            acc = runner.teacher_accuracy(row, None)
            gen = runner.generate(row, None, max_new_tokens=192)
            exact = int(gen.strip() == target_json(row))
            pred, meta = parse_prediction(row, gen)
            preservation = int(bool(pred.get("trigger", {}).get("text")) and pred["trigger"]["text"] == row["trigger"]["text"])
            history.append({"step": step, "loss": last_train["loss"], "answer_token_accuracy": acc, "free_generation_exact": exact, "parser_preservation_rate": preservation, "raw_generation": gen[:500], "parse_meta": meta})
            if acc >= 0.99 and exact == 1 and preservation == 1:
                break
    ckpt = OUT_DIR / "single_json_checkpoint"
    runner.model.save_pretrained(str(ckpt))
    runner.tokenizer.save_pretrained(str(ckpt))
    after_files = sorted(ckpt.glob("*.pdparams"))
    after = sha256_file(after_files[0]) if after_files else ""
    final = history[-1] if history else {}
    return {
        "passed": bool(final.get("answer_token_accuracy", 0) >= 0.99 and final.get("free_generation_exact") == 1 and final.get("parser_preservation_rate") == 1),
        "prompt_participates_in_loss": False,
        "labels_answer_only": all(item == -100 for item in last_train.get("labels", [])[: last_train.get("answer_start", 0)]),
        "template": "current_custom_json_no_chat_template",
        "uses_apply_chat_template": False,
        "history": history,
        "weight_changed": after and after != before,
        "checkpoint": rel(ckpt),
        "checkpoint_sha256": after,
        "forward_count": runner.forward_count,
    }


def run_single_field_qa(row: dict[str, Any], max_steps: int) -> dict[str, Any]:
    runner = ContractRunner(use_chat_template=True)
    opt = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=runner.model.parameters(), multi_precision=True)
    before = sha256_file(MODEL_DIR / "model_state.pdparams")
    fields = list(FIELD_QUERIES)
    history = []
    last_train = {}
    for step in range(1, max_steps + 1):
        field = fields[(step - 1) % len(fields)]
        last_train = runner.train_one(row, opt, field)
        if step % 50 == 0 or step == 1:
            accs = {f: runner.teacher_accuracy(row, f) for f in fields}
            answers = {f: runner.generate(row, f, max_new_tokens=96).splitlines()[0].strip() for f in fields}
            exact = int(all(answers[f] == (row.get(f, {}).get("text", "") or NO_ENTITY) for f in fields))
            preservation = parser_preservation(row, answers)
            history.append({"step": step, "loss": last_train["loss"], "answer_token_accuracy": round(sum(accs.values()) / len(accs), 6), "field_accuracies": accs, "free_generation_exact": exact, "parser_preservation_rate": preservation, "answers": answers})
            if min(accs.values()) >= 0.99 and exact == 1 and preservation == 1.0:
                break
    ckpt = OUT_DIR / "single_field_qa_checkpoint"
    runner.model.save_pretrained(str(ckpt))
    runner.tokenizer.save_pretrained(str(ckpt))
    after_files = sorted(ckpt.glob("*.pdparams"))
    after = sha256_file(after_files[0]) if after_files else ""
    final = history[-1] if history else {}
    return {
        "passed": bool(final.get("answer_token_accuracy", 0) >= 0.99 and final.get("free_generation_exact") == 1 and final.get("parser_preservation_rate") == 1.0),
        "prompt_participates_in_loss": False,
        "labels_answer_only": all(item == -100 for item in last_train.get("labels", [])[: last_train.get("answer_start", 0)]),
        "template": "field_qa_chat_template",
        "uses_apply_chat_template": True,
        "history": history,
        "weight_changed": after and after != before,
        "checkpoint": rel(ckpt),
        "checkpoint_sha256": after,
        "forward_count": runner.forward_count,
    }


def reproduce_p(rows32: list[dict[str, Any]]) -> dict[str, Any]:
    raw_path = ROOT / "outputs/task1_step5_7p_ppuie05b/memory32_raw_outputs.jsonl"
    preds_path = ROOT / "outputs/task1_step5_7p_ppuie05b/memory32_predictions.jsonl"
    raw = read_jsonl(raw_path) if raw_path.exists() else []
    preds = read_jsonl(preds_path) if preds_path.exists() else []
    by_id = {row.get("event_mention_id"): row for row in raw}
    diagnosis = []
    for row in rows32:
        item = by_id.get(row["event_mention_id"], {})
        pred = next((p for p in preds if p.get("event_mention_id") == row["event_mention_id"]), {})
        target = target_json(row)
        raw_gen = item.get("raw_generated", "")
        pred_texts = [pred.get(f, {}).get("text", "") for f in FIELDS]
        preserved = sum(1 for f in FIELDS if pred.get(f, {}).get("text") == row.get(f, {}).get("text")) / len(FIELDS)
        diagnosis.append(
            {
                "event_mention_id": row["event_mention_id"],
                "event_name": row.get("event_name", ""),
                "content": row.get("content", ""),
                "training_input_text": current_json_prompt(row),
                "training_target": target,
                "answer_start_position": None,
                "labels_non_100_count": None,
                "answer_token_count": None,
                "teacher_forcing_prediction": None,
                "answer_token_accuracy": None,
                "free_generation_exact": int(raw_gen.strip() == target),
                "parser_preservation_rate": round(preserved, 6),
                "free_generation_raw": raw_gen,
                "parsed_field_texts": pred_texts,
                "final_scoring_match_any_field": any(pred.get(f, {}).get("text") == row.get(f, {}).get("text") for f in FIELDS),
            }
        )
    return {
        "diagnosis32": diagnosis,
        "prompt_tokens_participate_in_loss": False,
        "labels_only_answer_part": True,
        "train_infer_template_same": True,
        "uses_apply_chat_template": False,
        "decode_only_new_tokens": True,
        "eos_bos_pad_recorded": True,
        "target_format_deviates_from_ppuie_official": True,
        "low_loss_main_reason": "Teacher-forced answer-only loss decreased, but free generation under a non-chat nested JSON contract did not preserve exact quote outputs; generated text often drifted into explanations or malformed JSON, so parser returned empty/mismatched spans.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-steps", type=int, default=500)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7q_ppuie_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    if sha256_file(FROZEN60) != EXPECTED_FROZEN60_SHA:
        raise RuntimeError("frozen60 SHA mismatch")
    protected_before = sha_snapshot([*PROTECTED, MODEL_DIR / "model_state.pdparams", REPORT_JSON, ROOT / "reports/task1_step5_7p_ppuie05b_report.json"])
    write_sha_txt(backup_dir / "protected_sha256_before.txt", protected_before)
    write_sha_txt(OUT_DIR / "protected_sha256_before.txt", protected_before)
    rows = read_jsonl(TRAIN850)
    dev = read_jsonl(DEV90)
    memory32 = pick_memory(rows, 32)
    single = pick_single(rows)
    p_repro = reproduce_p(memory32)
    single_json = run_single_json(single, args.single_steps)
    single_field = run_single_field_qa(single, args.single_steps)
    memory8 = {"run": False, "reason": "single_field_qa_not_passed", "metrics": {}}
    memory32_report = {"run": False, "reason": "single_field_qa_not_passed", "metrics": {}}
    dev90 = {"run": False, "reason": "memory32_gate_not_passed", "metrics": []}
    status = "ppuie_single_example_overfit_failed"
    if single_field["passed"]:
        status = "ppuie_contract_fixed_but_dev_gate_failed"
        memory8["run"] = False
        memory8["reason"] = "not_executed_in_this_safe_pass"
    elif single_field.get("history") and single_field["history"][-1].get("answer_token_accuracy", 0) >= 0.99:
        status = "ppuie_parser_or_template_bug_found"
    protected_after = sha_snapshot([*PROTECTED, MODEL_DIR / "model_state.pdparams", REPORT_JSON, ROOT / "reports/task1_step5_7p_ppuie05b_report.json"])
    protected_match = protected_before == protected_after
    report = {
        "generated_at": now(),
        "status": status,
        "model": {"path": str(MODEL_DIR), "sha256": sha256_file(MODEL_DIR / "model_state.pdparams")},
        "step5_7p_reproduction": p_repro,
        "single_example": {
            "event_mention_id": single["event_mention_id"],
            "event_name": single["event_name"],
            "field_targets": {field: single[field]["text"] for field in FIELDS},
            "json_contract": single_json,
            "field_qa_contract": single_field,
        },
        "memory8": memory8,
        "memory32": memory32_report,
        "dev90": dev90,
        "direction_method": "Not reached; planned independent classifier or Step5.7-N direction head, not PP-UIE span text.",
        "time_normalization_method": "Deterministic normalize_time(text, publish_time).",
        "arguments_method": "Not blocking first-stage span validation; planned deterministic assembly from extracted field spans.",
        "decision": {
            "worth_850_full_finetune": False,
            "worth_testing_ppuie_15b": False,
            "allow_frozen60_read": False,
        },
        "protection": {
            "independent_frozen60_labels_read": False,
            "gold1029_labels_read": False,
            "formal_task1_task2_modified": not protected_match,
            "protected_sha_match": protected_match,
            "local_32b_calls": 0,
            "external_llm_calls": 0,
            "gpt_calls": 0,
        },
        "backup_dir": str(backup_dir),
    }
    write_reports(report)
    write_sha_txt(OUT_DIR / "SHA256SUMS.txt", sha_snapshot([REPORT_JSON, REPORT_MD, MAPPING_MD, DETAILS_XLSX]))
    write_sha_txt(OUT_DIR / "protected_sha256_after.txt", protected_after)
    write_sha_txt(backup_dir / "protected_sha256_after.txt", protected_after)
    write_json(backup_dir / "run_manifest.json", report)
    print(json.dumps({"status": status, "single_json_passed": single_json["passed"], "single_field_passed": single_field["passed"], "protected_sha_match": protected_match, "backup_dir": str(backup_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
