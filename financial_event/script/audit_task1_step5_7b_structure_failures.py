#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-B offline structure failure and offset-adapter audit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_5_candidate_v2 import protected_file_list, rel, sha256_file, write_json, write_jsonl, write_xlsx  # noqa: E402
from src.event_extraction.task1_conditional_structure_32b import evaluate_structure_predictions, read_jsonl, validate_confirmed_annotations  # noqa: E402
from src.event_extraction.task1_structure_offset_adapter import (  # noqa: E402
    adapt_raw_response,
    classify_text_locator,
    field_raw_offset,
    field_text,
    strict_json_loads,
)

EXPECTED_CONFIRMED_SHA = "414174fdef7b8aeb89edecdde4a85f43ac08d86093664cb4a6d4541732f1186c"
STEP57A_DIR = ROOT / "outputs/task1_step5_7a_conditional_structure_32b"
CONFIRMED = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"
OUT = ROOT / "outputs/task1_step5_7b_structure_failure_audit"
REPORT_JSON = ROOT / "reports/task1_step5_7b_structure_failure_audit.json"
REPORT_MD = ROOT / "reports/task1_step5_7b_structure_failure_audit.md"
REPORT_XLSX = ROOT / "reports/task1_step5_7b_structure_failure_audit.xlsx"


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        ROOT / "outputs/event_match_baseline/best_model",
        ROOT / "outputs/task1_limited_training_round1/model_epoch_2",
        ROOT / "outputs/task1_step5_6d_classifier_retraining/experiment_C/best_model",
        STEP57A_DIR,
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
        ["bash", "-lc", "ps -eo pid,ppid,stat,etime,cmd | rg -i 'vllm|DianJin-R1-32B|task1_step5_7a|task1_step5_7b|train_task1' | rg -v 'rg -i' || true\nss -ltnp '( sport = :8000 )' || true"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    suspicious = []
    current = os.getpid()
    for line in proc.stdout.splitlines():
        parts = line.split(None, 4)
        if parts and parts[0].isdigit() and int(parts[0]) == current:
            continue
        if "audit_task1_step5_7b_structure_failures.py" in line:
            continue
        lower = line.lower()
        if any(token in lower for token in ["vllm", "dianjin-r1-32b", "train_task1"]):
            suspicious.append(line.strip())
    return {"status": "ok" if not suspicious else "suspicious_processes_found", "suspicious": suspicious, "raw": proc.stdout.splitlines()}


def event_names(row: dict[str, Any]) -> list[str]:
    return [part.strip() for part in str(row.get("gold_event_names_for_context_after_classification_eval", "")).split("|") if part.strip()]


def target_event_name(row: dict[str, Any]) -> str:
    names = event_names(row)
    return names[0] if names else ""


def multi_event_alignment_decision(row: dict[str, Any], received: str) -> tuple[str, str]:
    aid = str(row["article_id"])
    span = str(row.get("annotate_event_span", ""))
    first = received
    # These are audit decisions over six fixed diagnostic rows.  They do not
    # modify labels and are only used to flag contract risk.
    if aid == "NEWS_MULTISOURCE_D-1120797906500":
        return "wrong_event_selected", "结构span明确围绕“技术突破与生态繁荣”，更贴近第二个事件“关键技术突破”，首个“产能布局调整”不适配。"
    if aid in {"NEWS_MULTISOURCE_D-1126130257189", "NEWS_MULTISOURCE_D-1127695210485"}:
        return "ambiguous_multi_event", "结构span同时包含白皮书/技术演进和中试基地/行动启动，首个事件与第二个事件均可解释，当前单条annotation缺少mention级event_name。"
    return "first_event_alignment_correct", f"结构trigger/span与首个事件“{first}”语义一致，未发现首个事件选择导致的明显错配。"


def audit_multi_event(rows: list[dict[str, Any]], split: dict[str, Any]) -> list[dict[str, Any]]:
    split_by_id = {aid: "prompt_dev" for aid in split["prompt_dev_article_ids"]}
    split_by_id.update({aid: "frozen_eval" for aid in split["frozen_eval_article_ids"]})
    out = []
    for row in rows:
        names = event_names(row)
        if len(names) <= 1:
            continue
        received = target_event_name(row)
        category, reason = multi_event_alignment_decision(row, received)
        out.append(
            {
                "article_id": str(row["article_id"]),
                "split": split_by_id.get(str(row["article_id"]), ""),
                "all_gold_event_names": "|".join(names),
                "model_received_event_name": received,
                "annotate_trigger": row.get("annotate_trigger", ""),
                "annotate_event_span": str(row.get("annotate_event_span", ""))[:260],
                "alignment_category": category,
                "reason": reason,
            }
        )
    return out


def raw_rows(split: str) -> list[dict[str, Any]]:
    return read_jsonl(STEP57A_DIR / f"{split}_raw_responses.jsonl")


def prediction_rows(split: str) -> list[dict[str, Any]]:
    return read_jsonl(STEP57A_DIR / f"{split}_results.jsonl")


def primary_failure_category(tags: list[str]) -> str:
    order = [
        "text_unresolvable",
        "empty_field_protocol_error",
        "trigger_not_in_event_span",
        "arguments_top_level_mismatch",
        "normalized_multi_non_strict",
        "normalized_unique_non_strict",
        "text_exists_offset_wrong_multi",
        "text_exists_offset_wrong_unique",
        "start_end_boundary_error",
    ]
    for item in order:
        if item in tags:
            return item
    return "other"


def classify_failure(split: str, pred: dict[str, Any], raw: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    final = raw.get("raw_outputs", [])[-1]["content"] if raw.get("raw_outputs") else ""
    content = str(annotation.get("content", ""))
    try:
        payload = strict_json_loads(final)
    except Exception as exc:
        return {"split": split, "article_id": pred["article_id"], "primary_failure_category": "strict_json_parse_failed", "failure_tags": "strict_json_parse_failed", "error": str(exc)}
    tags: list[str] = []
    field_statuses = {}
    for field, allow_empty in [
        ("trigger", False),
        ("event_span", False),
        ("subject_company", False),
        ("event_time", False),
        ("object_target", True),
    ]:
        text = field_text(payload, field)
        start, end = field_raw_offset(payload, field)
        status = classify_text_locator(content, text, start, end, allow_empty=allow_empty)
        field_statuses[field] = status
        if status == "text_exists_offset_wrong_unique":
            tags.append("text_exists_offset_wrong_unique")
        elif status == "text_exists_offset_wrong_multi":
            tags.append("text_exists_offset_wrong_multi")
        elif status == "normalized_unique_non_strict":
            tags.append("normalized_unique_non_strict")
        elif status == "normalized_multi_non_strict":
            tags.append("normalized_multi_non_strict")
        elif status == "text_unresolvable":
            tags.append("text_unresolvable")
        elif status == "empty_field_protocol_error":
            tags.append("empty_field_protocol_error")
        elif status not in {"original_offset_strict_match", "empty_ok"} and status.startswith("start_end"):
            tags.append("start_end_boundary_error")
    trig_text = field_text(payload, "trigger")
    span_text = field_text(payload, "event_span")
    if trig_text and span_text and trig_text not in span_text:
        tags.append("trigger_not_in_event_span")
    args = payload.get("arguments") if isinstance(payload.get("arguments"), list) else []
    arg_set = {(str(item.get("text", "") or ""), str(item.get("role", "") or "")) for item in args if isinstance(item, dict)}
    top_set = {
        (field_text(payload, "subject_company"), "subject_company"),
        (field_text(payload, "event_time"), "event_time"),
        (field_text(payload, "object_target"), "object_target"),
        (field_text(payload, "trigger"), "trigger"),
    }
    top_set = {item for item in top_set if item[0]}
    if arg_set and top_set and arg_set != top_set:
        tags.append("arguments_top_level_mismatch")
    unique_tags = sorted(set(tags)) or ["other"]
    return {
        "split": split,
        "article_id": pred["article_id"],
        "prompt_version": pred.get("prompt_version", ""),
        "event_name": payload.get("event_name", ""),
        "primary_failure_category": primary_failure_category(unique_tags),
        "failure_tags": "|".join(unique_tags),
        "field_statuses": json.dumps(field_statuses, ensure_ascii=False),
        "raw_response_parseable_json": True,
        "final_raw_output_excerpt": final[:500],
    }


def audit_failures(annotations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["prompt_dev", "frozen_eval"]:
        raws = {str(row["article_id"]): row for row in raw_rows(split)}
        for pred in prediction_rows(split):
            if pred.get("status") == "success":
                continue
            aid = str(pred["article_id"])
            out.append(classify_failure(split, pred, raws[aid], annotations[aid]))
    return out


def adapt_split(split: str, annotations: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    adapter_results = []
    field_rows = []
    repair_rows = []
    for raw in raw_rows(split):
        aid = str(raw["article_id"])
        annotation = annotations[aid]
        final = raw.get("raw_outputs", [])[-1]["content"] if raw.get("raw_outputs") else ""
        prediction, fields, meta = adapt_raw_response(str(annotation.get("content", "")), final, expected_event_name=target_event_name(annotation))
        row = {
            "input_index": raw.get("input_index"),
            "split": split,
            "prompt_version": raw.get("prompt_version", ""),
            "article_id": aid,
            "status": "success" if meta["success"] else "failed",
            "prediction": prediction if meta["success"] else None,
            "adapter_errors": meta.get("errors", []),
            "ambiguous_field_count": meta.get("ambiguous_field_count", 0),
            "unresolvable_field_count": meta.get("unresolvable_field_count", 0),
            "strict_json_parse_failed": meta.get("strict_json_parse_failed", False),
            "argument_top_level_mismatch": meta.get("argument_top_level_mismatch", False),
        }
        adapter_results.append(row)
        for field in fields:
            field_rows.append({"split": split, "article_id": aid, **field})
        repair_rows.append(
            {
                "split": split,
                "article_id": aid,
                "adapter_status": row["status"],
                "ambiguous_field_count": row["ambiguous_field_count"],
                "unresolvable_field_count": row["unresolvable_field_count"],
                "errors": "|".join(row["adapter_errors"]),
            }
        )
    return adapter_results, field_rows, repair_rows


def metric_summary(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric_set": prefix,
        "sample_count": metrics["sample_count"],
        "success_count": metrics["success_count"],
        "trigger_relaxed_f1": metrics["trigger"]["relaxed_overlap_f1"],
        "event_span_char_f1": metrics["event_span"]["character_overlap_f1"],
        "subject_norm_acc": metrics["subject_company"]["normalized_accuracy"],
        "event_time_norm_acc": metrics["event_time"]["normalized_date_accuracy"],
        "object_norm_acc": metrics["object_target"]["normalized_accuracy"],
        "direction_acc": metrics["direction"]["accuracy"],
        "argument_f1": metrics["arguments"]["argument_identification_f1"],
        "argument_role_f1": metrics["arguments"]["end_to_end_argument_role_f1"],
        "article_exact": metrics["complete_structure"]["article_exact_match_accuracy"],
        "at_least_one_ratio": metrics["complete_structure"]["at_least_one_structure_field_correct_ratio"],
    }


def diagnostic_keep_ids(split: dict[str, Any], multi_rows: list[dict[str, Any]]) -> list[str]:
    bad = {
        row["article_id"]
        for row in multi_rows
        if row["split"] == "frozen_eval" and row["alignment_category"] in {"wrong_event_selected", "ambiguous_multi_event", "annotation_contract_incomplete"}
    }
    return [aid for aid in split["frozen_eval_article_ids"] if aid not in bad]


def write_sha_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append((rel(path), sha256_file(path)))
    (output_dir / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.perf_counter()
    if sha256_file(CONFIRMED) != EXPECTED_CONFIRMED_SHA:
        raise RuntimeError("confirmed annotation SHA mismatch")
    required = [
        STEP57A_DIR / "run_manifest.json",
        STEP57A_DIR / "prompt_dev_raw_responses.jsonl",
        STEP57A_DIR / "frozen_eval_raw_responses.jsonl",
        STEP57A_DIR / "prompt_dev_results.jsonl",
        STEP57A_DIR / "frozen_eval_results.jsonl",
        STEP57A_DIR / "split_manifest.json",
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    annotations = read_jsonl(CONFIRMED)
    validation = validate_confirmed_annotations(annotations)
    if validation["status"] != "valid":
        raise RuntimeError(f"confirmed annotation invalid: {validation}")
    ann_by_id = {str(row["article_id"]): row for row in annotations}
    split = read_json(STEP57A_DIR / "split_manifest.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7b_structure_failure_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    before = sha_snapshot(args.output_dir / "protected_and_frozen_sha256_before.txt", protected_and_frozen_files())
    shutil.copy2(args.output_dir / "protected_and_frozen_sha256_before.txt", backup_dir / "protected_and_frozen_sha256_before.txt")
    safety = {
        "generated_at": now(),
        "backup_dir": str(backup_dir),
        "process_check_before": process_check(),
        "before_sha_count": len(before),
        "confirmed_sha256": sha256_file(CONFIRMED),
        "step5_7a_run_manifest_sha256": sha256_file(STEP57A_DIR / "run_manifest.json"),
        "vllm_started": False,
        "local_32b_calls": 0,
        "external_llm_calls": 0,
        "gold_test_labels_read": False,
        "formal_outputs_modified": False,
    }
    write_json(args.output_dir / "safety_manifest.json", safety)
    write_json(backup_dir / "safety_manifest.json", safety)

    multi_rows = audit_multi_event(annotations, split)
    failure_rows = audit_failures(ann_by_id)
    adapter_prompt, prompt_field_rows, prompt_repair = adapt_split("prompt_dev", ann_by_id)
    adapter_frozen, frozen_field_rows, frozen_repair = adapt_split("frozen_eval", ann_by_id)
    field_rows = prompt_field_rows + frozen_field_rows
    repair_rows = prompt_repair + frozen_repair

    frozen_annotations = [ann_by_id[aid] for aid in split["frozen_eval_article_ids"]]
    original_frozen = prediction_rows("frozen_eval")
    original_metrics, original_detail, original_direction = evaluate_structure_predictions(frozen_annotations, original_frozen, "original_frozen_eval_replay")
    adapter_metrics, adapter_detail, adapter_direction = evaluate_structure_predictions(frozen_annotations, adapter_frozen, "adapter_frozen_eval")
    keep_ids = diagnostic_keep_ids(split, multi_rows)
    diagnostic_annotations = [ann_by_id[aid] for aid in keep_ids]
    diagnostic_original_rows = [row for row in original_frozen if str(row["article_id"]) in set(keep_ids)]
    diagnostic_adapter_rows = [row for row in adapter_frozen if str(row["article_id"]) in set(keep_ids)]
    diagnostic_original_metrics, _, _ = evaluate_structure_predictions(diagnostic_annotations, diagnostic_original_rows, "diagnostic_original_aligned_only")
    diagnostic_adapter_metrics, _, _ = evaluate_structure_predictions(diagnostic_annotations, diagnostic_adapter_rows, "diagnostic_adapter_aligned_only")

    step57a_report = read_json(STEP57A_DIR / "run_manifest.json")
    reproduced = {
        "success_count_matches": original_metrics["success_count"] == step57a_report["frozen_eval_metrics"]["success_count"],
        "trigger_relaxed_f1_delta": abs(original_metrics["trigger"]["relaxed_overlap_f1"] - step57a_report["frozen_eval_metrics"]["trigger"]["relaxed_overlap_f1"]),
        "event_span_char_f1_delta": abs(original_metrics["event_span"]["character_overlap_f1"] - step57a_report["frozen_eval_metrics"]["event_span"]["character_overlap_f1"]),
        "article_exact_delta": abs(original_metrics["complete_structure"]["article_exact_match_accuracy"] - step57a_report["frozen_eval_metrics"]["complete_structure"]["article_exact_match_accuracy"]),
    }
    originally_failed_ids = {str(row["article_id"]) for row in original_frozen if row.get("status") != "success"}
    adapter_success_ids = {str(row["article_id"]) for row in adapter_frozen if row.get("status") == "success"}
    repaired_original_failures = sorted(originally_failed_ids & adapter_success_ids)
    frozen_failures_before = len(originally_failed_ids)
    frozen_failures_after = sum(1 for row in adapter_frozen if row.get("status") != "success")
    failure_summary_rows = []
    for (split_name, category), count in sorted(Counter((row["split"], row["primary_failure_category"]) for row in failure_rows).items()):
        failure_summary_rows.append({"split": split_name, "primary_failure_category": category, "count": count})
    field_status_rows = []
    for (split_name, field, status), count in sorted(Counter((row.get("split"), row.get("field"), row.get("status")) for row in field_rows).items()):
        field_status_rows.append({"split": split_name, "field": field, "status": status, "count": count})
    multi_summary = dict(Counter(row["alignment_category"] for row in multi_rows))
    status = "contract_and_training_expansion_required"
    if not any(row["alignment_category"] in {"wrong_event_selected", "ambiguous_multi_event", "annotation_contract_incomplete"} for row in multi_rows):
        status = "structure_training_data_expansion_required"
    if adapter_metrics["success_count"] >= 60 and adapter_metrics["trigger"]["relaxed_overlap_f1"] >= 0.60 and adapter_metrics["event_span"]["character_overlap_f1"] >= 0.60:
        status = "offset_adapter_required"

    expansion_rows = [
        {"plan": "minimal_trial", "event_mentions": 200, "purpose": "验证mention级契约和offset适配后的结构抽取可训练性", "stratification": "覆盖已有24个事件类；每类至少3-5个EventMention，补足方向和未明确时间样本"},
        {"plan": "recommended_round1", "event_mentions": 500, "purpose": "训练/验证轻量结构抽取baseline", "stratification": "64类分层，优先补长尾；positive/negative/neutral/unclear按当前分布和困难样本均衡；明确/未明确时间各保留足量"},
        {"plan": "robust_round", "event_mentions": 800, "purpose": "稳定支持多事件文章展开和字段级误差审计", "stratification": "多事件文章拆成多条EventMention，至少15%-20%来自多事件文章；保留跨句span、专利、公告、政策类样本"},
    ]

    report = {
        "generated_at": now(),
        "status": status,
        "reference": "Offline Step 5.7-B audit over Step 5.7-A frozen raw responses; no new 32B calls.",
        "input_validation": validation,
        "safety": safety,
        "multi_event_alignment_summary": multi_summary,
        "multi_event_alignment_rows": multi_rows,
        "failure_summary_rows": failure_summary_rows,
        "failure_rows_count": len(failure_rows),
        "failure_rows_by_split": dict(Counter(row["split"] for row in failure_rows)),
        "adapter_summary": {
            "original_frozen_success_count": original_metrics["success_count"],
            "adapter_frozen_success_count": adapter_metrics["success_count"],
            "original_frozen_failed_count": frozen_failures_before,
            "adapter_frozen_failed_count": frozen_failures_after,
            "original_failed_rows_repaired_count": len(repaired_original_failures),
            "original_failed_rows_repaired_article_ids": repaired_original_failures,
            "adapter_unresolvable_field_count": sum(1 for row in field_rows if row.get("unresolvable")),
            "adapter_ambiguous_field_count": sum(1 for row in field_rows if row.get("ambiguous")),
        },
        "step5_7a_reproduction": reproduced,
        "metric_comparison_rows": [
            metric_summary("original_frozen_eval_replay", original_metrics),
            metric_summary("adapter_frozen_eval", adapter_metrics),
            metric_summary("diagnostic_original_aligned_only", diagnostic_original_metrics),
            metric_summary("diagnostic_adapter_aligned_only", diagnostic_adapter_metrics),
        ],
        "original_frozen_metrics": original_metrics,
        "adapter_frozen_metrics": adapter_metrics,
        "diagnostic_original_aligned_only_metrics": diagnostic_original_metrics,
        "diagnostic_adapter_aligned_only_metrics": diagnostic_adapter_metrics,
        "field_status_summary_rows": field_status_rows,
        "data_expansion_recommendations": expansion_rows,
        "recommendation": {
            "need_offset_adapter": adapter_metrics["success_count"] > original_metrics["success_count"],
            "need_event_mention_contract_rebuild": any(row["alignment_category"] in {"wrong_event_selected", "ambiguous_multi_event", "annotation_contract_incomplete"} for row in multi_rows),
            "need_structure_training_data_expansion": adapter_metrics["trigger"]["relaxed_overlap_f1"] < 0.60
            or adapter_metrics["event_span"]["character_overlap_f1"] < 0.60
            or adapter_metrics["arguments"]["end_to_end_argument_role_f1"] < 0.45,
            "do_not_generate_silver_now": True,
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
    }

    write_jsonl(args.output_dir / "multi_event_alignment_audit.jsonl", multi_rows)
    write_jsonl(args.output_dir / "offset_failure_rows.jsonl", failure_rows)
    write_jsonl(args.output_dir / "adapter_prompt_dev_results.jsonl", adapter_prompt)
    write_jsonl(args.output_dir / "adapter_frozen_eval_results.jsonl", adapter_frozen)
    write_jsonl(args.output_dir / "adapter_field_resolution_rows.jsonl", field_rows)
    write_jsonl(args.output_dir / "adapter_repair_rows.jsonl", repair_rows)
    write_json(args.output_dir / "evaluation_comparison.json", report)
    write_json(REPORT_JSON, report)

    md = [
        "# Task1 Step 5.7-B Structure Failure Audit",
        "",
        f"- status: {status}",
        f"- Step 5.7-A reproduction success: {reproduced['success_count_matches']} (success {original_metrics['success_count']}/60)",
        f"- adapter success: {adapter_metrics['success_count']}/60; repaired original frozen failures: {len(repaired_original_failures)}",
        f"- 30 failure rows by split: {dict(Counter(row['split'] for row in failure_rows))}",
        f"- failure categories: {dict(Counter(row['primary_failure_category'] for row in failure_rows))}",
        f"- multi-event alignment summary: {multi_summary}",
        f"- original trigger/span/argument+role: {original_metrics['trigger']['relaxed_overlap_f1']:.6f}/{original_metrics['event_span']['character_overlap_f1']:.6f}/{original_metrics['arguments']['end_to_end_argument_role_f1']:.6f}",
        f"- adapter trigger/span/argument+role: {adapter_metrics['trigger']['relaxed_overlap_f1']:.6f}/{adapter_metrics['event_span']['character_overlap_f1']:.6f}/{adapter_metrics['arguments']['end_to_end_argument_role_f1']:.6f}",
        "- recommendation: rebuild mention-level event contract and expand structure annotations before more 32B/Silver work.",
        "- LLM calls: local 32B 0, external 0.",
    ]
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_xlsx(
        REPORT_XLSX,
        {
            "多事件对齐审计": multi_rows,
            "Offset失败分类": failure_rows,
            "可修复记录": [row for row in repair_rows if row["adapter_status"] == "success"],
            "无法定位记录": [row for row in repair_rows if row["adapter_status"] != "success"],
            "字段级错误": field_status_rows,
            "修复前后指标": report["metric_comparison_rows"],
            "建议扩标分层": expansion_rows,
        },
    )
    after = sha_snapshot(args.output_dir / "protected_and_frozen_sha256_after.txt", protected_and_frozen_files())
    report["protection"] = {
        "formal_and_frozen_sha_unchanged": before == after,
        "confirmed_annotation_sha256": sha256_file(CONFIRMED),
        "confirmed_annotation_sha_unchanged": sha256_file(CONFIRMED) == EXPECTED_CONFIRMED_SHA,
        "step5_7a_frozen_response_sha_unchanged": before.get(rel(STEP57A_DIR / "frozen_eval_raw_responses.jsonl")) == after.get(rel(STEP57A_DIR / "frozen_eval_raw_responses.jsonl")),
        "backup_dir": str(backup_dir),
    }
    write_json(args.output_dir / "evaluation_comparison.json", report)
    write_json(REPORT_JSON, report)
    write_json(backup_dir / "final_summary.json", report)
    write_sha_manifest(args.output_dir)
    print(json.dumps({"status": status, "original_success": original_metrics["success_count"], "adapter_success": adapter_metrics["success_count"], "llm_calls": report["llm_calls"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

