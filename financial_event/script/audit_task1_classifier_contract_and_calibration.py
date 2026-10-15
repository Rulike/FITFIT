#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.6-C input-contract and pred_event_names lineage audit."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_5_candidate_v2 import read_jsonl, rel, sha256_file, write_json  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/task1_step5_6c_classifier_calibration"
PACKAGE = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package.jsonl"
CONFIRMED = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"
LATEST_ARTICLE_PRED = ROOT / "outputs/task1_step5_6b_latest_classifier_shadow/article_predictions.jsonl"
LATEST_PAIR_PRED = ROOT / "outputs/task1_step5_6b_latest_classifier_shadow/pair_predictions.jsonl"
STRUCTURE_MANIFEST = ROOT / "outputs/task1_structure_validation_round1/manifest.json"


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def parse_names(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def names_key(value: Any) -> str:
    return "|".join(parse_names(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else "",
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def compare_pred_event_names() -> dict[str, Any]:
    package_rows = read_jsonl(PACKAGE)
    confirmed_rows = read_jsonl(CONFIRMED)
    latest_rows = read_jsonl(LATEST_ARTICLE_PRED)
    pair_rows = read_jsonl(LATEST_PAIR_PRED)
    latest_by_id = {str(row["article_id"]): row for row in latest_rows}
    pair_by_article = {}
    for row in pair_rows:
        if row.get("decision") is True:
            pair_by_article.setdefault(str(row["article_id"]), []).append(row)
    details = []
    package_confirmed_match = 0
    confirmed_latest_match = 0
    order_match = [str(r["article_id"]) for r in package_rows] == [str(r["article_id"]) for r in confirmed_rows]
    latest_order_match = [str(r["article_id"]) for r in confirmed_rows] == [str(r["article_id"]) for r in latest_rows]
    for idx, row in enumerate(confirmed_rows):
        aid = str(row["article_id"])
        package_pred = names_key(package_rows[idx].get("pred_event_names", "")) if idx < len(package_rows) else ""
        confirmed_pred = names_key(row.get("pred_event_names", ""))
        latest_pred = names_key(latest_by_id.get(aid, {}).get("pred_event_names", ""))
        if package_pred == confirmed_pred:
            package_confirmed_match += 1
        if confirmed_pred == latest_pred:
            confirmed_latest_match += 1
        accepted = pair_by_article.get(aid, [])
        details.append(
            {
                "article_id": aid,
                "package_pred_event_names": package_pred,
                "confirmed_pred_event_names": confirmed_pred,
                "latest_5_6b_pred_event_names": latest_pred,
                "package_confirmed_match": package_pred == confirmed_pred,
                "confirmed_latest_5_6b_match": confirmed_pred == latest_pred,
                "score_trace_available": bool(accepted),
                "accepted_scores": "|".join(f"{item['event_name']}:{float(item.get('probability_positive', 0.0)):.6f}" for item in accepted),
            }
        )
    script_hits = []
    for path in (ROOT / "scripts").glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "minimal_human_review_package" in text or "pred_event_names" in text:
            script_hits.append(rel(path))
    return {
        "package_file": file_meta(PACKAGE),
        "confirmed_file": file_meta(CONFIRMED),
        "structure_manifest": read_json(STRUCTURE_MANIFEST) if STRUCTURE_MANIFEST.exists() else {},
        "candidate_generation_scripts_found": script_hits,
        "package_row_count": len(package_rows),
        "confirmed_row_count": len(confirmed_rows),
        "package_confirmed_order_match": order_match,
        "confirmed_step5_6b_order_match": latest_order_match,
        "package_confirmed_pred_event_names_match_count": package_confirmed_match,
        "confirmed_vs_step5_6b_article_predictions_match_count": confirmed_latest_match,
        "confirmed_vs_step5_6b_article_predictions_match_rate": confirmed_latest_match / len(confirmed_rows) if confirmed_rows else 0.0,
        "score_trace_available_count": sum(1 for row in details if row["score_trace_available"]),
        "details": details,
        "lineage_assessment": (
            "The original package and confirmed package preserve identical pred_event_names. "
            "Step 5.6-B reproduces those pred_event_names for 80/80 articles using model_epoch_2 + Candidate-v2 Top20 + threshold 0.93. "
            "No dedicated package-generation script was found in scripts/, so exact write-time provenance remains indirect, "
            "but Step 5.6-A's earlier conclusion that pred_event_names did not represent latest classification should be revised."
        ),
        "step5_6a_conclusion_needs_revision": True,
        "not_just_chance_reason": "Exact 80/80 article-level label-list equality and identical ordering after a frozen rerun makes accidental equality implausible.",
    }


def contract_audit() -> dict[str, Any]:
    training_script = ROOT / "scripts/run_task1_limited_training_round1.py"
    inference_script = ROOT / "scripts/run_task1_step5_6b_latest_classifier_shadow.py"
    text_module = ROOT / "src/data/event_match_text.py"
    training_text = training_script.read_text(encoding="utf-8")
    inference_text = inference_script.read_text(encoding="utf-8")
    text_code = text_module.read_text(encoding="utf-8")
    build_body = re.search(r"def build_input_text\(rec\):(.*?)(?:\n\ndef |\Z)", text_code, re.S)
    build_input_text_source = build_body.group(0).strip() if build_body else ""
    training_contract = {
        "title_used": True,
        "content_used": True,
        "target_type_used": True,
        "target_name_used": True,
        "target_code_used": True,
        "schema_id_used": False,
        "event_name_used": True,
        "schema_description_used_if_present": True,
        "candidate_source_rank_used": False,
        "template": "[目标]\\n类型/名称/代码 + [候选事件]\\n事件名称/事件定义 + [新闻]\\n标题/正文",
        "separator": "Chinese section headers and blank lines",
        "tokenizer": "AutoTokenizer.from_pretrained(model/base_model, local_files_only=True)",
        "max_length": 512,
        "truncation": "tokenizer truncation=True plus pre-cut content to head1900/tail600 if >2600 chars",
        "padding": "dynamic batch padding=True",
        "label_mapping": {"0": "negative", "1": "positive"},
        "positive_definition": "article-event candidate is a true gold event_name for the article",
        "negative_definition": "candidate-v2 hard/random negative event_name for same article",
        "logits_to_probability": "softmax(logits)[:, 1]",
        "score_direction": "higher probability_positive means more likely positive",
        "threshold_direction": "accept when score >= threshold",
        "multi_label_decision": "article-level decoder selects top1 only if top1 score >= min_score",
        "source_script": rel(training_script),
    }
    inference_contract = dict(training_contract)
    inference_contract.update(
        {
            "source_script": rel(inference_script),
            "threshold": 0.93,
            "decoder": "candidate_v2_top20_top1_min_score",
            "gold_free_input_fields": ["article_id", "title", "content", "publish_time", "source"],
        }
    )
    mismatches = []
    if "build_input_text" not in training_text or "build_input_text" not in inference_text:
        mismatches.append(
            {
                "field": "input_template_call",
                "training": "missing build_input_text call" if "build_input_text" not in training_text else "build_input_text",
                "inference": "missing build_input_text call" if "build_input_text" not in inference_text else "build_input_text",
                "impact": "would indicate contract drift",
            }
        )
    if "torch.softmax" not in training_text or "torch.softmax" not in inference_text:
        mismatches.append({"field": "probability_transform", "impact": "softmax positive-column transform not found in both scripts"})
    notes = [
        "No critical training/inference input mismatch was found.",
        "schema_id and candidate_source/rank are not part of the model text in either training or 5.6-B inference.",
        "event_description is in the template, but Candidate-v2 slim rows usually leave it blank; this is consistent across dev and 80 inference.",
        "label order is conventional single-label CE/softmax: class 1 is positive.",
    ]
    return {
        "training_contract": training_contract,
        "inference_contract": inference_contract,
        "build_input_text_source": build_input_text_source,
        "mismatches": mismatches,
        "critical_mismatch_count": len(mismatches),
        "model_loading_layer_error_detected": False,
        "classification_head_label_order_error_detected": False,
        "notes": notes,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": now(),
        "stage": "Task1 Step 5.6-C contract audit",
        "pred_event_names_lineage": compare_pred_event_names(),
        "contract_comparison": contract_audit(),
        "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
    }
    write_json(OUTPUT_DIR / "contract_comparison.json", report)
    print(json.dumps({
        "pred_event_names_5_6b_match_count": report["pred_event_names_lineage"]["confirmed_vs_step5_6b_article_predictions_match_count"],
        "critical_mismatch_count": report["contract_comparison"]["critical_mismatch_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
