#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.event_extraction.task1_supervised_structure_extractor import (  # noqa: E402
    CandidateRankerExtractor,
    split_sentences,
    validate_offsets,
)
from scripts.run_task1_step5_8hf_retrospective_eval import (  # noqa: E402
    HIER_MODEL_PKL,
    EMOTION_MODEL_PKL,
    TASK1_CANDIDATE_CONFIG,
    TAXONOMY_JSON,
    load_models,
    predict_rows as predict_task1_rows,
)
from scripts.task3_canonical_event_v2_resolver_v2_actual import (  # noqa: E402
    candidate_channels,
    feature_dict,
    threshold_predict,
)


OUT_DIR = ROOT / "outputs" / "task3_kg_v4_official1799_inference_v1"
RAW_TEST_XLSX = ROOT / "data" / "raw" / "news_with_tags_test.xlsx"
TASK1_OFFICIAL_REPORT = ROOT / "reports" / "task1_official_test_full_extraction_results_v2.xlsx"
PRIOR_MISMATCH_DIR = ROOT / "outputs" / "task3_kg_v4_official1799"
HF_FULL1799_DIR = ROOT / "outputs" / "task1_step5_8hf_full1799_retrospective"
HF_RETRO_DIR = ROOT / "outputs" / "task1_step5_8hf_retrospective_test"
HF_RETRO_PRED = HF_RETRO_DIR / "official_test_predictions_frozen.jsonl"
HF_RETRO_FREEZE = HF_RETRO_DIR / "final_freeze_manifest.json"
STRUCTURE_MODEL = ROOT / "outputs" / "task1_step5_7k_structure_extractor" / "experiment_C" / "model.joblib"
STRUCTURE_MODEL_ALT = ROOT / "outputs" / "task1_step5_7k_structure_extractor" / "experiment_C" / "model.pt"
STRUCTURE_MANIFEST = ROOT / "outputs" / "task1_step5_7k_structure_extractor" / "frozen_model_manifest.json"
RESOLVER_RUN = ROOT / "outputs" / "task3_canonical_event_v2_actual_repair" / "resolver_v2_run1"
RESOLVER_MODEL = RESOLVER_RUN / "resolver_v2_model.joblib"
RESOLVER_SELECTION = RESOLVER_RUN / "resolver_v2_selection.json"
BACKGROUND_TASK4 = RESOLVER_RUN / "task4_event_interface.jsonl"
KG_V3_GRAPH = ROOT / "outputs" / "task3_kg_v3_final" / "graph_nodes.jsonl"
KG_V3_TASK4 = ROOT / "outputs" / "task3_kg_v3_final" / "task4_event_interface.jsonl"

RAW_ALLOWED_COLUMNS = {
    "article_id": ["article_id", "article_file_id"],
    "title": ["title", "article_title"],
    "content": ["content"],
    "source": ["source", "article_source"],
    "publish_time": ["publish_time", "article_publish_time"],
}
FORBIDDEN_TEST_LABEL_COLUMNS = {
    "trading_code",
    "secu_abbr",
    "industry_code",
    "industry_name",
    "event_name",
    "event_emotion",
    "event_impact_analysis",
    "duplication_id",
    "target_type",
    "target_code",
    "target_name",
}
TASK1_REPORT_PREDICTION_SHEET = "模型预测全部1799条"
TASK1_REPORT_PREDICTION_COLUMNS = [
    "sheet_name",
    "sheet_row_index",
    "article_file_id",
    "article_title",
    "article_source",
    "article_publish_time",
    "content",
    "content_sha256",
    "title_sha256",
    "hierarchical_leaf_name",
    "event_layer",
    "event_parent",
    "event_group",
    "official_event_name",
    "hierarchical_confidence",
    "confidence",
    "event_name",
    "target_type",
    "target_code",
    "target_name",
    "candidate_score",
    "candidate_rank",
    "candidate_source",
    "route_count",
    "prediction_status",
    "prediction_notes",
]
TASK1_REPORT_FORBIDDEN_SHEETS = {"金标与预测逐条对照", "官方金标全部1799条"}
ALLOWED_SUPPORT = {
    "direct_fact",
    "confirmed_announcement",
    "regulatory_document",
    "financial_statement",
    "quoted_claim",
}


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"none", "null", "nan", "nat"} else value


def norm_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", text(value)).strip()


def norm_compact(value: Any) -> str:
    return re.sub(r"\s+", "", text(value)).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    value = norm_compact(value)
    return sha256_text(value) if value else ""


def stable_hash(*parts: Any, n: int = 24) -> str:
    payload = "||".join(text(part) for part in parts)
    return sha256_text(payload)[:n]


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_stat(path: Path) -> dict[str, Any]:
    item = {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path)}
    if path.exists():
        st = path.stat()
        item.update(
            {
                "size_bytes": st.st_size,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            }
        )
    return item


def excel_headers(path: Path) -> dict[str, list[str]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    out: dict[str, list[str]] = {}
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        out[sheet] = [text(cell.value) for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    wb.close()
    return out


def first_present(headers: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in headers:
            return candidate
    return ""


def read_label_blind_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    headers = excel_headers(RAW_TEST_XLSX)
    rows: list[dict[str, Any]] = []
    sheet_manifest: dict[str, Any] = {}
    global_idx = 0
    read_columns_by_sheet: dict[str, list[str]] = {}
    label_columns_present: dict[str, list[str]] = {}
    for sheet, sheet_headers in headers.items():
        source_cols = {key: first_present(sheet_headers, aliases) for key, aliases in RAW_ALLOWED_COLUMNS.items()}
        usecols = [col for col in source_cols.values() if col]
        missing_required = [key for key, col in source_cols.items() if not col and key != "article_id"]
        if missing_required:
            raise RuntimeError(f"missing allowed input columns in sheet={sheet}: {missing_required}")
        df = pd.read_excel(RAW_TEST_XLSX, sheet_name=sheet, usecols=usecols, dtype=object).fillna("")
        read_columns_by_sheet[sheet] = usecols
        label_columns_present[sheet] = [col for col in sheet_headers if col in FORBIDDEN_TEST_LABEL_COLUMNS]
        sheet_manifest[sheet] = {"rows": len(df), "source_columns": source_cols}
        for local_idx, record in enumerate(df.to_dict("records"), start=2):
            global_idx += 1
            official_row_id = f"OFF1799_{global_idx:04d}"
            article_id = text(record.get(source_cols.get("article_id", ""))) if source_cols.get("article_id") else ""
            if not article_id:
                article_id = official_row_id
            title = text(record.get(source_cols["title"]))
            content = text(record.get(source_cols["content"]))
            source = text(record.get(source_cols["source"]))
            publish_time = text(record.get(source_cols["publish_time"]))
            rows.append(
                {
                    "official_row_id": official_row_id,
                    "sheet_name": sheet,
                    "sheet_row_index": local_idx,
                    "article_id": article_id,
                    "article_file_id": article_id,
                    "title": title,
                    "article_title": title,
                    "content": content,
                    "source": source,
                    "article_source": source,
                    "publish_time": publish_time,
                    "article_publish_time": publish_time,
                    "content_hash": content_hash(content),
                    "title_hash": sha256_text(norm_compact(title)) if title else "",
                    "input_status": "missing_content" if not content else "ready",
                }
            )
    manifest = {
        "source_file": path_stat(RAW_TEST_XLSX),
        "allowed_logical_fields": list(RAW_ALLOWED_COLUMNS.keys()),
        "allowed_source_columns_by_sheet": read_columns_by_sheet,
        "sheet_manifest": sheet_manifest,
        "all_header_columns_inspected": headers,
        "forbidden_label_columns_present_but_not_read": label_columns_present,
        "forbidden_label_columns_read": [],
        "official_test_labels_read": False,
        "read_method": "pandas.read_excel(usecols=allowed_source_columns_only)",
        "row_count": len(rows),
        "unique_content_hashes_nonempty": len({r["content_hash"] for r in rows if r["content_hash"]}),
        "missing_content_rows": sum(1 for r in rows if not r["content"]),
    }
    return rows, manifest


def build_representatives(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    representative_by_hash: dict[str, str] = {}
    row_by_id = {row["official_row_id"]: row for row in rows}
    duplicate_of: dict[str, str] = {}
    duplicates_by_rep: dict[str, list[str]] = defaultdict(list)
    representatives: list[dict[str, Any]] = []
    for row in rows:
        key = row["content_hash"] or f"missing:{row['official_row_id']}"
        if key not in representative_by_hash:
            representative_by_hash[key] = row["official_row_id"]
            duplicate_of[row["official_row_id"]] = ""
            representatives.append(row)
        else:
            rep = representative_by_hash[key]
            duplicate_of[row["official_row_id"]] = rep
            duplicates_by_rep[rep].append(row["official_row_id"])
    for row in rows:
        rep = duplicate_of.get(row["official_row_id"], "")
        row["is_representative"] = not bool(rep)
        row["duplicate_of"] = rep
        row["representative_official_row_id"] = rep or row["official_row_id"]
    for rep, dup_ids in duplicates_by_rep.items():
        row_by_id[rep]["duplicate_row_ids"] = dup_ids
    return representatives, duplicate_of, duplicates_by_rep


def mark_prior_mismatch_audit() -> None:
    if not PRIOR_MISMATCH_DIR.exists():
        return
    marker = {
        "status": "corpus_mismatch_audit",
        "not_submission_product": True,
        "reason": "official1799 and current1707 were confirmed to be different pools; prior cross-pool mapping output is retained only as mismatch audit.",
        "replacement_output_dir": str(OUT_DIR),
        "marked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(PRIOR_MISMATCH_DIR / "corpus_mismatch_audit_marker.json", marker)


def existing_prediction_reuse_audit() -> dict[str, Any]:
    full_pred = HF_FULL1799_DIR / "official_test_predictions_frozen.jsonl"
    audit = {
        "checked_paths": {
            "preferred_full1799": path_stat(full_pred),
            "retrospective_8hf_prediction": path_stat(HF_RETRO_PRED),
            "retrospective_8hf_freeze_manifest": path_stat(HF_RETRO_FREEZE),
        },
        "reuse_existing_prediction_file": False,
        "reuse_reason": "",
    }
    if full_pred.exists():
        audit["reuse_existing_prediction_file"] = False
        audit["reuse_reason"] = "preferred path exists but this run reruns frozen models to produce a fresh label-blind inference artifact."
    elif HF_RETRO_PRED.exists() and HF_RETRO_FREEZE.exists():
        freeze = read_json(HF_RETRO_FREEZE)
        audit["retrospective_8hf_freeze_status"] = {
            "status": freeze.get("status"),
            "official_test_labels_read": freeze.get("official_test_labels_read"),
            "prediction_sha256": freeze.get("prediction_sha256"),
        }
        audit["reuse_reason"] = "8hf retrospective artifact was checked but not reused as final official inference because its freeze manifest records post-prediction official label evaluation; this run reruns frozen models from label-blind Excel input."
    else:
        audit["reuse_reason"] = "no reusable pure full1799 prediction file found; rerun frozen models from label-blind Excel input."
    return audit


def to_task1_model_input(row: dict[str, Any]) -> dict[str, Any]:
    content = text(row["content"])
    title = text(row["title"])
    return {
        "sheet_name": row["sheet_name"],
        "sheet_row_index": row["sheet_row_index"],
        "official_row_id": row["official_row_id"],
        "article_file_id": row["article_id"],
        "article_title": title,
        "article_source": row["source"],
        "article_publish_time": row["publish_time"],
        "content": content,
        "content_sha256": content_hash(content),
        "title_sha256": sha256_text(norm_compact(title)) if title else "",
    }


def run_frozen_task1(representatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hier, official_model, emotion_bundle, generator, maps, train_pairs, _ = load_models()
    prediction_inputs = [to_task1_model_input(row) for row in representatives]
    raw_predictions = predict_task1_rows(
        prediction_inputs,
        hier,
        official_model,
        emotion_bundle,
        generator,
        maps,
        train_pairs,
    )
    sanitized: list[dict[str, Any]] = []
    for original, pred in zip(representatives, raw_predictions):
        target_type = text(pred.get("target_type"))
        target_code = text(pred.get("target_code"))
        target_name = text(pred.get("target_name"))
        target_key = ""
        if target_type and target_code and target_type != "missing_prediction" and target_code != "missing_prediction":
            target_key = f"{target_type}|{target_code}"
        event_name = text(pred.get("event_name"))
        event_family = text(pred.get("event_group") or pred.get("event_parent") or event_name)
        sanitized.append(
            {
                "official_row_id": original["official_row_id"],
                "article_id": original["article_id"],
                "sheet_name": original["sheet_name"],
                "sheet_row_index": original["sheet_row_index"],
                "title": original["title"],
                "content": original["content"],
                "source": original["source"],
                "publish_time": original["publish_time"],
                "content_hash": original["content_hash"],
                "title_hash": original["title_hash"],
                "task1_target_type": target_type,
                "task1_target_code": target_code,
                "task1_target_name": target_name,
                "task1_target_key": target_key,
                "task1_event_name": event_name,
                "task1_event_family": event_family,
                "task1_event_parent": text(pred.get("event_parent")),
                "task1_hierarchical_leaf_name": text(pred.get("hierarchical_leaf_name")),
                "task1_confidence": float(pred.get("confidence") or 0.0),
                "task1_hierarchical_confidence": float(pred.get("hierarchical_confidence") or 0.0),
                "task1_candidate_score": float(pred.get("candidate_score") or 0.0),
                "task1_candidate_rank": int(pred.get("candidate_rank") or 0),
                "task1_candidate_source": text(pred.get("candidate_source")),
                "task1_route_count": int(pred.get("route_count") or 0),
                "task1_prediction_status": text(pred.get("prediction_status")),
                "task1_prediction_notes": text(pred.get("prediction_notes")),
                "task1_model_reused": True,
                "label_blind": True,
            }
        )
    return sanitized


def expand_predictions_to_all_rows(
    rows: list[dict[str, Any]],
    rep_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_by_rep = {row["official_row_id"]: row for row in rep_predictions}
    out: list[dict[str, Any]] = []
    for row in rows:
        rep_id = row["representative_official_row_id"]
        pred = dict(pred_by_rep[rep_id])
        pred.update(
            {
                "official_row_id": row["official_row_id"],
                "article_id": row["article_id"],
                "sheet_name": row["sheet_name"],
                "sheet_row_index": row["sheet_row_index"],
                "title": row["title"],
                "content": row["content"],
                "source": row["source"],
                "publish_time": row["publish_time"],
                "content_hash": row["content_hash"],
                "is_representative": row["is_representative"],
                "duplicate_of": row["duplicate_of"],
                "representative_official_row_id": rep_id,
                "inference_reused_from_representative": bool(row["duplicate_of"]),
            }
        )
        out.append(pred)
    return out


def span_ok(content: str, span: dict[str, Any] | None, allow_empty: bool = False) -> bool:
    if not isinstance(span, dict):
        return False
    span_text = text(span.get("text"))
    start = int(span.get("start", -1) if text(span.get("start", -1)) else -1)
    end = int(span.get("end", -1) if text(span.get("end", -1)) else -1)
    if not span_text:
        return allow_empty and start == -1 and end == -1
    return 0 <= start < end <= len(content) and content[start:end] == span_text


def sentence_window(content: str, start: int, end: int, radius_sentences: int = 1) -> dict[str, Any]:
    sents = split_sentences(content)
    if not sents or start < 0:
        return {"text": "", "start": -1, "end": -1, "sentence_id": ""}
    idx = None
    for i, sent in enumerate(sents):
        if sent["start"] <= start < end <= sent["end"]:
            idx = i
            break
    if idx is None:
        for i, sent in enumerate(sents):
            if sent["start"] <= start <= sent["end"]:
                idx = i
                break
    if idx is None:
        return {"text": "", "start": -1, "end": -1, "sentence_id": ""}
    lo = max(0, idx - radius_sentences)
    hi = min(len(sents) - 1, idx + radius_sentences)
    start_abs = sents[lo]["start"]
    end_abs = sents[hi]["end"]
    return {"text": content[start_abs:end_abs], "start": start_abs, "end": end_abs, "sentence_id": str(idx)}


def target_aliases(pred: dict[str, Any]) -> list[str]:
    aliases = [
        text(pred.get("task1_target_name")),
        text(pred.get("task1_target_code")),
    ]
    if text(pred.get("task1_target_type")) == "company":
        aliases.append(text(pred.get("task1_target_code")).zfill(6) if text(pred.get("task1_target_code")).isdigit() else "")
    out: list[str] = []
    for alias in aliases:
        alias = alias.strip()
        if alias and alias != "missing_prediction" and len(alias) >= 2 and alias not in out:
            out.append(alias)
    return out


def find_alias_span(content: str, aliases: list[str], start: int, end: int) -> tuple[dict[str, Any] | None, str]:
    if not content or start < 0 or end <= start:
        return None, "no_local_context"
    window = content[start:end]
    for alias in aliases:
        pos = window.find(alias)
        if pos >= 0:
            abs_start = start + pos
            return {"text": alias, "start": abs_start, "end": abs_start + len(alias)}, "local_context_exact"
    for alias in aliases:
        compact_alias = norm_compact(alias)
        if not compact_alias:
            continue
        compact_window = norm_compact(window)
        if compact_alias in compact_window:
            return None, "local_context_normalized_only"
    return None, "target_absent_from_local_context"


def find_alias_anywhere(content: str, aliases: list[str]) -> bool:
    return any(alias and alias in content for alias in aliases)


def modality_from_text(value: str) -> str:
    if any(word in value for word in ["传闻", "传言", "网传", "谣"]):
        return "rumor"
    if any(word in value for word in ["预计", "预期", "有望", "计划", "拟", "或将", "将", "未来", "展望"]):
        return "forecast"
    return "fact"


def support_type_from_text(value: str) -> str:
    if any(word in value for word in ["公告", "披露", "发布", "获批", "中标", "签署", "完成"]):
        return "confirmed_announcement"
    if any(word in value for word in ["表示", "称", "指出", "认为"]):
        return "quoted_claim"
    return "direct_fact"


def extract_structure(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extractor = CandidateRankerExtractor.load(STRUCTURE_MODEL)
    structured: list[dict[str, Any]] = []
    for pred in predictions:
        content = text(pred["content"])
        event_name = text(pred.get("task1_event_name"))
        title = text(pred.get("title"))
        target_key = text(pred.get("task1_target_key"))
        target_name = text(pred.get("task1_target_name"))
        target_type = text(pred.get("task1_target_type"))
        target_code = text(pred.get("task1_target_code"))
        structure = (
            extractor.predict_structure(content, event_name, title)
            if content and event_name and event_name != "missing_prediction"
            else {}
        )
        trigger = structure.get("trigger") if isinstance(structure.get("trigger"), dict) else {"text": "", "start": -1, "end": -1}
        event_span = structure.get("event_span") if isinstance(structure.get("event_span"), dict) else {"text": "", "start": -1, "end": -1}
        event_time = structure.get("event_time") if isinstance(structure.get("event_time"), dict) else {"text": "", "normalized": "未明确", "start": -1, "end": -1}
        object_target = structure.get("object_target") if isinstance(structure.get("object_target"), dict) else {"text": "", "start": -1, "end": -1}
        structure_offset_validation = validate_offsets(
            [
                {
                    "event_mention_id": pred["official_row_id"],
                    "article_id": pred["article_id"],
                    "content": content,
                    "event_name": event_name,
                    "title": title,
                }
            ],
            [{"event_mention_id": pred["official_row_id"], "article_id": pred["article_id"], **structure}],
        )
        event_span_valid = span_ok(content, event_span)
        trigger_valid = span_ok(content, trigger)
        trigger_inside_event = bool(
            trigger_valid
            and event_span_valid
            and int(event_span["start"]) <= int(trigger["start"]) < int(trigger["end"]) <= int(event_span["end"])
        )
        local = sentence_window(content, int(event_span.get("start", -1)), int(event_span.get("end", -1)), radius_sentences=1)
        aliases = target_aliases(pred)
        target_span, target_span_status = find_alias_span(content, aliases, int(local["start"]), int(local["end"]))
        target_found_anywhere = find_alias_anywhere(content, aliases)
        target_success = bool(target_key and target_name and target_name != "missing_prediction" and target_type != "missing_prediction")
        event_success = bool(event_name and event_name != "missing_prediction")
        structure_success = bool(trigger_valid and event_span_valid and trigger_inside_event)
        provenance_valid = bool(pred["article_id"] and pred["content_hash"])
        event_grounded = bool(trigger_valid and event_span_valid and trigger_inside_event)
        evidence_grounded = event_span_valid
        target_grounded = bool(target_span and target_span_status == "local_context_exact")
        contract_reasons: list[str] = []
        grounding_reasons: list[str] = []
        if not content:
            contract_reasons.append("missing_content")
        if not target_success:
            contract_reasons.append("task1_target_missing")
        if not event_success:
            contract_reasons.append("task1_event_missing")
        if not trigger_valid:
            contract_reasons.append("trigger_offset_unresolvable")
        if not event_span_valid:
            contract_reasons.append("event_fact_offset_unresolvable")
        if trigger_valid and event_span_valid and not trigger_inside_event:
            contract_reasons.append("trigger_not_inside_event_fact")
        if not provenance_valid:
            contract_reasons.append("provenance_missing")
        if target_success and not target_grounded:
            grounding_reasons.append(
                "target_not_local_to_evidence"
                if target_found_anywhere
                else "target_alias_not_found_in_content"
            )
        if event_success and not event_grounded:
            grounding_reasons.append("event_trigger_or_fact_not_grounded")
        if not evidence_grounded:
            grounding_reasons.append("evidence_span_not_grounded")
        if not provenance_valid:
            grounding_reasons.append("provenance_missing")
        if not content or any(reason.endswith("_unresolvable") or reason == "trigger_not_inside_event_fact" for reason in contract_reasons):
            contract_status = "quarantine"
        elif contract_reasons:
            contract_status = "review"
        else:
            contract_status = "pass"
        grounding_pass = bool(contract_status == "pass" and target_grounded and event_grounded and evidence_grounded and provenance_valid)
        evidence_text = text(event_span.get("text"))
        event_time_value = text(event_time.get("text")) or "未明确"
        event_time_normalized = text(event_time.get("normalized")) or event_time_value
        support_type = support_type_from_text(evidence_text)
        if support_type not in ALLOWED_SUPPORT:
            support_type = "direct_fact"
        modality = modality_from_text(" ".join([evidence_text, text(trigger.get("text")), event_name]))
        event_type = f"{target_type}_{target_code}_{event_name}" if target_type and target_code and event_name else event_name
        event_fact_signature = sha256_text(
            "||".join(
                [
                    target_key,
                    text(pred.get("task1_event_family")),
                    event_name,
                    text(trigger.get("text")),
                    text(object_target.get("text")),
                    event_time_normalized,
                    modality,
                    evidence_text,
                ]
            )
        )
        event_mention_id = "OFFT1E_" + stable_hash(pred["official_row_id"], target_key, event_fact_signature, event_span.get("start"), event_span.get("end"))
        evidence_unit_id = "OFFEU_" + stable_hash(event_mention_id, evidence_text, pred["content_hash"])
        sample_id = "OFFS_" + stable_hash(event_mention_id, pred["official_row_id"])
        record = {
            "schema_version": "structured_event_contract_v1",
            "official_row_id": pred["official_row_id"],
            "article_id": pred["article_id"],
            "publish_time": pred["publish_time"],
            "source": {
                "source_name": pred["source"],
                "source_dataset": "official1799_label_blind",
            },
            "target": {
                "target_key": target_key,
                "target_name": target_name,
                "target_type": target_type,
                "target_code": target_code,
                "mention_text": text(target_span.get("text")) if isinstance(target_span, dict) else "",
                "mention_span": target_span,
                "mention_span_status": target_span_status,
                "aliases_used": aliases,
            },
            "event": {
                "event_type": event_type,
                "event_name": event_name,
                "event_family": text(pred.get("task1_event_family")) or event_name,
                "event_fact": evidence_text,
                "event_fact_signature": event_fact_signature,
                "trigger_text": text(trigger.get("text")),
                "trigger_span": trigger,
                "event_time": event_time_normalized,
                "event_time_text": event_time_value,
                "event_time_span": event_time,
                "polarity_or_state": text(structure.get("direction")) or "unclear",
                "modality_status": modality,
                "object_text": text(object_target.get("text")),
                "object_span": object_target,
                "arguments": structure.get("arguments") if isinstance(structure.get("arguments"), list) else [],
            },
            "evidence": {
                "evidence_sentence": evidence_text,
                "local_context": text(local.get("text")),
                "local_context_span": {"start": local.get("start"), "end": local.get("end")},
                "evidence_span": event_span if event_span_valid else None,
                "support_type": support_type,
                "certainty": 0.9 if modality == "fact" else 0.55,
            },
            "provenance": {
                "document_hash": pred["content_hash"],
                "sentence_id": text(local.get("sentence_id")),
                "extraction_method": "official1799_label_blind_frozen_task1_structure_extractor",
                "extraction_time": datetime.now(timezone.utc).isoformat(),
                "upstream_model_or_rule_version": {
                    "target_event_runner": "scripts/run_task1_step5_8hf_retrospective_eval.py",
                    "structure_model": str(STRUCTURE_MODEL.relative_to(ROOT)),
                    "resolver_model": str(RESOLVER_MODEL.relative_to(ROOT)),
                },
                "source_file": str(RAW_TEST_XLSX.relative_to(ROOT)),
            },
            "quality": {
                "target_success": target_success,
                "event_success": event_success,
                "structure_success": structure_success,
                "target_grounded": target_grounded,
                "event_grounded": event_grounded,
                "evidence_grounded": evidence_grounded,
                "provenance_valid": provenance_valid,
                "grounding_pass": grounding_pass,
                "contract_status": contract_status,
                "contract_reasons": contract_reasons,
                "grounding_reasons": grounding_reasons,
                "offset_validation": structure_offset_validation,
                "target_found_anywhere": target_found_anywhere,
                "target_span_status": target_span_status,
            },
            "source_row": {
                "sample_id": sample_id,
                "event_id": event_mention_id,
                "evidence_unit_id": evidence_unit_id,
                "official_row_id": pred["official_row_id"],
                "article_file_id": pred["article_id"],
                "content_hash": pred["content_hash"],
                "sheet_name": pred["sheet_name"],
                "sheet_row_index": pred["sheet_row_index"],
                "dataset_split": "official1799_label_blind",
                "train_only": False,
                "validation_test_leakage": False,
            },
            "identity_features": {
                "event_family": text(pred.get("task1_event_family")) or event_name,
                "trigger_text": text(trigger.get("text")),
                "object_text": text(object_target.get("text")),
                "fact_tokens": sorted(set(re.findall(r"[0-9A-Za-z\u4e00-\u9fff]+", norm_compact(" ".join([evidence_text, text(trigger.get("text")), text(object_target.get("text")), event_time_normalized]))))),
                "provenance_key": f"{pred['article_id']}|{text(local.get('sentence_id'))}|{pred['content_hash'][:16]}",
                "modality_status": modality,
            },
        }
        structured.append(record)
    return structured


def expand_structured_to_all_rows(
    rows: list[dict[str, Any]],
    rep_structured: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_rep = {row["official_row_id"]: row for row in rep_structured}
    out: list[dict[str, Any]] = []
    for row in rows:
        rep_id = row["representative_official_row_id"]
        base = json.loads(json.dumps(by_rep[rep_id], ensure_ascii=False))
        if row["duplicate_of"]:
            base["official_row_id"] = row["official_row_id"]
            base["article_id"] = row["article_id"]
            base["source_row"]["official_row_id"] = row["official_row_id"]
            base["source_row"]["article_file_id"] = row["article_id"]
            base["quality"]["contract_status"] = "duplicate_of"
            base["quality"]["grounding_pass"] = False
            base["quality"]["duplicate_of"] = row["duplicate_of"]
            base["quality"]["inference_reused_from_representative"] = True
        else:
            base["quality"]["inference_reused_from_representative"] = False
        out.append(base)
    return out


def time_bucket(value: str) -> str:
    value = text(value)
    if not value or value == "未明确":
        return ""
    m = re.search(r"(20\d{2})[-年/](\d{1,2})", value)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.search(r"(\d{1,2})月", value)
    if m:
        return f"month-{int(m.group(1)):02d}"
    return value[:12]


def token_list(value: str) -> list[str]:
    return [tok for tok in re.findall(r"[0-9A-Za-z\u4e00-\u9fff]{2,}", norm_compact(value)) if tok]


def add_group_pairs(indices: list[int], pair_set: set[tuple[int, int]], max_group: int = 260) -> None:
    if len(indices) < 2:
        return
    indices = sorted(set(indices))
    if len(indices) > max_group:
        indices = indices[:max_group]
    for i, left in enumerate(indices):
        for right in indices[i + 1 :]:
            pair_set.add((left, right))


def generate_resolver_candidates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, event in enumerate(events):
        target = event["target"]
        ev = event["event"]
        src = event["source"]
        evidence = event["evidence"]
        target_key = text(target.get("target_key"))
        family = text(ev.get("event_family"))
        event_name = text(ev.get("event_name"))
        event_time = text(ev.get("event_time"))
        source_name = text(src.get("source_name"))
        object_text = text(ev.get("object_text"))
        trigger_text = text(ev.get("trigger_text"))
        evidence_text = text(evidence.get("evidence_sentence"))
        if target_key and family:
            groups[("target_family", target_key, family)].append(idx)
        if target_key and event_name:
            groups[("target_event", target_key, event_name)].append(idx)
        if target_key and time_bucket(event_time):
            groups[("target_time", target_key, time_bucket(event_time))].append(idx)
        if target_key and source_name:
            groups[("target_source", target_key, source_name)].append(idx)
        for tok in token_list(object_text)[:5]:
            groups[("target_object", target_key, tok)].append(idx)
        for tok in token_list(trigger_text)[:3]:
            groups[("target_trigger", target_key, family, tok)].append(idx)
        for tok in token_list(evidence_text)[:4]:
            groups[("target_evidence_token", target_key, family, tok)].append(idx)
    pair_set: set[tuple[int, int]] = set()
    for key, idxs in groups.items():
        add_group_pairs(idxs, pair_set, max_group=360 if key[0] in {"target_family", "target_event"} else 220)
    rows: list[dict[str, Any]] = []
    for left, right in sorted(pair_set):
        a = events[left]
        b = events[right]
        feats = feature_dict(a, b)
        channels = candidate_channels(a, b, feats)
        if not channels:
            continue
        rows.append(
            {
                "pair_id": "OFFPAIR_" + stable_hash(a["source_row"]["sample_id"], b["source_row"]["sample_id"]),
                "sample_id_a": a["source_row"]["sample_id"],
                "sample_id_b": b["source_row"]["sample_id"],
                "article_id_a": a["article_id"],
                "article_id_b": b["article_id"],
                "official_row_id_a": a["official_row_id"],
                "official_row_id_b": b["official_row_id"],
                "target_key_a": a["target"]["target_key"],
                "target_key_b": b["target"]["target_key"],
                "event_family_a": a["event"]["event_family"],
                "event_family_b": b["event"]["event_family"],
                "event_name_a": a["event"]["event_name"],
                "event_name_b": b["event"]["event_name"],
                "channels": "|".join(channels),
                "feature_dict": feats,
            }
        )
    return rows


class UnionFind:
    def __init__(self, members: list[str]) -> None:
        self.parent = {m: m for m in members}
        self.rank = {m: 0 for m in members}
        self.members = {m: {m} for m in members}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def can_merge(self, a: str, b: str, diff_pairs: set[tuple[str, str]]) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        for left in self.members[ra]:
            for right in self.members[rb]:
                if tuple(sorted((left, right))) in diff_pairs:
                    return False
        return True

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def run_frozen_resolver(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not events:
        return [], [], [], {
            "candidate_pairs": 0,
            "pair_decision_counts": {},
            "event_decision_counts": {},
        }
    resolver = joblib.load(RESOLVER_MODEL)
    selection = read_json(RESOLVER_SELECTION)
    thresholds = selection["thresholds"]
    classes = list(resolver.named_steps["clf"].classes_)
    candidates = generate_resolver_candidates(events)
    event_by_sid = {row["source_row"]["sample_id"]: row for row in events}
    resolved_pairs: list[dict[str, Any]] = []
    same_edges: list[dict[str, Any]] = []
    diff_pairs: set[tuple[str, str]] = set()
    uncertain_sids: set[str] = set()
    pair_decision_counter: Counter[str] = Counter()
    for row in candidates:
        sid_a = row["sample_id_a"]
        sid_b = row["sample_id_b"]
        a = event_by_sid[sid_a]
        b = event_by_sid[sid_b]
        feats = row["feature_dict"]
        prob = resolver.predict_proba([feats])[0]
        pred, conf, trace = threshold_predict(
            prob,
            classes,
            float(thresholds["same_thr"]),
            float(thresholds["diff_thr"]),
            float(thresholds.get("margin", 0.0)),
        )
        pair_decision_counter[pred] += 1
        pair_key = tuple(sorted((sid_a, sid_b)))
        if pred == "same_event" and conf >= float(thresholds["same_thr"]):
            same_edges.append({**row, "prediction": pred, "confidence": float(conf), "probabilities": trace})
        elif pred == "different_event" and conf >= max(0.72, float(thresholds["diff_thr"])):
            diff_pairs.add(pair_key)
        elif pred == "uncertain":
            uncertain_sids.update(pair_key)
        resolved_pairs.append(
            {
                **{k: v for k, v in row.items() if k != "feature_dict"},
                "prediction": pred,
                "confidence": round(float(conf), 6),
                "probabilities": trace,
                "hard_conflict_count": int(feats.get("hard_conflict_count", 0) or 0),
                "features": feats,
                "decision_reason": f"frozen_thresholds_same={thresholds['same_thr']}_diff={thresholds['diff_thr']}_margin={thresholds.get('margin', 0.0)}",
            }
        )
    same_edges.sort(key=lambda r: (-float(r["confidence"]), r["pair_id"]))
    uf = UnionFind(list(event_by_sid.keys()))
    for row in same_edges:
        if uf.can_merge(row["sample_id_a"], row["sample_id_b"], diff_pairs):
            uf.union(row["sample_id_a"], row["sample_id_b"])
    components: defaultdict[str, list[str]] = defaultdict(list)
    for sid in event_by_sid:
        components[uf.find(sid)].append(sid)
    cluster_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for _, members_raw in components.items():
        members = sorted(members_raw)
        component_uncertain = any(sid in uncertain_sids for sid in members)
        component_edges = [
            row for row in same_edges if row["sample_id_a"] in members and row["sample_id_b"] in members
        ]
        cluster_confidence = round(
            sum(float(row["confidence"]) for row in component_edges) / len(component_edges)
            if component_edges
            else 0.5,
            6,
        )
        layer = "review_uncertain_graph" if component_uncertain else "trusted_main_graph"
        resolver_decision = "uncertain" if component_uncertain else "accepted"
        prefix = "OFFCEV2REV" if component_uncertain else "OFFCEV2"
        canonical_event_id = f"{prefix}_{stable_hash(*members, n=20)}"
        event_thread_id = "OFFETF_" + stable_hash(canonical_event_id, n=20)
        representative = event_by_sid[members[0]]
        cluster_rows.append(
            {
                "canonical_event_id": canonical_event_id,
                "event_thread_id": event_thread_id,
                "graph_layer": layer,
                "resolver_decision": resolver_decision,
                "resolver_confidence": cluster_confidence,
                "member_count": len(members),
                "representative_sample_id": members[0],
                "representative_article_id": representative["article_id"],
                "representative_official_row_id": representative["official_row_id"],
                "target_key": representative["target"]["target_key"],
                "event_type": representative["event"]["event_type"],
                "event_family": representative["event"]["event_family"],
                "event_name": representative["event"]["event_name"],
                "event_fact_signature": representative["event"]["event_fact_signature"],
                "member_sample_ids": members,
                "member_official_row_ids": [event_by_sid[sid]["official_row_id"] for sid in members],
            }
        )
        for rank, sid in enumerate(members):
            event = event_by_sid[sid]
            mapping_rows.append(
                {
                    "canonical_event_id": canonical_event_id,
                    "event_thread_id": event_thread_id,
                    "graph_layer": layer,
                    "official_row_id": event["official_row_id"],
                    "article_id": event["article_id"],
                    "sample_id": sid,
                    "event_mention_id": event["source_row"]["event_id"],
                    "evidence_unit_id": event["source_row"]["evidence_unit_id"],
                    "target_key": event["target"]["target_key"],
                    "target_name": event["target"]["target_name"],
                    "target_type": event["target"]["target_type"],
                    "event_type": event["event"]["event_type"],
                    "event_family": event["event"]["event_family"],
                    "event_name": event["event"]["event_name"],
                    "trigger_text": event["event"]["trigger_text"],
                    "event_time": event["event"]["event_time"],
                    "polarity_or_state": event["event"]["polarity_or_state"],
                    "support_type": event["evidence"]["support_type"],
                    "event_fact_signature": event["event"]["event_fact_signature"],
                    "mention_span": event["target"]["mention_span"],
                    "trigger_span": event["event"]["trigger_span"],
                    "evidence_span": event["evidence"]["evidence_span"],
                    "provenance": event["provenance"],
                    "contract_status": event["quality"]["contract_status"],
                    "grounding_pass": event["quality"]["grounding_pass"],
                    "resolver_decision": resolver_decision,
                    "resolver_confidence": cluster_confidence,
                    "cluster_member_count": len(members),
                    "cluster_rank": rank,
                    "trust_score": round(min(0.99, max(0.0, 0.55 + 0.4 * cluster_confidence)), 6),
                    "acceptance_status": "pass" if layer == "trusted_main_graph" else "review",
                    "reject_reason": "" if layer == "trusted_main_graph" else "resolver_uncertain_pair_boundary",
                }
            )
    stats = {
        "candidate_pairs": len(candidates),
        "pair_decision_counts": dict(pair_decision_counter),
        "same_edges_used": len(same_edges),
        "cannot_link_pairs": len(diff_pairs),
        "uncertain_pair_count": pair_decision_counter.get("uncertain", 0),
        "event_decision_counts": dict(Counter(row["resolver_decision"] for row in mapping_rows)),
        "trusted_event_mentions": sum(1 for row in mapping_rows if row["graph_layer"] == "trusted_main_graph"),
        "review_event_mentions": sum(1 for row in mapping_rows if row["graph_layer"] == "review_uncertain_graph"),
        "trusted_canonical_events": sum(1 for row in cluster_rows if row["graph_layer"] == "trusted_main_graph"),
        "review_canonical_events": sum(1 for row in cluster_rows if row["graph_layer"] == "review_uncertain_graph"),
        "multi_mention_clusters": sum(1 for row in cluster_rows if int(row["member_count"]) > 1),
        "thresholds": thresholds,
        "resolver_model_sha256": sha256_file(RESOLVER_MODEL),
        "resolver_selection_sha256": sha256_file(RESOLVER_SELECTION),
    }
    return mapping_rows, cluster_rows, resolved_pairs, stats


def status_for_representative(
    structured: dict[str, Any],
    mapping_by_row: dict[str, dict[str, Any]],
) -> tuple[str, str, str, str]:
    quality = structured["quality"]
    oid = structured["official_row_id"]
    if not text(structured.get("source_row", {}).get("content_hash")):
        return "quarantine", "missing_content", "", ""
    if not quality.get("target_success"):
        return "no_target", "task1_target_missing", "", ""
    if not quality.get("event_success"):
        return "no_event", "task1_event_missing", "", ""
    if quality.get("contract_status") == "quarantine":
        return "quarantine", ";".join(quality.get("contract_reasons") or []) or "contract_quarantine", "", ""
    if not quality.get("grounding_pass"):
        return "grounding_failed", ";".join(quality.get("grounding_reasons") or quality.get("contract_reasons") or []) or "grounding_failed", "", ""
    mapping = mapping_by_row.get(oid)
    if not mapping:
        return "review", "resolver_not_mapped", "", "not_mapped"
    if mapping["graph_layer"] == "trusted_main_graph":
        return "accepted_to_main_graph", "", mapping["canonical_event_id"], mapping["resolver_decision"]
    return "review", mapping.get("reject_reason") or "resolver_uncertain", mapping["canonical_event_id"], mapping["resolver_decision"]


def build_processing_status(
    rows: list[dict[str, Any]],
    structured_all: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    structured_by_row = {row["official_row_id"]: row for row in structured_all}
    mapping_by_row = {row["official_row_id"]: row for row in mapping_rows}
    rep_status_cache: dict[str, tuple[str, str, str, str]] = {}
    status_rows: list[dict[str, Any]] = []
    for row in rows:
        structured = structured_by_row[row["official_row_id"]]
        rep_id = row["representative_official_row_id"]
        if row["duplicate_of"]:
            if rep_id not in rep_status_cache:
                rep_status_cache[rep_id] = status_for_representative(structured_by_row[rep_id], mapping_by_row)
            rep_status, rep_reason, rep_cid, rep_resolver = rep_status_cache[rep_id]
            status = "duplicate_of"
            reason = f"duplicate_of:{rep_id};representative_status:{rep_status};representative_reason:{rep_reason}"
            canonical_event_id = rep_cid
            resolver_decision = rep_resolver
            graph_layer = mapping_by_row.get(rep_id, {}).get("graph_layer", "")
        else:
            status, reason, canonical_event_id, resolver_decision = status_for_representative(structured, mapping_by_row)
            graph_layer = mapping_by_row.get(row["official_row_id"], {}).get("graph_layer", "")
            rep_status_cache[row["official_row_id"]] = (status, reason, canonical_event_id, resolver_decision)
        quality = structured["quality"]
        status_rows.append(
            {
                "official_row_id": row["official_row_id"],
                "article_id": row["article_id"],
                "sheet_name": row["sheet_name"],
                "sheet_row_index": row["sheet_row_index"],
                "content_hash": row["content_hash"],
                "is_representative": row["is_representative"],
                "duplicate_of": row["duplicate_of"],
                "representative_official_row_id": rep_id,
                "processing_status": status,
                "status_reason": reason,
                "task1_target_success": bool(quality.get("target_success")),
                "task1_event_success": bool(quality.get("event_success")),
                "task1_structure_success": bool(quality.get("structure_success")),
                "contract_status": text(quality.get("contract_status")),
                "grounding_pass": bool(quality.get("grounding_pass")),
                "target_grounded": bool(quality.get("target_grounded")),
                "event_grounded": bool(quality.get("event_grounded")),
                "evidence_grounded": bool(quality.get("evidence_grounded")),
                "provenance_valid": bool(quality.get("provenance_valid")),
                "contract_reasons": json.dumps(quality.get("contract_reasons") or [], ensure_ascii=False),
                "grounding_reasons": json.dumps(quality.get("grounding_reasons") or [], ensure_ascii=False),
                "target_span_status": text(quality.get("target_span_status")),
                "task1_target_key": structured["target"]["target_key"],
                "task1_target_name": structured["target"]["target_name"],
                "task1_target_type": structured["target"]["target_type"],
                "task1_event_name": structured["event"]["event_name"],
                "task1_event_family": structured["event"]["event_family"],
                "trigger_text": structured["event"]["trigger_text"],
                "event_time": structured["event"]["event_time"],
                "event_fact_signature": structured["event"]["event_fact_signature"],
                "event_mention_id": structured["source_row"]["event_id"],
                "canonical_event_id": canonical_event_id,
                "resolver_decision": resolver_decision,
                "graph_layer": graph_layer,
            }
        )
    return pd.DataFrame(status_rows)


def node(stable_id: str, node_type: str, layer: str, **props: Any) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "node_type": node_type,
        "graph_layer": layer,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **props,
    }


def edge(source_id: str, target_id: str, relation_type: str, layer: str, **props: Any) -> dict[str, Any]:
    return {
        "stable_id": "OFFED_" + stable_hash(source_id, target_id, relation_type, layer),
        "source_id": source_id,
        "target_id": target_id,
        "relation_type": relation_type,
        "graph_layer": layer,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **props,
    }


def build_graph(
    mapping_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    structured_by_row: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    canonical_events: list[dict[str, Any]] = []
    task4_rows: list[dict[str, Any]] = []
    for cluster in cluster_rows:
        layer = cluster["graph_layer"]
        cid = cluster["canonical_event_id"]
        cluster_props = {k: v for k, v in cluster.items() if k != "graph_layer"}
        nodes[cid] = node(cid, "CanonicalEvent", layer, **cluster_props)
        canonical_events.append(cluster)
    for mapping in mapping_rows:
        structured = structured_by_row[mapping["official_row_id"]]
        layer = mapping["graph_layer"]
        cid = mapping["canonical_event_id"]
        mid = mapping["event_mention_id"]
        eid = mapping["evidence_unit_id"]
        doc_id = "OFFDOC_" + structured["provenance"]["document_hash"][:24]
        source_id = "OFFSRC_" + stable_hash(structured["source"]["source_name"])
        target_type = text(structured["target"]["target_type"])
        target_node_type = "Company" if target_type == "company" else "Industry"
        target_id = "OFFTGT_" + stable_hash(structured["target"]["target_key"])
        nodes[mid] = node(
            mid,
            "EventMention",
            layer,
            official_row_id=mapping["official_row_id"],
            article_id=mapping["article_id"],
            canonical_event_id=cid,
            event_mention_id=mid,
            target_key=mapping["target_key"],
            event_type=mapping["event_type"],
            event_family=mapping["event_family"],
            event_name=mapping["event_name"],
            event_fact_signature=mapping["event_fact_signature"],
            event_time=mapping["event_time"],
            contract_status=mapping["contract_status"],
            resolver_decision=mapping["resolver_decision"],
            resolver_confidence=mapping["resolver_confidence"],
        )
        nodes[eid] = node(
            eid,
            "EvidenceUnit",
            layer,
            evidence_unit_id=eid,
            article_id=mapping["article_id"],
            evidence_span=mapping["evidence_span"],
            provenance=mapping["provenance"],
            support_type=mapping["support_type"],
        )
        nodes[doc_id] = node(
            doc_id,
            "Document",
            layer,
            article_id=mapping["article_id"],
            official_row_id=mapping["official_row_id"],
            document_hash=structured["provenance"]["document_hash"],
            title=structured.get("title", ""),
            publish_time=structured["publish_time"],
        )
        nodes[source_id] = node(source_id, "Source", layer, source_name=structured["source"]["source_name"])
        nodes[target_id] = node(
            target_id,
            target_node_type,
            layer,
            target_key=structured["target"]["target_key"],
            target_name=structured["target"]["target_name"],
            target_type=structured["target"]["target_type"],
            target_code=structured["target"]["target_code"],
        )
        for e in [
            edge(mid, cid, "INSTANCE_OF_EVENT", layer, confidence=mapping["resolver_confidence"]),
            edge(mid, eid, "BELONGS_TO_EVIDENCE_UNIT", layer),
            edge(eid, cid, "EVIDENCE_FOR", layer),
            edge(mid, doc_id, "EXTRACTED_FROM", layer),
            edge(doc_id, source_id, "PUBLISHED_BY", layer),
            edge(cid, target_id, "TARGETS_COMPANY" if target_type == "company" else "TARGETS_INDUSTRY", layer),
        ]:
            edges[e["stable_id"]] = e
        if layer == "trusted_main_graph":
            task4_rows.append(
                {
                    "canonical_event_id": cid,
                    "event_thread_id": mapping["event_thread_id"],
                    "event_mention_id": mid,
                    "official_row_id": mapping["official_row_id"],
                    "article_id": mapping["article_id"],
                    "schema_version": structured["schema_version"],
                    "target_key": mapping["target_key"],
                    "target_name": mapping["target_name"],
                    "target_type": mapping["target_type"],
                    "event_type": mapping["event_type"],
                    "event_family": mapping["event_family"],
                    "event_name": mapping["event_name"],
                    "trigger_text": mapping["trigger_text"],
                    "event_time": mapping["event_time"],
                    "polarity_or_state": mapping["polarity_or_state"],
                    "support_type": mapping["support_type"],
                    "event_fact_signature": mapping["event_fact_signature"],
                    "evidence_span": mapping["evidence_span"],
                    "provenance": mapping["provenance"],
                    "contract_status": mapping["contract_status"],
                    "resolver_decision": mapping["resolver_decision"],
                    "resolver_confidence": mapping["resolver_confidence"],
                    "trust_score": mapping["trust_score"],
                    "graph_layer": layer,
                    "acceptance_status": "pass",
                }
            )
    return list(nodes.values()), list(edges.values()), canonical_events, task4_rows


def build_review_sidecar(status_df: pd.DataFrame, structured_by_row: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in status_df.to_dict("records"):
        if rec["processing_status"] == "accepted_to_main_graph":
            continue
        structured = structured_by_row.get(rec["official_row_id"])
        rows.append(
            {
                "official_row_id": rec["official_row_id"],
                "article_id": rec["article_id"],
                "processing_status": rec["processing_status"],
                "status_reason": rec["status_reason"],
                "duplicate_of": rec["duplicate_of"],
                "representative_official_row_id": rec["representative_official_row_id"],
                "canonical_event_id": rec["canonical_event_id"],
                "resolver_decision": rec["resolver_decision"],
                "graph_layer": rec["graph_layer"],
                "target_key": rec["task1_target_key"],
                "event_name": rec["task1_event_name"],
                "event_family": rec["task1_event_family"],
                "trigger_text": rec["trigger_text"],
                "event_time": rec["event_time"],
                "quality": structured.get("quality") if structured else {},
                "evidence": structured.get("evidence") if structured else {},
                "provenance": structured.get("provenance") if structured else {},
            }
        )
    return rows


def graph_integrity(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], task4_rows: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = {row["stable_id"] for row in nodes}
    missing_edges = sum(1 for row in edges if row["source_id"] not in node_ids or row["target_id"] not in node_ids)
    provenance_ok = 0
    evidence_ok = 0
    for row in task4_rows:
        ev = row.get("evidence_span") if isinstance(row.get("evidence_span"), dict) else None
        prov = row.get("provenance") if isinstance(row.get("provenance"), dict) else None
        evidence_ok += int(bool(ev and text(ev.get("text")) and int(ev.get("start", -1)) >= 0))
        provenance_ok += int(bool(prov and text(prov.get("document_hash")) and text(prov.get("source_file"))))
    return {
        "edge_endpoint_missing": missing_edges,
        "edge_endpoint_integrity": missing_edges == 0,
        "task4_reference_count": len(task4_rows),
        "task4_evidence_span_rate": round(evidence_ok / max(1, len(task4_rows)), 6),
        "task4_provenance_rate": round(provenance_ok / max(1, len(task4_rows)), 6),
    }


def row_count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def output_stats() -> dict[str, Any]:
    paths = [
        "label_blind_input_manifest.json",
        "official1799_task1_predictions.jsonl",
        "official1799_structured_events.jsonl",
        "official1799_processing_status.parquet",
        "official1799_selection_funnel.json",
        "canonical_events.jsonl",
        "graph_nodes.jsonl",
        "graph_edges.jsonl",
        "review_sidecar.jsonl",
        "task4_event_interface.jsonl",
        "submission_manifest.json",
    ]
    return {name: path_stat(OUT_DIR / name) for name in paths}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mark_prior_mismatch_audit()

    label_rows, input_manifest = read_label_blind_inputs()
    representatives, duplicate_of, duplicates_by_rep = build_representatives(label_rows)
    input_manifest["deduplication"] = {
        "method": "content_hash_first_representative",
        "representative_rows": len(representatives),
        "duplicate_rows": sum(1 for row in label_rows if row["duplicate_of"]),
        "duplicate_groups": sum(1 for rows in duplicates_by_rep.values() if rows),
    }
    input_manifest["existing_prediction_reuse_audit"] = existing_prediction_reuse_audit()
    input_manifest["model_artifacts"] = {
        "task1_hierarchical_model": path_stat(HIER_MODEL_PKL),
        "task1_emotion_direction_model": path_stat(EMOTION_MODEL_PKL),
        "task1_candidate_config": path_stat(TASK1_CANDIDATE_CONFIG),
        "task1_taxonomy": path_stat(TAXONOMY_JSON),
        "task1_structure_model_joblib": path_stat(STRUCTURE_MODEL),
        "task1_structure_model_pt": path_stat(STRUCTURE_MODEL_ALT),
        "task1_structure_manifest": path_stat(STRUCTURE_MANIFEST),
        "resolver_v2_model": path_stat(RESOLVER_MODEL),
        "resolver_v2_selection": path_stat(RESOLVER_SELECTION),
    }
    write_json(OUT_DIR / "label_blind_input_manifest.json", input_manifest)

    rep_predictions = run_frozen_task1(representatives)
    all_predictions = expand_predictions_to_all_rows(label_rows, rep_predictions)
    write_jsonl(OUT_DIR / "official1799_task1_predictions.jsonl", all_predictions)

    rep_structured = extract_structure(rep_predictions)
    all_structured = expand_structured_to_all_rows(label_rows, rep_structured)
    write_jsonl(OUT_DIR / "official1799_structured_events.jsonl", all_structured)

    resolver_input_events = [
        row
        for row in rep_structured
        if row.get("quality", {}).get("grounding_pass")
        and row.get("quality", {}).get("contract_status") == "pass"
    ]
    mapping_rows, cluster_rows, resolver_pairs, resolver_stats = run_frozen_resolver(resolver_input_events)
    structured_by_row = {row["official_row_id"]: row for row in all_structured}
    status_df = build_processing_status(label_rows, all_structured, mapping_rows)
    status_df.to_parquet(OUT_DIR / "official1799_processing_status.parquet", index=False)

    graph_nodes, graph_edges, canonical_events, task4_rows = build_graph(mapping_rows, cluster_rows, structured_by_row)
    review_sidecar = build_review_sidecar(status_df, structured_by_row)
    write_jsonl(OUT_DIR / "canonical_events.jsonl", canonical_events)
    write_jsonl(OUT_DIR / "graph_nodes.jsonl", graph_nodes)
    write_jsonl(OUT_DIR / "graph_edges.jsonl", graph_edges)
    write_jsonl(OUT_DIR / "review_sidecar.jsonl", review_sidecar)
    write_jsonl(OUT_DIR / "task4_event_interface.jsonl", task4_rows)
    write_jsonl(OUT_DIR / "resolver_v2_pair_decisions.jsonl", resolver_pairs)

    status_counts = dict(Counter(status_df["processing_status"].tolist()))
    task1_target_success = sum(1 for row in rep_structured if row["quality"]["target_success"])
    task1_event_success = sum(1 for row in rep_structured if row["quality"]["event_success"])
    task1_structure_success = sum(1 for row in rep_structured if row["quality"]["structure_success"])
    contract_counts = dict(Counter(row["quality"]["contract_status"] for row in rep_structured))
    grounding_pass = sum(1 for row in rep_structured if row["quality"]["grounding_pass"])
    graph_layers = dict(Counter(row["graph_layer"] for row in graph_nodes))
    canonical_layer_counts = dict(Counter(row["graph_layer"] for row in canonical_events))
    background_rows = row_count_jsonl(BACKGROUND_TASK4)
    integrity = graph_integrity(graph_nodes, graph_edges, task4_rows)
    selection_funnel = {
        "status": "official1799_label_blind_inference_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "official1799_delivery_scope": True,
            "full_310k_production_inference_started": False,
            "prior_cross_pool_output_marked_as": "corpus_mismatch_audit",
            "background_204_strictly_separated": True,
            "background_204_count": background_rows,
            "background_204_counts_in_official_coverage": False,
        },
        "input": {
            "rows": len(label_rows),
            "unique_content_rows": len(representatives),
            "duplicate_rows_reused": sum(1 for row in label_rows if row["duplicate_of"]),
            "missing_content_rows": sum(1 for row in label_rows if not row["content"]),
        },
        "inference": {
            "actual_unique_rows_inferred": len(representatives),
            "duplicate_rows_reused": sum(1 for row in label_rows if row["duplicate_of"]),
            "task1_target_success": task1_target_success,
            "task1_event_success": task1_event_success,
            "task1_structure_success": task1_structure_success,
            "task1_target_success_rate": round(task1_target_success / max(1, len(representatives)), 6),
            "task1_event_success_rate": round(task1_event_success / max(1, len(representatives)), 6),
            "task1_structure_success_rate": round(task1_structure_success / max(1, len(representatives)), 6),
        },
        "contract_and_grounding": {
            "contract_status_counts_unique": contract_counts,
            "contract_pass_rows_unique": contract_counts.get("pass", 0),
            "contract_quarantine_rows_unique": contract_counts.get("quarantine", 0),
            "contract_review_rows_unique": contract_counts.get("review", 0),
            "grounding_pass_rows_unique": grounding_pass,
            "grounding_pass_rate_unique": round(grounding_pass / max(1, len(representatives)), 6),
        },
        "resolver_v2": resolver_stats,
        "official_graph": {
            "nodes": len(graph_nodes),
            "edges": len(graph_edges),
            "node_layer_counts": graph_layers,
            "canonical_event_layer_counts": canonical_layer_counts,
            "trusted_main_graph_event_mentions": sum(1 for row in mapping_rows if row["graph_layer"] == "trusted_main_graph"),
            "review_uncertain_graph_event_mentions": sum(1 for row in mapping_rows if row["graph_layer"] == "review_uncertain_graph"),
            "trusted_main_graph_canonical_events": canonical_layer_counts.get("trusted_main_graph", 0),
            "review_uncertain_graph_canonical_events": canonical_layer_counts.get("review_uncertain_graph", 0),
            "official_canonical_events_total": len(canonical_events),
            "multi_mention_cluster_count": sum(1 for row in canonical_events if int(row["member_count"]) > 1),
            "task4_interface_rows": len(task4_rows),
            "review_sidecar_rows": len(review_sidecar),
        },
        "official1799_processing_status_counts": status_counts,
        "integrity": integrity,
        "constraints": {
            "official_test_label_columns_read": [],
            "official_test_labels_read": False,
            "task34_benchmark_read": False,
            "severity_labels_read": False,
            "task1_retrained": False,
            "resolver_retrained_or_tuned": False,
            "resolver_thresholds_modified": False,
            "old_core_or_old_graph_modified": False,
        },
    }
    write_json(OUT_DIR / "official1799_selection_funnel.json", selection_funnel)

    protected_inputs = {
        "raw_official_test_xlsx": path_stat(RAW_TEST_XLSX),
        "task1_hierarchical_model": path_stat(HIER_MODEL_PKL),
        "task1_emotion_direction_model": path_stat(EMOTION_MODEL_PKL),
        "task1_structure_model": path_stat(STRUCTURE_MODEL),
        "resolver_v2_model": path_stat(RESOLVER_MODEL),
        "resolver_v2_selection": path_stat(RESOLVER_SELECTION),
        "background_task4_interface": path_stat(BACKGROUND_TASK4),
        "old_task3_kg_v3_graph": path_stat(KG_V3_GRAPH),
        "old_task3_kg_v3_task4_interface": path_stat(KG_V3_TASK4),
    }
    manifest = {
        "status": "official1799_inference_v1_ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_DIR),
        "repro_command": "python scripts/build_task3_kg_v4_official1799_inference_v1.py",
        "label_blind_input_manifest": str((OUT_DIR / "label_blind_input_manifest.json").relative_to(ROOT)),
        "input_paths": protected_inputs,
        "output_paths": output_stats(),
        "funnel_summary": selection_funnel,
        "background_context": {
            "path": str(BACKGROUND_TASK4),
            "rows": background_rows,
            "namespace": "background_context",
            "mixed_into_official_graph": False,
            "counted_in_official_coverage": False,
        },
        "constraints": selection_funnel["constraints"],
    }
    write_json(OUT_DIR / "submission_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(OUT_DIR),
                "official_rows": len(label_rows),
                "unique_content_rows": len(representatives),
                "actual_inference_rows": len(representatives),
                "duplicate_reuse_rows": sum(1 for row in label_rows if row["duplicate_of"]),
                "task1_target_success": task1_target_success,
                "task1_event_success": task1_event_success,
                "task1_structure_success": task1_structure_success,
                "contract_pass": contract_counts.get("pass", 0),
                "grounding_pass": grounding_pass,
                "resolver_event_decisions": resolver_stats.get("event_decision_counts", {}),
                "status_counts": status_counts,
                "official_canonical_events": len(canonical_events),
                "task4_interface_rows": len(task4_rows),
                "background_204_separated": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
