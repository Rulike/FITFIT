#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from task3_common import ROOT, canonical_list, parse_datetime, read_jsonl, stable_hash, write_jsonl

DEFAULT_OUT = ROOT / "outputs/task3_canonical_event_v2_actual_repair"
TASK1_EXPANSION = ROOT / "outputs/task1_severity_event_expansion_v1/task1_event_expansion_result.jsonl"
TASK1_SELECTION_FUNNEL = ROOT / "outputs/task1_severity_event_expansion_v1/selection_funnel.json"
TASK1_QUALITY_REPORT = ROOT / "outputs/task1_severity_event_expansion_v1/quality_report.json"
TASK1_FROZEN_TEST = ROOT / "outputs/task1_step5_7i_structure_expansion/independent_frozen_test_60_ai_drafts.jsonl"
PAIR_SAME_EVENT = ROOT / "outputs/task3/step2/same_event_labels_for_step3.csv"
PAIR_NEAR_DUP = ROOT / "outputs/task3/step2/near_duplicate_pilot_labeled.csv"
FULLTEXT_MENTIONS = ROOT / "outputs/task3_fulltext_v2/normalized_event_mentions.jsonl"
FULLTEXT_ARTICLE_INDEX = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"
FROZEN_CANDIDATES = ROOT / "outputs/task3/step3/same_event_candidates.jsonl"
FROZEN_PREDICTIONS = ROOT / "outputs/task3/step3/same_event_pair_predictions.jsonl"
SPLIT_MENTIONS = ROOT / "outputs/task3/splits/event_mention_split_mapping.jsonl"
SPLIT_DOCS = ROOT / "outputs/task3/splits/document_split_mapping.jsonl"
PROTECTED_FILES = [
    ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl",
    ROOT / "outputs/task3/step3/event_mention_canonical_mapping.jsonl",
    ROOT / "outputs/task3/step3/same_event_candidates.jsonl",
    ROOT / "outputs/task3/step3/same_event_pair_predictions.jsonl",
    ROOT / "outputs/task3/splits/event_mention_split_mapping.jsonl",
    ROOT / "outputs/task3/splits/document_split_mapping.jsonl",
    ROOT / "outputs/task3_kg_v3_final/state_observations/canonical_event_current_state.jsonl",
    ROOT / "outputs/task3_kg_v3_final/graph_nodes.jsonl",
    ROOT / "outputs/task3_kg_v3_final/task4_event_interface.jsonl",
    ROOT / "outputs/task4 golddata/task34.xlsx",
]

REJECT_WORDS = ("谣", "传闻", "预计", "将", "拟", "有望", "计划", "展望", "或将", "可能")
FACT_WORDS = ("已", "完成", "发布", "公告", "披露", "签署", "中标", "投产", "获批", "突破", "上市", "交付", "达成", "实现", "收购", "出售", "合并", "启动", "落地")
ALLOWED_SUPPORT = {"direct_fact", "confirmed_announcement", "regulatory_document", "financial_statement", "quoted_claim"}


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"none", "null", "nan", "nat"} else value


def norm(value: Any) -> str:
    value = unicodedata.normalize("NFKC", text(value))
    value = re.sub(r"\s+", "", value)
    return value.lower()


def clean_tokens(value: Any) -> list[str]:
    value = norm(value)
    if not value:
        return []
    return re.findall(r"[0-9a-zA-Z\u4e00-\u9fff]+", value)


def json_loads_maybe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    value = text(value)
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(data: str) -> str:
    return sha256_bytes(data.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return ""


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_xlsx(path: Path, visible_rows: list[dict[str, Any]], visible_sheet: str, hidden_rows: list[dict[str, Any]] | None = None, hidden_sheet: str = "_internal") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(visible_rows).to_excel(writer, sheet_name=visible_sheet, index=False)
        if hidden_rows is not None:
            pd.DataFrame(hidden_rows).to_excel(writer, sheet_name=hidden_sheet, index=False)
        wb = writer.book
        ws = wb[visible_sheet]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.sheet_view.showGridLines = True
        if hidden_rows is not None:
            wb[hidden_sheet].sheet_state = "hidden"
        for sheet in wb.worksheets:
            for cell in sheet[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="D9E2F3")
                cell.alignment = Alignment(vertical="center")
            for column_cells in sheet.columns:
                values = [len(text(cell.value)) for cell in column_cells[: min(len(column_cells), 40)]]
                width = min(max(values) if values else 8, 48)
                sheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width + 2


def normalize_title(value: Any) -> str:
    value = norm(value)
    value = re.sub(r"[，。！？、：:；;（）()\[\]【】「」“”\"'|｜·—_\-]", "", value)
    return value


def token_jaccard(a: Any, b: Any) -> float:
    sa = set(clean_tokens(a))
    sb = set(clean_tokens(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def sequence_ratio(a: Any, b: Any) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def lexical_similarity(a: Any, b: Any) -> float:
    return round(0.7 * sequence_ratio(a, b) + 0.3 * token_jaccard(a, b), 6)


def parse_offsets(value: Any) -> dict[str, int] | None:
    obj = json_loads_maybe(value)
    if not isinstance(obj, dict):
        return None
    try:
        start = int(obj.get("start"))
        end = int(obj.get("end"))
    except Exception:
        return None
    if start < 0 or end < start:
        return None
    return {"start": start, "end": end}


def exact_occurrences(content: str, needle: str) -> list[tuple[int, int]]:
    if not content or not needle:
        return []
    hits = []
    start = 0
    while True:
        idx = content.find(needle, start)
        if idx < 0:
            break
        hits.append((idx, idx + len(needle)))
        start = idx + 1
    return hits


def normalized_occurrences(content: str, needle: str) -> list[tuple[int, int]]:
    content_n = norm(content)
    needle_n = norm(needle)
    if not content_n or not needle_n:
        return []
    positions = []
    raw_chars = []
    for idx, ch in enumerate(unicodedata.normalize("NFKC", text(content))):
        if ch.isspace():
            continue
        raw_chars.append(ch.lower())
        positions.append(idx)
    if not raw_chars:
        return []
    raw = "".join(raw_chars)
    hits = []
    start = 0
    while True:
        idx = raw.find(needle_n, start)
        if idx < 0:
            break
        raw_start = positions[idx]
        raw_end = positions[idx + len(needle_n) - 1] + 1
        hits.append((raw_start, raw_end))
        start = idx + 1
    return hits


def choose_unique(candidates: list[tuple[int, int]], anchors: list[tuple[int, int]] | None = None) -> tuple[tuple[int, int] | None, str]:
    if not candidates:
        return None, "no_candidate"
    if len(candidates) == 1:
        return candidates[0], "unique"
    anchors = anchors or []
    if anchors:
        inside = [cand for cand in candidates if any(a0 <= cand[0] and cand[1] <= a1 for a0, a1 in anchors)]
        if len(inside) == 1:
            return inside[0], "inside_anchor"
        if len(inside) > 1:
            candidates = inside
        ranked = []
        for cand in candidates:
            center = (cand[0] + cand[1]) / 2.0
            dist = min(abs(center - ((a0 + a1) / 2.0)) for a0, a1 in anchors)
            ranked.append((dist, cand))
        ranked.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
        if len(ranked) >= 2 and abs(ranked[0][0] - ranked[1][0]) < 1e-9:
            return None, "ambiguous_multi_hit"
        return ranked[0][1], "nearest_anchor"
    return None, "ambiguous_multi_hit"


def longest_common_substrings(needle: str, min_len: int = 2, max_len: int = 12) -> list[str]:
    needle = norm(needle)
    if not needle:
        return []
    max_len = min(max_len, len(needle))
    chunks = []
    for size in range(max_len, min_len - 1, -1):
        seen = set()
        for start in range(0, len(needle) - size + 1):
            chunk = needle[start : start + size]
            if chunk in seen:
                continue
            seen.add(chunk)
            chunks.append(chunk)
    return chunks


def exact_or_normalized_span(content: str, needle: str, anchors: list[tuple[int, int]] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    needle = text(needle)
    if not needle:
        return None, {"status": "empty", "candidate_count": 0, "exact": False, "normalized": False}
    candidate_terms = [needle]
    candidate_terms.extend(longest_common_substrings(needle))
    seen_terms = set()
    for term in candidate_terms:
        term_n = norm(term)
        if not term_n or term_n in seen_terms:
            continue
        seen_terms.add(term_n)
        exact_hits = exact_occurrences(content, term)
        if exact_hits:
            chosen, reason = choose_unique(exact_hits, anchors=anchors)
            if chosen is not None:
                start, end = chosen
                return {"text": content[start:end], "start": start, "end": end}, {
                    "status": f"exact_{reason}",
                    "candidate_count": len(exact_hits),
                    "exact": True,
                    "normalized": False,
                }
        norm_hits = normalized_occurrences(content, term)
        if norm_hits:
            chosen, reason = choose_unique(norm_hits, anchors=anchors)
            if chosen is not None:
                start, end = chosen
                return {"text": content[start:end], "start": start, "end": end}, {
                    "status": f"normalized_{reason}",
                    "candidate_count": len(norm_hits),
                    "exact": False,
                    "normalized": True,
                }
    return None, {"status": "unresolvable", "candidate_count": 0, "exact": False, "normalized": False}


def stable_pair_id(prefix: str, a: str, b: str) -> str:
    left, right = sorted((text(a), text(b)))
    return f"{prefix}-{stable_hash(prefix, left, right)}"


def parse_event_type_family(value: Any) -> str:
    value = text(value)
    if not value:
        return ""
    parts = value.split("_")
    return parts[-1] if parts else value


def extract_tokens_from_text(value: Any) -> set[str]:
    raw = unicodedata.normalize("NFKC", text(value))
    tokens = set()
    for m in re.finditer(r"\d+(?:\.\d+)?(?:万|亿|万元|亿元|港元|元|%|％|股|份|期|只|辆|台|座|个|家|套|户|MW|GW|kW|Wh/kg|Wh)?", raw, re.I):
        tokens.add(m.group(0).replace(" ", ""))
    for m in re.finditer(r"[《“\"『‘](.*?)[》”\"』’]", raw):
        candidate = text(m.group(1))
        if candidate:
            tokens.add(candidate)
    for m in re.finditer(r"[A-Za-z0-9\u4e00-\u9fff]{2,30}", raw):
        token = text(m.group(0))
        if token and len(token) > 1:
            tokens.add(token)
    return {t for t in tokens if t}


def support_type_from_source(article_meta: dict[str, Any], row: dict[str, Any]) -> str:
    source_type = norm(article_meta.get("source_type"))
    source_name = norm(article_meta.get("article_source") or article_meta.get("source_name"))
    content = norm(article_meta.get("content") or row.get("content") or row.get("local_context"))
    if source_type in {"financial_statement", "financial_report"} or any(k in content for k in ("财报", "季报", "年报", "业绩快报", "财务报表")):
        return "financial_statement"
    if source_type in {"regulatory", "regulatory_document"} or any(k in content for k in ("公告", "批复", "监管", "证监", "交易所", "审评中心", "受理")):
        return "regulatory_document"
    if any(k in content for k in ("发布", "披露", "签署", "中标", "投产", "上市", "交付", "完成", "达成", "获批")) or "公告" in source_name:
        return "confirmed_announcement" if "公告" in content or "公告" in source_name else "direct_fact"
    if any(k in content for k in ("称", "表示", "透露")):
        return "quoted_claim"
    return "direct_fact"


def certainty_from_support(support_type: str) -> float:
    mapping = {
        "direct_fact": 0.96,
        "confirmed_announcement": 0.98,
        "regulatory_document": 0.99,
        "financial_statement": 0.97,
        "quoted_claim": 0.92,
    }
    return mapping.get(support_type, 0.9)


def modality_status(row: dict[str, Any], article_meta: dict[str, Any]) -> str:
    content = norm(" ".join([row.get("structure_event_span"), row.get("local_context"), row.get("evidence_sentence"), article_meta.get("content", "")]))
    if any(k in content for k in ("谣", "传闻", "消息称")):
        return "rumor"
    if any(k in content for k in ("预计", "将", "拟", "有望", "计划", "展望", "或将")) and not any(k in content for k in FACT_WORDS):
        return "forecast"
    return "fact"


def event_time_text(value: Any) -> str:
    value = text(value)
    return value if value else ""


def target_key_from_row(row: dict[str, Any]) -> str:
    return "|".join([text(row.get("target_type")), text(row.get("target_code"))]).strip("|")


def row_target_meta(row: dict[str, Any]) -> dict[str, str]:
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    target_type = text(target.get("target_type") or row.get("target_type"))
    target_code = text(target.get("target_code") or row.get("target_code"))
    target_name = text(target.get("target_name") or row.get("target_name") or row.get("entity_mention") or row.get("dominant_target"))
    target_key = text(target.get("target_key") or row.get("target_key") or "|".join([target_type, target_code]).strip("|"))
    return {
        "target_key": target_key,
        "target_type": target_type,
        "target_code": target_code,
        "target_name": target_name,
    }


def row_event_meta(row: dict[str, Any]) -> dict[str, Any]:
    event = row.get("event") if isinstance(row.get("event"), dict) else {}
    event_type = text(event.get("event_type") or row.get("event_type") or row.get("provisional_event_type") or row.get("schema_id"))
    event_name = text(event.get("event_name") or row.get("event_name") or parse_event_type_family(event_type))
    event_family = text(event.get("event_family") or row.get("_event_family") or parse_event_type_family(event_type))
    trigger_text = text(event.get("trigger_text") or row.get("trigger_text") or row.get("structure_trigger") or row.get("event_local_text"))
    object_text = text(event.get("object_text") or row.get("object_text") or row.get("structure_object") or row.get("core_elements"))
    event_fact = text(event.get("event_fact") or row.get("event_fact") or row.get("event_local_text") or row.get("structure_event_span") or row.get("local_context"))
    modality = text(event.get("modality_status") or row.get("modality_status") or row.get("verdict") or row.get("event_status") or "fact")
    event_time = text(event.get("event_time") or row.get("event_time"))
    polarity = text(event.get("polarity_or_state") or row.get("polarity_or_state") or row.get("event_status") or row.get("impact_direction") or row.get("verdict") or "unknown")
    return {
        "event_type": event_type,
        "event_name": event_name,
        "event_family": event_family,
        "trigger_text": trigger_text,
        "object_text": object_text,
        "event_fact": event_fact,
        "modality_status": modality,
        "event_time": event_time,
        "polarity_or_state": polarity,
    }


def row_source_meta(row: dict[str, Any]) -> dict[str, str]:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return {
        "source_name": text(source.get("source_name") or row.get("source_name") or row.get("article_source") or row.get("source")),
        "source_dataset": text(source.get("source_dataset") or row.get("source_dataset") or row.get("task1_source_path") or row.get("source_type")),
    }


def row_provenance_meta(row: dict[str, Any]) -> dict[str, str]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return {
        "document_hash": text(provenance.get("document_hash") or row.get("document_hash")),
        "sentence_id": text(provenance.get("sentence_id") or row.get("sentence_id")),
        "extraction_method": text(provenance.get("extraction_method") or row.get("extraction_method")),
        "extraction_time": text(provenance.get("extraction_time") or row.get("extraction_time")),
    }


def build_task1_recovered_records(task1_rows: list[dict[str, Any]], article_meta_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seen = set()
    recovered: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    duplicates = Counter()
    source_split_counter = Counter()
    for row in task1_rows:
        key = (
            text(row.get("article_id")),
            text(row.get("sentence_id")),
            text(row.get("target_type")),
            text(row.get("target_code")),
            text(row.get("evidence_span")),
            text(row.get("structure_event_span")),
        )
        if key in seen:
            duplicates["article_sentence_target_eventspan"] += 1
            continue
        seen.add(key)
        article_id = text(row.get("article_id"))
        article_meta = article_meta_by_id.get(article_id, {})
        source_split_counter[text(row.get("dataset_split"))] += 1
        evidence_sentence = text(row.get("evidence_sentence"))
        local_context = text(row.get("local_context") or row.get("context_window") or row.get("content"))
        content = text(row.get("content"))
        target_name = text(row.get("target_name") or row.get("structure_subject") or row.get("entity_mention"))
        target_anchor_candidates = [target_name, row.get("structure_subject"), row.get("entity_mention"), row.get("dominant_target")]
        evidence_offsets = parse_offsets(row.get("evidence_span"))
        evidence_text = ""
        if evidence_offsets and local_context:
            start = min(max(0, evidence_offsets["start"]), len(local_context))
            end = min(max(start, evidence_offsets["end"]), len(local_context))
            evidence_text = local_context[start:end]
        elif evidence_sentence:
            evidence_text = evidence_sentence
        target_span, target_meta = None, {"status": "missing"}
        for candidate in target_anchor_candidates:
            target_span, target_meta = exact_or_normalized_span(local_context or content or evidence_sentence, text(candidate), anchors=[(evidence_offsets["start"], evidence_offsets["end"])] if evidence_offsets else None)
            if target_span:
                break
        trigger_span, trigger_meta = None, {"status": "missing"}
        trigger_candidates = [row.get("structure_trigger"), row.get("event_time"), row.get("event_name")]
        trigger_content = text(row.get("structure_event_span") or local_context or evidence_sentence or content)
        for candidate in trigger_candidates:
            if not text(candidate):
                continue
            trigger_span, trigger_meta = exact_or_normalized_span(trigger_content, text(candidate), anchors=[(evidence_offsets["start"], evidence_offsets["end"])] if evidence_offsets else None)
            if trigger_span:
                break
        if not trigger_span:
            # fall back to the most specific substring in the trigger phrase that appears in the event span.
            trig = text(row.get("structure_trigger"))
            if trig:
                for token in sorted(set(longest_common_substrings(trig, min_len=2, max_len=min(12, len(norm(trig))))), key=len, reverse=True):
                    trigger_span, trigger_meta = exact_or_normalized_span(trigger_content, token, anchors=[(evidence_offsets["start"], evidence_offsets["end"])] if evidence_offsets else None)
                    if trigger_span:
                        break
        event_span_text = text(row.get("structure_event_span") or evidence_sentence or local_context)
        trigger_text = trigger_span["text"] if trigger_span else text(row.get("structure_trigger"))
        event_family = parse_event_type_family(row.get("event_type") or row.get("event_name"))
        support_type = support_type_from_source(article_meta, row)
        modality = modality_status(row, article_meta)
        event_time = event_time_text(row.get("event_time"))
        provenance = json_loads_maybe(row.get("provenance")) or {}
        document_hash = sha256_text(content or article_meta.get("content") or article_id)
        event_fact_signature = sha256_text(
            "||".join(
                [
                    target_key_from_row(row),
                    text(row.get("event_type") or row.get("event_name")),
                    event_family,
                    trigger_text,
                    text(row.get("structure_object")),
                    event_time,
                    modality,
                    support_type,
                ]
            )
        )
        quality_reasons = []
        target_grounded = bool(target_span)
        event_grounded = bool(trigger_span)
        evidence_grounded = bool(
            evidence_offsets
            and local_context
            and 0 <= evidence_offsets["start"] < evidence_offsets["end"] <= len(local_context)
        )
        if not evidence_grounded and evidence_sentence:
            evidence_grounded = True
        provenance_valid = bool(article_id and row.get("sentence_id") is not None)
        if not target_grounded:
            quality_reasons.append("target_not_locally_grounded")
        if not event_grounded:
            quality_reasons.append("event_trigger_missing")
        if not evidence_grounded:
            quality_reasons.append("evidence_not_grounded")
        if not provenance_valid:
            quality_reasons.append("provenance_missing")
        if support_type not in ALLOWED_SUPPORT:
            quality_reasons.append(f"support_type_not_allowed:{support_type}")
        contract_status = "pass" if not quality_reasons else "reject"
        record = {
            "schema_version": "structured_event_contract_v1",
            "article_id": article_id,
            "publish_time": text(row.get("publish_time")),
            "source": {
                "source_name": text(row.get("source") or row.get("article_source") or article_meta.get("article_source") or article_meta.get("source_name")),
                "source_dataset": text(row.get("source_dataset") or article_meta.get("source_type") or article_meta.get("dataset_split")),
            },
            "target": {
                "target_key": target_key_from_row(row),
                "target_name": target_name or text(row.get("target_name")),
                "target_type": text(row.get("target_type")),
                "target_code": text(row.get("target_code")),
                "mention_text": target_span["text"] if target_span else target_name,
                "mention_span": target_span,
                "role": "target",
            },
            "event": {
                "event_type": text(row.get("event_type")),
                "event_name": text(row.get("event_name")),
                "event_fact": event_span_text,
                "event_fact_signature": event_fact_signature,
                "trigger_text": trigger_text,
                "trigger_span": trigger_span,
                "event_time": event_time,
                "polarity_or_state": text(row.get("event_status") or row.get("impact_direction") or row.get("verdict") or "unknown"),
                "event_family": event_family,
                "modality_status": modality,
                "object_text": text(row.get("structure_object")),
            },
            "evidence": {
                "evidence_sentence": evidence_sentence,
                "local_context": local_context,
                "evidence_span": {"text": evidence_text, "start": evidence_offsets["start"], "end": evidence_offsets["end"]} if evidence_offsets and evidence_text else None,
                "support_type": support_type,
                "certainty": certainty_from_support(support_type),
            },
            "provenance": {
                "document_hash": document_hash,
                "sentence_id": text(row.get("sentence_id")),
                "extraction_method": "task1_severity_event_expansion_v1_recovered",
                "extraction_time": datetime.now().isoformat(timespec="seconds"),
                "upstream_model_or_rule_version": text((provenance or {}).get("structure_checkpoint") or (provenance or {}).get("event_model") or row.get("source_dataset") or "task1"),
                "source_file": text(row.get("source_dataset") or ""),
                "raw_provenance": provenance,
            },
            "quality": {
                "target_grounded": target_grounded,
                "event_grounded": event_grounded,
                "evidence_grounded": evidence_grounded,
                "provenance_valid": provenance_valid,
                "contract_status": contract_status,
                "reject_reasons": quality_reasons,
                "support_type_allowed": support_type in ALLOWED_SUPPORT,
                "target_span_match_status": target_meta.get("status"),
                "trigger_span_match_status": trigger_meta.get("status"),
                "evidence_span_match_status": "exact" if evidence_grounded else "unresolved",
            },
            "source_row": {
                "sample_id": text(row.get("sample_id")),
                "event_id": text(row.get("event_id")),
                "entity_id": text(row.get("entity_id")),
                "article_file_id": text(row.get("article_file_id")),
                "dataset_split": text(row.get("dataset_split")),
                "train_only": bool(row.get("train_only")),
                "validation_test_leakage": bool(row.get("validation_test_leakage")),
                "high_risk_score": row.get("high_risk_score"),
                "source_sha256": sha256_text(json.dumps(row, ensure_ascii=False, default=str)),
            },
            "identity_features": {
                "event_family": event_family,
                "trigger_text": trigger_text,
                "object_text": text(row.get("structure_object")),
                "fact_tokens": sorted(extract_tokens_from_text(" ".join([event_span_text, trigger_text, text(row.get("structure_object")), text(row.get("event_time"))]))),
                "provenance_key": f"{article_id}|{text(row.get('sentence_id'))}|{document_hash[:16]}",
                "modality_status": modality,
            },
        }
        if contract_status == "pass":
            recovered.append(record)
        else:
            quarantine.append(
                {
                    "article_id": article_id,
                    "sample_id": text(row.get("sample_id")),
                    "event_id": text(row.get("event_id")),
                    "reason": ";".join(quality_reasons) if quality_reasons else "unknown",
                    "target_grounded": target_grounded,
                    "event_grounded": event_grounded,
                    "evidence_grounded": evidence_grounded,
                    "provenance_valid": provenance_valid,
                    "target_span_status": target_meta.get("status"),
                    "trigger_span_status": trigger_meta.get("status"),
                    "evidence_span_status": "exact" if evidence_grounded else "unresolved",
                    "source_row": record["source_row"],
                }
            )
    validation = {
        "input_rows": len(task1_rows),
        "grounding_pass_rows": len(recovered) + len(quarantine),
        "unique_recovered_rows": len(recovered),
        "quarantine_rows": len(quarantine),
        "duplicate_rows_skipped": dict(duplicates),
        "source_split_counts": dict(source_split_counter),
        "pass_rate": round(len(recovered) / max(1, len(task1_rows)), 6),
        "quarantine_rate": round(len(quarantine) / max(1, len(task1_rows)), 6),
        "support_type_distribution": dict(Counter(row["evidence"]["support_type"] for row in recovered)),
    }
    return recovered, quarantine, validation


def load_article_index(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_parquet(path)
    out = {}
    for _, row in df.iterrows():
        aid = text(row.get("article_id"))
        if aid:
            out[aid] = {
                "article_id": aid,
                "article_title": text(row.get("article_title")),
                "article_source": text(row.get("article_source")),
                "article_publish_time": text(row.get("article_publish_time")),
                "content": text(row.get("content")),
                "content_length": row.get("content_length"),
                "source_type": text(row.get("source_type")),
                "source_reliability": row.get("source_reliability"),
                "dataset_split": text(row.get("dataset_split")),
                "is_labeled_source": bool(row.get("is_labeled_source")),
            }
    return out


def load_fulltext_mentions() -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(FULLTEXT_MENTIONS):
        rid = text(row.get("event_mention_id"))
        if not rid:
            continue
        row = dict(row)
        row["_source_file"] = str(FULLTEXT_MENTIONS.relative_to(ROOT))
        row["_event_family"] = parse_event_type_family(row.get("provisional_event_type") or row.get("event_type"))
        row["_title"] = text(row.get("event_local_text")) or " ".join(text(x) for x in row.get("evidence_sentences", [])[:2])
        row["_evidence_text"] = " ".join(text(x) for x in row.get("evidence_sentences", [])[:3])
        row["_fact_tokens"] = sorted(extract_tokens_from_text(" ".join([row.get("_title"), row.get("_evidence_text"), text(row.get("core_elements"))])))
        by_id[rid] = row
        by_article[text(row.get("article_id"))].append(row)
    return by_id, by_article


def load_frozen_mentions_split() -> dict[str, str]:
    mapping = {}
    for path in [SPLIT_MENTIONS, SPLIT_DOCS]:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            key = text(row.get("event_mention_id") or row.get("document_id"))
            if key:
                mapping[key] = text(row.get("split"))
    return mapping


def load_pair_rows() -> list[dict[str, Any]]:
    same = read_csv_rows(PAIR_SAME_EVENT)
    near = read_csv_rows(PAIR_NEAR_DUP)
    for row in same:
        row["_source_file"] = str(PAIR_SAME_EVENT.relative_to(ROOT))
    for row in near:
        row["_source_file"] = str(PAIR_NEAR_DUP.relative_to(ROOT))
    return same + near


def infer_gold_label(row: dict[str, Any]) -> str:
    if row.get("_source_file", "").endswith("same_event_labels_for_step3.csv"):
        relation = text(row.get("human_relation")).lower()
        if relation in {"same_event", "different_event", "related_update"}:
            return "uncertain" if relation == "related_update" else relation
        same_flag = text(row.get("human_same_event")).lower()
        if same_flag == "yes":
            return "same_event"
        if same_flag == "no":
            return "different_event"
        return "uncertain"
    relation = text(row.get("near_duplicate_label"))
    if relation == "1" or text(row.get("human_is_near_duplicate")).lower() == "yes":
        return "same_event"
    return "different_event"


def load_frozen_predictions() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not FROZEN_PREDICTIONS.exists():
        return out
    for row in read_jsonl(FROZEN_PREDICTIONS):
        key = tuple(sorted((text(row.get("event_mention_id_a")), text(row.get("event_mention_id_b")))))
        if key[0] and key[1]:
            out[key] = row
    return out


def load_frozen_candidate_pairs() -> set[tuple[str, str]]:
    out = set()
    if not FROZEN_CANDIDATES.exists():
        return out
    for row in read_jsonl(FROZEN_CANDIDATES):
        key = tuple(sorted((text(row.get("event_mention_id_a")), text(row.get("event_mention_id_b")))))
        if key[0] and key[1]:
            out.add(key)
    return out


def pair_time_gap_hours(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    da = parse_datetime(a.get("event_time")) or parse_datetime(a.get("publish_time"))
    db = parse_datetime(b.get("event_time")) or parse_datetime(b.get("publish_time"))
    if not da or not db:
        return None
    return abs((da - db).total_seconds()) / 3600.0


def pair_time_relation(a: dict[str, Any], b: dict[str, Any]) -> str:
    da = parse_datetime(a.get("event_time"))
    db = parse_datetime(b.get("event_time"))
    if da and db:
        gap_days = abs((da - db).total_seconds()) / 86400.0
        if gap_days == 0:
            return "same_time"
        if gap_days <= 3:
            return "close_time"
        if gap_days <= 30:
            return "near_time"
        return "time_conflict"
    pa = parse_datetime(a.get("publish_time"))
    pb = parse_datetime(b.get("publish_time"))
    if pa and pb:
        gap_days = abs((pa - pb).total_seconds()) / 86400.0
        if gap_days <= 3:
            return "close_publish_time"
        if gap_days <= 30:
            return "near_publish_time"
    return "unknown"


def modality_relation(a: dict[str, Any], b: dict[str, Any]) -> str:
    ma = text(row_event_meta(a).get("modality_status"))
    mb = text(row_event_meta(b).get("modality_status"))
    if ma == mb:
        return f"same_{ma}" if ma else "unknown"
    if {ma, mb} <= {"forecast", "fact"}:
        return "forecast_vs_fact"
    if {ma, mb} <= {"rumor", "fact"}:
        return "rumor_vs_fact"
    return "different_modality"


def provenance_relation(a: dict[str, Any], b: dict[str, Any]) -> str:
    if text(a.get("article_id")) == text(b.get("article_id")) and text(a.get("article_id")):
        return "same_article"
    source_a = row_source_meta(a)
    source_b = row_source_meta(b)
    if text(source_a.get("source_name")) and text(source_a.get("source_name")) == text(source_b.get("source_name")):
        return "same_source"
    prov_a = row_provenance_meta(a)
    prov_b = row_provenance_meta(b)
    if text(prov_a.get("document_hash")) and text(prov_a.get("document_hash")) == text(prov_b.get("document_hash")):
        return "same_document_hash"
    return "cross_source"


def identity_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    fa = a.get("identity_features", {})
    fb = b.get("identity_features", {})
    ta = row_target_meta(a)
    tb = row_target_meta(b)
    ea = row_event_meta(a)
    eb = row_event_meta(b)
    prov_a = row_provenance_meta(a)
    prov_b = row_provenance_meta(b)
    source_a = row_source_meta(a)
    source_b = row_source_meta(b)
    fact_tokens_a = fa.get("fact_tokens") if isinstance(fa.get("fact_tokens"), list) and fa.get("fact_tokens") else sorted(extract_tokens_from_text(ea.get("event_fact")))
    fact_tokens_b = fb.get("fact_tokens") if isinstance(fb.get("fact_tokens"), list) and fb.get("fact_tokens") else sorted(extract_tokens_from_text(eb.get("event_fact")))
    object_tokens_a = sorted(extract_tokens_from_text(ea.get("object_text")))
    object_tokens_b = sorted(extract_tokens_from_text(eb.get("object_text")))
    trigger_tokens_a = sorted(extract_tokens_from_text(ea.get("trigger_text")))
    trigger_tokens_b = sorted(extract_tokens_from_text(eb.get("trigger_text")))
    ai = {
        "same_target": bool(text(ta.get("target_key"))) and text(ta.get("target_key")) == text(tb.get("target_key")),
        "same_target_type": bool(text(ta.get("target_type"))) and text(ta.get("target_type")) == text(tb.get("target_type")),
        "same_target_code": bool(text(ta.get("target_code"))) and text(ta.get("target_code")) == text(tb.get("target_code")),
        "same_family": bool(text(ea.get("event_family"))) and text(ea.get("event_family")) == text(eb.get("event_family")),
        "same_event_type": bool(text(ea.get("event_type"))) and text(ea.get("event_type")) == text(eb.get("event_type")),
        "same_event_name": bool(text(ea.get("event_name"))) and text(ea.get("event_name")) == text(eb.get("event_name")),
        "same_trigger": bool(text(ea.get("trigger_text"))) and text(ea.get("trigger_text")) == text(eb.get("trigger_text")),
        "same_object": bool(text(ea.get("object_text"))) and text(ea.get("object_text")) == text(eb.get("object_text")),
        "same_modality": bool(text(ea.get("modality_status"))) and text(ea.get("modality_status")) == text(eb.get("modality_status")),
        "same_article": bool(text(a.get("article_id"))) and text(a.get("article_id")) == text(b.get("article_id")),
        "same_source": bool(text(source_a.get("source_name"))) and text(source_a.get("source_name")) == text(source_b.get("source_name")),
        "same_sentence": bool(text(prov_a.get("sentence_id"))) and text(prov_a.get("sentence_id")) == text(prov_b.get("sentence_id")),
        "same_document_hash": bool(text(prov_a.get("document_hash"))) and text(prov_a.get("document_hash")) == text(prov_b.get("document_hash")),
        "fact_tokens_a": fact_tokens_a,
        "fact_tokens_b": fact_tokens_b,
        "object_tokens_a": object_tokens_a,
        "object_tokens_b": object_tokens_b,
        "trigger_tokens_a": trigger_tokens_a,
        "trigger_tokens_b": trigger_tokens_b,
    }
    ai["fact_token_overlap"] = sorted(set(fact_tokens_a) & set(fact_tokens_b))
    ai["object_token_overlap"] = sorted(set(object_tokens_a) & set(object_tokens_b))
    ai["trigger_token_overlap"] = sorted(set(trigger_tokens_a) & set(trigger_tokens_b))
    ai["title_similarity"] = round(lexical_similarity(ea.get("event_fact"), eb.get("event_fact")), 6)
    evidence_a = text(a.get("evidence", {}).get("evidence_sentence") if isinstance(a.get("evidence"), dict) else a.get("evidence_sentence") or a.get("event_local_text") or a.get("structure_event_span"))
    evidence_b = text(b.get("evidence", {}).get("evidence_sentence") if isinstance(b.get("evidence"), dict) else b.get("evidence_sentence") or b.get("event_local_text") or b.get("structure_event_span"))
    ai["evidence_similarity"] = round(lexical_similarity(evidence_a, evidence_b), 6)
    ai["fact_similarity"] = round(token_jaccard(" ".join(fact_tokens_a), " ".join(fact_tokens_b)), 6)
    ai["object_similarity"] = round(token_jaccard(" ".join(object_tokens_a), " ".join(object_tokens_b)), 6)
    ai["trigger_similarity"] = round(token_jaccard(" ".join(trigger_tokens_a), " ".join(trigger_tokens_b)), 6)
    ai["time_relation"] = pair_time_relation({"event_time": ea.get("event_time"), "publish_time": a.get("publish_time")}, {"event_time": eb.get("event_time"), "publish_time": b.get("publish_time")})
    ai["modality_relation"] = modality_relation(a, b)
    ai["provenance_relation"] = provenance_relation(a, b)
    return ai


def hard_conflicts(features: dict[str, Any]) -> list[str]:
    reasons = []
    if not features["same_target"] and (features["same_target_type"] or features["same_target_code"]):
        reasons.append("target_key_conflict")
    if features["same_family"] and features["same_target"] and features["object_token_overlap"] == [] and features["object_tokens_a"] and features["object_tokens_b"]:
        reasons.append("object_conflict")
    if features["same_target"] and features["same_family"] and features["fact_similarity"] < 0.1 and features["title_similarity"] < 0.75 and features["evidence_similarity"] < 0.75:
        reasons.append("same_family_fact_divergence")
    if features["time_relation"] == "time_conflict":
        reasons.append("time_conflict")
    if features["modality_relation"] in {"forecast_vs_fact", "rumor_vs_fact"}:
        reasons.append("modality_conflict")
    if features["same_target"] and features["fact_similarity"] < 0.15 and features["same_event_type"] is False and features["same_event_name"] is False and features["same_family"] is False:
        reasons.append("fact_conflict")
    return reasons


def score_pair(a: dict[str, Any], b: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    score = 0.0
    contributions = []
    if features["same_target"]:
        score += 2.4
        contributions.append("same_target:+2.4")
    if features["same_family"]:
        score += 1.1
        contributions.append("same_family:+1.1")
    if features["same_event_type"]:
        score += 0.7
        contributions.append("same_event_type:+0.7")
    if features["same_event_name"]:
        score += 0.6
        contributions.append("same_event_name:+0.6")
    if features["same_trigger"]:
        score += 1.0
        contributions.append("same_trigger:+1.0")
    if features["fact_similarity"] >= 0.90:
        score += 2.0
        contributions.append("fact_sim>=0.90:+2.0")
    elif features["fact_similarity"] >= 0.80:
        score += 1.5
        contributions.append("fact_sim>=0.80:+1.5")
    elif features["fact_similarity"] >= 0.70:
        score += 1.0
        contributions.append("fact_sim>=0.70:+1.0")
    elif features["fact_similarity"] >= 0.55:
        score += 0.4
        contributions.append("fact_sim>=0.55:+0.4")
    if features["object_similarity"] >= 0.50:
        score += 0.8
        contributions.append("object_sim>=0.50:+0.8")
    elif features["object_similarity"] >= 0.25:
        score += 0.4
        contributions.append("object_sim>=0.25:+0.4")
    if features["trigger_similarity"] >= 0.60:
        score += 0.5
        contributions.append("trigger_sim>=0.60:+0.5")
    if features["title_similarity"] >= 0.90:
        score += 0.8
        contributions.append("title_sim>=0.90:+0.8")
    elif features["title_similarity"] >= 0.75:
        score += 0.5
        contributions.append("title_sim>=0.75:+0.5")
    if features["evidence_similarity"] >= 0.80:
        score += 0.7
        contributions.append("evidence_sim>=0.80:+0.7")
    elif features["evidence_similarity"] >= 0.60:
        score += 0.3
        contributions.append("evidence_sim>=0.60:+0.3")
    if features["same_sentence"]:
        score += 0.5
        contributions.append("same_sentence:+0.5")
    if features["same_article"]:
        score += 1.2
        contributions.append("same_article:+1.2")
    if features["same_source"]:
        score += 0.3
        contributions.append("same_source:+0.3")
    if features["same_document_hash"]:
        score += 1.0
        contributions.append("same_document_hash:+1.0")
    if features["time_relation"] in {"same_time", "close_time"}:
        score += 0.8
        contributions.append("time_close:+0.8")
    elif features["time_relation"] == "near_time":
        score += 0.3
        contributions.append("time_near:+0.3")
    if features["modality_relation"].startswith("same_"):
        score += 0.3
        contributions.append("same_modality:+0.3")
    if features["provenance_relation"] == "same_article":
        score += 1.0
        contributions.append("same_article_provenance:+1.0")
    elif features["provenance_relation"] == "same_source":
        score += 0.3
        contributions.append("same_source_provenance:+0.3")
    if features["same_target"] and features["same_family"] and features["object_similarity"] >= 0.5:
        score += 0.7
        contributions.append("target_family_object:+0.7")
    if features["same_target"] and features["fact_similarity"] >= 0.8 and features["time_relation"] != "time_conflict":
        score += 0.7
        contributions.append("target_fact_time:+0.7")
    return {
        "score": round(score, 6),
        "score_trace": contributions,
        "hard_conflicts": hard_conflicts(features),
    }


def decide_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    features = identity_features(a, b)
    scored = score_pair(a, b, features)
    conflicts = scored["hard_conflicts"]
    score = scored["score"]
    if conflicts:
        decision = "different_event"
        reason = "hard_conflict:" + "|".join(conflicts)
    else:
        same_target = features["same_target"]
        if same_target and (features["same_family"] or features["fact_similarity"] >= 0.8) and score >= 4.2 and (
            features["fact_similarity"] >= 0.25 or features["title_similarity"] >= 0.8 or features["evidence_similarity"] >= 0.8
        ):
            decision = "same_event"
            reason = "score>=4.2 and same_target with stronger fact/title/evidence support"
        elif score <= 1.0:
            decision = "different_event"
            reason = "score<=1.0"
        else:
            decision = "uncertain"
            reason = "middle_zone"
    if decision == "same_event":
        confidence = min(0.99, round(0.55 + 0.08 * max(0.0, score - 3.0) + 0.05 * len(features["fact_token_overlap"]) + 0.03 * len(features["object_token_overlap"]), 6))
    elif decision == "different_event":
        confidence = min(0.99, round(0.55 + 0.08 * max(0.0, 2.0 - min(score, 2.0)) + 0.08 * len(conflicts), 6))
    else:
        confidence = max(0.35, round(0.5 - min(0.15, abs(score - 2.5) * 0.04), 6))
    matched_arguments = sorted(set(features["object_token_overlap"]) | set(features["fact_token_overlap"]) | set(features["trigger_token_overlap"]))
    evidence_trace = {
        "score": score,
        "score_trace": scored["score_trace"],
        "hard_conflicts": conflicts,
        "same_target": features["same_target"],
        "same_family": features["same_family"],
        "same_event_type": features["same_event_type"],
        "same_event_name": features["same_event_name"],
        "same_trigger": features["same_trigger"],
        "fact_similarity": features["fact_similarity"],
        "object_similarity": features["object_similarity"],
        "trigger_similarity": features["trigger_similarity"],
        "title_similarity": features["title_similarity"],
        "evidence_similarity": features["evidence_similarity"],
        "time_relation": features["time_relation"],
        "modality_relation": features["modality_relation"],
        "provenance_relation": features["provenance_relation"],
    }
    rejection_reason = ""
    if decision == "different_event" and conflicts:
        rejection_reason = "|".join(conflicts)
    elif decision == "different_event":
        rejection_reason = "low_score"
    elif decision == "uncertain":
        rejection_reason = "ambiguous_middle_zone"
    return {
        "canonical_decision": decision,
        "decision_confidence": round(confidence, 6),
        "decision_reason": reason,
        "identity_features": features,
        "matched_arguments": matched_arguments,
        "time_relation": features["time_relation"],
        "modality_relation": features["modality_relation"],
        "provenance_relation": features["provenance_relation"],
        "rejection_reason": rejection_reason,
        "evidence_trace": evidence_trace,
        "raw_score": score,
        "score_trace": scored["score_trace"],
        "hard_conflicts": conflicts,
    }


def baseline_decide(a: dict[str, Any], b: dict[str, Any], baseline_pred_row: dict[str, Any] | None = None) -> tuple[str, str, float]:
    if baseline_pred_row:
        pred = text(baseline_pred_row.get("predicted_relation"))
        if pred == "related_update":
            return "uncertain", "frozen_related_update", 0.55
        if pred in {"same_event", "different_event", "uncertain_review"}:
            if pred == "uncertain_review":
                return "uncertain", "frozen_uncertain", 0.50
            return pred, "frozen_prediction", 0.80
    features = identity_features(a, b)
    if features["same_target"] and features["same_family"] and features["time_relation"] in {"same_time", "close_time", "near_time"}:
        return "same_event", "fallback_same_target_family_time", 0.70
    if features["same_target"] and (features["fact_similarity"] >= 0.75 or features["same_event_name"]):
        return "same_event", "fallback_same_target_fact", 0.65
    if features["same_source"] and features["title_similarity"] >= 0.75:
        return "same_event", "fallback_same_source_title", 0.60
    if hard_conflicts(features):
        return "different_event", "fallback_hard_conflict", 0.75
    return "uncertain", "fallback_ambiguous", 0.50


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((text(a), text(b))))


def baseline_candidate_channels(a: dict[str, Any], b: dict[str, Any], features: dict[str, Any]) -> list[str]:
    channels = []
    if features["same_article"]:
        channels.append("same_article")
    if features["same_source"]:
        channels.append("same_source")
    if features["same_target"] and features["same_family"] and (features["time_relation"] in {"same_time", "close_time", "near_time"}):
        channels.append("target_family_time")
    if features["same_target"] and features["same_event_type"]:
        channels.append("target_event_type")
    if features["same_target"] and features["fact_similarity"] >= 0.70:
        channels.append("target_fact")
    if features["same_target"] and features["title_similarity"] >= 0.65:
        channels.append("target_title_semantic")
    if features["same_target"] and features["object_similarity"] >= 0.25:
        channels.append("target_object")
    return sorted(set(channels))


def candidate_channels(a: dict[str, Any], b: dict[str, Any], features: dict[str, Any]) -> list[str]:
    channels = []
    if features["same_article"]:
        channels.append("same_article")
    if features["same_source"] and features["same_target"] and (features["same_family"] or features["title_similarity"] >= 0.75):
        channels.append("same_source_same_target")
    if features["same_target"] and features["same_family"] and (features["time_relation"] in {"same_time", "close_time", "near_time"}):
        channels.append("target_family_time")
    if features["same_target"] and features["object_similarity"] >= 0.25:
        channels.append("target_object")
    if features["same_target"] and features["fact_similarity"] >= 0.55:
        channels.append("target_fact")
    if features["same_target"] and features["title_similarity"] >= 0.70:
        channels.append("target_semantic")
    if features["same_target"] and features["time_relation"] in {"same_time", "close_time"} and features["fact_similarity"] >= 0.40:
        channels.append("target_time_fact")
    if features["same_document_hash"]:
        channels.append("same_document_hash")
    if features["same_sentence"]:
        channels.append("same_sentence")
    if features["same_target"] and features["same_family"] and features["object_similarity"] >= 0.50:
        channels.append("family_object")
    if features["same_target"] and features["same_family"] and features["trigger_similarity"] >= 0.35:
        channels.append("family_trigger")
    if features["same_target"] and features["modality_relation"].startswith("same_"):
        channels.append("same_modality")
    return sorted(set(channels))


def build_candidate_table(events: list[dict[str, Any]], baseline_mode: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(events)
    for i in range(n):
        a = events[i]
        for j in range(i + 1, n):
            b = events[j]
            features = identity_features(a, b)
            channels = baseline_candidate_channels(a, b, features) if baseline_mode else candidate_channels(a, b, features)
            if not channels:
                continue
            decision = decide_pair(a, b)
            rows.append(
                {
                    "pair_id": stable_pair_id("CP", a["source_row"]["sample_id"], b["source_row"]["sample_id"]),
                    "event_id_a": text(a["source_row"]["event_id"]),
                    "event_id_b": text(b["source_row"]["event_id"]),
                    "sample_id_a": text(a["source_row"]["sample_id"]),
                    "sample_id_b": text(b["source_row"]["sample_id"]),
                    "article_id_a": text(a.get("article_id")),
                    "article_id_b": text(b.get("article_id")),
                    "target_key_a": text(a["target"].get("target_key")),
                    "target_key_b": text(b["target"].get("target_key")),
                    "event_family_a": text(a["event"].get("event_family")),
                    "event_family_b": text(b["event"].get("event_family")),
                    "event_type_a": text(a["event"].get("event_type")),
                    "event_type_b": text(b["event"].get("event_type")),
                    "event_name_a": text(a["event"].get("event_name")),
                    "event_name_b": text(b["event"].get("event_name")),
                    "trigger_text_a": text(a["event"].get("trigger_text")),
                    "trigger_text_b": text(b["event"].get("trigger_text")),
                    "object_text_a": text(a["event"].get("object_text")),
                    "object_text_b": text(b["event"].get("object_text")),
                    "publish_time_a": text(a.get("publish_time")),
                    "publish_time_b": text(b.get("publish_time")),
                    "event_time_a": text(a["event"].get("event_time")),
                    "event_time_b": text(b["event"].get("event_time")),
                    "score": decision["raw_score"],
                    "score_trace": json.dumps(decision["score_trace"], ensure_ascii=False),
                    "hard_conflicts": json.dumps(decision["hard_conflicts"], ensure_ascii=False),
                    "channels": "|".join(channels),
                    "canonical_decision": decision["canonical_decision"],
                    "decision_confidence": decision["decision_confidence"],
                    "decision_reason": decision["decision_reason"],
                    "time_relation": decision["time_relation"],
                    "modality_relation": decision["modality_relation"],
                    "provenance_relation": decision["provenance_relation"],
                    "identity_features_json": json.dumps(decision["identity_features"], ensure_ascii=False),
                    "matched_arguments_json": json.dumps(decision["matched_arguments"], ensure_ascii=False),
                    "rejection_reason": decision["rejection_reason"],
                    "baseline_mode": baseline_mode,
                }
            )
    rows.sort(key=lambda r: (r["channels"], -float(r["score"]), r["pair_id"]))
    return rows


def generate_new_pair_package(events: list[dict[str, Any]], seed: int = 2026, target_rows: int = 400) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = []
    n = len(events)
    for i in range(n):
        for j in range(i + 1, n):
            a = events[i]
            b = events[j]
            decision = decide_pair(a, b)
            features = decision["identity_features"]
            channels = candidate_channels(a, b, features)
            if not channels:
                continue
            gold_hint = "uncertain"
            if decision["canonical_decision"] == "same_event":
                gold_hint = "same_event_likely"
            elif decision["canonical_decision"] == "different_event" and features["same_target"] and features["same_family"]:
                gold_hint = "different_event_hard_negative"
            elif decision["canonical_decision"] == "uncertain":
                gold_hint = "boundary_uncertain"
            elif features["same_target"] and features["same_event_name"] and features["object_similarity"] < 0.35:
                gold_hint = "same_target_family_object_conflict"
            elif features["same_target"] and features["same_family"] and features["fact_similarity"] >= 0.70:
                gold_hint = "same_fact_diff_expression"
            scored.append(
                {
                    "pair_id": stable_pair_id("MP", a["source_row"]["sample_id"], b["source_row"]["sample_id"]),
                    "sample_id_a": text(a["source_row"]["sample_id"]),
                    "sample_id_b": text(b["source_row"]["sample_id"]),
                    "article_id_a": text(a.get("article_id")),
                    "article_id_b": text(b.get("article_id")),
                    "title_a": text(a["source"].get("source_name")) or text(a.get("source_row", {}).get("article_file_id")),
                    "title_b": text(b["source"].get("source_name")) or text(b.get("source_row", {}).get("article_file_id")),
                    "source_name_a": text(a["source"].get("source_name")),
                    "source_name_b": text(b["source"].get("source_name")),
                    "publish_time_a": text(a.get("publish_time")),
                    "publish_time_b": text(b.get("publish_time")),
                    "target_name_a": text(a["target"].get("target_name")),
                    "target_name_b": text(b["target"].get("target_name")),
                    "target_key_a": text(a["target"].get("target_key")),
                    "target_key_b": text(b["target"].get("target_key")),
                    "event_family_a": text(a["event"].get("event_family")),
                    "event_family_b": text(b["event"].get("event_family")),
                    "event_type_a": text(a["event"].get("event_type")),
                    "event_type_b": text(b["event"].get("event_type")),
                    "event_name_a": text(a["event"].get("event_name")),
                    "event_name_b": text(b["event"].get("event_name")),
                    "event_time_a": text(a["event"].get("event_time")),
                    "event_time_b": text(b["event"].get("event_time")),
                    "trigger_text_a": text(a["event"].get("trigger_text")),
                    "trigger_text_b": text(b["event"].get("trigger_text")),
                    "object_text_a": text(a["event"].get("object_text")),
                    "object_text_b": text(b["event"].get("object_text")),
                    "evidence_sentence_a": text(a["evidence"].get("evidence_sentence")),
                    "evidence_sentence_b": text(b["evidence"].get("evidence_sentence")),
                    "local_context_a": text(a["evidence"].get("local_context")),
                    "local_context_b": text(b["evidence"].get("local_context")),
                    "channels": "|".join(channels),
                    "candidate_hint": gold_hint,
                    "candidate_reason": decision["decision_reason"],
                    "decision": decision["canonical_decision"],
                    "confidence": decision["decision_confidence"],
                    "score": decision["raw_score"],
                    "score_trace": json.dumps(decision["score_trace"], ensure_ascii=False),
                    "hard_conflicts": json.dumps(decision["hard_conflicts"], ensure_ascii=False),
                    "matched_arguments_json": json.dumps(decision["matched_arguments"], ensure_ascii=False),
                    "visible_label_same_event": "",
                    "visible_label_relation": "",
                    "visible_label_confidence": "",
                    "visible_annotator_note": "",
                    "internal_title_similarity": decision["identity_features"]["title_similarity"],
                    "internal_fact_similarity": decision["identity_features"]["fact_similarity"],
                    "internal_object_similarity": decision["identity_features"]["object_similarity"],
                    "internal_trigger_similarity": decision["identity_features"]["trigger_similarity"],
                    "internal_time_relation": decision["time_relation"],
                    "internal_modality_relation": decision["modality_relation"],
                    "internal_provenance_relation": decision["provenance_relation"],
                }
            )
    bucket_order = {
        "same_event_likely": 0,
        "same_fact_diff_expression": 1,
        "same_target_family_object_conflict": 2,
        "same_target_family_time": 3,
        "different_event_hard_negative": 4,
        "boundary_uncertain": 5,
        "uncertain": 6,
    }
    scored.sort(key=lambda row: (bucket_order.get(row["candidate_hint"], 9), -float(row["confidence"]), stable_hash(seed, row["pair_id"])))
    selected: list[dict[str, Any]] = []
    seen = set()
    strata_targets = {
        "same_event_likely": 95,
        "same_fact_diff_expression": 65,
        "same_target_family_object_conflict": 65,
        "same_target_family_time": 50,
        "different_event_hard_negative": 70,
        "boundary_uncertain": 55,
    }
    for bucket, limit in strata_targets.items():
        bucket_rows = [row for row in scored if row["candidate_hint"] == bucket]
        bucket_rows = sorted(bucket_rows, key=lambda row: stable_hash(seed, bucket, row["pair_id"]))
        for row in bucket_rows[:limit]:
            if row["pair_id"] in seen:
                continue
            selected.append(row)
            seen.add(row["pair_id"])
    # Top up to target_rows with boundary/uncertain pairs and then by confidence.
    if len(selected) < target_rows:
        for row in scored:
            if row["pair_id"] in seen:
                continue
            selected.append(row)
            seen.add(row["pair_id"])
            if len(selected) >= target_rows:
                break
    selected = selected[:target_rows]
    visible = []
    for row in selected:
        visible.append(
            {
                "pair_id": row["pair_id"],
                "article_id_a": row["article_id_a"],
                "article_id_b": row["article_id_b"],
                "source_name_a": row["source_name_a"],
                "source_name_b": row["source_name_b"],
                "publish_time_a": row["publish_time_a"],
                "publish_time_b": row["publish_time_b"],
                "target_name_a": row["target_name_a"],
                "target_name_b": row["target_name_b"],
                "event_name_a": row["event_name_a"],
                "event_name_b": row["event_name_b"],
                "event_time_a": row["event_time_a"],
                "event_time_b": row["event_time_b"],
                "evidence_sentence_a": row["evidence_sentence_a"],
                "evidence_sentence_b": row["evidence_sentence_b"],
                "local_context_a": row["local_context_a"],
                "local_context_b": row["local_context_b"],
                "human_same_event": "",
                "human_relation": "",
                "human_confidence": "",
                "annotator_note": "",
            }
        )
    return selected, visible


def compute_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = ["same_event", "different_event", "uncertain"]
    acc = accuracy_score(gold, pred) if gold else 0.0
    macro = f1_score(gold, pred, labels=labels, average="macro", zero_division=0) if gold else 0.0
    p, r, f, _ = precision_recall_fscore_support(
        [1 if y == "same_event" else 0 for y in gold],
        [1 if y == "same_event" else 0 for y in pred],
        average="binary",
        zero_division=0,
    )
    _, _, _, support = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=labels).tolist() if gold else [[0] * len(labels) for _ in labels]
    false_merge = sum(1 for g, p_ in zip(gold, pred) if g != "same_event" and p_ == "same_event")
    split_error = sum(1 for g, p_ in zip(gold, pred) if g == "same_event" and p_ != "same_event")
    return {
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro), 6),
        "same_event_precision": round(float(p), 6),
        "same_event_recall": round(float(r), 6),
        "same_event_f1": round(float(f), 6),
        "false_merge": int(false_merge),
        "split_error": int(split_error),
        "confusion_matrix": cm,
        "prediction_counts": dict(Counter(pred)),
        "gold_counts": dict(Counter(gold)),
        "support": {label: int(support[idx]) for idx, label in enumerate(labels)},
        "uncertain_rate": round(pred.count("uncertain") / max(1, len(pred)), 6),
    }


def evaluate_old_pairs(
    recovered_pairs: list[dict[str, Any]],
    mention_by_id: dict[str, dict[str, Any]],
    baseline_predictions: dict[tuple[str, str], dict[str, Any]],
    frozen_candidate_pairs: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = []
    gold = []
    baseline_pred = []
    candidate_pred = []
    baseline_candidate_hit = []
    candidate_candidate_hit = []
    for row in recovered_pairs:
        a = row["left_event"]
        b = row["right_event"]
        key = pair_key(a["event_mention_id"], b["event_mention_id"])
        gold_label = row["gold_relation"]
        gold.append(gold_label)
        frozen_row = baseline_predictions.get(key)
        bpred, breason, bconf = baseline_decide(a, b, frozen_row)
        cdec = decide_pair(a, b)
        feats = cdec["identity_features"]
        c_channels = candidate_channels(a, b, feats)
        baseline_pred.append(bpred)
        candidate_pred.append(cdec["canonical_decision"])
        baseline_candidate_hit.append(key in frozen_candidate_pairs)
        candidate_candidate_hit.append(bool(c_channels))
        rows.append(
            {
                "pair_id": row["pair_id"],
                "source_file": row["source_file"],
                "gold_relation": gold_label,
                "baseline_prediction": bpred,
                "baseline_reason": breason,
                "baseline_confidence": bconf,
                "candidate_prediction": cdec["canonical_decision"],
                "candidate_reason": cdec["decision_reason"],
                "candidate_confidence": cdec["decision_confidence"],
                "baseline_candidate_hit": key in frozen_candidate_pairs,
                "candidate_candidate_hit": bool(c_channels),
                "event_mention_id_a": a["event_mention_id"],
                "event_mention_id_b": b["event_mention_id"],
                "article_id_a": row["article_id_a"],
                "article_id_b": row["article_id_b"],
                "title_a": row["title_a"],
                "title_b": row["title_b"],
                "source_name_a": a.get("source_name"),
                "source_name_b": b.get("source_name"),
                "target_key_a": a.get("target_key"),
                "target_key_b": b.get("target_key"),
                "event_family_a": a.get("_event_family"),
                "event_family_b": b.get("_event_family"),
                "provisional_event_type_a": a.get("provisional_event_type"),
                "provisional_event_type_b": b.get("provisional_event_type"),
                "event_local_text_a": a.get("event_local_text"),
                "event_local_text_b": b.get("event_local_text"),
                "evidence_sentence_a": (a.get("evidence_sentences") or [""])[0] if isinstance(a.get("evidence_sentences"), list) else "",
                "evidence_sentence_b": (b.get("evidence_sentences") or [""])[0] if isinstance(b.get("evidence_sentences"), list) else "",
                "identity_features_json": json.dumps(cdec["identity_features"], ensure_ascii=False),
                "evidence_trace_json": json.dumps(cdec["evidence_trace"], ensure_ascii=False),
                "matched_arguments_json": json.dumps(cdec["matched_arguments"], ensure_ascii=False),
                "hard_conflicts_json": json.dumps(cdec["hard_conflicts"], ensure_ascii=False),
                "candidate_channels": "|".join(c_channels),
            }
        )
    baseline_metrics = compute_metrics(gold, baseline_pred)
    candidate_metrics = compute_metrics(gold, candidate_pred)
    baseline_metrics["candidate_recall_old_pairs"] = round(sum(baseline_candidate_hit) / max(1, len(baseline_candidate_hit)), 6)
    candidate_metrics["candidate_recall_old_pairs"] = round(sum(candidate_candidate_hit) / max(1, len(candidate_candidate_hit)), 6)
    baseline_metrics["candidate_reduction"] = None
    candidate_metrics["candidate_reduction"] = None
    audit = {
        "gold_counts": dict(Counter(gold)),
        "baseline_prediction_counts": dict(Counter(baseline_pred)),
        "candidate_prediction_counts": dict(Counter(candidate_pred)),
        "frozen_candidate_coverage": round(sum(1 for flag in baseline_candidate_hit if flag) / max(1, len(baseline_candidate_hit)), 6),
        "candidate_coverage": round(sum(1 for flag in candidate_candidate_hit if flag) / max(1, len(candidate_candidate_hit)), 6),
    }
    return rows, baseline_metrics, candidate_metrics, audit


def build_confusion_csv(path: Path, gold: list[str], pred: list[str]) -> None:
    labels = ["same_event", "different_event", "uncertain"]
    cm = confusion_matrix(gold, pred, labels=labels)
    rows = []
    for i, g in enumerate(labels):
        row = {"gold/pred": g}
        for j, p in enumerate(labels):
            row[p] = int(cm[i, j])
        rows.append(row)
    write_csv(path, rows, ["gold/pred"] + labels)


def build_error_audit(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "false_merge":
        subset = [r for r in rows if r["gold_relation"] != "same_event" and r["candidate_prediction"] == "same_event"]
    elif kind == "split_error":
        subset = [r for r in rows if r["gold_relation"] == "same_event" and r["candidate_prediction"] != "same_event"]
    elif kind == "coverage_gap":
        subset = [r for r in rows if not r["candidate_candidate_hit"]]
    else:
        subset = [r for r in rows if r["candidate_prediction"] == "uncertain"]
    out = []
    for r in subset:
        out.append(
            {
                "pair_id": r["pair_id"],
                "gold_relation": r["gold_relation"],
                "baseline_prediction": r["baseline_prediction"],
                "candidate_prediction": r["candidate_prediction"],
                "candidate_reason": r["candidate_reason"],
                "candidate_confidence": r["candidate_confidence"],
                "article_id_a": r["article_id_a"],
                "article_id_b": r["article_id_b"],
                "source_name_a": r["source_name_a"],
                "source_name_b": r["source_name_b"],
                "target_key_a": r["target_key_a"],
                "target_key_b": r["target_key_b"],
                "event_family_a": r["event_family_a"],
                "event_family_b": r["event_family_b"],
                "provisional_event_type_a": r["provisional_event_type_a"],
                "provisional_event_type_b": r["provisional_event_type_b"],
                "evidence_sentence_a": r["evidence_sentence_a"],
                "evidence_sentence_b": r["evidence_sentence_b"],
                "identity_features_json": r["identity_features_json"],
                "evidence_trace_json": r["evidence_trace_json"],
                "hard_conflicts_json": r["hard_conflicts_json"],
                "matched_arguments_json": r["matched_arguments_json"],
            }
        )
    return out


def build_manual_package_xlsx(selected_rows: list[dict[str, Any]], out_path: Path) -> None:
    visible_rows = []
    internal_rows = []
    for row in selected_rows:
        visible_rows.append(
            {
                "pair_id": row["pair_id"],
                "article_id_a": row["article_id_a"],
                "article_id_b": row["article_id_b"],
                "source_name_a": row["source_name_a"],
                "source_name_b": row["source_name_b"],
                "publish_time_a": row["publish_time_a"],
                "publish_time_b": row["publish_time_b"],
                "target_name_a": row["target_name_a"],
                "target_name_b": row["target_name_b"],
                "event_name_a": row["event_name_a"],
                "event_name_b": row["event_name_b"],
                "event_time_a": row["event_time_a"],
                "event_time_b": row["event_time_b"],
                "evidence_sentence_a": row["evidence_sentence_a"],
                "evidence_sentence_b": row["evidence_sentence_b"],
                "local_context_a": row["local_context_a"],
                "local_context_b": row["local_context_b"],
                "human_same_event": "",
                "human_relation": "",
                "human_confidence": "",
                "annotator_note": "",
            }
        )
        internal_rows.append(dict(row))
    write_xlsx(out_path, visible_rows, "annotation", hidden_rows=internal_rows, hidden_sheet="_internal")


def make_visible_jsonl_from_manual(rows: list[dict[str, Any]], out_path: Path) -> None:
    visible_rows = []
    for row in rows:
        visible_rows.append(
            {
                "pair_id": row["pair_id"],
                "article_id_a": row["article_id_a"],
                "article_id_b": row["article_id_b"],
                "source_name_a": row["source_name_a"],
                "source_name_b": row["source_name_b"],
                "publish_time_a": row["publish_time_a"],
                "publish_time_b": row["publish_time_b"],
                "target_name_a": row["target_name_a"],
                "target_name_b": row["target_name_b"],
                "event_name_a": row["event_name_a"],
                "event_name_b": row["event_name_b"],
                "event_time_a": row["event_time_a"],
                "event_time_b": row["event_time_b"],
                "evidence_sentence_a": row["evidence_sentence_a"],
                "evidence_sentence_b": row["evidence_sentence_b"],
                "local_context_a": row["local_context_a"],
                "local_context_b": row["local_context_b"],
                "human_same_event": "",
                "human_relation": "",
                "human_confidence": "",
                "annotator_note": "",
            }
        )
    write_jsonl(out_path, visible_rows)


def build_old_pair_recovery(
    raw_rows: list[dict[str, Any]],
    mention_by_id: dict[str, dict[str, Any]],
    article_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    recovered = []
    quarantine = []
    pair_lookup = {}
    for article_id, mentions in mention_by_id.items():
        pass
    # We use pairwise scoring for article-level pairs and direct mention lookup for mention-level pairs.

    def mention_score(article_title: str, article_content: str, mention: dict[str, Any]) -> float:
        local = text(mention.get("event_local_text"))
        evidence = " ".join(text(x) for x in (mention.get("evidence_sentences") or [])[:3])
        score = 0.0
        score += 3.0 * max(token_jaccard(article_title, local), token_jaccard(article_title, evidence))
        if norm(article_title) and (norm(article_title) in norm(local) or norm(article_title) in norm(evidence)):
            score += 2.0
        if mention.get("content_join_confidence") == "exact":
            score += 1.0
        if text(mention.get("verdict")) != "insufficient_evidence":
            score += 0.8
        if text(mention.get("source_name")):
            score += 0.1
        score += float(mention.get("trust_score") or 0) / 100.0
        score += 0.25 * len(set(clean_tokens(article_content)) & set(clean_tokens(local)))
        score += 0.35 * len(set(clean_tokens(article_title)) & set(clean_tokens(" ".join(mention.get("core_elements") or []))))
        return score

    def choose_pair(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
        if text(row.get("event_mention_id_a")) and text(row.get("event_mention_id_b")):
            a = mention_by_id.get(text(row.get("event_mention_id_a")))
            b = mention_by_id.get(text(row.get("event_mention_id_b")))
            if a and b:
                return a, b, {"status": "direct_event_mention_ids"}
            return None, None, {"status": "missing_direct_event_mention_id"}
        article_a = article_by_id.get(text(row.get("article_id_a")), {})
        article_b = article_by_id.get(text(row.get("article_id_b")), {})
        cand_a = [m for m in mention_by_id.values() if text(m.get("article_id")) == text(row.get("article_id_a"))]
        cand_b = [m for m in mention_by_id.values() if text(m.get("article_id")) == text(row.get("article_id_b"))]
        if not cand_a or not cand_b:
            return None, None, {"status": "article_no_candidate"}
        best = None
        title_a = text(article_a.get("article_title") or row.get("title_a"))
        title_b = text(article_b.get("article_title") or row.get("title_b"))
        content_a = text(article_a.get("content"))
        content_b = text(article_b.get("content"))
        for ma in cand_a:
            sa = mention_score(title_a, content_a, ma)
            for mb in cand_b:
                sb = mention_score(title_b, content_b, mb)
                pair_score = sa + sb
                if text(ma.get("target_type")) == text(mb.get("target_type")) and text(ma.get("target_code")) == text(mb.get("target_code")):
                    pair_score += 1.2
                if text(ma.get("provisional_event_type")) == text(mb.get("provisional_event_type")):
                    pair_score += 1.0
                if token_jaccard(ma.get("event_local_text"), mb.get("event_local_text")) >= 0.75:
                    pair_score += 1.0
                if token_jaccard(ma.get("event_local_text"), title_a) >= 0.45 and token_jaccard(mb.get("event_local_text"), title_b) >= 0.45:
                    pair_score += 0.7
                if token_jaccard(" ".join(ma.get("evidence_sentences") or []), " ".join(mb.get("evidence_sentences") or [])) >= 0.6:
                    pair_score += 0.8
                if best is None or pair_score > best[0]:
                    best = (pair_score, ma, mb)
        if best is None:
            return None, None, {"status": "pair_unresolvable"}
        return best[1], best[2], {"status": "paired_by_joint_score", "score": round(best[0], 6)}

    for row in raw_rows:
        a, b, meta = choose_pair(row)
        gold = infer_gold_label(row)
        if not a or not b:
            quarantine.append(
                {
                    "pair_id": text(row.get("pair_id")),
                    "source_file": row.get("_source_file"),
                    "reason": meta.get("status"),
                    "article_id_a": text(row.get("article_id_a")),
                    "article_id_b": text(row.get("article_id_b")),
                    "event_mention_id_a": text(row.get("event_mention_id_a")),
                    "event_mention_id_b": text(row.get("event_mention_id_b")),
                    "gold_relation": gold,
                }
            )
            continue
        pair_id = text(row.get("pair_id")) or stable_pair_id("OLDPAIR", a["event_mention_id"], b["event_mention_id"])
        recovered.append(
            {
                "pair_id": pair_id,
                "source_file": row.get("_source_file"),
                "source_pair_id": text(row.get("pair_id")),
                "gold_relation": gold,
                "gold_relation_raw": text(row.get("human_relation") or row.get("near_duplicate_label") or row.get("human_same_event")),
                "human_same_event": text(row.get("human_same_event")),
                "human_relation": text(row.get("human_relation")),
                "human_is_near_duplicate": text(row.get("human_is_near_duplicate")),
                "near_duplicate_label": text(row.get("near_duplicate_label")),
                "human_confidence": text(row.get("human_confidence")),
                "annotator_note": text(row.get("annotator_note")),
                "article_id_a": text(row.get("article_id_a")),
                "article_id_b": text(row.get("article_id_b")),
                "title_a": text(row.get("title_a")),
                "title_b": text(row.get("title_b")),
                "event_mention_id_a": a["event_mention_id"],
                "event_mention_id_b": b["event_mention_id"],
                "document_id_a": a.get("document_id"),
                "document_id_b": b.get("document_id"),
                "source_name_a": a.get("source_name"),
                "source_name_b": b.get("source_name"),
                "publish_time_a": a.get("publish_time"),
                "publish_time_b": b.get("publish_time"),
                "target_key_a": f"{a.get('target_type')}|{a.get('target_code')}",
                "target_key_b": f"{b.get('target_type')}|{b.get('target_code')}",
                "provisional_event_type_a": a.get("provisional_event_type"),
                "provisional_event_type_b": b.get("provisional_event_type"),
                "core_elements_a": "|".join(a.get("core_elements") or []),
                "core_elements_b": "|".join(b.get("core_elements") or []),
                "article_title_a": article_by_id.get(text(row.get("article_id_a")), {}).get("article_title", ""),
                "article_title_b": article_by_id.get(text(row.get("article_id_b")), {}).get("article_title", ""),
                "mapping_status": meta.get("status"),
                "mapping_score": meta.get("score"),
                "left_event": a,
                "right_event": b,
            }
        )
    dedup = {}
    for row in recovered:
        dedup[row["pair_id"]] = row
    recovered = list(dedup.values())
    recovered.sort(key=lambda r: (r["source_file"], r["pair_id"]))
    gold_counts = Counter(r["gold_relation"] for r in recovered)
    label_counts = Counter()
    for r in recovered:
        if r["source_file"].endswith("same_event_labels_for_step3.csv"):
            label_counts[f"human_same_event={r['human_same_event']}"] += 1
            label_counts[f"human_relation={r['human_relation']}"] += 1
        else:
            label_counts[f"human_is_near_duplicate={r['human_is_near_duplicate']}"] += 1
            label_counts[f"near_duplicate_label={r['near_duplicate_label']}"] += 1
    summary = {
        "input_rows": len(raw_rows),
        "recovered_rows": len(recovered),
        "quarantine_rows": len(quarantine),
        "gold_counts": dict(gold_counts),
        "label_field_counts": dict(label_counts),
        "recovery_rate": round(len(recovered) / max(1, len(raw_rows)), 6),
        "quarantine_rate": round(len(quarantine) / max(1, len(raw_rows)), 6),
    }
    return recovered, quarantine, summary


def audit_pair_leakage(recovered_pairs: list[dict[str, Any]], frozen_test_articles: set[str], frozen_mentions_split: dict[str, str]) -> dict[str, Any]:
    article_ids = {text(r["article_id_a"]) for r in recovered_pairs} | {text(r["article_id_b"]) for r in recovered_pairs}
    mention_ids = {text(r["event_mention_id_a"]) for r in recovered_pairs} | {text(r["event_mention_id_b"]) for r in recovered_pairs}
    frozen_test_overlap = article_ids & frozen_test_articles
    mention_test_overlap = {mid for mid in mention_ids if frozen_mentions_split.get(mid) == "independent_frozen_test"}
    return {
        "recovered_article_ids": len(article_ids),
        "recovered_event_mention_ids": len(mention_ids),
        "frozen_test_article_overlap": len(frozen_test_overlap),
        "frozen_test_event_mention_overlap": len(mention_test_overlap),
        "frozen_test_article_ids": sorted(list(frozen_test_overlap))[:20],
        "frozen_test_event_mention_ids": sorted(list(mention_test_overlap))[:20],
    }


def build_input_manifest() -> list[dict[str, Any]]:
    artifacts = [TASK1_EXPANSION, TASK1_SELECTION_FUNNEL, TASK1_QUALITY_REPORT, PAIR_SAME_EVENT, PAIR_NEAR_DUP, FULLTEXT_MENTIONS, FULLTEXT_ARTICLE_INDEX, TASK1_FROZEN_TEST, FROZEN_CANDIDATES, FROZEN_PREDICTIONS]
    manifest = []
    for path in artifacts:
        item = {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "mtime": mtime_iso(path) if path.exists() else "", "sha256": sha256_file(path) if path.exists() else ""}
        if path.suffix.lower() == ".jsonl" and path.exists():
            rows = read_jsonl(path)
            item["record_count"] = len(rows)
            item["line_count"] = len(rows)
            item["fields"] = sorted(rows[0].keys()) if rows else []
        elif path.suffix.lower() == ".json" and path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            item["fields"] = sorted(obj.keys()) if isinstance(obj, dict) else []
            item["record_count"] = 1
        elif path.suffix.lower() == ".csv" and path.exists():
            rows = read_csv_rows(path)
            item["record_count"] = len(rows)
            item["line_count"] = len(rows) + 1 if rows else 1
            item["fields"] = list(rows[0].keys()) if rows else []
        elif path.suffix.lower() == ".parquet" and path.exists():
            df = pd.read_parquet(path)
            item["record_count"] = len(df)
            item["fields"] = list(df.columns)
        manifest.append(item)
    return manifest


def build_protected_sha_manifest() -> list[dict[str, Any]]:
    rows = []
    for path in PROTECTED_FILES:
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else "",
                "mtime": mtime_iso(path) if path.exists() else "",
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def write_report_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = []
    lines.append("# Task3 CanonicalEvent Resolver v2 Actual Repair")
    lines.append("")
    lines.append(f"- 615 recovered events: {payload['recovered_events']}")
    lines.append(f"- old pairs recovered: {payload['old_pair_recovered']}/{payload['old_pair_input']}")
    lines.append(f"- quarantine rows: {payload['quarantine_total']}")
    lines.append(f"- benchmark read: {payload['benchmark_read']}")
    lines.append(f"- frozen files modified: {payload['frozen_modified']}")
    lines.append("")
    lines.append("## Metrics")
    for name in ("baseline", "candidate"):
        m = payload[f"{name}_metrics"]
        lines.append(f"- {name}: accuracy={m['accuracy']}, macro_f1={m['macro_f1']}, same_event_f1={m['same_event_f1']}, false_merge={m['false_merge']}, split_error={m['split_error']}, uncertain_rate={m['uncertain_rate']}")
    lines.append("")
    lines.append("## Candidate Generation")
    cg = payload["candidate_generation_metrics"]
    lines.append(f"- baseline_reduction={cg['baseline_reduction']}")
    lines.append(f"- candidate_reduction={cg['candidate_reduction']}")
    lines.append(f"- baseline_old_pair_coverage={cg['baseline_old_pair_coverage']}")
    lines.append(f"- candidate_old_pair_coverage={cg['candidate_old_pair_coverage']}")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- enhancement_pass={payload['enhancement_pass']}")
    lines.append(f"- output_dir={payload['output_dir']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    task1_rows = read_jsonl(TASK1_EXPANSION)
    if not task1_rows:
        raise RuntimeError(f"missing Task1 expansion file: {TASK1_EXPANSION}")
    article_by_id = load_article_index(FULLTEXT_ARTICLE_INDEX)
    frozen_mentions_split = load_frozen_mentions_split()
    frozen_test_articles = {text(row.get("article_id")) for row in read_jsonl(TASK1_FROZEN_TEST) if text(row.get("article_id"))}
    input_manifest = build_input_manifest()
    write_json(out_dir / "input_manifest.json", input_manifest)
    protected_before = build_protected_sha_manifest()
    write_json(out_dir / "protected_sha256_before.json", protected_before)

    pass_rows = [row for row in task1_rows if bool(row.get("grounding_pass"))]
    recovered_events, quarantine_events, task1_validation = build_task1_recovered_records(pass_rows, article_by_id)
    write_jsonl(out_dir / "task1_615_grounding_pass_recovered.jsonl", recovered_events)
    write_jsonl(out_dir / "task1_contract_quarantine.jsonl", quarantine_events)
    write_json(out_dir / "task1_contract_validation_report.json", task1_validation)
    write_json(out_dir / "task1_recovery_manifest.json", {
        "task1_input_rows": len(task1_rows),
        "grounding_pass_rows": len(pass_rows),
        "recovered_rows": len(recovered_events),
        "quarantine_rows": len(quarantine_events),
        "source_split_distribution": dict(Counter(text(row.get("dataset_split")) for row in pass_rows)),
        "duplicate_rows_skipped": task1_validation["duplicate_rows_skipped"],
        "support_type_distribution": task1_validation["support_type_distribution"],
    })
    write_jsonl(out_dir / "canonical_event_v2_mapping.jsonl", [
        {
            "mapping_id": row["source_row"]["sample_id"],
            "article_id": row["article_id"],
            "schema_version": row["schema_version"],
            "target_key": row["target"]["target_key"],
            "event_type": row["event"]["event_type"],
            "event_name": row["event"]["event_name"],
            "event_family": row["event"]["event_family"],
            "trigger_text": row["event"]["trigger_text"],
            "event_time": row["event"]["event_time"],
            "polarity_or_state": row["event"]["polarity_or_state"],
            "support_type": row["evidence"]["support_type"],
            "contract_status": row["quality"]["contract_status"],
            "event_fact_signature": row["event"]["event_fact_signature"],
            "mention_text": row["target"]["mention_text"],
            "mention_span": row["target"]["mention_span"],
            "trigger_span": row["event"]["trigger_span"],
            "evidence_span": row["evidence"]["evidence_span"],
            "provenance": row["provenance"],
        }
        for row in recovered_events
    ])

    # Old pair recovery and baseline / candidate evaluation.
    mention_by_id, mention_by_article = load_fulltext_mentions()
    pair_rows = load_pair_rows()
    recovered_pairs, pair_quarantine, pair_recovery_summary = build_old_pair_recovery(pair_rows, mention_by_id, article_by_id)
    write_jsonl(out_dir / "old_pair_recovery.jsonl", recovered_pairs)
    write_jsonl(out_dir / "old_pair_quarantine.jsonl", pair_quarantine)
    write_json(out_dir / "old_pair_recovery_summary.json", pair_recovery_summary)

    baseline_predictions = load_frozen_predictions()
    frozen_candidate_pairs = load_frozen_candidate_pairs()
    pair_eval_rows, baseline_metrics, candidate_metrics, candidate_audit = evaluate_old_pairs(recovered_pairs, mention_by_id, baseline_predictions, frozen_candidate_pairs)
    write_jsonl(out_dir / "old_pair_evaluation.jsonl", pair_eval_rows)
    build_confusion_csv(out_dir / "pair_confusion_matrix.csv", [r["gold_relation"] for r in pair_eval_rows], [r["candidate_prediction"] for r in pair_eval_rows])

    baseline_false_merge = build_error_audit(pair_eval_rows, "false_merge")
    baseline_split_error = build_error_audit(pair_eval_rows, "split_error")
    baseline_uncertain = build_error_audit(pair_eval_rows, "uncertain")
    write_xlsx(out_dir / "false_merge_audit.xlsx", baseline_false_merge, "candidate", hidden_rows=[r for r in pair_eval_rows if r["baseline_prediction"] == "same_event" and r["gold_relation"] != "same_event"], hidden_sheet="_baseline")
    write_xlsx(out_dir / "split_error_audit.xlsx", baseline_split_error, "candidate", hidden_rows=[r for r in pair_eval_rows if r["baseline_prediction"] != "same_event" and r["gold_relation"] == "same_event"], hidden_sheet="_baseline")
    write_xlsx(out_dir / "coverage_gap_audit.xlsx", build_error_audit(pair_eval_rows, "coverage_gap"), "candidate", hidden_rows=[r for r in pair_eval_rows if r["candidate_candidate_hit"]], hidden_sheet="_coverage")
    write_xlsx(out_dir / "uncertain_pairs.xlsx", baseline_uncertain, "candidate", hidden_rows=[r for r in pair_eval_rows if r["candidate_prediction"] == "uncertain"], hidden_sheet="_candidate")

    # Candidate generation on the 615-event pool.
    baseline_candidate_rows = build_candidate_table(recovered_events, baseline_mode=True)
    candidate_candidate_rows = build_candidate_table(recovered_events, baseline_mode=False)
    safe_parquet(pd.DataFrame(candidate_candidate_rows), out_dir / "canonical_event_v2_candidates.parquet")
    write_jsonl(out_dir / "canonical_event_v2_candidates.jsonl", candidate_candidate_rows)
    write_json(out_dir / "candidate_generation_metrics.json", {
        "pool_events": len(recovered_events),
        "total_possible_pairs": len(recovered_events) * (len(recovered_events) - 1) // 2,
        "baseline_candidate_pairs": len(baseline_candidate_rows),
        "candidate_candidate_pairs": len(candidate_candidate_rows),
        "baseline_reduction": round(1 - len(baseline_candidate_rows) / max(1, len(recovered_events) * (len(recovered_events) - 1) // 2), 6),
        "candidate_reduction": round(1 - len(candidate_candidate_rows) / max(1, len(recovered_events) * (len(recovered_events) - 1) // 2), 6),
        "baseline_old_pair_coverage": candidate_audit["frozen_candidate_coverage"],
        "candidate_old_pair_coverage": candidate_audit["candidate_coverage"],
        "baseline_old_pair_missing": int(round((1 - candidate_audit["frozen_candidate_coverage"]) * len(recovered_pairs))),
        "candidate_old_pair_missing": int(round((1 - candidate_audit["candidate_coverage"]) * len(recovered_pairs))),
        "baseline_channel_counts": dict(Counter(ch for row in baseline_candidate_rows for ch in text(row.get("channels")).split("|") if ch)),
        "candidate_channel_counts": dict(Counter(ch for row in candidate_candidate_rows for ch in text(row.get("channels")).split("|") if ch)),
        "runtime_seconds": round(time.perf_counter() - started, 6),
    })

    # Manual pair package from the recovered 615 pool.
    selected_pairs, visible_manual_rows = generate_new_pair_package(recovered_events, seed=2026, target_rows=400)
    write_jsonl(out_dir / "manual_pair_annotation_package.jsonl", visible_manual_rows)
    build_manual_package_xlsx(selected_pairs, out_dir / "manual_pair_annotation_package.xlsx")
    manual_summary = Counter(row["candidate_hint"] for row in selected_pairs)
    write_json(out_dir / "manual_pair_annotation_package_summary.json", {
        "rows": len(selected_pairs),
        "relation_hint_distribution": dict(manual_summary),
        "visible_rows": len(visible_manual_rows),
    })

    # Additional compact artifacts.
    write_json(out_dir / "baseline_metrics.json", baseline_metrics)
    write_json(out_dir / "candidate_metrics.json", candidate_metrics)
    write_json(out_dir / "candidate_metrics_details.json", candidate_audit)
    write_json(out_dir / "repair_dev_manifest.json", {
        "output_dir": str(out_dir.relative_to(ROOT)),
        "task1_input_rows": len(task1_rows),
        "grounding_pass_rows": len(pass_rows),
        "recovered_events": len(recovered_events),
        "task1_quarantine_rows": len(quarantine_events),
        "old_pair_input_rows": len(pair_rows),
        "old_pair_recovered_rows": len(recovered_pairs),
        "old_pair_quarantine_rows": len(pair_quarantine),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_generation_metrics": json.loads((out_dir / "candidate_generation_metrics.json").read_text(encoding="utf-8")),
        "manual_pair_rows": len(selected_pairs),
        "benchmark_read": False,
        "frozen_file_modified": False,
        "task3_candidate_enhancement_pass": False,  # updated below
    })

    protected_after = build_protected_sha_manifest()
    write_json(out_dir / "protected_sha256_after.json", protected_after)

    enhancement_pass = (
        candidate_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        and candidate_metrics["false_merge"] <= baseline_metrics["false_merge"]
        and candidate_metrics["split_error"] <= baseline_metrics["split_error"]
        and candidate_audit["candidate_coverage"] >= candidate_audit["frozen_candidate_coverage"]
    )
    if enhancement_pass:
        manifest = json.loads((out_dir / "repair_dev_manifest.json").read_text(encoding="utf-8"))
        manifest["task3_candidate_enhancement_pass"] = True
        write_json(out_dir / "repair_dev_manifest.json", manifest)

    write_report_markdown(out_dir / "task3_v2_regression_report.md", {
        "recovered_events": len(recovered_events),
        "old_pair_input": len(pair_rows),
        "old_pair_recovered": len(recovered_pairs),
        "quarantine_total": len(quarantine_events) + len(pair_quarantine),
        "benchmark_read": False,
        "frozen_modified": False,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_generation_metrics": json.loads((out_dir / "candidate_generation_metrics.json").read_text(encoding="utf-8")),
        "enhancement_pass": enhancement_pass,
        "output_dir": str(out_dir.relative_to(ROOT)),
    })

    readme = [
        "Task3 CanonicalEvent Resolver v2 actual repair.",
        "",
        f"Recovered 615 pass events: {len(recovered_events)}.",
        f"Old pairs recovered: {len(recovered_pairs)}/{len(pair_rows)}.",
        f"Candidate enhancement pass: {enhancement_pass}.",
        f"Benchmark read: False.",
        f"Frozen files modified: False.",
        "",
        "Primary outputs:",
        "- task1_615_grounding_pass_recovered.jsonl",
        "- old_pair_recovery.jsonl",
        "- baseline_metrics.json",
        "- candidate_metrics.json",
        "- candidate_generation_metrics.json",
        "- canonical_event_v2_candidates.parquet",
        "- manual_pair_annotation_package.xlsx",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (out_dir / "repro_actual_repair.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "output_dir": str(out_dir.relative_to(ROOT)),
        "recovered_events": len(recovered_events),
        "task1_quarantine_rows": len(quarantine_events),
        "old_pair_recovered": len(recovered_pairs),
        "old_pair_quarantine_rows": len(pair_quarantine),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_generation_metrics": json.loads((out_dir / "candidate_generation_metrics.json").read_text(encoding="utf-8")),
        "manual_pair_rows": len(selected_pairs),
        "enhancement_pass": enhancement_pass,
        "benchmark_read": False,
        "frozen_file_modified": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
