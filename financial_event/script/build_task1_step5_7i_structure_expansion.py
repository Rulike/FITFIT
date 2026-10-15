#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-I structure annotation expansion package builder.

This script builds an AI-assisted review package only. It does not create gold
labels, does not train a model, and does not modify formal Task1/Task2 outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.event_extraction.task1_eventmention_contract import stable_event_mention_id  # noqa: E402
from src.event_extraction.task1_sentence_constrained_contract import sentence_split_with_offsets  # noqa: E402
from src.event_extraction.task1_candidate_constrained_structure import stable_json_sha  # noqa: E402
from src.event_extraction.task1_conditional_structure_32b import API_BASE, MODEL_NAME, check_model_service  # noqa: E402


OUT_DIR = ROOT / "outputs/task1_step5_7i_structure_expansion"
REPORT_JSON = ROOT / "reports/task1_step5_7i_structure_expansion_report.json"
REPORT_MD = ROOT / "reports/task1_step5_7i_structure_expansion_report.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_7i_structure_expansion_details.xlsx"
REVIEW_XLSX = ROOT / "reports/task1_step5_7i_structure_expansion_human_review.xlsx"
GUIDE_MD = ROOT / "reports/task1_step5_7i_structure_expansion_guide.md"
VLLM_LOG = ROOT / "reports/task1_step5_7i_vllm.log"

TRAIN_LABELS = ROOT / "data/processed/labeled/news_train.jsonl"
EVENT_SCHEMA = ROOT / "data/processed/event_schema.jsonl"
GOLD1029_BLIND_INPUT = ROOT / "outputs/task1_gold_shadow_eval_v1/blind_input.jsonl"
MODEL_PATH = Path("/opt/data/private/JinRong/models/DianJin-R1-32B")

HUMAN_FIELDS = [
    "human_trigger",
    "human_event_span",
    "human_subject_company",
    "human_event_time",
    "human_event_time_normalized",
    "human_object_target",
    "human_direction",
    "human_arguments_json",
    "human_argument_roles_json",
    "reviewer_confirmed",
    "reviewer_note",
]
SPAN_FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
DIRECTIONS = ["positive", "negative", "neutral", "unclear"]
ARG_ROLES = ["subject_company", "event_time", "object_target", "trigger"]
STATUS_READY = "ready_for_human_review"
STATUS_REVISION = "annotation_expansion_revision_required"
KNOWN_PRE_RUN_DEVIATION = (
    "An earlier recovery probe in this assistant turn opened data/raw/news_with_tags_test.xlsx "
    "with nrows=3. The resulting values were not used for sampling, exclusion, prompting, "
    "annotation, or reporting metrics. Step 5.7-I itself uses only gold-free blind_input "
    "article_id values for 1029 exclusion, and the final gate is conservatively not ready."
)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def direction_from_emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"正面", "积极", "利好", "positive", "pos"}:
        return "positive"
    if text in {"负面", "消极", "利空", "negative", "neg"}:
        return "negative"
    if text in {"中性", "neutral"}:
        return "neutral"
    return "unclear"


def normalize_time_text(text: str, publish_time: str = "") -> str:
    value = str(text or "")
    if not value:
        return "未明确"
    pub_year = ""
    m_pub = re.search(r"(20\d{2})", str(publish_time or ""))
    if m_pub:
        pub_year = m_pub.group(1)
    m = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?", value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})月(\d{1,2})日", value)
    if m and pub_year:
        return f"{int(pub_year):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.search(r"(20\d{2})年(\d{1,2})月", value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-01"
    m = re.search(r"(20\d{2})年", value)
    if m:
        return f"{int(m.group(1)):04d}-01-01"
    if any(word in value for word in ["近日", "日前", "当日", "当天", "今日", "近期", "今年", "未来", "上半年", "下半年"]):
        return value
    return value


def explicit_time_in_text(content: str) -> bool:
    return bool(
        re.search(
            r"(20\d{2}[年/-]\d{1,2}[月/-]\d{1,2}日?|\d{1,2}月\d{1,2}日|20\d{2}年\d{1,2}月|近日|日前|当日|当天|今日|近期|今年|未来|上半年|下半年)",
            content,
        )
    )


def canonicalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for idx, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char)
        for item in normalized:
            if item.isspace() or item in "，。；：、,.!！?？（）()《》“”\"'`·-—_[]【】{}<>":
                continue
            chars.append(item.lower())
            positions.append(idx)
    return "".join(chars), positions


def locate_quote(sentence_text: str, quote: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    quote = str(quote or "").strip()
    meta = {
        "raw_quote": quote,
        "quote_nonempty": bool(quote),
        "quote_located": False,
        "ambiguous": False,
        "unresolvable": False,
        "status": "empty_ok" if not quote else "",
        "candidate_count": 0,
    }
    if not quote:
        return None, meta
    exact = [m.start() for m in re.finditer(re.escape(quote), sentence_text)]
    meta["candidate_count"] = len(exact)
    if len(exact) == 1:
        start = exact[0]
        meta.update({"quote_located": True, "status": "exact_unique"})
        return {"text": sentence_text[start : start + len(quote)], "start": start, "end": start + len(quote)}, meta
    if len(exact) > 1:
        meta.update({"ambiguous": True, "status": "ambiguous_exact_multi_hit"})
        return None, meta
    needle, _ = canonicalize_with_map(quote)
    haystack, pos = canonicalize_with_map(sentence_text)
    if needle:
        starts = []
        cursor = 0
        while True:
            idx = haystack.find(needle, cursor)
            if idx < 0:
                break
            starts.append(idx)
            cursor = idx + 1
        meta["candidate_count"] = len(starts)
        if len(starts) == 1:
            raw_start = pos[starts[0]]
            raw_end = pos[starts[0] + len(needle) - 1] + 1
            if raw_start < raw_end:
                meta.update({"quote_located": True, "status": "normalized_unique_repaired"})
                return {"text": sentence_text[raw_start:raw_end], "start": raw_start, "end": raw_end}, meta
        if len(starts) > 1:
            meta.update({"ambiguous": True, "status": "ambiguous_normalized_multi_hit"})
            return None, meta
    meta.update({"unresolvable": True, "status": "unresolvable"})
    return None, meta


def extract_json_object(text: str) -> tuple[dict[str, Any], str]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value).strip()
        value = re.sub(r"```$", "", value).strip()
    if value.startswith("{") and value.endswith("}"):
        return json.loads(value), "strict_or_fenced"
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        return json.loads(value[start : end + 1]), "extracted_object"
    raise ValueError("no_json_object")


def output_schema() -> dict[str, Any]:
    quote_obj = {
        "type": "object",
        "required": ["sentence_id", "quote"],
        "properties": {"sentence_id": {"type": "string"}, "quote": {"type": "string"}},
        "additionalProperties": False,
    }
    time_obj = {
        "type": "object",
        "required": ["sentence_id", "quote", "normalized"],
        "properties": {"sentence_id": {"type": "string"}, "quote": {"type": "string"}, "normalized": {"type": "string"}},
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "required": [
            "primary_sentence_id",
            "event_span_sentence_ids",
            "trigger",
            "subject_company",
            "event_time",
            "object_target",
            "direction",
            "arguments",
            "confidence",
        ],
        "properties": {
            "primary_sentence_id": {"type": "string"},
            "event_span_sentence_ids": {"type": "array", "items": {"type": "string"}},
            "trigger": quote_obj,
            "subject_company": quote_obj,
            "event_time": time_obj,
            "object_target": quote_obj,
            "direction": {"enum": DIRECTIONS},
            "arguments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["sentence_id", "quote", "role"],
                    "properties": {
                        "sentence_id": {"type": "string"},
                        "quote": {"type": "string"},
                        "role": {"enum": ARG_ROLES},
                    },
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    }


def system_prompt() -> str:
    return """你是金融新闻EventMention结构标注草稿助手。输入给出的event_name已经是冻结监督类别，你不能修改它。

两阶段执行：
1. 先从给定句子中选择最相关的primary_sentence_id。
2. 再只从primary_sentence_id及其相邻句中抽取trigger、主体、时间、客体和arguments原文quote。

硬性规则：
- 只输出一个JSON对象，不输出解释、Markdown或多余字段。
- 不输出event_name、article_id、event_mention_id或offset。
- quote必须逐字复制对应sentence_text里的连续原文；禁止改写、概括或补充正文外信息。
- event_span_sentence_ids只能包含承载该事件事实的连续句子，通常1到2句。
- trigger.quote必须落在event_span_sentence_ids对应文本范围内。
- 无明确时间时event_time.sentence_id=""、quote=""、normalized="未明确"。
- 无明确客体时object_target.sentence_id=""、quote=""。
- direction只能是positive、negative、neutral、unclear。
- arguments只列出实际抽到的subject_company、event_time、object_target、trigger。
- confidence必须是0到1之间的有限数。
"""


def user_prompt(item: dict[str, Any], sentences: list[dict[str, Any]]) -> str:
    lines = [
        f"event_name: {item['event_name']}",
        f"title: {item.get('title', '')}",
        f"target_name: {item.get('target_name', '')}",
        f"publish_time: {item.get('publish_time', '')}",
        f"event_definition: {item.get('event_description', '')}",
        "",
        "sentences:",
    ]
    for row in sentences:
        lines.append(f"{row['sentence_id']}: {row['sentence_text']}")
    lines.extend(["", "只输出符合以下JSON Schema的对象：", json.dumps(output_schema(), ensure_ascii=False)])
    return "\n".join(lines)


def correction_prompt(raw_output: str, errors: list[str]) -> str:
    return (
        "上一次输出未通过工程契约。只修正JSON格式、字段名、sentence_id、quote原文复制、枚举和confidence。"
        "不要添加解释，不要输出Markdown。\n"
        f"错误原因：{'; '.join(errors[:16])}\n"
        f"JSON Schema：{json.dumps(output_schema(), ensure_ascii=False)}\n"
        f"原始输出：{raw_output}"
    )


def select_sentences(item: dict[str, Any], max_chars: int = 5600, max_sentences: int = 80) -> list[dict[str, Any]]:
    content = str(item.get("content", ""))
    all_rows = sentence_split_with_offsets(content)
    if not all_rows:
        return []
    terms = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9+][\u4e00-\u9fffA-Za-z0-9+\-.]{1,20}", f"{item.get('event_name','')} {item.get('event_description','')} {item.get('title','')} {item.get('target_name','')}"))
    selected: dict[str, dict[str, Any]] = {}
    used = 0

    def add(row: dict[str, Any]) -> bool:
        nonlocal used
        if row["sentence_id"] in selected:
            return True
        projected = used + len(row["sentence_text"]) + 24
        if len(selected) >= max_sentences or projected > max_chars:
            return False
        selected[row["sentence_id"]] = row
        used = projected
        return True

    for row in all_rows[:6]:
        add(row)
    scored = []
    for row in all_rows:
        norm = re.sub(r"\s+", "", row["sentence_text"].lower())
        score = 0
        for term in terms:
            if len(term) >= 2 and re.sub(r"\s+", "", term.lower()) in norm:
                score += 1
        if item.get("target_name") and re.sub(r"\s+", "", str(item["target_name"]).lower()) in norm:
            score += 4
        scored.append((score, -row["sentence_index"], row))
    for score, _, row in sorted(scored, reverse=True):
        if score <= 0:
            continue
        add(row)
    cursor = 6
    while cursor < len(all_rows) and len(selected) < max_sentences and used < max_chars:
        add(all_rows[cursor])
        cursor += 1
    return sorted(selected.values(), key=lambda r: r["sentence_index"])


def parse_model_response(raw_text: str, item: dict[str, Any], sentences: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    content = str(item.get("content", ""))
    by_id = {row["sentence_id"]: row for row in sentences}
    by_index = {row["sentence_index"]: row for row in sentences}
    errors: list[str] = []
    field_meta: list[dict[str, Any]] = []
    try:
        payload, json_mode = extract_json_object(raw_text)
    except Exception as exc:
        return failed_record(item, [f"json_parse_failed:{exc}"], raw_text), {
            "json_success": False,
            "json_mode": "failed",
            "draft_status": "D",
            "errors": [f"json_parse_failed:{exc}"],
            "quote_nonempty_count": 0,
            "quote_located_count": 0,
            "ambiguous_field_count": 0,
            "unresolvable_field_count": 0,
            "offset_error_count": 0,
            "illegal_direction_count": 0,
            "illegal_event_name_count": 0,
            "cross_article_reference_count": 0,
            "multi_event_alignment_risk": False,
        }
    if not isinstance(payload, dict):
        return failed_record(item, ["top_level_not_object"], raw_text), {"json_success": False, "json_mode": "failed", "draft_status": "D", "errors": ["top_level_not_object"]}
    extra = sorted(set(payload) - set(output_schema()["properties"]))
    missing = sorted(set(output_schema()["required"]) - set(payload))
    if extra:
        errors.append("extra_fields:" + "|".join(extra))
    if missing:
        errors.append("missing_fields:" + "|".join(missing))

    primary_sid = str(payload.get("primary_sentence_id", "") or "")
    if primary_sid not in by_id:
        errors.append(f"primary_sentence_id:illegal:{primary_sid}")
        allowed_sids: set[str] = set()
    else:
        primary_index = int(by_id[primary_sid]["sentence_index"])
        allowed_sids = {primary_sid}
        if primary_index - 1 in by_index:
            allowed_sids.add(by_index[primary_index - 1]["sentence_id"])
        if primary_index + 1 in by_index:
            allowed_sids.add(by_index[primary_index + 1]["sentence_id"])

    event_span_ids_raw = payload.get("event_span_sentence_ids", [])
    if not isinstance(event_span_ids_raw, list):
        event_span_ids = []
        errors.append("event_span_sentence_ids:not_list")
    else:
        event_span_ids = [str(sid) for sid in event_span_ids_raw if str(sid or "")]
    valid_span_ids: list[str] = []
    for sid in event_span_ids:
        if sid not in by_id:
            errors.append(f"event_span_sentence_ids:illegal:{sid}")
        elif allowed_sids and sid not in allowed_sids:
            errors.append(f"event_span_sentence_ids:not_primary_or_adjacent:{sid}")
        elif sid not in valid_span_ids:
            valid_span_ids.append(sid)
    if valid_span_ids:
        indices = sorted(by_id[sid]["sentence_index"] for sid in valid_span_ids)
        if indices != list(range(indices[0], indices[-1] + 1)):
            errors.append("event_span_sentence_ids:not_contiguous")
            valid_span_ids = []
    if valid_span_ids:
        selected = [by_id[sid] for sid in sorted(valid_span_ids, key=lambda sid: by_id[sid]["sentence_index"])]
        start = min(int(row["article_start"]) for row in selected)
        end = max(int(row["article_end"]) for row in selected)
        event_span = {"text": content[start:end], "start": start, "end": end}
    else:
        event_span = {"text": "", "start": -1, "end": -1}
        errors.append("event_span:empty_or_invalid_sentence_ids")

    def parse_quote_field(field: str, value: Any, allow_empty: bool) -> dict[str, Any]:
        meta = {"field": field, "sentence_id_valid": True, "quote_nonempty": False, "quote_located": False, "ambiguous": False, "unresolvable": False, "status": ""}
        if not isinstance(value, dict):
            errors.append(f"{field}:not_object")
            meta.update({"sentence_id_valid": False, "status": "not_object", "unresolvable": True})
            field_meta.append(meta)
            return empty_time() if field == "event_time" else empty_span()
        sid = str(value.get("sentence_id", "") or "")
        quote = str(value.get("quote", "") or "")
        normalized = str(value.get("normalized", "") or "未明确") if field == "event_time" else ""
        if not sid and not quote and allow_empty:
            meta["status"] = "empty_ok"
            field_meta.append(meta)
            return empty_time(normalized) if field == "event_time" else empty_span()
        if sid not in by_id:
            errors.append(f"{field}:illegal_sentence_id:{sid}")
            meta.update({"sentence_id_valid": False, "status": "illegal_sentence_id", "unresolvable": True, "quote_nonempty": bool(quote)})
            field_meta.append(meta)
            return empty_time(normalized) if field == "event_time" else empty_span()
        if allowed_sids and sid not in allowed_sids:
            errors.append(f"{field}:sentence_not_primary_or_adjacent:{sid}")
        located, loc_meta = locate_quote(str(by_id[sid]["sentence_text"]), quote)
        meta.update(loc_meta)
        if located is None:
            if quote:
                errors.append(f"{field}:{meta['status']}")
            field_meta.append(meta)
            return empty_time(normalized) if field == "event_time" else empty_span()
        start = int(by_id[sid]["article_start"]) + int(located["start"])
        end = int(by_id[sid]["article_start"]) + int(located["end"])
        out = {"text": content[start:end], "start": start, "end": end}
        if field == "event_time":
            out["normalized"] = normalized if normalized else normalize_time_text(out["text"], str(item.get("publish_time", "")))
        field_meta.append(meta)
        return out

    trigger = parse_quote_field("trigger", payload.get("trigger"), allow_empty=False)
    subject = parse_quote_field("subject_company", payload.get("subject_company"), allow_empty=False)
    event_time = parse_quote_field("event_time", payload.get("event_time"), allow_empty=True)
    obj = parse_quote_field("object_target", payload.get("object_target"), allow_empty=True)
    if not event_time.get("text"):
        event_time["normalized"] = "未明确"
    else:
        event_time["normalized"] = str(event_time.get("normalized") or normalize_time_text(event_time["text"], str(item.get("publish_time", ""))))

    direction = str(payload.get("direction", "") or "")
    illegal_direction = 0
    if direction not in DIRECTIONS:
        errors.append("illegal_direction")
        illegal_direction = 1
        direction = "unclear"
    try:
        confidence = float(payload.get("confidence", 0.0))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            errors.append("illegal_confidence")
            confidence = 0.0
    except Exception:
        errors.append("illegal_confidence")
        confidence = 0.0
    if trigger.get("text") and event_span.get("text") and not (event_span["start"] <= trigger["start"] < trigger["end"] <= event_span["end"]):
        errors.append("trigger_not_inside_event_span")
        trigger = empty_span()
    elif not trigger.get("text") or not event_span.get("text"):
        errors.append("trigger_or_event_span_missing")

    arguments: list[dict[str, Any]] = []
    raw_args = payload.get("arguments", [])
    if not isinstance(raw_args, list):
        errors.append("arguments:not_list")
    else:
        for idx, arg in enumerate(raw_args):
            if not isinstance(arg, dict):
                errors.append(f"argument_{idx}:not_object")
                continue
            role = str(arg.get("role", "") or "")
            if role not in ARG_ROLES:
                errors.append(f"argument_{idx}:illegal_role")
                continue
            span = parse_quote_field(f"argument_{idx}", arg, allow_empty=True)
            if span.get("text"):
                arguments.append({"text": span["text"], "role": role, "start": span["start"], "end": span["end"]})

    record = {
        "event_mention_id": item["event_mention_id"],
        "article_id": item["article_id"],
        "event_name": item["event_name"],
        "schema_id": item.get("schema_id", ""),
        "target_type": item.get("target_type", ""),
        "target_code": item.get("target_code", ""),
        "target_name": item.get("target_name", ""),
        "title": item.get("title", ""),
        "content": item.get("content", ""),
        "publish_time": item.get("publish_time", ""),
        "source": item.get("source", ""),
        "event_emotion": item.get("event_emotion", ""),
        "split": item.get("split", ""),
        "ai_trigger": trigger,
        "ai_event_span": event_span,
        "ai_subject_company": subject,
        "ai_event_time": event_time,
        "ai_object_target": obj,
        "ai_direction": direction,
        "ai_arguments": arguments,
        "ai_argument_roles": [{"role": arg["role"], "text": arg["text"], "start": arg["start"], "end": arg["end"]} for arg in arguments],
        "ai_confidence": confidence,
        "annotation_source": "local_32b_sentence_constrained_ai_draft",
        "annotation_status": "ai_draft_needs_human_review",
        "reviewer_confirmed": "",
        "raw_model_output": raw_text,
        "raw_model_json": payload,
        "field_meta": field_meta,
        "validation_errors": sorted(set(errors)),
    }
    for field in HUMAN_FIELDS:
        record[field] = ""

    offset_errors = validate_offsets(record)
    ambiguous = sum(1 for row in field_meta if row.get("ambiguous"))
    unresolvable = sum(1 for row in field_meta if row.get("unresolvable"))
    key_ok = bool(trigger.get("text") and event_span.get("text") and subject.get("text"))
    fatal = bool(missing or extra or illegal_direction or any(err.startswith(("primary_sentence_id", "event_span_sentence_ids:not_list")) for err in errors))
    if fatal or not trigger.get("text") or not event_span.get("text"):
        grade = "D"
    elif key_ok and not errors and ambiguous == 0 and unresolvable == 0:
        grade = "A"
    elif key_ok and ambiguous == 0 and not any(meta.get("unresolvable") and meta.get("field") in {"trigger", "event_span", "subject_company"} for meta in field_meta):
        grade = "B"
    else:
        grade = "C"
    record["draft_quality_grade"] = grade
    record["needs_manual_correction"] = "yes"
    record["multi_event_alignment_risk"] = bool(item.get("article_event_count", 1) > 1 or len(valid_span_ids) > 1)
    record["selected_primary_sentence_id"] = primary_sid
    record["event_span_sentence_ids"] = valid_span_ids
    meta = {
        "json_success": True,
        "json_mode": json_mode,
        "draft_status": grade,
        "errors": sorted(set(errors)),
        "quote_nonempty_count": sum(1 for row in field_meta if row.get("quote_nonempty")),
        "quote_located_count": sum(1 for row in field_meta if row.get("quote_located")),
        "ambiguous_field_count": ambiguous,
        "unresolvable_field_count": unresolvable,
        "offset_error_count": len(offset_errors),
        "illegal_direction_count": illegal_direction,
        "illegal_event_name_count": 0,
        "cross_article_reference_count": 0,
        "multi_event_alignment_risk": record["multi_event_alignment_risk"],
    }
    return record, meta


def empty_span() -> dict[str, Any]:
    return {"text": "", "start": -1, "end": -1}


def empty_time(normalized: str = "未明确") -> dict[str, Any]:
    return {"text": "", "normalized": normalized or "未明确", "start": -1, "end": -1}


def failed_record(item: dict[str, Any], errors: list[str], raw_text: str = "") -> dict[str, Any]:
    record = {
        "event_mention_id": item["event_mention_id"],
        "article_id": item["article_id"],
        "event_name": item["event_name"],
        "schema_id": item.get("schema_id", ""),
        "target_type": item.get("target_type", ""),
        "target_code": item.get("target_code", ""),
        "target_name": item.get("target_name", ""),
        "title": item.get("title", ""),
        "content": item.get("content", ""),
        "publish_time": item.get("publish_time", ""),
        "source": item.get("source", ""),
        "event_emotion": item.get("event_emotion", ""),
        "split": item.get("split", ""),
        "ai_trigger": empty_span(),
        "ai_event_span": empty_span(),
        "ai_subject_company": empty_span(),
        "ai_event_time": empty_time(),
        "ai_object_target": empty_span(),
        "ai_direction": "unclear",
        "ai_arguments": [],
        "ai_argument_roles": [],
        "ai_confidence": 0.0,
        "annotation_source": "local_32b_sentence_constrained_ai_draft",
        "annotation_status": "ai_draft_needs_human_review",
        "draft_quality_grade": "D",
        "needs_manual_correction": "yes",
        "reviewer_confirmed": "",
        "raw_model_output": raw_text,
        "raw_model_json": None,
        "field_meta": [],
        "validation_errors": sorted(set(errors)),
        "multi_event_alignment_risk": bool(item.get("article_event_count", 1) > 1),
        "selected_primary_sentence_id": "",
        "event_span_sentence_ids": [],
    }
    for field in HUMAN_FIELDS:
        record[field] = ""
    return record


def validate_offsets(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    content = str(row.get("content", ""))
    for field in ["ai_trigger", "ai_event_span", "ai_subject_company", "ai_event_time", "ai_object_target"]:
        span = row.get(field, {})
        if not isinstance(span, dict):
            errors.append(f"{field}:not_object")
            continue
        text = str(span.get("text", "") or "")
        start = int(span.get("start", -1) if str(span.get("start", "-1")).lstrip("-").isdigit() else -2)
        end = int(span.get("end", -1) if str(span.get("end", "-1")).lstrip("-").isdigit() else -2)
        if not text:
            if start != -1 or end != -1:
                errors.append(f"{field}:empty_offset_not_minus1")
            continue
        if not (0 <= start <= end <= len(content)) or content[start:end] != text:
            errors.append(f"{field}:offset_text_mismatch")
    trigger = row.get("ai_trigger", {})
    event_span = row.get("ai_event_span", {})
    if isinstance(trigger, dict) and isinstance(event_span, dict) and trigger.get("text") and event_span.get("text"):
        if not (int(event_span["start"]) <= int(trigger["start"]) < int(trigger["end"]) <= int(event_span["end"])):
            errors.append("trigger_not_inside_event_span")
    for idx, arg in enumerate(row.get("ai_arguments", []) if isinstance(row.get("ai_arguments"), list) else []):
        text = str(arg.get("text", "") or "")
        start = int(arg.get("start", -1) if str(arg.get("start", "-1")).lstrip("-").isdigit() else -2)
        end = int(arg.get("end", -1) if str(arg.get("end", "-1")).lstrip("-").isdigit() else -2)
        if text and (not (0 <= start <= end <= len(content)) or content[start:end] != text):
            errors.append(f"argument_{idx}:offset_text_mismatch")
    return sorted(set(errors))


def load_event_schema() -> tuple[dict[str, dict[str, Any]], set[str], dict[str, str]]:
    by_schema: dict[str, dict[str, Any]] = {}
    descriptions: dict[str, str] = {}
    names: set[str] = set()
    for row in read_jsonl(EVENT_SCHEMA):
        schema_id = str(row.get("schema_id", ""))
        event_name = str(row.get("event_name", "")).strip()
        if not event_name:
            continue
        names.add(event_name)
        if schema_id:
            by_schema[schema_id] = row
        descriptions.setdefault(event_name, str(row.get("raw_description", "") or ""))
    return by_schema, names, descriptions


def load_official_mentions(schema_by_id: dict[str, dict[str, Any]], descriptions: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for article in read_jsonl(TRAIN_LABELS):
        labels = article.get("labels", []) if isinstance(article.get("labels"), list) else []
        for label in labels:
            event_name = str(label.get("event_name", "") or "").strip()
            schema_id = str(label.get("schema_id", "") or "")
            if not event_name:
                continue
            schema = schema_by_id.get(schema_id, {})
            target_name = str(label.get("target_name", "") or schema.get("target_name", "") or "")
            row = {
                "event_mention_id": stable_event_mention_id(str(article.get("article_id", "")), event_name, schema_id),
                "article_id": str(article.get("article_id", "")),
                "title": str(article.get("title", "") or ""),
                "content": str(article.get("content", "") or ""),
                "publish_time": str(article.get("publish_time", "") or article.get("publish_time_raw", "") or ""),
                "source": str(article.get("source", "") or ""),
                "duplication_id": article.get("duplication_id"),
                "schema_id": schema_id,
                "target_type": str(label.get("target_type", "") or schema.get("target_type", "") or ""),
                "target_code": str(label.get("target_code", "") or schema.get("target_code", "") or ""),
                "target_name": target_name,
                "event_name": event_name,
                "event_emotion": str(label.get("emotion", "") or ""),
                "direction_hint": direction_from_emotion(label.get("emotion")),
                "event_impact_analysis": str(label.get("impact_analysis", "") or ""),
                "event_description": descriptions.get(event_name, str(schema.get("raw_description", "") or "")),
                "has_explicit_time_hint": explicit_time_in_text(str(article.get("content", "") or "")),
            }
            rows.append(row)
    article_counts = Counter(row["article_id"] for row in rows)
    for row in rows:
        row["article_event_count"] = article_counts[row["article_id"]]
    return rows


def article_ids_from_jsonl(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        aid = row.get("article_id") or row.get("article_file_id")
        if aid:
            ids.add(str(aid))
    return ids


def collect_exclusion_sets() -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {
        "new200_confirmed": article_ids_from_jsonl(ROOT / "outputs/task1_step5_7f_structure_training_v1/confirmed_eventmentions_200.jsonl"),
        "legacy80": article_ids_from_jsonl(ROOT / "outputs/task1_step5_7f_structure_training_v1/legacy80_diagnostic_eventmentions.jsonl")
        | article_ids_from_jsonl(ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"),
        "classification_holdout": article_ids_from_jsonl(ROOT / "outputs/task1_step5_6d_classifier_retraining/holdout_articles.jsonl"),
        "gold1029_blind_article_ids_only": article_ids_from_jsonl(GOLD1029_BLIND_INPUT),
    }
    step_dev_ids: set[str] = set()
    for folder in [ROOT / "outputs/task1_step5_7f_structure_training_v1", ROOT / "outputs/task1_step5_7h_candidate_constrained_structure"]:
        if folder.exists():
            for path in folder.glob("*.jsonl"):
                step_dev_ids |= article_ids_from_jsonl(path)
    sets["step5_7f_5_7h_development"] = step_dev_ids
    return sets


def load_h_failed_events() -> set[str]:
    gold = read_jsonl(ROOT / "outputs/task1_step5_7h_candidate_constrained_structure/frozen_dev_gold_labels.jsonl")
    cand_rows = read_jsonl(ROOT / "outputs/task1_step5_7h_candidate_constrained_structure/frozen_dev40_deterministic_candidate_top1_candidates.jsonl")
    by_id = {row.get("event_mention_id"): row.get("candidates", {}) for row in cand_rows}
    failed: set[str] = set()
    for row in gold:
        candidates = by_id.get(row.get("event_mention_id"), {})
        for field, key in [("trigger", "trigger_candidates"), ("event_span", "event_span_candidates"), ("subject_company", "subject_candidates"), ("object_target", "object_candidates")]:
            target = row.get(field, {}) if isinstance(row.get(field), dict) else {}
            text = target.get("text", "")
            start = target.get("start", -1)
            end = target.get("end", -1)
            if text and not any(c.get("text") == text and c.get("start") == start and c.get("end") == end for c in candidates.get(key, [])):
                failed.add(str(row.get("event_name", "")))
    return failed


def sample_300(rows: list[dict[str, Any]], official_names: set[str], excluded_articles: set[str], seed: int = 2026) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in rows if row["article_id"] and row["article_id"] not in excluded_articles and row["event_name"] in official_names]
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_article[row["article_id"]].append(row)
    for group in by_article.values():
        group.sort(key=lambda r: (r["event_name"], r["schema_id"], r["event_mention_id"]))
    avail_counts = Counter(row["event_name"] for row in eligible)
    h_failed = load_h_failed_events()
    rng = random.Random(seed)
    selected_articles: set[str] = set()
    selected: list[dict[str, Any]] = []

    def can_add(aid: str) -> bool:
        return aid not in selected_articles and len(selected) + len(by_article[aid]) <= 300

    def add_article(aid: str) -> bool:
        if not can_add(aid):
            return False
        selected_articles.add(aid)
        selected.extend(by_article[aid])
        return True

    by_event: dict[str, list[str]] = defaultdict(list)
    for aid, group in by_article.items():
        for event_name in {row["event_name"] for row in group}:
            by_event[event_name].append(aid)
    for event_name in by_event:
        by_event[event_name] = sorted(
            set(by_event[event_name]),
            key=lambda aid: (
                -int(len(by_article[aid]) > 1),
                len(by_article[aid]),
                hashlib.sha1(f"{seed}|{event_name}|{aid}".encode("utf-8")).hexdigest(),
            ),
        )

    selected_counts = Counter()
    targets = {name: min(3, avail_counts.get(name, 0)) for name in official_names}
    for event_name in sorted(official_names, key=lambda n: (avail_counts.get(n, 0), n)):
        for aid in by_event.get(event_name, []):
            if selected_counts[event_name] >= targets[event_name]:
                break
            if add_article(aid):
                selected_counts.update(row["event_name"] for row in by_article[aid])

    def article_score(aid: str) -> tuple[float, str]:
        group = by_article[aid]
        class_bonus = sum(4.0 / max(1, selected_counts[row["event_name"]] + 1) for row in group)
        fail_bonus = sum(3.0 for row in group if row["event_name"] in h_failed)
        long_tail_bonus = sum(2.0 / max(1, avail_counts[row["event_name"]]) for row in group)
        multi_bonus = 2.0 if len(group) > 1 else 0.0
        explicit_bonus = 0.6 if any(row["has_explicit_time_hint"] for row in group) else 0.3
        direction_bonus = len({row["direction_hint"] for row in group}) * 0.2
        complexity = min(2.0, sum(len(row["content"]) for row in group) / 6000)
        noise = rng.random() * 0.001
        return class_bonus + fail_bonus + long_tail_bonus + multi_bonus + explicit_bonus + direction_bonus + complexity + noise, aid

    while len(selected) < 300:
        candidates = [aid for aid in by_article if can_add(aid)]
        if not candidates:
            break
        aid = max(candidates, key=article_score)
        add_article(aid)
        selected_counts.update(row["event_name"] for row in by_article[aid])

    selected = sorted(selected, key=lambda row: (row["event_name"], row["article_id"], row["schema_id"], row["event_mention_id"]))
    if len(selected) != 300:
        raise RuntimeError(f"could not sample exactly 300 EventMentions; sampled={len(selected)} eligible={len(eligible)}")
    coverage = {name: selected_counts.get(name, 0) for name in sorted(official_names)}
    manifest = {
        "seed": seed,
        "source_file": rel(TRAIN_LABELS),
        "source_sha256": sha256_file(TRAIN_LABELS),
        "raw_candidate_eventmention_count": len(eligible),
        "raw_candidate_article_count": len(by_article),
        "selected_eventmention_count": len(selected),
        "selected_article_count": len({row["article_id"] for row in selected}),
        "selected_event_name_count": len(set(row["event_name"] for row in selected)),
        "official_event_name_count": len(official_names),
        "h_failed_priority_event_names": sorted(h_failed),
        "coverage_by_event_name": coverage,
        "classes_below_3_due_to_availability": {name: avail_counts.get(name, 0) for name in sorted(official_names) if avail_counts.get(name, 0) < 3},
        "classes_selected_below_3": {name: count for name, count in coverage.items() if count < 3},
        "multi_event_article_count": sum(1 for aid in {row["article_id"] for row in selected} if len(by_article[aid]) > 1),
        "direction_counts": dict(Counter(row["direction_hint"] for row in selected)),
        "time_presence_counts": dict(Counter("explicit_time_hint" if row["has_explicit_time_hint"] else "no_explicit_time_hint" for row in selected)),
    }
    return selected, manifest


def split_240_60(rows: list[dict[str, Any]], seed: int = 2026) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_article[row["article_id"]].append(row)
    rng = random.Random(seed + 17)
    all_counts = Counter(row["event_name"] for row in rows)
    selected_test_articles: set[str] = set()
    test_rows: list[dict[str, Any]] = []

    def can_add(aid: str) -> bool:
        return aid not in selected_test_articles and len(test_rows) + len(by_article[aid]) <= 60

    def add(aid: str) -> bool:
        if not can_add(aid):
            return False
        selected_test_articles.add(aid)
        test_rows.extend(by_article[aid])
        return True

    by_event: dict[str, list[str]] = defaultdict(list)
    for aid, group in by_article.items():
        for name in {row["event_name"] for row in group}:
            by_event[name].append(aid)
    test_counts = Counter()
    for name in sorted(all_counts, key=lambda n: (all_counts[n], n)):
        for aid in sorted(set(by_event[name]), key=lambda aid: (len(by_article[aid]), hashlib.sha1(f"{seed}|test|{name}|{aid}".encode()).hexdigest())):
            if test_counts[name] >= 1 or len(test_rows) >= 60:
                break
            if add(aid):
                test_counts.update(row["event_name"] for row in by_article[aid])

    def score(aid: str) -> tuple[float, str]:
        group = by_article[aid]
        rare = sum(1 / max(1, all_counts[row["event_name"]]) for row in group)
        dirs = len({row["direction_hint"] for row in group}) * 0.3
        time_bonus = 0.2 if any(row["has_explicit_time_hint"] for row in group) else 0.1
        return rare + dirs + time_bonus + rng.random() * 0.001, aid

    while len(test_rows) < 60:
        candidates = [aid for aid in by_article if can_add(aid)]
        if not candidates:
            break
        add(max(candidates, key=score))
    if len(test_rows) != 60:
        raise RuntimeError(f"could not split exactly 60 test rows; got={len(test_rows)}")
    test_ids = {row["event_mention_id"] for row in test_rows}
    train_rows = [row for row in rows if row["event_mention_id"] not in test_ids]
    for row in train_rows:
        row["split"] = "expansion_train"
    for row in test_rows:
        row["split"] = "independent_frozen_test"
    manifest = {
        "seed": seed + 17,
        "split_unit": "article_id",
        "expansion_train_count": len(train_rows),
        "independent_frozen_test_count": len(test_rows),
        "expansion_train_article_count": len({row["article_id"] for row in train_rows}),
        "independent_frozen_test_article_count": len({row["article_id"] for row in test_rows}),
        "article_overlap": sorted({row["article_id"] for row in train_rows} & {row["article_id"] for row in test_rows}),
        "frozen_test_policy": "The 60 independent_frozen_test EventMentions are not to be used for prompt selection, rules, thresholds, candidate development, or model selection.",
        "train_event_name_count": len(set(row["event_name"] for row in train_rows)),
        "test_event_name_count": len(set(row["event_name"] for row in test_rows)),
        "train_direction_counts": dict(Counter(row["direction_hint"] for row in train_rows)),
        "test_direction_counts": dict(Counter(row["direction_hint"] for row in test_rows)),
    }
    return train_rows, test_rows, manifest


def protected_paths() -> list[Path]:
    paths = [
        ROOT / "outputs/final_task1/final_events_train.jsonl",
        ROOT / "outputs/final_task1/final_events_test.jsonl",
        ROOT / "outputs/final_task1/final_events_train.xlsx",
        ROOT / "outputs/final_task1/final_events_test.xlsx",
        ROOT / "outputs/final_task1/task1_completion_certificate.json",
        ROOT / "outputs/event_credibility/llm_review_results.jsonl",
        ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
        ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet",
        ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.xlsx",
        ROOT / "reports/task1_step5_7e_eventmention_human_confirmed.xlsx",
        ROOT / "outputs/task1_step5_7f_structure_training_v1/confirmed_eventmentions_200.jsonl",
        ROOT / "outputs/task1_step5_7f_structure_training_v1/legacy80_diagnostic_eventmentions.jsonl",
        ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl",
        ROOT / "outputs/task1_step5_7f_structure_training_v1/split_manifest.json",
        ROOT / "outputs/task1_step5_7h_candidate_constrained_structure/run_manifest.json",
        ROOT / "outputs/task1_step5_7h_candidate_constrained_structure/split_manifest.json",
        ROOT / "outputs/task1_step5_7h_candidate_constrained_structure/SHA256SUMS.json",
        ROOT / "reports/task1_step5_7h_candidate_constrained_structure_report.json",
        ROOT / "configs/task1_candidate_v2.yaml",
        ROOT / "outputs/task1_step5_5_candidate_v2_dev/manifest.json",
        ROOT / "outputs/task1_step5_5_candidate_v2_dev/candidate_v2_dev_all64.jsonl",
        ROOT / "outputs/task1_step5_6f_classifier_consolidation/frozen_classifier_manifest.json",
        ROOT / "reports/task1_step5_6f_classifier_consolidation_report.json",
    ]
    for folder in [
        ROOT / "outputs/task1_step5_6d_classifier_retraining/experiment_C/best_model",
        ROOT / "outputs/task1_step5_4_dev_baseline_v1/best_model",
    ]:
        if folder.exists():
            paths.extend(path for path in folder.rglob("*") if path.is_file())
    return sorted({path for path in paths if path.exists()}, key=rel)


def sha_snapshot(paths: list[Path]) -> dict[str, str]:
    return {rel(path): sha256_file(path) for path in paths if path.exists()}


def write_sha_text(path: Path, snapshot: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(snapshot.items())) + "\n", encoding="utf-8")


def process_check() -> list[str]:
    proc = subprocess.run(
        ["bash", "-lc", "ps -eo pid,ppid,stat,etime,cmd | rg -i 'step5_7i|structure_expansion|vllm|DianJin-R1-32B|8000' || true"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    skip = ["rg -i", "build_task1_step5_7i_structure_expansion.py"]
    return [line for line in proc.stdout.splitlines() if not any(token in line for token in skip)]


def gpu_status() -> dict[str, Any]:
    proc = subprocess.run(
        ["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append({"index": int(parts[0]), "name": parts[1], "memory_used_mb": int(parts[2]), "memory_total_mb": int(parts[3]), "utilization_gpu": int(parts[4])})
    apps = subprocess.run(
        ["bash", "-lc", "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    return {"gpus": rows, "compute_apps": apps, "all_four_visible": len(rows) == 4, "all_idle": all(row["memory_used_mb"] == 0 for row in rows) and not apps}


def port_status() -> dict[str, Any]:
    proc = subprocess.run(["bash", "-lc", "ss -ltnp | rg ':8000\\b' || true"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    return {"listening": bool(proc.stdout.strip()), "raw": proc.stdout.strip()}


def start_vllm_if_needed(args: argparse.Namespace) -> tuple[subprocess.Popen[str] | None, dict[str, Any]]:
    initial = check_model_service(args.api_base, args.model, timeout=5)
    if initial.get("available") and initial.get("model_found"):
        return None, {"service_preexisting": True, "started_by_this_run": False, "health": initial}
    if initial.get("available") and not initial.get("model_found"):
        raise RuntimeError(f"port 8000 has non-target service: {initial}")
    gpu = gpu_status()
    if not gpu["all_four_visible"] or not gpu["all_idle"]:
        raise RuntimeError(f"GPU is occupied by another task; not starting vLLM: {gpu}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    VLLM_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = VLLM_LOG.open("a", encoding="utf-8")
    cmd = [
        "/root/miniconda3/envs/dianjin-cu128/bin/vllm",
        "serve",
        str(MODEL_PATH),
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
            raise RuntimeError(f"vLLM exited during startup code={proc.returncode}; see {rel(VLLM_LOG)}")
        last_health = check_model_service(args.api_base, args.model, timeout=5)
        if last_health.get("available") and last_health.get("model_found"):
            return proc, {"service_preexisting": False, "started_by_this_run": True, "pid": proc.pid, "command": "CUDA_VISIBLE_DEVICES=0,1,2,3 " + " ".join(cmd), "log": rel(VLLM_LOG), "health": last_health}
        time.sleep(10)
    raise RuntimeError(f"vLLM health check failed after {args.service_wait_seconds}s; last={last_health}; see {rel(VLLM_LOG)}")


def stop_vllm(proc: subprocess.Popen[str] | None) -> dict[str, Any]:
    if proc is None:
        return {"stopped": False, "reason": "preexisting_or_not_started"}
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


def request_chat(messages: list[dict[str, str]], args: argparse.Namespace) -> tuple[str, dict[str, Any], float]:
    payload = {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": args.max_tokens}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{args.api_base.rstrip('/')}/chat/completions", data=data, headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.perf_counter()
    try:
        with opener.open(req, timeout=args.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            elapsed = time.perf_counter() - start
            parsed = json.loads(body)
            return str(parsed["choices"][0]["message"]["content"]), parsed, elapsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"http_{exc.code}:{body[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}:{exc}") from exc


def annotate_item(item: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sentences = select_sentences(item, max_chars=args.max_prompt_chars, max_sentences=args.max_sentences)
    raw_record: dict[str, Any] = {"event_mention_id": item["event_mention_id"], "article_id": item["article_id"], "attempts": []}
    logical = 1
    http_attempts = 0
    retries = 0
    http_5xx = timeout = oom = 0
    final_errors: list[str] = []
    final_record = failed_record(item, ["not_run"])
    final_meta: dict[str, Any] = {}
    for attempt in range(2):
        if attempt:
            retries += 1
        messages = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user_prompt(item, sentences)}]
        if attempt:
            messages = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": correction_prompt(raw_record["attempts"][-1].get("raw_text", ""), final_errors)}]
        try:
            http_attempts += 1
            raw_text, payload, elapsed = request_chat(messages, args)
            usage = payload.get("usage") or {}
            record, meta = parse_model_response(raw_text, item, sentences)
            raw_record["attempts"].append(
                {
                    "attempt": attempt + 1,
                    "elapsed_seconds": round(elapsed, 6),
                    "usage": usage,
                    "raw_text": raw_text,
                    "json_success": bool(meta.get("json_success")),
                    "draft_status": meta.get("draft_status"),
                    "errors": meta.get("errors", []),
                }
            )
            final_record, final_meta = record, meta
            final_errors = list(meta.get("errors", []))
            if meta.get("json_success") and record.get("draft_quality_grade") in {"A", "B", "C"}:
                break
        except RuntimeError as exc:
            msg = str(exc)
            if "http_5" in msg:
                http_5xx += 1
            if "timeout" in msg.lower() or "Timeout" in msg:
                timeout += 1
            if "oom" in msg.lower() or "cuda" in msg.lower():
                oom += 1
            final_errors = [msg]
            raw_record["attempts"].append({"attempt": attempt + 1, "valid": False, "errors": final_errors})
            final_record = failed_record(item, final_errors)
            final_meta = {"json_success": False, "draft_status": "D", "errors": final_errors}
        except Exception as exc:
            final_errors = [f"{type(exc).__name__}:{exc}"]
            raw_record["attempts"].append({"attempt": attempt + 1, "valid": False, "errors": final_errors})
            final_record = failed_record(item, final_errors)
            final_meta = {"json_success": False, "draft_status": "D", "errors": final_errors}
    attempts = raw_record["attempts"]
    metric = {
        "event_mention_id": item["event_mention_id"],
        "article_id": item["article_id"],
        "split": item.get("split", ""),
        "logical_request_count": logical,
        "http_attempt_count": http_attempts,
        "retry_count": retries,
        "http_5xx_count": http_5xx,
        "timeout_count": timeout,
        "cuda_oom_count": oom,
        "json_success": bool(final_meta.get("json_success")),
        "draft_status": final_record.get("draft_quality_grade", "D"),
        "elapsed_seconds": sum(float(a.get("elapsed_seconds", 0.0) or 0.0) for a in attempts),
        "total_tokens": sum(int((a.get("usage") or {}).get("total_tokens", 0) or 0) for a in attempts),
        "prompt_tokens": sum(int((a.get("usage") or {}).get("prompt_tokens", 0) or 0) for a in attempts),
        "completion_tokens": sum(int((a.get("usage") or {}).get("completion_tokens", 0) or 0) for a in attempts),
        "quote_nonempty_count": int(final_meta.get("quote_nonempty_count", 0) or 0),
        "quote_located_count": int(final_meta.get("quote_located_count", 0) or 0),
        "ambiguous_field_count": int(final_meta.get("ambiguous_field_count", 0) or 0),
        "unresolvable_field_count": int(final_meta.get("unresolvable_field_count", 0) or 0),
        "offset_error_count": len(validate_offsets(final_record)),
        "illegal_direction_count": int(final_meta.get("illegal_direction_count", 0) or 0),
        "illegal_event_name_count": int(final_meta.get("illegal_event_name_count", 0) or 0),
        "cross_article_reference_count": int(final_meta.get("cross_article_reference_count", 0) or 0),
        "multi_event_alignment_risk": bool(final_meta.get("multi_event_alignment_risk")),
        "validation_errors": final_record.get("validation_errors", []),
    }
    sentence_input = {
        "event_mention_id": item["event_mention_id"],
        "article_id": item["article_id"],
        "event_name": item["event_name"],
        "selected_sentence_count": len(sentences),
        "sentences": sentences,
        "sentence_input_sha256": stable_json_sha(sentences),
    }
    return final_record, raw_record, metric | {"sentence_input_sha256": sentence_input["sentence_input_sha256"]}, sentence_input


def annotate_all(items: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    draft_path = OUT_DIR / "annotation_drafts_300.jsonl"
    raw_path = OUT_DIR / "raw_responses.jsonl"
    metrics_path = OUT_DIR / "request_metrics.jsonl"
    sent_path = OUT_DIR / "sentence_inputs.jsonl"
    existing = {row["event_mention_id"]: row for row in read_jsonl(draft_path)}
    raw_existing = {row["event_mention_id"]: row for row in read_jsonl(raw_path)}
    metric_existing = {row["event_mention_id"]: row for row in read_jsonl(metrics_path)}
    sent_existing = {row["event_mention_id"]: row for row in read_jsonl(sent_path)}
    ordered_ids = [row["event_mention_id"] for row in items]
    for idx, item in enumerate(items, start=1):
        mid = item["event_mention_id"]
        if mid in existing:
            continue
        draft, raw, metric, sentence_input = annotate_item(item, args)
        append_jsonl(draft_path, draft)
        append_jsonl(raw_path, raw)
        append_jsonl(metrics_path, metric)
        append_jsonl(sent_path, sentence_input)
        print(json.dumps({"progress": idx, "total": len(items), "event_mention_id": mid, "grade": draft.get("draft_quality_grade"), "errors": len(draft.get("validation_errors", []))}, ensure_ascii=False), flush=True)
    drafts = {row["event_mention_id"]: row for row in read_jsonl(draft_path)}
    raws = {row["event_mention_id"]: row for row in read_jsonl(raw_path)}
    metrics = {row["event_mention_id"]: row for row in read_jsonl(metrics_path)}
    sents = {row["event_mention_id"]: row for row in read_jsonl(sent_path)}
    missing = [mid for mid in ordered_ids if mid not in drafts]
    if missing:
        raise RuntimeError(f"missing drafts after annotation: {missing[:10]}")
    return [drafts[mid] for mid in ordered_ids], [raws.get(mid, {}) for mid in ordered_ids], [metrics.get(mid, {}) for mid in ordered_ids], [sents.get(mid, {}) for mid in ordered_ids]


def summarize_quality(drafts: list[dict[str, Any]], metrics: list[dict[str, Any]], official_names: set[str], split_manifest: dict[str, Any]) -> dict[str, Any]:
    n = len(drafts)
    grade_counts = Counter(row.get("draft_quality_grade", "") for row in drafts)
    field_rates: dict[str, Any] = {}
    for field in ["ai_trigger", "ai_event_span", "ai_subject_company", "ai_event_time", "ai_object_target"]:
        nonempty = 0
        strict = 0
        for row in drafts:
            span = row.get(field, {})
            if isinstance(span, dict) and span.get("text"):
                nonempty += 1
                content = str(row.get("content", ""))
                if 0 <= int(span.get("start", -1)) <= int(span.get("end", -1)) <= len(content) and content[int(span["start"]) : int(span["end"])] == span["text"]:
                    strict += 1
        field_rates[field] = {"nonempty": nonempty, "strict": strict, "strict_rate": strict / n if n else 0.0}
    total_tokens = [int(row.get("total_tokens", 0) or 0) for row in metrics]
    elapsed = [float(row.get("elapsed_seconds", 0.0) or 0.0) for row in metrics]
    json_success = sum(int(bool(row.get("json_success"))) for row in metrics)
    offset_errors = sum(int(row.get("offset_error_count", 0) or 0) for row in metrics)
    cross = sum(int(row.get("cross_article_reference_count", 0) or 0) for row in metrics)
    illegal_event = sum(1 for row in drafts if row.get("event_name") not in official_names) + sum(int(row.get("illegal_event_name_count", 0) or 0) for row in metrics)
    illegal_dir = sum(1 for row in drafts if row.get("ai_direction") not in DIRECTIONS) + sum(int(row.get("illegal_direction_count", 0) or 0) for row in metrics)
    ab = grade_counts.get("A", 0) + grade_counts.get("B", 0)
    gate = {
        "exactly_300": n == 300,
        "event_mention_id_unique": len({row["event_mention_id"] for row in drafts}) == 300,
        "train_test_article_isolated": not split_manifest.get("article_overlap"),
        "history_and_1029_intersection_zero": True,
        "class_coverage_reported": True,
        "json_success_rate_ge_0_95": json_success / n >= 0.95 if n else False,
        "a_b_ratio_ge_0_75": ab / n >= 0.75 if n else False,
        "trigger_strict_rate_ge_0_80": field_rates["ai_trigger"]["strict_rate"] >= 0.80,
        "event_span_strict_rate_ge_0_80": field_rates["ai_event_span"]["strict_rate"] >= 0.80,
        "offset_error_zero": offset_errors == 0,
        "cross_article_reference_zero": cross == 0,
        "illegal_event_name_zero": illegal_event == 0,
        "known_pre_run_gold_test_probe_absent": False,
    }
    return {
        "record_count": n,
        "grade_counts": dict(grade_counts),
        "grade_ratios": {grade: grade_counts.get(grade, 0) / n if n else 0.0 for grade in ["A", "B", "C", "D"]},
        "json_success_count": json_success,
        "json_success_rate": json_success / n if n else 0.0,
        "field_strict_location": field_rates,
        "ambiguous_count": sum(int(row.get("ambiguous_field_count", 0) or 0) for row in metrics),
        "unresolvable_count": sum(int(row.get("unresolvable_field_count", 0) or 0) for row in metrics),
        "illegal_event_name_count": illegal_event,
        "illegal_direction_count": illegal_dir,
        "offset_error_count": offset_errors,
        "cross_article_reference_count": cross,
        "multi_event_alignment_risk_count": sum(1 for row in drafts if row.get("multi_event_alignment_risk")),
        "needs_manual_correction_count": sum(1 for row in drafts if row.get("needs_manual_correction") == "yes" or row.get("draft_quality_grade") in {"B", "C", "D"}),
        "request_summary": {
            "logical_calls": sum(int(row.get("logical_request_count", 0) or 0) for row in metrics),
            "http_attempts": sum(int(row.get("http_attempt_count", 0) or 0) for row in metrics),
            "retries": sum(int(row.get("retry_count", 0) or 0) for row in metrics),
            "http_5xx": sum(int(row.get("http_5xx_count", 0) or 0) for row in metrics),
            "timeout": sum(int(row.get("timeout_count", 0) or 0) for row in metrics),
            "cuda_oom": sum(int(row.get("cuda_oom_count", 0) or 0) for row in metrics),
            "total_tokens": sum(total_tokens),
            "avg_total_tokens": sum(total_tokens) / len(total_tokens) if total_tokens else 0.0,
            "p95_total_tokens": percentile(total_tokens, 95),
            "total_elapsed_seconds": sum(elapsed),
            "avg_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else 0.0,
            "p95_elapsed_seconds": percentile(elapsed, 95),
        },
        "gate": gate,
    }


def percentile(values: list[float] | list[int], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def dataframe_rows(drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in drafts:
        out.append(
            {
                "event_mention_id": row.get("event_mention_id", ""),
                "article_id": row.get("article_id", ""),
                "split": row.get("split", ""),
                "event_name": row.get("event_name", ""),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "publish_time": row.get("publish_time", ""),
                "target_name": row.get("target_name", ""),
                "source": row.get("source", ""),
                "annotation_notice": "AI辅助审核模板，不是独立人工金标，未经确认不得训练或评估。",
                "draft_quality_grade": row.get("draft_quality_grade", ""),
                "needs_manual_correction": row.get("needs_manual_correction", ""),
                "multi_event_alignment_risk": row.get("multi_event_alignment_risk", ""),
                "validation_errors": ";".join(row.get("validation_errors", [])),
                "ai_trigger": json.dumps(row.get("ai_trigger", {}), ensure_ascii=False),
                "ai_event_span": json.dumps(row.get("ai_event_span", {}), ensure_ascii=False),
                "ai_subject_company": json.dumps(row.get("ai_subject_company", {}), ensure_ascii=False),
                "ai_event_time": json.dumps(row.get("ai_event_time", {}), ensure_ascii=False),
                "ai_event_time_normalized": (row.get("ai_event_time", {}) or {}).get("normalized", ""),
                "ai_object_target": json.dumps(row.get("ai_object_target", {}), ensure_ascii=False),
                "ai_direction": row.get("ai_direction", ""),
                "ai_arguments_json": json.dumps(row.get("ai_arguments", []), ensure_ascii=False),
                "ai_argument_roles_json": json.dumps(row.get("ai_argument_roles", []), ensure_ascii=False),
                "ai_confidence": row.get("ai_confidence", ""),
                **{field: "" for field in HUMAN_FIELDS},
            }
        )
    return out


def write_sheet(wb: Workbook, title: str, rows: list[dict[str, Any]], headers: list[str] | None = None, color: bool = True) -> None:
    ws = wb.create_sheet(title)
    if headers is None:
        headers = sorted({key for row in rows for key in row.keys()}) if rows else []
    fills = {
        "header": PatternFill("solid", fgColor="1F4E78"),
        "ai": PatternFill("solid", fgColor="D9EAF7"),
        "human": PatternFill("solid", fgColor="FCE4D6"),
        "risk": PatternFill("solid", fgColor="F4CCCC"),
        "multi": PatternFill("solid", fgColor="FFF2CC"),
    }
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(1, col, header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = fills["header"]
    for r_idx, row in enumerate(rows, start=2):
        grade = str(row.get("draft_quality_grade", ""))
        multi = str(row.get("multi_event_alignment_risk", "")).lower() in {"true", "1", "yes"}
        for c_idx, header in enumerate(headers, start=1):
            value = row.get(header, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            cell = ws.cell(r_idx, c_idx, value)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if color and header.startswith("ai_"):
                cell.fill = fills["ai"]
            if color and header.startswith("human_") or header in {"reviewer_confirmed", "reviewer_note"}:
                cell.fill = fills["human"]
            if color and grade in {"C", "D"}:
                cell.fill = fills["risk"]
            if color and multi:
                cell.fill = fills["multi"]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    widths = {"content": 80, "title": 36, "validation_errors": 40, "ai_event_span": 60}
    for idx, header in enumerate(headers, start=1):
        ws.column_dimensions[ws.cell(1, idx).column_letter].width = widths.get(header, 18)
    if "human_direction" in headers:
        col = headers.index("human_direction") + 1
        dv = DataValidation(type="list", formula1='"positive,negative,neutral,unclear"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{ws.cell(2, col).coordinate}:{ws.cell(max(2, len(rows) + 1), col).coordinate}")
    if "reviewer_confirmed" in headers:
        col = headers.index("reviewer_confirmed") + 1
        dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{ws.cell(2, col).coordinate}:{ws.cell(max(2, len(rows) + 1), col).coordinate}")


def write_workbooks(drafts: list[dict[str, Any]], sample_manifest: dict[str, Any], quality: dict[str, Any]) -> None:
    rows = dataframe_rows(drafts)
    main_headers = list(rows[0].keys()) if rows else []
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    instructions = [
        {"item": "用途", "description": "AI辅助审核模板，不是独立人工金标，未经确认不得训练或评估。"},
        {"item": "填写", "description": "只填写human_*、reviewer_confirmed、reviewer_note；AI字段只作参考。"},
        {"item": "reviewer_confirmed", "description": "人工逐条确认后填yes；需要修正或不确定填no并说明。"},
        {"item": "offset", "description": "所有非空文本必须是content中的逐字连续片段。"},
        {"item": "independent_frozen_test_60", "description": "从现在起不得用于Prompt、规则、阈值或模型选择。"},
    ]
    write_sheet(wb, "使用说明", instructions, ["item", "description"], color=False)
    write_sheet(wb, "300条主审核表", rows, main_headers)
    write_sheet(wb, "expansion_train_240", [r for r in rows if r["split"] == "expansion_train"], main_headers)
    write_sheet(wb, "independent_frozen_test_60", [r for r in rows if r["split"] == "independent_frozen_test"], main_headers)
    write_sheet(wb, "A级低风险", [r for r in rows if r["draft_quality_grade"] == "A"], main_headers)
    write_sheet(wb, "B／C／D高风险", [r for r in rows if r["draft_quality_grade"] in {"B", "C", "D"}], main_headers)
    write_sheet(wb, "多事件文章", [r for r in rows if str(r["multi_event_alignment_risk"]).lower() in {"true", "1", "yes"}], main_headers)
    offset_rows = []
    for row in drafts:
        for field in ["ai_trigger", "ai_event_span", "ai_subject_company", "ai_event_time", "ai_object_target"]:
            span = row.get(field, {})
            offset_rows.append({"event_mention_id": row["event_mention_id"], "article_id": row["article_id"], "field": field, "text": span.get("text", "") if isinstance(span, dict) else "", "start": span.get("start", "") if isinstance(span, dict) else "", "end": span.get("end", "") if isinstance(span, dict) else "", "strict": not validate_offsets({**row, field: span})})
    write_sheet(wb, "字段与Offset详情", offset_rows, ["event_mention_id", "article_id", "field", "text", "start", "end", "strict"], color=False)
    cov_rows = [{"event_name": name, "selected_count": count} for name, count in sample_manifest["coverage_by_event_name"].items()]
    write_sheet(wb, "类别覆盖统计", cov_rows, ["event_name", "selected_count"], color=False)
    err_counter = Counter()
    for row in drafts:
        err_counter.update(row.get("validation_errors", []))
    err_rows = [{"error_type": key, "count": val} for key, val in sorted(err_counter.items(), key=lambda x: (-x[1], x[0]))]
    err_rows.extend([
        {"error_type": "ambiguous_total", "count": quality["ambiguous_count"]},
        {"error_type": "unresolvable_total", "count": quality["unresolvable_count"]},
    ])
    write_sheet(wb, "错误类型统计", err_rows, ["error_type", "count"], color=False)
    REVIEW_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(REVIEW_XLSX)
    shutil.copy2(REVIEW_XLSX, DETAILS_XLSX)


def write_markdown(report: dict[str, Any]) -> None:
    quality = report["quality"]
    lines = [
        "# Task1 Step 5.7-I 结构训练数据扩充报告",
        "",
        f"- final_status: `{report['status']}`",
        f"- artifact_type: `{report['artifact_type']}`",
        f"- sampled_eventmentions: {report['sampling']['selected_eventmention_count']}",
        f"- split: expansion_train={report['split']['expansion_train_count']}, independent_frozen_test={report['split']['independent_frozen_test_count']}",
        f"- official_event_name_count: {report['sampling']['official_event_name_count']}",
        f"- grade_counts: {json.dumps(quality['grade_counts'], ensure_ascii=False)}",
        f"- json_success_rate: {quality['json_success_rate']:.4f}",
        f"- A+B ratio: {(quality['grade_counts'].get('A',0)+quality['grade_counts'].get('B',0))/max(1,quality['record_count']):.4f}",
        f"- trigger_strict_rate: {quality['field_strict_location']['ai_trigger']['strict_rate']:.4f}",
        f"- event_span_strict_rate: {quality['field_strict_location']['ai_event_span']['strict_rate']:.4f}",
        f"- known_protection_deviation: {report['protection']['known_pre_run_gold_test_probe']}",
        "",
        "本产物是AI辅助审核模板，不是独立人工金标，未经确认不得训练或评估。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    GUIDE_MD.write_text(
        "# Task1 Step 5.7-I 审核指南\n\n"
        "本工作簿为AI辅助审核模板，不是独立人工金标，未经确认不得训练或评估。\n\n"
        "请逐条检查AI字段是否与正文逐字匹配，并在human_*字段填写最终人工确认结果。"
        "`reviewer_confirmed`只可填写`yes`或`no`。独立冻结测试集60条不得用于Prompt、规则、阈值或模型选择。\n",
        encoding="utf-8",
    )


def sha_outputs() -> dict[str, str]:
    paths = [
        OUT_DIR / "sampled_eventmentions_300.jsonl",
        OUT_DIR / "sampling_manifest.json",
        OUT_DIR / "split_manifest.json",
        OUT_DIR / "expansion_train_240_source.jsonl",
        OUT_DIR / "independent_frozen_test_60_source.jsonl",
        OUT_DIR / "annotation_drafts_300.jsonl",
        OUT_DIR / "expansion_train_240_ai_drafts.jsonl",
        OUT_DIR / "independent_frozen_test_60_ai_drafts.jsonl",
        OUT_DIR / "raw_responses.jsonl",
        OUT_DIR / "request_metrics.jsonl",
        OUT_DIR / "sentence_inputs.jsonl",
        REPORT_JSON,
        REPORT_MD,
        DETAILS_XLSX,
        REVIEW_XLSX,
        GUIDE_MD,
    ]
    return {rel(path): sha256_file(path) for path in paths if path.exists()}


def existing_complete() -> bool:
    return (OUT_DIR / "annotation_drafts_300.jsonl").exists() and len(read_jsonl(OUT_DIR / "annotation_drafts_300.jsonl")) == 300 and REPORT_JSON.exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--service-wait-seconds", type=int, default=900)
    parser.add_argument("--max-prompt-chars", type=int, default=5600)
    parser.add_argument("--max-sentences", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if existing_complete():
        print(json.dumps({"status": read_json(REPORT_JSON).get("status"), "message": "existing_complete_no_regeneration", "report": rel(REPORT_JSON)}, ensure_ascii=False))
        return 0
    if process_check():
        raise RuntimeError(f"another Step 5.7-I/vLLM related process is running: {process_check()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7i_structure_expansion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    recovery = {
        "generated_at": now(),
        "existing_step57i_artifacts_before_run": sorted(rel(path) for path in list(OUT_DIR.glob("*")) + [REPORT_JSON, REPORT_MD, DETAILS_XLSX, REVIEW_XLSX, GUIDE_MD] if path.exists()),
        "process_check": process_check(),
        "gpu_status": gpu_status(),
        "port_status": port_status(),
        "backup_dir": str(backup_dir),
    }
    if not recovery["gpu_status"]["all_four_visible"]:
        raise RuntimeError(f"expected four GPUs; got {recovery['gpu_status']}")
    if recovery["gpu_status"]["compute_apps"]:
        raise RuntimeError(f"GPU is occupied by another task: {recovery['gpu_status']}")

    protected_before = sha_snapshot(protected_paths())
    write_sha_text(OUT_DIR / "protected_sha256_before.txt", protected_before)
    write_json(backup_dir / "protected_sha256_before.json", protected_before)

    schema_by_id, official_names, descriptions = load_event_schema()
    all_mentions = load_official_mentions(schema_by_id, descriptions)
    exclusion_sets = collect_exclusion_sets()
    excluded_articles = set().union(*exclusion_sets.values())
    sampled, sampling_manifest = sample_300(all_mentions, official_names, excluded_articles, seed=args.seed)
    overlap = {name: sorted({row["article_id"] for row in sampled} & ids) for name, ids in exclusion_sets.items()}
    sampling_manifest["exclusion_set_counts"] = {name: len(ids) for name, ids in exclusion_sets.items()}
    sampling_manifest["exclusion_intersections"] = {name: len(values) for name, values in overlap.items()}
    sampling_manifest["exclusion_intersection_ids"] = {name: values[:100] for name, values in overlap.items() if values}
    sampling_manifest["gold_test_labels_read_by_step57i_script"] = False
    sampling_manifest["gold_test_article_ids_source"] = rel(GOLD1029_BLIND_INPUT)
    write_jsonl(OUT_DIR / "sampled_eventmentions_300.jsonl", sampled)
    write_json(OUT_DIR / "sampling_manifest.json", sampling_manifest)

    train_rows, test_rows, split_manifest = split_240_60(sampled, seed=args.seed)
    write_jsonl(OUT_DIR / "expansion_train_240_source.jsonl", train_rows)
    write_jsonl(OUT_DIR / "independent_frozen_test_60_source.jsonl", test_rows)
    split_manifest["independent_frozen_test_60_source_sha256"] = sha256_file(OUT_DIR / "independent_frozen_test_60_source.jsonl")
    split_manifest["gold_test_labels_read"] = False
    write_json(OUT_DIR / "split_manifest.json", split_manifest)

    vllm_proc: subprocess.Popen[str] | None = None
    vllm_start: dict[str, Any] = {}
    vllm_stop: dict[str, Any] = {}
    try:
        vllm_proc, vllm_start = start_vllm_if_needed(args)
        drafts, raw_rows, metrics, sentence_inputs = annotate_all(train_rows + test_rows, args)
    finally:
        vllm_stop = stop_vllm(vllm_proc)
    write_jsonl(OUT_DIR / "expansion_train_240_ai_drafts.jsonl", [row for row in drafts if row.get("split") == "expansion_train"])
    write_jsonl(OUT_DIR / "independent_frozen_test_60_ai_drafts.jsonl", [row for row in drafts if row.get("split") == "independent_frozen_test"])

    quality = summarize_quality(drafts, metrics, official_names, split_manifest)
    if any(sampling_manifest["exclusion_intersections"].values()):
        quality["gate"]["history_and_1029_intersection_zero"] = False
    status = STATUS_READY if all(quality["gate"].values()) else STATUS_REVISION
    protected_after = sha_snapshot(protected_paths())
    write_sha_text(OUT_DIR / "protected_sha256_after.txt", protected_after)
    protection_match = protected_before == protected_after
    if not protection_match:
        status = STATUS_REVISION
    report = {
        "generated_at": now(),
        "status": status,
        "artifact_type": "ai_assisted_human_review_template_only",
        "gold": False,
        "confirmed": False,
        "training_ready": False,
        "recovery": recovery,
        "sampling": sampling_manifest,
        "split": split_manifest,
        "quality": quality,
        "protection": {
            "protected_sha256_before": protected_before,
            "protected_sha256_after": protected_after,
            "protected_sha256_match": protection_match,
            "formal_task1_task2_modified": not protection_match,
            "gold_test_labels_read_by_step57i_script": False,
            "known_pre_run_gold_test_probe": True,
            "known_pre_run_gold_test_probe_note": KNOWN_PRE_RUN_DEVIATION,
        },
        "vllm": {"start": vllm_start, "stop": vllm_stop, "final_port_status": port_status(), "final_gpu_status": gpu_status()},
        "outputs": {},
        "backup_dir": str(backup_dir),
    }
    write_workbooks(drafts, sampling_manifest, quality)
    write_markdown(report)
    write_json(REPORT_JSON, report)
    output_sha = sha_outputs()
    write_json(OUT_DIR / "SHA256SUMS.json", output_sha)
    report["outputs"] = output_sha
    write_json(REPORT_JSON, report)
    write_json(backup_dir / "run_manifest.json", report)
    print(json.dumps({"status": status, "sampled": len(sampled), "train": len(train_rows), "test": len(test_rows), "report": rel(REPORT_JSON), "backup_dir": str(backup_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
