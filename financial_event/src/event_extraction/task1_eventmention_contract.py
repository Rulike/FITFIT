#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventMention-level contract helpers for Task1 structure annotations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Any

from src.event_extraction.task1_structure_offset_adapter import adapt_text_only_response

DIRECTION_LABELS = ["positive", "negative", "neutral", "unclear"]
ARGUMENT_ROLES = ["subject_company", "event_time", "object_target", "trigger"]
SPAN_FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
HUMAN_REVIEW_FIELDS = [
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


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def stable_event_mention_id(article_id: str, event_name: str, schema_id: str | None = None) -> str:
    key = f"{article_id}|{event_name}|{schema_id or ''}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"EVM_{digest}"


def empty_span() -> dict[str, Any]:
    return {"text": "", "start": -1, "end": -1}


def empty_time() -> dict[str, Any]:
    return {"text": "", "normalized": "未明确", "start": -1, "end": -1}


def contract_schema() -> dict[str, Any]:
    span = {
        "type": "object",
        "required": ["text", "start", "end"],
        "properties": {"text": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
        "additionalProperties": False,
    }
    time_span = {
        "type": "object",
        "required": ["text", "normalized", "start", "end"],
        "properties": {
            "text": {"type": "string"},
            "normalized": {"type": "string"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Task1 EventMention Structure Annotation Contract",
        "type": "object",
        "required": [
            "event_mention_id",
            "article_id",
            "event_name",
            "title",
            "content",
            "publish_time",
            "trigger",
            "event_span",
            "subject_company",
            "event_time",
            "object_target",
            "direction",
            "arguments",
            "annotation_source",
            "reviewer_confirmed",
            "annotation_status",
        ],
        "properties": {
            "event_mention_id": {"type": "string"},
            "article_id": {"type": "string"},
            "event_name": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "publish_time": {"type": "string"},
            "trigger": span,
            "event_span": span,
            "subject_company": span,
            "event_time": time_span,
            "object_target": span,
            "direction": {"enum": DIRECTION_LABELS},
            "arguments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "role", "start", "end"],
                    "properties": {
                        "text": {"type": "string"},
                        "role": {"enum": ARGUMENT_ROLES},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
            },
            "annotation_source": {"type": "string"},
            "reviewer_confirmed": {"enum": ["yes", "no", ""]},
            "annotation_status": {"type": "string"},
        },
        "additionalProperties": True,
    }


def teacher_output_schema() -> dict[str, Any]:
    return {
        "event_name": "必须与输入event_name完全一致",
        "trigger_text": "正文原文中的触发词或短语",
        "event_span_text": "正文原文中的连续事件事实片段，必须包含trigger_text",
        "subject_company_text": "正文原文中的主体公司/机构/产品/行业主体",
        "event_time_text": "正文原文中的事件时间；没有明确时间则为空字符串",
        "event_time_normalized": "YYYY-MM-DD|未明确",
        "object_target_text": "正文原文中的客体/标的/产品/项目；没有明确客体则为空字符串",
        "direction": "positive|negative|neutral|unclear",
        "arguments": [{"text": "正文原文片段", "role": "subject_company|event_time|object_target|trigger"}],
        "confidence": 0.0,
    }


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。；：、,.!！?？（）()《》“”\"'`·\-—_<>【】\[\]{}]", "", text)
    return text


def normalize_direction(value: Any) -> str:
    text = normalize_text(value)
    if text in {"正面", "积极", "利好", "positive", "pos"}:
        return "positive"
    if text in {"负面", "消极", "利空", "negative", "neg"}:
        return "negative"
    if text in {"中性", "neutral"}:
        return "neutral"
    if text in {"不明确", "未明确", "不清楚", "unclear", "unknown", ""}:
        return "unclear"
    return value if value in DIRECTION_LABELS else "unclear"


def split_sentences(text: str) -> list[str]:
    value = str(text or "").replace("\r\n", "\n")
    pieces = re.split(r"(?<=[。！？；;!\?\n])", value)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def terms_from_schema(event_name: str, event_description: str) -> list[str]:
    source = f"{event_name}\n{event_description}"
    terms = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9\-.]{1,18}", source))
    stop = {"事件定义", "事件示例", "企业", "公司", "行业", "通过", "影响", "用于", "关注", "相关", "进行"}
    return sorted((term for term in terms if term not in stop), key=lambda item: (-len(item), item))[:40]


def build_content_excerpt(title: str, content: str, event_name: str, event_description: str, max_chars: int = 5600) -> dict[str, Any]:
    full = str(content or "")
    if len(full) <= max_chars:
        return {"excerpt": full, "mode": "full", "content_truncated": False, "content_length": len(full)}
    terms = terms_from_schema(event_name, event_description)
    sentences = split_sentences(full)
    selected: list[str] = []
    selected_set: set[str] = set()
    if sentences:
        selected.append(sentences[0])
        selected_set.add(sentences[0])
    scored = []
    for idx, sentence in enumerate(sentences):
        score = 0
        norm_sentence = normalize_text(sentence)
        for term in terms:
            if normalize_text(term) and normalize_text(term) in norm_sentence:
                score += 1
        if event_name and normalize_text(event_name) in norm_sentence:
            score += 3
        scored.append((score, -idx, sentence))
    for score, _, sentence in sorted(scored, reverse=True):
        if score <= 0:
            continue
        if sentence in selected_set:
            continue
        selected.append(sentence)
        selected_set.add(sentence)
        if sum(len(item) for item in selected) >= max_chars * 0.72:
            break
    cursor = 1
    while sum(len(item) for item in selected) < max_chars and cursor < len(sentences):
        sentence = sentences[cursor]
        if sentence not in selected_set:
            selected.append(sentence)
            selected_set.add(sentence)
        cursor += 1
    excerpt = "\n".join(selected)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    return {"excerpt": excerpt, "mode": "schema_relevance_excerpt", "content_truncated": True, "content_length": len(full)}


def build_teacher_prompt() -> str:
    return """你是金融新闻事件结构抽取标注助手。输入已经给出正确的event_name；你只需要为这一类事件抽取一条EventMention结构草稿。

硬性规则：
1. 只输出一个严格JSON对象，不输出Markdown、解释、前后缀文本或额外字段。
2. event_name必须与输入event_name逐字一致。
3. 不要输出start/end/offset。所有非空text字段必须逐字复制输入content中的连续原文片段，禁止改写、概括、补充正文外信息。
4. event_span_text必须是content中的连续片段，覆盖该event_name对应的核心事实，不要整篇新闻。
5. trigger_text必须是event_span_text内部的原文片段。
6. subject_company_text填写事件主体公司、机构、产品或行业主体；必须来自content原文。
7. 正文未明确事件时间时，event_time_text=""，event_time_normalized="未明确"。
8. 正文不存在明确object_target时，object_target_text=""。
9. direction只能是positive、negative、neutral、unclear之一；无法判断时用unclear。
10. arguments只列出实际抽到的subject_company、event_time、object_target、trigger，role必须使用合法枚举。
11. confidence必须是0到1之间的有限数。

字段定义：
- trigger_text：最能表达事件发生的触发词或短语。
- event_span_text：包含事件事实的最小连续文本片段。
- subject_company_text：事件主体公司、机构、产品或行业主体。
- event_time_text/event_time_normalized：事件发生、发布或公告时间；没有明确时间则标为未明确。
- object_target_text：事件影响对象、产品、项目、交易标的或主题对象；没有明确对象时留空。
- direction：事件对主体或行业的影响方向。
"""


def build_teacher_user_message(item: dict[str, Any]) -> str:
    return (
        f"event_mention_id: {item['event_mention_id']}\n"
        f"article_id: {item['article_id']}\n"
        f"event_name: {item['event_name']}\n"
        f"title: {item.get('title', '')}\n"
        f"publish_time: {item.get('publish_time', '')}\n"
        f"字段定义: {item.get('event_description', '')}\n\n"
        "content:\n"
        f"{item.get('content_excerpt', item.get('content', ''))}\n\n"
        "只输出符合以下Schema的严格JSON：\n"
        + json.dumps(teacher_output_schema(), ensure_ascii=False, indent=2)
    )


def correction_user_message(raw_output: str, errors: list[str]) -> str:
    return (
        "你的上一次输出未通过格式校验。只修正JSON格式、字段名、枚举、confidence范围和原文复制问题；不要添加解释，不要输出Markdown。\n"
        f"错误原因：{'; '.join(errors[:12])}\n"
        f"JSON Schema：{json.dumps(teacher_output_schema(), ensure_ascii=False)}\n"
        f"原始模型输出：{raw_output}"
    )


def validate_teacher_payload(payload: Any, expected_event_name: str, content: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return False, ["top_level_not_object"]
    required = [
        "event_name",
        "trigger_text",
        "event_span_text",
        "subject_company_text",
        "event_time_text",
        "event_time_normalized",
        "object_target_text",
        "direction",
        "arguments",
        "confidence",
    ]
    for field in required:
        if field not in payload:
            errors.append(f"missing:{field}")
    if str(payload.get("event_name", "")) != expected_event_name:
        errors.append("illegal_event_name")
    if payload.get("direction") not in DIRECTION_LABELS:
        errors.append("illegal_direction")
    try:
        confidence = float(payload.get("confidence"))
        if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
            errors.append("illegal_confidence")
    except Exception:
        errors.append("illegal_confidence")
    if not isinstance(payload.get("arguments"), list):
        errors.append("arguments_not_list")
    else:
        for idx, arg in enumerate(payload.get("arguments", [])):
            if not isinstance(arg, dict):
                errors.append(f"argument_{idx}_not_object")
                continue
            if str(arg.get("role", "")) not in ARGUMENT_ROLES:
                errors.append(f"argument_{idx}_illegal_role")
            text = str(arg.get("text", "") or "")
            if text and text not in content:
                errors.append(f"argument_{idx}_text_not_in_content")
    span = str(payload.get("event_span_text", "") or "")
    trigger = str(payload.get("trigger_text", "") or "")
    for field in ["trigger_text", "event_span_text", "subject_company_text", "event_time_text", "object_target_text"]:
        text = str(payload.get(field, "") or "")
        if text and text not in content:
            errors.append(f"{field}_not_exact_in_content")
    if trigger and span and trigger not in span:
        errors.append("trigger_not_in_event_span_text")
    if not str(payload.get("event_time_text", "") or "") and str(payload.get("event_time_normalized", "") or "") != "未明确":
        errors.append("empty_time_requires_un明确")
    return not errors, sorted(set(errors))


def eventmention_from_spans(
    base: dict[str, Any],
    spans: dict[str, Any] | None = None,
    annotation_source: str = "",
    reviewer_confirmed: str = "",
    annotation_status: str = "",
) -> dict[str, Any]:
    spans = spans or {}
    return {
        "event_mention_id": base["event_mention_id"],
        "article_id": str(base.get("article_id", "")),
        "event_name": str(base.get("event_name", "")),
        "schema_id": str(base.get("schema_id", "")),
        "target_type": str(base.get("target_type", "")),
        "target_code": str(base.get("target_code", "")),
        "target_name": str(base.get("target_name", "")),
        "title": str(base.get("title", "")),
        "content": str(base.get("content", "")),
        "publish_time": str(base.get("publish_time", "")),
        "trigger": spans.get("trigger", empty_span()),
        "event_span": spans.get("event_span", empty_span()),
        "subject_company": spans.get("subject_company", empty_span()),
        "event_time": spans.get("event_time", empty_time()),
        "object_target": spans.get("object_target", empty_span()),
        "direction": spans.get("direction", "unclear") if spans.get("direction", "unclear") in DIRECTION_LABELS else "unclear",
        "arguments": spans.get("arguments", []),
        "annotation_source": annotation_source,
        "reviewer_confirmed": reviewer_confirmed,
        "annotation_status": annotation_status,
    }


def sanitize_prediction_for_contract(content: str, prediction: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop non-strict spans from contract fields while preserving diagnostics elsewhere."""
    out = dict(prediction or {})
    dropped: list[str] = []
    for field in SPAN_FIELDS:
        span = out.get(field)
        if not isinstance(span, dict):
            out[field] = empty_time() if field == "event_time" else empty_span()
            dropped.append(field)
            continue
        text = str(span.get("text", "") or "")
        try:
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
        except Exception:
            start, end = -1, -1
        strict = bool(text) and 0 <= start <= end <= len(content) and content[start:end] == text
        if text and not strict:
            replacement = empty_time() if field == "event_time" else empty_span()
            if field == "event_time":
                replacement["normalized"] = str(span.get("normalized", "") or "未明确")
            out[field] = replacement
            dropped.append(field)
        elif not text:
            replacement = empty_time() if field == "event_time" else empty_span()
            if field == "event_time":
                replacement["normalized"] = str(span.get("normalized", "") or "未明确")
            out[field] = replacement
        else:
            out[field] = {**span, "start": start, "end": end}
    trigger = out.get("trigger", {})
    event_span = out.get("event_span", {})
    if (
        isinstance(trigger, dict)
        and isinstance(event_span, dict)
        and trigger.get("text")
        and event_span.get("text")
        and not (int(event_span.get("start", -1)) <= int(trigger.get("start", -1)) < int(trigger.get("end", -1)) <= int(event_span.get("end", -1)))
    ):
        out["trigger"] = empty_span()
        dropped.append("trigger")
    safe_arguments = []
    for role in ARGUMENT_ROLES:
        span = out.get(role)
        if isinstance(span, dict) and span.get("text") and int(span.get("start", -1)) >= 0:
            safe_arguments.append({"text": span["text"], "role": role, "start": int(span["start"]), "end": int(span["end"])})
    out["arguments"] = safe_arguments
    return out, sorted(set(dropped))


def validate_eventmention_record(
    row: dict[str, Any],
    official_event_names: set[str],
    require_human_fields_empty: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if not str(row.get("event_mention_id", "")).strip():
        errors.append("missing_event_mention_id")
    if not str(row.get("article_id", "")).strip():
        errors.append("missing_article_id")
    if row.get("event_name") not in official_event_names:
        errors.append("illegal_event_name")
    content = str(row.get("content", ""))
    for field in SPAN_FIELDS:
        span = row.get(field)
        if not isinstance(span, dict):
            errors.append(f"{field}_not_object")
            continue
        text = str(span.get("text", "") or "")
        try:
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
        except Exception:
            errors.append(f"{field}_offset_not_int")
            continue
        if not text:
            if start != -1 or end != -1:
                errors.append(f"{field}_empty_offset_not_minus1")
            continue
        if not (0 <= start <= end <= len(content)):
            errors.append(f"{field}_offset_out_of_range")
        elif content[start:end] != text:
            errors.append(f"{field}_offset_text_mismatch")
    trigger = row.get("trigger") if isinstance(row.get("trigger"), dict) else {}
    span = row.get("event_span") if isinstance(row.get("event_span"), dict) else {}
    if trigger.get("text") and span.get("text"):
        if not (int(span.get("start", -1)) <= int(trigger.get("start", -1)) < int(trigger.get("end", -1)) <= int(span.get("end", -1))):
            errors.append("trigger_not_inside_event_span")
    if row.get("direction") not in DIRECTION_LABELS:
        errors.append("illegal_direction")
    if not isinstance(row.get("arguments"), list):
        errors.append("arguments_not_list")
    else:
        for idx, arg in enumerate(row.get("arguments", [])):
            if not isinstance(arg, dict):
                errors.append(f"argument_{idx}_not_object")
                continue
            if arg.get("role") not in ARGUMENT_ROLES:
                errors.append(f"argument_{idx}_illegal_role")
            text = str(arg.get("text", "") or "")
            start = int(arg.get("start", -1)) if str(arg.get("start", "-1")).lstrip("-").isdigit() else -2
            end = int(arg.get("end", -1)) if str(arg.get("end", "-1")).lstrip("-").isdigit() else -2
            if text and (not (0 <= start <= end <= len(content)) or content[start:end] != text):
                errors.append(f"argument_{idx}_offset_text_mismatch")
    if require_human_fields_empty:
        for field in HUMAN_REVIEW_FIELDS:
            if str(row.get(field, "") or "") != "":
                errors.append(f"{field}_not_empty")
    return {"valid": not errors, "errors": sorted(set(errors))}


def grade_draft(prediction: dict[str, Any], field_rows: list[dict[str, Any]], meta: dict[str, Any], payload_errors: list[str]) -> dict[str, Any]:
    contract_errors = list(payload_errors) + list(meta.get("errors", []))
    ambiguous_fields = sorted({row["field"] for row in field_rows if row.get("ambiguous")})
    unresolvable_fields = sorted({row["field"] for row in field_rows if row.get("unresolvable")})
    key_fields = ["trigger", "event_span", "subject_company"]
    key_located = [
        bool(prediction.get(field, {}).get("text")) and int(prediction.get(field, {}).get("start", -1)) >= 0
        for field in key_fields
    ]
    located_rate = sum(1 for item in key_located if item) / len(key_fields)
    if meta.get("strict_json_parse_failed"):
        grade = "D"
    elif unresolvable_fields and any(field in unresolvable_fields for field in key_fields):
        grade = "C"
    elif ambiguous_fields and any(field in ambiguous_fields for field in key_fields):
        grade = "C"
    elif contract_errors and located_rate < 0.75:
        grade = "C"
    elif ambiguous_fields or unresolvable_fields or contract_errors:
        grade = "B"
    else:
        grade = "A"
    return {
        "draft_status": grade,
        "field_validation": {row["field"]: row.get("status", "") for row in field_rows},
        "unresolvable_fields": unresolvable_fields,
        "ambiguous_fields": ambiguous_fields,
        "contract_errors": sorted(set(contract_errors)),
        "key_field_located_rate": located_rate,
        "needs_human_review": True,
        "review_priority": {"A": "normal", "B": "medium", "C": "high", "D": "urgent"}[grade],
    }


def adapt_text_draft(content: str, raw_content: str, expected_event_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str]]:
    payload_errors: list[str] = []
    try:
        payload = json.loads(raw_content.strip())
        _, payload_errors = validate_teacher_payload(payload, expected_event_name, content)
    except Exception as exc:
        payload_errors = [f"json_parse_failed:{exc}"]
    prediction, field_rows, meta = adapt_text_only_response(content, raw_content, expected_event_name=expected_event_name)
    return prediction, field_rows, meta, payload_errors


def summary_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))
