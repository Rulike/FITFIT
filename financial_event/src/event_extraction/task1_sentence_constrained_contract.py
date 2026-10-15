#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentence-constrained EventMention structure draft contract for Task1 Step 5.7-D."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

DIRECTION_LABELS = ["positive", "negative", "neutral", "unclear"]
ARGUMENT_ROLES = ["subject_company", "event_time", "object_target", "trigger"]
OUTPUT_FIELDS = {
    "event_sentence_ids",
    "trigger",
    "subject_company",
    "event_time",
    "object_target",
    "direction",
    "arguments",
    "confidence",
}


def strict_json_loads(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        raise ValueError("markdown_code_fence")
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError("not_single_json_object")
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("top_level_not_object")
    return payload


def trim_span(content: str, start: int, end: int) -> tuple[int, int]:
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end


def sentence_split_with_offsets(content: str) -> list[dict[str, Any]]:
    text = str(content or "")
    rows: list[dict[str, Any]] = []
    start = 0

    def flush(raw_start: int, raw_end: int) -> None:
        s, e = trim_span(text, raw_start, raw_end)
        if e <= s:
            return
        rows.append(
            {
                "sentence_id": f"S{len(rows) + 1:03d}",
                "sentence_index": len(rows),
                "sentence_text": text[s:e],
                "article_start": s,
                "article_end": e,
            }
        )

    for idx, char in enumerate(text):
        if char in "。！？；;!?":
            flush(start, idx + 1)
            start = idx + 1
        elif char == "\n":
            flush(start, idx)
            start = idx + 1
    flush(start, len(text))
    return rows


def normalize(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def terms(event_name: str, description: str, title: str, target_name: str = "") -> list[str]:
    source = f"{event_name}\n{description}\n{title}\n{target_name}"
    raw = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9+][\u4e00-\u9fffA-Za-z0-9+\-.]{1,18}", source))
    stop = {"事件定义", "事件示例", "公司", "企业", "行业", "通过", "影响", "用于", "关注", "进行", "相关"}
    return sorted((item for item in raw if item not in stop), key=lambda item: (-len(item), item))[:48]


def select_sentences_for_prompt(item: dict[str, Any], max_chars: int = 5600, max_sentences: int = 80) -> list[dict[str, Any]]:
    sentences = sentence_split_with_offsets(str(item.get("content", "")))
    if not sentences:
        return []
    selected: dict[str, dict[str, Any]] = {}
    used_chars = 0

    def add(row: dict[str, Any]) -> bool:
        nonlocal used_chars
        sid = row["sentence_id"]
        if sid in selected:
            return True
        projected = used_chars + len(row["sentence_text"]) + 32
        if len(selected) >= max_sentences or projected > max_chars:
            return False
        selected[sid] = row
        used_chars = projected
        return True

    for row in sentences[:8]:
        add(row)
    term_list = terms(str(item.get("event_name", "")), str(item.get("event_description", "")), str(item.get("title", "")), str(item.get("target_name", "")))
    scored = []
    for row in sentences:
        norm_sentence = normalize(row["sentence_text"])
        score = 0
        for term in term_list:
            if normalize(term) and normalize(term) in norm_sentence:
                score += 1
        if str(item.get("target_name", "")) and normalize(item.get("target_name", "")) in norm_sentence:
            score += 3
        scored.append((score, -row["sentence_index"], row))
    for score, _, row in sorted(scored, reverse=True):
        if score <= 0:
            continue
        add(row)
    cursor = 8
    while cursor < len(sentences) and len(selected) < max_sentences and used_chars < max_chars:
        add(sentences[cursor])
        cursor += 1
    return sorted(selected.values(), key=lambda row: row["sentence_index"])


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
        "required": ["event_sentence_ids", "trigger", "subject_company", "event_time", "object_target", "direction", "arguments", "confidence"],
        "properties": {
            "event_sentence_ids": {"type": "array", "items": {"type": "string"}},
            "trigger": quote_obj,
            "subject_company": quote_obj,
            "event_time": time_obj,
            "object_target": quote_obj,
            "direction": {"enum": DIRECTION_LABELS},
            "arguments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["sentence_id", "quote", "role"],
                    "properties": {
                        "sentence_id": {"type": "string"},
                        "quote": {"type": "string"},
                        "role": {"enum": ARGUMENT_ROLES},
                    },
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    }


def build_prompt() -> str:
    return """你是金融新闻EventMention结构标注助手。输入会给出正确event_name和带编号的正文句子。你只输出JSON，不输出解释。

硬性规则：
1. 不要输出event_name、article_id、event_mention_id或offset；这些由程序继承。
2. 只能从给定sentence_id中选择证据句。
3. quote必须是对应sentence_text中的连续原文子串，禁止改写、概括、增删标点或跨句拼接。
4. event_sentence_ids填写承载该事件事实的句子ID，通常1到2句；event span由程序用这些原句确定。
5. trigger.quote必须来自event_sentence_ids之一。
6. 无明确字段时sentence_id=""且quote=""；时间不明确时normalized="未明确"。
7. direction只能是positive、negative、neutral、unclear。
8. arguments只允许引用合法sentence_id中的连续原文，role只能是subject_company、event_time、object_target、trigger。
9. confidence是0到1之间的有限数。
10. 严格输出一个JSON对象，不要Markdown代码块、注释或额外字段。
"""


def build_user_message(item: dict[str, Any], sentences: list[dict[str, Any]]) -> str:
    lines = [
        f"event_name: {item['event_name']}",
        f"title: {item.get('title', '')}",
        f"publish_time: {item.get('publish_time', '')}",
        f"event_definition: {item.get('event_description', '')}",
        "",
        "sentences:",
    ]
    for row in sentences:
        lines.append(f"{row['sentence_id']}: {row['sentence_text']}")
    lines.extend(["", "只输出符合以下JSON Schema的对象：", json.dumps(output_schema(), ensure_ascii=False)])
    return "\n".join(lines)


def correction_message(raw_output: str, errors: list[str]) -> str:
    return (
        "上一次输出没有通过工程契约。只修正JSON格式、字段名、sentence_id、quote原文复制、枚举和confidence，不添加解释。\n"
        f"错误原因：{'; '.join(errors[:16])}\n"
        f"JSON Schema：{json.dumps(output_schema(), ensure_ascii=False)}\n"
        f"原始输出：{raw_output}"
    )


def empty_span() -> dict[str, Any]:
    return {"text": "", "start": -1, "end": -1}


def empty_time(normalized: str = "未明确") -> dict[str, Any]:
    return {"text": "", "normalized": normalized or "未明确", "start": -1, "end": -1}


def quote_to_span(field: str, value: Any, sentence_by_id: dict[str, dict[str, Any]], allow_empty: bool = True) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    errors: list[str] = []
    meta = {"field": field, "sentence_id_valid": True, "quote_located": False, "ambiguous": False, "unresolvable": False, "quote_nonempty": False}
    if not isinstance(value, dict):
        return empty_time() if field == "event_time" else empty_span(), [f"{field}:not_object"], {**meta, "sentence_id_valid": False}
    sid = str(value.get("sentence_id", "") or "")
    quote = str(value.get("quote", "") or "")
    normalized = str(value.get("normalized", "") or "未明确") if field == "event_time" else ""
    if not sid and not quote:
        return empty_time(normalized) if field == "event_time" else empty_span(), [], meta
    meta["quote_nonempty"] = bool(quote)
    if not sid or sid not in sentence_by_id:
        errors.append(f"{field}:illegal_sentence_id")
        meta["sentence_id_valid"] = False
        return empty_time(normalized) if field == "event_time" else empty_span(), errors, meta
    sentence = sentence_by_id[sid]["sentence_text"]
    positions = [m.start() for m in re.finditer(re.escape(quote), sentence)] if quote else []
    if not quote:
        if allow_empty:
            return empty_time(normalized) if field == "event_time" else empty_span(), [], meta
        errors.append(f"{field}:empty_quote")
        return empty_time(normalized) if field == "event_time" else empty_span(), errors, meta
    if len(positions) == 1:
        start = int(sentence_by_id[sid]["article_start"]) + positions[0]
        end = start + len(quote)
        meta["quote_located"] = True
        if field == "event_time":
            return {"text": quote, "normalized": normalized or "未明确", "start": start, "end": end}, [], meta
        return {"text": quote, "start": start, "end": end}, [], meta
    if len(positions) > 1:
        errors.append(f"{field}:ambiguous_quote")
        meta["ambiguous"] = True
    else:
        errors.append(f"{field}:quote_not_in_sentence")
        meta["unresolvable"] = True
    return empty_time(normalized) if field == "event_time" else empty_span(), errors, meta


def parse_sentence_response(
    raw_content: str,
    item: dict[str, Any],
    sentences: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sentence_by_id = {row["sentence_id"]: row for row in sentences}
    errors: list[str] = []
    field_meta: list[dict[str, Any]] = []
    try:
        payload = strict_json_loads(raw_content)
    except Exception as exc:
        return failed_record(item, f"json_parse_failed:{exc}"), {
            "success": False,
            "draft_status": "D",
            "errors": [f"json_parse_failed:{exc}"],
            "json_success": False,
            "sentence_id_illegal_count": 0,
            "quote_nonempty_count": 0,
            "quote_located_count": 0,
            "ambiguous_field_count": 0,
            "unresolvable_field_count": 0,
        }
    extra = sorted(set(payload) - OUTPUT_FIELDS)
    missing = sorted(OUTPUT_FIELDS - set(payload))
    if extra:
        errors.append("extra_fields:" + "|".join(extra))
    if missing:
        errors.append("missing_fields:" + "|".join(missing))
    event_sentence_ids_raw = payload.get("event_sentence_ids", [])
    if not isinstance(event_sentence_ids_raw, list):
        event_sentence_ids = []
        errors.append("event_sentence_ids:not_list")
    else:
        event_sentence_ids = [str(sid) for sid in event_sentence_ids_raw]
    valid_event_sentence_ids = []
    for sid in event_sentence_ids:
        if sid not in sentence_by_id:
            errors.append(f"event_sentence_ids:illegal_sentence_id:{sid}")
        elif sid not in valid_event_sentence_ids:
            valid_event_sentence_ids.append(sid)
    direction = str(payload.get("direction", "") or "")
    if direction not in DIRECTION_LABELS:
        errors.append("illegal_direction")
        direction = "unclear"
    try:
        confidence = float(payload.get("confidence", 0.0))
        if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
            errors.append("illegal_confidence")
            confidence = 0.0
    except Exception:
        errors.append("illegal_confidence")
        confidence = 0.0

    spans: dict[str, Any] = {
        "event_name": item["event_name"],
        "direction": direction,
        "confidence": confidence,
        "event_sentences": [sentence_by_id[sid] for sid in valid_event_sentence_ids],
    }
    if valid_event_sentence_ids:
        selected = [sentence_by_id[sid] for sid in valid_event_sentence_ids]
        start = min(int(row["article_start"]) for row in selected)
        end = max(int(row["article_end"]) for row in selected)
        spans["event_span"] = {"text": str(item["content"])[start:end], "start": start, "end": end}
    else:
        spans["event_span"] = empty_span()
        errors.append("event_sentence_ids:empty_or_invalid")

    for field in ["trigger", "subject_company", "event_time", "object_target"]:
        span, field_errors, meta = quote_to_span(field, payload.get(field), sentence_by_id, allow_empty=True)
        spans[field] = span
        errors.extend(field_errors)
        field_meta.append(meta)
    trigger = spans["trigger"]
    event_span = spans["event_span"]
    trigger_in_event_sentence = False
    if trigger.get("text") and event_span.get("text"):
        trigger_in_event_sentence = int(event_span["start"]) <= int(trigger["start"]) < int(trigger["end"]) <= int(event_span["end"])
        if not trigger_in_event_sentence:
            errors.append("trigger_not_in_event_sentence")
    else:
        errors.append("trigger_or_event_span_missing")

    arguments = []
    raw_args = payload.get("arguments", [])
    if not isinstance(raw_args, list):
        errors.append("arguments:not_list")
    else:
        for idx, arg in enumerate(raw_args):
            role = str(arg.get("role", "") or "") if isinstance(arg, dict) else ""
            if role not in ARGUMENT_ROLES:
                errors.append(f"argument_{idx}:illegal_role")
                continue
            span, field_errors, meta = quote_to_span(f"argument_{idx}", arg, sentence_by_id, allow_empty=True)
            errors.extend(field_errors)
            field_meta.append(meta)
            if span.get("text") and span.get("start", -1) >= 0:
                arguments.append({"text": span["text"], "role": role, "start": span["start"], "end": span["end"]})
    spans["arguments"] = arguments

    quote_nonempty_count = sum(1 for row in field_meta if row["quote_nonempty"])
    quote_located_count = sum(1 for row in field_meta if row["quote_located"])
    sentence_id_illegal_count = sum(1 for row in field_meta if not row["sentence_id_valid"]) + sum(1 for sid in event_sentence_ids if sid not in sentence_by_id)
    ambiguous_count = sum(1 for row in field_meta if row["ambiguous"])
    unresolvable_count = sum(1 for row in field_meta if row["unresolvable"])
    key_ok = bool(spans["trigger"].get("text") and spans["event_span"].get("text") and spans["subject_company"].get("text"))
    fatal_format = any(err.startswith(("json_parse_failed", "missing_fields", "extra_fields", "illegal_direction", "illegal_confidence", "event_sentence_ids:not_list")) for err in errors)
    if fatal_format:
        grade = "D"
    elif key_ok and not errors:
        grade = "A"
    elif key_ok and ambiguous_count == 0 and unresolvable_count == 0:
        grade = "B"
    elif key_ok:
        grade = "B"
    else:
        grade = "C"
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
        "trigger": spans["trigger"],
        "event_span": spans["event_span"],
        "event_sentences": spans["event_sentences"],
        "subject_company": spans["subject_company"],
        "event_time": spans["event_time"],
        "object_target": spans["object_target"],
        "direction": direction,
        "arguments": arguments,
        "annotation_source": "local_32b_sentence_constrained_draft",
        "reviewer_confirmed": "",
        "annotation_status": "ai_draft_needs_human_review",
        "needs_human_review": True,
        "draft_status": grade,
        "sentence_contract_errors": sorted(set(errors)),
        "sentence_field_meta": field_meta,
        "raw_sentence_output": payload,
    }
    for field in [
        "human_trigger",
        "human_event_span",
        "human_subject_company",
        "human_event_time",
        "human_event_time_normalized",
        "human_object_target",
        "human_direction",
        "human_arguments_json",
        "human_argument_roles_json",
        "reviewer_note",
    ]:
        record[field] = ""
    meta = {
        "success": grade in {"A", "B", "C"} and not fatal_format,
        "draft_status": grade,
        "errors": sorted(set(errors)),
        "json_success": True,
        "sentence_id_illegal_count": sentence_id_illegal_count,
        "quote_nonempty_count": quote_nonempty_count,
        "quote_located_count": quote_located_count,
        "quote_located_rate": quote_located_count / quote_nonempty_count if quote_nonempty_count else 1.0,
        "ambiguous_field_count": ambiguous_count,
        "unresolvable_field_count": unresolvable_count,
        "trigger_in_event_sentence": trigger_in_event_sentence,
        "final_format_failed": grade == "D",
    }
    return record, meta


def failed_record(item: dict[str, Any], error: str) -> dict[str, Any]:
    return {
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
        "trigger": empty_span(),
        "event_span": empty_span(),
        "event_sentences": [],
        "subject_company": empty_span(),
        "event_time": empty_time(),
        "object_target": empty_span(),
        "direction": "unclear",
        "arguments": [],
        "annotation_source": "local_32b_sentence_constrained_draft",
        "reviewer_confirmed": "",
        "annotation_status": "ai_draft_needs_human_review",
        "needs_human_review": True,
        "draft_status": "D",
        "sentence_contract_errors": [error],
        "sentence_field_meta": [],
        "raw_sentence_output": None,
        "human_trigger": "",
        "human_event_span": "",
        "human_subject_company": "",
        "human_event_time": "",
        "human_event_time_normalized": "",
        "human_object_target": "",
        "human_direction": "",
        "human_arguments_json": "",
        "human_argument_roles_json": "",
        "reviewer_note": "",
    }


def contract_summary(rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    grade_counts = dict(Counter(str(row.get("draft_status", "")) for row in rows))
    quote_nonempty = sum(int(row.get("quote_nonempty_count", 0) or 0) for row in metrics)
    quote_located = sum(int(row.get("quote_located_count", 0) or 0) for row in metrics)
    latencies = [float(row.get("elapsed_seconds", 0.0) or 0.0) for row in metrics]
    tokens = [float(row.get("total_tokens", 0) or 0) for row in metrics]
    return {
        "record_count": count,
        "grade_counts": grade_counts,
        "json_success_count": sum(int(row.get("json_success", 0) or 0) for row in metrics),
        "json_success_rate": sum(int(row.get("json_success", 0) or 0) for row in metrics) / count if count else 0.0,
        "sentence_id_illegal_count": sum(int(row.get("sentence_id_illegal_count", 0) or 0) for row in metrics),
        "quote_nonempty_count": quote_nonempty,
        "quote_located_count": quote_located,
        "quote_located_rate": quote_located / quote_nonempty if quote_nonempty else 1.0,
        "trigger_in_event_sentence_count": sum(int(row.get("trigger_in_event_sentence", 0) or 0) for row in metrics),
        "ambiguous_field_count": sum(int(row.get("ambiguous_field_count", 0) or 0) for row in metrics),
        "unresolvable_field_count": sum(int(row.get("unresolvable_field_count", 0) or 0) for row in metrics),
        "final_format_failed_count": sum(int(row.get("final_format_failed", 0) or 0) for row in metrics),
        "request_count": sum(int(row.get("request_count", 0) or 0) for row in metrics),
        "retry_count": sum(int(row.get("retry_count", 0) or 0) for row in metrics),
        "timeout_count": sum(int(row.get("timeout", 0) or 0) for row in metrics),
        "http_5xx_count": sum(int(row.get("http_5xx", 0) or 0) for row in metrics),
        "cuda_oom_count": sum(int(row.get("cuda_oom", 0) or 0) for row in metrics),
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_seconds": percentile(latencies, 95),
        "avg_total_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
        "p95_total_tokens": percentile(tokens, 95),
    }


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[idx])
