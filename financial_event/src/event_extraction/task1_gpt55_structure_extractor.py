#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-M GPT-5.5 quote-based structure extraction helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIR_LABELS = ["positive", "negative", "neutral"]
ROLE_LABELS = ["subject_company", "event_time", "object_target", "trigger"]
SPAN_FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
FORBIDDEN_INPUT_PREFIXES = (
    "ai_",
    "human_",
    "reviewer_",
)
FORBIDDEN_INPUT_FIELDS = {
    "reference_raw_text",
    "draft_quality_grade",
    "quality_grade",
    "validation_errors",
    "annotation_notice",
    "annotation_source",
    "confirmed_at",
    "trigger",
    "event_span",
    "subject_company",
    "event_time",
    "object_target",
    "direction",
    "arguments",
    "argument_roles",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_responses_url(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if not base:
        raise ValueError("base_url is empty")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return base + "/responses"
    if base == "https://mdkj.lol":
        return base + "/responses"
    return base + "/responses"


def split_sentences(content: str) -> list[dict[str, Any]]:
    text = str(content or "")
    sentences: list[dict[str, Any]] = []
    start = 0
    for idx, ch in enumerate(text):
        if ch in "。！？!?；;\n":
            end = idx + 1
            add_sentence(sentences, text, start, end)
            start = end
    if start < len(text):
        add_sentence(sentences, text, start, len(text))
    if not sentences and text:
        sentences.append({"sentence_id": "S0001", "text": text, "start": 0, "end": len(text)})
    return sentences


def add_sentence(sentences: list[dict[str, Any]], content: str, start: int, end: int) -> None:
    segment = content[start:end]
    if not segment.strip():
        return
    lead = len(segment) - len(segment.lstrip())
    trail = len(segment) - len(segment.rstrip())
    real_start = start + lead
    real_end = end - trail
    if real_start >= real_end:
        return
    sentences.append(
        {
            "sentence_id": f"S{len(sentences) + 1:04d}",
            "text": content[real_start:real_end],
            "start": real_start,
            "end": real_end,
        }
    )


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", "", text.lower())
    text = re.sub(r"[，。；：、,.!！?？（）()《》“”\"'`·\-—_<>【】\[\]{}]", "", text)
    for suffix in ["股份有限公司", "有限责任公司", "有限公司", "集团股份", "集团", "公司"]:
        text = text.replace(suffix, "")
    return text


def normalize_time(text: str, publish_time: str = "") -> str:
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
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    m = re.search(r"(20\d{2})年", value)
    if m:
        return m.group(1)
    if any(word in value for word in ["未明确", "近日", "日前", "当日", "当天", "今日", "近期", "今年", "未来"]):
        return value
    return value


def empty_span(field: str = "") -> dict[str, Any]:
    if field == "event_time":
        return {"text": "", "start": -1, "end": -1, "normalized": "未明确"}
    return {"text": "", "start": -1, "end": -1}


def response_schema() -> dict[str, Any]:
    nullable_text = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    nullable_sentence = {"anyOf": [{"type": "string", "pattern": "^S\\d{4}$"}, {"type": "null"}]}
    quote_obj = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"sentence_id": nullable_sentence, "quote": nullable_text},
        "required": ["sentence_id", "quote"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_name_echo": {"type": "string"},
            "selected_sentence_ids": {"type": "array", "items": {"type": "string", "pattern": "^S\\d{4}$"}, "maxItems": 5},
            "trigger": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"sentence_id": {"type": "string", "pattern": "^S\\d{4}$"}, "quote": {"type": "string"}},
                "required": ["sentence_id", "quote"],
            },
            "event_span": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sentence_ids": {"type": "array", "items": {"type": "string", "pattern": "^S\\d{4}$"}, "minItems": 1, "maxItems": 5},
                    "quote": {"type": "string"},
                },
                "required": ["sentence_ids", "quote"],
            },
            "subject_company": quote_obj,
            "event_time": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"sentence_id": nullable_sentence, "quote": nullable_text, "normalized": {"type": "string"}},
                "required": ["sentence_id", "quote", "normalized"],
            },
            "object_target": quote_obj,
            "direction": {"type": "string", "enum": DIR_LABELS},
            "arguments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sentence_id": {"type": "string", "pattern": "^S\\d{4}$"},
                        "quote": {"type": "string"},
                        "role": {"type": "string", "enum": ROLE_LABELS},
                    },
                    "required": ["sentence_id", "quote", "role"],
                },
                "maxItems": 12,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_short": {"type": "string", "maxLength": 80},
        },
        "required": [
            "event_name_echo",
            "selected_sentence_ids",
            "trigger",
            "event_span",
            "subject_company",
            "event_time",
            "object_target",
            "direction",
            "arguments",
            "confidence",
            "reason_short",
        ],
    }


def request_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": "gpt-5.5",
        "store": False,
        "input": prompt,
        "reasoning": {"effort": "xhigh"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "task1_eventmention_structure",
                "strict": True,
                "schema": response_schema(),
            }
        },
        "max_output_tokens": 2400,
    }


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def safe_model_info(response: dict[str, Any]) -> Any:
    return response.get("model")


@dataclass
class ApiResult:
    ok: bool
    http_status: int | None
    latency_seconds: float
    response: dict[str, Any] | None
    response_text: str
    error: str
    usage: dict[str, Any]


def call_responses_api(api_key: str, url: str, payload: dict[str, Any], timeout: int = 120) -> ApiResult:
    started = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key, "User-Agent": "codex-cli"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            data = json.loads(body)
            return ApiResult(
                ok=200 <= resp.status < 300,
                http_status=resp.status,
                latency_seconds=round(time.time() - started, 3),
                response=data,
                response_text=extract_response_text(data),
                error="",
                usage=data.get("usage", {}) if isinstance(data, dict) else {},
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:1000]
        return ApiResult(False, exc.code, round(time.time() - started, 3), None, "", f"http_error:{body}", {})
    except Exception as exc:  # noqa: BLE001
        return ApiResult(False, None, round(time.time() - started, 3), None, "", f"{type(exc).__name__}:{str(exc)[:500]}", {})


def row_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = scan_forbidden_fields(row)
    if forbidden:
        # The caller may pass gold rows; only project allowed fields into prompts.
        pass
    return {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "event_name": row.get("event_name", ""),
        "title": row.get("title", ""),
        "publish_time": row.get("publish_time", ""),
        "sentences": [{"sentence_id": s["sentence_id"], "text": s["text"]} for s in split_sentences(row.get("content", ""))],
    }


def scan_forbidden_fields(row: dict[str, Any]) -> list[str]:
    hits: list[str] = []
    for key in row:
        if key in FORBIDDEN_INPUT_FIELDS or any(key.startswith(prefix) for prefix in FORBIDDEN_INPUT_PREFIXES):
            hits.append(key)
    return sorted(hits)


def gold_as_example(row: dict[str, Any]) -> dict[str, Any]:
    sentences = split_sentences(row.get("content", ""))
    sentence_id = sentence_id_for_span(sentences, row.get("event_span", {}).get("start", -1), row.get("event_span", {}).get("end", -1))
    trigger_sid = sentence_id_for_span(sentences, row.get("trigger", {}).get("start", -1), row.get("trigger", {}).get("end", -1))
    subject_sid = sentence_id_for_span(sentences, row.get("subject_company", {}).get("start", -1), row.get("subject_company", {}).get("end", -1))
    time_sid = sentence_id_for_span(sentences, row.get("event_time", {}).get("start", -1), row.get("event_time", {}).get("end", -1))
    object_sid = sentence_id_for_span(sentences, row.get("object_target", {}).get("start", -1), row.get("object_target", {}).get("end", -1))
    return {
        "event_name": row.get("event_name", ""),
        "title": row.get("title", ""),
        "answer": {
            "event_name_echo": row.get("event_name", ""),
            "selected_sentence_ids": [sentence_id] if sentence_id else [],
            "trigger": {"sentence_id": trigger_sid or sentence_id or "S0001", "quote": row.get("trigger", {}).get("text", "")},
            "event_span": {"sentence_ids": [sentence_id] if sentence_id else [], "quote": row.get("event_span", {}).get("text", "")},
            "subject_company": {"sentence_id": subject_sid, "quote": row.get("subject_company", {}).get("text") or None},
            "event_time": {
                "sentence_id": time_sid,
                "quote": row.get("event_time", {}).get("text") or None,
                "normalized": row.get("event_time", {}).get("normalized", "未明确"),
            },
            "object_target": {"sentence_id": object_sid, "quote": row.get("object_target", {}).get("text") or None},
            "direction": row.get("direction", "neutral") if row.get("direction") in DIR_LABELS else "neutral",
            "arguments": [
                {"sentence_id": sentence_id_for_span(sentences, a.get("start", -1), a.get("end", -1)) or "S0001", "quote": a.get("text", ""), "role": a.get("role", "trigger")}
                for a in row.get("arguments", [])
                if a.get("role") in ROLE_LABELS and a.get("text")
            ][:6],
            "confidence": 0.9,
            "reason_short": "训练示例",
        },
    }


def sentence_id_for_span(sentences: list[dict[str, Any]], start: int, end: int) -> str | None:
    if start is None or end is None or start < 0 or end < start:
        return None
    for sent in sentences:
        if sent["start"] <= start < end <= sent["end"]:
            return sent["sentence_id"]
    for sent in sentences:
        if not (end <= sent["start"] or start >= sent["end"]):
            return sent["sentence_id"]
    return None


def build_prompt(row: dict[str, Any], examples: list[dict[str, Any]], variant: str) -> str:
    payload = row_for_prompt(row)
    examples_payload = [gold_as_example(example) for example in examples]
    self_check = ""
    if variant == "strict_quote_extraction_with_self_check":
        self_check = "\nBefore finalizing, silently verify: every non-null quote is copied exactly from the listed sentence, trigger is inside event_span, event_name_echo equals input event_name, and all roles/enums are legal."
    return (
        "You are doing Task1 financial EventMention structure extraction.\n"
        "Return exactly one JSON object that matches the provided schema. Do not use markdown.\n"
        "Use the configured model gpt-5.5. The input event_name is already correct and frozen; echo it exactly.\n"
        "Do not create offsets. Select sentence_id values and copy exact quotes from the provided sentences.\n"
        "If subject, time, or object is not explicit, use null quote and null sentence_id. Use direction only from positive, negative, neutral.\n"
        "Allowed argument roles: subject_company, event_time, object_target, trigger.\n"
        "The trigger quote must be contained in the event_span quote.\n"
        f"{self_check}\n\n"
        "Few-shot examples from training data:\n"
        f"{json.dumps(examples_payload, ensure_ascii=False)}\n\n"
        "Now extract this EventMention:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def parse_model_json(text: str) -> tuple[dict[str, Any] | None, str]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except Exception as exc:  # noqa: BLE001
        return None, f"json_parse_error:{exc}"
    if not isinstance(value, dict):
        return None, "json_not_object"
    return value, ""


def resolve_quote(
    content: str,
    sentences: list[dict[str, Any]],
    quote: Any,
    sentence_id: str | None,
    allowed_sentence_ids: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    if quote is None or quote == "":
        return {"text": "", "start": -1, "end": -1}, ""
    text = str(quote)
    search_ranges: list[tuple[int, int]] = []
    by_id = {s["sentence_id"]: s for s in sentences}
    if sentence_id and sentence_id in by_id:
        sent = by_id[sentence_id]
        search_ranges.append((sent["start"], sent["end"]))
    elif allowed_sentence_ids:
        selected = [by_id[sid] for sid in allowed_sentence_ids if sid in by_id]
        if selected:
            search_ranges.append((min(s["start"] for s in selected), max(s["end"] for s in selected)))
    if not search_ranges:
        search_ranges.append((0, len(content)))
    matches: list[tuple[int, int]] = []
    for start, end in search_ranges:
        segment = content[start:end]
        pos = segment.find(text)
        while pos != -1:
            matches.append((start + pos, start + pos + len(text)))
            pos = segment.find(text, pos + 1)
    if len(matches) == 1:
        start, end = matches[0]
        if content[start:end] == text:
            return {"text": text, "start": start, "end": end}, ""
    if len(matches) > 1:
        return {"text": text, "start": -1, "end": -1}, "ambiguous_quote"
    return {"text": text, "start": -1, "end": -1}, "quote_not_found"


def validate_model_object(obj: dict[str, Any], row: dict[str, Any], sentences: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    sentence_ids = {s["sentence_id"] for s in sentences}
    if obj.get("event_name_echo") != row.get("event_name"):
        errors.append("event_name_echo_mismatch")
    if obj.get("direction") not in DIR_LABELS:
        errors.append("illegal_direction")
    confidence = obj.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not (0 <= float(confidence) <= 1):
        errors.append("invalid_confidence")
    for sid in obj.get("selected_sentence_ids", []) or []:
        if sid not in sentence_ids:
            errors.append("invalid_selected_sentence_id")
    for field in ["trigger", "subject_company", "event_time", "object_target"]:
        sid = (obj.get(field) or {}).get("sentence_id")
        if sid is not None and sid not in sentence_ids:
            errors.append(f"invalid_{field}_sentence_id")
    for sid in (obj.get("event_span") or {}).get("sentence_ids", []) or []:
        if sid not in sentence_ids:
            errors.append("invalid_event_span_sentence_id")
    for arg in obj.get("arguments", []) or []:
        if arg.get("sentence_id") not in sentence_ids:
            errors.append("invalid_argument_sentence_id")
        if arg.get("role") not in ROLE_LABELS:
            errors.append("illegal_argument_role")
    return errors


def resolve_prediction(row: dict[str, Any], obj: dict[str, Any] | None, raw_text: str, variant: str, api_meta: dict[str, Any]) -> dict[str, Any]:
    content = row.get("content", "")
    sentences = split_sentences(content)
    base: dict[str, Any] = {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "event_name": row.get("event_name", ""),
        "scheme": variant,
        "trigger": empty_span("trigger"),
        "event_span": empty_span("event_span"),
        "subject_company": empty_span("subject_company"),
        "event_time": empty_span("event_time"),
        "object_target": empty_span("object_target"),
        "direction": "neutral",
        "arguments": [],
        "argument_roles": [],
        "status": "failed",
        "parse_errors": [],
        "api_meta": api_meta,
        "raw_response_text": raw_text,
    }
    if obj is None:
        base["parse_errors"].append("missing_model_json")
        return base
    errors = validate_model_object(obj, row, sentences)
    base["direction"] = obj.get("direction") if obj.get("direction") in DIR_LABELS else "neutral"
    span_ids = (obj.get("event_span") or {}).get("sentence_ids", []) or []
    event_span, err = resolve_quote(content, sentences, (obj.get("event_span") or {}).get("quote"), None, span_ids)
    if err:
        errors.append(f"event_span_{err}")
    base["event_span"] = event_span
    trigger, err = resolve_quote(content, sentences, (obj.get("trigger") or {}).get("quote"), (obj.get("trigger") or {}).get("sentence_id"))
    if err:
        errors.append(f"trigger_{err}")
    base["trigger"] = trigger
    subject, err = resolve_quote(content, sentences, (obj.get("subject_company") or {}).get("quote"), (obj.get("subject_company") or {}).get("sentence_id"))
    if err:
        errors.append(f"subject_company_{err}")
    base["subject_company"] = subject
    event_time, err = resolve_quote(content, sentences, (obj.get("event_time") or {}).get("quote"), (obj.get("event_time") or {}).get("sentence_id"))
    if err:
        errors.append(f"event_time_{err}")
    event_time["normalized"] = (obj.get("event_time") or {}).get("normalized") or normalize_time(event_time.get("text", ""), row.get("publish_time", ""))
    if not event_time.get("text"):
        event_time["normalized"] = "未明确"
    base["event_time"] = event_time
    obj_span, err = resolve_quote(content, sentences, (obj.get("object_target") or {}).get("quote"), (obj.get("object_target") or {}).get("sentence_id"))
    if err:
        errors.append(f"object_target_{err}")
    base["object_target"] = obj_span
    args: list[dict[str, Any]] = []
    for arg in obj.get("arguments", []) or []:
        if arg.get("role") not in ROLE_LABELS:
            errors.append("illegal_argument_role")
            continue
        span, arg_err = resolve_quote(content, sentences, arg.get("quote"), arg.get("sentence_id"))
        if arg_err:
            errors.append(f"argument_{arg_err}")
            continue
        if span.get("text"):
            span["role"] = arg.get("role")
            args.append(span)
    base["arguments"] = args
    base["argument_roles"] = list(args)
    if trigger.get("text") and event_span.get("text") and not (event_span["start"] <= trigger["start"] < trigger["end"] <= event_span["end"]):
        errors.append("trigger_not_in_event_span")
    for field in SPAN_FIELDS:
        span = base[field]
        if span.get("text") and content[span.get("start", -1) : span.get("end", -1)] != span.get("text"):
            errors.append(f"{field}_invalid_offset")
    for arg in args:
        if content[arg.get("start", -1) : arg.get("end", -1)] != arg.get("text"):
            errors.append("argument_invalid_offset")
    base["parse_errors"] = sorted(set(errors))
    base["status"] = "success" if not errors and base["trigger"].get("text") and base["event_span"].get("text") else "failed"
    return base


def select_few_shot_examples(train_rows: list[dict[str, Any]], count: int = 3) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for row in train_rows:
        if row.get("event_name") in seen_events:
            continue
        if row.get("trigger", {}).get("text") and row.get("event_span", {}).get("text"):
            selected.append(row)
            seen_events.add(row.get("event_name", ""))
        if len(selected) >= count:
            return selected
    return selected or train_rows[:count]


def project_gold_free_input(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "event_name": row.get("event_name", ""),
        "title": row.get("title", ""),
        "content": row.get("content", ""),
        "publish_time": row.get("publish_time", ""),
    }

