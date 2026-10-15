#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic offset adapter for Task1 structure extraction predictions.

The adapter only consumes model output text and the original article content.
It never reads gold structure fields and never invents text that the model did
not predict.  A repaired span is accepted only when content[start:end] exactly
equals the model text.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

DIR_LABELS = {"positive", "negative", "neutral", "unclear"}
ARG_ROLES = {"subject_company", "event_time", "object_target", "trigger"}
SPAN_FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
PUNCT_TO_DROP = set(" \t\r\n，。；：、,.!！?？（）()《》“”\"'`·-—_[]【】{}<>")


@dataclass(frozen=True)
class CandidateSpan:
    start: int
    end: int
    source: str

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


def strict_json_loads(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError("not a single JSON object")
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("JSON top-level is not object")
    return payload


def coerce_span(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"text": "", "start": -1, "end": -1, "protocol_error": "not_object"}
    try:
        start = int(value.get("start", -1))
        end = int(value.get("end", -1))
    except Exception:
        start, end = -1, -1
    out = {"text": str(value.get("text", "") or ""), "start": start, "end": end}
    if "normalized" in value:
        out["normalized"] = str(value.get("normalized", "") or "")
    return out


def exact_occurrences(content: str, text: str) -> list[CandidateSpan]:
    if not text:
        return []
    return [CandidateSpan(match.start(), match.end(), "exact") for match in re.finditer(re.escape(text), content)]


def canonicalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for idx, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char)
        for item in normalized:
            if item in PUNCT_TO_DROP or item.isspace():
                continue
            chars.append(item.lower())
            positions.append(idx)
    return "".join(chars), positions


def normalized_occurrences(content: str, text: str) -> list[CandidateSpan]:
    needle, _ = canonicalize_with_map(text)
    haystack, positions = canonicalize_with_map(content)
    if not needle:
        return []
    out: list[CandidateSpan] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        raw_start = positions[idx]
        raw_end = positions[idx + len(needle) - 1] + 1
        out.append(CandidateSpan(raw_start, raw_end, "normalized"))
        start = idx + 1
    return out


def span_strict_match(content: str, span: dict[str, Any]) -> bool:
    text = str(span.get("text", "") or "")
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    return bool(text) and 0 <= start <= end <= len(content) and content[start:end] == text


def field_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, dict):
        return ""
    return str(value.get("text", "") or "")


def field_raw_offset(payload: dict[str, Any], field: str) -> tuple[int, int]:
    value = payload.get(field)
    if not isinstance(value, dict):
        return -1, -1
    try:
        return int(value.get("start", -1)), int(value.get("end", -1))
    except Exception:
        return -1, -1


def choose_unique(candidates: list[CandidateSpan], anchors: list[CandidateSpan] | None = None) -> tuple[CandidateSpan | None, str]:
    if not candidates:
        return None, "no_candidate"
    if len(candidates) == 1:
        return candidates[0], "unique"
    anchors = anchors or []
    if anchors:
        inside = [
            candidate
            for candidate in candidates
            if any(anchor.start <= candidate.start and candidate.end <= anchor.end for anchor in anchors)
        ]
        if len(inside) == 1:
            return inside[0], "multi_context_inside_anchor"
        if len(inside) > 1:
            candidates = inside
        distances = []
        for candidate in candidates:
            distance = min(abs(candidate.center - anchor.center) for anchor in anchors)
            distances.append((distance, candidate))
        distances.sort(key=lambda item: (item[0], item[1].start, item[1].end))
        if len(distances) >= 2 and abs(distances[0][0] - distances[1][0]) < 1e-9:
            return None, "ambiguous_multi_hit"
        return distances[0][1], "multi_context_nearest_anchor"
    return None, "ambiguous_multi_hit"


def classify_text_locator(content: str, text: str, start: int, end: int, allow_empty: bool = False) -> str:
    if not text:
        return "empty_ok" if allow_empty and start == -1 and end == -1 else "empty_field_protocol_error"
    if 0 <= start <= end <= len(content) and content[start:end] == text:
        return "original_offset_strict_match"
    exact = exact_occurrences(content, text)
    if exact:
        return "text_exists_offset_wrong_unique" if len(exact) == 1 else "text_exists_offset_wrong_multi"
    normalized = normalized_occurrences(content, text)
    if normalized:
        # If no exact raw match exists, final strict output cannot be guaranteed.
        return "normalized_unique_non_strict" if len(normalized) == 1 else "normalized_multi_non_strict"
    return "text_unresolvable"


def resolve_field(
    content: str,
    text: str,
    raw_start: int,
    raw_end: int,
    field: str,
    anchors: list[CandidateSpan] | None = None,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not text:
        if allow_empty:
            resolved = {"text": "", "start": -1, "end": -1}
            if field == "event_time":
                resolved["normalized"] = "未明确"
            return resolved, {"field": field, "status": "empty_ok", "resolved": True, "ambiguous": False, "unresolvable": False}
        return {"text": "", "start": -1, "end": -1}, {"field": field, "status": "empty_field_protocol_error", "resolved": False, "ambiguous": False, "unresolvable": True}
    raw_span = {"text": text, "start": raw_start, "end": raw_end}
    if span_strict_match(content, raw_span):
        return raw_span, {"field": field, "status": "original_offset_strict_match", "resolved": True, "ambiguous": False, "unresolvable": False}
    exact = exact_occurrences(content, text)
    if exact:
        chosen, reason = choose_unique(exact, anchors=anchors)
        if chosen is None:
            return {"text": text, "start": -1, "end": -1}, {"field": field, "status": reason, "resolved": False, "ambiguous": True, "unresolvable": False, "candidate_count": len(exact)}
        return {"text": text, "start": chosen.start, "end": chosen.end}, {"field": field, "status": "text_exists_offset_wrong_" + reason, "resolved": True, "ambiguous": False, "unresolvable": False, "candidate_count": len(exact)}
    normalized = normalized_occurrences(content, text)
    if normalized:
        chosen, reason = choose_unique(normalized, anchors=anchors)
        if chosen is not None and content[chosen.start : chosen.end] == text:
            return {"text": text, "start": chosen.start, "end": chosen.end}, {"field": field, "status": "normalized_strict_" + reason, "resolved": True, "ambiguous": False, "unresolvable": False, "candidate_count": len(normalized)}
        status = "normalized_unique_non_strict" if len(normalized) == 1 else "normalized_multi_non_strict"
        return {"text": text, "start": -1, "end": -1}, {"field": field, "status": status, "resolved": False, "ambiguous": len(normalized) > 1, "unresolvable": len(normalized) == 1, "candidate_count": len(normalized)}
    return {"text": text, "start": -1, "end": -1}, {"field": field, "status": "text_unresolvable", "resolved": False, "ambiguous": False, "unresolvable": True}


def resolve_trigger_and_span(content: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    trigger_text = field_text(payload, "trigger")
    span_text = field_text(payload, "event_span")
    trigger_start, trigger_end = field_raw_offset(payload, "trigger")
    span_start, span_end = field_raw_offset(payload, "event_span")
    field_rows: list[dict[str, Any]] = []
    if trigger_text and span_text and trigger_text not in span_text:
        trigger, trig_meta = resolve_field(content, trigger_text, trigger_start, trigger_end, "trigger")
        span, span_meta = resolve_field(content, span_text, span_start, span_end, "event_span")
        trig_meta["relation_error"] = "trigger_text_not_in_event_span_text"
        span_meta["relation_error"] = "trigger_text_not_in_event_span_text"
        field_rows.extend([trig_meta, span_meta])
        return trigger, span, field_rows

    trigger_candidates = exact_occurrences(content, trigger_text)
    span_candidates = exact_occurrences(content, span_text)
    if trigger_text and span_text and trigger_candidates and span_candidates:
        pairs = [
            (span, trigger)
            for span in span_candidates
            for trigger in trigger_candidates
            if span.start <= trigger.start and trigger.end <= span.end
        ]
        if len(pairs) == 1:
            span, trigger = pairs[0]
            trigger_out = {"text": trigger_text, "start": trigger.start, "end": trigger.end}
            span_out = {"text": span_text, "start": span.start, "end": span.end}
            field_rows.append({"field": "trigger", "status": "paired_trigger_span_unique", "resolved": True, "ambiguous": False, "unresolvable": False, "candidate_count": len(trigger_candidates)})
            field_rows.append({"field": "event_span", "status": "paired_trigger_span_unique", "resolved": True, "ambiguous": False, "unresolvable": False, "candidate_count": len(span_candidates)})
            return trigger_out, span_out, field_rows

    span_out, span_meta = resolve_field(content, span_text, span_start, span_end, "event_span", allow_empty=False)
    anchors = []
    if span_meta.get("resolved"):
        anchors.append(CandidateSpan(span_out["start"], span_out["end"], "event_span"))
    trigger_out, trig_meta = resolve_field(content, trigger_text, trigger_start, trigger_end, "trigger", anchors=anchors, allow_empty=False)
    if span_meta.get("resolved") and trig_meta.get("resolved") and not (span_out["start"] <= trigger_out["start"] < trigger_out["end"] <= span_out["end"]):
        trig_meta["relation_error"] = "trigger_not_inside_event_span"
        span_meta["relation_error"] = "trigger_not_inside_event_span"
    field_rows.extend([trig_meta, span_meta])
    return trigger_out, span_out, field_rows


def adapt_prediction(content: str, payload: dict[str, Any], expected_event_name: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    field_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    event_name = str(payload.get("event_name", "") or "")
    if expected_event_name is not None and event_name != expected_event_name:
        errors.append("event_name_mismatch")
    direction = str(payload.get("direction", "") or "")
    if direction not in DIR_LABELS:
        errors.append("illegal_direction")
        direction = "unclear"
    try:
        confidence = float(payload.get("confidence", 0.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            errors.append("illegal_confidence")
            confidence = 0.0
    except Exception:
        errors.append("illegal_confidence")
        confidence = 0.0

    trigger, event_span, trigger_span_rows = resolve_trigger_and_span(content, payload)
    field_rows.extend(trigger_span_rows)
    anchors = []
    if event_span["start"] >= 0:
        anchors.append(CandidateSpan(event_span["start"], event_span["end"], "event_span"))
    if trigger["start"] >= 0:
        anchors.append(CandidateSpan(trigger["start"], trigger["end"], "trigger"))

    resolved: dict[str, Any] = {
        "event_name": event_name,
        "trigger": trigger,
        "event_span": event_span,
        "direction": direction,
        "confidence": confidence,
    }
    for field in ["subject_company", "event_time", "object_target"]:
        text = field_text(payload, field)
        start, end = field_raw_offset(payload, field)
        allow_empty = field in {"event_time", "object_target"}
        span, meta = resolve_field(content, text, start, end, field, anchors=anchors, allow_empty=allow_empty)
        if field == "event_time":
            source = payload.get("event_time") if isinstance(payload.get("event_time"), dict) else {}
            span["normalized"] = span.get("normalized") or str(source.get("normalized", "") or ("未明确" if not span["text"] else ""))
        resolved[field] = span
        field_rows.append(meta)

    # Rebuild arguments from model-predicted top-level fields that were safely located.
    arguments = []
    top_roles = {
        "subject_company": resolved["subject_company"],
        "event_time": resolved["event_time"],
        "object_target": resolved["object_target"],
        "trigger": resolved["trigger"],
    }
    for role, span in top_roles.items():
        if span.get("text") and span.get("start", -1) >= 0 and content[span["start"] : span["end"]] == span["text"]:
            arguments.append({"text": span["text"], "role": role, "start": span["start"], "end": span["end"]})
    resolved["arguments"] = arguments

    original_args = payload.get("arguments") if isinstance(payload.get("arguments"), list) else []
    original_arg_set = {(str(item.get("text", "") or ""), str(item.get("role", "") or "")) for item in original_args if isinstance(item, dict)}
    rebuilt_arg_set = {(item["text"], item["role"]) for item in arguments}
    argument_mismatch = bool(original_arg_set and rebuilt_arg_set and original_arg_set != rebuilt_arg_set)

    for meta in field_rows:
        field = meta["field"]
        if not meta.get("resolved"):
            errors.append(f"{field}:{meta.get('status')}")
        if meta.get("relation_error"):
            errors.append(f"{field}:{meta['relation_error']}")
    if trigger["start"] >= 0 and event_span["start"] >= 0 and not (event_span["start"] <= trigger["start"] < trigger["end"] <= event_span["end"]):
        errors.append("trigger_not_inside_event_span")

    for field in ["trigger", "event_span", "subject_company"]:
        span = resolved[field]
        if not span.get("text") or span.get("start", -1) < 0 or content[span["start"] : span["end"]] != span["text"]:
            errors.append(f"{field}:missing_or_non_strict")
    for field in ["event_time", "object_target"]:
        span = resolved[field]
        if span.get("text") and (span.get("start", -1) < 0 or content[span["start"] : span["end"]] != span["text"]):
            errors.append(f"{field}:non_strict")
    strict_offsets_ok = all(
        (not resolved[field].get("text") and resolved[field].get("start") == -1 and resolved[field].get("end") == -1)
        or (resolved[field].get("start", -1) >= 0 and content[resolved[field]["start"] : resolved[field]["end"]] == resolved[field]["text"])
        for field in SPAN_FIELDS
    )
    meta = {
        "success": not errors and strict_offsets_ok,
        "strict_offsets_ok": strict_offsets_ok,
        "errors": sorted(set(errors)),
        "ambiguous_field_count": sum(1 for row in field_rows if row.get("ambiguous")),
        "unresolvable_field_count": sum(1 for row in field_rows if row.get("unresolvable")),
        "argument_top_level_mismatch": argument_mismatch,
        "field_rows": field_rows,
    }
    return resolved, field_rows, meta


def adapt_raw_response(content: str, raw_content: str, expected_event_name: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = strict_json_loads(raw_content)
    except Exception as exc:
        failed = {"event_name": expected_event_name or "", "prediction": None}
        return failed, [], {"success": False, "strict_json_parse_failed": True, "errors": [f"strict_json_parse_failed:{exc}"], "ambiguous_field_count": 0, "unresolvable_field_count": 0}
    prediction, field_rows, meta = adapt_prediction(content, payload, expected_event_name=expected_event_name)
    meta["strict_json_parse_failed"] = False
    return prediction, field_rows, meta


def adapt_text_only_response(content: str, raw_content: str, expected_event_name: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Adapt a text-only structure draft.

    Expected JSON keys:
    - event_name
    - trigger_text
    - event_span_text
    - subject_company_text
    - event_time_text
    - event_time_normalized
    - object_target_text
    - direction
    - arguments
    - confidence
    """
    try:
        payload = strict_json_loads(raw_content)
    except Exception as exc:
        failed = {"event_name": expected_event_name or "", "prediction": None}
        return failed, [], {"success": False, "strict_json_parse_failed": True, "errors": [f"strict_json_parse_failed:{exc}"], "ambiguous_field_count": 0, "unresolvable_field_count": 0}
    if not isinstance(payload, dict):
        return {"event_name": expected_event_name or "", "prediction": None}, [], {"success": False, "strict_json_parse_failed": True, "errors": ["top_level_not_object"], "ambiguous_field_count": 0, "unresolvable_field_count": 0}
    mapped = {
        "event_name": payload.get("event_name", ""),
        "trigger": {"text": payload.get("trigger_text", ""), "start": -1, "end": -1},
        "event_span": {"text": payload.get("event_span_text", ""), "start": -1, "end": -1},
        "subject_company": {"text": payload.get("subject_company_text", ""), "start": -1, "end": -1},
        "event_time": {
            "text": payload.get("event_time_text", ""),
            "normalized": payload.get("event_time_normalized", ""),
            "start": -1,
            "end": -1,
        },
        "object_target": {"text": payload.get("object_target_text", ""), "start": -1, "end": -1},
        "direction": payload.get("direction", ""),
        "arguments": payload.get("arguments", []),
        "confidence": payload.get("confidence", 0.0),
    }
    prediction, field_rows, meta = adapt_prediction(content, mapped, expected_event_name=expected_event_name)
    meta["strict_json_parse_failed"] = False
    meta["text_only_payload"] = True
    return prediction, field_rows, meta


def status_counts(field_rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("status", "")) for row in field_rows))
