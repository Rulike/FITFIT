#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic role assembly for Task1 discriminative structure extraction."""

from __future__ import annotations

from typing import Any


ROLE_FIELDS = ["subject_company", "event_time", "object_target", "trigger"]


def assemble_arguments(pred: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args: list[dict[str, Any]] = []
    mapped = 0
    for role in ROLE_FIELDS:
        span = pred.get(role, {}) or {}
        if span.get("text"):
            args.append({"text": span["text"], "start": int(span["start"]), "end": int(span["end"]), "role": role})
            mapped += 1
    pred["arguments"] = args
    pred["argument_roles"] = list(args)
    return args, {"deterministic_mapping_count": mapped, "unmapped_extra_argument_count": 0}


def empty_span(field: str = "") -> dict[str, Any]:
    if field == "event_time":
        return {"text": "", "start": -1, "end": -1, "normalized": "未明确"}
    return {"text": "", "start": -1, "end": -1}


def enforce_offsets(content: str, pred: dict[str, Any], span_fields: list[str]) -> tuple[dict[str, Any], int]:
    illegal = 0
    for field in span_fields:
        span = pred.get(field, {}) or {}
        text = str(span.get("text", "") or "")
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
        if not text:
            pred[field] = empty_span(field)
            continue
        if not (0 <= start <= end <= len(content) and content[start:end] == text):
            illegal += 1
            pred[field] = empty_span(field)
    return pred, illegal

