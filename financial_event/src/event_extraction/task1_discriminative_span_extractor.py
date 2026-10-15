#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discriminative span extraction helpers for Task1 Step 5.7-X."""

from __future__ import annotations

import re
from typing import Any


SPAN_FIELDS = ["trigger", "subject_company", "event_time", "object_target"]
FIELD_PROMPTS = {
    "trigger": "触发词",
    "subject_company": "主体公司或机构",
    "event_time": "事件时间",
    "object_target": "事件对象",
}
OPTIONAL_FIELDS = {"subject_company", "event_time", "object_target"}


def empty_span(field: str = "") -> dict[str, Any]:
    if field == "event_time":
        return {"text": "", "start": -1, "end": -1, "normalized": "未明确"}
    return {"text": "", "start": -1, "end": -1}


def normalize_time(text: str, publish_time: str = "") -> str:
    value = str(text or "").strip()
    if not value:
        return "未明确"
    m = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?", value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(20\d{2})年(\d{1,2})月", value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    m = re.search(r"(20\d{2})年", value)
    if m:
        return m.group(1)
    if value in {"今日", "当日", "当天"}:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(publish_time or ""))
        return m.group(0) if m else "未明确"
    if value in {"昨日"}:
        return value
    if value in {"近日", "近期", "日前", "今年", "未来", "上半年", "下半年"}:
        return value
    return value


def as_span(content: str, start: int, end: int, field: str = "", publish_time: str = "") -> dict[str, Any]:
    if not (0 <= start < end <= len(content)):
        return empty_span(field)
    out: dict[str, Any] = {"text": content[start:end], "start": int(start), "end": int(end)}
    if field == "event_time":
        out["normalized"] = normalize_time(out["text"], publish_time)
    return out


def token_span_from_offsets(offsets: list[tuple[int, int]], content_mask: list[int], span: dict[str, Any], window: dict[str, Any]) -> tuple[int, int, int]:
    text = str(span.get("text", "") or "")
    if not text:
        return 0, 0, 0
    start = int(span.get("start", -1)) - int(window.get("start", 0))
    end = int(span.get("end", -1)) - int(window.get("start", 0))
    covered = [i for i, ((s, e), m) in enumerate(zip(offsets, content_mask)) if m and not (e <= start or s >= end)]
    if not covered:
        return 0, 0, 0
    if offsets[covered[0]][0] > start or offsets[covered[-1]][1] < end:
        return 0, 0, 0
    return covered[0], covered[-1], 1

