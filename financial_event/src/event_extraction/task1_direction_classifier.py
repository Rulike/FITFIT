#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch direction classifier helpers for Task1 Step 5.7-X."""

from __future__ import annotations

from collections import Counter
from typing import Any


DIR_LABELS = ["positive", "negative", "neutral"]


def normalize_direction(value: str) -> str:
    return value if value in DIR_LABELS else "neutral"


def direction_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(normalize_direction(str(row.get("direction", "neutral"))) for row in rows))


def direction_text(row: dict[str, Any], window: dict[str, Any], pred: dict[str, Any] | None = None) -> str:
    fields = ""
    if pred:
        fields = " ".join(str((pred.get(k) or {}).get("text", "")) for k in ["trigger", "subject_company", "event_time", "object_target"])
    return f"事件：{row.get('event_name','')} 标题：{row.get('title','')} 句窗：{window.get('text','')} 字段：{fields}"

