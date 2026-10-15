"""Shared IO and normalization helpers for task 3 audit tooling."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_datetime(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().replace("T", " ")
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    text = text.replace("Z", "")
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    match = re.search(r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})", text)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(clean_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(clean_text(item) for item in value.values())
    return re.sub(r"\s+", " ", str(value)).strip()


def as_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [clean_text(item) for item in parsed if clean_text(item)]
    except (TypeError, json.JSONDecodeError):
        pass
    return [text]


def tokens(value: Any) -> set[str]:
    text = clean_text(value).lower()
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text))


def lexical_similarity(left: Any, right: Any) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(clean_text(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def flatten_unique(values: Iterable[Any]) -> list[str]:
    result: set[str] = set()
    for value in values:
        result.update(as_list(value))
    return sorted(item for item in result if item)


def group_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = clean_text(row.get(key))
        if value:
            grouped[value].append(row)
    return dict(grouped)


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
