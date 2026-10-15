#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared IO, normalization, and validation helpers for task 3 tooling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "task3_annotation.yaml"


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
            rows.append(value)
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False, default=str) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    text = text.replace("T", " ").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    match = re.search(r"(20\d{2})[-年](\d{1,2})[-月](\d{1,2})", text)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(clean_text(item) for item in value if clean_text(item))
    if isinstance(value, dict):
        return " ".join(clean_text(item) for item in value.values() if clean_text(item))
    return re.sub(r"\s+", " ", str(value)).strip()


def as_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [clean_text(item) for item in value if clean_text(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return as_list(decoded)
        return [part.strip() for part in re.split(r"[,，;；|、]", text) if part.strip()]
    return [clean_text(value)]


def normalize_tokens(value: Any) -> set[str]:
    text = clean_text(value).lower()
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text))


def lexical_similarity(left: Any, right: Any) -> float:
    left_tokens = normalize_tokens(left)
    right_tokens = normalize_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def normalized_fingerprint(value: Any) -> str:
    return clean_text(value).lower().replace(" ", "")


def fingerprint_equal(left: Any, right: Any) -> bool:
    left_value = normalized_fingerprint(left)
    right_value = normalized_fingerprint(right)
    return bool(left_value and right_value and left_value == right_value)


def stable_pair_id(article_a: str, article_b: str, prefix: str) -> str:
    first, second = sorted((str(article_a), str(article_b)))
    digest = hashlib.sha1(f"{first}\0{second}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def first_nonempty(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return value
    return ""


def load_task1_events(paths: list[str | Path] | None = None) -> list[dict[str, Any]]:
    paths = paths or [
        ROOT / "outputs/final_task1/final_events_train.jsonl",
        ROOT / "outputs/final_task1/final_events_test.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    for path_value in paths:
        path = resolve_path(path_value)
        split = "test" if path.stem.endswith("_test") else "train"
        for row in read_jsonl(path):
            item = dict(row)
            item["dataset_split"] = item.get("dataset_split") or split
            item["article_id"] = clean_text(item.get("article_id"))
            item["source_name"] = clean_text(first_nonempty(item, "source_name", "source"))
            item["publish_dt"] = parse_datetime(item.get("publish_time"))
            item["body_for_matching"] = clean_text(
                first_nonempty(item, "content", "evidence_sentences", "title")
            )
            rows.append(item)
    if not rows:
        raise ValueError("任务一输入为空，无法构建任务三数据")
    return rows


def load_task2_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(resolve_path(path)):
        cluster_id = clean_text(row.get("cluster_id"))
        if cluster_id:
            result[cluster_id] = row
    return result


def load_cluster_members(path: str | Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    by_event: dict[str, str] = {}
    articles_by_cluster: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(resolve_path(path)):
        cluster_id = clean_text(row.get("cluster_id"))
        article_id = clean_text(row.get("article_id"))
        event_record_id = clean_text(row.get("event_record_id"))
        if cluster_id and article_id:
            articles_by_cluster[cluster_id].add(article_id)
        if cluster_id and event_record_id:
            by_event[event_record_id] = cluster_id
        if cluster_id and article_id:
            by_event.setdefault(article_id, cluster_id)
    return by_event, dict(articles_by_cluster)


def load_cluster_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    return {clean_text(row.get("cluster_id")): row for row in read_jsonl(resolve_path(path)) if clean_text(row.get("cluster_id"))}


def make_article_rows(events: list[dict[str, Any]], member_by_event: dict[str, str], scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        article_id = clean_text(row.get("article_id"))
        if article_id:
            grouped[article_id].append(row)
    result: list[dict[str, Any]] = []
    for article_id, article_events in grouped.items():
        first = next((row for row in article_events if clean_text(row.get("title")) or clean_text(row.get("publish_time"))), article_events[0])
        cluster_ids = set()
        for event in article_events:
            event_id = clean_text(event.get("event_id"))
            for key in (event_id, f"{event.get('dataset_split', '')}:{event_id}", article_id):
                if key in member_by_event:
                    cluster_ids.add(member_by_event[key])
        score_rows = [scores[cid] for cid in sorted(cluster_ids) if cid in scores]
        score = max(score_rows, key=lambda item: float(item.get("credibility_score", -1) or -1)) if score_rows else {}
        article = {
            "article_id": article_id,
            "title": clean_text(first.get("title")),
            "content": clean_text(first.get("content")),
            "source_name": clean_text(first.get("source_name") or first.get("source")),
            "publish_time": clean_text(first.get("publish_time")),
            "publish_dt": first.get("publish_dt"),
            "dataset_split": clean_text(first.get("dataset_split")),
            "event_records": article_events,
            "cluster_ids": sorted(cluster_ids),
            "trust_score": score.get("credibility_score", ""),
            "verdict": score.get("verification_status", ""),
            "source_level": score.get("source_level", ""),
            "conflicting_reports": score.get("risk_reasons", ""),
            "verification_hints": score.get("supporting_reasons", ""),
        }
        result.append(article)
    return result


def date_range(rows: Iterable[dict[str, Any]], fields: tuple[str, ...] = ("publish_time",)) -> tuple[datetime | None, datetime | None]:
    dates: list[datetime] = []
    for row in rows:
        for field in fields:
            value = row.get(field)
            dt = value if isinstance(value, datetime) else parse_datetime(value)
            if dt:
                dates.append(dt)
                break
    return (min(dates), max(dates)) if dates else (None, None)


def counter_distribution(rows: Iterable[dict[str, Any]], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = clean_text(row.get(field)) or "<missing>"
        counts[value] += 1
    return counts
