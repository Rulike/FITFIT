#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gold-free sentence/window candidates for Task1 structure extraction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_sentences(content: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    pattern = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", re.M)
    sid = 1
    for match in pattern.finditer(content or ""):
        segment = match.group(0)
        if not segment:
            continue
        leading = len(segment) - len(segment.lstrip())
        trailing_end = len(segment.rstrip())
        start = match.start() + leading
        end = match.start() + trailing_end
        if start >= end:
            continue
        sentences.append({"sentence_id": f"S{sid:04d}", "text": content[start:end], "start": start, "end": end})
        sid += 1
    if not sentences and content:
        sentences.append({"sentence_id": "S0001", "text": content, "start": 0, "end": len(content)})
    return sentences


def paragraph_spans(content: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    start = 0
    pid = 1
    for part in re.split(r"(\n+)", content or ""):
        if not part:
            continue
        end = start + len(part)
        if not part.isspace():
            text = content[start:end].strip()
            lead = len(content[start:end]) - len(content[start:end].lstrip())
            trail = len(content[start:end].rstrip())
            real_start = start + lead
            real_end = start + trail
            if real_start < real_end:
                spans.append({"sentence_id": f"P{pid:04d}", "text": content[real_start:real_end], "start": real_start, "end": real_end})
                pid += 1
        start = end
    return spans


def make_sentence_windows(row: dict[str, Any], max_size: int = 3, max_candidates: int = 100000) -> list[dict[str, Any]]:
    content = str(row.get("content", "") or "")
    sentences = split_sentences(content)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    def add(start: int, end: int, ids: list[str], source_type: str) -> None:
        if not (0 <= start < end <= len(content)):
            return
        key = (start, end, source_type)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "candidate_id": f"C{len(candidates) + 1:04d}",
                "sentence_ids": ids,
                "text": content[start:end],
                "start": start,
                "end": end,
                "source_type": source_type,
                "offset_valid": content[start:end] == content[start:end],
            }
        )

    for i in range(len(sentences)):
        for size in range(1, max_size + 1):
            j = i + size - 1
            if j >= len(sentences):
                continue
            add(sentences[i]["start"], sentences[j]["end"], [sentences[k]["sentence_id"] for k in range(i, j + 1)], f"{size}_sentence_window")

    for para in paragraph_spans(content):
        add(para["start"], para["end"], [para["sentence_id"]], "paragraph")

    if sentences:
        add(sentences[0]["start"], sentences[0]["end"], ["TITLE", sentences[0]["sentence_id"]], "title_first_sentence")

    candidates.sort(key=lambda x: (x["start"], x["end"], x["source_type"]))
    return candidates[:max_candidates]


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def char_f1(gold: dict[str, Any], pred: dict[str, Any]) -> float:
    gs, ge = int(gold.get("start", -1)), int(gold.get("end", -1))
    ps, pe = int(pred.get("start", -1)), int(pred.get("end", -1))
    if gs < 0 or ge <= gs or ps < 0 or pe <= ps:
        return 0.0
    overlap = overlap_len(gs, ge, ps, pe)
    if overlap <= 0:
        return 0.0
    precision = overlap / max(1, pe - ps)
    recall = overlap / max(1, ge - gs)
    return 2 * precision * recall / max(1e-12, precision + recall)


def contains_gold_span(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    span = row.get("event_span", {}) or {}
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    return start >= 0 and end > start and int(candidate.get("start", 10**9)) <= start and end <= int(candidate.get("end", -1))


def best_gold_candidate(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {}
    best = max(candidates, key=lambda cand: (1 if contains_gold_span(row, cand) else 0, char_f1(row.get("event_span", {}), cand), -len(cand.get("text", ""))))
    out = dict(best)
    out["contains_gold_span"] = contains_gold_span(row, out)
    out["gold_char_f1"] = round(char_f1(row.get("event_span", {}), out), 6)
    return out


def candidate_pool_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered = 0
    illegal = 0
    total_yes = 0
    source_counts: dict[str, int] = {}
    for row in rows:
        candidates = make_sentence_windows(row)
        for cand in candidates:
            source_counts[cand["source_type"]] = source_counts.get(cand["source_type"], 0) + 1
            if row.get("content", "")[cand["start"] : cand["end"]] != cand["text"]:
                illegal += 1
        if row.get("event_exists", "yes") == "no" or not row.get("event_span", {}).get("text"):
            continue
        total_yes += 1
        covered += int(any(contains_gold_span(row, cand) for cand in candidates))
    return {
        "event_exists_yes_count": total_yes,
        "covered_count": covered,
        "coverage": round(covered / max(1, total_yes), 6),
        "candidate_offset_illegal_count": illegal,
        "source_type_counts": source_counts,
    }
