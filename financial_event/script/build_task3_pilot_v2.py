#!/usr/bin/env python3
"""Build diversified task3 step 1.7 pilot annotation packages.

The outputs are candidate pairs only. Human label fields are deliberately blank.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from task3_common import ROOT, parse_datetime, read_jsonl, stable_hash

DOCS = ROOT / "outputs/task3/normalized/normalized_documents.jsonl"
MENTIONS = ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl"
ROUTING = ROOT / "outputs/task3/normalized/event_mention_routing.json"
DOC_SPLITS = ROOT / "outputs/task3/splits/document_split_mapping.jsonl"
MENTION_SPLITS = ROOT / "outputs/task3/splits/event_mention_split_mapping.jsonl"
NEAR_OUT = ROOT / "outputs/task3/annotation_pilot_v2/near_duplicate_pilot_v2.csv"
SAME_OUT = ROOT / "outputs/task3/annotation_pilot_v2/same_event_pilot_v2.csv"
REPORT_OUT = ROOT / "reports/task3_pilot_v2_sampling_report.json"

SEED = 2026
RELATION_KEYWORDS = ("传闻", "回应", "确认", "澄清", "否认", "终止", "恢复", "撤回", "反转", "辟谣", "进展")

NEAR_FIELDS = [
    "pair_id", "document_id_a", "document_id_b", "article_id_a", "article_id_b",
    "title_a", "title_b", "source_name_a", "source_name_b", "publish_time_a", "publish_time_b",
    "provisional_event_type_a", "provisional_event_type_b", "target_a", "target_b",
    "core_elements_a", "core_elements_b", "processing_channel_a", "processing_channel_b",
    "sampling_stratum", "title_similarity", "is_same_source", "entity_match", "entity_conflict",
    "core_fact_conflict", "time_diff_hours", "text_basis", "information_sufficiency",
    "candidate_reason", "human_is_near_duplicate", "human_confidence", "annotator_note",
]

SAME_FIELDS = [
    "pair_id", "event_mention_id_a", "event_mention_id_b", "document_id_a", "document_id_b",
    "article_id_a", "article_id_b", "title_a", "title_b", "source_name_a", "source_name_b",
    "publish_time_a", "publish_time_b", "target_a", "target_b",
    "provisional_event_type_a", "provisional_event_type_b", "core_elements_a", "core_elements_b",
    "processing_channel_a", "processing_channel_b", "sampling_stratum", "title_similarity",
    "target_match", "event_type_match", "core_fact_conflict", "time_diff_hours",
    "keyword_relation_hint", "text_basis", "information_sufficiency", "candidate_reason",
    "human_same_event", "human_relation", "human_confidence", "annotator_note",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def norm_title(value: Any) -> str:
    text = clean(value).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、：:；;（）()\[\]【】「」“”\"'|｜·—_-]", "", text)
    return text


def grams(text: str, n: int) -> set[str]:
    text = norm_title(text)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def title_similarity(a: Any, b: Any) -> float:
    scores = []
    for n in (2, 3):
        left, right = grams(clean(a), n), grams(clean(b), n)
        scores.append(len(left & right) / len(left | right) if left or right else 0.0)
    return round(max(scores), 6)


def template_key(title: str) -> str:
    text = norm_title(title)
    text = re.sub(r"[a-z]{0,3}\d{2,6}[a-z]{0,3}", "N", text)
    text = re.sub(r"\d+(?:\.\d+)?(?:万|亿|元|港元|%|％|日|月|年|点|分|股|份)?", "N", text)
    text = re.sub(r"^[\u4e00-\u9fff]{2,8}[：:]", "X:", text)
    text = re.sub(r"^[\u4e00-\u9fff]{2,10}", "X", text)
    return text[:40]


def dt(value: Any) -> datetime | None:
    return parse_datetime(value)


def hours_between(a: Any, b: Any) -> float | None:
    da, db = dt(a), dt(b)
    if not da or not db:
        return None
    return round(abs((da - db).total_seconds()) / 3600, 6)


def fact_tokens(title: str) -> set[str]:
    text = clean(title)
    tokens = set(re.findall(r"\d+(?:\.\d+)?\s*(?:万|亿|万元|亿元|港元|%|％|日|月|年|号|点|分|股|份)?", text))
    tokens.update(re.findall(r"[A-Z]{0,4}\d{2,8}[A-Z]{0,4}", text, flags=re.I))
    tokens.update(re.findall(r"[《“\"]([^《》“”\"]{2,40})[》”\"]", text))
    return {t.strip() for t in tokens if t.strip()}


def core_fact_conflict(a: str, b: str) -> bool:
    left, right = fact_tokens(a), fact_tokens(b)
    if not left or not right:
        return False
    common = left & right
    return len(common) < min(len(left), len(right))


def keyword_hint(a: str, b: str) -> str:
    hits = [kw for kw in RELATION_KEYWORDS if kw in clean(a) or kw in clean(b)]
    return "|".join(hits)


def target_key(row: dict[str, Any]) -> str:
    return "|".join([clean(row.get("target_type")), clean(row.get("target_code"))])


def target_label(row: dict[str, Any]) -> str:
    core = row.get("core_elements") if isinstance(row.get("core_elements"), list) else []
    name = next((clean(item) for item in core if clean(item) and clean(item) != clean(row.get("target_code"))), "")
    return "|".join(x for x in [clean(row.get("target_type")), clean(row.get("target_code")), name] if x)


def core_text(row: dict[str, Any]) -> str:
    value = row.get("core_elements")
    if isinstance(value, list):
        return "|".join(clean(item) for item in value if clean(item))
    return clean(value)


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def read_route_map() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(ROUTING)
    return {row["event_mention_id"]: row for row in rows}


def enriched_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, str]]:
    docs = read_jsonl(DOCS)
    mentions = read_jsonl(MENTIONS)
    routes = read_route_map()
    doc_splits = {row["document_id"]: row["split"] for row in read_jsonl(DOC_SPLITS)}
    mention_splits = {row["event_mention_id"]: row["split"] for row in read_jsonl(MENTION_SPLITS)}
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        route = routes.get(mention["event_mention_id"], {})
        mention["processing_channel"] = clean(route.get("processing_channel"))
        mention["split"] = mention_splits.get(mention["event_mention_id"], "")
        by_doc[mention["document_id"]].append(mention)
    for doc in docs:
        doc["split"] = doc_splits.get(doc["document_id"], "")
        doc_mentions = sorted(by_doc.get(doc["document_id"], []), key=lambda row: row["event_mention_id"])
        doc["_mentions"] = doc_mentions
        targets = sorted({target_key(m) for m in doc_mentions if target_key(m)})
        types = sorted({clean(m.get("provisional_event_type")) for m in doc_mentions if clean(m.get("provisional_event_type"))})
        channels = sorted({clean(m.get("processing_channel")) for m in doc_mentions if clean(m.get("processing_channel"))})
        cores = sorted({core_text(m) for m in doc_mentions if core_text(m)})
        doc["target_key"] = "|".join(targets)
        doc["target_label"] = " ; ".join(target_label(m) for m in doc_mentions if target_label(m))
        doc["provisional_event_type"] = "|".join(types)
        doc["processing_channel"] = "|".join(channels)
        doc["core_elements"] = " ; ".join(cores)
    return docs, mentions, doc_splits, mention_splits


def add_near_candidate(pool: dict[str, dict[str, dict[str, Any]]], a: dict[str, Any], b: dict[str, Any], stratum: str, reason: str) -> None:
    if a["document_id"] == b["document_id"] or a.get("split") != b.get("split") or a.get("split") not in {"development", "validation"}:
        return
    aa, bb = sorted((a, b), key=lambda row: row["document_id"])
    key = pair_key(aa["document_id"], bb["document_id"])
    sim = title_similarity(aa.get("title"), bb.get("title"))
    same_source = clean(aa.get("source_name")) == clean(bb.get("source_name")) and bool(clean(aa.get("source_name")))
    entity_match = bool(aa.get("target_key")) and aa.get("target_key") == bb.get("target_key")
    entity_conflict = bool(aa.get("target_key")) and bool(bb.get("target_key")) and aa.get("target_key") != bb.get("target_key")
    fact_conflict = core_fact_conflict(aa.get("title", ""), bb.get("title", ""))
    sufficiency = "insufficient" if stratum == "insufficient_information" else "reviewable_title_only"
    row = {
        "pair_id": "DOCV2-" + stable_hash("near_v2", aa["document_id"], bb["document_id"]),
        "document_id_a": aa["document_id"],
        "document_id_b": bb["document_id"],
        "article_id_a": aa.get("article_id", ""),
        "article_id_b": bb.get("article_id", ""),
        "title_a": clean(aa.get("title")),
        "title_b": clean(bb.get("title")),
        "source_name_a": clean(aa.get("source_name")),
        "source_name_b": clean(bb.get("source_name")),
        "publish_time_a": clean(aa.get("publish_time")),
        "publish_time_b": clean(bb.get("publish_time")),
        "provisional_event_type_a": clean(aa.get("provisional_event_type")),
        "provisional_event_type_b": clean(bb.get("provisional_event_type")),
        "target_a": clean(aa.get("target_label")),
        "target_b": clean(bb.get("target_label")),
        "core_elements_a": clean(aa.get("core_elements")),
        "core_elements_b": clean(bb.get("core_elements")),
        "processing_channel_a": clean(aa.get("processing_channel")),
        "processing_channel_b": clean(bb.get("processing_channel")),
        "sampling_stratum": stratum,
        "title_similarity": f"{sim:.6f}",
        "is_same_source": str(same_source),
        "entity_match": str(entity_match),
        "entity_conflict": str(entity_conflict),
        "core_fact_conflict": str(fact_conflict),
        "time_diff_hours": "" if hours_between(aa.get("publish_time"), bb.get("publish_time")) is None else str(hours_between(aa.get("publish_time"), bb.get("publish_time"))),
        "text_basis": "title_only",
        "information_sufficiency": sufficiency,
        "candidate_reason": reason,
        "human_is_near_duplicate": "",
        "human_confidence": "",
        "annotator_note": "",
        "_split": aa.get("split"),
        "_sort_score": sim,
    }
    existing = pool[stratum].get("|".join(key))
    if not existing or (row["_sort_score"], row["pair_id"]) > (existing["_sort_score"], existing["pair_id"]):
        pool[stratum]["|".join(key)] = row


def build_near_candidates(docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pool: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    usable = [d for d in docs if d.get("split") in {"development", "validation"} and clean(d.get("title"))]
    by_split = defaultdict(list)
    by_exact = defaultdict(list)
    by_template = defaultdict(list)
    by_target = defaultdict(list)
    by_day = defaultdict(list)
    for doc in usable:
        by_split[doc["split"]].append(doc)
        by_exact[(doc["split"], norm_title(doc.get("title")))].append(doc)
        by_template[(doc["split"], template_key(doc.get("title")))].append(doc)
        if clean(doc.get("target_key")):
            by_target[(doc["split"], doc["target_key"])].append(doc)
        date = (dt(doc.get("publish_time")) or datetime.min).strftime("%Y-%m-%d")
        by_day[(doc["split"], date)].append(doc)

    for group in by_exact.values():
        for i, left in enumerate(sorted(group, key=lambda r: r["document_id"])):
            for right in group[i + 1 :]:
                if clean(left.get("source_name")) == clean(right.get("source_name")):
                    add_near_candidate(pool, left, right, "same_source_high_similarity", "normalized_title_exact_match_same_source")
                else:
                    add_near_candidate(pool, left, right, "cross_source_high_similarity", "normalized_title_exact_match_cross_source")

    for group in by_template.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda r: stable_hash(SEED, "tpl", r["document_id"]))
        for i, left in enumerate(ordered[:35]):
            for right in ordered[i + 1 : i + 8]:
                if left.get("target_key") and right.get("target_key") and left.get("target_key") != right.get("target_key"):
                    add_near_candidate(pool, left, right, "template_title_different_entity", "template_title_same_pattern_entity_conflict")

    for group in by_target.values():
        ordered = sorted(group, key=lambda r: (dt(r.get("publish_time")) or datetime.min, r["document_id"]))
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 : i + 18]:
                h = hours_between(left.get("publish_time"), right.get("publish_time"))
                if h is not None and h > 72:
                    break
                sim = title_similarity(left.get("title"), right.get("title"))
                if 0.28 <= sim < 0.72:
                    add_near_candidate(pool, left, right, "same_entity_medium_similarity", "same_entity_medium_title_overlap_close_time")
                if sim >= 0.45 and core_fact_conflict(left.get("title", ""), right.get("title", "")):
                    add_near_candidate(pool, left, right, "high_similarity_core_conflict", "same_entity_title_overlap_numeric_or_product_conflict")

    for group in by_day.values():
        ordered = sorted(group, key=lambda r: stable_hash(SEED, "day", r["document_id"]))
        for i, left in enumerate(ordered[:240]):
            for right in ordered[i + 1 : i + 35]:
                sim = title_similarity(left.get("title"), right.get("title"))
                if sim >= 0.58 and left.get("target_key") != right.get("target_key"):
                    add_near_candidate(pool, left, right, "high_similarity_core_conflict", "same_day_high_title_overlap_entity_or_fact_conflict")
                if 0.12 <= sim <= 0.35 and left.get("processing_channel") == "shadow":
                    add_near_candidate(pool, left, right, "insufficient_information", "title_only_low_context_shadow_candidate")

    return {key: list(value.values()) for key, value in pool.items()}


def add_same_candidate(pool: dict[str, dict[str, dict[str, Any]]], a: dict[str, Any], b: dict[str, Any], docs: dict[str, dict[str, Any]], stratum: str, reason: str) -> None:
    if a["event_mention_id"] == b["event_mention_id"] or a.get("split") != b.get("split") or a.get("split") not in {"development", "validation"}:
        return
    aa, bb = sorted((a, b), key=lambda row: row["event_mention_id"])
    key = pair_key(aa["event_mention_id"], bb["event_mention_id"])
    doc_a, doc_b = docs.get(aa["document_id"], {}), docs.get(bb["document_id"], {})
    sim = title_similarity(doc_a.get("title", ""), doc_b.get("title", ""))
    target_match = bool(target_key(aa)) and target_key(aa) == target_key(bb)
    event_type_match = clean(aa.get("provisional_event_type")) == clean(bb.get("provisional_event_type")) and bool(clean(aa.get("provisional_event_type")))
    fact_conflict = core_fact_conflict(doc_a.get("title", ""), doc_b.get("title", ""))
    hint = keyword_hint(doc_a.get("title", ""), doc_b.get("title", ""))
    sufficiency = "insufficient" if stratum == "insufficient_information" else "reviewable_title_only"
    row = {
        "pair_id": "EMV2-" + stable_hash("same_v2", aa["event_mention_id"], bb["event_mention_id"]),
        "event_mention_id_a": aa["event_mention_id"],
        "event_mention_id_b": bb["event_mention_id"],
        "document_id_a": aa["document_id"],
        "document_id_b": bb["document_id"],
        "article_id_a": aa.get("article_id", ""),
        "article_id_b": bb.get("article_id", ""),
        "title_a": clean(doc_a.get("title")),
        "title_b": clean(doc_b.get("title")),
        "source_name_a": clean(aa.get("source_name")),
        "source_name_b": clean(bb.get("source_name")),
        "publish_time_a": clean(aa.get("publish_time")),
        "publish_time_b": clean(bb.get("publish_time")),
        "target_a": target_label(aa),
        "target_b": target_label(bb),
        "provisional_event_type_a": clean(aa.get("provisional_event_type")),
        "provisional_event_type_b": clean(bb.get("provisional_event_type")),
        "core_elements_a": core_text(aa),
        "core_elements_b": core_text(bb),
        "processing_channel_a": clean(aa.get("processing_channel")),
        "processing_channel_b": clean(bb.get("processing_channel")),
        "sampling_stratum": stratum,
        "title_similarity": f"{sim:.6f}",
        "target_match": str(target_match),
        "event_type_match": str(event_type_match),
        "core_fact_conflict": str(fact_conflict),
        "time_diff_hours": "" if hours_between(aa.get("publish_time"), bb.get("publish_time")) is None else str(hours_between(aa.get("publish_time"), bb.get("publish_time"))),
        "keyword_relation_hint": hint,
        "text_basis": "title_only",
        "information_sufficiency": sufficiency,
        "candidate_reason": reason,
        "human_same_event": "",
        "human_relation": "",
        "human_confidence": "",
        "annotator_note": "",
        "_split": aa.get("split"),
        "_sort_score": sim,
    }
    existing = pool[stratum].get("|".join(key))
    if not existing or (row["_sort_score"], row["pair_id"]) > (existing["_sort_score"], existing["pair_id"]):
        pool[stratum]["|".join(key)] = row


def build_same_candidates(mentions: list[dict[str, Any]], docs: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pool: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    usable = [m for m in mentions if m.get("split") in {"development", "validation"}]
    by_target = defaultdict(list)
    by_target_type = defaultdict(list)
    by_document = defaultdict(list)
    macro = []
    by_template = defaultdict(list)
    for mention in usable:
        by_target[(mention["split"], target_key(mention))].append(mention)
        by_target_type[(mention["split"], target_key(mention), clean(mention.get("provisional_event_type")))].append(mention)
        by_document[mention["document_id"]].append(mention)
        if mention.get("processing_channel") == "macro" or mention.get("target_type") == "industry":
            macro.append(mention)
        doc = docs.get(mention["document_id"], {})
        by_template[(mention["split"], template_key(doc.get("title", "")))].append(mention)

    for group in by_target_type.values():
        ordered = sorted(group, key=lambda r: (dt(r.get("publish_time")) or datetime.min, r["event_mention_id"]))
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 : i + 18]:
                h = hours_between(left.get("publish_time"), right.get("publish_time"))
                if h is not None and h > 72:
                    break
                add_same_candidate(pool, left, right, docs, "same_target_same_type_close_time", "same_target_same_type_within_72h")
                if core_fact_conflict(docs.get(left["document_id"], {}).get("title", ""), docs.get(right["document_id"], {}).get("title", "")):
                    add_same_candidate(pool, left, right, docs, "same_target_same_type_core_conflict", "same_target_same_type_numeric_or_product_conflict")
                if keyword_hint(docs.get(left["document_id"], {}).get("title", ""), docs.get(right["document_id"], {}).get("title", "")):
                    add_same_candidate(pool, left, right, docs, "clarification_refutation_reversal", "relation_keyword_same_target_pair")

    for group in by_target.values():
        ordered = sorted(group, key=lambda r: stable_hash(SEED, "target", r["event_mention_id"]))
        for i, left in enumerate(ordered[:60]):
            for right in ordered[i + 1 : i + 12]:
                if clean(left.get("provisional_event_type")) != clean(right.get("provisional_event_type")):
                    add_same_candidate(pool, left, right, docs, "same_target_different_type", "same_target_different_provisional_event_type")

    for group in by_document.values():
        if len(group) > 1:
            ordered = sorted(group, key=lambda r: r["event_mention_id"])
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    add_same_candidate(pool, left, right, docs, "same_document_distinct_mentions", "same_document_multiple_event_mentions")

    for group in by_template.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda r: stable_hash(SEED, "same_tpl", r["event_mention_id"]))
        for i, left in enumerate(ordered[:50]):
            for right in ordered[i + 1 : i + 10]:
                if target_key(left) != target_key(right):
                    add_same_candidate(pool, left, right, docs, "different_target_high_similarity", "same_title_template_different_target")

    macro_sorted = sorted(macro, key=lambda r: (r.get("split", ""), clean(r.get("provisional_event_type")), r["event_mention_id"]))
    for i, left in enumerate(macro_sorted):
        for right in macro_sorted[i + 1 : i + 25]:
            if left.get("split") == right.get("split") and target_key(left) != target_key(right):
                add_same_candidate(pool, left, right, docs, "macro_cross_target", "macro_or_industry_channel_cross_target")

    keyword_mentions = [m for m in usable if keyword_hint(docs.get(m["document_id"], {}).get("title", ""), "")]
    for group in by_target.values():
        kws = [m for m in group if m in keyword_mentions]
        for left in kws[:20]:
            for right in group[:20]:
                if left["event_mention_id"] != right["event_mention_id"]:
                    add_same_candidate(pool, left, right, docs, "clarification_refutation_reversal", "relation_keyword_nearby_target_pair")

    return {key: list(value.values()) for key, value in pool.items()}


def selected(rows_by_stratum: dict[str, list[dict[str, Any]]], quotas: list[tuple[str, int]], max_count: int, near: bool) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    same_source_count = 0

    def ordered(stratum: str) -> list[dict[str, Any]]:
        rows = rows_by_stratum.get(stratum, [])
        return sorted(rows, key=lambda row: (0 if row["_split"] == "development" else 1, -float(row["_sort_score"]), stable_hash(SEED, stratum, row["pair_id"])))

    def can_take(row: dict[str, Any]) -> bool:
        if row["pair_id"] in seen:
            return False
        if near and row["is_same_source"] == "True" and same_source_count >= 20:
            return False
        return True

    for stratum, quota in quotas:
        taken = 0
        for row in ordered(stratum):
            if taken >= quota:
                break
            if not can_take(row):
                continue
            chosen.append(row)
            seen.add(row["pair_id"])
            if near and row["is_same_source"] == "True":
                same_source_count += 1
            taken += 1

    leftovers = []
    for stratum in rows_by_stratum:
        leftovers.extend(ordered(stratum))
    for row in sorted(leftovers, key=lambda r: (0 if r["_split"] == "development" else 1, -float(r["_sort_score"]), stable_hash(SEED, "fill", r["pair_id"]))):
        if len(chosen) >= max_count:
            break
        if not can_take(row):
            continue
        chosen.append(row)
        seen.add(row["pair_id"])
        if near and row["is_same_source"] == "True":
            same_source_count += 1

    if near and sum(1 for row in chosen if row["is_same_source"] == "False") < 15:
        raise RuntimeError("not enough cross-source near-duplicate candidates")
    if len(chosen) != max_count:
        raise RuntimeError(f"selected {len(chosen)} candidates, expected {max_count}")
    return chosen


def clean_private_fields(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in fields} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_private_fields(rows, fields))


def csv_hash(rows: list[dict[str, Any]], fields: list[str]) -> str:
    payload = "\n".join(",".join(str(row.get(field, "")) for field in fields) for row in clean_private_fields(rows, fields))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def metrics(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    human_fields = ["human_is_near_duplicate", "human_confidence", "annotator_note"] if kind == "near" else ["human_same_event", "human_relation", "human_confidence", "annotator_note"]
    result = {
        "count": len(rows),
        "sampling_stratum_counts": dict(sorted(Counter(row["sampling_stratum"] for row in rows).items())),
        "title_only_count": sum(row.get("text_basis") == "title_only" for row in rows),
        "human_field_non_empty_count": sum(1 for row in rows for field in human_fields if clean(row.get(field))),
        "core_fact_conflict_count": sum(row.get("core_fact_conflict") == "True" for row in rows),
        "same_source_count": sum(clean(row.get("source_name_a")) == clean(row.get("source_name_b")) for row in rows),
        "cross_source_count": sum(clean(row.get("source_name_a")) != clean(row.get("source_name_b")) for row in rows),
        "development_pair_count": sum(row.get("_split") == "development" for row in rows),
        "validation_pair_count": sum(row.get("_split") == "validation" for row in rows),
    }
    if kind == "near":
        result["entity_match_count"] = sum(row.get("entity_match") == "True" for row in rows)
        result["entity_conflict_count"] = sum(row.get("entity_conflict") == "True" for row in rows)
        result["event_type_match_count"] = sum(clean(row.get("provisional_event_type_a")) == clean(row.get("provisional_event_type_b")) for row in rows)
        result["event_type_different_count"] = sum(clean(row.get("provisional_event_type_a")) != clean(row.get("provisional_event_type_b")) for row in rows)
    else:
        result["target_match_count"] = sum(row.get("target_match") == "True" for row in rows)
        result["target_conflict_count"] = sum(row.get("target_match") != "True" for row in rows)
        result["event_type_match_count"] = sum(row.get("event_type_match") == "True" for row in rows)
        result["event_type_different_count"] = sum(row.get("event_type_match") != "True" for row in rows)
    return result


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    docs, mentions, _, _ = enriched_inputs()
    docs_by_id = {doc["document_id"]: doc for doc in docs}
    near_pool = build_near_candidates(docs)
    same_pool = build_same_candidates(mentions, docs_by_id)
    near_quotas = [
        ("same_source_high_similarity", 10),
        ("cross_source_high_similarity", 10),
        ("template_title_different_entity", 8),
        ("same_entity_medium_similarity", 8),
        ("high_similarity_core_conflict", 8),
        ("insufficient_information", 6),
    ]
    same_quotas = [
        ("same_target_same_type_close_time", 10),
        ("same_target_same_type_core_conflict", 10),
        ("same_target_different_type", 8),
        ("different_target_high_similarity", 8),
        ("same_document_distinct_mentions", 6),
        ("macro_cross_target", 4),
        ("clarification_refutation_reversal", 4),
    ]
    near_rows = selected(near_pool, near_quotas, 50, near=True)
    same_rows = selected(same_pool, same_quotas, 50, near=False)
    report = {
        "step": "1.7",
        "random_seed": SEED,
        "used_test": False,
        "input_counts": {
            "documents": len(docs),
            "event_mentions": len(mentions),
            "documents_by_split": dict(sorted(Counter(doc.get("split") for doc in docs).items())),
            "event_mentions_by_split": dict(sorted(Counter(m.get("split") for m in mentions).items())),
        },
        "candidate_pool_counts": {
            "near_duplicate": {key: len(value) for key, value in sorted(near_pool.items())},
            "same_event": {key: len(value) for key, value in sorted(same_pool.items())},
        },
        "near_duplicate": metrics(near_rows, "near"),
        "same_event": metrics(same_rows, "same"),
        "development_samples_used": {
            "near_duplicate_pairs": sum(row.get("_split") == "development" for row in near_rows),
            "same_event_pairs": sum(row.get("_split") == "development" for row in same_rows),
            "unique_documents": len({row["document_id_a"] for row in near_rows + same_rows if row.get("_split") == "development"} | {row["document_id_b"] for row in near_rows + same_rows if row.get("_split") == "development"}),
            "unique_event_mentions": len({row.get("event_mention_id_a") for row in same_rows if row.get("_split") == "development"} | {row.get("event_mention_id_b") for row in same_rows if row.get("_split") == "development"}),
        },
        "validation_samples_used": {
            "near_duplicate_pairs": sum(row.get("_split") == "validation" for row in near_rows),
            "same_event_pairs": sum(row.get("_split") == "validation" for row in same_rows),
            "unique_documents": len({row["document_id_a"] for row in near_rows + same_rows if row.get("_split") == "validation"} | {row["document_id_b"] for row in near_rows + same_rows if row.get("_split") == "validation"}),
            "unique_event_mentions": len({row.get("event_mention_id_a") for row in same_rows if row.get("_split") == "validation"} | {row.get("event_mention_id_b") for row in same_rows if row.get("_split") == "validation"}),
        },
        "determinism": {
            "near_duplicate_csv_hash": csv_hash(near_rows, NEAR_FIELDS),
            "same_event_csv_hash": csv_hash(same_rows, SAME_FIELDS),
            "repeatable_with_fixed_seed": True,
        },
        "data_limitations": [
            "All normalized documents and event mentions have missing content; pilot rows are title_only.",
            "Clarification/refutation/reversal candidates are sparse because only titles are available.",
            "Candidate strata are heuristic discovery strata, not human truth labels.",
        ],
    }
    return near_rows, same_rows, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--near-output", default=str(NEAR_OUT))
    parser.add_argument("--same-output", default=str(SAME_OUT))
    parser.add_argument("--report-output", default=str(REPORT_OUT))
    args = parser.parse_args()
    near_rows, same_rows, report = build()
    write_csv(Path(args.near_output), near_rows, NEAR_FIELDS)
    write_csv(Path(args.same_output), same_rows, SAME_FIELDS)
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "near_duplicate_pilot_v2": len(near_rows),
        "same_event_pilot_v2": len(same_rows),
        "used_test": False,
        "near_strata": report["near_duplicate"]["sampling_stratum_counts"],
        "same_strata": report["same_event"]["sampling_stratum_counts"],
        "report": str(Path(args.report_output).relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
