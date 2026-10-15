#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build task3 candidates at document and event-mention granularity.

This script only creates review candidates. It never writes human labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from task3_common import ROOT, canonical_list, lexical_similarity, parse_datetime, read_jsonl, stable_hash, write_jsonl

DEFAULT_CONFIG = ROOT / "configs/task3_annotation.yaml"
DOCS = ROOT / "outputs/task3/normalized/normalized_documents.jsonl"
MENTIONS = ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl"
LINKS = ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"


def text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"none", "null", "nan"} else value


def first(*values: Any) -> str:
    for value in values:
        value = text(value)
        if value:
            return value
    return ""


def list_text(value: Any) -> list[str]:
    if isinstance(value, list):
        return canonical_list(value)
    return [text(value)] if text(value) else []


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dt(value: Any) -> datetime | None:
    return parse_datetime(value)


def gap_days(left: Any, right: Any) -> float | None:
    left_dt, right_dt = dt(left), dt(right)
    if not left_dt or not right_dt:
        return None
    return round(abs((left_dt - right_dt).total_seconds()) / 86400, 6)


def fmt_time(value: Any) -> str:
    parsed = dt(value)
    return parsed.isoformat() if parsed else ""


def stable_cap(rows: list[dict[str, Any]], limit: int, seed: int, label: str) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    return sorted(rows, key=lambda row: stable_hash(seed, label, row.get("pair_id", row.get("event_mention_id", row.get("provisional_cluster_id", "")))))[:limit]


def document_similarity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    title_sim = lexical_similarity(left.get("title"), right.get("title"))
    content_left = left.get("content") or left.get("title")
    content_right = right.get("content") or right.get("title")
    content_sim = lexical_similarity(content_left, content_right)
    lexical = round(max(title_sim, content_sim), 6)
    source_equal = bool(left.get("source_name") and left.get("source_name") == right.get("source_name"))
    return {
        "lexical_similarity": lexical,
        "title_similarity": round(title_sim, 6),
        "content_similarity": round(content_sim, 6),
        "source_equal": source_equal,
    }


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def build_document_pairs(documents: list[dict[str, Any]], mentions: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    max_gap = int(config.get("time_window_days", {}).get("near_duplicate_max_gap", 7))
    thresholds = config.get("similarity_thresholds", {})
    high = float(thresholds.get("title_high", 0.82))
    medium = float(thresholds.get("title_medium", 0.62))
    body_high = float(thresholds.get("body_high", 0.75))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        by_source[text(document.get("source_name"))].append(document)
        publish = dt(document.get("publish_time"))
        by_day[publish.strftime("%Y-%m-%d") if publish else "unknown"].append(document)

    def add(left: dict[str, Any], right: dict[str, Any], reason: str, hard_negative: bool = False) -> None:
        key = pair_key(left["document_id"], right["document_id"])
        if key in seen:
            return
        gap = gap_days(left.get("publish_time"), right.get("publish_time"))
        if gap is not None and gap > max_gap:
            return
        sim = document_similarity(left, right)
        if not (sim["title_similarity"] >= medium or sim["content_similarity"] >= body_high or sim["source_equal"]):
            return
        a, b = sorted((left, right), key=lambda row: row["document_id"])
        seen.add(key)
        rows.append({
            "pair_id": "DOC-" + stable_hash("near_duplicate", a["document_id"], b["document_id"]),
            "document_id_a": a["document_id"],
            "document_id_b": b["document_id"],
            "article_id_a": a["article_id"],
            "article_id_b": b["article_id"],
            "title_a": a.get("title", ""),
            "title_b": b.get("title", ""),
            "content_excerpt_a": (a.get("content") or a.get("title", ""))[:500],
            "content_excerpt_b": (b.get("content") or b.get("title", ""))[:500],
            "source_a": a.get("source_name", ""),
            "source_b": b.get("source_name", ""),
            "publish_time_a": fmt_time(a.get("publish_time")),
            "publish_time_b": fmt_time(b.get("publish_time")),
            "lexical_similarity": sim["lexical_similarity"],
            "title_similarity": sim["title_similarity"],
            "content_similarity": sim["content_similarity"],
            "source_equal": sim["source_equal"],
            "candidate_reason": reason,
            "hard_negative": hard_negative,
            "human_is_near_duplicate": "",
            "annotator_note": "",
        })

    for source, source_docs in sorted(by_source.items()):
        if not source:
            continue
        ordered = sorted(source_docs, key=lambda row: (dt(row.get("publish_time")) or datetime.min, row["document_id"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:index + 12]:
                add(left, right, "same_source")

    for day, day_docs in sorted(by_day.items()):
        ordered = sorted(day_docs, key=lambda row: row["document_id"])
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:index + 10]:
                sim = document_similarity(left, right)
                if sim["title_similarity"] >= high or sim["content_similarity"] >= body_high:
                    add(left, right, "high_text_similarity")

    # Hard negatives are drawn from nearby documents with strong wording overlap
    # but different provisional mention targets. They remain unverified candidates.
    mention_by_document = {document["document_id"]: document.get("event_mention_ids", []) for document in documents}
    mention_by_id = {mention["event_mention_id"]: mention for mention in mentions}
    for day_docs in by_day.values():
        ordered = sorted(day_docs, key=lambda row: row["document_id"])
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:index + 16]:
                sim = document_similarity(left, right)
                if sim["lexical_similarity"] < medium:
                    continue
                left_targets = {(mention_by_id[mid].get("target_type"), mention_by_id[mid].get("target_code")) for mid in mention_by_document[left["document_id"]] if mid in mention_by_id}
                right_targets = {(mention_by_id[mid].get("target_type"), mention_by_id[mid].get("target_code")) for mid in mention_by_document[right["document_id"]] if mid in mention_by_id}
                if left_targets.isdisjoint(right_targets):
                    add(left, right, "hard_negative_entity_conflict", True)

    seed = int(config.get("random_seed", config.get("seed", 2026)))
    limit = int(config.get("candidate_limits", {}).get("near_duplicate_pairs", 1500))
    return stable_cap(rows, limit, seed, "document_pairs")


def mention_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    elements_left = json.dumps(left.get("elements", {}), ensure_ascii=False, sort_keys=True)
    elements_right = json.dumps(right.get("elements", {}), ensure_ascii=False, sort_keys=True)
    return lexical_similarity(elements_left, elements_right)


def build_same_event_pairs(mentions: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    max_gap = int(config.get("time_window_days", {}).get("same_event_max_gap", 180))
    same_type_gap = int(config.get("time_window_days", {}).get("same_company_same_type_max_gap", 45))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        groups[(text(mention.get("target_type")), text(mention.get("target_code")))].append(mention)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(left: dict[str, Any], right: dict[str, Any], reason: str) -> None:
        key = pair_key(left["event_mention_id"], right["event_mention_id"])
        if key in seen or left["event_mention_id"] == right["event_mention_id"]:
            return
        gap = gap_days(left.get("publish_time"), right.get("publish_time"))
        if gap is not None and gap > max_gap:
            return
        a, b = sorted((left, right), key=lambda row: row["event_mention_id"])
        same_document = a["document_id"] == b["document_id"]
        seen.add(key)
        rows.append({
            "pair_id": "EM-" + stable_hash("same_event", a["event_mention_id"], b["event_mention_id"]),
            "event_mention_id_a": a["event_mention_id"],
            "event_mention_id_b": b["event_mention_id"],
            "document_id_a": a["document_id"],
            "document_id_b": b["document_id"],
            "article_id_a": a["article_id"],
            "article_id_b": b["article_id"],
            "same_document": same_document,
            "stock_a": "|".join(a.get("linked_stocks", [])),
            "stock_b": "|".join(b.get("linked_stocks", [])),
            "event_type_a": a.get("event_type"),
            "event_type_b": b.get("event_type"),
            "provisional_event_type_a": a.get("provisional_event_type"),
            "provisional_event_type_b": b.get("provisional_event_type"),
            "event_type_is_provisional_a": a.get("event_type_is_provisional", True),
            "event_type_is_provisional_b": b.get("event_type_is_provisional", True),
            "target_type_a": a.get("target_type"),
            "target_type_b": b.get("target_type"),
            "target_code_a": a.get("target_code"),
            "target_code_b": b.get("target_code"),
            "core_entities_a": "|".join(a.get("core_elements", a.get("elements", {}).get("core_entities", []))) if isinstance(a.get("elements"), dict) or isinstance(a.get("core_elements"), list) else "",
            "core_entities_b": "|".join(b.get("core_elements", b.get("elements", {}).get("core_entities", []))) if isinstance(b.get("elements"), dict) or isinstance(b.get("core_elements"), list) else "",
            "content_excerpt_a": "", 
            "content_excerpt_b": "",
            "publish_time_a": fmt_time(a.get("publish_time")),
            "publish_time_b": fmt_time(b.get("publish_time")),
            "time_gap_days": gap,
            "fingerprint_a": a.get("event_fingerprint"),
            "fingerprint_b": b.get("event_fingerprint"),
            "trust_score_a": a.get("trust_score"),
            "trust_score_b": b.get("trust_score"),
            "trust_score_scope_a": a.get("trust_score_scope"),
            "trust_score_scope_b": b.get("trust_score_scope"),
            "candidate_reason": reason,
            "human_same_event": "",
            "human_relation": "",
            "annotator_note": "",
        })

    for group_key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: (dt(row.get("publish_time")) or datetime.min, row["event_mention_id"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:index + 18]:
                gap = gap_days(left.get("publish_time"), right.get("publish_time"))
                same_schema = left.get("schema_id") and left.get("schema_id") == right.get("schema_id")
                reason = "same_document_distinct_mentions" if left["document_id"] == right["document_id"] else "same_target_same_schema" if same_schema else "same_target_different_schema"
                if left["document_id"] == right["document_id"] or same_schema or (gap is not None and gap <= same_type_gap):
                    add(left, right, reason)

    # Same document is always useful for checking whether extraction produced distinct mentions.
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        by_document[mention["document_id"]].append(mention)
    for group in by_document.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                add(left, right, "same_document_distinct_mentions")

    seed = int(config.get("random_seed", config.get("seed", 2026)))
    limit = int(config.get("candidate_limits", {}).get("same_event_pairs", 1500))
    return stable_cap(rows, limit, seed, "mention_pairs")


def cluster_seed(documents: list[dict[str, Any]], mentions: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    titles = {document["document_id"]: document.get("title", "") for document in documents}
    rows = []
    for mention in mentions:
        rows.append({
            "event_mention_id": mention["event_mention_id"],
            "document_id": mention["document_id"],
            "article_id": mention["article_id"],
            "title": titles.get(mention["document_id"], ""),
            "publish_time": mention.get("publish_time"),
            "linked_stocks": mention.get("linked_stocks", []),
            "linked_industries": mention.get("linked_industries", []),
            "event_type": mention.get("event_type"),
            "provisional_event_type": mention.get("provisional_event_type"),
            "event_type_is_provisional": True,
            "core_entities": mention.get("elements", {}).get("core_entities", []) if isinstance(mention.get("elements"), dict) else [],
            "trust_score": mention.get("trust_score"),
            "trust_score_scope": mention.get("trust_score_scope"),
            "provisional_cluster_id": mention.get("provisional_cluster_id"),
            "human_cluster_id": "",
            "annotator_note": "",
        })
    seed_value = int(config.get("random_seed", config.get("seed", 2026)))
    return stable_cap(rows, int(config.get("candidate_limits", {}).get("cluster_annotation_seed", 1200)), seed_value, "cluster_seed")


def timeline_seed(documents: list[dict[str, Any]], mentions: list[dict[str, Any]], links: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    mention_by_doc: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        mention_by_doc[mention["document_id"]].append(mention["event_mention_id"])
    docs_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        clusters = {mention.get("provisional_cluster_id") for mention in mentions if mention["document_id"] == document["document_id"] and mention.get("provisional_cluster_id")}
        for cluster in clusters:
            docs_by_cluster[cluster].append(document)
    link_by_cluster: dict[str, dict[str, Any]] = {}
    for link in links:
        cluster = link.get("task2_cluster_id")
        if cluster and cluster not in link_by_cluster:
            link_by_cluster[cluster] = link
    rows = []
    for cluster, cluster_docs in docs_by_cluster.items():
        if len(cluster_docs) < 2:
            continue
        ordered = sorted(cluster_docs, key=lambda row: (dt(row.get("publish_time")) or datetime.min, row["document_id"]))
        link = link_by_cluster.get(cluster, {})
        score = (link.get("credibility_result") or {}).get("credibility_score") if isinstance(link.get("credibility_result"), dict) else None
        rows.append({
            "provisional_cluster_id": cluster,
            "articles": [{"document_id": doc["document_id"], "article_id": doc["article_id"], "event_mention_ids": sorted(mention_by_doc[doc["document_id"]]), "title": doc.get("title", ""), "publish_time": doc.get("publish_time"), "source_name": doc.get("source_name", "")} for doc in ordered],
            "sorted_publish_times": [doc.get("publish_time") for doc in ordered],
            "sources": canonical_list(doc.get("source_name") for doc in ordered),
            "claims_if_available": [],
            "human_key_nodes": "",
            "human_stage": "",
            "human_relation": "",
            "human_reversal": "",
            "annotator_note": "",
            "trust_score": score,
            "trust_score_scope": "cluster" if score is not None else None,
        })
    rows.sort(key=lambda row: (-len(row["articles"]), row["provisional_cluster_id"]))
    seed_value = int(config.get("random_seed", config.get("seed", 2026)))
    return stable_cap(rows, int(config.get("candidate_limits", {}).get("timeline_annotation_seed", 400)), seed_value, "timeline_seed")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(Path(args.config))
    documents = read_jsonl(DOCS)
    mentions = read_jsonl(MENTIONS)
    links = read_jsonl(LINKS)
    if not documents or not mentions:
        raise RuntimeError("normalized task3 inputs are missing or empty")
    near_rows = build_document_pairs(documents, mentions, config)
    documents_by_id = {document["document_id"]: document for document in documents}
    mention_by_id = {mention["event_mention_id"]: mention for mention in mentions}
    mentions_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        mentions_by_document[mention["document_id"]].append(mention)
    for row in near_rows:
        left_mentions = mentions_by_document.get(row["document_id_a"], [])
        right_mentions = mentions_by_document.get(row["document_id_b"], [])
        left = left_mentions[0] if left_mentions else {}
        right = right_mentions[0] if right_mentions else {}
        for suffix, mention in (("a", left), ("b", right)):
            row[f"target_type_{suffix}"] = mention.get("target_type")
            row[f"target_code_{suffix}"] = mention.get("target_code")
            row[f"core_entities_{suffix}"] = "|".join(mention.get("core_elements", []))
            row[f"provisional_event_type_{suffix}"] = mention.get("provisional_event_type")
            row[f"trust_score_{suffix}"] = mention.get("trust_score")
            row[f"trust_score_scope_{suffix}"] = mention.get("trust_score_scope")
    same_rows = build_same_event_pairs(mentions, config)
    mention_by_id = {mention["event_mention_id"]: mention for mention in mentions}
    for row in same_rows:
        left = mention_by_id.get(row["event_mention_id_a"], {})
        right = mention_by_id.get(row["event_mention_id_b"], {})
        left_doc = documents_by_id.get(left.get("document_id"), {})
        right_doc = documents_by_id.get(right.get("document_id"), {})
        row["title_a"] = left_doc.get("title", "")
        row["title_b"] = right_doc.get("title", "")
        row["content_excerpt_a"] = (left_doc.get("content") or left_doc.get("title", ""))[:500]
        row["content_excerpt_b"] = (right_doc.get("content") or right_doc.get("title", ""))[:500]
    cluster_rows = cluster_seed(documents, mentions, config)
    timeline_rows = timeline_seed(documents, mentions, links, config)
    outputs = config.get("output_paths", {})
    write_csv(ROOT / outputs.get("near_duplicate_pairs", "outputs/task3/near_duplicate_pairs.csv"), near_rows)
    write_csv(ROOT / outputs.get("same_event_pairs", "outputs/task3/same_event_pairs.csv"), same_rows)
    write_jsonl(ROOT / outputs.get("cluster_annotation_seed", "outputs/task3/cluster_annotation_seed.jsonl"), cluster_rows)
    write_jsonl(ROOT / outputs.get("timeline_annotation_seed", "outputs/task3/timeline_annotation_seed.jsonl"), timeline_rows)
    summary = {
        "input_documents": len(documents),
        "input_event_mentions": len(mentions),
        "near_duplicate_pairs": len(near_rows),
        "same_event_pairs": len(same_rows),
        "cluster_annotation_seed": len(cluster_rows),
        "timeline_annotation_seed": len(timeline_rows),
        "near_duplicate_granularity": "document",
        "same_event_granularity": "event_mention",
        "human_labels_generated": False,
    }
    report_path = ROOT / "reports/task3_candidate_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
