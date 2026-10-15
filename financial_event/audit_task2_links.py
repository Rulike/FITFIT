#!/usr/bin/env python3
"""Audit and normalize Task2 evidence links for task3 step 1.6."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from task3_common import ROOT, canonical_list, read_jsonl, resolve_path, write_jsonl


def clean(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "none", "null", "nan"} else value


def load_jsonl_map(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path, allow_missing=True)
    return {clean(row.get(key_field)): row for row in rows if clean(row.get(key_field))}


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def determine_scope(mention: dict[str, Any], cluster_row: dict[str, Any] | None, link_stat: str) -> str:
    if link_stat == "ambiguous":
        return "document"
    if cluster_row is None:
        return "event_mention"
    if len(canonical_list(mention.get("core_elements", []))) >= 2:
        return "event_mention"
    return "document"


def link_status(mention: dict[str, Any], cluster_row: dict[str, Any] | None) -> tuple[str, str]:
    if cluster_row is None:
        return "unresolved", "no_cluster_score"
    if not clean(mention.get("schema_id")) or not clean(mention.get("target_type")) or not clean(mention.get("target_code")):
        return "ambiguous", "missing_target_identity"
    same_schema = clean(mention.get("schema_id")) == clean(cluster_row.get("schema_id"))
    same_target = clean(mention.get("target_type")) == clean(cluster_row.get("target_type")) and clean(mention.get("target_code")) == clean(cluster_row.get("target_code"))
    core = canonical_list(mention.get("core_elements", []))
    if same_schema and same_target and core:
        return "resolved", "schema_target_core_alignment"
    if same_target and core:
        return "resolved", "target_and_core_elements_match"
    return "ambiguous", "cluster_only_no_event_mention_disambiguation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", default="outputs/task3/normalized/task2_evidence_links.jsonl")
    parser.add_argument("--mentions", default="outputs/task3/normalized/normalized_event_mentions.jsonl")
    parser.add_argument("--clusters", default="outputs/event_credibility/event_credibility_scores.jsonl")
    parser.add_argument("--output-links", default="outputs/task3/normalized/task2_evidence_links.jsonl")
    parser.add_argument("--ambiguous-output", default="outputs/task3/normalized/task2_ambiguous_links.jsonl")
    parser.add_argument("--report", default="reports/task3_task2_link_audit.json")
    args = parser.parse_args()

    link_rows = read_jsonl(args.links)
    mentions = read_jsonl(args.mentions)
    mentions_by_id = {clean(row.get("event_mention_id")): row for row in mentions if clean(row.get("event_mention_id"))}
    clusters = load_jsonl_map(resolve_path(args.clusters), "cluster_id")

    deduped = []
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_rows = 0
    ambiguous_rows = []
    resolved_count = 0
    ambiguous_count = 0
    unresolved_count = 0
    scope_counts = Counter()
    mention_link_counts = Counter()
    doc_cluster_counts: dict[str, set[str]] = defaultdict(set)
    doc_mention_counts: dict[str, set[str]] = defaultdict(set)
    doc_cartesian_flags: dict[str, bool] = defaultdict(bool)
    reason_counts = Counter()

    for row in link_rows:
        mention_id = clean(row.get("event_mention_id"))
        cluster_id = clean(row.get("task2_cluster_id"))
        if not mention_id:
            continue
        mention = mentions_by_id.get(mention_id, {})
        cluster_row = clusters.get(cluster_id) if cluster_id else None
        key = pair_key(mention_id, cluster_id)
        if key in seen_pairs:
            duplicate_rows += 1
            continue
        seen_pairs.add(key)
        link_stat, reason = link_status(mention, cluster_row)
        scope = determine_scope(mention, cluster_row, link_stat)
        document_id = clean(row.get("document_id") or mention.get("document_id"))
        if document_id:
            doc_cluster_counts[document_id].add(cluster_id)
            doc_mention_counts[document_id].add(mention_id)
            if len(doc_mention_counts[document_id]) > 1 and len(doc_cluster_counts[document_id]) > 1:
                doc_cartesian_flags[document_id] = True
        output = {
            "document_id": document_id,
            "event_mention_id": mention_id,
            "task2_cluster_id": cluster_id or None,
            "link_scope": scope,
            "link_status": link_stat,
            "link_reason": reason,
            "trust_score": row.get("trust_score") if link_stat == "resolved" else None,
            "credibility_score": row.get("credibility_score") if link_stat == "resolved" else None,
            "verdict": row.get("verdict") if link_stat == "resolved" else None,
            "trust_score_scope": row.get("trust_score_scope") if link_stat == "resolved" else None,
            "link_method": row.get("link_method") or "article_id_to_event_cluster_members_to_cluster_id",
            "source_path": row.get("source_path") or "outputs/event_clustering/event_cluster_members.jsonl",
        }
        if link_stat == "resolved":
            resolved_count += 1
            mention_link_counts[mention_id] += 1
        elif link_stat == "ambiguous":
            ambiguous_count += 1
            ambiguous_rows.append(output)
        else:
            unresolved_count += 1
        scope_counts[scope] += 1
        reason_counts[reason] += 1
        deduped.append(output)

    ambiguous_path = resolve_path(args.ambiguous_output)
    write_jsonl(ambiguous_path, ambiguous_rows)
    output_path = resolve_path(args.output_links)
    write_jsonl(output_path, deduped)

    docs_with_multiple_mentions = sum(1 for values in doc_mention_counts.values() if len(values) > 1)
    cartesian_docs = sum(1 for flag in doc_cartesian_flags.values() if flag)
    mention_scope_distribution = Counter()
    for mention_id, count in mention_link_counts.items():
        mention_scope_distribution[count] += 1
    mentions_without_links = sum(1 for mention in mentions if mention.get("event_mention_id") not in mention_link_counts)
    report = {
        "input_link_rows": len(link_rows),
        "deduplicated_link_rows": len(deduped),
        "duplicate_rows_removed": duplicate_rows,
        "resolved_count": resolved_count,
        "ambiguous_count": ambiguous_count,
        "unresolved_count": unresolved_count,
        "event_mention_link_distribution": {str(key): value for key, value in sorted(mention_scope_distribution.items())},
        "mentions_without_links": mentions_without_links,
        "docs_with_multiple_mentions": docs_with_multiple_mentions,
        "cartesian_expansion_docs": cartesian_docs,
        "link_scope_counts": dict(scope_counts),
        "link_reason_counts": dict(reason_counts),
        "ambiguous_output": str(ambiguous_path.relative_to(ROOT)),
        "canonical_output": str(output_path.relative_to(ROOT)),
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
