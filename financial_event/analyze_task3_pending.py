#!/usr/bin/env python3
"""Create row-level pending reasons and processing-channel statistics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from task3_common import ROOT, parse_datetime, read_jsonl, resolve_path


def clean(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "none", "null", "nan"} else value


def classify(row: dict[str, Any], document_mentions: list[dict[str, Any]]) -> tuple[str, str]:
    publish = clean(row.get("publish_time"))
    parseable = bool(parse_datetime(publish))
    if not publish:
        return "missing_publish_time", "pending"
    if not parseable:
        return "unparseable_publish_time", "pending"
    if not clean(row.get("event_mention_ids")):
        return "unresolved_document_link", "pending"
    if not row.get("has_linked_task2_evidence"):
        return "unresolved_cluster_link", "pending"
    if any(float(mention.get("trust_score") or 0) < 60 for mention in document_mentions if mention.get("trust_score") is not None):
        return "", "shadow"
    if not clean(row.get("title")) or not clean(row.get("source_name")) or not clean(row.get("content")):
        return "insufficient_core_fields", "repair"
    if any("industry" == clean(mention.get("target_type")) or any(token in clean(mention.get("provisional_event_type")) for token in ("政策", "行业", "宏观", "监管")) for mention in document_mentions):
        return "", "macro"
    return "", "main"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", default="outputs/task3/normalized/normalized_documents.jsonl")
    parser.add_argument("--mentions", default="outputs/task3/normalized/normalized_event_mentions.jsonl")
    parser.add_argument("--links", default="outputs/task3/normalized/task2_evidence_links.jsonl")
    parser.add_argument("--pending-output", default="outputs/task3/pending/pending_records.csv")
    parser.add_argument("--report", default="reports/task3_pending_analysis.json")
    args = parser.parse_args()
    documents = read_jsonl(args.documents)
    mentions = read_jsonl(args.mentions)
    links = read_jsonl(args.links)
    linked_mentions = {clean(row.get("event_mention_id")) for row in links if clean(row.get("event_mention_id")) and row.get("link_status") in {"linked", "resolved", "ambiguous"}}
    mention_by_document = {}
    for row in mentions:
        mention_by_document.setdefault(clean(row.get("document_id")), []).append(row)

    rows = []
    reason_counts = Counter()
    channel_counts = Counter()
    repaired_count = 0
    original_missing_publish_count = 0
    unresolved_time_rows = 0
    for document in documents:
        row = dict(document)
        if not parse_datetime(row.get("publish_time")):
            unresolved_time_rows += 1
        if row.get("metadata_repaired"):
            original_missing_publish_count += 1
        row["document_id"] = clean(row.get("document_id"))
        row["article_id"] = clean(row.get("article_id"))
        row["event_mention_count"] = len(mention_by_document.get(row["document_id"], []))
        row["has_linked_task2_evidence"] = any(clean(mention.get("event_mention_id")) in linked_mentions for mention in mention_by_document.get(row["document_id"], []))
        row["metadata_repaired"] = bool(row.get("metadata_repaired"))
        if row["metadata_repaired"]:
            repaired_count += 1
        reason, channel = classify(row, mention_by_document.get(row["document_id"], []))
        row["pending_reason"] = reason
        row["processing_channel"] = channel
        channel_counts[channel] += 1
        if channel == "pending":
            if reason:
                reason_counts[reason] += 1
            rows.append(row)

    output = resolve_path(args.pending_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["document_id", "article_id", "title", "source_name", "publish_time", "metadata_repaired", "missing_fields", "normalization_warnings", "event_mention_count", "has_linked_task2_evidence", "pending_reason", "processing_channel"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: json.dumps(row.get(column), ensure_ascii=False) if isinstance(row.get(column), (list, dict)) else row.get(column, "") for column in columns})

    report = {
        "input_documents": len(documents),
        "input_event_mentions": len(mentions),
        "original_pending_count": original_missing_publish_count + unresolved_time_rows,
        "successful_metadata_repair_count": repaired_count,
        "post_repair_pending_count": channel_counts.get("pending", 0),
        "issue_queue_count": len(rows),
        "pending_reason_distribution": dict(reason_counts),
        "processing_channel_counts": dict(channel_counts),
        "pending_output": str(output.relative_to(ROOT)),
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
