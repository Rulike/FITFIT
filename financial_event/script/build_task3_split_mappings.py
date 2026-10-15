#!/usr/bin/env python3
"""Emit document and event-mention split mappings."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from task3_common import ROOT, read_jsonl, resolve_path, write_jsonl


def clean(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "none", "null", "nan"} else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="outputs/task3/splits")
    parser.add_argument("--documents", default="outputs/task3/normalized/normalized_documents.jsonl")
    parser.add_argument("--mentions", default="outputs/task3/normalized/normalized_event_mentions.jsonl")
    parser.add_argument("--output-doc", default="outputs/task3/splits/document_split_mapping.jsonl")
    parser.add_argument("--output-mention", default="outputs/task3/splits/event_mention_split_mapping.jsonl")
    parser.add_argument("--report", default="reports/task3_split_mapping_report.json")
    args = parser.parse_args()
    documents = read_jsonl(args.documents)
    mentions = read_jsonl(args.mentions)
    split_dir = resolve_path(args.split_dir)
    split_docs: dict[str, dict[str, Any]] = {}
    for split_name in ("development", "validation", "test"):
        for row in read_jsonl(split_dir / f"{split_name}.jsonl", allow_missing=True):
            split_docs[clean(row.get("document_id"))] = {"split": split_name, "split_group_id": clean(row.get("split_group_id"))}
    doc_rows = []
    for document in documents:
        document_id = clean(document.get("document_id"))
        split_info = split_docs.get(document_id, {"split": "pending", "split_group_id": ""})
        doc_rows.append({
            "document_id": document_id,
            "article_id": clean(document.get("article_id")),
            "split": split_info["split"],
            "split_group_id": split_info["split_group_id"],
        })
    mention_rows = []
    document_split_map = {row["document_id"]: row["split"] for row in doc_rows}
    for mention in mentions:
        document_id = clean(mention.get("document_id"))
        mention_rows.append({
            "event_mention_id": clean(mention.get("event_mention_id")),
            "document_id": document_id,
            "article_id": clean(mention.get("article_id")),
            "split": document_split_map.get(document_id, "pending"),
        })
    write_jsonl(args.output_doc, doc_rows)
    write_jsonl(args.output_mention, mention_rows)
    doc_counts = defaultdict(int)
    for row in doc_rows:
        doc_counts[row["split"]] += 1
    mention_counts = defaultdict(int)
    for row in mention_rows:
        mention_counts[row["split"]] += 1
    report = {
        "document_split_counts": dict(doc_counts),
        "event_mention_split_counts": dict(mention_counts),
        "output_doc": args.output_doc,
        "output_mention": args.output_mention,
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
