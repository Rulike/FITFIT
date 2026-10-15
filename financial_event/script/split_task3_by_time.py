#!/usr/bin/env python3
"""Create a time split without treating unlabeled candidate edges as truth."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task3_common import ROOT, parse_datetime, read_csv_rows, read_jsonl, resolve_path, write_jsonl


def parse_required_date(value: str, name: str) -> datetime:
    parsed = parse_datetime(value)
    if parsed is None:
        raise ValueError(f"{name} 不是可解析日期: {value}")
    return parsed.replace(tzinfo=timezone.utc)


def parse_with_timezone(value: Any) -> datetime | None:
    parsed = parse_datetime(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed else None


def union_find(values: list[str]):
    parent = {value: value for value in values}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    return find, union


def load_edge_inventory(path: Path, document_ids: set[str]) -> tuple[list[tuple[str, str, dict[str, Any]]], int]:
    used: list[tuple[str, str, dict[str, Any]]] = []
    ignored = 0
    if not path.exists():
        return used, ignored
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            left = str(row.get("document_id_a") or row.get("article_id_a") or "").strip()
            right = str(row.get("document_id_b") or row.get("article_id_b") or "").strip()
            if not left or not right or left not in document_ids or right not in document_ids:
                continue
            human = str(row.get("human_is_near_duplicate") or "").strip().lower()
            if human in {"true", "1", "yes", "positive"}:
                used.append((left, right, {"edge_type": "human_near_duplicate", "edge_source": str(path.relative_to(ROOT)), "is_human_verified": True, "confidence": "human", "used_for_split": True}))
            else:
                ignored += 1
    return used, ignored


def load_human_same_event_edges(path: Path, mention_to_document: dict[str, str], document_ids: set[str]) -> tuple[list[tuple[str, str, dict[str, Any]]], int]:
    used: list[tuple[str, str, dict[str, Any]]] = []
    if not path.exists():
        return used, 0
    ignored = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            human = str(row.get("human_same_event") or "").strip().lower()
            left_mention = str(row.get("event_mention_id_a") or "").strip()
            right_mention = str(row.get("event_mention_id_b") or "").strip()
            left, right = mention_to_document.get(left_mention), mention_to_document.get(right_mention)
            if not left or not right or left not in document_ids or right not in document_ids:
                continue
            if human in {"true", "1", "yes", "positive", "same_event", "related_update", "clarification_or_refutation"}:
                used.append((left, right, {"edge_type": "human_same_event", "edge_source": str(path.relative_to(ROOT)), "is_human_verified": True, "confidence": "human", "used_for_split": True}))
            else:
                ignored += 1
    return used, ignored


def load_provisional_edges(mentions: list[dict[str, Any]], allow: bool) -> list[tuple[str, str, dict[str, Any]]]:
    if not allow:
        return []
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        cluster = str(mention.get("provisional_cluster_id") or "").strip()
        document = str(mention.get("document_id") or "").strip()
        if cluster and document and document not in by_cluster[cluster]:
            by_cluster[cluster].append(document)
    edges = []
    for cluster, documents in by_cluster.items():
        for right in documents[1:]:
            edges.append((documents[0], right, {"edge_type": "provisional_cluster", "edge_source": "normalized_event_mentions.jsonl", "is_human_verified": False, "confidence": "provisional", "used_for_split": True, "provisional_cluster_id": cluster}))
    return edges


def assign(dt: datetime, dev_end: datetime, validation_end: datetime) -> str:
    if dt <= dev_end:
        return "development"
    if dt <= validation_end:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--near-pairs", default="outputs/task3/near_duplicate_pairs.csv")
    parser.add_argument("--same-event-pairs", default="outputs/task3/same_event_pairs.csv")
    parser.add_argument("--documents", default="outputs/task3/normalized/normalized_documents.jsonl")
    parser.add_argument("--mentions", default="outputs/task3/normalized/normalized_event_mentions.jsonl")
    parser.add_argument("--output-dir", default="outputs/task3/splits")
    parser.add_argument("--report", default="reports/task3_time_split_report.json")
    parser.add_argument("--allow-provisional-cluster-edges", action="store_true")
    args = parser.parse_args()
    dev_end = parse_required_date(args.dev_end, "--dev-end")
    validation_end = parse_required_date(args.validation_end, "--validation-end")
    if validation_end <= dev_end:
        raise ValueError("--validation-end 必须晚于 --dev-end")

    documents = read_jsonl(args.documents)
    mentions = read_jsonl(args.mentions)
    if not documents:
        raise RuntimeError("normalized_documents.jsonl 为空")
    document_by_id = {str(row.get("document_id")): row for row in documents if str(row.get("document_id") or "").strip()}
    article_ids = set(document_by_id)
    mention_to_document = {str(row.get("event_mention_id")): str(row.get("document_id")) for row in mentions if row.get("event_mention_id") and row.get("document_id")}

    near_edges, ignored_near = load_edge_inventory(resolve_path(args.near_pairs), article_ids)
    same_edges, ignored_same = load_human_same_event_edges(resolve_path(args.same_event_pairs), mention_to_document, article_ids)
    provisional_edges = load_provisional_edges(mentions, args.allow_provisional_cluster_edges)

    find, union = union_find(sorted(article_ids))
    edge_inventory = []
    for left, right, metadata in near_edges + same_edges + provisional_edges:
        if left != right:
            union(left, right)
            edge_inventory.append({"left": left, "right": right, **metadata})

    components: dict[str, list[str]] = defaultdict(list)
    for document_id in sorted(article_ids):
        components[find(document_id)].append(document_id)

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending_rows: list[dict[str, Any]] = []
    boundary_groups = 0
    for component_id, member_ids in sorted(components.items()):
        parsed = [(document_id, parse_with_timezone(document_by_id[document_id].get("publish_time"))) for document_id in member_ids]
        missing = [document_id for document_id, publish_time in parsed if publish_time is None]
        if missing:
            for document_id in member_ids:
                row = dict(document_by_id[document_id])
                row.update({"temporary_split": "pending", "split_group_id": component_id, "split_group_basis": "document_id_plus_verified_edges", "pending_reason": "component_contains_missing_or_unparseable_publish_time", "human_cluster_id_available": False})
                pending_rows.append(row)
            continue
        raw_splits = {assign(publish_time, dev_end, validation_end) for _, publish_time in parsed if publish_time}
        if len(raw_splits) > 1:
            boundary_groups += 1
        chosen = assign(max(publish_time for _, publish_time in parsed if publish_time), dev_end, validation_end)
        for document_id in member_ids:
            row = dict(document_by_id[document_id])
            row.update({"temporary_split": chosen, "split_group_id": component_id, "split_group_basis": "document_id_plus_verified_edges", "human_cluster_id_available": False})
            split_rows[chosen].append(row)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("development", "validation", "test"):
        write_jsonl(output_dir / f"{split_name}.jsonl", sorted(split_rows[split_name], key=lambda row: (row.get("publish_time") or "", row["document_id"])))
    write_jsonl(output_dir / "pending.jsonl", sorted(pending_rows, key=lambda row: row["document_id"]))

    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for split, rows in split_rows.items():
        groups_by_split[split] = {row["split_group_id"] for row in rows}
    overlaps = {}
    split_names = ("development", "validation", "test")
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = sorted(groups_by_split[left] & groups_by_split[right])
            if overlap:
                overlaps[f"{left}__{right}"] = overlap

    report = {
        "status": "temporary_without_human_cluster_ids",
        "input_document_count": len(documents),
        "input_event_mention_count": len(mentions),
        "human_cluster_id_available": False,
        "split_policy": "publish_time_with_document_and_verified_edge_components",
        "dev_end_inclusive": args.dev_end,
        "validation_end_inclusive": args.validation_end,
        "development_document_count": len(split_rows["development"]),
        "validation_document_count": len(split_rows["validation"]),
        "test_document_count": len(split_rows["test"]),
        "pending_document_count": len(pending_rows),
        "component_count": len(components),
        "edge_counts": {
            "exact_document_edges": 0,
            "human_near_duplicate_edges": sum(1 for edge in edge_inventory if edge["edge_type"] == "human_near_duplicate"),
            "human_same_event_edges": sum(1 for edge in edge_inventory if edge["edge_type"] == "human_same_event"),
            "provisional_cluster_edges": sum(1 for edge in edge_inventory if edge["edge_type"] == "provisional_cluster"),
            "ignored_unlabeled_candidate_edges": ignored_near + ignored_same,
        },
        "edge_sources": sorted({edge["edge_source"] for edge in edge_inventory}),
        "ignored_candidate_edge_counts": {"near_duplicate_unlabeled": ignored_near, "same_event_unlabeled": ignored_same},
        "pending_reason_distribution": {"component_contains_missing_or_unparseable_publish_time": len(pending_rows)},
        "date_boundary_component_count": boundary_groups,
        "split_is_human_truth": False,
        "edge_inventory": edge_inventory,
        "boundary_group_count": boundary_groups,
        "cross_split_group_overlap_count": sum(len(value) for value in overlaps.values()),
        "cross_split_group_overlaps": overlaps,
        "output_dir": str(output_dir.relative_to(ROOT)),
        "note": "仅完整 document_id、人工确认正边和显式允许的 provisional_program_cluster 边参与组件构建；未标注候选边不参与。",
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "edge_inventory"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
