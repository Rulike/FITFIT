#!/usr/bin/env python3
"""Stage-1 audit for task3 KG v2 full data universe."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from task3_common import ROOT, clean_text, read_jsonl

FULLTEXT_SOURCE = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"
TASK1_TRAIN = ROOT / "outputs/final_task1/final_events_train.jsonl"
TASK1_TEST = ROOT / "outputs/final_task1/final_events_test.jsonl"
TASK2_FULLTEXT = ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl"
TASK2_BASE = ROOT / "outputs/event_credibility/event_credibility_scores.jsonl"
TASK3_DOCS = ROOT / "outputs/task3/normalized/normalized_documents.jsonl"
TASK3_MENTIONS = ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl"
TASK3_DOC_SPLITS = ROOT / "outputs/task3/splits/document_split_mapping.jsonl"
TASK3_MENTION_SPLITS = ROOT / "outputs/task3/splits/event_mention_split_mapping.jsonl"
OUT_DIR = ROOT / "outputs/task3_kg_v2"
REGISTRY_PATH = OUT_DIR / "document_registry.parquet"
UNRESOLVED_QUEUE = OUT_DIR / "document_registry_unresolved_queue.jsonl"
CONFLICT_QUEUE = OUT_DIR / "document_registry_conflict_queue.jsonl"
REPORT_PATH = ROOT / "reports/task3_kg_v2_full_data_audit.json"
DOC_PATH = ROOT / "docs/task3_kg_v2_full_data_audit.md"

NAV_RE = re.compile(r"(当前位置|面包屑|登录|注册|版权所有|免责声明|返回首页|app下载|扫码|分享到|网站地图|相关阅读|广告)")
HTML_RE = re.compile(r"<[^>]+>")
BAD_ENCODING_RE = re.compile(r"(?:�|锟斤拷|\ufffd)")
TITLE_ONLY_INPUTS = [
    TASK3_DOCS,
    TASK3_MENTIONS,
    TASK3_DOC_SPLITS,
    TASK3_MENTION_SPLITS,
    ROOT / "outputs/task3_fulltext_v2/normalized_documents.jsonl",
    ROOT / "outputs/task3_fulltext_v2/normalized_event_mentions.jsonl",
    ROOT / "outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl",
    ROOT / "outputs/task3_fulltext_v2/content_ambiguity_queue.jsonl",
    TASK1_TRAIN,
    TASK1_TEST,
    TASK2_FULLTEXT,
    TASK2_BASE,
    FULLTEXT_SOURCE,
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    text = re.sub(r"<\s*(br|/p|/div|/li|/tr|/h\d)\b[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return clean_text(text)


def split_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def first_item(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return None


def collect_task1_rows() -> list[dict[str, Any]]:
    rows = []
    for path in (TASK1_TRAIN, TASK1_TEST):
        split = "train" if path.name.endswith("train.jsonl") else "test"
        for idx, row in enumerate(read_jsonl(path)):
            item = dict(row)
            item["task1_split"] = split
            item["task1_source_file"] = str(path.relative_to(ROOT))
            item["task1_row_index"] = idx
            rows.append(item)
    return rows


def build_document_registry() -> dict[str, Any]:
    docs = read_jsonl(TASK3_DOCS)
    mentions = read_jsonl(TASK3_MENTIONS)
    doc_splits = {r["document_id"]: r["split"] for r in read_jsonl(TASK3_DOC_SPLITS)}
    mention_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in mentions:
        mention_by_doc[m["document_id"]].append(m)

    fulltext_df = pd.read_parquet(FULLTEXT_SOURCE)
    fulltext_df = fulltext_df[["article_id", "content"]].fillna("")
    fulltext_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(fulltext_df.to_dict(orient="records")):
        aid = clean_text(row.get("article_id"))
        raw_content = "" if row.get("content") is None else str(row.get("content"))
        clean_content = normalize_content(raw_content)
        fulltext_by_article[aid].append({
            "article_id": aid,
            "raw_content": raw_content,
            "clean_content": clean_content,
            "raw_content_hash": sha256_text(raw_content),
            "clean_content_hash": sha256_text(clean_content),
            "row_index": idx,
        })

    task1_rows = collect_task1_rows()
    task1_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task1_rows:
        task1_by_article[clean_text(row.get("article_id"))].append(row)

    registry_rows: list[dict[str, Any]] = []
    unresolved_queue: list[dict[str, Any]] = []
    conflict_queue: list[dict[str, Any]] = []
    source_doc_hashes = Counter()
    doc_quality = Counter()
    split_counts = Counter()
    doc_mentions_missing = 0
    doc_content_empty = 0
    doc_title_only = 0
    duplicate_clean_hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    article_id_to_sources: dict[str, set[str]] = defaultdict(set)
    article_id_to_task1_events: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        article_id = clean_text(doc.get("article_id"))
        split = doc_splits.get(doc.get("document_id"), "unknown")
        split_counts[split] += 1
        fulltext_rows = fulltext_by_article.get(article_id, [])
        fulltext_row = fulltext_rows[0] if fulltext_rows else None
        task1_rows_for_article = task1_by_article.get(article_id, [])
        if task1_rows_for_article:
            article_id_to_task1_events[article_id].extend(clean_text(r.get("event_id")) for r in task1_rows_for_article if clean_text(r.get("event_id")))
        source_rows = doc.get("source_rows") or []
        source_row = first_item(source_rows) or {}
        raw_source_file = clean_text(source_row.get("path")) or (task1_rows_for_article[0]["task1_source_file"] if task1_rows_for_article else "")
        raw_row_index = source_row.get("line_number")
        content = fulltext_row["clean_content"] if fulltext_row else ""
        raw_content = fulltext_row["raw_content"] if fulltext_row else ""
        raw_hash = fulltext_row["raw_content_hash"] if fulltext_row else ""
        clean_hash = fulltext_row["clean_content_hash"] if fulltext_row else ""
        content_length = len(content)
        content_availability = "fulltext" if content else "empty_content"
        provenance = {
            "document_source": {
                "source_file": raw_source_file,
                "row_index": raw_row_index,
                "article_id": article_id,
                "document_id": doc.get("document_id"),
                "task1_event_ids": article_id_to_task1_events.get(article_id, []),
            },
            "fulltext_source": {
                "source_file": str(FULLTEXT_SOURCE.relative_to(ROOT)),
                "join_key": "article_id",
                "join_confidence": "exact",
                "raw_content_hash": raw_hash,
                "clean_content_hash": clean_hash,
            },
        }
        if not fulltext_row:
            unresolved_queue.append({
                "document_id": doc.get("document_id"),
                "article_id": article_id,
                "issue": "unmatched_fulltext_source",
                "raw_source_file": raw_source_file,
                "raw_row_index": raw_row_index,
                "split": split,
            })
        elif not content:
            unresolved_queue.append({
                "document_id": doc.get("document_id"),
                "article_id": article_id,
                "issue": "empty_content_after_minimal_cleaning",
                "raw_source_file": raw_source_file,
                "raw_row_index": raw_row_index,
                "split": split,
                "raw_content_hash": raw_hash,
                "clean_content_hash": clean_hash,
            })
            doc_content_empty += 1
            content_availability = "title_structured_only"
        else:
            source_doc_hashes[clean_hash] += 1
        if content and content == clean_text(doc.get("title")):
            doc_title_only += 1
            unresolved_queue.append({
                "document_id": doc.get("document_id"),
                "article_id": article_id,
                "issue": "title_equals_content",
                "raw_source_file": raw_source_file,
                "raw_row_index": raw_row_index,
                "split": split,
            })
            content_availability = "title_structured_only"
        if not task1_rows_for_article:
            unresolved_queue.append({
                "document_id": doc.get("document_id"),
                "article_id": article_id,
                "issue": "missing_task1_cover",
                "raw_source_file": raw_source_file,
                "raw_row_index": raw_row_index,
                "split": split,
            })
        if len(task1_rows_for_article) == 0:
            doc_mentions_missing += 1
        mention_ids = [m.get("event_mention_id") for m in mention_by_doc.get(doc.get("document_id"), []) if clean_text(m.get("event_mention_id"))]
        if not mention_ids:
            doc_mentions_missing += 1
        row = {
            "document_id": doc.get("document_id"),
            "article_id": article_id,
            "title": doc.get("title", ""),
            "content": content,
            "source_name": doc.get("source_name", ""),
            "publish_time": doc.get("publish_time", ""),
            "split": split,
            "raw_source_file": raw_source_file,
            "raw_row_index": int(raw_row_index) if raw_row_index not in [None, ""] else None,
            "raw_content_hash": raw_hash,
            "clean_content_hash": clean_hash,
            "content_length": content_length,
            "content_availability": content_availability,
            "field_provenance": json.dumps(provenance, ensure_ascii=False, default=str),
            "task1_event_ids": "|".join(article_id_to_task1_events.get(article_id, [])),
            "task1_event_count": len(article_id_to_task1_events.get(article_id, [])),
            "event_mention_count": len(mention_by_doc.get(doc.get("document_id"), [])),
            "content_quality": "fulltext_high_quality" if content and content_length >= 300 and not NAV_RE.search(content) else (
                "fulltext_contaminated" if content and (NAV_RE.search(content) or HTML_RE.search(content) or BAD_ENCODING_RE.search(content)) else (
                    "fulltext_short" if content and content_length < 80 else ("title_structured_only" if not content else "fulltext_usable")
                )
            ),
        }
        registry_rows.append(row)
        if clean_hash:
            duplicate_clean_hash_groups[clean_hash].append(row)
        if article_id:
            article_id_to_sources[article_id].add(raw_source_file)
    # final pass for conflict queue on duplicate fulltext rows or split leakage
    duplicate_clean_hash_docs = [group for group in duplicate_clean_hash_groups.values() if len(group) > 1]
    for group in duplicate_clean_hash_docs:
        splits = {r["split"] for r in group}
        article_ids = {r["article_id"] for r in group}
        if len(article_ids) > 1:
            conflict_queue.append({
                "issue": "duplicate_clean_content_hash",
                "clean_content_hash": group[0]["clean_content_hash"],
                "group_size": len(group),
                "article_ids": list(article_ids)[:50],
                "splits": sorted(splits),
                "raw_source_files": sorted({r["raw_source_file"] for r in group}),
            })
    # build task2 audit context
    task2_rows = read_jsonl(TASK2_FULLTEXT)
    task2_base_rows = read_jsonl(TASK2_BASE)
    task2_scope_distribution = Counter("cluster" if clean_text(r.get("cluster_id")) else "none" for r in task2_rows)
    task2_split_distribution = Counter(clean_text(r.get("split")) for r in task2_rows)
    task2_verification_distribution = Counter(clean_text(r.get("verification_status")) for r in task2_rows)
    task2_fulltext_verification_distribution = Counter(clean_text(r.get("fulltext_verification_status")) for r in task2_rows)
    task2_score_scope_distribution = Counter(clean_text(r.get("trust_score_scope")) for r in read_jsonl(TASK3_MENTIONS))
    # task1 coverage and event mention counts
    task1_rows = collect_task1_rows()
    task1_unique_docs = len({clean_text(r.get("article_id")) for r in task1_rows if clean_text(r.get("article_id"))})
    task1_events = len(task1_rows)
    task3_mentions = read_jsonl(TASK3_MENTIONS)
    task3_mention_splits = Counter(clean_text(r.get("split")) for r in read_jsonl(TASK3_MENTION_SPLITS))
    task3_mention_with_cluster = sum(1 for r in task3_mentions if clean_text(r.get("provisional_cluster_id")))
    task3_mention_with_task2 = sum(1 for r in task3_mentions if r.get("task2_cluster_ids"))
    task3_mention_has_trust = sum(1 for r in task3_mentions if r.get("trust_score") not in [None, ""])
    # content leakage / missingness stats on mentions
    mention_missing_rates = {}
    for field in ["target_type", "target_code", "provisional_event_type", "event_type", "core_elements", "event_time", "publish_time"]:
        missing = 0
        for row in task3_mentions:
            value = row.get(field)
            if field == "event_time":
                missing += 1
            elif field == "core_elements":
                missing += 1 if not value else 0
            elif field == "event_type":
                missing += 1 if value in [None, ""] else 0
            else:
                missing += 1 if not clean_text(value) else 0
        mention_missing_rates[field] = round(missing / len(task3_mentions), 6)
    # article_id / clean content / split leakage
    article_id_split_map: dict[str, set[str]] = defaultdict(set)
    clean_hash_split_map: dict[str, set[str]] = defaultdict(set)
    component_split_map: dict[str, set[str]] = defaultdict(set)
    component_by_article: dict[str, str] = {}
    for doc in registry_rows:
        article_id_split_map[doc["article_id"]].add(doc["split"])
        if doc["clean_content_hash"]:
            clean_hash_split_map[doc["clean_content_hash"]].add(doc["split"])
    for m in task3_mentions:
        doc_id = m.get("document_id")
        key = clean_text(m.get("target_type")) + "|" + clean_text(m.get("target_code")) + "|" + clean_text(m.get("provisional_event_type"))
        split = clean_text(next((d["split"] for d in registry_rows if d["document_id"] == doc_id), ""))
        if key.strip("|"):
            component_split_map[key].add(split)
        if clean_text(m.get("article_id")):
            component_by_article[clean_text(m.get("article_id"))] = key
    cross_split_article_ids = [aid for aid, splits in article_id_split_map.items() if len(splits) > 1]
    cross_split_clean_hashes = [h for h, splits in clean_hash_split_map.items() if len(splits) > 1]
    cross_split_components = [k for k, splits in component_split_map.items() if len(splits) > 1 and k.strip("|")]
    # document coverage and no-event docs
    docs_with_mentions = {m.get("document_id") for m in task3_mentions if clean_text(m.get("document_id"))}
    no_event_docs = [d["document_id"] for d in registry_rows if d["document_id"] not in docs_with_mentions]
    # duplicate clean hash stats
    duplicate_hash_groups = [group for group in duplicate_clean_hash_groups.values() if len(group) > 1]
    same_source_duplicate_groups = 0
    cross_source_duplicate_groups = 0
    max_group_size = 0
    for group in duplicate_hash_groups:
        max_group_size = max(max_group_size, len(group))
        sources = {clean_text(r["source_name"]) for r in group}
        if len(sources) <= 1:
            same_source_duplicate_groups += 1
        else:
            cross_source_duplicate_groups += 1
    source_hash_counts = Counter(r["clean_content_hash"] for r in registry_rows if r["clean_content_hash"])
    registry_quality_distribution = Counter(r["content_quality"] for r in registry_rows)
    registry_df = pd.DataFrame(registry_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry_df.to_parquet(REGISTRY_PATH, index=False)
    write_json = json.dumps
    from task3_common import write_jsonl
    write_jsonl(UNRESOLVED_QUEUE, unresolved_queue)
    write_jsonl(CONFLICT_QUEUE, conflict_queue)
    report = {
        "version": "task3_kg_v2_stage1",
        "document_universe": {
            "complete_document_count": len(registry_rows),
            "unique_document_id_count": len({r["document_id"] for r in registry_rows}),
            "unique_article_id_count": len({r["article_id"] for r in registry_rows}),
            "article_id_unique": len({r["article_id"] for r in registry_rows}) == len(registry_rows),
            "title_only_input_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in TITLE_ONLY_INPUTS if p.exists()},
        },
        "split_distribution": dict(split_counts),
        "content_coverage": {
            "nonempty_content_count": sum(1 for r in registry_rows if r["content"]),
            "nonempty_content_rate": round(sum(1 for r in registry_rows if r["content"]) / len(registry_rows), 6),
            "no_event_document_count": len(no_event_docs),
            "no_event_document_sample": no_event_docs[:20],
        },
        "task1_coverage": {
            "task1_document_count": task1_unique_docs,
            "task1_event_mention_count": task1_events,
            "task1_covering_document_count": len({clean_text(r.get("article_id")) for r in task1_rows if clean_text(r.get("article_id")) in {d["article_id"] for d in registry_rows}}),
        },
        "task2_coverage": {
            "task2_cluster_count": len(task2_rows),
            "task2_cluster_split_distribution": dict(task2_split_distribution),
            "task2_verification_status_distribution": dict(task2_verification_distribution),
            "task2_fulltext_verification_status_distribution": dict(task2_fulltext_verification_distribution),
            "task2_score_scope_distribution": dict(task2_score_scope_distribution),
            "task2_covered_event_mention_count": task3_mention_with_task2,
            "task2_covered_event_mention_rate": round(task3_mention_with_task2 / len(task3_mentions), 6),
            "task2_trust_score_scope_distribution": dict(task2_scope_distribution),
            "task2_mentions_with_trust_score_count": task3_mention_has_trust,
        },
        "registry_quality": {
            "quality_distribution": dict(registry_quality_distribution),
            "title_structured_only_count": sum(1 for r in registry_rows if r["content_availability"] == "title_structured_only"),
            "content_length": {
                "min": int(registry_df[registry_df["content"].astype(str) != ""]["content_length"].min()) if (registry_df["content"].astype(str) != "").any() else 0,
                "median": float(median([r["content_length"] for r in registry_rows if r["content"]])) if any(r["content"] for r in registry_rows) else 0,
                "p95": int(sorted([r["content_length"] for r in registry_rows if r["content"]])[max(0, round((sum(1 for r in registry_rows if r["content"]) - 1) * 0.95))]) if any(r["content"] for r in registry_rows) else 0,
                "max": max([r["content_length"] for r in registry_rows if r["content"]], default=0),
            },
            "raw_content_hash_unique_count": len({r["raw_content_hash"] for r in registry_rows if r["raw_content_hash"]}),
            "clean_content_hash_unique_count": len({r["clean_content_hash"] for r in registry_rows if r["clean_content_hash"]}),
            "duplicate_clean_content_hash_document_count": sum(len(g) for g in duplicate_hash_groups),
            "duplicate_clean_content_hash_group_count": len(duplicate_hash_groups),
            "duplicate_clean_content_hash_max_group_size": max_group_size,
            "same_source_duplicate_group_count": same_source_duplicate_groups,
            "cross_source_duplicate_group_count": cross_source_duplicate_groups,
        },
        "coverage_audit": {
            "task1_document_coverage_count": task1_unique_docs,
            "task1_event_mention_count": task1_events,
            "task2_event_mention_coverage_count": task3_mention_with_task2,
            "task2_event_mention_coverage_rate": round(task3_mention_with_task2 / len(task3_mentions), 6),
            "unresolved_connection_count": len(unresolved_queue),
            "conflict_count": len(conflict_queue),
            "unmatched_article_ids": [r["article_id"] for r in unresolved_queue if r.get("issue") == "unmatched_fulltext_source"],
            "empty_content_article_ids": [r["article_id"] for r in unresolved_queue if r.get("issue") == "empty_content_after_minimal_cleaning"],
            "title_only_article_ids": [r["article_id"] for r in unresolved_queue if r.get("issue") == "title_equals_content"],
        },
        "missing_rates": mention_missing_rates,
        "leakage_audit": {
            "cross_split_same_article_id_count": len(cross_split_article_ids),
            "cross_split_same_clean_content_hash_count": len(cross_split_clean_hashes),
            "cross_split_same_event_component_count": len(cross_split_components),
            "cross_split_same_article_id_sample": cross_split_article_ids[:20],
            "cross_split_same_clean_content_hash_sample": cross_split_clean_hashes[:20],
            "cross_split_same_event_component_sample": cross_split_components[:20],
        },
        "upstream_scope": {
            "task1_output_rows": len(task1_rows),
            "task2_output_rows": len(task2_rows),
            "task3_document_rows": len(registry_rows),
            "task3_event_mention_rows": len(task3_mentions),
        },
        "provenance": {
            "registry_columns": list(registry_df.columns),
            "raw_source_file_column": True,
            "raw_row_index_column": True,
            "field_provenance_column": True,
        },
        "limitations": [
            "本阶段只做数据宇宙审计与Document注册，不运行Embedding、聚类、GNN或大模型推理。",
            "2条fulltext清洗后为空，因此保留在注册表并进入unresolved队列。",
            "task1事件数与task3 EventMention数一致，但task2仍以cluster级评分为主，需要通过cluster映射折算到EventMention。",
        ],
        "protected_input_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in TITLE_ONLY_INPUTS if p.exists()},
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        f"""# Task3 KG v2 第一阶段全量数据宇宙审计

- 完整Document：{len(registry_rows)}
- 任务1覆盖Document：{task1_unique_docs}
- 任务1 EventMention：{task1_events}
- 任务2覆盖EventMention：{task3_mention_with_task2}
- 无事件Document：{len(no_event_docs)}
- 无法连接记录：{len(unresolved_queue)}

## 注册表输出

`{REGISTRY_PATH.relative_to(ROOT)}`

## 独立队列

- `{UNRESOLVED_QUEUE.relative_to(ROOT)}`
- `{CONFLICT_QUEUE.relative_to(ROOT)}`

## 限制

- 不运行新的Embedding/聚类/GNN/大模型推理。
- 不修改任务1、任务2和旧任务3正式输出。
- 同正文多article_id保留，只计数不删除。
""",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build_document_registry(), ensure_ascii=False, indent=2))
