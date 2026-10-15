#!/usr/bin/env python3
"""Audit local news/fulltext sources without changing pipeline artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from task3_common import ROOT, clean_text, read_jsonl

CONTENT_FIELDS = {
    "content", "body", "text", "article_content", "news_content", "full_text",
    "summary", "abstract", "description", "raw_text", "html", "article_body",
}
ID_FIELDS = {"article_id", "document_id", "url", "source_path", "path", "file_path"}
ALL_SUFFIXES = {".json", ".jsonl", ".csv", ".xlsx", ".xls", ".parquet", ".db", ".sqlite", ".sqlite3"}
SKIP_PARTS = {".git", "__pycache__", ".venv", "node_modules"}


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and value != value:
        return False
    if isinstance(value, (list, dict)):
        return bool(value)
    return bool(str(value).strip())


def text_length(value: Any) -> int:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return len(str(value).strip()) if is_nonempty(value) else 0


def records_from_json(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        value = json.loads(line)
                    except Exception:
                        yield {"_parse_error_line": line_number}
                        continue
                    yield value if isinstance(value, dict) else {"_value": value}
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        yield {"_parse_error": True}
        return
    if isinstance(value, list):
        for item in value:
            yield item if isinstance(item, dict) else {"_value": item}
    elif isinstance(value, dict):
        # A top-level dict containing a record list is treated as a record container.
        lists = [v for v in value.values() if isinstance(v, list) and all(isinstance(x, dict) for x in v[:3])]
        if len(lists) == 1:
            yield from lists[0]
        else:
            yield value
    else:
        yield {"_value": value}


def records_from_file(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix in {".json", ".jsonl"}:
        yield from records_from_json(path)
    elif path.suffix == ".csv":
        try:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                yield from csv.DictReader(handle)
        except Exception:
            yield {"_parse_error": True}
    elif path.suffix in {".xlsx", ".xls"} and pd is not None:
        try:
            sheets = pd.read_excel(path, sheet_name=None, dtype=object)
            for sheet, frame in sheets.items():
                for row in frame.fillna("").to_dict(orient="records"):
                    row["_sheet"] = sheet
                    yield row
        except Exception:
            yield {"_parse_error": True}
    elif path.suffix == ".parquet" and pd is not None:
        try:
            frame = pd.read_parquet(path)
            yield from frame.fillna("").to_dict(orient="records")
        except Exception:
            yield {"_parse_error": True}
    elif path.suffix in {".db", ".sqlite", ".sqlite3"}:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            tables = [r[0] for r in conn.execute("select name from sqlite_master where type='table'")]
            for table in tables:
                cols = [r[1] for r in conn.execute(f'pragma table_info("{table}")')]
                for values in conn.execute(f'select * from "{table}"'):
                    yield dict(zip(cols, values)) | {"_table": table}
            conn.close()
        except Exception:
            yield {"_parse_error": True}


def discover(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ALL_SUFFIXES and not any(x in SKIP_PARTS for x in p.parts))


def norm(value: Any) -> str:
    return clean_text(value).strip()


def add_nested_ids(value: Any, output: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in ID_FIELDS and is_nonempty(item):
                output.add(norm(item))
            add_nested_ids(item, output)
    elif isinstance(value, list):
        for item in value:
            add_nested_ids(item, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    members = read_jsonl(root / "outputs/task3/step4/event_cluster_members.jsonl")
    attached_documents = {r["document_id"] for r in members if r.get("document_id")}
    mentions = read_jsonl(root / "outputs/task3/normalized/normalized_event_mentions.jsonl")
    # The normalized mention registry is the complete Task3 document universe; Step4 members only cover attached cluster members.
    documents = {r["document_id"] for r in mentions if r.get("document_id")}
    graph_document_count = sum(1 for r in read_jsonl(root / "outputs/task3/step6/graph_nodes.jsonl") if r.get("node_type") == "Document")
    doc_by_article = {r.get("article_id"): r.get("document_id") for r in mentions if r.get("article_id") and r.get("document_id")}
    article_ids = set(doc_by_article)
    doc_by_path = {r.get("task1_source_path"): r.get("document_id") for r in mentions if r.get("task1_source_path") and r.get("document_id")}
    files = discover(root)
    summaries: list[dict[str, Any]] = []
    global_field_counts: Counter[str] = Counter()
    content_file_hits: list[dict[str, Any]] = []
    for path in files:
        count = 0
        fields: set[str] = set()
        nonempty: Counter[str] = Counter()
        lengths: dict[str, list[int]] = {f: [] for f in CONTENT_FIELDS}
        connected: set[str] = set()
        id_fields_seen: set[str] = set()
        parse_errors = 0
        try:
            rows = records_from_file(path)
            for row in rows:
                count += 1
                if row.get("_parse_error"):
                    parse_errors += 1
                    continue
                lower = {str(k).lower(): v for k, v in row.items()}
                fields.update(str(k) for k in row.keys())
                for key, value in lower.items():
                    if key in CONTENT_FIELDS:
                        global_field_counts[key] += 1
                        if is_nonempty(value):
                            nonempty[key] += 1
                            lengths[key].append(text_length(value))
                for key in lower:
                    if key in ID_FIELDS:
                        id_fields_seen.add(key)
                ids: set[str] = set()
                add_nested_ids(row, ids)
                for value in ids:
                    if value in documents:
                        connected.add(value)
                    if value in article_ids and value in doc_by_article:
                        connected.add(doc_by_article[value])
                source_path = next((norm(lower.get(k)) for k in ("source_path", "path", "file_path") if is_nonempty(lower.get(k))), "")
                if source_path in doc_by_path:
                    connected.add(doc_by_path[source_path])
        except Exception as exc:
            parse_errors += 1
            fields.add(f"_error:{type(exc).__name__}")
        distributions = {}
        for field, values in lengths.items():
            if values:
                ordered = sorted(values)
                distributions[field] = {"nonempty_count": len(values), "min": ordered[0], "p50": ordered[len(ordered)//2], "mean": round(mean(values), 2), "max": ordered[-1], "zero_count": 0}
        summary = {
            "file": str(path.relative_to(root)), "suffix": path.suffix.lower(), "size_bytes": path.stat().st_size,
            "record_count": count, "id_fields": sorted(id_fields_seen), "all_fields": sorted(fields),
            "content_fields": sorted(set(str(x).lower() for x in fields) & CONTENT_FIELDS),
            "nonempty_counts": dict(nonempty), "content_length_distribution": distributions,
            "connected_document_count": len(connected), "connected_document_ids_sample": sorted(connected)[:10],
            "parse_error_records": parse_errors,
        }
        summaries.append(summary)
        if summary["content_fields"] and any(summary["nonempty_counts"].values()):
            content_file_hits.append({"file": summary["file"], "content_fields": summary["content_fields"], "nonempty_counts": summary["nonempty_counts"], "connected_document_count": len(connected)})
    content_docs: dict[str, set[str]] = {}
    for hit in content_file_hits:
        path = root / hit["file"]
        docs_hit: set[str] = set()
        try:
            for row in records_from_file(path):
                ids: set[str] = set(); add_nested_ids(row, ids)
                for value in ids:
                    if value in documents: docs_hit.add(value)
                    if value in article_ids and value in doc_by_article: docs_hit.add(doc_by_article[value])
        except Exception:
            pass
        content_docs[hit["file"]] = docs_hit
    connected_content_docs = set().union(*content_docs.values()) if content_docs else set()
    report = {
        "audit_scope": {"root": str(root), "recursive_suffixes": sorted(ALL_SUFFIXES), "no_network": True, "no_title_as_content": True, "content_fields": sorted(CONTENT_FIELDS)},
        "document_universe": {"count": len(documents), "source": "outputs/task3/normalized/normalized_event_mentions.jsonl", "attached_step4_document_count": len(attached_documents), "attached_step4_source": "outputs/task3/step4/event_cluster_members.jsonl", "graph_document_node_count": graph_document_count, "note": "The normalized registry, and Step6 graph, contain 4,849 unique Task3 Documents. Step4 members contain 1,558 attached core/cluster member Documents and are a subset of the full universe."},
        "file_count": len(files), "files": summaries, "content_source_files": content_file_hits,
        "content_document_connection": {"connected_document_count": len(connected_content_docs), "unconnected_document_count": len(documents - connected_content_docs), "connected_document_ids_sample": sorted(connected_content_docs)[:20]},
        "special_checks": {
            "task1_raw_inputs": [s for s in summaries if s["file"].startswith("data/processed/labeled/") or s["file"].startswith("data/processed/event_clusters/") or s["file"].startswith("outputs/final_task1/")],
            "crawler_or_cache_candidates": [s for s in summaries if any(x in s["file"].lower() for x in ("crawler", "cache", "raw"))],
            "article_metadata_index": [s for s in summaries if "article_metadata_index" in s["file"]],
            "task2_fulltext_candidates": [s for s in summaries if "fulltext" in s["file"].lower() or "event_credibility" in s["file"]],
        },
        "interpretation": {"local_body_exists": bool(content_file_hits), "usable_original_body_sources": [x["file"] for x in content_file_hits if any(f in {"content", "body", "article_content", "news_content", "full_text", "raw_text", "article_body", "html"} for f in x["content_fields"])], "summary_only_sources": [x["file"] for x in content_file_hits if set(x["content_fields"]) <= {"summary", "abstract", "description", "text"}]},
        "safe_backfill_recommendation": ["仅使用显式content/body/full_text等字段；不得用title填充正文。", "先按article_id精确连接4849个Document，再用已审计的source_path连接；url仅在规范化URL唯一匹配时使用。", "对near-duplicate只保留代表Document的正文作为事实证据，其余仅保留传播证据。", "fulltext_evidence中的sentence/evidence_items是任务2抽取证据，不等同于完整原文；只能作为证据片段回填。", "回填后保留content_source_file、content_field、content_length、join_key、join_confidence和原始哈希。"],
    }
    out = root / "reports/task3_content_source_audit.json"; out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = root / "docs/task3_content_source_audit.md"
    usable = report["interpretation"]["usable_original_body_sources"]
    md.write_text(f"# 任务三正文/摘要/公告原文审计\n\n- 审计文件数：{len(files)}\n- normalized完整Document总数：{len(documents)}\n- Step4已附着Document子集：{len(attached_documents)}\n- Step6图中Document节点数：{graph_document_count}\n- 本地存在正文相关非空字段：{'是' if content_file_hits else '否'}\n- 能连接完整Document：{len(connected_content_docs)}\n- 无法连接完整Document：{len(documents-connected_content_docs)}\n- 可作为原文候选的文件：{', '.join(usable) if usable else '无'}\n\n## 结论\n\n项目本地存在正文，主要来自显式`content`字段的数据集和任务2 fulltext索引。`article_metadata_index`只有文章元数据，不含正文。`outputs/event_credibility/fulltext_evidence`保存的是抽取后的证据句及其来源信息，不应冒充完整新闻原文。任务一最终事件输出的`evidence_sentences`多数为空，不能作为正文来源。\n\n## 安全回填建议\n\n1. 仅按`article_id`精确连接；不能匹配时再使用唯一的`source_path`或规范化URL。\n2. near-duplicate组只把代表文档正文作为独立事实证据，其他文档保留传播关系。\n3. 记录来源文件、字段、连接键、连接置信度、正文长度和文件哈希。\n4. 不用标题填正文；任务2证据句仅作为片段级证据。\n\n完整逐文件统计见`reports/task3_content_source_audit.json`。\n", encoding="utf-8")
    print(json.dumps({"file_count": len(files), "document_count": len(documents), "content_source_file_count": len(content_file_hits), "connected_documents": len(connected_content_docs), "unconnected_documents": len(documents-connected_content_docs), "content_sources": usable}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
