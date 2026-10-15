#!/usr/bin/env python3
"""Build task3 fulltext_v2 artifacts with exact article_id joins only."""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from task3_common import ROOT, clean_text, read_jsonl, write_jsonl

SOURCE_REL = "data/processed/event_credibility/fulltext_article_index.parquet"
SOURCE_PATH = ROOT / SOURCE_REL
INPUT_DOCUMENTS = ROOT / "outputs/task3/normalized/normalized_documents.jsonl"
INPUT_MENTIONS = ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl"
INPUT_DOC_SPLITS = ROOT / "outputs/task3/splits/document_split_mapping.jsonl"
INPUT_MENTION_SPLITS = ROOT / "outputs/task3/splits/event_mention_split_mapping.jsonl"
OUTPUT_DIR = ROOT / "outputs/task3_fulltext_v2"
REPORT_PATH = ROOT / "reports/task3_fulltext_v2_report.json"
DOC_PATH = ROOT / "docs/task3_fulltext_v2.md"

CONTENT_FIELD = "content"
TOP_K = 5
SHORT_CONTENT_THRESHOLD = 80
NAVIGATION_PATTERNS = re.compile(r"(当前位置|面包屑|登录|注册|版权所有|免责声明|返回首页|app下载|扫码|分享到|网站地图|相关阅读)")
HTML_TAG_RE = re.compile(r"<\s*/?\s*(?:html|body|p|div|span|br|table|tr|td|script|style|a|img|section|article|ul|li|strong|em|h\d)\b[^>]*>", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
BAD_ENCODING_RE = re.compile(r"(?:�|锟斤拷|\ufffd)")


def sha256_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    text = re.sub(r"<\s*(br|/p|/div|/li|/tr|/h\d)\b[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[\u3000\xa0]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    meaningful = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text)
    if not meaningful:
        return 0.0
    return len(CHINESE_RE.findall(text)) / len(meaningful)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []
    start = 0
    for part in SENTENCE_SPLIT_RE.split(text):
        sent = clean_text(part)
        if not sent:
            start += len(part)
            continue
        idx = text.find(part, start)
        if idx < 0:
            idx = start
        end = idx + len(part)
        result.append((sent, idx, end))
        start = end
    if not result and clean_text(text):
        result.append((clean_text(text), 0, len(text)))
    return result


def keyword_set(mention: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("target_code", "schema_id", "provisional_event_type", "event_type"):
        value = clean_text(mention.get(field))
        if value:
            keys.add(value)
            keys.update(x for x in re.split(r"[_\s|/,-]+", value) if x)
    for field in ("core_elements", "linked_stocks", "linked_industries"):
        value = mention.get(field)
        if isinstance(value, list):
            keys.update(clean_text(x) for x in value if clean_text(x))
    elements = mention.get("elements") or {}
    if isinstance(elements, dict):
        for value in elements.values():
            if isinstance(value, list):
                keys.update(clean_text(x) for x in value if clean_text(x))
            else:
                text = clean_text(value)
                if text:
                    keys.add(text)
                    keys.update(x for x in re.split(r"[_\s|/,-]+", text) if x)
    return {k for k in keys if len(k) >= 2}


def score_sentence(sentence: str, keys: set[str]) -> float:
    score = 0.0
    for key in keys:
        if key and key in sentence:
            score += 1.0 + min(len(key), 8) / 8
    if any(char.isdigit() for char in sentence):
        score += 0.2
    return score


def select_evidence_sentences(content: str, mention: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str, bool]:
    sentences = split_sentences(content)
    keys = keyword_set(mention)
    scored = []
    for sent, start, end in sentences:
        score = score_sentence(sent, keys)
        if score > 0:
            scored.append((score, start, end, sent))
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = []
    for score, start, end, sent in scored[:TOP_K]:
        if len(sent) > 320:
            sent = sent[:320].rstrip() + "…"
        selected.append({"sentence": sent, "start": start, "end": end, "score": round(score, 4)})
    if selected:
        return selected, "keyword_overlap_top_k", "medium", False
    fallback = []
    for sent, start, end in sentences[: min(TOP_K, len(sentences))]:
        if len(sent) > 320:
            sent = sent[:320].rstrip() + "…"
        fallback.append({"sentence": sent, "start": start, "end": end, "score": 0.0})
    return fallback, "no_keyword_overlap_fallback_lead_sentences", "low", True


def build_content_index() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_parquet(SOURCE_PATH)
    if "article_id" not in df.columns or CONTENT_FIELD not in df.columns:
        raise ValueError("fulltext source must contain article_id and content")
    source_hash = file_sha256(SOURCE_PATH)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quality_rows: list[dict[str, Any]] = []
    content_hash_to_article_ids: dict[str, set[str]] = defaultdict(set)
    for row in df[["article_id", CONTENT_FIELD]].fillna("").to_dict(orient="records"):
        article_id = clean_text(row.get("article_id"))
        raw_content = "" if row.get(CONTENT_FIELD) is None else str(row.get(CONTENT_FIELD))
        clean_content = normalize_content(raw_content)
        raw_hash = sha256_text(raw_content)
        clean_hash = sha256_text(clean_content)
        entry = {
            "article_id": article_id,
            "content": clean_content,
            "raw_content_hash": raw_hash,
            "clean_content_hash": clean_hash,
            "content_length": len(clean_content),
            "content_source_file": SOURCE_REL,
            "content_source_hash": source_hash,
            "content_field": CONTENT_FIELD,
        }
        if article_id:
            grouped[article_id].append(entry)
            if clean_content:
                content_hash_to_article_ids[clean_hash].add(article_id)
        quality_rows.append({
            "article_id": article_id,
            "raw_content_hash": raw_hash,
            "clean_content_hash": clean_hash,
            "length": len(clean_content),
            "nonempty": bool(clean_content),
            "html_pollution": bool(HTML_TAG_RE.search(raw_content) or NAVIGATION_PATTERNS.search(clean_content)),
            "bad_encoding": bool(BAD_ENCODING_RE.search(raw_content) or BAD_ENCODING_RE.search(clean_content)),
            "chinese_ratio": chinese_ratio(clean_content),
        })
    index: dict[str, dict[str, Any]] = {}
    ambiguity_queue: list[dict[str, Any]] = []
    duplicate_article_id_count = 0
    identical_content_duplicate_count = 0
    conflicting_content_duplicate_count = 0
    for article_id, rows in grouped.items():
        if len(rows) > 1:
            duplicate_article_id_count += 1
        hashes = sorted({r["clean_content_hash"] for r in rows})
        if len(rows) > 1 and len(hashes) == 1:
            identical_content_duplicate_count += 1
        elif len(hashes) > 1:
            conflicting_content_duplicate_count += 1
            ambiguity_queue.append({
                "article_id": article_id,
                "issue": "ambiguous_content",
                "record_count": len(rows),
                "clean_content_hashes": hashes,
                "content_lengths": sorted({r["content_length"] for r in rows}),
                "policy": "quarantined_no_arbitrary_selection",
            })
            continue
        # Deterministic de-dup when all hashes match: pick first sorted stable tuple without changing content.
        chosen = sorted(rows, key=lambda r: (r["raw_content_hash"], r["content_length"]))[0]
        index[article_id] = chosen
    multi_article_same_content = sum(1 for ids in content_hash_to_article_ids.values() if len(ids) > 1)
    source_stats = {
        "source_records": len(df),
        "source_article_id_count": len(grouped),
        "duplicate_article_id_count": duplicate_article_id_count,
        "identical_content_duplicate_count": identical_content_duplicate_count,
        "conflicting_content_duplicate_count": conflicting_content_duplicate_count,
        "same_content_multiple_article_id_count": multi_article_same_content,
        "source_hash": source_hash,
    }
    return index, ambiguity_queue, {"quality_rows": quality_rows, "source_stats": source_stats}


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def build() -> dict[str, Any]:
    documents = read_jsonl(INPUT_DOCUMENTS)
    mentions = read_jsonl(INPUT_MENTIONS)
    doc_splits = {r["document_id"]: r["split"] for r in read_jsonl(INPUT_DOC_SPLITS)}
    mention_splits = {r["event_mention_id"]: r["split"] for r in read_jsonl(INPUT_MENTION_SPLITS)}
    title_only_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in [INPUT_DOCUMENTS, INPUT_MENTIONS, INPUT_DOC_SPLITS, INPUT_MENTION_SPLITS]}
    content_index, ambiguity_queue, source_audit = build_content_index()
    ambiguous_article_ids = {r["article_id"] for r in ambiguity_queue}

    out_docs: list[dict[str, Any]] = []
    doc_coverage_by_split: Counter[str] = Counter()
    doc_exact_join_by_split: Counter[str] = Counter()
    unmatched_docs: list[str] = []
    exact_join_docs = 0
    nonempty_join_docs = 0
    for doc in documents:
        row = dict(doc)
        article_id = clean_text(row.get("article_id"))
        split = doc_splits.get(row.get("document_id"), "unknown")
        content = content_index.get(article_id)
        provenance = dict(row.get("field_provenance") or {})
        if content:
            exact_join_docs += 1
            doc_exact_join_by_split[split] += 1
            has_clean_content = bool(clean_text(content["content"]))
            if has_clean_content:
                nonempty_join_docs += 1
                doc_coverage_by_split[split] += 1
            row.update({
                "content": content["content"],
                "content_source_file": content["content_source_file"],
                "content_source_hash": content["content_source_hash"],
                "content_field": content["content_field"],
                "content_length": content["content_length"],
                "raw_content_hash": content["raw_content_hash"],
                "clean_content_hash": content["clean_content_hash"],
                "join_key": "article_id",
                "join_confidence": "exact",
                "content_availability": "fulltext" if has_clean_content else "empty_content",
                "evidence_scope": "fulltext_structured" if has_clean_content else "metadata_only",
            })
            missing = [x for x in row.get("missing_fields", []) if x != "content"] if has_clean_content else list(row.get("missing_fields", []))
            row["missing_fields"] = missing
            provenance["content"] = {"source": SOURCE_REL, "field": CONTENT_FIELD, "join_key": "article_id", "join_confidence": "exact", "raw_content_hash": content["raw_content_hash"], "clean_content_hash": content["clean_content_hash"], "availability": "fulltext" if has_clean_content else "empty_content"}
        else:
            unmatched_docs.append(row.get("document_id"))
            availability = "ambiguous_content" if article_id in ambiguous_article_ids else "missing_content"
            row.update({
                "content_source_file": "", "content_source_hash": "", "content_field": "", "content_length": 0,
                "raw_content_hash": "", "clean_content_hash": "", "join_key": "article_id", "join_confidence": "none",
                "content_availability": availability, "evidence_scope": "title_structured_only",
            })
            provenance["content"] = {"source": availability, "field": CONTENT_FIELD, "join_key": "article_id", "join_confidence": "none"}
        row["field_provenance"] = provenance
        out_docs.append(row)

    doc_by_id = {d["document_id"]: d for d in out_docs}
    mentions_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        mentions_by_doc[mention["document_id"]].append(mention)
    multi_event_document_count = sum(1 for vals in mentions_by_doc.values() if len(vals) > 1)

    out_mentions: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    mention_coverage_by_split: Counter[str] = Counter()
    review_count = 0
    for mention in mentions:
        row = dict(mention)
        doc = doc_by_id[row["document_id"]]
        split = mention_splits.get(row["event_mention_id"], "unknown")
        content = doc.get("content", "") if doc.get("content_availability") == "fulltext" else ""
        if content:
            mention_coverage_by_split[split] += 1
            selected, reason, confidence, needs_review = select_evidence_sentences(content, row)
            text_parts = [clean_text(doc.get("title")), clean_text(row.get("provisional_event_type")), " ".join(clean_text(x) for x in row.get("core_elements", []) if clean_text(x))]
            elems = row.get("elements") or {}
            if isinstance(elems, dict):
                for value in elems.values():
                    if isinstance(value, list):
                        text_parts.append(" ".join(clean_text(x) for x in value if clean_text(x)))
                    else:
                        text_parts.append(clean_text(value))
            text_parts.extend(x["sentence"] for x in selected)
            event_local_text = clean_text("\n".join(x for x in text_parts if x))
            # In multi-event documents, identical local text means the sentence selector did not distinguish mentions.
            if len(mentions_by_doc[row["document_id"]]) > 1:
                temp_texts = []
                for peer in mentions_by_doc[row["document_id"]]:
                    peer_sel, _, _, _ = select_evidence_sentences(content, peer)
                    temp_texts.append(clean_text("\n".join([clean_text(doc.get("title")), clean_text(peer.get("provisional_event_type")), " ".join(x["sentence"] for x in peer_sel)])))
                if len(set(temp_texts)) < len(temp_texts):
                    confidence = "low"
                    needs_review = True
            row.update({
                "event_local_text": event_local_text,
                "evidence_sentences": [x["sentence"] for x in selected],
                "sentence_selection_reason": reason,
                "sentence_offsets": [{"start": x["start"], "end": x["end"], "score": x["score"]} for x in selected],
                "local_text_scope": "fulltext_structured",
                "local_text_confidence": confidence,
                "needs_human_review": needs_review,
                "evidence_scope": "fulltext_structured",
                "content_join_key": "article_id",
                "content_join_confidence": "exact",
                "content_source_file": doc.get("content_source_file"),
                "clean_content_hash": doc.get("clean_content_hash"),
            })
            for rank, item in enumerate(selected, 1):
                evidence_rows.append({
                    "event_mention_id": row["event_mention_id"],
                    "document_id": row["document_id"],
                    "article_id": row.get("article_id"),
                    "rank": rank,
                    "sentence": item["sentence"],
                    "start": item["start"],
                    "end": item["end"],
                    "score": item["score"],
                    "selection_reason": reason,
                    "content_source_file": doc.get("content_source_file"),
                    "clean_content_hash": doc.get("clean_content_hash"),
                    "evidence_scope": "fulltext_structured",
                })
        else:
            review_count += 1
            row.update({
                "event_local_text": clean_text("\n".join([clean_text(doc.get("title")), clean_text(row.get("provisional_event_type")), " ".join(clean_text(x) for x in row.get("core_elements", []) if clean_text(x))])),
                "evidence_sentences": [], "sentence_selection_reason": "no_fulltext_available", "sentence_offsets": [],
                "local_text_scope": "title_structured_only", "local_text_confidence": "none", "needs_human_review": True,
                "evidence_scope": "title_structured_only", "content_join_key": "article_id", "content_join_confidence": "none",
                "content_source_file": "", "clean_content_hash": "",
            })
        out_mentions.append(row)
        if row.get("needs_human_review"):
            review_count += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "normalized_documents.jsonl", out_docs)
    write_jsonl(OUTPUT_DIR / "normalized_event_mentions.jsonl", out_mentions)
    write_jsonl(OUTPUT_DIR / "event_mention_evidence_sentences.jsonl", evidence_rows)
    write_jsonl(OUTPUT_DIR / "content_ambiguity_queue.jsonl", ambiguity_queue)

    lengths = [d["content_length"] for d in out_docs if d.get("content_availability") == "fulltext"]
    source_quality = source_audit["quality_rows"]
    quality_lengths = [r["length"] for r in source_quality if r["nonempty"]]
    source_article_content_counter = Counter(r["clean_content_hash"] for r in source_quality if r["nonempty"])
    clean_hash_to_article_ids: dict[str, set[str]] = defaultdict(set)
    for r in source_quality:
        if r["nonempty"]:
            clean_hash_to_article_ids[r["clean_content_hash"]].add(r["article_id"])
    multi_doc_diffs = []
    for document_id, vals in mentions_by_doc.items():
        if len(vals) <= 1:
            continue
        texts = [m for m in out_mentions if m["document_id"] == document_id]
        unique_texts = len({m.get("event_local_text", "") for m in texts})
        multi_doc_diffs.append({"document_id": document_id, "event_mention_count": len(vals), "unique_local_text_count": unique_texts, "distinguishable": unique_texts == len(vals), "low_confidence_count": sum(1 for m in texts if m.get("local_text_confidence") == "low")})
    report = {
        "version": "fulltext_v2",
        "inputs": {"normalized_documents": str(INPUT_DOCUMENTS.relative_to(ROOT)), "normalized_event_mentions": str(INPUT_MENTIONS.relative_to(ROOT)), "document_split_mapping": str(INPUT_DOC_SPLITS.relative_to(ROOT)), "event_mention_split_mapping": str(INPUT_MENTION_SPLITS.relative_to(ROOT)), "content_source": SOURCE_REL},
        "title_only_input_hashes": title_only_hashes,
        "document_counts": {"normalized_documents": len(documents), "unique_document_ids": len({d["document_id"] for d in documents}), "exact_join_documents": exact_join_docs, "nonempty_fulltext_documents": nonempty_join_docs, "unmatched_documents": len(unmatched_docs), "unmatched_document_ids_sample": unmatched_docs[:20]},
        "event_mention_counts": {"normalized_event_mentions": len(mentions), "unique_event_mention_ids": len({m["event_mention_id"] for m in mentions}), "covered_event_mentions": sum(mention_coverage_by_split.values()), "coverage_rate": round(sum(mention_coverage_by_split.values()) / len(mentions), 6)},
        "split_coverage": {"documents_exact_join": dict(doc_exact_join_by_split), "documents_nonempty_fulltext": dict(doc_coverage_by_split), "event_mentions": dict(mention_coverage_by_split)},
        "article_id_uniqueness": source_audit["source_stats"],
        "content_quality": {
            "source_content_nonempty_rate": round(sum(1 for r in source_quality if r["nonempty"]) / len(source_quality), 6),
            "document_content_nonempty_rate": round(nonempty_join_docs / len(documents), 6),
            "document_content_length": {"min": min(lengths) if lengths else 0, "median": median(lengths) if lengths else 0, "p95": percentile(lengths, 0.95), "max": max(lengths) if lengths else 0},
            "source_content_length": {"min": min(quality_lengths) if quality_lengths else 0, "median": median(quality_lengths) if quality_lengths else 0, "p95": percentile(quality_lengths, 0.95), "max": max(quality_lengths) if quality_lengths else 0},
            "short_document_content_count": sum(1 for x in lengths if x < SHORT_CONTENT_THRESHOLD),
            "content_equals_title_count": sum(1 for d in out_docs if clean_text(d.get("content")) and clean_text(d.get("content")) == clean_text(d.get("title"))),
            "html_or_navigation_pollution_source_count": sum(1 for r in source_quality if r["html_pollution"]),
            "duplicate_clean_content_hash_count": sum(1 for _, c in source_article_content_counter.items() if c > 1),
            "bad_encoding_source_count": sum(1 for r in source_quality if r["bad_encoding"]),
            "avg_chinese_ratio_source": round(sum(r["chinese_ratio"] for r in source_quality) / len(source_quality), 6),
            "same_content_multiple_article_id_count": source_audit["source_stats"]["same_content_multiple_article_id_count"],
        },
        "multi_event_documents": {"document_count": multi_event_document_count, "distinguishable_count": sum(1 for r in multi_doc_diffs if r["distinguishable"]), "needs_review_count": sum(1 for r in multi_doc_diffs if not r["distinguishable"] or r["low_confidence_count"] > 0), "samples": multi_doc_diffs[:20]},
        "evidence_sentence_count": len(evidence_rows),
        "human_review": {"event_mentions_needing_review": sum(1 for m in out_mentions if m.get("needs_human_review"))},
        "leakage_policy": {"content_join_all_splits": True, "threshold_and_model_fit": "development_and_validation_only", "test_usage": "final_inference_only", "task2_label_not_used_for_local_text": True},
        "limitations": ["句子选择为关键词重叠式抽取，不生成新事实。", "同一article_id多正文冲突时隔离到ambiguity_queue，不任意选最长正文。", "无法保证Top-K句子覆盖事件因果全貌；低区分度多事件文档已标记人工复核。"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(f"# Task3 fulltext_v2\n\n本版本只生成全文规范化和EventMention局部文本，不重跑步骤2—6，不覆盖原title-only输出。\n\n## 覆盖率\n\n- Document精确连接：{exact_join_docs}/{len(documents)}\n- Document非空全文：{nonempty_join_docs}/{len(documents)}\n- EventMention非空全文：{sum(mention_coverage_by_split.values())}/{len(mentions)}\n- unmatched Document：{len(unmatched_docs)}\n\n## 正文质量\n\n- Document正文长度：min={report['content_quality']['document_content_length']['min']}，median={report['content_quality']['document_content_length']['median']}，p95={report['content_quality']['document_content_length']['p95']}，max={report['content_quality']['document_content_length']['max']}\n- duplicate article_id：{source_audit['source_stats']['duplicate_article_id_count']}\n- identical duplicate article_id：{source_audit['source_stats']['identical_content_duplicate_count']}\n- conflicting duplicate article_id：{source_audit['source_stats']['conflicting_content_duplicate_count']}\n\n## 安全约束\n\n仅通过`article_id`精确连接；不使用标题、模糊匹配或evidence sentence拼接正文；冲突正文进入`content_ambiguity_queue.jsonl`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
