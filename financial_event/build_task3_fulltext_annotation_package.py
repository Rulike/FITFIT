#!/usr/bin/env python3
"""Build fulltext_v2 round-1 human annotation package without auto labels."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import median
from typing import Any

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from task3_common import ROOT, clean_text, read_jsonl

SEED = 2026
RNG = Random(SEED)
OUT_DIR = ROOT / "outputs/task3_fulltext_v2/annotation_round1"
WORKBOOK = OUT_DIR / "task3_fulltext_round1.xlsx"
MULTI_EVENT_XLSX = OUT_DIR / "multi_event_document_audit.xlsx"
FIELD_DICT = OUT_DIR / "field_dictionary.md"
QUALITY_REPORT = ROOT / "reports/task3_fulltext_quality_report.json"
ANNOTATION_REPORT = ROOT / "reports/task3_fulltext_annotation_round1.json"
GUIDE = ROOT / "docs/task3_fulltext_annotation_guide.md"

DOCS_PATH = ROOT / "outputs/task3_fulltext_v2/normalized_documents.jsonl"
MENTIONS_PATH = ROOT / "outputs/task3_fulltext_v2/normalized_event_mentions.jsonl"
EVIDENCE_PATH = ROOT / "outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl"
PILOT_XLSX = ROOT / "outputs/task3/task3_pilot_v2_annotated.xlsx"
STEP2_PAIRS = ROOT / "outputs/task3/step2/near_duplicate_pair_predictions.jsonl"
STEP3_PAIRS = ROOT / "outputs/task3/step3/same_event_pair_predictions.jsonl"
DOC_SPLITS = ROOT / "outputs/task3/splits/document_split_mapping.jsonl"
MENTION_SPLITS = ROOT / "outputs/task3/splits/event_mention_split_mapping.jsonl"

TITLE_ONLY_INPUTS = [
    ROOT / "outputs/task3/normalized/normalized_documents.jsonl",
    ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl",
    ROOT / "outputs/task3/splits/document_split_mapping.jsonl",
    ROOT / "outputs/task3/splits/event_mention_split_mapping.jsonl",
    ROOT / "outputs/task3/step2/near_duplicate_pair_predictions.jsonl",
    ROOT / "outputs/task3/step3/same_event_pair_predictions.jsonl",
    ROOT / "outputs/task3/step4/event_cluster_members.jsonl",
    ROOT / "outputs/task3/step5/cluster_timelines.jsonl",
    ROOT / "outputs/task3/step6/graph_nodes.jsonl",
]

NAV_RE = re.compile(r"(当前位置|面包屑|登录|注册|版权所有|免责声明|返回首页|app下载|扫码|分享到|网站地图|相关阅读|广告)")
TEMPLATE_TITLE_RE = re.compile(r"(股市必读|公告|快讯|异动|涨停|跌停|融资净|主力资金|龙虎榜)")
TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[\u4e00-\u9fff]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_pair_id(prefix: str, a: str, b: str, suffix: str = "") -> str:
    x, y = sorted([str(a), str(b)])
    return f"{prefix}-{hashlib.sha1((x+'||'+y+'||'+suffix).encode('utf-8')).hexdigest()[:16]}"


def tokens(text: Any) -> set[str]:
    return set(TOKEN_RE.findall(clean_text(text).lower()))


def jaccard(a: Any, b: Any) -> float:
    sa, sb = tokens(a), tokens(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return round(len(sa & sb) / len(sa | sb), 6)


def list_text(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(clean_text(x) for x in value if clean_text(x))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return clean_text(value)


def first_sentences_summary(text: str, max_chars: int = 480) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def evidence_text(mention: dict[str, Any]) -> str:
    return "\n".join(clean_text(x) for x in mention.get("evidence_sentences", []) if clean_text(x))


def content_quality(doc: dict[str, Any]) -> str:
    content = clean_text(doc.get("content"))
    if doc.get("content_availability") != "fulltext" or not content:
        return "title_structured_only"
    if len(content) < 80:
        return "fulltext_short"
    if NAV_RE.search(content):
        return "fulltext_contaminated"
    if len(content) >= 300 and doc.get("source_name"):
        return "fulltext_high_quality"
    return "fulltext_usable"


def target_text(row: dict[str, Any]) -> str:
    return "|".join(clean_text(x) for x in [row.get("target_type"), row.get("target_code"), list_text(row.get("core_elements"))] if clean_text(x))


def side(prefix: str, doc: dict[str, Any], mention: dict[str, Any] | None, split: str) -> dict[str, Any]:
    mention = mention or {}
    return {
        f"article_id_{prefix}": doc.get("article_id", ""),
        f"document_id_{prefix}": doc.get("document_id", ""),
        f"event_mention_id_{prefix}": mention.get("event_mention_id", ""),
        f"title_{prefix}": doc.get("title", ""),
        f"source_{prefix}": doc.get("source_name", ""),
        f"publish_time_{prefix}": doc.get("publish_time", ""),
        f"event_type_{prefix}": mention.get("provisional_event_type", ""),
        f"target_company_stock_{prefix}": target_text(mention),
        f"core_elements_{prefix}": list_text(mention.get("core_elements", [])),
        f"event_local_text_{prefix}": mention.get("event_local_text", ""),
        f"evidence_sentences_{prefix}": evidence_text(mention),
        f"content_summary_{prefix}": first_sentences_summary(doc.get("content", "")),
        f"content_quality_{prefix}": doc.get("content_quality", ""),
        f"split_{prefix}": split,
        f"content_source_file_{prefix}": doc.get("content_source_file", ""),
        f"clean_content_hash_{prefix}": doc.get("clean_content_hash", ""),
    }


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([str(a), str(b)]))


def load_all() -> dict[str, Any]:
    docs = read_jsonl(DOCS_PATH)
    mentions = read_jsonl(MENTIONS_PATH)
    evidence = read_jsonl(EVIDENCE_PATH)
    doc_splits = {r["document_id"]: r["split"] for r in read_jsonl(DOC_SPLITS)}
    mention_splits = {r["event_mention_id"]: r["split"] for r in read_jsonl(MENTION_SPLITS)}
    for d in docs:
        d["content_quality"] = content_quality(d)
    mentions_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in mentions:
        mentions_by_doc[m["document_id"]].append(m)
    return {
        "docs": docs,
        "doc_by_id": {d["document_id"]: d for d in docs},
        "mentions": mentions,
        "mention_by_id": {m["event_mention_id"]: m for m in mentions},
        "mentions_by_doc": mentions_by_doc,
        "evidence": evidence,
        "doc_splits": doc_splits,
        "mention_splits": mention_splits,
    }


def quality_reports(data: dict[str, Any]) -> dict[str, Any]:
    docs = data["docs"]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in docs:
        if d.get("clean_content_hash"):
            by_hash[d["clean_content_hash"]].append(d)
    duplicate_groups = [v for v in by_hash.values() if len(v) > 1]
    same_source_groups = 0
    cross_source_groups = 0
    for group in duplicate_groups:
        sources = {clean_text(x.get("source_name")) for x in group}
        if len(sources) <= 1:
            same_source_groups += 1
        else:
            cross_source_groups += 1
    short = [d.get("article_id") for d in docs if d.get("content_quality") == "fulltext_short"]
    contaminated = [d.get("article_id") for d in docs if d.get("content_quality") == "fulltext_contaminated"]
    report = {
        "document_count": len(docs),
        "quality_distribution": dict(Counter(d["content_quality"] for d in docs)),
        "short_content_count": len(short),
        "short_article_ids": short,
        "contaminated_count": len(contaminated),
        "contaminated_article_ids_sample": contaminated[:100],
        "duplicate_clean_content_hash_document_count": sum(len(g) for g in duplicate_groups),
        "duplicate_content_group_count": len(duplicate_groups),
        "max_duplicate_group_size": max([len(g) for g in duplicate_groups] or [0]),
        "same_source_duplicate_group_count": same_source_groups,
        "cross_source_duplicate_group_count": cross_source_groups,
        "duplicate_group_samples": [
            {"clean_content_hash": g[0].get("clean_content_hash"), "size": len(g), "sources": sorted({clean_text(x.get("source_name")) for x in g}), "article_ids": [x.get("article_id") for x in g[:10]]}
            for g in duplicate_groups[:20]
        ],
    }
    QUALITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def multi_event_audit(data: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    sample_doc_ids = []
    for doc_id, mentions in sorted(data["mentions_by_doc"].items()):
        if len(mentions) <= 1:
            continue
        sets = [set(m.get("evidence_sentences", [])) for m in mentions]
        texts = [m.get("event_local_text", "") for m in mentions]
        evidence_same = len({"\n".join(sorted(s)) for s in sets}) == 1
        ev_j = 1.0 if len(sets) < 2 else jaccard(" ".join(sets[0]), " ".join(sets[1]))
        txt_j = 1.0 if len(texts) < 2 else jaccard(texts[0], texts[1])
        shared_sentences = set.intersection(*sets) if sets else set()
        only_prefix_diff = ev_j >= 0.999 and txt_j < 0.999
        rows.append({
            "document_id": doc_id,
            "article_id": data["doc_by_id"][doc_id].get("article_id"),
            "title": data["doc_by_id"][doc_id].get("title"),
            "source": data["doc_by_id"][doc_id].get("source_name"),
            "event_mention_count": len(mentions),
            "evidence_sentence_sets_identical": evidence_same,
            "evidence_sentence_jaccard_first_two": ev_j,
            "event_local_text_jaccard_first_two": txt_j,
            "only_structured_prefix_diff": only_prefix_diff,
            "same_evidence_supports_multiple_mentions": bool(shared_sentences),
            "shared_evidence_count": len(shared_sentences),
            "event_mention_ids": "|".join(m["event_mention_id"] for m in mentions),
            "event_types": "|".join(clean_text(m.get("provisional_event_type")) for m in mentions),
            "human_review_relation": "",
            "human_confidence": "",
            "human_note": "",
        })
    # deterministic stratified: highest jaccard and shared evidence first, then hash order.
    rows_sorted = sorted(rows, key=lambda r: (-float(r["evidence_sentence_jaccard_first_two"]), -int(r["shared_evidence_count"]), r["document_id"]))
    sample_doc_ids = [r["document_id"] for r in rows_sorted[:20]]
    return pd.DataFrame(rows), sample_doc_ids


def make_near_row(pair: dict[str, Any], data: dict[str, Any], stratum: str, source: str, old: dict[str, Any] | None = None) -> dict[str, Any] | None:
    da, db = pair.get("document_id_a"), pair.get("document_id_b")
    if da not in data["doc_by_id"] or db not in data["doc_by_id"]:
        return None
    if data["doc_splits"].get(da) == "test" or data["doc_splits"].get(db) == "test":
        return None
    ma = data["mentions_by_doc"].get(da, [{}])[0]
    mb = data["mentions_by_doc"].get(db, [{}])[0]
    d1, d2 = data["doc_by_id"][da], data["doc_by_id"][db]
    row = {"pair_id": pair.get("pair_id") or stable_pair_id("FTND", da, db, stratum), "package_section": source}
    row.update(side("a", d1, ma, data["doc_splits"].get(da, "")))
    row.update(side("b", d2, mb, data["doc_splits"].get(db, "")))
    row.update({
        "title_similarity": pair.get("edit_similarity", pair.get("title_similarity", jaccard(d1.get("title"), d2.get("title")))),
        "fulltext_similarity": jaccard(d1.get("content", ""), d2.get("content", "")),
        "candidate_reason": pair.get("candidate_reason") or pair.get("candidate_source") or stratum,
        "sampling_stratum": stratum,
        "old_agent_label": "",
        "old_agent_confidence": "",
        "old_agent_note": "",
        "human_is_near_duplicate": "",
        "human_fulltext_label": "",
        "human_confidence": "",
        "human_changed_from_title_label": "",
        "human_change_reason": "",
        "human_note": "",
    })
    if old:
        row["old_agent_label"] = old.get("human_is_near_duplicate", "")
        row["old_agent_confidence"] = old.get("human_confidence", "")
        row["old_agent_note"] = old.get("annotator_note", old.get("human_note", ""))
    return row


def make_same_row(pair: dict[str, Any], data: dict[str, Any], stratum: str, source: str, old: dict[str, Any] | None = None) -> dict[str, Any] | None:
    ea, eb = pair.get("event_mention_id_a"), pair.get("event_mention_id_b")
    if ea not in data["mention_by_id"] or eb not in data["mention_by_id"]:
        return None
    ma, mb = data["mention_by_id"][ea], data["mention_by_id"][eb]
    if data["mention_splits"].get(ea) == "test" or data["mention_splits"].get(eb) == "test":
        return None
    da, db = ma["document_id"], mb["document_id"]
    d1, d2 = data["doc_by_id"][da], data["doc_by_id"][db]
    row = {"pair_id": pair.get("pair_id") or stable_pair_id("FTSE", ea, eb, stratum), "package_section": source}
    row.update(side("a", d1, ma, data["mention_splits"].get(ea, "")))
    row.update(side("b", d2, mb, data["mention_splits"].get(eb, "")))
    row.update({
        "title_similarity": pair.get("semantic_score", pair.get("title_similarity", jaccard(d1.get("title"), d2.get("title")))),
        "fulltext_similarity": jaccard(ma.get("event_local_text", ""), mb.get("event_local_text", "")),
        "candidate_reason": pair.get("candidate_reason") or stratum,
        "sampling_stratum": stratum,
        "old_agent_relation": "",
        "old_agent_confidence": "",
        "old_agent_note": "",
        "human_relation": "",
        "human_confidence": "",
        "human_note": "",
    })
    if old:
        row["old_agent_relation"] = old.get("human_relation", old.get("human_same_event", ""))
        row["old_agent_confidence"] = old.get("human_confidence", "")
        row["old_agent_note"] = old.get("annotator_note", old.get("human_note", ""))
    return row


def take_unique(rows: list[dict[str, Any]], n: int, used_keys: set[tuple[str, str]], id_a: str, id_b: str) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        key = pair_key(r.get(id_a), r.get(id_b))
        if key in used_keys:
            continue
        used_keys.add(key)
        out.append(r)
        if len(out) >= n:
            break
    return out


def build_package() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    quality = quality_reports(data)
    multi_df, multi_sample_doc_ids = multi_event_audit(data)
    near_old_df = pd.read_excel(PILOT_XLSX, sheet_name="Near Duplicate", dtype=object).fillna("")
    same_old_df = pd.read_excel(PILOT_XLSX, sheet_name="Same Event", dtype=object).fillna("")
    near_step2 = read_jsonl(STEP2_PAIRS)
    same_step3 = read_jsonl(STEP3_PAIRS)

    used_doc_pairs: set[tuple[str, str]] = set()
    used_em_pairs: set[tuple[str, str]] = set()
    near_reaudit_candidates = []
    for old in near_old_df.to_dict(orient="records"):
        pair = {"pair_id": old.get("pair_id"), "document_id_a": old.get("document_id_a"), "document_id_b": old.get("document_id_b"), "title_similarity": old.get("title_similarity"), "candidate_reason": old.get("candidate_reason")}
        r = make_near_row(pair, data, "original_near_duplicate_fulltext_reaudit", "near_duplicate_reaudit", old)
        if r:
            near_reaudit_candidates.append(r)
    near_reaudit = take_unique(near_reaudit_candidates, 50, used_doc_pairs, "document_id_a", "document_id_b")

    same_reaudit_candidates = []
    for old in same_old_df.to_dict(orient="records"):
        pair = {"pair_id": old.get("pair_id"), "event_mention_id_a": old.get("event_mention_id_a"), "event_mention_id_b": old.get("event_mention_id_b"), "title_similarity": old.get("title_similarity"), "candidate_reason": old.get("candidate_reason")}
        r = make_same_row(pair, data, "original_same_event_fulltext_reaudit", "same_event_reaudit", old)
        if r:
            same_reaudit_candidates.append(r)
    same_reaudit = take_unique(same_reaudit_candidates, 50, used_em_pairs, "event_mention_id_a", "event_mention_id_b")

    # New near-duplicate strata from step2 plus hash-derived duplicate pairs.
    near_pool: list[dict[str, Any]] = []
    for p in near_step2:
        da, db = p.get("document_id_a"), p.get("document_id_b")
        if da not in data["doc_by_id"] or db not in data["doc_by_id"]:
            continue
        d1, d2 = data["doc_by_id"][da], data["doc_by_id"][db]
        title_sim = float(p.get("edit_similarity", 0) or 0)
        full_sim = jaccard(d1.get("content", ""), d2.get("content", ""))
        if d1.get("clean_content_hash") and d1.get("clean_content_hash") == d2.get("clean_content_hash") and d1.get("article_id") != d2.get("article_id"):
            stratum = "same_clean_content_hash_different_article_id"
        elif title_sim >= 0.75 and full_sim < 0.25:
            stratum = "title_similar_fulltext_dissimilar"
        elif title_sim < 0.5 and full_sim >= 0.65:
            stratum = "title_dissimilar_fulltext_similar"
        elif d1.get("source_name") != d2.get("source_name") and full_sim >= 0.55:
            stratum = "cross_source_high_fulltext_similarity"
        elif TEMPLATE_TITLE_RE.search(clean_text(d1.get("title"))) or TEMPLATE_TITLE_RE.search(clean_text(d2.get("title"))):
            stratum = "template_title_different_body"
        elif d1.get("content_quality") in {"fulltext_short", "fulltext_contaminated"} or d2.get("content_quality") in {"fulltext_short", "fulltext_contaminated"}:
            stratum = "short_or_contaminated_content"
        elif p.get("core_fact_conflict"):
            stratum = "core_fact_conflict"
        else:
            stratum = "insufficient_or_general_review"
        r = make_near_row(p, data, stratum, "near_duplicate_new")
        if r:
            near_pool.append(r)
    RNG.shuffle(near_pool)
    # prioritize covering strata
    near_new = []
    for stratum in ["title_similar_fulltext_dissimilar", "title_dissimilar_fulltext_similar", "same_clean_content_hash_different_article_id", "cross_source_high_fulltext_similarity", "template_title_different_body", "core_fact_conflict", "short_or_contaminated_content", "insufficient_or_general_review"]:
        near_new.extend(take_unique([r for r in near_pool if r["sampling_stratum"] == stratum], max(3, 100 // 8), used_doc_pairs, "document_id_a", "document_id_b"))
    near_new.extend(take_unique(near_pool, 100 - len(near_new), used_doc_pairs, "document_id_a", "document_id_b"))
    near_new = near_new[:100]

    same_pool: list[dict[str, Any]] = []
    for p in same_step3:
        ea, eb = p.get("event_mention_id_a"), p.get("event_mention_id_b")
        if ea not in data["mention_by_id"] or eb not in data["mention_by_id"]:
            continue
        ma, mb = data["mention_by_id"][ea], data["mention_by_id"][eb]
        d1, d2 = data["doc_by_id"][ma["document_id"]], data["doc_by_id"][mb["document_id"]]
        same_target = clean_text(ma.get("target_code")) == clean_text(mb.get("target_code"))
        same_type = clean_text(ma.get("provisional_event_type")) == clean_text(mb.get("provisional_event_type"))
        ft_sim = jaccard(ma.get("event_local_text", ""), mb.get("event_local_text", ""))
        title_sim = jaccard(d1.get("title"), d2.get("title"))
        if ma["document_id"] == mb["document_id"]:
            stratum = "same_document_multiple_event_mentions"
        elif same_target and same_type and ft_sim >= 0.65:
            stratum = "same_target_same_type_fulltext_consistent"
        elif same_target and same_type and (p.get("hard_conflicts") or ft_sim < 0.25):
            stratum = "same_target_same_type_core_fact_conflict"
        elif same_target and not same_type:
            stratum = "same_target_different_event_type"
        elif title_sim >= 0.65 and ft_sim < 0.35:
            stratum = "title_similar_fulltext_different_event"
        elif title_sim < 0.35 and ft_sim >= 0.6:
            stratum = "title_different_fulltext_same_event"
        elif "related" in clean_text(p.get("predicted_relation")) or "update" in clean_text(p.get("candidate_reason")):
            stratum = "related_update"
        elif re.search(r"澄清|否认|不实|反转|更正", evidence_text(ma) + evidence_text(mb) + d1.get("title", "") + d2.get("title", "")):
            stratum = "clarification_denial_reversal"
        elif clean_text(ma.get("target_type")) in {"macro", "industry"} or clean_text(mb.get("target_type")) in {"macro", "industry"}:
            stratum = "macro_event_cross_subject"
        else:
            stratum = "insufficient_or_general_review"
        r = make_same_row(p, data, stratum, "same_event_new")
        if r:
            same_pool.append(r)
    # add explicit same-document pairs if not in step3 pool
    for doc_id, mentions in data["mentions_by_doc"].items():
        if len(mentions) > 1:
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    p = {"pair_id": stable_pair_id("FTSEMD", mentions[i]["event_mention_id"], mentions[j]["event_mention_id"]), "event_mention_id_a": mentions[i]["event_mention_id"], "event_mention_id_b": mentions[j]["event_mention_id"], "candidate_reason": "same_document_multiple_event_mentions"}
                    r = make_same_row(p, data, "same_document_multiple_event_mentions", "same_event_new")
                    if r:
                        same_pool.append(r)
    RNG.shuffle(same_pool)
    same_new = []
    same_strata = ["same_target_same_type_fulltext_consistent", "same_target_same_type_core_fact_conflict", "same_target_different_event_type", "title_similar_fulltext_different_event", "title_different_fulltext_same_event", "related_update", "clarification_denial_reversal", "macro_event_cross_subject", "same_document_multiple_event_mentions", "insufficient_or_general_review"]
    for stratum in same_strata:
        same_new.extend(take_unique([r for r in same_pool if r["sampling_stratum"] == stratum], max(3, 100 // len(same_strata)), used_em_pairs, "event_mention_id_a", "event_mention_id_b"))
    same_new.extend(take_unique(same_pool, 100 - len(same_new), used_em_pairs, "event_mention_id_a", "event_mention_id_b"))
    same_new = same_new[:100]

    sheets = {
        "near_duplicate_reaudit": pd.DataFrame(near_reaudit),
        "same_event_reaudit": pd.DataFrame(same_reaudit),
        "near_duplicate_new": pd.DataFrame(near_new),
        "same_event_new": pd.DataFrame(same_new),
    }
    instructions = pd.DataFrame([
        {"item": "purpose", "description": "fulltext_v2第一轮人工标注包，只用于人工复核，不含自动gold标签。"},
        {"item": "leakage", "description": "仅development/validation进入本包；test不用于阈值训练候选。"},
        {"item": "human fields", "description": "所有human_*字段初始为空，请标注员填写。"},
    ])
    field_dict_df = pd.DataFrame([
        {"field": "human_is_near_duplicate", "description": "近重复人工判断：yes/no/uncertain"},
        {"field": "human_relation", "description": "同事件人工关系：same_event/related_update/different_event/uncertain"},
        {"field": "human_confidence", "description": "人工置信度，可填high/medium/low或0-1数值"},
        {"field": "human_note", "description": "人工备注"},
        {"field": "old_agent_*", "description": "旧pilot agent-assisted标签，仅供审计对比，不是gold"},
        {"field": "content_quality", "description": "正文质量等级"},
    ])
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)
        field_dict_df.to_excel(writer, sheet_name="field_dictionary", index=False)
    format_workbook(WORKBOOK)
    multi_df.to_excel(MULTI_EVENT_XLSX, index=False)
    format_workbook(MULTI_EVENT_XLSX)

    FIELD_DICT.write_text("# fulltext_v2 round1字段字典\n\n- `human_*`：人工标注字段，生成时必须为空。\n- `old_agent_*`：原pilot中agent-assisted标签，仅作审计对比，不是gold。\n- `content_summary_*`：正文前300～500字摘要，来自Document完整正文截断。\n- `event_local_text_*`：结构化事件信息 + Top-K相关证据句。\n- `clean_content_hash_*`：清洗正文hash，用于追溯和重复正文识别。\n", encoding="utf-8")
    GUIDE.parent.mkdir(parents=True, exist_ok=True)
    GUIDE.write_text("# 任务三fulltext_v2第一轮标注指南\n\n请仅依据展示的标题、正文摘要、事件局部文本和证据句进行判断。`old_agent_*`字段只表示旧试标建议，不是标准答案。所有`human_*`字段由人工填写。若正文过短、污染或证据不足，请选择`uncertain`并说明原因。\n", encoding="utf-8")

    report = {
        "version": "task3_fulltext_annotation_round1",
        "random_seed": SEED,
        "input_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in TITLE_ONLY_INPUTS if p.exists()},
        "sheet_counts": {name: len(df) for name, df in sheets.items()},
        "quality_report_path": str(QUALITY_REPORT.relative_to(ROOT)),
        "annotation_workbook": str(WORKBOOK.relative_to(ROOT)),
        "multi_event_audit": {
            "document_count": len(multi_df),
            "sample_count": len(multi_sample_doc_ids),
            "sample_document_ids": multi_sample_doc_ids,
            "identical_evidence_set_count": int(multi_df["evidence_sentence_sets_identical"].sum()) if not multi_df.empty else 0,
            "same_evidence_supports_multiple_mentions_count": int(multi_df["same_evidence_supports_multiple_mentions"].sum()) if not multi_df.empty else 0,
            "avg_evidence_sentence_jaccard_first_two": float(multi_df["evidence_sentence_jaccard_first_two"].mean()) if not multi_df.empty else 0.0,
            "avg_event_local_text_jaccard_first_two": float(multi_df["event_local_text_jaccard_first_two"].mean()) if not multi_df.empty else 0.0,
        },
        "stratum_counts": {name: dict(Counter(df["sampling_stratum"])) if not df.empty and "sampling_stratum" in df else {} for name, df in sheets.items()},
        "leakage_policy": {"test_excluded": True, "human_fields_empty": True, "old_agent_labels_are_audit_only": True},
    }
    ANNOTATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def format_workbook(path: Path) -> None:
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for col in ws.columns:
            name = str(col[0].value or "")
            width = 18
            if "title" in name:
                width = 34
            elif "event_local_text" in name or "evidence_sentences" in name or "content_summary" in name:
                width = 55
            elif "note" in name or "reason" in name:
                width = 35
            ws.column_dimensions[col[0].column_letter].width = width
        if ws.title.startswith("near_duplicate"):
            dv = DataValidation(type="list", formula1='"yes,no,uncertain"', allow_blank=True)
            ws.add_data_validation(dv)
            for idx, cell in enumerate(ws[1], 1):
                if cell.value in {"human_is_near_duplicate", "human_fulltext_label", "human_changed_from_title_label"}:
                    dv.add(f"{cell.column_letter}2:{cell.column_letter}1000")
        if ws.title.startswith("same_event"):
            dv = DataValidation(type="list", formula1='"same_event,related_update,different_event,uncertain"', allow_blank=True)
            ws.add_data_validation(dv)
            for idx, cell in enumerate(ws[1], 1):
                if cell.value == "human_relation":
                    dv.add(f"{cell.column_letter}2:{cell.column_letter}1000")
    wb.save(path)


if __name__ == "__main__":
    print(json.dumps(build_package(), ensure_ascii=False, indent=2))
