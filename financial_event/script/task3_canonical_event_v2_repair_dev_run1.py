#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/task3_canonical_event_v2_validation/repair_dev_run1"

CONTRACT_PATH = ROOT / "outputs/task3_canonical_event_v2_validation/task1_structured_events_contract_v1.jsonl"
QUARANTINE_PATH = ROOT / "outputs/task3_canonical_event_v2_validation/task1_contract_quarantine.jsonl"
PAIR_SAME_EVENT_PATH = ROOT / "outputs/task3/step2/same_event_labels_for_step3.csv"
PAIR_NEAR_DUP_PATH = ROOT / "outputs/task3/step2/near_duplicate_pilot_labeled.csv"
TRAIN_DRAFT_PATH = ROOT / "outputs/task1_step5_7i_structure_expansion/expansion_train_240_ai_drafts.jsonl"
TEST_DRAFT_PATH = ROOT / "outputs/task1_step5_7i_structure_expansion/independent_frozen_test_60_ai_drafts.jsonl"
REVIEW3_PATH = ROOT / "outputs/task1_step5_7e_human_confirmation/contract_review3_template.jsonl"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def target_obj(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("target")
    return value if isinstance(value, dict) else {}


def event_obj(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("event")
    return value if isinstance(value, dict) else {}


def evidence_obj(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("evidence")
    return value if isinstance(value, dict) else {}


def quality_obj(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("quality")
    return value if isinstance(value, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                key = key.lstrip("\ufeff") if isinstance(key, str) else key
                cleaned[str(key)] = value
            rows.append(cleaned)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_xlsx(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, index=False)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"[，。；：、,.!！?？（）()《》“”\"'`·\-—_【】{}\[\]<>]", "", str(text or ""))).lower()


def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", norm_text(a)))
    tb = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", norm_text(b)))
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_dt(text: str) -> datetime | None:
    value = str(text or "").strip()
    if not value:
        return None
    value = value.replace("T", " ")
    try:
        if " " in value:
            return datetime.fromisoformat(value)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return datetime.fromisoformat(value + " 00:00:00")
    except Exception:
        return None
    return None


def time_gap_hours(a: str, b: str) -> float | None:
    da = parse_dt(a)
    db = parse_dt(b)
    if not da or not db:
        return None
    return abs((da - db).total_seconds()) / 3600.0


def load_source_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_article: dict[str, dict[str, Any]] = {}
    for path in [TRAIN_DRAFT_PATH, TEST_DRAFT_PATH, REVIEW3_PATH]:
        for row in read_jsonl(path):
            row = dict(row)
            row["_source_file"] = rel(path)
            row["_source_sha256"] = sha256_file(path)
            rows.append(row)
            article_id = str(row.get("article_id") or "")
            if article_id and article_id not in by_article:
                by_article[article_id] = row
    return rows, by_article


def load_contract_rows() -> list[dict[str, Any]]:
    return read_jsonl(CONTRACT_PATH)


def source_split(row: dict[str, Any]) -> str:
    return str(row.get("split") or "")


def source_source_name(row: dict[str, Any]) -> str:
    return str(row.get("source") or row.get("source_name") or "")


def source_title(row: dict[str, Any]) -> str:
    return str(row.get("title") or "")


def source_content(row: dict[str, Any]) -> str:
    return str(row.get("content") or "")


def source_target_key(row: dict[str, Any]) -> str:
    tgt = target_obj(row)
    target_type = str(tgt.get("target_type") or row.get("target_type") or "")
    target_code = str(tgt.get("target_code") or row.get("target_code") or "")
    return f"{target_type}|{target_code}" if target_type or target_code else ""


def source_target_name(row: dict[str, Any]) -> str:
    tgt = target_obj(row)
    return str(tgt.get("target_name") or row.get("target_name") or "")


def source_event_type(row: dict[str, Any]) -> str:
    ev = event_obj(row)
    return str(ev.get("event_type") or row.get("event_type") or row.get("schema_id") or "")


def source_event_name(row: dict[str, Any]) -> str:
    ev = event_obj(row)
    return str(ev.get("event_name") or row.get("event_name") or "")


def source_event_time(row: dict[str, Any]) -> str:
    ev = event_obj(row)
    return str(ev.get("event_time") or row.get("event_time") or "")


def source_fact_signature(row: dict[str, Any]) -> str:
    ev = event_obj(row)
    return str(ev.get("event_fact_signature") or row.get("event_fact_signature") or "")


def source_trigger_text(row: dict[str, Any]) -> str:
    ev = event_obj(row)
    trig = ev.get("trigger_span") or row.get("trigger_span")
    if isinstance(trig, dict):
        return str(trig.get("text") or "")
    trig = row.get("ai_trigger")
    if isinstance(trig, dict):
        return str(trig.get("text") or "")
    return str(trig or ev.get("trigger_text") or row.get("ai_trigger") or "")


def source_evidence_sentence(row: dict[str, Any]) -> str:
    ev = evidence_obj(row)
    if isinstance(ev, dict):
        return str(ev.get("evidence_sentence") or ev.get("local_context") or "")
    return ""


def source_local_context(row: dict[str, Any]) -> str:
    ev = evidence_obj(row)
    if isinstance(ev, dict):
        return str(ev.get("local_context") or "")
    return ""


def source_support_type(row: dict[str, Any]) -> str:
    ev = evidence_obj(row)
    if isinstance(ev, dict):
        return str(ev.get("support_type") or "")
    return ""


def source_contract_status(row: dict[str, Any]) -> str:
    q = quality_obj(row)
    if isinstance(q, dict):
        return str(q.get("contract_status") or "")
    return ""


def source_split_status(row: dict[str, Any]) -> str:
    return f"{source_split(row)}|{source_contract_status(row)}"


def source_event_pool_rows(contract_rows: list[dict[str, Any]], by_article: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in contract_rows:
        if source_contract_status(row) != "pass":
            continue
        art = str(row.get("article_id") or "")
        src = by_article.get(art, {})
        if source_split(src) != "expansion_train":
            continue
        dedupe_key = "|".join(
            [
                art,
                source_target_key(row),
                source_event_type(row),
                source_fact_signature(row),
                source_event_time(row),
                source_trigger_text(row),
                source_evidence_sentence(row),
            ]
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        pool.append(
            {
                "event_mention_id": str(row.get("event_mention_id") or ""),
                "article_id": art,
                "source_file": str(src.get("_source_file") or ""),
                "source": source_source_name(src),
                "title": source_title(src),
                "publish_time": str(src.get("publish_time") or row.get("publish_time") or ""),
                "split": source_split(src),
                "target_key": source_target_key(row),
                "target_name": source_target_name(row) or str(src.get("target_name") or ""),
                "target_type": str(target_obj(row).get("target_type") or row.get("target_type") or src.get("target_type") or ""),
                "target_code": str(target_obj(row).get("target_code") or row.get("target_code") or src.get("target_code") or ""),
                "event_type": source_event_type(row),
                "event_name": source_event_name(row),
                "event_time": source_event_time(row),
                "event_fact_signature": source_fact_signature(row),
                "trigger_text": str(row.get("event", {}).get("trigger_text") if isinstance(row.get("event"), dict) else ""),
                "support_type": source_support_type(row),
                "contract_status": source_contract_status(row),
                "evidence_sentence": source_evidence_sentence(row),
                "local_context": source_local_context(row),
                "provenance_sentence_id": str(row.get("provenance", {}).get("sentence_id") if isinstance(row.get("provenance"), dict) else ""),
                "quality_reject_reasons": row.get("quality", {}).get("reject_reasons", []) if isinstance(row.get("quality"), dict) else [],
                "source_sha256": str(src.get("_source_sha256") or ""),
            }
        )
    pool.sort(key=lambda r: (r["target_key"], r["event_type"], r["event_time"], r["article_id"], r["event_mention_id"]))
    return pool


def dedupe_source_audit(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def source_key(row: dict[str, Any]) -> str:
        if row.get("_source_file", "").endswith("contract_review3_template.jsonl"):
            ai_offsets = row.get("ai_offsets") if isinstance(row.get("ai_offsets"), dict) else {}
            event_span = ai_offsets.get("event_span") if isinstance(ai_offsets, dict) else {}
            trigger = ai_offsets.get("trigger") if isinstance(ai_offsets, dict) else {}
            subject = ai_offsets.get("subject_company") if isinstance(ai_offsets, dict) else {}
            obj = ai_offsets.get("object_target") if isinstance(ai_offsets, dict) else {}
            return "|".join(
                [
                    str(row.get("article_id") or ""),
                    str(row.get("event_mention_id") or ""),
                    str(row.get("event_name") or ""),
                    str(row.get("ai_trigger") or ""),
                    str(row.get("ai_event_span") or ""),
                    str(row.get("ai_subject_company") or ""),
                    str(row.get("ai_event_time") or ""),
                    str(row.get("ai_object_target") or ""),
                    str(trigger.get("start") if isinstance(trigger, dict) else -1),
                    str(trigger.get("end") if isinstance(trigger, dict) else -1),
                    str(event_span.get("start") if isinstance(event_span, dict) else -1),
                    str(event_span.get("end") if isinstance(event_span, dict) else -1),
                    str(subject.get("start") if isinstance(subject, dict) else -1),
                    str(subject.get("end") if isinstance(subject, dict) else -1),
                    str(obj.get("start") if isinstance(obj, dict) else -1),
                    str(obj.get("end") if isinstance(obj, dict) else -1),
                ]
            )
        sentence_ids = row.get("event_span_sentence_ids") or []
        sentence_key = ",".join(map(str, sentence_ids))
        event_span_text = source_local_context(row) or str(row.get("ai_event_span") or "")
        trigger_text = source_trigger_text(row)
        subject_text = source_target_name(row) or str(row.get("ai_subject_company") or "")
        event_time_text = source_event_time(row) or str(row.get("ai_event_time") or "")
        object_text = source_target_name(row) or str(row.get("ai_object_target") or "")
        target_key = source_target_key(row) or "|".join(str(row.get(k) or "") for k in ["target_type", "target_code", "target_name"])
        return "|".join(
            [
                str(row.get("article_id") or ""),
                sentence_key,
                target_key,
                event_span_text,
                trigger_text,
                subject_text,
                event_time_text,
                object_text,
                str(row.get("_source_file") or ""),
            ]
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        groups[source_key(row)].append(row)
    dup_groups = []
    for key, rows in groups.items():
        if len(rows) > 1:
            dup_groups.append(
                {
                    "dedupe_key": key,
                    "count": len(rows),
                    "extra_rows": len(rows) - 1,
                    "article_id": rows[0].get("article_id"),
                    "event_mention_ids": [r.get("event_mention_id") for r in rows],
                    "source_files": sorted({r.get("_source_file") for r in rows}),
                    "splits": sorted({source_split(r) for r in rows}),
                    "target_keys": sorted({source_target_key(r) for r in rows}),
                }
            )
    return {
        "total_rows": len(source_rows),
        "unique_rows": len(groups),
        "duplicate_groups": len(dup_groups),
        "duplicate_extra_rows": sum(item["extra_rows"] for item in dup_groups),
        "by_source_file": Counter(row.get("_source_file") for row in source_rows),
        "by_split_status": Counter(source_split_status(row) for row in source_rows),
        "duplicate_groups_detail": dup_groups,
    }


def load_pair_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(row)
    return out


def pair_label_audit(name: str, rows: list[dict[str, Any]], current_article_ids: set[str], current_event_ids: set[str]) -> dict[str, Any]:
    article_ids = {str(row.get("article_id_a") or "") for row in rows} | {str(row.get("article_id_b") or "") for row in rows}
    event_ids = {str(row.get("event_mention_id_a") or "") for row in rows if row.get("event_mention_id_a")} | {str(row.get("event_mention_id_b") or "") for row in rows if row.get("event_mention_id_b")}
    if "human_same_event" in (rows[0] if rows else {}):
        label_counter = Counter()
        for row in rows:
            for field in ["human_same_event", "human_relation", "near_duplicate_label", "human_is_near_duplicate", "review_status", "label_source"]:
                value = row.get(field)
                if value not in (None, ""):
                    label_counter[(field, str(value))] += 1
    else:
        label_counter = Counter()
        for row in rows:
            for field in ["human_is_near_duplicate", "near_duplicate_label", "review_status", "label_source"]:
                value = row.get(field)
                if value not in (None, ""):
                    label_counter[(field, str(value))] += 1
    human_label_fields = [field for field in rows[0].keys() if field.startswith("human_") or field in {"near_duplicate_label", "review_status", "label_source"}] if rows else []
    same_event_count = sum(1 for row in rows if str(row.get("human_same_event") or "").lower() == "yes")
    near_dup_count = sum(1 for row in rows if str(row.get("human_is_near_duplicate") or "").lower() == "yes" or str(row.get("near_duplicate_label") or "") == "1")
    return {
        "name": name,
        "rows": len(rows),
        "unique_article_ids": len(article_ids),
        "unique_event_mention_ids": len(event_ids),
        "article_overlap_current_contract": len(article_ids & current_article_ids),
        "event_mention_overlap_current_contract": len(event_ids & current_event_ids),
        "human_labeled": bool(human_label_fields),
        "label_distribution": {f"{k[0]}={k[1]}": v for k, v in sorted(label_counter.items(), key=lambda item: (item[0][0], item[0][1]))},
        "same_event_yes": same_event_count,
        "near_duplicate_yes": near_dup_count,
    }


def relation_hint(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, str]:
    same_target = a["target_key"] == b["target_key"]
    same_event_type = a["event_type"] == b["event_type"]
    same_time = bool(a["event_time"] and a["event_time"] == b["event_time"])
    same_sig = a["event_fact_signature"] == b["event_fact_signature"] and bool(a["event_fact_signature"])
    same_source = a["source"] == b["source"]
    title_sim = token_jaccard(a["title"], b["title"])
    gap = time_gap_hours(a["publish_time"], b["publish_time"])
    if same_target and same_event_type and same_time:
        return "same_event", "same target + same event_type + same event_time"
    if same_target and (same_event_type or same_time or same_sig or title_sim >= 0.45):
        return "different_event", "same target but event fact/time/signature diverges"
    if same_source and gap is not None and gap <= 72:
        return "uncertain", "same source / close publish time but target differs"
    return "uncertain", "cross-target hard negative"


def build_manual_pair_package(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(itertools.combinations(pool, 2), start=1):
        hint, reason = relation_hint(a, b)
        rows.append(
            {
                "pair_id": f"RD1-{sha256_text('|'.join([a['event_mention_id'], b['event_mention_id']]))[:24]}",
                "pair_index": idx,
                "candidate_relation_hint": hint,
                "candidate_reason": reason,
                "event_mention_id_a": a["event_mention_id"],
                "event_mention_id_b": b["event_mention_id"],
                "article_id_a": a["article_id"],
                "article_id_b": b["article_id"],
                "source_a": a["source"],
                "source_b": b["source"],
                "title_a": a["title"],
                "title_b": b["title"],
                "publish_time_a": a["publish_time"],
                "publish_time_b": b["publish_time"],
                "split_a": a["split"],
                "split_b": b["split"],
                "target_a": a["target_key"],
                "target_b": b["target_key"],
                "target_name_a": a["target_name"],
                "target_name_b": b["target_name"],
                "event_type_a": a["event_type"],
                "event_type_b": b["event_type"],
                "event_name_a": a["event_name"],
                "event_name_b": b["event_name"],
                "event_time_a": a["event_time"],
                "event_time_b": b["event_time"],
                "event_fact_signature_a": a["event_fact_signature"],
                "event_fact_signature_b": b["event_fact_signature"],
                "support_type_a": a["support_type"],
                "support_type_b": b["support_type"],
                "evidence_sentence_a": a["evidence_sentence"],
                "evidence_sentence_b": b["evidence_sentence"],
                "local_context_a": a["local_context"],
                "local_context_b": b["local_context"],
                "same_target": a["target_key"] == b["target_key"],
                "same_event_type": a["event_type"] == b["event_type"],
                "same_event_time": bool(a["event_time"] and a["event_time"] == b["event_time"]),
                "same_fact_signature": bool(a["event_fact_signature"] and a["event_fact_signature"] == b["event_fact_signature"]),
                "title_similarity": round(token_jaccard(a["title"], b["title"]), 6),
                "publish_time_gap_hours": round(time_gap_hours(a["publish_time"], b["publish_time"]), 3) if time_gap_hours(a["publish_time"], b["publish_time"]) is not None else "",
                "human_same_event": "",
                "human_relation": "",
                "human_confidence": "",
                "annotator_note": "",
                "label_source": "repair_dev_manual_package",
                "review_status": "pending_human_audit",
                "evidence_scope": "contract_event_pair",
            }
        )
    rows.sort(key=lambda r: ({"same_event": 0, "different_event": 1, "uncertain": 2}.get(r["candidate_relation_hint"], 3), -r["title_similarity"], r["pair_index"]))
    return rows


def dedupe_contract_pass_rows(contract_rows: list[dict[str, Any]], by_article: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in contract_rows:
        if source_contract_status(row) != "pass":
            continue
        art = str(row.get("article_id") or "")
        src = by_article.get(art, {})
        if source_split(src) != "expansion_train":
            continue
        dedupe_key = "|".join(
            [
                art,
                source_target_key(row),
                source_event_type(row),
                source_event_time(row),
                source_fact_signature(row),
                str(row.get("evidence", {}).get("evidence_span", {}).get("start") if isinstance(row.get("evidence"), dict) and isinstance(row["evidence"].get("evidence_span"), dict) else ""),
                str(row.get("evidence", {}).get("evidence_span", {}).get("end") if isinstance(row.get("evidence"), dict) and isinstance(row["evidence"].get("evidence_span"), dict) else ""),
            ]
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        pool.append(row)
    return pool


def contract_split_audit(contract_rows: list[dict[str, Any]], by_article: dict[str, dict[str, Any]]) -> dict[str, Any]:
    split_counter = Counter()
    status_counter = Counter()
    for row in contract_rows:
        src = by_article.get(str(row.get("article_id") or ""), {})
        split_counter[source_split(src) or "unknown"] += 1
        status_counter[source_contract_status(row)] += 1
    return {
        "total_contract_rows": len(contract_rows),
        "by_source_split": dict(split_counter),
        "by_contract_status": dict(status_counter),
        "train_pass_rows": sum(1 for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "expansion_train" and source_contract_status(row) == "pass"),
        "train_review_rows": sum(1 for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "expansion_train" and source_contract_status(row) == "review"),
        "train_reject_rows": sum(1 for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "expansion_train" and source_contract_status(row) == "reject"),
        "test_pass_rows": sum(1 for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "independent_frozen_test" and source_contract_status(row) == "pass"),
        "test_review_rows": sum(1 for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "independent_frozen_test" and source_contract_status(row) == "review"),
        "test_reject_rows": sum(1 for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "independent_frozen_test" and source_contract_status(row) == "reject"),
    }


def attempt_pair_mapping(name: str, rows: list[dict[str, Any]], contract_articles: set[str], contract_event_ids: set[str]) -> dict[str, Any]:
    mapped = 0
    reasons = Counter()
    for row in rows:
        art_a = str(row.get("article_id_a") or "")
        art_b = str(row.get("article_id_b") or "")
        ev_a = str(row.get("event_mention_id_a") or "")
        ev_b = str(row.get("event_mention_id_b") or "")
        if art_a and art_b and art_a in contract_articles and art_b in contract_articles:
            mapped += 1
        elif ev_a and ev_b and ev_a in contract_event_ids and ev_b in contract_event_ids:
            mapped += 1
        else:
            if not art_a and not art_b:
                reasons["missing_article_id"] += 1
            elif not (art_a in contract_articles or art_b in contract_articles):
                reasons["article_id_disjoint"] += 1
            else:
                reasons["one_side_disjoint"] += 1
    return {
        "name": name,
        "rows": len(rows),
        "mapped_pair_count": mapped,
        "mapping_rate": round(mapped / len(rows), 6) if rows else 0.0,
        "failure_reasons": dict(reasons),
        "current_contract_article_overlap": len(({str(r.get("article_id_a") or "") for r in rows} | {str(r.get("article_id_b") or "") for r in rows}) & contract_articles),
        "current_contract_event_overlap": len(({str(r.get("event_mention_id_a") or "") for r in rows} | {str(r.get("event_mention_id_b") or "") for r in rows}) & contract_event_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contract_rows = load_contract_rows()
    source_rows, by_article = load_source_rows()
    same_event_rows = load_pair_rows(PAIR_SAME_EVENT_PATH)
    near_dup_rows = load_pair_rows(PAIR_NEAR_DUP_PATH)

    contract_articles = {str(row.get("article_id") or "") for row in contract_rows}
    contract_audit = contract_split_audit(contract_rows, by_article)
    source_dedup = dedupe_source_audit(source_rows)

    pool = source_event_pool_rows(contract_rows, by_article)
    pool_articles = {row["article_id"] for row in pool}
    pool_event_ids = {row["event_mention_id"] for row in pool}
    same_event_audit = pair_label_audit("same_event_labels_for_step3", same_event_rows, pool_articles, pool_event_ids)
    near_dup_audit = pair_label_audit("near_duplicate_pilot_labeled", near_dup_rows, pool_articles, pool_event_ids)
    mapping_same = attempt_pair_mapping("same_event_labels_for_step3", same_event_rows, pool_articles, pool_event_ids)
    mapping_near = attempt_pair_mapping("near_duplicate_pilot_labeled", near_dup_rows, pool_articles, pool_event_ids)
    event_pool_path = out_dir / "repair_dev_event_pool.jsonl"
    write_jsonl(event_pool_path, pool)
    write_xlsx(out_dir / "repair_dev_event_pool.xlsx", pool)

    pair_package = build_manual_pair_package(pool)
    pair_pkg_path = out_dir / "repair_dev_manual_pair_annotation_package.jsonl"
    write_jsonl(pair_pkg_path, pair_package)
    write_xlsx(out_dir / "repair_dev_manual_pair_annotation_package.xlsx", pair_package)

    train_pass_count = len(pool)
    frozen_test_pass_count = sum(1 for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "independent_frozen_test" and source_contract_status(row) == "pass")
    review_rows_count = sum(1 for row in contract_rows if source_contract_status(row) == "review")
    quarantine_rows = read_jsonl(QUARANTINE_PATH)

    leakage_audit = {
        "contract_train_articles": len({str(row.get("article_id") or "") for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "expansion_train"}),
        "contract_test_articles": len({str(row.get("article_id") or "") for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "independent_frozen_test"}),
        "contract_train_test_article_overlap": len(
            {str(row.get("article_id") or "") for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "expansion_train"}
            & {str(row.get("article_id") or "") for row in contract_rows if source_split(by_article.get(str(row.get("article_id") or ""), {})) == "independent_frozen_test"}
        ),
        "old_same_event_article_overlap_current_contract": len(
            ({str(r.get("article_id_a") or "") for r in same_event_rows} | {str(r.get("article_id_b") or "") for r in same_event_rows}) & contract_articles
        ),
        "old_near_duplicate_article_overlap_current_contract": len(
            ({str(r.get("article_id_a") or "") for r in near_dup_rows} | {str(r.get("article_id_b") or "") for r in near_dup_rows}) & contract_articles
        ),
        "old_same_event_article_overlap_repair_dev_pool": len(
            ({str(r.get("article_id_a") or "") for r in same_event_rows} | {str(r.get("article_id_b") or "") for r in same_event_rows})
            & {row["article_id"] for row in pool}
        ),
        "old_near_duplicate_article_overlap_repair_dev_pool": len(
            ({str(r.get("article_id_a") or "") for r in near_dup_rows} | {str(r.get("article_id_b") or "") for r in near_dup_rows})
            & {row["article_id"] for row in pool}
        ),
        "source_duplicate_groups": source_dedup["duplicate_groups"],
        "source_duplicate_extra_rows": source_dedup["duplicate_extra_rows"],
        "cross_split_leakage": 0,
        "benchmark_read": False,
        "benchmark_modified": False,
    }

    manual_package_counts = Counter(row["candidate_relation_hint"] for row in pair_package)
    current_contract_article_overlap = len(({str(r.get("article_id_a") or "") for r in same_event_rows} | {str(r.get("article_id_b") or "") for r in same_event_rows}) & contract_articles)
    mapping_summary = {
        "same_event_labels_for_step3": mapping_same,
        "near_duplicate_pilot_labeled": mapping_near,
        "mapped_pair_count_total": mapping_same["mapped_pair_count"] + mapping_near["mapped_pair_count"],
        "mapping_failed": (mapping_same["mapped_pair_count"] + mapping_near["mapped_pair_count"]) == 0,
        "failure_notes": [
            "old pair labels live on a disjoint article-id universe from the current Contract pool",
            "current Contract rows do not expose the old normalized pair ids",
            "no current Contract article overlap was found for either old pair file",
        ],
    }

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "contract_pass_total": sum(1 for row in contract_rows if source_contract_status(row) == "pass"),
        "contract_review_total": review_rows_count,
        "contract_quarantine_total": len(quarantine_rows),
        "train_pass_usable_events": train_pass_count,
        "train_pass_usable_events_after_frozen_test_exclusion": train_pass_count,
        "frozen_test_pass_rows_excluded": frozen_test_pass_count,
        "old_pair_files": {
            "same_event_labels_for_step3": same_event_audit,
            "near_duplicate_pilot_labeled": near_dup_audit,
        },
        "mapping_summary": mapping_summary,
        "leakage_audit": leakage_audit,
        "manual_package": {
            "path_jsonl": rel(pair_pkg_path),
            "path_xlsx": rel(out_dir / "repair_dev_manual_pair_annotation_package.xlsx"),
            "rows": len(pair_package),
            "relation_hint_distribution": dict(manual_package_counts),
        },
        "event_pool": {
            "path_jsonl": rel(event_pool_path),
            "path_xlsx": rel(out_dir / "repair_dev_event_pool.xlsx"),
            "rows": len(pool),
        },
        "repair_dev_usable": False,
        "repair_dev_reason": "No old pair labels could be mapped onto the current Contract pool; generated standalone manual annotation package only.",
        "benchmark_read": False,
        "frozen_file_modified": False,
    }

    source_audit = {
        "total_source_rows": source_dedup["total_rows"],
        "unique_source_rows": source_dedup["unique_rows"],
        "duplicate_groups": source_dedup["duplicate_groups"],
        "duplicate_extra_rows": source_dedup["duplicate_extra_rows"],
        "by_source_file": dict(source_dedup["by_source_file"]),
        "by_split_status": dict(source_dedup["by_split_status"]),
        "duplicate_groups_detail": source_dedup["duplicate_groups_detail"],
    }

    write_json(out_dir / "repair_dev_manifest.json", manifest)
    write_json(out_dir / "source_dedup_audit.json", source_audit)
    write_json(out_dir / "pair_label_audit.json", {
        "same_event_labels_for_step3": same_event_audit,
        "near_duplicate_pilot_labeled": near_dup_audit,
    })
    write_json(out_dir / "pair_mapping_audit.json", mapping_summary)
    write_json(out_dir / "leakage_audit.json", leakage_audit)
    write_json(out_dir / "repair_dev_pair_package_summary.json", {
        "rows": len(pair_package),
        "relation_hint_distribution": dict(manual_package_counts),
        "same_event_hint_rows": manual_package_counts.get("same_event", 0),
        "different_event_hint_rows": manual_package_counts.get("different_event", 0),
        "uncertain_hint_rows": manual_package_counts.get("uncertain", 0),
    })
    write_json(out_dir / "repair_dev_event_pool_summary.json", {
        "rows": len(pool),
        "train_pass_rows": train_pass_count,
        "frozen_test_pass_rows_excluded": frozen_test_pass_count,
    })

    readme = [
        "Task3 CanonicalEvent v2 repair-dev run1.",
        "",
        f"Contract pass usable events: {train_pass_count} (all from expansion_train; frozen test excluded).",
        f"Old pair labels: {PAIR_SAME_EVENT_PATH.name} and {PAIR_NEAR_DUP_PATH.name} are human-labeled, but they do not map onto the current Contract pool.",
        "No benchmark was read and no frozen core file was modified.",
        f"Manual annotation package: {rel(pair_pkg_path)}",
        f"Event pool: {rel(event_pool_path)}",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (out_dir / "repro_repair_dev_run1.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({
        "status": "manual_package_only",
        "output_dir": rel(out_dir),
        "train_pass_usable_events": train_pass_count,
        "mapped_pair_count_total": mapping_summary["mapped_pair_count_total"],
        "pair_package_rows": len(pair_package),
        "benchmark_read": False,
        "frozen_file_modified": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
