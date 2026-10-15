#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPAIR_DIR = ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1"
REPAIR_PAIRS = REPAIR_DIR / "repair_dev_500.parquet"
V2_CANDIDATES = REPAIR_DIR / "resolver_v2_candidate_pairs.parquet"
MANUAL_ADJUDICATION = REPAIR_DIR / "manual_review_adjudication.jsonl"
TASK4_EVENT_INTERFACE = REPAIR_DIR / "task4_event_interface.jsonl"
TASK4_FEATURE_TABLE = REPAIR_DIR / "task4_feature_table.parquet"
DEFAULT_OUT = ROOT / "outputs/task3_resolver_v3_candidate"
QWEN_EMBED = ROOT.parent / "downloads/Qwen/Qwen3-Embedding-0.6B"
MARAG_PYTHON = Path("/root/miniconda3/envs/marag_retriever/bin/python")

CHANNEL_ORDER_V3 = [
    "same_document_hash",
    "target_fact",
    "target_time_fact",
    "target_object",
    "family_object",
    "family_trigger",
    "same_source_same_target",
    "target_semantic",
    "same_modality",
    "same_sentence",
]
R_ORDER = ["R1", "R2", "R4", "R3"]
MISSING = "__MISSING__"
R3_SEMANTIC_SCORE_FLOOR = 0.58


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"none", "null", "nan", "nat"} else value


def norm(value: Any) -> str:
    value = unicodedata.normalize("NFKC", text(value))
    value = re.sub(r"\s+", "", value)
    return value.lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(*parts: Any, length: int = 16) -> str:
    return sha256_text("||".join(text(p) for p in parts))[:length]


def pair_key(a: Any, b: Any) -> tuple[str, str]:
    aa, bb = text(a), text(b)
    return tuple(sorted((aa, bb)))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def json_loads_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    raw = text(value)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_dt(value: Any) -> datetime | None:
    raw = text(value)
    if not raw or raw in {"未明确", "不明", "未知", "近日", "日前"}:
        return None
    try:
        ts = pd.to_datetime(raw, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def clean_tokens(value: Any) -> list[str]:
    raw = unicodedata.normalize("NFKC", text(value))
    if not raw:
        return []
    return re.findall(r"[0-9a-zA-Z\u4e00-\u9fff]+", raw.lower())


def extract_tokens(value: Any) -> set[str]:
    raw = unicodedata.normalize("NFKC", text(value))
    out: set[str] = set()
    for m in re.finditer(r"\d+(?:\.\d+)?(?:万|亿|万元|亿元|港元|元|%|％|股|份|期|只|辆|台|座|个|家|套|户|mw|gw|kw|wh/kg)?", raw, re.I):
        out.add(m.group(0).replace(" ", ""))
    for m in re.finditer(r"[《“\"『‘](.*?)[》”\"』’]", raw):
        token = text(m.group(1))
        if token:
            out.add(token)
    for m in re.finditer(r"[A-Za-z0-9\u4e00-\u9fff]{2,30}", raw):
        token = text(m.group(0))
        if len(token) > 1:
            out.add(token)
    return {x for x in out if x}


def token_jaccard(a: Any, b: Any) -> float:
    sa, sb = set(clean_tokens(a)), set(clean_tokens(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def sequence_ratio(a: Any, b: Any) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def lexical_similarity(a: Any, b: Any) -> float:
    return round(0.7 * sequence_ratio(a, b) + 0.3 * token_jaccard(a, b), 6)


def target_meta(event: dict[str, Any]) -> dict[str, str]:
    target = event.get("target") if isinstance(event.get("target"), dict) else {}
    return {
        "target_key": text(target.get("target_key") or event.get("target_key")),
        "target_type": text(target.get("target_type") or event.get("target_type")),
        "target_code": text(target.get("target_code") or event.get("target_code")),
        "target_name": text(target.get("target_name") or event.get("target_name")),
    }


def event_meta(event: dict[str, Any]) -> dict[str, str]:
    ev = event.get("event") if isinstance(event.get("event"), dict) else {}
    event_type = text(ev.get("event_type") or event.get("event_type"))
    family = text(ev.get("event_family") or event.get("event_family") or (event_type.split("_")[-1] if event_type else ""))
    return {
        "event_type": event_type,
        "event_name": text(ev.get("event_name") or event.get("event_name") or family),
        "event_family": family,
        "trigger_text": text(ev.get("trigger_text") or event.get("trigger_text")),
        "object_text": text(ev.get("object_text") or event.get("object_text")),
        "event_fact": text(ev.get("event_fact") or event.get("event_fact")),
        "event_fact_signature": text(ev.get("event_fact_signature") or event.get("event_fact_signature")),
        "event_time": text(ev.get("event_time") or event.get("event_time")),
        "modality_status": text(ev.get("modality_status") or event.get("modality_status") or "fact"),
        "polarity_or_state": text(ev.get("polarity_or_state") or event.get("polarity_or_state") or "unknown"),
    }


def evidence_meta(event: dict[str, Any]) -> dict[str, str]:
    ev = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    return {
        "evidence_sentence": text(ev.get("evidence_sentence") or event.get("evidence_sentence")),
        "local_context": text(ev.get("local_context") or event.get("local_context")),
        "support_type": text(ev.get("support_type") or event.get("support_type")),
    }


def source_meta(event: dict[str, Any]) -> dict[str, str]:
    src = event.get("source") if isinstance(event.get("source"), dict) else {}
    return {
        "source_name": text(src.get("source_name") or event.get("source_name") or event.get("source")),
        "source_dataset": text(src.get("source_dataset") or event.get("source_dataset")),
    }


def provenance_meta(event: dict[str, Any]) -> dict[str, str]:
    prov = event.get("provenance") if isinstance(event.get("provenance"), dict) else {}
    return {
        "document_hash": text(prov.get("document_hash") or event.get("document_hash")),
        "sentence_id": text(prov.get("sentence_id") or event.get("sentence_id")),
        "source_file": text(prov.get("source_file") or event.get("source_file")),
    }


def sample_id(event: dict[str, Any], fallback: str = "") -> str:
    source_row = event.get("source_row") if isinstance(event.get("source_row"), dict) else {}
    return text(source_row.get("sample_id") or event.get("sample_id") or fallback)


def event_id(event: dict[str, Any]) -> str:
    source_row = event.get("source_row") if isinstance(event.get("source_row"), dict) else {}
    sid = sample_id(event)
    return text(source_row.get("event_id") or event.get("event_id") or sid)


def pair_time_gap_days(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    ea, eb = event_meta(a), event_meta(b)
    da = parse_dt(ea.get("event_time")) or parse_dt(a.get("publish_time"))
    db = parse_dt(eb.get("event_time")) or parse_dt(b.get("publish_time"))
    if not da or not db:
        return None
    return abs((da - db).total_seconds()) / 86400.0


def pair_time_relation(a: dict[str, Any], b: dict[str, Any]) -> str:
    ea, eb = event_meta(a), event_meta(b)
    da, db = parse_dt(ea.get("event_time")), parse_dt(eb.get("event_time"))
    if da and db:
        gap = abs((da - db).total_seconds()) / 86400.0
        if gap == 0:
            return "same_time"
        if gap <= 3:
            return "close_time"
        if gap <= 30:
            return "near_time"
        return "time_conflict"
    pa, pb = parse_dt(a.get("publish_time")), parse_dt(b.get("publish_time"))
    if pa and pb:
        gap = abs((pa - pb).total_seconds()) / 86400.0
        if gap <= 3:
            return "close_publish_time"
        if gap <= 30:
            return "near_publish_time"
    return "unknown"


def identity_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ta, tb = target_meta(a), target_meta(b)
    ea, eb = event_meta(a), event_meta(b)
    pa, pb = provenance_meta(a), provenance_meta(b)
    sa, sb = source_meta(a), source_meta(b)
    iva = a.get("identity_features") if isinstance(a.get("identity_features"), dict) else {}
    ivb = b.get("identity_features") if isinstance(b.get("identity_features"), dict) else {}
    fact_a = iva.get("fact_tokens") if isinstance(iva.get("fact_tokens"), list) and iva.get("fact_tokens") else sorted(extract_tokens(ea.get("event_fact")))
    fact_b = ivb.get("fact_tokens") if isinstance(ivb.get("fact_tokens"), list) and ivb.get("fact_tokens") else sorted(extract_tokens(eb.get("event_fact")))
    obj_a, obj_b = sorted(extract_tokens(ea.get("object_text"))), sorted(extract_tokens(eb.get("object_text")))
    trig_a, trig_b = sorted(extract_tokens(ea.get("trigger_text"))), sorted(extract_tokens(eb.get("trigger_text")))
    evia, evib = evidence_meta(a), evidence_meta(b)
    f: dict[str, Any] = {
        "same_target": bool(ta["target_key"]) and ta["target_key"] == tb["target_key"],
        "same_target_type": bool(ta["target_type"]) and ta["target_type"] == tb["target_type"],
        "same_target_code": bool(ta["target_code"]) and ta["target_code"] == tb["target_code"],
        "same_family": bool(ea["event_family"]) and ea["event_family"] == eb["event_family"],
        "same_event_type": bool(ea["event_type"]) and ea["event_type"] == eb["event_type"],
        "same_event_name": bool(ea["event_name"]) and ea["event_name"] == eb["event_name"],
        "same_trigger": bool(ea["trigger_text"]) and ea["trigger_text"] == eb["trigger_text"],
        "same_object": bool(ea["object_text"]) and ea["object_text"] == eb["object_text"],
        "same_modality": bool(ea["modality_status"]) and ea["modality_status"] == eb["modality_status"],
        "same_article": bool(text(a.get("article_id"))) and text(a.get("article_id")) == text(b.get("article_id")),
        "same_source": bool(sa["source_name"]) and sa["source_name"] == sb["source_name"],
        "same_sentence": bool(pa["sentence_id"]) and pa["sentence_id"] == pb["sentence_id"],
        "same_document_hash": bool(pa["document_hash"]) and pa["document_hash"] == pb["document_hash"],
        "same_event_fact_signature": bool(ea["event_fact_signature"]) and ea["event_fact_signature"] == eb["event_fact_signature"],
        "fact_tokens_a": fact_a,
        "fact_tokens_b": fact_b,
        "object_tokens_a": obj_a,
        "object_tokens_b": obj_b,
        "trigger_tokens_a": trig_a,
        "trigger_tokens_b": trig_b,
    }
    f["fact_token_overlap"] = sorted(set(fact_a) & set(fact_b))
    f["object_token_overlap"] = sorted(set(obj_a) & set(obj_b))
    f["trigger_token_overlap"] = sorted(set(trig_a) & set(trig_b))
    f["title_similarity"] = lexical_similarity(ea.get("event_fact"), eb.get("event_fact"))
    f["fact_similarity"] = token_jaccard(" ".join(fact_a), " ".join(fact_b))
    f["object_similarity"] = token_jaccard(" ".join(obj_a), " ".join(obj_b))
    f["trigger_similarity"] = token_jaccard(" ".join(trig_a), " ".join(trig_b))
    f["evidence_similarity"] = lexical_similarity(evia.get("evidence_sentence"), evib.get("evidence_sentence"))
    f["time_relation"] = pair_time_relation(a, b)
    if ea["modality_status"] == eb["modality_status"]:
        f["modality_relation"] = f"same_{ea['modality_status']}" if ea["modality_status"] else "unknown"
    elif {ea["modality_status"], eb["modality_status"]} <= {"forecast", "fact"}:
        f["modality_relation"] = "forecast_vs_fact"
    elif {ea["modality_status"], eb["modality_status"]} <= {"rumor", "fact"}:
        f["modality_relation"] = "rumor_vs_fact"
    else:
        f["modality_relation"] = "different_modality"
    if f["same_article"]:
        f["provenance_relation"] = "same_article"
    elif f["same_source"]:
        f["provenance_relation"] = "same_source"
    elif f["same_document_hash"]:
        f["provenance_relation"] = "same_document_hash"
    else:
        f["provenance_relation"] = "cross_source"
    return f


def hard_conflicts(features: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not features["same_target"] and (features["same_target_type"] or features["same_target_code"]):
        reasons.append("target_key_conflict")
    if features["same_family"] and features["same_target"] and not features["object_token_overlap"] and features["object_tokens_a"] and features["object_tokens_b"]:
        reasons.append("object_conflict")
    if features["same_target"] and features["same_family"] and features["fact_similarity"] < 0.1 and features["title_similarity"] < 0.75 and features["evidence_similarity"] < 0.75:
        reasons.append("same_family_fact_divergence")
    if features["time_relation"] == "time_conflict":
        reasons.append("time_conflict")
    if features["modality_relation"] in {"forecast_vs_fact", "rumor_vs_fact"}:
        reasons.append("modality_conflict")
    if features["same_target"] and features["fact_similarity"] < 0.15 and not features["same_event_type"] and not features["same_event_name"] and not features["same_family"]:
        reasons.append("fact_conflict")
    return reasons


def event_text_for_hash(event: dict[str, Any]) -> str:
    ev, ee = event_meta(event), evidence_meta(event)
    return "\n".join(
        [
            text(event.get("article_id")),
            text(event.get("publish_time")),
            target_meta(event).get("target_key", ""),
            ev.get("event_family", ""),
            ev.get("event_fact", ""),
            ev.get("trigger_text", ""),
            ev.get("object_text", ""),
            ee.get("evidence_sentence", ""),
        ]
    )


def content_hash(event: dict[str, Any]) -> str:
    prov = provenance_meta(event)
    if prov["document_hash"]:
        return prov["document_hash"]
    ee = evidence_meta(event)
    return sha256_text(ee["local_context"] or ee["evidence_sentence"] or event_text_for_hash(event))


def duplication_key(event: dict[str, Any]) -> str:
    return content_hash(event) or text(event.get("article_id"))


def strong_keys(event: dict[str, Any]) -> set[str]:
    ev, ee = event_meta(event), evidence_meta(event)
    tm = target_meta(event)
    raw = unicodedata.normalize("NFKC", " ".join([ev["object_text"], ev["event_fact"], ee["evidence_sentence"]]))
    raw = re.sub(r"\bcompany[_|]?\d{3,8}(?:[_|][A-Za-z0-9\u4e00-\u9fff]+)?", " ", raw, flags=re.I)
    out: set[str] = set()
    for m in re.finditer(r"[《“\"『‘](.*?)[》”\"』’]", raw):
        token = norm(m.group(1))
        if len(token) >= 2:
            out.add(token)
    amount_patterns = [
        r"\d+(?:\.\d+)?(?:万|亿)?元",
        r"\d+(?:\.\d+)?(?:万|亿)?股",
    ]
    for pattern in amount_patterns:
        for m in re.finditer(pattern, raw, flags=re.I):
            out.add(norm(m.group(0)))
    phrase_suffixes = "项目|合同|协议|产品|地块|订单|产线|基地|工厂|车型|药品|平台|基金|股权|债券|专利|牌照|矿区|装置|产能|园区"
    for m in re.finditer(rf"[\u4e00-\u9fffA-Za-z0-9\-]{{2,40}}(?:{phrase_suffixes})", raw):
        token = norm(m.group(0))
        if len(token) >= 4:
            out.add(token)
    for m in re.finditer(r"[A-Za-z]{1,8}[-_ ]?\d{2,}[A-Za-z0-9\-]*|\d{4,}[A-Za-z][A-Za-z0-9\-]*", raw):
        out.add(norm(m.group(0)))
    generic = {"公司", "项目", "产品", "合同", "协议", "公告", "股份", "股权"}
    target_terms = {norm(tm.get("target_name")), norm(tm.get("target_code")), norm(tm.get("target_key"))}
    cleaned = set()
    for item in out:
        if not item or item in generic or item in target_terms:
            continue
        if re.fullmatch(r"company[_|]?\d{3,8}.*", item, flags=re.I):
            continue
        if len(item) < 4 and not re.search(r"\d", item):
            continue
        cleaned.add(item)
    return cleaned


def has_core_argument_support(a: dict[str, Any], b: dict[str, Any], f: dict[str, Any]) -> bool:
    overlap = strong_keys(a) & strong_keys(b)
    return bool(
        overlap
        or f["object_similarity"] >= 0.6
        or (len(f["object_token_overlap"]) >= 2 and f["trigger_similarity"] >= 0.25)
        or (f["same_trigger"] and f["object_similarity"] >= 0.25)
        or f["fact_similarity"] >= 0.65
    )


def r1_reasons(a: dict[str, Any], b: dict[str, Any], f: dict[str, Any]) -> list[str]:
    if not f["same_target"]:
        return []
    reasons: list[str] = []
    strong_overlap = strong_keys(a) & strong_keys(b)
    if f["same_event_fact_signature"]:
        reasons.append("same_target_fact_signature")
    if f["same_document_hash"] and (f["same_event_fact_signature"] or f["same_trigger"] or f["object_similarity"] >= 0.5 or f["fact_similarity"] >= 0.55 or strong_overlap):
        reasons.append("same_target_document_hash_with_fact_or_argument")
    if f["same_article"] and f["same_sentence"] and (f["same_event_fact_signature"] or f["same_trigger"] or f["object_similarity"] >= 0.5 or f["fact_similarity"] >= 0.55 or strong_overlap):
        reasons.append("same_sentence_with_target_and_strong_support")
    if strong_overlap and (f["same_trigger"] or f["object_similarity"] >= 0.5 or f["fact_similarity"] >= 0.75) and f["time_relation"] != "time_conflict":
        reasons.append("shared_announcement_project_product_contract_key")
    return sorted(set(reasons))


def r2_reasons(a: dict[str, Any], b: dict[str, Any], f: dict[str, Any], conflicts: list[str]) -> list[str]:
    if not f["same_target"] or not f["same_family"] or conflicts:
        return []
    reasons: list[str] = []
    strong_overlap = strong_keys(a) & strong_keys(b)
    if f["fact_similarity"] >= 0.60:
        reasons.append("target_family_fact_lexical")
    if f["evidence_similarity"] >= 0.65:
        reasons.append("target_family_evidence_lexical")
    if strong_overlap and f["object_similarity"] >= 0.35 and f["trigger_similarity"] >= 0.35 and (f["fact_similarity"] >= 0.35 or f["evidence_similarity"] >= 0.45):
        reasons.append("target_family_argument_trigger")
    if f["time_relation"] in {"same_time", "close_time", "near_time", "close_publish_time"} and (
        f["fact_similarity"] >= 0.55 or f["evidence_similarity"] >= 0.60 or (strong_overlap and f["trigger_similarity"] >= 0.50)
    ):
        reasons.append("target_family_time_with_fact_or_argument")
    return sorted(set(reasons))


def r4_reasons(a: dict[str, Any], b: dict[str, Any], f: dict[str, Any], conflicts: list[str]) -> list[str]:
    if not f["same_target"] or conflicts:
        return []
    reasons: list[str] = []
    if f["same_source"] and f["same_family"] and (f["fact_similarity"] >= 0.35 or f["evidence_similarity"] >= 0.45 or f["trigger_similarity"] >= 0.35 or f["object_similarity"] >= 0.25):
        reasons.append("same_source_same_target_structured_support")
    if f["same_modality"] and f["same_family"] and f["fact_similarity"] >= 0.50:
        reasons.append("same_modality_with_family_fact_support")
    if f["time_relation"] in {"same_time", "close_time"} and f["fact_similarity"] >= 0.40:
        reasons.append("target_time_fact")
    return sorted(set(reasons))


class DSU:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra
        return True

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for value in self.parent:
            out[self.find(value)].append(value)
        return {key: sorted(vals) for key, vals in out.items()}


def normalize_event(raw: Any, fallback_sample_id: str, fallback_article_id: str) -> dict[str, Any]:
    obj = json_loads_maybe(raw)
    if not isinstance(obj, dict):
        obj = {}
    obj = dict(obj)
    if not text(obj.get("article_id")):
        obj["article_id"] = fallback_article_id
    source_row = obj.get("source_row") if isinstance(obj.get("source_row"), dict) else {}
    source_row = dict(source_row)
    if not text(source_row.get("sample_id")):
        source_row["sample_id"] = fallback_sample_id
    if not text(source_row.get("event_id")):
        source_row["event_id"] = fallback_sample_id
    obj["source_row"] = source_row
    return obj


def build_events(pair_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for row in pair_df.itertuples(index=False):
        left = normalize_event(getattr(row, "left_event"), text(getattr(row, "sample_id_a")), text(getattr(row, "article_id_a")))
        right = normalize_event(getattr(row, "right_event"), text(getattr(row, "sample_id_b")), text(getattr(row, "article_id_b")))
        for ev, fallback in [(left, text(getattr(row, "sample_id_a"))), (right, text(getattr(row, "sample_id_b")))]:
            sid = sample_id(ev, fallback)
            if sid and sid not in events:
                events[sid] = ev
    return events


def add_group_edges(dsu: DSU, groups: dict[str, list[str]], reason: str, edge_rows: list[dict[str, Any]]) -> None:
    for key, ids in groups.items():
        unique = sorted(set(i for i in ids if i))
        if len(unique) < 2:
            continue
        anchor = unique[0]
        for other in unique[1:]:
            changed = dsu.union(anchor, other)
            edge_rows.append({"a": anchor, "b": other, "reason": reason, "key": key, "changed": changed})


def build_components(events: dict[str, dict[str, Any]], pair_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    ids = sorted(events)
    canonical = DSU(ids)
    isolation = DSU(ids)
    canonical_edges: list[dict[str, Any]] = []
    isolation_edges: list[dict[str, Any]] = []

    for row in pair_df.itertuples(index=False):
        a, b = text(getattr(row, "sample_id_a")), text(getattr(row, "sample_id_b"))
        if text(getattr(row, "gold_relation")) == "same_event":
            changed = canonical.union(a, b)
            isolation.union(a, b)
            edge = {"a": a, "b": b, "reason": "human_same_event", "key": text(getattr(row, "pair_id")), "changed": changed}
            canonical_edges.append(edge)
            isolation_edges.append(dict(edge))

    sig_groups: dict[str, list[str]] = defaultdict(list)
    content_groups: dict[str, list[str]] = defaultdict(list)
    article_groups: dict[str, list[str]] = defaultdict(list)
    for sid, ev in events.items():
        tm, em = target_meta(ev), event_meta(ev)
        sig = em["event_fact_signature"]
        if sig:
            sig_groups[f"{tm['target_key']}|{sig}"].append(sid)
        ch = content_hash(ev)
        if ch:
            content_groups[ch].append(sid)
        if text(ev.get("article_id")):
            article_groups[text(ev.get("article_id"))].append(sid)

    add_group_edges(canonical, sig_groups, "same_target_fact_signature", canonical_edges)
    add_group_edges(isolation, sig_groups, "same_target_fact_signature", isolation_edges)
    add_group_edges(canonical, content_groups, "same_content_hash", canonical_edges)
    add_group_edges(isolation, content_groups, "same_content_hash", isolation_edges)
    add_group_edges(isolation, article_groups, "same_article_isolation", isolation_edges)

    by_target_family: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid, ev in events.items():
        by_target_family[(target_meta(ev)["target_key"], event_meta(ev)["event_family"])].append(sid)

    for (target_key, family), group in by_target_family.items():
        if not target_key or not family or len(group) < 2:
            continue
        for a, b in combinations(sorted(group), 2):
            ev_a, ev_b = events[a], events[b]
            f = identity_features(ev_a, ev_b)
            if f["time_relation"] not in {"same_time", "close_time", "near_time", "close_publish_time", "near_publish_time"}:
                continue
            if not has_core_argument_support(ev_a, ev_b, f) and not f["same_event_fact_signature"] and not f["same_document_hash"]:
                continue
            reason = "near_time_duplicate_with_fact_or_argument_support"
            key = f"{target_key}|{family}|{f['time_relation']}"
            changed = canonical.union(a, b)
            isolation.union(a, b)
            edge = {"a": a, "b": b, "reason": reason, "key": key, "changed": changed}
            canonical_edges.append(edge)
            isolation_edges.append(dict(edge))

    canonical_map = {
        sid: f"CEC-{stable_hash(','.join(vals), length=18)}"
        for _, vals in canonical.components().items()
        for sid in vals
    }
    isolation_map = {
        sid: f"ISC-{stable_hash(','.join(vals), length=18)}"
        for _, vals in isolation.components().items()
        for sid in vals
    }
    return canonical_map, isolation_map, canonical_edges, isolation_edges


def component_distribution(component_map: dict[str, str]) -> dict[str, Any]:
    counts = Counter(component_map.values())
    sizes = sorted(counts.values(), reverse=True)
    if not sizes:
        return {"component_count": 0, "max_size": 0}
    arr = np.array(sizes, dtype=np.float64)
    hist = Counter(sizes)
    return {
        "component_count": len(sizes),
        "max_size": int(max(sizes)),
        "p50_size": float(np.percentile(arr, 50)),
        "p90_size": float(np.percentile(arr, 90)),
        "p95_size": float(np.percentile(arr, 95)),
        "singleton_components": int(hist.get(1, 0)),
        "size_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "top10_sizes": [int(x) for x in sizes[:10]],
    }


def assign_splits(events: dict[str, dict[str, Any]], isolation_map: dict[str, str], pair_df: pd.DataFrame) -> dict[str, str]:
    by_comp: dict[str, list[str]] = defaultdict(list)
    for sid, comp in isolation_map.items():
        by_comp[comp].append(sid)
    comp_same_counts = Counter()
    for row in pair_df.itertuples(index=False):
        if text(getattr(row, "gold_relation")) != "same_event":
            continue
        a, b = text(getattr(row, "sample_id_a")), text(getattr(row, "sample_id_b"))
        if isolation_map.get(a) == isolation_map.get(b):
            comp_same_counts[isolation_map[a]] += 1
    total_events = len(events)
    total_same = sum(comp_same_counts.values())
    targets = {
        "train": {"events": total_events * 0.60, "same": total_same * 0.60},
        "dev": {"events": total_events * 0.20, "same": total_same * 0.20},
        "final_holdout": {"events": total_events * 0.20, "same": total_same * 0.20},
    }
    current = {name: {"events": 0, "same": 0} for name in targets}
    comp_split: dict[str, str] = {}
    comps = sorted(by_comp, key=lambda c: (-len(by_comp[c]), -comp_same_counts[c], stable_hash(c)))
    for comp in comps:
        size = len(by_comp[comp])
        same = comp_same_counts[comp]
        best_split = None
        best_score = None
        for split in ["train", "dev", "final_holdout"]:
            event_over = max(0.0, current[split]["events"] + size - targets[split]["events"])
            same_over = max(0.0, current[split]["same"] + same - targets[split]["same"])
            event_deficit = abs(targets[split]["events"] - (current[split]["events"] + size))
            same_deficit = abs(targets[split]["same"] - (current[split]["same"] + same))
            score = (event_over * 10.0 + same_over * 3.0 + event_deficit + same_deficit * 0.25, stable_hash(split, comp))
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        assert best_split is not None
        comp_split[comp] = best_split
        current[best_split]["events"] += size
        current[best_split]["same"] += same
    return {sid: comp_split[comp] for sid, comp in isolation_map.items()}


def split_pair_rows(pair_df: pd.DataFrame, event_split: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in pair_df.itertuples(index=False):
        a, b = text(getattr(row, "sample_id_a")), text(getattr(row, "sample_id_b"))
        sa, sb = event_split.get(a, ""), event_split.get(b, "")
        pair_split = sa if sa == sb and sa else "cross_split"
        rows.append(
            {
                "pair_id": text(getattr(row, "pair_id")),
                "sample_id_a": a,
                "sample_id_b": b,
                "gold_relation": text(getattr(row, "gold_relation")),
                "gold_relation_origin": text(getattr(row, "gold_relation_origin")),
                "split_a": sa,
                "split_b": sb,
                "pair_split": pair_split,
            }
        )
    return pd.DataFrame(rows)


def split_leakage_audit(events: dict[str, dict[str, Any]], event_split: dict[str, str], canonical_map: dict[str, str], isolation_map: dict[str, str], pair_df: pd.DataFrame) -> dict[str, Any]:
    def cross_count(groups: dict[str, list[str]]) -> dict[str, Any]:
        bad = []
        for key, ids in groups.items():
            splits = sorted({event_split.get(i, "") for i in ids if event_split.get(i, "")})
            if len(splits) > 1:
                bad.append({"key": key, "splits": splits, "event_count": len(set(ids))})
        return {"cross_split_group_count": len(bad), "examples": bad[:20]}

    article_groups: dict[str, list[str]] = defaultdict(list)
    content_groups: dict[str, list[str]] = defaultdict(list)
    sig_groups: dict[str, list[str]] = defaultdict(list)
    canonical_groups: dict[str, list[str]] = defaultdict(list)
    isolation_groups: dict[str, list[str]] = defaultdict(list)
    for sid, ev in events.items():
        if text(ev.get("article_id")):
            article_groups[text(ev.get("article_id"))].append(sid)
        ch = content_hash(ev)
        if ch:
            content_groups[ch].append(sid)
        sig = event_meta(ev)["event_fact_signature"]
        if sig:
            sig_groups[f"{target_meta(ev)['target_key']}|{sig}"].append(sid)
        canonical_groups[canonical_map[sid]].append(sid)
        isolation_groups[isolation_map[sid]].append(sid)

    same_cross = 0
    same_total = 0
    for row in pair_df.itertuples(index=False):
        if text(getattr(row, "gold_relation")) != "same_event":
            continue
        same_total += 1
        a, b = text(getattr(row, "sample_id_a")), text(getattr(row, "sample_id_b"))
        if event_split.get(a) != event_split.get(b):
            same_cross += 1

    return {
        "human_same_event": {"total": same_total, "cross_split": same_cross},
        "article": cross_count(article_groups),
        "content_hash": cross_count(content_groups),
        "event_fact_signature": cross_count(sig_groups),
        "canonical_component": cross_count(canonical_groups),
        "isolation_component": cross_count(isolation_groups),
        "future_official1799_read": 0,
    }


def event_mapping_rows(events: dict[str, dict[str, Any]], event_split: dict[str, str], canonical_map: dict[str, str], isolation_map: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid, ev in sorted(events.items()):
        tm, em, pm = target_meta(ev), event_meta(ev), provenance_meta(ev)
        rows.append(
            {
                "event_uid": sid,
                "event_id": event_id(ev),
                "split": event_split[sid],
                "canonical_component_id": canonical_map[sid],
                "isolation_component_id": isolation_map[sid],
                "article_id": text(ev.get("article_id")),
                "content_hash": content_hash(ev),
                "duplication_key": duplication_key(ev),
                "target_key": tm["target_key"],
                "target_name": tm["target_name"],
                "event_family": em["event_family"],
                "event_type": em["event_type"],
                "event_name": em["event_name"],
                "event_fact_signature": em["event_fact_signature"],
                "trigger_text": em["trigger_text"],
                "object_text": em["object_text"],
                "publish_time": text(ev.get("publish_time")),
                "event_time": em["event_time"],
                "document_hash": pm["document_hash"],
                "sentence_id": pm["sentence_id"],
            }
        )
    return rows


def split_counts(event_map_df: pd.DataFrame, pair_split_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["train", "dev", "final_holdout"]:
        pairs = pair_split_df[pair_split_df["pair_split"].eq(split)]
        events = event_map_df[event_map_df["split"].eq(split)]
        unique_pairs: dict[tuple[str, str], str] = {}
        duplicate_label_rows = 0
        conflicting_duplicate_labels = 0
        for row in pairs.itertuples(index=False):
            key = pair_key(row.sample_id_a, row.sample_id_b)
            if key in unique_pairs:
                duplicate_label_rows += 1
                if unique_pairs[key] != text(row.gold_relation):
                    conflicting_duplicate_labels += 1
            else:
                unique_pairs[key] = text(row.gold_relation)
        unique_distribution = Counter(unique_pairs.values())
        out[split] = {
            "event_count": int(len(events)),
            "pair_count": int(len(pairs)),
            "unique_unordered_pair_count": int(len(unique_pairs)),
            "duplicate_label_rows": int(duplicate_label_rows),
            "conflicting_duplicate_labels": int(conflicting_duplicate_labels),
            "gold_relation_distribution": {k: int(v) for k, v in pairs["gold_relation"].value_counts().to_dict().items()},
            "unique_gold_relation_distribution": {k: int(v) for k, v in unique_distribution.items()},
            "positive_same_event_count": int((pairs["gold_relation"] == "same_event").sum()),
            "negative_different_event_count": int((pairs["gold_relation"] == "different_event").sum()),
            "uncertain_count": int((pairs["gold_relation"] == "uncertain").sum()),
            "unique_positive_same_event_count": int(unique_distribution.get("same_event", 0)),
            "unique_negative_different_event_count": int(unique_distribution.get("different_event", 0)),
            "unique_uncertain_count": int(unique_distribution.get("uncertain", 0)),
        }
    cross = pair_split_df[pair_split_df["pair_split"].eq("cross_split")]
    cross_unique: dict[tuple[str, str], str] = {}
    for row in cross.itertuples(index=False):
        cross_unique.setdefault(pair_key(row.sample_id_a, row.sample_id_b), text(row.gold_relation))
    out["cross_split_labeled_pairs_excluded_from_train_dev_final_metrics"] = {
        "pair_count": int(len(cross)),
        "unique_unordered_pair_count": int(len(cross_unique)),
        "gold_relation_distribution": {k: int(v) for k, v in cross["gold_relation"].value_counts().to_dict().items()},
        "unique_gold_relation_distribution": {k: int(v) for k, v in Counter(cross_unique.values()).items()},
    }
    return out


def parse_channel_cell(value: Any) -> set[str]:
    raw = text(value)
    if not raw:
        return set()
    return {x for x in raw.split("|") if x}


def v2_candidate_dedup_audit(v2_df: pd.DataFrame, pair_df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    key_rows: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_pairs = 0
    channel_to_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in v2_df.itertuples(index=False):
        key = pair_key(getattr(row, "sample_id_a"), getattr(row, "sample_id_b"))
        if not key[0] or not key[1]:
            continue
        if key in key_rows:
            duplicate_pairs += 1
        channels = parse_channel_cell(getattr(row, "channels"))
        merged = set(key_rows.get(key, {}).get("channels_set", set())) | channels
        key_rows[key] = {"channels_set": merged}
        for ch in channels:
            channel_to_pairs[ch].add(key)

    unique_pairs = set(key_rows)
    same_pairs_all = {
        pair_key(row.sample_id_a, row.sample_id_b)
        for row in pair_df.itertuples(index=False)
        if text(row.gold_relation) == "same_event"
    }
    independent = {
        ch: {
            "unique_pair_count": len(pairs),
            "human_same_event_hits_all_labeled": len(pairs & same_pairs_all),
        }
        for ch, pairs in sorted(channel_to_pairs.items())
    }
    incremental: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str]] = set()
    ordered = [ch for ch in CHANNEL_ORDER_V3 if ch in channel_to_pairs] + sorted(ch for ch in channel_to_pairs if ch not in CHANNEL_ORDER_V3)
    for ch in ordered:
        pairs = channel_to_pairs[ch]
        new_pairs = pairs - seen
        incremental[ch] = {
            "new_unique_pair_count": len(new_pairs),
            "new_human_same_event_hits_all_labeled": len(new_pairs & same_pairs_all),
        }
        seen |= pairs
    audit = {
        "status": "v2_candidate_table_dedup_channel_audit_complete",
        "scope": "v2_candidate_table_dedup_channel_audit_not_v3_result",
        "v2_candidate_rows": int(len(v2_df)),
        "unique_unordered_pair_count": int(len(unique_pairs)),
        "duplicate_pair_rows": int(len(v2_df) - len(unique_pairs)),
        "duplicate_pair_count_after_merge": int(duplicate_pairs),
        "candidate_ratio_over_515_full_combination": round(len(unique_pairs) / 132355, 6),
        "candidate_reduction_over_515_full_combination": round(1.0 - len(unique_pairs) / 132355, 6),
        "channel_order_for_incremental_count": ordered,
        "channel_independent_counts": independent,
        "channel_incremental_counts": incremental,
        "human_same_event_total_all_labeled": int(len(same_pairs_all)),
        "note": "same_sentence is audited as an old channel only; it is not allowed as a standalone v3 recall rule.",
    }
    write_json(out_dir / "v2_candidate_table_dedup_channel_audit.json", audit)
    return audit


def build_event_template(event: dict[str, Any]) -> str:
    tm, em, ee, sm = target_meta(event), event_meta(event), evidence_meta(event), source_meta(event)
    def field(name: str, value: Any) -> str:
        v = text(value)
        return f"{name}: {v if v else MISSING}"
    return "\n".join(
        [
            field("target_key", tm["target_key"]),
            field("target_name", tm["target_name"]),
            field("event_family", em["event_family"]),
            field("event_name", em["event_name"]),
            field("trigger", em["trigger_text"]),
            field("object", em["object_text"]),
            field("event_time", em["event_time"]),
            field("publish_time", event.get("publish_time")),
            field("source", sm["source_name"]),
            field("event_fact", em["event_fact"]),
            field("evidence", ee["evidence_sentence"]),
        ]
    )


def worker_python(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    try:
        import faiss  # type: ignore  # noqa: F401
        import transformers  # type: ignore  # noqa: F401

        return sys.executable
    except Exception:
        pass
    if MARAG_PYTHON.exists():
        return str(MARAG_PYTHON)
    return None


def encode_qwen_current_python(texts: list[str], cache_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    text_digest = sha256_text("\n".join(sha256_text(t) for t in texts))
    meta_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    if not (QWEN_EMBED / "model.safetensors").exists():
        return {"status": "blocked_qwen_embedding_missing", "model_path": str(QWEN_EMBED)}
    try:
        if cache_path.exists():
            arr = np.load(cache_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            if arr.shape[0] == len(texts) and meta.get("input_text_digest") == text_digest:
                return {
                    "status": "ok",
                    "cache_hit": True,
                    "cache_path": str(cache_path),
                    "input_text_digest": text_digest,
                    "embedding_shape": list(arr.shape),
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                }
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(QWEN_EMBED), trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(str(QWEN_EMBED), trust_remote_code=True, torch_dtype=torch.float32).eval()
        batches: list[np.ndarray] = []
        batch_size = 4
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                out = model(**encoded)
                hidden = out.last_hidden_state
                lengths = encoded["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0]), lengths]
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                batches.append(pooled.cpu().numpy().astype("float32"))
        embeddings = np.vstack(batches).astype("float32")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        report = {
            "status": "ok",
            "cache_hit": False,
            "model_path": str(QWEN_EMBED),
            "model_safetensors_sha256": sha256_file(QWEN_EMBED / "model.safetensors"),
            "cache_path": str(cache_path),
            "input_text_digest": text_digest,
            "embedding_shape": list(embeddings.shape),
            "finite": bool(np.isfinite(embeddings).all()),
            "normalization_ok": bool(np.allclose(np.linalg.norm(embeddings, axis=1), np.ones(embeddings.shape[0]), atol=1e-2)),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
        write_json(meta_path, report)
        return report
    except Exception as exc:
        return {
            "status": "blocked_qwen_encoding_failed",
            "model_path": str(QWEN_EMBED),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }


def ann_worker(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    input_path = Path(args.ann_input)
    output_path = Path(args.ann_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import faiss  # type: ignore
    except Exception as exc:
        write_json(output_path, {"status": "blocked_dependency_missing", "error": f"{type(exc).__name__}: {exc}", "pairs": []})
        return

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload["events"]
    model_path = Path(payload["model_path"])
    cache_path = Path(payload["cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    texts = [row["text"] for row in rows]
    try:
        if payload.get("precomputed_embeddings"):
            embeddings = np.load(cache_path)
            cache_hit = True
        elif cache_path.exists():
            embeddings = np.load(cache_path)
            cache_hit = bool(embeddings.shape[0] == len(texts))
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer

            cache_hit = False
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, torch_dtype=torch.float32).eval()
            batches: list[np.ndarray] = []
            batch_size = int(payload.get("batch_size") or 8)
            with torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    encoded = tokenizer(batch, padding=True, truncation=True, max_length=int(payload.get("max_length") or 512), return_tensors="pt")
                    out = model(**encoded)
                    hidden = out.last_hidden_state
                    lengths = encoded["attention_mask"].sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.shape[0]), lengths]
                    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                    batches.append(pooled.cpu().numpy().astype("float32"))
            embeddings = np.vstack(batches).astype("float32")
            np.save(cache_path, embeddings)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        id_to_idx = {row["event_uid"]: idx for idx, row in enumerate(rows)}
        pairs: dict[tuple[str, str], dict[str, Any]] = {}
        by_target: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            by_target[text(row.get("target_key"))].append(idx)
        for target_key, idxs in by_target.items():
            if len(idxs) < 2:
                continue
            vecs = np.ascontiguousarray(embeddings[idxs])
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            size = len(idxs)
            date_values = []
            for i in idxs:
                raw = text(rows[i].get("publish_time"))
                try:
                    ts = pd.to_datetime(raw, errors="coerce")
                    if not pd.isna(ts):
                        date_values.append(ts.to_pydatetime())
                except Exception:
                    pass
            span_days = 0.0
            if len(date_values) >= 2:
                span_days = (max(date_values) - min(date_values)).total_seconds() / 86400.0
            base_k = 2
            if size >= 8:
                base_k = 3
            if size >= 20:
                base_k = 5
            if span_days <= 7 and size >= 5:
                base_k += 1
            k = min(size, max(2, base_k + 1))
            scores, nbrs = index.search(vecs, k)
            for local_i, source_global_idx in enumerate(idxs):
                source = rows[source_global_idx]
                for rank in range(1, k):
                    local_j = int(nbrs[local_i, rank])
                    if local_j < 0 or local_j >= len(idxs):
                        continue
                    other_global_idx = idxs[local_j]
                    other = rows[other_global_idx]
                    key = pair_key(source["event_uid"], other["event_uid"])
                    if key[0] == key[1]:
                        continue
                    sim = float(scores[local_i, rank])
                    if sim < R3_SEMANTIC_SCORE_FLOOR:
                        continue
                    prev = pairs.get(key)
                    if prev is None or sim > float(prev["semantic_score"]):
                        pairs[key] = {
                            "a": key[0],
                            "b": key[1],
                            "semantic_score": round(sim, 6),
                            "target_key": target_key,
                            "dynamic_top_k": int(k - 1),
                            "target_density": int(size),
                            "target_time_span_days": round(float(span_days), 6),
                        }
        write_json(
            output_path,
            {
                "status": "ok",
                "backend": "faiss.IndexFlatIP",
                "model_path": str(model_path),
                "model_safetensors_sha256": sha256_file(model_path / "model.safetensors") if (model_path / "model.safetensors").exists() else "",
                "semantic_score_floor": R3_SEMANTIC_SCORE_FLOOR,
                "cache_path": str(cache_path),
                "cache_hit": cache_hit,
                "event_count": len(rows),
                "pair_count": len(pairs),
                "pairs": list(pairs.values()),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            },
        )
    except Exception as exc:
        write_json(
            output_path,
            {
                "status": "blocked_runtime_error",
                "backend": "faiss.IndexFlatIP",
                "error": f"{type(exc).__name__}: {exc}",
                "event_count": len(rows),
                "pairs": [],
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            },
        )


def run_r3_ann(dev_events: dict[str, dict[str, Any]], out_dir: Path, explicit_worker: str | None) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    ann_input = out_dir / "r3_ann_dev_input.json"
    ann_output = out_dir / "r3_ann_dev_output.json"
    rows = []
    for sid, ev in sorted(dev_events.items()):
        rows.append(
            {
                "event_uid": sid,
                "text": build_event_template(ev),
                "target_key": target_meta(ev)["target_key"],
                "event_family": event_meta(ev)["event_family"],
                "publish_time": text(ev.get("publish_time")),
                "event_time": event_meta(ev)["event_time"],
            }
        )
    cache_path = out_dir / "r3_qwen_dev_embeddings.npy"
    encode_report = encode_qwen_current_python([row["text"] for row in rows], cache_path)
    payload = {
        "events": rows,
        "model_path": str(QWEN_EMBED),
        "cache_path": str(cache_path),
        "batch_size": 4,
        "max_length": 512,
        "precomputed_embeddings": encode_report.get("status") == "ok",
        "encode_report": encode_report,
    }
    write_json(ann_input, payload)
    py = worker_python(explicit_worker)
    if not py:
        report = {"status": "blocked_no_python_with_faiss", "event_count": len(rows), "pairs": []}
        write_json(ann_output, report)
        return set(), report
    if not (QWEN_EMBED / "model.safetensors").exists():
        report = {"status": "blocked_qwen_embedding_missing", "model_path": str(QWEN_EMBED), "event_count": len(rows), "pairs": []}
        write_json(ann_output, report)
        return set(), report
    cmd = [py, str(Path(__file__).resolve()), "--ann-worker", "--ann-input", str(ann_input), "--ann-output", str(ann_output)]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    if completed.returncode != 0:
        report = {
            "status": "blocked_ann_worker_failed",
            "worker_python": py,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-4000:],
            "pairs": [],
        }
        write_json(ann_output, report)
        return set(), report
    report = json.loads(ann_output.read_text(encoding="utf-8"))
    pairs = {pair_key(row.get("a"), row.get("b")) for row in report.get("pairs", []) if row.get("a") and row.get("b")}
    report["encode_report"] = encode_report
    report["worker_python"] = py
    report["valid_r3_backend"] = report.get("status") == "ok" and text(report.get("backend")).startswith("faiss")
    write_json(ann_output, report)
    return pairs, report


def candidate_metrics(candidate_pairs: set[tuple[str, str]], same_pairs: set[tuple[str, str]], all_possible: int) -> dict[str, Any]:
    hits = candidate_pairs & same_pairs
    return {
        "candidate_pairs": int(len(candidate_pairs)),
        "all_possible_pairs": int(all_possible),
        "candidate_ratio": round(len(candidate_pairs) / max(1, all_possible), 6),
        "candidate_reduction": round(1.0 - len(candidate_pairs) / max(1, all_possible), 6),
        "human_same_event_denominator": int(len(same_pairs)),
        "human_same_event_hits": int(len(hits)),
        "candidate_recall": round(len(hits) / max(1, len(same_pairs)), 6),
        "missed_same_event_pairs": int(len(same_pairs - candidate_pairs)),
    }


def diagnose_candidates(events: dict[str, dict[str, Any]], event_split: dict[str, str], pair_df: pd.DataFrame, out_dir: Path, explicit_worker: str | None) -> dict[str, Any]:
    dev_ids = sorted(sid for sid, split in event_split.items() if split == "dev")
    dev_events = {sid: events[sid] for sid in dev_ids}
    all_possible = len(dev_ids) * (len(dev_ids) - 1) // 2
    dev_same_pairs = {
        pair_key(row.sample_id_a, row.sample_id_b)
        for row in pair_df.itertuples(index=False)
        if text(row.gold_relation) == "same_event" and event_split.get(text(row.sample_id_a)) == "dev" and event_split.get(text(row.sample_id_b)) == "dev"
    }
    r3_pairs, r3_report = run_r3_ann(dev_events, out_dir, explicit_worker)
    candidate_rows: list[dict[str, Any]] = []
    sets: dict[str, set[tuple[str, str]]] = {r: set() for r in R_ORDER}
    r_reasons: dict[tuple[str, str], dict[str, list[str]]] = {}
    for a, b in combinations(dev_ids, 2):
        ev_a, ev_b = dev_events[a], dev_events[b]
        f = identity_features(ev_a, ev_b)
        conflicts = hard_conflicts(f)
        reasons = {
            "R1": r1_reasons(ev_a, ev_b, f),
            "R2": r2_reasons(ev_a, ev_b, f, conflicts),
            "R4": r4_reasons(ev_a, ev_b, f, conflicts),
            "R3": ["qwen_faiss_dynamic_topk"] if pair_key(a, b) in r3_pairs else [],
        }
        key = pair_key(a, b)
        r_reasons[key] = reasons
        for r, vals in reasons.items():
            if vals:
                sets[r].add(key)
        if any(reasons.values()):
            candidate_rows.append(
                {
                    "sample_id_a": key[0],
                    "sample_id_b": key[1],
                    "R1": bool(reasons["R1"]),
                    "R2": bool(reasons["R2"]),
                    "R3": bool(reasons["R3"]),
                    "R4": bool(reasons["R4"]),
                    "r1_reasons": "|".join(reasons["R1"]),
                    "r2_reasons": "|".join(reasons["R2"]),
                    "r3_reasons": "|".join(reasons["R3"]),
                    "r4_reasons": "|".join(reasons["R4"]),
                    "same_target": bool(f["same_target"]),
                    "same_family": bool(f["same_family"]),
                    "same_sentence": bool(f["same_sentence"]),
                    "same_document_hash": bool(f["same_document_hash"]),
                    "same_event_fact_signature": bool(f["same_event_fact_signature"]),
                    "fact_similarity": round(float(f["fact_similarity"]), 6),
                    "object_similarity": round(float(f["object_similarity"]), 6),
                    "trigger_similarity": round(float(f["trigger_similarity"]), 6),
                    "evidence_similarity": round(float(f["evidence_similarity"]), 6),
                    "time_relation": f["time_relation"],
                    "hard_conflicts": "|".join(conflicts),
                    "gold_relation_if_labeled": "",
                }
            )
    label_lookup = {pair_key(row.sample_id_a, row.sample_id_b): text(row.gold_relation) for row in pair_df.itertuples(index=False)}
    for row in candidate_rows:
        row["gold_relation_if_labeled"] = label_lookup.get(pair_key(row["sample_id_a"], row["sample_id_b"]), "")

    safe_parquet(pd.DataFrame(candidate_rows), out_dir / "r_channel_candidate_pairs_dev.parquet")
    write_csv(out_dir / "r_channel_candidate_pairs_dev.csv", candidate_rows)

    metrics: dict[str, Any] = {}
    for r in R_ORDER:
        metrics[r] = candidate_metrics(sets[r], dev_same_pairs, all_possible)
    union = set().union(*(sets[r] for r in R_ORDER)) if sets else set()
    metrics["R1_R2_R3_R4_union"] = candidate_metrics(union, dev_same_pairs, all_possible)
    incremental_sets = {
        "R1": sets["R1"],
        "R1_plus_R2_plus_R4": sets["R1"] | sets["R2"] | sets["R4"],
        "R1_plus_R2_plus_R4_plus_R3": sets["R1"] | sets["R2"] | sets["R4"] | sets["R3"],
    }
    incremental = {name: candidate_metrics(vals, dev_same_pairs, all_possible) for name, vals in incremental_sets.items()}
    channel_incremental: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str]] = set()
    for r in R_ORDER:
        new_pairs = sets[r] - seen
        channel_incremental[r] = {
            "new_unique_pair_count": int(len(new_pairs)),
            "new_human_same_event_hits": int(len(new_pairs & dev_same_pairs)),
        }
        seen |= sets[r]

    same_detail: list[dict[str, Any]] = []
    for key in sorted(dev_same_pairs):
        reasons = r_reasons.get(key, {r: [] for r in R_ORDER})
        same_detail.append(
            {
                "sample_id_a": key[0],
                "sample_id_b": key[1],
                "hit_any": key in union,
                "R1": bool(reasons.get("R1")),
                "R2": bool(reasons.get("R2")),
                "R3": bool(reasons.get("R3")),
                "R4": bool(reasons.get("R4")),
                "r1_reasons": "|".join(reasons.get("R1", [])),
                "r2_reasons": "|".join(reasons.get("R2", [])),
                "r3_reasons": "|".join(reasons.get("R3", [])),
                "r4_reasons": "|".join(reasons.get("R4", [])),
            }
        )
    write_csv(out_dir / "dev_same_event_candidate_recall_detail.csv", same_detail)

    report = {
        "stage": "split_and_recall_diagnosis_only_no_cross_encoder_no_final_holdout_metrics",
        "dev_event_count": int(len(dev_ids)),
        "dev_all_possible_pairs": int(all_possible),
        "dev_human_same_event_pair_denominator": int(len(dev_same_pairs)),
        "r3_ann_report": {k: v for k, v in r3_report.items() if k != "pairs"},
        "r_channel_independent_metrics": metrics,
        "r_channel_incremental_new_counts": channel_incremental,
        "incremental_experiments": incremental,
        "recall_gate": {
            "candidate_recall_min": 0.85,
            "candidate_reduction_min": 0.99,
            "passed_by_union": bool(
                metrics["R1_R2_R3_R4_union"]["candidate_recall"] >= 0.85
                and metrics["R1_R2_R3_R4_union"]["candidate_reduction"] >= 0.99
                and r3_report.get("valid_r3_backend", False)
            ),
            "not_passed_reason": "" if (
                metrics["R1_R2_R3_R4_union"]["candidate_recall"] >= 0.85
                and metrics["R1_R2_R3_R4_union"]["candidate_reduction"] >= 0.99
                and r3_report.get("valid_r3_backend", False)
            ) else "recall_or_reduction_gate_failed_or_r3_backend_invalid",
        },
        "decision_pairwise_metrics": "not_evaluated_current_stage_no_cross_encoder_no_threshold_selection",
        "b3_metrics": "not_evaluated_pair_labels_only_cluster_gold_not_established",
    }
    write_json(out_dir / "candidate_recall_diagnosis.json", report)
    return report


def write_readme(out_dir: Path) -> None:
    readme = """# Task3 Resolver v3 Candidate Development

This directory contains the non-official Resolver v3 split freeze and candidate-recall diagnostics.

Current stage only freezes the repair split and evaluates R1/R2/R3/R4 candidate recall on the frozen dev split. It does not train a Cross-Encoder, does not update KG/graph outputs, does not replace Resolver v2, and does not evaluate final_holdout.

Reproduce:

```bash
/root/miniconda3/envs/dianjin-cu128/bin/python scripts/build_task3_resolver_v3_candidate.py --out outputs/task3_resolver_v3_candidate
```

R channels:

- R1: strong deterministic keys with target compatibility.
- R2: structured lexical rules with target/family compatibility and no hard conflict.
- R3: local Qwen3-Embedding-0.6B + FAISS dynamic Top-K dense retriever.
- R4: constrained legacy source/time/modality support; same_sentence is not standalone recall evidence.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def write_repro_script(out_dir: Path) -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail
cd /opt/data/private/JinRong/kechuang/financial_event_project
/root/miniconda3/envs/dianjin-cu128/bin/python scripts/build_task3_resolver_v3_candidate.py --out outputs/task3_resolver_v3_candidate
"""
    path = out_dir / "reproduce_split_and_recall_diagnosis.sh"
    path.write_text(script, encoding="utf-8")
    os.chmod(path, 0o755)


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_parquet(REPAIR_PAIRS)
    v2_df = pd.read_parquet(V2_CANDIDATES)
    events = build_events(pair_df)
    canonical_map, isolation_map, canonical_edges, isolation_edges = build_components(events, pair_df)
    canonical_dist = component_distribution(canonical_map)
    isolation_dist = component_distribution(isolation_map)
    max_allowed = max(50, int(len(events) * 0.20))
    abnormal = isolation_dist["max_size"] > max_allowed
    if abnormal:
        write_json(
            out_dir / "split_manifest.json",
            {
                "status": "blocked_max_component_abnormal_split_not_frozen",
                "event_count": len(events),
                "max_allowed_component_size": max_allowed,
                "canonical_component_distribution": canonical_dist,
                "isolation_component_distribution": isolation_dist,
            },
        )
        raise RuntimeError(f"max isolation component {isolation_dist['max_size']} exceeds {max_allowed}; split not frozen")

    event_split = assign_splits(events, isolation_map, pair_df)
    event_rows = event_mapping_rows(events, event_split, canonical_map, isolation_map)
    event_map_df = pd.DataFrame(event_rows)
    pair_split_df = split_pair_rows(pair_df, event_split)
    safe_parquet(event_map_df, out_dir / "event_split_mapping.parquet")
    safe_parquet(pair_split_df, out_dir / "repair_pair_split_mapping.parquet")
    write_csv(out_dir / "event_split_mapping.csv", event_rows)
    write_csv(out_dir / "repair_pair_split_mapping.csv", pair_split_df.to_dict(orient="records"))
    write_jsonl(out_dir / "component_edges_canonical.jsonl", canonical_edges)
    write_jsonl(out_dir / "component_edges_isolation.jsonl", isolation_edges)

    leakage = split_leakage_audit(events, event_split, canonical_map, isolation_map, pair_df)
    write_json(out_dir / "split_leakage_audit.json", leakage)
    counts = split_counts(event_map_df, pair_split_df)

    input_manifest = {
        "status": "inputs_frozen_no_official1799_read",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            rel(REPAIR_PAIRS): {"sha256": sha256_file(REPAIR_PAIRS), "rows": int(len(pair_df))},
            rel(V2_CANDIDATES): {"sha256": sha256_file(V2_CANDIDATES), "rows": int(len(v2_df))},
            rel(MANUAL_ADJUDICATION): {"sha256": sha256_file(MANUAL_ADJUDICATION) if MANUAL_ADJUDICATION.exists() else "", "rows": "not_read_current_stage"},
            rel(TASK4_EVENT_INTERFACE): {"sha256": sha256_file(TASK4_EVENT_INTERFACE) if TASK4_EVENT_INTERFACE.exists() else "", "rows": "not_read_current_stage"},
            rel(TASK4_FEATURE_TABLE): {"sha256": sha256_file(TASK4_FEATURE_TABLE) if TASK4_FEATURE_TABLE.exists() else "", "rows": "not_read_current_stage"},
        },
        "explicitly_not_read": [
            "outputs/task34_official1799_final_evaluation_v1",
            "outputs/task4 golddata/*official1799*_filled*",
            "official1799 filled human gold files",
        ],
        "script": rel(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(out_dir / "input_manifest.json", input_manifest)
    write_json(
        out_dir / "gold_provenance.json",
        {
            "status": "repair_pair_gold_only",
            "repair_pair_rows": int(len(pair_df)),
            "gold_relation_distribution": {k: int(v) for k, v in pair_df["gold_relation"].value_counts().to_dict().items()},
            "gold_relation_origin_distribution": {k: int(v) for k, v in pair_df["gold_relation_origin"].value_counts().to_dict().items()},
            "official1799_gold_read": 0,
            "weak_labels_used_as_gold": 0,
            "final_holdout_usage": "split_membership_frozen_only_no_candidate_or_decision_metrics",
        },
    )

    split_manifest = {
        "status": "repair_split_frozen",
        "event_count": int(len(events)),
        "repair_pair_count": int(len(pair_df)),
        "canonical_component_distribution": canonical_dist,
        "isolation_component_distribution": isolation_dist,
        "max_component_abnormal": False,
        "max_allowed_component_size": max_allowed,
        "split_counts": counts,
        "leakage_summary": leakage,
        "artifact_sha256": {
            "event_split_mapping.parquet": sha256_file(out_dir / "event_split_mapping.parquet"),
            "repair_pair_split_mapping.parquet": sha256_file(out_dir / "repair_pair_split_mapping.parquet"),
            "component_edges_canonical.jsonl": sha256_file(out_dir / "component_edges_canonical.jsonl"),
            "component_edges_isolation.jsonl": sha256_file(out_dir / "component_edges_isolation.jsonl"),
        },
    }
    write_json(out_dir / "split_manifest.json", split_manifest)

    v2_audit = v2_candidate_dedup_audit(v2_df, pair_df, out_dir)
    recall_report = diagnose_candidates(events, event_split, pair_df, out_dir, args.r3_worker_python)
    write_readme(out_dir)
    write_repro_script(out_dir)

    phase_summary = {
        "status": "split_and_r_recall_diagnosis_complete",
        "input_size": {"events": int(len(events)), "repair_pairs": int(len(pair_df)), "v2_candidate_rows": int(len(v2_df))},
        "positive_negative_counts": {
            "all_repair": {k: int(v) for k, v in pair_df["gold_relation"].value_counts().to_dict().items()},
            "dev_label_rows": counts["dev"]["gold_relation_distribution"],
            "dev_unique_unordered_pairs": counts["dev"]["unique_gold_relation_distribution"],
            "dev_human_same_event_recall_denominator_unique_pairs": recall_report["dev_human_same_event_pair_denominator"],
        },
        "leakage_check": {
            "human_same_event_cross_split": leakage["human_same_event"]["cross_split"],
            "article_cross_split_groups": leakage["article"]["cross_split_group_count"],
            "content_hash_cross_split_groups": leakage["content_hash"]["cross_split_group_count"],
            "event_fact_signature_cross_split_groups": leakage["event_fact_signature"]["cross_split_group_count"],
            "official1799_read": 0,
        },
        "candidate_count": recall_report["r_channel_independent_metrics"]["R1_R2_R3_R4_union"]["candidate_pairs"],
        "candidate_recall": recall_report["r_channel_independent_metrics"]["R1_R2_R3_R4_union"]["candidate_recall"],
        "candidate_reduction": recall_report["r_channel_independent_metrics"]["R1_R2_R3_R4_union"]["candidate_reduction"],
        "pairwise_metrics": recall_report["decision_pairwise_metrics"],
        "split_false_merge": {
            "split_error": recall_report["r_channel_independent_metrics"]["R1_R2_R3_R4_union"]["missed_same_event_pairs"],
            "false_merge": "not_evaluated_current_stage_candidate_generation_only",
        },
        "current_blocker": recall_report["recall_gate"]["not_passed_reason"],
        "artifact_path": rel(out_dir),
        "v2_unique_pair_audit": {
            "unique_pairs": v2_audit["unique_unordered_pair_count"],
            "duplicate_rows": v2_audit["duplicate_pair_rows"],
            "candidate_reduction_over_515_full_combination": v2_audit["candidate_reduction_over_515_full_combination"],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(out_dir / "phase_summary_split_and_recall.json", phase_summary)
    print(json.dumps(phase_summary, ensure_ascii=False, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--r3-worker-python", default=None)
    parser.add_argument("--ann-worker", action="store_true")
    parser.add_argument("--ann-input", default="")
    parser.add_argument("--ann-output", default="")
    args = parser.parse_args()
    if args.ann_worker and (not args.ann_input or not args.ann_output):
        parser.error("--ann-worker requires --ann-input and --ann-output")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.ann_worker:
        ann_worker(parsed)
    else:
        main(parsed)
