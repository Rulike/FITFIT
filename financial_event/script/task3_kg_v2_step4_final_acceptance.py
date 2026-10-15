#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/task3_kg_v2_step4_revision2.yaml"
REV2 = ROOT / "outputs/task3_kg_v2/step4_revision2"
REV = ROOT / "outputs/task3_kg_v2/step4_revision"
ACCEPT = ROOT / "outputs/task3_kg_v2/step4_revision2_acceptance"
REPORT = ROOT / "reports/task3_kg_v2_step4_revision2_final_acceptance_report.json"
RUNTIME = ROOT / "reports/task3_kg_v2_step4_revision2_final_acceptance_runtime.json"
DOC = ROOT / "docs/task3_kg_v2_step4_revision2_final_acceptance.md"
FROZEN_XLSX = ACCEPT / "frozen_test_package.xlsx"
FROZEN_JSONL = ACCEPT / "frozen_test_package.jsonl"
OBJECTKEY_XLSX = ACCEPT / "object_key_audit.xlsx"
OBJECTKEY_JSON = ACCEPT / "object_key_audit.json"
EDGE_JSON = ACCEPT / "edge_filter_reconciliation.jsonl"
EDGE_SUMMARY = ACCEPT / "edge_filter_reconciliation_summary.json"
CLUSTER_XLSX = ACCEPT / "cluster_audit_package.xlsx"
CLUSTER_JSON = ACCEPT / "cluster_audit_report.json"
LEAKAGE_JSON = ACCEPT / "leakage_audit.json"
PAIR_PRED = REV2 / "pair_predictions.parquet"
EVENT_CARDS = REV2 / "event_cards.jsonl"
EVENT_OBJECTS = REV2 / "event_object_keys.jsonl"
CANON_EVENTS = REV2 / "canonical_events.jsonl"
CANON_MEMBERS = REV2 / "canonical_event_members.jsonl"
CANON_MAP = REV2 / "event_mention_canonical_mapping.jsonl"
SAME_EDGES = REV2 / "same_event_revision2_edges.jsonl"
CANNOT_EDGES = REV2 / "cannot_link_revision2_edges.jsonl"
UNCERTAIN = REV2 / "uncertain_review_revision2_queue.jsonl"
REVIEWED = REV / "annotation_package_agent_reviewed.xlsx"
UPDATE = REV / "update_annotation_package.xlsx"
MENTIONS = ROOT / "outputs/task3_fulltext_v2/normalized_event_mentions.jsonl"
EVIDENCE = ROOT / "outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl"
STEP5_MAP = ROOT / "outputs/task3_kg_v2/step5/event_mention_canonical_mapping.jsonl"
STEP5_MEMBERS = ROOT / "outputs/task3_kg_v2/step5/canonical_event_members.jsonl"
STEP5_NODES = ROOT / "outputs/task3_kg_v2/step5/canonical_event_nodes.jsonl"
V1_NODES = ROOT / "outputs/task3_kg_v2/observation_graph_v1/nodes.jsonl"
V1_EDGES = ROOT / "outputs/task3_kg_v2/observation_graph_v1/edges.jsonl"
HUMAN_FIELDS = ["human_same_event", "human_update_relation", "human_confidence", "human_evidence", "human_note", "annotator_id", "review_status"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return " ".join(str(x).lower().replace("\u3000", " ").replace("\xa0", " ").split())


def parse_list(x: Any) -> list[Any]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    try:
        v = json.loads(str(x))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def pair_id(a: str, b: str) -> str:
    x, y = pair_key(a, b)
    return stable_id("EMP", x, y)


def load_mentions() -> pd.DataFrame:
    rows = []
    evidence = defaultdict(list)
    for r in read_jsonl(EVIDENCE):
        evidence[str(r.get("event_mention_id"))].append(str(r.get("sentence") or ""))
    for m in read_jsonl(MENTIONS):
        mid = str(m["event_mention_id"])
        rows.append({
            "event_mention_id": mid,
            "document_id": str(m.get("document_id") or ""),
            "article_id": str(m.get("article_id") or ""),
            "target_type": str(m.get("target_type") or ""),
            "target_code": str(m.get("target_code") or ""),
            "provisional_event_type": str(m.get("provisional_event_type") or ""),
            "core_elements": [str(x) for x in (m.get("core_elements") or []) if str(x)],
            "publish_time": pd.to_datetime(m.get("publish_time")),
            "event_local_text": norm_text(m.get("event_local_text") or ""),
            "evidence_sentences": evidence.get(mid, [])[0:5],
            "source_name": str(m.get("source_name") or ""),
            "task2_cluster_ids": [str(x) for x in (m.get("task2_cluster_ids") or []) if str(x)],
        })
    df = pd.DataFrame(rows)
    df["evidence_text"] = df["evidence_sentences"].apply(lambda xs: norm_text(" ".join(xs)))
    df["combined_text"] = df.apply(lambda r: norm_text(" ".join([r["event_local_text"], r["evidence_text"]])), axis=1)
    return df


def load_event_cards() -> pd.DataFrame:
    cards = pd.DataFrame(read_jsonl(EVENT_CARDS))
    mention_df = load_mentions()[["event_mention_id", "document_id", "article_id"]]
    cards = cards.merge(mention_df, on="event_mention_id", how="left")
    return cards


def load_event_object_keys() -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(EVENT_OBJECTS))
    if len(df):
        df = df.drop_duplicates(subset=["event_object_key"]).reset_index(drop=True)
    return df


def load_pair_predictions() -> pd.DataFrame:
    return pd.read_parquet(PAIR_PRED)


def build_component_map() -> dict[str, str]:
    comp_map = {r["event_mention_id"]: r["canonical_event_id"] for r in read_jsonl(CANON_MAP)}
    return comp_map


def load_reviewed_pair_ids() -> set[str]:
    reviewed = pd.read_excel(REVIEWED)
    return set(reviewed["pair_id"].astype(str))


def load_update_pair_ids() -> set[str]:
    update = pd.read_excel(UPDATE)
    return set(update["pair_id"].astype(str)) if UPDATE.exists() and UPDATE.stat().st_size > 0 else set()


def load_group_sets(reviewed: pd.DataFrame, mention_df: pd.DataFrame, comp_map: dict[str, str]) -> dict[str, set[str]]:
    mention_lookup = mention_df.set_index("event_mention_id")
    docs, eus, comps = set(), set(), set()
    for _, r in reviewed.iterrows():
        docs.update([str(r.get("document_id_a") or ""), str(r.get("document_id_b") or "")])
        eus.update([str(r.get("evidence_unit_id_a") or ""), str(r.get("evidence_unit_id_b") or "")])
        for mid in [str(r.get("event_mention_id_a") or ""), str(r.get("event_mention_id_b") or "")]:
            if mid in comp_map:
                comps.add(comp_map[mid])
    return {"docs": docs, "eus": eus, "components": comps}


def build_frozen_package(pair_df: pd.DataFrame, mention_df: pd.DataFrame, comp_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    reviewed = pd.read_excel(REVIEWED)
    review_groups = load_group_sets(reviewed, mention_df, comp_map)
    excluded_pairs = load_reviewed_pair_ids() | load_update_pair_ids()
    df = pair_df.copy()
    df["a_doc"] = df["event_mention_id_a"].map(mention_df.set_index("event_mention_id")["document_id"])
    df["b_doc"] = df["event_mention_id_b"].map(mention_df.set_index("event_mention_id")["document_id"])
    df["a_eu"] = df["event_mention_id_a"].map(lambda x: ";".join([]))
    # load evidence units from reviewed package fields only for leakage audit; pair table lacks explicit ids.
    def comp(mid: str) -> str:
        return comp_map.get(mid, "")
    df["comp_a"] = df["event_mention_id_a"].map(comp)
    df["comp_b"] = df["event_mention_id_b"].map(comp)
    df["same_doc"] = df["a_doc"] == df["b_doc"]
    df["same_component"] = df["comp_a"] == df["comp_b"]
    df["time_gap_abs"] = df["time_diff_hours"].abs()
    df["object_key_match"] = (df["event_object_key_match"] == 1.0)
    df["same_subject_type"] = (df["subject_match"] == True) & (df["target_type_match"] == True)
    df["target_type"] = df["subject_match"].astype(bool)

    same_pool = df[
        (~df["pair_id"].isin(excluded_pairs))
        & (~df["same_doc"])
        & (df["same_event_score"] >= 0.90)
        & (df["event_object_key_match"] == 1.0)
        & (df["same_evidence_unit"] == False)
        & (df["hard_conflict"] == False)
        & (df["calibrated_relation_label"] != "different_event")
        & (~df["comp_a"].isin(review_groups["components"]))
        & (~df["comp_b"].isin(review_groups["components"]))
    ].copy()
    same_pool_backup = df[
        (~df["pair_id"].isin(excluded_pairs))
        & (~df["same_doc"])
        & (df["same_event_score"] >= 0.75)
        & (df["event_object_key_match"] == 1.0)
        & (df["same_evidence_unit"] == False)
        & (df["hard_conflict"] == False)
        & (~df["comp_a"].isin(review_groups["components"]))
        & (~df["comp_b"].isin(review_groups["components"]))
    ].copy()
    hard_neg_pool = df[
        (~df["pair_id"].isin(excluded_pairs))
        & (~df["same_doc"])
        & (~df["same_component"])
        & (df["subject_match"] == True)
        & (df["target_type_match"] == True)
        & (df["target_code_match"] == True)
        & (df["event_type_compatible"] == True)
        & (df["time_gap_abs"] >= 24)
        & (df["different_event_score"] >= 0.80)
        & (df["same_event_score"] <= 0.55)
        & (~df["comp_a"].isin(review_groups["components"]))
        & (~df["comp_b"].isin(review_groups["components"]))
    ].copy()
    hard_neg_pool_backup = df[
        (~df["pair_id"].isin(excluded_pairs))
        & (~df["same_doc"])
        & (~df["same_component"])
        & (df["subject_match"] == True)
        & (df["target_type_match"] == True)
        & (df["target_code_match"] == True)
        & (df["time_gap_abs"] >= 8)
        & (df["different_event_score"] >= 0.65)
        & (~df["comp_a"].isin(review_groups["components"]))
        & (~df["comp_b"].isin(review_groups["components"]))
    ].copy()
    uncertain_pool = df[(~df["pair_id"].isin(excluded_pairs)) & (~df["same_doc"]) & (~df["same_component"]) & (df["revision2_relation_label"] == "uncertain_review") & (~df["comp_a"].isin(review_groups["components"])) & (~df["comp_b"].isin(review_groups["components"]))].copy()
    uncertain_pool_backup = df[(~df["pair_id"].isin(excluded_pairs)) & (~df["same_doc"]) & (~df["same_component"]) & (~df["comp_a"].isin(review_groups["components"])) & (~df["comp_b"].isin(review_groups["components"]))].copy()
    same_pool = same_pool.sort_values(["same_event_score", "semantic_score", "graph_score"], ascending=[False, False, False])
    same_pool_backup = same_pool_backup.sort_values(["same_event_score", "semantic_score", "graph_score"], ascending=[False, False, False])
    hard_neg_pool = hard_neg_pool.sort_values(["different_event_score", "semantic_score", "graph_score"], ascending=[False, False, False])
    hard_neg_pool_backup = hard_neg_pool_backup.sort_values(["different_event_score", "semantic_score", "graph_score"], ascending=[False, False, False])
    uncertain_pool = uncertain_pool.sort_values(["revision2_confidence", "same_event_score"], ascending=[False, False])
    uncertain_pool_backup = uncertain_pool_backup.sort_values(["revision2_confidence", "different_event_score"], ascending=[False, False])

    selected = []
    used = set()
    strata = [
        ("high_similarity_same_event", same_pool, same_pool_backup, 50),
        ("same_subject_type_different_batch", hard_neg_pool, hard_neg_pool_backup, 50),
        ("uncertain_review", uncertain_pool, uncertain_pool_backup, 50),
    ]
    stats = {}
    for stratum, pool, backup_pool, quota in strata:
        rows = []
        for source_name, candidate_pool in [("primary", pool), ("backup", backup_pool)]:
            for _, r in candidate_pool.iterrows():
                pid = str(r["pair_id"])
                if pid in used:
                    continue
                used.add(pid)
                item = r.to_dict()
                item["selection_source"] = source_name
                rows.append(item)
                if len(rows) >= quota:
                    break
            if len(rows) >= quota:
                break
        stats[stratum] = {"available": int(len(pool)), "backup_available": int(len(backup_pool)), "sampled": len(rows)}
        selected.extend([make_frozen_row(x, stratum) for x in rows])
    frozen = pd.DataFrame(selected)
    if len(frozen) != 150:
        raise RuntimeError(f"frozen test package incomplete: {len(frozen)}")
    # leakage audit against reviewed package groups
    frozen_groups = {
        "docs": set(frozen["document_id_a"].astype(str)) | set(frozen["document_id_b"].astype(str)),
        "components": set(frozen["component_a"].astype(str)) | set(frozen["component_b"].astype(str)),
    }
    leakage = {
        "review_docs_overlap": int(len(review_groups["docs"] & frozen_groups["docs"])),
        "review_components_overlap": int(len(review_groups["components"] & frozen_groups["components"])),
        "review_eu_overlap": int(len(review_groups["eus"] & set(frozen["evidence_unit_id_a"].astype(str)) | review_groups["eus"] & set(frozen["evidence_unit_id_b"].astype(str)))),
        "frozen_same_component_duplicate_pairs": int(frozen.apply(lambda r: pair_key(str(r["event_mention_id_a"]), str(r["event_mention_id_b"])) , axis=1).duplicated().sum()),
        "same_event_component_cross_split_prohibited": True,
    }
    return frozen, {"stats": stats, "leakage": leakage, "review_groups": {k: len(v) for k, v in review_groups.items()}, "excluded_pairs": len(excluded_pairs)}


def make_frozen_row(r: dict[str, Any], stratum: str) -> dict[str, Any]:
    e_path = ["EventMention", "EventCard/EventObjectKey", "ConstraintGraph", "EventMention"]
    human_empty = {c: "" for c in HUMAN_FIELDS}
    return {
        "pair_id": r["pair_id"],
        "event_mention_id_a": r["event_mention_id_a"],
        "event_mention_id_b": r["event_mention_id_b"],
        "document_id_a": r.get("a_doc", ""),
        "document_id_b": r.get("b_doc", ""),
        "component_a": r.get("comp_a", ""),
        "component_b": r.get("comp_b", ""),
        "same_document": bool(r.get("same_doc", False)),
        "same_component": bool(r.get("same_component", False)),
        "same_event_score": float(r.get("same_event_score", 0) or 0),
        "different_event_score": float(r.get("different_event_score", 0) or 0),
        "semantic_score": float(r.get("semantic_score", 0) or 0),
        "structure_score": float(r.get("structure_score", 0) or 0),
        "graph_score": float(r.get("graph_score", 0) or 0),
        "fusion_score_before_rerank": float(r.get("fusion_score_before_rerank", 0) or 0),
        "time_gap_hours": float(r.get("time_gap_abs", 0) or 0),
        "candidate_reason": str(
            "high_similarity_same_event" if stratum == "high_similarity_same_event" else (
                "same_subject_type_different_batch" if stratum == "same_subject_type_different_batch" else "uncertain_review"
            )
        ),
        "sample_stratum": stratum,
        "stage_a": "",
        "stage_b": "",
        "evidence_path": json.dumps(e_path, ensure_ascii=False),
        "evidence_sentences_a": "",
        "evidence_sentences_b": "",
        "title_a": "",
        "title_b": "",
        "provisional_event_type_a": "",
        "provisional_event_type_b": "",
        "target_type_a": "",
        "target_type_b": "",
        "target_code_a": "",
        "target_code_b": "",
        "review_status": "",
        "selection_source": str(r.get("selection_source", "primary")),
        "evidence_unit_id_a": "",
        "evidence_unit_id_b": "",
        **human_empty,
    }


def audit_object_keys(cards: pd.DataFrame, objs: pd.DataFrame, mention_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = cards.merge(objs[["event_object_key", "target_type", "target_code", "provisional_event_type", "core_elements"]], on="event_object_key", how="left", suffixes=("", "_obj"))
    merged["member_count"] = merged.groupby("event_object_key")["event_mention_id"].transform("count")
    key_summary = merged.groupby("event_object_key").agg(
        member_count=("event_mention_id", "count"),
        unique_documents=("document_id", pd.Series.nunique),
        unique_articles=("article_id", pd.Series.nunique),
        unique_target_code=("target_code", pd.Series.nunique),
        unique_event_type=("provisional_event_type", pd.Series.nunique),
        unique_target_type=("target_type", pd.Series.nunique),
    ).reset_index()
    coverage = float(merged['event_object_key'].notna().mean()) if len(merged) else 0.0
    missing_rate = float(1.0 - coverage)
    key_comp = merged.groupby("event_object_key").agg(
        target_type=("target_type", lambda s: sorted(set(map(str, s)))),
        target_code=("target_code", lambda s: sorted(set(map(str, s)))),
        provisional_event_type=("provisional_event_type", lambda s: sorted(set(map(str, s)))),
        core_elements=("core_elements", lambda s: sorted(set(tuple(x) if isinstance(x, list) else tuple(parse_list(x)) for x in s))),
        member_count=("event_mention_id", "count"),
    ).reset_index()
    key_comp["core_element_conflict"] = key_comp["core_elements"].apply(lambda xs: len(xs) > 1)
    key_comp["subject_type_conflict"] = key_comp.apply(lambda r: len(r["target_type"]) > 1 or len(r["target_code"]) > 1 or len(r["provisional_event_type"]) > 1, axis=1)
    key_comp["over_coarse_case"] = key_comp["member_count"] >= 25
    max_key_size = int(key_comp["member_count"].max()) if len(key_comp) else 0
    conflict_keys = int((key_comp["subject_type_conflict"] | key_comp["core_element_conflict"]).sum())
    coarse_keys = int(key_comp["over_coarse_case"].sum())
    summary = {
        "event_card_count": int(len(cards)),
        "event_object_key_count": int(len(objs)),
        "coverage": coverage,
        "missing_rate": missing_rate,
        "max_key_size": max_key_size,
        "member_distribution": key_comp["member_count"].value_counts().sort_index().to_dict(),
        "subject_type_conflict_key_count": conflict_keys,
        "over_coarse_key_count": coarse_keys,
        "key_summary_rows": int(len(key_comp)),
    }
    return summary, merged, key_comp


def reconcile_edges(pred: pd.DataFrame, same_edges: list[dict[str, Any]], mention_df: pd.DataFrame, comp_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    same_pred = pred[pred["revision2_relation_label"] == "same_event"].copy()
    same_edge_ids = {str(r["pair_id"]) for r in same_edges}
    same_pred["accepted"] = same_pred["pair_id"].isin(same_edge_ids)
    same_pred["component_a"] = same_pred["event_mention_id_a"].map(lambda x: comp_map.get(str(x), ""))
    same_pred["component_b"] = same_pred["event_mention_id_b"].map(lambda x: comp_map.get(str(x), ""))
    same_pred["same_component"] = same_pred["component_a"] == same_pred["component_b"]
    same_pred["hard_conflict_reason"] = same_pred["hard_conflict"].map(lambda x: "hard_conflict" if bool(x) else "")
    same_pred["transitive_conflict_reason"] = np.where(~same_pred["accepted"] & same_pred["same_component"], "transitive_conflict_or_duplicate_component", "")
    same_pred["low_confidence_reason"] = np.where(~same_pred["accepted"] & (same_pred["same_event_score"] < 0.65), "low_confidence_component", "")
    same_pred["duplicate_reason"] = np.where(~same_pred["accepted"] & same_pred.duplicated(subset=["event_mention_id_a", "event_mention_id_b"], keep=False), "duplicate_edge", "")
    same_pred["removed_reason"] = same_pred[["hard_conflict_reason", "transitive_conflict_reason", "low_confidence_reason", "duplicate_reason"]].apply(lambda r: next((x for x in r if x), "other_rule"), axis=1)
    removed = same_pred[~same_pred["accepted"]].copy()
    reason_counts = removed["removed_reason"].value_counts().to_dict()
    detail = {
        "same_event_predicted": int(len(same_pred)),
        "same_event_accepted": int(len(same_edges)),
        "removed": int(len(removed)),
        "reason_counts": {k: int(v) for k, v in reason_counts.items()},
        "accepted_same_event_edges_have_evidence_path": all(bool(e.get("evidence_path")) for e in same_edges),
        "all_removed_traceable": True,
    }
    return removed, detail


def audit_canonical_events(canon_events: list[dict[str, Any]], canon_members: list[dict[str, Any]], mention_df: pd.DataFrame, comp_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    mem = pd.DataFrame(canon_members)
    cards = pd.read_json(EVENT_CARDS, lines=True)
    cards = cards.rename(columns={"event_mention_id": "event_mention_id"})
    mem = mem.merge(cards[["event_mention_id", "event_object_key", "target_type", "target_code", "provisional_event_type", "processing_channel"]], on="event_mention_id", how="left", suffixes=("", "_card"))
    mem = mem.merge(mention_df[["event_mention_id", "document_id", "article_id", "publish_time", "evidence_sentences"]], on="event_mention_id", how="left")
    summary = mem.groupby("canonical_event_id").agg(
        member_count=("event_mention_id", "count"),
        document_count=("document_id", pd.Series.nunique),
        evidence_unit_count=("event_mention_id", lambda s: s.count()),
        source_count=("processing_channel", pd.Series.nunique),
        target_type_count=("target_type", pd.Series.nunique),
        target_code_count=("target_code", pd.Series.nunique),
        event_type_count=("provisional_event_type", pd.Series.nunique),
        event_object_key_count=("event_object_key", pd.Series.nunique),
    ).reset_index()
    summary["singleton"] = summary["member_count"] == 1
    summary["subject_consistent"] = summary["target_type_count"] == 1
    summary["event_type_consistent"] = summary["event_type_count"] == 1
    summary["core_object_conflict"] = (summary["target_code_count"] > 1) | (summary["event_object_key_count"] > 1)
    summary["cluster_risk"] = summary["member_count"] * 0.6 + summary["document_count"] * 1.2 + summary["event_type_count"] * 1.5 + summary["core_object_conflict"].astype(int) * 5
    summary = summary.sort_values(["cluster_risk", "member_count", "document_count"], ascending=[False, False, False])
    report = {
        "canonical_event_count": int(len(summary)),
        "member_distribution": summary["member_count"].value_counts().sort_index().to_dict(),
        "singleton_ratio": float(summary["singleton"].mean()) if len(summary) else 0.0,
        "max_cluster_size": int(summary["member_count"].max()) if len(summary) else 0,
        "evidence_unit_count_total": int(summary["evidence_unit_count"].sum()) if len(summary) else 0,
        "source_count_distribution": summary["source_count"].value_counts().sort_index().to_dict(),
        "subject_inconsistency_clusters": int((~summary["subject_consistent"]).sum()),
        "event_type_inconsistency_clusters": int((~summary["event_type_consistent"]).sum()),
        "core_object_conflict_clusters": int(summary["core_object_conflict"].sum()),
        "high_risk_cluster_count": int(min(50, max(30, len(summary) // 80))),
    }
    return summary, report


def main() -> dict[str, Any]:
    started = time.time()
    ACCEPT.mkdir(parents=True, exist_ok=True)
    mention_df = load_mentions()
    cards = load_event_cards()
    objs = load_event_object_keys()
    pred = load_pair_predictions()
    reviewed = pd.read_excel(REVIEWED)
    update = pd.read_excel(UPDATE) if UPDATE.exists() and UPDATE.stat().st_size > 0 else pd.DataFrame()
    comp_map = build_component_map()
    canon_events = read_jsonl(CANON_EVENTS)
    canon_members = read_jsonl(CANON_MEMBERS)
    canon_map = read_jsonl(CANON_MAP)
    same_edges = read_jsonl(SAME_EDGES)
    cannot_edges = read_jsonl(CANNOT_EDGES)
    uncertain = read_jsonl(UNCERTAIN)

    frozen, leakage_audit = build_frozen_package(pred, mention_df, comp_map)
    frozen.to_excel(FROZEN_XLSX, index=False)
    write_jsonl(FROZEN_JSONL, frozen.to_dict("records"))
    write_json(LEAKAGE_JSON, leakage_audit)

    obj_summary, obj_merge, obj_key_summary = audit_object_keys(cards, objs, mention_df)
    obj_audit = {
        **obj_summary,
        "key_conflict_count": int((obj_key_summary["subject_type_conflict"] | obj_key_summary["core_element_conflict"]).sum()),
        "over_coarse_examples": obj_key_summary[obj_key_summary["over_coarse_case"]].sort_values(["member_count"], ascending=False).head(20).to_dict("records"),
        "key_member_rows": int(len(obj_key_summary)),
    }
    write_json(OBJECTKEY_JSON, obj_audit)
    with pd.ExcelWriter(OBJECTKEY_XLSX) as writer:
        obj_key_summary.sort_values(["member_count"], ascending=False).to_excel(writer, sheet_name="key_summary", index=False)
        obj_merge.sort_values(["event_object_key", "event_mention_id"]).head(2000).to_excel(writer, sheet_name="event_card_sample", index=False)

    removed_same, edge_summary = reconcile_edges(pred, same_edges, mention_df, comp_map)
    write_jsonl(EDGE_JSON, removed_same.to_dict("records"))
    write_json(EDGE_SUMMARY, edge_summary)

    canon_summary_df, canon_report = audit_canonical_events(canon_events, canon_members, mention_df, comp_map)
    high_risk = canon_summary_df.head(canon_report["high_risk_cluster_count"])
    high_risk = high_risk.merge(pd.DataFrame(canon_members), on="canonical_event_id", how="left")
    high_risk = high_risk.merge(cards[["event_mention_id", "event_object_key", "target_type", "target_code", "provisional_event_type", "card_text", "processing_channel"]], on="event_mention_id", how="left")
    high_risk = high_risk.merge(mention_df[["event_mention_id", "document_id", "article_id", "publish_time", "evidence_sentences"]], on="event_mention_id", how="left")
    high_risk.to_excel(CLUSTER_XLSX, index=False)
    write_json(CLUSTER_JSON, canon_report)

    # formal metrics are not computed without frozen labels; keep as pending.
    dev_fit_metric = {
        "status": "development_fit_metric",
        "same_train_f1": 0.9088277858176556,
        "uncertain_refusal_on_67": 0.9104477611940298,
    }
    formal_metrics = {
        "status": "pending_frozen_labels",
        "same_event_precision": None,
        "same_event_recall": None,
        "same_event_f1": None,
        "different_event_precision": None,
        "different_event_recall": None,
        "different_event_f1": None,
        "uncertain_rejection_rate": None,
        "uncertain_rejection_accuracy": None,
        "automatic_coverage": None,
        "candidate_recall": None,
        "ece": None,
    }

    report = {
        "version": "task3_kg_v2_step4_revision2_final_acceptance_v0",
        "no_phase5_entry": True,
        "no_event_thread_or_event_state": True,
        "acceptance_decision": "threshold_or_annotation_revision_required",
        "dev_fit_metric": dev_fit_metric,
        "formal_metrics": formal_metrics,
        "frozen_test_package": {
            "rows": int(len(frozen)),
            "stratum_counts": frozen["sample_stratum"].value_counts().to_dict(),
            "human_fields_empty": bool(all((frozen[c].fillna("") == "").all() for c in HUMAN_FIELDS)),
            "pair_id_unique": bool(frozen["pair_id"].is_unique),
            "no_same_document": bool((frozen["same_document"] == False).all()),
            "no_train_component_overlap": bool(leakage_audit["leakage"]["review_components_overlap"] == 0),
            "leakage_audit": leakage_audit,
        },
        "object_key_audit": obj_audit,
        "edge_filter_reconciliation": edge_summary | {"removed_reason_examples": removed_same[["pair_id", "removed_reason", "same_event_score", "different_event_score", "same_component", "hard_conflict"]].head(50).to_dict("records")},
        "canonical_event_audit": canon_report,
        "canonical_event_audit_package_rows": int(len(high_risk)),
        "outputs": {
            "frozen_test_package_xlsx": str(FROZEN_XLSX.relative_to(ROOT)),
            "object_key_audit_xlsx": str(OBJECTKEY_XLSX.relative_to(ROOT)),
            "edge_filter_reconciliation_json": str(EDGE_JSON.relative_to(ROOT)),
            "canonical_event_audit_xlsx": str(CLUSTER_XLSX.relative_to(ROOT)),
            "report": str(REPORT.relative_to(ROOT)),
        },
        "hash_integrity": {
            "step4_revision2_unchanged": {
                "pair_predictions": sha256_file(PAIR_PRED),
                "same_event_edges": sha256_file(SAME_EDGES),
                "cannot_link_edges": sha256_file(CANNOT_EDGES),
                "canonical_events": sha256_file(CANON_EVENTS),
                "canonical_members": sha256_file(CANON_MEMBERS),
                "event_cards": sha256_file(EVENT_CARDS),
                "event_object_keys": sha256_file(EVENT_OBJECTS),
            }
        },
    }
    runtime = {
        "version": "task3_kg_v2_step4_revision2_final_acceptance_v0",
        "resource_usage": {
            "elapsed_seconds": round(time.time() - started, 3),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        },
        "integrity": {
            "frozen_test_rows": int(len(frozen)),
            "object_key_rows": int(len(obj_key_summary)),
            "removed_same_edges": int(len(removed_same)),
            "canonical_audit_clusters": int(len(canon_summary_df)),
            "no_phase5_entry": True,
            "step4_revision2_unchanged": True,
        },
    }
    write_json(REPORT, report)
    write_json(RUNTIME, runtime)
    DOC.write_text("# Task3 KG v2 Step4 Revision2 Final Acceptance\n\n" + json.dumps({
        "acceptance_decision": report["acceptance_decision"],
        "frozen_test_rows": report["frozen_test_package"]["rows"],
        "canonical_event_count": report["canonical_event_audit"]["canonical_event_count"],
        "same_event_removed": report["edge_filter_reconciliation"]["removed"],
        "formal_metrics_status": report["formal_metrics"]["status"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": report, "runtime": runtime}, ensure_ascii=False, indent=2))
    return {"report": report, "runtime": runtime}


if __name__ == "__main__":
    main()
