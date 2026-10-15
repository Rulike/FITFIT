#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP4 = ROOT / "outputs/task3_kg_v2/step4"
REV = ROOT / "outputs/task3_kg_v2/step4_revision"
NODES_V2 = ROOT / "outputs/task3_kg_v2/observation_graph_v2/nodes.jsonl"
EDGES_V2 = ROOT / "outputs/task3_kg_v2/observation_graph_v2/edges.jsonl"
CANDIDATES = STEP4 / "same_event_candidates.parquet"
MENTIONS = ROOT / "outputs/task3_fulltext_v2/normalized_event_mentions.jsonl"
EVIDENCE = ROOT / "outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl"
FREEZE_REPORT = REV / "step4_v0_relation_classifier_revision_required.json"
PREV_REPORT = REV / "sampling_report.json"
OUT_XLSX = REV / "update_annotation_package.xlsx"
OUT_REPORT = ROOT / "reports/task3_kg_v2_step4_update_sampling_report.json"
OUT_GUIDE = ROOT / "docs/task3_kg_v2_step4_update_annotation_guide.md"
OUT_ROUTES = REV / "processing_channel_repaired_event_mentions.parquet"
OUT_SUMMARY = REV / "update_annotation_summary.json"
HUMAN_FIELDS = ["human_same_event", "human_update_relation", "human_confidence", "human_evidence", "human_note", "annotator_id", "review_status"]

STAGE_MAP = {
    "rumor_to_confirmation": {
        "stage_a": "rumor",
        "stage_b": "confirmation",
        "source_keywords": ["传闻", "网传", "据悉", "消息称", "爆料", "或将", "可能"],
        "target_keywords": ["官方", "确认", "证实", "公告", "回应", "披露"],
    },
    "announcement_to_progress": {
        "stage_a": "announcement",
        "stage_b": "progress",
        "source_keywords": ["公告", "披露", "宣布", "发布", "通报"],
        "target_keywords": ["进展", "推进", "最新", "跟进", "持续", "更新"],
    },
    "progress_to_completion": {
        "stage_a": "progress",
        "stage_b": "completion",
        "source_keywords": ["进展", "推进", "跟进", "持续", "更新"],
        "target_keywords": ["完成", "达成", "落地", "交割", "获批", "竣工", "实现"],
    },
    "investigation_to_penalty": {
        "stage_a": "investigation",
        "stage_b": "penalty",
        "source_keywords": ["调查", "立案", "核查", "问询", "审查"],
        "target_keywords": ["处罚", "罚款", "处分", "警示", "监管", "问责"],
    },
    "clarification_or_refutation": {
        "stage_a": "report_or_rumor",
        "stage_b": "clarification_or_refutation",
        "source_keywords": ["报道", "消息", "传闻", "网传", "市场消息", "据悉"],
        "target_keywords": ["澄清", "否认", "辟谣", "回应", "说明"],
    },
    "plan_to_termination": {
        "stage_a": "plan",
        "stage_b": "termination",
        "source_keywords": ["拟", "计划", "方案", "预案", "将", "考虑"],
        "target_keywords": ["终止", "取消", "停止", "中止", "撤回", "放弃"],
    },
}
RELATION_QUOTAS = {
    "rumor_to_confirmation": 34,
    "announcement_to_progress": 34,
    "progress_to_completion": 33,
    "investigation_to_penalty": 33,
    "clarification_or_refutation": 33,
    "plan_to_termination": 33,
}
TARGET_SAMPLE_COUNT = 200


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
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
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


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def sentence_fallback(text: str) -> str:
    parts = [p.strip() for p in re_split_sentences(text) if p.strip()]
    return " | ".join(parts[:3])


def re_split_sentences(text: str) -> list[str]:
    import re
    return re.split(r"[。！？；;\n]+", text or "")


def stable_pair_id(a: str, b: str) -> str:
    x, y = sorted([a, b])
    return "UPD_" + hashlib.sha256(f"{x}|{y}".encode("utf-8")).hexdigest()[:24]


def listify_value(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(i) for i in v if str(i)]
    if v is None:
        return []
    try:
        parsed = json.loads(str(v))
        if isinstance(parsed, list):
            return [str(i) for i in parsed if str(i)]
    except Exception:
        pass
    return [s for s in str(v).replace("|", ",").split(",") if s]


def build_mention_frame() -> pd.DataFrame:
    mentions = read_jsonl(MENTIONS)
    evidence_rows = read_jsonl(EVIDENCE)
    nodes = read_jsonl(NODES_V2)
    doc_titles = {str(n.get("article_id")): str(n.get("title") or "") for n in nodes if n.get("node_type") == "Document"}
    doc_stable = {str(n.get("article_id")): str(n.get("stable_id")) for n in nodes if n.get("node_type") == "Document"}
    doc_by_mid: dict[str, str] = {}
    eu_by_doc: dict[str, set[str]] = defaultdict(set)
    for e in read_jsonl(EDGES_V2):
        if e.get("relation_type") == "CONTAINS_MENTION":
            doc_by_mid[str(e.get("target_stable_id"))] = str(e.get("source_stable_id"))
        elif e.get("relation_type") == "BELONGS_TO_EVIDENCE_UNIT":
            eu_by_doc[str(e.get("source_stable_id"))].add(str(e.get("target_stable_id")))
    sent_map: dict[str, list[str]] = defaultdict(list)
    for r in evidence_rows:
        sent_map[str(r.get("event_mention_id"))].append(str(r.get("sentence") or ""))
    rows: list[dict[str, Any]] = []
    for m in mentions:
        mid = str(m["event_mention_id"])
        aid = str(m.get("article_id") or "")
        doc_id = str(m.get("document_id") or "")
        doc_stable_id = doc_by_mid.get(mid, "")
        title = doc_titles.get(aid, "")
        local_text = norm_text(m.get("event_local_text") or "")
        evidence_sentences = sent_map.get(mid) or list(m.get("evidence_sentences") or [])
        evidence_text = norm_text(" ".join(evidence_sentences[:5]))
        combined = norm_text(" ".join([title, local_text, evidence_text]))
        rows.append({
            "event_mention_id": mid,
            "article_id": aid,
            "document_id": doc_id,
            "document_stable_id": doc_stable_id,
            "document_title": title,
            "publish_time": pd.to_datetime(m.get("publish_time")),
            "target_type": str(m.get("target_type") or ""),
            "target_code": str(m.get("target_code") or ""),
            "provisional_event_type": str(m.get("provisional_event_type") or ""),
            "core_elements": listify_value(m.get("core_elements")),
            "event_local_text": local_text,
            "evidence_sentences": evidence_sentences,
            "evidence_text": evidence_text,
            "combined_text": combined,
            "same_document_evidence_unit_ids": sorted(eu_by_doc.get(doc_stable_id, set())),
            "same_document_id": doc_id,
            "source_name": str(m.get("source_name") or ""),
        })
    df = pd.DataFrame(rows)
    df["stage_rumor"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["rumor_to_confirmation"]["source_keywords"]))
    df["stage_confirmation"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["rumor_to_confirmation"]["target_keywords"]))
    df["stage_announcement"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["announcement_to_progress"]["source_keywords"]))
    df["stage_progress"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["announcement_to_progress"]["target_keywords"]) or contains_any(t, STAGE_MAP["progress_to_completion"]["source_keywords"]))
    df["stage_completion"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["progress_to_completion"]["target_keywords"]))
    df["stage_investigation"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["investigation_to_penalty"]["source_keywords"]))
    df["stage_penalty"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["investigation_to_penalty"]["target_keywords"]))
    df["stage_clarification_or_refutation"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["clarification_or_refutation"]["target_keywords"]))
    df["stage_report_or_rumor"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["clarification_or_refutation"]["source_keywords"]))
    df["stage_plan"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["plan_to_termination"]["source_keywords"]))
    df["stage_termination"] = df["combined_text"].apply(lambda t: contains_any(t, STAGE_MAP["plan_to_termination"]["target_keywords"]))
    df["evidence_unit_ids"] = df["same_document_evidence_unit_ids"]
    return df


def build_pair_frame(mention_df: pd.DataFrame) -> pd.DataFrame:
    cand = pd.read_parquet(CANDIDATES)
    cols = [
        "pair_id", "event_mention_id_a", "event_mention_id_b",
        "semantic_score", "title_similarity", "local_evidence_similarity",
        "core_element_similarity", "subject_match", "target_type_match",
        "target_code_match", "event_type_compatible", "company_overlap",
        "stock_overlap", "industry_overlap", "shared_entity_count",
        "shared_entity_jaccard", "time_diff_hours", "same_evidence_unit",
        "shared_evidence_document_count", "task2_assessment_neighbor_overlap",
        "evidence_independent_unit_count", "graph_common_neighbor_count",
        "graph_neighborhood_jaccard", "structure_score", "graph_score",
        "fusion_score_before_rerank", "hard_conflict", "candidate_source_class",
        "calibrated_relation_label", "calibrated_edge_relation_type",
        "calibrated_confidence", "calibrated_decision_reason",
        "constraint_results_json", "candidate_sources_json",
    ]
    cand = cand[[c for c in cols if c in cand.columns]].copy()
    a = mention_df.add_prefix("a_")
    b = mention_df.add_prefix("b_")
    cand = cand.merge(a, left_on="event_mention_id_a", right_on="a_event_mention_id", how="left")
    cand = cand.merge(b, left_on="event_mention_id_b", right_on="b_event_mention_id", how="left")
    cand["same_document"] = cand["a_document_id"] == cand["b_document_id"]
    cand["same_evidence_unit_any"] = cand["same_evidence_unit"]
    cand["time_gap_hours_abs"] = cand["time_diff_hours"].abs()
    cand["ordered_time_ok"] = cand["a_publish_time"] < cand["b_publish_time"]
    cand["pair_unique_key"] = cand.apply(lambda r: stable_pair_id(str(r["event_mention_id_a"]), str(r["event_mention_id_b"])), axis=1)
    cand["a_evidence_unit_ids"] = cand["a_evidence_unit_ids"].apply(parse_list)
    cand["b_evidence_unit_ids"] = cand["b_evidence_unit_ids"].apply(parse_list)
    return cand


def score_stage_pair_pool(cand: pd.DataFrame, relation: str) -> pd.DataFrame:
    cfg = STAGE_MAP[relation]
    if relation == "rumor_to_confirmation":
        mask = cand["a_stage_rumor"] & cand["b_stage_confirmation"]
    elif relation == "announcement_to_progress":
        mask = cand["a_stage_announcement"] & cand["b_stage_progress"]
    elif relation == "progress_to_completion":
        mask = cand["a_stage_progress"] & cand["b_stage_completion"]
    elif relation == "investigation_to_penalty":
        mask = cand["a_stage_investigation"] & cand["b_stage_penalty"]
    elif relation == "clarification_or_refutation":
        mask = cand["a_stage_report_or_rumor"] & cand["b_stage_clarification_or_refutation"]
    elif relation == "plan_to_termination":
        mask = cand["a_stage_plan"] & cand["b_stage_termination"]
    else:
        mask = pd.Series(False, index=cand.index)
    pool = cand[mask].copy()
    pool = pool[
        (pool["a_target_code"] == pool["b_target_code"])
        & (pool["a_target_type"] == pool["b_target_type"])
        & (~pool["same_document"])
        & (~pool["same_evidence_unit_any"])
        & (pool["ordered_time_ok"])
        & (pool["title_similarity"] < 0.95)
        & (pool["local_evidence_similarity"] < 0.95)
        & (pool["candidate_source_class"].isin(["graph_only", "graph_and_semantic"]))
        & (~pool["hard_conflict"])
    ].copy()
    if relation == "clarification_or_refutation":
        pool = pool[pool["semantic_score"] >= 0.25].copy()
    if relation in {"rumor_to_confirmation", "announcement_to_progress", "progress_to_completion", "investigation_to_penalty", "plan_to_termination"}:
        pool = pool[pool["semantic_score"] >= 0.20].copy()
    pool["stage_a"] = cfg["stage_a"]
    pool["stage_b"] = cfg["stage_b"]
    pool["relation_type"] = relation
    pool["time_gap_hours"] = (pool["b_publish_time"] - pool["a_publish_time"]).dt.total_seconds() / 3600
    pool["stage_signal_strength"] = (
        pool[[c for c in pool.columns if c.startswith("a_stage_") or c.startswith("b_stage_")]].sum(axis=1)
    )
    pool["selection_score"] = (
        0.45 * pool["semantic_score"]
        + 0.20 * pool["structure_score"]
        + 0.20 * pool["graph_score"]
        + 0.15 * pool["stage_signal_strength"]
        - 0.05 * pool["title_similarity"]
    )
    pool = pool.sort_values(["selection_score", "time_gap_hours", "semantic_score"], ascending=[False, True, False])
    return pool


def choose_rows(pool: pd.DataFrame, n: int, used_pairs: set[str]) -> list[dict[str, Any]]:
    rows = []
    for _, r in pool.iterrows():
        pid = str(r["pair_id"])
        if pid in used_pairs:
            continue
        used_pairs.add(pid)
        rows.append(r.to_dict())
        if len(rows) >= n:
            break
    return rows


def build_row(r: dict[str, Any]) -> dict[str, Any]:
    evidence_path = [
        "EventMention",
        "Document",
        "EvidenceSentence",
        "EvidenceUnit",
        "EvidenceSentence",
        "Document",
        "EventMention",
    ]
    e_a = r.get("a_evidence_sentences") or []
    e_b = r.get("b_evidence_sentences") or []
    if not e_a:
        e_a = [sentence_fallback(str(r.get("a_event_local_text") or ""))]
    if not e_b:
        e_b = [sentence_fallback(str(r.get("b_event_local_text") or ""))]
    eu_a = r.get("a_evidence_unit_ids") or []
    eu_b = r.get("b_evidence_unit_ids") or []
    return {
        "pair_id": r["pair_id"],
        "relation_type": r["relation_type"],
        "stage_a": r["stage_a"],
        "stage_b": r["stage_b"],
        "candidate_reason": f"same_subject_same_core_object + ordered_{r['relation_type']} + non_duplicate_cache_candidate",
        "source_event_mention_id": r["event_mention_id_a"],
        "target_event_mention_id": r["event_mention_id_b"],
        "document_id_a": r["a_document_id"],
        "document_id_b": r["b_document_id"],
        "article_id_a": r["a_article_id"],
        "article_id_b": r["b_article_id"],
        "same_document": bool(r["same_document"]),
        "same_evidence_unit": bool(r["same_evidence_unit_any"]),
        "same_evidence_unit_ids_a": "|".join([str(x) for x in eu_a]),
        "same_evidence_unit_ids_b": "|".join([str(x) for x in eu_b]),
        "title_a": r.get("a_document_title", ""),
        "title_b": r.get("b_document_title", ""),
        "source_name_a": r.get("a_source_name", ""),
        "source_name_b": r.get("b_source_name", ""),
        "publish_time_a": r.get("a_publish_time").isoformat() if pd.notna(r.get("a_publish_time")) else "",
        "publish_time_b": r.get("b_publish_time").isoformat() if pd.notna(r.get("b_publish_time")) else "",
        "time_gap_hours": float(r.get("time_gap_hours", 0) or 0),
        "target_type_a": r.get("a_target_type", ""),
        "target_type_b": r.get("b_target_type", ""),
        "target_code_a": r.get("a_target_code", ""),
        "target_code_b": r.get("b_target_code", ""),
        "provisional_event_type_a": r.get("a_provisional_event_type", ""),
        "provisional_event_type_b": r.get("b_provisional_event_type", ""),
        "core_elements_a": "|".join([str(x) for x in r.get("a_core_elements", [])]),
        "core_elements_b": "|".join([str(x) for x in r.get("b_core_elements", [])]),
        "evidence_sentences_a": " | ".join([str(x) for x in e_a[:5]]),
        "evidence_sentences_b": " | ".join([str(x) for x in e_b[:5]]),
        "evidence_path": json.dumps(evidence_path, ensure_ascii=False),
        "semantic_score": float(r.get("semantic_score", 0) or 0),
        "structure_score": float(r.get("structure_score", 0) or 0),
        "graph_score": float(r.get("graph_score", 0) or 0),
        "title_similarity": float(r.get("title_similarity", 0) or 0),
        "local_evidence_similarity": float(r.get("local_evidence_similarity", 0) or 0),
        "fusion_score_before_rerank": float(r.get("fusion_score_before_rerank", 0) or 0),
        "candidate_source_class": r.get("candidate_source_class", ""),
        "hard_conflict": bool(r.get("hard_conflict", False)),
        "related_update_stage_evidence_present": bool(r.get("related_update_stage_evidence_present", False)),
        "label_source": "agent_assisted_update_candidate_generation",
        "review_status": "",
        "human_same_event": "",
        "human_update_relation": "",
        "human_confidence": "",
        "human_evidence": "",
        "human_note": "",
        "annotator_id": "",
        "candidate_stratum": "update_stage_relation",
        "selection_score": float(r.get("selection_score", 0) or 0),
        "verification_status": "pending_human_audit",
        "rule_version": "task3_kg_v2_step4_update_annotation_v0",
    }


def main() -> dict[str, Any]:
    started = time.time()
    REV.mkdir(parents=True, exist_ok=True)
    mention_df = build_mention_frame()
    cand = build_pair_frame(mention_df)
    report_before = {
        "task1_train": sha256_file(ROOT / "outputs/final_task1/final_events_train.jsonl"),
        "task1_test": sha256_file(ROOT / "outputs/final_task1/final_events_test.jsonl"),
        "task2_links": sha256_file(ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"),
        "v1_nodes": sha256_file(ROOT / "outputs/task3_kg_v2/observation_graph_v1/nodes.jsonl"),
        "v1_edges": sha256_file(ROOT / "outputs/task3_kg_v2/observation_graph_v1/edges.jsonl"),
        "step4_candidates": sha256_file(CANDIDATES),
        "step4_revision_freeze": sha256_file(FREEZE_REPORT),
        "existing_revision_report": sha256_file(PREV_REPORT) if PREV_REPORT.exists() else None,
    }
    used_pairs: set[str] = set()
    sampled_rows: list[dict[str, Any]] = []
    pools_summary = {}
    for relation, quota in RELATION_QUOTAS.items():
        pool = score_stage_pair_pool(cand, relation)
        pools_summary[relation] = {
            "available": int(len(pool)),
            "quota": quota,
        }
        rows = choose_rows(pool, quota, used_pairs)
        for row in rows:
            sampled_rows.append(build_row(row))
        pools_summary[relation]["sampled"] = len(rows)
    if len(sampled_rows) < TARGET_SAMPLE_COUNT:
        supplement_pool = cand[
            (~cand["pair_id"].isin(used_pairs))
            & (~cand["same_document"])
            & (~cand["same_evidence_unit_any"])
            & (cand["ordered_time_ok"])
            & (cand["a_target_code"] == cand["b_target_code"])
            & (cand["a_target_type"] == cand["b_target_type"])
            & (cand["title_similarity"] < 0.95)
            & (cand["local_evidence_similarity"] < 0.95)
            & (cand["candidate_source_class"].isin(["graph_only", "graph_and_semantic"]))
        ].copy()
        supplement_pool = supplement_pool.sort_values(["selection_score"], ascending=[False])
        for _, row in supplement_pool.iterrows():
            if len(sampled_rows) >= TARGET_SAMPLE_COUNT:
                break
            if str(row["pair_id"]) in used_pairs:
                continue
            row = row.to_dict()
            row.update({"relation_type": "announcement_to_progress", "stage_a": "announcement", "stage_b": "progress", "candidate_reason": "supplemental_same_subject_stage_sequence_candidate"})
            sampled_rows.append(build_row(row))
            used_pairs.add(str(row["pair_id"]))
    sampled_rows = sampled_rows[:TARGET_SAMPLE_COUNT]
    df = pd.DataFrame(sampled_rows)
    if df.empty:
        raise RuntimeError("no update candidates generated")
    df.to_excel(OUT_XLSX, index=False)

    relation_counts = df["relation_type"].value_counts().to_dict()
    stage_a_counts = df["stage_a"].value_counts().to_dict()
    stage_b_counts = df["stage_b"].value_counts().to_dict()
    candidate_reason_counts = df["candidate_reason"].value_counts().to_dict()
    time_gap_stats = {
        "count": int(df["time_gap_hours"].count()),
        "mean": float(df["time_gap_hours"].mean()),
        "median": float(df["time_gap_hours"].median()),
        "min": float(df["time_gap_hours"].min()),
        "max": float(df["time_gap_hours"].max()),
    }
    report_after = {
        "task1_train": sha256_file(ROOT / "outputs/final_task1/final_events_train.jsonl"),
        "task1_test": sha256_file(ROOT / "outputs/final_task1/final_events_test.jsonl"),
        "task2_links": sha256_file(ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"),
        "v1_nodes": sha256_file(ROOT / "outputs/task3_kg_v2/observation_graph_v1/nodes.jsonl"),
        "v1_edges": sha256_file(ROOT / "outputs/task3_kg_v2/observation_graph_v1/edges.jsonl"),
        "step4_candidates": sha256_file(CANDIDATES),
        "step4_revision_freeze": sha256_file(FREEZE_REPORT),
        "existing_revision_report": sha256_file(PREV_REPORT) if PREV_REPORT.exists() else None,
    }
    guide = (
        "# Task3 KG v2 Step4 Update Annotation Guide\n\n"
        "本包仅用于补充阶段关系候选，不生成最终金标。\n\n"
        "判定顺序：\n"
        "1. 先检查两端主体和核心事件对象是否一致。\n"
        "2. 再检查是否满足明确阶段推进：rumor_to_confirmation / announcement_to_progress / progress_to_completion / investigation_to_penalty / clarification_or_refutation / plan_to_termination。\n"
        "3. 必须排除近重复、同标题转载、同一Document、同一EvidenceUnit。\n"
        "4. 只有当两端证据句分别支持前后阶段时，才可作为 related_update 复核候选。\n"
        "\n"
        "human 字段必须保持为空。\n"
        "`candidate_reason` 仅是抽样与解释字段，不是标签。\n"
        "`stage_a` / `stage_b` 是阶段候选，不是最终真值。\n"
        "`evidence_path` 用于记录可解释路径，不得省略。\n"
    )
    OUT_GUIDE.parent.mkdir(parents=True, exist_ok=True)
    OUT_GUIDE.write_text(guide, encoding="utf-8")

    summary = {
        "version": "task3_kg_v2_step4_update_annotation_v0",
        "sample_count": int(len(df)),
        "relation_counts": {k: int(v) for k, v in relation_counts.items()},
        "stage_a_counts": {k: int(v) for k, v in stage_a_counts.items()},
        "stage_b_counts": {k: int(v) for k, v in stage_b_counts.items()},
        "candidate_reason_counts": {k: int(v) for k, v in candidate_reason_counts.items()},
        "time_gap_stats": time_gap_stats,
        "pools_summary": pools_summary,
        "human_fields_empty": bool(all((df[c].fillna("") == "").all() for c in HUMAN_FIELDS)),
        "pair_id_unique": bool(df["pair_id"].is_unique),
        "no_same_document": bool((df["same_document"] == False).all()),
        "no_same_evidence_unit": bool((df["same_evidence_unit"] == False).all()),
        "time_order_clear": bool((df["time_gap_hours"] > 0).all()),
        "original_hashes_unchanged": report_before == report_after,
        "hashes_before": report_before,
        "hashes_after": report_after,
        "output_paths": {
            "annotation_package": str(OUT_XLSX.relative_to(ROOT)),
            "sampling_report": str(OUT_REPORT.relative_to(ROOT)),
            "guide": str(OUT_GUIDE.relative_to(ROOT)),
        },
        "resource_usage": {
            "elapsed_seconds": round(time.time() - started, 3),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        },
        "note": "human fields remain empty; no CanonicalEvent or phase-5 outputs were modified",
    }
    write_json(OUT_REPORT, summary)
    write_json(OUT_SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
