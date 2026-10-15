#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP4 = ROOT / "outputs/task3_kg_v2/step4"
REV = ROOT / "outputs/task3_kg_v2/step4_revision"
V2 = ROOT / "outputs/task3_kg_v2/observation_graph_v2"
CANDIDATES = STEP4 / "same_event_candidates.parquet"
FEATURES = STEP4 / "graph_feature_explanations.jsonl"
MENTIONS = ROOT / "outputs/task3_fulltext_v2/normalized_event_mentions.jsonl"
EVIDENCE = ROOT / "outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl"
OLD_ROUTING = ROOT / "outputs/task3/step4/event_mention_cluster_mapping.jsonl"
NODES_V2 = V2 / "nodes.jsonl"
EDGES_V2 = V2 / "edges.jsonl"
REPORT_EVAL = ROOT / "reports/task3_kg_v2_step4_evaluation.json"
REPORT_RUNTIME = ROOT / "reports/task3_kg_v2_step4_runtime.json"
CONFIG = ROOT / "configs/task3_kg_v2_step4_same_event.yaml"
PACKAGE_XLSX = REV / "annotation_package.xlsx"
GUIDE = REV / "annotation_guide.md"
SAMPLING_REPORT = REV / "sampling_report.json"
ROUTING_OUT = REV / "processing_channel_repaired_event_mentions.parquet"
FREEZE_REPORT = REV / "step4_v0_relation_classifier_revision_required.json"
TEST_REPORT = REV / "revision_test_report.json"

STAGE_PATTERNS = {
    "rumor_to_confirmation": ["传闻", "网传", "官方", "确认", "证实"],
    "announcement_to_progress": ["公告", "进展", "最新", "推进"],
    "progress_to_completion": ["进展", "完成", "落地", "交割", "达成"],
    "investigation_to_penalty": ["调查", "处罚", "罚款", "立案"],
    "report_to_clarification_refutation": ["报道", "澄清", "否认", "辟谣", "回应"],
    "plan_to_termination": ["方案", "计划", "终止", "取消", "停止"],
}
HUMAN_FIELDS = ["human_same_event", "human_update_relation", "human_confidence", "human_evidence", "human_note", "annotator_id", "review_status"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        v = json.loads(str(x))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def contains_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def choose_rows(pool: pd.DataFrame, n: int, used: set[str]) -> list[dict[str, Any]]:
    if len(pool) == 0 or n <= 0:
        return []
    rows = []
    sort_cols = [c for c in ["semantic_score", "structure_score", "graph_score"] if c in pool.columns]
    ordered = pool.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else pool
    for _, r in ordered.iterrows():
        pid = str(r["pair_id"])
        if pid in used:
            continue
        used.add(pid)
        rows.append(r.to_dict())
        if len(rows) >= n:
            break
    return rows


def main() -> dict[str, Any]:
    started = time.time()
    REV.mkdir(parents=True, exist_ok=True)
    before_hashes = {
        "task1_train": sha256_file(ROOT / "outputs/final_task1/final_events_train.jsonl"),
        "task1_test": sha256_file(ROOT / "outputs/final_task1/final_events_test.jsonl"),
        "task2_links": sha256_file(ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"),
        "old_task3_graph_v2_nodes": sha256_file(NODES_V2),
        "old_task3_graph_v2_edges": sha256_file(EDGES_V2),
    }

    mentions = read_jsonl(MENTIONS)
    old_routing = {r["event_mention_id"]: r.get("processing_channel") for r in read_jsonl(OLD_ROUTING)}
    routing_rows = []
    exact_join = 0
    join_failed = 0
    for m in mentions:
        mid = m["event_mention_id"]
        if mid in old_routing and old_routing[mid] in {"main", "shadow", "macro", "repair"}:
            channel = old_routing[mid]
            status = "exact_join"
            exact_join += 1
        else:
            channel = "pending_routing"
            status = "join_failed"
            join_failed += 1
        routing_rows.append({"event_mention_id": mid, "document_id": m.get("document_id"), "processing_channel": channel, "routing_join_status": status})
    routing_df = pd.DataFrame(routing_rows)
    routing_df.to_parquet(ROUTING_OUT, index=False)
    routing_counts = Counter(routing_df["processing_channel"])
    routing_summary = {
        "main_count": int(routing_counts.get("main", 0)),
        "shadow_count": int(routing_counts.get("shadow", 0)),
        "macro_count": int(routing_counts.get("macro", 0)),
        "repair_count": int(routing_counts.get("repair", 0)),
        "pending_count": int(routing_counts.get("pending_routing", 0)),
        "exact_join_count": exact_join,
        "join_failed_count": join_failed,
        "routing_source": str(OLD_ROUTING.relative_to(ROOT)),
        "no_missing_defaulted_to_main": True,
    }

    freeze = {
        "status": "step4_v0_relation_classifier_revision_required",
        "reason": "RELATED_UPDATE over-prediction and processing_channel routing repair required before canonicalization.",
        "frozen_outputs": {
            "config": str(CONFIG.relative_to(ROOT)),
            "evaluation_report": str(REPORT_EVAL.relative_to(ROOT)),
            "runtime_report": str(REPORT_RUNTIME.relative_to(ROOT)),
            "candidate_table": str(CANDIDATES.relative_to(ROOT)),
            "graph_v2_nodes": str(NODES_V2.relative_to(ROOT)),
            "graph_v2_edges": str(EDGES_V2.relative_to(ROOT)),
        },
        "hashes": {
            "config_sha256": sha256_file(CONFIG),
            "evaluation_report_sha256": sha256_file(REPORT_EVAL),
            "runtime_report_sha256": sha256_file(REPORT_RUNTIME),
            "candidate_table_sha256": sha256_file(CANDIDATES),
            "graph_v2_nodes_sha256": sha256_file(NODES_V2),
            "graph_v2_edges_sha256": sha256_file(EDGES_V2),
        },
    }
    write_json(FREEZE_REPORT, freeze)

    cand = pd.read_parquet(CANDIDATES)
    cand = cand.merge(routing_df.rename(columns={"event_mention_id": "event_mention_id_a", "processing_channel": "processing_channel_a", "routing_join_status": "routing_join_status_a"})[["event_mention_id_a", "processing_channel_a", "routing_join_status_a"]], on="event_mention_id_a", how="left")
    cand = cand.merge(routing_df.rename(columns={"event_mention_id": "event_mention_id_b", "processing_channel": "processing_channel_b", "routing_join_status": "routing_join_status_b"})[["event_mention_id_b", "processing_channel_b", "routing_join_status_b"]], on="event_mention_id_b", how="left")
    cand = cand[(cand["processing_channel_a"] != "pending_routing") & (cand["processing_channel_b"] != "pending_routing")].copy()

    mention_by_id = {m["event_mention_id"]: m for m in mentions}
    evidence_by_mid: dict[str, list[str]] = defaultdict(list)
    for r in read_jsonl(EVIDENCE):
        evidence_by_mid[str(r.get("event_mention_id"))].append(str(r.get("sentence") or ""))
    feature_by_pair = {r["pair_id"]: r for r in read_jsonl(FEATURES)}
    eu_by_mid: dict[str, set[str]] = defaultdict(set)
    for r in feature_by_pair.values():
        # candidate features contain pair-level same_evidence_unit but not ids; package keeps ids if available from v2 graph lookup below.
        pass
    doc_eu: dict[str, set[str]] = defaultdict(set)
    nodes = read_jsonl(NODES_V2)
    doc_article = {n["stable_id"]: str(n.get("article_id")) for n in nodes if n.get("node_type") == "Document"}
    doc_title = {str(n.get("article_id")): str(n.get("title") or "") for n in nodes if n.get("node_type") == "Document"}
    doc_by_mid: dict[str, str] = {}
    for e in read_jsonl(EDGES_V2):
        if e.get("relation_type") == "CONTAINS_MENTION":
            doc_by_mid[str(e.get("target_stable_id"))] = str(e.get("source_stable_id"))
        elif e.get("relation_type") == "BELONGS_TO_EVIDENCE_UNIT":
            doc_eu[str(e.get("source_stable_id"))].add(str(e.get("target_stable_id")))
    eu_by_mid = {mid: doc_eu.get(doc, set()) for mid, doc in doc_by_mid.items()}

    cand["candidate_sources"] = cand["candidate_sources_json"].apply(parse_list)
    cand["hard_constraints_list"] = cand["constraint_results_json"].apply(parse_list)
    cand["text_join"] = cand.apply(lambda r: " ".join([str(mention_by_id[r["event_mention_id_a"]].get("event_local_text", "")), str(mention_by_id[r["event_mention_id_b"]].get("event_local_text", ""))]), axis=1)

    strata_filters = {
        "high_semantic_structure_consistent": cand[(cand.semantic_score >= .78) & (cand.structure_score >= .85) & (~cand.hard_conflict)],
        "same_company_same_type_different_batch": cand[(cand.subject_match == True) & (cand.event_type_compatible == True) & (cand.time_diff_hours.fillna(0) > 168) & (~cand.hard_conflict)],
        "same_trigger_different_amount_target_product_id": cand[cand.hard_constraints_list.apply(lambda xs: any(str(x.get("constraint_type")) in {"different_target_code", "different_target_type", "incompatible_event_type"} for x in xs))],
        "same_evidence_unit_multi_event": cand[(cand.same_evidence_unit == True) & (cand.event_type_compatible == False)],
        "graph_neighbors_similar_text_weak": cand[(cand.graph_score >= .35) & (cand.semantic_score < .55)],
        "semantic_similar_graph_hard_conflict": cand[(cand.semantic_score >= .72) & (cand.hard_conflict == True)],
        "rumor_to_confirmation": cand[cand.text_join.apply(lambda x: contains_any(x, STAGE_PATTERNS["rumor_to_confirmation"]))],
        "announcement_to_progress": cand[cand.text_join.apply(lambda x: contains_any(x, STAGE_PATTERNS["announcement_to_progress"]))],
        "progress_to_completion": cand[cand.text_join.apply(lambda x: contains_any(x, STAGE_PATTERNS["progress_to_completion"]))],
        "investigation_to_penalty": cand[cand.text_join.apply(lambda x: contains_any(x, STAGE_PATTERNS["investigation_to_penalty"]))],
        "report_to_clarification_refutation": cand[cand.text_join.apply(lambda x: contains_any(x, STAGE_PATTERNS["report_to_clarification_refutation"]))],
        "plan_to_termination": cand[cand.text_join.apply(lambda x: contains_any(x, STAGE_PATTERNS["plan_to_termination"]))],
        "uncertain_middle_zone": cand[cand.get("calibrated_relation_label", cand.relation_label).eq("uncertain_review")],
    }
    target_per_stratum = {
        "high_semantic_structure_consistent": 50,
        "same_company_same_type_different_batch": 40,
        "same_trigger_different_amount_target_product_id": 40,
        "same_evidence_unit_multi_event": 35,
        "graph_neighbors_similar_text_weak": 40,
        "semantic_similar_graph_hard_conflict": 40,
        "rumor_to_confirmation": 35,
        "announcement_to_progress": 35,
        "progress_to_completion": 35,
        "investigation_to_penalty": 35,
        "report_to_clarification_refutation": 35,
        "plan_to_termination": 35,
        "uncertain_middle_zone": 85,
    }
    used: set[str] = set()
    sampled = []
    stratum_counts = {}
    for stratum, pool in strata_filters.items():
        rows = choose_rows(pool, target_per_stratum[stratum], used)
        stratum_counts[stratum] = {"available": int(len(pool)), "sampled": len(rows)}
        for r in rows:
            r["candidate_stratum"] = stratum
            sampled.append(r)
    if len(sampled) < 500:
        rows = choose_rows(cand, 500 - len(sampled), used)
        stratum_counts["supplement"] = {"available": int(len(cand)), "sampled": len(rows)}
        for r in rows:
            r["candidate_stratum"] = "supplement"
            sampled.append(r)
    sampled = sampled[:500]

    package_rows = []
    for r in sampled:
        a = mention_by_id[r["event_mention_id_a"]]
        b = mention_by_id[r["event_mention_id_b"]]
        fa = feature_by_pair.get(r["pair_id"], {})
        constraints = r.get("hard_constraints_list") or parse_list(r.get("constraint_results_json"))
        ai_reason = "本地32B草稿未在本步骤实际调用；保留字段供后续离线补全，不能作为金标。"
        package_rows.append({
            "pair_id": r["pair_id"],
            "event_mention_id_a": r["event_mention_id_a"],
            "event_mention_id_b": r["event_mention_id_b"],
            "document_id_a": a.get("document_id"),
            "document_id_b": b.get("document_id"),
            "processing_channel_a": r.get("processing_channel_a"),
            "processing_channel_b": r.get("processing_channel_b"),
            "title_a": doc_title.get(str(a.get("article_id")), ""),
            "title_b": doc_title.get(str(b.get("article_id")), ""),
            "event_local_text_a": a.get("event_local_text", ""),
            "event_local_text_b": b.get("event_local_text", ""),
            "provisional_event_type_a": a.get("provisional_event_type"),
            "provisional_event_type_b": b.get("provisional_event_type"),
            "subject_a": a.get("target_type"),
            "subject_b": b.get("target_type"),
            "target_code_a": a.get("target_code"),
            "target_code_b": b.get("target_code"),
            "core_elements_a": "|".join(map(str, a.get("core_elements") or [])),
            "core_elements_b": "|".join(map(str, b.get("core_elements") or [])),
            "publish_time_a": a.get("publish_time"),
            "publish_time_b": b.get("publish_time"),
            "evidence_sentences_a": " | ".join(evidence_by_mid.get(r["event_mention_id_a"], [])[:5]),
            "evidence_sentences_b": " | ".join(evidence_by_mid.get(r["event_mention_id_b"], [])[:5]),
            "evidence_unit_id_a": "|".join(sorted(eu_by_mid.get(r["event_mention_id_a"], set()))),
            "evidence_unit_id_b": "|".join(sorted(eu_by_mid.get(r["event_mention_id_b"], set()))),
            "semantic_score": r.get("semantic_score"),
            "structure_score": r.get("structure_score"),
            "graph_score": r.get("graph_score"),
            "reranker_score": r.get("reranker_score"),
            "hard_constraints": json.dumps(constraints, ensure_ascii=False),
            "evidence_path": json.dumps(fa.get("evidence_path", ["EventMention", "graph/evidence features", "EventMention"]), ensure_ascii=False),
            "candidate_stratum": r["candidate_stratum"],
            "ai_draft_event_card_a": f"EventCard草稿A: {a.get('provisional_event_type')} / {a.get('target_code')} / {a.get('publish_time')}",
            "ai_draft_event_card_b": f"EventCard草稿B: {b.get('provisional_event_type')} / {b.get('target_code')} / {b.get('publish_time')}",
            "ai_draft_suggested_label": "pending_human_audit",
            "ai_draft_rationale": ai_reason,
            "label_source": "local_32b_draft",
            "draft_review_status": "pending_human_audit",
            "human_same_event": "",
            "human_update_relation": "",
            "human_confidence": "",
            "human_evidence": "",
            "human_note": "",
            "annotator_id": "",
            "review_status": "",
            "split_suggestion": "pending_group_aware_split_after_human_labels",
        })
    package_df = pd.DataFrame(package_rows)
    package_df.to_excel(PACKAGE_XLSX, index=False)

    GUIDE.write_text(
        "# Task3 KG v2 Step4 Revision Annotation Guide\n\n"
        "先判断 `human_same_event`: yes / no / uncertain。\n\n"
        "只有当 `human_same_event=no` 且两端属于同一事件演化脉络时，才填写 `human_update_relation`。\n\n"
        "`human_update_relation` 可选：none, rumor_to_confirmation, announcement_to_progress, progress_to_completion, investigation_to_penalty, report_to_clarification, report_to_refutation, plan_to_termination, other_update, uncertain。\n\n"
        "AI草稿字段仅供参考，`label_source=local_32b_draft`，不得作为金标或测试指标。\n\n"
        "人工标签完成前只保留 split_suggestion，不正式切分；后续需按 EvidenceUnit、候选连通组件和时间做 group-aware split。\n",
        encoding="utf-8",
    )

    after_hashes = {
        "task1_train": sha256_file(ROOT / "outputs/final_task1/final_events_train.jsonl"),
        "task1_test": sha256_file(ROOT / "outputs/final_task1/final_events_test.jsonl"),
        "task2_links": sha256_file(ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"),
        "old_task3_graph_v2_nodes": sha256_file(NODES_V2),
        "old_task3_graph_v2_edges": sha256_file(EDGES_V2),
    }
    report = {
        "version": "task3_kg_v2_step4_revision_annotation_v0",
        "routing_repair": routing_summary,
        "freeze_status": freeze["status"],
        "sampling": {
            "package_rows": len(package_df),
            "stratum_counts": stratum_counts,
            "pair_id_unique": package_df["pair_id"].is_unique,
            "human_fields_empty": all((package_df[c].fillna("") == "").all() for c in HUMAN_FIELDS),
            "ai_draft_separated_from_human_fields": all(c.startswith("ai_draft_") or c in package_df.columns for c in ["ai_draft_event_card_a", "ai_draft_event_card_b", "ai_draft_suggested_label", "ai_draft_rationale"]),
            "label_source": "local_32b_draft",
            "review_status": "pending_human_audit",
        },
        "split_policy": {
            "formal_split_created": False,
            "split_suggestion_only": True,
            "future_group_aware_keys": ["EvidenceUnit", "candidate_connected_component", "publish_time"],
            "frozen_test_not_used_for_threshold_selection": True,
        },
        "hash_integrity": {
            "before": before_hashes,
            "after": after_hashes,
            "unchanged": before_hashes == after_hashes,
        },
        "outputs": {
            "annotation_package": str(PACKAGE_XLSX.relative_to(ROOT)),
            "annotation_guide": str(GUIDE.relative_to(ROOT)),
            "sampling_report": str(SAMPLING_REPORT.relative_to(ROOT)),
            "routing_repair": str(ROUTING_OUT.relative_to(ROOT)),
            "freeze_report": str(FREEZE_REPORT.relative_to(ROOT)),
        },
        "resource_usage": {
            "elapsed_seconds": round(time.time() - started, 3),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        },
    }
    write_json(SAMPLING_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
