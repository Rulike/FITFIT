#!/usr/bin/env python3
"""Build the task3 KG v2 observation/evidence graph v0.

This is a read-only materialization over existing artifacts. It does not run
schema construction, embedding, clustering, GNN, or LLM inference.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULE_VERSION = "task3_kg_v2_observation_graph_v0"
SCHEMA_VERSION = "task3_kg_v2_schema_v0"
NOW_STAMP = None  # keep deterministic; temporal fields remain null when not known

# Inputs
FULLTEXT_INDEX = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"
META_INDEX = ROOT / "data/processed/event_clusters/article_metadata_index.jsonl"
TASK1_TRAIN = ROOT / "outputs/final_task1/final_events_train.jsonl"
TASK1_TEST = ROOT / "outputs/final_task1/final_events_test.jsonl"
TASK1_CERT = ROOT / "outputs/final_task1/task1_completion_certificate.json"
MENTIONS_PATH = ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl"
DOCS_PATH = ROOT / "outputs/task3/normalized/normalized_documents.jsonl"
EVIDENCE_SENTENCES_PATH = ROOT / "outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl"
TASK2_LINKS_PATH = ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"
TASK1_TRAIN = ROOT / "outputs/final_task1/final_events_train.jsonl"
TASK1_TEST = ROOT / "outputs/final_task1/final_events_test.jsonl"
FULLTEXT_ONLY_AUDIT = ROOT / "outputs/task3_kg_v2/fulltext_only_2200_audit.parquet"
FULL_UNIVERSE_PATH = ROOT / "outputs/task3_kg_v2/full_document_universe.parquet"
LEDGER_PATH = ROOT / "outputs/task3_kg_v2/task1_article_processing_ledger.parquet"
AMBIGUOUS_TASK2_PATH = ROOT / "outputs/task3_kg_v2/observation_graph/ambiguous_task2_links.jsonl"

# Outputs
SCHEMA_DIR = ROOT / "outputs/task3_kg_v2/schema"
GRAPH_DIR = ROOT / "outputs/task3_kg_v2/observation_graph"
SCHEMA_JSON = SCHEMA_DIR / "graph_schema.json"
NODES_PATH = GRAPH_DIR / "nodes.jsonl"
EDGES_PATH = GRAPH_DIR / "edges.jsonl"
NODE_MAPPING_PATH = GRAPH_DIR / "node_mapping.parquet"
EDGE_MAPPING_PATH = GRAPH_DIR / "edge_mapping.parquet"
MANIFEST_PATH = GRAPH_DIR / "materialization_manifest.json"
SCHEMA_REPORT_PATH = ROOT / "reports/task3_kg_v2_schema_report.json"
GRAPH_REPORT_PATH = ROOT / "reports/task3_kg_v2_observation_graph_report.json"
SCHEMA_DOC_PATH = ROOT / "docs/task3_kg_v2_schema.md"
GRAPH_DOC_PATH = ROOT / "docs/task3_kg_v2_observation_graph.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(value: Any) -> str:
    return hashlib.sha256(("" if value is None else str(value)).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(part.split()) for part in text.split("\n")).strip()


def load_fulltext() -> pd.DataFrame:
    df = pd.read_parquet(FULLTEXT_INDEX)
    return df.copy()


def load_universe() -> pd.DataFrame:
    return pd.read_parquet(FULL_UNIVERSE_PATH)


def load_task1_final() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(TASK1_TRAIN) + read_jsonl(TASK1_TEST)
    by_event_id: dict[str, dict[str, Any]] = {}
    by_article_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event_id[str(row["event_id"])] = row
        by_article_id[str(row["article_id"])].append(row)
    return {"by_event_id": by_event_id, "by_article_id": by_article_id}


def build_schema() -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "layers": {
            "observation_evidence": {
                "description": "只读观测证据层，承载 Document/Source/EventMention/Claim/EvidenceSentence/Task2Assessment/Entity/Company/Stock/Industry。",
                "node_instances_allowed": True,
            },
            "canonical_event": {
                "description": "规范事件层，仅定义 Schema，不生成实例。",
                "node_instances_allowed": False,
            },
        },
        "node_types": {
            "Document": {"graph_layer": "observation_evidence"},
            "Source": {"graph_layer": "observation_evidence"},
            "EventMention": {"graph_layer": "observation_evidence"},
            "Claim": {"graph_layer": "observation_evidence"},
            "EvidenceSentence": {"graph_layer": "observation_evidence"},
            "Task2Assessment": {"graph_layer": "observation_evidence"},
            "Entity": {"graph_layer": "observation_evidence"},
            "Company": {"graph_layer": "observation_evidence"},
            "Stock": {"graph_layer": "observation_evidence"},
            "Industry": {"graph_layer": "observation_evidence"},
            "CanonicalEvent": {"graph_layer": "canonical_event"},
            "EventThread": {"graph_layer": "canonical_event"},
            "EventState": {"graph_layer": "canonical_event"},
        },
        "edge_types": {
            "PUBLISHED_BY": {"graph_layer": "observation_evidence", "source_type": "Document", "target_type": "Source"},
            "CONTAINS_MENTION": {"graph_layer": "observation_evidence", "source_type": "Document", "target_type": "EventMention"},
            "EXPRESSES_CLAIM": {"graph_layer": "observation_evidence", "source_type": "EventMention", "target_type": "Claim"},
            "SUPPORTED_BY_SENTENCE": {"graph_layer": "observation_evidence", "source_type": "Claim", "target_type": "EvidenceSentence"},
            "EXTRACTED_FROM": {"graph_layer": "observation_evidence", "source_type": "EvidenceSentence", "target_type": "Document"},
            "MENTIONS_ENTITY": {"graph_layer": "observation_evidence", "source_type": ["EventMention", "Claim"], "target_type": "Entity"},
            "TARGETS_COMPANY": {"graph_layer": "observation_evidence", "source_type": ["EventMention", "Claim"], "target_type": "Company"},
            "TARGETS_STOCK": {"graph_layer": "observation_evidence", "source_type": ["EventMention", "Claim"], "target_type": "Stock"},
            "TARGETS_INDUSTRY": {"graph_layer": "observation_evidence", "source_type": ["EventMention", "Claim"], "target_type": "Industry"},
            "ASSESSED_BY": {"graph_layer": "observation_evidence", "source_type": ["Document", "EventMention"], "target_type": "Task2Assessment"},
            "EXACT_CONTENT_MATCH": {"graph_layer": "observation_evidence", "source_type": "Document", "target_type": "Document", "constraint": "clean_content_hash_exact_match_only"},
            "EVIDENCE_FOR": {"graph_layer": "observation_evidence", "source_type": "Document", "target_type": ["EventMention", "Claim"], "constraint": "explicit_upstream_evidence_only"},
            "INSTANCE_OF_EVENT": {"graph_layer": "canonical_event", "source_type": "EventMention", "target_type": "CanonicalEvent"},
            "RELATED_UPDATE": {"graph_layer": "canonical_event", "source_type": "CanonicalEvent", "target_type": "CanonicalEvent"},
            "PART_OF_THREAD": {"graph_layer": "canonical_event", "source_type": "CanonicalEvent", "target_type": "EventThread"},
            "PRECEDES": {"graph_layer": "canonical_event", "source_type": "CanonicalEvent", "target_type": "CanonicalEvent"},
            "SUPPORTS": {"graph_layer": "canonical_event", "source_type": "Claim", "target_type": "Claim"},
            "CONTRADICTS": {"graph_layer": "canonical_event", "source_type": "Claim", "target_type": "Claim"},
            "CLARIFIES": {"graph_layer": "canonical_event", "source_type": "Claim", "target_type": "Claim"},
            "REFUTES": {"graph_layer": "canonical_event", "source_type": "Claim", "target_type": "Claim"},
            "REVERSES": {"graph_layer": "canonical_event", "source_type": ["Claim", "CanonicalEvent"], "target_type": ["Claim", "CanonicalEvent"]},
            "AFFECTS": {"graph_layer": "canonical_event", "source_type": "CanonicalEvent", "target_type": ["Company", "Stock", "Industry"]},
        },
        "contracts": {
            "task1": str((ROOT / "configs/task3_kg_v2_task1_contract.yaml").relative_to(ROOT)),
            "task2": str((ROOT / "configs/task3_kg_v2_task2_contract.yaml").relative_to(ROOT)),
        },
        "constraints": {
            "canonical_event_instances_must_be_zero": True,
            "event_thread_instances_must_be_zero": True,
            "instance_of_event_instances_must_be_zero": True,
            "task2_cluster_id_must_not_define_same_event": True,
            "ambiguous_task2_links_must_remain_ambiguous": True,
            "auxiliary_documents_must_not_generate_core_event_mentions": True,
            "stable_id_deterministic": True,
            "all_nodes_and_edges_require_provenance_and_temporal_fields": True,
        },
    }


def main() -> dict[str, Any]:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    fulltext_df = load_fulltext()
    task1 = load_task1_final()
    mentions = read_jsonl(MENTIONS_PATH)
    docs = read_jsonl(DOCS_PATH)
    evidence_sentences = read_jsonl(EVIDENCE_SENTENCES_PATH)
    task2_links = read_jsonl(TASK2_LINKS_PATH)
    fulltext_only_audit = pd.read_parquet(FULLTEXT_ONLY_AUDIT)
    task1_cert = json.loads(TASK1_CERT.read_text(encoding="utf-8")) if TASK1_CERT.exists() else {}

    # Universe joins.
    fulltext_by_id = {str(r["article_id"]): r for _, r in fulltext_df.iterrows()}
    universe_by_id = {str(r["article_id"]): r for _, r in universe.iterrows()}
    fulltext_only_candidates = fulltext_only_audit[fulltext_only_audit["exist_in_task2_evidence"] == True].copy()  # noqa: E712
    aux_ids = [str(x) for x in fulltext_only_candidates["article_id"].tolist()]
    core_doc_ids = [str(r["article_id"]) for r in docs]
    core_doc_set = set(core_doc_ids)
    aux_doc_set = set(aux_ids)
    graph_doc_ids = core_doc_ids + [aid for aid in aux_ids if aid not in core_doc_set]

    # Helper maps.
    mention_by_id = {str(r["event_mention_id"]): r for r in mentions}
    doc_to_mentions: dict[str, list[str]] = defaultdict(list)
    for m in mentions:
        doc_to_mentions[str(m["document_id"])].append(str(m["event_mention_id"]))

    evidence_by_mention: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_sentences:
        evidence_by_mention[str(row["event_mention_id"])].append(row)

    final_rows = read_jsonl(TASK1_TRAIN) + read_jsonl(TASK1_TEST)
    event_id_to_confidence: dict[str, float] = {}
    event_id_to_final_status: dict[str, str] = {}
    article_to_event_ids: dict[str, list[str]] = defaultdict(list)
    for row in final_rows:
        event_id = str(row["event_id"])
        article_id = str(row["article_id"])
        event_id_to_confidence[event_id] = float(row.get("final_confidence_score", 0.0))
        event_id_to_final_status[event_id] = str(row.get("final_status", ""))
        article_to_event_ids[article_id].append(event_id)

    # Build core document source rows.
    nodes: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []

    # Source nodes.
    sources = sorted({str(r["source_name"]) for r in docs if r.get("source_name")} | {str(fulltext_by_id[aid]["article_source"]) for aid in aux_ids if aid in fulltext_by_id})
    source_node_ids: dict[str, str] = {}
    for source_name in sources:
        sid = stable_id("SRC", source_name)
        source_node_ids[source_name] = sid
        node = {
            "stable_id": sid,
            "node_type": "Source",
            "graph_layer": "observation_evidence",
            "provenance_source": str(META_INDEX.relative_to(ROOT)),
            "source_record_id": source_name,
            "confidence": 1.0,
            "confidence_scope": "source_registry",
            "verification_status": "observed",
            "evidence_scope": "record",
            "rule_version": RULE_VERSION,
            "created_at": None,
            "valid_from": None,
            "valid_to": None,
            "source_name": source_name,
        }
        nodes.append(node)
        node_rows.append({"stable_id": sid, "node_type": "Source", "graph_layer": "observation_evidence", "source_record_id": source_name, "provenance_source": node["provenance_source"]})

    # Document nodes: 4849 core docs + 238 auxiliary docs with explicit task2 evidence.
    # core docs come from normalized_documents.jsonl; aux docs are from the fulltext-only audit and task2 evidence overlap.
    fulltext_only_lookup = {str(r["article_id"]): r for _, r in fulltext_only_candidates.iterrows()}

    def doc_node_from_article(article_id: str, doc_kind: str) -> dict[str, Any]:
        row = fulltext_by_id.get(article_id)
        if row is None:
            row = universe_by_id.get(article_id)
        if row is None:
            raise KeyError(article_id)
        source_name = str(row.get("article_source", row.get("source_name", "")))
        publish_time = str(row.get("article_publish_time", row.get("publish_time", "")))
        content = row.get("content", "")
        clean = normalize_content(content)
        clean_hash = str(row.get("clean_content_hash", sha256_text(clean)))
        raw_hash = str(row.get("raw_content_hash", sha256_text(content)))
        sid = stable_id("DOC", article_id)
        source_record_id = article_id
        provenance_source = str(FULLTEXT_INDEX.relative_to(ROOT))
        evidence_scope = "task1_core" if doc_kind == "core" else "evidence_only"
        verification_status = "observed" if doc_kind == "core" else "evidence_only_candidate"
        confidence = 1.0
        confidence_scope = "record_presence"
        return {
            "stable_id": sid,
            "node_type": "Document",
            "graph_layer": "observation_evidence",
            "provenance_source": provenance_source,
            "source_record_id": source_record_id,
            "confidence": confidence,
            "confidence_scope": confidence_scope,
            "verification_status": verification_status,
            "evidence_scope": evidence_scope,
            "rule_version": RULE_VERSION,
            "created_at": None,
            "valid_from": None,
            "valid_to": None,
            "document_id": article_id,
            "article_id": article_id,
            "title": str(row.get("article_title", row.get("title", ""))),
            "source_name": source_name,
            "publish_time": publish_time,
            "content_length": int(row.get("content_length", len(clean))) if row is not None else len(clean),
            "raw_content_hash": raw_hash,
            "clean_content_hash": clean_hash,
            "document_kind": doc_kind,
        }

    core_doc_rows: list[dict[str, Any]] = []
    for d in docs:
        aid = str(d["article_id"])
        node = doc_node_from_article(aid, "core")
        core_doc_rows.append(node)
    aux_doc_rows: list[dict[str, Any]] = []
    for aid in aux_ids:
        if aid in core_doc_set:
            continue
        node = doc_node_from_article(aid, "auxiliary")
        aux_doc_rows.append(node)
    doc_rows = core_doc_rows + aux_doc_rows
    for node in doc_rows:
        nodes.append(node)
        node_rows.append({"stable_id": node["stable_id"], "node_type": "Document", "graph_layer": node["graph_layer"], "source_record_id": node["source_record_id"], "provenance_source": node["provenance_source"]})

    # EventMention nodes.
    mention_nodes: list[dict[str, Any]] = []
    for m in mentions:
        mid = str(m["event_mention_id"])
        aid = str(m["document_id"])
        source_rows = m.get("source_rows", []) or []
        event_id = None
        if source_rows:
            event_id = str(source_rows[0].get("event_id", ""))
        confidence = event_id_to_confidence.get(event_id, float(m.get("trust_score", 0.0)) / 100.0 if m.get("trust_score") is not None else None)
        if confidence is not None and confidence > 1.0:
            confidence = confidence / 100.0
        verification_status = event_id_to_final_status.get(event_id, str(m.get("verdict", ""))) or None
        evidence_scope = "task1_structured"
        node = {
            "stable_id": mid,
            "node_type": "EventMention",
            "graph_layer": "observation_evidence",
            "provenance_source": str(MENTIONS_PATH.relative_to(ROOT)),
            "source_record_id": mid,
            "confidence": confidence,
            "confidence_scope": "event_mention",
            "verification_status": verification_status,
            "evidence_scope": evidence_scope,
            "rule_version": RULE_VERSION,
            "created_at": None,
            "valid_from": None,
            "valid_to": None,
            "document_id": aid,
            "article_id": aid,
            "event_type": m.get("event_type"),
            "provisional_event_type": m.get("provisional_event_type"),
            "event_type_is_provisional": bool(m.get("event_type_is_provisional")),
            "target_type": m.get("target_type"),
            "target_code": m.get("target_code"),
            "linked_stocks": m.get("linked_stocks", []),
            "linked_industries": m.get("linked_industries", []),
            "core_elements": m.get("core_elements", []),
            "publish_time": m.get("publish_time"),
            "source_name": m.get("source_name"),
            "task1_source_path": m.get("task1_source_path"),
            "task2_cluster_ids": m.get("task2_cluster_ids", []),
            "trust_score": m.get("trust_score"),
            "trust_score_scope": m.get("trust_score_scope"),
            "verdict": m.get("verdict"),
        }
        mention_nodes.append(node)
        nodes.append(node)
        node_rows.append({"stable_id": mid, "node_type": "EventMention", "graph_layer": "observation_evidence", "source_record_id": mid, "provenance_source": str(MENTIONS_PATH.relative_to(ROOT))})

    # Claim nodes: one per EventMention.
    claim_nodes: dict[str, dict[str, Any]] = {}
    claim_id_by_mention: dict[str, str] = {}
    for m in mention_nodes:
        mid = m["stable_id"]
        claim_id = stable_id("CLM", mid)
        claim_id_by_mention[mid] = claim_id
        claim_text = json.dumps({
            "subject": m.get("target_name") or m.get("core_elements", [None])[0],
            "provisional_event_type": m.get("provisional_event_type"),
            "target_type": m.get("target_type"),
            "target_code": m.get("target_code"),
            "core_elements": m.get("core_elements", []),
            "event_time": None,
        }, ensure_ascii=False, sort_keys=True)
        evidence_sentence_ids = []
        for s in evidence_by_mention.get(mid, []):
            evidence_sentence_ids.append(stable_id("SENT", mid, s.get("document_id"), s.get("rank"), s.get("start"), s.get("end"), s.get("sentence")))
        node = {
            "stable_id": claim_id,
            "node_type": "Claim",
            "graph_layer": "observation_evidence",
            "provenance_source": str(MENTIONS_PATH.relative_to(ROOT)),
            "source_record_id": mid,
            "confidence": m.get("confidence"),
            "confidence_scope": "claim_provisional",
            "verification_status": m.get("verification_status") or "provisional",
            "evidence_scope": "task1_structured",
            "rule_version": RULE_VERSION,
            "created_at": None,
            "valid_from": None,
            "valid_to": None,
            "claim_status": "provisional",
            "source_event_mention_id": mid,
            "subject": m.get("target_name") or (m.get("core_elements", [None])[0]),
            "provisional_event_type": m.get("provisional_event_type"),
            "target_type": m.get("target_type"),
            "target_code": m.get("target_code"),
            "core_elements": m.get("core_elements", []),
            "event_time": None,
            "publish_time": m.get("publish_time"),
            "evidence_sentence_ids": evidence_sentence_ids,
            "claim_text": claim_text,
        }
        claim_nodes[mid] = node
        nodes.append(node)
        node_rows.append({"stable_id": claim_id, "node_type": "Claim", "graph_layer": "observation_evidence", "source_record_id": mid, "provenance_source": str(MENTIONS_PATH.relative_to(ROOT))})

    # EvidenceSentence nodes.
    evidence_sentence_nodes: list[dict[str, Any]] = []
    for s in evidence_sentences:
        sid = stable_id("SENT", s["event_mention_id"], s["document_id"], s.get("rank"), s.get("start"), s.get("end"), s.get("sentence"))
        node = {
            "stable_id": sid,
            "node_type": "EvidenceSentence",
            "graph_layer": "observation_evidence",
            "provenance_source": str(EVIDENCE_SENTENCES_PATH.relative_to(ROOT)),
            "source_record_id": f"{s['event_mention_id']}|{s['document_id']}|{s.get('rank')}|{s.get('start')}|{s.get('end')}",
            "confidence": float(s.get("score", 0.0)),
            "confidence_scope": "sentence_selection",
            "verification_status": "selected",
            "evidence_scope": s.get("evidence_scope"),
            "rule_version": RULE_VERSION,
            "created_at": None,
            "valid_from": None,
            "valid_to": None,
            "event_mention_id": s.get("event_mention_id"),
            "document_id": s.get("document_id"),
            "article_id": s.get("article_id"),
            "rank": s.get("rank"),
            "start": s.get("start"),
            "end": s.get("end"),
            "sentence": s.get("sentence"),
            "clean_content_hash": s.get("clean_content_hash"),
        }
        evidence_sentence_nodes.append(node)
        nodes.append(node)
        node_rows.append({"stable_id": sid, "node_type": "EvidenceSentence", "graph_layer": "observation_evidence", "source_record_id": node["source_record_id"], "provenance_source": node["provenance_source"]})

    # Task2Assessment nodes from the normalized 5221 link file.
    # Business key: (task2_cluster_id, assessment_version); if missing, fall back to task2_output_version.
    assessment_nodes: list[dict[str, Any]] = []
    ambiguous_link_rows: list[dict[str, Any]] = []
    assessment_key_to_node: dict[tuple[str, str], dict[str, Any]] = {}
    assessment_key_to_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    assessment_version_source = Path(str(TASK2_LINKS_PATH.name)).stem
    for row in task2_links:
        row = dict(row)
        cluster_id = str(row.get("task2_cluster_id") or "")
        assessment_version = str(
            row.get("assessment_version")
            or row.get("task2_output_version")
            or Path(str(row.get("source_path") or assessment_version_source)).stem
        )
        business_key = (cluster_id, assessment_version)
        assessment_key_to_rows[business_key].append(row)
        if business_key not in assessment_key_to_node:
            assessment_id = stable_id("AST", cluster_id, assessment_version)
            node = {
                "stable_id": assessment_id,
                "node_type": "Task2Assessment",
                "graph_layer": "observation_evidence",
                "provenance_source": str(TASK2_LINKS_PATH.relative_to(ROOT)),
                "source_record_id": f"{cluster_id}|{assessment_version}",
                "confidence": (float(row["credibility_score"]) / 100.0) if row.get("credibility_score") is not None else None,
                "confidence_scope": "cluster",
                "verification_status": row.get("link_status"),
                "evidence_scope": "cluster_assessment",
                "rule_version": RULE_VERSION,
                "created_at": None,
                "valid_from": None,
                "valid_to": None,
                "task2_cluster_id": cluster_id,
                "assessment_version": assessment_version,
                "assessment_business_key": [cluster_id, assessment_version],
                "assessment_row_count": 0,
                "trust_score": row.get("trust_score"),
                "credibility_score": row.get("credibility_score"),
                "trust_score_scope": row.get("trust_score_scope") or "cluster",
                "verification_status_cluster": row.get("verification_status"),
                "fulltext_verification_status": row.get("fulltext_verification_status"),
                "task2_output_version": assessment_version,
                "uses_evidence_link_id_for_id": False,
                "provenance_source": str(TASK2_LINKS_PATH.relative_to(ROOT)),
                "source_record_id": f"{cluster_id}|{assessment_version}",
            }
            assessment_key_to_node[business_key] = node
            assessment_nodes.append(node)
            nodes.append(node)
            node_rows.append({"stable_id": assessment_id, "node_type": "Task2Assessment", "graph_layer": "observation_evidence", "source_record_id": node["source_record_id"], "provenance_source": node["provenance_source"]})
        if row.get("link_status") == "ambiguous":
            ambiguous_link_rows.append(row)
    for business_key, node in assessment_key_to_node.items():
        node["assessment_row_count"] = len(assessment_key_to_rows[business_key])
    write_jsonl(AMBIGUOUS_TASK2_PATH, ambiguous_link_rows)

    # Entity / Company / Stock / Industry nodes and edges.
    entity_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    company_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    stock_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    industry_nodes: dict[tuple[str, str], dict[str, Any]] = {}

    def ensure_node(store: dict[tuple[str, str], dict[str, Any]], ntype: str, key: str, **extra: Any) -> dict[str, Any]:
        k = (ntype, key)
        if k not in store:
            sid = stable_id(ntype[:3].upper(), key)
            node = {
                "stable_id": sid,
                "node_type": ntype,
                "graph_layer": "observation_evidence",
                "provenance_source": str(MENTIONS_PATH.relative_to(ROOT)),
                "source_record_id": key,
                "confidence": 1.0,
                "confidence_scope": "entity_reference",
                "verification_status": "derived",
                "evidence_scope": "task1_structured",
                "rule_version": RULE_VERSION,
                "created_at": None,
                "valid_from": None,
                "valid_to": None,
            }
            node.update(extra)
            store[k] = node
            nodes.append(node)
            node_rows.append({"stable_id": sid, "node_type": ntype, "graph_layer": "observation_evidence", "source_record_id": key, "provenance_source": str(MENTIONS_PATH.relative_to(ROOT))})
        return store[k]

    # Node creation from mentions.
    mention_to_entity_keys: dict[str, set[str]] = defaultdict(set)
    for m in mention_nodes:
        mid = m["stable_id"]
        target_type = m.get("target_type")
        target_code = str(m.get("target_code") or "")
        target_name = str(m.get("target_name") or "")
        core_elements = [str(x) for x in (m.get("core_elements") or []) if x]
        # Generic entities from core elements.
        for ce in core_elements:
            ent = ensure_node(entity_nodes, "Entity", ce, entity_name=ce)
            mention_to_entity_keys[mid].add(ent["stable_id"])
        # A generic entity for target name when present.
        if target_name:
            ent = ensure_node(entity_nodes, "Entity", target_name, entity_name=target_name)
            mention_to_entity_keys[mid].add(ent["stable_id"])
        # Typed nodes.
        if target_type == "company" and target_code:
            comp = ensure_node(company_nodes, "Company", target_code, company_code=target_code, company_name=target_name)
            mention_to_entity_keys[mid].add(comp["stable_id"])
        elif target_type == "stock" and target_code:
            stk = ensure_node(stock_nodes, "Stock", target_code, stock_code=target_code, stock_name=target_name)
            mention_to_entity_keys[mid].add(stk["stable_id"])
        elif target_type == "industry" and target_code:
            ind = ensure_node(industry_nodes, "Industry", target_code, industry_code=target_code, industry_name=target_name)
            mention_to_entity_keys[mid].add(ind["stable_id"])

    # Add typed nodes to the node list.
    for store in [entity_nodes, company_nodes, stock_nodes, industry_nodes]:
        for node in store.values():
            nodes.append(node)
            node_rows.append({"stable_id": node["stable_id"], "node_type": node["node_type"], "graph_layer": node["graph_layer"], "source_record_id": node["source_record_id"], "provenance_source": node["provenance_source"]})

    # Edges helpers.
    def add_edge(rel_type: str, src_id: str, dst_id: str, src_type: str, dst_type: str, provenance_source: str, source_record_id: str, confidence: Any = None, confidence_scope: str | None = None, verification_status: Any = None, evidence_scope: str | None = None, link_status: str | None = None, extra: dict[str, Any] | None = None) -> None:
        edge_id = stable_id("EDG", rel_type, src_id, dst_id, source_record_id)
        edge = {
            "stable_id": edge_id,
            "relation_type": rel_type,
            "graph_layer": "observation_evidence" if rel_type not in {"INSTANCE_OF_EVENT", "RELATED_UPDATE", "PART_OF_THREAD", "PRECEDES", "SUPPORTS", "CONTRADICTS", "CLARIFIES", "REFUTES", "REVERSES", "AFFECTS"} else "canonical_event",
            "source_stable_id": src_id,
            "target_stable_id": dst_id,
            "source_node_type": src_type,
            "target_node_type": dst_type,
            "provenance_source": provenance_source,
            "source_record_id": source_record_id,
            "confidence": confidence,
            "confidence_scope": confidence_scope,
            "verification_status": verification_status,
            "evidence_scope": evidence_scope,
            "rule_version": RULE_VERSION,
            "created_at": None,
            "valid_from": None,
            "valid_to": None,
        }
        if link_status is not None:
            edge["link_status"] = link_status
        if extra:
            edge.update(extra)
        edges.append(edge)
        edge_rows.append({"stable_id": edge_id, "relation_type": rel_type, "graph_layer": edge["graph_layer"], "source_stable_id": src_id, "target_stable_id": dst_id, "source_record_id": source_record_id, "provenance_source": provenance_source})

    # PUBLISHED_BY edges.
    for d in doc_rows:
        source_name = d.get("source_name") or ""
        if source_name in source_node_ids:
            add_edge(
                "PUBLISHED_BY",
                d["stable_id"],
                source_node_ids[source_name],
                "Document",
                "Source",
                d["provenance_source"],
                d["source_record_id"],
                confidence=1.0,
                confidence_scope="record_presence",
                verification_status="observed",
                evidence_scope=d.get("document_kind"),
            )

    # CONTAINS_MENTION edges.
    doc_node_map = {d["article_id"]: d for d in doc_rows}
    for m in mention_nodes:
        d = doc_node_map.get(m["article_id"])
        if d is not None:
            add_edge(
                "CONTAINS_MENTION",
                d["stable_id"],
                m["stable_id"],
                "Document",
                "EventMention",
                str(MENTIONS_PATH.relative_to(ROOT)),
                m["stable_id"],
                confidence=m.get("confidence"),
                confidence_scope="event_mention",
                verification_status=m.get("verification_status"),
                evidence_scope="task1_structured",
            )

    # EXPRESSES_CLAIM + typed entity edges.
    for m in mention_nodes:
        claim_id = claim_id_by_mention[m["stable_id"]]
        add_edge("EXPRESSES_CLAIM", m["stable_id"], claim_id, "EventMention", "Claim", str(MENTIONS_PATH.relative_to(ROOT)), m["stable_id"], confidence=m.get("confidence"), confidence_scope="claim_provisional", verification_status=m.get("verification_status"), evidence_scope="task1_structured")
        # MENTIONS_ENTITY for generic entities
        for ce in [str(x) for x in (m.get("core_elements") or []) if x]:
            ent = entity_nodes[("Entity", ce)]
            add_edge("MENTIONS_ENTITY", m["stable_id"], ent["stable_id"], "EventMention", "Entity", str(MENTIONS_PATH.relative_to(ROOT)), m["stable_id"], confidence=m.get("confidence"), confidence_scope="event_mention", verification_status=m.get("verification_status"), evidence_scope="task1_structured")
        # Target typed edges for company / stock / industry.
        if m.get("target_type") == "company" and str(m.get("target_code") or ""):
            comp = company_nodes[("Company", str(m.get("target_code")))]
            add_edge("TARGETS_COMPANY", m["stable_id"], comp["stable_id"], "EventMention", "Company", str(MENTIONS_PATH.relative_to(ROOT)), m["stable_id"], confidence=m.get("confidence"), confidence_scope="event_mention", verification_status=m.get("verification_status"), evidence_scope="task1_structured")
        elif m.get("target_type") == "stock" and str(m.get("target_code") or ""):
            stk = stock_nodes[("Stock", str(m.get("target_code")))]
            add_edge("TARGETS_STOCK", m["stable_id"], stk["stable_id"], "EventMention", "Stock", str(MENTIONS_PATH.relative_to(ROOT)), m["stable_id"], confidence=m.get("confidence"), confidence_scope="event_mention", verification_status=m.get("verification_status"), evidence_scope="task1_structured")
        elif m.get("target_type") == "industry" and str(m.get("target_code") or ""):
            ind = industry_nodes[("Industry", str(m.get("target_code")))]
            add_edge("TARGETS_INDUSTRY", m["stable_id"], ind["stable_id"], "EventMention", "Industry", str(MENTIONS_PATH.relative_to(ROOT)), m["stable_id"], confidence=m.get("confidence"), confidence_scope="event_mention", verification_status=m.get("verification_status"), evidence_scope="task1_structured")

    # Claim->EvidenceSentence and EvidenceSentence->Document.
    for mid, claim in claim_nodes.items():
        for s in evidence_by_mention.get(mid, []):
            sid = stable_id("SENT", s["event_mention_id"], s["document_id"], s.get("rank"), s.get("start"), s.get("end"), s.get("sentence"))
            add_edge("SUPPORTED_BY_SENTENCE", claim["stable_id"], sid, "Claim", "EvidenceSentence", str(EVIDENCE_SENTENCES_PATH.relative_to(ROOT)), claim["source_record_id"], confidence=float(s.get("score", 0.0)), confidence_scope="sentence_selection", verification_status="selected", evidence_scope=s.get("evidence_scope"))
            d = doc_node_map.get(str(s["article_id"]))
            if d is not None:
                add_edge("EXTRACTED_FROM", sid, d["stable_id"], "EvidenceSentence", "Document", str(EVIDENCE_SENTENCES_PATH.relative_to(ROOT)), f"{s['event_mention_id']}|{s['document_id']}|{s.get('rank')}|{s.get('start')}|{s.get('end')}", confidence=float(s.get("score", 0.0)), confidence_scope="sentence_selection", verification_status="selected", evidence_scope=s.get("evidence_scope"))

    # Task2Assessment edges.
    # All 5,221 original assessment rows remain traceable, but the assessment node count is reduced
    # to unique business keys. We keep every original row as an ASSESSED_BY edge to its business-key node.
    task2_assessment_rows: list[dict[str, Any]] = []
    for business_key, rows in assessment_key_to_rows.items():
        node = assessment_key_to_node[business_key]
        node_doc_ids = {r.get("document_id") for r in rows}
        node_em_ids = {r.get("event_mention_id") for r in rows}
        for row in rows:
            task2_assessment_rows.append(row)
            src_doc_id = str(row.get("document_id"))
            src_em_id = str(row.get("event_mention_id"))
            link_status = str(row.get("link_status")) if row.get("link_status") is not None else None
            confidence = row.get("confidence")
            evidence_scope = row.get("evidence_scope")
            if link_status == "resolved" and src_em_id in mention_by_id:
                add_edge(
                    "ASSESSED_BY",
                    src_em_id,
                    node["stable_id"],
                    "EventMention",
                    "Task2Assessment",
                    row.get("provenance_source", str(TASK2_LINKS_PATH.relative_to(ROOT))),
                    f"{row.get('document_id')}|{row.get('event_mention_id')}|{row.get('task2_cluster_id')}|{row.get('source_path')}|{row.get('link_status')}",
                    confidence=confidence,
                    confidence_scope=row.get("confidence_scope"),
                    verification_status=row.get("verification_status"),
                    evidence_scope=evidence_scope,
                    link_status=link_status,
                )
            else:
                d = doc_node_map.get(src_doc_id)
                if d is not None:
                    add_edge(
                        "ASSESSED_BY",
                        d["stable_id"],
                        node["stable_id"],
                        "Document",
                        "Task2Assessment",
                        row.get("provenance_source", str(TASK2_LINKS_PATH.relative_to(ROOT))),
                        f"{row.get('document_id')}|{row.get('event_mention_id')}|{row.get('task2_cluster_id')}|{row.get('source_path')}|{row.get('link_status')}",
                        confidence=confidence,
                        confidence_scope=row.get("confidence_scope"),
                        verification_status=row.get("verification_status"),
                        evidence_scope=evidence_scope,
                        link_status=link_status,
                    )

    # Auxiliary EVIDENCE_FOR edges: only explicit evidence-item origins are allowed.
    # We connect auxiliary evidence documents to claims only when the evidence item is explicitly
    # listed in the source evidence JSON. This is a pure provenance edge and not a same-event signal.
    evidence_rows = read_jsonl(ROOT / "outputs/event_credibility/fulltext_evidence.jsonl")
    aux_edge_count = 0
    evidence_origin_types = Counter()
    evidence_for_source_doc_kind = Counter()
    cluster_to_resolved_mentions: dict[str, set[str]] = defaultdict(set)
    for business_key, rows in assessment_key_to_rows.items():
        cid = business_key[0]
        for row in rows:
            if row.get("link_status") == "resolved" and row.get("event_mention_id"):
                cluster_to_resolved_mentions[cid].add(str(row.get("event_mention_id")))
    for row in evidence_rows:
        cid = str(row.get("cluster_id") or "")
        ev = row.get("evidence_items")
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except Exception:
                ev = []
        if not isinstance(ev, list):
            continue
        for item in ev:
            if not isinstance(item, dict):
                continue
            aid = str(item.get("article_id") or "")
            if aid not in aux_doc_set or aid in core_doc_set:
                continue
            d = doc_node_map.get(aid)
            if d is None:
                continue
            evidence_origin_types[str(item.get("source_type") or "unknown")] += 1
            evidence_for_source_doc_kind[str(d.get("document_kind") or "unknown")] += 1
            # Preserve only explicit provenance edges from the evidence document to mentions in the same
            # assessed cluster. This does not create or imply same-event or must-link relationships.
            for mid in sorted(cluster_to_resolved_mentions.get(cid, set())):
                claim = claim_nodes.get(mid)
                if claim is None:
                    continue
                add_edge(
                    "EVIDENCE_FOR",
                    d["stable_id"],
                    claim["stable_id"],
                    "Document",
                    "Claim",
                    str(ROOT / "outputs/event_credibility/fulltext_evidence.jsonl"),
                    f"{aid}|{cid}|{mid}|evidence_item",
                    confidence=None,
                    confidence_scope=None,
                    verification_status="explicit_upstream_evidence",
                    evidence_scope="evidence_only",
                )
                aux_edge_count += 1

    # EXACT_CONTENT_MATCH edges only for identical clean_content_hash across instantiated docs.
    hash_to_docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in doc_rows:
        h = str(d.get("clean_content_hash") or "")
        if h:
            hash_to_docs[h].append(d)
    exact_match_pairs = 0
    for h, ds in hash_to_docs.items():
        if len(ds) < 2:
            continue
        ds_sorted = sorted(ds, key=lambda x: x["stable_id"])
        for i in range(len(ds_sorted)):
            for j in range(i + 1, len(ds_sorted)):
                a = ds_sorted[i]
                b = ds_sorted[j]
                add_edge(
                    "EXACT_CONTENT_MATCH",
                    a["stable_id"],
                    b["stable_id"],
                    "Document",
                    "Document",
                    a["provenance_source"],
                    f"{a['source_record_id']}|{b['source_record_id']}|{h}",
                    confidence=1.0,
                    confidence_scope="exact_hash",
                    verification_status="exact_content_match",
                    evidence_scope="clean_content_hash",
                )
                exact_match_pairs += 1

    # No canonical event instances in this phase.
    canonical_instance_counts = {
        "CanonicalEvent": 0,
        "EventThread": 0,
        "EventState": 0,
        "INSTANCE_OF_EVENT": 0,
    }

    # De-duplicate by stable_id while preserving first-seen order.
    dedup_nodes: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    for node in nodes:
        sid = str(node["stable_id"])
        if sid in seen_node_ids:
            continue
        seen_node_ids.add(sid)
        dedup_nodes.append(node)
    dedup_edges: list[dict[str, Any]] = []
    seen_edge_ids: set[str] = set()
    for edge in edges:
        sid = str(edge["stable_id"])
        if sid in seen_edge_ids:
            continue
        seen_edge_ids.add(sid)
        dedup_edges.append(edge)
    nodes = dedup_nodes
    edges = dedup_edges
    node_rows = [
        {"stable_id": n["stable_id"], "node_type": n["node_type"], "graph_layer": n["graph_layer"], "source_record_id": n["source_record_id"], "provenance_source": n["provenance_source"]}
        for n in nodes
    ]
    edge_rows = [
        {"stable_id": e["stable_id"], "relation_type": e["relation_type"], "graph_layer": e["graph_layer"], "source_stable_id": e["source_stable_id"], "target_stable_id": e["target_stable_id"], "source_record_id": e["source_record_id"], "provenance_source": e["provenance_source"]}
        for e in edges
    ]

    # Write outputs.
    write_jsonl(NODES_PATH, nodes)
    write_jsonl(EDGES_PATH, edges)
    pd.DataFrame(node_rows).to_parquet(NODE_MAPPING_PATH, index=False)
    pd.DataFrame(edge_rows).to_parquet(EDGE_MAPPING_PATH, index=False)

    schema = build_schema()
    SCHEMA_JSON.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    # Reports.
    node_type_counts = Counter(n["node_type"] for n in nodes)
    edge_type_counts = Counter(e["relation_type"] for e in edges)
    doc_counts = {
        "core_document_count": len(core_doc_rows),
        "auxiliary_document_count": len(aux_doc_rows),
        "graph_document_count": len(core_doc_rows) + len(aux_doc_rows),
        "fulltext_only_evidence_candidates": len(fulltext_only_candidates),
        "fulltext_only_uninstantiated": len(fulltext_only_audit) - len(fulltext_only_candidates),
    }
    mention_counts = {
        "event_mention_count": len(mention_nodes),
        "claim_count": len(claim_nodes),
        "evidence_sentence_count": len(evidence_sentence_nodes),
        "task2_assessment_count": len(task2_assessment_rows),
        "task2_resolved_count": sum(1 for r in task2_assessment_rows if r.get("link_status") == "resolved"),
        "task2_ambiguous_count": sum(1 for r in task2_assessment_rows if r.get("link_status") == "ambiguous"),
    }
    missing_event_type = sum(1 for m in mention_nodes if m.get("event_type") is None)
    missing_provisional_event_type = sum(1 for m in mention_nodes if not m.get("provisional_event_type"))
    missing_event_time = sum(1 for c in claim_nodes.values() if c.get("event_time") is None)
    task2_business_keys = list(assessment_key_to_node.keys())
    task2_cluster_counts = Counter(k[0] for k in task2_business_keys)
    assessment_id_uses_evidence_link_id = False
    report = {
        "version": RULE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "full_document_universe": str(FULL_UNIVERSE_PATH.relative_to(ROOT)),
            "task1_final_events_train": str(TASK1_TRAIN.relative_to(ROOT)),
            "task1_final_events_test": str(TASK1_TEST.relative_to(ROOT)),
            "normalized_documents": str(DOCS_PATH.relative_to(ROOT)),
            "normalized_event_mentions": str(MENTIONS_PATH.relative_to(ROOT)),
            "task2_evidence_links": str(TASK2_LINKS_PATH.relative_to(ROOT)),
            "evidence_sentences": str(EVIDENCE_SENTENCES_PATH.relative_to(ROOT)),
        },
        "counts": {
            "nodes_total": len(nodes),
            "edges_total": len(edges),
            "node_type_counts": dict(node_type_counts),
            "edge_type_counts": dict(edge_type_counts),
            "canonical_instance_counts": canonical_instance_counts,
        },
        "documents": doc_counts,
        "mentions": mention_counts,
        "task2_assessment_modeling": {
            "task2_assessment_node_count": len(assessment_nodes),
            "unique_task2_cluster_id_count": len(task2_cluster_counts),
            "unique_business_key_count": len(task2_business_keys),
            "assessment_business_key_duplicates": sum(1 for c in task2_cluster_counts.values() if c > 1),
            "max_assessments_per_cluster": max(task2_cluster_counts.values()) if task2_cluster_counts else 0,
            "uses_evidence_link_id_for_assessment_id": assessment_id_uses_evidence_link_id,
        },
        "field_gaps": {
            "missing_formal_event_type_count": missing_event_type,
            "missing_provisional_event_type_count": missing_provisional_event_type,
            "missing_event_time_count": missing_event_time,
        },
        "integrity": {
            "node_id_unique": len({n["stable_id"] for n in nodes}) == len(nodes),
            "edge_id_unique": len({e["stable_id"] for e in edges}) == len(edges),
            "dangling_edge_count": 0,  # filled below
            "task2_score_scope_cluster_maintained": all((r.get("trust_score_scope") or "cluster") == "cluster" for r in task2_assessment_rows),
            "ambiguous_links_preserved": sum(1 for r in task2_assessment_rows if r.get("link_status") == "ambiguous") == len(ambiguous_link_rows),
            "auxiliary_docs_no_core_mentions": all(aid not in doc_to_mentions for aid in aux_ids),
            "event_mentions_traceable_to_document": all(m["article_id"] in doc_node_map for m in mention_nodes),
            "claims_traceable_to_event_mentions": all(c["source_event_mention_id"] in mention_by_id for c in claim_nodes.values()),
            "evidence_sentences_traceable_to_document": all(s["document_id"] in doc_node_map for s in evidence_sentence_nodes),
            "stable_id_formula_deterministic": True,
            "assessment_business_keys_unique": len(task2_business_keys) == len(assessment_nodes),
            "assessment_business_key_duplicates": sum(1 for c in task2_cluster_counts.values() if c > 1),
            "task2_assessment_nodes_use_evidence_link_id": assessment_id_uses_evidence_link_id,
            "evidence_for_has_explicit_provenance": all(e.get("provenance_source") and e.get("source_record_id") for e in edges if e.get("relation_type") == "EVIDENCE_FOR"),
        },
        "stability": {
            "exact_content_match_edges": exact_match_pairs,
            "auxiliary_evidence_for_edges": aux_edge_count,
            "node_mapping_rows": len(node_rows),
            "edge_mapping_rows": len(edge_rows),
            "source_node_count": len(source_node_ids),
            "exactly_materialized_auxiliary_documents": len(aux_doc_rows),
            "evidence_for_source_doc_kind": dict(evidence_for_source_doc_kind),
            "evidence_origin_source_type_distribution": dict(evidence_origin_types),
        },
        "limitations": [
            "No canonical-event layer instances were generated by design.",
            "Task2 trust/credibility remains cluster-scoped and was not promoted to document or mention truth scores.",
            "A small number of EventMention records have no selected evidence sentences in the fulltext evidence file; they remain traceable to their source document without fabricated offsets.",
            "The 2,200 fulltext-only documents were not wholesale materialized; only the 238 with explicit task2 evidence linkage were brought into the observation graph as auxiliary evidence docs.",
            "EVIDENCE_FOR edges are restricted to explicit provenance from the fulltext evidence audit and do not encode same-event or must-link semantics.",
            "Task1, Task2, and old Task3 outputs were read only; no prior outputs were modified.",
        ],
        "provenance": {
            "schema_file": str(SCHEMA_JSON.relative_to(ROOT)),
            "nodes_file": str(NODES_PATH.relative_to(ROOT)),
            "edges_file": str(EDGES_PATH.relative_to(ROOT)),
            "node_mapping_file": str(NODE_MAPPING_PATH.relative_to(ROOT)),
            "edge_mapping_file": str(EDGE_MAPPING_PATH.relative_to(ROOT)),
            "ambiguous_task2_links_file": str(AMBIGUOUS_TASK2_PATH.relative_to(ROOT)),
        },
        "hashes": {
            "schema_sha256": sha256_text(SCHEMA_JSON.read_text(encoding="utf-8")),
            "nodes_sha256": sha256_text(NODES_PATH.read_text(encoding="utf-8")),
            "edges_sha256": sha256_text(EDGES_PATH.read_text(encoding="utf-8")),
            "node_mapping_sha256": sha256_bytes(NODE_MAPPING_PATH.read_bytes()),
            "edge_mapping_sha256": sha256_bytes(EDGE_MAPPING_PATH.read_bytes()),
            "ambiguous_task2_sha256": sha256_text(AMBIGUOUS_TASK2_PATH.read_text(encoding="utf-8")),
        },
    }

    # Compute dangling edges now that node set exists.
    node_ids = {n["stable_id"] for n in nodes}
    dangling = 0
    for e in edges:
        if e["source_stable_id"] not in node_ids or e["target_stable_id"] not in node_ids:
            dangling += 1
    report["integrity"]["dangling_edge_count"] = dangling
    # make source provenance explicit for Task2 assessment nodes in the final report
    report["task2_assessment_modeling"]["assessment_provenance_source"] = str(TASK2_LINKS_PATH.relative_to(ROOT))
    report["task2_assessment_modeling"]["assessment_source_record_id_template"] = "task2_cluster_id|assessment_version"

    schema_report = {
        "schema_version": SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "node_type_count": len(schema["node_types"]),
        "edge_type_count": len(schema["edge_types"]),
        "canonical_event_instances_must_be_zero": schema["constraints"]["canonical_event_instances_must_be_zero"],
        "event_thread_instances_must_be_zero": schema["constraints"]["event_thread_instances_must_be_zero"],
        "instance_of_event_instances_must_be_zero": schema["constraints"]["instance_of_event_instances_must_be_zero"],
        "task1_contract": schema["contracts"]["task1"],
        "task2_contract": schema["contracts"]["task2"],
        "task2_assessment_business_key": "(task2_cluster_id, assessment_version or task2_output_version)",
        "task2_assessment_id_restrictions": ["evidence_link_id", "event_mention_id", "document_id"],
    }

    SCHEMA_REPORT_PATH.write_text(json.dumps(schema_report, ensure_ascii=False, indent=2), encoding="utf-8")
    GRAPH_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    node_map_bytes = NODE_MAPPING_PATH.read_bytes()
    edge_map_bytes = EDGE_MAPPING_PATH.read_bytes()
    MANIFEST_PATH.write_text(json.dumps({
        "schema_sha256": report["hashes"]["schema_sha256"],
        "nodes_sha256": report["hashes"]["nodes_sha256"],
        "edges_sha256": report["hashes"]["edges_sha256"],
        "node_mapping_sha256": sha256_bytes(node_map_bytes),
        "edge_mapping_sha256": sha256_bytes(edge_map_bytes),
        "ambiguous_task2_sha256": report["hashes"]["ambiguous_task2_sha256"],
        "nodes_total": len(nodes),
        "edges_total": len(edges),
        "node_type_counts": dict(node_type_counts),
        "edge_type_counts": dict(edge_type_counts),
        "task2_resolved_count": mention_counts["task2_resolved_count"],
        "task2_ambiguous_count": mention_counts["task2_ambiguous_count"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    SCHEMA_DOC_PATH.write_text(
        "# Task3 KG v2 Schema\n\n"
        f"- Schema version: `{SCHEMA_VERSION}`\n"
        f"- Rule version: `{RULE_VERSION}`\n"
        "- Observation layer materialized; canonical layer is schema-only.\n\n"
        "## Constraints\n\n"
        "- CanonicalEvent / EventThread / INSTANCE_OF_EVENT instances: 0\n"
        "- Task2 scores remain cluster-scoped\n"
        "- Ambiguous task2 links remain ambiguous\n"
        "- Auxiliary docs do not generate core event mentions\n",
        encoding="utf-8",
    )
    GRAPH_DOC_PATH.write_text(
        "# Task3 KG v2 Observation Graph v0\n\n"
        f"- Core Documents: {doc_counts['core_document_count']}\n"
        f"- Auxiliary Documents: {doc_counts['auxiliary_document_count']}\n"
        f"- EventMentions: {mention_counts['event_mention_count']}\n"
        f"- Claims: {mention_counts['claim_count']}\n"
        f"- EvidenceSentences: {mention_counts['evidence_sentence_count']}\n"
        f"- Task2Assessments (unique business keys): {mention_counts['task2_assessment_count']}\n"
        f"- Resolved task2 links: {mention_counts['task2_resolved_count']}\n"
        f"- Ambiguous task2 links: {mention_counts['task2_ambiguous_count']}\n"
        f"- EVIDENCE_FOR edges: {aux_edge_count}\n"
        f"- CanonicalEvent instances: 0\n",
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
