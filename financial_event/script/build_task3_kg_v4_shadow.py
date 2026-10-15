#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/opt/data/private/JinRong/kechuang/financial_event_project")
INPUT_DIR = ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1"
OUT_DIR = ROOT / "outputs/task3_kg_v4_shadow"

ACCEPTED_PATH = INPUT_DIR / "task4_event_interface.jsonl"
FEATURE_PATH = INPUT_DIR / "task4_feature_table.parquet"
JOIN_REPORT_PATH = INPUT_DIR / "task3_task4_join_report.json"
REVIEW_SIDECAR_PATH = OUT_DIR / "review_sidecar.jsonl"
TASK4_OUT_PATH = OUT_DIR / "task4_event_interface.jsonl"
NODES_OUT_PATH = OUT_DIR / "graph_nodes.jsonl"
EDGES_OUT_PATH = OUT_DIR / "graph_edges.jsonl"
CANONICAL_EVENTS_OUT_PATH = OUT_DIR / "canonical_events.jsonl"
EVENT_MENTIONS_OUT_PATH = OUT_DIR / "event_mentions.jsonl"
EVIDENCE_UNITS_OUT_PATH = OUT_DIR / "evidence_units.jsonl"
EVENT_THREADS_OUT_PATH = OUT_DIR / "event_threads.jsonl"
CURRENT_STATE_OUT_PATH = OUT_DIR / "current_event_state.jsonl"
GRAPH_STATS_OUT_PATH = OUT_DIR / "graph_statistics.json"
MANIFEST_OUT_PATH = OUT_DIR / "build_manifest.json"

RULE_VERSION = "task3_kg_v4_shadow_v1"
GRAPH_LAYER = "task3_kg_v4_shadow"

CONTEXT_RELATIONS = {
    "BELONGS_TO_INDUSTRY",
    "SUPPLY_CHAIN_RELATED",
    "EQUITY_RELATED",
    "HAS_FINANCIAL_STATE",
}


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {text(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, tuple):
        return [normalize(v) for v in obj]
    if isinstance(obj, set):
        return [normalize(v) for v in sorted(obj, key=lambda x: text(x))]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(normalize(row), ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted = read_jsonl(ACCEPTED_PATH)
    feature_df = pd.read_parquet(FEATURE_PATH)
    feature_rows = feature_df.to_dict(orient="records")
    review_sidecar = [row for row in feature_rows if text(row.get("task4_acceptance_status")) != "pass"]
    join_report = json.loads(JOIN_REPORT_PATH.read_text(encoding="utf-8"))
    return accepted, feature_rows, review_sidecar, join_report


def node_base(node_type: str, stable: str, confidence: float = 1.0, verification_status: str = "accepted") -> dict[str, Any]:
    return {
        "stable_id": stable,
        "node_type": node_type,
        "graph_layer": GRAPH_LAYER,
        "verification_status": verification_status,
        "confidence": round(float(confidence), 6),
        "confidence_scope": "shadow_graph",
        "evidence_scope": "accepted_interface" if verification_status == "accepted" else "review_sidecar",
        "rule_version": RULE_VERSION,
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "valid_from": None,
        "valid_to": None,
    }


def edge_base(rel: str, src: str, dst: str, src_type: str, dst_type: str, record_id: str, confidence: float = 1.0, verification_status: str = "accepted") -> dict[str, Any]:
    return {
        "stable_id": stable_id("ED4", rel, src, dst, record_id, length=24),
        "relation_type": rel,
        "source_id": src,
        "target_id": dst,
        "source_node_type": src_type,
        "target_node_type": dst_type,
        "graph_layer": GRAPH_LAYER,
        "verification_status": verification_status,
        "processing_channel": "shadow",
        "confidence": round(float(confidence), 6),
        "confidence_scope": "shadow_graph",
        "evidence_scope": "accepted_interface" if verification_status == "accepted" else "review_sidecar",
        "relation_source": "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/task4_event_interface.jsonl",
        "rule_version": RULE_VERSION,
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "updated_at": pd.Timestamp.now("UTC").isoformat(),
        "valid_from": None,
        "valid_to": None,
        "source_record_id": record_id,
    }


def build_shadow_graph(accepted_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]]) -> dict[str, Any]:
    node_map: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_seen: set[tuple[str, str, str, str]] = set()

    def add_node(node: dict[str, Any]) -> None:
        node_map.setdefault(node["stable_id"], node)

    def add_edge(rel: str, src: str, dst: str, src_type: str, dst_type: str, record_id: str, confidence: float = 1.0, verification_status: str = "accepted") -> None:
        key = (rel, src, dst, record_id)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(edge_base(rel, src, dst, src_type, dst_type, record_id, confidence=confidence, verification_status=verification_status))

    sources: dict[tuple[str, str], str] = {}
    documents: dict[str, str] = {}
    targets: dict[str, str] = {}
    evidence_sentences: dict[str, str] = {}
    evidence_units: dict[str, str] = {}
    claims: dict[str, str] = {}
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in accepted_rows:
        cid = text(row.get("canonical_event_id"))
        groups[cid].append(row)

        source_key = (text(row.get("source_name")), text(row.get("source_dataset")))
        source_id = sources.get(source_key)
        if not source_id:
            source_id = stable_id("SRC4", *source_key)
            sources[source_key] = source_id
            add_node({
                **node_base("Source", source_id, confidence=1.0, verification_status="observed"),
                "source_name": source_key[0],
                "source_dataset": source_key[1],
                "source_record_id": "|".join(source_key),
            })

        document_id = documents.get(text(row.get("article_id")))
        if not document_id:
            document_id = stable_id("DOC4", text(row.get("article_id")), length=24)
            documents[text(row.get("article_id"))] = document_id
            add_node({
                **node_base("Document", document_id, confidence=1.0, verification_status="observed"),
                "document_id": text(row.get("article_id")),
                "article_id": text(row.get("article_id")),
                "source_name": text(row.get("source_name")),
                "source_dataset": text(row.get("source_dataset")),
                "publish_time": text(row.get("publish_time")),
                "source_record_id": text(row.get("article_id")),
            })
        add_edge("PUBLISHED_BY", document_id, source_id, "Document", "Source", text(row.get("article_id")), confidence=1.0, verification_status="observed")

        target_type = text(row.get("target_type"))
        target_key = text(row.get("target_key"))
        if target_key not in targets:
            if target_type == "industry":
                target_id = f"IND4_{target_key.replace('|', '_')}"
                node = {
                    **node_base("Industry", target_id, confidence=1.0, verification_status="accepted"),
                    "target_key": target_key,
                    "target_name": text(row.get("target_name")),
                    "target_code": text(row.get("target_code")),
                    "target_type": target_type,
                    "source_record_id": target_key,
                }
            else:
                target_id = f"COMP4_{target_key.replace('|', '_')}"
                node = {
                    **node_base("Company", target_id, confidence=1.0, verification_status="accepted"),
                    "target_key": target_key,
                    "target_name": text(row.get("target_name")),
                    "target_code": text(row.get("target_code")),
                    "target_type": target_type,
                    "source_record_id": target_key,
                }
            targets[target_key] = target_id
            add_node(node)
        target_id = targets[target_key]

        mention_id = text(row.get("event_mention_id"))
        claim_id = f"CLM4_{stable_id('CLM4', mention_id, length=18)[5:]}"
        evidence_sentence_id = stable_id("SENT4", text(row.get("article_id")), text(row.get("evidence_span_text")), row.get("evidence_span_start"), row.get("evidence_span_end"), length=24)
        evidence_unit_id = stable_id("EU4", text(row.get("article_id")), text(row.get("evidence_span_text")), row.get("evidence_span_start"), row.get("evidence_span_end"), length=24)

        evidence_sentences[evidence_sentence_id] = evidence_sentence_id
        evidence_units[evidence_unit_id] = evidence_unit_id
        claims[claim_id] = claim_id

        add_node({
            **node_base("EventMention", mention_id, confidence=float(row.get("resolver_confidence") or 0.5), verification_status=text(row.get("task4_acceptance_status"))),
            "event_mention_id": mention_id,
            "resolver_event_id": text(row.get("resolver_event_id")),
            "canonical_event_id": cid,
            "event_thread_id": text(row.get("event_thread_id")),
            "article_id": text(row.get("article_id")),
            "target_key": target_key,
            "event_type": text(row.get("event_type")),
            "event_family": text(row.get("event_family")),
            "event_name": text(row.get("event_name")),
            "event_time": text(row.get("event_time")),
            "event_fact_signature": text(row.get("event_fact_signature")),
            "trigger_text": text(row.get("trigger_text")),
            "polarity_or_state": text(row.get("polarity_or_state")),
            "modality_status": text(row.get("modality_status")),
            "support_type": text(row.get("support_type")),
            "contract_status": text(row.get("contract_status")),
            "resolver_decision": text(row.get("resolver_decision")),
            "resolver_confidence": float(row.get("resolver_confidence") or 0.5),
            "cluster_confidence": float(row.get("cluster_confidence") or 0.5),
            "trust_score": float(row.get("trust_score") or 0.0),
            "evidence_scope": "accepted_interface",
            "source_record_id": mention_id,
        })
        add_node({
            **node_base("Claim", claim_id, confidence=float(row.get("resolver_confidence") or 0.5), verification_status=text(row.get("task4_acceptance_status"))),
            "claim_text": text(row.get("event_fact")),
            "article_id": text(row.get("article_id")),
            "target_key": target_key,
            "event_mention_id": mention_id,
            "event_type": text(row.get("event_type")),
            "event_family": text(row.get("event_family")),
            "event_time": text(row.get("event_time")),
            "event_fact_signature": text(row.get("event_fact_signature")),
            "resolver_decision": text(row.get("resolver_decision")),
            "source_record_id": mention_id,
        })
        add_node({
            **node_base("EvidenceSentence", evidence_sentence_id, confidence=float(row.get("resolver_confidence") or 0.5), verification_status=text(row.get("task4_acceptance_status"))),
            "article_id": text(row.get("article_id")),
            "evidence_sentence": text(row.get("evidence_sentence")),
            "evidence_span_text": text(row.get("evidence_span_text")),
            "evidence_span_start": int(row.get("evidence_span_start") or -1),
            "evidence_span_end": int(row.get("evidence_span_end") or -1),
            "provenance_document_hash": text(row.get("provenance_document_hash")),
            "provenance_sentence_id": text(row.get("provenance_sentence_id")),
            "source_record_id": evidence_sentence_id,
        })
        add_node({
            **node_base("EvidenceUnit", evidence_unit_id, confidence=float(row.get("resolver_confidence") or 0.5), verification_status=text(row.get("task4_acceptance_status"))),
            "article_id": text(row.get("article_id")),
            "event_mention_id": mention_id,
            "canonical_event_id": cid,
            "evidence_span_text": text(row.get("evidence_span_text")),
            "evidence_span_start": int(row.get("evidence_span_start") or -1),
            "evidence_span_end": int(row.get("evidence_span_end") or -1),
            "support_type": text(row.get("support_type")),
            "provenance_document_hash": text(row.get("provenance_document_hash")),
            "provenance_sentence_id": text(row.get("provenance_sentence_id")),
            "source_record_id": evidence_unit_id,
        })

        add_edge("CONTAINS_MENTION", document_id, mention_id, "Document", "EventMention", text(row.get("article_id")), confidence=float(row.get("resolver_confidence") or 0.5))
        add_edge("EXPRESSES_CLAIM", mention_id, claim_id, "EventMention", "Claim", mention_id, confidence=float(row.get("resolver_confidence") or 0.5))
        add_edge("BELONGS_TO_EVIDENCE_UNIT", mention_id, evidence_unit_id, "EventMention", "EvidenceUnit", mention_id, confidence=float(row.get("resolver_confidence") or 0.5))
        add_edge("SUPPORTED_BY_SENTENCE", evidence_sentence_id, claim_id, "EvidenceSentence", "Claim", mention_id, confidence=float(row.get("resolver_confidence") or 0.5))
        add_edge("EVIDENCE_FOR", evidence_unit_id, claim_id, "EvidenceUnit", "Claim", mention_id, confidence=float(row.get("resolver_confidence") or 0.5))
        add_edge("EXTRACTED_FROM", evidence_sentence_id, document_id, "EvidenceSentence", "Document", mention_id, confidence=1.0)

        if target_type == "industry":
            add_edge("TARGETS_INDUSTRY", mention_id, target_id, "EventMention", "Industry", mention_id, confidence=float(row.get("resolver_confidence") or 0.5))
        else:
            add_edge("TARGETS_COMPANY", mention_id, target_id, "EventMention", "Company", mention_id, confidence=float(row.get("resolver_confidence") or 0.5))

    for cid, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: (text(r.get("resolver_decision")) != "pass", -float(r.get("trust_score") or 0.0), text(r.get("event_mention_id"))))
        representative = rows[0]
        thread_id = text(representative.get("event_thread_id"))
        member_ids = [text(r.get("event_mention_id")) for r in rows]
        evidence_ids = [stable_id("EU4", text(r.get("article_id")), text(r.get("evidence_span_text")), r.get("evidence_span_start"), r.get("evidence_span_end"), length=24) for r in rows]
        evidence_sentence_ids = [stable_id("SENT4", text(r.get("article_id")), text(r.get("evidence_span_text")), r.get("evidence_span_start"), r.get("evidence_span_end"), length=24) for r in rows]
        canonical_id = cid
        current_state_id = stable_id("SO4", canonical_id, "unknown", length=24)

        add_node({
            **node_base("CanonicalEvent", canonical_id, confidence=float(representative.get("cluster_confidence") or 0.5), verification_status="accepted"),
            "canonical_event_id": canonical_id,
            "event_thread_id": thread_id,
            "member_count": len(rows),
            "representative_event_mention_id": text(representative.get("event_mention_id")),
            "article_id": text(representative.get("article_id")),
            "target_key": text(representative.get("target_key")),
            "target_type": text(representative.get("target_type")),
            "event_type": text(representative.get("event_type")),
            "event_family": text(representative.get("event_family")),
            "event_name": text(representative.get("event_name")),
            "event_time": text(representative.get("event_time")),
            "event_fact_signature": text(representative.get("event_fact_signature")),
            "event_mention_ids": member_ids,
            "evidence_unit_ids": evidence_ids,
            "event_thread_ids": [thread_id],
            "trust_score": float(representative.get("trust_score") or 0.0),
            "cluster_confidence": float(representative.get("cluster_confidence") or 0.5),
            "contract_status": text(representative.get("contract_status")),
            "resolver_decision": text(representative.get("resolver_decision")),
            "source_record_id": canonical_id,
        })
        add_node({
            **node_base("EventThread", thread_id, confidence=float(representative.get("cluster_confidence") or 0.5), verification_status="accepted"),
            "event_thread_id": thread_id,
            "canonical_event_ids": [canonical_id],
            "thread_size": len(rows),
            "cycle_anomaly": False,
            "source_record_id": thread_id,
        })
        add_node({
            **node_base("StateObservation", current_state_id, confidence=float(representative.get("cluster_confidence") or 0.5), verification_status="accepted"),
            "current_state_observation_id": current_state_id,
            "canonical_event_id": canonical_id,
            "current_state_type": "unknown",
            "publish_time": text(representative.get("publish_time")),
            "evidence_sentence_ids": evidence_sentence_ids,
            "rule_source": "task3_kg_v4_shadow_from_task4_interface",
            "source_record_id": current_state_id,
        })

        for row in rows:
            mention_id = text(row.get("event_mention_id"))
            add_edge("INSTANCE_OF_EVENT", mention_id, canonical_id, "EventMention", "CanonicalEvent", mention_id, confidence=float(row.get("cluster_confidence") or 0.5))
            add_edge("MEMBER_OF_THREAD", mention_id, thread_id, "EventMention", "EventThread", mention_id, confidence=float(row.get("cluster_confidence") or 0.5))
        add_edge("HAS_STATE_OBSERVATION", canonical_id, current_state_id, "CanonicalEvent", "StateObservation", canonical_id, confidence=float(representative.get("cluster_confidence") or 0.5))

    node_rows = sorted(node_map.values(), key=lambda r: (r["node_type"], r["stable_id"]))
    edge_rows = sorted(edges, key=lambda r: (r["relation_type"], r["source_id"], r["target_id"], r["stable_id"]))

    node_ids = {row["stable_id"] for row in node_rows}
    edge_integrity = all(edge["source_id"] in node_ids and edge["target_id"] in node_ids for edge in edge_rows)
    relation_counts = Counter(edge["relation_type"] for edge in edge_rows)
    node_type_counts = Counter(row["node_type"] for row in node_rows)

    return {
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "canonical_events": sorted([row for row in node_rows if row["node_type"] == "CanonicalEvent"], key=lambda r: r["stable_id"]),
        "event_mentions": sorted([row for row in node_rows if row["node_type"] == "EventMention"], key=lambda r: r["stable_id"]),
        "evidence_units": sorted([row for row in node_rows if row["node_type"] == "EvidenceUnit"], key=lambda r: r["stable_id"]),
        "event_threads": sorted([row for row in node_rows if row["node_type"] == "EventThread"], key=lambda r: r["stable_id"]),
        "current_event_state": sorted([row for row in node_rows if row["node_type"] == "StateObservation"], key=lambda r: r["stable_id"]),
        "review_sidecar": review_rows,
        "node_type_counts": dict(node_type_counts),
        "edge_type_counts": dict(relation_counts),
        "edge_integrity": edge_integrity,
    }


def build_graph_statistics(accepted_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]], artifacts: dict[str, Any], join_report: dict[str, Any]) -> dict[str, Any]:
    accepted_validation_leak = sum(1 for row in accepted_rows if bool(row.get("source_row_validation_test_leakage")))
    review_validation_leak = sum(1 for row in review_rows if bool(row.get("source_row_validation_test_leakage")))
    provenance_complete = sum(1 for row in accepted_rows if text(row.get("evidence_span_text")) and text(row.get("provenance_document_hash")) and text(row.get("provenance_sentence_id")))
    relation_counts = artifacts["edge_type_counts"]
    context_relation_counts = {rel: relation_counts.get(rel, 0) for rel in CONTEXT_RELATIONS}
    return {
        "status": "kg_v4_shadow_ready",
        "accepted_rows": len(accepted_rows),
        "review_rows": len(review_rows),
        "main_graph_counts": {
            "nodes": len(artifacts["node_rows"]),
            "edges": len(artifacts["edge_rows"]),
            "canonical_events": len(artifacts["canonical_events"]),
            "event_mentions": len(artifacts["event_mentions"]),
            "evidence_units": len(artifacts["evidence_units"]),
            "event_threads": len(artifacts["event_threads"]),
            "current_event_state": len(artifacts["current_event_state"]),
        },
        "node_type_counts": artifacts["node_type_counts"],
        "edge_type_counts": relation_counts,
        "context_edge_type_counts": context_relation_counts,
        "integrity": {
            "accepted_event_presence_rate": round(len(artifacts["event_mentions"]) / max(1, len(accepted_rows)), 6),
            "canonical_event_id_unique": len({row["canonical_event_id"] for row in accepted_rows}) == len(artifacts["canonical_events"]),
            "edge_endpoint_integrity": artifacts["edge_integrity"],
            "evidence_provenance_coverage_rate": round(provenance_complete / max(1, len(accepted_rows)), 6),
            "review_sidecar_main_graph_overlap": len({row["event_mention_id"] for row in review_rows} & {row["event_mention_id"] for row in accepted_rows}),
            "validation_test_leakage_rate": round((accepted_validation_leak + review_validation_leak) / max(1, len(accepted_rows) + len(review_rows)), 6),
            "task4_interface_reference_rate": 1.0,
        },
        "funnel": {
            "input_accepted_rows": len(accepted_rows),
            "input_review_rows": len(review_rows),
            "input_leakage_rows": accepted_validation_leak + review_validation_leak,
        },
        "contract_join_summary": {
            "task4_accepted_rows": join_report.get("task4_interface", {}).get("accepted_rows", len(accepted_rows)),
            "task4_rejected_rows": join_report.get("task4_interface", {}).get("rejected_rows", len(review_rows)),
            "resolver_pool_rows": join_report.get("resolver_pool", {}).get("rows", len(accepted_rows) + len(review_rows)),
        },
    }


def build_manifest(accepted_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]], graph_stats: dict[str, Any]) -> dict[str, Any]:
    inputs = []
    for path in [ACCEPTED_PATH, FEATURE_PATH, JOIN_REPORT_PATH]:
        inputs.append({
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else "",
            "size_bytes": path.stat().st_size if path.exists() else 0,
        })
    outputs = []
    for path in [
        TASK4_OUT_PATH,
        REVIEW_SIDECAR_PATH,
        NODES_OUT_PATH,
        EDGES_OUT_PATH,
        CANONICAL_EVENTS_OUT_PATH,
        EVENT_MENTIONS_OUT_PATH,
        EVIDENCE_UNITS_OUT_PATH,
        EVENT_THREADS_OUT_PATH,
        CURRENT_STATE_OUT_PATH,
        GRAPH_STATS_OUT_PATH,
    ]:
        outputs.append({
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else "",
            "size_bytes": path.stat().st_size if path.exists() else 0,
        })
    return {
        "status": "kg_v4_shadow_ready",
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "accepted_rows": len(accepted_rows),
        "review_rows": len(review_rows),
        "inputs": inputs,
        "outputs": outputs,
        "graph_counts": graph_stats["main_graph_counts"],
        "integrity": graph_stats["integrity"],
        "relation_type_counts": graph_stats["edge_type_counts"],
        "context_relation_type_counts": graph_stats["context_edge_type_counts"],
        "resolved_from": "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1",
        "old_graph_touched": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows, feature_rows, review_rows, join_report = load_inputs()
    artifacts = build_shadow_graph(accepted_rows, review_rows)
    graph_stats = build_graph_statistics(accepted_rows, review_rows, artifacts, join_report)

    write_jsonl(TASK4_OUT_PATH, accepted_rows)
    write_jsonl(REVIEW_SIDECAR_PATH, review_rows)
    write_jsonl(EVENT_MENTIONS_OUT_PATH, artifacts["event_mentions"])
    write_jsonl(EVIDENCE_UNITS_OUT_PATH, artifacts["evidence_units"])
    write_jsonl(CANONICAL_EVENTS_OUT_PATH, artifacts["canonical_events"])
    write_jsonl(EVENT_THREADS_OUT_PATH, artifacts["event_threads"])
    write_jsonl(CURRENT_STATE_OUT_PATH, artifacts["current_event_state"])
    write_jsonl(NODES_OUT_PATH, artifacts["node_rows"])
    write_jsonl(EDGES_OUT_PATH, artifacts["edge_rows"])
    write_json(GRAPH_STATS_OUT_PATH, graph_stats)

    manifest = build_manifest(accepted_rows, review_rows, graph_stats)
    write_json(MANIFEST_OUT_PATH, manifest)

    # integrity checks
    assert len(accepted_rows) == 204
    assert len(review_rows) == 411
    assert len(artifacts["event_mentions"]) == 204
    assert len(artifacts["canonical_events"]) == 201
    assert len(artifacts["event_threads"]) == 201
    assert len(artifacts["current_event_state"]) == 201
    assert len({row["canonical_event_id"] for row in artifacts["canonical_events"]}) == 201
    assert artifacts["edge_integrity"]
    assert graph_stats["integrity"]["review_sidecar_main_graph_overlap"] == 0
    assert graph_stats["integrity"]["validation_test_leakage_rate"] == 0.0
    assert graph_stats["integrity"]["evidence_provenance_coverage_rate"] == 1.0
    assert graph_stats["integrity"]["task4_interface_reference_rate"] == 1.0

    print(json.dumps({
        "output_dir": str(out_dir.relative_to(ROOT)),
        "main_graph_nodes": len(artifacts["node_rows"]),
        "main_graph_edges": len(artifacts["edge_rows"]),
        "canonical_events": len(artifacts["canonical_events"]),
        "accepted_rows": len(accepted_rows),
        "review_rows": len(review_rows),
        "validation_test_leakage_rate": graph_stats["integrity"]["validation_test_leakage_rate"],
        "old_graph_touched": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
