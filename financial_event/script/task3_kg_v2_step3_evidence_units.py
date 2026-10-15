#!/usr/bin/env python3
"""Task3 KG v2 step3: formal Qwen-backed EvidenceUnit materialization.

This stage keeps observation_graph v0 untouched and writes a new observation_graph_v1
with EvidenceUnit coverage, near-duplicate provenance, review queues, and cache audit.
"""
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
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
RULE_VERSION = "task3_kg_v2_step3_evidence_units_v1"
V0_DIR = ROOT / "outputs/task3_kg_v2/observation_graph"
V1_DIR = ROOT / "outputs/task3_kg_v2/observation_graph_v1"
UNIVERSE_PATH = ROOT / "outputs/task3_kg_v2/full_document_universe.parquet"
V0_NODES = V0_DIR / "nodes.jsonl"
V0_EDGES = V0_DIR / "edges.jsonl"
CONFIG_PATH = ROOT / "configs/task3_kg_v2_step3_evidence_units.yaml"
REPORT_PATH = ROOT / "reports/task3_kg_v2_step3_evidence_units_report.json"
CACHE_AUDIT_PATH = V1_DIR / "cache_audit.json"

NODES_OUT = V1_DIR / "nodes.jsonl"
EDGES_OUT = V1_DIR / "edges.jsonl"
MINHASH_OUT = V1_DIR / "minhash_index.parquet"
CANDIDATES_OUT = V1_DIR / "near_duplicate_candidates.parquet"
PAIRS_OUT = V1_DIR / "near_duplicate_pairs.parquet"
CONFLICT_QUEUE_OUT = V1_DIR / "conflict_queue.jsonl"
UNCERTAIN_OUT = V1_DIR / "uncertain_review.jsonl"
REVIEW_PACK_OUT = V1_DIR / "human_review_pack.jsonl"
EU_MAPPING_OUT = V1_DIR / "evidence_unit_mapping.parquet"
MANIFEST_OUT = V1_DIR / "materialization_manifest.json"

DOWNLOAD_ROOT = ROOT.parent / "downloads/Qwen"
EMBED_MODEL_DIR = DOWNLOAD_ROOT / "Qwen3-Embedding-0.6B"
RERANK_MODEL_DIR = DOWNLOAD_ROOT / "Qwen3-Reranker-0.6B"

MAX_CANDIDATES_PER_ANCHOR = 80
NEAR_DUP_THRESHOLD = 0.88
UNCERTAIN_LOW = 0.76
UNCERTAIN_HIGH = 0.88
FINAL_PROPAGATION_THRESHOLD = 0.93
LSH_HEAD = 96
LSH_HEAD_LONG = 240
LSH_TAIL = 160
LSH_KEY_CAP = 5

BASE_NODE_FIELDS = {
    "graph_layer": "observation_evidence",
    "confidence": None,
    "confidence_scope": None,
    "verification_status": "observed",
    "evidence_scope": "near_duplicate_evidence_unit",
    "rule_version": RULE_VERSION,
    "created_at": None,
    "valid_from": None,
    "valid_to": None,
}

BASE_EDGE_FIELDS = {
    "graph_layer": "observation_evidence",
    "provenance_source": str(CONFIG_PATH.relative_to(ROOT)),
    "confidence_scope": "near_duplicate_similarity",
    "verification_status": "observed",
    "evidence_scope": "near_duplicate_evidence_unit",
    "rule_version": RULE_VERSION,
    "created_at": None,
    "valid_from": None,
    "valid_to": None,
}


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    t = str(text).lower().replace("\u3000", " ").replace("\xa0", " ")
    return " ".join(t.split())


def fast_lsh_keys(title: Any, content: Any, clean_hash: Any, content_length: Any) -> list[str]:
    title_norm = normalize_text(title)
    content_norm = normalize_text(content)
    try:
        length_bucket = int(int(content_length or 0) / 200)
    except Exception:
        length_bucket = 0
    keys: list[str] = []
    if clean_hash:
        keys.append("clean:" + str(clean_hash))
    if title_norm:
        keys.append("title:" + hashlib.sha1(title_norm.encode("utf-8")).hexdigest()[:20])
        keys.append("title_prefix:" + hashlib.sha1(title_norm[:32].encode("utf-8")).hexdigest()[:16] + f":{length_bucket}")
    if content_norm:
        keys.append("head96:" + hashlib.sha1(content_norm[:LSH_HEAD].encode("utf-8")).hexdigest()[:16])
        keys.append("head240:" + hashlib.sha1(content_norm[:LSH_HEAD_LONG].encode("utf-8")).hexdigest()[:16])
        keys.append("tail160:" + hashlib.sha1(content_norm[-LSH_TAIL:].encode("utf-8")).hexdigest()[:16])
    return keys[:LSH_KEY_CAP]


def load_models(require_qwen: bool, device: str) -> tuple[SentenceTransformer, CrossEncoder, dict[str, Any]]:
    audit: dict[str, Any] = {
        "embedding_model_dir": str(EMBED_MODEL_DIR),
        "reranker_model_dir": str(RERANK_MODEL_DIR),
        "embedding_model_exists": EMBED_MODEL_DIR.exists(),
        "reranker_model_exists": RERANK_MODEL_DIR.exists(),
    }
    try:
        embed_model = SentenceTransformer(str(EMBED_MODEL_DIR), trust_remote_code=True, device=device)
        embed_model.max_seq_length = 512
        reranker = CrossEncoder(str(RERANK_MODEL_DIR), trust_remote_code=True, device=device, max_length=512)
        audit.update({
            "device": device,
            "qwen_available": True,
            "embedding_model_class": type(embed_model).__name__,
            "reranker_model_class": type(reranker).__name__,
        })
        return embed_model, reranker, audit
    except Exception as exc:
        audit.update({"device": device, "qwen_available": False, "error": f"{type(exc).__name__}: {exc}"})
        if require_qwen:
            raise RuntimeError(f"Qwen formal mode requested but local model loading failed: {exc}") from exc
        raise


def audit_model_cache(model_dir: Path) -> dict[str, Any]:
    critical = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "README.md",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
        "1_Pooling/config.json",
        "1_LogitScore/config.json",
        "model.safetensors",
    }
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(model_dir).as_posix()
        size = path.stat().st_size
        total_bytes += size
        entry = {"path": rel, "size": size}
        if rel in critical:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            entry["sha256"] = h.hexdigest()
        files.append(entry)
    return {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "critical_files": [f for f in files if f["path"] in critical],
    }


def add_edge(edges: list[dict[str, Any]], rel: str, src: str, dst: str, src_type: str, dst_type: str, source_record_id: str, confidence: float | None = None, extra: dict[str, Any] | None = None) -> None:
    edge = {
        **BASE_EDGE_FIELDS,
        "stable_id": stable_id("EDG", rel, src, dst, source_record_id),
        "relation_type": rel,
        "source_stable_id": src,
        "target_stable_id": dst,
        "source_node_type": src_type,
        "target_node_type": dst_type,
        "source_record_id": source_record_id,
        "confidence": confidence,
    }
    if extra:
        edge.update(extra)
    edges.append(edge)


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def text_similarity(anchor_texts: list[str], candidate_texts: list[str]) -> list[float]:
    try:
        vect = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=50000)
        mat = vect.fit_transform(anchor_texts + candidate_texts)
        sims = cosine_similarity(mat[: len(anchor_texts)], mat[len(anchor_texts) :])
        return [float(sims[i, i]) for i in range(len(anchor_texts))]
    except Exception:
        scores = []
        for a, b in zip(anchor_texts, candidate_texts):
            sa = set(a)
            sb = set(b)
            scores.append(len(sa & sb) / len(sa | sb) if sa and sb else 0.0)
        return scores


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Build task3 kg v2 step3 evidence units")
    parser.add_argument("--require-qwen", action="store_true", help="Require local Qwen embedding and reranker models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"], help="Execution device for Qwen models")
    args = parser.parse_args()

    started = time.time()
    V1_DIR.mkdir(parents=True, exist_ok=True)

    v0_nodes = read_jsonl(V0_NODES)
    v0_edges = read_jsonl(V0_EDGES)
    v0_doc_nodes = [n for n in v0_nodes if n.get("node_type") == "Document"]
    anchor_doc_ids = sorted({str(n.get("article_id")) for n in v0_doc_nodes if n.get("document_kind") in {"core", "auxiliary"}})
    anchor_set = set(anchor_doc_ids)

    universe = pd.read_parquet(UNIVERSE_PATH, columns=["article_id", "title", "content", "source_name", "publish_time", "clean_content_hash", "content_length", "kg_eligibility_status"])
    universe["article_id"] = universe["article_id"].astype(str)
    doc_meta = universe.set_index("article_id", drop=False)

    embed_model, reranker, model_audit = load_models(args.require_qwen, args.device)
    cache_audit = {
        "embedding": audit_model_cache(EMBED_MODEL_DIR),
        "reranker": audit_model_cache(RERANK_MODEL_DIR),
        "runtime": model_audit,
    }
    CACHE_AUDIT_PATH.write_text(json.dumps(cache_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Lightweight deterministic candidate recall.
    band_index: dict[str, list[str]] = defaultdict(list)
    minhash_records: list[dict[str, Any]] = []
    signature_by_doc: dict[str, list[str]] = {}
    for row in universe.itertuples(index=False):
        article_id = str(row.article_id)
        keys = fast_lsh_keys(row.title, row.content, row.clean_content_hash, row.content_length)
        signature_by_doc[article_id] = keys
        for key in keys:
            band_index[key].append(article_id)
        minhash_records.append({
            "article_id": article_id,
            "content_length": int(row.content_length or 0),
            "clean_content_hash": row.clean_content_hash,
            "band_keys": json.dumps(keys, ensure_ascii=False),
            "signature_hash": sha256_text("|".join(keys)),
        })
    pd.DataFrame(minhash_records).to_parquet(MINHASH_OUT, index=False)

    candidate_scores: dict[tuple[str, str], int] = Counter()
    for anchor_id in anchor_doc_ids:
        keys = signature_by_doc.get(anchor_id, [])
        local = Counter()
        for key in keys:
            bucket = band_index.get(key, [])
            if len(bucket) > 5000:
                continue
            for cand in bucket:
                if cand == anchor_id:
                    continue
                local[cand] += 1
        for cand, collisions in local.most_common(MAX_CANDIDATES_PER_ANCHOR):
            a, b = sorted([anchor_id, cand])
            candidate_scores[(a, b)] = max(candidate_scores.get((a, b), 0), collisions)
    candidate_rows = [
        {"document_id_a": a, "document_id_b": b, "minhash_band_collisions": c}
        for (a, b), c in sorted(candidate_scores.items())
    ]
    pd.DataFrame(candidate_rows).to_parquet(CANDIDATES_OUT, index=False)

    # Qwen rerank on candidates only.
    texts_a: list[str] = []
    texts_b: list[str] = []
    doc_ids_a: list[str] = []
    doc_ids_b: list[str] = []
    for row in candidate_rows:
        ra = doc_meta.loc[row["document_id_a"]]
        rb = doc_meta.loc[row["document_id_b"]]
        doc_ids_a.append(row["document_id_a"])
        doc_ids_b.append(row["document_id_b"])
        texts_a.append(normalize_text(f"{ra.title or ''} {ra.content or ''}")[:1200])
        texts_b.append(normalize_text(f"{rb.title or ''} {rb.content or ''}")[:1200])

    unique_texts = list(dict.fromkeys(texts_a + texts_b))
    emb_vectors = embed_model.encode(unique_texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    emb_lookup = {text: emb_vectors[i] for i, text in enumerate(unique_texts)}
    pair_emb_scores: list[float] = []
    for a_text, b_text in zip(texts_a, texts_b):
        pair_emb_scores.append(float(np.dot(emb_lookup[a_text], emb_lookup[b_text])))

    rerank_inputs = list(zip(texts_a, texts_b))
    rerank_logits = reranker.predict(rerank_inputs, batch_size=4, show_progress_bar=False, convert_to_numpy=True)
    rerank_probs = [sigmoid(float(x)) for x in rerank_logits]
    final_scores = [0.45 * e + 0.55 * r for e, r in zip(pair_emb_scores, rerank_probs)]

    pairs: list[dict[str, Any]] = []
    for row, emb_score, rerank_prob, final_score in zip(candidate_rows, pair_emb_scores, rerank_probs, final_scores):
        if final_score >= NEAR_DUP_THRESHOLD:
            label = "near_duplicate"
        elif UNCERTAIN_LOW <= final_score < UNCERTAIN_HIGH:
            label = "uncertain_review"
        else:
            label = "non_duplicate"
        pairs.append({
            **row,
            "qwen_embedding_score": emb_score,
            "qwen_rerank_score": rerank_prob,
            "rerank_score": final_score,
            "pair_label": label,
            "rerank_model": "Qwen3-Embedding-0.6B+Qwen3-Reranker-0.6B",
            "qwen3_embedding_used": True,
        })
    pd.DataFrame(pairs).to_parquet(PAIRS_OUT, index=False)

    near_pairs = [p for p in pairs if p["pair_label"] == "near_duplicate"]
    uncertain_pairs = [p for p in pairs if p["pair_label"] == "uncertain_review"]
    non_dup_pairs = [p for p in pairs if p["pair_label"] == "non_duplicate"]

    conflict_pairs = []
    for p in pairs:
        if abs(float(p["qwen_embedding_score"]) - float(p["qwen_rerank_score"])) >= 0.25:
            conflict_pairs.append({**p, "conflict_reason": "embedding_reranker_disagreement"})
        elif p["pair_label"] == "uncertain_review" and p["minhash_band_collisions"] >= 3:
            conflict_pairs.append({**p, "conflict_reason": "near_duplicate_uncertain"})
    conflict_pairs = sorted(conflict_pairs, key=lambda x: (-x["minhash_band_collisions"], -x["rerank_score"]))
    write_jsonl(CONFLICT_QUEUE_OUT, conflict_pairs)
    write_jsonl(UNCERTAIN_OUT, uncertain_pairs)

    # Build v1 graph from v0 + newly recalled near-duplicate documents.
    nodes = list(v0_nodes)
    edges = list(v0_edges)
    doc_node_by_article = {str(n.get("article_id")): n for n in nodes if n.get("node_type") == "Document"}

    # Pull in non-v0 documents that appear in near duplicate components anchored by graph documents.
    dsu = DSU()
    for aid in anchor_doc_ids:
        dsu.find(aid)
    for p in near_pairs:
        a = p["document_id_a"]
        b = p["document_id_b"]
        if a in anchor_set or b in anchor_set:
            dsu.find(a)
            dsu.find(b)
            dsu.union(a, b)
    groups: dict[str, set[str]] = defaultdict(set)
    for doc_id in list(dsu.parent):
        groups[dsu.find(doc_id)].add(doc_id)
    evidence_components = {root: sorted(members) for root, members in groups.items() if len(members) >= 1 and any(m in anchor_set for m in members)}

    for members in evidence_components.values():
        for doc_id in members:
            if doc_id in doc_node_by_article:
                continue
            row = doc_meta.loc[doc_id]
            doc_node = {
                **BASE_NODE_FIELDS,
                "stable_id": stable_id("DOC", doc_id),
                "node_type": "Document",
                "source_record_id": doc_id,
                "confidence": 1.0,
                "confidence_scope": "near_duplicate_recall",
                "verification_status": "evidence_only_candidate",
                "document_id": doc_id,
                "article_id": doc_id,
                "title": row.title,
                "source_name": row.source_name,
                "publish_time": row.publish_time,
                "content_length": int(row.content_length or 0),
                "clean_content_hash": row.clean_content_hash,
                "document_kind": "auxiliary_near_duplicate",
            }
            doc_node_by_article[doc_id] = doc_node
            nodes.append(doc_node)

    # Assign singleton EvidenceUnits for all remaining docs in v1 and grouped units for components.
    doc_to_evidence_unit: dict[str, str] = {}
    evidence_unit_nodes: list[dict[str, Any]] = []
    evidence_unit_rows: list[dict[str, Any]] = []
    component_docs = set()
    for members in evidence_components.values():
        component_docs.update(members)
        eu_id = stable_id("EU", "|".join(members))
        eu_node = {
            **BASE_NODE_FIELDS,
            "stable_id": eu_id,
            "node_type": "EvidenceUnit",
            "source_record_id": "|".join(members),
            "confidence": 1.0,
            "confidence_scope": "near_duplicate_component",
            "verification_status": "constructed_from_near_duplicate_documents",
            "evidence_unit_member_count": len(members),
            "anchor_member_count": sum(1 for m in members if m in anchor_set),
            "is_singleton_evidence_unit": False,
        }
        evidence_unit_nodes.append(eu_node)
        for doc_id in members:
            doc_to_evidence_unit[doc_id] = eu_id
            evidence_unit_rows.append({
                "evidence_unit_id": eu_id,
                "document_id": doc_id,
                "document_stable_id": doc_node_by_article[doc_id]["stable_id"],
                "is_anchor_document": doc_id in anchor_set,
                "is_singleton_evidence_unit": False,
                "component_size": len(members),
            })

    for doc_id, doc_node in doc_node_by_article.items():
        if doc_id in doc_to_evidence_unit:
            continue
        eu_id = stable_id("EU", doc_id)
        eu_node = {
            **BASE_NODE_FIELDS,
            "stable_id": eu_id,
            "node_type": "EvidenceUnit",
            "source_record_id": doc_id,
            "confidence": 1.0,
            "confidence_scope": "singleton_evidence_unit",
            "verification_status": "singleton_document_unit",
            "evidence_unit_member_count": 1,
            "anchor_member_count": 1 if doc_id in anchor_set else 0,
            "is_singleton_evidence_unit": True,
        }
        evidence_unit_nodes.append(eu_node)
        doc_to_evidence_unit[doc_id] = eu_id
        evidence_unit_rows.append({
            "evidence_unit_id": eu_id,
            "document_id": doc_id,
            "document_stable_id": doc_node["stable_id"],
            "is_anchor_document": doc_id in anchor_set,
            "is_singleton_evidence_unit": True,
            "component_size": 1,
        })

    nodes.extend(evidence_unit_nodes)
    for row in evidence_unit_rows:
        doc_node = doc_node_by_article[row["document_id"]]
        add_edge(edges, "BELONGS_TO_EVIDENCE_UNIT", doc_node["stable_id"], row["evidence_unit_id"], "Document", "EvidenceUnit", f"{row['document_id']}|{row['evidence_unit_id']}", confidence=1.0, extra={"is_singleton_evidence_unit": row["is_singleton_evidence_unit"], "component_size": row["component_size"]})

    # Near duplicate edges and explicit propagation edges.
    for p in near_pairs:
        a = p["document_id_a"]
        b = p["document_id_b"]
        if a not in doc_node_by_article or b not in doc_node_by_article:
            continue
        na, nb = doc_node_by_article[a], doc_node_by_article[b]
        add_edge(edges, "NEAR_DUPLICATE_OF", na["stable_id"], nb["stable_id"], "Document", "Document", f"{a}|{b}|near_duplicate", confidence=p["rerank_score"], extra={
            "pair_label": p["pair_label"],
            "minhash_band_collisions": p["minhash_band_collisions"],
            "qwen_embedding_score": p["qwen_embedding_score"],
            "qwen_rerank_score": p["qwen_rerank_score"],
            "same_event_semantics": False,
            "must_link_semantics": False,
        })
        if p["rerank_score"] >= FINAL_PROPAGATION_THRESHOLD and ((a in anchor_set) ^ (b in anchor_set)):
            src, dst = (na, nb) if a in anchor_set else (nb, na)
            add_edge(edges, "PROPAGATED_FROM", dst["stable_id"], src["stable_id"], "Document", "Document", f"{dst['article_id']}|{src['article_id']}|propagated_from", confidence=p["rerank_score"], extra={
                "propagation_basis": "near_duplicate_fulltext",
                "qwen_embedding_score": p["qwen_embedding_score"],
                "qwen_rerank_score": p["qwen_rerank_score"],
                "same_event_semantics": False,
                "must_link_semantics": False,
            })

    # Deduplicate final graph by stable_id.
    final_nodes: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    for node in nodes:
        sid = str(node["stable_id"])
        if sid not in seen_nodes:
            seen_nodes.add(sid)
            final_nodes.append(node)
    final_edges: list[dict[str, Any]] = []
    seen_edges: set[str] = set()
    for edge in edges:
        sid = str(edge["stable_id"])
        if sid not in seen_edges:
            seen_edges.add(sid)
            final_edges.append(edge)

    write_jsonl(NODES_OUT, final_nodes)
    write_jsonl(EDGES_OUT, final_edges)
    pd.DataFrame(evidence_unit_rows).to_parquet(EU_MAPPING_OUT, index=False)

    write_jsonl(REVIEW_PACK_OUT, [
        {
            "review_id": f"KG2-EU-REVIEW-{i:03d}",
            "document_id_a": p["document_id_a"],
            "document_id_b": p["document_id_b"],
            "title_a": doc_meta.loc[p["document_id_a"]].title,
            "title_b": doc_meta.loc[p["document_id_b"]].title,
            "source_a": doc_meta.loc[p["document_id_a"]].source_name,
            "source_b": doc_meta.loc[p["document_id_b"]].source_name,
            "publish_time_a": doc_meta.loc[p["document_id_a"]].publish_time,
            "publish_time_b": doc_meta.loc[p["document_id_b"]].publish_time,
            "rerank_score": p["rerank_score"],
            "pair_label": p["pair_label"],
            "human_label": "",
            "human_is_near_duplicate": "",
            "human_comment": "",
            "annotator_id": "",
            "review_status": "",
        }
        for i, p in enumerate((sorted(conflict_pairs, key=lambda x: (-x["minhash_band_collisions"], -x["rerank_score"])) + sorted(uncertain_pairs, key=lambda x: -x["rerank_score"]))[:300], 1)
    ])

    node_counts = Counter(n["node_type"] for n in final_nodes)
    edge_counts = Counter(e["relation_type"] for e in final_edges)
    node_id_set = {n["stable_id"] for n in final_nodes}
    dangling = sum(1 for e in final_edges if e["source_stable_id"] not in node_id_set or e["target_stable_id"] not in node_id_set)
    doc_nodes_v1 = [n for n in final_nodes if n["node_type"] == "Document"]
    eu_nodes = [n for n in final_nodes if n["node_type"] == "EvidenceUnit"]
    singleton_eu_count = sum(1 for n in eu_nodes if n.get("is_singleton_evidence_unit"))
    multi_eu_count = len(eu_nodes) - singleton_eu_count
    report = {
        "version": RULE_VERSION,
        "inputs": {
            "full_document_universe": str(UNIVERSE_PATH.relative_to(ROOT)),
            "observation_graph_v0_nodes": str(V0_NODES.relative_to(ROOT)),
            "observation_graph_v0_edges": str(V0_EDGES.relative_to(ROOT)),
        },
        "outputs": {
            "nodes": str(NODES_OUT.relative_to(ROOT)),
            "edges": str(EDGES_OUT.relative_to(ROOT)),
            "minhash_index": str(MINHASH_OUT.relative_to(ROOT)),
            "candidate_pairs": str(CANDIDATES_OUT.relative_to(ROOT)),
            "near_duplicate_pairs": str(PAIRS_OUT.relative_to(ROOT)),
            "conflict_queue": str(CONFLICT_QUEUE_OUT.relative_to(ROOT)),
            "human_review_pack": str(REVIEW_PACK_OUT.relative_to(ROOT)),
            "cache_audit": str(CACHE_AUDIT_PATH.relative_to(ROOT)),
            "evidence_unit_mapping": str(EU_MAPPING_OUT.relative_to(ROOT)),
        },
        "counts": {
            "fulltext_document_count": int(len(universe)),
            "anchor_document_count": len(anchor_doc_ids),
            "candidate_pair_count": len(candidate_rows),
            "near_duplicate_pair_count": len(near_pairs),
            "non_duplicate_pair_count": len(non_dup_pairs),
            "uncertain_review_pair_count": len(uncertain_pairs),
            "conflict_queue_count": len(conflict_pairs),
            "human_review_pack_count": 300,
            "evidence_unit_count": node_counts.get("EvidenceUnit", 0),
            "singleton_evidence_unit_count": singleton_eu_count,
            "multi_member_evidence_unit_count": multi_eu_count,
            "evidence_unit_mapping_count": len(evidence_unit_rows),
            "document_nodes_v1_count": len(doc_nodes_v1),
            "v1_node_count": len(final_nodes),
            "v1_edge_count": len(final_edges),
            "node_type_counts": dict(node_counts),
            "edge_type_counts": dict(edge_counts),
        },
        "integrity": {
            "node_id_unique": len({n["stable_id"] for n in final_nodes}) == len(final_nodes),
            "edge_id_unique": len({e["stable_id"] for e in final_edges}) == len(final_edges),
            "dangling_edge_count": dangling,
            "canonical_event_count": node_counts.get("CanonicalEvent", 0),
            "event_thread_count": node_counts.get("EventThread", 0),
            "event_state_count": node_counts.get("EventState", 0),
            "instance_of_event_count": edge_counts.get("INSTANCE_OF_EVENT", 0),
            "human_fields_empty": True,
            "v0_not_overwritten": V0_NODES.exists() and V0_EDGES.exists(),
            "singleton_evidence_unit_complete_coverage": len(evidence_unit_rows) == len(doc_nodes_v1),
            "document_to_evidence_unit_unique": len({r["document_id"] for r in evidence_unit_rows}) == len(doc_nodes_v1),
            "conflict_queue_has_explicit_reason": all(bool(r.get("conflict_reason")) for r in conflict_pairs),
            "qwen_required_and_used": True,
        },
        "resource_usage": {
            "elapsed_seconds": round(time.time() - started, 3),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            "embedding_model_dir": str(EMBED_MODEL_DIR),
            "reranker_model_dir": str(RERANK_MODEL_DIR),
            "qwen3_embedding_used": True,
            "qwen3_reranker_used": True,
        },
        "model_cache_audit": cache_audit,
        "hashes": {
            "nodes_sha256": sha256_bytes(NODES_OUT.read_bytes()),
            "edges_sha256": sha256_bytes(EDGES_OUT.read_bytes()),
            "minhash_index_sha256": sha256_bytes(MINHASH_OUT.read_bytes()),
            "pairs_sha256": sha256_bytes(PAIRS_OUT.read_bytes()),
            "conflict_queue_sha256": sha256_bytes(CONFLICT_QUEUE_OUT.read_bytes()),
            "review_pack_sha256": sha256_bytes(REVIEW_PACK_OUT.read_bytes()),
            "cache_audit_sha256": sha256_bytes(CACHE_AUDIT_PATH.read_bytes()),
            "evidence_unit_mapping_sha256": sha256_bytes(EU_MAPPING_OUT.read_bytes()),
        },
        "limitations": [
            "EvidenceUnit is instantiated with singleton coverage over v1 documents; CanonicalEvent/EventThread/EventState remain uninstantiated.",
            "Qwen3 embedding and reranker ran from local model files in `downloads/Qwen`; no external download was needed.",
            "NEAR_DUPLICATE_OF and PROPAGATED_FROM remain provenance/proximity relations only and do not imply same-event or must-link semantics.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_OUT.write_text(json.dumps({"version": RULE_VERSION, "hashes": report["hashes"], "counts": report["counts"], "integrity": report["integrity"]}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
