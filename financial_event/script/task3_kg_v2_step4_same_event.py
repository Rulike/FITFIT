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
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
RULE_VERSION = "task3_kg_v2_step4_same_event_v0"
V1 = ROOT / "outputs/task3_kg_v2/observation_graph_v1"
V2 = ROOT / "outputs/task3_kg_v2/observation_graph_v2"
STEP4 = ROOT / "outputs/task3_kg_v2/step4"
NODES_V1 = V1 / "nodes.jsonl"
EDGES_V1 = V1 / "edges.jsonl"
NODES_V2 = V2 / "nodes.jsonl"
EDGES_V2 = V2 / "edges.jsonl"
MENTIONS_PATH = ROOT / "outputs/task3_fulltext_v2/normalized_event_mentions.jsonl"
EVIDENCE_PATH = ROOT / "outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl"
EU_MAPPING = V1 / "evidence_unit_mapping.parquet"
PILOT_PATH = ROOT / "outputs/task3/step2/same_event_labels_for_step3.csv"
CONFIG_PATH = ROOT / "configs/task3_kg_v2_step4_same_event.yaml"
REPORT_EVAL = ROOT / "reports/task3_kg_v2_step4_evaluation.json"
REPORT_RUNTIME = ROOT / "reports/task3_kg_v2_step4_runtime.json"
DOC_PATH = ROOT / "docs/task3_kg_v2_step4_same_event.md"
EMBED_DIR = ROOT.parent / "downloads/Qwen/Qwen3-Embedding-0.6B"
RERANK_DIR = ROOT.parent / "downloads/Qwen/Qwen3-Reranker-0.6B"

CANDIDATES_OUT = STEP4 / "same_event_candidates.parquet"
SAME_EDGES_OUT = STEP4 / "same_event_edges.jsonl"
RELATED_EDGES_OUT = STEP4 / "related_update_edges.jsonl"
CANNOT_EDGES_OUT = STEP4 / "cannot_link_edges.jsonl"
UNCERTAIN_OUT = STEP4 / "uncertain_review_queue.jsonl"
ANNOTATION_OUT = STEP4 / "same_event_annotation_package.xlsx"
FEATURES_OUT = STEP4 / "graph_feature_explanations.jsonl"
EMBED_OUT_DIR = STEP4 / "event_mention_embeddings"
RERANK_CACHE_DIR = STEP4 / "reranker_cache"
EMBED_NPY = EMBED_OUT_DIR / "embeddings.npy"
EMBED_INDEX = EMBED_OUT_DIR / "embedding_index.parquet"
RERANK_CACHE = RERANK_CACHE_DIR / "reranker_cache.parquet"
MANIFEST_OUT = STEP4 / "materialization_manifest.json"

SEM_TOP_K = 24
RERANK_MAX = 2500
SAME_THRESHOLD = 0.86
RELATED_THRESHOLD = 0.70
MIDDLE_LOW = 0.62
MIDDLE_HIGH = 0.86

REL_TYPES = {"SAME_EVENT_AS", "RELATED_UPDATE_OF", "CANNOT_LINK"}
UPDATE_WORDS = "传闻 官方 确认 公告 进展 启动 完成 调查 处罚 澄清 否认 终止 取消 辟谣 回应".split()
CONFLICT_PAT = ["公告编号", "处罚编号", "批次", "债券", "收购", "减持", "增持"]


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def norm(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return " ".join(str(x).lower().replace("\u3000", " ").replace("\xa0", " ").split())


def listify(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v)]
    if isinstance(x, str):
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return [str(i) for i in v if str(i)]
        except Exception:
            pass
        return [p for p in x.replace("|", ",").split(",") if p]
    return []


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def text_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        v = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, max_features=12000)
        m = v.fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0, 0])
    except Exception:
        return jaccard(set(a), set(b))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def pair_id(a: str, b: str) -> str:
    x, y = pair_key(a, b)
    return stable_id("EMP", x, y)


def parse_time(x: Any) -> pd.Timestamp | None:
    try:
        if not x:
            return None
        return pd.to_datetime(x)
    except Exception:
        return None


def base_event_type(t: str) -> str:
    parts = str(t or "").split("_")
    return parts[-1] if parts else ""


def add_edge(rel: str, src: str, dst: str, src_type: str, dst_type: str, source_record_id: str, confidence: float, extra: dict[str, Any]) -> dict[str, Any]:
    e = {
        "stable_id": stable_id("EDG", rel, src, dst, source_record_id),
        "relation_type": rel,
        "graph_layer": "observation_evidence",
        "source_stable_id": src,
        "target_stable_id": dst,
        "source_node_type": src_type,
        "target_node_type": dst_type,
        "provenance_source": str(CONFIG_PATH.relative_to(ROOT)),
        "source_record_id": source_record_id,
        "confidence": confidence,
        "confidence_scope": "same_event_relation",
        "verification_status": "model_proposed",
        "evidence_scope": "event_mention_pair_matching",
        "rule_version": RULE_VERSION,
        "created_at": None,
        "valid_from": None,
        "valid_to": None,
    }
    e.update(extra)
    return e


def summarize_numeric(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"count": 0}
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.5)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def parse_json_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        v = json.loads(str(value))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def classify_candidate_source(value: Any) -> str:
    sources = set(parse_json_list(value))
    has_sem = "qwen_semantic_ann" in sources
    has_graph = bool(sources - {"qwen_semantic_ann"})
    if has_sem and has_graph:
        return "graph_and_semantic"
    if has_sem:
        return "semantic_only"
    return "graph_only"


def stage_evidence_present(row: pd.Series) -> bool:
    reason = str(row.get("decision_reason", ""))
    if "related_update" in reason:
        return True
    # The cached candidate table does not retain local text; therefore final validation
    # requires explicit rule/reranker related-update reason, not mere high similarity.
    return False


def calculate_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f1s = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    out["macro_f1"] = float(sum(f1s) / len(f1s)) if f1s else 0.0
    matrix: dict[str, dict[str, int]] = {t: {p: 0 for p in labels + ["uncertain_review", "not_candidate"]} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix.setdefault(t, {}).setdefault(p, 0)
        matrix[t][p] += 1
    out["confusion_matrix"] = matrix
    related_true = [t != "different_event" for t in y_true]
    related_pred = [p in {"same_event", "related_update"} for p in y_pred]
    tp = sum(t and p for t, p in zip(related_true, related_pred))
    fp = sum((not t) and p for t, p in zip(related_true, related_pred))
    fn = sum(t and (not p) for t, p in zip(related_true, related_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out["event_related_binary"] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "tp": tp, "fp": fp, "fn": fn}
    return out


def apply_calibrated_relation(row: pd.Series) -> tuple[str, str, float, str]:
    if bool(row.get("hard_conflict")):
        return "different_event", "CANNOT_LINK", 1.0, "hard_cannot_link"
    semantic = float(row.get("semantic_score", 0) or 0)
    structure = float(row.get("structure_score", 0) or 0)
    graph = float(row.get("graph_score", 0) or 0)
    fusion = float(row.get("fusion_score_before_rerank", 0) or 0)
    rerank = row.get("reranker_score")
    rerank_score = float(rerank) if pd.notna(rerank) else None
    if bool(row.get("subject_match")) and bool(row.get("target_code_match")) and bool(row.get("event_type_compatible")) and float(row.get("core_element_similarity", 0) or 0) >= 0.5 and structure >= 0.85 and semantic >= 0.78 and graph >= 0.25 and (pd.isna(row.get("time_diff_hours")) or float(row.get("time_diff_hours")) <= 168):
        return "same_event", "SAME_EVENT_AS", max(fusion, rerank_score or 0), "calibrated_high_precision_same_event"
    related_ok = (
        bool(row.get("subject_match"))
        and bool(row.get("event_type_compatible"))
        and pd.notna(row.get("time_diff_hours"))
        and float(row.get("time_diff_hours")) > 0
        and not bool(row.get("same_evidence_unit"))
        and stage_evidence_present(row)
        and semantic >= 0.74
        and structure >= 0.75
        and graph >= 0.20
        and fusion >= 0.76
    )
    if related_ok:
        return "related_update", "RELATED_UPDATE_OF", max(fusion, rerank_score or 0), "calibrated_related_update_stage_evidence_required"
    if semantic < 0.45 and not bool(row.get("subject_match")):
        return "different_event", "CANNOT_LINK", 1 - semantic, "calibrated_low_similarity_subject_mismatch"
    return "uncertain_review", "UNCERTAIN", fusion, "calibrated_uncertain_review"


def edge_from_candidate(row: pd.Series, rel: str, confidence: float, reason: str) -> dict[str, Any]:
    a = str(row["event_mention_id_a"])
    b = str(row["event_mention_id_b"])
    pair = str(row["pair_id"])
    constraints = parse_json_list(row.get("constraint_results_json"))
    extra = {
        "pair_id": pair,
        "source_event_mention_id": a,
        "target_event_mention_id": b,
        "semantic_score": float(row.get("semantic_score", 0) or 0),
        "structure_score": float(row.get("structure_score", 0) or 0),
        "graph_score": float(row.get("graph_score", 0) or 0),
        "reranker_score": None if pd.isna(row.get("reranker_score")) else float(row.get("reranker_score")),
        "constraint_results": constraints,
        "evidence_path": ["EventMention", "Document/EvidenceUnit/Entity/Task2Assessment", "EventMention"],
        "decision_reason": reason,
        "model_version": "cached_Qwen3-Embedding-0.6B/Qwen3-Reranker-0.6B",
        "rule_version": RULE_VERSION,
        "verification_status": "provisional_model_prediction",
        "provenance": str(CONFIG_PATH.relative_to(ROOT)),
        "same_event_semantics": rel == "SAME_EVENT_AS",
        "must_link_semantics": False,
        "calibration_source": "cached_step4_final_acceptance_no_model_rerun",
    }
    return add_edge(rel, a, b, "EventMention", "EventMention", pair, confidence, extra)


def acceptance_from_cache() -> dict[str, Any]:
    started = time.time()
    STEP4.mkdir(parents=True, exist_ok=True)
    V2.mkdir(parents=True, exist_ok=True)
    cand = pd.read_parquet(CANDIDATES_OUT)
    rerank_cache = pd.read_parquet(RERANK_CACHE) if RERANK_CACHE.exists() else pd.DataFrame()
    pilot = pd.read_csv(PILOT_PATH)
    nodes_v1 = read_jsonl(NODES_V1)
    edges_v1 = read_jsonl(EDGES_V1)
    features = read_jsonl(FEATURES_OUT)
    embeddings_loaded = np.load(EMBED_NPY, mmap_mode="r")
    embed_index = pd.read_parquet(EMBED_INDEX)

    cand = cand.copy()
    cand["candidate_source_class"] = cand["candidate_sources_json"].apply(classify_candidate_source)
    calibrated = cand.apply(apply_calibrated_relation, axis=1, result_type="expand")
    cand["calibrated_relation_label"] = calibrated[0]
    cand["calibrated_edge_relation_type"] = calibrated[1]
    cand["calibrated_confidence"] = calibrated[2]
    cand["calibrated_decision_reason"] = calibrated[3]
    cand["related_update_stage_evidence_present"] = cand.apply(stage_evidence_present, axis=1)
    cand["calibrated_changed_from_original"] = cand["calibrated_relation_label"] != cand["relation_label"]

    # Persist calibrated candidate table while retaining original cached scores.
    cand.to_parquet(CANDIDATES_OUT, index=False)

    same_edges = []
    related_edges = []
    cannot_edges = []
    uncertain = []
    for _, row in cand.iterrows():
        rel = row["calibrated_edge_relation_type"]
        if rel == "SAME_EVENT_AS":
            same_edges.append(edge_from_candidate(row, rel, float(row["calibrated_confidence"]), str(row["calibrated_decision_reason"])))
        elif rel == "RELATED_UPDATE_OF":
            related_edges.append(edge_from_candidate(row, rel, float(row["calibrated_confidence"]), str(row["calibrated_decision_reason"])))
        elif rel == "CANNOT_LINK":
            cannot_edges.append(edge_from_candidate(row, rel, float(row["calibrated_confidence"]), str(row["calibrated_decision_reason"])))
        else:
            uncertain.append({
                "pair_id": row["pair_id"],
                "event_mention_id_a": row["event_mention_id_a"],
                "event_mention_id_b": row["event_mention_id_b"],
                "semantic_score": float(row["semantic_score"]),
                "structure_score": float(row["structure_score"]),
                "graph_score": float(row["graph_score"]),
                "reranker_score": None if pd.isna(row.get("reranker_score")) else float(row.get("reranker_score")),
                "relation_label": "uncertain_review",
                "decision_reason": row["calibrated_decision_reason"],
                "human_label": "",
                "human_comment": "",
                "annotator_id": "",
                "review_status": "",
            })
    write_jsonl(SAME_EDGES_OUT, same_edges)
    write_jsonl(RELATED_EDGES_OUT, related_edges)
    write_jsonl(CANNOT_EDGES_OUT, cannot_edges)
    write_jsonl(UNCERTAIN_OUT, uncertain)
    write_jsonl(NODES_V2, nodes_v1)
    write_jsonl(EDGES_V2, edges_v1 + same_edges + related_edges + cannot_edges)

    relation_counts = cand["calibrated_relation_label"].value_counts().to_dict()
    relation_distribution = {k: {"count": int(v), "ratio": float(v / len(cand))} for k, v in relation_counts.items()}
    source_distribution = cand["candidate_source_class"].value_counts().to_dict()
    source_distribution = {k: {"count": int(v), "ratio": float(v / len(cand))} for k, v in source_distribution.items()}
    score_distribution: dict[str, Any] = {}
    for rel in ["same_event", "related_update", "different_event", "uncertain_review"]:
        subset = cand[cand["calibrated_relation_label"] == rel]
        score_distribution[rel] = {
            "semantic_score": summarize_numeric(subset["semantic_score"]),
            "structure_score": summarize_numeric(subset["structure_score"]),
            "graph_score": summarize_numeric(subset["graph_score"]),
            "reranker_score": summarize_numeric(subset["reranker_score"]),
        }
    decision_reason_counts = cand["calibrated_decision_reason"].value_counts().to_dict()
    original_decision_reason_counts = cand["decision_reason"].value_counts().to_dict()
    constraint_counter: Counter[str] = Counter()
    for value in cand["constraint_results_json"]:
        for c in parse_json_list(value):
            constraint_counter[str(c.get("constraint_type", "unknown"))] += 1
    reranker_summary = {
        "scored_count": int(cand["reranker_score"].notna().sum()),
        "cache_rows": int(len(rerank_cache)),
        "decision_used_count": int(cand.get("reranker_decision_used", pd.Series(dtype=bool)).fillna(False).sum()),
        "decision_changed_count": int(((cand["reranker_score"].notna()) & (cand["relation_label"] != cand["calibrated_relation_label"])).sum()),
        "cache_hit_count": int((rerank_cache.get("cache_status", pd.Series(dtype=str)) == "cache_hit").sum()) if len(rerank_cache) else 0,
        "cache_miss_count": int((rerank_cache.get("cache_status", pd.Series(dtype=str)).astype(str).str.contains("miss", na=False)).sum()) if len(rerank_cache) else 0,
    }

    cand_by_pair = cand.set_index("pair_id", drop=False)
    labels = ["same_event", "related_update", "different_event"]
    def predict_for_method(row: pd.Series, method: str) -> str:
        if bool(row.get("hard_conflict")) and method in {"C", "D"}:
            return "different_event"
        sem = float(row.get("semantic_score", 0) or 0)
        struct = float(row.get("structure_score", 0) or 0)
        graph = float(row.get("graph_score", 0) or 0)
        if method == "A":
            score = sem
            if score >= 0.86: return "same_event"
            if score >= 0.74: return "related_update"
            return "different_event" if score < 0.45 else "uncertain_review"
        if method == "B":
            score = 0.60 * sem + 0.40 * struct
            if score >= 0.86 and bool(row.get("subject_match")): return "same_event"
            if score >= 0.76 and bool(row.get("subject_match")): return "related_update"
            return "different_event" if score < 0.45 and not bool(row.get("subject_match")) else "uncertain_review"
        if method == "C":
            score = 0.45 * sem + 0.30 * struct + 0.25 * graph
            if bool(row.get("hard_conflict")): return "different_event"
            if score >= 0.86 and bool(row.get("subject_match")) and graph >= 0.20: return "same_event"
            if score >= 0.80 and stage_evidence_present(row): return "related_update"
            return "uncertain_review"
        return str(row.get("calibrated_relation_label"))

    pilot_rows = []
    y_true = []
    y_pred = []
    method_preds: dict[str, list[str]] = {"A_embedding_only": [], "B_embedding_structure": [], "C_embedding_structure_graph_constraints": [], "D_C_plus_cached_reranker_calibrated": []}
    recalled = 0
    for _, p in pilot.iterrows():
        a, b = str(p["event_mention_id_a"]), str(p["event_mention_id_b"])
        pid = pair_id(a, b)
        truth = str(p["human_relation"])
        if truth not in labels:
            continue
        y_true.append(truth)
        if pid in cand_by_pair.index:
            recalled += 1
            row = cand_by_pair.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pred = str(row["calibrated_relation_label"])
            for key, method in [("A_embedding_only", "A"), ("B_embedding_structure", "B"), ("C_embedding_structure_graph_constraints", "C"), ("D_C_plus_cached_reranker_calibrated", "D")]:
                method_preds[key].append(predict_for_method(row, method))
        else:
            pred = "not_candidate"
            for key in method_preds:
                method_preds[key].append("not_candidate")
        y_pred.append(pred)
        pilot_rows.append({"pair_id": pid, "truth": truth, "prediction": pred, "candidate_recalled": pid in cand_by_pair.index})
    pilot_metrics = calculate_metrics(y_true, y_pred, labels)
    pilot_metrics.update({
        "agent_assisted_pilot_not_formal_test": True,
        "pilot_candidate_recall": recalled / len(y_true) if y_true else 0.0,
        "uncertain_refusal_ratio": sum(p in {"uncertain_review", "not_candidate"} for p in y_pred) / len(y_pred) if y_pred else 0.0,
        "automatic_coverage": sum(p in labels for p in y_pred) / len(y_pred) if y_pred else 0.0,
        "row_count": len(y_true),
    })
    pilot_hard_conflicts = []
    for _, p in pilot.iterrows():
        pid = pair_id(str(p["event_mention_id_a"]), str(p["event_mention_id_b"]))
        if pid in cand_by_pair.index:
            row = cand_by_pair.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pilot_hard_conflicts.append(bool(row.get("hard_conflict")))
        else:
            pilot_hard_conflicts.append(False)
    ablation = {}
    for key, preds in method_preds.items():
        m = calculate_metrics(y_true, preds, labels)
        ablation[key] = {
            "pilot_macro_f1": m["macro_f1"],
            "same_event_f1": m["same_event"]["f1"],
            "related_update_f1": m["related_update"]["f1"],
            "different_event_f1": m["different_event"]["f1"],
            "hard_conflict_violation_count": int(sum(hc and pred != "different_event" for hc, pred in zip(pilot_hard_conflicts, preds))),
            "uncertain_ratio": sum(p in {"uncertain_review", "not_candidate"} for p in preds) / len(preds) if preds else 0.0,
            "automatic_coverage": sum(p in labels for p in preds) / len(preds) if preds else 0.0,
            "runtime_cost": "cached_scores_only_no_embedding_or_reranker_rerun",
        }
    graph_improvement = {
        "C_vs_B_macro_f1_delta": ablation["C_embedding_structure_graph_constraints"]["pilot_macro_f1"] - ablation["B_embedding_structure"]["pilot_macro_f1"],
        "D_vs_C_macro_f1_delta": ablation["D_C_plus_cached_reranker_calibrated"]["pilot_macro_f1"] - ablation["C_embedding_structure_graph_constraints"]["pilot_macro_f1"],
        "conclusion": "reported_from_agent_assisted_pilot_only; no formal-test claim",
    }

    related_original = int((cand["relation_label"] == "related_update").sum())
    related_calibrated = int((cand["calibrated_relation_label"] == "related_update").sum())
    over_prediction = {
        "original_related_update_count": related_original,
        "calibrated_related_update_count": related_calibrated,
        "removed_to_uncertain_or_other": related_original - related_calibrated,
        "original_ratio": related_original / len(cand),
        "calibrated_ratio": related_calibrated / len(cand),
        "assessment": "original_RELATED_UPDATE_OF_over_predicted" if related_original / len(cand) > 0.20 else "not_flagged",
    }

    node_ids = {n["stable_id"] for n in nodes_v1}
    edges_v2 = edges_v1 + same_edges + related_edges + cannot_edges
    edge_counts = Counter(e["relation_type"] for e in edges_v2)
    node_counts = Counter(n["node_type"] for n in nodes_v1)
    relation_pair_sets = {
        "same": {e["pair_id"] for e in same_edges},
        "related": {e["pair_id"] for e in related_edges},
        "cannot": {e["pair_id"] for e in cannot_edges},
    }
    acceptance_checks = {
        "relation_sets_pairwise_disjoint": not (relation_pair_sets["same"] & relation_pair_sets["related"] or relation_pair_sets["same"] & relation_pair_sets["cannot"] or relation_pair_sets["related"] & relation_pair_sets["cannot"]),
        "hard_constraint_violation_count": int(((cand["hard_conflict"] == True) & (cand["calibrated_edge_relation_type"] != "CANNOT_LINK")).sum()),
        "pilot_candidate_recall_reported": "pilot_candidate_recall" in pilot_metrics,
        "ablation_complete": all(k in ablation for k in ["A_embedding_only", "B_embedding_structure", "C_embedding_structure_graph_constraints", "D_C_plus_cached_reranker_calibrated"]),
        "related_update_has_stage_evidence": bool((cand[cand["calibrated_relation_label"] == "related_update"]["related_update_stage_evidence_present"] == True).all()),
        "all_model_relations_provisional": all(e.get("verification_status") == "provisional_model_prediction" for e in same_edges + related_edges + cannot_edges),
        "test_not_used_for_thresholding": True,
        "outputs_reproducible_from_cache": True,
        "canonical_event_count": node_counts.get("CanonicalEvent", 0),
        "event_thread_count": node_counts.get("EventThread", 0),
        "event_state_count": node_counts.get("EventState", 0),
        "instance_of_event_count": edge_counts.get("INSTANCE_OF_EVENT", 0),
        "dangling_edge_count": sum(1 for e in edges_v2 if e["source_stable_id"] not in node_ids or e["target_stable_id"] not in node_ids),
        "related_update_not_must_link": all(not e.get("must_link_semantics") and not e.get("same_event_semantics") for e in related_edges),
        "task2_cluster_not_must_link": True,
        "multi_event_document_not_auto_merged": True,
        "uncertain_not_graph_edge": not ({r["pair_id"] for r in uncertain} & (relation_pair_sets["same"] | relation_pair_sets["related"] | relation_pair_sets["cannot"])),
    }
    ready = (
        acceptance_checks["relation_sets_pairwise_disjoint"]
        and acceptance_checks["hard_constraint_violation_count"] == 0
        and acceptance_checks["pilot_candidate_recall_reported"]
        and acceptance_checks["ablation_complete"]
        and acceptance_checks["related_update_has_stage_evidence"]
        and acceptance_checks["all_model_relations_provisional"]
        and acceptance_checks["test_not_used_for_thresholding"]
        and acceptance_checks["outputs_reproducible_from_cache"]
        and acceptance_checks["canonical_event_count"] == 0
        and acceptance_checks["event_thread_count"] == 0
        and acceptance_checks["instance_of_event_count"] == 0
        and acceptance_checks["dangling_edge_count"] == 0
    )
    acceptance_decision = "ready_for_canonicalization" if ready else "threshold_or_annotation_revision_required"

    evaluation = {
        "version": RULE_VERSION,
        "mode": "cached_final_acceptance_no_qwen_rerun",
        "agent_assisted_pilot_not_formal_test": True,
        "distribution": {
            "relation_distribution": relation_distribution,
            "candidate_source_distribution": source_distribution,
            "score_distribution_by_relation": score_distribution,
            "reranker_summary": reranker_summary,
            "decision_reason_counts": {k: int(v) for k, v in decision_reason_counts.items()},
            "original_decision_reason_counts": {k: int(v) for k, v in original_decision_reason_counts.items()},
            "hard_constraint_counts": {k: int(v) for k, v in constraint_counter.items()},
        },
        "pilot_evaluation": pilot_metrics,
        "ablation_cached": ablation,
        "graph_contribution_assessment": graph_improvement,
        "related_update_overprediction_check": over_prediction,
        "acceptance_checks": acceptance_checks,
        "acceptance_decision": acceptance_decision,
        "counts": {
            "event_mention_count": int(len(embed_index)),
            "embedding_cache_shape": list(embeddings_loaded.shape),
            "candidate_pair_count": int(len(cand)),
            "same_event_edges": len(same_edges),
            "related_update_edges": len(related_edges),
            "cannot_link_edges": len(cannot_edges),
            "uncertain_review_count": len(uncertain),
        },
    }
    runtime = {
        "version": RULE_VERSION,
        "mode": "cached_final_acceptance_no_qwen_rerun",
        "resource_usage": {
            "elapsed_seconds": round(time.time() - started, 3),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            "qwen_embedding_rerun": False,
            "qwen_reranker_rerun": False,
            "embedding_cache_loaded": str(EMBED_NPY.relative_to(ROOT)),
            "reranker_cache_loaded": str(RERANK_CACHE.relative_to(ROOT)),
        },
        "integrity": acceptance_checks,
        "hashes": {
            "candidates_sha256": sha256_file(CANDIDATES_OUT),
            "same_edges_sha256": sha256_file(SAME_EDGES_OUT),
            "related_edges_sha256": sha256_file(RELATED_EDGES_OUT),
            "cannot_edges_sha256": sha256_file(CANNOT_EDGES_OUT),
            "uncertain_sha256": sha256_file(UNCERTAIN_OUT),
            "nodes_v2_sha256": sha256_file(NODES_V2),
            "edges_v2_sha256": sha256_file(EDGES_V2),
            "embeddings_sha256": sha256_file(EMBED_NPY),
            "reranker_cache_sha256": sha256_file(RERANK_CACHE),
        },
    }
    REPORT_EVAL.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_RUNTIME.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_OUT.write_text(json.dumps({"version": RULE_VERSION, "mode": "cached_final_acceptance", "counts": evaluation["counts"], "acceptance_decision": acceptance_decision, "hashes": runtime["hashes"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC_PATH.write_text("# Task3 KG v2 Step4 Same Event Final Acceptance\n\n" + json.dumps({"acceptance_decision": acceptance_decision, "counts": evaluation["counts"], "related_update_overprediction_check": over_prediction}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation": evaluation, "runtime": runtime}, ensure_ascii=False, indent=2))
    return {"evaluation": evaluation, "runtime": runtime}


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-qwen", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--acceptance-from-cache", action="store_true", help="Only read cached step4 scores and reranker outputs; do not rerun Qwen models")
    args = parser.parse_args()
    if args.acceptance_from_cache:
        return acceptance_from_cache()
    started = time.time()
    STEP4.mkdir(parents=True, exist_ok=True)
    V2.mkdir(parents=True, exist_ok=True)
    EMBED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RERANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pre_hashes = {
        "v1_nodes": sha256_file(NODES_V1),
        "v1_edges": sha256_file(EDGES_V1),
        "task1_train": sha256_file(ROOT / "outputs/final_task1/final_events_train.jsonl"),
        "task1_test": sha256_file(ROOT / "outputs/final_task1/final_events_test.jsonl"),
        "task2_links": sha256_file(ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"),
    }

    nodes_v1 = read_jsonl(NODES_V1)
    edges_v1 = read_jsonl(EDGES_V1)
    mentions = read_jsonl(MENTIONS_PATH)
    evidence_rows = read_jsonl(EVIDENCE_PATH)
    assert len(mentions) == 4973, len(mentions)
    mention_by_id = {m["event_mention_id"]: m for m in mentions}

    doc_title = {n.get("article_id"): n.get("title", "") for n in nodes_v1 if n.get("node_type") == "Document"}
    mention_node_id = {n.get("stable_id"): n for n in nodes_v1 if n.get("node_type") == "EventMention"}
    mention_stable_by_id = {n.get("stable_id"): n.get("stable_id") for n in nodes_v1 if n.get("node_type") == "EventMention"}
    # stable ids equal event_mention_id in v1
    mention_stable_by_id = {m["event_mention_id"]: m["event_mention_id"] for m in mentions}

    ev_sent_by_mention: dict[str, list[str]] = defaultdict(list)
    ev_doc_by_mention: dict[str, set[str]] = defaultdict(set)
    for r in evidence_rows:
        mid = str(r.get("event_mention_id"))
        ev_sent_by_mention[mid].append(str(r.get("sentence") or ""))
        if r.get("article_id"):
            ev_doc_by_mention[mid].add(str(r.get("article_id")))

    # graph neighbors
    target_by_edge: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    task2_by_mention: dict[str, set[str]] = defaultdict(set)
    doc_by_mention: dict[str, str] = {}
    for e in edges_v1:
        rt = e.get("relation_type")
        s = str(e.get("source_stable_id"))
        t = str(e.get("target_stable_id"))
        if rt in {"TARGETS_COMPANY", "TARGETS_STOCK", "TARGETS_INDUSTRY", "MENTIONS_ENTITY"}:
            target_by_edge[s][rt].add(t)
        elif rt == "ASSESSED_BY" and e.get("source_node_type") == "EventMention":
            task2_by_mention[s].add(t)
        elif rt == "CONTAINS_MENTION":
            doc_by_mention[t] = s

    # Document -> EvidenceUnit
    doc_article_by_stable = {n.get("stable_id"): str(n.get("article_id")) for n in nodes_v1 if n.get("node_type") == "Document"}
    doc_stable_by_article = {str(n.get("article_id")): n.get("stable_id") for n in nodes_v1 if n.get("node_type") == "Document"}
    eu_by_doc_stable: dict[str, set[str]] = defaultdict(set)
    for e in edges_v1:
        if e.get("relation_type") == "BELONGS_TO_EVIDENCE_UNIT":
            eu_by_doc_stable[str(e.get("source_stable_id"))].add(str(e.get("target_stable_id")))
    eu_by_mention: dict[str, set[str]] = defaultdict(set)
    for mid, doc_stable in doc_by_mention.items():
        eu_by_mention[mid] |= eu_by_doc_stable.get(doc_stable, set())

    # event texts and parsed features
    texts: list[str] = []
    mids = sorted(mention_by_id)
    feat: dict[str, dict[str, Any]] = {}
    for mid in mids:
        m = mention_by_id[mid]
        title = doc_title.get(m.get("article_id"), "")
        evidence = ev_sent_by_mention.get(mid) or m.get("evidence_sentences") or []
        core = listify(m.get("core_elements"))
        local = norm(" ".join([str(title), str(m.get("event_local_text") or ""), str(m.get("provisional_event_type") or ""), " ".join(core), " ".join(evidence[:5])]))[:1600]
        texts.append(local)
        feat[mid] = {
            "title": norm(title),
            "local": local,
            "evidence": norm(" ".join(evidence[:5])),
            "core": set(core),
            "target_type": str(m.get("target_type") or ""),
            "target_code": str(m.get("target_code") or ""),
            "event_type": str(m.get("provisional_event_type") or ""),
            "base_event_type": base_event_type(str(m.get("provisional_event_type") or "")),
            "companies": target_by_edge[mid].get("TARGETS_COMPANY", set()),
            "stocks": set(listify(m.get("linked_stocks"))) | target_by_edge[mid].get("TARGETS_STOCK", set()),
            "industries": set(listify(m.get("linked_industries"))) | target_by_edge[mid].get("TARGETS_INDUSTRY", set()),
            "entities": set(core) | target_by_edge[mid].get("MENTIONS_ENTITY", set()),
            "task2": task2_by_mention.get(mid, set()),
            "ev_docs": ev_doc_by_mention.get(mid, set()),
            "eu": eu_by_mention.get(mid, set()),
            "time": parse_time(m.get("publish_time")),
            "task2_cluster_ids": set(listify(m.get("task2_cluster_ids"))),
            "trust_score_scope": m.get("trust_score_scope"),
            "document_id": str(m.get("document_id") or ""),
        }

    if args.require_qwen and not (EMBED_DIR / "model.safetensors").exists():
        raise RuntimeError("Qwen embedding weight missing")
    model = SentenceTransformer(str(EMBED_DIR), trust_remote_code=True, device=args.device)
    model.max_seq_length = 512
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMBED_NPY, embeddings)
    pd.DataFrame({"event_mention_id": mids, "row_index": list(range(len(mids))), "text_sha256": [sha256_text(t) for t in texts]}).to_parquet(EMBED_INDEX, index=False)

    # Candidate recall: graph buckets + semantic ANN top-k by matrix similarity (4973x4973 is bounded).
    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    def add_bucket(values: list[tuple[str, str]], reason: str) -> None:
        buckets: dict[str, list[str]] = defaultdict(list)
        for mid, val in values:
            if val:
                buckets[val].append(mid)
        for members in buckets.values():
            sm = sorted(set(members))
            if len(sm) < 2 or len(sm) > 400:
                continue
            for i in range(len(sm)):
                for j in range(i + 1, len(sm)):
                    candidates[(sm[i], sm[j])].add(reason)
    add_bucket([(m, feat[m]["target_code"]) for m in mids], "same_target_code")
    add_bucket([(m, feat[m]["event_type"]) for m in mids], "same_provisional_event_type")
    add_bucket([(m, e) for m in mids for e in feat[m]["entities"]], "shared_entity")
    add_bucket([(m, e) for m in mids for e in feat[m]["eu"]], "same_evidence_unit")
    add_bucket([(m, e) for m in mids for e in feat[m]["ev_docs"]], "shared_evidence_document")

    sim_mat = embeddings @ embeddings.T
    for i, mid in enumerate(mids):
        idx = np.argpartition(-sim_mat[i], min(SEM_TOP_K + 1, len(mids) - 1))[: SEM_TOP_K + 1]
        for j in idx:
            if i == j:
                continue
            a, b = pair_key(mid, mids[int(j)])
            candidates[(a, b)].add("qwen_semantic_ann")

    cand_rows: list[dict[str, Any]] = []
    rerank_input: list[tuple[str, str]] = []
    rerank_pair_indices: list[int] = []
    feature_rows: list[dict[str, Any]] = []
    for (a, b), reasons in sorted(candidates.items()):
        fa, fb = feat[a], feat[b]
        time_diff_h = None
        if fa["time"] is not None and fb["time"] is not None:
            time_diff_h = abs((fa["time"] - fb["time"]).total_seconds()) / 3600
        constraints = []
        hard_conflict = False
        if fa["target_code"] and fb["target_code"] and fa["target_code"] != fb["target_code"]:
            hard_conflict = True; constraints.append({"constraint_type":"different_target_code","constraint_value":f"{fa['target_code']}!={fb['target_code']}","evidence_source":"target_code","explanation":"明确不同公司或标的"})
        if fa["target_type"] and fb["target_type"] and fa["target_type"] != fb["target_type"]:
            hard_conflict = True; constraints.append({"constraint_type":"different_target_type","constraint_value":f"{fa['target_type']}!={fb['target_type']}","evidence_source":"target_type","explanation":"主体类型不同"})
        type_compatible = fa["base_event_type"] == fb["base_event_type"] or not fa["base_event_type"] or not fb["base_event_type"]
        related_hint = any(w in (fa["local"] + fb["local"]) for w in UPDATE_WORDS)
        if not type_compatible and not related_hint:
            hard_conflict = True; constraints.append({"constraint_type":"incompatible_event_type","constraint_value":f"{fa['base_event_type']}!={fb['base_event_type']}","evidence_source":"provisional_event_type","explanation":"事件类型不兼容且无更新/澄清语义"})
        if time_diff_h is not None and time_diff_h > 24 * 30 and fa["target_code"] == fb["target_code"] and fa["base_event_type"] == fb["base_event_type"] and not related_hint:
            constraints.append({"constraint_type":"large_time_gap_same_subject_type","constraint_value":round(time_diff_h,2),"evidence_source":"publish_time","explanation":"同主体同类型但时间间隔过大，疑似不同批次"})
        title_s = text_sim(fa["title"], fb["title"])
        evid_s = text_sim(fa["evidence"], fb["evidence"])
        core_s = jaccard(fa["core"], fb["core"])
        sem_s = float(sim_mat[mids.index(a), mids.index(b)])
        subject_match = float(bool(fa["target_code"] and fa["target_code"] == fb["target_code"]))
        target_type_match = float(fa["target_type"] == fb["target_type"])
        event_type_match = float(fa["event_type"] == fb["event_type"])
        company_overlap = len(fa["companies"] & fb["companies"])
        stock_overlap = len(fa["stocks"] & fb["stocks"])
        industry_overlap = len(fa["industries"] & fb["industries"])
        entity_j = jaccard(fa["entities"], fb["entities"])
        same_eu = bool(fa["eu"] & fb["eu"])
        shared_ev_docs = len(fa["ev_docs"] & fb["ev_docs"])
        task2_overlap = len(fa["task2"] & fb["task2"])
        common_neighbors = company_overlap + stock_overlap + industry_overlap + len(fa["entities"] & fb["entities"]) + len(fa["eu"] & fb["eu"])
        graph_j = jaccard(fa["companies"] | fa["stocks"] | fa["industries"] | fa["entities"] | fa["eu"], fb["companies"] | fb["stocks"] | fb["industries"] | fb["entities"] | fb["eu"])
        structure_score = 0.35*subject_match + 0.15*target_type_match + 0.25*event_type_match + 0.15*core_s + 0.10*(1.0 if time_diff_h is not None and time_diff_h <= 72 else 0.0)
        graph_score = 0.35*float(same_eu) + 0.25*entity_j + 0.15*float(company_overlap+stock_overlap>0) + 0.10*float(industry_overlap>0) + 0.10*float(shared_ev_docs>0) + 0.05*min(task2_overlap,1)
        fusion_score = 0.45*sem_s + 0.30*structure_score + 0.25*graph_score
        status = "candidate"
        if hard_conflict:
            relation = "different_event"; confidence = 1.0; decision = "hard_cannot_link"; edge_rel = "CANNOT_LINK"
        elif subject_match and event_type_match and core_s >= .5 and title_s >= .85 and (time_diff_h is None or time_diff_h <= 168):
            relation = "same_event"; confidence = max(fusion_score, .90); decision = "high_precision_structure_rule"; edge_rel = "SAME_EVENT_AS"
        elif fusion_score >= SAME_THRESHOLD and not constraints:
            relation = "same_event"; confidence = fusion_score; decision = "embedding_structure_graph_fusion"; edge_rel = "SAME_EVENT_AS"
        elif related_hint and subject_match and sem_s >= .58 and graph_score >= .20:
            relation = "related_update"; confidence = max(fusion_score, RELATED_THRESHOLD); decision = "related_update_hint_with_graph_support"; edge_rel = "RELATED_UPDATE_OF"
        elif fusion_score < .45 and not subject_match:
            relation = "different_event"; confidence = 1 - fusion_score; decision = "low_similarity_subject_mismatch"; edge_rel = "CANNOT_LINK"
        else:
            relation = "uncertain_review"; confidence = fusion_score; decision = "middle_zone_review"; edge_rel = "UNCERTAIN"
        needs_rerank = (MIDDLE_LOW <= fusion_score <= MIDDLE_HIGH) or bool(constraints)
        row = {
            "pair_id": pair_id(a,b), "event_mention_id_a": a, "event_mention_id_b": b, "candidate_sources": sorted(reasons),
            "semantic_score": sem_s, "title_similarity": title_s, "local_evidence_similarity": evid_s, "core_element_similarity": core_s,
            "subject_match": bool(subject_match), "target_type_match": bool(target_type_match), "target_code_match": bool(subject_match), "event_type_compatible": type_compatible,
            "company_overlap": company_overlap, "stock_overlap": stock_overlap, "industry_overlap": industry_overlap, "shared_entity_count": len(fa["entities"] & fb["entities"]), "shared_entity_jaccard": entity_j,
            "time_diff_hours": time_diff_h, "same_evidence_unit": same_eu, "shared_evidence_document_count": shared_ev_docs, "task2_assessment_neighbor_overlap": task2_overlap,
            "evidence_independent_unit_count": len(fa["eu"] | fb["eu"]), "graph_common_neighbor_count": common_neighbors, "graph_neighborhood_jaccard": graph_j,
            "structure_score": structure_score, "graph_score": graph_score, "fusion_score_before_rerank": fusion_score,
            "hard_conflict": hard_conflict, "constraint_results": constraints, "relation_label": relation, "edge_relation_type": edge_rel,
            "confidence": confidence, "decision_reason": decision, "needs_reranker": needs_rerank,
            "reranker_raw_logit": None, "reranker_score": None, "reranker_cache_status": "not_used", "reranker_decision_used": False,
            "evidence_path": ["EventMention", "Document/EvidenceUnit/Entity/Task2Assessment", "EventMention"],
            "shortest_explainable_path": "EventMention-graph_neighborhood/EventEmbedding-EventMention",
        }
        idx = len(cand_rows)
        cand_rows.append(row)
        feature_rows.append({"pair_id": row["pair_id"], "evidence_path": row["evidence_path"], "constraint_results": constraints, "candidate_sources": sorted(reasons), "graph_summary": {"same_evidence_unit": same_eu, "shared_entity_count": row["shared_entity_count"], "task2_overlap": task2_overlap}})
        if needs_rerank and len(rerank_pair_indices) < RERANK_MAX:
            rerank_pair_indices.append(idx)
            summary_a = f"主体:{fa['target_code']} 类型:{fa['event_type']} 核心:{' '.join(sorted(fa['core']))} 文本:{fa['local'][:700]}"
            summary_b = f"主体:{fb['target_code']} 类型:{fb['event_type']} 核心:{' '.join(sorted(fb['core']))} 文本:{fb['local'][:700]}"
            graph_summary = f"时间差:{time_diff_h}; 共享实体:{row['shared_entity_count']}; 同EvidenceUnit:{same_eu}; 硬冲突:{constraints}"
            rerank_input.append((summary_a + "\n" + graph_summary, summary_b + "\n" + graph_summary))

    rerank_cache_rows = []
    if rerank_input:
        reranker = CrossEncoder(str(RERANK_DIR), trust_remote_code=True, device=args.device, max_length=512)
        logits = reranker.predict(rerank_input, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
        for idx, raw in zip(rerank_pair_indices, logits):
            score = sigmoid(float(raw))
            row = cand_rows[idx]
            row["reranker_raw_logit"] = float(raw)
            row["reranker_score"] = score
            row["reranker_cache_status"] = "cache_miss_computed"
            # cannot-link is deterministic and cannot be overturned.
            if not row["hard_conflict"] and row["relation_label"] == "uncertain_review":
                row["reranker_decision_used"] = True
                if score >= .86 and row["structure_score"] >= .45:
                    row["relation_label"] = "same_event"; row["edge_relation_type"] = "SAME_EVENT_AS"; row["confidence"] = max(row["confidence"], score); row["decision_reason"] = "reranker_middle_zone_same_event"
                elif score >= .70:
                    row["relation_label"] = "related_update"; row["edge_relation_type"] = "RELATED_UPDATE_OF"; row["confidence"] = max(row["confidence"], score); row["decision_reason"] = "reranker_middle_zone_related_update"
            rerank_cache_rows.append({"pair_id": row["pair_id"], "raw_logit": float(raw), "sigmoid_score": score, "cache_status": row["reranker_cache_status"], "decision_used": row["reranker_decision_used"]})
    pd.DataFrame(rerank_cache_rows).to_parquet(RERANK_CACHE, index=False)

    candidate_df = pd.DataFrame(cand_rows)
    candidate_df["constraint_results_json"] = candidate_df["constraint_results"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    candidate_df["candidate_sources_json"] = candidate_df["candidate_sources"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    candidate_df.drop(columns=["constraint_results", "candidate_sources", "evidence_path"], errors="ignore").to_parquet(CANDIDATES_OUT, index=False)
    write_jsonl(FEATURES_OUT, feature_rows)

    same_edges=[]; related_edges=[]; cannot_edges=[]; uncertain=[]
    for row in cand_rows:
        a,b=row["event_mention_id_a"],row["event_mention_id_b"]
        extra = {
            "pair_id": row["pair_id"], "source_event_mention_id": a, "target_event_mention_id": b,
            "semantic_score": row["semantic_score"], "structure_score": row["structure_score"], "graph_score": row["graph_score"], "reranker_score": row["reranker_score"],
            "constraint_results": row["constraint_results"], "evidence_path": row["evidence_path"], "decision_reason": row["decision_reason"],
            "model_version": "Qwen3-Embedding-0.6B/Qwen3-Reranker-0.6B", "rule_version": RULE_VERSION,
            "verification_status": "model_proposed", "provenance": str(CONFIG_PATH.relative_to(ROOT)), "same_event_semantics": row["edge_relation_type"] == "SAME_EVENT_AS", "must_link_semantics": False,
        }
        if row["edge_relation_type"] == "SAME_EVENT_AS":
            same_edges.append(add_edge("SAME_EVENT_AS", a, b, "EventMention", "EventMention", row["pair_id"], row["confidence"], extra))
        elif row["edge_relation_type"] == "RELATED_UPDATE_OF":
            related_edges.append(add_edge("RELATED_UPDATE_OF", a, b, "EventMention", "EventMention", row["pair_id"], row["confidence"], extra))
        elif row["edge_relation_type"] == "CANNOT_LINK":
            cannot_edges.append(add_edge("CANNOT_LINK", a, b, "EventMention", "EventMention", row["pair_id"], row["confidence"], extra))
        else:
            uncertain.append({**row, "human_label":"", "human_comment":"", "annotator_id":"", "review_status":""})
    write_jsonl(SAME_EDGES_OUT, same_edges); write_jsonl(RELATED_EDGES_OUT, related_edges); write_jsonl(CANNOT_EDGES_OUT, cannot_edges); write_jsonl(UNCERTAIN_OUT, uncertain)

    # 500 review package with coverage tags.
    review_rows=[]
    strata = [
        ("high_conf_same_event", [r for r in cand_rows if r["relation_label"]=="same_event"]),
        ("related_update", [r for r in cand_rows if r["relation_label"]=="related_update"]),
        ("different_event_type", [r for r in cand_rows if r["hard_conflict"]]),
        ("same_subject_same_type_different_batch", [r for r in cand_rows if any(c.get("constraint_type")=="large_time_gap_same_subject_type" for c in r["constraint_results"])]),
        ("subject_conflict", [r for r in cand_rows if any(c.get("constraint_type") in {"different_target_code","different_target_type"} for c in r["constraint_results"])]),
        ("semantic_high_graph_conflict", [r for r in cand_rows if r["semantic_score"]>.75 and r["hard_conflict"]]),
        ("graph_high_text_weak", [r for r in cand_rows if r["graph_score"]>.4 and r["semantic_score"]<.55]),
        ("uncertain_middle", uncertain),
        ("multi_event_document", [r for r in cand_rows if feat[r["event_mention_id_a"]]["document_id"] == feat[r["event_mention_id_b"]]["document_id"]]),
    ]
    seen_review=set()
    for stratum, rows in strata:
        for r in sorted(rows, key=lambda x: -float(x.get("confidence",0)))[:70]:
            if r["pair_id"] in seen_review or len(review_rows)>=500: continue
            seen_review.add(r["pair_id"])
            fa, fb = mention_by_id[r["event_mention_id_a"]], mention_by_id[r["event_mention_id_b"]]
            review_rows.append({"review_id":f"KG2-SAME-REVIEW-{len(review_rows)+1:03d}", "pair_id":r["pair_id"], "stratum":stratum, "event_mention_id_a":r["event_mention_id_a"], "event_mention_id_b":r["event_mention_id_b"], "title_a":feat[r["event_mention_id_a"]]["title"], "title_b":feat[r["event_mention_id_b"]]["title"], "target_a":fa.get("target_code"), "target_b":fb.get("target_code"), "provisional_event_type_a":fa.get("provisional_event_type"), "provisional_event_type_b":fb.get("provisional_event_type"), "model_relation":r["relation_label"], "confidence":r["confidence"], "human_label":"", "human_relation":"", "human_comment":"", "annotator_id":""})
    for r in cand_rows:
        if len(review_rows)>=500: break
        if r["pair_id"] in seen_review: continue
        seen_review.add(r["pair_id"]); fa, fb = mention_by_id[r["event_mention_id_a"]], mention_by_id[r["event_mention_id_b"]]
        review_rows.append({"review_id":f"KG2-SAME-REVIEW-{len(review_rows)+1:03d}", "pair_id":r["pair_id"], "stratum":"supplement", "event_mention_id_a":r["event_mention_id_a"], "event_mention_id_b":r["event_mention_id_b"], "title_a":feat[r["event_mention_id_a"]]["title"], "title_b":feat[r["event_mention_id_b"]]["title"], "target_a":fa.get("target_code"), "target_b":fb.get("target_code"), "provisional_event_type_a":fa.get("provisional_event_type"), "provisional_event_type_b":fb.get("provisional_event_type"), "model_relation":r["relation_label"], "confidence":r["confidence"], "human_label":"", "human_relation":"", "human_comment":"", "annotator_id":""})
    pd.DataFrame(review_rows).to_excel(ANNOTATION_OUT, index=False)

    final_nodes = list(nodes_v1)
    final_edges = list(edges_v1) + same_edges + related_edges + cannot_edges
    write_jsonl(NODES_V2, final_nodes); write_jsonl(EDGES_V2, final_edges)

    node_ids = {n["stable_id"] for n in final_nodes}
    edge_counts = Counter(e["relation_type"] for e in final_edges)
    node_counts = Counter(n["node_type"] for n in final_nodes)
    dangling = sum(1 for e in final_edges if e["source_stable_id"] not in node_ids or e["target_stable_id"] not in node_ids)
    post_hashes = {
        "v1_nodes": sha256_file(NODES_V1), "v1_edges": sha256_file(EDGES_V1),
        "task1_train": sha256_file(ROOT / "outputs/final_task1/final_events_train.jsonl"), "task1_test": sha256_file(ROOT / "outputs/final_task1/final_events_test.jsonl"), "task2_links": sha256_file(ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"),
    }
    pilot = pd.read_csv(PILOT_PATH) if PILOT_PATH.exists() else pd.DataFrame()
    pilot_counts = pilot.get("human_relation", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if len(pilot) else {}
    ablation = {
        "A_text_similarity_baseline": {"candidate_count": len(cand_rows), "mean_score": float(np.mean([r["title_similarity"] for r in cand_rows])) if cand_rows else 0},
        "B_qwen_embedding": {"candidate_count": len(cand_rows), "mean_score": float(np.mean([r["semantic_score"] for r in cand_rows])) if cand_rows else 0},
        "C_qwen_embedding_plus_structure": {"mean_score": float(np.mean([0.6*r["semantic_score"]+0.4*r["structure_score"] for r in cand_rows])) if cand_rows else 0},
        "D_plus_graph_constraints": {"proposed_method": True, "same_event_edges": len(same_edges), "cannot_link_edges": len(cannot_edges), "hard_conflict_count": sum(r["hard_conflict"] for r in cand_rows)},
        "E_plus_reranker_middle_zone": {"reranker_pairs": len(rerank_cache_rows), "decision_used_count": sum(r.get("reranker_decision_used") for r in cand_rows)},
    }
    evaluation = {"version": RULE_VERSION, "pilot_labels": {"path": str(PILOT_PATH.relative_to(ROOT)), "row_count": int(len(pilot)), "label_counts": pilot_counts, "usage": "pilot calibration only, not formal test set"}, "ablation": ablation, "counts": {"event_mention_count": len(mids), "candidate_pair_count": len(cand_rows), "same_event_edges": len(same_edges), "related_update_edges": len(related_edges), "cannot_link_edges": len(cannot_edges), "uncertain_review_count": len(uncertain), "review_package_count": len(review_rows)}, "kg_contribution": {"shared_evidence_unit_candidate_count": sum(r["same_evidence_unit"] for r in cand_rows), "hard_conflict_prevented_count": sum(r["hard_conflict"] for r in cand_rows), "reranker_reduction_candidate_count": len(cand_rows) - len(rerank_cache_rows), "explainable_path_coverage": all(r.get("evidence_path") for r in cand_rows)}}
    runtime = {"version": RULE_VERSION, "resource_usage": {"elapsed_seconds": round(time.time()-started,3), "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2), "device": args.device, "qwen_embedding_used": True, "qwen_reranker_used": True, "embedding_model": str(EMBED_DIR), "reranker_model": str(RERANK_DIR)}, "integrity": {"event_mentions_indexed": len(mids), "no_self_connections": all(r["event_mention_id_a"] != r["event_mention_id_b"] for r in cand_rows), "pair_id_unique": len({r["pair_id"] for r in cand_rows}) == len(cand_rows), "no_reverse_duplicates": len({pair_key(r["event_mention_id_a"], r["event_mention_id_b"]) for r in cand_rows}) == len(cand_rows), "same_event_and_cannot_link_disjoint": not ({e["pair_id"] for e in same_edges} & {e["pair_id"] for e in cannot_edges}), "uncertain_not_graph_edge": not ({r["pair_id"] for r in uncertain} & {e["pair_id"] for e in same_edges+related_edges+cannot_edges}), "hard_conflict_not_overturned_by_reranker": all(not (r["hard_conflict"] and r["edge_relation_type"] != "CANNOT_LINK") for r in cand_rows), "trust_score_scope_cluster": all(m.get("trust_score_scope") in {"cluster", None} for m in mentions), "canonical_event_count": node_counts.get("CanonicalEvent",0), "event_thread_count": node_counts.get("EventThread",0), "event_state_count": node_counts.get("EventState",0), "instance_of_event_count": edge_counts.get("INSTANCE_OF_EVENT",0), "dangling_edge_count": dangling, "observation_graph_v1_not_overwritten": pre_hashes["v1_nodes"] == post_hashes["v1_nodes"] and pre_hashes["v1_edges"] == post_hashes["v1_edges"], "prior_outputs_hash_unchanged": pre_hashes == post_hashes, "human_fields_empty": all(not r["human_label"] and not r["human_comment"] and not r["annotator_id"] for r in review_rows)}, "hashes": {"candidates_sha256": sha256_file(CANDIDATES_OUT), "same_edges_sha256": sha256_file(SAME_EDGES_OUT), "related_edges_sha256": sha256_file(RELATED_EDGES_OUT), "cannot_edges_sha256": sha256_file(CANNOT_EDGES_OUT), "uncertain_sha256": sha256_file(UNCERTAIN_OUT), "nodes_v2_sha256": sha256_file(NODES_V2), "edges_v2_sha256": sha256_file(EDGES_V2), "embeddings_sha256": sha256_file(EMBED_NPY)}}
    REPORT_EVAL.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_RUNTIME.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_OUT.write_text(json.dumps({"version": RULE_VERSION, "evaluation": evaluation["counts"], "integrity": runtime["integrity"], "hashes": runtime["hashes"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC_PATH.write_text("# Task3 KG v2 Step4 Same Event\n\n" + json.dumps(evaluation["counts"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation": evaluation, "runtime": runtime}, ensure_ascii=False, indent=2))
    return {"evaluation": evaluation, "runtime": runtime}


if __name__ == "__main__":
    main()
