#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import build_task3_resolver_v3_candidate as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/task3_resolver_v3_candidate"
REPAIR_PAIRS = ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/repair_dev_500.parquet"
RECOVERED_EVENTS = ROOT / "outputs/task3_canonical_event_v2_actual_repair/task1_615_grounding_pass_recovered.jsonl"
V2_MODEL = ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_model.joblib"
V2_SELECTION = ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_selection.json"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pair_label_lookup(candidate_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    for row in candidate_df.itertuples(index=False):
        label = base.text(getattr(row, "gold_relation_if_labeled"))
        if not label:
            continue
        key = base.pair_key(getattr(row, "sample_id_a"), getattr(row, "sample_id_b"))
        old = labels.get(key)
        if old and old != label:
            labels[key] = "conflict"
        else:
            labels[key] = label
    return labels


def channel_sets(candidate_df: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    out: dict[str, set[tuple[str, str]]] = {r: set() for r in base.R_ORDER}
    for row in candidate_df.itertuples(index=False):
        key = base.pair_key(getattr(row, "sample_id_a"), getattr(row, "sample_id_b"))
        for r in base.R_ORDER:
            if bool(getattr(row, r)):
                out[r].add(key)
    return out


def channel_overlap_report(candidate_df: pd.DataFrame) -> dict[str, Any]:
    sets = channel_sets(candidate_df)
    independent = {r: len(sets[r]) for r in base.R_ORDER}
    combination_counts: Counter[str] = Counter()
    for row in candidate_df.itertuples(index=False):
        active = [r for r in base.R_ORDER if bool(getattr(row, r))]
        combination_counts["+".join(active) if active else "none"] += 1
    pairwise = {}
    for a in base.R_ORDER:
        for b in base.R_ORDER:
            pairwise[f"{a}&{b}"] = len(sets[a] & sets[b])
    seen: set[tuple[str, str]] = set()
    incremental = {}
    for r in ["R1", "R2", "R4", "R3"]:
        new_pairs = sets[r] - seen
        incremental[r] = {"new_candidate_pairs": len(new_pairs)}
        seen |= sets[r]
    return {
        "candidate_pairs": int(len(candidate_df)),
        "independent_channel_counts": independent,
        "channel_combination_counts": dict(sorted(combination_counts.items())),
        "pairwise_intersections": pairwise,
        "incremental_order_R1_R2_R4_R3": incremental,
    }


def candidate_count_stats(candidate_pairs: set[tuple[str, str]], event_ids: list[str]) -> dict[str, Any]:
    counts = Counter()
    for a, b in candidate_pairs:
        counts[a] += 1
        counts[b] += 1
    arr = np.array([counts[e] for e in event_ids], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
    return {
        "mean": round(float(arr.mean()), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": int(arr.max()),
        "zero_candidate_events": int((arr == 0).sum()),
    }


def light_prerank_score(row: Any) -> float:
    score = 0.0
    if bool(row.R1):
        score += 4.0
    if bool(row.R2):
        score += 2.5
    if bool(row.R4):
        score += 1.5
    if bool(row.R3):
        score += 1.2
    score += 2.0 * float(row.fact_similarity)
    score += 1.2 * float(row.object_similarity)
    score += 0.8 * float(row.trigger_similarity)
    score += 0.8 * float(row.evidence_similarity)
    if bool(row.same_event_fact_signature):
        score += 2.0
    if bool(row.same_document_hash):
        score += 1.0
    if base.text(row.time_relation) in {"same_time", "close_time", "close_publish_time"}:
        score += 0.3
    if base.text(row.hard_conflicts):
        score -= 3.0
    return round(score, 6)


def resolver_feature_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    f = base.identity_features(left, right)
    support_a = base.evidence_meta(left)["support_type"] or base.MISSING
    support_b = base.evidence_meta(right)["support_type"] or base.MISSING
    polarity_a = base.event_meta(left)["polarity_or_state"] or base.MISSING
    polarity_b = base.event_meta(right)["polarity_or_state"] or base.MISSING
    sig_a = base.event_meta(left)["event_fact_signature"] or base.MISSING
    sig_b = base.event_meta(right)["event_fact_signature"] or base.MISSING
    return {
        "same_target": int(bool(f["same_target"])),
        "same_target_type": int(bool(f["same_target_type"])),
        "same_target_code": int(bool(f["same_target_code"])),
        "same_family": int(bool(f["same_family"])),
        "same_event_type": int(bool(f["same_event_type"])),
        "same_event_name": int(bool(f["same_event_name"])),
        "same_trigger": int(bool(f["same_trigger"])),
        "same_object": int(bool(f["same_object"])),
        "same_modality": int(bool(f["same_modality"])),
        "same_article": int(bool(f["same_article"])),
        "same_source": int(bool(f["same_source"])),
        "same_sentence": int(bool(f["same_sentence"])),
        "same_document_hash": int(bool(f["same_document_hash"])),
        "same_event_fact_signature": int(bool(f["same_event_fact_signature"])),
        "same_support_type": int(support_a != base.MISSING and support_a == support_b),
        "same_polarity_or_state": int(polarity_a != base.MISSING and polarity_a == polarity_b),
        "title_similarity": float(f["title_similarity"]),
        "fact_similarity": float(f["fact_similarity"]),
        "object_similarity": float(f["object_similarity"]),
        "trigger_similarity": float(f["trigger_similarity"]),
        "evidence_similarity": float(f["evidence_similarity"]),
        "fact_token_overlap_count": len(f["fact_token_overlap"]),
        "object_token_overlap_count": len(f["object_token_overlap"]),
        "trigger_token_overlap_count": len(f["trigger_token_overlap"]),
        "hard_conflict_count": len(base.hard_conflicts(f)),
        "time_relation": base.text(f["time_relation"]),
        "modality_relation": base.text(f["modality_relation"]),
        "provenance_relation": base.text(f["provenance_relation"]),
        "support_relation": "same" if support_a != base.MISSING and support_a == support_b else "different" if support_a != base.MISSING and support_b != base.MISSING else "missing",
        "polarity_relation": "same" if polarity_a != base.MISSING and polarity_a == polarity_b else "different" if polarity_a != base.MISSING and polarity_b != base.MISSING else "missing",
        "support_type_a": support_a,
        "support_type_b": support_b,
        "polarity_or_state_a": polarity_a,
        "polarity_or_state_b": polarity_b,
        "event_fact_signature_a": sig_a,
        "event_fact_signature_b": sig_b,
    }


def threshold_predict(prob_row: np.ndarray, classes: list[str], same_thr: float, diff_thr: float, margin: float) -> tuple[str, float, dict[str, float]]:
    idx = {c: i for i, c in enumerate(classes)}
    p_same = float(prob_row[idx["same_event"]])
    p_diff = float(prob_row[idx["different_event"]])
    p_unc = float(prob_row[idx["uncertain"]])
    if p_same >= same_thr and (p_same - max(p_diff, p_unc)) >= margin:
        return "same_event", p_same, {"same_event": p_same, "different_event": p_diff, "uncertain": p_unc}
    if p_diff >= diff_thr and (p_diff - max(p_same, p_unc)) >= margin:
        return "different_event", p_diff, {"same_event": p_same, "different_event": p_diff, "uncertain": p_unc}
    return "uncertain", max(p_same, p_diff, p_unc), {"same_event": p_same, "different_event": p_diff, "uncertain": p_unc}


def evaluate_binary_same(gold_same_total: int, gold_diff_total: int, candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for r in candidate_rows if r["gold_relation"] == "same_event" and r["resolver_prediction"] == "same_event")
    fp = sum(1 for r in candidate_rows if r["gold_relation"] == "different_event" and r["resolver_prediction"] == "same_event")
    candidate_same_gold = sum(1 for r in candidate_rows if r["gold_relation"] == "same_event")
    fn = gold_same_total - tp
    tn = gold_diff_total - fp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, gold_same_total)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "gold_same_event_pairs": int(gold_same_total),
        "gold_different_event_pairs": int(gold_diff_total),
        "candidate_gold_same_event_pairs": int(candidate_same_gold),
        "tp": int(tp),
        "fp_false_merge": int(fp),
        "fn_split_error": int(fn),
        "tn_assuming_non_candidates_different": int(tn),
        "same_event_precision": round(float(precision), 6),
        "same_event_recall": round(float(recall), 6),
        "pairwise_f1": round(float(f1), 6),
    }


def run_quality_pool(events: dict[str, dict[str, Any]], candidate_df: pd.DataFrame, diagnosis: dict[str, Any]) -> dict[str, Any]:
    selection = json.loads(V2_SELECTION.read_text(encoding="utf-8"))
    thresholds = selection.get("thresholds", {"same_thr": 0.45, "diff_thr": 0.45, "margin": 0.0})
    model = joblib.load(V2_MODEL)
    classes = list(model.named_steps["clf"].classes_)
    label_map = pair_label_lookup(candidate_df)
    candidate_pairs = {base.pair_key(r.sample_id_a, r.sample_id_b) for r in candidate_df.itertuples(index=False)}
    sets = channel_sets(candidate_df)
    pred_rows: list[dict[str, Any]] = []
    for row in candidate_df.itertuples(index=False):
        key = base.pair_key(row.sample_id_a, row.sample_id_b)
        left = events[key[0]]
        right = events[key[1]]
        feats = resolver_feature_dict(left, right)
        proba = model.predict_proba([feats])[0]
        pred, conf, trace = threshold_predict(proba, classes, float(thresholds["same_thr"]), float(thresholds["diff_thr"]), float(thresholds.get("margin", 0.0)))
        pred_rows.append(
            {
                "sample_id_a": key[0],
                "sample_id_b": key[1],
                "R1": bool(row.R1),
                "R2": bool(row.R2),
                "R3": bool(row.R3),
                "R4": bool(row.R4),
                "light_prerank_score": light_prerank_score(row),
                "resolver_prediction": pred,
                "resolver_confidence": round(float(conf), 6),
                "prob_same_event": round(float(trace["same_event"]), 6),
                "prob_different_event": round(float(trace["different_event"]), 6),
                "prob_uncertain": round(float(trace["uncertain"]), 6),
                "gold_relation": label_map.get(key, ""),
                "hard_conflicts": base.text(row.hard_conflicts),
                "fact_similarity": float(row.fact_similarity),
                "object_similarity": float(row.object_similarity),
                "trigger_similarity": float(row.trigger_similarity),
                "evidence_similarity": float(row.evidence_similarity),
            }
        )
    pred_rows.sort(key=lambda r: (-r["light_prerank_score"], -r["prob_same_event"], r["sample_id_a"], r["sample_id_b"]))
    write_parquet(OUT / "quality_pool_prerank_resolver_predictions.parquet", pred_rows)
    write_csv(OUT / "quality_pool_prerank_resolver_predictions.csv", pred_rows)

    recalled_same = {base.pair_key(r["sample_id_a"], r["sample_id_b"]) for r in pred_rows if r["gold_relation"] == "same_event"}
    prefix_seen: set[tuple[str, str]] = set()
    prefix_size = 0
    for idx, row in enumerate(pred_rows, start=1):
        key = base.pair_key(row["sample_id_a"], row["sample_id_b"])
        if key in recalled_same:
            prefix_seen.add(key)
        if prefix_seen == recalled_same:
            prefix_size = idx
            break
    gold_same_total = int(diagnosis["r_channel_independent_metrics"]["R1_R2_R3_R4_union"]["human_same_event_denominator"])
    gold_diff_total = 22
    labeled_rows = [r for r in pred_rows if r["gold_relation"] in {"same_event", "different_event"}]
    metrics = evaluate_binary_same(gold_same_total, gold_diff_total, labeled_rows)
    metrics.update(
        {
            "candidate_pairs": int(len(candidate_pairs)),
            "candidate_recalled_same_event_pairs": int(len(recalled_same)),
            "oracle_min_candidate_set_to_recall_79_same_pairs": int(len(recalled_same)),
            "light_prerank_prefix_size_to_cover_recalled_same_pairs": int(prefix_size),
            "resolver_prediction_counts_on_323": dict(Counter(r["resolver_prediction"] for r in pred_rows)),
            "resolver_same_predictions_unlabeled": int(sum(1 for r in pred_rows if not r["gold_relation"] and r["resolver_prediction"] == "same_event")),
            "uncertain_predictions_on_323": int(sum(1 for r in pred_rows if r["resolver_prediction"] == "uncertain")),
            "b3_status": "unavailable_pair_labels_only_no_complete_cluster_member_gold",
        }
    )
    channel_report = channel_overlap_report(candidate_df)
    event_ids = sorted({x for key in candidate_pairs for x in key})
    count_stats = candidate_count_stats(candidate_pairs, event_ids)
    contribution = {
        "independent": diagnosis["r_channel_independent_metrics"],
        "incremental": diagnosis["incremental_experiments"],
        "incremental_new_counts": diagnosis["r_channel_incremental_new_counts"],
    }
    write_json(OUT / "quality_pool_resolver_metrics.json", metrics)
    write_json(OUT / "r_channel_contribution.json", contribution)
    write_json(OUT / "candidate_channel_overlap.json", channel_report)
    write_json(OUT / "quality_pool_candidate_count_stats.json", count_stats)
    return {
        "metrics": metrics,
        "channel_report": channel_report,
        "candidate_count_stats": count_stats,
        "contribution": contribution,
    }


def missed_reason(left: dict[str, Any], right: dict[str, Any], r3_pairs: set[tuple[str, str]]) -> dict[str, Any]:
    f = base.identity_features(left, right)
    conflicts = base.hard_conflicts(f)
    r1 = base.r1_reasons(left, right, f)
    r2 = base.r2_reasons(left, right, f, conflicts)
    r4 = base.r4_reasons(left, right, f, conflicts)
    reasons = []
    if not f["same_target"]:
        reasons.append("target_key_not_compatible")
    if not f["same_family"]:
        reasons.append("event_family_mismatch_or_missing")
    if not f["same_event_fact_signature"]:
        reasons.append("fact_signature_not_equal")
    if f["object_similarity"] < 0.35:
        reasons.append("argument_object_similarity_low")
    if f["trigger_similarity"] < 0.35:
        reasons.append("trigger_similarity_low")
    if f["fact_similarity"] < 0.60:
        reasons.append("fact_similarity_below_R2_fact_threshold")
    if f["evidence_similarity"] < 0.65:
        reasons.append("evidence_similarity_below_R2_evidence_threshold")
    if f["time_relation"] == "time_conflict":
        reasons.append("time_conflict")
    if conflicts:
        reasons.append("hard_conflict:" + "|".join(conflicts))
    key = base.pair_key(base.sample_id(left), base.sample_id(right))
    if key not in r3_pairs:
        reasons.append("R3_not_in_qwen_faiss_dynamic_topk")
    return {
        "same_target": bool(f["same_target"]),
        "same_family": bool(f["same_family"]),
        "same_event_fact_signature": bool(f["same_event_fact_signature"]),
        "same_document_hash": bool(f["same_document_hash"]),
        "same_sentence": bool(f["same_sentence"]),
        "fact_similarity": round(float(f["fact_similarity"]), 6),
        "object_similarity": round(float(f["object_similarity"]), 6),
        "trigger_similarity": round(float(f["trigger_similarity"]), 6),
        "evidence_similarity": round(float(f["evidence_similarity"]), 6),
        "time_relation": f["time_relation"],
        "hard_conflicts": "|".join(conflicts),
        "r1_reasons_if_any": "|".join(r1),
        "r2_reasons_if_any": "|".join(r2),
        "r4_reasons_if_any": "|".join(r4),
        "missing_reason": "|".join(reasons) if reasons else "not_selected_by_any_channel",
    }


def analyze_missed_same_events(events: dict[str, dict[str, Any]], diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    detail = pd.read_csv(OUT / "dev_same_event_candidate_recall_detail.csv")
    missed = detail[detail["hit_any"] == False].copy()  # noqa: E712
    r3_report = json.loads((OUT / "r3_ann_dev_output.json").read_text(encoding="utf-8"))
    r3_pairs = {base.pair_key(row["a"], row["b"]) for row in r3_report.get("pairs", [])}
    rows = []
    for row in missed.itertuples(index=False):
        a = base.text(row.sample_id_a)
        b = base.text(row.sample_id_b)
        left = events[a]
        right = events[b]
        lm = base.event_meta(left)
        rm = base.event_meta(right)
        reason = missed_reason(left, right, r3_pairs)
        rows.append(
            {
                "sample_id_a": a,
                "sample_id_b": b,
                "target_a": base.target_meta(left)["target_key"],
                "target_b": base.target_meta(right)["target_key"],
                "family_a": lm["event_family"],
                "family_b": rm["event_family"],
                "trigger_a": lm["trigger_text"],
                "trigger_b": rm["trigger_text"],
                "object_a": lm["object_text"],
                "object_b": rm["object_text"],
                "event_time_a": lm["event_time"],
                "event_time_b": rm["event_time"],
                "event_fact_a": lm["event_fact"][:240],
                "event_fact_b": rm["event_fact"][:240],
                **reason,
            }
        )
    write_csv(OUT / "missed_same_event_analysis.csv", rows)
    write_json(OUT / "missed_same_event_analysis.json", {"missed_same_event_count": len(rows), "rows": rows})
    return rows


def load_efficiency_pool(event_split_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    final_event_ids = set(event_split_df.loc[event_split_df["split"] == "final_holdout", "event_uid"].astype(str))
    final_articles = set(event_split_df.loc[event_split_df["split"] == "final_holdout", "article_id"].astype(str))
    rows = read_jsonl(RECOVERED_EVENTS)
    pool: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw = json.dumps(row, ensure_ascii=False)
        if "official1799" in raw.lower():
            continue
        sid = base.sample_id(row)
        if not sid or sid in final_event_ids:
            continue
        if base.text(row.get("article_id")) in final_articles:
            continue
        if sid in pool:
            continue
        pool[sid] = row
    return pool


def build_candidate_sets_for_events(events: dict[str, dict[str, Any]], r3_pairs: set[tuple[str, str]]) -> tuple[dict[str, set[tuple[str, str]]], list[dict[str, Any]]]:
    ids = sorted(events)
    sets = {r: set() for r in base.R_ORDER}
    rows: list[dict[str, Any]] = []
    for a, b in combinations(ids, 2):
        left, right = events[a], events[b]
        f = base.identity_features(left, right)
        conflicts = base.hard_conflicts(f)
        reasons = {
            "R1": base.r1_reasons(left, right, f),
            "R2": base.r2_reasons(left, right, f, conflicts),
            "R4": base.r4_reasons(left, right, f, conflicts),
            "R3": ["qwen_faiss_dynamic_topk"] if base.pair_key(a, b) in r3_pairs else [],
        }
        if not any(reasons.values()):
            continue
        key = base.pair_key(a, b)
        for r in base.R_ORDER:
            if reasons[r]:
                sets[r].add(key)
        rows.append(
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
                "fact_similarity": round(float(f["fact_similarity"]), 6),
                "object_similarity": round(float(f["object_similarity"]), 6),
                "trigger_similarity": round(float(f["trigger_similarity"]), 6),
                "evidence_similarity": round(float(f["evidence_similarity"]), 6),
            }
        )
    return sets, rows


def run_efficiency_pool(event_split_df: pd.DataFrame) -> dict[str, Any]:
    started = time.perf_counter()
    pool = load_efficiency_pool(event_split_df)
    pool_dir = OUT / "production_like_efficiency_pool"
    r3_pairs, r3_report = base.run_r3_ann(pool, pool_dir, None)
    sets, rows = build_candidate_sets_for_events(pool, r3_pairs)
    candidate_pairs = set().union(*(sets[r] for r in base.R_ORDER))
    all_possible = len(pool) * (len(pool) - 1) // 2
    write_parquet(pool_dir / "efficiency_pool_candidate_pairs.parquet", rows)
    write_csv(pool_dir / "efficiency_pool_candidate_pairs.csv", rows)
    event_stats = candidate_count_stats(candidate_pairs, sorted(pool))
    independent = {
        r: {
            "candidate_pairs": int(len(sets[r])),
            "candidate_ratio": round(len(sets[r]) / max(1, all_possible), 6),
            "candidate_reduction": round(1.0 - len(sets[r]) / max(1, all_possible), 6),
        }
        for r in base.R_ORDER
    }
    union_metrics = {
        "candidate_pairs": int(len(candidate_pairs)),
        "all_possible_pairs": int(all_possible),
        "candidate_ratio": round(len(candidate_pairs) / max(1, all_possible), 6),
        "candidate_reduction": round(1.0 - len(candidate_pairs) / max(1, all_possible), 6),
        "passes_production_reduction_0_99": bool(1.0 - len(candidate_pairs) / max(1, all_possible) >= 0.99),
    }
    report = {
        "status": "production_like_efficiency_pool_complete",
        "pool_definition": "task1_615_grounding_pass_recovered excluding frozen final_holdout event/article ids and official1799 references; labels not read or used",
        "candidate_policy": {
            "R1_R2_R4": "structured rule channels unchanged",
            "R3": "qwen_faiss_dynamic_topk with semantic_score floor",
            "r3_semantic_score_floor": base.R3_SEMANTIC_SCORE_FLOOR,
        },
        "input_events": int(len(pool)),
        "all_possible_pairs": int(all_possible),
        "r3_report": {k: v for k, v in r3_report.items() if k != "pairs"},
        "independent_channel_efficiency": independent,
        "union_efficiency": union_metrics,
        "candidate_count_per_event": event_stats,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(pool_dir / "production_like_efficiency_report.json", report)
    return report


def main() -> None:
    started = time.perf_counter()
    pair_df = pd.read_parquet(
        REPAIR_PAIRS,
        columns=["sample_id_a", "sample_id_b", "article_id_a", "article_id_b", "left_event", "right_event"],
    )
    events_all = base.build_events(pair_df)
    event_split = pd.read_parquet(OUT / "event_split_mapping.parquet")
    dev_ids = set(event_split.loc[event_split["split"] == "dev", "event_uid"].astype(str))
    events = {sid: ev for sid, ev in events_all.items() if sid in dev_ids}
    candidate_df = pd.read_parquet(OUT / "r_channel_candidate_pairs_dev.parquet")
    diagnosis = json.loads((OUT / "candidate_recall_diagnosis.json").read_text(encoding="utf-8"))

    quality = run_quality_pool(events, candidate_df, diagnosis)
    missed = analyze_missed_same_events(events, diagnosis)
    efficiency = run_efficiency_pool(event_split)

    summary = {
        "status": "quality_and_efficiency_analysis_complete",
        "input_size": {
            "quality_dev_events": int(len(events)),
            "quality_candidate_pairs": int(len(candidate_df)),
            "production_like_events": efficiency["input_events"],
        },
        "positive_negative_counts": {
            "quality_dev_unique_same_event": int(diagnosis["dev_human_same_event_pair_denominator"]),
            "quality_dev_unique_different_event": 22,
            "production_like_labels_used": 0,
        },
        "leakage_check": {
            "official1799_read": 0,
            "final_holdout_evaluated": 0,
            "production_pool_final_holdout_events_excluded": True,
        },
        "candidate_count": {
            "quality_pool": int(len(candidate_df)),
            "production_like_pool": efficiency["union_efficiency"]["candidate_pairs"],
        },
        "candidate_recall": diagnosis["r_channel_independent_metrics"]["R1_R2_R3_R4_union"]["candidate_recall"],
        "candidate_reduction": {
            "quality_pool_not_efficiency_gate": diagnosis["r_channel_independent_metrics"]["R1_R2_R3_R4_union"]["candidate_reduction"],
            "production_like_pool": efficiency["union_efficiency"]["candidate_reduction"],
        },
        "candidate_policy": {
            "quality_pool": "frozen 323 repair-dev candidates; R3 floor does not remove any dev candidate",
            "production_like_pool": f"R1/R2/R4 plus R3 semantic_score >= {base.R3_SEMANTIC_SCORE_FLOOR}",
        },
        "pairwise_metrics": quality["metrics"],
        "split_false_merge": {
            "split_error": quality["metrics"]["fn_split_error"],
            "false_merge": quality["metrics"]["fp_false_merge"],
        },
        "current_blocker": "" if efficiency["union_efficiency"]["passes_production_reduction_0_99"] else "production_like_efficiency_reduction_below_0_99",
        "artifact_path": rel(OUT),
        "missed_same_event_count": len(missed),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(OUT / "quality_efficiency_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
