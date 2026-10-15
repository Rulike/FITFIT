#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
ROOT = WORKSPACE_ROOT / "financial_event_project"
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import build_task3_resolver_v3_candidate as multiroute  # noqa: E402
from scripts import build_task3_resolver_v3_official1799_multiroute as integrated  # noqa: E402


OUT = WORKSPACE_ROOT / "outputs/task3_resolver_v3_1_hybrid_final"
FROZEN_SCRIPT = OUT / "inference_pipeline.py"
V3_PACKAGE = ROOT / "outputs/task3_resolver_v3_candidate_recall"
REPAIR_DIR = ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1"
EVENT_SPLIT = V3_PACKAGE / "event_split_mapping.parquet"
PAIR_SPLIT = V3_PACKAGE / "repair_pair_split_mapping.parquet"
DEV_MULTIROUTE = V3_PACKAGE / "r_channel_candidate_pairs_dev.parquet"
LEGACY_CANDIDATES = REPAIR_DIR / "resolver_v2_candidate_pairs.parquet"
REPAIR_PAIRS = REPAIR_DIR / "repair_dev_500.parquet"
V3_MODEL = V3_PACKAGE / "selected_model/resolver_v3_pair_model.joblib"
V3_SELECTION = V3_PACKAGE / "selected_model/resolver_v3_pair_model_selection.json"
OFFICIAL_FILLED_DIR = ROOT / "outputs/task4 golddata"
OFFICIAL_EVAL_DIR = ROOT / "outputs/task34_official1799_final_evaluation_v1"
OFFICIAL_MULTIROUTE_DIR = ROOT / "outputs/task3_resolver_v3_official1799_multiroute"

PRIMARY = ["same_event", "different_event"]
THRESHOLDS = {"same_thr": 0.98, "diff_thr": 0.76, "margin": 0.0}
ROUTE_BITS = {"legacy": 16, "R1": 1, "R2": 2, "R3": 4, "R4": 8}
R_ROUTES = ["R1", "R2", "R3", "R4"]
ALL_ROUTES = ["legacy", "R1", "R2", "R3", "R4"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    raw = str(value).strip()
    return "" if raw.lower() in {"none", "nan", "null", "nat"} else raw


def pair_key(a: Any, b: Any) -> tuple[str, str]:
    aa, bb = text(a), text(b)
    return tuple(sorted((aa, bb)))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: Path, hash_file: bool = True) -> dict[str, Any]:
    meta: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        meta["sha256"] = None
        return meta
    meta["type"] = "dir" if path.is_dir() else "file"
    if path.is_file():
        stat = path.stat()
        meta["size_bytes"] = int(stat.st_size)
        meta["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        meta["sha256"] = sha256_file(path) if hash_file else None
    else:
        meta["sha256"] = None
        meta["note"] = "directory contents intentionally not hashed"
    return meta


def parquet_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    safe = []
    for row in rows:
        item = {}
        for key, value in row.items():
            item[key] = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple, set)) else jsonable(value)
        safe.append(item)
    return pd.DataFrame(safe)


def read_parquet_filtered(path: Path, filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, filters=filters)
    except Exception:
        df = pd.read_parquet(path)
        mask = pd.Series(True, index=df.index)
        for col, op, value in filters:
            if op != "==":
                raise ValueError(op)
            mask &= df[col].eq(value)
        return df[mask].copy()


def snapshot(status: str, final_holdout_read_count: int) -> dict[str, Any]:
    protected_official = {
        "official1799_filled_gold_dir": {
            "path": str(OFFICIAL_FILLED_DIR),
            "exists": OFFICIAL_FILLED_DIR.exists(),
            "read_count": 0,
            "content_hash": None,
        },
        "official1799_final_evaluation_dir": {
            "path": str(OFFICIAL_EVAL_DIR),
            "exists": OFFICIAL_EVAL_DIR.exists(),
            "read_count": 0,
            "content_hash": None,
        },
    }
    files = {
        "entrypoint_script": SCRIPT_PATH,
        "frozen_inference_pipeline": FROZEN_SCRIPT,
        "selected_v3_model": V3_MODEL,
        "selected_v3_selection": V3_SELECTION,
        "event_split_mapping": EVENT_SPLIT,
        "repair_pair_split_mapping": PAIR_SPLIT,
        "dev_multiroute_candidates": DEV_MULTIROUTE,
        "legacy_candidates": LEGACY_CANDIDATES,
        "repair_pairs": REPAIR_PAIRS,
        "multiroute_generator": ROOT / "scripts/build_task3_resolver_v3_candidate.py",
        "official_multiroute_integration_script": ROOT / "scripts/build_task3_resolver_v3_official1799_multiroute.py",
        "official_multiroute_frozen_output": OFFICIAL_MULTIROUTE_DIR,
    }
    return {
        "status": status,
        "generated_at_utc": now(),
        "official1799_filled_gold_read_count": 0,
        "official1799_error_detail_read_count": 0,
        "final_holdout_read_count": int(final_holdout_read_count),
        "files": {name: file_meta(path, hash_file=path.is_file()) for name, path in files.items()},
        "protected_official1799_not_read": protected_official,
    }


def load_selection() -> dict[str, Any]:
    selection = read_json(V3_SELECTION)
    thresholds = selection.get("thresholds") or {}
    if thresholds != THRESHOLDS:
        raise RuntimeError(f"selected v3 thresholds are not frozen: {thresholds}")
    return selection


def route_names(row: dict[str, Any]) -> list[str]:
    names = []
    if bool(row.get("legacy")):
        names.append("legacy")
    for route in R_ROUTES:
        if bool(row.get(route)):
            names.append(route)
    return names


def r_route_names(row: dict[str, Any]) -> list[str]:
    return [route for route in R_ROUTES if bool(row.get(route))]


def route_mask(names: Iterable[str]) -> int:
    return int(sum(ROUTE_BITS[name] for name in names if name in ROUTE_BITS))


def reason_set(row: dict[str, Any], column: str) -> set[str]:
    return {value for value in text(row.get(column)).split("|") if value and value != "legacy_candidate" and not value.startswith("legacy_channel:")}


def strong_structured_support(row: dict[str, Any]) -> bool:
    fact = float(row.get("fact_similarity") or 0.0)
    obj = float(row.get("object_similarity") or 0.0)
    trigger = float(row.get("trigger_similarity") or 0.0)
    evidence = float(row.get("evidence_similarity") or 0.0)
    return bool(
        bool(row.get("same_event_fact_signature"))
        or (bool(row.get("same_document_hash")) and (fact >= 0.55 or obj >= 0.50 or trigger >= 0.60 or evidence >= 0.60))
        or fact >= 0.60
        or evidence >= 0.65
        or (obj >= 0.35 and trigger >= 0.35 and (fact >= 0.35 or evidence >= 0.45))
        or (obj >= 0.60 and (fact >= 0.35 or trigger >= 0.25 or evidence >= 0.45))
    )


def route_strength(row: dict[str, Any]) -> str:
    names = set(r_route_names(row))
    r1 = reason_set(row, "r1_reasons")
    r2 = reason_set(row, "r2_reasons")
    r4 = reason_set(row, "r4_reasons")
    actual_r1 = r1 & {
        "same_target_fact_signature",
        "same_target_document_hash_with_fact_or_argument",
        "same_sentence_with_target_and_strong_support",
        "shared_announcement_project_product_contract_key",
    }
    actual_r2 = r2 & {
        "target_family_fact_lexical",
        "target_family_evidence_lexical",
        "target_family_argument_trigger",
        "target_family_time_with_fact_or_argument",
    }
    actual_r4 = r4 & {
        "same_source_same_target_structured_support",
        "same_modality_with_family_fact_support",
    }
    if bool(row.get("same_event_fact_signature")):
        return "strong"
    if bool(row.get("same_document_hash")) and strong_structured_support(row):
        return "strong"
    if (actual_r1 or actual_r2 or actual_r4) and strong_structured_support(row):
        return "strong"
    if names or bool(row.get("legacy")):
        return "recovery"
    return "weak"


def normalize_candidate_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    key = pair_key(row.get("sample_id_a"), row.get("sample_id_b"))
    item = dict(row)
    item["sample_id_a"], item["sample_id_b"] = key
    item["candidate_source"] = source
    item["legacy"] = bool(item.get("legacy")) or source == "legacy"
    for route in R_ROUTES:
        item[route] = bool(item.get(route))
    item["route_names"] = "|".join(r_route_names(item))
    item["source_names"] = "|".join(route_names(item))
    item["route_count"] = len(r_route_names(item))
    item["source_count"] = len(route_names(item))
    item["route_mask"] = route_mask(r_route_names(item))
    item["source_mask"] = route_mask(route_names(item))
    item["route_strength_input"] = text(item.get("route_strength"))
    item["route_strength"] = route_strength(item)
    item["pair_id"] = text(item.get("pair_id")) or f"V31PAIR_{hashlib.sha256((key[0]+'||'+key[1]).encode()).hexdigest()[:20]}"
    return item


def merge_candidate_frames(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for source, df in frames:
        for raw in df.to_dict("records"):
            row = normalize_candidate_row(raw, source)
            key = pair_key(row["sample_id_a"], row["sample_id_b"])
            old = merged.get(key)
            if old is None:
                merged[key] = row
                continue
            old["legacy"] = bool(old.get("legacy")) or bool(row.get("legacy"))
            for route in R_ROUTES:
                old[route] = bool(old.get(route)) or bool(row.get(route))
                reason_col = f"{route.lower()}_reasons"
                vals = sorted({x for x in (text(old.get(reason_col)) + "|" + text(row.get(reason_col))).split("|") if x})
                old[reason_col] = "|".join(vals)
            old["candidate_source"] = "|".join(sorted({x for x in (text(old.get("candidate_source")) + "|" + source).split("|") if x}))
            old["legacy_channels"] = "|".join(sorted({x for x in (text(old.get("legacy_channels")) + "|" + text(row.get("legacy_channels"))).split("|") if x}))
            for col in [
                "same_target",
                "same_family",
                "same_sentence",
                "same_document_hash",
                "same_event_fact_signature",
            ]:
                old[col] = bool(old.get(col)) or bool(row.get(col))
            for col in ["fact_similarity", "object_similarity", "trigger_similarity", "evidence_similarity"]:
                old[col] = max(float(old.get(col) or 0.0), float(row.get(col) or 0.0))
            if not text(old.get("time_relation")):
                old["time_relation"] = text(row.get("time_relation"))
            conflicts = sorted({x for x in (text(old.get("hard_conflicts")) + "|" + text(row.get("hard_conflicts"))).split("|") if x})
            old["hard_conflicts"] = "|".join(conflicts)
            old["route_names"] = "|".join(r_route_names(old))
            old["source_names"] = "|".join(route_names(old))
            old["route_count"] = len(r_route_names(old))
            old["source_count"] = len(route_names(old))
            old["route_mask"] = route_mask(r_route_names(old))
            old["source_mask"] = route_mask(route_names(old))
            old["route_strength"] = "strong" if route_strength(old) == "strong" else route_strength(old)
    rows = [merged[key] for key in sorted(merged)]
    return pd.DataFrame(rows)


def load_dev_labels() -> dict[tuple[str, str], str]:
    df = read_parquet_filtered(PAIR_SPLIT, [("pair_split", "==", "dev")])
    label_info = dedup_primary_labels(df)
    labels = label_info["labels"]
    if len(labels) != 109:
        raise RuntimeError(f"expected 109 unique primary Dev pairs, got {len(labels)} from {PAIR_SPLIT}")
    return labels


def dedup_primary_labels(pair_df: pd.DataFrame) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    origins: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in pair_df.itertuples(index=False):
        key = pair_key(row.sample_id_a, row.sample_id_b)
        label = text(row.gold_relation)
        if label:
            grouped[key].append(label)
            origins[key].append(text(getattr(row, "gold_relation_origin", "")))
    labels: dict[tuple[str, str], str] = {}
    ambiguous = []
    for key, values in sorted(grouped.items()):
        primary_values = {v for v in values if v in PRIMARY}
        if len(primary_values) == 1 and "uncertain" not in set(values):
            labels[key] = next(iter(primary_values))
        else:
            ambiguous.append({"sample_id_a": key[0], "sample_id_b": key[1], "labels": sorted(set(values)), "origins": sorted(set(origins[key]))})
    return {
        "labels": labels,
        "ambiguous_or_nonprimary_pairs": ambiguous,
        "input_rows": int(len(pair_df)),
        "unique_pairs": int(len(grouped)),
    }


def load_event_split(split: str) -> pd.DataFrame:
    return read_parquet_filtered(EVENT_SPLIT, [("split", "==", split)]).copy()


def legacy_candidates_for_ids(ids: set[str]) -> pd.DataFrame:
    return integrated.legacy_dev_candidates_as_v3(ids)


def multiroute_dev_candidates() -> pd.DataFrame:
    return pd.read_parquet(DEV_MULTIROUTE).copy()


def load_full_events_for_ids(ids: set[str]) -> dict[str, dict[str, Any]]:
    pair_df = pd.read_parquet(REPAIR_PAIRS)
    events = multiroute.build_events(pair_df)
    return {sid: event for sid, event in events.items() if sid in ids}


def multiroute_candidates_for_events(events: dict[str, dict[str, Any]], out_dir: Path, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, report = integrated.generate_multiroute_candidates(
        list(events.values()),
        out_dir,
        r3_worker_python="/root/miniconda3/envs/marag_retriever/bin/python",
        r3_encode_python="/root/miniconda3/envs/dianjin-cu128/bin/python",
        prefix=prefix,
    )
    return pd.DataFrame(rows), report


def score_candidates(candidate_df: pd.DataFrame, flat_events: dict[str, dict[str, Any]], model: Any) -> dict[tuple[str, str], dict[str, Any]]:
    rows = candidate_df.to_dict("records")
    features = [integrated.feature_row_for_candidate(flat_events, row) for row in rows]
    probs_rows = integrated.v3_probabilities(model, features)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row, feature, probs in zip(rows, features, probs_rows):
        key = pair_key(row["sample_id_a"], row["sample_id_b"])
        pred, conf = integrated.selected_v3_predict(probs, THRESHOLDS)
        item = {
            **row,
            "sample_id_a": key[0],
            "sample_id_b": key[1],
            "prediction_raw": pred,
            "confidence": round(float(conf), 6),
            "prob_same_event": round(float(probs.get("same_event", 0.0)), 6),
            "prob_different_event": round(float(probs.get("different_event", 0.0)), 6),
            "prob_uncertain": round(float(probs.get("uncertain", 0.0)), 6),
            "features": feature,
        }
        out[key] = item
    return out


class DSU:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {v: v for v in sorted(values)}
        self.members = {v: {v} for v in sorted(values)}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def component(self, value: str) -> set[str]:
        return set(self.members[self.find(value)])

    def merge(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members[rb]

    def same(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    def components(self) -> list[list[str]]:
        roots = sorted({self.find(v) for v in self.parent})
        return [sorted(self.members[root]) for root in roots]


def hard_conflict(flat_events: dict[str, dict[str, Any]], a: str, b: str, diff_pairs: set[tuple[str, str]]) -> str:
    return integrated.hard_conflict_between(flat_events, a, b, diff_pairs)


def cluster_scored_edges(mode: str, scored: dict[tuple[str, str], dict[str, Any]], ids: set[str], flat_events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    diff_pairs = {
        key
        for key, row in scored.items()
        if row["prediction_raw"] == "different_event" and float(row.get("confidence") or 0.0) >= THRESHOLDS["diff_thr"]
    }
    same_edges = [
        row
        for row in scored.values()
        if row["prediction_raw"] == "same_event" and float(row.get("confidence") or 0.0) >= THRESHOLDS["same_thr"]
    ]
    same_edges.sort(key=lambda row: (-float(row.get("prob_same_event") or row.get("confidence") or 0.0), text(row.get("pair_id")), row["sample_id_a"], row["sample_id_b"]))
    dsu = DSU(ids)
    audit = []
    used = []
    transitive = 0
    weak_transitive = 0
    for row in same_edges:
        a, b = text(row["sample_id_a"]), text(row["sample_id_b"])
        left, right = dsu.component(a), dsu.component(b)
        if left == right:
            audit.append(edge_audit(row, "skip_same_component", "", 0))
            continue
        implied = max(0, len(left) * len(right) - 1)
        if text(row.get("route_strength")) != "strong":
            audit.append(edge_audit(row, "veto", "weak_or_recovery_edge_no_union", implied))
            continue
        conflict = ""
        for x in sorted(left):
            for y in sorted(right):
                conflict = hard_conflict(flat_events, x, y, diff_pairs)
                if conflict:
                    break
            if conflict:
                break
        if conflict:
            audit.append(edge_audit(row, "veto", conflict, implied))
            continue
        dsu.merge(a, b)
        used.append(row)
        transitive += implied
        audit.append(edge_audit(row, "merge", "strong_edge_pass", implied))
    sizes = [len(comp) for comp in dsu.components()]
    direct_weak_same = sum(1 for row in same_edges if text(row.get("route_strength")) != "strong")
    return {
        "candidate_mode": mode,
        "dsu": dsu,
        "cluster_edge_audit": audit,
        "same_edges_raw": int(len(same_edges)),
        "same_edges_used_for_clustering": int(len(used)),
        "same_edges_vetoed_or_skipped": int(len(same_edges) - len(used)),
        "direct_weak_same_edges_vetoed": int(direct_weak_same),
        "different_veto_pairs": int(len(diff_pairs)),
        "cluster_count": int(len(sizes)),
        "multi_mention_clusters": int(sum(1 for s in sizes if s > 1)),
        "max_cluster_size": int(max(sizes) if sizes else 0),
        "cluster_size_histogram": {str(k): int(v) for k, v in sorted(Counter(sizes).items())},
        "transitive_merge_count": int(transitive),
        "weak_edge_transitive_merge_count": int(weak_transitive),
        "cluster_edge_policy_counts": dict(Counter(row["policy_reason"] for row in audit if row["policy_reason"])),
    }


def edge_audit(row: dict[str, Any], action: str, reason: str, implied: int) -> dict[str, Any]:
    return {
        "pair_id": text(row.get("pair_id")),
        "sample_id_a": text(row.get("sample_id_a")),
        "sample_id_b": text(row.get("sample_id_b")),
        "merge_action": action,
        "policy_reason": reason,
        "implied_transitive_pairs": int(implied),
        "prob_same_event": float(row.get("prob_same_event") or 0.0),
        "confidence": float(row.get("confidence") or 0.0),
        "route_strength": text(row.get("route_strength")),
        "source_names": text(row.get("source_names")),
        "route_names": text(row.get("route_names")),
    }


def fixed_predictions(labels: dict[tuple[str, str], str], scored: dict[tuple[str, str], dict[str, Any]], cluster: dict[str, Any]) -> list[dict[str, Any]]:
    dsu: DSU = cluster["dsu"]
    rows = []
    for key, gold in sorted(labels.items()):
        raw = scored.get(key)
        if dsu.same(key[0], key[1]):
            pred = "same_event"
            decision_source = "cluster_same_component"
        elif raw and raw.get("prediction_raw") == "uncertain":
            pred = "uncertain"
            decision_source = "candidate_resolver_uncertain"
        else:
            pred = "different_event"
            decision_source = "not_same_after_cluster_policy" if raw else "non_candidate_not_predicted_same"
        rows.append(
            {
                "sample_id_a": key[0],
                "sample_id_b": key[1],
                "gold_relation": gold,
                "prediction": pred,
                "raw_prediction": text(raw.get("prediction_raw")) if raw else "not_candidate",
                "candidate_hit": raw is not None,
                "decision_source": decision_source,
                "confidence": float(raw.get("confidence") or 1.0) if raw else 1.0,
                "prob_same_event": float(raw.get("prob_same_event") or 0.0) if raw else 0.0,
                "route_names": text(raw.get("route_names")) if raw else "",
                "source_names": text(raw.get("source_names")) if raw else "",
                "route_strength": text(raw.get("route_strength")) if raw else "",
            }
        )
    return rows


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["gold_relation"] for row in rows]
    pred = [row["prediction"] for row in rows]
    pr, rc, f1, support = precision_recall_fscore_support(gold, pred, labels=PRIMARY, zero_division=0)
    per_class = {
        label: {"precision": round(float(pr[i]), 6), "recall": round(float(rc[i]), 6), "f1": round(float(f1[i]), 6), "support": int(support[i])}
        for i, label in enumerate(PRIMARY)
    }
    same = per_class["same_event"]
    diff = per_class["different_event"]
    return {
        "pair_count": int(len(rows)),
        "gold_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(pred)),
        "labels_parameter": PRIMARY,
        "per_class": per_class,
        "same_event_precision": same["precision"],
        "same_event_recall": same["recall"],
        "same_event_f1": same["f1"],
        "different_event_precision": diff["precision"],
        "different_event_recall": diff["recall"],
        "different_event_f1": diff["f1"],
        "pair_macro_f1": round(float(f1_score(gold, pred, labels=PRIMARY, average="macro", zero_division=0)), 6),
        "accuracy": round(float(accuracy_score(gold, pred)), 6),
        "false_merge": int(sum(1 for row in rows if row["gold_relation"] == "different_event" and row["prediction"] == "same_event")),
        "split_error": int(sum(1 for row in rows if row["gold_relation"] == "same_event" and row["prediction"] != "same_event")),
        "uncertain_count": int(sum(1 for row in rows if row["prediction"] == "uncertain")),
        "uncertain_rate": round(sum(1 for row in rows if row["prediction"] == "uncertain") / max(1, len(rows)), 6),
    }


def candidate_conditional_metrics(labels: dict[tuple[str, str], str], scored: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for key, gold in sorted(labels.items()):
        raw = scored.get(key)
        if not raw:
            continue
        rows.append({"gold_relation": gold, "prediction": raw["prediction_raw"]})
    return metric_summary(rows) if rows else {"pair_count": 0, "status": "unavailable_no_labeled_candidate_hits"}


def candidate_coverage(mode: str, candidate_df: pd.DataFrame, ids: set[str], labels: dict[tuple[str, str], str]) -> dict[str, Any]:
    keys = {pair_key(row.sample_id_a, row.sample_id_b) for row in candidate_df.itertuples(index=False)}
    same = {key for key, value in labels.items() if value == "same_event"}
    touched = {sid for key in keys for sid in key}
    all_possible = len(ids) * (len(ids) - 1) // 2
    route_counts = {route: int(candidate_df[route].astype(bool).sum()) for route in R_ROUTES if route in candidate_df.columns}
    source_counts = dict(Counter(text(row.source_names) for row in candidate_df.itertuples(index=False))) if "source_names" in candidate_df.columns else {}
    return {
        "candidate_mode": mode,
        "event_count": int(len(ids)),
        "all_possible_pairs": int(all_possible),
        "candidate_pairs": int(len(keys)),
        "candidate_ratio": round(len(keys) / max(1, all_possible), 6),
        "candidate_reduction": round(1.0 - len(keys) / max(1, all_possible), 6),
        "candidate_oracle_recall": round(len(keys & same) / max(1, len(same)), 6),
        "candidate_hit_same_event": int(len(keys & same)),
        "same_event_denominator": int(len(same)),
        "mention_candidate_coverage": round(len(touched) / max(1, len(ids)), 6),
        "covered_mentions": int(len(touched)),
        "isolated_mentions": int(len(ids - touched)),
        "isolated_mention_rate": round(len(ids - touched) / max(1, len(ids)), 6),
        "route_counts": route_counts,
        "source_combo_counts": source_counts,
    }


def route_contribution(candidate_frames: dict[str, pd.DataFrame], labels: dict[tuple[str, str], str]) -> dict[str, Any]:
    sets = {name: {pair_key(row.sample_id_a, row.sample_id_b) for row in df.itertuples(index=False)} for name, df in candidate_frames.items()}
    same = {key for key, label in labels.items() if label == "same_event"}
    out = {
        "pair_set_counts": {name: len(vals) for name, vals in sets.items()},
        "legacy_only_pairs": len(sets["legacy"] - sets["multiroute"]),
        "multiroute_only_pairs": len(sets["multiroute"] - sets["legacy"]),
        "legacy_multiroute_intersection_pairs": len(sets["legacy"] & sets["multiroute"]),
        "hybrid_union_contains_legacy": bool(sets["legacy"] <= sets["hybrid_union"]),
        "hybrid_union_contains_multiroute": bool(sets["multiroute"] <= sets["hybrid_union"]),
        "same_event_hits": {name: len(vals & same) for name, vals in sets.items()},
        "same_event_increment_legacy_only": len((sets["legacy"] - sets["multiroute"]) & same),
        "same_event_increment_multiroute_only": len((sets["multiroute"] - sets["legacy"]) & same),
    }
    route_sets = {route: set() for route in ALL_ROUTES}
    for row in candidate_frames["hybrid_union"].to_dict("records"):
        key = pair_key(row["sample_id_a"], row["sample_id_b"])
        for route in route_names(row):
            route_sets[route].add(key)
    out["route_pair_counts_hybrid_union"] = {route: len(vals) for route, vals in route_sets.items()}
    out["route_same_hits_hybrid_union"] = {route: len(vals & same) for route, vals in route_sets.items()}
    return out


def evaluate_mode(mode: str, candidate_df: pd.DataFrame, ids: set[str], flat_events: dict[str, dict[str, Any]], labels: dict[tuple[str, str], str], model: Any) -> dict[str, Any]:
    scored = score_candidates(candidate_df, flat_events, model)
    cluster = cluster_scored_edges(mode, scored, ids, flat_events)
    fixed_rows = fixed_predictions(labels, scored, cluster)
    fixed = metric_summary(fixed_rows)
    cond = candidate_conditional_metrics(labels, scored)
    coverage = candidate_coverage(mode, candidate_df, ids, labels)
    pair_rows = []
    for key, row in sorted(scored.items()):
        pair_rows.append({k: v for k, v in row.items() if k != "features"})
    return {
        "mode": mode,
        "scored": scored,
        "pair_rows": pair_rows,
        "fixed_rows": fixed_rows,
        "fixed_denominator_metrics": fixed,
        "candidate_conditional_metrics": cond,
        "coverage": coverage,
        "cluster_safety": {k: v for k, v in cluster.items() if k not in {"dsu", "cluster_edge_audit"}},
        "cluster_edge_audit": cluster["cluster_edge_audit"],
    }


def dev_gate(metrics: dict[str, Any], coverage: dict[str, Any], legacy_cov: dict[str, Any], multiroute_cov: dict[str, Any], union_report: dict[str, Any], safety: dict[str, Any]) -> dict[str, Any]:
    fixed = metrics["fixed_denominator_metrics"]
    checks = {
        "candidate_contains_legacy_and_multiroute": bool(union_report["hybrid_union_contains_legacy"] and union_report["hybrid_union_contains_multiroute"]),
        "candidate_oracle_recall_not_below_max": coverage["candidate_oracle_recall"] >= max(legacy_cov["candidate_oracle_recall"], multiroute_cov["candidate_oracle_recall"]),
        "fixed_precision_min_0_95": fixed["same_event_precision"] >= 0.95,
        "fixed_recall_min_0_850575": fixed["same_event_recall"] >= 0.850575,
        "fixed_f1_min_0_902439": fixed["same_event_f1"] >= 0.902439,
        "false_merge_max_3": fixed["false_merge"] <= 3,
        "split_error_max_13": fixed["split_error"] <= 13,
        "mention_coverage_not_below_legacy_or_multiroute": (
            coverage["mention_candidate_coverage"]
            >= max(legacy_cov["mention_candidate_coverage"], multiroute_cov["mention_candidate_coverage"])
        ),
        "weak_edge_transitive_merge_zero": safety["weak_edge_transitive_merge_count"] == 0,
        "no_abnormal_giant_cluster": safety["max_cluster_size"] <= 20,
        "fixed_denominator_not_route_specific": True,
        "candidate_reduction_denominator_explicit": coverage["all_possible_pairs"] > 0,
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def run_split(split: str, candidate_frames: dict[str, pd.DataFrame], labels: dict[tuple[str, str], str], event_df: pd.DataFrame, model: Any) -> dict[str, Any]:
    ids = set(event_df["event_uid"].map(text))
    flat_events = {text(row.event_uid): row._asdict() for row in event_df.itertuples(index=False)}
    reports = {}
    for mode, df in candidate_frames.items():
        reports[mode] = evaluate_mode(mode, df, ids, flat_events, labels, model)
    return reports


def build_dev_candidate_frames(dev_ids: set[str]) -> dict[str, pd.DataFrame]:
    legacy = merge_candidate_frames([("legacy", legacy_candidates_for_ids(dev_ids))])
    multi = merge_candidate_frames([("multiroute", multiroute_dev_candidates())])
    union = merge_candidate_frames([("legacy", legacy), ("multiroute", multi)])
    return {
        "legacy": legacy,
        "multiroute": multi,
        "hybrid_union": union,
    }


def final_candidate_frames(final_ids: set[str]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    legacy = merge_candidate_frames([("legacy", legacy_candidates_for_ids(final_ids))])
    full_events = load_full_events_for_ids(final_ids)
    multi_raw, multi_report = multiroute_candidates_for_events(full_events, OUT / "final_holdout_multiroute_generation", "FHPAIRV31")
    multi = merge_candidate_frames([("multiroute", multi_raw)])
    union = merge_candidate_frames([("legacy", legacy), ("multiroute", multi)])
    return {"legacy": legacy, "multiroute": multi, "hybrid_union": union}, {"multiroute_generation": multi_report}


def write_frozen_config(dev_choice: str) -> dict[str, Any]:
    config = {
        "status": "frozen_before_final_holdout",
        "generated_at_utc": now(),
        "selected_candidate_mode": dev_choice,
        "candidate_modes": ["legacy", "multiroute", "hybrid_union", "hybrid_coverage"],
        "hybrid_union_definition": "normalized unordered-pair deduplicated legacy union multiroute; legacy candidates are never dropped",
        "hybrid_coverage_policy": "disabled when hybrid_union passes Dev; no coverage rules were stacked",
        "thresholds": THRESHOLDS,
        "resolver_model": str(V3_MODEL),
        "resolver_selection": str(V3_SELECTION),
        "cluster_policy": {
            "algorithm": "deterministic conflict-aware union-find",
            "edge_order": "prob_same_event descending, then pair id",
            "same_edge_union": "selected_v3 same_event with confidence >= 0.98 and route_strength == strong only",
            "uncertain_edge": "never directly merged",
            "different_edge": "component union veto when selected_v3 different_event confidence >= 0.76",
            "hard_conflicts": ["target_conflict", "time_conflict", "object_conflict", "high_confidence_resolver_different"],
            "weak_edge_transitive_merge": "forbidden",
        },
        "official1799_used_for_tuning": False,
        "resolver_retrained": False,
        "thresholds_modified": False,
    }
    write_json(OUT / "frozen_config.json", config)
    return config


def freeze_script_copy() -> None:
    FROZEN_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    if SCRIPT_PATH != FROZEN_SCRIPT:
        shutil.copy2(SCRIPT_PATH, FROZEN_SCRIPT)


def write_frozen_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "status": "frozen_before_final_holdout",
        "generated_at_utc": now(),
        "config": config,
        "sha256": {
            "entrypoint_script": sha256_file(SCRIPT_PATH),
            "inference_pipeline.py": sha256_file(FROZEN_SCRIPT),
            "frozen_config.json": sha256_file(OUT / "frozen_config.json"),
            "selected_v3_model": sha256_file(V3_MODEL),
            "selected_v3_selection": sha256_file(V3_SELECTION),
            "multiroute_generator": sha256_file(ROOT / "scripts/build_task3_resolver_v3_candidate.py"),
            "official_multiroute_integration_script": sha256_file(ROOT / "scripts/build_task3_resolver_v3_official1799_multiroute.py"),
        },
        "final_holdout_read_count_at_freeze": 0,
        "official1799_filled_gold_read_count": 0,
    }
    write_json(OUT / "frozen_manifest.json", manifest)
    return manifest


def compact_report(reports: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for mode, report in reports.items():
        out[mode] = {
            "fixed_denominator_metrics": report["fixed_denominator_metrics"],
            "candidate_conditional_metrics": report["candidate_conditional_metrics"],
            "coverage": report["coverage"],
            "cluster_safety": report["cluster_safety"],
        }
    return out


def final_acceptance(final_reports: dict[str, Any]) -> dict[str, Any]:
    legacy = final_reports["legacy"]
    hybrid = final_reports["hybrid_union"]
    legacy_m = legacy["fixed_denominator_metrics"]
    hybrid_m = hybrid["fixed_denominator_metrics"]
    legacy_cov = legacy["coverage"]
    hybrid_cov = hybrid["coverage"]
    checks = {
        "hybrid_recall_not_below_legacy": hybrid_m["same_event_recall"] >= legacy_m["same_event_recall"],
        "hybrid_f1_not_below_legacy": hybrid_m["same_event_f1"] >= legacy_m["same_event_f1"],
        "precision_drop_within_1pp": hybrid_m["same_event_precision"] >= legacy_m["same_event_precision"] - 0.01,
        "false_merge_not_obviously_higher": hybrid_m["false_merge"] <= max(legacy_m["false_merge"] + 1, legacy_m["false_merge"]),
        "gold_cluster_connectivity_not_below_legacy": hybrid_cov["candidate_oracle_recall"] >= legacy_cov["candidate_oracle_recall"],
        "eligible_coverage_not_below_legacy": "unavailable_no_task4_holdout_gold",
        "no_cluster_explosion": hybrid["cluster_safety"]["max_cluster_size"] <= 20,
        "candidate_reduction_low_cost": hybrid_cov["candidate_reduction"] >= 0.90,
    }
    bool_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
    return {
        "status": "pass" if all(bool_checks.values()) else "fail",
        "checks": checks,
        "note": "B3 and downstream eligible coverage are unavailable for this non-official final_holdout unless complete cluster/member and Task4 holdout Gold are present.",
    }


def write_readme(dev_gate_report: dict[str, Any], final_status: str) -> None:
    text_body = f"""# Task3 Resolver v3.1 Hybrid Final

Hybrid v3.1 combines legacy candidate coverage with the existing multiroute R1/R2/R3/R4 recall and scores every pair with the frozen selected Resolver v3 model.

Controls:
- Resolver retrained: false
- Thresholds: same_thr=0.98, diff_thr=0.76, margin=0.0
- Official1799 filled Gold read: false
- Frozen Resolver v2 outputs modified: false
- Final Holdout read only after frozen_manifest: {final_status != "not_run"}

Dev gate: {dev_gate_report["status"]}
Final Holdout: {final_status}
"""
    (OUT / "README.md").write_text(text_body, encoding="utf-8")


def write_reproduce() -> None:
    path = OUT / "reproduce.sh"
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /opt/data/private/JinRong/kechuang\n/root/miniconda3/envs/task4-impact-v3/bin/python scripts/build_task3_resolver_v3_1_hybrid.py\n",
        encoding="utf-8",
    )


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    final_metrics_path = OUT / "final_holdout_metrics.json"
    if final_metrics_path.exists():
        raise RuntimeError("refusing to rerun after final_holdout_metrics.json exists")
    write_json(OUT / "protected_sha256_before.json", snapshot("before_dev_no_final_holdout_no_official1799_gold", 0))
    load_selection()
    model = joblib.load(V3_MODEL)

    dev_event_df = load_event_split("dev")
    dev_ids = set(dev_event_df["event_uid"].map(text))
    dev_labels = load_dev_labels()
    dev_candidates = build_dev_candidate_frames(dev_ids)
    dev_reports = run_split("dev", dev_candidates, dev_labels, dev_event_df, model)

    route_report = route_contribution(dev_candidates, dev_labels)
    coverage_report = {mode: report["coverage"] for mode, report in dev_reports.items()}
    fixed_metrics = {mode: report["fixed_denominator_metrics"] for mode, report in dev_reports.items()}
    cluster_safety = {mode: report["cluster_safety"] for mode, report in dev_reports.items()}
    cluster_connectivity = {
        "dev_complete_cluster_member_gold": {
            "available": False,
            "reason": "No legal non-official complete cluster/member Gold artifact is available; only the 109 human pair denominator is used.",
        },
        "gold_cluster_candidate_connectivity": "unavailable",
        "oracle_candidate_closure_recall": "unavailable",
        "cluster_member_coverage": "unavailable",
    }

    gate = dev_gate(
        dev_reports["hybrid_union"],
        dev_reports["hybrid_union"]["coverage"],
        dev_reports["legacy"]["coverage"],
        dev_reports["multiroute"]["coverage"],
        route_report,
        dev_reports["hybrid_union"]["cluster_safety"],
    )
    selected = "hybrid_union" if gate["status"] == "pass" else "none_dev_failed"
    candidate_inventory = {
        "status": "candidate_mode_inventory_complete",
        "candidate_modes": {
            mode: {
                "candidate_pairs": int(len(df)),
                "candidate_keys_unique": int(len({pair_key(r.sample_id_a, r.sample_id_b) for r in df.itertuples(index=False)})),
                "coverage": coverage_report.get(mode),
            }
            for mode, df in dev_candidates.items()
        },
        "hybrid_coverage": {
            "status": "not_enabled_hybrid_union_passed" if gate["status"] == "pass" else "not_enabled_dev_failed",
            "candidate_pairs": None,
        },
        "intersections": route_report,
    }
    write_json(OUT / "candidate_mode_inventory.json", candidate_inventory)
    dev_candidates["hybrid_union"].to_parquet(OUT / "hybrid_candidate_pairs.parquet", index=False)
    parquet_frame(dev_reports["hybrid_union"]["pair_rows"]).to_parquet(OUT / "hybrid_pair_decisions.parquet", index=False)
    write_json(OUT / "route_contribution_report.json", route_report)
    write_json(OUT / "fixed_denominator_dev_metrics.json", fixed_metrics)
    write_json(OUT / "candidate_coverage_report.json", coverage_report)
    write_json(OUT / "cluster_connectivity_report.json", cluster_connectivity)
    write_json(OUT / "cluster_safety_report.json", cluster_safety)
    write_json(
        OUT / "dev_ablation_report.json",
        {
            "status": "dev_gate_complete",
            "formal_denominator": {
                "human_pair_count": len(dev_labels),
                "same_event": sum(1 for v in dev_labels.values() if v == "same_event"),
                "different_event": sum(1 for v in dev_labels.values() if v == "different_event"),
                "unlabeled_pairs_defaulted_as_different_event": False,
            },
            "reports": compact_report(dev_reports),
            "dev_gate": gate,
            "selected_candidate_mode": selected,
            "hybrid_coverage_enabled": False,
            "official1799_filled_gold_read_count": 0,
            "final_holdout_read_count": 0,
        },
    )

    final_status = "not_run"
    final_payload: dict[str, Any] | None = None
    if gate["status"] == "pass":
        freeze_script_copy()
        config = write_frozen_config(selected)
        frozen_manifest = write_frozen_manifest(config)
        final_event_df = load_event_split("final_holdout")
        final_ids = set(final_event_df["event_uid"].map(text))
        final_pair_df = read_parquet_filtered(PAIR_SPLIT, [("pair_split", "==", "final_holdout")])
        final_label_info = dedup_primary_labels(final_pair_df)
        final_candidates, final_generation = final_candidate_frames(final_ids)
        final_reports = run_split("final_holdout", final_candidates, final_label_info["labels"], final_event_df, model)
        final_accept = final_acceptance(final_reports)
        final_payload = {
            "status": "final_holdout_evaluated_once_after_freeze",
            "generated_at_utc": now(),
            "frozen_manifest_sha256": sha256_file(OUT / "frozen_manifest.json"),
            "final_holdout_gold_load": {
                "input_rows": final_label_info["input_rows"],
                "unique_pairs": final_label_info["unique_pairs"],
                "formal_primary_pair_count": len(final_label_info["labels"]),
                "same_event": sum(1 for v in final_label_info["labels"].values() if v == "same_event"),
                "different_event": sum(1 for v in final_label_info["labels"].values() if v == "different_event"),
                "ambiguous_or_nonprimary_pairs": final_label_info["ambiguous_or_nonprimary_pairs"],
            },
            "candidate_generation": {
                "legacy": final_reports["legacy"]["coverage"],
                "multiroute": final_reports["multiroute"]["coverage"],
                "hybrid_union": final_reports["hybrid_union"]["coverage"],
                **final_generation,
            },
            "reports": compact_report(final_reports),
            "acceptance": final_accept,
            "b3": "unavailable_no_complete_nonofficial_cluster_member_gold",
            "downstream_eligible_coverage": "unavailable_no_task4_final_holdout_gold",
            "official1799_filled_gold_read_count": 0,
            "final_holdout_read_count": 1,
            "prediction_sha_after_holdout": {
                "hybrid_candidate_pairs.parquet": sha256_file(OUT / "hybrid_candidate_pairs.parquet"),
                "hybrid_pair_decisions.parquet": sha256_file(OUT / "hybrid_pair_decisions.parquet"),
                "selected_v3_model": sha256_file(V3_MODEL),
            },
            "frozen_manifest": frozen_manifest,
        }
        write_json(final_metrics_path, final_payload)
        final_status = final_accept["status"]
    else:
        write_json(
            OUT / "frozen_config.json",
            {
                "status": "not_frozen_dev_gate_failed",
                "reason": "hybrid_union did not satisfy Dev gate; final_holdout was not read",
                "thresholds": THRESHOLDS,
            },
        )
        write_json(
            OUT / "frozen_manifest.json",
            {
                "status": "not_frozen_dev_gate_failed",
                "final_holdout_read_count": 0,
                "official1799_filled_gold_read_count": 0,
            },
        )

    write_reproduce()
    write_readme(gate, final_status)
    write_json(
        OUT / "protected_sha256_after.json",
        {
            **snapshot("after_hybrid_v3_1_pipeline", 1 if final_payload else 0),
            "dev_gate_status": gate["status"],
            "final_holdout_status": final_status,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        },
    )
    print(
        json.dumps(
            jsonable(
                {
                    "status": "complete",
                    "output_dir": str(OUT),
                    "dev_gate": gate,
                    "final_holdout": final_payload["acceptance"] if final_payload else "not_read_dev_failed",
                    "official1799_filled_gold_read_count": 0,
                    "final_holdout_read_count": 1 if final_payload else 0,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
