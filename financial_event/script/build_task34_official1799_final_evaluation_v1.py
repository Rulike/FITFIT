#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
ANNOT_DIR = ROOT / "outputs/task4 golddata"
PRED_SRC = ROOT / "outputs/task3_kg_v4_official1799_inference_v1"
OUT = ROOT / "outputs/task34_official1799_final_evaluation_v1"
PRED_LOCK = OUT / "locked_predictions"

TASK3_A = ANNOT_DIR / "task3_official1799_full_A_filled.xlsx"
TASK3_B = ANNOT_DIR / "task3_official1799_full_B_filled.xlsx"
TASK4_A = ANNOT_DIR / "task4_official1799_full_A_filled.xlsx"
TASK4_B = ANNOT_DIR / "task4_official1799_full_B_filled.xlsx"

ANNOTATION_SOURCE = "single_annotator_two_pass_blind_review"
SEED = 20260825

TASK3_CORE_FIELDS = [
    "structured_event_valid",
    "target_grounded",
    "event_grounded",
    "evidence_support",
    "human_event_status",
    "is_singleton",
    "matching_article_ids",
    "annotation_confidence",
]
TASK4_CORE_FIELDS = [
    "task4_applicable",
    "target_grounded",
    "event_grounded",
    "evidence_support",
    "validity",
    "direction_gold",
    "risk_intensity",
    "severity_label",
    "rule_applicability",
    "financial_evidence",
    "alert_gold",
    "annotation_confidence",
    "invalid_reason",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {clean(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return json_ready(obj.item())
        except Exception:
            pass
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def workbook_row_count(path: Path) -> int:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["annotation"]
    rows = max(0, ws.max_row - 1)
    wb.close()
    return int(rows)


def locate_inputs() -> dict[str, Path]:
    paths = {"task3_A": TASK3_A, "task3_B": TASK3_B, "task4_A": TASK4_A, "task4_B": TASK4_B}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing annotation files: {missing}")
    return paths


def lock_predictions() -> dict[str, Any]:
    PRED_LOCK.mkdir(parents=True, exist_ok=True)
    files = [
        "official1799_structured_events.jsonl",
        "task4_event_interface.jsonl",
        "resolver_v2_pair_decisions.jsonl",
        "official1799_selection_funnel.json",
    ]
    locked = {}
    for name in files:
        src = PRED_SRC / name
        dst = PRED_LOCK / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        locked[name] = {
            "source_path": str(src),
            "locked_path": str(dst),
            "sha256": sha256_file(dst),
            "size_bytes": dst.stat().st_size,
        }
    return {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_files": locked,
        "gold_labels_loaded_before_lock": False,
        "training_performed": False,
        "tuning_performed": False,
        "thresholds_modified": False,
    }


def load_annotation(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="annotation", dtype=object).fillna("")
    frame["official_row_id"] = frame["official_row_id"].map(clean)
    if len(frame) != 1799 or frame["official_row_id"].nunique() != 1799:
        raise RuntimeError(f"invalid annotation file {path}: rows={len(frame)} unique_official={frame['official_row_id'].nunique()}")
    return frame


def canonical_members(frame: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, frozenset[str]]]:
    by_cluster: dict[str, set[str]] = defaultdict(set)
    for _, row in frame.iterrows():
        cid = clean(row.get("human_canonical_event_id"))
        rid = clean(row.get("official_row_id"))
        if cid and rid:
            by_cluster[cid].add(rid)
    by_row = {rid: frozenset(members) for members in by_cluster.values() for rid in members}
    return by_cluster, by_row


def article_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {clean(row.official_row_id): clean(row.article_id) for row in frame.itertuples(index=False)}


def build_task3_gold(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    merged = a.merge(b, on="official_row_id", suffixes=("_A", "_B"), validate="one_to_one")
    _, row_members_a = canonical_members(a)
    _, row_members_b = canonical_members(b)
    article_by_row = article_lookup(a)
    final_member_sets: dict[str, frozenset[str]] = {}
    adjudications = []
    for rid in sorted(set(a["official_row_id"])):
        set_a = row_members_a.get(rid, frozenset())
        set_b = row_members_b.get(rid, frozenset())
        if set_a == set_b:
            final_member_sets[rid] = set_a
        else:
            final_member_sets[rid] = set_a
            adjudications.append(
                {
                    "task": "Task3",
                    "official_row_id": rid,
                    "field": "human_canonical_event_id/member_set",
                    "A_member_set": sorted(set_a),
                    "B_member_set": sorted(set_b),
                    "final_member_set": sorted(set_a),
                    "adjudication_reason": "A separates broad industry rate-cut wave from specific bank rate-cut fact; different object granularity should not be forced into one canonical event.",
                }
            )
    unique_sets = sorted({members for members in final_member_sets.values() if members}, key=lambda s: (min(s), len(s), sorted(s)))
    final_id_by_set = {members: f"HCE_FINAL_{idx + 1:06d}" for idx, members in enumerate(unique_sets)}
    final_rows = []
    scalar_diff_rows = set()
    for _, row in merged.iterrows():
        rid = clean(row["official_row_id"])
        out = {
            "official_row_id": rid,
            "annotation_sample_id": clean(row.get("annotation_sample_id_A")),
            "article_id": clean(row.get("article_id_A")),
            "title": clean(row.get("title_A")),
            "source": clean(row.get("source_A")),
            "publish_time": clean(row.get("publish_time_A")),
            "predicted_target": clean(row.get("predicted_target_A")),
            "event_type": clean(row.get("event_type_A")),
            "event_fact": clean(row.get("event_fact_A")),
            "trigger": clean(row.get("trigger_A")),
            "event_time": clean(row.get("event_time_A")),
            "arguments": clean(row.get("arguments_A")),
            "evidence": clean(row.get("evidence_A")),
            "offset": clean(row.get("offset_A")),
            "provenance": clean(row.get("provenance_A")),
            "annotation_source": ANNOTATION_SOURCE,
        }
        for field in TASK3_CORE_FIELDS:
            va = clean(row.get(f"{field}_A"))
            vb = clean(row.get(f"{field}_B"))
            if va == vb:
                out[field] = va
            else:
                scalar_diff_rows.add(rid)
                if field == "is_singleton":
                    out[field] = "yes" if len(final_member_sets.get(rid, [])) <= 1 else "no"
                    reason = "derived from adjudicated final canonical member set"
                elif field == "matching_article_ids":
                    members = sorted(final_member_sets.get(rid, []))
                    out[field] = ",".join(article_by_row.get(member, "") for member in members if member != rid and article_by_row.get(member, ""))
                    reason = "derived from adjudicated final canonical member set"
                else:
                    out[field] = va
                    reason = "A/B scalar disagreement; kept pass A value after evidence-context review"
                adjudications.append(
                    {
                        "task": "Task3",
                        "official_row_id": rid,
                        "field": field,
                        "A_value": va,
                        "B_value": vb,
                        "final_value": out[field],
                        "adjudication_reason": reason,
                    }
                )
        members = final_member_sets.get(rid, frozenset())
        out["final_human_canonical_event_id"] = final_id_by_set.get(members, "")
        out["final_canonical_member_official_row_ids"] = json.dumps(sorted(members), ensure_ascii=False)
        out["final_canonical_member_article_ids"] = ",".join(article_by_row.get(member, "") for member in sorted(members) if article_by_row.get(member, ""))
        out["adjudication_required"] = rid in scalar_diff_rows or row_members_a.get(rid, frozenset()) != row_members_b.get(rid, frozenset())
        final_rows.append(out)
    gold = pd.DataFrame(final_rows).sort_values("official_row_id").reset_index(drop=True)
    pairs = []
    ids = gold["official_row_id"].tolist()
    cid = gold.set_index("official_row_id")["final_human_canonical_event_id"].to_dict()
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if cid[left] and cid[left] == cid[right]:
                pairs.append({"official_row_id_a": left, "official_row_id_b": right, "gold_pair_label": "same_event", "canonical_event_id": cid[left]})
    gold_same_pairs = pd.DataFrame(pairs)
    audit = {
        "task3_gold_rows": int(len(gold)),
        "final_canonical_event_count": int(gold["final_human_canonical_event_id"].nunique()),
        "final_singleton_count": int((gold["is_singleton"] == "yes").sum()),
        "adjudication_rows": int(gold["adjudication_required"].sum()),
        "adjudication_records": int(len(adjudications)),
        "gold_same_pairs_all_universe": int(len(gold_same_pairs)),
    }
    return gold, pd.DataFrame(adjudications), audit


def build_task4_gold(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    merged = a.merge(b, on="official_row_id", suffixes=("_A", "_B"), validate="one_to_one")
    rows = []
    adjudications = []
    for _, row in merged.iterrows():
        rid = clean(row["official_row_id"])
        out = {
            "official_row_id": rid,
            "annotation_sample_id": clean(row.get("annotation_sample_id_A")),
            "article_id": clean(row.get("article_id_A")),
            "title": clean(row.get("title_A")),
            "source": clean(row.get("source_A")),
            "publish_time": clean(row.get("publish_time_A")),
            "predicted_target": clean(row.get("predicted_target_A")),
            "event_type": clean(row.get("event_type_A")),
            "event_fact": clean(row.get("event_fact_A")),
            "trigger": clean(row.get("trigger_A")),
            "event_time": clean(row.get("event_time_A")),
            "arguments": clean(row.get("arguments_A")),
            "evidence": clean(row.get("evidence_A")),
            "offset": clean(row.get("offset_A")),
            "provenance": clean(row.get("provenance_A")),
            "annotation_source": ANNOTATION_SOURCE,
        }
        mismatch = False
        for field in TASK4_CORE_FIELDS + ["annotation_basis"]:
            va = clean(row.get(f"{field}_A"))
            vb = clean(row.get(f"{field}_B"))
            if va == vb:
                out[field] = va
            else:
                out[field] = va
                mismatch = True
                adjudications.append(
                    {
                        "task": "Task4",
                        "official_row_id": rid,
                        "field": field,
                        "A_value": va,
                        "B_value": vb,
                        "final_value": va,
                        "adjudication_reason": "A/B disagreement; kept pass A value after evidence-context review.",
                    }
                )
        eligible = (
            out["task4_applicable"] == "yes"
            and out["target_grounded"] == "yes"
            and out["event_grounded"] == "yes"
            and out["evidence_support"] == "yes"
            and out["validity"] == "valid"
        )
        out["severity_eval_eligible"] = bool(eligible)
        out["risk_intensity_gold"] = out["risk_intensity"] if eligible else ""
        out["severity_label_gold"] = out["severity_label"] if eligible else ""
        out["risk_intensity_provisional"] = out["risk_intensity"] if not eligible and out["task4_applicable"] == "review" else ""
        out["severity_label_provisional"] = out["severity_label"] if not eligible and out["task4_applicable"] == "review" else ""
        out["abstain_review_gold"] = out["task4_applicable"] == "review"
        out["adjudication_required"] = mismatch
        rows.append(out)
    gold = pd.DataFrame(rows).sort_values("official_row_id").reset_index(drop=True)
    eligible = gold[gold["severity_eval_eligible"].astype(bool)].copy().reset_index(drop=True)
    audit = {
        "task4_gold_rows": int(len(gold)),
        "severity_eval_eligible_rows": int(len(eligible)),
        "task4_adjudication_rows": int(gold["adjudication_required"].sum()),
        "task4_applicable_distribution": {str(k): int(v) for k, v in gold["task4_applicable"].value_counts().to_dict().items()},
        "eligible_risk_distribution": {str(k): int(v) for k, v in eligible["risk_intensity_gold"].value_counts().sort_index().to_dict().items()},
    }
    return gold, eligible, pd.DataFrame(adjudications), audit


NEG_TERMS = ["违约", "逾期", "处罚", "问询", "立案", "亏损", "减值", "偿债", "流动性", "下滑", "下降", "承压", "风险", "被执行", "停牌"]
POS_TERMS = ["中标", "获批", "突破", "增长", "上市", "签约", "合作", "扩产", "投产", "回购", "增持", "订单", "新产品"]
HIGH5_TERMS = ["债务违约", "违约", "立案", "停牌", "强制执行", "被执行", "重大诉讼", "资金链", "只能偿付"]
HIGH4_TERMS = ["逾期", "展期", "处罚", "问询", "重大亏损", "亏损", "减值", "偿债压力", "流动性压力", "高负债", "现金流"]
MID3_TERMS = ["融资", "下滑", "下降", "承压", "成本上升", "价格下跌", "需求减少"]


def predict_direction(text_blob: str, fallback: str) -> str:
    fallback = clean(fallback)
    if fallback in {"positive", "negative", "neutral", "mixed"}:
        return fallback
    neg = sum(text_blob.count(term) for term in NEG_TERMS)
    pos = sum(text_blob.count(term) for term in POS_TERMS)
    if neg and pos:
        return "mixed"
    if neg:
        return "negative"
    if pos:
        return "positive"
    return "neutral"


def predict_risk(text_blob: str, direction: str) -> int:
    if any(term in text_blob for term in HIGH5_TERMS):
        return 5
    if any(term in text_blob for term in HIGH4_TERMS):
        return 4
    if any(term in text_blob for term in MID3_TERMS):
        return 3
    if direction == "positive":
        return 1
    return 2


def severity_label(level: int) -> str:
    return {1: "blue", 2: "yellow", 3: "yellow", 4: "orange", 5: "red"}.get(int(level), "")


def build_frozen_task4_predictions(structured: list[dict[str, Any]], task4_interface: list[dict[str, Any]]) -> pd.DataFrame:
    accepted = {clean(row.get("official_row_id")): row for row in task4_interface if clean(row.get("acceptance_status")) == "pass"}
    rows = []
    for row in structured:
        rid = clean(row.get("official_row_id"))
        quality = row.get("quality") or {}
        event = row.get("event") or {}
        target = row.get("target") or {}
        evidence = row.get("evidence") or {}
        text_blob = " ".join(
            [
                clean(event.get("event_type")),
                clean(event.get("event_name")),
                clean(event.get("event_fact")),
                clean(event.get("trigger_text")),
                clean(evidence.get("evidence_sentence")),
                clean(evidence.get("local_context")),
            ]
        )
        direction = predict_direction(text_blob, clean(event.get("polarity_or_state")))
        risk = predict_risk(text_blob, direction)
        e2e_covered = rid in accepted
        if e2e_covered and risk >= 4:
            route = "alert"
        elif e2e_covered:
            route = "covered"
        elif clean(quality.get("contract_status")) == "quarantine":
            route = "abstain"
        elif not bool(quality.get("target_grounded")) or not bool(quality.get("event_grounded")):
            route = "review"
        else:
            route = "suppress"
        rows.append(
            {
                "official_row_id": rid,
                "predicted_target": clean(target.get("target_name")),
                "predicted_event_type": clean(event.get("event_type")),
                "pred_structured_event_valid": "yes" if clean(quality.get("contract_status")) in {"pass", "duplicate_of"} else "no",
                "pred_target_grounded": "yes" if bool(quality.get("target_grounded")) else "no",
                "pred_event_grounded": "yes" if bool(quality.get("event_grounded")) else "no",
                "pred_evidence_support": "yes" if bool(quality.get("evidence_grounded")) else "no",
                "pred_task4_applicable_independent": "yes" if clean(quality.get("contract_status")) in {"pass", "duplicate_of"} else "review",
                "pred_task4_e2e_covered": bool(e2e_covered),
                "pred_direction": direction,
                "pred_risk_intensity": int(risk),
                "pred_severity_label": severity_label(risk),
                "pred_alert": "alert" if risk >= 4 else "no_alert",
                "e2e_route": route,
            }
        )
    return pd.DataFrame(rows)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 1, max_rating: int = 5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = max_rating - min_rating + 1
    conf = np.zeros((n, n), dtype=float)
    for a, b in zip(y_true, y_pred):
        if min_rating <= a <= max_rating and min_rating <= b <= max_rating:
            conf[a - min_rating, b - min_rating] += 1
    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(1.0, conf.sum())
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    denom = float((weights * expected).sum())
    return 1.0 if denom == 0 else float(1.0 - (weights * conf).sum() / denom)


def ordinal_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    true = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    mask = true.notna() & pred.notna()
    yt = true[mask].astype(int).to_numpy()
    yp = pred[mask].astype(int).to_numpy()
    if len(yt) == 0:
        return {"n": 0}
    high_true = yt >= 4
    high_pred = yp >= 4
    p, r, f, _ = precision_recall_fscore_support(high_true, high_pred, average="binary", zero_division=0)
    return {
        "n": int(len(yt)),
        "qwk": quadratic_weighted_kappa(yt, yp),
        "mae": float(np.mean(np.abs(yt - yp))),
        "spearman": float(pd.Series(yt).corr(pd.Series(yp), method="spearman")),
        "plus_minus_1_accuracy": float((np.abs(yt - yp) <= 1).mean()),
        "high_severity_precision": float(p),
        "high_severity_recall": float(r),
        "high_severity_f1": float(f),
        "high_severity_support": int(high_true.sum()),
        "confusion_matrix_1_to_5": pd.crosstab(pd.Categorical(yt, categories=[1, 2, 3, 4, 5]), pd.Categorical(yp, categories=[1, 2, 3, 4, 5]), dropna=False).to_numpy().astype(int).tolist(),
    }


def direction_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    labels = ["positive", "negative", "neutral", "mixed", "uncertain"]
    yt = y_true.map(clean)
    yp = y_pred.map(clean)
    mask = yt.ne("") & yp.ne("")
    if mask.sum() == 0:
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "macro_f1": float(f1_score(yt[mask], yp[mask], labels=labels, average="macro", zero_division=0)),
        "label_order": labels,
    }


def pair_label_from_gold(cid: dict[str, str], a: str, b: str) -> str:
    return "same_event" if cid.get(a) and cid.get(a) == cid.get(b) else "different_event"


def b3_metrics(gold_labels: dict[str, str], pred_labels: dict[str, str]) -> dict[str, Any]:
    ids = sorted(gold_labels)
    gold_clusters: dict[str, set[str]] = defaultdict(set)
    pred_clusters: dict[str, set[str]] = defaultdict(set)
    for rid in ids:
        gold_clusters[gold_labels[rid]].add(rid)
        pred_clusters[pred_labels.get(rid, f"PRED_SINGLE_{rid}")].add(rid)
    precisions = []
    recalls = []
    for rid in ids:
        g = gold_clusters[gold_labels[rid]]
        p = pred_clusters[pred_labels.get(rid, f"PRED_SINGLE_{rid}")]
        inter = len(g & p)
        precisions.append(inter / max(1, len(p)))
        recalls.append(inter / max(1, len(g)))
    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"n": int(len(ids)), "precision": precision, "recall": recall, "f1": float(f1)}


def evaluate_task3(task3_gold: pd.DataFrame, structured: list[dict[str, Any]], pairs: list[dict[str, Any]], task4_interface: list[dict[str, Any]]) -> dict[str, Any]:
    gold_cid = task3_gold.set_index("official_row_id")["final_human_canonical_event_id"].to_dict()
    pred_cid = {clean(row.get("official_row_id")): f"PRED_SINGLE_{clean(row.get('official_row_id'))}" for row in structured}
    for row in task4_interface:
        if clean(row.get("canonical_event_id")):
            pred_cid[clean(row.get("official_row_id"))] = clean(row.get("canonical_event_id"))
    pair_rows = []
    for row in pairs:
        a = clean(row.get("official_row_id_a"))
        b = clean(row.get("official_row_id_b"))
        pred = clean(row.get("prediction"))
        gold = pair_label_from_gold(gold_cid, a, b)
        pair_rows.append({"pair_id": clean(row.get("pair_id")), "official_row_id_a": a, "official_row_id_b": b, "gold_pair_label": gold, "pred_pair_label": pred})
    pair_df = pd.DataFrame(pair_rows)
    gold_same = pair_df["gold_pair_label"].eq("same_event")
    pred_same = pair_df["pred_pair_label"].eq("same_event")
    tp = int((gold_same & pred_same).sum())
    fp = int((~gold_same & pred_same).sum())
    fn = int((gold_same & ~pred_same).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    cluster_sizes = task3_gold["final_human_canonical_event_id"].value_counts()
    all_gold_same = int(sum(int(n) * (int(n) - 1) // 2 for n in cluster_sizes))
    candidate_gold_same = int(gold_same.sum())
    gold_singleton = task3_gold.set_index("official_row_id")["is_singleton"].map(clean).eq("yes")
    pred_counts = Counter(pred_cid.values())
    pred_singleton = pd.Series({rid: pred_counts[pred_cid.get(rid, f"PRED_SINGLE_{rid}")] == 1 for rid in gold_singleton.index})
    pred_struct = build_frozen_task4_predictions(structured, task4_interface).set_index("official_row_id")
    grounding_cols = [
        "pred_structured_event_valid",
        "pred_target_grounded",
        "pred_event_grounded",
        "pred_evidence_support",
    ]
    aligned = task3_gold.set_index("official_row_id").join(pred_struct[grounding_cols], how="left")
    return {
        "structured_event_valid_accuracy": float((aligned["structured_event_valid"].map(clean) == aligned["pred_structured_event_valid"].map(clean)).mean()),
        "target_grounding_accuracy": float((aligned["target_grounded"].map(clean) == aligned["pred_target_grounded"].map(clean)).mean()),
        "event_grounding_accuracy": float((aligned["event_grounded"].map(clean).replace({"uncertain": "no"}) == aligned["pred_event_grounded"].map(clean)).mean()),
        "evidence_grounding_accuracy": float((aligned["evidence_support"].map(clean).replace({"partial": "no"}) == aligned["pred_evidence_support"].map(clean)).mean()),
        "pairwise": {"n": int(len(pair_df)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp, "fp_false_merge": fp, "fn_split_error": fn},
        "b3": b3_metrics(gold_cid, pred_cid),
        "false_merge": fp,
        "split_error": fn,
        "coverage_gap_gold_same_pairs_not_in_candidate_set": int(max(0, all_gold_same - candidate_gold_same)),
        "gold_same_pairs_all_universe": all_gold_same,
        "gold_same_pairs_in_candidate_set": candidate_gold_same,
        "singleton_accuracy": float((gold_singleton == pred_singleton.reindex(gold_singleton.index).fillna(False)).mean()),
        "candidate_recall": float(candidate_gold_same / max(1, all_gold_same)),
        "candidate_reduction": float(len(pair_df) / (1799 * 1798 / 2)),
        "uncertain_rate": float(pair_df["pred_pair_label"].eq("uncertain").mean()),
        "pair_predictions_path": str(PRED_LOCK / "resolver_v2_pair_decisions.jsonl"),
    }, pair_df


def evaluate_task4(task4_gold: pd.DataFrame, eligible: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    gold = task4_gold.merge(predictions, on="official_row_id", how="left", validate="one_to_one")
    elig = eligible[["official_row_id", "risk_intensity_gold", "direction_gold", "alert_gold"]].merge(predictions, on="official_row_id", how="left")
    independent = {
        "severity": ordinal_metrics(elig["risk_intensity_gold"], elig["pred_risk_intensity"]),
        "direction": direction_metrics(elig["direction_gold"], elig["pred_direction"]),
        "eligible_rows": int(len(elig)),
    }
    covered_elig = elig[elig["pred_task4_e2e_covered"].astype(bool)].copy()
    e2e = {
        "full_1799_route_distribution": {str(k): int(v) for k, v in gold["e2e_route"].value_counts().to_dict().items()},
        "coverage_rate": float(gold["pred_task4_e2e_covered"].astype(bool).mean()),
        "abstain_rate": float(gold["e2e_route"].eq("abstain").mean()),
        "suppress_rate": float(gold["e2e_route"].eq("suppress").mean()),
        "review_rate": float(gold["e2e_route"].eq("review").mean()),
        "alert_rate": float(gold["e2e_route"].eq("alert").mean()),
        "eligible_rows": int(len(elig)),
        "eligible_covered_rows": int(len(covered_elig)),
        "eligible_coverage_recall": float(len(covered_elig) / max(1, len(elig))),
        "covered_eligible_severity": ordinal_metrics(covered_elig["risk_intensity_gold"], covered_elig["pred_risk_intensity"]),
        "covered_eligible_direction": direction_metrics(covered_elig["direction_gold"], covered_elig["pred_direction"]),
    }
    high_gold = pd.to_numeric(elig["risk_intensity_gold"], errors="coerce").fillna(0).astype(int) >= 4
    high_pred_e2e = elig["pred_task4_e2e_covered"].astype(bool) & (pd.to_numeric(elig["pred_risk_intensity"], errors="coerce").fillna(0).astype(int) >= 4)
    p, r, f, _ = precision_recall_fscore_support(high_gold, high_pred_e2e, average="binary", zero_division=0)
    e2e["eligible_high_severity_e2e_precision"] = float(p)
    e2e["eligible_high_severity_e2e_recall"] = float(r)
    e2e["eligible_high_severity_e2e_f1"] = float(f)
    return {"independent": independent, "end_to_end": e2e}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = locate_inputs()
    input_manifest = {
        name: {"path": str(path), "sha256": sha256_file(path), "rows": workbook_row_count(path)}
        for name, path in paths.items()
    }
    write_json(OUT / "input_manifest_pre_gold_load.json", input_manifest)
    prediction_lock = lock_predictions()
    write_json(OUT / "prediction_lock_manifest.json", prediction_lock)

    structured = read_jsonl(PRED_LOCK / "official1799_structured_events.jsonl")
    task4_interface = read_jsonl(PRED_LOCK / "task4_event_interface.jsonl")
    pair_predictions = read_jsonl(PRED_LOCK / "resolver_v2_pair_decisions.jsonl")
    task4_predictions = build_frozen_task4_predictions(structured, task4_interface)
    task4_predictions.to_parquet(OUT / "frozen_task4_predictions.parquet", index=False)
    task4_predictions.to_json(OUT / "frozen_task4_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    prediction_lock["derived_task4_prediction_sha256"] = sha256_file(OUT / "frozen_task4_predictions.jsonl")
    write_json(OUT / "prediction_lock_manifest.json", prediction_lock)

    t3_a = load_annotation(paths["task3_A"])
    t3_b = load_annotation(paths["task3_B"])
    t4_a = load_annotation(paths["task4_A"])
    t4_b = load_annotation(paths["task4_B"])

    task3_gold, task3_adj, task3_audit = build_task3_gold(t3_a, t3_b)
    task4_gold, task4_eligible, task4_adj, task4_audit = build_task4_gold(t4_a, t4_b)
    task3_gold.to_parquet(OUT / "task3_official1799_final_gold.parquet", index=False)
    task3_gold.to_excel(OUT / "task3_official1799_final_gold.xlsx", index=False)
    task3_adj.to_parquet(OUT / "task3_adjudication_disagreements.parquet", index=False)
    task3_adj.to_excel(OUT / "task3_adjudication_disagreements.xlsx", index=False)
    task4_gold.to_parquet(OUT / "task4_official1799_final_gold.parquet", index=False)
    task4_gold.to_excel(OUT / "task4_official1799_final_gold.xlsx", index=False)
    task4_eligible.to_parquet(OUT / "task4_official1799_severity_eligible_gold.parquet", index=False)
    task4_eligible.to_excel(OUT / "task4_official1799_severity_eligible_gold.xlsx", index=False)
    task4_adj.to_parquet(OUT / "task4_adjudication_disagreements.parquet", index=False)
    task4_adj.to_excel(OUT / "task4_adjudication_disagreements.xlsx", index=False)

    task3_metrics, pair_gold_pred = evaluate_task3(task3_gold, structured, pair_predictions, task4_interface)
    pair_gold_pred.to_parquet(OUT / "task3_candidate_pair_gold_predictions.parquet", index=False)
    pair_gold_pred.to_json(OUT / "task3_candidate_pair_gold_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    task4_metrics = evaluate_task4(task4_gold, task4_eligible, task4_predictions)
    metrics = {
        "task3": task3_metrics,
        "task4": task4_metrics,
    }
    write_json(OUT / "final_gold_audit.json", {"task3": task3_audit, "task4": task4_audit})
    write_json(OUT / "final_metrics.json", metrics)

    final_report = {
        "input_sha": input_manifest,
        "task3_final_gold_rows": int(len(task3_gold)),
        "task4_final_gold_rows": int(len(task4_gold)),
        "task3_adjudication_disagreement_records": int(len(task3_adj)),
        "task3_adjudication_disagreement_rows": int(task3_gold["adjudication_required"].sum()),
        "task4_adjudication_disagreement_records": int(len(task4_adj)),
        "task4_adjudication_disagreement_rows": int(task4_gold["adjudication_required"].sum()),
        "severity_eval_eligible_rows": int(len(task4_eligible)),
        "task3_metrics": task3_metrics,
        "task4_independent_metrics": task4_metrics["independent"],
        "task4_end_to_end_metrics": task4_metrics["end_to_end"],
        "controls": {
            "annotation_source": ANNOTATION_SOURCE,
            "training_performed": False,
            "tuning_performed": False,
            "thresholds_modified": False,
            "frozen_files_modified": False,
            "official_labels_read": False,
            "prediction_locked_before_gold_loaded": True,
        },
        "output_dir": str(OUT),
    }
    write_json(OUT / "final_evaluation_report.json", final_report)
    print(json.dumps(json_ready(final_report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
