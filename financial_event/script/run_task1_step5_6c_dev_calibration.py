#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.6-C dev-only classifier calibration.

This script only uses the frozen Step 5.4 dev split for calibration. 80-row
labels are intentionally not loaded here; after a single dev-selected strategy
is frozen, the strategy is applied to the already frozen Step 5.6-B 80-row pair
scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_5_candidate_v2 import (  # noqa: E402
    candidate_rows_for_top_k,
    load_config,
    load_frozen_split,
    protected_file_list,
    read_jsonl,
    rel,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)
from src.evaluation.task1_task_metrics import (  # noqa: E402
    candidate_recall_at_k,
    derive_per_class_thresholds,
    evaluate_event_predictions,
)

OUTPUT_DIR = ROOT / "outputs/task1_step5_6c_classifier_calibration"
CONFIG_PATH = ROOT / "configs/task1_candidate_v2.yaml"
CONTRACT_JSON = OUTPUT_DIR / "contract_comparison.json"
STEP56B_DIR = ROOT / "outputs/task1_step5_6b_latest_classifier_shadow"
LATEST_MODEL_DIR = ROOT / "outputs/task1_limited_training_round1/model_epoch_2"
LATEST_MODEL_SHA = "8ccd5ec466d39325762f7bbdf2da6e0c435a0c42dd0f871e84df7fc33cc6a35d"
FORBIDDEN_GOLD_LABEL_PATHS = {
    "data/processed/labeled/news_test.jsonl",
    "data/processed/labeled/event_pairs_test.jsonl",
    "data/processed/model_data/event_match_official_test.jsonl",
}


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_sha_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append((path.name, sha256_file(path)))
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in rows),
        encoding="utf-8",
    )


def protected_snapshot(path: Path) -> dict[str, str]:
    rows = {rel(item): sha256_file(item) for item in protected_file_list()}
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(rows.items())), encoding="utf-8")
    return rows


def process_check() -> dict[str, Any]:
    try:
        proc = subprocess.run(["ps", "-eo", "pid,ppid,stat,cmd"], check=True, text=True, capture_output=True)
    except Exception as exc:  # pragma: no cover
        return {"status": "unknown", "error": repr(exc), "suspicious": []}
    current = str(os.getpid())
    suspicious = []
    tokens = ["task1", "event_match", "candidate_v2"]
    for line in proc.stdout.splitlines()[1:]:
        if current in line:
            continue
        text = line.lower()
        if any(token in text for token in tokens) and "run_task1_step5_6c_dev_calibration.py" not in text:
            if "rg " in text or "sed " in text or "cat " in text:
                continue
            suspicious.append(line.strip())
    return {"status": "ok" if not suspicious else "suspicious_processes_found", "suspicious": suspicious}


def assert_no_forbidden_path(path: Path) -> None:
    rel_path = rel(path) if path.exists() or path.is_absolute() else path.as_posix()
    if rel_path in FORBIDDEN_GOLD_LABEL_PATHS:
        raise RuntimeError(f"forbidden gold-test label path requested: {rel_path}")


def load_event_names() -> list[str]:
    return sorted({str(row["event_name"]) for row in read_jsonl(ROOT / "data/processed/event_schema.jsonl") if row.get("event_name")})


def train_support(split: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(str(row["event_name"]) for row in split["train_pairs"] if row.get("event_name")))


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ["min", "p10", "p25", "p50", "p75", "p90", "max", "mean"]}
    xs = sorted(float(v) for v in values)

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {
        "min": xs[0],
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "max": xs[-1],
        "mean": sum(xs) / len(xs),
    }


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    sum_pos_ranks = sum(rank for rank, label in zip(ranks, labels) if label)
    return (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)


def pr_auc_average_precision(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(labels)
    if pos == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _, label in pairs:
        if label:
            tp += 1
            recall = tp / pos
            precision = tp / (tp + fp)
            ap += (recall - prev_recall) * precision
            prev_recall = recall
        else:
            fp += 1
    return ap


def gold_by_article(gold_pairs: Iterable[dict[str, Any]], article_ids: Iterable[str]) -> dict[str, set[str]]:
    by_article = {str(aid): set() for aid in article_ids}
    for row in gold_pairs:
        by_article.setdefault(str(row["article_id"]), set()).add(str(row["event_name"]))
    return by_article


def add_gold_labels(rows: list[dict[str, Any]], gold_pairs: list[dict[str, Any]], article_ids: list[str]) -> list[dict[str, Any]]:
    gold = gold_by_article(gold_pairs, article_ids)
    out = []
    for row in rows:
        item = dict(row)
        item["gold_label"] = int(str(row.get("event_name") or "") in gold.get(str(row.get("article_id") or ""), set()))
        out.append(item)
    return out


def score_distribution(rows: list[dict[str, Any]], article_ids: list[str], gold_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = add_gold_labels(rows, gold_pairs, article_ids)
    labels = [int(row["gold_label"]) for row in labeled]
    scores = [float(row.get("probability_positive", 0.0) or 0.0) for row in labeled]
    pos_scores = [score for label, score in zip(labels, scores) if label]
    neg_scores = [score for label, score in zip(labels, scores) if not label]
    return {
        "candidate_pair_count": len(labeled),
        "positive_pair_count": len(pos_scores),
        "negative_pair_count": len(neg_scores),
        "positive_score_distribution": quantiles(pos_scores),
        "negative_score_distribution": quantiles(neg_scores),
        "roc_auc": roc_auc(labels, scores),
        "pr_auc_average_precision": pr_auc_average_precision(labels, scores),
        "threshold_0_93_pair_confusion": pair_confusion(labeled, 0.93),
        "gold_model_rank_recall": gold_rank_recall(rows, article_ids, gold_pairs),
    }


def pair_confusion(labeled_rows: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for row in labeled_rows:
        pred = float(row.get("probability_positive", 0.0) or 0.0) >= threshold
        label = bool(row.get("gold_label"))
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def gold_rank_recall(rows: list[dict[str, Any]], article_ids: list[str], gold_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_article[str(row["article_id"])].append(row)
    rank_rows = []
    hit = Counter()
    total = 0
    for aid in article_ids:
        ordered = sorted(
            by_article.get(str(aid), []),
            key=lambda item: (-float(item.get("probability_positive", 0.0) or 0.0), int(item.get("candidate_rank", 10**9) or 10**9), str(item.get("event_name", ""))),
        )
        ranks = {str(row.get("event_name") or ""): idx for idx, row in enumerate(ordered, 1)}
        scores = {str(row.get("event_name") or ""): float(row.get("probability_positive", 0.0) or 0.0) for row in ordered}
        cand_ranks = {str(row.get("event_name") or ""): row.get("candidate_rank", "") for row in ordered}
        for pair in gold_pairs:
            if str(pair["article_id"]) != str(aid):
                continue
            event_name = str(pair["event_name"])
            total += 1
            rank = ranks.get(event_name)
            for k in [1, 3, 5]:
                if rank is not None and rank <= k:
                    hit[k] += 1
            rank_rows.append(
                {
                    "article_id": aid,
                    "event_name": event_name,
                    "candidate_rank": cand_ranks.get(event_name, ""),
                    "model_rank": rank if rank is not None else "",
                    "model_score": scores.get(event_name, ""),
                }
            )
    return {
        "gold_event_count": total,
        "top1_recall": hit[1] / total if total else 0.0,
        "top3_recall": hit[3] / total if total else 0.0,
        "top5_recall": hit[5] / total if total else 0.0,
        "rank_rows": rank_rows,
    }


def top20(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("candidate_rank", 10**9) or 10**9) <= 20]


def model_specs() -> list[dict[str, Any]]:
    return [
        {
            "model_id": "event_match_baseline_best_model",
            "model_dir": "outputs/event_match_baseline/best_model",
            "label_space": "binary_pair",
            "score_path": "outputs/task1_existing_checkpoint_comparison/scores/outputs_event_match_baseline_best_model/candidate_v2_top20_scores.jsonl",
            "input_contract": "article-target-event pair text",
            "original_threshold": 0.38,
        },
        {
            "model_id": "step5_4_checkpoint_849",
            "model_dir": "outputs/task1_step5_4_dev_baseline_v1/model/checkpoint-849",
            "label_space": "binary_pair",
            "score_path": "outputs/task1_existing_checkpoint_comparison/scores/outputs_task1_step5_4_dev_baseline_v1_model_checkpoint-849/candidate_v2_top20_scores.jsonl",
            "input_contract": "article-target-event pair text",
            "original_threshold": 0.38,
        },
        {
            "model_id": "step5_5_class_balanced_bce_epoch_2",
            "model_dir": "outputs/task1_step5_5_candidate_v2_dev/new_classifier_models/class_balanced_bce_epoch_2",
            "label_space": "64_event_name_multilabel",
            "score_path": "outputs/task1_step5_5_candidate_v2_dev/new_classifier_all64_predictions.jsonl",
            "input_contract": "article text -> 64 logits",
            "original_threshold": 0.53,
        },
        {
            "model_id": "limited_round1_model_epoch_2",
            "model_dir": "outputs/task1_limited_training_round1/model_epoch_2",
            "label_space": "binary_pair",
            "score_path": "outputs/task1_limited_training_round1/round1_epoch_2_dev_scores.jsonl",
            "input_contract": "article-target-event pair text",
            "original_threshold": 0.93,
        },
        {
            "model_id": "event_match_hard_negative_best_model",
            "model_dir": "outputs/event_match_hard_negative/best_model",
            "label_space": "binary_pair",
            "score_path": "outputs/task1_existing_checkpoint_comparison/scores/outputs_event_match_hard_negative_best_model/candidate_v2_top20_scores.jsonl",
            "input_contract": "article-target-event pair text",
            "original_threshold": 0.45,
            "extra_comparison": True,
        },
    ]


def load_model_rows(spec: dict[str, Any], article_ids: list[str]) -> list[dict[str, Any]]:
    path = ROOT / spec["score_path"]
    assert_no_forbidden_path(path)
    rows = read_jsonl(path)
    ids = set(map(str, article_ids))
    rows = [row for row in rows if str(row.get("article_id") or "") in ids]
    rows = top20(rows)
    if not rows:
        raise RuntimeError(f"no score rows loaded for {spec['model_id']} from {spec['score_path']}")
    return rows


def rows_by_article(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["article_id"])].append(row)
    return grouped


def normalize_candidate_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = rows_by_article(rows)
    out = []
    for _, article_rows in grouped.items():
        values = [float(row.get("candidate_score", 0.0) or 0.0) for row in article_rows]
        lo = min(values) if values else 0.0
        hi = max(values) if values else 0.0
        for row in article_rows:
            item = dict(row)
            raw = float(row.get("candidate_score", 0.0) or 0.0)
            item["candidate_score_norm"] = 1.0 if hi == lo else (raw - lo) / (hi - lo)
            out.append(item)
    return out


def select_rows(rows: list[dict[str, Any]], article_ids: list[str], strategy: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    score_key = str(params.get("score_key", "probability_positive"))
    selected: list[dict[str, Any]] = []
    grouped = rows_by_article(rows)
    for aid in article_ids:
        article_rows = grouped.get(str(aid), [])
        ordered = sorted(
            article_rows,
            key=lambda item: (-float(item.get(score_key, 0.0) or 0.0), int(item.get("candidate_rank", 10**9) or 10**9), str(item.get("event_name", ""))),
        )
        if strategy in {"original_threshold_0_93", "original_decoder_threshold", "top1_only", "top1_min_score", "top1_min_score_margin", "fusion_top1_min_score"}:
            if not ordered:
                continue
            top = ordered[0]
            top_score = float(top.get(score_key, 0.0) or 0.0)
            second_score = float(ordered[1].get(score_key, 0.0) or 0.0) if len(ordered) > 1 else -1.0
            min_score = float(params.get("min_score", 0.0))
            margin = float(params.get("margin", 0.0))
            if strategy == "top1_only":
                min_score = -1.0
                margin = 0.0
            if top_score >= min_score and top_score - second_score >= margin:
                item = dict(top)
                item["selection_score"] = top_score
                item["decision"] = True
                item["decode_rule"] = strategy
                selected.append(item)
        elif strategy == "global_threshold":
            threshold = float(params["threshold"])
            for row in ordered:
                if float(row.get(score_key, 0.0) or 0.0) >= threshold:
                    item = dict(row)
                    item["selection_score"] = float(row.get(score_key, 0.0) or 0.0)
                    item["decision"] = True
                    item["decode_rule"] = strategy
                    selected.append(item)
        elif strategy == "global_threshold_top1_fallback":
            threshold = float(params["threshold"])
            hits = []
            for row in ordered:
                if float(row.get(score_key, 0.0) or 0.0) >= threshold:
                    hits.append(row)
            if not hits and ordered:
                hits = [ordered[0]]
            for row in hits:
                item = dict(row)
                item["selection_score"] = float(row.get(score_key, 0.0) or 0.0)
                item["decision"] = True
                item["decode_rule"] = strategy
                selected.append(item)
        elif strategy == "per_class_threshold":
            thresholds = params["thresholds"]
            default = float(thresholds.get("__default__", params.get("default", 0.5)))
            for row in ordered:
                threshold = float(thresholds.get(str(row.get("event_name") or ""), default))
                if float(row.get(score_key, 0.0) or 0.0) >= threshold:
                    item = dict(row)
                    item["selection_score"] = float(row.get(score_key, 0.0) or 0.0)
                    item["decision"] = True
                    item["decode_rule"] = strategy
                    selected.append(item)
        elif strategy == "global_threshold_max_labels_2":
            threshold = float(params["threshold"])
            kept = [row for row in ordered if float(row.get(score_key, 0.0) or 0.0) >= threshold][:2]
            for row in kept:
                item = dict(row)
                item["selection_score"] = float(row.get(score_key, 0.0) or 0.0)
                item["decision"] = True
                item["decode_rule"] = strategy
                selected.append(item)
        else:
            raise ValueError(strategy)
    return selected


def metric_row(
    model_id: str,
    model_dir: str,
    label_space: str,
    strategy: str,
    params: dict[str, Any],
    article_ids: list[str],
    gold_pairs: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    event_names: list[str],
    candidate_source: str,
) -> dict[str, Any]:
    metrics = evaluate_event_predictions(article_ids, gold_pairs, selected, event_names, threshold=0.0)
    counts = [int(row["pred_count"]) for row in metrics["article_rows"]]
    return {
        "model_id": model_id,
        "model_dir": model_dir,
        "label_space": label_space,
        "candidate_source": candidate_source,
        "strategy": strategy,
        "strategy_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "micro_precision": round(float(metrics["micro_precision"]), 6),
        "micro_recall": round(float(metrics["micro_recall"]), 6),
        "micro_f1": round(float(metrics["micro_f1"]), 6),
        "macro_f1_fixed_64": round(float(metrics["macro_f1_fixed_64"]), 6),
        "macro_f1_observed": round(float(metrics["macro_f1_observed"]), 6),
        "weighted_f1": round(float(metrics["weighted_f1"]), 6),
        "article_exact_accuracy": round(float(metrics["article_exact_accuracy"]), 6),
        "at_least_one_correct_article_ratio": round(float(metrics["at_least_one_correct_article_ratio"]), 6),
        "avg_predicted_label_count": round(sum(counts) / len(counts), 6) if counts else 0.0,
        "pred_article_count_0": sum(1 for value in counts if value == 0),
        "pred_article_count_1": sum(1 for value in counts if value == 1),
        "pred_article_count_2": sum(1 for value in counts if value == 2),
        "pred_article_count_3plus": sum(1 for value in counts if value >= 3),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "predicted_event_count": metrics["predicted_event_count"],
        "gold_event_count": metrics["gold_event_count"],
    }


def strategy_grid(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    article_ids: list[str],
    gold_pairs: list[dict[str, Any]],
    event_names: list[str],
    support: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    thresholds = [round(i / 100.0, 2) for i in range(0, 100)]
    all_rows: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def add(strategy: str, params: dict[str, Any]) -> None:
        selected = select_rows(rows, article_ids, strategy, params)
        row = metric_row(
            spec["model_id"],
            spec["model_dir"],
            spec["label_space"],
            strategy,
            params,
            article_ids,
            gold_pairs,
            selected,
            event_names,
            "candidate_v2_top20",
        )
        all_rows.append(row)
        candidates.append((row, selected))

    add("original_threshold_0_93" if spec["model_id"] == "limited_round1_model_epoch_2" else "original_decoder_threshold", {"min_score": float(spec["original_threshold"])})
    add("top1_only", {})
    for threshold in thresholds:
        add("global_threshold", {"threshold": threshold})
        add("top1_min_score", {"min_score": threshold})
        add("global_threshold_top1_fallback", {"threshold": threshold})
        add("global_threshold_max_labels_2", {"threshold": threshold})
    for min_score in thresholds:
        for margin in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]:
            add("top1_min_score_margin", {"min_score": min_score, "margin": margin})

    global_best = max((row for row in all_rows if row["strategy"] == "global_threshold"), key=lambda row: (row["micro_f1"], row["micro_recall"], -row["avg_predicted_label_count"]))
    grid = [round(i / 100.0, 2) for i in range(1, 100)]
    per_class = derive_per_class_thresholds(article_ids, gold_pairs, rows, event_names, grid, support, min_support=5, default_threshold=float(json.loads(global_best["strategy_params"])["threshold"]))
    add("per_class_threshold", {"thresholds": per_class, "default": per_class.get("__default__", 0.5)})

    fused_rows = normalize_candidate_scores(rows)
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        augmented = []
        for row in fused_rows:
            item = dict(row)
            prob = float(row.get("probability_positive", 0.0) or 0.0)
            cand = float(row.get("candidate_score_norm", 0.0) or 0.0)
            item["fusion_score"] = alpha * prob + (1.0 - alpha) * cand
            augmented.append(item)
        for threshold in thresholds:
            selected = select_rows(augmented, article_ids, "fusion_top1_min_score", {"score_key": "fusion_score", "min_score": threshold, "alpha": alpha})
            row = metric_row(
                spec["model_id"],
                spec["model_dir"],
                spec["label_space"],
                "fusion_top1_min_score",
                {"score_key": "fusion_score", "min_score": threshold, "alpha": alpha},
                article_ids,
                gold_pairs,
                selected,
                event_names,
                "candidate_v2_top20",
            )
            all_rows.append(row)
            candidates.append((row, selected))

    best_row, best_selected = select_best_candidate(candidates)
    return all_rows, best_row, best_selected


def simplicity_priority(row: dict[str, Any]) -> int:
    order = {
        "original_threshold_0_93": 0,
        "original_decoder_threshold": 0,
        "top1_only": 1,
        "top1_min_score": 2,
        "global_threshold": 3,
        "top1_min_score_margin": 4,
        "global_threshold_max_labels_2": 5,
        "global_threshold_top1_fallback": 6,
        "per_class_threshold": 7,
        "fusion_top1_min_score": 8,
    }
    return order.get(str(row["strategy"]), 99)


def select_best_candidate(candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = [(row, selected) for row, selected in candidates if float(row["micro_recall"]) >= 0.60]
    pool = eligible or candidates
    max_f1 = max(float(row["micro_f1"]) for row, _ in pool)
    near = [(row, selected) for row, selected in pool if max_f1 - float(row["micro_f1"]) < 0.01]
    return sorted(
        near,
        key=lambda item: (
            float(item[0]["avg_predicted_label_count"]),
            simplicity_priority(item[0]),
            -float(item[0]["micro_precision"]),
            item[0]["model_id"],
            item[0]["strategy_params"],
        ),
    )[0]


def compare_with_history(reproduced: dict[str, Any]) -> dict[str, Any]:
    history_path = ROOT / "outputs/task1_limited_training_round1/round1_training_history.csv"
    if not history_path.exists():
        return {"history_found": False}
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epoch2 = next((row for row in rows if row.get("checkpoint_id") == "round1_epoch_2"), None)
    if not epoch2:
        return {"history_found": True, "round1_epoch_2_found": False}
    keys = ["micro_precision", "micro_recall", "micro_f1", "macro_f1_fixed_64", "article_exact_accuracy"]
    diffs = {key: round(float(reproduced[key]) - float(epoch2[key]), 9) for key in keys}
    return {
        "history_found": True,
        "round1_epoch_2_found": True,
        "history_metrics": {key: float(epoch2[key]) for key in keys},
        "reproduced_metrics": {key: reproduced[key] for key in keys},
        "diffs": diffs,
        "dev_reproduction_passed": all(abs(value) <= 1e-6 for value in diffs.values()),
    }


def decode_article_predictions(article_ids: list[str], pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = rows_by_article(pair_rows)
    out = []
    for aid in article_ids:
        selected = sorted(
            grouped.get(str(aid), []),
            key=lambda row: (-float(row.get("selection_score", row.get("probability_positive", 0.0)) or 0.0), str(row.get("event_name", ""))),
        )
        out.append(
            {
                "article_id": str(aid),
                "predicted_event_count": len(selected),
                "pred_event_names": "|".join(str(row.get("event_name") or "") for row in selected),
                "selected_events": [
                    {
                        "event_name": row.get("event_name", ""),
                        "schema_id": row.get("schema_id", ""),
                        "probability_positive": row.get("probability_positive", ""),
                        "selection_score": row.get("selection_score", row.get("probability_positive", "")),
                        "candidate_rank": row.get("candidate_rank", ""),
                        "candidate_source": row.get("candidate_source", ""),
                    }
                    for row in selected
                ],
                "processing_status": "success_with_events" if selected else "success_no_event",
            }
        )
    return out


def apply_selected_strategy_to_80(selection: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if selection["model_id"] != "limited_round1_model_epoch_2":
        raise RuntimeError(
            "selected model does not have a frozen 80-row score file in this Step 5.6-C run; "
            "run gold-free inference for that model before applying the decoder"
        )
    pair_path = STEP56B_DIR / "pair_predictions.jsonl"
    input_path = STEP56B_DIR / "gold_free_inputs.jsonl"
    for path in [pair_path, input_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    article_rows = read_jsonl(input_path)
    article_ids = [str(row["article_id"]) for row in article_rows]
    source_pairs = read_jsonl(pair_path)
    params = json.loads(selection["strategy_params"])
    strategy = str(selection["strategy"])
    if strategy == "original_threshold_0_93":
        strategy = "top1_min_score"
        params = {"min_score": 0.93}
    selected = select_rows(source_pairs, article_ids, strategy, params)
    for row in selected:
        row["decode_rule"] = selection["strategy"]
        row["decode_params"] = selection["strategy_params"]
        row["step5_6c_selected"] = True
    write_jsonl(output_dir / "frozen_80_pair_predictions.jsonl", selected)
    article_predictions = decode_article_predictions(article_ids, selected)
    write_jsonl(output_dir / "frozen_80_predictions.jsonl", article_predictions)
    return {
        "source_pair_predictions": rel(pair_path),
        "source_pair_predictions_sha256": sha256_file(pair_path),
        "gold_free_inputs": rel(input_path),
        "gold_free_inputs_sha256": sha256_file(input_path),
        "article_count": len(article_ids),
        "selected_event_count": len(selected),
        "frozen_80_predictions_sha256": sha256_file(output_dir / "frozen_80_predictions.jsonl"),
        "frozen_80_pair_predictions_sha256": sha256_file(output_dir / "frozen_80_pair_predictions.jsonl"),
        "gold_labels_read_in_prediction_phase": False,
    }


def verify_step56b_sha() -> dict[str, Any]:
    sha_path = STEP56B_DIR / "SHA256SUMS.txt"
    if not sha_path.exists():
        raise FileNotFoundError(sha_path)
    expected = {}
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        if "  " in line:
            digest, name = line.split("  ", 1)
            expected[name] = digest
    required = ["gold_free_inputs.jsonl", "candidates_top20.jsonl", "pair_predictions.jsonl", "article_predictions.jsonl", "run_manifest.json"]
    mismatches = []
    for name in required:
        path = STEP56B_DIR / name
        digest = sha256_file(path)
        if expected.get(name) != digest:
            mismatches.append({"file": name, "expected": expected.get(name), "actual": digest})
    if mismatches:
        raise RuntimeError(f"Step 5.6-B frozen SHA mismatch: {mismatches}")
    return {"required_files": required, "sha256": {name: expected[name] for name in required}, "mismatch_count": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--candidate-config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--backup-dir", type=Path, default=None)
    args = parser.parse_args()

    start = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "frozen_selection_manifest.json").exists() and not (output_dir / "evaluation_80.json").exists():
        raise FileExistsError("partial Step 5.6-C output exists; inspect before rerun to avoid mixed frozen selection")

    process = process_check()
    write_json(output_dir / "process_check.json", process)
    if process["status"] == "suspicious_processes_found":
        raise RuntimeError(f"suspicious Task1 processes found: {process['suspicious'][:5]}")

    protected_before = protected_snapshot(output_dir / "formal_protected_sha256_before.txt")
    step56b_sha = verify_step56b_sha()
    config = load_config(args.candidate_config)
    split = load_frozen_split(config)
    event_names = load_event_names()
    support = train_support(split)
    dev_ids = [str(aid) for aid in split["dev_ids"]]
    dev_pairs = split["dev_pairs"]
    candidate_all64 = read_jsonl(ROOT / "outputs/task1_step5_5_candidate_v2_dev/candidate_v2_dev_all64.jsonl")
    candidate_top20 = candidate_rows_for_top_k(candidate_all64, 20)
    candidate_recall = candidate_recall_at_k(dev_ids, dev_pairs, candidate_top20, event_names, [1, 3, 5, 10, 20])

    score_distributions = {}
    strategy_rows: list[dict[str, Any]] = []
    best_by_model: list[dict[str, Any]] = []
    best_selected_by_key: dict[str, list[dict[str, Any]]] = {}
    source_files = []
    latest_reproduced_metrics: dict[str, Any] | None = None

    for spec in model_specs():
        rows = load_model_rows(spec, dev_ids)
        source_files.append({"model_id": spec["model_id"], "score_path": spec["score_path"], "score_sha256": sha256_file(ROOT / spec["score_path"]), "row_count": len(rows)})
        score_distributions[spec["model_id"]] = score_distribution(rows, dev_ids, dev_pairs)
        model_strategy_rows, best_row, best_selected = strategy_grid(spec, rows, dev_ids, dev_pairs, event_names, support)
        strategy_rows.extend(model_strategy_rows)
        best_by_model.append(best_row)
        best_selected_by_key[f"{best_row['model_id']}::{best_row['strategy']}::{best_row['strategy_params']}"] = best_selected
        if spec["model_id"] == "limited_round1_model_epoch_2":
            latest_reproduced_metrics = best_row

    global_best, global_best_selected = select_best_candidate(
        [
            (row, best_selected_by_key[f"{row['model_id']}::{row['strategy']}::{row['strategy_params']}"])
            for row in best_by_model
        ]
    )
    if global_best["model_id"] != "limited_round1_model_epoch_2":
        # Keep the best row, but still avoid accidental 80 inference with the wrong model.
        global_best["requires_fresh_gold_free_80_inference"] = True
    else:
        global_best["requires_fresh_gold_free_80_inference"] = False

    latest_original_rows = load_model_rows(next(spec for spec in model_specs() if spec["model_id"] == "limited_round1_model_epoch_2"), dev_ids)
    latest_original_selected = select_rows(latest_original_rows, dev_ids, "top1_min_score", {"min_score": 0.93})
    latest_original_metrics = metric_row(
        "limited_round1_model_epoch_2",
        "outputs/task1_limited_training_round1/model_epoch_2",
        "binary_pair",
        "candidate_v2_top20_top1_min_score",
        {"min_score": 0.93},
        dev_ids,
        dev_pairs,
        latest_original_selected,
        event_names,
        "training_time_candidate_v2_top20",
    )
    history_reproduction = compare_with_history(latest_original_metrics)
    dev_reproduction = {
        "generated_at": now(),
        "status": "dev_reproduced" if history_reproduction.get("dev_reproduction_passed") else "dev_reproduction_failed",
        "source": "frozen training-time dev score file; no model retraining and no 80-label use",
        "training_time_original_candidate_source": "candidate_v2_top20_hard_negatives",
        "training_time_input_format": "src.data.event_match_text.build_input_text",
        "candidate_v2_top20_reproduction_uses_same_candidate_set": True,
        "model_epoch_2_score_path": "outputs/task1_limited_training_round1/round1_epoch_2_dev_scores.jsonl",
        "model_epoch_2_score_sha256": sha256_file(ROOT / "outputs/task1_limited_training_round1/round1_epoch_2_dev_scores.jsonl"),
        "historical_compare": history_reproduction,
        "reproduced_metrics": latest_original_metrics,
        "candidate_recall": candidate_recall,
        "threshold_0_93_semantics": {
            "rule": "top1_min_score",
            "meaning": "rank each article's Candidate-v2 Top20 candidates by model positive probability and emit only the top candidate if its probability is >= 0.93",
            "not_a_full_multilabel_threshold": True,
            "not_a_candidate_rank_threshold": True,
            "pair_probability_transform": "softmax(logits)[:, 1]",
        },
    }
    write_json(output_dir / "dev_reproduction.json", dev_reproduction)
    write_json(output_dir / "dev_score_distribution.json", {"generated_at": now(), "models": score_distributions})
    write_json(
        output_dir / "dev_strategy_comparison.json",
        {
            "generated_at": now(),
            "selection_rule": "exclude recall < 0.60; maximize micro-F1; if within 0.01 choose lower avg predicted labels; then simpler rule",
            "model_best_rows": best_by_model,
            "selected": global_best,
            "all_strategy_rows": strategy_rows,
        },
    )
    write_csv(output_dir / "dev_strategy_comparison.csv", strategy_rows)

    selected_80 = apply_selected_strategy_to_80(global_best, output_dir)
    protected_after = protected_snapshot(output_dir / "formal_protected_sha256_after.txt")

    manifest = {
        "generated_at": now(),
        "stage": "Task1 Step 5.6-C dev-only calibration",
        "status": "selection_frozen_prediction_frozen",
        "gold_labels_read_in_calibration": False,
        "gold_labels_read_in_80_prediction_phase": False,
        "gold_test_labels_read": False,
        "train_or_finetune_executed": False,
        "candidate_v2_modified": False,
        "model_modified": False,
        "formal_output_modified": protected_before != protected_after,
        "backup_dir": str(args.backup_dir) if args.backup_dir else "",
        "step5_6b_frozen_sha_verified": step56b_sha,
        "split_checks": split["checks"],
        "event_class_count": len(event_names),
        "candidate_recall_dev": candidate_recall,
        "source_score_files": source_files,
        "selected_model_strategy": global_best,
        "selected_80_prediction": selected_80,
        "model_epoch_2_sha256": sha256_file(LATEST_MODEL_DIR / "model.safetensors"),
        "expected_model_epoch_2_sha256": LATEST_MODEL_SHA,
        "candidate_v2": {
            "code": "src/routing/task1_candidate_v2.py",
            "code_sha256": sha256_file(ROOT / "src/routing/task1_candidate_v2.py"),
            "config": rel(args.candidate_config),
            "config_sha256": sha256_file(args.candidate_config),
            "top_k": 20,
        },
        "contract_comparison": rel(CONTRACT_JSON) if CONTRACT_JSON.exists() else "",
        "runtime_seconds": round(time.time() - start, 3),
        "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
    }
    write_json(output_dir / "frozen_selection_manifest.json", manifest)
    write_sha_manifest(output_dir)
    print(json.dumps({"status": manifest["status"], "selected_model": global_best["model_id"], "selected_strategy": global_best["strategy"], "runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
