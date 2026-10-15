#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 article-level multi-label metrics for the 64 event-name taxonomy."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


def f1_from_counts(tp: int, pred: int, gold: int) -> tuple[float, float, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def gold_event_sets(gold_pairs: Iterable[dict[str, Any]], article_ids: Iterable[str]) -> dict[str, set[str]]:
    article_set = {str(aid) for aid in article_ids}
    by_article: dict[str, set[str]] = {aid: set() for aid in article_set}
    for row in gold_pairs:
        aid = str(row.get("article_id", ""))
        if aid in article_set:
            name = str(row.get("event_name") or "")
            if name:
                by_article.setdefault(aid, set()).add(name)
    return by_article


def prediction_event_sets(
    prediction_rows: Iterable[dict[str, Any]],
    article_ids: Iterable[str],
    threshold: float | dict[str, float],
    top_k: int | None = None,
) -> dict[str, set[str]]:
    rows_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    article_list = [str(aid) for aid in article_ids]
    for row in prediction_rows:
        aid = str(row.get("article_id", ""))
        if aid:
            rows_by_article[aid].append(row)
    pred: dict[str, set[str]] = {aid: set() for aid in article_list}
    for aid in article_list:
        rows = sorted(
            rows_by_article.get(aid, []),
            key=lambda item: (
                -float(item.get("probability_positive", item.get("candidate_score", 0.0)) or 0.0),
                int(item.get("candidate_rank", 10**9) or 10**9),
                str(item.get("event_name", "")),
            ),
        )
        if top_k is not None:
            rows = rows[:top_k]
        for row in rows:
            event_name = str(row.get("event_name") or "")
            if not event_name:
                continue
            score = float(row.get("probability_positive", row.get("candidate_score", 0.0)) or 0.0)
            if isinstance(threshold, dict):
                th = float(threshold.get(event_name, threshold.get("__default__", 0.5)))
            else:
                th = float(threshold)
            if score >= th:
                pred[aid].add(event_name)
    return pred


def evaluate_event_predictions(
    article_ids: list[str],
    gold_pairs: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    fixed_event_names: list[str],
    threshold: float | dict[str, float],
    top_k: int | None = None,
) -> dict[str, Any]:
    """Compute article-level multi-label metrics over fixed event-name classes."""
    fixed = sorted({str(name) for name in fixed_event_names})
    gold_by_article = gold_event_sets(gold_pairs, article_ids)
    pred_by_article = prediction_event_sets(prediction_rows, article_ids, threshold, top_k=top_k)

    class_tp = {name: 0 for name in fixed}
    class_gold = {name: 0 for name in fixed}
    class_pred = {name: 0 for name in fixed}
    article_rows: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    tp_total = 0
    pred_total = 0
    gold_total = 0
    exact_count = 0
    at_least_one_count = 0

    pred_rows_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        pred_rows_by_article[str(row.get("article_id", ""))].append(row)

    for aid in article_ids:
        gold_set = set(gold_by_article.get(str(aid), set()))
        pred_set = set(pred_by_article.get(str(aid), set()))
        tp_set = gold_set & pred_set
        tp_total += len(tp_set)
        pred_total += len(pred_set)
        gold_total += len(gold_set)
        exact = gold_set == pred_set
        exact_count += int(exact)
        at_least_one_count += int(bool(tp_set))
        for event_name in gold_set:
            if event_name in class_gold:
                class_gold[event_name] += 1
        for event_name in pred_set:
            if event_name in class_pred:
                class_pred[event_name] += 1
        for event_name in tp_set:
            if event_name in class_tp:
                class_tp[event_name] += 1

        top_pred = "__NO_PRED__"
        eligible_rows = []
        for row in pred_rows_by_article.get(str(aid), []):
            event_name = str(row.get("event_name") or "")
            if not event_name:
                continue
            score = float(row.get("probability_positive", row.get("candidate_score", 0.0)) or 0.0)
            if isinstance(threshold, dict):
                th = float(threshold.get(event_name, threshold.get("__default__", 0.5)))
            else:
                th = float(threshold)
            if score >= th:
                eligible_rows.append(row)
        if eligible_rows:
            best = max(
                eligible_rows,
                key=lambda item: (
                    float(item.get("probability_positive", item.get("candidate_score", 0.0)) or 0.0),
                    -int(item.get("candidate_rank", 10**9) or 10**9),
                ),
            )
            top_pred = str(best.get("event_name") or "__NO_PRED__")
        for event_name in sorted(gold_set):
            confusion[event_name][event_name if event_name in pred_set else top_pred] += 1

        article_rows.append(
            {
                "article_id": aid,
                "gold_event_names": "|".join(sorted(gold_set)),
                "pred_event_names": "|".join(sorted(pred_set)),
                "gold_count": len(gold_set),
                "pred_count": len(pred_set),
                "tp": len(tp_set),
                "fp": max(0, len(pred_set) - len(tp_set)),
                "fn": max(0, len(gold_set) - len(tp_set)),
                "exact_match": exact,
                "at_least_one_correct": bool(tp_set),
            }
        )

    precision, recall, micro_f1 = f1_from_counts(tp_total, pred_total, gold_total)
    class_rows = []
    fixed_f1: list[float] = []
    observed_f1: list[float] = []
    weighted_terms: list[tuple[float, int]] = []
    for event_name in fixed:
        tp = class_tp[event_name]
        pred = class_pred[event_name]
        gold = class_gold[event_name]
        p, r, f1 = f1_from_counts(tp, pred, gold)
        fixed_f1.append(f1)
        if gold or pred:
            observed_f1.append(f1)
        weighted_terms.append((f1, gold))
        class_rows.append(
            {
                "event_name": event_name,
                "gold_support": gold,
                "prediction_support": pred,
                "tp": tp,
                "fp": max(0, pred - tp),
                "fn": max(0, gold - tp),
                "precision": round(p, 6),
                "recall": round(r, 6),
                "f1": round(f1, 6),
            }
        )

    total_gold = sum(class_gold.values())
    return {
        "threshold": threshold,
        "top_k": top_k,
        "tp": tp_total,
        "fp": max(0, pred_total - tp_total),
        "fn": max(0, gold_total - tp_total),
        "predicted_event_count": pred_total,
        "gold_event_count": gold_total,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "macro_f1_fixed_64": sum(fixed_f1) / len(fixed_f1) if fixed_f1 else 0.0,
        "macro_f1_observed": sum(observed_f1) / len(observed_f1) if observed_f1 else 0.0,
        "weighted_f1": sum(f * w for f, w in weighted_terms) / total_gold if total_gold else 0.0,
        "article_exact_accuracy": exact_count / len(article_ids) if article_ids else 0.0,
        "at_least_one_correct_article_ratio": at_least_one_count / len(article_ids) if article_ids else 0.0,
        "class_rows": class_rows,
        "article_rows": article_rows,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def sweep_global_thresholds(
    article_ids: list[str],
    gold_pairs: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    fixed_event_names: list[str],
    thresholds: Iterable[float],
    top_k: int | None = None,
    primary: str = "macro_f1_fixed_64",
    secondary: str = "micro_f1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = evaluate_event_predictions(
            article_ids=article_ids,
            gold_pairs=gold_pairs,
            prediction_rows=prediction_rows,
            fixed_event_names=fixed_event_names,
            threshold=float(threshold),
            top_k=top_k,
        )
        row = {
            "threshold": round(float(threshold), 6),
            "top_k": top_k if top_k is not None else "all",
            "micro_precision": round(metrics["micro_precision"], 6),
            "micro_recall": round(metrics["micro_recall"], 6),
            "micro_f1": round(metrics["micro_f1"], 6),
            "macro_f1_fixed_64": round(metrics["macro_f1_fixed_64"], 6),
            "macro_f1_observed": round(metrics["macro_f1_observed"], 6),
            "weighted_f1": round(metrics["weighted_f1"], 6),
            "article_exact_accuracy": round(metrics["article_exact_accuracy"], 6),
            "at_least_one_correct_article_ratio": round(metrics["at_least_one_correct_article_ratio"], 6),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
        rows.append(row)
        if best is None or (
            row[primary],
            row[secondary],
            row["micro_recall"],
            -abs(float(row["threshold"]) - 0.5),
        ) > (
            best[primary],
            best[secondary],
            best["micro_recall"],
            -abs(float(best["threshold"]) - 0.5),
        ):
            best = row
    return rows, best or {}


def derive_per_class_thresholds(
    article_ids: list[str],
    gold_pairs: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    fixed_event_names: list[str],
    thresholds: Iterable[float],
    train_support: dict[str, int],
    min_support: int,
    default_threshold: float,
) -> dict[str, float]:
    gold_by_article = gold_event_sets(gold_pairs, article_ids)
    rows_by_event: dict[str, list[tuple[int, float]]] = {name: [] for name in fixed_event_names}
    for row in prediction_rows:
        aid = str(row.get("article_id", ""))
        event_name = str(row.get("event_name") or "")
        if event_name in rows_by_event:
            label = int(event_name in gold_by_article.get(aid, set()))
            score = float(row.get("probability_positive", row.get("candidate_score", 0.0)) or 0.0)
            rows_by_event[event_name].append((label, score))
    out = {"__default__": float(default_threshold)}
    for event_name in fixed_event_names:
        if int(train_support.get(event_name, 0)) < min_support:
            out[event_name] = float(default_threshold)
            continue
        best_threshold = float(default_threshold)
        best_tuple = (-1.0, -1.0, -1.0)
        pairs = rows_by_event.get(event_name, [])
        for threshold in thresholds:
            tp = sum(1 for label, score in pairs if label and score >= threshold)
            pred = sum(1 for _, score in pairs if score >= threshold)
            gold = sum(1 for label, _ in pairs if label)
            p, r, f1 = f1_from_counts(tp, pred, gold)
            key = (f1, r, p)
            if key > best_tuple:
                best_tuple = key
                best_threshold = float(threshold)
        out[event_name] = best_threshold
    return out


def candidate_recall_at_k(
    article_ids: list[str],
    gold_pairs: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    fixed_event_names: list[str],
    ks: Iterable[int],
) -> dict[str, Any]:
    rows_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_article[str(row.get("article_id", ""))].append(row)
    gold_by_article = gold_event_sets(gold_pairs, article_ids)
    ks_sorted = sorted(int(k) for k in ks)
    hits = Counter()
    total = 0
    class_total = Counter()
    class_hits = {k: Counter() for k in ks_sorted}
    missing_rows = []
    for aid in article_ids:
        ranked = sorted(
            rows_by_article.get(str(aid), []),
            key=lambda item: (
                int(item.get("candidate_rank", 10**9) or 10**9),
                -float(item.get("candidate_score", 0.0) or 0.0),
                str(item.get("event_name", "")),
            ),
        )
        ranks = {str(row.get("event_name")): idx for idx, row in enumerate(ranked, 1)}
        for event_name in sorted(gold_by_article.get(str(aid), set())):
            total += 1
            class_total[event_name] += 1
            rank = ranks.get(event_name)
            for k in ks_sorted:
                if rank is not None and rank <= k:
                    hits[k] += 1
                    class_hits[k][event_name] += 1
            if rank is None:
                reason = "candidate_missing"
            elif rank > max(ks_sorted):
                reason = f"rank_gt_{max(ks_sorted)}"
            else:
                reason = "candidate_present"
            if reason != "candidate_present":
                missing_rows.append(
                    {
                        "article_id": aid,
                        "event_name": event_name,
                        "candidate_rank": rank if rank is not None else "",
                        "fn_reason": reason,
                    }
                )
    metrics = {
        "gold_event_count": total,
        **{f"candidate_recall_at_{k}": hits[k] / total if total else 0.0 for k in ks_sorted},
        "candidate_missing_at_max_k": total - hits[max(ks_sorted)] if total else 0,
        "missing_rows": missing_rows,
    }
    class_rows = []
    for event_name in sorted(fixed_event_names):
        row = {"event_name": event_name, "gold_support": class_total[event_name]}
        for k in ks_sorted:
            row[f"candidate_hits_at_{k}"] = class_hits[k][event_name]
            row[f"candidate_recall_at_{k}"] = (
                class_hits[k][event_name] / class_total[event_name] if class_total[event_name] else 0.0
            )
        class_rows.append(row)
    metrics["class_rows"] = class_rows
    return metrics
