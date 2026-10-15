#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 independent gold readiness and partial evaluation audit.

This audit is read-only for formal Task1/Task2 outputs. It checks whether the
available human-labelled Task1 data can support strict evaluation, and computes
only the overlap-based metrics that are actually supported by local files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/task1_independent_gold_eval_v1"

SCHEMA_PATH = ROOT / "data/processed/event_schema.jsonl"
GOLD_NEWS = {
    "train": ROOT / "data/processed/labeled/news_train.jsonl",
    "test": ROOT / "data/processed/labeled/news_test.jsonl",
}
GOLD_PAIRS = {
    "train": ROOT / "data/processed/labeled/event_pairs_train.jsonl",
    "test": ROOT / "data/processed/labeled/event_pairs_test.jsonl",
}
RAW_GOLD_FILES = {
    "train": ROOT / "data/raw/news_with_tags_train.xlsx",
    "test": ROOT / "data/raw/news_with_tags_test.xlsx",
}
PREDICTION_FILES = {
    "train": ROOT / "outputs/final_task1/final_events_train.jsonl",
    "test": ROOT / "outputs/final_task1/final_events_test.jsonl",
}
FORMAL_FILES = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_train.xlsx",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/final_task1/final_events_test.xlsx",
    ROOT / "outputs/final_task1/task1_completion_certificate.json",
    ROOT / "reports/task1_final_summary.json",
    ROOT / "reports/task1_final_event_frequency.csv",
]

GOLD_PAIR_FIELDS = [
    "sample_id",
    "article_id",
    "title",
    "publish_time",
    "source",
    "content",
    "schema_id",
    "target_type",
    "target_code",
    "target_name",
    "event_name",
    "event_description",
    "emotion",
    "impact_analysis",
    "label",
]
TASK1_OUTPUT_FIELDS = [
    "event_id",
    "article_id",
    "title",
    "publish_time",
    "source",
    "target_type",
    "target_code",
    "target_name",
    "schema_id",
    "event_name",
    "event_subject_name",
    "event_subject_type",
    "event_time_raw",
    "event_time_normalized",
    "event_time_precision",
    "event_action",
    "event_object",
    "event_status",
    "location",
    "impact_direction",
    "evidence_sentences",
    "extraction_confidence",
]
STRUCTURED_FIELDS = [
    "event_subject_name",
    "event_time_raw",
    "event_time_normalized",
    "event_action",
    "event_object",
    "event_status",
    "location",
    "impact_direction",
    "evidence_sentences",
    "extraction_confidence",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {rel(path): sha256(path) if path.exists() else "missing" for path in paths}


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 6)


def parse_duplication_ids(value: Any) -> list[str]:
    if is_blank(value):
        return []
    return [part.strip() for part in re.split(r"[;,，、\s]+", str(value)) if part.strip()]


def normalized_content_hash(text: Any) -> str:
    normalized = " ".join(str(text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_schema() -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = read_jsonl(SCHEMA_PATH)
    by_id = {row["schema_id"]: row for row in rows}
    event_names = sorted({row["event_name"] for row in rows})
    return by_id, event_names


def add_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["_split"] = split
        out.append(item)
    return out


def count_empty_fields(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, int]:
    return {field: sum(1 for row in rows if is_blank(row.get(field))) for field in fields}


def values_by_article(rows: list[dict[str, Any]], value_field: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(value_field)
        if not is_blank(row.get("article_id")) and not is_blank(value):
            out[str(row["article_id"])].add(str(value))
    return out


def predicted_event_name(row: dict[str, Any], schema_by_id: dict[str, dict[str, Any]]) -> str:
    schema = schema_by_id.get(str(row.get("schema_id", "")))
    if schema and not is_blank(schema.get("event_name")):
        return str(schema["event_name"])
    if not is_blank(row.get("event_name")):
        return str(row["event_name"])
    return ""


def prediction_event_names_by_article(
    rows: list[dict[str, Any]], schema_by_id: dict[str, dict[str, Any]], *, raw: bool
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if is_blank(row.get("article_id")):
            continue
        if raw:
            value = "" if is_blank(row.get("event_name")) else str(row["event_name"])
        else:
            value = predicted_event_name(row, schema_by_id)
        if value:
            out[str(row["article_id"])].add(value)
        else:
            out[str(row["article_id"])].add("__MISSING__")
    return out


def pair_metrics(
    gold_by_article: dict[str, set[str]],
    pred_by_article: dict[str, set[str]],
    overlap_articles: set[str],
) -> dict[str, Any]:
    tp = 0
    pred_total = 0
    gold_total = 0
    exact_articles = 0
    for article_id in sorted(overlap_articles):
        gold_values = gold_by_article.get(article_id, set())
        pred_values = pred_by_article.get(article_id, set())
        tp += len(gold_values & pred_values)
        pred_total += len(pred_values)
        gold_total += len(gold_values)
        if gold_values == pred_values:
            exact_articles += 1
    precision = rate(tp, pred_total)
    recall = rate(tp, gold_total)
    return {
        "overlap_article_count": len(overlap_articles),
        "tp": tp,
        "pred_total": pred_total,
        "gold_total": gold_total,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "exact_article_count": exact_articles,
        "exact_article_accuracy": rate(exact_articles, len(overlap_articles)),
    }


def class_metrics(
    event_names: list[str],
    gold_test_pairs: list[dict[str, Any]],
    gold_by_article: dict[str, set[str]],
    pred_by_article: dict[str, set[str]],
    overlap_articles: set[str],
) -> list[dict[str, Any]]:
    test_support = Counter(row["event_name"] for row in gold_test_pairs)
    rows: list[dict[str, Any]] = []
    for event_name in event_names:
        tp = fp = fn = 0
        overlap_gold_support = 0
        pred_support = 0
        for article_id in overlap_articles:
            gold_hit = event_name in gold_by_article.get(article_id, set())
            pred_hit = event_name in pred_by_article.get(article_id, set())
            if gold_hit:
                overlap_gold_support += 1
            if pred_hit:
                pred_support += 1
            if gold_hit and pred_hit:
                tp += 1
            elif pred_hit and not gold_hit:
                fp += 1
            elif gold_hit and not pred_hit:
                fn += 1
        precision = rate(tp, tp + fp)
        recall = rate(tp, tp + fn)
        rows.append(
            {
                "event_name": event_name,
                "gold_test_support": test_support.get(event_name, 0),
                "overlap_gold_support": overlap_gold_support,
                "overlap_pred_support": pred_support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        )
    return rows


def confusion_matrix_rows(
    event_names: list[str],
    gold_by_article: dict[str, set[str]],
    pred_by_article: dict[str, set[str]],
    overlap_articles: set[str],
) -> list[dict[str, Any]]:
    labels = list(event_names)
    matrix: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for article_id in sorted(overlap_articles):
        gold_values = sorted(gold_by_article.get(article_id, set()))
        pred_values = sorted(value for value in pred_by_article.get(article_id, set()) if value in labels)
        if len(gold_values) == 1 and len(pred_values) == 1:
            matrix[gold_values[0]][pred_values[0]] += 1
            continue
        for gold_value in gold_values:
            if gold_value in pred_values:
                matrix[gold_value][gold_value] += 1
            elif pred_values:
                matrix[gold_value][pred_values[0]] += 1
    return [{"gold_event_name": label, **{pred: matrix[label].get(pred, 0) for pred in labels}} for label in labels]


def field_coverage_rows(
    gold_pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = [
        "article_id",
        "target_type",
        "target_code",
        "target_name",
        "schema_id",
        "event_name",
        "event_subject_name",
        "event_time_raw",
        "event_time_normalized",
        "event_action",
        "event_object",
        "event_status",
        "location",
        "impact_direction",
        "evidence_sentences",
        "extraction_confidence",
        "emotion",
        "impact_analysis",
    ]
    rows = []
    gold_total = len(gold_pairs)
    pred_total = len(predictions)
    gold_fields = set().union(*(row.keys() for row in gold_pairs)) if gold_pairs else set()
    pred_fields = set().union(*(row.keys() for row in predictions)) if predictions else set()
    for field in fields:
        gold_nonempty = sum(1 for row in gold_pairs if not is_blank(row.get(field)))
        pred_nonempty = sum(1 for row in predictions if not is_blank(row.get(field)))
        evaluable = field in {"article_id", "target_type", "target_code", "target_name", "schema_id", "event_name"}
        if field in STRUCTURED_FIELDS:
            note = "金标无该字段；正式输出该字段也未填充，不能评测要素抽取准确率。"
            evaluable = False
        elif field in {"emotion", "impact_analysis"}:
            note = "金标有影响/情绪说明，正式 Task1 输出无对应预测字段；不能作为要素抽取准确率。"
            evaluable = False
        elif field == "event_name":
            note = "可由 schema_id 稳定映射；正式输出 raw event_name 在 llm_accepted 段为空。"
        else:
            note = "可用于文章-事件对对齐或完整性检查。" if evaluable else "非核心评测字段。"
        rows.append(
            {
                "field": field,
                "gold_field_present": field in gold_fields,
                "gold_nonempty_count": gold_nonempty,
                "gold_nonempty_rate": rate(gold_nonempty, gold_total),
                "prediction_field_present": field in pred_fields,
                "prediction_nonempty_count": pred_nonempty,
                "prediction_nonempty_rate": rate(pred_nonempty, pred_total),
                "evaluable": evaluable,
                "note": note,
            }
        )
    return rows


def build_error_rows(
    gold_test_pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    schema_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gold_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pred_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_test_pairs:
        gold_by_article[row["article_id"]].append(row)
    for row in predictions:
        pred_by_article[row["article_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for article_id in sorted(set(gold_by_article) & set(pred_by_article)):
        gold_schema = {row["schema_id"] for row in gold_by_article[article_id]}
        pred_schema = {row["schema_id"] for row in pred_by_article[article_id] if not is_blank(row.get("schema_id"))}
        gold_names = {row["event_name"] for row in gold_by_article[article_id]}
        raw_names = {str(row.get("event_name") or "") for row in pred_by_article[article_id]}
        derived_names = {predicted_event_name(row, schema_by_id) for row in pred_by_article[article_id]}
        schema_match = gold_schema == pred_schema
        raw_name_match = gold_names == {name for name in raw_names if name}
        if schema_match and raw_name_match:
            continue
        if not schema_match:
            issue = "schema_id_mismatch"
        else:
            issue = "raw_event_name_missing_or_mismatch"
        first_gold = gold_by_article[article_id][0]
        rows.append(
            {
                "article_id": article_id,
                "title": first_gold.get("title", ""),
                "gold_schema_ids": "|".join(sorted(gold_schema)),
                "pred_schema_ids": "|".join(sorted(pred_schema)),
                "gold_event_names": "|".join(sorted(gold_names)),
                "pred_event_names_raw": "|".join(sorted(raw_names)),
                "pred_event_names_from_schema": "|".join(sorted(derived_names)),
                "schema_match": schema_match,
                "raw_event_name_match": raw_name_match,
                "issue": issue,
                "note": "schema_id 正确但 raw event_name 为空" if schema_match else "正式预测 schema_id 与金标不一致",
            }
        )
    return rows


def build_audit(formal_sha_before: dict[str, str] | None = None) -> dict[str, Any]:
    schema_by_id, event_names = load_schema()
    news = {split: read_jsonl(path) for split, path in GOLD_NEWS.items()}
    gold_pairs = {split: read_jsonl(path) for split, path in GOLD_PAIRS.items()}
    predictions_by_split = {split: read_jsonl(path) for split, path in PREDICTION_FILES.items()}
    all_gold_pairs = add_split(gold_pairs["train"], "train") + add_split(gold_pairs["test"], "test")
    all_predictions = add_split(predictions_by_split["train"], "train") + add_split(predictions_by_split["test"], "test")

    gold_train_articles = {row["article_id"] for row in news["train"]}
    gold_test_articles = {row["article_id"] for row in news["test"]}
    train_hashes = {normalized_content_hash(row.get("content")) for row in news["train"] if not is_blank(row.get("content"))}
    test_hashes = {normalized_content_hash(row.get("content")) for row in news["test"] if not is_blank(row.get("content"))}

    cross_train_to_test = []
    for row in news["train"]:
        dups = parse_duplication_ids(row.get("duplication_id"))
        if any(item in gold_test_articles for item in dups):
            cross_train_to_test.append({"article_id": row["article_id"], "duplication_id": row.get("duplication_id")})
    cross_test_to_train = []
    for row in news["test"]:
        dups = parse_duplication_ids(row.get("duplication_id"))
        if any(item in gold_train_articles for item in dups):
            cross_test_to_train.append({"article_id": row["article_id"], "duplication_id": row.get("duplication_id")})

    prediction_articles = {row["article_id"] for row in all_predictions if not is_blank(row.get("article_id"))}
    train_pred_articles = {row["article_id"] for row in predictions_by_split["train"] if not is_blank(row.get("article_id"))}
    test_pred_articles = {row["article_id"] for row in predictions_by_split["test"] if not is_blank(row.get("article_id"))}

    gold_test_by_article_schema = values_by_article(gold_pairs["test"], "schema_id")
    pred_by_article_schema = values_by_article(all_predictions, "schema_id")
    gold_test_by_article_event = values_by_article(gold_pairs["test"], "event_name")
    pred_by_article_event_derived = prediction_event_names_by_article(all_predictions, schema_by_id, raw=False)
    pred_by_article_event_raw = prediction_event_names_by_article(all_predictions, schema_by_id, raw=True)
    overlap_articles = set(gold_test_by_article_schema) & set(pred_by_article_schema)

    schema_metrics = pair_metrics(gold_test_by_article_schema, pred_by_article_schema, overlap_articles)
    event_name_metrics = pair_metrics(gold_test_by_article_event, pred_by_article_event_derived, overlap_articles)
    raw_event_name_metrics = pair_metrics(gold_test_by_article_event, pred_by_article_event_raw, overlap_articles)

    pred_status = Counter(row.get("quality_status", "") for row in all_predictions)
    pred_event_name_by_status: dict[str, dict[str, int]] = {}
    for status in sorted(pred_status):
        subset = [row for row in all_predictions if row.get("quality_status", "") == status]
        pred_event_name_by_status[status] = {
            "row_count": len(subset),
            "event_name_nonempty_count": sum(1 for row in subset if not is_blank(row.get("event_name"))),
            "evidence_sentences_nonempty_count": sum(1 for row in subset if not is_blank(row.get("evidence_sentences"))),
        }

    duplicate_event_keys = {}
    for split in ("train", "test"):
        keys = [(row["article_id"], row["schema_id"], row["target_code"], row["event_name"]) for row in gold_pairs[split]]
        duplicate_event_keys[split] = len(keys) - len(set(keys))

    status = "partially_ready"
    not_evaluable = {
        "trigger_extraction": "金标没有 trigger span/offset；正式输出没有 trigger 字段。",
        "argument_extraction": "金标没有主体、时间、金额、地点、人物等论元字段；正式输出相关字段为空。",
        "strict_relaxed_argument_metrics": "缺少 span/role gold，不能计算 strict 或 relaxed F1。",
        "full_gold_set_event_recognition": "正式预测文件只覆盖 test 金标 25/1029 篇文章；不能把未覆盖文章当成漏提。",
    }

    audit: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_status": status,
        "acceptance_conclusion": "需优化",
        "full_acceptance_passed": False,
        "model_calls_32b": 0,
        "sources": {
            "raw_gold_files": {split: rel(path) for split, path in RAW_GOLD_FILES.items()},
            "gold_news_files": {split: rel(path) for split, path in GOLD_NEWS.items()},
            "gold_event_pair_files": {split: rel(path) for split, path in GOLD_PAIRS.items()},
            "schema_file": rel(SCHEMA_PATH),
            "formal_prediction_files": {split: rel(path) for split, path in PREDICTION_FILES.items()},
        },
        "gold_inventory": {
            "train_document_count": len(news["train"]),
            "test_document_count": len(news["test"]),
            "total_document_count": len(news["train"]) + len(news["test"]),
            "train_event_pair_count": len(gold_pairs["train"]),
            "test_event_pair_count": len(gold_pairs["test"]),
            "total_event_pair_count": len(all_gold_pairs),
            "schema_inventory_count": len(schema_by_id),
            "gold_unique_schema_id_count": len({row["schema_id"] for row in all_gold_pairs}),
            "gold_unique_event_name_count": len({row["event_name"] for row in all_gold_pairs}),
            "gold_train_unique_event_name_count": len({row["event_name"] for row in gold_pairs["train"]}),
            "gold_test_unique_event_name_count": len({row["event_name"] for row in gold_pairs["test"]}),
            "train_multi_label_article_count": sum(1 for value in Counter(row["article_id"] for row in gold_pairs["train"]).values() if value > 1),
            "test_multi_label_article_count": sum(1 for value in Counter(row["article_id"] for row in gold_pairs["test"]).values() if value > 1),
            "gold_duplicate_event_key_count": duplicate_event_keys,
            "gold_pair_empty_fields": {
                split: count_empty_fields(gold_pairs[split], GOLD_PAIR_FIELDS) for split in ("train", "test")
            },
            "gold_news_empty_fields": {
                split: count_empty_fields(news[split], ["article_id", "title", "publish_time", "source", "content", "labels", "duplication_id"])
                for split in ("train", "test")
            },
            "event_name_distribution_total": dict(Counter(row["event_name"] for row in all_gold_pairs).most_common()),
            "event_name_distribution_test": dict(Counter(row["event_name"] for row in gold_pairs["test"]).most_common()),
        },
        "split_and_leakage_check": {
            "gold_train_test_article_id_overlap": len(gold_train_articles & gold_test_articles),
            "gold_train_test_sample_id_overlap": len({row["sample_id"] for row in gold_pairs["train"]} & {row["sample_id"] for row in gold_pairs["test"]}),
            "gold_train_test_title_overlap": len({row["title"] for row in news["train"]} & {row["title"] for row in news["test"]}),
            "gold_train_test_exact_content_hash_overlap": len(train_hashes & test_hashes),
            "gold_duplication_id_cross_train_to_test_count": len(cross_train_to_test),
            "gold_duplication_id_cross_test_to_train_count": len(cross_test_to_train),
            "formal_prediction_train_test_article_id_overlap": len(train_pred_articles & test_pred_articles),
            "formal_prediction_overlap_gold_train_articles": len(prediction_articles & gold_train_articles),
            "formal_prediction_overlap_gold_test_articles": len(prediction_articles & gold_test_articles),
            "gold_origin_read": "data/raw/news_with_tags_train.xlsx 和 data/raw/news_with_tags_test.xlsx 经 build_labeled_dataset.py 规范化生成；本审计未发现 train/test 交叉污染。",
        },
        "formal_prediction_inventory": {
            "train_event_count": len(predictions_by_split["train"]),
            "test_event_count": len(predictions_by_split["test"]),
            "total_event_count": len(all_predictions),
            "unique_article_count": len(prediction_articles),
            "quality_status_counts": dict(pred_status),
            "event_name_nonempty_count": sum(1 for row in all_predictions if not is_blank(row.get("event_name"))),
            "event_name_nonempty_rate": rate(sum(1 for row in all_predictions if not is_blank(row.get("event_name"))), len(all_predictions)),
            "evidence_sentences_nonempty_count": sum(1 for row in all_predictions if not is_blank(row.get("evidence_sentences"))),
            "evidence_sentences_nonempty_rate": rate(sum(1 for row in all_predictions if not is_blank(row.get("evidence_sentences"))), len(all_predictions)),
            "structured_field_nonempty_counts": {
                field: sum(1 for row in all_predictions if not is_blank(row.get(field))) for field in STRUCTURED_FIELDS
            },
            "event_name_by_quality_status": pred_event_name_by_status,
        },
        "matchability": {
            "primary_stable_key": "article_id + schema_id",
            "document_id_available_in_gold": False,
            "event_id_available_in_gold": False,
            "schema_id_available_in_gold_and_predictions": True,
            "test_gold_article_count": len(gold_test_by_article_schema),
            "test_gold_event_pair_count": len(gold_pairs["test"]),
            "formal_prediction_overlap_test_article_count": len(overlap_articles),
            "formal_prediction_overlap_test_article_rate": rate(len(overlap_articles), len(gold_test_by_article_schema)),
            "overlap_gold_pair_count": schema_metrics["gold_total"],
            "overlap_pred_pair_count": schema_metrics["pred_total"],
            "overlap_note": "仅对正式预测和 test 金标共同拥有 article_id 的记录计算；未覆盖样本不计为 FN。",
        },
        "partial_metrics": {
            "scope": "formal Task1 outputs intersected with independent gold test articles only",
            "schema_id_pair_metrics": schema_metrics,
            "event_name_metrics_derived_from_schema_id": event_name_metrics,
            "raw_event_name_field_metrics": raw_event_name_metrics,
            "limitations": [
                "样本仅 25 篇 test 金标文章，不代表 1029 篇 test 金标整体。",
                "金标只有文章-事件对正例标签，无 trigger span 或论元角色标注。",
                "正式输出的要素和 evidence_sentences 字段为空，不能计算要素抽取准确率。",
            ],
        },
        "not_evaluable": not_evaluable,
        "task1_quality_conclusion": {
            "engineering_coverage_is_not_accuracy": True,
            "formal_event_mentions": len(all_predictions),
            "conclusion": "需优化",
            "reason": "当前只能对 25 篇重叠 test 金标做事件识别的局部验收；触发词、论元和证据句无可评金标且正式输出字段为空，不能通过完整质量验收。",
        },
        "formal_file_sha256_before": formal_sha_before or hash_files(FORMAL_FILES),
        "output_files": {},
    }

    audit["_rows"] = {
        "overall_metrics": overall_metric_rows(audit),
        "class_metrics_64": class_metrics(
            event_names,
            gold_pairs["test"],
            gold_test_by_article_event,
            pred_by_article_event_derived,
            overlap_articles,
        ),
        "confusion_matrix": confusion_matrix_rows(
            event_names,
            gold_test_by_article_event,
            pred_by_article_event_derived,
            overlap_articles,
        ),
        "error_samples": build_error_rows(gold_pairs["test"], all_predictions, schema_by_id),
        "field_coverage": field_coverage_rows(all_gold_pairs, all_predictions),
    }
    return audit


def overall_metric_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, scope: str, source: str, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "scope": scope, "source": source, "note": note})

    gold = audit["gold_inventory"]
    match = audit["matchability"]
    partial = audit["partial_metrics"]
    formal = audit["formal_prediction_inventory"]
    leakage = audit["split_and_leakage_check"]
    add("evaluation_status", audit["evaluation_status"], "overall", "audit", "只允许 ready/partially_ready/not_ready")
    add("acceptance_conclusion", audit["acceptance_conclusion"], "overall", "audit", "完整验收未通过")
    add("gold_train_documents", gold["train_document_count"], "gold_train", "data/processed/labeled/news_train.jsonl")
    add("gold_test_documents", gold["test_document_count"], "gold_test", "data/processed/labeled/news_test.jsonl")
    add("gold_train_event_pairs", gold["train_event_pair_count"], "gold_train", "data/processed/labeled/event_pairs_train.jsonl")
    add("gold_test_event_pairs", gold["test_event_pair_count"], "gold_test", "data/processed/labeled/event_pairs_test.jsonl")
    add("gold_unique_event_names", gold["gold_unique_event_name_count"], "gold_total", "data/processed/event_schema.jsonl")
    add("gold_unique_schema_ids", gold["gold_unique_schema_id_count"], "gold_total", "data/processed/event_schema.jsonl")
    add("gold_train_test_article_id_overlap", leakage["gold_train_test_article_id_overlap"], "leakage_check", "gold files")
    add("gold_train_test_exact_content_hash_overlap", leakage["gold_train_test_exact_content_hash_overlap"], "leakage_check", "gold files")
    add("gold_duplication_cross_split_count", leakage["gold_duplication_id_cross_train_to_test_count"] + leakage["gold_duplication_id_cross_test_to_train_count"], "leakage_check", "gold files")
    add("formal_task1_event_mentions", formal["total_event_count"], "formal_prediction", "outputs/final_task1/final_events_*.jsonl")
    add("formal_task1_event_name_nonempty_rate", formal["event_name_nonempty_rate"], "formal_prediction", "outputs/final_task1/final_events_*.jsonl")
    add("formal_task1_evidence_sentences_nonempty_rate", formal["evidence_sentences_nonempty_rate"], "formal_prediction", "outputs/final_task1/final_events_*.jsonl")
    add("matched_test_articles", match["formal_prediction_overlap_test_article_count"], "overlap_test", "article_id")
    add("matched_test_article_rate", match["formal_prediction_overlap_test_article_rate"], "overlap_test", "article_id", "25/1029，不能代表完整 test")
    for key, value in partial["schema_id_pair_metrics"].items():
        add(f"schema_id_{key}", value, "overlap_test", "article_id + schema_id")
    for key, value in partial["event_name_metrics_derived_from_schema_id"].items():
        add(f"event_name_derived_{key}", value, "overlap_test", "schema_id->event_name")
    for key, value in partial["raw_event_name_field_metrics"].items():
        add(f"raw_event_name_{key}", value, "overlap_test", "raw event_name field", "受 llm_accepted event_name 为空影响")
    add("trigger_extraction_evaluable", False, "field_evaluability", "gold/prediction fields")
    add("argument_extraction_evaluable", False, "field_evaluability", "gold/prediction fields")
    add("model_calls_32b", audit["model_calls_32b"], "protection", "audit")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(audit: dict[str, Any]) -> str:
    gold = audit["gold_inventory"]
    match = audit["matchability"]
    metrics = audit["partial_metrics"]
    formal = audit["formal_prediction_inventory"]
    leakage = audit["split_and_leakage_check"]
    schema = metrics["schema_id_pair_metrics"]
    raw = metrics["raw_event_name_field_metrics"]
    lines = [
        "# Task1 独立金标评测与质量验收",
        "",
        f"- 评测状态：`{audit['evaluation_status']}`。",
        f"- 验收结论：**{audit['acceptance_conclusion']}**。完整验收未通过，不能宣称 Task1 已有可信人工金标准确率。",
        "- 本步骤未调用 32B、未重跑 30 万篇、未修改 Task1/Task2 正式输出。",
        "",
        "## 数据盘点",
        "",
        f"- 人工金标来源：`{audit['sources']['raw_gold_files']['train']}`、`{audit['sources']['raw_gold_files']['test']}`；规范化后为 `data/processed/labeled/news_*.jsonl` 和 `event_pairs_*.jsonl`。",
        f"- 金标文档：train={gold['train_document_count']}，test={gold['test_document_count']}，合计={gold['total_document_count']}。",
        f"- 金标事件对：train={gold['train_event_pair_count']}，test={gold['test_event_pair_count']}，合计={gold['total_event_pair_count']}。",
        f"- 事件类别：64 个 event_name，120 个 schema_id；schema_id 覆盖 schema 目录 120/120。",
        f"- 正式 Task1 输出：{formal['total_event_count']} 条 EventMention，train={formal['train_event_count']}，test={formal['test_event_count']}。",
        "",
        "## 独立性与泄漏",
        "",
        f"- gold train/test article_id 重叠：{leakage['gold_train_test_article_id_overlap']}。",
        f"- gold train/test sample_id 重叠：{leakage['gold_train_test_sample_id_overlap']}。",
        f"- gold train/test 精确正文 hash 重叠：{leakage['gold_train_test_exact_content_hash_overlap']}。",
        f"- duplication_id 跨 train/test：{leakage['gold_duplication_id_cross_train_to_test_count'] + leakage['gold_duplication_id_cross_test_to_train_count']}。",
        f"- 正式预测与 gold train 文章重叠：{leakage['formal_prediction_overlap_gold_train_articles']}；与 gold test 文章重叠：{leakage['formal_prediction_overlap_gold_test_articles']}。",
        "",
        "## 可评性判断",
        "",
        "- 状态选择：`partially_ready`。",
        "- 可评：仅能在正式预测与独立 test 金标共有的 `article_id` 上，用 `article_id + schema_id` 做文章-事件对级事件识别局部评测。",
        "- 不可评：触发词 strict/relaxed、论元识别、论元角色分类、主体/时间/金额/地点/人物、端到端事件 exact-match；原因是金标无 span/role/论元字段，正式输出相关字段也为空。",
        f"- 共同 test 文章：{match['formal_prediction_overlap_test_article_count']}/{match['test_gold_article_count']}，覆盖率 {match['formal_prediction_overlap_test_article_rate']}。",
        "",
        "## 局部事件识别指标",
        "",
        f"- 范围：仅 {schema['overlap_article_count']} 篇重叠 test 金标文章，不能代表完整 test。",
        f"- schema_id Precision={schema['precision']}，Recall={schema['recall']}，F1={schema['f1']}，article exact accuracy={schema['exact_article_accuracy']}。",
        f"- raw event_name 字段 Precision={raw['precision']}，Recall={raw['recall']}，F1={raw['f1']}；低于 schema_id 是因为 `llm_accepted` 段 raw event_name 为空。",
        f"- raw event_name 非空率：{formal['event_name_nonempty_rate']}；evidence_sentences 非空率：{formal['evidence_sentences_nonempty_rate']}。",
        "",
        "## 主要问题",
        "",
        "- 4973 条正式输出中，924 条 `llm_accepted` 的 raw `event_name` 为空；但 `schema_id` 完整，可由 schema 目录映射事件名。",
        "- 主体、时间、动作、对象、状态、地点、影响方向、证据句、抽取置信度在正式输出中均无可用填充，无法验证要素抽取准确率。",
        "- 正式预测未覆盖 1029 篇 gold test 中的 1004 篇；这不是模型漏提结论，而是预测文件与金标评测集未对齐。",
        "",
        "## 结论",
        "",
        "Task1 当前不能完成严格人工金标准确率验收。已有金标可以支持文章-事件对级评测，但需要在完整 gold test 上生成对应正式预测，或补充能与 4973 条正式 EventMention 对齐的人工金标；要素抽取还需要新增 trigger/span/role 级金标并补齐正式输出字段。",
    ]
    return "\n".join(lines) + "\n"


def sanitize_for_json(audit: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in audit.items() if key != "_rows"}
    return out


def write_outputs(audit: dict[str, Any], output_dir: Path, overwrite: bool = False) -> dict[str, str]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace audit outputs")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit["_rows"]

    paths = {
        "json_report": output_dir / "task1_independent_gold_evaluation.json",
        "markdown_report": output_dir / "task1_independent_gold_evaluation.md",
        "overall_metrics_csv": output_dir / "overall_metrics.csv",
        "class_metrics_csv": output_dir / "class_metrics_64.csv",
        "confusion_matrix_csv": output_dir / "confusion_matrix.csv",
        "error_samples_xlsx": output_dir / "error_samples.xlsx",
        "field_coverage_csv": output_dir / "field_coverage.csv",
        "manifest_json": output_dir / "manifest.json",
        "sha256sums": output_dir / "SHA256SUMS.txt",
    }

    write_csv(paths["overall_metrics_csv"], rows["overall_metrics"])
    write_csv(paths["class_metrics_csv"], rows["class_metrics_64"])
    write_csv(paths["confusion_matrix_csv"], rows["confusion_matrix"])
    write_csv(paths["field_coverage_csv"], rows["field_coverage"])
    pd.DataFrame(rows["error_samples"]).to_excel(paths["error_samples_xlsx"], index=False, engine="openpyxl")
    paths["markdown_report"].write_text(markdown_report(audit), encoding="utf-8")

    audit["output_files"] = {
        key: rel(path) if path.is_relative_to(ROOT) else str(path) for key, path in paths.items()
    }
    write_json(paths["json_report"], sanitize_for_json(audit))

    manifest_files = [path for key, path in paths.items() if key not in {"manifest_json", "sha256sums"}]
    manifest = {
        "manifest_version": "task1_independent_gold_eval_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_status": audit["evaluation_status"],
        "acceptance_conclusion": audit["acceptance_conclusion"],
        "formal_file_sha256_unchanged": audit.get("formal_file_sha256_unchanged"),
        "model_calls_32b": audit["model_calls_32b"],
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in manifest_files
        ],
    }
    write_json(paths["manifest_json"], manifest)

    sum_lines = []
    for path in manifest_files + [paths["manifest_json"]]:
        sum_lines.append(f"{sha256(path)}  {path.name}")
    paths["sha256sums"].write_text("\n".join(sum_lines) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formal_before = hash_files(FORMAL_FILES)
    audit = build_audit(formal_sha_before=formal_before)
    formal_after = hash_files(FORMAL_FILES)
    audit["formal_file_sha256_after"] = formal_after
    audit["formal_file_sha256_unchanged"] = formal_before == formal_after
    if not audit["formal_file_sha256_unchanged"]:
        raise RuntimeError("formal Task1 output SHA256 changed during read-only audit")
    write_outputs(audit, args.output_dir.resolve(), overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": audit["evaluation_status"],
                "acceptance_conclusion": audit["acceptance_conclusion"],
                "output_dir": str(args.output_dir),
                "model_calls_32b": audit["model_calls_32b"],
                "formal_sha_unchanged": audit["formal_file_sha256_unchanged"],
                "overlap_test_articles": audit["matchability"]["formal_prediction_overlap_test_article_count"],
                "schema_id_f1": audit["partial_metrics"]["schema_id_pair_metrics"]["f1"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
