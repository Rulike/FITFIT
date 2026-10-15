#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5.4: leakage-free Task1 development baseline and recall attribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.data.event_match_text import build_input_text, smart_truncate
from src.routing.target_router import TargetRouter


OUTPUT_DIR = ROOT / "outputs/task1_step5_4_dev_baseline_v1"
NEWS_TRAIN = ROOT / "data/processed/labeled/news_train.jsonl"
PAIRS_TRAIN = ROOT / "data/processed/labeled/event_pairs_train.jsonl"
NEWS_TEST = ROOT / "data/processed/labeled/news_test.jsonl"
PAIRS_TEST = ROOT / "data/processed/labeled/event_pairs_test.jsonl"
SCHEMA_PATH = ROOT / "data/processed/event_schema.jsonl"
ALIAS_PATH = ROOT / "configs/target_aliases.json"
BASE_CONFIG_PATH = ROOT / "configs/event_match_macbert_base.json"
EXISTING_MODEL_DIR = ROOT / "outputs/event_match_baseline/best_model"
EXISTING_TRAIN_DIR = ROOT / "outputs/event_match_baseline"
PRETRAINED_MODEL_DIR = OUTPUT_DIR / "pretrained/hfl_chinese_macbert_base"
SEED = 2026
THRESHOLD = 0.38

PROTECTED_PATHS = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/final_task1/final_events_train.xlsx",
    ROOT / "outputs/final_task1/final_events_test.xlsx",
    ROOT / "outputs/final_task1/task1_completion_certificate.json",
    ROOT / "reports/task1_final_summary.json",
    ROOT / "reports/task1_final_event_frequency.csv",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
    ROOT / "outputs/task1_gold_shadow_eval_v1/shadow_predictions_raw.jsonl",
    ROOT / "outputs/task1_gold_shadow_eval_v1/shadow_predictions_backfilled.jsonl",
    ROOT / "outputs/task1_gold_shadow_eval_v1/evaluation_report.json",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def optional_sha256(path: Path) -> str:
    return sha256_file(path) if path.exists() else "missing"


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def text_hash(*parts: Any) -> str:
    text = "\n".join(norm_text(x) for x in parts if norm_text(x))
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def protected_sha_snapshot() -> dict[str, str]:
    return {rel(path): sha256_file(path) if path.exists() else "missing" for path in PROTECTED_PATHS}


def load_schema() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    rows = read_jsonl(SCHEMA_PATH)
    by_id = {row["schema_id"]: row for row in rows}
    by_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[(row["target_type"], str(row["target_code"]))].append(row)
    return by_id, by_target


def audit_existing_training(output_dir: Path) -> dict[str, Any]:
    model_data_train = ROOT / "data/processed/model_data/event_match_train.jsonl"
    model_data_val = ROOT / "data/processed/model_data/event_match_validation.jsonl"
    model_data_official = ROOT / "data/processed/model_data/event_match_official_test.jsonl"
    augmented_train = ROOT / "data/processed/augmented/news_train_augmented.jsonl"
    augmented_test = ROOT / "data/processed/augmented/news_test_augmented.jsonl"
    train_rows = read_jsonl(model_data_train)
    val_rows = read_jsonl(model_data_val)
    official_rows = read_jsonl(model_data_official)
    train_ids = {row["article_id"] for row in train_rows}
    val_ids = {row["article_id"] for row in val_rows}
    official_ids = {row["article_id"] for row in official_rows}
    gold_test_ids = {row["article_id"] for row in read_jsonl(NEWS_TEST)}
    gold_train_ids = {row["article_id"] for row in read_jsonl(NEWS_TRAIN)}
    augmented_train_ids = {row["article_id"] for row in read_jsonl(augmented_train)}
    augmented_test_ids = {row["article_id"] for row in read_jsonl(augmented_test)}
    config = read_json(BASE_CONFIG_PATH)
    trainer_state = read_json(EXISTING_TRAIN_DIR / "checkpoint-312/trainer_state.json")
    train_metrics = read_json(EXISTING_TRAIN_DIR / "train_metrics.json")
    threshold = read_json(EXISTING_TRAIN_DIR / "best_threshold.json")
    leakage = {
        "train_article_overlap_gold_test": len(train_ids & gold_test_ids),
        "validation_article_overlap_gold_test": len(val_ids & gold_test_ids),
        "official_test_article_overlap_gold_test": len(official_ids & gold_test_ids),
        "train_article_not_in_gold_train": len(train_ids - gold_train_ids),
        "validation_article_not_in_gold_train": len(val_ids - gold_train_ids),
        "train_article_not_in_augmented_train": len(train_ids - augmented_train_ids),
        "validation_article_not_in_augmented_train": len(val_ids - augmented_train_ids),
        "train_article_overlap_augmented_test": len(train_ids & augmented_test_ids),
        "validation_article_overlap_augmented_test": len(val_ids & augmented_test_ids),
        "train_validation_article_overlap": len(train_ids & val_ids),
    }
    if (
        leakage["train_article_overlap_gold_test"]
        or leakage["validation_article_overlap_gold_test"]
        or leakage["train_article_overlap_augmented_test"]
        or leakage["validation_article_overlap_augmented_test"]
    ):
        conclusion = "test_leakage"
    elif leakage["train_validation_article_overlap"]:
        conclusion = "train_dev_contaminated"
    elif leakage["train_article_not_in_augmented_train"] or leakage["validation_article_not_in_augmented_train"]:
        conclusion = "provenance_unknown"
    else:
        conclusion = "clean_training"
    audit = {
        "generated_at": now(),
        "conclusion": conclusion,
        "best_model_path": rel(EXISTING_MODEL_DIR),
        "training_code": rel(ROOT / "scripts/train_event_match_baseline.py"),
        "evaluation_code": rel(ROOT / "scripts/evaluate_event_match_baseline.py"),
        "base_config_path": rel(BASE_CONFIG_PATH),
        "base_config": config,
        "actual_model_data": {
            "train_file": rel(model_data_train),
            "validation_file": rel(model_data_val),
            "official_test_file": rel(model_data_official),
            "augmented_train_file": rel(augmented_train),
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "official_test_rows": len(official_rows),
            "train_articles": len(train_ids),
            "validation_articles": len(val_ids),
            "official_test_articles": len(official_ids),
        },
        "seed": config.get("seed"),
        "checkpoint_selection": {
            "metric_for_best_model": config.get("metric_for_best_model"),
            "best_checkpoint": trainer_state.get("best_model_checkpoint"),
            "best_metric": trainer_state.get("best_metric"),
            "best_threshold": threshold,
        },
        "train_metrics": train_metrics,
        "leakage_checks": leakage,
        "sha256": {
            "best_model_config": sha256_file(EXISTING_MODEL_DIR / "config.json"),
            "best_model_weights": sha256_file(EXISTING_MODEL_DIR / "model.safetensors"),
            "base_config": sha256_file(BASE_CONFIG_PATH),
            "train_file": sha256_file(model_data_train),
            "validation_file": sha256_file(model_data_val),
            "official_test_file": sha256_file(model_data_official),
            "augmented_train_file": sha256_file(augmented_train),
        },
        "strict_reproduction_possible": True,
        "notes": [
            "现有 best_model 使用历史 data/processed/model_data/event_match_train/validation 训练，训练/验证文章均属于train侧增强集合。",
            "本审计只读检查，不将1029篇gold test标签用于训练、选模型或阈值。",
        ],
    }
    write_json(output_dir / "training_provenance_audit.json", audit)
    return audit


def group_key(row: dict[str, Any]) -> str:
    aid = str(row.get("article_id") or "")
    title_h = text_hash(row.get("title", ""))
    content_h = text_hash(row.get("content", ""))
    dup = str(row.get("duplication_id") or "")
    return dup or content_h or title_h or aid


def build_dev_split(output_dir: Path) -> dict[str, Any]:
    rng = random.Random(SEED)
    news = read_jsonl(NEWS_TRAIN)
    pairs = read_jsonl(PAIRS_TRAIN)
    by_article = {row["article_id"]: row for row in news}
    pairs_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        pairs_by_article[pair["article_id"]].append(pair)
    groups: dict[str, set[str]] = defaultdict(set)
    for article in news:
        groups[group_key(article)].add(article["article_id"])
    group_stats = []
    for key, aids in groups.items():
        schema_ids = set()
        event_names = set()
        pair_count = 0
        for aid in aids:
            for pair in pairs_by_article.get(aid, []):
                schema_ids.add(pair["schema_id"])
                event_names.add(pair["event_name"])
                pair_count += 1
        group_stats.append({
            "group_key": key,
            "article_ids": sorted(aids),
            "article_count": len(aids),
            "event_pair_count": pair_count,
            "schema_ids": sorted(schema_ids),
            "event_names": sorted(event_names),
        })
    group_stats.sort(key=lambda x: (x["event_pair_count"], x["group_key"]), reverse=True)

    total_articles = len(news)
    target_val_articles = round(total_articles * 0.2)
    schema_group_count: dict[str, int] = Counter()
    for group in group_stats:
        for sid in group["schema_ids"]:
            schema_group_count[sid] += 1

    val_groups: set[str] = set()
    train_groups: set[str] = set()
    val_article_count = 0

    eligible_by_schema: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in group_stats:
        for sid in group["schema_ids"]:
            eligible_by_schema[sid].append(group)
    for sid, items in sorted(eligible_by_schema.items(), key=lambda x: (len(x[1]), x[0])):
        if schema_group_count[sid] < 2:
            continue
        shuffled = list(items)
        rng.shuffle(shuffled)
        if any(g["group_key"] in val_groups for g in shuffled):
            continue
        chosen = shuffled[0]
        if val_article_count + chosen["article_count"] <= target_val_articles * 1.1:
            val_groups.add(chosen["group_key"])
            val_article_count += chosen["article_count"]

    remaining = [g for g in group_stats if g["group_key"] not in val_groups]
    rng.shuffle(remaining)
    for group in remaining:
        if val_article_count >= target_val_articles:
            break
        val_groups.add(group["group_key"])
        val_article_count += group["article_count"]

    for group in group_stats:
        if group["group_key"] not in val_groups:
            train_groups.add(group["group_key"])

    train_ids = sorted(aid for group in group_stats if group["group_key"] in train_groups for aid in group["article_ids"])
    val_ids = sorted(aid for group in group_stats if group["group_key"] in val_groups for aid in group["article_ids"])
    train_set = set(train_ids)
    val_set = set(val_ids)
    train_pairs = [row for row in pairs if row["article_id"] in train_set]
    val_pairs = [row for row in pairs if row["article_id"] in val_set]

    train_hashes = {(text_hash(by_article[aid].get("title", "")), text_hash(by_article[aid].get("content", ""))) for aid in train_ids}
    val_hashes = {(text_hash(by_article[aid].get("title", "")), text_hash(by_article[aid].get("content", ""))) for aid in val_ids}
    train_dup = {str(by_article[aid].get("duplication_id") or "") for aid in train_ids if by_article[aid].get("duplication_id")}
    val_dup = {str(by_article[aid].get("duplication_id") or "") for aid in val_ids if by_article[aid].get("duplication_id")}
    rare = []
    for sid in sorted({p["schema_id"] for p in pairs}):
        train_c = sum(1 for p in train_pairs if p["schema_id"] == sid)
        val_c = sum(1 for p in val_pairs if p["schema_id"] == sid)
        total_c = train_c + val_c
        if train_c == 0 or val_c == 0:
            rare.append({
                "schema_id": sid,
                "total_event_pairs": total_c,
                "train_event_pairs": train_c,
                "validation_event_pairs": val_c,
                "reason": "too_few_or_group_constrained",
            })
    write_jsonl(output_dir / "development_train_ids.jsonl", [{"article_id": aid} for aid in train_ids])
    write_jsonl(output_dir / "development_validation_ids.jsonl", [{"article_id": aid} for aid in val_ids])
    manifest = {
        "generated_at": now(),
        "seed": SEED,
        "source_files": {
            "news_train": rel(NEWS_TRAIN),
            "event_pairs_train": rel(PAIRS_TRAIN),
        },
        "source_sha256": {
            "news_train": sha256_file(NEWS_TRAIN),
            "event_pairs_train": sha256_file(PAIRS_TRAIN),
        },
        "development_train": {
            "article_count": len(train_ids),
            "event_pair_count": len(train_pairs),
            "event_name_count": len({p["event_name"] for p in train_pairs}),
            "schema_id_count": len({p["schema_id"] for p in train_pairs}),
        },
        "development_validation": {
            "article_count": len(val_ids),
            "event_pair_count": len(val_pairs),
            "event_name_count": len({p["event_name"] for p in val_pairs}),
            "schema_id_count": len({p["schema_id"] for p in val_pairs}),
        },
        "leakage_checks": {
            "article_id_overlap": len(train_set & val_set),
            "title_content_hash_overlap": len(train_hashes & val_hashes),
            "duplication_id_overlap": len(train_dup & val_dup),
        },
        "rare_or_uncovered_schema_report": rare,
        "validation_ratio_by_article": round(len(val_ids) / len(news), 6),
        "notes": [
            "article_id为最小分组单位；相同duplication_id、标题hash、正文hash不跨集合。",
            "未复制样本制造类别覆盖。",
        ],
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return {
        "news": news,
        "pairs": pairs,
        "by_article": by_article,
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "train_pairs": train_pairs,
        "validation_pairs": val_pairs,
        "manifest": manifest,
    }


def build_candidates_for_articles(
    articles: list[dict[str, Any]],
    gold_pairs_by_article: dict[str, list[dict[str, Any]]],
    schema_by_id: dict[str, dict[str, Any]],
    schemas_by_target: dict[tuple[str, str], list[dict[str, Any]]],
    router: TargetRouter,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    candidates = []
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in sorted(articles, key=lambda x: x["article_id"]):
        routes = router.route(article.get("title", ""), article.get("content", ""))
        gold_schema_ids = {p["schema_id"] for p in gold_pairs_by_article.get(article["article_id"], [])}
        for route in routes:
            target_key = (route["target_type"], str(route["target_code"]))
            for schema in schemas_by_target.get(target_key, []):
                row = {
                    "sample_id": f"dev_{article['article_id']}_{schema['schema_id']}",
                    "article_id": article["article_id"],
                    "title": article.get("title", ""),
                    "content": article.get("content", ""),
                    "publish_time": article.get("publish_time", ""),
                    "source": article.get("source", ""),
                    "target_type": route["target_type"],
                    "target_code": route["target_code"],
                    "target_name": route["target_name"],
                    "route_method": route["route_method"],
                    "route_confidence": route["route_confidence"],
                    "matched_alias": route["matched_alias"],
                    "schema_id": schema["schema_id"],
                    "event_name": schema["event_name"],
                    "event_description": schema.get("raw_description", ""),
                    "label": 1 if schema["schema_id"] in gold_schema_ids else 0,
                    "negative_type": None if schema["schema_id"] in gold_schema_ids else "routed_non_gold_candidate",
                    "is_duplicate_news": bool(article.get("duplication_id")),
                    "canonical_group_id": str(article.get("duplication_id") or article["article_id"]),
                }
                candidates.append(row)
                by_article[article["article_id"]].append(row)
    return candidates, by_article


def jaccard_chars(a: str, b: str, n: int = 2) -> float:
    if not a or not b:
        return 0.0
    sa = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    sb = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def schema_similarity(schema_by_id: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    sids = sorted(schema_by_id)
    sims: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for i, sid in enumerate(sids):
        a = schema_by_id[sid]
        a_text = a.get("event_name", "") + " " + a.get("raw_description", "")
        for other in sids[i + 1:]:
            b = schema_by_id[other]
            b_text = b.get("event_name", "") + " " + b.get("raw_description", "")
            score = 0.7 * jaccard_chars(a.get("event_name", ""), b.get("event_name", ""), 2) + 0.3 * jaccard_chars(a_text, b_text, 3)
            if score > 0:
                sims[sid].append((other, score))
                sims[other].append((sid, score))
    for sid in sims:
        sims[sid].sort(key=lambda x: -x[1])
    return sims


def target_name_in_text(target_name: str, title: str, content: str) -> bool:
    if not target_name:
        return False
    return target_name.lower() in (title + " " + content).lower()


def make_positive_record(pair: dict[str, Any], article: dict[str, Any], split_name: str) -> dict[str, Any]:
    item = dict(pair)
    item.update({
        "article_id": article["article_id"],
        "title": article.get("title", pair.get("title", "")),
        "content": article.get("content", pair.get("content", "")),
        "publish_time": article.get("publish_time", pair.get("publish_time", "")),
        "source": article.get("source", pair.get("source", "")),
        "canonical_group_id": str(article.get("duplication_id") or article["article_id"]),
        "canonical_article_id": str(article.get("duplication_id") or article["article_id"]),
        "is_duplicate_news": bool(article.get("duplication_id")),
        "internal_split": split_name,
        "split_seed": SEED,
        "label": 1,
        "negative_type": None,
        "negative_confidence": None,
        "schema_similarity": None,
        "source_positive_sample_id": None,
    })
    return item


def build_training_pair_data(
    split: dict[str, Any],
    schema_by_id: dict[str, dict[str, Any]],
    train_articles: list[dict[str, Any]],
    train_gold_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sim = schema_similarity(schema_by_id)
    pairs_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in train_gold_pairs:
        pairs_by_article[pair["article_id"]].append(pair)
    records = []
    neg_keys = set()
    for article in sorted(train_articles, key=lambda x: x["article_id"]):
        aid = article["article_id"]
        pos_pairs = pairs_by_article.get(aid, [])
        pos_schema_ids = {p["schema_id"] for p in pos_pairs}
        pos_targets = {(p["target_type"], str(p["target_code"])) for p in pos_pairs}
        pos_event_names = {p["event_name"] for p in pos_pairs}
        for pair in pos_pairs:
            records.append(make_positive_record(pair, article, "development_train"))
        if not pos_pairs:
            continue
        selected_negs: list[dict[str, Any]] = []
        for tt, tc in sorted(pos_targets):
            candidates = [
                sid for sid, schema in schema_by_id.items()
                if schema["target_type"] == tt and str(schema["target_code"]) == tc and sid not in pos_schema_ids
            ]
            rng.shuffle(candidates)
            for sid in candidates[:2]:
                selected_negs.append({"schema_id": sid, "negative_type": "same_target_other_event", "negative_confidence": "medium", "schema_similarity": None})
        pos_types = {tt for tt, _ in pos_targets}
        for event_name in sorted(pos_event_names):
            candidates = []
            for sid, schema in schema_by_id.items():
                if sid in pos_schema_ids or schema["event_name"] != event_name or schema["target_type"] not in pos_types:
                    continue
                if (schema["target_type"], str(schema["target_code"])) in pos_targets:
                    continue
                if target_name_in_text(schema["target_name"], article.get("title", ""), article.get("content", "")):
                    continue
                candidates.append(sid)
            rng.shuffle(candidates)
            for sid in candidates[:1]:
                selected_negs.append({"schema_id": sid, "negative_type": "same_event_wrong_target", "negative_confidence": "high", "schema_similarity": None})
        for pos_sid in sorted(pos_schema_ids):
            pos_schema = schema_by_id.get(pos_sid, {})
            for sid, score in sim.get(pos_sid, [])[:20]:
                schema = schema_by_id[sid]
                if sid in pos_schema_ids or schema["target_type"] != pos_schema.get("target_type"):
                    continue
                if (schema["target_type"], str(schema["target_code"])) in pos_targets:
                    continue
                if target_name_in_text(schema["target_name"], article.get("title", ""), article.get("content", "")):
                    continue
                selected_negs.append({"schema_id": sid, "negative_type": "similar_schema_wrong_target", "negative_confidence": "high", "schema_similarity": round(score, 4)})
                break
        for neg in selected_negs:
            sid = neg["schema_id"]
            key = (aid, sid)
            if key in neg_keys or sid in pos_schema_ids:
                continue
            neg_keys.add(key)
            schema = schema_by_id[sid]
            source_id = pos_pairs[0]["sample_id"] if pos_pairs else ""
            records.append({
                "sample_id": f"dev_train_neg_{aid}_{sid}",
                "split": "train",
                "article_id": aid,
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "publish_time": article.get("publish_time", ""),
                "source": article.get("source", ""),
                "canonical_group_id": str(article.get("duplication_id") or aid),
                "canonical_article_id": str(article.get("duplication_id") or aid),
                "is_duplicate_news": bool(article.get("duplication_id")),
                "schema_id": sid,
                "target_type": schema["target_type"],
                "target_code": str(schema["target_code"]),
                "target_name": schema["target_name"],
                "event_name": schema["event_name"],
                "event_description": schema.get("raw_description", ""),
                "label": 0,
                "negative_type": neg["negative_type"],
                "negative_confidence": neg["negative_confidence"],
                "schema_similarity": neg["schema_similarity"],
                "source_positive_sample_id": source_id,
                "internal_split": "development_train",
                "split_seed": SEED,
            })
    return records


def prepare_model_data(split: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    schema_by_id, schemas_by_target = load_schema()
    router = TargetRouter(schema_file=str(SCHEMA_PATH), alias_file=str(ALIAS_PATH))
    gold_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in split["pairs"]:
        gold_by_article[pair["article_id"]].append(pair)
    train_articles = [split["by_article"][aid] for aid in split["train_ids"]]
    val_articles = [split["by_article"][aid] for aid in split["validation_ids"]]
    train_candidates = build_training_pair_data(split, schema_by_id, train_articles, split["train_pairs"])
    train_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_candidates:
        train_by_article[row["article_id"]].append(row)
    val_candidates, val_by_article = build_candidates_for_articles(val_articles, gold_by_article, schema_by_id, schemas_by_target, router)
    write_jsonl(output_dir / "model_data/event_match_train.jsonl", train_candidates)
    write_jsonl(output_dir / "model_data/event_match_validation.jsonl", val_candidates)
    return {
        "schema_by_id": schema_by_id,
        "schemas_by_target": schemas_by_target,
        "train_candidates": train_candidates,
        "validation_candidates": val_candidates,
        "train_candidates_by_article": train_by_article,
        "validation_candidates_by_article": val_by_article,
    }


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_baseline(output_dir: Path, model_data: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

    class EventMatchDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int):
            self.records = records
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            rec = self.records[idx]
            text = smart_truncate(build_input_text(rec), self.tokenizer, self.max_length)
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": int(rec["label"])}

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred = np.argmax(logits, axis=-1)
        labels_arr = np.array(labels)
        pred_arr = np.array(pred)
        acc = float((labels_arr == pred_arr).mean()) if len(labels_arr) else 0.0
        tp = int(((labels_arr == 1) & (pred_arr == 1)).sum())
        fp = int(((labels_arr == 0) & (pred_arr == 1)).sum())
        fn = int(((labels_arr == 1) & (pred_arr == 0)).sum())
        tn = int(((labels_arr == 0) & (pred_arr == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        p0 = tn / (tn + fn) if tn + fn else 0.0
        r0 = tn / (tn + fp) if tn + fp else 0.0
        f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
        macro_f1 = (f0 + f1) / 2.0
        support0 = int((labels_arr == 0).sum())
        support1 = int((labels_arr == 1).sum())
        weighted_f1 = (f0 * support0 + f1 * support1) / (support0 + support1) if support0 + support1 else 0.0
        auc = 0.5
        pr_auc = 0.0
        return {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "positive_f1": float(f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
        }

    config = read_json(BASE_CONFIG_PATH)
    pretrained_dir = output_dir / "pretrained/hfl_chinese_macbert_base"
    if not pretrained_dir.exists():
        default_pretrained_dir = PRETRAINED_MODEL_DIR
        if default_pretrained_dir.exists():
            pretrained_dir = default_pretrained_dir
    pretrained_source = {
        "requested_model_name_or_path": config["model_name_or_path"],
        "actual_model_name_or_path": str(pretrained_dir if pretrained_dir.exists() else config["model_name_or_path"]),
        "actual_model_path_relative": rel(pretrained_dir) if pretrained_dir.exists() and pretrained_dir.is_relative_to(ROOT) else "",
        "download_endpoint": "https://hf-mirror.com",
        "local_pretrained_available": pretrained_dir.exists(),
        "config_sha256": optional_sha256(pretrained_dir / "config.json") if pretrained_dir.exists() else "remote_not_materialized",
        "pytorch_model_sha256": optional_sha256(pretrained_dir / "pytorch_model.bin") if pretrained_dir.exists() else "remote_not_materialized",
        "vocab_sha256": optional_sha256(pretrained_dir / "vocab.txt") if pretrained_dir.exists() else "remote_not_materialized",
    }
    training_config = dict(config)
    training_config.update({
        "step": "5.4",
        "output_dir": rel(output_dir / "model"),
        "train_file": rel(output_dir / "model_data/event_match_train.jsonl"),
        "validation_file": rel(output_dir / "model_data/event_match_validation.jsonl"),
        "gold_test_usage": "forbidden_not_read",
        "pretrained_source": pretrained_source,
    })
    write_json(output_dir / "baseline_training_config.json", training_config)

    set_seed(SEED)
    start = time.time()
    model_name_or_path = pretrained_source["actual_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=pretrained_dir.exists())
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2, local_files_only=pretrained_dir.exists())
    train_records = model_data["train_candidates"]
    val_records = model_data["validation_candidates"]
    max_length = int(config["max_length"])
    train_dataset = EventMatchDataset(train_records, tokenizer, max_length)
    val_dataset = EventMatchDataset(val_records, tokenizer, max_length)
    model_output_dir = output_dir / "model"
    args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        fp16=bool(config["fp16"] and torch.cuda.is_available()),
        seed=SEED,
        eval_strategy=config.get("evaluation_strategy", "epoch"),
        save_strategy=config["save_strategy"],
        logging_steps=config["logging_steps"],
        save_total_limit=config["save_total_limit"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])],
    )
    train_result = trainer.train()
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    prediction_output = trainer.predict(val_dataset)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    val_preds = []
    for rec, prob in zip(val_records, probs):
        item = dict(rec)
        item.update({
            "probability_negative": float(prob[0]),
            "probability_positive": float(prob[1]),
            "prediction_threshold_0_5": int(float(prob[1]) >= 0.5),
            "prediction_current_threshold": int(float(prob[1]) >= THRESHOLD),
        })
        val_preds.append(item)
    write_jsonl(output_dir / "validation_predictions.jsonl", val_preds)

    eval_history = [row for row in trainer.state.log_history if "eval_loss" in row]
    history_rows = []
    for row in trainer.state.log_history:
        history_rows.append({k: row.get(k, "") for k in [
            "step", "epoch", "loss", "learning_rate", "grad_norm",
            "eval_loss", "eval_accuracy", "eval_precision", "eval_recall",
            "eval_positive_f1", "eval_macro_f1", "eval_weighted_f1", "eval_roc_auc", "eval_pr_auc",
        ]})
    write_csv(output_dir / "baseline_training_history.csv", history_rows, [
        "step", "epoch", "loss", "learning_rate", "grad_norm",
        "eval_loss", "eval_accuracy", "eval_precision", "eval_recall",
        "eval_positive_f1", "eval_macro_f1", "eval_weighted_f1", "eval_roc_auc", "eval_pr_auc",
    ])
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    summary = {
        "train_runtime_seconds": round(time.time() - start, 3),
        "train_loss": float(train_result.training_loss),
        "global_step": int(train_result.global_step),
        "eval_history": eval_history,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "parameter_count": int(sum(p.numel() for p in trainer.model.parameters())),
        "train_candidate_count": len(train_records),
        "train_positive_count": sum(1 for r in train_records if r["label"] == 1),
        "train_negative_count": sum(1 for r in train_records if r["label"] == 0),
        "validation_candidate_count": len(val_records),
        "validation_positive_count": sum(1 for r in val_records if r["label"] == 1),
        "validation_negative_count": sum(1 for r in val_records if r["label"] == 0),
        "device_info": device_info,
        "pretrained_source": pretrained_source,
        "config_sha256": sha256_file(output_dir / "baseline_training_config.json"),
        "checkpoint_sha256": sha256_file(best_model_dir / "model.safetensors"),
        "best_model_config_sha256": sha256_file(best_model_dir / "config.json"),
    }
    write_json(output_dir / "baseline_training_summary.json", summary)
    return {"summary": summary, "validation_predictions": val_preds}


def predicted_schema_set(preds: list[dict[str, Any]], threshold: float) -> set[str]:
    return {row["schema_id"] for row in preds if float(row.get("probability_positive", 0.0)) >= threshold}


def compute_pair_metrics(
    gold_pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    schema_by_id: dict[str, dict[str, Any]],
    article_ids: list[str],
    threshold: float,
) -> dict[str, Any]:
    gold_by_article: dict[str, set[str]] = defaultdict(set)
    pred_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_pairs:
        gold_by_article[row["article_id"]].add(row["schema_id"])
    for row in predictions:
        pred_by_article[row["article_id"]].append(row)
    all_event_names = sorted({row["event_name"] for row in schema_by_id.values()})
    class_tp = {name: 0 for name in all_event_names}
    class_pred = {name: 0 for name in all_event_names}
    class_gold = {name: 0 for name in all_event_names}
    article_exact = 0
    at_least_one = 0
    success_no_event = 0
    complete_miss = 0
    overpredicted = 0
    misclassified = 0
    article_rows = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    tp_total = pred_total = gold_total = 0
    for aid in article_ids:
        gold_set = set(gold_by_article.get(aid, set()))
        rows = pred_by_article.get(aid, [])
        pred_set = predicted_schema_set(rows, threshold)
        tp_set = gold_set & pred_set
        tp_total += len(tp_set)
        pred_total += len(pred_set)
        gold_total += len(gold_set)
        exact = gold_set == pred_set
        article_exact += int(exact)
        at_least_one += int(bool(tp_set))
        success_no_event += int(len(pred_set) == 0)
        complete_miss += int(bool(gold_set) and not tp_set)
        overpredicted += int(len(pred_set) > len(gold_set))
        misclassified += int(bool(pred_set) and not exact and bool(tp_set))
        for sid in gold_set:
            name = schema_by_id.get(sid, {}).get("event_name", sid)
            if name in class_gold:
                class_gold[name] += 1
        for sid in pred_set:
            name = schema_by_id.get(sid, {}).get("event_name", sid)
            if name in class_pred:
                class_pred[name] += 1
        for sid in tp_set:
            name = schema_by_id.get(sid, {}).get("event_name", sid)
            if name in class_tp:
                class_tp[name] += 1
        top_pred = "__NO_PRED__"
        if rows:
            top = max(rows, key=lambda x: float(x.get("probability_positive", 0.0)))
            if float(top.get("probability_positive", 0.0)) >= threshold:
                top_pred = schema_by_id.get(top["schema_id"], {}).get("event_name", top["schema_id"])
        for sid in sorted(gold_set):
            gold_name = schema_by_id.get(sid, {}).get("event_name", sid)
            confusion[gold_name][top_pred] += 1
        article_rows.append({
            "article_id": aid,
            "gold_schema_ids": "|".join(sorted(gold_set)),
            "pred_schema_ids": "|".join(sorted(pred_set)),
            "gold_count": len(gold_set),
            "pred_count": len(pred_set),
            "tp": len(tp_set),
            "fp": max(0, len(pred_set) - len(tp_set)),
            "fn": max(0, len(gold_set) - len(tp_set)),
            "exact_match": exact,
            "at_least_one_correct": bool(tp_set),
            "success_no_event": len(pred_set) == 0,
        })
    precision = tp_total / pred_total if pred_total else 0.0
    recall = tp_total / gold_total if gold_total else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    class_rows = []
    fixed_f1 = []
    observed_f1 = []
    weighted = []
    for name in all_event_names:
        tp = class_tp[name]
        pred = class_pred[name]
        gold = class_gold[name]
        p = tp / pred if pred else 0.0
        r = tp / gold if gold else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        fixed_f1.append(f)
        if gold or pred:
            observed_f1.append(f)
        weighted.append((f, gold))
        class_rows.append({
            "event_name": name,
            "gold_support": gold,
            "prediction_support": pred,
            "tp": tp,
            "fp": max(0, pred - tp),
            "fn": max(0, gold - tp),
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f, 6),
        })
    total_gold = sum(class_gold.values())
    return {
        "threshold": threshold,
        "tp": tp_total,
        "fp": max(0, pred_total - tp_total),
        "fn": max(0, gold_total - tp_total),
        "predicted_event_pair_count": pred_total,
        "gold_event_pair_count": gold_total,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "macro_f1_fixed_64": float(np.mean(fixed_f1)) if fixed_f1 else 0.0,
        "macro_f1_observed": float(np.mean(observed_f1)) if observed_f1 else 0.0,
        "weighted_f1": sum(f * w for f, w in weighted) / total_gold if total_gold else 0.0,
        "article_exact_accuracy": article_exact / len(article_ids) if article_ids else 0.0,
        "at_least_one_correct_article_ratio": at_least_one / len(article_ids) if article_ids else 0.0,
        "complete_miss_article_count": complete_miss,
        "overpredicted_article_count": overpredicted,
        "misclassified_article_count": misclassified,
        "success_no_event_count": success_no_event,
        "class_rows": sorted(class_rows, key=lambda x: (x["gold_support"], x["f1"]), reverse=True),
        "article_rows": article_rows,
        "confusion": confusion,
    }


def attribute_bottlenecks(
    gold_pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    schema_by_id: dict[str, dict[str, Any]],
    article_ids: list[str],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        pred_by_article[row["article_id"]].append(row)
    rows = []
    rank_hits = Counter()
    rank_values = []
    reciprocal_ranks = []
    threshold_pre_recall_hits = 0
    threshold_post_recall_hits = 0
    for gold in gold_pairs:
        aid = gold["article_id"]
        sid = gold["schema_id"]
        candidates = sorted(pred_by_article.get(aid, []), key=lambda x: float(x.get("probability_positive", 0.0)), reverse=True)
        schema_rank = None
        schema_prob = None
        for idx, cand in enumerate(candidates, 1):
            if cand["schema_id"] == sid:
                schema_rank = idx
                schema_prob = float(cand.get("probability_positive", 0.0))
                break
        accepted = [c for c in candidates if float(c.get("probability_positive", 0.0)) >= threshold]
        accepted_schema_ids = {c["schema_id"] for c in accepted}
        top_schema = candidates[0]["schema_id"] if candidates else ""
        top_prob = float(candidates[0].get("probability_positive", 0.0)) if candidates else None
        if schema_rank is None:
            reason = "candidate_missing"
        else:
            rank_values.append(schema_rank)
            reciprocal_ranks.append(1.0 / schema_rank)
            for k in [1, 3, 5, 10]:
                if schema_rank <= k:
                    rank_hits[k] += 1
            threshold_pre_recall_hits += 1
            if sid in accepted_schema_ids:
                threshold_post_recall_hits += 1
                reason = "true_positive"
            elif schema_rank == 1 and (schema_prob or 0.0) < threshold:
                reason = "correct_top1_below_threshold"
            elif schema_rank is not None:
                reason = "candidate_present_classifier_wrong"
            else:
                reason = "unknown"
        rows.append({
            "article_id": aid,
            "schema_id": sid,
            "event_name": schema_by_id.get(sid, {}).get("event_name", gold.get("event_name", "")),
            "target_type": gold.get("target_type", ""),
            "target_code": gold.get("target_code", ""),
            "gold_candidate_rank": schema_rank if schema_rank is not None else "",
            "gold_candidate_probability": round(schema_prob, 6) if schema_prob is not None else "",
            "top_schema_id": top_schema,
            "top_probability": round(top_prob, 6) if top_prob is not None else "",
            "accepted_prediction_count": len(accepted),
            "bottleneck_reason": reason,
        })
    total = len(gold_pairs)
    reason_counts = Counter(row["bottleneck_reason"] for row in rows)
    candidate_metrics = {
        "gold_event_pair_count": total,
        "candidate_recall_at_1": rank_hits[1] / total if total else 0.0,
        "candidate_recall_at_3": rank_hits[3] / total if total else 0.0,
        "candidate_recall_at_5": rank_hits[5] / total if total else 0.0,
        "candidate_recall_at_10": rank_hits[10] / total if total else 0.0,
        "top1_classification_accuracy": rank_hits[1] / total if total else 0.0,
        "gold_candidate_mean_rank": float(np.mean(rank_values)) if rank_values else None,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "threshold_pre_recall": threshold_pre_recall_hits / total if total else 0.0,
        "threshold_post_recall": threshold_post_recall_hits / total if total else 0.0,
        "bottleneck_counts": dict(reason_counts),
    }
    return rows, candidate_metrics


def xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def xlsx_cell(value: Any, row_idx: int, col_idx: int) -> str:
    ref = f"{xlsx_col_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool) and (not isinstance(value, float) or math.isfinite(value)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = xml_escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def worksheet_xml(rows: list[dict[str, Any]], columns: list[str]) -> str:
    sheet_rows = ['<row r="1">' + "".join(xlsx_cell(col, 1, i + 1) for i, col in enumerate(columns)) + "</row>"]
    for r_idx, row in enumerate(rows, 2):
        sheet_rows.append(f'<row r="{r_idx}">' + "".join(xlsx_cell(row.get(col, ""), r_idx, i + 1) for i, col in enumerate(columns)) + "</row>")
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(sheet_rows) + "</sheetData></worksheet>"


def write_xlsx(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> str:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet, rows in sheets.items():
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet[:31], index=False)
        return "openpyxl"
    except ModuleNotFoundError:
        items = list(sheets.items())
        content_types = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        ]
        for i, _ in enumerate(items, 1):
            content_types.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        content_types.append("</Types>")
        workbook_sheets = []
        workbook_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
        for i, (sheet, _) in enumerate(items, 1):
            workbook_sheets.append(f'<sheet name="{xml_escape(sheet[:31])}" sheetId="{i}" r:id="rId{i}"/>')
            workbook_rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
        workbook_rels.append("</Relationships>")
        workbook_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(workbook_sheets) + "</sheets></workbook>"
        root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", "".join(content_types))
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("xl/workbook.xml", workbook_xml)
            zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
            for i, (_, rows) in enumerate(items, 1):
                cols = sorted({k for row in rows for k in row.keys()})
                zf.writestr(f"xl/worksheets/sheet{i}.xml", worksheet_xml(rows, cols))
        return "minimal_xlsx"


def finalize_reports(
    output_dir: Path,
    audit: dict[str, Any],
    split: dict[str, Any],
    model_data: dict[str, Any],
    train_result: dict[str, Any],
    protected_before: dict[str, str],
) -> dict[str, Any]:
    schema_by_id = model_data["schema_by_id"]
    val_predictions = train_result["validation_predictions"]
    val_ids = split["validation_ids"]
    val_pairs = split["validation_pairs"]
    metrics = compute_pair_metrics(val_pairs, val_predictions, schema_by_id, val_ids, THRESHOLD)
    bottleneck_rows, candidate_metrics = attribute_bottlenecks(val_pairs, val_predictions, schema_by_id, val_ids, THRESHOLD)

    write_csv(output_dir / "candidate_recall_metrics.csv", [
        {"metric": k, "value": v} for k, v in candidate_metrics.items() if k != "bottleneck_counts"
    ], ["metric", "value"])
    write_csv(output_dir / "bottleneck_attribution.csv", bottleneck_rows, [
        "article_id", "schema_id", "event_name", "target_type", "target_code",
        "gold_candidate_rank", "gold_candidate_probability", "top_schema_id", "top_probability",
        "accepted_prediction_count", "bottleneck_reason",
    ])
    write_csv(output_dir / "class_metrics_64.csv", metrics["class_rows"], [
        "event_name", "gold_support", "prediction_support", "tp", "fp", "fn", "precision", "recall", "f1",
    ])
    col_labels = sorted({row["event_name"] for row in schema_by_id.values()}) + ["__NO_PRED__"]
    matrix_rows = []
    for gold_name in sorted({row["event_name"] for row in schema_by_id.values()}):
        item = {"gold_event_name": gold_name}
        for col in col_labels:
            item[col] = int(metrics["confusion"][gold_name].get(col, 0))
        matrix_rows.append(item)
    write_csv(output_dir / "confusion_matrix.csv", matrix_rows, ["gold_event_name", *col_labels])

    sweep_rows = []
    best_sweep = None
    for t in [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]:
        m = compute_pair_metrics(val_pairs, val_predictions, schema_by_id, val_ids, t)
        row = {
            "threshold": t,
            "precision": round(m["micro_precision"], 6),
            "recall": round(m["micro_recall"], 6),
            "micro_f1": round(m["micro_f1"], 6),
            "fixed_64_macro_f1": round(m["macro_f1_fixed_64"], 6),
            "article_exact_accuracy": round(m["article_exact_accuracy"], 6),
            "success_no_event_count": m["success_no_event_count"],
        }
        sweep_rows.append(row)
        if best_sweep is None or (row["micro_f1"], row["recall"]) > (best_sweep["micro_f1"], best_sweep["recall"]):
            best_sweep = row
    write_csv(output_dir / "threshold_sweep.csv", sweep_rows, [
        "threshold", "precision", "recall", "micro_f1", "fixed_64_macro_f1", "article_exact_accuracy", "success_no_event_count",
    ])

    article_error_rows = [row for row in metrics["article_rows"] if not row["exact_match"]]
    xlsx_writer = write_xlsx(output_dir / "error_samples.xlsx", {
        "article_errors": article_error_rows,
        "bottlenecks": bottleneck_rows,
        "class_metrics": metrics["class_rows"],
    })

    all_bottleneck_reasons = [
        "candidate_missing",
        "candidate_present_classifier_wrong",
        "correct_top1_below_threshold",
        "correct_candidate_removed_postprocess",
        "export_or_mapping_loss",
        "true_positive",
    ]
    reason_counts = Counter(row["bottleneck_reason"] for row in bottleneck_rows)
    for reason in all_bottleneck_reasons:
        reason_counts.setdefault(reason, 0)
    fn_reasons = {k: v for k, v in reason_counts.items() if k != "true_positive"}
    total_fn = sum(fn_reasons.values())
    fn_reason_rows = [
        {"reason": k, "count": v, "ratio": round(v / total_fn, 6) if total_fn else 0.0}
        for k, v in sorted(fn_reasons.items(), key=lambda x: (-x[1], x[0]))
    ]
    if not fn_reason_rows:
        main_bottleneck = "mixed_bottlenecks"
    else:
        top_reason, top_count = max(fn_reasons.items(), key=lambda x: x[1])
        if top_count / total_fn < 0.5:
            main_bottleneck = "mixed_bottlenecks"
        elif top_reason == "candidate_missing":
            main_bottleneck = "candidate_generation_bottleneck"
        elif top_reason == "candidate_present_classifier_wrong":
            main_bottleneck = "classifier_bottleneck"
        elif top_reason == "correct_top1_below_threshold":
            main_bottleneck = "threshold_bottleneck"
        elif top_reason == "correct_candidate_removed_postprocess":
            main_bottleneck = "postprocess_bottleneck"
        else:
            main_bottleneck = "mixed_bottlenecks"
    protected_after = protected_sha_snapshot()
    report = {
        "generated_at": now(),
        "output_dir": rel(output_dir),
        "training_provenance_conclusion": audit["conclusion"],
        "gold_test_usage": {
            "training": False,
            "model_selection": False,
            "threshold_selection": False,
            "rerun_1029_gold_test": False,
        },
        "development_split": split["manifest"],
        "baseline_training": train_result["summary"],
        "development_validation_metrics_at_current_threshold": {
            k: metrics[k]
            for k in [
                "threshold", "tp", "fp", "fn", "predicted_event_pair_count", "gold_event_pair_count",
                "micro_precision", "micro_recall", "micro_f1", "macro_f1_fixed_64",
                "macro_f1_observed", "weighted_f1", "article_exact_accuracy",
                "success_no_event_count",
            ]
        },
        "candidate_recall_metrics": candidate_metrics,
        "fn_bottleneck_distribution": fn_reason_rows,
        "threshold_sweep_best_theoretical": best_sweep,
        "main_bottleneck_conclusion": main_bottleneck,
        "error_samples_xlsx_writer": xlsx_writer,
        "formal_file_sha256_before": protected_before,
        "formal_file_sha256_after": protected_after,
        "formal_file_sha256_unchanged": protected_before == protected_after,
        "calls": {
            "dianjin_r1_32b_calls": 0,
            "external_llm_calls": 0,
        },
        "notes": [
            "本步骤dev指标不是最终test成绩。",
            "1029篇gold test标签未用于训练、checkpoint选择或阈值推广。",
        ],
    }
    write_json(output_dir / "step5_4_report.json", report)
    md = [
        "# Step 5.4 Task1 无泄漏训练/验证基线与低召回瓶颈归因",
        "",
        f"- 现有 best_model 训练来源结论：{audit['conclusion']}。",
        f"- development_train：{split['manifest']['development_train']['article_count']}篇，{split['manifest']['development_train']['event_pair_count']}个事件对。",
        f"- development_validation：{split['manifest']['development_validation']['article_count']}篇，{split['manifest']['development_validation']['event_pair_count']}个事件对。",
        f"- dev micro-F1：{metrics['micro_f1']:.6f}；fixed-64 macro-F1：{metrics['macro_f1_fixed_64']:.6f}；weighted-F1：{metrics['weighted_f1']:.6f}。",
        f"- candidate recall@1/@3/@5/@10：{candidate_metrics['candidate_recall_at_1']:.6f}/{candidate_metrics['candidate_recall_at_3']:.6f}/{candidate_metrics['candidate_recall_at_5']:.6f}/{candidate_metrics['candidate_recall_at_10']:.6f}。",
        f"- 主要瓶颈：{main_bottleneck}。",
        f"- 阈值扫描最佳理论：threshold={best_sweep['threshold']} micro-F1={best_sweep['micro_f1']} recall={best_sweep['recall']}。",
        f"- 正式文件SHA保持：{report['formal_file_sha256_unchanged']}。",
        f"- 32B调用次数：0。",
        "",
        "## FN瓶颈分布",
        "",
    ]
    for row in fn_reason_rows:
        md.append(f"- {row['reason']}：{row['count']} ({row['ratio']:.6f})。")
    md.extend([
        "",
        "## 说明",
        "",
        "本报告只使用gold train内部开发划分；dev指标不得写成最终gold test成绩。",
        "Step 5.3完整gold test预测和正式Task1/Task2文件未修改。",
    ])
    (output_dir / "step5_4_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def write_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS.txt"}:
            rel_path = path.relative_to(output_dir).as_posix()
            files.append({"path": rel_path, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    with (output_dir / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for item in files:
            handle.write(f"{item['sha256']}  {item['path']}\n")
    files.append({"path": "SHA256SUMS.txt", "size_bytes": (output_dir / "SHA256SUMS.txt").stat().st_size, "sha256": sha256_file(output_dir / "SHA256SUMS.txt")})
    manifest = {
        "manifest_version": "task1_step5_4_dev_baseline_v1",
        "generated_at": now(),
        "output_dir": rel(output_dir),
        "file_count": len(files),
        "files": files,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def run(output_dir: Path, resume: bool = False, audit_only: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "step5_4_report.json").exists() and not resume:
        raise FileExistsError(f"existing Step 5.4 output without --resume: {output_dir}")
    protected_before = protected_sha_snapshot()
    audit = audit_existing_training(output_dir)
    if audit["conclusion"] == "test_leakage":
        write_manifest(output_dir)
        return {"status": "stopped_test_leakage", "audit": audit}
    if audit_only:
        write_manifest(output_dir)
        return {"status": "audit_only", "audit": audit}
    split = build_dev_split(output_dir)
    model_data = prepare_model_data(split, output_dir)
    train_result = train_baseline(output_dir, model_data)
    report = finalize_reports(output_dir, audit, split, model_data, train_result, protected_before)
    write_manifest(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = run(args.output_dir.resolve(), resume=args.resume, audit_only=args.audit_only)
    if report.get("status", ""):
        print(json.dumps({"status": report["status"], "conclusion": report["audit"]["conclusion"]}, ensure_ascii=False))
    else:
        print(json.dumps({
            "output_dir": str(args.output_dir.resolve()),
            "training_provenance": report["training_provenance_conclusion"],
            "dev_articles": report["development_split"]["development_validation"]["article_count"],
            "dev_micro_f1": report["development_validation_metrics_at_current_threshold"]["micro_f1"],
            "candidate_recall_at_1": report["candidate_recall_metrics"]["candidate_recall_at_1"],
            "main_bottleneck": report["main_bottleneck_conclusion"],
            "formal_sha_unchanged": report["formal_file_sha256_unchanged"],
            "dianjin_r1_32b_calls": report["calls"]["dianjin_r1_32b_calls"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
