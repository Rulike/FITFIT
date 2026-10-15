#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for Task1 Step 5.8-HJ training and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_task1_official_format_shadow import official_enums_and_maps  # noqa: E402
from scripts.build_task1_step5_8hc_taxonomy import (  # noqa: E402
    BACKUP_ROOT,
    CONFIG_JSON,
    INCREMENTAL_XLSX,
    MANUAL_5000_FALLBACK_XLSX,
    MANUAL_5000_LOCAL_XLSX,
    MANUAL_5000_PRIMARY_XLSX,
    OFFICIAL_TRAIN_XLSX,
    SHADOW_BASE_JSONL,
    cell_value,
    load_incremental_rows,
    load_manual_5000_rows,
    load_official_rows,
    now_iso,
    read_jsonl,
    resolve_manual_5000_xlsx,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from scripts.run_task1_step5_8e_final_optimization import (  # noqa: E402
    deterministic_duplication,
    evidence_quote,
    impact_text,
    metric_dict,
    per_class_rows,
    predict_allowed_event,
    predict_emotion,
    row_text,
    split_official_rows,
    train_emotion_models,
    train_sgd_pipeline,
)
from scripts.train_task1_step5_8hc_hierarchical_model import combine_rows, split_all_rows  # noqa: E402
from src.routing.task1_candidate_v2 import Task1CandidateGeneratorV2  # noqa: E402

LOCAL_MACBERT = ROOT / "outputs/task1_step5_4_dev_baseline_v1/pretrained/hfl_chinese_macbert_base"
OUT_DIR = ROOT / "outputs/task1_step5_8hj_training"
MODEL_DIR = OUT_DIR / "models"
REPORT_DIR = ROOT / "reports"
DOC_DIR = ROOT / "docs"
OFFICIAL_TEST_XLSX = ROOT / "data/raw/news_with_tags_test.xlsx"
UNLABELED_TEST_XLSX = ROOT / "data/raw/news_without_tags_test.xlsx"
OFFICIAL_TRAIN_JSONL = ROOT / "data/processed/labeled/news_train.jsonl"
OFFICIAL_TRAIN_PAIRS = ROOT / "data/processed/labeled/event_pairs_train.jsonl"
EVENT_SCHEMA_JSONL = ROOT / "data/processed/event_schema.jsonl"
TARGET_ALIAS_JSON = ROOT / "configs/target_aliases.json"
HIER_MODEL_PKL = ROOT / "outputs/task1_step5_8hc_hierarchical_event/models/best_hierarchical_model.pkl"
FROZEN_EMOTION_MODEL = ROOT / "outputs/task1_step5_8f_final_candidate/frozen_emotion_duplication/event_emotion_model.pkl"
HIST_HF_REPORT = ROOT / "reports/task1_step5_8hf_retrospective_official_test_report.json"
PER_TAG_REPORT = ROOT / "reports/task1_step5_8hi_per_tag_metrics.xlsx"

SEEDS = [2026, 2027, 2028]
MAIN_SHEETS = {"个股新闻", "行业新闻"}
COMPANY_SHEET = "个股新闻"
INDUSTRY_SHEET = "行业新闻"
LEGAL_EMOTIONS = ["正面", "中性", "负面"]


def sha256_file_text(path: Path) -> str:
    return sha256_file(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def norm_code(value: Any, width: int | None = 6) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if width and text.isdigit():
        text = text.zfill(width)
    return text


def article_group_key(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("article_file_id") or ""),
        str(row.get("duplication_id") or ""),
        str(row.get("content_sha256") or ""),
        str(row.get("title_sha256") or row.get("title_norm_sha256") or ""),
    ]
    return "|".join(parts)


def build_article_excerpt(row: dict[str, Any], max_chars: int = 1800) -> str:
    title = str(row.get("article_title") or row.get("title") or "")
    content = str(row.get("content") or "")
    if len(content) <= max_chars:
        body = content
    else:
        head = content[: max_chars // 2]
        tail = content[-max(0, max_chars // 3) :]
        body = f"{head}\n{tail}"
    return "\n".join(
        [
            f"标题：{title}",
            f"来源：{row.get('article_source') or row.get('source') or ''}",
            f"正文：{body}",
        ]
    )


def build_event_pair_text(row: dict[str, Any], event_name: str, event_desc: str, event_examples: str = "") -> str:
    return "\n".join(
        [
            f"[事件候选] {event_name}",
            f"定义：{event_desc}",
            f"示例：{event_examples}",
            build_article_excerpt(row),
        ]
    )


def build_emotion_text(row: dict[str, Any], oof_event_name: str) -> str:
    target = str(row.get("secu_abbr") or row.get("industry_name") or "")
    target_type = str(row.get("target_type") or ("company" if str(row.get("trading_code") or "") else "industry"))
    return "\n".join(
        [
            f"主体类型：{target_type}",
            f"主体：{target}",
            f"事件：{oof_event_name}",
            build_article_excerpt(row, 1300),
        ]
    )


def build_target_text(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[候选主体] {candidate.get('target_type', '')}",
            f"代码：{candidate.get('target_code', '')}",
            f"名称：{candidate.get('target_name', '')}",
            f"别名：{candidate.get('matched_alias', '')}",
            f"命中方式：{candidate.get('route_method', '')}",
            f"置信度：{candidate.get('route_confidence', 0.0)}",
            build_article_excerpt(row),
        ]
    )


def build_taxonomy_maps() -> dict[str, Any]:
    config = read_json(CONFIG_JSON)
    leaf_nodes = config.get("leaf_nodes", [])
    leaf_by_name = {str(node["event_name"]): node for node in leaf_nodes if node.get("event_name")}
    official64_names = [str(node["event_name"]) for node in leaf_nodes if node.get("label_namespace") == "official64"]
    trainable_leaf_names = [str(node["event_name"]) for node in leaf_nodes if str(node.get("event_layer") or "") != "unresolved_meta"]
    l1_names = [str(node["l1_parent_name"]) for node in config.get("l1_nodes", []) if node.get("l1_parent_name")]
    l2_names = [str(node["l2_group_name"]) for node in config.get("l2_nodes", []) if node.get("l2_group_name")]
    leaves_by_l1: dict[str, list[str]] = defaultdict(list)
    leaves_by_l2: dict[str, list[str]] = defaultdict(list)
    leaves_by_layer: dict[str, list[str]] = defaultdict(list)
    leaves_by_target: dict[str, list[str]] = defaultdict(list)
    for node in leaf_nodes:
        name = str(node.get("event_name") or "")
        if not name:
            continue
        leaves_by_l1[str(node.get("l1_parent_name") or "")].append(name)
        leaves_by_l2[str(node.get("l2_group_name") or "")].append(name)
        leaves_by_layer[str(node.get("event_layer") or "")].append(name)
        scope = str(node.get("target_scope") or "both")
        if scope == "both":
            leaves_by_target["company"].append(name)
            leaves_by_target["industry"].append(name)
        elif scope in {"company", "industry"}:
            leaves_by_target[scope].append(name)
        else:
            leaves_by_target["company"].append(name)
            leaves_by_target["industry"].append(name)
    return {
        "config": config,
        "leaf_by_name": leaf_by_name,
        "official64_names": sorted(official64_names),
        "trainable_leaf_names": sorted(trainable_leaf_names),
        "l1_names": sorted(l1_names),
        "l2_names": sorted(l2_names),
        "leaves_by_l1": {k: sorted(set(v)) for k, v in leaves_by_l1.items()},
        "leaves_by_l2": {k: sorted(set(v)) for k, v in leaves_by_l2.items()},
        "leaves_by_layer": {k: sorted(set(v)) for k, v in leaves_by_layer.items()},
        "leaves_by_target": {k: sorted(set(v)) for k, v in leaves_by_target.items()},
        "source_forward_map": dict(config.get("source_label_forward_map", {})),
    }


def build_training_rows() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    taxonomy = build_taxonomy_maps()
    official_rows = load_official_rows()
    incremental_rows, incremental_stats = load_incremental_rows()
    manual_5000_rows, manual_5000_stats = load_manual_5000_rows()
    rows = combine_rows(official_rows, incremental_rows + manual_5000_rows, taxonomy)
    rows_by_split, split_manifest = split_all_rows(rows)
    if "train" not in rows_by_split:
        rows_by_split = {
            "train": list(rows_by_split.get("hierarchical_train", [])),
            "model_selection_dev": list(rows_by_split.get("hierarchical_dev", [])),
            "untouched_internal_holdout": list(rows_by_split.get("hierarchical_internal_holdout", [])),
            **{k: v for k, v in rows_by_split.items() if k not in {"hierarchical_train", "hierarchical_dev", "hierarchical_internal_holdout"}},
        }
    source_stats = {
        "official_rows": len(official_rows),
        "incremental_rows": incremental_stats.get("rows", len(incremental_rows)),
        "manual_5000_rows": manual_5000_stats.get("rows", len(manual_5000_rows)),
    }
    return rows, rows_by_split, {"official": official_rows, "incremental": incremental_rows, "manual5000": manual_5000_rows}, {
        "taxonomy": taxonomy,
        "split_manifest": split_manifest,
        "source_stats": source_stats,
        "incremental_stats": incremental_stats,
        "manual5000_stats": manual_5000_stats,
    }


def load_candidate_generator(train_rows: list[dict[str, Any]]) -> Task1CandidateGeneratorV2:
    train_articles = []
    train_pairs = []
    seen_articles: set[str] = set()
    for row in train_rows:
        aid = str(row.get("article_file_id") or row.get("article_id") or "")
        if not aid:
            continue
        if aid not in seen_articles:
            train_articles.append(
                {
                    "article_id": aid,
                    "title": str(row.get("article_title") or row.get("title") or ""),
                    "content": str(row.get("content") or ""),
                    "publish_time": str(row.get("article_publish_time") or row.get("publish_time") or ""),
                    "source": str(row.get("article_source") or row.get("source") or ""),
                }
            )
            seen_articles.add(aid)
        train_pairs.append(
            {
                "article_id": aid,
                "event_name": str(row.get("event_name") or ""),
                "target_type": str(row.get("target_type") or ("company" if str(row.get("trading_code") or "") else "industry")),
                "target_code": str(row.get("trading_code") or row.get("industry_code") or ""),
                "target_name": str(row.get("secu_abbr") or row.get("industry_name") or ""),
                "title": str(row.get("article_title") or row.get("title") or ""),
                "content": str(row.get("content") or ""),
            }
        )
    return Task1CandidateGeneratorV2.from_paths(
        EVENT_SCHEMA_JSONL,
        train_articles,
        train_pairs,
        alias_path=TARGET_ALIAS_JSON,
        config=read_json(ROOT / "configs/task1_candidate_v2.yaml") if (ROOT / "configs/task1_candidate_v2.yaml").exists() else {},
    )


def load_official_schemas() -> tuple[list[str], list[str], dict[str, dict[str, str]], dict[str, str]]:
    official_event_names, official_emotions, company_map, industry_map = official_enums_and_maps()
    return sorted(official_event_names), sorted(official_emotions), company_map, industry_map


def event_schema_map() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(EVENT_SCHEMA_JSONL)
    return {str(row["event_name"]): row for row in rows if row.get("event_name")}


def allowed_official_events_by_type() -> dict[str, list[str]]:
    schema = event_schema_map()
    company = sorted({name for name, row in schema.items() if str(row.get("target_type") or "").find("company") >= 0 or str(row.get("target_type") or "").find("标的") >= 0})
    industry = sorted({name for name, row in schema.items() if str(row.get("target_type") or "").find("industry") >= 0 or str(row.get("target_type") or "").find("行业") >= 0})
    return {"company": company, "industry": industry}


def build_target_rank_examples(
    rows: list[dict[str, Any]],
    generator: Task1CandidateGeneratorV2,
    per_row_negatives: int = 4,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for row in rows:
        article = {
            "article_id": str(row.get("article_file_id") or row.get("article_id") or ""),
            "title": str(row.get("article_title") or row.get("title") or ""),
            "content": str(row.get("content") or ""),
            "publish_time": str(row.get("article_publish_time") or row.get("publish_time") or ""),
            "source": str(row.get("article_source") or row.get("source") or ""),
        }
        candidates, _ = generator.generate(article, include_all64=True)
        gold_type = str(row.get("target_type") or ("company" if str(row.get("trading_code") or "") else "industry"))
        gold_code = norm_code(row.get("trading_code") or row.get("industry_code") or "")
        gold_name = norm_text(row.get("secu_abbr") or row.get("industry_name") or "")
        pos = []
        neg = []
        for cand in candidates:
            cand_type = str(cand.get("target_type") or "")
            cand_code = norm_code(cand.get("target_code"))
            cand_name = norm_text(cand.get("target_name"))
            is_pos = cand_type == gold_type and cand_code == gold_code and cand_name == gold_name
            item = {
                "article_id": article["article_id"],
                "target_type": cand_type,
                "target_code": cand_code,
                "target_name": cand_name,
                "matched_alias": str(cand.get("matched_alias") or ""),
                "route_method": str(cand.get("route_method") or ""),
                "route_confidence": float(cand.get("route_confidence") or 0.0),
                "text": build_target_text(row, cand),
                "label": 1 if is_pos else 0,
            }
            if is_pos:
                pos.append(item)
            else:
                neg.append(item)
        if not pos:
            pos = [{
                "article_id": article["article_id"],
                "target_type": gold_type,
                "target_code": gold_code,
                "target_name": gold_name,
                "matched_alias": gold_name,
                "route_method": "gold_fallback",
                "route_confidence": 1.0,
                "text": build_target_text(row, {"target_type": gold_type, "target_code": gold_code, "target_name": gold_name, "matched_alias": gold_name, "route_method": "gold_fallback", "route_confidence": 1.0}),
                "label": 1,
            }]
        neg.sort(key=lambda x: (-x["route_confidence"], x["target_type"], x["target_code"], x["target_name"]))
        if len(neg) > per_row_negatives:
            hard = neg[: per_row_negatives - 1]
            rand_pick = rng.choice(neg[per_row_negatives - 1 :]) if len(neg) > per_row_negatives - 1 else None
            neg = hard + ([rand_pick] if rand_pick else [])
        examples.extend(pos + neg)
    rng.shuffle(examples)
    return examples


def build_event_rank_examples(
    rows: list[dict[str, Any]],
    generator: Task1CandidateGeneratorV2,
    allowed_by_type: dict[str, list[str]],
    per_row_negatives: int = 4,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    schema = event_schema_map()
    examples: list[dict[str, Any]] = []
    for row in rows:
        target_type = str(row.get("target_type") or ("company" if str(row.get("trading_code") or "") else "industry"))
        article = {
            "article_id": str(row.get("article_file_id") or row.get("article_id") or ""),
            "title": str(row.get("article_title") or row.get("title") or ""),
            "content": str(row.get("content") or ""),
            "publish_time": str(row.get("article_publish_time") or row.get("publish_time") or ""),
            "source": str(row.get("article_source") or row.get("source") or ""),
        }
        candidates, _ = generator.generate(article, include_all64=True)
        gold = str(row.get("event_name") or "")
        gold_desc = str(schema.get(gold, {}).get("raw_description") or schema.get(gold, {}).get("description") or "")
        gold_examples = str(schema.get(gold, {}).get("raw_examples") or schema.get(gold, {}).get("examples") or "")
        pos = [{
            "article_id": article["article_id"],
            "event_name": gold,
            "target_type": target_type,
            "text": build_event_pair_text(row, gold, gold_desc, gold_examples),
            "label": 1,
        }]
        neg = []
        same_type_pool = [ev for ev in allowed_by_type.get(target_type, []) if ev != gold]
        cand_order = [str(c.get("event_name") or "") for c in candidates if str(c.get("event_name") or "") != gold and str(c.get("event_name") or "") in same_type_pool]
        if not cand_order:
            cand_order = same_type_pool
        seen = set()
        for ev in cand_order:
            if ev in seen or ev == gold:
                continue
            seen.add(ev)
            info = schema.get(ev, {})
            neg.append(
                {
                    "article_id": article["article_id"],
                    "event_name": ev,
                    "target_type": target_type,
                    "text": build_event_pair_text(row, ev, str(info.get("raw_description") or info.get("description") or ""), str(info.get("raw_examples") or info.get("examples") or "")),
                    "label": 0,
                }
            )
            if len(neg) >= per_row_negatives:
                break
        if len(neg) < per_row_negatives:
            same_l1 = []
            same_l2 = []
            gold_leaf = None
            # the taxonomy map is only available during orchestration; fall back to random negatives here.
        examples.extend(pos + neg)
    rng.shuffle(examples)
    return examples


def build_emotion_examples(rows: list[dict[str, Any]], oof_event_map: dict[str, str], seed: int = 2026) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for row in rows:
        aid = str(row.get("article_file_id") or row.get("article_id") or "")
        examples.append(
            {
                "article_id": aid,
                "label": str(row.get("event_emotion") or ""),
                "text": build_emotion_text(row, oof_event_map.get(aid, str(row.get("event_name") or ""))),
            }
        )
    rng.shuffle(examples)
    return examples


def groups_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [article_group_key(row) for row in rows]


def train_linear_target_ranker(
    train_rows: list[dict[str, Any]],
    seed: int = 2026,
    analyzer: str = "char",
    ngram_range: tuple[int, int] = (2, 4),
    max_features: int = 80000,
) -> Pipeline:
    generator = load_candidate_generator(train_rows)
    examples = build_target_rank_examples(train_rows, generator, per_row_negatives=4, seed=seed)
    if analyzer == "hash_char":
        vectorizer = HashingVectorizer(analyzer="char", ngram_range=ngram_range, n_features=2**18, alternate_sign=False, norm="l2")
    else:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1, max_features=max_features, sublinear_tf=True)
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, class_weight="balanced", max_iter=1400, tol=1e-3, random_state=seed, n_jobs=4)
    model = Pipeline([("vectorizer", vectorizer), ("classifier", clf)])
    model.fit([x["text"] for x in examples], [x["label"] for x in examples])
    return model


def predict_target_ranker(
    model: Pipeline,
    rows: list[dict[str, Any]],
    generator: Task1CandidateGeneratorV2,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        article = {
            "article_id": str(row.get("article_file_id") or row.get("article_id") or ""),
            "title": str(row.get("article_title") or row.get("title") or ""),
            "content": str(row.get("content") or ""),
            "publish_time": str(row.get("article_publish_time") or row.get("publish_time") or ""),
            "source": str(row.get("article_source") or row.get("source") or ""),
        }
        candidates, _ = generator.generate(article, include_all64=True)
        if not candidates:
            out.append(
                {
                    "target_type": "missing_prediction",
                    "target_code": "missing_prediction",
                    "target_name": "missing_prediction",
                    "route_confidence": 0.0,
                    "candidate_count": 0,
                }
            )
            continue
        texts = [build_target_text(row, cand) for cand in candidates]
        probs = model.predict_proba(texts)[:, 1]
        best_idx = int(np.argmax(probs))
        best = dict(candidates[best_idx])
        best["score"] = float(probs[best_idx])
        best["candidate_count"] = len(candidates)
        out.append(best)
    return out


def train_linear_event_ranker(
    train_rows: list[dict[str, Any]],
    seed: int = 2026,
    analyzer: str = "char",
    ngram_range: tuple[int, int] = (2, 4),
    max_features: int = 80000,
) -> Pipeline:
    y = [str(r.get("event_name") or "") for r in train_rows]
    if analyzer == "hash_char":
        vectorizer = HashingVectorizer(analyzer="char", ngram_range=ngram_range, n_features=2**18, alternate_sign=False, norm="l2")
    else:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1, max_features=max_features, sublinear_tf=True)
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, class_weight="balanced", max_iter=1400, tol=1e-3, random_state=seed, n_jobs=4)
    model = Pipeline([("vectorizer", vectorizer), ("classifier", clf)])
    model.fit([row_text(r) for r in train_rows], y)
    return model


def predict_linear_event(
    model: Pipeline,
    rows: list[dict[str, Any]],
    allowed_by_type: dict[str, list[str]],
) -> tuple[list[str], list[list[str]], list[float]]:
    classes = list(model.named_steps["classifier"].classes_)
    probs = model.predict_proba([row_text(r) for r in rows])
    preds: list[str] = []
    top20: list[list[str]] = []
    confs: list[float] = []
    for row, prob in zip(rows, probs):
        allowed = set(allowed_by_type.get(str(row.get("target_type") or "company"), []))
        order = sorted(range(len(classes)), key=lambda i: float(prob[i]), reverse=True)
        allowed_order = [classes[i] for i in order if classes[i] in allowed]
        if not allowed_order:
            allowed_order = [classes[i] for i in order]
        pred = allowed_order[0] if allowed_order else ""
        preds.append(pred)
        top20.append(allowed_order[:20])
        confs.append(float(prob[classes.index(pred)]) if pred in classes else float(np.max(prob)))
    return preds, top20, confs


def build_event_oof_predictions(
    train_rows: list[dict[str, Any]],
    allowed_by_type: dict[str, list[str]],
    seed: int = 2026,
    n_splits: int = 3,
) -> tuple[dict[str, str], dict[str, Any]]:
    groups = groups_for_rows(train_rows)
    labels = [str(r.get("event_name") or "") for r in train_rows]
    splitter = GroupKFold(n_splits=n_splits)
    oof: dict[str, str] = {}
    fold_metrics = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(train_rows, labels, groups=groups), 1):
        tr_rows = [train_rows[i] for i in tr_idx]
        va_rows = [train_rows[i] for i in va_idx]
        model = train_linear_event_ranker(tr_rows, seed=seed + fold_idx)
        pred, _, _ = predict_linear_event(model, va_rows, allowed_by_type)
        for row, p in zip(va_rows, pred):
            oof[str(row.get("article_file_id") or row.get("article_id") or "")] = p
        fold_metrics.append(
            {
                "fold": fold_idx,
                "support": len(va_rows),
                "micro_f1": round(float(f1_score([str(r.get("event_name") or "") for r in va_rows], pred, average="micro", zero_division=0)), 6),
                "macro_f1": round(float(f1_score([str(r.get("event_name") or "") for r in va_rows], pred, average="macro", zero_division=0)), 6),
            }
        )
    return oof, {"fold_metrics": fold_metrics, "fold_count": n_splits}


def event_metrics_from_predictions(
    rows: list[dict[str, Any]],
    pred_leaf: list[str],
    pred_top20: list[list[str]],
    pred_conf: list[float],
    official_labels: list[str],
    company_labels: list[str],
    industry_labels: list[str],
) -> dict[str, Any]:
    y_true = [str(r.get("event_name") or "") for r in rows]
    labels = sorted(set(y_true) | set(pred_leaf) | set(official_labels))
    p, r, f, s = precision_recall_fscore_support(y_true, pred_leaf, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, pred_leaf, labels=labels)
    company_idx = [i for i, row in enumerate(rows) if str(row.get("target_type") or "") == "company"]
    industry_idx = [i for i, row in enumerate(rows) if str(row.get("target_type") or "") == "industry"]
    out = {
        "support": len(rows),
        "micro_precision": round(float(precision_recall_fscore_support(y_true, pred_leaf, labels=labels, average="micro", zero_division=0)[0]), 6),
        "micro_recall": round(float(precision_recall_fscore_support(y_true, pred_leaf, labels=labels, average="micro", zero_division=0)[1]), 6),
        "micro_f1": round(float(precision_recall_fscore_support(y_true, pred_leaf, labels=labels, average="micro", zero_division=0)[2]), 6),
        "fixed_64_macro_f1": round(float(f1_score(y_true, pred_leaf, labels=official_labels, average="macro", zero_division=0)), 6),
        "observed_label_macro_f1": round(float(f1_score(y_true, pred_leaf, labels=sorted(set(y_true)), average="macro", zero_division=0)), 6) if y_true else 0.0,
        "weighted_f1": round(float(f1_score(y_true, pred_leaf, labels=labels, average="weighted", zero_division=0)), 6),
        "company44_macro_f1": round(float(f1_score([y_true[i] for i in company_idx], [pred_leaf[i] for i in company_idx], labels=company_labels, average="macro", zero_division=0)), 6) if company_idx else 0.0,
        "industry20_macro_f1": round(float(f1_score([y_true[i] for i in industry_idx], [pred_leaf[i] for i in industry_idx], labels=industry_labels, average="macro", zero_division=0)), 6) if industry_idx else 0.0,
        "article_exact_match": round(float(accuracy_score(y_true, pred_leaf)), 6) if y_true else 0.0,
        "at_least_one_hit_rate": round(float(np.mean([g in top for g, top in zip(y_true, pred_top20)])), 6) if y_true else 0.0,
        "complete_miss_count": int(sum(1 for x in pred_leaf if not x)),
        "multi_prediction_count": 0,
        "misclassification_count": int(sum(1 for g, p_ in zip(y_true, pred_leaf) if g != p_)),
        "per_class": [
            {"label": labels[i], "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "f1": round(float(f[i]), 6), "support": int(s[i])}
            for i in range(len(labels))
        ],
        "confusion_matrix": {labels[i]: {labels[j]: int(cm[i, j]) for j in range(len(labels))} for i in range(len(labels))},
        "candidate_recall_at_20": round(float(np.mean([g in top for g, top in zip(y_true, pred_top20)])), 6) if y_true else 0.0,
        "illegal_event_name": int(sum(1 for x in pred_leaf if x not in official_labels)),
        "top1_confidence_mean": round(float(np.mean(pred_conf)), 6) if pred_conf else 0.0,
    }
    return out


def target_metrics_from_predictions(pred_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = []
    y_pred = []
    company_idx = []
    industry_idx = []
    missing = wrong = extra = 0
    for idx, (pred, gold) in enumerate(zip(pred_rows, gold_rows)):
        gt_type = str(gold.get("target_type") or ("company" if str(gold.get("trading_code") or "") else "industry"))
        gp = gt_type
        pp = str(pred.get("target_type") or "missing_prediction")
        y_true.append(gp)
        y_pred.append(pp)
        if gt_type == "company":
            company_idx.append(idx)
        else:
            industry_idx.append(idx)
        if not pp or pp == "missing_prediction":
            missing += 1
        elif pp != gp:
            wrong += 1
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "support": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6) if y_true else 0.0,
        "micro_precision": round(float(precision_recall_fscore_support(y_true, y_pred, labels=labels, average="micro", zero_division=0)[0]), 6) if y_true else 0.0,
        "micro_recall": round(float(precision_recall_fscore_support(y_true, y_pred, labels=labels, average="micro", zero_division=0)[1]), 6) if y_true else 0.0,
        "micro_f1": round(float(precision_recall_fscore_support(y_true, y_pred, labels=labels, average="micro", zero_division=0)[2]), 6) if y_true else 0.0,
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6) if y_true else 0.0,
        "missing_prediction_count": missing,
        "wrong_prediction_count": wrong,
        "extra_prediction_count": extra,
        "per_class": [
            {"label": labels[i], "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "f1": round(float(f[i]), 6), "support": int(s[i])}
            for i in range(len(labels))
        ],
        "confusion_matrix": {labels[i]: {labels[j]: int(cm[i, j]) for j in range(len(labels))} for i in range(len(labels))},
    }


def infer_main_sheet(row: dict[str, Any], default: str = COMPANY_SHEET) -> str:
    sheet = str(row.get("sheet_name") or "")
    if sheet in MAIN_SHEETS:
        return sheet
    target_type = str(row.get("target_type") or "")
    if target_type == "company" or str(row.get("trading_code") or ""):
        return COMPANY_SHEET
    if target_type == "industry" or str(row.get("industry_code") or ""):
        return INDUSTRY_SHEET
    return default


def target_full_metrics(pred_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = list(zip(pred_rows, gold_rows))
    company_pred = [r for r, g in rows if infer_main_sheet(g) == COMPANY_SHEET]
    company_gold = [g for _, g in rows if infer_main_sheet(g) == COMPANY_SHEET]
    industry_pred = [r for r, g in rows if infer_main_sheet(g) == INDUSTRY_SHEET]
    industry_gold = [g for _, g in rows if infer_main_sheet(g) == INDUSTRY_SHEET]

    def _field(rows_pred: list[dict[str, Any]], rows_gold: list[dict[str, Any]], field: str, is_code: bool = False) -> dict[str, Any]:
        y_true = [norm_code(g.get(field)) if is_code else norm_text(g.get(field)) for g in rows_gold]
        y_pred = [norm_code(p.get(field)) if is_code else norm_text(p.get(field)) for p in rows_pred]
        labels = sorted(set(y_true) | set(y_pred))
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "support": len(y_true),
            "exact_accuracy": round(float(accuracy_score(y_true, y_pred)), 6) if y_true else 0.0,
            "normalized_accuracy": round(float(accuracy_score(y_true, y_pred)), 6) if y_true else 0.0,
            "missing_prediction_count": int(sum(1 for x in y_pred if not x)),
            "wrong_prediction_count": int(sum(1 for g, p_ in zip(y_true, y_pred) if g != p_)),
            "extra_prediction_count": 0,
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6) if y_true else 0.0,
            "per_class": [
                {"label": labels[i], "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "f1": round(float(f[i]), 6), "support": int(s[i])}
                for i in range(len(labels))
            ],
            "confusion_matrix": {labels[i]: {labels[j]: int(cm[i, j]) for j in range(len(labels))} for i in range(len(labels))},
        }

    main_pred = [r for r, g in rows if infer_main_sheet(g) in MAIN_SHEETS]
    main_gold = [g for _, g in rows if infer_main_sheet(g) in MAIN_SHEETS]
    target_type_true = ["company" if infer_main_sheet(g) == COMPANY_SHEET else "industry" for g in main_gold]
    target_type_pred = [str(p.get("target_type") or "missing_prediction") for p in main_pred]
    target_labels = sorted(set(target_type_true) | set(target_type_pred))
    tp, tr, tf, ts = precision_recall_fscore_support(target_type_true, target_type_pred, labels=target_labels, zero_division=0)
    target_cm = confusion_matrix(target_type_true, target_type_pred, labels=target_labels)

    return {
        "target_type": {
            "support": len(target_type_true),
            "accuracy": round(float(accuracy_score(target_type_true, target_type_pred)), 6) if target_type_true else 0.0,
            "micro_precision": round(float(precision_recall_fscore_support(target_type_true, target_type_pred, labels=target_labels, average="micro", zero_division=0)[0]), 6) if target_type_true else 0.0,
            "micro_recall": round(float(precision_recall_fscore_support(target_type_true, target_type_pred, labels=target_labels, average="micro", zero_division=0)[1]), 6) if target_type_true else 0.0,
            "micro_f1": round(float(precision_recall_fscore_support(target_type_true, target_type_pred, labels=target_labels, average="micro", zero_division=0)[2]), 6) if target_type_true else 0.0,
            "macro_f1": round(float(f1_score(target_type_true, target_type_pred, labels=target_labels, average="macro", zero_division=0)), 6) if target_type_true else 0.0,
            "per_class": [
                {"label": target_labels[i], "precision": round(float(tp[i]), 6), "recall": round(float(tr[i]), 6), "f1": round(float(tf[i]), 6), "support": int(ts[i])}
                for i in range(len(target_labels))
            ],
            "confusion_matrix": {target_labels[i]: {target_labels[j]: int(target_cm[i, j]) for j in range(len(target_labels))} for i in range(len(target_labels))},
        },
        "company": {
            "trading_code": _field(company_pred, company_gold, "trading_code", is_code=True),
            "secu_abbr": _field(company_pred, company_gold, "secu_abbr"),
        },
        "industry": {
            "industry_code": _field(industry_pred, industry_gold, "industry_code", is_code=True),
            "industry_name": _field(industry_pred, industry_gold, "industry_name"),
        },
        "main_target_exact": round(
            float(
                np.mean(
                    [
                        (
                            (infer_main_sheet(g) == COMPANY_SHEET and norm_code(p.get("trading_code")) == norm_code(g.get("trading_code")) and norm_text(p.get("secu_abbr")) == norm_text(g.get("secu_abbr")))
                            or (infer_main_sheet(g) == INDUSTRY_SHEET and norm_code(p.get("industry_code")) == norm_code(g.get("industry_code")) and norm_text(p.get("industry_name")) == norm_text(g.get("industry_name")))
                        )
                        for p, g in rows
                        if infer_main_sheet(g) in MAIN_SHEETS
                    ]
                )
            ),
            6,
        )
        if rows
        else 0.0,
    }



def code_metrics(pred_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]], field: str, target_sheet: str) -> dict[str, Any]:
    y_true = []
    y_pred = []
    for pred, gold in zip(pred_rows, gold_rows):
        if infer_main_sheet(gold, default=target_sheet) != target_sheet:
            continue
        if field == "trading_code":
            y_true.append(norm_code(gold.get(field)))
            y_pred.append(norm_code(pred.get(field)))
        else:
            y_true.append(norm_text(gold.get(field)))
            y_pred.append(norm_text(pred.get(field)))
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "support": len(y_true),
        "exact_accuracy": round(float(accuracy_score(y_true, y_pred)), 6) if y_true else 0.0,
        "normalized_accuracy": round(float(accuracy_score(y_true, y_pred)), 6) if y_true else 0.0,
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6) if y_true else 0.0,
        "missing_prediction_count": int(sum(1 for p_ in y_pred if not p_)),
        "wrong_prediction_count": int(sum(1 for g, p_ in zip(y_true, y_pred) if g != p_)),
        "extra_prediction_count": 0,
        "per_class": [
            {"label": labels[i], "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "f1": round(float(f[i]), 6), "support": int(s[i])}
            for i in range(len(labels))
        ],
    }


def write_sha_manifest(path: Path, targets: list[Path]) -> dict[str, str]:
    sha_map = {}
    lines = []
    for p in targets:
        if p.exists():
            digest = sha256_file(p)
            sha_map[str(p)] = digest
            lines.append(f"{digest}  {p}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return sha_map


def read_official_test_input() -> list[dict[str, Any]]:
    df = pd.read_excel(OFFICIAL_TEST_XLSX, sheet_name=0, usecols=["article_file_id", "article_title", "article_source", "content", "article_publish_time"], dtype=object).fillna("")
    rows = []
    for idx, rec in enumerate(df.to_dict("records")):
        row = {k: ("" if pd.isna(v) else v) for k, v in rec.items()}
        row["sheet_name"] = "official_test"
        row["sheet_row_index"] = idx + 2
        row["article_file_id"] = str(row.get("article_file_id") or "")
        row["article_title"] = str(row.get("article_title") or "")
        row["article_source"] = str(row.get("article_source") or "")
        row["content"] = str(row.get("content") or "")
        row["article_publish_time"] = str(row.get("article_publish_time") or "")
        row["content_sha256"] = sha256_text(norm_text(row["content"]))
        row["title_sha256"] = sha256_text(norm_text(row["article_title"]))
        rows.append(row)
    return rows


def official_test_header_audit() -> dict[str, Any]:
    xls = pd.ExcelFile(OFFICIAL_TEST_XLSX)
    unlabeled = pd.ExcelFile(UNLABELED_TEST_XLSX)
    return {
        "labeled_sheets": xls.sheet_names,
        "unlabeled_sheets": unlabeled.sheet_names,
        "unlabeled_first_sheet_columns": list(pd.read_excel(UNLABELED_TEST_XLSX, sheet_name=unlabeled.sheet_names[0], nrows=1).columns),
    }
