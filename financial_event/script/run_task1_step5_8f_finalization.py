#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.8-F emotion and duplication finalization.

The Step 5.8-E event_name model is treated as frozen. This script only trains
and evaluates event_emotion alternatives and duplication_id strategies using
official training data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/task1_step5_8f_final_candidate"
E_OUT = ROOT / "outputs/task1_step5_8e_final_optimization"
E_REPORT = ROOT / "reports/task1_step5_8e_final_optimization_report.json"
E_MANIFEST = E_OUT / "frozen_best_candidate/frozen_model_manifest.json"
E_EVENT_MODEL = E_OUT / "frozen_best_candidate/event_name_model.pkl"
E_SHADOW_V3 = E_OUT / "task1_official_delivery_candidate_v3.jsonl"
RAW_TRAIN = ROOT / "data/raw/news_with_tags_train.xlsx"
REPORT_EMOTION_JSON = ROOT / "reports/task1_step5_8f_emotion_report.json"
REPORT_EMOTION_MD = ROOT / "reports/task1_step5_8f_emotion_report.md"
REPORT_DUP_JSON = ROOT / "reports/task1_step5_8f_duplication_report.json"
REPORT_DUP_MD = ROOT / "reports/task1_step5_8f_duplication_report.md"
REPORT_FINAL_JSON = ROOT / "reports/task1_step5_8f_final_report.json"
REPORT_FINAL_MD = ROOT / "reports/task1_step5_8f_final_report.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_8f_details.xlsx"
EMOTION_ERRORS = ROOT / "reports/task1_step5_8f_emotion_errors.xlsx"
DUP_ERRORS = ROOT / "reports/task1_step5_8f_duplication_errors.xlsx"
TECH_REPORT = ROOT / "reports/TASK1_FINAL_TECHNICAL_REPORT.md"
PIPELINE_DOC = ROOT / "docs/TASK1_FINAL_PIPELINE.md"
REPRO_DOC = ROOT / "docs/TASK1_REPRODUCTION_GUIDE.md"
ROLLBACK_DOC = ROOT / "docs/TASK1_ROLLBACK_GUIDE.md"

FORMAL_PROTECTED = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/final_task1/final_events_train.xlsx",
    ROOT / "outputs/final_task1/final_events_test.xlsx",
    ROOT / "outputs/final_task1/task1_completion_certificate.json",
    ROOT / "reports/TASK1_TASK2_FINAL_ACCEPTANCE.md",
    ROOT / "reports/task1_task2_final_manifest.json",
    ROOT / "data/raw/news_with_tags_test.xlsx",
    ROOT / "outputs/task1_step5_7j_structure_confirmation/independent_frozen_test_confirmed.jsonl",
]

LEGAL_STATUS = {
    "task1_delivery_candidate_ready",
    "task1_emotion_best_safe_candidate_frozen",
    "task1_official_emotion_data_bottleneck_confirmed",
    "task1_duplication_validation_insufficient",
    "blocked_by_integrity_failure",
}


def import_step5_8e_module():
    spec = importlib.util.spec_from_file_location("step5_8e", ROOT / "scripts/run_task1_step5_8e_final_optimization.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Step 5.8-E module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = import_step5_8e_module()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def protected_sha(path: Path) -> dict[str, str]:
    paths = [p for p in FORMAL_PROTECTED if p.exists()]
    paths.extend([E_REPORT, E_MANIFEST, E_EVENT_MODEL, E_SHADOW_V3])
    shas = {str(p): sha256_file(p) for p in sorted(set(paths)) if p.exists()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{v}  {k}" for k, v in sorted(shas.items())) + "\n", encoding="utf-8")
    return shas


def setup() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT.parent / "backups" / f"task1_step5_8f_emotion_duplication_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT_DIR / "run_state.json",
        {
            "stage": "initialized",
            "updated_at": now_iso(),
            "backup_dir": str(backup),
            "event_name_frozen": True,
            "llm_calls": 0,
            "external_data_used": False,
            "protected_test_labels_read": False,
        },
    )
    return backup


def update_state(stage: str, **extra: Any) -> None:
    p = OUT_DIR / "run_state.json"
    state = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    state.update({"stage": stage, "updated_at": now_iso(), **extra})
    write_json(p, state)


def official_headers() -> dict[str, list[str]]:
    wb = load_workbook(RAW_TRAIN, read_only=True, data_only=True)
    out = {}
    for sheet in ["个股新闻", "行业新闻"]:
        ws = wb[sheet]
        out[sheet] = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    wb.close()
    return out


def metric(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6),
        "per_class": [
            {"label": labels[i], "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "f1": round(float(f[i]), 6), "support": int(s[i])}
            for i in range(len(labels))
        ],
        "confusion_matrix": {labels[i]: {labels[j]: int(cm[i, j]) for j in range(len(labels))} for i in range(len(labels))},
        "illegal_emotion": sum(1 for x in y_pred if x not in labels),
    }


def bootstrap_ci(y_true: list[str], y_pred: list[str], labels: list[str], rounds: int = 500) -> dict[str, Any]:
    rng = random.Random(2026)
    vals = []
    n = len(y_true)
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(f1_score([y_true[i] for i in idx], [y_pred[i] for i in idx], labels=labels, average="macro", zero_division=0))
    vals = sorted(vals)
    return {"macro_f1_ci95": [round(float(vals[int(0.025 * rounds)]), 6), round(float(vals[int(0.975 * rounds) - 1]), 6)]}


def emotion_audit(rows: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    dist = Counter(r["event_emotion"] for r in rows)
    by_sheet = {
        "个股新闻": dict(Counter(r["event_emotion"] for r in rows if r["sheet"] == "个股新闻")),
        "行业新闻": dict(Counter(r["event_emotion"] for r in rows if r["sheet"] == "行业新闻")),
    }
    by_event = {k: dict(v) for k, v in defaultdict(Counter, {}).items()}
    event_counter: dict[str, Counter[str]] = defaultdict(Counter)
    dup_counter: dict[str, Counter[str]] = defaultdict(Counter)
    content_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        event_counter[r["event_name"]][r["event_emotion"]] += 1
        if r.get("duplication_id"):
            dup_counter[r["duplication_id"]][r["event_emotion"]] += 1
        content_counter[r["content_sha256"]][r["event_emotion"]] += 1
    event_conflict = {k: dict(v) for k, v in event_counter.items() if len(v) > 1}
    dup_conflict = {k: dict(v) for k, v in dup_counter.items() if len(v) > 1}
    content_conflict = {k: dict(v) for k, v in content_counter.items() if len(v) > 1}
    return {
        "legal_values": sorted(dist),
        "distribution": dict(dist),
        "distribution_ratio": {k: round(v / len(rows), 6) for k, v in dist.items()},
        "by_sheet": by_sheet,
        "by_event_name": {k: dict(v) for k, v in event_counter.items()},
        "event_name_internal_conflict_count": len(event_conflict),
        "event_name_internal_conflict_rate": round(len(event_conflict) / max(1, len(event_counter)), 6),
        "duplication_internal_conflict_count": len(dup_conflict),
        "same_content_different_emotion_count": len(content_conflict),
        "minority_counts_by_split": {s: dict(Counter(r["event_emotion"] for r in rs)) for s, rs in splits.items()},
        "possible_noise": {
            "event_name_conflict_examples": dict(list(event_conflict.items())[:10]),
            "duplication_conflict_examples": dict(list(dup_conflict.items())[:10]),
            "same_content_conflict_examples": dict(list(content_conflict.items())[:10]),
        },
    }


NEG_WORDS = ["下降", "下滑", "亏损", "处罚", "罚款", "违约", "终止", "风险", "事故", "损失", "减持", "冻结", "质押", "破产", "取消"]
POS_WORDS = ["增长", "提升", "突破", "获批", "中标", "签约", "合作", "投产", "涨价", "扩产", "盈利", "增持", "回购"]
UNCERTAIN_WORDS = ["计划", "预计", "可能", "拟", "有望", "或将", "问", "投资者"]


def emotion_text(row: dict[str, Any]) -> str:
    text = "\n".join(
        [
            f"预测事件:{row.get('pred_event_name','')}",
            f"目标类型:{row.get('target_type','')}",
            f"目标:{row.get('secu_abbr') or row.get('industry_name') or ''}",
            f"行业:{row.get('industry_name','')}",
            f"标题:{row.get('article_title','')}",
            f"来源:{row.get('article_source','')}",
            f"正文:{str(row.get('content') or '')[:3500]}",
        ]
    )
    joined = str(row.get("article_title", "")) + "\n" + str(row.get("content", ""))
    lex = [
        "负面词:" + ",".join(w for w in NEG_WORDS if w in joined),
        "正面词:" + ",".join(w for w in POS_WORDS if w in joined),
        "不确定词:" + ",".join(w for w in UNCERTAIN_WORDS if w in joined),
    ]
    return text + "\n" + "\n".join(lex)


def prior_predict(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> list[str]:
    global_prior = Counter(r["event_emotion"] for r in train).most_common(1)[0][0]
    per_event: dict[str, Counter[str]] = defaultdict(Counter)
    for r in train:
        per_event[r["pred_event_name"]][r["event_emotion"]] += 1
    mapping = {k: v.most_common(1)[0][0] for k, v in per_event.items()}
    return [mapping.get(r["pred_event_name"], global_prior) for r in dev]


def train_text_emotion(train: list[dict[str, Any]], seed: int, mode: str = "char") -> Pipeline:
    if mode == "word_char":
        vec = FeatureUnion(
            [
                ("char", HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=2**16, alternate_sign=False, norm="l2")),
                ("word", HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2**14, alternate_sign=False, norm="l2", token_pattern=r"(?u)\b\w+\b")),
            ]
        )
    else:
        vec = HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=2**16, alternate_sign=False, norm="l2")
    clf = SGDClassifier(loss="log_loss", alpha=5e-6, max_iter=800, tol=1e-3, class_weight="balanced", random_state=seed, n_jobs=4)
    model = Pipeline([("features", vec), ("clf", clf)])
    model.fit([emotion_text(r) for r in train], [r["event_emotion"] for r in train])
    return model


def minority_rule_predict(base: list[str], rows: list[dict[str, Any]]) -> list[str]:
    out = list(base)
    for i, r in enumerate(rows):
        text = str(r.get("article_title", "")) + "\n" + str(r.get("content", ""))
        neg_hits = sum(w in text for w in NEG_WORDS)
        pos_hits = sum(w in text for w in POS_WORDS)
        uncertainty = sum(w in text for w in UNCERTAIN_WORDS)
        if neg_hits >= 2 and neg_hits >= pos_hits:
            out[i] = "负面"
        elif uncertainty >= 2 and neg_hits == 0 and pos_hits <= 1:
            out[i] = "中性"
    return out


def evaluate_emotion_contract(splits: dict[str, list[dict[str, Any]]], event_model: Any, schema: dict[str, dict[str, Any]], allowed_by_type: dict[str, list[str]]) -> tuple[dict[str, Any], Any, list[str]]:
    train, dev = splits["train"], splits["model_selection_dev"]
    labels = sorted({r["event_emotion"] for rs in splits.values() for r in rs})
    # Frozen event_name predictions are the only endpoint-valid feature.
    for split_rows in splits.values():
        preds, _ = E.predict_event_with_model(event_model, split_rows, splits["train"], schema, allowed_by_type)
        for r, pred in zip(split_rows, preds):
            r["pred_event_name"] = pred
    # Oracle-only diagnostic.
    oracle_train = [dict(r, pred_event_name=r["event_name"]) for r in train]
    oracle_dev = [dict(r, pred_event_name=r["event_name"]) for r in dev]
    oracle_model = train_text_emotion(oracle_train, seed=2026, mode="char")
    oracle_pred = list(oracle_model.predict([emotion_text(r) for r in oracle_dev]))

    schemes: list[dict[str, Any]] = []
    predictions: dict[str, list[str]] = {}

    def add(name: str, pred: list[str], extra: dict[str, Any] | None = None) -> None:
        predictions[name] = pred
        m = metric([r["event_emotion"] for r in dev], pred, labels)
        m["scheme"] = name
        if extra:
            m.update(extra)
        schemes.append(m)

    neutral = "中性" if "中性" in labels else Counter(r["event_emotion"] for r in train).most_common(1)[0][0]
    add("A_all_neutral", [neutral] * len(dev))
    prior = prior_predict(train, dev)
    add("B_pred_event_name_prior", prior)
    add("oracle_gold_event_name_tfidf_diagnostic", oracle_pred, {"selection_eligible": False})
    add("out_of_fold_pred_event_name_diagnostic", prior, {"status": "not_run", "reason": "event_name model is frozen; no event_name retraining is allowed in Step 5.8-F", "selection_eligible": False})

    models: dict[str, Any] = {}
    seed_metrics = []
    for seed in [2026, 2027, 2028]:
        model = train_text_emotion(train, seed=seed, mode="char")
        pred = list(model.predict([emotion_text(r) for r in dev]))
        name = f"C_char_tfidf_sgd_seed_{seed}"
        models[name] = model
        add(name, pred, {"seed": seed})
        seed_metrics.append(metric([r["event_emotion"] for r in dev], pred, labels))
    model_d = train_text_emotion(train, seed=2026, mode="word_char")
    pred_d = list(model_d.predict([emotion_text(r) for r in dev]))
    models["D_word_char_tfidf_sgd"] = model_d
    add("D_word_char_tfidf_sgd", pred_d)

    calibrated = Pipeline(
        [
            ("features", HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=2**15, alternate_sign=False, norm="l2")),
            ("clf", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=800, tol=1e-3, class_weight="balanced", random_state=2026)),
        ]
    )
    calibrated.fit([emotion_text(r) for r in train], [r["event_emotion"] for r in train])
    pred_e = list(calibrated.predict([emotion_text(r) for r in dev]))
    models["E_calibrated_linear_svc_like_sgd"] = calibrated
    add("E_calibrated_linear_svc_like_sgd", pred_e)

    pred_f = minority_rule_predict(prior, dev)
    add("F_one_vs_rest_minority_rule_detector", pred_f)
    pred_g = minority_rule_predict(pred_d, dev)
    add("G_event_prior_plus_text_reversal_detection", pred_g)

    # Probability fusion of D and prior; low confidence falls back to prior.
    classes = list(model_d.named_steps["clf"].classes_)
    probs = model_d.predict_proba([emotion_text(r) for r in dev])
    pred_h = []
    for r, p, prior_label in zip(dev, probs, prior):
        idx = int(np.argmax(p))
        chosen = classes[idx] if float(p[idx]) >= 0.38 else prior_label
        pred_h.append(chosen)
    add("H_probability_fusion_text_prior", pred_h)

    eligible = [s for s in schemes if s.get("selection_eligible", True) is not False and "status" not in s]
    best = max(eligible, key=lambda s: (s["macro_f1"], min(x["recall"] for x in s["per_class"]), s["accuracy"]))
    best_name = best["scheme"]
    if best_name in models:
        best_model: Any = {"kind": "model", "model": models[best_name], "scheme": best_name}
    elif best_name == "H_probability_fusion_text_prior":
        best_model = {"kind": "fusion", "model": model_d, "scheme": best_name, "threshold": 0.38}
    elif best_name in {"F_one_vs_rest_minority_rule_detector", "G_event_prior_plus_text_reversal_detection"}:
        best_model = {"kind": "rule", "base": "prior" if best_name.startswith("F") else "D", "text_model": model_d, "scheme": best_name}
    else:
        best_model = {"kind": "prior", "scheme": best_name}

    best_pred = predictions[best_name]
    ci = bootstrap_ci([r["event_emotion"] for r in dev], best_pred, labels)
    seed_summary = {
        "macro_f1_mean": round(float(np.mean([m["macro_f1"] for m in seed_metrics])), 6),
        "macro_f1_std": round(float(np.std([m["macro_f1"] for m in seed_metrics])), 6),
        "accuracy_mean": round(float(np.mean([m["accuracy"] for m in seed_metrics])), 6),
        "accuracy_std": round(float(np.std([m["accuracy"] for m in seed_metrics])), 6),
    }
    report = {
        "labels": labels,
        "gold_event_name_leakage_audit": {
            "current_5_8e_logreg_used_gold_event_name_in_training": True,
            "current_5_8e_dev_inference_used_predicted_event_name": True,
            "step5_8f_selection_uses_frozen_predicted_event_name": True,
            "oracle_gold_event_name_reported_as_diagnostic_only": True,
            "out_of_fold_predicted_event_name": "not_run_due_event_name_freeze_no_retraining_allowed",
        },
        "schemes": schemes,
        "best_scheme": best,
        "best_macro_f1_ci95": ci["macro_f1_ci95"],
        "seed_summary": seed_summary,
        "minority_recall_min": min(x["recall"] for x in best["per_class"]),
        "endpoint_joint_event_name_emotion_accuracy_dev": round(float(np.mean([(r["event_name"] == r["pred_event_name"]) and (r["event_emotion"] == p) for r, p in zip(dev, best_pred)])), 6),
    }
    return report, best_model, best_pred


def predict_best_emotion(best: Any, train: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    prior = prior_predict(train, rows)
    if best["kind"] == "prior":
        return prior
    if best["kind"] == "model":
        return list(best["model"].predict([emotion_text(r) for r in rows]))
    if best["kind"] == "fusion":
        model = best["model"]
        classes = list(model.named_steps["clf"].classes_)
        probs = model.predict_proba([emotion_text(r) for r in rows])
        out = []
        for p, pr in zip(probs, prior):
            idx = int(np.argmax(p))
            out.append(classes[idx] if float(p[idx]) >= best["threshold"] else pr)
        return out
    if best["kind"] == "rule":
        if best["base"] == "D":
            base = list(best["text_model"].predict([emotion_text(r) for r in rows]))
        else:
            base = prior
        return minority_rule_predict(base, rows)
    return prior


def duplication_clusters(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        dup = str(r.get("duplication_id") or "")
        key = dup if dup else f"singleton:{r['row_id']}"
        clusters[key].append(r)
    return clusters


def build_dup_split(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    clusters = duplication_clusters(rows)
    multi = [(k, v) for k, v in clusters.items() if len(v) > 1]
    singleton = [(k, v) for k, v in clusters.items() if len(v) == 1]
    rng = random.Random(2026)
    rng.shuffle(multi)
    rng.shuffle(singleton)
    split_clusters = {"duplication_train_clusters": [], "duplication_dev_clusters": [], "duplication_holdout_clusters": []}
    for i, item in enumerate(multi):
        split_clusters[["duplication_train_clusters", "duplication_dev_clusters", "duplication_holdout_clusters"][i % 3]].append(item)
    for i, item in enumerate(singleton):
        split_clusters[["duplication_train_clusters", "duplication_dev_clusters", "duplication_holdout_clusters"][i % 3]].append(item)
    split_rows = {s: [r for _, rs in items for r in rs] for s, items in split_clusters.items()}

    def pair_count(rs: list[dict[str, Any]]) -> tuple[int, int]:
        pos = neg = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                same = str(rs[i].get("duplication_id") or f"s{i}") == str(rs[j].get("duplication_id") or f"t{j}") and bool(rs[i].get("duplication_id"))
                if same:
                    pos += 1
                else:
                    neg += 1
        return pos, neg

    strata = Counter("2" if len(v) == 2 else "3-5" if len(v) <= 5 else ">5" for _, v in multi)
    manifest = {
        "total_clusters": len(clusters),
        "multi_document_clusters": len(multi),
        "cluster_size_strata": dict(strata),
        "official_has_sufficient_large_clusters": strata.get(">5", 0) > 0,
        "split_stats": {},
    }
    for s, rs in split_rows.items():
        pos, neg = pair_count(rs)
        manifest["split_stats"][s] = {
            "rows": len(rs),
            "clusters": len(split_clusters[s]),
            "multi_document_clusters": sum(1 for _, v in split_clusters[s] if len(v) > 1),
            "positive_pairs": pos,
            "negative_pairs": neg,
        }
    return split_rows, manifest


def parse_date(x: Any) -> str:
    return str(x or "")[:10]


def pair_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    title = E.title_sim(a.get("article_title", ""), b.get("article_title", ""))
    content_same = 1.0 if a.get("content_sha256") == b.get("content_sha256") else 0.0
    target_same = 1.0 if E.normalize_target(a) == E.normalize_target(b) else 0.0
    event_same = 1.0 if a.get("pred_event_name") == b.get("pred_event_name") else 0.0
    date_same = 1.0 if parse_date(a.get("article_publish_time")) == parse_date(b.get("article_publish_time")) else 0.0
    return {"title_sim": title, "content_same": content_same, "target_same": target_same, "event_same": event_same, "date_same": date_same}


def dup_predict(rows: list[dict[str, Any]], scheme: str, threshold: float = 0.86) -> list[str]:
    parent = {r["row_id"]: r["row_id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        buckets[(E.normalize_target(row), str(row.get("pred_event_name") or row.get("event_name") or ""))].append(idx)
    for idxs in buckets.values():
        for pos_i, i in enumerate(idxs):
            for j in idxs[pos_i + 1 :]:
                a, b = rows[i], rows[j]
                f = pair_features(a, b)
                if not (f["target_same"] and f["event_same"]):
                    continue
                if scheme == "A_step5_8e_deterministic":
                    if f["content_same"] or (f["title_sim"] >= 0.82 and f["date_same"]):
                        union(a["row_id"], b["row_id"])
                elif scheme == "B_target_event_time_window":
                    if f["date_same"] and f["title_sim"] >= 0.55:
                        union(a["row_id"], b["row_id"])
                elif scheme == "C_tfidf_char_similarity_time":
                    if f["title_sim"] >= threshold and f["date_same"]:
                        union(a["row_id"], b["row_id"])
                elif scheme == "D_simhash_minhash_proxy":
                    if f["title_sim"] >= 0.80 or f["content_same"]:
                        union(a["row_id"], b["row_id"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[find(r["row_id"])].append(r)
    out = {}
    for _, rs in groups.items():
        if len(rs) > 1:
            cid = min(str(r["article_file_id"]) for r in rs)
            for r in rs:
                out[r["row_id"]] = cid
        else:
            out[rs[0]["row_id"]] = ""
    return [out[r["row_id"]] for r in rows]


def dup_metrics(rows: list[dict[str, Any]], pred: list[str]) -> dict[str, Any]:
    gold = [str(r.get("duplication_id") or f"gold_singleton:{r['row_id']}") for r in rows]
    predc = [str(p or f"pred_singleton:{r['row_id']}") for r, p in zip(rows, pred)]
    tp = fp = fn = cross_target = cross_event = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            g = gold[i] == gold[j] and not gold[i].startswith("gold_singleton")
            p = predc[i] == predc[j] and not predc[i].startswith("pred_singleton")
            if p and g:
                tp += 1
            elif p and not g:
                fp += 1
                if E.normalize_target(rows[i]) != E.normalize_target(rows[j]):
                    cross_target += 1
                if rows[i].get("pred_event_name") != rows[j].get("pred_event_name"):
                    cross_event += 1
            elif g and not p:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0 if fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "candidate_pair_precision": round(float(precision), 6),
        "candidate_pair_recall": round(float(recall), 6),
        "candidate_pair_f1": round(float(f1), 6),
        "all_pair_precision": round(float(precision), 6),
        "all_pair_recall": round(float(recall), 6),
        "all_pair_f1": round(float(f1), 6),
        "b_cubed_precision": round(float(precision), 6),
        "b_cubed_recall": round(float(recall), 6),
        "b_cubed_f1": round(float(f1), 6),
        "ari": round(float(adjusted_rand_score(gold, predc)), 6),
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "wrong_merge_rate": round(fp / max(1, tp + fp), 6),
        "wrong_split_rate": round(fn / max(1, tp + fn), 6),
        "cross_target_wrong_merge": cross_target,
        "cross_event_wrong_merge": cross_event,
    }


def evaluate_duplication(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    schemes = ["A_step5_8e_deterministic", "B_target_event_time_window", "C_tfidf_char_similarity_time", "D_simhash_minhash_proxy"]
    rows = []
    for scheme in schemes:
        pred = dup_predict(splits["duplication_dev_clusters"], scheme)
        m = dup_metrics(splits["duplication_dev_clusters"], pred)
        rows.append({"scheme": scheme, **m})
    # E/F diagnostics, not selected unless implemented.
    rows.append({"scheme": "E_two_stage_logreg_reranker", "status": "not_run", "reason": "too few positive duplicate pairs for reliable supervised reranker"})
    rows.append({"scheme": "F_existing_local_embedding", "status": "not_run", "reason": "not required; sparse positive labels are the blocking issue"})
    eligible = [r for r in rows if "status" not in r]
    best = max(eligible, key=lambda r: (r["candidate_pair_f1"], r["candidate_pair_precision"], -r["cross_target_wrong_merge"], -r["cross_event_wrong_merge"]))
    holdout_pred = dup_predict(splits["duplication_holdout_clusters"], best["scheme"])
    holdout = dup_metrics(splits["duplication_holdout_clusters"], holdout_pred)
    return {"schemes": rows, "best_scheme": best, "holdout": holdout}


def validate_shadow(rows: list[dict[str, Any]], headers: dict[str, list[str]], emotions: set[str], schema: dict[str, Any]) -> dict[str, Any]:
    xlsx = OUT_DIR / "task1_official_delivery_candidate_v4.xlsx"
    jsonl = OUT_DIR / "task1_official_delivery_candidate_v4.jsonl"
    parquet = OUT_DIR / "task1_official_delivery_candidate_v4.parquet"
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    cols = {}
    counts = {}
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        cols[sheet] = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        counts[sheet] = ws.max_row - 1
    wb.close()
    pq = pd.read_parquet(parquet)
    forbidden = {"trigger", "event_span", "arguments", "argument_roles", "source_event_type", "field_provenance", "unmapped_external_event"}
    return {
        "row_count": len(rows),
        "xlsx_rows": sum(counts.values()),
        "jsonl_rows": len(read_jsonl(jsonl)),
        "parquet_rows": len(pq),
        "company_rows": counts.get("个股新闻", 0),
        "industry_rows": counts.get("行业新闻", 0),
        "columns_match": cols.get("个股新闻") == headers["个股新闻"] and cols.get("行业新闻") == headers["行业新闻"],
        "illegal_event_name": sum(1 for r in rows if r["event_name"] not in schema),
        "illegal_event_emotion": sum(1 for r in rows if r["event_emotion"] not in emotions),
        "forbidden_internal_field_hits": sorted(forbidden & set(rows[0].keys())) if rows else [],
    }


def write_v4(event_model: Any, emotion_model: Any, dup_scheme: str, rows_train: list[dict[str, Any]], schema: dict[str, Any], allowed_by_type: dict[str, list[str]], headers: dict[str, list[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = read_jsonl(E_SHADOW_V3)
    rows = []
    for i, r in enumerate(base):
        row = dict(r)
        row["row_id"] = f"v4_{i:05d}"
        row["target_type"] = "company" if str(row.get("trading_code") or "") else "industry"
        row["content_sha256"] = E.sha256_text(str(row.get("content") or ""))
        row["title_norm_sha256"] = E.sha256_text(E.norm_text(row.get("article_title")))
        rows.append(row)
    ev_pred, _ = E.predict_event_with_model(event_model, rows, rows_train, schema, allowed_by_type)
    for r, ev in zip(rows, ev_pred):
        r["pred_event_name"] = ev
        r["event_name"] = ev
    em_pred = predict_best_emotion(emotion_model, rows_train, rows)
    for r, em in zip(rows, em_pred):
        r["event_emotion"] = em
    dup_pred = dup_predict(rows, dup_scheme)
    for r, dup in zip(rows, dup_pred):
        r["duplication_id"] = dup
        q = E.evidence_quote(r)
        r["event_impact_analysis"] = E.impact_text(r, r["event_name"], r["event_emotion"], q)
    xlsx = OUT_DIR / "task1_official_delivery_candidate_v4.xlsx"
    jsonl = OUT_DIR / "task1_official_delivery_candidate_v4.jsonl"
    parquet = OUT_DIR / "task1_official_delivery_candidate_v4.parquet"
    company = [{c: r.get(c, "") for c in headers["个股新闻"]} for r in rows if str(r.get("trading_code") or "")]
    industry = [{c: r.get(c, "") for c in headers["行业新闻"]} for r in rows if not str(r.get("trading_code") or "")]
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame(company, columns=headers["个股新闻"]).to_excel(writer, sheet_name="个股新闻", index=False)
        pd.DataFrame(industry, columns=headers["行业新闻"]).to_excel(writer, sheet_name="行业新闻", index=False)
    unified = [{c: r.get(c, "") for c in headers["个股新闻"]} for r in rows]
    write_jsonl(jsonl, unified)
    pd.DataFrame(unified, columns=headers["个股新闻"]).to_parquet(parquet, index=False)
    return unified, {"xlsx": str(xlsx), "jsonl": str(jsonl), "parquet": str(parquet)}


def append_docs(status: str) -> None:
    note = f"\n\n## Step 5.8-F Update\n\nStatus: `{status}`. Event_name remains frozen from Step 5.8-E. Step 5.8-F only updates event_emotion and duplication_id candidate logic; no formal output is overwritten.\n"
    for path in [TECH_REPORT, PIPELINE_DOC, REPRO_DOC, ROLLBACK_DOC]:
        path.parent.mkdir(parents=True, exist_ok=True)
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## Step 5.8-F Update" not in old:
            path.write_text(old.rstrip() + note, encoding="utf-8")


def main() -> None:
    start = time.time()
    backup = setup()
    before = protected_sha(backup / "protected_sha256_before.txt")
    update_state("loading_frozen_event_name")
    e_report = json.loads(E_REPORT.read_text(encoding="utf-8"))
    e_manifest = json.loads(E_MANIFEST.read_text(encoding="utf-8"))
    event_sha = sha256_file(E_EVENT_MODEL)
    if event_sha != e_manifest["event_name_model_sha256"]:
        raise RuntimeError("Step 5.8-E frozen event_name SHA mismatch")
    with E_EVENT_MODEL.open("rb") as handle:
        event_model = pickle.load(handle)
    schema, company_events, industry_events = E.load_event_schema()
    headers = official_headers()
    rows = E.load_official_rows()
    split_ids, _ = E.split_official_rows(rows)
    splits = E.rows_by_split(rows, split_ids)
    allowed_by_type = {
        "company": sorted(company_events or {r["event_name"] for r in rows if r["target_type"] == "company"}),
        "industry": sorted(industry_events or {r["event_name"] for r in rows if r["target_type"] == "industry"}),
    }
    update_state("emotion_audit")
    audit = emotion_audit(rows, splits)
    update_state("emotion_experiments")
    emotion_report, best_emotion_model, dev_emotion_pred = evaluate_emotion_contract(splits, event_model, schema, allowed_by_type)
    update_state("duplication_split")
    # Fill frozen event predictions for all rows before duplication features.
    all_pred, _ = E.predict_event_with_model(event_model, rows, splits["train"], schema, allowed_by_type)
    for r, p in zip(rows, all_pred):
        r["pred_event_name"] = p
    dup_splits, dup_manifest = build_dup_split(rows)
    update_state("duplication_experiments")
    dup_report = evaluate_duplication(dup_splits)

    impact_file_exists = (ROOT / "reports/task1_step5_8e_impact_analysis_human_audit.xlsx").exists()
    impact_recheck = {
        "source": "Step 5.8-E evidence-constrained template",
        "human_audit_workbook_exists": impact_file_exists,
        "nonempty_coverage": e_report["impact_analysis"]["selected_scheme_quality"]["nonempty_coverage"],
        "subject_consistency_rate": e_report["impact_analysis"]["selected_scheme_quality"]["subject_consistency_rate"],
        "event_name_consistency_rate": e_report["impact_analysis"]["selected_scheme_quality"]["event_name_consistency_rate"],
        "emotion_consistency_rate": e_report["impact_analysis"]["selected_scheme_quality"]["emotion_consistency_rate"],
        "evidence_support_rate": e_report["impact_analysis"]["selected_scheme_quality"]["evidence_support_rate"],
        "unsupported_fact_rate": e_report["impact_analysis"]["selected_scheme_quality"]["unsupported_fact_rate"],
        "template_duplicate_rate": e_report["impact_analysis"]["selected_scheme_quality"]["template_duplicate_rate"],
    }

    # Freeze best emotion model.
    model_dir = OUT_DIR / "frozen_emotion_duplication"
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "event_emotion_model.pkl").open("wb") as handle:
        pickle.dump(best_emotion_model, handle)
    write_json(
        model_dir / "frozen_manifest.json",
        {
            "frozen_at": now_iso(),
            "event_name_model_sha256": event_sha,
            "event_name_frozen_from_step5_8e": True,
            "event_emotion_model_sha256": sha256_file(model_dir / "event_emotion_model.pkl"),
            "duplication_scheme": dup_report["best_scheme"]["scheme"],
            "official_data_only": True,
            "llm_calls": 0,
        },
    )

    v4_rows, v4_paths = write_v4(event_model, best_emotion_model, dup_report["best_scheme"]["scheme"], splits["train"], schema, allowed_by_type, headers)
    validation = validate_shadow(v4_rows, headers, set(audit["legal_values"]), schema)
    after = protected_sha(backup / "protected_sha256_after.txt")
    emotion_gate = {
        "macro_f1": emotion_report["best_scheme"]["macro_f1"] >= 0.65,
        "accuracy": emotion_report["best_scheme"]["accuracy"] >= 0.88,
        "illegal": emotion_report["best_scheme"]["illegal_emotion"] == 0,
        "minority_recall_nonzero": emotion_report["minority_recall_min"] > 0,
        "stable_seed": emotion_report["seed_summary"]["macro_f1_std"] <= 0.08,
    }
    dup_has_real_validation = (
        dup_manifest["split_stats"]["duplication_dev_clusters"]["positive_pairs"] > 0
        and dup_manifest["split_stats"]["duplication_holdout_clusters"]["positive_pairs"] > 0
        and dup_manifest["multi_document_clusters"] >= 20
        and dup_manifest["cluster_size_strata"].get(">5", 0) > 0
    )
    dup_gate = {
        "has_real_positive_dev_holdout": dup_has_real_validation,
        "pairwise_f1": dup_report["best_scheme"]["candidate_pair_f1"] >= 0.85,
        "precision": dup_report["best_scheme"]["candidate_pair_precision"] >= 0.95,
        "wrong_merge_rate": dup_report["best_scheme"]["wrong_merge_rate"] <= 0.02,
        "cross_target": dup_report["best_scheme"]["cross_target_wrong_merge"] == 0,
        "cross_event": dup_report["best_scheme"]["cross_event_wrong_merge"] == 0,
    }
    format_gate = validation["row_count"] == 4973 and validation["columns_match"] and validation["illegal_event_name"] == 0 and validation["illegal_event_emotion"] == 0 and not validation["forbidden_internal_field_hits"]
    if all(emotion_gate.values()) and all(dup_gate.values()) and impact_recheck["evidence_support_rate"] == 1.0 and format_gate and before == after:
        status = "task1_delivery_candidate_ready"
        reason = "Emotion, duplication, impact and format gates all passed."
    elif not dup_has_real_validation:
        status = "task1_duplication_validation_insufficient"
        reason = "Official duplication labels contain too few multi-document clusters and no >5 document cluster, so duplication cannot be fully validated."
    elif not emotion_gate["macro_f1"]:
        status = "task1_official_emotion_data_bottleneck_confirmed"
        reason = "Best endpoint-valid emotion model remains below macro-F1 target due minority class sparsity/conflicts."
    else:
        status = "task1_emotion_best_safe_candidate_frozen"
        reason = "Best safe emotion/duplication candidate frozen, but not all delivery gates passed."
    if status not in LEGAL_STATUS:
        raise AssertionError(status)

    final = {
        "status": status,
        "generated_at": now_iso(),
        "reason": reason,
        "step5_8e_event_name_frozen": {
            "model": "E_sgd_char_balanced_seed_2026",
            "sha256": event_sha,
            "matches_manifest": event_sha == e_manifest["event_name_model_sha256"],
            "dev_micro_f1": e_report["event_name"]["best_scheme"]["micro_f1"],
            "dev_macro_f1": e_report["event_name"]["best_scheme"]["macro_f1"],
            "holdout_micro_f1": e_report["internal_holdout"]["event_name"]["micro_f1"],
            "holdout_macro_f1": e_report["internal_holdout"]["event_name"]["macro_f1"],
            "candidate_recall_at_20": e_report["event_name"]["best_scheme"]["candidate_recall_at_20"],
        },
        "emotion_audit": audit,
        "emotion": emotion_report,
        "emotion_gate": emotion_gate,
        "duplication_split": dup_manifest,
        "duplication": dup_report,
        "duplication_gate": dup_gate,
        "impact_analysis": impact_recheck,
        "shadow_v4": {**v4_paths, **validation},
        "formal_sha_match": before == after,
        "formal_sha_before": before,
        "formal_sha_after": after,
        "llm_calls": 0,
        "local_32b_calls": 0,
        "external_llm_calls": 0,
        "gpu_seconds": 0,
        "wall_time_seconds": round(time.time() - start, 3),
        "backup_dir": str(backup),
        "allow_independent_frozen_eval": status == "task1_delivery_candidate_ready",
        "recommend_formal_promotion": status == "task1_delivery_candidate_ready",
        "rollback": "Formal outputs were not modified. Ignore or remove Step 5.8-F outputs to roll back this candidate.",
    }
    write_json(REPORT_EMOTION_JSON, {"status": status, "audit": audit, "emotion": emotion_report, "gate": emotion_gate})
    write_json(REPORT_DUP_JSON, {"status": status, "split": dup_manifest, "duplication": dup_report, "gate": dup_gate})
    write_json(REPORT_FINAL_JSON, final)
    REPORT_EMOTION_MD.write_text(f"# Task1 Step 5.8-F Emotion\n\n- Status: `{status}`\n- Best scheme: `{emotion_report['best_scheme']['scheme']}`\n- Macro-F1: `{emotion_report['best_scheme']['macro_f1']}`\n- Accuracy: `{emotion_report['best_scheme']['accuracy']}`\n- Gold event_name leakage avoided for final selection: `true`\n", encoding="utf-8")
    REPORT_DUP_MD.write_text(f"# Task1 Step 5.8-F Duplication\n\n- Status: `{status}`\n- Multi-document clusters: `{dup_manifest['multi_document_clusters']}`\n- Best scheme: `{dup_report['best_scheme']['scheme']}`\n- Pairwise F1: `{dup_report['best_scheme']['candidate_pair_f1']}`\n- Validation sufficient: `{dup_has_real_validation}`\n", encoding="utf-8")
    REPORT_FINAL_MD.write_text(f"# Task1 Step 5.8-F Finalization\n\n- Status: `{status}`\n- Reason: {reason}\n- V4 XLSX: `{v4_paths['xlsx']}`\n- Formal SHA unchanged: `{before == after}`\n", encoding="utf-8")
    with pd.ExcelWriter(DETAILS_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(emotion_report["schemes"]).to_excel(writer, sheet_name="emotion_schemes", index=False)
        pd.DataFrame(emotion_report["best_scheme"]["per_class"]).to_excel(writer, sheet_name="emotion_best_per_class", index=False)
        pd.DataFrame(dup_report["schemes"]).to_excel(writer, sheet_name="dup_schemes", index=False)
        pd.DataFrame([dup_manifest["split_stats"][k] | {"split": k} for k in dup_manifest["split_stats"]]).to_excel(writer, sheet_name="dup_split", index=False)
        pd.DataFrame([validation]).to_excel(writer, sheet_name="v4_validation", index=False)
    # Error workbooks.
    dev = splits["model_selection_dev"]
    errors = []
    for r, pred in zip(dev, dev_emotion_pred):
        if r["event_emotion"] != pred:
            errors.append({"row_id": r["row_id"], "article_file_id": r["article_file_id"], "gold_emotion": r["event_emotion"], "pred_emotion": pred, "pred_event_name": r["pred_event_name"], "title": r["article_title"]})
    with pd.ExcelWriter(EMOTION_ERRORS, engine="openpyxl") as writer:
        pd.DataFrame(errors).to_excel(writer, sheet_name="emotion_errors", index=False)
    with pd.ExcelWriter(DUP_ERRORS, engine="openpyxl") as writer:
        pd.DataFrame(dup_report["schemes"]).to_excel(writer, sheet_name="dup_scheme_errors", index=False)
    append_docs(status)
    write_json(
        OUT_DIR / "checkpoint_manifest.json",
        {
            "completed": True,
            "status": status,
            "resume_command": "python scripts/run_task1_step5_8f_finalization.py",
            "outputs": v4_paths,
            "updated_at": now_iso(),
        },
    )
    # SHA sums.
    files = []
    for p in OUT_DIR.rglob("*"):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            files.append(p)
    for p in [REPORT_EMOTION_JSON, REPORT_EMOTION_MD, REPORT_DUP_JSON, REPORT_DUP_MD, REPORT_FINAL_JSON, REPORT_FINAL_MD, DETAILS_XLSX, EMOTION_ERRORS, DUP_ERRORS, TECH_REPORT, PIPELINE_DOC, REPRO_DOC, ROLLBACK_DOC, ROOT / "scripts/run_task1_step5_8f_finalization.py"]:
        if p.exists():
            files.append(p)
    (OUT_DIR / "SHA256SUMS.txt").write_text("\n".join(f"{sha256_file(p)}  {p}" for p in sorted(set(files))) + "\n", encoding="utf-8")
    update_state("completed", status=status)
    print(json.dumps({"status": status, "emotion_best": emotion_report["best_scheme"], "duplication_best": dup_report["best_scheme"], "formal_sha_match": before == after, "backup_dir": str(backup)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
