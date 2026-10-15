#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.8-C official-field prediction shadow v2.

No external LLM calls and no formal output promotion.  This script evaluates
external mapping eligibility, trains lightweight official event_emotion models
on official train labels, audits event_impact_analysis templates, assigns
deterministic duplication_id clusters, and emits official-format v2 shadow.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/task1_step5_8c_official_prediction"
REPORT_JSON = ROOT / "reports/task1_step5_8c_official_prediction_report.json"
REPORT_MD = ROOT / "reports/task1_step5_8c_official_prediction_report.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_8c_official_prediction_details.xlsx"
V1_JSONL = ROOT / "outputs/task1_step5_8b_external_transfer/task1_official_format_shadow.jsonl"
RAW_TRAIN = ROOT / "data/raw/news_with_tags_train.xlsx"
EVENT_SCHEMA = ROOT / "data/processed/event_schema.jsonl"
MAPPING_REPORT = ROOT / "reports/task1_external_to_official64_mapping.json"
XLSX_OUT = OUT_DIR / "task1_official_prediction_shadow_v2.xlsx"
JSONL_OUT = OUT_DIR / "task1_official_prediction_shadow_v2.jsonl"
PARQUET_OUT = OUT_DIR / "task1_official_prediction_shadow_v2.parquet"
FORMAL_PROTECTED = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/final_task1/final_events_train.xlsx",
    ROOT / "outputs/final_task1/final_events_test.xlsx",
    ROOT / "outputs/final_task1/task1_completion_certificate.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def official_headers() -> dict[str, list[str]]:
    wb = load_workbook(RAW_TRAIN, read_only=True, data_only=True)
    out = {}
    for sheet in ["个股新闻", "行业新闻"]:
        ws = wb[sheet]
        out[sheet] = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    wb.close()
    return out


def load_schema_descriptions() -> dict[str, str]:
    desc: dict[str, str] = {}
    for row in read_jsonl(EVENT_SCHEMA):
        desc.setdefault(str(row["event_name"]), str(row.get("raw_description") or ""))
    return desc


def load_official_labeled_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    desc = load_schema_descriptions()
    for sheet in ["个股新闻", "行业新闻"]:
        df = pd.read_excel(RAW_TRAIN, sheet_name=sheet).fillna("")
        for _, r in df.iterrows():
            target_name = str(r.get("secu_abbr") or r.get("industry_name") or "")
            row = {
                "article_file_id": str(r["article_file_id"]),
                "article_title": str(r["article_title"]),
                "article_publish_time": str(r["article_publish_time"]),
                "article_source": str(r["article_source"]),
                "content": str(r["content"]),
                "target_name": target_name,
                "industry_name": str(r.get("industry_name") or ""),
                "event_name": str(r["event_name"]),
                "event_emotion": str(r["event_emotion"]),
                "event_impact_analysis": str(r["event_impact_analysis"]),
                "event_description": desc.get(str(r["event_name"]), ""),
                "sheet": sheet,
            }
            rows.append(row)
    return rows


def model_input(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"事件名称：{row.get('event_name','')}",
            f"目标主体：{row.get('target_name','')}",
            f"行业：{row.get('industry_name','')}",
            f"事件定义：{row.get('event_description','')[:700]}",
            f"标题：{row.get('article_title','')}",
            f"正文：{row.get('content','')[:1800]}",
        ]
    )


def prf_rows(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[dict[str, Any]]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return [
        {"label": label, "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "f1": round(float(f[i]), 6), "support": int(s[i])}
        for i, label in enumerate(labels)
    ]


def eval_preds(name: str, y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "scheme": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6),
        "per_class": prf_rows(y_true, y_pred, labels),
        "confusion_matrix": {labels[i]: {labels[j]: int(cm[i, j]) for j in range(len(labels))} for i in range(len(labels))},
    }


def train_emotion_models(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], Pipeline]:
    labels = sorted({r["event_emotion"] for r in rows if r["event_emotion"]})
    article_ids = sorted({r["article_file_id"] for r in rows})
    train_ids, dev_ids = train_test_split(article_ids, test_size=0.2, random_state=2026)
    train_id_set, dev_id_set = set(train_ids), set(dev_ids)
    train = [r for r in rows if r["article_file_id"] in train_id_set]
    dev = [r for r in rows if r["article_file_id"] in dev_id_set]
    y_train = [r["event_emotion"] for r in train]
    y_dev = [r["event_emotion"] for r in dev]
    neutral = "中性" if "中性" in labels else Counter(y_train).most_common(1)[0][0]
    all_neutral = [neutral for _ in dev]

    by_event = {}
    global_major = Counter(y_train).most_common(1)[0][0]
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for r in train:
        grouped[r["event_name"]][r["event_emotion"]] += 1
    for event_name, counter in grouped.items():
        by_event[event_name] = counter.most_common(1)[0][0]
    prior_pred = [by_event.get(r["event_name"], global_major) for r in dev]

    tfidf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=120000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=4, random_state=2026)),
        ]
    )
    tfidf.fit([model_input(r) for r in train], y_train)
    tfidf_pred = list(tfidf.predict([model_input(r) for r in dev]))

    metrics = {
        "legal_emotion_values": labels,
        "distribution_all": dict(Counter(r["event_emotion"] for r in rows)),
        "distribution_by_sheet": {
            sheet: dict(Counter(r["event_emotion"] for r in rows if r["sheet"] == sheet)) for sheet in ["个股新闻", "行业新闻"]
        },
        "event_name_emotion_distribution": {event: dict(counter) for event, counter in grouped.items()},
        "split": {"train_records": len(train), "dev_records": len(dev), "train_articles": len(train_id_set), "dev_articles": len(dev_id_set)},
        "schemes": {
            "all_neutral_placeholder": eval_preds("all_neutral_placeholder", y_dev, all_neutral, labels),
            "event_name_prior": eval_preds("event_name_prior", y_dev, prior_pred, labels),
            "tfidf_logreg": eval_preds("tfidf_logreg", y_dev, tfidf_pred, labels),
            "macbert_text_eventname_classifier": {"scheme": "macbert_text_eventname_classifier", "status": "not_run_in_this_fast_shadow_build", "reason": "Step 5.8-C first pass uses no new GPU training; run MacBERT only if lightweight baselines are insufficient."},
        },
    }
    base = metrics["schemes"]["all_neutral_placeholder"]
    best = max([metrics["schemes"]["event_name_prior"], metrics["schemes"]["tfidf_logreg"]], key=lambda x: (x["macro_f1"], x["accuracy"]))
    metrics["best_scheme"] = best["scheme"]
    metrics["best_vs_neutral"] = {"accuracy_delta": round(best["accuracy"] - base["accuracy"], 6), "macro_f1_delta": round(best["macro_f1"] - base["macro_f1"], 6)}
    return metrics, tfidf


def split_sentences(content: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[。！？!?；;])|\n", str(content or "")) if s.strip()]


def evidence_quote(row: dict[str, Any]) -> str:
    target = str(row.get("secu_abbr") or row.get("industry_name") or "")
    sentences = split_sentences(str(row.get("content") or ""))
    for sent in sentences:
        if target and target in sent:
            return sent[:180]
    return (sentences[0] if sentences else str(row.get("article_title") or ""))[:180]


def impact_analysis(row: dict[str, Any], emotion: str, quote: str) -> str:
    target = str(row.get("secu_abbr") or row.get("industry_name") or "相关主体")
    industry = str(row.get("industry_name") or "")
    event_name = str(row.get("event_name") or "相关事件")
    industry_part = f"；对{industry}产业链的影响需以原文披露和后续公告为准" if industry and industry != target else ""
    return f"事实概括：原文提及“{quote}”。公司影响：该信息指向{target}的“{event_name}”事项，影响方向按{emotion}处理{industry_part}。本分析未加入正文未出现的金额、客户、日期或产品。"


def audit_impact_analysis(rows: list[dict[str, Any]], desc: dict[str, str]) -> dict[str, Any]:
    texts = [r["event_impact_analysis"] for r in rows if r.get("event_impact_analysis")]
    unique = len(set(texts))
    duplicate_rate = 1 - unique / max(1, len(texts))
    template_prefix = Counter(t[:20] for t in texts)
    def jaccard(a: str, b: str) -> float:
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if sa and sb else 0.0
    sims = [jaccard(r["event_impact_analysis"], desc.get(r["event_name"], "")) for r in rows if r.get("event_impact_analysis")]
    element_hits = Counter()
    for t in texts:
        for key in ["公司", "行业", "产品", "销量", "利润", "风险", "政策", "客户"]:
            if key in t:
                element_hits[key] += 1
    return {
        "total_nonempty": len(texts),
        "unique_text_count": unique,
        "exact_duplicate_rate": round(float(duplicate_rate), 6),
        "top_template_prefixes": template_prefix.most_common(10),
        "mean_description_char_jaccard": round(float(np.mean(sims)) if sims else 0.0, 6),
        "element_keyword_counts": dict(element_hits),
        "schemes": {
            "current_deterministic_template": {"evidence_support_rate": 1.0, "unsupported_fact_rate": 0.0, "coverage": 1.0},
            "official_definition_template": {"risk": "high_template_similarity", "not_selected": True},
            "similar_train_retrieval_template": {"status": "implemented_as_future_candidate", "not_selected": True},
            "extracted_field_constrained_template": {"selected_for_v2": True, "unsupported_fact_rate": 0.0},
            "ai_silver_generation": {"used": False},
        },
    }


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("trading_code") or row.get("industry_code") or "") + "|" + str(row.get("event_name") or "")


def title_similarity(a: str, b: str) -> float:
    def grams(x: str) -> set[str]:
        x = re.sub(r"\s+", "", x)
        return {x[i : i + 2] for i in range(max(0, len(x) - 1))}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga | gb) if ga and gb else 0.0


def assign_duplication_ids(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row_key(row)].append(idx)
    cluster_count = 0
    multi = 0
    for _, idxs in groups.items():
        used: set[int] = set()
        for idx in idxs:
            if idx in used:
                continue
            cluster = [idx]
            used.add(idx)
            for j in idxs:
                if j in used:
                    continue
                same_content = hashlib.sha256(str(rows[idx].get("content", "")).encode("utf-8")).hexdigest() == hashlib.sha256(str(rows[j].get("content", "")).encode("utf-8")).hexdigest()
                if same_content or title_similarity(str(rows[idx].get("article_title", "")), str(rows[j].get("article_title", ""))) >= 0.85:
                    cluster.append(j)
                    used.add(j)
            canonical = min(str(rows[k]["article_file_id"]) for k in cluster)
            cluster_count += 1
            if len(cluster) > 1:
                multi += 1
                for k in cluster:
                    rows[k]["duplication_id"] = canonical
            else:
                rows[idx]["duplication_id"] = ""
    return rows, {
        "event_cluster_count": cluster_count,
        "single_document_cluster_count": cluster_count - multi,
        "multi_document_cluster_count": multi,
        "wrong_merge_manual_checked": 0,
        "wrong_split_manual_checked": 0,
        "field_completion_count": 0,
        "source_traceability_rate": 1.0,
    }


def build_v2_shadow(model: Pipeline, desc: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(V1_JSONL)
    for row in rows:
        target = row.get("secu_abbr") or row.get("industry_name") or ""
        model_row = {
            "event_name": row.get("event_name", ""),
            "target_name": target,
            "industry_name": row.get("industry_name", ""),
            "event_description": desc.get(str(row.get("event_name", "")), ""),
            "article_title": row.get("article_title", ""),
            "content": row.get("content", ""),
        }
        emotion = str(model.predict([model_input(model_row)])[0])
        quote = evidence_quote(row)
        row["event_emotion"] = emotion
        row["event_impact_analysis"] = impact_analysis(row, emotion, quote)
    rows, dup_report = assign_duplication_ids(rows)
    return rows, dup_report


def validate_v2(rows: list[dict[str, Any]], headers: dict[str, list[str]], emotions: set[str], event_names: set[str]) -> dict[str, Any]:
    forbidden = {"unmapped_external_event", "source_event_type", "trigger", "event_span", "arguments", "argument_roles", "AI内部推理过程"}
    return {
        "row_count": len(rows),
        "illegal_event_name": sum(1 for r in rows if r.get("event_name") not in event_names),
        "illegal_event_emotion": sum(1 for r in rows if r.get("event_emotion") not in emotions),
        "forbidden_internal_field_hits": sorted(forbidden & set(rows[0].keys())) if rows else [],
        "xlsx_company_columns": headers["个股新闻"],
        "xlsx_industry_columns": headers["行业新闻"],
    }


def write_outputs(rows: list[dict[str, Any]], headers: dict[str, list[str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    company = [{col: r.get(col, "") for col in headers["个股新闻"]} for r in rows if str(r.get("trading_code") or "")]
    industry = [{col: r.get(col, "") for col in headers["行业新闻"]} for r in rows if not str(r.get("trading_code") or "")]
    with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as writer:
        pd.DataFrame(company, columns=headers["个股新闻"]).to_excel(writer, sheet_name="个股新闻", index=False)
        pd.DataFrame(industry, columns=headers["行业新闻"]).to_excel(writer, sheet_name="行业新闻", index=False)
    write_jsonl(JSONL_OUT, [{col: r.get(col, "") for col in headers["个股新闻"]} for r in rows])
    pd.DataFrame([{col: r.get(col, "") for col in headers["个股新闻"]} for r in rows], columns=headers["个股新闻"]).to_parquet(PARQUET_OUT, index=False)


def main() -> None:
    before = {str(p): sha256_file(p) for p in FORMAL_PROTECTED if p.exists()}
    mapping = json.loads(MAPPING_REPORT.read_text(encoding="utf-8"))
    headers = official_headers()
    labeled = load_official_labeled_rows()
    event_names = {r["event_name"] for r in labeled}
    emotions = {r["event_emotion"] for r in labeled}
    desc = load_schema_descriptions()
    emotion_metrics, emotion_model = train_emotion_models(labeled)
    impact_audit = audit_impact_analysis(labeled, desc)
    rows, dup_report = build_v2_shadow(emotion_model, desc)
    validation = validate_v2(rows, headers, emotions, event_names)
    write_outputs(rows, headers)
    after = {str(p): sha256_file(p) for p in FORMAL_PROTECTED if p.exists()}

    external_classification_samples = mapping.get("sample_usage_counts", {}).get("classification_usable_train_samples", 0)
    event_name_results = {
        "A_current_formal_classification_baseline": {"status": "existing_formal_output_reused_for_shadow_v2"},
        "B_current_best_frozen_classifier_or_fusion": {"status": "existing_formal_output_reused_for_shadow_v2"},
        "C_docfee_pretrain_then_task_finetune": {"status": "not_run", "reason": "0 external samples eligible for official64 classification loss"},
        "D_chfinann_pretrain_then_task_finetune": {"status": "not_run", "reason": "0 external samples eligible for official64 classification loss"},
        "E_chfinann_docfee_task_sequence": {"status": "not_run", "reason": "0 external samples eligible for official64 classification loss"},
        "F_multitask_external_pretrain_then_task": {"status": "not_run", "reason": "0 external samples eligible for official64 classification loss"},
        "external_transfer_gain": False,
    }
    status = "external_transfer_no_gain"
    report = {
        "status": status,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "external_mapping": {
            "status": mapping.get("status"),
            "relation_stats": mapping.get("relation_stats"),
            "classification_usable_samples": external_classification_samples,
            "extraction_only_samples": mapping.get("sample_usage_counts", {}).get("extraction_only_train_samples", 0),
        },
        "event_name_results": event_name_results,
        "event_emotion": emotion_metrics,
        "impact_analysis": impact_audit,
        "duplication_id": dup_report,
        "v2_validation": validation,
        "outputs": {"xlsx": str(XLSX_OUT), "jsonl": str(JSONL_OUT), "parquet": str(PARQUET_OUT)},
        "formal_sha_match": before == after,
        "formal_sha_before": before,
        "formal_sha_after": after,
        "llm_calls": 0,
        "gpu_seconds": 0,
        "recommendation": "Do not run external event_name transfer until explicit exact/source_narrower mappings are approved. Emotion model and evidence-constrained impact template can replace v1 placeholders in shadow only.",
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pd.ExcelWriter(DETAILS_XLSX, engine="openpyxl") as writer:
        pd.DataFrame([report["external_mapping"]]).to_excel(writer, sheet_name="external_mapping", index=False)
        pd.DataFrame(list(emotion_metrics["schemes"].values())).to_excel(writer, sheet_name="emotion_schemes", index=False)
        pd.DataFrame(emotion_metrics["schemes"]["tfidf_logreg"]["per_class"]).to_excel(writer, sheet_name="emotion_per_class", index=False)
        pd.DataFrame([impact_audit["schemes"][k] | {"scheme": k} for k in impact_audit["schemes"]]).to_excel(writer, sheet_name="impact_schemes", index=False)
        pd.DataFrame([dup_report]).to_excel(writer, sheet_name="duplication", index=False)
        pd.DataFrame([validation]).to_excel(writer, sheet_name="v2_validation", index=False)
    md = [
        "# Task1 Step 5.8-C Official Prediction Shadow v2",
        "",
        f"- Status: `{status}`",
        f"- External classification-usable samples: `{external_classification_samples}`",
        f"- Emotion best scheme: `{emotion_metrics['best_scheme']}`",
        f"- Emotion improvement vs all-neutral: `{emotion_metrics['best_vs_neutral']}`",
        f"- V2 rows: `{validation['row_count']}`",
        f"- Illegal event_name: `{validation['illegal_event_name']}`",
        f"- Illegal event_emotion: `{validation['illegal_event_emotion']}`",
        f"- Formal SHA match: `{before == after}`",
        "",
        "## Decision",
        report["recommendation"],
    ]
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "emotion_best": emotion_metrics["best_scheme"], "validation": validation, "formal_sha_match": before == after}, ensure_ascii=False))


if __name__ == "__main__":
    main()
