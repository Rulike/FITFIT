#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.8-HI.

Build a frozen candidate from the existing retrospective predictions on the
user-supplied labeled test set, normalize singleton duplication IDs, and emit
HI-specific reports without retraining or rerunning model inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/task1_step5_8hi_tag_optimization"
REPORT_JSON = ROOT / "reports/task1_step5_8hi_final_report.json"
REPORT_MD = ROOT / "reports/task1_step5_8hi_final_report.md"
PER_TAG_XLSX = ROOT / "reports/task1_step5_8hi_per_tag_metrics.xlsx"
PER_TAG_MD = ROOT / "reports/task1_step5_8hi_per_tag_metrics.md"
BASELINE_VS_OPT_XLSX = ROOT / "reports/task1_step5_8hi_baseline_vs_optimized.xlsx"
ERROR_XLSX = ROOT / "reports/task1_step5_8hi_error_analysis.xlsx"
IMPACT_AUDIT_XLSX = ROOT / "reports/task1_step5_8hi_impact_human_audit.xlsx"
DUP_ERRORS_XLSX = ROOT / "reports/task1_step5_8hi_duplication_errors.xlsx"
TECH_REPORT = ROOT / "reports/TASK1_FINAL_OPTIMIZED_TECHNICAL_REPORT.md"
PIPELINE_DOC = ROOT / "docs/TASK1_FINAL_OPTIMIZED_PIPELINE.md"
REPRO_DOC = ROOT / "docs/TASK1_FINAL_REPRODUCTION_GUIDE.md"
ROLLBACK_DOC = ROOT / "docs/TASK1_FINAL_ROLLBACK_GUIDE.md"
RUN_STATE = OUT_DIR / "run_state.json"
CHECKPOINT = OUT_DIR / "checkpoint_manifest.json"
RUN_MANIFEST = OUT_DIR / "run_manifest.json"
SHA_SUMS = OUT_DIR / "SHA256SUMS.txt"
FROZEN_MANIFEST = OUT_DIR / "frozen_model_manifest.json"
PRED_JSONL = OUT_DIR / "official_labeled_test_predictions.jsonl"
PRED_XLSX = OUT_DIR / "official_labeled_test_predictions.xlsx"
PRED_PARQUET = OUT_DIR / "official_labeled_test_predictions.parquet"

HF_REPORT = ROOT / "reports/task1_step5_8hf_retrospective_official_test_report.json"
HF_PER_TAG = ROOT / "reports/task1_step5_8hf_per_tag_metrics.xlsx"
HF_PRED = ROOT / "outputs/task1_step5_8hf_retrospective_test/official_test_predictions_frozen.jsonl"
HC_REPORT = ROOT / "reports/task1_step5_8hc_hierarchical_event_report.json"

USER_TEST_CANDIDATES = [
    ROOT / "data/evaluation/news_with_tags_test_user_supplied.xlsx",
    ROOT / "data/raw/news_with_tags_test.xlsx",
    Path("/opt/data/private/JinRong/kechuang/news_with_tags_test.xlsx"),
]

FORMAL_PROTECTED = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/final_task1/final_events_train.xlsx",
    ROOT / "outputs/final_task1/final_events_test.xlsx",
    ROOT / "outputs/final_task1/task1_completion_certificate.json",
    ROOT / "reports/task1_task2_final_manifest.json",
    ROOT / "reports/TASK1_TASK2_FINAL_ACCEPTANCE.md",
    ROOT / "reports/TASK2_FINAL_TECHNICAL_REPORT.md",
    ROOT / "task1_task2_final_delivery_20260809.tar.gz",
    ROOT / "task1234_delivery_20260811.tar.gz",
]

OUTPUT_COLS = [
    "article_file_id",
    "article_title",
    "article_publish_time",
    "article_source",
    "content",
    "target_type",
    "trading_code",
    "secu_abbr",
    "industry_code",
    "industry_name",
    "event_name",
    "event_emotion",
    "event_impact_analysis",
    "duplication_id",
]

MAIN_SHEETS = ["个股新闻", "行业新闻"]
DUP_SHEETS = ["个股重复新闻", "行业重复新闻"]
ALL_SHEETS = MAIN_SHEETS + DUP_SHEETS
LEGAL_STATUSES = {
    "task1_optimized_tag_evaluation_complete",
    "task1_best_safe_candidate_frozen",
    "task1_optimization_gain_not_confirmed",
    "blocked_by_integrity_or_environment_failure",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def norm_code(value: Any, width: int | None = 6) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if width and text.isdigit():
        text = text.zfill(width)
    return text


def sheet_gold_type(sheet_name: str) -> str:
    return "industry" if "行业" in sheet_name else "company"


def choose_test_file() -> tuple[Path, str]:
    for path in USER_TEST_CANDIDATES:
        if path.exists():
            sha = sha256_file(path)
            return path, sha
    raise FileNotFoundError("No labeled test workbook found in the configured search paths.")


def read_headers(path: Path) -> dict[str, list[str]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    headers: dict[str, list[str]] = {}
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        headers[sheet] = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    wb.close()
    return headers


def load_gold_rows(path: Path) -> list[dict[str, Any]]:
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    rows: list[dict[str, Any]] = []
    for sheet_name in ALL_SHEETS:
        df = sheets[sheet_name].fillna("")
        for idx, rec in enumerate(df.to_dict("records"), start=2):
            row = {k: ("" if pd.isna(v) else v) for k, v in rec.items()}
            row["sheet_name"] = sheet_name
            row["sheet_row_index"] = idx
            rows.append(row)
    return rows


def load_predictions() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in HF_PRED.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("Frozen prediction file is empty.")
    return rows


def singleton_dup_id(row: dict[str, Any]) -> str:
    return f"singleton::{row['sheet_name']}::{row['article_file_id']}"


def build_candidate_rows(pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pred_rows:
        out = {k: row.get(k, "") for k in OUTPUT_COLS}
        out["target_type"] = str(out.get("target_type") or "missing_prediction")
        out["trading_code"] = str(out.get("trading_code") or "")
        out["secu_abbr"] = str(out.get("secu_abbr") or "")
        out["industry_code"] = str(out.get("industry_code") or "")
        out["industry_name"] = str(out.get("industry_name") or "")
        out["event_name"] = str(out.get("event_name") or "missing_prediction")
        out["event_emotion"] = str(out.get("event_emotion") or "missing_prediction")
        out["event_impact_analysis"] = str(out.get("event_impact_analysis") or "")
        out["duplication_id"] = str(out.get("duplication_id") or "").strip()
        if not out["duplication_id"]:
            out["duplication_id"] = singleton_dup_id(row)
        rows.append(out)
    return rows


def load_gold_map(gold_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["sheet_name"], int(row["sheet_row_index"])): row for row in gold_rows}


def as_main(row: dict[str, Any]) -> bool:
    return row["sheet_name"] in MAIN_SHEETS


def candidate_to_output_rows(candidate_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_sheet: dict[str, list[dict[str, Any]]] = {s: [] for s in ALL_SHEETS}
    flat: list[dict[str, Any]] = []
    for pred, gold in zip(candidate_rows, gold_rows):
        row = {col: pred.get(col, "") for col in OUTPUT_COLS}
        row["target_type"] = pred.get("target_type", "")
        if sheet_gold_type(gold["sheet_name"]) == "company":
            row["trading_code"] = pred.get("trading_code", "")
            row["secu_abbr"] = pred.get("secu_abbr", "")
        else:
            row["trading_code"] = ""
            row["secu_abbr"] = ""
        if sheet_gold_type(gold["sheet_name"]) == "industry" or gold["sheet_name"] == "个股新闻":
            row["industry_code"] = pred.get("industry_code", "")
            row["industry_name"] = pred.get("industry_name", "")
        else:
            row["industry_code"] = pred.get("industry_code", "")
            row["industry_name"] = pred.get("industry_name", "")
        row["event_name"] = pred.get("event_name", "")
        row["event_emotion"] = pred.get("event_emotion", "")
        row["event_impact_analysis"] = pred.get("event_impact_analysis", "")
        row["duplication_id"] = pred.get("duplication_id", "")
        flat.append(row)
        by_sheet[gold["sheet_name"]].append(row)
    return by_sheet, flat


def parse_quote(text: str) -> str:
    match = re.search(r"原文证据(?:为)?[：:，, ]*[“\"](.+?)[”\"]", text)
    return match.group(1).strip() if match else ""


def rouge_l(ref: str, hyp: str) -> float:
    if not ref or not hyp:
        return 0.0
    prev = [0] * (len(hyp) + 1)
    for ch_ref in ref:
        cur = [0]
        for j, ch_hyp in enumerate(hyp, 1):
            cur.append(prev[j - 1] + 1 if ch_ref == ch_hyp else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    return round(float((2 * lcs) / (len(ref) + len(hyp))), 6)


def official_labels() -> tuple[list[str], list[str], list[str], list[str], dict[str, str]]:
    df = pd.read_excel(HF_PER_TAG, sheet_name="EventName_64类")
    company = sorted(df[df["target_scope"] == "company"]["event_name"].astype(str).tolist())
    industry = sorted(df[df["target_scope"] == "industry"]["event_name"].astype(str).tolist())
    labels = sorted(df[df["target_scope"].isin(["company", "industry"])]["event_name"].astype(str).tolist())
    scope_map = {str(row["event_name"]): str(row["target_scope"]) for _, row in df[df["target_scope"].isin(["company", "industry"])].iterrows()}
    return company, industry, labels, sorted({"正面", "中性", "负面"}), scope_map


def metric_rows(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    labels64: list[str],
    company44: list[str],
    industry20: list[str],
    all_sheets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_idx = [i for i, row in enumerate(gold_rows) if as_main(row)]
    company_idx = [i for i, row in enumerate(gold_rows) if row["sheet_name"] == "个股新闻"]
    industry_idx = [i for i, row in enumerate(gold_rows) if row["sheet_name"] == "行业新闻"]
    company_all_idx = [i for i, row in enumerate(gold_rows) if sheet_gold_type(row["sheet_name"]) == "company"]
    industry_all_idx = [i for i, row in enumerate(gold_rows) if sheet_gold_type(row["sheet_name"]) == "industry"]

    def gold_value(row: dict[str, Any], field: str) -> str:
        if field == "target_type":
            return sheet_gold_type(row["sheet_name"])
        if field == "trading_code":
            return norm_code(row.get("trading_code"))
        if field == "secu_abbr":
            return norm_text(row.get("secu_abbr"))
        if field == "industry_code":
            return norm_code(row.get("industry_code"))
        if field == "industry_name":
            return norm_text(row.get("industry_name"))
        return norm_text(row.get(field))

    def pred_value(row: dict[str, Any], field: str) -> str:
        if field == "target_type":
            return str(row.get("target_type") or "")
        if field == "trading_code":
            return norm_code(row.get("trading_code"))
        if field == "secu_abbr":
            return norm_text(row.get("secu_abbr"))
        if field == "industry_code":
            return norm_code(row.get("industry_code"))
        if field == "industry_name":
            return norm_text(row.get("industry_name"))
        return norm_text(row.get(field))

    def is_missing(v: str) -> bool:
        return str(v).strip() in {"", "missing_prediction", "not_implemented"}

    def is_extra(v: str) -> bool:
        return any(sep in str(v) for sep in ["|", ",", "、", ";", "/", "；"])

    def row_metric(tag: str, idxs: list[int], definition: str, labels: list[str] | None = None) -> dict[str, Any]:
        y_true = [gold_value(gold_rows[i], tag) for i in idxs]
        y_pred = [pred_value(pred_rows[i], tag) for i in idxs]
        missing = sum(1 for p in y_pred if is_missing(p))
        extra = sum(1 for p in y_pred if (not is_missing(p)) and is_extra(p))
        wrong = sum(1 for g, p in zip(y_true, y_pred) if (not is_missing(p)) and (not is_extra(p)) and g != p)
        predicted_count = len(y_pred) - missing
        label_set = labels or sorted(set(y_true) | set(y_pred))
        if y_true:
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=label_set, average="micro", zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, labels=label_set, average="macro", zero_division=0)
            acc = accuracy_score(y_true, y_pred)
        else:
            p = r = f = macro_f1 = acc = 0.0
        return {
            "Tag": tag,
            "sample_count": len(gold_rows),
            "applicable_count": len(y_true),
            "predicted_count": predicted_count,
            "missing_count": missing,
            "wrong_count": wrong,
            "extra_count": extra,
            "coverage": round(predicted_count / len(y_true), 6) if y_true else "NA",
            "accuracy_or_exact_accuracy": round(float(acc), 6) if y_true else "NA",
            "micro_precision": round(float(p), 6) if y_true else "NA",
            "micro_recall": round(float(r), 6) if y_true else "NA",
            "micro_f1": round(float(f), 6) if y_true else "NA",
            "macro_f1": round(float(macro_f1), 6) if y_true else "NA",
            "metric_definition": definition,
        }

    tag_summary = pd.DataFrame(
        [
            row_metric("target_type", list(range(len(gold_rows))), "适用于全部1799行；Gold target_type由sheet类型推导，预测来自冻结候选。"),
            row_metric("trading_code", company_idx, "仅适用于个股新闻主sheet，不使用全部1799行为分母。"),
            row_metric("secu_abbr", company_idx, "仅适用于个股新闻主sheet；exact为去空白后的简称匹配。"),
            row_metric("industry_code", main_idx, "适用于个股新闻与行业新闻主sheet。"),
            row_metric("industry_name", main_idx, "适用于个股新闻与行业新闻主sheet。"),
            row_metric("event_name", main_idx, "仅适用于主标签sheet；macro_f1为官方64类fixed macro-F1。", labels=labels64),
            row_metric("event_emotion", main_idx, "仅适用于主标签sheet；三分类情绪。", labels=["正面", "中性", "负面"]),
            row_metric("event_impact_analysis", main_idx, "生成式影响分析不按分类accuracy/F1评价；使用非空率、一致率、证据支持率、ROUGE-L和人审包。"),
            row_metric("duplication_id", list(range(len(gold_rows))), "duplication_id为聚类任务；不使用行级ID exact/F1代表质量，详见Duplication指标sheet。"),
        ]
    )
    metric_display_cols = ["accuracy_or_exact_accuracy", "micro_precision", "micro_recall", "micro_f1", "macro_f1"]
    for col in metric_display_cols:
        tag_summary[col] = tag_summary[col].astype(object)
    event_true_for_row = [gold_rows[i]["event_name"] for i in main_idx]
    event_pred_for_row = [pred_rows[i]["event_name"] for i in main_idx]
    event_exact_micro = round(float(accuracy_score(event_true_for_row, event_pred_for_row)), 6)
    event_fixed64_macro = round(float(f1_score(event_true_for_row, event_pred_for_row, labels=labels64, average="macro", zero_division=0)), 6)
    event_mask = tag_summary["Tag"] == "event_name"
    for col in ["accuracy_or_exact_accuracy", "micro_precision", "micro_recall", "micro_f1"]:
        tag_summary.loc[event_mask, col] = event_exact_micro
    tag_summary.loc[event_mask, "macro_f1"] = event_fixed64_macro

    # Event name sheet
    y_true = [gold_rows[i]["event_name"] for i in main_idx]
    y_pred = [pred_rows[i]["event_name"] for i in main_idx]
    true_cnt = Counter(y_true)
    pred_cnt = Counter(y_pred)
    event_rows: list[dict[str, Any]] = []
    for label in labels64:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1v = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        conf = Counter(p for g, p in zip(y_true, y_pred) if g == label and p != label)
        event_rows.append(
            {
                "event_name": label,
                "target_scope": "company" if label in company44 else "industry" if label in industry20 else "mixed",
                "support": true_cnt[label],
                "predicted_count": pred_cnt[label],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "F1": round(float(f1v), 6),
                "macro_F1": round(float(f1v), 6),
                "主要混淆类别": "; ".join(f"{k}:{v}" for k, v in conf.most_common(5)) or "NA",
                "metric_definition": "每类one-vs-rest；主要混淆类别为Gold为该类但预测为其他类的Top类别。",
            }
        )
    for name, labels in [("SUMMARY_个股44类", company44), ("SUMMARY_行业20类", industry20), ("SUMMARY_总体64类", labels64)]:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == p and g in labels)
        fp = sum(1 for g, p in zip(y_true, y_pred) if p in labels and g != p)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g in labels and g != p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        event_rows.append(
            {
                "event_name": name,
                "target_scope": "summary",
                "support": sum(true_cnt[l] for l in labels),
                "predicted_count": sum(pred_cnt[l] for l in labels),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "F1": round(float(micro_f1), 6),
                "macro_F1": round(float(macro_f1), 6),
                "主要混淆类别": "NA",
                "metric_definition": "summary row: precision/recall/F1为micro汇总，macro_F1为指定类集合宏平均。",
            }
        )
    event_df = pd.DataFrame(event_rows)

    # Event confusion matrix
    event_cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels64), index=[f"Gold_{x}" for x in labels64], columns=[f"Pred_{x}" for x in labels64])

    # Emotion sheet
    emo_true = [gold_rows[i]["event_emotion"] for i in main_idx]
    emo_pred = [pred_rows[i]["event_emotion"] for i in main_idx]
    emo_labels = ["正面", "中性", "负面"]
    p, r, f, s = precision_recall_fscore_support(emo_true, emo_pred, labels=emo_labels, zero_division=0)
    emotion_df = pd.DataFrame(
        [
            {"event_emotion": emo_labels[i], "support": int(s[i]), "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "F1": round(float(f[i]), 6)}
            for i in range(len(emo_labels))
        ]
    )
    emotion_cm = pd.DataFrame(confusion_matrix(emo_true, emo_pred, labels=emo_labels), index=[f"Gold_{x}" for x in emo_labels], columns=[f"Pred_{x}" for x in emo_labels])

    # Target metrics
    def target_tuple(row: dict[str, Any], use_gold: bool = False) -> str:
        if use_gold:
            typ = sheet_gold_type(row["sheet_name"])
            if row["sheet_name"] == "个股新闻":
                return f"{typ}|{norm_code(row.get('trading_code'))}|{norm_text(row.get('secu_abbr'))}|{norm_code(row.get('industry_code'))}|{norm_text(row.get('industry_name'))}"
            return f"{typ}|{norm_code(row.get('industry_code'))}|{norm_text(row.get('industry_name'))}"
        typ = str(row.get("target_type") or "")
        if row["sheet_name"] == "个股新闻":
            return f"{typ}|{norm_code(row.get('trading_code'))}|{norm_text(row.get('secu_abbr'))}|{norm_code(row.get('industry_code'))}|{norm_text(row.get('industry_name'))}"
        return f"{typ}|{norm_code(row.get('industry_code'))}|{norm_text(row.get('industry_name'))}"

    target_rows = []
    for tag, idxs, definition in [
        ("target_type", list(range(len(gold_rows))), "target_type适用于全部1799行。"),
        ("target_type", company_all_idx, "个股新闻及个股重复新闻。"),
        ("target_type", industry_all_idx, "行业新闻及行业重复新闻。"),
        ("trading_code", company_idx, "证券代码仅适用于个股新闻主sheet。"),
        ("secu_abbr", company_idx, "证券简称仅适用于个股新闻主sheet。"),
        ("industry_code", company_idx, "个股新闻中的行业代码字段。"),
        ("industry_code", industry_idx, "行业新闻中的行业代码字段。"),
        ("industry_code", main_idx, "主sheet总体行业代码字段。"),
        ("industry_name", company_idx, "个股新闻中的行业名称字段。"),
        ("industry_name", industry_idx, "行业新闻中的行业名称字段。"),
        ("industry_name", main_idx, "主sheet总体行业名称字段。"),
    ]:
        y_true = []
        y_pred = []
        for i in idxs:
            g = gold_rows[i]
            p = pred_rows[i]
            if tag == "target_type":
                y_true.append(sheet_gold_type(g["sheet_name"]))
                y_pred.append(str(p.get("target_type") or ""))
            elif tag == "trading_code":
                y_true.append(norm_code(g.get("trading_code")))
                y_pred.append(norm_code(p.get("trading_code")))
            elif tag == "secu_abbr":
                y_true.append(norm_text(g.get("secu_abbr")))
                y_pred.append(norm_text(p.get("secu_abbr")))
            elif tag == "industry_code":
                y_true.append(norm_code(g.get("industry_code")))
                y_pred.append(norm_code(p.get("industry_code")))
            else:
                y_true.append(norm_text(g.get("industry_name")))
                y_pred.append(norm_text(p.get("industry_name")))
        missing = sum(1 for x in y_pred if str(x).strip() in {"", "missing_prediction", "not_implemented"})
        extra = sum(1 for x in y_pred if any(sep in str(x) for sep in ["|", ",", "、", ";", "/", "；"]))
        wrong = sum(1 for g, p in zip(y_true, y_pred) if str(p).strip() not in {"", "missing_prediction", "not_implemented"} and not any(sep in str(p) for sep in ["|", ",", "、", ";", "/", "；"]) and g != p)
        labels_local = sorted(set(y_true) | set(y_pred))
        if y_true:
            p0, r0, f0, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels_local, average="micro", zero_division=0)
            macro = f1_score(y_true, y_pred, labels=labels_local, average="macro", zero_division=0)
            acc = accuracy_score(y_true, y_pred)
        else:
            p0 = r0 = f0 = macro = acc = 0.0
        target_rows.append(
            {
                "tag": tag,
                "target_scope": definition.replace("。", ""),
                "applicable_count": len(y_true),
                "predicted_count": len(y_pred) - missing,
                "missing_count": missing,
                "wrong_count": wrong,
                "extra_count": extra,
                "coverage": round((len(y_pred) - missing) / len(y_true), 6) if y_true else "NA",
                "exact_accuracy": round(float(acc), 6) if y_true else "NA",
                "normalized_accuracy": round(float(acc), 6) if y_true else "NA",
                "micro_precision": round(float(p0), 6) if y_true else "NA",
                "micro_recall": round(float(r0), 6) if y_true else "NA",
                "micro_f1": round(float(f0), 6) if y_true else "NA",
                "macro_f1": round(float(macro), 6) if y_true else "NA",
                "metric_definition": definition,
            }
        )
    target_df = pd.DataFrame(target_rows)

    # Duplication metrics
    gold_dup = [str(r.get("duplication_id") or f"singleton_gold::{r['sheet_name']}::{r['article_file_id']}") for r in gold_rows]
    pred_dup = [str(r.get("duplication_id") or f"singleton_pred::{r['sheet_name']}::{r['article_file_id']}") for r in pred_rows]
    tp = fp = fn = 0
    cross_target = cross_event = 0
    for i in range(len(gold_rows)):
        for j in range(i + 1, len(gold_rows)):
            g = gold_dup[i] == gold_dup[j]
            p = pred_dup[i] == pred_dup[j]
            if g and p:
                tp += 1
            elif p and not g:
                fp += 1
                if sheet_gold_type(gold_rows[i]["sheet_name"]) != sheet_gold_type(gold_rows[j]["sheet_name"]):
                    cross_target += 1
                if str(gold_rows[i].get("event_name") or "") != str(gold_rows[j].get("event_name") or ""):
                    cross_event += 1
            elif g and not p:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    pairwise_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gold_pairs = list(gold_dup)
    pred_pairs = list(pred_dup)
    gold_counts = Counter(gold_pairs)
    pred_counts = Counter(pred_pairs)
    b_prec_vals = []
    b_rec_vals = []
    for i in range(len(gold_rows)):
        gset = {j for j, gid in enumerate(gold_pairs) if gid == gold_pairs[i]}
        pset = {j for j, pid in enumerate(pred_pairs) if pid == pred_pairs[i]}
        inter = len(gset & pset)
        b_prec_vals.append(inter / len(pset) if pset else 1.0)
        b_rec_vals.append(inter / len(gset) if gset else 1.0)
    bcubed_precision = float(np.mean(b_prec_vals)) if b_prec_vals else 1.0
    bcubed_recall = float(np.mean(b_rec_vals)) if b_rec_vals else 1.0
    bcubed_f1 = 2 * bcubed_precision * bcubed_recall / (bcubed_precision + bcubed_recall) if bcubed_precision + bcubed_recall else 0.0

    dup_df = pd.DataFrame(
        [
            {"metric": "pairwise_precision", "value": round(float(precision), 6), "metric_definition": "按Gold/Pred重复簇的文档对计算precision。"},
            {"metric": "pairwise_recall", "value": round(float(recall), 6), "metric_definition": "按Gold/Pred重复簇的文档对计算recall。"},
            {"metric": "pairwise_f1", "value": round(float(pairwise_f1), 6), "metric_definition": "pairwise precision/recall的F1。"},
            {"metric": "bcubed_precision", "value": round(float(bcubed_precision), 6), "metric_definition": "B-cubed precision。"},
            {"metric": "bcubed_recall", "value": round(float(bcubed_recall), 6), "metric_definition": "B-cubed recall。"},
            {"metric": "bcubed_f1", "value": round(float(bcubed_f1), 6), "metric_definition": "B-cubed F1。"},
            {"metric": "ari", "value": round(float(adjusted_rand_score(gold_pairs, pred_pairs)), 6), "metric_definition": "Adjusted Rand Index。"},
            {"metric": "missing_count", "value": sum(1 for x in pred_rows if not str(x.get("duplication_id") or "").strip()), "metric_definition": "预测duplication_id为空或missing_prediction的行数。"},
            {"metric": "singleton_count", "value": sum(1 for x in pred_counts.values() if x == 1), "metric_definition": "预测单文档簇数量。"},
            {"metric": "multi_document_cluster_count", "value": sum(1 for x in pred_counts.values() if x > 1), "metric_definition": "预测多文档簇数量。"},
            {"metric": "gold_multi_document_cluster_count", "value": sum(1 for x in gold_counts.values() if x > 1), "metric_definition": "Gold多文档簇数量。"},
            {"metric": "wrong_merge_count", "value": fp, "metric_definition": "预测合并但Gold未合并的文档对数量。"},
            {"metric": "wrong_split_count", "value": fn, "metric_definition": "Gold同簇但预测拆分的文档对数量。"},
            {"metric": "cross_target_wrong_merge", "value": cross_target, "metric_definition": "跨主体错误合并对数量。"},
            {"metric": "cross_event_wrong_merge", "value": cross_event, "metric_definition": "跨event_name错误合并对数量。"},
            {"metric": "pred_cluster_count", "value": len(set(pred_pairs)), "metric_definition": "预测簇数量。"},
            {"metric": "gold_cluster_count", "value": len(set(gold_pairs)), "metric_definition": "Gold簇数量。"},
        ]
    )

    # Impact metrics
    impact_rows = []
    for i in main_idx:
        gold = gold_rows[i]
        pred = pred_rows[i]
        hyp = str(pred.get("event_impact_analysis") or "")
        quote = parse_quote(hyp)
        target = str(gold.get("secu_abbr") or gold.get("industry_name") or "")
        subject_ok = bool(target and target in hyp)
        event_ok = str(gold.get("event_name") or "") in hyp
        emotion_ok = str(gold.get("event_emotion") or "") in hyp
        evidence_ok = bool(quote and quote in str(gold.get("content") or ""))
        unsupported = bool(hyp and quote and quote not in str(gold.get("content") or ""))
        impact_rows.append(
            {
                "gold": str(gold.get("event_impact_analysis") or ""),
                "prediction": hyp,
                "subject_ok": subject_ok,
                "event_ok": event_ok,
                "emotion_ok": emotion_ok,
                "evidence_ok": evidence_ok,
                "unsupported": unsupported,
                "rouge_l": rouge_l(str(gold.get("event_impact_analysis") or ""), hyp),
            }
        )
    impact_df = pd.DataFrame(
        [
            {
                "metric": "event_impact_analysis",
                "support": len(impact_rows),
                "nonempty_coverage": round(float(np.mean([bool(x["prediction"]) for x in impact_rows])), 6),
                "subject_consistency_rate": round(float(np.mean([x["subject_ok"] for x in impact_rows])), 6),
                "event_name_consistency_rate": round(float(np.mean([x["event_ok"] for x in impact_rows])), 6),
                "emotion_consistency_rate": round(float(np.mean([x["emotion_ok"] for x in impact_rows])), 6),
                "evidence_quote_support_rate": round(float(np.mean([x["evidence_ok"] for x in impact_rows])), 6),
                "unsupported_fact_rate": round(float(np.mean([x["unsupported"] for x in impact_rows])), 6),
                "rouge_l": round(float(np.mean([x["rouge_l"] for x in impact_rows])), 6),
                "template_duplicate_rate": round(float(1 - len(set(x["prediction"] for x in impact_rows)) / max(1, len(impact_rows))), 6),
            }
        ]
    )
    impact_mask = tag_summary["Tag"] == "event_impact_analysis"
    impact_predicted = sum(1 for x in impact_rows if bool(str(x["prediction"]).strip()))
    impact_quality_wrong = sum(
        1
        for x in impact_rows
        if bool(str(x["prediction"]).strip())
        and not (bool(x["subject_ok"]) and bool(x["event_ok"]) and bool(x["emotion_ok"]) and bool(x["evidence_ok"]))
    )
    tag_summary.loc[impact_mask, "predicted_count"] = impact_predicted
    tag_summary.loc[impact_mask, "missing_count"] = len(impact_rows) - impact_predicted
    tag_summary.loc[impact_mask, "wrong_count"] = impact_quality_wrong
    tag_summary.loc[impact_mask, "coverage"] = round(impact_predicted / max(1, len(impact_rows)), 6)
    for col in ["accuracy_or_exact_accuracy", "micro_precision", "micro_recall", "micro_f1", "macro_f1"]:
        tag_summary.loc[impact_mask, col] = "NA"

    dup_mask = tag_summary["Tag"] == "duplication_id"
    tag_summary.loc[dup_mask, "predicted_count"] = len(gold_rows)
    tag_summary.loc[dup_mask, "missing_count"] = 0
    tag_summary.loc[dup_mask, "coverage"] = 1.0
    for col in ["accuracy_or_exact_accuracy", "micro_precision", "micro_recall", "micro_f1", "macro_f1"]:
        tag_summary.loc[dup_mask, col] = "NA"

    # Corrected exact metrics
    corrected_rows = []
    clusters_gold: dict[str, list[int]] = defaultdict(list)
    clusters_pred: dict[str, list[int]] = defaultdict(list)
    for i in range(len(gold_rows)):
        clusters_gold[gold_dup[i]].append(i)
        clusters_pred[pred_dup[i]].append(i)
    for i in main_idx:
        g = gold_rows[i]
        p = pred_rows[i]
        target_ok = sheet_gold_type(g["sheet_name"]) == str(p.get("target_type") or "")
        if g["sheet_name"] == "个股新闻":
            target_ok = target_ok and norm_code(g.get("trading_code")) == norm_code(p.get("trading_code")) and norm_text(g.get("secu_abbr")) == norm_text(p.get("secu_abbr")) and norm_code(g.get("industry_code")) == norm_code(p.get("industry_code")) and norm_text(g.get("industry_name")) == norm_text(p.get("industry_name"))
        else:
            target_ok = target_ok and norm_code(g.get("industry_code")) == norm_code(p.get("industry_code")) and norm_text(g.get("industry_name")) == norm_text(p.get("industry_name"))
        event_ok = str(g.get("event_name") or "") == str(p.get("event_name") or "")
        emotion_ok = str(g.get("event_emotion") or "") == str(p.get("event_emotion") or "")
        hyp = str(p.get("event_impact_analysis") or "")
        quote = parse_quote(hyp)
        subject = str(g.get("secu_abbr") or g.get("industry_name") or "")
        impact_ok = bool(
            hyp
            and quote
            and quote in str(g.get("content") or "")
            and subject
            and subject in hyp
            and str(g.get("event_name") or "") in hyp
            and str(g.get("event_emotion") or "") in hyp
        )
        gold_set = {gold_rows[j]["article_file_id"] for j in clusters_gold[gold_dup[i]]}
        pred_set = {gold_rows[j]["article_file_id"] for j in clusters_pred[pred_dup[i]]}
        dup_ok = gold_set == pred_set
        corrected_rows.append(
            {
                "article_file_id": g["article_file_id"],
                "sheet_name": g["sheet_name"],
                "event_only_exact": event_ok and emotion_ok,
                "target_exact": target_ok,
                "core_discrete_exact": target_ok and event_ok and emotion_ok,
                "full_quality_pass": target_ok and event_ok and emotion_ok and impact_ok and dup_ok,
                "impact_ok": impact_ok,
                "duplication_cluster_exact": dup_ok,
            }
        )
    corrected_df = pd.DataFrame(
        [
            {
                "metric": "event_only_exact",
                "value": round(float(np.mean([x["event_only_exact"] for x in corrected_rows])), 6),
                "metric_definition": "event_name和event_emotion同时正确。",
            },
            {
                "metric": "target_exact",
                "value": round(float(np.mean([x["target_exact"] for x in corrected_rows])), 6),
                "metric_definition": "适用target字段全部正确。",
            },
            {
                "metric": "core_discrete_exact",
                "value": round(float(np.mean([x["core_discrete_exact"] for x in corrected_rows])), 6),
                "metric_definition": "target字段、event_name和event_emotion全部正确。",
            },
            {
                "metric": "full_quality_pass",
                "value": round(float(np.mean([x["full_quality_pass"] for x in corrected_rows])), 6),
                "metric_definition": "core_discrete_exact正确，且impact满足证据约束、duplication簇关系正确。",
            },
            {
                "metric": "legacy_complete_record_exact_match",
                "value": read_json(HF_REPORT)["full_record"]["complete_record_exact_match_accuracy"],
                "metric_definition": "HF历史完整离散Tags exact-match，保留为legacy指标。",
            },
        ]
    )
    return tag_summary, event_df, event_cm, emotion_df, emotion_cm, target_df, dup_df, impact_df, corrected_df, pd.DataFrame(corrected_rows)


def collect_errors(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> pd.DataFrame:
    errors: list[dict[str, Any]] = []

    def add(row: dict[str, Any], field: str, gold_value: Any, pred_value: Any, error_type: str) -> None:
        errors.append(
            {
                "sheet_name": row["sheet_name"],
                "sheet_row_index": row["sheet_row_index"],
                "article_file_id": row.get("article_file_id", ""),
                "article_title": str(row.get("article_title", ""))[:1000],
                "error_field": field,
                "error_type": error_type,
                "gold_value": str(gold_value)[:2000],
                "prediction_value": str(pred_value)[:2000],
            }
        )

    for gold, pred in zip(gold_rows, pred_rows):
        gt_type = sheet_gold_type(gold["sheet_name"])
        if str(pred.get("target_type") or "") != gt_type:
            add(gold, "target_type", gt_type, pred.get("target_type"), "missing" if not str(pred.get("target_type") or "").strip() else "wrong")
        if gold["sheet_name"] == "个股新闻":
            for field in ["trading_code", "secu_abbr"]:
                gt = norm_code(gold.get(field)) if field == "trading_code" else norm_text(gold.get(field))
                pr = norm_code(pred.get(field)) if field == "trading_code" else norm_text(pred.get(field))
                if gt != pr:
                    add(gold, field, gold.get(field), pred.get(field), "missing" if not str(pr).strip() else "wrong")
        if gold["sheet_name"] in MAIN_SHEETS:
            for field in ["industry_code", "industry_name", "event_name", "event_emotion"]:
                gt = norm_code(gold.get(field)) if field == "industry_code" else norm_text(gold.get(field))
                pr = norm_code(pred.get(field)) if field == "industry_code" else norm_text(pred.get(field))
                if gt != pr:
                    add(gold, field, gold.get(field), pred.get(field), "missing" if not str(pr).strip() else "wrong")
            hyp = str(pred.get("event_impact_analysis") or "")
            quote = parse_quote(hyp)
            evidence_ok = bool(quote and quote in str(gold.get("content") or ""))
            subject = str(gold.get("secu_abbr") or gold.get("industry_name") or "")
            if not (hyp and subject and subject in hyp and str(gold.get("event_name") or "") in hyp and str(gold.get("event_emotion") or "") in hyp and evidence_ok):
                add(gold, "event_impact_analysis", gold.get("event_impact_analysis", ""), hyp, "missing" if not str(hyp).strip() else "wrong")
        if str(pred.get("duplication_id") or "").strip() != str(gold.get("duplication_id") or ""):
            add(gold, "duplication_id", gold.get("duplication_id"), pred.get("duplication_id"), "missing" if not str(pred.get("duplication_id") or "").strip() else "wrong")
    return pd.DataFrame(errors)


def build_baseline_vs_optimized(hf_report: dict[str, Any], optimized: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("target_type_accuracy", hf_report["target_type"]["accuracy"], optimized["target_type_accuracy"], "overall target_type accuracy"),
        ("target_exact_main", hf_report["full_record"]["target_exact_rate"], optimized["target_exact_main"], "main labeled target tuple exact"),
        ("event_name_micro_f1", hf_report["event_name"]["micro_f1"], optimized["event_name_micro_f1"], "official 64-class micro-F1"),
        ("event_name_fixed64_macro_f1", hf_report["event_name"]["fixed_64_macro_f1"], optimized["event_name_fixed64_macro_f1"], "official 64-class macro-F1"),
        ("event_emotion_accuracy", hf_report["event_emotion"]["accuracy"], optimized["event_emotion_accuracy"], "3-class emotion accuracy"),
        ("event_emotion_macro_f1", hf_report["event_emotion"]["macro_f1"], optimized["event_emotion_macro_f1"], "3-class emotion macro-F1"),
        ("target_event_joint_micro_f1", hf_report["target_event_joint"]["micro_f1"], optimized["target_event_joint_micro_f1"], "target+event joint micro-F1"),
        ("event_only_exact", hf_report["full_record"]["event_exact_rate"] if "event_exact_rate" in hf_report["full_record"] else None, optimized["event_only_exact"], "event_name+emotion exact"),
        ("core_discrete_exact", hf_report["full_record"]["target_event_emotion_exact_rate"], optimized["core_discrete_exact"], "target+event+emotion exact"),
        ("full_quality_pass", None, optimized["full_quality_pass"], "corrected full-quality pass"),
        ("dup_pairwise_f1", hf_report["duplication"]["pairwise_f1"], optimized["dup_pairwise_f1"], "duplication pairwise F1"),
        ("dup_bcubed_f1", hf_report["duplication"]["bcubed_f1"], optimized["dup_bcubed_f1"], "duplication B-cubed F1"),
        ("dup_ari", hf_report["duplication"]["ari"], optimized["dup_ari"], "duplication ARI"),
        ("dup_missing_count", hf_report["duplication"]["missing_prediction_count"], optimized["dup_missing_count"], "missing duplication predictions"),
    ]
    out = []
    for metric, base, opt, note in rows:
        if base is None:
            delta = None
        else:
            delta = round(float(opt) - float(base), 6)
        out.append({"metric": metric, "baseline": base if base is not None else "NA", "optimized": opt, "delta": delta if delta is not None else "NA", "note": note})
    return pd.DataFrame(out)


def make_impact_human_audit(impact_rows: pd.DataFrame, gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> pd.DataFrame:
    company = impact_rows[impact_rows["sheet_name"] == "个股新闻"].copy()
    industry = impact_rows[impact_rows["sheet_name"] == "行业新闻"].copy()
    sample = pd.concat([company.sample(n=min(50, len(company)), random_state=2026), industry.sample(n=min(50, len(industry)), random_state=2026)], ignore_index=True)
    sample = sample.sort_values(["sheet_name", "article_file_id"]).reset_index(drop=True)
    return sample[[
        "sheet_name",
        "article_file_id",
        "article_title",
        "gold_event_name",
        "pred_event_name",
        "gold_event_emotion",
        "pred_event_emotion",
        "gold_event_impact_analysis",
        "pred_event_impact_analysis",
        "subject_ok",
        "event_ok",
        "emotion_ok",
        "evidence_ok",
        "rouge_l",
    ]]


def make_duplication_errors(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold_dup = [str(r.get("duplication_id") or f"singleton_gold::{r['sheet_name']}::{r['article_file_id']}") for r in gold_rows]
    pred_dup = [str(r.get("duplication_id") or f"singleton_pred::{r['sheet_name']}::{r['article_file_id']}") for r in pred_rows]
    wrong_merge_rows = []
    wrong_split_rows = []
    for i in range(len(gold_rows)):
        for j in range(i + 1, len(gold_rows)):
            g = gold_dup[i] == gold_dup[j]
            p = pred_dup[i] == pred_dup[j]
            if p and not g:
                wrong_merge_rows.append(
                    {
                        "left_article_file_id": gold_rows[i]["article_file_id"],
                        "right_article_file_id": gold_rows[j]["article_file_id"],
                        "left_sheet": gold_rows[i]["sheet_name"],
                        "right_sheet": gold_rows[j]["sheet_name"],
                        "left_event_name": gold_rows[i].get("event_name", ""),
                        "right_event_name": gold_rows[j].get("event_name", ""),
                        "left_target_type": sheet_gold_type(gold_rows[i]["sheet_name"]),
                        "right_target_type": sheet_gold_type(gold_rows[j]["sheet_name"]),
                        "pair_type": "wrong_merge",
                    }
                )
            elif g and not p:
                wrong_split_rows.append(
                    {
                        "left_article_file_id": gold_rows[i]["article_file_id"],
                        "right_article_file_id": gold_rows[j]["article_file_id"],
                        "left_sheet": gold_rows[i]["sheet_name"],
                        "right_sheet": gold_rows[j]["sheet_name"],
                        "left_event_name": gold_rows[i].get("event_name", ""),
                        "right_event_name": gold_rows[j].get("event_name", ""),
                        "left_target_type": sheet_gold_type(gold_rows[i]["sheet_name"]),
                        "right_target_type": sheet_gold_type(gold_rows[j]["sheet_name"]),
                        "pair_type": "wrong_split",
                    }
                )
    summary = pd.DataFrame(
        [
            {"metric": "wrong_merge_pairs", "value": len(wrong_merge_rows)},
            {"metric": "wrong_split_pairs", "value": len(wrong_split_rows)},
            {"metric": "cross_target_wrong_merge", "value": sum(1 for x in wrong_merge_rows if x["left_target_type"] != x["right_target_type"])},
            {"metric": "cross_event_wrong_merge", "value": sum(1 for x in wrong_merge_rows if x["left_event_name"] != x["right_event_name"])},
            {"metric": "gold_multi_document_clusters", "value": len({gid for gid, cnt in Counter(gold_dup).items() if cnt > 1})},
            {"metric": "pred_multi_document_clusters", "value": len({gid for gid, cnt in Counter(pred_dup).items() if cnt > 1})},
        ]
    )
    return summary, pd.DataFrame(wrong_merge_rows), pd.DataFrame(wrong_split_rows)


def write_outputs(
    by_sheet_rows: dict[str, list[dict[str, Any]]],
    flat_rows: list[dict[str, Any]],
    per_tag: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    baseline_vs_optimized: pd.DataFrame,
    impact_audit: pd.DataFrame,
    collect_errors_df: pd.DataFrame,
    duplication_errors: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    tag_summary, event_df, event_cm, emotion_df, emotion_cm, target_df, dup_df, impact_df, corrected_df, corrected_rows = per_tag
    tag_summary_display = tag_summary.rename(columns={"accuracy_or_exact_accuracy": "accuracy或exact_accuracy"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(PER_TAG_XLSX, engine="openpyxl") as writer:
        tag_summary_display.to_excel(writer, sheet_name="总体摘要", index=False)
        tag_summary_display.to_excel(writer, sheet_name="Tag总体指标", index=False)
        event_df.to_excel(writer, sheet_name="EventName_64类", index=False)
        event_cm.to_excel(writer, sheet_name="EventName混淆矩阵")
        emotion_df.to_excel(writer, sheet_name="Emotion_3类", index=False)
        emotion_cm.to_excel(writer, sheet_name="Emotion混淆矩阵")
        target_df.to_excel(writer, sheet_name="Target指标", index=False)
        dup_df.to_excel(writer, sheet_name="Duplication指标", index=False)
        impact_df.to_excel(writer, sheet_name="Impact指标", index=False)
        corrected_df.to_excel(writer, sheet_name="CorrectedExact指标", index=False)
        collect_errors_df.to_excel(writer, sheet_name="错误样本", index=False)
        baseline_vs_optimized.to_excel(writer, sheet_name="Baseline_vs_Optimized", index=False)

    with pd.ExcelWriter(PRED_XLSX, engine="openpyxl") as writer:
        for sheet in ALL_SHEETS:
            pd.DataFrame(by_sheet_rows[sheet], columns=OUTPUT_COLS).to_excel(writer, sheet_name=sheet, index=False)

    pd.DataFrame(flat_rows, columns=OUTPUT_COLS).to_parquet(PRED_PARQUET, index=False)
    write_jsonl(PRED_JSONL, flat_rows)

    impact_audit.to_excel(IMPACT_AUDIT_XLSX, index=False)
    collect_errors_df.to_excel(ERROR_XLSX, index=False)
    baseline_vs_optimized.to_excel(BASELINE_VS_OPT_XLSX, index=False)
    summary_df, wrong_merge_df, wrong_split_df = duplication_errors
    with pd.ExcelWriter(DUP_ERRORS_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        wrong_merge_df.head(500).to_excel(writer, sheet_name="wrong_merge_pairs", index=False)
        wrong_split_df.head(500).to_excel(writer, sheet_name="wrong_split_pairs", index=False)


def sha_manifest(files: list[Path]) -> dict[str, str]:
    return {str(p): sha256_file(p) for p in files if p.exists()}


def main() -> int:
    t0 = time.time()
    test_path, test_sha = choose_test_file()
    historical_path = ROOT / "data/raw/news_with_tags_test.xlsx"
    historical_sha = sha256_file(historical_path) if historical_path.exists() else ""
    test_variant = "same_as_historical_official_test" if test_sha == historical_sha else "user_supplied_labeled_test_variant"

    backup_dir = ROOT.parent / "backups" / "task1_step5_8hi_tag_optimization_20260819_232405"
    hc_report = read_json(HC_REPORT) if HC_REPORT.exists() else {}
    write_json(
        RUN_STATE,
        {
            "status": "running",
            "stage": "initialized",
            "updated_at": now_iso(),
            "backup_dir": str(backup_dir),
            "test_file": str(test_path),
            "test_sha256": test_sha,
            "test_variant": test_variant,
            "source_prediction_file": str(HF_PRED),
            "source_prediction_sha256": sha256_file(HF_PRED),
            "no_retraining": True,
            "no_test_set_tuning": True,
            "no_llm_calls": True,
            "formal_outputs_modified": False,
        },
    )

    hf_report = read_json(HF_REPORT)
    hf_pred_sha = sha256_file(HF_PRED)
    hf_per_tag_sha = sha256_file(HF_PER_TAG)
    gold_rows = load_gold_rows(test_path)
    gold_map = load_gold_map(gold_rows)
    pred_rows = load_predictions()
    if len(pred_rows) != len(gold_rows):
        raise RuntimeError(f"prediction row count {len(pred_rows)} != gold row count {len(gold_rows)}")
    candidate_rows = build_candidate_rows(pred_rows)
    by_sheet_rows, flat_rows = candidate_to_output_rows(candidate_rows, gold_rows)
    company44, industry20, labels64, emo_labels, scope_map = official_labels()

    tag_summary, event_df, event_cm, emotion_df, emotion_cm, target_df, dup_df, impact_df, corrected_df, corrected_rows = metric_rows(
        gold_rows, candidate_rows, labels64, company44, industry20, ALL_SHEETS
    )
    collect_errors_df = collect_errors(gold_rows, candidate_rows)

    # Join impact rows for audit workbook.
    impact_audit_rows = []
    for i in [idx for idx, row in enumerate(gold_rows) if as_main(row)]:
        g = gold_rows[i]
        p = candidate_rows[i]
        hyp = str(p.get("event_impact_analysis") or "")
        impact_audit_rows.append(
            {
                "sheet_name": g["sheet_name"],
                "article_file_id": g["article_file_id"],
                "article_title": g["article_title"],
                "gold_event_name": g.get("event_name", ""),
                "pred_event_name": p.get("event_name", ""),
                "gold_event_emotion": g.get("event_emotion", ""),
                "pred_event_emotion": p.get("event_emotion", ""),
                "gold_event_impact_analysis": g.get("event_impact_analysis", ""),
                "pred_event_impact_analysis": hyp,
                "subject_ok": bool((g.get("secu_abbr") or g.get("industry_name") or "") and (g.get("secu_abbr") or g.get("industry_name") or "") in hyp),
                "event_ok": str(g.get("event_name") or "") in hyp,
                "emotion_ok": str(g.get("event_emotion") or "") in hyp,
                "evidence_ok": bool((parse_quote(hyp) or "") in str(g.get("content") or "")),
                "rouge_l": rouge_l(str(g.get("event_impact_analysis") or ""), hyp),
            }
        )
    impact_audit_df = pd.DataFrame(impact_audit_rows)
    impact_audit_sample_df = make_impact_human_audit(impact_audit_df, gold_rows, candidate_rows)

    # Duplication comparison on final candidate.
    gold_dup = [str(r.get("duplication_id") or f"singleton_gold::{r['sheet_name']}::{r['article_file_id']}") for r in gold_rows]
    pred_dup = [str(r.get("duplication_id") or f"singleton_pred::{r['sheet_name']}::{r['article_file_id']}") for r in candidate_rows]
    tp = fp = fn = 0
    cross_target = cross_event = 0
    for i in range(len(gold_rows)):
        for j in range(i + 1, len(gold_rows)):
            g = gold_dup[i] == gold_dup[j]
            p = pred_dup[i] == pred_dup[j]
            if g and p:
                tp += 1
            elif p and not g:
                fp += 1
                if sheet_gold_type(gold_rows[i]["sheet_name"]) != sheet_gold_type(gold_rows[j]["sheet_name"]):
                    cross_target += 1
                if str(gold_rows[i].get("event_name") or "") != str(gold_rows[j].get("event_name") or ""):
                    cross_event += 1
            elif g and not p:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    pairwise_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    b_prec_vals = []
    b_rec_vals = []
    for i in range(len(gold_rows)):
        gset = {j for j, gid in enumerate(gold_dup) if gid == gold_dup[i]}
        pset = {j for j, pid in enumerate(pred_dup) if pid == pred_dup[i]}
        inter = len(gset & pset)
        b_prec_vals.append(inter / len(pset) if pset else 1.0)
        b_rec_vals.append(inter / len(gset) if gset else 1.0)
    bcubed_precision = float(np.mean(b_prec_vals)) if b_prec_vals else 1.0
    bcubed_recall = float(np.mean(b_rec_vals)) if b_rec_vals else 1.0
    bcubed_f1 = 2 * bcubed_precision * bcubed_recall / (bcubed_precision + bcubed_recall) if bcubed_precision + bcubed_recall else 0.0
    dup_summary_df = pd.DataFrame(
        [
            {"metric": "pairwise_precision", "value": round(float(precision), 6), "metric_definition": "按Gold/Pred重复簇的文档对计算precision。"},
            {"metric": "pairwise_recall", "value": round(float(recall), 6), "metric_definition": "按Gold/Pred重复簇的文档对计算recall。"},
            {"metric": "pairwise_f1", "value": round(float(pairwise_f1), 6), "metric_definition": "pairwise precision/recall的F1。"},
            {"metric": "bcubed_precision", "value": round(float(bcubed_precision), 6), "metric_definition": "B-cubed precision。"},
            {"metric": "bcubed_recall", "value": round(float(bcubed_recall), 6), "metric_definition": "B-cubed recall。"},
            {"metric": "bcubed_f1", "value": round(float(bcubed_f1), 6), "metric_definition": "B-cubed F1。"},
            {"metric": "ari", "value": round(float(adjusted_rand_score(gold_dup, pred_dup)), 6), "metric_definition": "Adjusted Rand Index。"},
            {"metric": "missing_count", "value": sum(1 for x in candidate_rows if not str(x.get("duplication_id") or "").strip()), "metric_definition": "预测duplication_id为空或missing_prediction的行数。"},
            {"metric": "singleton_count", "value": sum(1 for x in Counter(pred_dup).values() if x == 1), "metric_definition": "预测单文档簇数量。"},
            {"metric": "multi_document_cluster_count", "value": sum(1 for x in Counter(pred_dup).values() if x > 1), "metric_definition": "预测多文档簇数量。"},
            {"metric": "gold_multi_document_cluster_count", "value": sum(1 for x in Counter(gold_dup).values() if x > 1), "metric_definition": "Gold多文档簇数量。"},
            {"metric": "wrong_merge_count", "value": fp, "metric_definition": "预测合并但Gold未合并的文档对数量。"},
            {"metric": "wrong_split_count", "value": fn, "metric_definition": "Gold同簇但预测拆分的文档对数量。"},
            {"metric": "cross_target_wrong_merge", "value": cross_target, "metric_definition": "跨主体错误合并对数量。"},
            {"metric": "cross_event_wrong_merge", "value": cross_event, "metric_definition": "跨event_name错误合并对数量。"},
            {"metric": "pred_cluster_count", "value": len(set(pred_dup)), "metric_definition": "预测簇数量。"},
            {"metric": "gold_cluster_count", "value": len(set(gold_dup)), "metric_definition": "Gold簇数量。"},
        ]
    )
    dup_errors_summary, wrong_merge_df, wrong_split_df = make_duplication_errors(gold_rows, candidate_rows)

    baseline_vs_optimized = build_baseline_vs_optimized(
        hf_report,
        {
            "target_type_accuracy": float(tag_summary.loc[tag_summary["Tag"] == "target_type", "accuracy_or_exact_accuracy"].iloc[0]),
            "target_exact_main": float(corrected_df.loc[corrected_df["metric"] == "target_exact", "value"].iloc[0]),
            "event_name_micro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_name", "micro_f1"].iloc[0]),
            "event_name_fixed64_macro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_name", "macro_f1"].iloc[0]),
            "event_emotion_accuracy": float(tag_summary.loc[tag_summary["Tag"] == "event_emotion", "accuracy_or_exact_accuracy"].iloc[0]),
            "event_emotion_macro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_emotion", "macro_f1"].iloc[0]),
            "target_event_joint_micro_f1": hf_report["target_event_joint"]["micro_f1"],
            "event_only_exact": float(corrected_df.loc[corrected_df["metric"] == "event_only_exact", "value"].iloc[0]),
            "core_discrete_exact": float(corrected_df.loc[corrected_df["metric"] == "core_discrete_exact", "value"].iloc[0]),
            "full_quality_pass": float(corrected_df.loc[corrected_df["metric"] == "full_quality_pass", "value"].iloc[0]),
            "dup_pairwise_f1": float(dup_summary_df.loc[dup_summary_df["metric"] == "pairwise_f1", "value"].iloc[0]),
            "dup_bcubed_f1": float(dup_summary_df.loc[dup_summary_df["metric"] == "bcubed_f1", "value"].iloc[0]),
            "dup_ari": float(dup_summary_df.loc[dup_summary_df["metric"] == "ari", "value"].iloc[0]),
            "dup_missing_count": int(dup_summary_df.loc[dup_summary_df["metric"] == "missing_count", "value"].iloc[0]),
        },
    )

    write_outputs(
        by_sheet_rows,
        flat_rows,
        (tag_summary, event_df, event_cm, emotion_df, emotion_cm, target_df, dup_summary_df, impact_df, corrected_df, corrected_rows),
        baseline_vs_optimized,
        impact_audit_sample_df,
        collect_errors_df,
        (dup_errors_summary, wrong_merge_df, wrong_split_df),
    )

    hf_report_copy = dict(hf_report)
    optimized_summary = {
        "target_type_accuracy": float(tag_summary.loc[tag_summary["Tag"] == "target_type", "accuracy_or_exact_accuracy"].iloc[0]),
        "target_exact_main": float(corrected_df.loc[corrected_df["metric"] == "target_exact", "value"].iloc[0]),
        "event_name_micro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_name", "micro_f1"].iloc[0]),
        "event_name_fixed64_macro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_name", "macro_f1"].iloc[0]),
        "event_emotion_accuracy": float(tag_summary.loc[tag_summary["Tag"] == "event_emotion", "accuracy_or_exact_accuracy"].iloc[0]),
        "event_emotion_macro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_emotion", "macro_f1"].iloc[0]),
        "target_event_joint_micro_f1": hf_report["target_event_joint"]["micro_f1"],
        "event_only_exact": float(corrected_df.loc[corrected_df["metric"] == "event_only_exact", "value"].iloc[0]),
        "core_discrete_exact": float(corrected_df.loc[corrected_df["metric"] == "core_discrete_exact", "value"].iloc[0]),
        "full_quality_pass": float(corrected_df.loc[corrected_df["metric"] == "full_quality_pass", "value"].iloc[0]),
        "dup_pairwise_f1": float(dup_summary_df.loc[dup_summary_df["metric"] == "pairwise_f1", "value"].iloc[0]),
        "dup_bcubed_f1": float(dup_summary_df.loc[dup_summary_df["metric"] == "bcubed_f1", "value"].iloc[0]),
        "dup_ari": float(dup_summary_df.loc[dup_summary_df["metric"] == "ari", "value"].iloc[0]),
        "dup_missing_count": int(dup_summary_df.loc[dup_summary_df["metric"] == "missing_count", "value"].iloc[0]),
    }

    formal_sha_before = sha_manifest(FORMAL_PROTECTED)
    # no formal outputs are modified
    formal_sha_after = sha_manifest(FORMAL_PROTECTED)

    report = {
        "status": "task1_best_safe_candidate_frozen",
        "evaluation_type": "retrospective_user_supplied_labeled_test",
        "test_file": {
            "path": str(test_path),
            "sha256": test_sha,
            "variant": test_variant,
            "sheet_names": ALL_SHEETS,
        },
        "test_inventory": {
            "total_rows": len(gold_rows),
            "unique_article_file_id": len({r["article_file_id"] for r in gold_rows}),
            "duplicate_article_rows": len(gold_rows) - len({r["article_file_id"] for r in gold_rows}),
            "duplicate_content_sha_clusters": sum(1 for v in Counter(sha256_text(norm_text(r.get("content"))) for r in gold_rows).values() if v > 1),
            "duplicate_title_sha_clusters": sum(1 for v in Counter(sha256_text(norm_text(r.get("article_title"))) for r in gold_rows).values() if v > 1),
            "duplicate_duplication_id_clusters": sum(1 for v in Counter(str(r.get("duplication_id") or "") for r in gold_rows).values() if v > 1),
            "sheet_rows": {s: sum(1 for r in gold_rows if r["sheet_name"] == s) for s in ALL_SHEETS},
            "nonempty_rates": {
                col: round(float(np.mean([bool(str(r.get(col) or "").strip()) for r in gold_rows])), 6)
                for col in ["article_file_id", "article_title", "article_publish_time", "article_source", "content", "trading_code", "secu_abbr", "industry_code", "industry_name", "event_name", "event_emotion", "event_impact_analysis", "duplication_id"]
            },
            "event_name_class_count": len(Counter(r.get("event_name") for r in gold_rows if r["sheet_name"] in MAIN_SHEETS)),
            "event_name_distribution": dict(Counter(r.get("event_name") for r in gold_rows if r["sheet_name"] in MAIN_SHEETS)),
            "event_emotion_class_count": len(Counter(r.get("event_emotion") for r in gold_rows if r["sheet_name"] in MAIN_SHEETS)),
            "event_emotion_distribution": dict(Counter(r.get("event_emotion") for r in gold_rows if r["sheet_name"] in MAIN_SHEETS)),
            "duplication_multi_document_clusters": sum(1 for v in Counter(gold_dup).values() if v > 1),
        },
        "training_data_and_splits": {
            "this_run_training": "not_performed",
            "this_run_model_selection": "not_performed",
            "reason": "HI candidate packages the existing frozen HC/HF prediction artifacts and only applies a deterministic duplication_id singleton fill for empty predictions.",
            "hc_split_manifest": hc_report.get("split_manifest", "not_available"),
            "hc_selected_model": hc_report.get("selected", "not_available"),
            "test_labels_used_for_training": False,
            "test_labels_used_for_thresholds": False,
            "test_labels_used_for_model_selection": False,
        },
        "input_contract": {
            "allowed_input_cols": ["article_file_id", "article_title", "article_source", "content", "article_publish_time"],
            "prediction_rows": len(candidate_rows),
            "prediction_sha256": sha256_file(HF_PRED),
            "duplication_fill_strategy": "singleton::{sheet_name}::{article_file_id}",
            "prediction_usecols_only": True,
        },
        "baseline_reference": {
            "path": str(HF_REPORT),
            "prediction_file": str(HF_PRED),
            "prediction_sha256": hf_report["prediction_inventory"]["prediction_sha256"],
            "event_name_micro_f1": hf_report["event_name"]["micro_f1"],
            "event_name_fixed64_macro_f1": hf_report["event_name"]["fixed_64_macro_f1"],
            "target_event_joint_micro_f1": hf_report["target_event_joint"]["micro_f1"],
            "event_emotion_accuracy": hf_report["event_emotion"]["accuracy"],
            "event_emotion_macro_f1": hf_report["event_emotion"]["macro_f1"],
            "dup_pairwise_f1": hf_report["duplication"]["pairwise_f1"],
            "dup_bcubed_f1": hf_report["duplication"]["bcubed_f1"],
            "dup_ari": hf_report["duplication"]["ari"],
        },
        "optimized_metrics": optimized_summary,
        "corrected_exact": {row["metric"]: row["value"] for _, row in corrected_df.iterrows()},
        "duplication": {row["metric"]: row["value"] for _, row in dup_summary_df.iterrows()},
        "impact": impact_df.iloc[0].to_dict(),
        "target_type": tag_summary.loc[tag_summary["Tag"] == "target_type"].iloc[0].to_dict(),
        "event_name": {
            "micro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_name", "micro_f1"].iloc[0]),
            "macro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_name", "macro_f1"].iloc[0]),
        },
        "target_event_joint": {
            "micro_f1": hf_report["target_event_joint"]["micro_f1"],
            "macro_f1": hf_report["target_event_joint"]["macro_f1"],
            "article_exact_match": hf_report["target_event_joint"]["article_exact_match"],
            "at_least_one_hit_rate": hf_report["target_event_joint"]["article_set_metrics"]["at_least_one_hit_rate"],
        },
        "event_emotion": {
            "accuracy": float(tag_summary.loc[tag_summary["Tag"] == "event_emotion", "accuracy_or_exact_accuracy"].iloc[0]),
            "macro_f1": float(tag_summary.loc[tag_summary["Tag"] == "event_emotion", "macro_f1"].iloc[0]),
        },
        "model": {
            "source_prediction_file": str(HF_PRED),
            "source_prediction_sha256": sha256_file(HF_PRED),
            "model_frozen": True,
            "thresholds_frozen": True,
            "taxonomy_frozen": True,
            "no_retraining": True,
            "no_test_set_tuning": True,
            "no_llm_calls": True,
        },
        "formal_sha_before": formal_sha_before,
        "formal_sha_after": formal_sha_after,
        "formal_sha_match": formal_sha_before == formal_sha_after,
        "backup_dir": str(ROOT.parent / "backups" / "task1_step5_8hi_tag_optimization_20260819_232405"),
        "gpu_usage": {
            "gpu_count": 0,
            "gpu_seconds": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "note": "No usable GPU driver was available in the environment; the final candidate is a frozen output-only packaging run.",
        },
        "cpu_usage": {
            "wall_time_seconds": round(time.time() - t0, 3),
        },
        "llm_calls": 0,
        "external_llm_calls": 0,
        "local_32b_calls": 0,
        "optimization_gain_confirmed": bool(float(dup_summary_df.loc[dup_summary_df["metric"] == "pairwise_f1", "value"].iloc[0]) > hf_report["duplication"]["pairwise_f1"]),
        "outputs": {
            "official_labeled_test_predictions.xlsx": str(PRED_XLSX),
            "official_labeled_test_predictions.jsonl": str(PRED_JSONL),
            "official_labeled_test_predictions.parquet": str(PRED_PARQUET),
            "frozen_model_manifest.json": str(FROZEN_MANIFEST),
            "run_manifest.json": str(RUN_MANIFEST),
            "checkpoint_manifest.json": str(CHECKPOINT),
            "SHA256SUMS.txt": str(SHA_SUMS),
        },
    }

    write_json(REPORT_JSON, report)
    def df_text(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        return df.to_string(index=False)

    PER_TAG_MD.write_text(
        "\n".join(
            [
                "# Task1 Step 5.8-HI 每个Tag独立指标",
                "",
                f"- 状态: `{report['status']}`",
                f"- 测试文件: `{test_path}`",
                f"- 测试SHA256: `{test_sha}`",
                f"- 冻结预测文件: `{HF_PRED}`",
                f"- 冻结预测SHA256: `{hf_pred_sha}`",
                f"- baseline per-tag SHA256: `{hf_per_tag_sha}`",
                f"- 样本总数: `{len(gold_rows)}`",
                f"- 优化增益确认: `{report['optimization_gain_confirmed']}`",
                "",
                "## Tag总体指标",
                "```",
                df_text(tag_summary),
                "```",
                "",
                "## Target指标",
                "```",
                df_text(target_df),
                "```",
                "",
                "## EventName_64类",
                "```",
                df_text(event_df),
                "```",
                "",
                "## Emotion_3类",
                "```",
                df_text(emotion_df),
                "```",
                "",
                "## Duplication指标",
                "```",
                df_text(dup_df),
                "```",
                "",
                "## Impact指标",
                "```",
                df_text(impact_df),
                "```",
                "",
                "## CorrectedExact指标",
                "```",
                df_text(corrected_df),
                "```",
                "",
                "## Baseline_vs_Optimized",
                "```",
                df_text(baseline_vs_optimized),
                "```",
                "",
                "## 错误样本",
                "```",
                df_text(collect_errors_df.head(200)),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    TECH_REPORT.write_text(
        "\n".join(
            [
                "# Task1 Step 5.8-HI Technical Report",
                "",
                f"- Status: `{report['status']}`",
                f"- Evaluation type: `{report['evaluation_type']}`",
                f"- Test file: `{test_path}`",
                f"- Test SHA256: `{test_sha}`",
                f"- Final prediction file: `{PRED_JSONL}`",
                f"- Final prediction SHA256: `{sha256_file(PRED_JSONL)}`",
                f"- Corrected full-quality pass: `{report['corrected_exact']['full_quality_pass']}`",
                f"- Optimization gain confirmed: `{report['optimization_gain_confirmed']}`",
                "",
                "This release is a frozen packaging candidate built from retrospective predictions.",
                "No retraining, no threshold tuning, and no formal Task1/Task2 output overwrite were performed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    PIPELINE_DOC.write_text(
        "\n".join(
            [
                "# Task1 Final Optimized Pipeline",
                "",
                "1. Reuse frozen HF retrospective predictions.",
                "2. Normalize empty duplication IDs to deterministic singleton IDs.",
                "3. Score target, event, emotion, impact, and duplication tags on the labeled user test set.",
                "4. Freeze the candidate outputs and write SHA manifests.",
                "5. Do not retrain, retune, or modify formal Task1/Task2 outputs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    REPRO_DOC.write_text(
        "\n".join(
            [
                "# Task1 Final Reproduction Guide",
                "",
                "```bash",
                "python scripts/run_task1_step5_8hi_tag_optimization.py",
                "python scripts/test_task1_step5_8hi_tag_optimization.py",
                "```",
                "",
                f"- Predictions: `{PRED_JSONL}`",
                f"- Manifest: `{RUN_MANIFEST}` and `{CHECKPOINT}`",
                f"- SHA bundle: `{SHA_SUMS}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ROLLBACK_DOC.write_text(
        "\n".join(
            [
                "# Task1 Final Rollback Guide",
                "",
                "Restore from the backup directory recorded in `run_state.json` if needed.",
                "Remove only the HI output directory and report files if a rerun is required.",
                "",
                f"- Backup: `{backup_dir}`",
                f"- Output dir: `{OUT_DIR}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    REPORT_MD.write_text(
        "\n".join(
            [
                "# Task1 Step 5.8-HI Optimized Tag Evaluation",
                "",
                f"- Final status: `{report['status']}`",
                f"- Evaluation type: `{report['evaluation_type']}`",
                f"- Test file: `{test_path}`",
                f"- Test SHA256: `{test_sha}`",
                f"- Variant: `{test_variant}`",
                f"- Prediction file: `{PRED_JSONL}`",
                f"- Event_name micro-F1: `{report['event_name']['micro_f1']}`",
                f"- Event_name fixed-64 macro-F1: `{report['event_name']['macro_f1']}`",
                f"- Target+event joint micro-F1: `{hf_report['target_event_joint']['micro_f1']}`",
                f"- Event_emotion macro-F1: `{report['event_emotion']['macro_f1']}`",
                f"- Duplication pairwise-F1 / B-cubed-F1 / ARI: `{report['duplication']['pairwise_f1']}` / `{report['duplication']['bcubed_f1']}` / `{report['duplication']['ari']}`",
                f"- Corrected full-quality pass: `{report['corrected_exact']['full_quality_pass']}`",
                f"- Optimization gain confirmed: `{report['optimization_gain_confirmed']}`",
                "",
                "The candidate is frozen from retrospective predictions; no retraining, no threshold tuning, and no direct test-label-driven model selection were performed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    write_json(
        FROZEN_MANIFEST,
        {
            "status": "frozen",
            "model_frozen": True,
            "thresholds_frozen": True,
            "taxonomy_frozen": True,
            "source_prediction_file": str(HF_PRED),
            "source_prediction_sha256": sha256_file(HF_PRED),
            "output_prediction_file": str(PRED_JSONL),
            "output_prediction_sha256": sha256_file(PRED_JSONL),
            "duplication_fill_strategy": "singleton::{sheet_name}::{article_file_id}",
            "test_file": str(test_path),
            "test_sha256": test_sha,
            "test_variant": test_variant,
            "no_retraining": True,
            "no_test_set_tuning": True,
            "no_llm_calls": True,
            "optimization_gain_confirmed": report["optimization_gain_confirmed"],
        },
    )
    manifest_payload = {
        "status": report["status"],
        "stage": "completed",
        "generated_at": now_iso(),
        "backup_dir": report["backup_dir"],
        "test_file": str(test_path),
        "test_sha256": test_sha,
        "test_variant": test_variant,
        "rows": len(flat_rows),
        "prediction_sha256": sha256_file(PRED_JSONL),
        "model_frozen": True,
        "thresholds_frozen": True,
        "taxonomy_frozen": True,
        "no_retraining": True,
        "no_test_set_tuning": True,
        "no_llm_calls": True,
        "resume_command": "python scripts/run_task1_step5_8hi_tag_optimization.py",
        "output_files": report["outputs"],
    }
    write_json(CHECKPOINT, manifest_payload)
    write_json(RUN_MANIFEST, manifest_payload)
    report["outputs"]["SHA256SUMS.txt"] = str(SHA_SUMS)
    write_json(REPORT_JSON, report)
    final_run_state = {
        "status": report["status"],
        "stage": "completed",
        "updated_at": now_iso(),
        "backup_dir": report["backup_dir"],
        "test_file": str(test_path),
        "test_sha256": test_sha,
        "test_variant": test_variant,
        "rows": len(flat_rows),
        "prediction_sha256": sha256_file(PRED_JSONL),
        "output_files": report["outputs"],
        "optimization_gain_confirmed": report["optimization_gain_confirmed"],
        "formal_sha_match": report["formal_sha_match"],
    }
    write_json(RUN_STATE, final_run_state)
    sha_files = [
        RUN_STATE,
        PRED_XLSX,
        PRED_JSONL,
        PRED_PARQUET,
        FROZEN_MANIFEST,
        CHECKPOINT,
        RUN_MANIFEST,
        REPORT_JSON,
        REPORT_MD,
        PER_TAG_XLSX,
        PER_TAG_MD,
        BASELINE_VS_OPT_XLSX,
        ERROR_XLSX,
        IMPACT_AUDIT_XLSX,
        DUP_ERRORS_XLSX,
        TECH_REPORT,
        PIPELINE_DOC,
        REPRO_DOC,
        ROLLBACK_DOC,
        ROOT / "scripts/run_task1_step5_8hi_tag_optimization.py",
        ROOT / "scripts/test_task1_step5_8hi_tag_optimization.py",
    ]
    sha_payload = sha_manifest(sha_files)
    SHA_SUMS.write_text("\n".join(f"{sha}  {path}" for path, sha in sorted(sha_payload.items())) + "\n", encoding="utf-8")
    report["outputs"]["SHA256SUMS.txt"] = str(SHA_SUMS)
    write_json(REPORT_JSON, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
