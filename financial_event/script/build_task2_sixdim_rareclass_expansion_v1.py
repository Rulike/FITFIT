#!/usr/bin/env python3
"""Build Task2 six-dimensional rare-class blind annotation candidates.

This script registers Independent500-v2 as a human-gold development set and
builds a new blind annotation package for rare-class expansion. It performs no
training, weight fitting, threshold tuning, prediction generation, or formal
evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 20260828
DATASET_ID = "task2_sixdim_rareclass_expansion_v1"
DEV_DATASET_ID = "human_gold_development_v2"
OUT_DIR = ROOT / "outputs/event_credibility/task2_sixdim_rareclass_expansion_v1"
REPORT_400 = ROOT / "reports/task2_sixdim_rareclass_annotation_400.xlsx"
REPORT_1000 = ROOT / "reports/task2_sixdim_rareclass_annotation_1000.xlsx"
GUIDE = ROOT / "reports/task2_sixdim_rareclass_annotation_guide.md"
DEV_CONFIG = ROOT / "configs/task2_human_gold_development_v2.json"
PROTOCOL_CONFIG = ROOT / "configs/task2_sixdim_data_protocol_v1.json"
PROTOCOL_MD = ROOT / "reports/task2_sixdim_data_protocol_v1.md"
GOLD_WORKBOOK = ROOT / "reports/task2_independent500_v2_double_blind_adjudicated.xlsx"
GOLD_SAMPLES = ROOT / "outputs/event_credibility/task2_independent500_v2/independent_samples.jsonl"
ARTICLE_INDEX = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"
MARKET_CACHE = ROOT / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v1/resset_market_daily_cache.parquet"
FINANCIAL_CACHE = ROOT / "data/processed/event_credibility/csmar_financial_cache_v2/financial_records.parquet"
FULLTEXT_EVIDENCE = ROOT / "outputs/event_credibility/fulltext_evidence.parquet"
ALL_EVENTS = ROOT / "data/processed/event_clusters/all_final_events_with_sources.jsonl"

EXPECTED_GOLD_WORKBOOK_SHA = "699f2e9bedb3a9b925f33be752166db6cf02e984c9444c85d29eee4284b27052"
EXPECTED_GOLD_SAMPLES_SHA = "0ea6679aeb3fc9c3d5d7dd311832d1e652ded2d92adb428e2481e39f6c11e18b"
EXPECTED_LABEL_DISTRIBUTION = {
    "support": 110,
    "contradict": 1,
    "insufficient": 376,
    "unclear": 13,
}
LABEL_VALUES = ["support", "contradict", "insufficient", "unclear"]
CONFIDENCE_VALUES = ["high", "medium", "low"]
YES_NO_VALUES = ["yes", "no"]

POOL_ROUTE_TARGETS = {
    "contradict-prone": 1100,
    "unclear-prone": 1000,
    "support/insufficient control": 400,
}
FINAL_ROUTE_TARGETS = {
    "contradict-prone": 450,
    "unclear-prone": 400,
    "support/insufficient control": 150,
}
SMOKE_POOL_ROUTE_TARGETS = {
    "contradict-prone": 45,
    "unclear-prone": 40,
    "support/insufficient control": 15,
}
SMOKE_FINAL_ROUTE_TARGETS = {
    "contradict-prone": 45,
    "unclear-prone": 40,
    "support/insufficient control": 15,
}

HUMAN_FIELDS = [
    "annotator_a_label",
    "annotator_a_confidence",
    "annotator_a_reason",
    "annotator_a_evidence_ids",
    "annotator_b_label",
    "annotator_b_confidence",
    "annotator_b_reason",
    "annotator_b_evidence_ids",
    "disagreement",
    "adjudicated_label",
    "adjudication_reason",
    "needs_expert_review",
]
VISIBLE_COLUMNS = [
    "sample_id",
    "cluster_id",
    "article_file_id",
    "target",
    "event_name",
    "claim_title",
    "claim_text",
    "publish_time",
    "structured_event_arguments",
    "closed_track_evidence",
    "evidence_sources",
    "evidence_times",
    "financial_context",
    "market_context",
    *HUMAN_FIELDS,
]
FORBIDDEN_VISIBLE_PATTERNS = [
    "candidate_route",
    "route=",
    "contradict-prone",
    "unclear-prone",
    "candidate_quality_score",
    "dianjin",
    "round1",
    "model_prediction",
    "predicted_label",
    "prediction",
    "sixdim_score",
    "六维评分",
    "old_label",
    "gold_label",
    "gold=",
    "rule_conclusion",
    "rule_verdict",
    "market_consistency_verdict",
    "financial_consistency_verdict",
]

CODE_RE = re.compile(
    r"(?<!\d)(000\d{3}|001\d{3}|002\d{3}|003\d{3}|300\d{3}|301\d{3}|600\d{3}|601\d{3}|603\d{3}|605\d{3}|688\d{3}|689\d{3})(?:\.(?:SH|SZ|sh|sz))?(?!\d)"
)
DATE_RE = re.compile(r"(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})")
MONTH_DAY_RE = re.compile(r"(\d{1,2})月(\d{1,2})日")
FUND_OR_INDEX_TERMS = [
    "ETF",
    "etf",
    "指数基金",
    "成份股",
    "成分股",
    "中证",
    "国证",
    "沪深300",
    "深证成指",
    "上证指数",
]
REVERSE_TERMS = [
    "澄清",
    "否认",
    "不属实",
    "不实",
    "失实",
    "不存在",
    "终止",
    "取消",
    "撤回",
    "更正",
    "未参与",
    "未通过",
    "尚未完成",
    "仅为意向",
    "传闻不实",
    "下调",
    "修正",
    "问询回复",
    "辟谣",
]
AMBIGUITY_TERMS = [
    "子公司",
    "母公司",
    "集团",
    "旗下",
    "简称",
    "意向",
    "计划",
    "拟",
    "预计",
    "有望",
    "尚未",
    "目前",
    "截至",
    "口径",
    "同比",
    "环比",
    "传闻",
    "是否",
    "问询",
    "回应",
    "多个",
    "不同",
    "分歧",
    "混淆",
]
SUPPORT_TERMS = [
    "公告",
    "披露",
    "发布",
    "获批",
    "批准",
    "中标",
    "签订",
    "签署",
    "完成",
    "落地",
    "上市",
    "增长",
    "上涨",
    "涨停",
    "取得",
    "通过",
]
MARKET_TERMS = [
    "涨停",
    "跌停",
    "收涨",
    "收跌",
    "上涨",
    "下跌",
    "大涨",
    "大跌",
    "涨幅",
    "跌幅",
    "涨超",
    "跌超",
    "涨逾",
    "跌逾",
]
FINANCIAL_TERMS = [
    "营业收入",
    "主营收入",
    "营收",
    "收入",
    "经营现金流",
    "现金流",
    "净资产收益率",
    "ROE",
    "roe",
]
POSITIVE_FINANCIAL_TERMS = [
    "同比增长",
    "增长",
    "增加",
    "提升",
    "改善",
    "转正",
    "上升",
    "同比增",
    "大增",
    "回暖",
    "修复",
]
NEGATIVE_FINANCIAL_TERMS = [
    "同比下降",
    "下降",
    "减少",
    "下滑",
    "降低",
    "恶化",
    "转负",
    "亏损扩大",
    "同比降",
    "大降",
    "承压",
    "亏损",
]
OFFICIAL_SOURCE_HINTS = ["巨潮", "交易所", "深交所", "上交所", "互动易", "公告", "证监会", "公司"]


@dataclass(frozen=True)
class HistorySource:
    name: str
    category: str
    path: Path
    trust_status: str
    notes: str = ""


HISTORY_SOURCES = [
    HistorySource("independent500_v2_unlabeled_input", "current_development_set_exclusion", GOLD_SAMPLES, "trusted_identifier_source"),
    HistorySource("independent500_v2_adjudicated_workbook", "current_development_set_exclusion", GOLD_WORKBOOK, "trusted_identifier_source"),
    HistorySource("independent500_v2_locked_ids", "current_development_set_exclusion", ROOT / "configs/task2_independent500_v2_locked_ids.json", "trusted_identifier_source"),
    HistorySource("independent500_v1_unlabeled_input", "retired_independent_test_exclusion", ROOT / "outputs/event_credibility/task2_independent500_v1/independent_samples.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("independent500_v1_locked_ids", "retired_independent_test_exclusion", ROOT / "configs/task2_independent500_locked_ids.json", "unverified_for_sixdim_gold"),
    HistorySource("hybrid_v3_annotation_candidates", "historical_task2_annotation_or_training_asset", ROOT / "outputs/event_credibility/task2_hybrid_v3/annotation_candidates.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("hybrid_v3_human_annotation_workbook", "historical_task2_annotation_or_training_asset", ROOT / "reports/task2_hybrid_v3_human_annotation_1500.xlsx", "unverified_for_sixdim_gold"),
    HistorySource("hybrid_v3_clean_evidence", "historical_task2_annotation_or_training_asset", ROOT / "outputs/event_credibility/task2_hybrid_v3/clean_evidence_v1/clean_evidence_1500.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("hybrid_v3_silver_final", "historical_task2_annotation_or_training_asset", ROOT / "outputs/event_credibility/task2_hybrid_v3/heterogeneous_silver_v2/silver_final_1500.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("hybrid_v3_silver_training_eligible", "historical_task2_annotation_or_training_asset", ROOT / "outputs/event_credibility/task2_hybrid_v3/heterogeneous_silver_v2/silver_training_eligible.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("hybrid_v3_fusion_train", "historical_task2_train_dev_test_asset", ROOT / "outputs/event_credibility/task2_hybrid_v3/fusion_model_v1/train.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("hybrid_v3_fusion_dev", "historical_task2_train_dev_test_asset", ROOT / "outputs/event_credibility/task2_hybrid_v3/fusion_model_v1/dev.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("hybrid_v3_fusion_internal_test", "historical_task2_train_dev_test_asset", ROOT / "outputs/event_credibility/task2_hybrid_v3/fusion_model_v1/internal_test.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("dea_v1_gray_samples", "historical_task2_audit_asset", ROOT / "outputs/event_credibility/dea_v1_gray/sample_rows.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("dea_v2_gray_samples", "historical_task2_audit_asset", ROOT / "outputs/event_credibility/dea_v2_gray/sample_rows.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("dea_v2_cached_gray_samples", "historical_task2_audit_asset", ROOT / "outputs/event_credibility/dea_v2_cached_gray/sample_rows.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("dea_cached_manual_audit", "historical_task2_human_audit_asset", ROOT / "reports/task2_dea_cached_human_audit.xlsx", "unverified_for_sixdim_gold"),
    HistorySource("dea_manual_audit_jsonl", "historical_task2_human_audit_asset", ROOT / "reports/task2_dea_manual_audit.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("dea_v2_manual_audit_jsonl", "historical_task2_human_audit_asset", ROOT / "reports/task2_dea_v2_manual_audit.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("dea_v2_cached_manual_audit_jsonl", "historical_task2_human_audit_asset", ROOT / "reports/task2_dea_v2_cached_manual_audit.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("safety_gate_independent_samples", "historical_task2_human_audit_asset", ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent/independent_samples.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("safety_gate_primary_samples", "historical_task2_human_audit_asset", ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent/primary_samples.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("safety_gate_stress_samples", "historical_task2_human_audit_asset", ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent/contradiction_stress_samples.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("isr_candidates", "historical_task2_unlabeled_candidate_asset", ROOT / "outputs/event_credibility/isr_candidate_audit_v1/isr_candidates.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("isr_claim_fix_candidates", "historical_task2_unlabeled_candidate_asset", ROOT / "outputs/event_credibility/isr_candidate_audit_v1/claim_fix_candidates.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("isr_safety_gate_fix_candidates", "historical_task2_unlabeled_candidate_asset", ROOT / "outputs/event_credibility/isr_candidate_audit_v1/safety_gate_fix_candidates.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("task1_structure_independent60_source", "task1_task3_same_source_exclusion", ROOT / "outputs/task1_step5_7i_structure_expansion/independent_frozen_test_60_source.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("task1_structure_confirmed60", "task1_task3_same_source_exclusion", ROOT / "outputs/task1_step5_7j_structure_confirmation/independent_frozen_test_confirmed.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("task3_split_test", "task1_task3_same_source_exclusion", ROOT / "outputs/task3/splits/test.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("task3_task2_links", "task1_task3_same_source_exclusion", ROOT / "outputs/task3/task2_evidence_links.jsonl", "unverified_for_sixdim_gold"),
    HistorySource("event_cluster_all_events", "task1_structured_event_output_exclusion", ALL_EVENTS, "unverified_for_sixdim_gold"),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def norm(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def compact_text(value: Any, limit: int = 100000) -> str:
    text = norm(value).lower()
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
    return text[:limit]


def text_sha256(value: Any) -> str:
    text = compact_text(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def first_date(value: Any) -> str:
    text = norm(value)
    match = DATE_RE.search(text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return ""


def parse_datetime(value: Any) -> datetime | None:
    text = norm(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().replace(tzinfo=None)


def deterministic_key(*parts: Any, seed: int = SEED) -> str:
    payload = "|".join(norm(part) for part in parts) + f"|seed={seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sequence_similarity(left: Any, right: Any) -> float:
    a = compact_text(left, 360)
    b = compact_text(right, 360)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    max_possible = 2.0 * min(len(a), len(b)) / max(1, len(a) + len(b))
    if max_possible < 0.86:
        return max_possible
    matcher = SequenceMatcher(None, a, b)
    quick = matcher.quick_ratio()
    if quick < 0.86:
        return quick
    return matcher.ratio()


def date_distance_days(left: Any, right: Any) -> int | None:
    a = parse_datetime(first_date(left) or left)
    b = parse_datetime(first_date(right) or right)
    if a is None or b is None:
        return None
    return abs((a.date() - b.date()).days)


def is_near_duplicate(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, str, float]:
    if left.get("target_key") and right.get("target_key") and left["target_key"] != right["target_key"]:
        return False, "", 0.0
    title_sim = sequence_similarity(left.get("title_text"), right.get("title_text"))
    claim_sim = sequence_similarity(left.get("claim_text"), right.get("claim_text"))
    distance = date_distance_days(left.get("date"), right.get("date"))
    if left.get("date") and left.get("date") == right.get("date") and title_sim >= 0.92:
        return True, "same_target_date_title_similarity_ge_0.92", title_sim
    if distance is not None and distance <= 2 and max(title_sim, claim_sim) >= 0.94:
        return True, "same_target_near_date_text_similarity_ge_0.94", max(title_sim, claim_sim)
    if title_sim >= 0.98:
        return True, "same_target_high_title_similarity_ge_0.98", title_sim
    return False, "", max(title_sim, claim_sim)


def source_is_official(source: str, source_type: str) -> bool:
    text = f"{source} {source_type}"
    return source_type == "official" or any(term in text for term in OFFICIAL_SOURCE_HINTS)


def split_sentences(title: str, content: str, limit: int = 3600) -> list[str]:
    text = f"{title}。{content[:limit]}".replace("\n", "。")
    return [norm(s) for s in re.split(r"(?<=[。！？!?；;])", text) if norm(s)]


def sentence_with_terms(title: str, content: str, terms: Iterable[str], target_name: str = "", target_code: str = "") -> str:
    target_terms = [target_name, target_code, "公司"]
    for sentence in split_sentences(title, content):
        if any(term in sentence for term in terms):
            if target_name or target_code:
                if not any(term and term in sentence for term in target_terms) and not any(term and term in title for term in (target_name, target_code)):
                    continue
            return sentence[:520]
    return (norm(content) or norm(title))[:520]


def format_evidence(evidence_items: list[dict[str, Any]]) -> str:
    lines = []
    for item in evidence_items:
        parts = [
            norm(item.get("evidence_id")),
            norm(item.get("evidence_source")),
            norm(item.get("evidence_time")),
            norm(item.get("source_type")),
            "independent=yes" if item.get("is_independent_source") else "independent=no",
            norm(item.get("evidence_text")),
        ]
        lines.append(" | ".join(part for part in parts if part))
    return "\n".join(lines)


def make_evidence(
    evidence_id: str,
    source: str,
    evidence_time: str,
    text: str,
    source_type: str,
    trace_path: str,
    independent: bool = True,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "evidence_source": norm(source),
        "evidence_time": norm(evidence_time),
        "evidence_text": norm(text)[:900],
        "source_type": norm(source_type) or "unknown",
        "is_independent_source": bool(independent),
        "trace_path": trace_path,
    }


def visible_row(sample: dict[str, Any]) -> dict[str, Any]:
    row = {
        "sample_id": sample["sample_id"],
        "cluster_id": sample["cluster_id"],
        "article_file_id": sample["article_file_id"],
        "target": sample["target"],
        "event_name": sample["event_name"],
        "claim_title": sample["claim_title"],
        "claim_text": sample["claim_text"],
        "publish_time": sample["publish_time"],
        "structured_event_arguments": json.dumps(sample["structured_event_arguments"], ensure_ascii=False, sort_keys=True),
        "closed_track_evidence": format_evidence(sample["closed_track_evidence"]),
        "evidence_sources": "; ".join(dict.fromkeys(item["evidence_source"] for item in sample["closed_track_evidence"] if item.get("evidence_source"))),
        "evidence_times": "; ".join(dict.fromkeys(item["evidence_time"] for item in sample["closed_track_evidence"] if item.get("evidence_time"))),
        "financial_context": sample.get("financial_context", ""),
        "market_context": sample.get("market_context", ""),
    }
    for field in HUMAN_FIELDS:
        row[field] = ""
    return row


def safe_cell(value: Any, limit: int = 32000) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    else:
        text = norm(value)
    return text if len(text) <= limit else text[: limit - 40] + " ...[truncated]"


def write_workbook(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "annotation"
    rows = [visible_row(sample) for sample in samples]
    ws.append(VISIBLE_COLUMNS)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    auto_fill = PatternFill("solid", fgColor="F2F2F2")
    human_fill = PatternFill("solid", fgColor="FCE4D6")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in rows:
        ws.append([safe_cell(row.get(column, "")) for column in VISIBLE_COLUMNS])
    for col_idx, header in enumerate(VISIBLE_COLUMNS, 1):
        fill = human_fill if header in HUMAN_FIELDS else auto_fill
        for row_idx in range(2, ws.max_row + 1):
            cell = ws.cell(row_idx, col_idx)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        width = {
            "claim_title": 54,
            "claim_text": 72,
            "structured_event_arguments": 50,
            "closed_track_evidence": 95,
            "financial_context": 60,
            "market_context": 60,
        }.get(header)
        if width is None:
            values = [safe_cell(row.get(header, "")) for row in rows[:150]]
            width = min(max(max([len(header)] + [min(len(value), 60) for value in values]) + 2, 12), 42)
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    validations = {
        "annotator_a_label": LABEL_VALUES,
        "annotator_b_label": LABEL_VALUES,
        "adjudicated_label": LABEL_VALUES,
        "annotator_a_confidence": CONFIDENCE_VALUES,
        "annotator_b_confidence": CONFIDENCE_VALUES,
        "disagreement": YES_NO_VALUES,
        "needs_expert_review": YES_NO_VALUES,
    }
    for col_idx, header in enumerate(VISIBLE_COLUMNS, 1):
        if header not in validations:
            continue
        dv = DataValidation(type="list", formula1=f'"{",".join(validations[header])}"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{get_column_letter(col_idx)}2:{get_column_letter(col_idx)}{max(ws.max_row, 2)}")
    wb.save(path)


def collect_identifiers_from_value(value: Any, identifiers: dict[str, set[str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"sample_id", "cluster_id", "event_id", "candidate_id", "source_event_id"} or "cluster_id" in lowered:
                if isinstance(item, str) and item:
                    identifiers["ids"].add(item)
                    for match in re.findall(r"EC-\d{6}(?:__v\d+)?|I500V2-\d{6}", item):
                        identifiers["cluster_ids"].add(match.split("__", 1)[0])
            if lowered in {"article_id", "article_file_id", "document_id", "canonical_article_id", "trace_id", "claim_article_id", "evidence_article_id"} or "article_id" in lowered or "article_file_id" in lowered:
                if isinstance(item, str) and item:
                    identifiers["article_ids"].add(item)
            if "title_sha256" in lowered or "normalized_title_sha256" in lowered:
                if isinstance(item, str) and item:
                    identifiers["title_hashes"].add(item)
            if "content_sha256" in lowered or "normalized_content_sha256" in lowered:
                if isinstance(item, str) and item:
                    identifiers["content_hashes"].add(item)
            if lowered in {"title", "claim_title", "article_title", "selected_article_title"} and isinstance(item, str):
                digest = text_sha256(item)
                if digest:
                    identifiers["title_hashes"].add(digest)
            if lowered == "title_samples" and isinstance(item, str):
                for part in re.split(r"\s*\|\|\s*|\n+", item):
                    digest = text_sha256(part)
                    if digest:
                        identifiers["title_hashes"].add(digest)
            if lowered == "content" and isinstance(item, str):
                digest = text_sha256(item)
                if digest:
                    identifiers["content_hashes"].add(digest)
            collect_identifiers_from_value(item, identifiers)
    elif isinstance(value, list):
        for item in value:
            collect_identifiers_from_value(item, identifiers)


def profile_from_row(row: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    claim = row.get("claim") if isinstance(row.get("claim"), dict) else {}
    title = norm(row.get("claim_title") or row.get("title") or row.get("article_title") or claim.get("title") or row.get("title_samples"))
    claim_text = norm(row.get("claim_text") or row.get("event_claim") or claim.get("event_claim") or row.get("claim"))
    target_text = norm(row.get("target_code") or row.get("trading_code") or row.get("target") or claim.get("target"))
    match = CODE_RE.search(target_text)
    target_key = match.group(1) if match else compact_text(norm(row.get("target_name") or row.get("target_entity") or claim.get("target")), 80)
    if not (target_key or title or claim_text):
        return None
    return {
        "source_name": source_name,
        "id": norm(row.get("sample_id") or row.get("cluster_id") or row.get("event_id") or row.get("candidate_id")),
        "target_key": target_key,
        "event_name": norm(row.get("event_name") or row.get("event_type") or row.get("schema_id") or claim.get("event_type")),
        "date": first_date(row.get("publish_time") or row.get("article_publish_time") or row.get("event_time") or claim.get("publish_time")),
        "title_text": title,
        "claim_text": claim_text,
    }


def rows_from_workbook(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    rows: list[dict[str, Any]] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        iterator = ws.iter_rows(values_only=True)
        try:
            headers = [norm(value) for value in next(iterator)]
        except StopIteration:
            continue
        if not any(headers):
            continue
        for values in iterator:
            row = {header: value for header, value in zip(headers, values) if header}
            if row:
                rows.append(row)
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def rows_from_csv(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def load_history_rows(source: HistorySource) -> list[dict[str, Any]]:
    if not source.path.exists():
        return []
    suffix = source.path.suffix.lower()
    try:
        if suffix == ".jsonl":
            return read_jsonl(source.path)
        if suffix == ".json":
            payload = read_json(source.path)
            if isinstance(payload, dict):
                return [payload]
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        if suffix == ".csv":
            return rows_from_csv(source.path)
        if suffix == ".xlsx":
            return rows_from_workbook(source.path)
    except Exception:
        return []
    return []


def collect_history() -> dict[str, Any]:
    identifiers: dict[str, set[str]] = {
        "ids": set(),
        "cluster_ids": set(),
        "article_ids": set(),
        "title_hashes": set(),
        "content_hashes": set(),
    }
    profiles: list[dict[str, Any]] = []
    source_stats = []
    for source in HISTORY_SOURCES:
        before = {key: len(values) for key, values in identifiers.items()}
        rows = load_history_rows(source)
        for row in rows:
            collect_identifiers_from_value(row, identifiers)
            profile = profile_from_row(row, source.name)
            if profile:
                profiles.append(profile)
        after = {key: len(values) for key, values in identifiers.items()}
        source_stats.append(
            {
                "name": source.name,
                "category": source.category,
                "path": rel(source.path),
                "exists": source.path.exists(),
                "trust_status": source.trust_status,
                "row_count": len(rows),
                "new_identifier_counts": {key: after[key] - before[key] for key in before},
                "notes": source.notes,
            }
        )
    profile_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        profile_buckets[profile.get("target_key", "")].append(profile)
    return {
        "identifiers": identifiers,
        "profiles": profiles,
        "profile_buckets": profile_buckets,
        "source_stats": source_stats,
        "identifier_counts": {key: len(values) for key, values in identifiers.items()},
    }


def discover_unverified_annotation_assets() -> list[dict[str, Any]]:
    roots = [ROOT / "reports", ROOT / "outputs/event_credibility", ROOT / "configs", ROOT / "data/processed"]
    patterns = re.compile(r"(gold|human|annotation|annotat|audit)", re.I)
    out = []
    seen: set[Path] = set()
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path in seen or not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".xlsx", ".md", ".yaml", ".yml"}:
                continue
            if not patterns.search(path.name):
                continue
            seen.add(path)
            trusted = path == GOLD_WORKBOOK
            out.append(
                {
                    "path": rel(path),
                    "exists": True,
                    "size_bytes": path.stat().st_size,
                    "role_in_this_round": "registered_development_gold" if trusted else "audited_unverified_not_used_as_gold",
                    "trust_status": "trusted_only_for_human_gold_development_v2" if trusted else "unverified",
                }
            )
    out.sort(key=lambda item: item["path"])
    return out


def validate_and_register_development_set(generated_at: str) -> tuple[dict[str, Any], dict[str, int]]:
    workbook_sha = sha256_file(GOLD_WORKBOOK)
    samples_sha = sha256_file(GOLD_SAMPLES)
    if workbook_sha != EXPECTED_GOLD_WORKBOOK_SHA:
        raise RuntimeError(f"Independent500-v2 adjudicated workbook SHA mismatch: {workbook_sha}")
    if samples_sha != EXPECTED_GOLD_SAMPLES_SHA:
        raise RuntimeError(f"Independent500-v2 sample input SHA mismatch: {samples_sha}")

    wb = load_workbook(GOLD_WORKBOOK, read_only=True, data_only=True)
    ws = wb["annotation"]
    headers = [norm(value) for value in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    label_index = headers.index("human_label")
    labels = Counter()
    for row in ws.iter_rows(min_row=2, values_only=True):
        label = norm(row[label_index])
        if label:
            labels[label] += 1
    label_distribution = {label: labels.get(label, 0) for label in LABEL_VALUES}
    if label_distribution != EXPECTED_LABEL_DISTRIBUTION:
        raise RuntimeError(f"Independent500-v2 label distribution mismatch: {label_distribution}")
    config = {
        "dataset_id": DEV_DATASET_ID,
        "previous_dataset_id": "task2_independent500_v2",
        "status": "human_gold_development_set",
        "effective_time": generated_at,
        "source_paths": {
            "adjudicated_workbook": rel(GOLD_WORKBOOK),
            "unlabeled_input": rel(GOLD_SAMPLES),
        },
        "source_sha256": {
            "adjudicated_workbook": workbook_sha,
            "unlabeled_input": samples_sha,
        },
        "row_count": sum(label_distribution.values()),
        "label_distribution": label_distribution,
        "human_process": {
            "annotation_mode": "double_blind",
            "adjudication": "adjudicated_v2_workbook",
            "trusted_task2_human_four_class_gold": True,
            "other_gold_human_annotation_assets_auto_trusted": False,
        },
        "historical_evaluation_usage": [
            "Independent500-v2 was previously used as a first blind human evaluation package.",
            "Round1 and DianJin post-hoc reports remain historical records and are not modified by this run.",
            "The historical 85.6% result must not be re-used as a new independent score after development tuning.",
        ],
        "development_reason": (
            "The labels are now known and have been used for retrospective analysis; the set is therefore "
            "registered as development data for six-dimensional rubric and calibration work, not as future unseen test data."
        ),
        "allowed_uses": [
            "six-dimensional scoring rubric development",
            "six-dimensional weight training",
            "tuning and calibration",
            "error analysis",
            "ablation experiments",
        ],
        "forbidden_uses": [
            "calling the set unseen independent test data again",
            "calling the set a first blind test again",
            "using it as a future model's final official test set",
            "citing the historical 85.6% result as a new independent score after tuning",
        ],
        "constraints": {
            "original_files_preserved": True,
            "historical_evaluation_outputs_preserved": True,
            "training_performed_by_this_run": False,
            "weight_fitting_performed_by_this_run": False,
            "threshold_tuning_performed_by_this_run": False,
            "new_formal_test_result_generated_by_this_run": False,
        },
    }
    write_json(DEV_CONFIG, config)
    return config, label_distribution


def load_stock_name_maps() -> tuple[dict[str, str], dict[str, str], dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    market_df = pd.read_parquet(
        MARKET_CACHE,
        columns=["trading_code", "trading_date", "stock_name", "close", "previous_close", "daily_return", "turnover_full", "valid_price"],
    )
    stock_names: dict[str, str] = {}
    market_records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in market_df.itertuples(index=False):
        code = str(row.trading_code).zfill(6)
        name = norm(row.stock_name)
        if name and code not in stock_names:
            stock_names[code] = name
        date = str(row.trading_date)[:10]
        market_records[(code, date)] = {
            "trading_code": code,
            "trading_date": date,
            "stock_name": name,
            "close": row.close,
            "previous_close": row.previous_close,
            "daily_return": row.daily_return,
            "turnover_full": row.turnover_full,
            "valid_price": bool(row.valid_price),
        }

    financial_columns = [
        "stock_code",
        "company_name",
        "report_period",
        "estimated_available_date",
        "revenue_yoy",
        "ocf_yoy",
        "roe",
        "parent_roe",
        "net_profit_cash_content",
        "record_id",
    ]
    fin_df = pd.read_parquet(FINANCIAL_CACHE, columns=financial_columns)
    financial_names: dict[str, str] = {}
    financial_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fin_df.itertuples(index=False):
        code = re.sub(r"\D", "", norm(row.stock_code)).zfill(6)[:6]
        if not code:
            continue
        company = norm(row.company_name)
        if company:
            financial_names.setdefault(code, company)
            stock_names.setdefault(code, company)
        available_dt = parse_datetime(row.estimated_available_date)
        if available_dt is None:
            continue
        financial_records[code].append(
            {
                "stock_code": code,
                "company_name": company,
                "report_period": str(row.report_period)[:10],
                "estimated_available_date": available_dt,
                "estimated_available_date_text": available_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "revenue_yoy": row.revenue_yoy,
                "ocf_yoy": row.ocf_yoy,
                "roe": row.roe,
                "parent_roe": row.parent_roe,
                "net_profit_cash_content": row.net_profit_cash_content,
                "record_id": norm(row.record_id),
            }
        )
    for code in financial_records:
        financial_records[code].sort(key=lambda item: (item["estimated_available_date"], item["report_period"], item.get("record_id", "")))
    return stock_names, financial_names, financial_records, market_records


def source_code_penalty(code: str, text: str) -> int:
    head = text[:160]
    if code == "300033" and "同花顺" in head and ("金融研究中心" in head or "财经" in head):
        return 25
    if code == "300059" and "东方财富" in head and ("财经" in head or "Choice" in head):
        return 18
    return 0


def choose_target_code(title: str, content: str, stock_names: dict[str, str]) -> tuple[str, list[str], str]:
    text = f"{title} {content[:1600]}"
    codes = list(dict.fromkeys(match.group(1) for match in CODE_RE.finditer(text)))
    if not codes:
        return "", [], "no_stock_code_detected"
    scored = []
    for pos, code in enumerate(codes):
        name = stock_names.get(code, "")
        score = max(0, 6 - pos)
        if code in title:
            score += 12
        if name and name in title:
            score += 24
        if name and re.search(re.escape(name) + r"\s*[（(]\s*" + re.escape(code), text):
            score += 28
        if name and name in content[:500]:
            score += 8
        if re.search(r"[（(]\s*" + re.escape(code) + r"(?:\.(?:SH|SZ|sh|sz))?\s*[）)]", title):
            score += 10
        score -= source_code_penalty(code, text)
        scored.append((score, code))
    scored.sort(key=lambda item: (-item[0], codes.index(item[1]), item[1]))
    best_score, best_code = scored[0]
    if best_score <= 0:
        return "", codes, "all_detected_codes_look_like_source_or_index"
    return best_code, codes, f"selected_by_code_name_score_{best_score}"


def infer_event_name(title: str, content: str) -> str:
    text = f"{title} {content[:1200]}"
    if any(term in text for term in MARKET_TERMS):
        return "行情价格变动"
    if any(term in text for term in FINANCIAL_TERMS):
        return "财务业绩口径"
    if any(term in text for term in REVERSE_TERMS):
        return "公告澄清与否认"
    if any(term in text for term in ["中标", "订单", "合同", "项目", "采购"]):
        return "项目订单进展"
    if any(term in text for term in ["并购", "重组", "融资", "定增", "收购", "股权"]):
        return "资本市场运作"
    if any(term in text for term in AMBIGUITY_TERMS):
        return "主体或口径不清"
    return "未分类财经事件"


def target_aligned_sentence(sentence: str, title: str, target_name: str, target_code: str) -> bool:
    if target_code and target_code in sentence:
        return True
    if target_name and target_name in sentence:
        return True
    if (target_code and target_code in title) or (target_name and target_name in title):
        return "公司" in sentence or any(term in sentence for term in FINANCIAL_TERMS + MARKET_TERMS + SUPPORT_TERMS + REVERSE_TERMS)
    return False


def parse_financial_sentence(sentence: str) -> tuple[str, str] | None:
    metric = ""
    if any(term in sentence for term in ["营业收入", "主营收入", "营收", "收入"]):
        metric = "revenue_yoy"
    elif any(term in sentence for term in ["经营现金流", "现金流"]):
        metric = "ocf_yoy"
    elif any(term in sentence for term in ["净资产收益率", "ROE", "roe"]):
        metric = "roe"
    if not metric:
        return None
    positive = any(term in sentence for term in POSITIVE_FINANCIAL_TERMS)
    negative = any(term in sentence for term in NEGATIVE_FINANCIAL_TERMS)
    if positive and not negative:
        return metric, "up"
    if negative and not positive:
        return metric, "down"
    return None


def parse_financial_period(sentence: str, publish_dt: datetime) -> str:
    years = [int(year) for year in re.findall(r"(20\d{2})年", sentence)]
    year = years[-1] if years else publish_dt.year
    if any(term in sentence for term in ["一季度", "第一季度", "Q1", "q1"]):
        return f"{year:04d}-03-31"
    if any(term in sentence for term in ["半年", "上半年", "二季度", "第二季度", "Q2", "q2"]):
        return f"{year:04d}-06-30"
    if any(term in sentence for term in ["前三季度", "三季度", "第三季度", "Q3", "q3"]):
        return f"{year:04d}-09-30"
    if any(term in sentence for term in ["全年", "年报", "年度", "年末"]):
        return f"{year:04d}-12-31"
    return ""


def select_financial_record(
    records: list[dict[str, Any]],
    metric: str,
    period: str,
    claim_dt: datetime,
) -> dict[str, Any] | None:
    eligible = []
    for record in records:
        if record["estimated_available_date"] > claim_dt:
            continue
        if period and record.get("report_period") != period:
            continue
        value = record.get(metric)
        try:
            value_f = float(value)
        except Exception:
            continue
        if not math.isfinite(value_f):
            continue
        eligible.append(record)
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: (item["estimated_available_date"], item["report_period"], item.get("record_id", "")))[-1]


def financial_direction_conflict(direction: str, value: float) -> bool:
    return (direction == "up" and value < -0.02) or (direction == "down" and value > 0.02)


def financial_direction_consistent(direction: str, value: float) -> bool:
    return (direction == "up" and value > 0.02) or (direction == "down" and value < -0.02)


def financial_metric_name(metric: str) -> str:
    return {
        "revenue_yoy": "营业收入同比",
        "ocf_yoy": "经营现金流同比",
        "roe": "净资产收益率",
    }.get(metric, metric)


def parse_market_claim(title: str) -> tuple[str, bool, str] | None:
    if any(term in title for term in ["主力资金", "净流入", "净流出", "融资", "融券", "转债", "股东户数"]):
        return None
    direction = ""
    limit_claim = False
    if "涨停" in title and "收盘" in title:
        direction, limit_claim = "up", True
    elif "跌停" in title and "收盘" in title:
        direction, limit_claim = "down", True
    elif any(term in title for term in ["收涨", "上涨", "大涨", "涨幅", "涨超", "涨逾"]):
        direction = "up"
    elif any(term in title for term in ["收跌", "下跌", "大跌", "跌幅", "跌超", "跌逾"]):
        direction = "down"
    if not direction:
        return None
    return direction, limit_claim, "explicit_close_or_direction_claim"


def parse_market_reference_date(title: str, publish_dt: datetime) -> str:
    title_head = title[:100]
    full = DATE_RE.search(title_head)
    if full:
        return f"{int(full.group(1)):04d}-{int(full.group(2)):02d}-{int(full.group(3)):02d}"
    partial = MONTH_DAY_RE.search(title_head)
    if partial:
        candidate = datetime(publish_dt.year, int(partial.group(1)), int(partial.group(2)))
        if candidate.date() > publish_dt.date() + timedelta(days=3):
            candidate = datetime(publish_dt.year - 1, int(partial.group(1)), int(partial.group(2)))
        return candidate.strftime("%Y-%m-%d")
    if publish_dt.time() >= time(15, 30):
        return publish_dt.strftime("%Y-%m-%d")
    return ""


def market_evidence_datetime(ref_date: str) -> datetime | None:
    if not ref_date:
        return None
    dt = parse_datetime(ref_date)
    if dt is None:
        return None
    return datetime.combine(dt.date(), time(15, 0))


def market_relation(direction: str, limit_claim: bool, daily_return: float) -> str:
    if direction == "up" and daily_return < -0.003:
        return "opposite_direction"
    if direction == "down" and daily_return > 0.003:
        return "opposite_direction"
    if limit_claim and direction == "up" and daily_return < 0.085:
        return "limit_mismatch"
    if limit_claim and direction == "down" and daily_return > -0.085:
        return "limit_mismatch"
    if direction == "up" and daily_return > 0.003:
        return "same_direction"
    if direction == "down" and daily_return < -0.003:
        return "same_direction"
    return "weak_or_flat"


def build_financial_evidence(
    article: dict[str, Any],
    record: dict[str, Any],
    metric: str,
    direction: str,
    sentence: str,
) -> tuple[dict[str, Any], str]:
    value = float(record[metric])
    metric_cn = financial_metric_name(metric)
    text = (
        f"{record.get('company_name') or article['target_name']} {record['report_period']} 报告期"
        f"{metric_cn}={value:.4f}，估计可获得日期为{record['estimated_available_date_text']}；"
        f"该方向与待核主张中的“{sentence[:120]}”相反或可直接校验。"
    )
    evidence = make_evidence(
        f"E1",
        "CSMAR_FINANCIAL_V2",
        record["estimated_available_date_text"],
        text,
        "financial_data",
        f"{rel(FINANCIAL_CACHE)}#record_id={record.get('record_id')}",
        True,
    )
    context = f"{metric_cn}={value:.4f}; report_period={record['report_period']}; available={record['estimated_available_date_text']}"
    return evidence, context


def build_market_evidence(article: dict[str, Any], market_record: dict[str, Any], relation: str) -> tuple[dict[str, Any], str]:
    daily_return = float(market_record["daily_return"])
    ref_date = market_record["trading_date"]
    evidence_time = market_evidence_datetime(ref_date).strftime("%Y-%m-%d %H:%M:%S")
    text = (
        f"{market_record.get('stock_name') or article['target_name']} {ref_date} 收盘价={float(market_record['close']):.4f}，"
        f"前收盘价={float(market_record['previous_close']):.4f}，日收益率={daily_return:.4%}；"
        f"该行情数据可校验待核主张的价格方向或涨跌停表述。"
    )
    evidence = make_evidence(
        "E1",
        "RESSET_DRESSTK_2026",
        evidence_time,
        text,
        "market_data",
        f"{rel(MARKET_CACHE)}#trading_code={article['target_code']}&trading_date={ref_date}&relation={relation}",
        True,
    )
    context = (
        f"trading_date={ref_date}; close={float(market_record['close']):.4f}; "
        f"previous_close={float(market_record['previous_close']):.4f}; daily_return={daily_return:.4%}"
    )
    return evidence, context


def build_article_evidence(source_article: dict[str, Any], evidence_index: int, relation: str) -> dict[str, Any]:
    terms = REVERSE_TERMS if relation == "reverse_prior_article" else SUPPORT_TERMS + AMBIGUITY_TERMS
    text = sentence_with_terms(
        source_article["claim_title"],
        source_article["content"],
        terms,
        source_article["target_name"],
        source_article["target_code"],
    )
    return make_evidence(
        f"E{evidence_index}",
        source_article["source"],
        source_article["publish_time"],
        f"{source_article['claim_title']}。{text}",
        source_article["source_type"],
        f"{rel(ARTICLE_INDEX)}#article_id={source_article['article_file_id']}",
        True,
    )


def build_sample(
    article: dict[str, Any],
    route: str,
    route_reason: str,
    evidence: list[dict[str, Any]],
    quality: dict[str, float],
    claim_sentence: str = "",
    financial_context: str = "",
    market_context: str = "",
    future_discovery_only: bool = False,
) -> dict[str, Any]:
    event_name = article["event_name"]
    claim_text = f"关于{article['target_name']}的{event_name}主张：{claim_sentence or article['claim_title']}"
    identity = deterministic_key(article["article_file_id"], article["target_code"], event_name, article["normalized_title_sha256"])
    return {
        "sample_id": "",
        "cluster_id": f"SIXDIM-RC1-{identity[:16]}",
        "article_file_id": article["article_file_id"],
        "target": article["target"],
        "target_code": article["target_code"],
        "target_name": article["target_name"],
        "event_name": event_name,
        "claim_title": article["claim_title"],
        "claim_text": claim_text,
        "publish_time": article["publish_time"],
        "publish_datetime": article["publish_datetime"],
        "source": article["source"],
        "source_type": article["source_type"],
        "dataset_split": article["dataset_split"],
        "structured_event_arguments": {
            "target_code": article["target_code"],
            "target_name": article["target_name"],
            "event_name": event_name,
            "claim_time": article["publish_time"],
            "claim_article_id": article["article_file_id"],
            "argument_source": "fulltext_article_index_plus_structured_market_financial_evidence",
        },
        "closed_track_evidence": evidence,
        "financial_context": financial_context,
        "market_context": market_context,
        "normalized_title_sha256": article["normalized_title_sha256"],
        "normalized_content_sha256": article["normalized_content_sha256"],
        "candidate_route": route,
        "route_reason": route_reason,
        "candidate_quality_score": round(sum(quality.values()), 6),
        "quality_components": {key: round(value, 6) for key, value in quality.items()},
        "future_discovery_only": future_discovery_only,
    }


def candidate_profile(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_key": sample.get("target_code", ""),
        "event_name": sample.get("event_name", ""),
        "date": first_date(sample.get("publish_time")),
        "title_text": sample.get("claim_title", ""),
        "claim_text": sample.get("claim_text", ""),
        "id": sample.get("sample_id") or sample.get("cluster_id"),
    }


def exact_exclusion_reason(article: dict[str, Any], history: dict[str, Any]) -> str:
    identifiers = history["identifiers"]
    if article["article_file_id"] in identifiers["article_ids"]:
        return "article_file_id_exact_overlap"
    if article["normalized_title_sha256"] in identifiers["title_hashes"]:
        return "title_sha256_exact_overlap"
    if article["normalized_content_sha256"] in identifiers["content_hashes"]:
        return "content_sha256_exact_overlap"
    return ""


def near_history_overlap(sample: dict[str, Any], history: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    profile = candidate_profile(sample)
    for historical in history["profile_buckets"].get(profile["target_key"], []):
        duplicate, method, score = is_near_duplicate(profile, historical)
        if duplicate:
            return True, {
                "candidate_cluster_id": sample["cluster_id"],
                "historical_id": historical.get("id"),
                "historical_source": historical.get("source_name"),
                "method": method,
                "similarity": round(score, 6),
            }
    return False, {}


def row_near_selected(sample: dict[str, Any], selected_buckets: dict[str, list[dict[str, Any]]]) -> tuple[bool, dict[str, Any]]:
    profile = candidate_profile(sample)
    for selected in selected_buckets.get(profile["target_key"], []):
        duplicate, method, score = is_near_duplicate(profile, selected)
        if duplicate:
            return True, {
                "candidate_cluster_id": sample["cluster_id"],
                "selected_id": selected.get("id"),
                "method": method,
                "similarity": round(score, 6),
            }
    return False, {}


def source_dominance_limit(route: str, target_count: int) -> int:
    # Keep a soft source cap high enough to avoid starving routes while still
    # preventing a single source from dominating a final package.
    if target_count <= 150:
        return max(180, int(target_count * 0.9))
    return max(240, int(target_count * 0.5))


def select_by_route(
    route_candidates: dict[str, list[dict[str, Any]]],
    route_targets: dict[str, int],
    history: dict[str, Any],
    final_mode: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_article_ids: set[str] = set()
    used_cluster_ids: set[str] = set()
    used_title_hashes: set[str] = set()
    used_content_hashes: set[str] = set()
    target_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    near_rejections: list[dict[str, Any]] = []
    route_shortfalls: dict[str, dict[str, Any]] = {}

    def can_add(sample: dict[str, Any], target_route: str, source_limit: int) -> tuple[bool, str]:
        if sample["article_file_id"] in used_article_ids:
            return False, "duplicate_article_file_id"
        if sample["cluster_id"] in used_cluster_ids:
            return False, "duplicate_cluster_id"
        if sample["normalized_title_sha256"] in used_title_hashes:
            return False, "duplicate_title_sha256"
        if sample["normalized_content_sha256"] in used_content_hashes:
            return False, "duplicate_content_sha256"
        if final_mode and target_counts[sample["target"]] >= 10:
            return False, "same_target_company_cap"
        if final_mode and source_counts[sample["source"] or "unknown"] >= source_limit:
            return False, "source_soft_cap"
        is_dup, detail = row_near_selected(sample, selected_buckets)
        if is_dup:
            near_rejections.append(detail)
            return False, "near_duplicate_selected"
        is_hist_dup, detail = near_history_overlap(sample, history)
        if is_hist_dup:
            near_rejections.append(detail)
            return False, "near_duplicate_history"
        return True, ""

    route_items = list(route_targets.items())
    if final_mode:
        route_items.sort(key=lambda item: (item[1], item[0]))

    for route, target in route_items:
        candidates = sorted(
            route_candidates.get(route, []),
            key=lambda row: (-float(row["candidate_quality_score"]), row["publish_time"], deterministic_key(row["cluster_id"], route)),
        )
        source_limit = source_dominance_limit(route, target)
        added = 0
        for sample in candidates:
            if added >= target:
                break
            ok, reason = can_add(sample, route, source_limit)
            if not ok:
                rejection_counts[f"{route}:{reason}"] += 1
                continue
            selected.append(sample)
            profile = candidate_profile(sample)
            selected_buckets[profile["target_key"]].append(profile)
            used_article_ids.add(sample["article_file_id"])
            used_cluster_ids.add(sample["cluster_id"])
            used_title_hashes.add(sample["normalized_title_sha256"])
            used_content_hashes.add(sample["normalized_content_sha256"])
            target_counts[sample["target"]] += 1
            source_counts[sample["source"] or "unknown"] += 1
            route_counts[route] += 1
            added += 1
        if added < target:
            route_shortfalls[route] = {
                "target": target,
                "actual": added,
                "reason": "insufficient_candidates_after_time_alignment_dedup_and_caps",
            }

    selected.sort(key=lambda row: deterministic_key("final_order" if final_mode else "pool_order", row["cluster_id"]))
    for idx, sample in enumerate(selected, 1):
        sample["sample_id"] = f"SIXDIM-RC1-{idx:06d}" if not final_mode else f"SIXDIM-ANN1-{idx:06d}"
        for field in HUMAN_FIELDS:
            sample[field] = ""
    summary = {
        "selected_count": len(selected),
        "route_counts": dict(route_counts),
        "route_shortfalls": route_shortfalls,
        "rejection_counts": dict(rejection_counts),
        "near_duplicate_rejection_examples": near_rejections[:20],
        "unique_article_file_ids": len(used_article_ids),
        "unique_cluster_ids": len(used_cluster_ids),
        "unique_targets": len(target_counts),
        "target_max_count": max(target_counts.values()) if target_counts else 0,
        "source_counts": dict(source_counts.most_common(30)),
    }
    return selected, summary


def load_and_normalize_articles(history: dict[str, Any], stock_names: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_parquet(
        ARTICLE_INDEX,
        columns=[
            "dataset_split",
            "article_id",
            "article_title",
            "article_source",
            "article_publish_time",
            "content",
            "content_length",
            "source_type",
            "source_reliability",
            "is_labeled_source",
        ],
    )
    articles: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    target_codes_seen: set[str] = set()
    all_article_ids: set[str] = set()
    for row in df.itertuples(index=False):
        article_id = norm(row.article_id)
        all_article_ids.add(article_id)
        title = norm(row.article_title)
        content = "" if pd.isna(row.content) else str(row.content)
        if not article_id or not title or not content:
            exclusion_counts["missing_article_id_title_or_content"] += 1
            continue
        if bool(row.is_labeled_source):
            exclusion_counts["raw_labeled_source_excluded"] += 1
            continue
        if any(term in title for term in FUND_OR_INDEX_TERMS):
            exclusion_counts["fund_or_index_title_excluded"] += 1
            continue
        publish_dt = parse_datetime(row.article_publish_time)
        if publish_dt is None:
            exclusion_counts["invalid_publish_time"] += 1
            continue
        target_code, all_codes, target_reason = choose_target_code(title, content, stock_names)
        if not target_code:
            exclusion_counts["no_company_target_code"] += 1
            continue
        target_name = stock_names.get(target_code, "")
        if not target_name:
            exclusion_counts["stock_name_missing"] += 1
            continue
        normalized_title_sha = text_sha256(title)
        normalized_content_sha = text_sha256(content)
        article = {
            "article_file_id": article_id,
            "claim_title": title,
            "content": content[:5000],
            "publish_time": publish_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "publish_datetime": publish_dt,
            "source": norm(row.article_source),
            "source_type": norm(row.source_type) or "unknown",
            "source_reliability": float(row.source_reliability) if not pd.isna(row.source_reliability) else 0.0,
            "dataset_split": norm(row.dataset_split),
            "target_code": target_code,
            "target_name": target_name,
            "target": f"{target_name} / {target_code}",
            "all_detected_codes": all_codes,
            "target_selection_reason": target_reason,
            "event_name": infer_event_name(title, content),
            "normalized_title_sha256": normalized_title_sha,
            "normalized_content_sha256": normalized_content_sha,
            "raw_text_head": f"{title} {content[:1200]}",
        }
        reason = exact_exclusion_reason(article, history)
        if reason:
            exclusion_counts[reason] += 1
            continue
        target_codes_seen.add(target_code)
        articles.append(article)
    articles.sort(key=lambda item: (item["target_code"], item["publish_time"], item["article_file_id"]))
    inventory = {
        "fulltext_article_rows": len(df),
        "fulltext_article_unique_ids": len(all_article_ids),
        "normalized_unused_company_articles": len(articles),
        "normalized_unused_company_targets": len(target_codes_seen),
        "normalization_exclusion_counts": dict(exclusion_counts),
    }
    return articles, inventory


def create_route_candidates(
    articles: list[dict[str, Any]],
    financial_records: dict[str, list[dict[str, Any]]],
    market_records: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    route_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    generation_counts: Counter[str] = Counter()
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        by_code[article["target_code"]].append(article)

    prior_articles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for code, code_articles in by_code.items():
        code_articles.sort(key=lambda item: (item["publish_datetime"], item["article_file_id"]))
        prior_reverse: list[dict[str, Any]] = []
        prior_support: list[dict[str, Any]] = []
        for article in code_articles:
            title = article["claim_title"]
            content = article["content"]
            text = article["raw_text_head"]
            claim_dt = article["publish_datetime"]

            made_route = False
            for sentence in split_sentences(title, content):
                parsed_fin = parse_financial_sentence(sentence)
                if not parsed_fin:
                    continue
                if not target_aligned_sentence(sentence, title, article["target_name"], article["target_code"]):
                    continue
                metric, direction = parsed_fin
                period = parse_financial_period(sentence, claim_dt)
                if not period:
                    generation_counts["financial_sentence_without_period"] += 1
                    continue
                record = select_financial_record(financial_records.get(code, []), metric, period, claim_dt)
                if not record:
                    generation_counts["financial_record_missing_or_future"] += 1
                    continue
                value = float(record[metric])
                evidence, financial_context = build_financial_evidence(article, record, metric, direction, sentence)
                if financial_direction_conflict(direction, value):
                    quality = {
                        "target_alignment": 1.0,
                        "event_alignment": 0.95,
                        "time_legality": 1.0,
                        "evidence_independence": 1.0,
                        "evidence_authority": 1.0,
                        "direct_contradiction_strength": min(1.0, abs(value) * 2.0 + 0.55),
                        "ambiguity_strength": 0.05,
                        "claim_completeness": 0.85 if len(sentence) >= 20 else 0.65,
                        "duplicate_risk": 0.0,
                    }
                    route_candidates["contradict-prone"].append(
                        build_sample(article, "contradict-prone", "period_aligned_financial_direction_conflict", [evidence], quality, sentence, financial_context, "")
                    )
                    made_route = True
                    generation_counts["financial_conflict_candidates"] += 1
                    break
                if financial_direction_consistent(direction, value):
                    quality = {
                        "target_alignment": 1.0,
                        "event_alignment": 0.85,
                        "time_legality": 1.0,
                        "evidence_independence": 1.0,
                        "evidence_authority": 1.0,
                        "direct_contradiction_strength": 0.0,
                        "ambiguity_strength": 0.0,
                        "claim_completeness": 0.75,
                        "duplicate_risk": 0.0,
                    }
                    route_candidates["support/insufficient control"].append(
                        build_sample(article, "support/insufficient control", "period_aligned_financial_direction_consistent", [evidence], quality, sentence, financial_context, "")
                    )
                    made_route = True
                    generation_counts["financial_control_candidates"] += 1
                    break

            if not made_route:
                parsed_market = parse_market_claim(title)
                if parsed_market:
                    direction, limit_claim, claim_type = parsed_market
                    ref_date = parse_market_reference_date(title, claim_dt)
                    mkt = market_records.get((code, ref_date))
                    evidence_dt = market_evidence_datetime(ref_date)
                    if mkt and evidence_dt and evidence_dt <= claim_dt and mkt.get("daily_return") is not None and pd.notna(mkt.get("daily_return")):
                        relation = market_relation(direction, limit_claim, float(mkt["daily_return"]))
                        evidence, market_context = build_market_evidence(article, mkt, relation)
                        if relation in {"opposite_direction", "limit_mismatch"}:
                            quality = {
                                "target_alignment": 1.0,
                                "event_alignment": 0.95,
                                "time_legality": 1.0,
                                "evidence_independence": 1.0,
                                "evidence_authority": 0.95,
                                "direct_contradiction_strength": 0.9 if relation == "opposite_direction" else 0.78,
                                "ambiguity_strength": 0.05,
                                "claim_completeness": 0.82,
                                "duplicate_risk": 0.0,
                            }
                            route_candidates["contradict-prone"].append(
                                build_sample(article, "contradict-prone", f"market_{relation}", [evidence], quality, title, "", market_context)
                            )
                            made_route = True
                            generation_counts["market_conflict_candidates"] += 1
                        elif relation == "same_direction":
                            quality = {
                                "target_alignment": 1.0,
                                "event_alignment": 0.85,
                                "time_legality": 1.0,
                                "evidence_independence": 1.0,
                                "evidence_authority": 0.95,
                                "direct_contradiction_strength": 0.0,
                                "ambiguity_strength": 0.0,
                                "claim_completeness": 0.72,
                                "duplicate_risk": 0.0,
                            }
                            route_candidates["support/insufficient control"].append(
                                build_sample(article, "support/insufficient control", "market_same_direction_control", [evidence], quality, title, "", market_context)
                            )
                            made_route = True
                            generation_counts["market_control_candidates"] += 1

            reverse_sources = {item["source"] for item in prior_reverse if item["source"] and item["source"] != article["source"]}
            reverse_official = any(source_is_official(item["source"], item["source_type"]) for item in prior_reverse)
            if not made_route and prior_reverse and (reverse_official or len(reverse_sources) >= 2) and any(term in text for term in SUPPORT_TERMS):
                selected_reverse = []
                seen_sources = set()
                for prior in sorted(
                    prior_reverse,
                    key=lambda item: (
                        0 if source_is_official(item["source"], item["source_type"]) else 1,
                        0 if item["source"] not in seen_sources else 1,
                        item["publish_time"],
                    ),
                ):
                    if prior["article_file_id"] == article["article_file_id"]:
                        continue
                    if prior["source"] in seen_sources and len(selected_reverse) >= 1:
                        continue
                    selected_reverse.append(prior)
                    seen_sources.add(prior["source"])
                    if len(selected_reverse) >= 2:
                        break
                if selected_reverse:
                    evidence = [build_article_evidence(prior, idx + 1, "reverse_prior_article") for idx, prior in enumerate(selected_reverse)]
                    quality = {
                        "target_alignment": 0.9,
                        "event_alignment": 0.72,
                        "time_legality": 1.0,
                        "evidence_independence": 0.85 if len(seen_sources) >= 2 else 0.65,
                        "evidence_authority": 0.95 if reverse_official else 0.72,
                        "direct_contradiction_strength": 0.72 if reverse_official else 0.6,
                        "ambiguity_strength": 0.12,
                        "claim_completeness": 0.7,
                        "duplicate_risk": 0.0,
                    }
                    route_candidates["contradict-prone"].append(
                        build_sample(article, "contradict-prone", "prior_reverse_authority_or_two_independent_sources", evidence, quality, title, "", "")
                    )
                    made_route = True
                    generation_counts["prior_reverse_candidates"] += 1

            prior_pool = prior_support[-6:] + prior_reverse[-6:]
            if not made_route and (any(term in text for term in AMBIGUITY_TERMS) or len(article["all_detected_codes"]) >= 2):
                evidence = []
                seen_sources: set[str] = set()
                for prior in sorted(prior_pool, key=lambda item: (item["publish_time"], item["article_file_id"]), reverse=True):
                    if prior["source"] in seen_sources and len(evidence) >= 1:
                        continue
                    evidence.append(build_article_evidence(prior, len(evidence) + 1, "ambiguous_prior_article"))
                    seen_sources.add(prior["source"])
                    if len(evidence) >= 2:
                        break
                if not evidence:
                    fin = financial_records.get(code, [])
                    record = next((item for item in reversed(fin) if item["estimated_available_date"] <= claim_dt), None)
                    if record:
                        text_ev = (
                            f"{article['target_name']}最近可用财务记录为{record['report_period']}，"
                            f"可获得时间{record['estimated_available_date_text']}；该记录仅提供背景，不直接预设标签。"
                        )
                        evidence.append(make_evidence("E1", "CSMAR_FINANCIAL_V2", record["estimated_available_date_text"], text_ev, "financial_data", rel(FINANCIAL_CACHE), True))
                if evidence:
                    ambiguity_strength = 0.65 + min(0.25, 0.08 * len(article["all_detected_codes"]))
                    quality = {
                        "target_alignment": 0.82 if len(article["all_detected_codes"]) >= 2 else 0.9,
                        "event_alignment": 0.65,
                        "time_legality": 1.0,
                        "evidence_independence": 0.75 if len(seen_sources) >= 2 else 0.55,
                        "evidence_authority": 0.65,
                        "direct_contradiction_strength": 0.15,
                        "ambiguity_strength": ambiguity_strength,
                        "claim_completeness": 0.68,
                        "duplicate_risk": 0.0,
                    }
                    route_candidates["unclear-prone"].append(
                        build_sample(article, "unclear-prone", "subject_time_or_metric_ambiguity_with_closed_track_context", evidence, quality, title, "", "")
                    )
                    generation_counts["unclear_candidates"] += 1

            if not made_route and prior_support and any(term in text for term in SUPPORT_TERMS):
                selected_support = []
                seen_sources = set()
                for prior in reversed(prior_support[-12:]):
                    if prior["source"] in seen_sources and len(selected_support) >= 1:
                        continue
                    selected_support.append(prior)
                    seen_sources.add(prior["source"])
                    if len(selected_support) >= 2:
                        break
                if selected_support:
                    evidence = [build_article_evidence(prior, idx + 1, "support_prior_article") for idx, prior in enumerate(selected_support)]
                    quality = {
                        "target_alignment": 0.9,
                        "event_alignment": 0.72,
                        "time_legality": 1.0,
                        "evidence_independence": 0.75 if len(seen_sources) >= 2 else 0.55,
                        "evidence_authority": 0.65,
                        "direct_contradiction_strength": 0.0,
                        "ambiguity_strength": 0.05,
                        "claim_completeness": 0.65,
                        "duplicate_risk": 0.0,
                    }
                    route_candidates["support/insufficient control"].append(
                        build_sample(article, "support/insufficient control", "same_target_prior_support_control", evidence, quality, title, "", "")
                    )
                    generation_counts["article_control_candidates"] += 1

            if any(term in text for term in REVERSE_TERMS):
                prior_reverse.append(article)
            if any(term in text for term in SUPPORT_TERMS):
                prior_support.append(article)
            prior_articles[code].append(article)

    # Keep the highest-quality route when the same article slipped into multiple buckets.
    seen_articles: set[str] = set()
    deduped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in ("contradict-prone", "unclear-prone", "support/insufficient control"):
        for sample in sorted(route_candidates.get(route, []), key=lambda row: (-float(row["candidate_quality_score"]), row["cluster_id"])):
            if sample["article_file_id"] in seen_articles:
                generation_counts["same_article_route_duplicate_removed"] += 1
                continue
            seen_articles.add(sample["article_file_id"])
            deduped[route].append(sample)
    summary = {
        "raw_route_candidate_counts": {route: len(values) for route, values in route_candidates.items()},
        "deduped_route_candidate_counts": {route: len(values) for route, values in deduped.items()},
        "generation_counts": dict(generation_counts),
    }
    return deduped, summary


def route_private_rows(samples: list[dict[str, Any]], package_name: str) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        rows.append(
            {
                "package": package_name,
                "sample_id": sample["sample_id"],
                "cluster_id": sample["cluster_id"],
                "article_file_id": sample["article_file_id"],
                "target": sample["target"],
                "event_name": sample["event_name"],
                "candidate_route": sample["candidate_route"],
                "route_reason": sample["route_reason"],
                "candidate_quality_score": sample["candidate_quality_score"],
                "quality_components": sample["quality_components"],
                "future_discovery_only": sample["future_discovery_only"],
                "dianjin_usage": "not_used; if used in future it is sampling_only and cannot become gold",
            }
        )
    return rows


def strip_private_fields(sample: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "sample_id",
        "cluster_id",
        "article_file_id",
        "target",
        "target_code",
        "target_name",
        "event_name",
        "claim_title",
        "claim_text",
        "publish_time",
        "source",
        "source_type",
        "dataset_split",
        "structured_event_arguments",
        "closed_track_evidence",
        "financial_context",
        "market_context",
        "normalized_title_sha256",
        "normalized_content_sha256",
        *HUMAN_FIELDS,
    ]
    out = {key: sample.get(key, "") for key in allowed}
    for field in HUMAN_FIELDS:
        out[field] = ""
    return out


def distribution_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    return {
        "total": total,
        "source_type_counts": dict(Counter(sample.get("source_type") or "unknown" for sample in samples).most_common()),
        "source_counts_top20": dict(Counter(sample.get("source") or "unknown" for sample in samples).most_common(20)),
        "event_name_counts": dict(Counter(sample.get("event_name") or "unknown" for sample in samples).most_common()),
        "target_counts_top20": dict(Counter(sample.get("target") or "unknown" for sample in samples).most_common(20)),
        "publish_month_counts": dict(Counter((first_date(sample.get("publish_time")) or "unknown")[:7] for sample in samples).most_common()),
    }


def build_blindness_audit(samples: list[dict[str, Any]], workbook_path: Path) -> dict[str, Any]:
    leaks = []
    human_nonempty = []
    for sample in samples:
        row = visible_row(sample)
        for key, value in row.items():
            if key in HUMAN_FIELDS:
                if norm(value):
                    human_nonempty.append({"sample_id": sample["sample_id"], "field": key})
                continue
            lower = safe_cell(value).lower()
            for pattern in FORBIDDEN_VISIBLE_PATTERNS:
                if pattern.lower() in lower:
                    leaks.append({"sample_id": sample["sample_id"], "field": key, "pattern": pattern})
                    break
    workbook_human_nonempty = []
    workbook_leaks = []
    if workbook_path.exists():
        wb = load_workbook(workbook_path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            headers = [norm(cell.value) for cell in next(ws.iter_rows(min_row=1, max_row=1))]
            for row in ws.iter_rows(min_row=2, values_only=True):
                for header, value in zip(headers, row):
                    text = norm(value)
                    if header in HUMAN_FIELDS:
                        if text:
                            workbook_human_nonempty.append({"sheet": ws.title, "field": header, "value": text[:80]})
                        continue
                    lower = text.lower()
                    for pattern in FORBIDDEN_VISIBLE_PATTERNS:
                        if pattern.lower() in lower:
                            workbook_leaks.append({"sheet": ws.title, "field": header, "pattern": pattern, "value": text[:80]})
                            break
    return {
        "workbook": rel(workbook_path),
        "visible_columns": VISIBLE_COLUMNS,
        "forbidden_visible_patterns": FORBIDDEN_VISIBLE_PATTERNS,
        "route_leakage_count": sum(1 for leak in leaks + workbook_leaks if "route" in leak["pattern"] or "prone" in leak["pattern"]),
        "candidate_quality_score_leakage_count": sum(1 for leak in leaks + workbook_leaks if leak["pattern"] == "candidate_quality_score"),
        "model_prediction_leakage_count": sum(1 for leak in leaks + workbook_leaks if "prediction" in leak["pattern"] or "dianjin" in leak["pattern"] or "round1" in leak["pattern"]),
        "visible_answer_leak_count": len(leaks) + len(workbook_leaks),
        "visible_answer_leak_examples": (leaks + workbook_leaks)[:20],
        "human_fields_blank": not human_nonempty and not workbook_human_nonempty,
        "human_nonempty_count": len(human_nonempty) + len(workbook_human_nonempty),
        "human_nonempty_examples": (human_nonempty + workbook_human_nonempty)[:20],
    }


def evidence_time_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    future = []
    invalid_ids = []
    missing = []
    for sample in samples:
        claim_dt = parse_datetime(sample.get("publish_time"))
        evidence_items = sample.get("closed_track_evidence") or []
        if not evidence_items:
            missing.append(sample.get("sample_id"))
            continue
        for item in evidence_items:
            eid = norm(item.get("evidence_id"))
            if not re.fullmatch(r"E\d+", eid):
                invalid_ids.append({"sample_id": sample.get("sample_id"), "evidence_id": eid})
            evidence_dt = parse_datetime(item.get("evidence_time"))
            if claim_dt is None or evidence_dt is None or evidence_dt > claim_dt:
                future.append({"sample_id": sample.get("sample_id"), "evidence_id": eid, "evidence_time": item.get("evidence_time"), "publish_time": sample.get("publish_time")})
    return {
        "future_evidence_visible_count": len(future),
        "future_evidence_visible_examples": future[:20],
        "invalid_evidence_id_count": len(invalid_ids),
        "invalid_evidence_id_examples": invalid_ids[:20],
        "missing_evidence_count": len(missing),
        "missing_evidence_examples": missing[:20],
    }


def build_leakage_audit(samples: list[dict[str, Any]], history: dict[str, Any], selection_summary: dict[str, Any]) -> dict[str, Any]:
    identifiers = history["identifiers"]
    exact_counts = Counter()
    near = []
    selected_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    intra_near = []
    for sample in samples:
        if sample["sample_id"] in identifiers["ids"]:
            exact_counts["sample_id_exact_overlap"] += 1
        if sample["cluster_id"] in identifiers["cluster_ids"] or sample["cluster_id"] in identifiers["ids"]:
            exact_counts["cluster_id_exact_overlap"] += 1
        if sample["article_file_id"] in identifiers["article_ids"]:
            exact_counts["article_file_id_exact_overlap"] += 1
        if sample["normalized_title_sha256"] in identifiers["title_hashes"]:
            exact_counts["title_sha256_exact_overlap"] += 1
        if sample["normalized_content_sha256"] in identifiers["content_hashes"]:
            exact_counts["content_sha256_exact_overlap"] += 1
        hist_dup, detail = near_history_overlap(sample, history)
        if hist_dup:
            near.append(detail)
        selected_dup, detail = row_near_selected(sample, selected_buckets)
        if selected_dup:
            intra_near.append(detail)
        profile = candidate_profile(sample)
        selected_buckets[profile["target_key"]].append(profile)
    for key in [
        "sample_id_exact_overlap",
        "cluster_id_exact_overlap",
        "article_file_id_exact_overlap",
        "title_sha256_exact_overlap",
        "content_sha256_exact_overlap",
    ]:
        exact_counts.setdefault(key, 0)
    return {
        "dataset_id": DATASET_ID,
        "sample_count": len(samples),
        "unique_sample_id_count": len({sample["sample_id"] for sample in samples}),
        "unique_cluster_id_count": len({sample["cluster_id"] for sample in samples}),
        "unique_article_file_id_count": len({sample["article_file_id"] for sample in samples}),
        "exact_overlap_counts": dict(exact_counts),
        "all_required_overlap_counts_zero": all(value == 0 for value in exact_counts.values()),
        "near_duplicate_overlap_count": len(near),
        "near_duplicate_overlap_examples": near[:20],
        "intra_package_near_duplicate_count": len(intra_near),
        "intra_package_near_duplicate_examples": intra_near[:20],
        "cluster_overlap_count": len(samples) - len({sample["cluster_id"] for sample in samples}),
        "selection_rejection_counts": selection_summary.get("rejection_counts", {}),
        "current500_excluded": True,
        "independent500_v1_v2_excluded": True,
        "historical_task2_human_audit_train_dev_test_assets_excluded": True,
        "task1_task3_same_source_assets_excluded": True,
        "model_predictions_read_for_sampling": False,
        "gold_labels_used_for_sampling": False,
        "training_performed": False,
        "weight_fitting_performed": False,
        "threshold_tuning_performed": False,
        "new_formal_test_result_generated": False,
    }


def build_data_quality_audit(samples: list[dict[str, Any]], private_rows: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = Counter(row["candidate_route"] for row in private_rows if row.get("package") == "final_annotation_1000")
    target_counts = Counter(sample["target"] for sample in samples)
    event_counts = Counter(sample["event_name"] for sample in samples)
    source_counts = Counter(sample["source"] or "unknown" for sample in samples)
    time_audit = evidence_time_audit(samples)
    missing_required = []
    required = [
        "sample_id",
        "cluster_id",
        "article_file_id",
        "target",
        "event_name",
        "claim_title",
        "claim_text",
        "publish_time",
        "structured_event_arguments",
        "closed_track_evidence",
        "financial_context",
        "market_context",
    ]
    for sample in samples:
        for field in required:
            if field == "closed_track_evidence":
                if not sample.get(field):
                    missing_required.append({"sample_id": sample["sample_id"], "field": field})
            elif field in {"financial_context", "market_context"}:
                continue
            elif not norm(sample.get(field)):
                missing_required.append({"sample_id": sample["sample_id"], "field": field})
    return {
        "sample_count": len(samples),
        "route_counts": dict(route_counts),
        "target_company_max_count": max(target_counts.values()) if target_counts else 0,
        "target_company_cap": 10,
        "target_company_cap_passed": (max(target_counts.values()) if target_counts else 0) <= 10,
        "unique_targets": len(target_counts),
        "unique_sources": len(source_counts),
        "unique_event_names": len(event_counts),
        "source_counts_top20": dict(source_counts.most_common(20)),
        "event_name_counts": dict(event_counts.most_common()),
        "target_counts_top20": dict(target_counts.most_common(20)),
        "publish_month_counts": dict(Counter((first_date(sample.get("publish_time")) or "unknown")[:7] for sample in samples).most_common()),
        "missing_required_field_count": len(missing_required),
        "missing_required_field_examples": missing_required[:20],
        "target_alignment_test_passed": all(sample.get("target_code") and sample.get("target_name") and sample.get("target_name") in sample.get("target", "") for sample in samples),
        "event_alignment_test_passed": all(norm(sample.get("event_name")) for sample in samples),
        **time_audit,
    }


def render_protocol_markdown(dev_config: dict[str, Any], source_inventory: dict[str, Any]) -> str:
    lines = [
        "# Task2 Six-Dimensional Data Protocol v1",
        "",
        "## Effective Scope",
        "",
        f"- Effective time: {dev_config['effective_time']}",
        f"- Registered development set: `{DEV_DATASET_ID}`",
        f"- Source workbook: `{dev_config['source_paths']['adjudicated_workbook']}`",
        f"- Source unlabeled input: `{dev_config['source_paths']['unlabeled_input']}`",
        f"- Workbook SHA256: `{dev_config['source_sha256']['adjudicated_workbook']}`",
        f"- Unlabeled input SHA256: `{dev_config['source_sha256']['unlabeled_input']}`",
        f"- Row count: {dev_config['row_count']}",
        f"- Label distribution: {json.dumps(dev_config['label_distribution'], ensure_ascii=False, sort_keys=True)}",
        "",
        "## Human Process",
        "",
        "- Independent500-v2 is the only verified Task2 human four-class credibility gold set currently available.",
        "- It was produced through double-blind annotation and adjudication in the v2 workbook.",
        "- Other files containing gold, human, annotation, or audit in their names are inventoried as unverified unless explicitly confirmed.",
        "",
        "## Historical Result Preservation",
        "",
        "- Existing Independent500-v2 Round1 and DianJin post-hoc reports are preserved as historical results.",
        "- This run does not overwrite the original workbook, original unlabeled input, or historical evaluation outputs.",
        "- The historical 85.6% result must not be cited as a new independent score for models tuned on this development set.",
        "",
        "## Allowed Uses",
        "",
        *[f"- {item}" for item in dev_config["allowed_uses"]],
        "",
        "## Forbidden Uses",
        "",
        *[f"- {item}" for item in dev_config["forbidden_uses"]],
        "",
        "## Label Contract",
        "",
        "- support: before the claim publication time, independent valid evidence aligns on subject, event, and time and directly supports the core claim.",
        "- contradict: before the claim publication time, authoritative or multiple independent evidence directly negates the core claim. Lack of supporting evidence is not contradiction.",
        "- insufficient: evidence available up to the claim publication time is not enough to support or refute the claim.",
        "- unclear: subject, event, metric, time, or evidence direction is conflicted or ambiguous enough that the claim cannot stably be assigned to the other classes.",
        "",
        "## Closed-Track Rules",
        "",
        "- Evidence used for labeling must satisfy `evidence_time <= claim_publish_time`.",
        "- Future clarifications may be used only for internal discovery and must not be shown as closed-track evidence.",
        "- Same-source reposts do not count as multiple independent support.",
        "- Similar events at different companies do not count as evidence for the target company.",
        "",
        "## Candidate Source Audit Summary",
        "",
        f"- Candidate source families inventoried: {len(source_inventory['candidate_sources'])}",
        f"- Unverified gold/human/annotation assets discovered: {len(source_inventory['unverified_gold_human_annotation_assets'])}",
        f"- FTS5 index status: {source_inventory['fts5_index_status']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_annotation_guide() -> str:
    examples = [
        ("support", "A公司公告已于5月1日签订合同；主张5月2日称A公司已签订该合同，证据时间早于主张。"),
        ("support", "交易所公告显示B公司债券兑付完成；随后新闻称B公司完成兑付，主体和时间一致。"),
        ("support", "监管许可文件先于新闻发布，文件明确批准C公司的具体产品。"),
        ("contradict", "新闻称D公司2026年一季度营收增长；此前可见财报显示同一报告期营收同比下降。"),
        ("contradict", "新闻称E公司参与某项目；此前公司公告明确否认参与同一项目。"),
        ("contradict", "新闻称F公司当日收涨；闭市后行情数据表明同一交易日收跌。"),
        ("insufficient", "新闻称G公司获得大额订单，但可见材料只有同一媒体转载，没有官方或独立证据。"),
        ("insufficient", "新闻称H公司利润改善，但证据没有对应报告期数字，也没有可核验来源。"),
        ("insufficient", "新闻称I公司将来可能推出产品，主张发布时间前只有市场传闻。"),
        ("unclear", "主张中的简称可对应两家公司，证据未能确定目标主体。"),
        ("unclear", "一个来源称项目已落地，另一个独立来源称仍为意向，且都在主张前可见。"),
        ("unclear", "主张使用集团口径，证据使用上市公司口径，核心数字方向相反但口径无法统一。"),
    ]
    lines = [
        "# Task2 Six-Dimensional Rare-Class Annotation Guide",
        "",
        "本指南用于 `task2_sixdim_rareclass_annotation_1000.xlsx` 双盲人工标注。只填写人工字段，不根据样本顺序或证据措辞猜测抽样路线。",
        "",
        "## 四分类定义",
        "",
        "- support：在主张发布时间之前，存在主体、事件和时间对齐的独立有效证据，能够直接支持核心主张。",
        "- contradict：在主张发布时间之前，已经存在主体、事件和时间对齐的权威或多源独立证据，能够直接否定核心主张。不能因为找不到支持证据就判为 contradict。",
        "- insufficient：截至主张发布时间，合法证据不足以支持或反驳核心主张，包括只有同源转载、证据数量不足、缺少关键官方信息或核心数字无法核实。",
        "- unclear：存在主体、事件、口径、时间或证据方向冲突，当前材料无法稳定归入 support、contradict 或 insufficient。",
        "",
        "## Closed-Track 时间要求",
        "",
        "- 只能使用证据时间不晚于 `publish_time` 的证据。",
        "- 后续澄清、后续财报、后续行情不能作为当前标签依据。",
        "- 如果证据文本提到未来状态，只判断其在主张发布时是否已经公开且可核验。",
        "",
        "## 边界规则",
        "",
        "- support 与 insufficient：证据必须直接覆盖主体、事件、时间和核心数字；只相关但不直接覆盖时标 insufficient。",
        "- contradict 与无证据：没有找到支持证据不是反驳；只有直接反向官方证据或两条独立反向证据才考虑 contradict。",
        "- unclear 与 insufficient：材料太少是 insufficient；材料存在合理冲突或多种解释且无法消解时是 unclear。",
        "- 主体错配：母公司、子公司、集团、品牌、供应商和客户不能自动互换；不能确定映射时优先 unclear 或 needs_expert_review。",
        "- 同源转载：同一原始来源的转载只能算一个来源，不能构成多源支持或多源反驳。",
        "- 财务数字冲突：必须核对报告期、合并口径、同比/环比、币种和单位；期别不一致不能直接判 contradict。",
        "- 行情时间：日涨跌、收盘、涨跌停主张必须匹配同一合法交易日；盘中触及与收盘涨跌停不是同一主张。",
        "- 官方澄清：主张发布前已经公开的公告、监管问询回复、互动易回复可作为强证据；发布后的澄清不能进入闭轨判断。",
        "",
        "## Selected Evidence",
        "",
        "- `closed_track_evidence` 中每行以证据编号开头，例如 E1、E2。",
        "- 在 `annotator_*_evidence_ids` 中填写实际使用的证据编号，多个编号用分号分隔。",
        "- 不使用的证据不需要填写；若所有证据都不足，理由中说明缺口。",
        "",
        "## Expert Review",
        "",
        "- 出现监管/会计/法律口径争议时标记 `needs_expert_review=yes`。",
        "- 主体映射存在两个以上合理解释时标记专家复核。",
        "- 关键金额、报告期、交易日无法稳定对齐时标记专家复核。",
        "",
        "## Synthetic Examples",
        "",
        "以下示例均为合成示例，不来自正式候选样本。",
    ]
    for idx, (label, text) in enumerate(examples, 1):
        lines.append(f"{idx}. {label}: {text}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_source_inventory(
    generated_at: str,
    history: dict[str, Any],
    article_inventory: dict[str, Any],
    route_generation: dict[str, Any],
    dev_config: dict[str, Any],
) -> dict[str, Any]:
    candidate_sources = [
        {
            "name": "fulltext_article_index",
            "path": rel(ARTICLE_INDEX),
            "exists": ARTICLE_INDEX.exists(),
            "source_type": "raw_news_fulltext",
            "used_for_candidate_generation": True,
            "row_count": article_inventory.get("fulltext_article_rows"),
            "eligible_after_exclusions": article_inventory.get("normalized_unused_company_articles"),
        },
        {
            "name": "task1_structured_event_outputs",
            "path": rel(ALL_EVENTS),
            "exists": ALL_EVENTS.exists(),
            "source_type": "Task1_structured_events",
            "used_for_candidate_generation": False,
            "used_for_exclusion": True,
            "reason": "historical structured event output is audited and excluded to avoid Task1/Task3 same-source repetition",
        },
        {
            "name": "task2_fulltext_evidence",
            "path": rel(FULLTEXT_EVIDENCE),
            "exists": FULLTEXT_EVIDENCE.exists(),
            "source_type": "Task2_evidence_library",
            "used_for_candidate_generation": False,
            "used_for_source_inventory": True,
        },
        {
            "name": "RESSet_market_daily_cache",
            "path": rel(MARKET_CACHE),
            "exists": MARKET_CACHE.exists(),
            "source_type": "market_data",
            "used_for_candidate_generation": True,
            "closed_track_filter": "evidence_time <= publish_time",
        },
        {
            "name": "CSMAR_financial_records",
            "path": rel(FINANCIAL_CACHE),
            "exists": FINANCIAL_CACHE.exists(),
            "source_type": "financial_data",
            "used_for_candidate_generation": True,
            "closed_track_filter": "estimated_available_date <= publish_time",
        },
        {
            "name": "historical_unlabeled_candidates",
            "path": "data/processed/inference_candidates/*.jsonl",
            "exists": (ROOT / "data/processed/inference_candidates").exists(),
            "source_type": "historical_unlabeled_task_candidates",
            "used_for_candidate_generation": False,
            "used_for_source_inventory": True,
            "reason": "audited as historical candidate assets; not trusted as gold",
        },
        {
            "name": "task3_event_clusters_and_links",
            "path": "outputs/task3",
            "exists": (ROOT / "outputs/task3").exists(),
            "source_type": "Task3_event_clusters",
            "used_for_candidate_generation": False,
            "used_for_exclusion": True,
        },
    ]
    return {
        "dataset_id": DATASET_ID,
        "generated_at": generated_at,
        "development_gold": {
            "dataset_id": dev_config["dataset_id"],
            "path": dev_config["source_paths"],
            "sha256": dev_config["source_sha256"],
            "label_distribution": dev_config["label_distribution"],
            "status": "registered_for_development_not_future_independent_test",
        },
        "candidate_sources": candidate_sources,
        "history_sources": history["source_stats"],
        "history_identifier_counts": history["identifier_counts"],
        "article_inventory": article_inventory,
        "route_generation": route_generation,
        "unverified_gold_human_annotation_assets": discover_unverified_annotation_assets(),
        "fts5_index_status": "not_found_no_sqlite_or_fts5_file_discovered",
        "unverified_policy": "Files named gold, human, annotation, or audit are inventoried as unverified unless explicitly registered; none are used as six-dimensional gold in this run.",
    }


def build_protocol_config(generated_at: str, dev_config: dict[str, Any], source_inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": "task2_sixdim_data_protocol_v1",
        "effective_time": generated_at,
        "development_set": dev_config,
        "label_contract": {
            "labels": LABEL_VALUES,
            "support": "Closed-track independent evidence aligns on subject, event, and time and directly supports the core claim.",
            "contradict": "Closed-track authoritative or multiple independent evidence directly negates the core claim; lack of support is not contradiction.",
            "insufficient": "Evidence available at publication time is not enough to support or refute the core claim.",
            "unclear": "Subject, event, metric, time, or evidence direction is conflicted or ambiguous.",
        },
        "allowed_uses": dev_config["allowed_uses"],
        "forbidden_uses": dev_config["forbidden_uses"],
        "candidate_expansion_policy": {
            "candidate_pool_route_targets": POOL_ROUTE_TARGETS,
            "final_annotation_route_targets": FINAL_ROUTE_TARGETS,
            "route_fields_visible_to_annotators": False,
            "candidate_quality_score_visible_to_annotators": False,
            "dianjin_predictions_used": False,
            "dianjin_allowed_future_role": "sampling_only_not_gold_not_visible",
            "closed_track_only": True,
            "same_target_company_max_final": 10,
            "same_event_cluster_max": 1,
        },
        "source_inventory_path": rel(OUT_DIR / "source_inventory.json"),
        "source_inventory_summary": {
            "candidate_source_families": len(source_inventory["candidate_sources"]),
            "unverified_annotation_assets": len(source_inventory["unverified_gold_human_annotation_assets"]),
        },
        "run_constraints": {
            "training_performed": False,
            "weight_fitting_performed": False,
            "threshold_tuning_performed": False,
            "new_formal_test_result_generated": False,
        },
    }


def sha_manifest(paths: list[Path], out_path: Path) -> dict[str, str]:
    hashes = {rel(path): sha256_file(path) for path in paths if path.exists() and path != out_path}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(f"{digest}  {path}" for path, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    return hashes


def check_workbook_reloads(path: Path, expected_rows: int) -> bool:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["annotation"]
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    row_count = sum(1 for _ in ws.iter_rows(min_row=2, values_only=True))
    return headers == VISIBLE_COLUMNS and row_count == expected_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = now_iso()
    log_lines = [f"{generated_at} start {DATASET_ID} mode={args.mode}", f"pid={os.getpid()}"]
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"task2_sixdim_rareclass_expansion_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    (log_dir / "task2_sixdim_rareclass_expansion_v1.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")

    if args.mode == "smoke":
        output_dir = OUT_DIR / "smoke100"
        workbook_1000 = ROOT / "reports/task2_sixdim_rareclass_annotation_smoke100.xlsx"
        workbook_400 = ROOT / "reports/task2_sixdim_rareclass_annotation_smoke100_first40.xlsx"
        pool_targets = SMOKE_POOL_ROUTE_TARGETS
        final_targets = SMOKE_FINAL_ROUTE_TARGETS
    else:
        output_dir = OUT_DIR
        workbook_1000 = REPORT_1000
        workbook_400 = REPORT_400
        pool_targets = POOL_ROUTE_TARGETS
        final_targets = FINAL_ROUTE_TARGETS
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_config, label_distribution = validate_and_register_development_set(generated_at)
    history = collect_history()
    stock_names, _financial_names, financial_records, market_records = load_stock_name_maps()
    articles, article_inventory = load_and_normalize_articles(history, stock_names)
    log_lines.append(f"normalized_articles={len(articles)}")
    route_candidates, route_generation = create_route_candidates(articles, financial_records, market_records)
    log_lines.append(f"route_generation={json.dumps(route_generation['deduped_route_candidate_counts'], ensure_ascii=False, sort_keys=True)}")

    candidate_pool, pool_summary = select_by_route(route_candidates, pool_targets, history, final_mode=False)
    pool_by_cluster = {sample["cluster_id"]: sample for sample in candidate_pool}
    pool_route_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in candidate_pool:
        pool_route_candidates[sample["candidate_route"]].append(copy.deepcopy(sample))
    final_samples, final_summary = select_by_route(pool_route_candidates, final_targets, history, final_mode=True)
    # Reassign final IDs without changing the pool IDs stored in candidate_pool.
    final_ids_by_cluster = {sample["cluster_id"]: sample["sample_id"] for sample in final_samples}
    for sample in candidate_pool:
        sample["candidate_pool_sample_id"] = sample["sample_id"]
        if sample["cluster_id"] in final_ids_by_cluster:
            sample["final_annotation_sample_id"] = final_ids_by_cluster[sample["cluster_id"]]

    source_inventory = build_source_inventory(generated_at, history, article_inventory, route_generation, dev_config)
    write_json(output_dir / "source_inventory.json", source_inventory)
    protocol_config = build_protocol_config(generated_at, dev_config, source_inventory)
    write_json(PROTOCOL_CONFIG, protocol_config)
    PROTOCOL_MD.write_text(render_protocol_markdown(dev_config, source_inventory), encoding="utf-8")

    private_rows = route_private_rows(candidate_pool, "candidate_pool") + route_private_rows(final_samples, "final_annotation_1000")
    write_json(output_dir / "private_sampling_manifest.json", {"generated_at": generated_at, "private_rows": private_rows})
    write_jsonl(output_dir / "candidate_pool.jsonl", [strip_private_fields(sample) for sample in candidate_pool])
    write_jsonl(output_dir / "final_annotation_samples.jsonl", [strip_private_fields(sample) for sample in final_samples])

    write_workbook(workbook_1000, final_samples)
    first_400 = final_samples[: min(400, len(final_samples))]
    write_workbook(workbook_400, first_400)
    GUIDE.write_text(render_annotation_guide(), encoding="utf-8")

    blindness = build_blindness_audit(final_samples, workbook_1000)
    write_json(output_dir / "annotation_blindness_audit.json", blindness)
    leakage = build_leakage_audit(final_samples, history, final_summary)
    write_json(output_dir / "leakage_audit.json", leakage)
    data_quality = build_data_quality_audit(final_samples, private_rows)
    write_json(output_dir / "data_quality_audit.json", data_quality)
    exclusion_manifest = {
        "generated_at": generated_at,
        "exclusion_policy": [
            "current Independent500-v2 unlabeled input and adjudicated workbook",
            "Independent500-v1/v2 identifiers",
            "historical Task2 human audit, training, development, and test assets",
            "same sample_id, article_file_id, cluster_id, normalized title SHA, normalized content SHA",
            "exact duplicate and near duplicate to historical or already selected samples",
            "Task1/Task3 same-source-derived identifiers",
            "one synthetic event cluster per selected article-target-event identity",
        ],
        "history_sources": history["source_stats"],
        "history_identifier_counts": history["identifier_counts"],
        "article_normalization_exclusion_counts": article_inventory["normalization_exclusion_counts"],
        "selection_rejection_counts": {
            "candidate_pool": pool_summary["rejection_counts"],
            "final_annotation": final_summary["rejection_counts"],
        },
    }
    write_json(output_dir / "exclusion_manifest.json", exclusion_manifest)

    sampling_manifest = {
        "dataset_id": DATASET_ID,
        "generated_at": generated_at,
        "mode": args.mode,
        "seed": SEED,
        "candidate_pool_target": sum(pool_targets.values()),
        "candidate_pool_route_targets": pool_targets,
        "candidate_pool_actual_count": len(candidate_pool),
        "candidate_pool_route_counts": pool_summary["route_counts"],
        "candidate_pool_shortfalls": pool_summary["route_shortfalls"],
        "final_annotation_target": sum(final_targets.values()),
        "final_annotation_route_targets": final_targets,
        "final_annotation_actual_count": len(final_samples),
        "final_annotation_route_counts": final_summary["route_counts"],
        "final_annotation_shortfalls": final_summary["route_shortfalls"],
        "same_target_company_max_final": data_quality["target_company_max_count"],
        "source_event_cluster_policy": "same synthetic cluster_id appears at most once",
        "route_visible_to_annotators": False,
        "candidate_quality_score_visible_to_annotators": False,
        "model_predictions_visible_to_annotators": False,
        "dianjin_predictions_used": False,
        "training_performed": False,
        "weight_fitting_performed": False,
        "threshold_tuning_performed": False,
        "new_formal_test_result_generated": False,
        "distributions": distribution_summary(final_samples),
        "outputs": {
            "candidate_pool": rel(output_dir / "candidate_pool.jsonl"),
            "final_annotation_samples": rel(output_dir / "final_annotation_samples.jsonl"),
            "annotation_workbook_1000": rel(workbook_1000),
            "annotation_workbook_400": rel(workbook_400),
            "annotation_guide": rel(GUIDE),
            "private_sampling_manifest": rel(output_dir / "private_sampling_manifest.json"),
        },
    }
    write_json(output_dir / "sampling_manifest.json", sampling_manifest)

    run_state = {
        "status": "ready_for_double_blind_human_annotation" if args.mode == "full" else "smoke_complete",
        "generated_at": generated_at,
        "mode": args.mode,
        "completed_stages": [
            "current500_development_protocol_registration",
            "candidate_source_inventory",
            "historical_exclusion_manifest",
            "closed_track_candidate_pool_generation",
            "final_blind_annotation_package_generation",
            "annotation_guide_generation",
            "deduplication_and_leakage_audit",
            "blindness_audit",
            "data_quality_audit",
            "sha256_manifest",
        ],
        "external_blocker": "waiting_for_double_blind_human_annotation_and_adjudication" if args.mode == "full" else "",
        "training_performed": False,
        "weight_fitting_performed": False,
        "threshold_tuning_performed": False,
        "new_formal_test_result_generated": False,
        "historical_evaluation_outputs_modified": False,
        "original_independent500_v2_files_modified": False,
        "workbook_reload_consistency": check_workbook_reloads(workbook_1000, len(final_samples)),
        "log_path": rel(log_path),
        "pid_file": rel(log_dir / "task2_sixdim_rareclass_expansion_v1.pid"),
    }
    write_json(output_dir / "run_state.json", run_state)

    sha_paths = [
        DEV_CONFIG,
        PROTOCOL_CONFIG,
        PROTOCOL_MD,
        GOLD_WORKBOOK,
        GOLD_SAMPLES,
        output_dir / "source_inventory.json",
        output_dir / "exclusion_manifest.json",
        output_dir / "leakage_audit.json",
        output_dir / "candidate_pool.jsonl",
        output_dir / "final_annotation_samples.jsonl",
        output_dir / "sampling_manifest.json",
        output_dir / "private_sampling_manifest.json",
        output_dir / "annotation_blindness_audit.json",
        output_dir / "data_quality_audit.json",
        output_dir / "run_state.json",
        workbook_400,
        workbook_1000,
        GUIDE,
    ]
    sha_hashes = sha_manifest(sha_paths, output_dir / "SHA256SUMS.txt")
    log_lines.append(f"candidate_pool={len(candidate_pool)} final={len(final_samples)}")
    log_lines.append(f"final_route_counts={json.dumps(final_summary['route_counts'], ensure_ascii=False, sort_keys=True)}")
    log_lines.append(f"sha_entries={len(sha_hashes)}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "mode": args.mode,
        "candidate_pool": len(candidate_pool),
        "final_annotation_samples": len(final_samples),
        "candidate_pool_route_counts": pool_summary["route_counts"],
        "final_annotation_route_counts": final_summary["route_counts"],
        "workbook": rel(workbook_1000),
        "guide": rel(GUIDE),
        "output_dir": rel(output_dir),
        "log_path": rel(log_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
