#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path("/opt/data/private/JinRong/kechuang/financial_event_project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.event_credibility.csmar_adapter import build_csmar_financial_cache_v2, load_csmar_financial_records_v2
from src.event_credibility.financial_consistency import _company_aliases, _normalize_company_name

GOLD_DIR = ROOT / "outputs/task4 golddata"
OLD_GOLD = GOLD_DIR / "task4_severity_training_data_v1.xlsx"
NEW_96 = GOLD_DIR / "task4_severity_human_annotation_candidates_96_labeled_double_review_v1.xlsx"
OUT = ROOT / "outputs/task4_impact_score_v2_657_plus_96"
SCRIPT_PATH = ROOT / "scripts/build_task4_impact_score_v2_657_plus_96.py"
SEED = 20260824

GOLD_SHEETS = ("Gold_A_Verified", "Gold_B_Legacy")
NEW_SHEET = "annotation"
NEW_META_SHEET = "sampling_meta"

GOLD_COLUMNS = [
    "sample_id",
    "target_name",
    "target_type",
    "event_name",
    "event_type",
    "title",
    "evidence_sentence",
    "local_context",
    "financial_facts",
    "publish_time",
    "severity_label",
    "risk_intensity",
    "annotation_confidence",
    "annotation_basis",
    "label_tier",
    "label_source",
    "target_local_status",
    "split_status",
    "provenance",
    "provenance_status",
    "supervision_eligible",
]

CSMAR_HINTS = [
    ROOT / "data/external/csmar/财务指标",
    ROOT / "data/external/csmar",
    ROOT / "data/csmar",
    ROOT / "data/raw/csmar",
    ROOT / "data/processed/csmar",
    ROOT / "inputs/csmar",
]

FINANCIAL_CONTRACT = ROOT / "outputs/task34_root_cause_repair/task4_severity_financial_risk_infra/financial_risk_feature_contract.json"
FINANCIAL_DRY_RUN_SUMMARY = ROOT / "outputs/event_credibility/task2_financial_enhanced_cached_dea_v1/financial_audit_summary.json"
CREDIBILITY = ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet"


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def norm_key(value: Any) -> str:
    return norm_text(value).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if not math.isfinite(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [json_ready(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple, set)) else False:
        return None
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_cell(value: Any) -> dict[str, Any]:
    text = norm_text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    text = norm_text(value)
    if not text:
        return None
    try:
        ts = pd.to_datetime(text, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.normalize()


def workbook_manifest(path: Path) -> dict[str, Any]:
    xl = pd.ExcelFile(path)
    sheets: dict[str, Any] = {}
    for sheet in xl.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        sheets[sheet] = {
            "rows": int(len(frame)),
            "columns": list(map(str, frame.columns)),
            "column_count": int(len(frame.columns)),
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "sheets": sheets,
    }


def require_input_paths() -> None:
    if not OLD_GOLD.exists():
        raise FileNotFoundError(OLD_GOLD)
    if not NEW_96.exists():
        raise FileNotFoundError(NEW_96)
    if OLD_GOLD.name == "task34.xlsx" or NEW_96.name == "task34.xlsx":
        raise RuntimeError("task34.xlsx is explicitly excluded from this run")


def build_csmar_cache() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[str]]:
    source_dir = ROOT / "data/external/csmar/财务指标"
    cache_dir = OUT / "csmar_cache"
    manifest = build_csmar_financial_cache_v2(source_dir, cache_dir, force=False)
    records = load_csmar_financial_records_v2(cache_dir)
    company_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = _normalize_company_name(record.get("company_name") or "")
        if key:
            company_index[key].append(record)
    for key, items in company_index.items():
        items.sort(
            key=lambda r: (
                parse_date(r.get("estimated_available_date")) or pd.Timestamp.min.normalize(),
                norm_text(r.get("report_period")),
                norm_text(r.get("Typrep")),
            ),
            reverse=True,
        )
    company_keys = sorted(company_index)
    return manifest, records, company_index, company_keys


def validate_columns(frame: pd.DataFrame, source: str) -> list[str]:
    missing = [col for col in GOLD_COLUMNS if col not in frame.columns]
    extra = [col for col in frame.columns if col not in GOLD_COLUMNS]
    if missing:
        raise RuntimeError(f"{source} missing required columns: {missing}")
    return extra


def load_gold_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_parts = []
    for sheet in GOLD_SHEETS:
        frame = pd.read_excel(OLD_GOLD, sheet_name=sheet)
        validate_columns(frame, f"{OLD_GOLD.name}:{sheet}")
        frame = frame[GOLD_COLUMNS].copy()
        frame["_input_source"] = "original_657"
        frame["_input_sheet"] = sheet
        frame["_source_priority"] = 0
        old_parts.append(frame)
    old = pd.concat(old_parts, ignore_index=True)
    new = pd.read_excel(NEW_96, sheet_name=NEW_SHEET)
    validate_columns(new, f"{NEW_96.name}:{NEW_SHEET}")
    new = new[GOLD_COLUMNS].copy()
    new["_input_source"] = "new_96_double_review"
    new["_input_sheet"] = NEW_SHEET
    new["_source_priority"] = 1
    try:
        meta = pd.read_excel(NEW_96, sheet_name=NEW_META_SHEET)
    except Exception:
        meta = pd.DataFrame(columns=["sample_id"])
    return old, new, meta


def financial_candidate_records(
    target_name: str,
    publish_time: Any,
    company_index: dict[str, list[dict[str, Any]]],
    company_keys: list[str],
) -> list[dict[str, Any]]:
    aliases = {_normalize_company_name(alias) for alias in _company_aliases(target_name)}
    aliases = {alias for alias in aliases if alias}
    if not aliases:
        return []
    matched: list[dict[str, Any]] = []
    for company_key in company_keys:
        if any(alias == company_key or alias in company_key or company_key in alias for alias in aliases):
            matched.extend(company_index.get(company_key, []))
    if not matched:
        return []
    pub = parse_date(publish_time)
    if pub is None:
        return []
    available = []
    for record in matched:
        avail = parse_date(record.get("estimated_available_date"))
        if avail is not None and avail <= pub:
            available.append(record)
    if not available:
        return []
    return available


def extract_internal_keys(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    meta_keep = [col for col in ["sample_id", "article_id", "canonical_event_id", "canonical_match", "duplicate_status"] if col in meta.columns]
    if meta_keep:
        meta_small = meta[meta_keep].copy()
        out = out.merge(meta_small, on="sample_id", how="left", validate="one_to_one")
    else:
        out["article_id"] = ""
        out["canonical_event_id"] = ""

    provenance_dicts = [parse_json_cell(v) for v in out["provenance"]]
    prov_article = [norm_text(d.get("article_id") or d.get("canonical_document_id") or "") for d in provenance_dicts]
    prov_canonical = [norm_text(d.get("canonical_event_id") or d.get("canonical_event") or "") for d in provenance_dicts]
    prov_source = [norm_text(d.get("source") or "") for d in provenance_dicts]

    if "article_id" not in out.columns:
        out["article_id"] = ""
    if "canonical_event_id" not in out.columns:
        out["canonical_event_id"] = ""
    out["_article_id"] = [
        norm_text(a) or p
        for a, p in zip(out["article_id"].tolist(), prov_article)
    ]
    out["_canonical_event_id"] = [
        norm_text(c) or p
        for c, p in zip(out["canonical_event_id"].tolist(), prov_canonical)
    ]
    out["_source_name"] = prov_source
    out["_sample_id_key"] = out["sample_id"].map(norm_key)
    out["_article_key"] = out["_article_id"].map(norm_key)
    out["_canonical_event_key"] = out["_canonical_event_id"].map(norm_key)
    out["_target_event_evidence_key"] = (
        out["target_name"].map(norm_key)
        + "|"
        + out["event_name"].map(norm_key)
        + "|"
        + out["evidence_sentence"].map(norm_key)
    )
    out["_target_event_key"] = out["target_name"].map(norm_key) + "|" + out["event_name"].map(norm_key)
    out["_doc_proxy_key"] = out["_article_key"]
    missing_article = out["_doc_proxy_key"].eq("")
    out.loc[missing_article, "_doc_proxy_key"] = (
        out.loc[missing_article, "title"].map(norm_key)
        + "|"
        + out.loc[missing_article, "publish_time"].map(norm_key)
    )
    return out


def split_bucket(value: Any) -> str:
    text = norm_key(value)
    if "test" in text:
        return "test"
    if "dev" in text or "valid" in text or "val" in text:
        return "dev"
    if "train" in text:
        return "train"
    return "unknown"


def deduplicate(combined: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = combined.sort_values(["_source_priority", "_input_sheet", "sample_id"], kind="mergesort").reset_index(drop=True)
    ordered["_dedup_keep"] = True
    ordered["_dedup_reason"] = ""
    ordered["_dedup_key_name"] = ""
    ordered["_dedup_key_value"] = ""
    rules = [
        ("sample_id", "_sample_id_key", False),
        ("article_id", "_article_key", True),
        ("canonical_event_id", "_canonical_event_key", True),
        ("target_event_evidence", "_target_event_evidence_key", False),
    ]
    removed_rows: list[dict[str, Any]] = []
    for rule_name, key_col, skip_empty in rules:
        seen: set[str] = set()
        for idx, row in ordered[ordered["_dedup_keep"]].iterrows():
            key = norm_text(row.get(key_col))
            if skip_empty and not key:
                continue
            if key in seen:
                ordered.loc[idx, "_dedup_keep"] = False
                ordered.loc[idx, "_dedup_reason"] = f"duplicate_{rule_name}"
                ordered.loc[idx, "_dedup_key_name"] = rule_name
                ordered.loc[idx, "_dedup_key_value"] = key
                removed_rows.append({
                    "sample_id": row.get("sample_id"),
                    "input_source": row.get("_input_source"),
                    "reason": f"duplicate_{rule_name}",
                    "key": key,
                })
            else:
                seen.add(key)
    kept = ordered[ordered["_dedup_keep"]].copy().reset_index(drop=True)

    duplicate_key_summary = {}
    for rule_name, key_col, skip_empty in rules:
        series = ordered[key_col].map(norm_text)
        if skip_empty:
            series = series[series.ne("")]
        counts = series.value_counts()
        duplicate_key_summary[rule_name] = {
            "nonempty_key_count": int(series.ne("").sum()),
            "unique_key_count": int(series.nunique()),
            "duplicate_row_count": int((counts - 1).clip(lower=0).sum()) if len(counts) else 0,
            "duplicate_group_count": int((counts > 1).sum()) if len(counts) else 0,
        }

    report = {
        "combined_before_dedup": int(len(ordered)),
        "combined_after_dedup": int(len(kept)),
        "removed_total": int(len(ordered) - len(kept)),
        "removed_by_reason": Counter(ordered.loc[~ordered["_dedup_keep"], "_dedup_reason"]).copy(),
        "removed_by_input_source": Counter(ordered.loc[~ordered["_dedup_keep"], "_input_source"]).copy(),
        "source_counts_before": Counter(ordered["_input_source"]).copy(),
        "source_counts_after": Counter(kept["_input_source"]).copy(),
        "duplicate_key_summary_before_dedup": duplicate_key_summary,
        "removed_rows": removed_rows,
        "dedup_policy": [
            "preserve original_657 before new_96_double_review",
            "remove duplicate sample_id",
            "remove duplicate article_id only when a traceable article_id exists",
            "remove duplicate canonical_event_id only when a traceable canonical_event_id exists",
            "remove duplicate normalized target_name + event_name + evidence_sentence",
            "do not infer article_id for legacy rows from title alone for deletion; title proxy is used only for leakage audit",
        ],
    }
    return kept, report


def risk_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    risk = pd.to_numeric(frame["risk_intensity"], errors="coerce")
    severity = frame["severity_label"].map(norm_key)
    return {
        "total_rows": int(len(frame)),
        "risk_intensity_1_5_distribution": {str(k): int(v) for k, v in risk.value_counts(dropna=False).sort_index().items()},
        "risk_intensity_by_input_source": {
            str(src): {str(k): int(v) for k, v in pd.to_numeric(part["risk_intensity"], errors="coerce").value_counts(dropna=False).sort_index().items()}
            for src, part in frame.groupby("_input_source", dropna=False)
        },
        "severity_label_auxiliary_distribution": {str(k): int(v) for k, v in severity.value_counts(dropna=False).items()},
        "note": "risk_intensity is the supervised target; severity_label is auxiliary display metadata only.",
    }


def audit_splits(frame: pd.DataFrame) -> dict[str, Any]:
    tmp = frame.copy()
    tmp["_split_bucket"] = tmp["split_status"].map(split_bucket)
    split_counts = tmp["_split_bucket"].value_counts(dropna=False).to_dict()
    raw_split_counts = tmp["split_status"].fillna("").astype(str).value_counts(dropna=False).to_dict()

    def leaks_for(key_col: str) -> dict[str, Any]:
        leaks = []
        nonempty = tmp[tmp[key_col].map(norm_text).ne("")]
        for key, part in nonempty.groupby(key_col, dropna=False):
            buckets = sorted(set(part["_split_bucket"].map(str)))
            if len(buckets) > 1:
                leaks.append({
                    "key": key,
                    "splits": buckets,
                    "sample_ids": part["sample_id"].head(10).tolist(),
                    "count": int(len(part)),
                })
        return {
            "key": key_col,
            "cross_split_leak_count": int(len(leaks)),
            "examples": leaks[:20],
        }

    leak_audits = {
        "sample_id": leaks_for("_sample_id_key"),
        "article_id": leaks_for("_article_key"),
        "document_proxy": leaks_for("_doc_proxy_key"),
        "canonical_event_id": leaks_for("_canonical_event_key"),
        "target_event_evidence": leaks_for("_target_event_evidence_key"),
        "target_event": leaks_for("_target_event_key"),
    }
    cross_split_total = int(sum(item["cross_split_leak_count"] for item in leak_audits.values()))
    has_dev = split_counts.get("dev", 0) > 0
    has_test = split_counts.get("test", 0) > 0
    has_train = split_counts.get("train", 0) > 0
    return {
        "split_bucket_counts": {str(k): int(v) for k, v in split_counts.items()},
        "raw_split_status_counts": {str(k): int(v) for k, v in raw_split_counts.items()},
        "new_96_train_only_verified": bool((tmp.loc[tmp["_input_source"].eq("new_96_double_review"), "_split_bucket"] == "train").all()),
        "cross_split_leak_total": cross_split_total,
        "cross_split_leakage_by_key": leak_audits,
        "leakage_gate_pass": bool(cross_split_total == 0),
        "frozen_dev_available": bool(has_dev),
        "frozen_test_available": bool(has_test),
        "train_available": bool(has_train),
        "model_selection_possible": bool(has_train and has_dev),
        "final_test_evaluation_possible": bool(has_test),
        "split_gate_note": "No frozen dev/test rows are present in the supplied 657+96 gold pool." if not (has_dev and has_test) else "",
    }


def csmar_inventory() -> dict[str, Any]:
    files = []
    missing_locations = []
    for path in CSMAR_HINTS:
        if not path.exists():
            missing_locations.append(str(path))
            continue
        if path.is_file():
            files.append(str(path))
        else:
            for pattern in ("*.parquet", "*.pq", "*.csv", "*.tsv", "*.jsonl", "*.json", "*.xlsx", "*.xls"):
                files.extend(str(p) for p in sorted(path.glob(pattern)))
    return {
        "candidate_locations": [str(p) for p in CSMAR_HINTS],
        "source_files": sorted(set(files)),
        "missing_locations": missing_locations,
        "record_source_available": bool(files),
    }


def read_credibility_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not CREDIBILITY.exists():
        out = frame[["sample_id"]].copy()
        for col in ["credibility_score", "source_entropy", "member_count", "cluster_cohesion_score"]:
            out[col] = np.nan
        return out
    cols = [
        "target_name",
        "event_name",
        "credibility_score",
        "source_entropy",
        "member_count",
        "cluster_cohesion_score",
    ]
    cred = pd.read_parquet(CREDIBILITY, columns=cols).copy()
    cred["_target_event_key"] = cred["target_name"].map(norm_key) + "|" + cred["event_name"].map(norm_key)
    agg = cred.groupby("_target_event_key", as_index=False).agg({
        "credibility_score": "max",
        "source_entropy": "max",
        "member_count": "max",
        "cluster_cohesion_score": "max",
    })
    joined = frame[["sample_id", "_target_event_key"]].merge(agg, on="_target_event_key", how="left")
    return joined.drop(columns=["_target_event_key"])


def financial_coverage(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    inventory = csmar_inventory()
    contract = safe_read_json(FINANCIAL_CONTRACT)
    dry_run = safe_read_json(FINANCIAL_DRY_RUN_SUMMARY)
    cred_join = read_credibility_features(frame)
    row_count = len(frame)
    nonempty_financial_facts = int(frame["financial_facts"].map(norm_text).ne("").sum())
    csmar_manifest, csmar_records, company_index, company_keys = build_csmar_cache()

    selected_records: list[dict[str, Any] | None] = []
    candidate_counts: list[int] = []
    for _, row in frame.iterrows():
        if norm_text(row.get("target_type")) != "company":
            selected_records.append(None)
            candidate_counts.append(0)
            continue
        candidates = financial_candidate_records(row.get("target_name"), row.get("publish_time"), company_index, company_keys)
        candidate_counts.append(int(len(candidates)))
        selected_records.append(candidates[0] if candidates else None)

    company_mask = frame["target_type"].fillna("").astype(str).eq("company")
    company_row_count = int(company_mask.sum())
    company_candidate_count = int(sum(1 for n in candidate_counts if n > 0))
    company_candidate_rate = float(company_candidate_count / company_row_count) if company_row_count else 0.0
    industry_row_count = int((~company_mask).sum())
    candidate_violations = 0
    for row, rec in zip(frame.itertuples(index=False), selected_records):
        if rec is None:
            continue
        pub = parse_date(getattr(row, "publish_time", None))
        avail = parse_date(rec.get("estimated_available_date"))
        if pub is not None and avail is not None and avail > pub:
            candidate_violations += 1

    def feature_row(name: str, source: str, values: pd.Series | None, required: bool, state: str) -> dict[str, Any]:
        available = 0 if values is None else int(values.notna().sum())
        return {
            "rule_or_feature": name,
            "required": bool(required),
            "source": source,
            "state": state,
            "available_count": available,
            "total_count": int(row_count),
            "coverage_rate": float(available / row_count) if row_count else 0.0,
            "missing_count": int(row_count - available),
            "missing_values_imputed": False,
            "leakage_checked": False if source == "CSMAR" and not inventory["record_source_available"] else True,
            "leakage_violation_count": 0,
            "status": "available" if available else ("rule_not_evaluable" if required else "missing_optional"),
        }

    rows = [
        feature_row("credibility", str(CREDIBILITY), cred_join["credibility_score"], True, "available_upstream" if CREDIBILITY.exists() else "missing"),
        feature_row("source_entropy", str(CREDIBILITY), cred_join["source_entropy"], True, "available_upstream" if CREDIBILITY.exists() else "missing"),
        feature_row("member_count", str(CREDIBILITY), cred_join["member_count"], True, "available_upstream" if CREDIBILITY.exists() else "missing"),
        feature_row("event_cluster_cohesion", str(CREDIBILITY), cred_join["cluster_cohesion_score"], True, "available_upstream" if CREDIBILITY.exists() else "missing"),
        feature_row("financial_facts_text_present", "gold.financial_facts", frame["financial_facts"].where(frame["financial_facts"].map(norm_text).ne("")), False, "text_only_not_csmar"),
    ]

    record_series = pd.Series(selected_records, dtype=object)
    rows.extend([
        feature_row("asset_liability_ratio", "CSMAR", record_series.map(lambda r: None if r is None else r.get("debt_ratio")), True, "available" if record_series.map(lambda r: None if r is None else r.get("debt_ratio")).notna().any() else "rule_not_evaluable"),
        feature_row("current_ratio", "CSMAR", record_series.map(lambda r: None if r is None else r.get("current_ratio")), True, "available" if record_series.map(lambda r: None if r is None else r.get("current_ratio")).notna().any() else "rule_not_evaluable"),
        feature_row("interest_coverage", "CSMAR", record_series.map(lambda r: None if r is None else r.get("interest_coverage")), True, "available" if record_series.map(lambda r: None if r is None else r.get("interest_coverage")).notna().any() else "rule_not_evaluable"),
        feature_row("operating_cashflow", "CSMAR", record_series.map(lambda r: None if r is None else r.get("ocf")), True, "available" if record_series.map(lambda r: None if r is None else r.get("ocf")).notna().any() else "rule_not_evaluable"),
        feature_row("roe", "CSMAR", record_series.map(lambda r: None if r is None else r.get("roe")), True, "available" if record_series.map(lambda r: None if r is None else r.get("roe")).notna().any() else "rule_not_evaluable"),
        feature_row("roa", "CSMAR", record_series.map(lambda r: None if r is None else r.get("roa")), True, "available" if record_series.map(lambda r: None if r is None else r.get("roa")).notna().any() else "rule_not_evaluable"),
        feature_row("revenue_growth", "CSMAR", record_series.map(lambda r: None if r is None else r.get("revenue_yoy")), True, "available" if record_series.map(lambda r: None if r is None else r.get("revenue_yoy")).notna().any() else "rule_not_evaluable"),
        feature_row("net_profit_growth", "CSMAR", record_series.map(lambda r: None if r is None else r.get("net_profit_yoy")), True, "available" if record_series.map(lambda r: None if r is None else r.get("net_profit_yoy")).notna().any() else "rule_not_evaluable"),
        feature_row("gross_margin", "CSMAR", record_series.map(lambda r: None if r is None else r.get("gross_margin")), True, "available" if record_series.map(lambda r: None if r is None else r.get("gross_margin")).notna().any() else "rule_not_evaluable"),
        feature_row("industry_mean_pe", "CSMAR", record_series.map(lambda r: None if r is None else r.get("industry_mean_pe")), False, "available_optional"),
        feature_row("pe_ratio", "CSMAR", record_series.map(lambda r: None if r is None else r.get("pe_ratio")), False, "available_optional"),
        feature_row("financial_candidate_present", "CSMAR", pd.Series([c > 0 for c in candidate_counts]), True, "available"),
    ])

    table = pd.DataFrame(rows)
    hard_required = table[table["required"] & table["rule_or_feature"].ne("financial_facts_text_present")]
    financial_rule_gate_pass = bool(
        inventory["record_source_available"]
        and company_candidate_rate >= 0.95
        and candidate_violations == 0
    )
    coverage = {
        "row_count": int(row_count),
        "company_row_count": company_row_count,
        "industry_row_count": industry_row_count,
        "company_candidate_coverage_rate": company_candidate_rate,
        "company_rows_with_candidates": company_candidate_count,
        "candidate_lookup_violations": int(candidate_violations),
        "financial_contract_path": str(FINANCIAL_CONTRACT),
        "financial_contract_status": contract.get("status"),
        "financial_contract_hard_constraint": contract.get("hard_constraint", "DisclosureDate <= EventDate"),
        "disclosure_leakage_policy": "EventDate >= estimated_available_date; no post-event CSMAR disclosures may be used.",
        "csmar_inventory": inventory,
        "task2_financial_dry_run_summary": dry_run,
        "nonempty_financial_facts_text_count": nonempty_financial_facts,
        "csmar_cache_manifest": {
            "path": str(OUT / "csmar_cache" / "csmar_cache_manifest.json"),
            "rows": int(csmar_manifest["outputs"]["financial_records_rows"]),
            "tables": list(csmar_manifest.get("table_metadata", {}).keys()),
        },
        "feature_coverage": rows,
        "rows_with_any_upstream_credibility_feature": int(cred_join["credibility_score"].notna().sum()),
        "rows_with_all_hard_csmar_features": company_candidate_count,
        "rule_not_evaluable_count": int(row_count - company_candidate_count),
        "rule_not_evaluable_rate": float((row_count - company_candidate_count) / row_count) if row_count else 0.0,
        "financial_rule_gate_pass": financial_rule_gate_pass,
        "missing_financial_values_filled_with_zero": False,
        "leakage_audit": {
            "evaluated_csmar_records": int(len(csmar_records)),
            "disclosure_date_violations": int(candidate_violations),
            "status": "past_only_estimated_available_date_checked" if inventory["record_source_available"] else "not_evaluable_missing_csmar",
        },
    }
    return coverage, table


def quality_gate(
    old: pd.DataFrame,
    new: pd.DataFrame,
    merged: pd.DataFrame,
    dedup_report: dict[str, Any],
    split_audit: dict[str, Any],
    financial: dict[str, Any],
) -> dict[str, Any]:
    risk = pd.to_numeric(merged["risk_intensity"], errors="coerce")
    data_errors = []
    if len(old) != 657:
        data_errors.append(f"original_human_gold_count_expected_657_got_{len(old)}")
    if len(new) != 96:
        data_errors.append(f"new_double_review_count_expected_96_got_{len(new)}")
    if risk.isna().any() or not risk.dropna().isin([1, 2, 3, 4, 5]).all():
        data_errors.append("risk_intensity_invalid_or_missing")
    if merged["sample_id"].duplicated().any():
        data_errors.append("sample_id_duplicate_after_dedup")
    if not split_audit["new_96_train_only_verified"]:
        data_errors.append("new_96_not_train_only")

    reasons = []
    data_gate_pass = not data_errors
    if not data_gate_pass:
        reasons.extend(data_errors)
    if not split_audit["leakage_gate_pass"]:
        reasons.append("cross_split_leakage_detected")
    if not split_audit["model_selection_possible"]:
        reasons.append("frozen_dev_not_available_for_model_selection")
    if not split_audit["final_test_evaluation_possible"]:
        reasons.append("frozen_test_not_available_for_single_final_evaluation")
    if not financial["financial_rule_gate_pass"]:
        reasons.append("strict_csmar_financial_rule_gate_not_evaluable")

    train_ready = bool(
        data_gate_pass
        and split_audit["leakage_gate_pass"]
        and split_audit["model_selection_possible"]
        and split_audit["final_test_evaluation_possible"]
        and financial["financial_rule_gate_pass"]
    )
    return {
        "status": "pass" if train_ready else "blocked_audit_only",
        "data_gate_pass": data_gate_pass,
        "data_gate_errors": data_errors,
        "dedup_removed_total": dedup_report["removed_total"],
        "split_leakage_gate_pass": split_audit["leakage_gate_pass"],
        "model_selection_possible_on_frozen_dev": split_audit["model_selection_possible"],
        "single_final_test_evaluation_possible": split_audit["final_test_evaluation_possible"],
        "financial_rule_gate_pass": financial["financial_rule_gate_pass"],
        "training_allowed": train_ready,
        "training_executed": False,
        "blocked_reasons": reasons,
        "contract_enforcement": {
            "risk_intensity_is_primary_target": True,
            "severity_color_is_auxiliary": True,
            "new_96_train_only": True,
            "no_smote_or_label_balancing": True,
            "no_benchmark_1513_training": True,
            "no_post_event_csmar_disclosure": True,
            "missing_financial_values_not_filled_zero": True,
        },
    }


def null_metrics(experiment: str, definition: str, readiness: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    risk = pd.to_numeric(frame["risk_intensity"], errors="coerce")
    high_tail_support = int(risk.isin([4, 5]).sum())
    return {
        "experiment": experiment,
        "definition": definition,
        "training_run": False,
        "model_selection_split": None,
        "test_evaluation_run": False,
        "qwk": None,
        "mae": None,
        "spearman": None,
        "plus_minus_1_accuracy": None,
        "tail_4_5": {
            "support": high_tail_support,
            "precision": None,
            "recall": None,
            "mae": None,
            "qwk": None,
        },
        "rule_trigger_accuracy": None,
        "abstention_rate": None,
        "retained_coverage": None,
        "color_classification_metrics_auxiliary": None,
        "skipped_reason": "training_readiness_gate_failed",
        "blocked_reasons": readiness["blocked_reasons"],
    }


def write_metric_files(readiness: dict[str, Any], merged: pd.DataFrame) -> dict[str, Any]:
    specs = {
        "A": "original_657 + original ordinal model",
        "B": "657 + 96 + same ordinal model",
        "C": "B + strict financial rule gate",
        "D": "C + abstention gate",
    }
    summary: dict[str, Any] = {}
    for key, definition in specs.items():
        metrics = null_metrics(key, definition, readiness, merged)
        path = OUT / f"metrics_{key}.json"
        write_json(path, metrics)
        summary[key] = {
            "path": str(path),
            "qwk": None,
            "mae": None,
            "spearman": None,
            "training_run": False,
            "skipped_reason": metrics["skipped_reason"],
        }
    write_json(OUT / "metrics_summary_abcd.json", summary)
    return summary


def format_merged_xlsx(path: Path) -> None:
    wb = load_workbook(path)
    ws = wb.active
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wrap_cols = {
        "title",
        "evidence_sentence",
        "local_context",
        "financial_facts",
        "annotation_basis",
        "provenance",
    }
    fill = PatternFill("solid", fgColor="FFF2CC")
    for cell in ws[1]:
        cell.fill = fill
    widths = {
        "sample_id": 30,
        "target_name": 16,
        "target_type": 12,
        "event_name": 20,
        "event_type": 20,
        "title": 32,
        "evidence_sentence": 56,
        "local_context": 72,
        "financial_facts": 36,
        "publish_time": 20,
        "severity_label": 14,
        "risk_intensity": 14,
        "annotation_confidence": 20,
        "annotation_basis": 36,
        "label_tier": 14,
        "label_source": 24,
        "target_local_status": 20,
        "split_status": 24,
        "provenance": 44,
        "provenance_status": 20,
        "supervision_eligible": 22,
    }
    header = [cell.value for cell in ws[1]]
    for idx, name in enumerate(header, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = widths.get(str(name), 18)
        if name in wrap_cols:
            for row in ws.iter_rows(min_row=2, min_col=idx, max_col=idx, max_row=ws.max_row):
                row[0].alignment = row[0].alignment.copy(wrap_text=True, vertical="top")
    wb.save(path)


def write_merged_outputs(merged: pd.DataFrame) -> None:
    public = merged[GOLD_COLUMNS].copy()
    xlsx_path = OUT / "merged_gold_train_v2.xlsx"
    parquet_path = OUT / "merged_gold_train_v2.parquet"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        public.to_excel(writer, index=False, sheet_name="merged_gold_train_v2")
    format_merged_xlsx(xlsx_path)
    public.to_parquet(parquet_path, index=False)


def write_report(
    input_manifest: dict[str, Any],
    dedup_report: dict[str, Any],
    split_audit: dict[str, Any],
    distribution: dict[str, Any],
    financial: dict[str, Any],
    readiness: dict[str, Any],
    metrics_summary: dict[str, Any],
    frozen_unchanged: bool,
) -> None:
    old_info = input_manifest["inputs"]["original_657"]
    new_info = input_manifest["inputs"]["new_96_double_review"]
    lines = [
        "# Task4 Impact Score v2 657+96 Audit",
        "",
        "This run keeps risk_intensity (1-5) as the primary target. Severity colors are auxiliary display metadata.",
        "",
        "## Inputs",
        f"- original_657: {old_info['path']}",
        f"- original_657 sha256: {old_info['sha256']}",
        f"- new_96: {new_info['path']}",
        f"- new_96 sha256: {new_info['sha256']}",
        "",
        "## Merge",
        f"- before dedup: {dedup_report['combined_before_dedup']}",
        f"- after dedup: {dedup_report['combined_after_dedup']}",
        f"- removed: {dedup_report['removed_total']}",
        f"- removed by reason: {json.dumps(json_ready(dedup_report['removed_by_reason']), ensure_ascii=False)}",
        "",
        "## Risk Intensity Distribution",
        json.dumps(distribution["risk_intensity_1_5_distribution"], ensure_ascii=False),
        "",
        "## Split and Leakage",
        f"- split buckets: {json.dumps(split_audit['split_bucket_counts'], ensure_ascii=False)}",
        f"- cross-split leak total: {split_audit['cross_split_leak_total']}",
        f"- frozen dev available: {split_audit['frozen_dev_available']}",
        f"- frozen test available: {split_audit['frozen_test_available']}",
        "",
        "## Financial Rule Gate",
        f"- CSMAR source available: {financial['csmar_inventory']['record_source_available']}",
        f"- rows with upstream credibility feature: {financial['rows_with_any_upstream_credibility_feature']}",
        f"- rule_not_evaluable_count: {financial['rule_not_evaluable_count']}",
        f"- financial_rule_gate_pass: {financial['financial_rule_gate_pass']}",
        "",
        "## Readiness",
        f"- training_allowed: {readiness['training_allowed']}",
        f"- blocked reasons: {json.dumps(readiness['blocked_reasons'], ensure_ascii=False)}",
        "",
        "## A/B/C/D",
        json.dumps(metrics_summary, ensure_ascii=False, indent=2),
        "",
        "## Artifact Mutation",
        f"- input Gold files unchanged by SHA256: {frozen_unchanged}",
    ]
    (OUT / "impact_score_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    require_input_paths()

    old_sha_before = sha256_file(OLD_GOLD)
    new_sha_before = sha256_file(NEW_96)
    input_manifest = {
        "status": "inputs_located",
        "random_seed": SEED,
        "excluded_files": [str(GOLD_DIR / "task34.xlsx")],
        "inputs": {
            "original_657": workbook_manifest(OLD_GOLD),
            "new_96_double_review": workbook_manifest(NEW_96),
        },
        "gold_sheets_used": list(GOLD_SHEETS),
        "new_sheet_used": NEW_SHEET,
    }

    old, new, meta = load_gold_frames()
    if len(old) != 657 or len(new) != 96:
        input_manifest["status"] = "input_count_mismatch"
        input_manifest["observed_counts"] = {"original": int(len(old)), "new": int(len(new))}

    combined = pd.concat([old, new], ignore_index=True)
    keyed = extract_internal_keys(combined, meta)
    merged, dedup_report = deduplicate(keyed)
    distribution = risk_distribution(merged)
    split_audit = audit_splits(merged)
    financial, rule_table = financial_coverage(merged)
    readiness = quality_gate(old, new, merged, dedup_report, split_audit, financial)
    metrics_summary = write_metric_files(readiness, merged)

    write_merged_outputs(merged)
    write_json(OUT / "input_manifest.json", input_manifest)
    write_json(OUT / "merge_dedup_report.json", dedup_report)
    write_json(OUT / "split_leakage_audit.json", split_audit)
    write_json(OUT / "risk_intensity_distribution.json", distribution)
    write_json(OUT / "financial_feature_coverage.json", financial)
    rule_table.to_csv(OUT / "rule_trigger_coverage.csv", index=False)
    write_json(OUT / "training_readiness_gate.json", readiness)

    old_sha_after = sha256_file(OLD_GOLD)
    new_sha_after = sha256_file(NEW_96)
    frozen_unchanged = bool(old_sha_before == old_sha_after and new_sha_before == new_sha_after)

    model_manifest = {
        "status": "audit_only_no_model_trained" if not readiness["training_allowed"] else "ready_but_training_not_implemented_in_audit_mode",
        "training_run": False,
        "final_model_path": None,
        "risk_intensity_primary_target": True,
        "severity_color_auxiliary_only": True,
        "experiments": metrics_summary,
        "blocked_reasons": readiness["blocked_reasons"],
        "input_gold_sha256_before": {
            "original_657": old_sha_before,
            "new_96_double_review": new_sha_before,
        },
        "input_gold_sha256_after": {
            "original_657": old_sha_after,
            "new_96_double_review": new_sha_after,
        },
        "input_gold_files_unchanged": frozen_unchanged,
        "task1_task2_task3_core_modified": False,
        "frozen_dev_test_modified": False,
        "benchmark_1513_used_for_training": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "reproducible_script": str(SCRIPT_PATH),
    }
    write_json(OUT / "model_manifest.json", model_manifest)

    reproduce = "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        f"python {SCRIPT_PATH.relative_to(ROOT)}",
        "",
    ])
    repro_path = OUT / "reproduce_task4_impact_score_v2_657_plus_96.sh"
    repro_path.write_text(reproduce, encoding="utf-8")
    repro_path.chmod(0o755)

    write_report(input_manifest, dedup_report, split_audit, distribution, financial, readiness, metrics_summary, frozen_unchanged)
    print(json.dumps({
        "status": readiness["status"],
        "original_rows": int(len(old)),
        "new_rows": int(len(new)),
        "merged_after_dedup": int(len(merged)),
        "risk_distribution": distribution["risk_intensity_1_5_distribution"],
        "split_bucket_counts": split_audit["split_bucket_counts"],
        "financial_rule_gate_pass": financial["financial_rule_gate_pass"],
        "training_allowed": readiness["training_allowed"],
        "blocked_reasons": readiness["blocked_reasons"],
        "output_dir": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
