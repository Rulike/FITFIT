#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path("/opt/data/private/JinRong/kechuang/financial_event_project")
OUT = ROOT / "outputs/task4_impact_v2_annotation_round1"
STAGE2 = ROOT / "outputs/task4/stage2/task4_event_level_samples.parquet"
IMPACT_FRAME = ROOT / "outputs/task4_impact_frame_v1/impact_frames_3127.parquet"
WORKLOAD = ROOT / "reports/task4_stage2_annotation_workload_list.csv"
GOLD_MASTER = ROOT / "outputs/task4_severity_ordinal_v2_actual_training/severity_master_gold.xlsx"
HIGHRISK = ROOT / "outputs/task4_highrisk_v2_reject_audit_pack/severity_highrisk_v2_reject_audit_200.parquet"
SOURCE_SUMMARY = ROOT / "outputs/task4_severity_weak_pretrain_gold_finetune_v1/source_summary.json"
SIDE_CAR = ROOT / "outputs/task34_raw_controlled_integration/task4_weak_label_sidecar.parquet"
TEST_RAW = ROOT / "data/raw/news_with_tags_test.xlsx"

SEED = 20260826

COMMON_FIELDS = [
    "sample_id",
    "article_id",
    "title",
    "content",
    "target_name",
    "target_type",
    "target_code",
    "event_name",
    "event_type",
    "evidence_sentence",
    "local_context",
    "financial_facts",
    "arguments",
    "publish_time",
    "source",
]

DIRECTION_FIELDS = [
    "human_direction",
    "impact_horizon",
    "target_role",
    "direction_confidence",
    "direction_evidence",
    "direction_reason",
    "valid_for_direction",
]

IMPACT_FIELDS = [
    "risk_intensity",
    "severity_axis",
    "scope_axis",
    "urgency_axis",
    "novelty_axis",
    "evolution_axis",
    "annotation_confidence",
    "annotation_basis",
    "evidence_ids",
    "exclusion_reason",
    "human_review_status",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def norm_key(value: Any) -> str:
    return clean_text(value).lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple, set)) else False:
        return None
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_test_fingerprints() -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    xl = pd.ExcelFile(TEST_RAW)
    for sheet in xl.sheet_names:
        frame = pd.read_excel(TEST_RAW, sheet_name=sheet)
        frame = frame.rename(
            columns={
                "article_file_id": "article_id",
                "article_title": "title",
                "article_publish_time": "publish_time",
                "article_source": "source",
            }
        )
        frame["article_id"] = frame.get("article_id", pd.Series(dtype=object)).map(clean_text)
        frame["title"] = frame.get("title", pd.Series(dtype=object)).map(clean_text)
        frame["content"] = frame.get("content", pd.Series(dtype=object)).map(clean_text)
        frame["duplication_id"] = frame.get("duplication_id", pd.Series(dtype=object)).map(clean_text)
        frame["content_hash"] = [
            sha256_bytes((clean_text(title) + "\n" + clean_text(content)).encode("utf-8"))
            for title, content in zip(frame["title"], frame["content"])
        ]
        frames.append(frame[["article_id", "title", "content", "duplication_id", "content_hash"]])
    test = pd.concat(frames, ignore_index=True)
    return {
        "path": str(TEST_RAW),
        "sha256": sha256_file(TEST_RAW),
        "sheet_names": xl.sheet_names,
        "row_count": int(len(test)),
        "article_id_count": int(test["article_id"].nunique()),
        "content_hash_count": int(test["content_hash"].nunique()),
        "article_ids": set(test["article_id"].tolist()),
        "content_hashes": set(test["content_hash"].tolist()),
        "duplication_ids": set(test["duplication_id"].tolist()) - {""},
        "sheets": {sheet: int(len(pd.read_excel(TEST_RAW, sheet_name=sheet))) for sheet in xl.sheet_names},
    }


def table_manifest(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path)
        sheets: dict[str, Any] = {}
        for sheet in xl.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet)
            sheets[sheet] = {
                "rows": int(len(frame)),
                "columns": [str(c) for c in frame.columns],
                "column_count": int(len(frame.columns)),
            }
        return {
            "path": str(path),
            "kind": "excel_workbook",
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "sheets": sheets,
        }
    if suffix == ".csv":
        frame = pd.read_csv(path)
        return {
            "path": str(path),
            "kind": "csv",
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "column_count": int(len(frame.columns)),
        }
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return {
            "path": str(path),
            "kind": "parquet",
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "column_count": int(len(frame.columns)),
        }
    return {
        "path": str(path),
        "kind": suffix.lstrip(".") or "unknown",
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def load_gold_inventory() -> dict[str, Any]:
    xl = pd.ExcelFile(GOLD_MASTER)
    out: dict[str, Any] = {
        "path": str(GOLD_MASTER),
        "sha256": sha256_file(GOLD_MASTER),
        "sheet_rows": {},
        "risk_intensity_distribution": {},
    }
    for sheet in xl.sheet_names:
        frame = pd.read_excel(GOLD_MASTER, sheet_name=sheet)
        out["sheet_rows"][sheet] = int(len(frame))
        if sheet == "master_gold" and "risk_intensity" in frame.columns:
            out["risk_intensity_distribution"] = {
                str(k): int(v) for k, v in pd.to_numeric(frame["risk_intensity"], errors="coerce").value_counts(dropna=False).sort_index().items()
            }
    return out


def read_stage2() -> pd.DataFrame:
    stage2 = pd.read_parquet(STAGE2)
    stage2["publish_time"] = stage2["publish_time"].map(clean_text)
    stage2["title"] = stage2["title"].map(clean_text)
    stage2["content"] = stage2["content"].map(clean_text)
    stage2["target_name"] = stage2["target_name"].map(clean_text)
    stage2["target_type"] = stage2["target_type"].map(clean_text)
    stage2["target_code"] = stage2["target_code"].map(clean_text)
    stage2["event_name"] = stage2["event_name"].map(clean_text)
    stage2["source"] = stage2["source"].map(clean_text)
    stage2["emotion_label"] = stage2["emotion_label"].map(clean_text)
    stage2["duplication_id"] = stage2["duplication_id"].map(clean_text)
    stage2["clean_content_hash"] = stage2["clean_content_hash"].map(clean_text)
    stage2["month"] = stage2["publish_time"].str[:7]
    return stage2


def read_impact_features() -> pd.DataFrame:
    impact = pd.read_parquet(IMPACT_FRAME, columns=["task4_sample_id", "risk_or_opportunity", "frame_confidence"])
    impact["risk_or_opportunity"] = impact["risk_or_opportunity"].map(clean_text)
    return impact


def read_workload() -> pd.DataFrame:
    wl = pd.read_csv(WORKLOAD)
    wl["task4_sample_id"] = wl["task4_sample_id"].map(clean_text)
    wl["annotatability_bucket"] = wl["annotatability_bucket"].map(clean_text)
    wl["document_mapping_status"] = wl["document_mapping_status"].map(clean_text)
    wl["mapping_type"] = wl["mapping_type"].map(clean_text)
    return wl


def read_sidecar_inventory() -> dict[str, Any]:
    side = pd.read_parquet(
        SIDE_CAR,
        columns=["contract_valid", "test_overlap", "allowed_usage", "provenance_status", "normalized_label"],
    )
    return {
        "path": str(SIDE_CAR),
        "sha256": sha256_file(SIDE_CAR),
        "row_count": int(len(side)),
        "contract_valid_distribution": {str(k): int(v) for k, v in side["contract_valid"].value_counts(dropna=False).to_dict().items()},
        "test_overlap_distribution": {str(k): int(v) for k, v in side["test_overlap"].value_counts(dropna=False).to_dict().items()},
        "allowed_usage_distribution": {str(k): int(v) for k, v in side["allowed_usage"].value_counts(dropna=False).to_dict().items()},
        "provenance_status_distribution": {str(k): int(v) for k, v in side["provenance_status"].value_counts(dropna=False).to_dict().items()},
        "normalized_label_distribution": {str(k): int(v) for k, v in side["normalized_label"].value_counts(dropna=False).to_dict().items()},
    }


def load_source_summary() -> dict[str, Any]:
    summary = safe_read_json(SOURCE_SUMMARY)
    return {
        "path": str(SOURCE_SUMMARY),
        "sha256": sha256_file(SOURCE_SUMMARY) if SOURCE_SUMMARY.exists() else "",
        "payload": summary,
    }


def build_non_test_inventory(stage2: pd.DataFrame, impact: pd.DataFrame, wl: pd.DataFrame, gold: dict[str, Any], sidecar: dict[str, Any], source_summary: dict[str, Any], highrisk: pd.DataFrame, test_info: dict[str, Any]) -> dict[str, Any]:
    train = stage2[stage2["official_split"].eq("train")].copy()
    test = stage2[stage2["official_split"].eq("test")].copy()
    train_emotion = {str(k): int(v) for k, v in train["emotion_label"].value_counts(dropna=False).to_dict().items()}
    train_target = {str(k): int(v) for k, v in train["target_type"].value_counts(dropna=False).to_dict().items()}
    train_source = {str(k): int(v) for k, v in train["source"].value_counts(dropna=False).head(15).to_dict().items()}
    train_month = {str(k): int(v) for k, v in train["month"].value_counts(dropna=False).sort_index().to_dict().items()}
    impact_conf = {str(k): int(v) for k, v in pd.to_numeric(impact["frame_confidence"], errors="coerce").value_counts(dropna=False).sort_index().to_dict().items()}
    impact_roo = {str(k): int(v) for k, v in impact["risk_or_opportunity"].value_counts(dropna=False).to_dict().items()}
    workload_bucket = {str(k): int(v) for k, v in wl["annotatability_bucket"].value_counts(dropna=False).to_dict().items()}
    workload_target = {str(k): int(v) for k, v in wl["target_type"].value_counts(dropna=False).to_dict().items()}
    workload_emotion = {str(k): int(v) for k, v in wl["emotion_label"].value_counts(dropna=False).to_dict().items()}
    highrisk_bucket = {str(k): int(v) for k, v in highrisk["audit_bucket"].value_counts(dropna=False).to_dict().items()}
    highrisk_priority = {str(k): int(v) for k, v in highrisk["candidate_priority"].value_counts(dropna=False).to_dict().items()}
    return {
        "verified_human_gold": {
            "path": gold["path"],
            "sha256": gold["sha256"],
            "sheets": gold["sheet_rows"],
            "risk_intensity_distribution": gold["risk_intensity_distribution"],
            "count": int(gold["sheet_rows"].get("master_gold", 0)),
        },
        "verified_train_only_human": {
            "count": int(gold["sheet_rows"].get("train_only_gold", 0)),
            "source": "severity_master_gold.xlsx:train_only_gold",
        },
        "weak_severity": source_summary["payload"].get("W", {}),
        "direction_emotion_old_labels": source_summary["payload"].get("D_direction", {}),
        "validity_old_labels": source_summary["payload"].get("D_validity", {}),
        "task4_event_pool": {
            "path": str(STAGE2),
            "sha256": sha256_file(STAGE2),
            "row_count": int(len(stage2)),
            "train_count": int(len(train)),
            "test_count": int(len(test)),
            "train_emotion_distribution": train_emotion,
            "train_target_type_distribution": train_target,
            "train_source_top15_distribution": train_source,
            "train_month_distribution": train_month,
        },
        "impact_frame": {
            "path": str(IMPACT_FRAME),
            "sha256": sha256_file(IMPACT_FRAME),
            "row_count": int(len(impact)),
            "frame_confidence_distribution": impact_conf,
            "risk_or_opportunity_distribution": impact_roo,
        },
        "workload_list": {
            "path": str(WORKLOAD),
            "sha256": sha256_file(WORKLOAD),
            "row_count": int(len(wl)),
            "annotatability_bucket_distribution": workload_bucket,
            "target_type_distribution": workload_target,
            "emotion_label_distribution": workload_emotion,
        },
        "highrisk_audit_pack": {
            "path": str(HIGHRISK),
            "sha256": sha256_file(HIGHRISK),
            "row_count": int(len(highrisk)),
            "audit_bucket_distribution": highrisk_bucket,
            "candidate_priority_distribution": highrisk_priority,
        },
        "weak_sidecar": sidecar,
        "frozen_official_test": {
            "path": test_info["path"],
            "sha256": test_info["sha256"],
            "row_count": test_info["row_count"],
            "article_id_count": test_info["article_id_count"],
            "content_hash_count": test_info["content_hash_count"],
            "sheet_counts": test_info["sheets"],
        },
    }


def build_evidence_sentence(title: str, content: str, target: str, event: str) -> tuple[str, str]:
    text = f"{clean_text(title)}\n{clean_text(content)}"
    sentences = [s.strip() for s in re.split(r"(?<=[。！？\n])", text) if s.strip()]
    if not sentences:
        return clean_text(title)[:1200], clean_text(title)[:2000]
    target = clean_text(target)
    event = clean_text(event)

    def score(sent: str) -> float:
        local = sent
        prev = ""
        nxt = ""
        idx = sentences.index(sent)
        if idx > 0:
            prev = sentences[idx - 1]
        if idx + 1 < len(sentences):
            nxt = sentences[idx + 1]
        blob = local + prev + nxt
        hits = 0
        for token in [target, event]:
            if token and token in blob:
                hits += 3
            elif token and token in local:
                hits += 2
        hits += sum(blob.count(tok) for tok in ["增长", "下降", "亏损", "利润", "销量", "订单", "交付", "融资", "债务", "现金流", "处罚", "问询", "立案", "风险", "利好", "突破", "投产", "合作", "并购"])
        hits += 1 if re.search(r"\d+(?:\.\d+)?%?|亿元|万元|亿", blob) else 0
        return float(hits)

    ranked = sorted(sentences, key=lambda s: (-score(s), len(s), s))
    best = ranked[0]
    idx = sentences.index(best)
    prev = sentences[idx - 1] if idx > 0 else ""
    nxt = sentences[idx + 1] if idx + 1 < len(sentences) else ""
    local_context = " ".join(x for x in [prev, best, nxt] if x)
    if not best:
        best = clean_text(title)
        local_context = clean_text(content)[:2000]
    return best, local_context


def extract_financial_facts(content: str) -> str:
    text = clean_text(content)
    sentences = [s.strip() for s in re.split(r"(?<=[。！？\n])", text) if s.strip()]
    terms = ["利润", "营收", "收入", "销量", "交付", "净利", "净利润", "现金流", "负债", "资产", "债务", "融资", "债券", "亏损", "增速", "同比", "环比", "亿元", "万元", "%"]
    picked = []
    for sent in sentences:
        if re.search(r"\d+(?:\.\d+)?%?|亿元|万元|万台|万辆|亿", sent) and any(term in sent for term in terms):
            picked.append(sent)
        if len(picked) >= 2:
            break
    if not picked:
        for sent in sentences[:2]:
            if any(term in sent for term in terms):
                picked.append(sent)
    return " ".join(picked[:2])


def build_arguments(row: pd.Series) -> str:
    payload = {
        "article_id": row["article_id"],
        "target_name": row["target_name"],
        "target_type": row["target_type"],
        "target_code": row["target_code"],
        "event_name": row["event_name"],
        "event_type": row["event_type"],
        "source": row["source"],
        "clean_content_hash": row["clean_content_hash"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def compute_positive_priority(df: pd.DataFrame) -> pd.Series:
    frame_conf = pd.to_numeric(df["frame_confidence"], errors="coerce").fillna(1.0)
    roo = df["risk_or_opportunity"].fillna("").map({"uncertain": 2.0, "risk": 1.0, "opportunity": 0.0}).fillna(0.0)
    target_boost = df["target_type"].fillna("").eq("industry").astype(float)
    dup_boost = df["duplication_id"].fillna("").ne("").astype(float)
    event_freq = df.groupby("event_name")["event_name"].transform("count").astype(float)
    source_freq = df.groupby("source")["source"].transform("count").astype(float)
    month_freq = df.groupby("month")["month"].transform("count").astype(float)
    content_len = df["content"].map(len).astype(float)
    score = (
        3.2 * (1.0 - frame_conf)
        + 1.8 * roo
        + 0.9 * target_boost
        + 0.5 * dup_boost
        + 0.25 * (1.0 / np.log1p(event_freq))
        + 0.20 * (1.0 / np.log1p(source_freq))
        + 0.15 * (1.0 / np.log1p(month_freq))
        + 0.05 * (content_len < 400).astype(float)
    )
    return score


def select_samples(stage2: pd.DataFrame, workload: pd.DataFrame, direction_total: int = 1200, impact_total: int = 800) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = stage2[stage2["official_split"].eq("train")].copy().reset_index(drop=True)
    workload_ids = set(workload["task4_sample_id"].tolist())
    train["is_seed"] = train["task4_sample_id"].isin(workload_ids)
    train["is_positive"] = train["emotion_label"].eq("正面")
    train["is_nonpositive"] = ~train["is_positive"]

    impact_small = pd.read_parquet(IMPACT_FRAME, columns=["task4_sample_id", "risk_or_opportunity", "frame_confidence"])
    train = train.merge(impact_small, on="task4_sample_id", how="left")
    train = train.merge(
        workload[
            [
                "task4_sample_id",
                "annotatability_bucket",
                "needs_double_review",
                "needs_arbitration_risk",
                "high_effort_case",
                "document_mapping_status",
                "mapping_type",
                "mapping_confidence",
                "label_count",
                "is_multi_event_document",
                "has_impact_rationale",
                "impact_rationale_len",
                "content_len",
                "evidence_unit_count",
                "evidence_sentence_count",
                "graph_document_available",
                "graph_event_available",
                "graph_evidence_available",
                "graph_state_available",
                "source_missing",
                "publish_time_missing",
                "target_code_missing",
                "target_name_missing",
                "event_name_missing",
                "title_missing",
                "content_missing",
            ]
        ],
        on="task4_sample_id",
        how="left",
        validate="one_to_one",
    )

    # Full train set and seed rows are never used for official test.
    train["official_test_overlap"] = False

    seed = train[train["is_seed"]].copy().reset_index(drop=True)
    extra = train[~train["is_seed"]].copy().reset_index(drop=True)
    extra_nonpositive = extra[extra["is_nonpositive"]].copy().reset_index(drop=True)
    extra_positive = extra[extra["is_positive"]].copy().reset_index(drop=True)

    if len(seed) != len(workload):
        raise RuntimeError(f"seed/workload mismatch: seed={len(seed)} workload={len(workload)}")

    # Keep all remaining neutral/negative rows; prioritize them for both packs.
    extra_nonpositive = extra_nonpositive.sort_values(
        ["frame_confidence", "target_type", "event_name", "source", "task4_sample_id"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    extra_positive["priority_score"] = compute_positive_priority(extra_positive)
    extra_positive = extra_positive.sort_values(
        ["priority_score", "frame_confidence", "target_type", "event_name", "source", "task4_sample_id"],
        ascending=[False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    need_nonpositive = max(0, min(len(extra_nonpositive), direction_total - len(seed) - len(extra_positive)))
    if need_nonpositive == 0:
        need_nonpositive = len(extra_nonpositive)
    # We want all remaining non-positive rows if possible; otherwise as many as possible.
    extra_nonpositive_sel = extra_nonpositive.head(min(len(extra_nonpositive), need_nonpositive)).copy()

    remaining_direction = direction_total - len(seed) - len(extra_nonpositive_sel)
    remaining_impact = impact_total - len(seed) - len(extra_nonpositive_sel)
    if remaining_direction < 0 or remaining_impact < 0:
        raise RuntimeError("seed plus non-positive rows exceed requested total")
    if len(extra_positive) < remaining_direction:
        raise RuntimeError(f"not enough positive rows for direction: need {remaining_direction}, have {len(extra_positive)}")
    if len(extra_positive) < remaining_impact:
        raise RuntimeError(f"not enough positive rows for impact: need {remaining_impact}, have {len(extra_positive)}")

    extra_positive_dir = extra_positive.head(remaining_direction).copy()
    extra_positive_imp = extra_positive.head(remaining_impact).copy()

    direction = pd.concat([seed, extra_nonpositive_sel, extra_positive_dir], ignore_index=True)
    impact = pd.concat([seed, extra_nonpositive_sel, extra_positive_imp], ignore_index=True)

    direction = direction.drop_duplicates("task4_sample_id").reset_index(drop=True)
    impact = impact.drop_duplicates("task4_sample_id").reset_index(drop=True)

    if len(direction) != direction_total:
        raise RuntimeError(f"direction sample count mismatch: {len(direction)} != {direction_total}")
    if len(impact) != impact_total:
        raise RuntimeError(f"impact sample count mismatch: {len(impact)} != {impact_total}")
    if not set(impact["task4_sample_id"]).issubset(set(direction["task4_sample_id"])):
        raise RuntimeError("impact sample set is not a subset of direction set")

    selection_detail = {
        "seed_rows": int(len(seed)),
        "seed_nonpositive_rows": int(len(seed[seed["is_nonpositive"]])),
        "seed_positive_rows": int(len(seed[seed["is_positive"]])),
        "extra_nonpositive_available": int(len(extra_nonpositive)),
        "extra_nonpositive_selected": int(len(extra_nonpositive_sel)),
        "extra_positive_available": int(len(extra_positive)),
        "direction_positive_selected": int(len(extra_positive_dir)),
        "impact_positive_selected": int(len(extra_positive_imp)),
        "direction_total": int(len(direction)),
        "impact_total": int(len(impact)),
    }

    return direction, impact, selection_detail


def build_common_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["evidence_sentence"] = [
        build_evidence_sentence(title, content, target, event)[0]
        for title, content, target, event in zip(out["title"], out["content"], out["target_name"], out["event_name"])
    ]
    out["local_context"] = [
        build_evidence_sentence(title, content, target, event)[1]
        for title, content, target, event in zip(out["title"], out["content"], out["target_name"], out["event_name"])
    ]
    out["financial_facts"] = [extract_financial_facts(content) for content in out["content"]]
    if "event_type" not in out.columns:
        out["event_type"] = ""
    out["event_type"] = out["event_type"].fillna("")
    if "schema_id" in out.columns:
        mask = out["event_type"].eq("")
        out.loc[mask, "event_type"] = out.loc[mask, "schema_id"].fillna("")
    out["arguments"] = [build_arguments(row) for _, row in out.iterrows()]
    return out


def build_direction_pack(frame: pd.DataFrame) -> pd.DataFrame:
    out = build_common_context(frame)
    out = out[
        [
            "task4_sample_id",
            "article_id",
            "title",
            "content",
            "target_name",
            "target_type",
            "target_code",
            "event_name",
            "event_type",
            "evidence_sentence",
            "local_context",
            "financial_facts",
            "arguments",
            "publish_time",
            "source",
        ]
    ].copy()
    out = out.rename(columns={"task4_sample_id": "sample_id"})
    for col in DIRECTION_FIELDS:
        out[col] = ""
    return out[COMMON_FIELDS + DIRECTION_FIELDS]


def build_impact_pack(frame: pd.DataFrame) -> pd.DataFrame:
    out = build_common_context(frame)
    out = out[
        [
            "task4_sample_id",
            "article_id",
            "title",
            "content",
            "target_name",
            "target_type",
            "target_code",
            "event_name",
            "event_type",
            "evidence_sentence",
            "local_context",
            "financial_facts",
            "arguments",
            "publish_time",
            "source",
        ]
    ].copy()
    out = out.rename(columns={"task4_sample_id": "sample_id"})
    for col in IMPACT_FIELDS:
        out[col] = ""
    return out[COMMON_FIELDS + IMPACT_FIELDS]


def sort_for_version(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    def rank_value(sample_id: str) -> str:
        return hashlib.sha256(f"{sample_id}|{seed}".encode("utf-8")).hexdigest()

    out = df.copy()
    out["_order_key"] = out["sample_id"].map(rank_value)
    out = out.sort_values(["_order_key", "sample_id"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["_order_key"])


def write_main_workbook(path: Path, df: pd.DataFrame, sheet_name: str, guide_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]], example_rows: list[dict[str, Any]], df_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    wb = load_workbook(path)
    ws = wb[sheet_name]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    example_fill = PatternFill("solid", fgColor="F7F2D9")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    widths = {
        "sample_id": 30,
        "article_id": 34,
        "title": 40,
        "content": 78,
        "target_name": 16,
        "target_type": 12,
        "target_code": 12,
        "event_name": 24,
        "event_type": 26,
        "evidence_sentence": 60,
        "local_context": 72,
        "financial_facts": 38,
        "arguments": 48,
        "publish_time": 20,
        "source": 22,
        "human_direction": 18,
        "impact_horizon": 18,
        "target_role": 20,
        "direction_confidence": 20,
        "direction_evidence": 28,
        "direction_reason": 28,
        "valid_for_direction": 18,
        "risk_intensity": 16,
        "severity_axis": 14,
        "scope_axis": 14,
        "urgency_axis": 14,
        "novelty_axis": 14,
        "evolution_axis": 14,
        "annotation_confidence": 20,
        "annotation_basis": 28,
        "evidence_ids": 24,
        "exclusion_reason": 24,
        "human_review_status": 20,
    }
    for idx, column in enumerate(df.columns, start=1):
        col_letter = get_column_letter(idx)
        ws.column_dimensions[col_letter].width = widths.get(column, 18)
        if column in {"content", "evidence_sentence", "local_context", "financial_facts", "arguments", "direction_reason", "direction_evidence", "annotation_basis"}:
            for row in ws.iter_rows(min_row=2, min_col=idx, max_col=idx, max_row=ws.max_row):
                row[0].alignment = Alignment(vertical="top", wrap_text=True)

    # selection of validation rules for blank human fields.
    if df_type == "direction":
        dv_map = {
            "human_direction": '"positive,negative,neutral,uncertain"',
            "impact_horizon": '"short,medium,long,not_evaluable"',
            "target_role": '"direct_beneficiary,direct_exposed,indirect_chain,industry_level,unclear"',
            "direction_confidence": '"high,medium,low"',
            "valid_for_direction": '"yes,no,review"',
        }
    else:
        dv_map = {
            "risk_intensity": '"1,2,3,4,5,not_evaluable"',
            "severity_axis": '"1,2,3,4,5,not_evaluable"',
            "scope_axis": '"1,2,3,4,5,not_evaluable"',
            "urgency_axis": '"1,2,3,4,5,not_evaluable"',
            "novelty_axis": '"1,2,3,4,5,not_evaluable"',
            "evolution_axis": '"1,2,3,4,5,not_evaluable"',
            "annotation_confidence": '"high,medium,low"',
            "human_review_status": '"pass,review,abstain"',
        }
    for field, quoted in dv_map.items():
        idx = list(df.columns).index(field) + 1
        dv = DataValidation(type="list", formula1=f"={quoted}", allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{get_column_letter(idx)}2:{get_column_letter(idx)}{ws.max_row}")

    ws.title = sheet_name

    if "instructions" in wb.sheetnames:
        del wb["instructions"]
    if "field_dictionary" in wb.sheetnames:
        del wb["field_dictionary"]
    if "examples" in wb.sheetnames:
        del wb["examples"]

    inst = wb.create_sheet("instructions", 1)
    inst["A1"] = f"Task4 {'Direction' if df_type == 'direction' else 'Five-Axis'} Round 1 Annotation Instructions"
    inst["A1"].font = Font(bold=True, size=14)
    inst["A1"].alignment = Alignment(wrap_text=True)

    row_cursor = 3
    inst.cell(row=row_cursor, column=1, value="item").font = Font(bold=True)
    inst.cell(row=row_cursor, column=2, value="description").font = Font(bold=True)
    row_cursor += 1
    instruction_rows = guide_rows
    for row in instruction_rows:
        inst.cell(row=row_cursor, column=1, value=row["item"])
        inst.cell(row=row_cursor, column=2, value=row["description"])
        row_cursor += 1

    row_cursor += 2
    inst.cell(row=row_cursor, column=1, value="field").font = Font(bold=True)
    inst.cell(row=row_cursor, column=2, value="allowed_values").font = Font(bold=True)
    inst.cell(row=row_cursor, column=3, value="notes").font = Font(bold=True)
    row_cursor += 1
    for row in field_rows:
        inst.cell(row=row_cursor, column=1, value=row["field"])
        inst.cell(row=row_cursor, column=2, value=row["allowed_values"])
        inst.cell(row=row_cursor, column=3, value=row.get("notes", ""))
        row_cursor += 1

    row_cursor += 2
    if df_type == "direction":
        inst.cell(row=row_cursor, column=1, value="direction_examples").font = Font(bold=True)
        row_cursor += 1
        headers = ["example_id", "target_name", "event_name", "evidence", "human_direction", "impact_horizon", "target_role", "direction_confidence", "direction_evidence", "direction_reason", "valid_for_direction"]
        for col, header in enumerate(headers, start=1):
            inst.cell(row=row_cursor, column=col, value=header).font = Font(bold=True)
        row_cursor += 1
        for example in example_rows:
            for col, header in enumerate(headers, start=1):
                value = example.get(header, "")
                inst.cell(row=row_cursor, column=col, value=value)
            for col in range(1, len(headers) + 1):
                inst.cell(row=row_cursor, column=col).fill = example_fill
            row_cursor += 1
    else:
        inst.cell(row=row_cursor, column=1, value="five_axis_examples").font = Font(bold=True)
        row_cursor += 1
        headers = [
            "example_id",
            "target_name",
            "event_name",
            "evidence",
            "risk_intensity",
            "severity_axis",
            "scope_axis",
            "urgency_axis",
            "novelty_axis",
            "evolution_axis",
            "annotation_confidence",
            "annotation_basis",
            "evidence_ids",
            "exclusion_reason",
            "human_review_status",
        ]
        for col, header in enumerate(headers, start=1):
            inst.cell(row=row_cursor, column=col, value=header).font = Font(bold=True)
        row_cursor += 1
        for example in example_rows:
            for col, header in enumerate(headers, start=1):
                value = example.get(header, "")
                inst.cell(row=row_cursor, column=col, value=value)
            for col in range(1, len(headers) + 1):
                inst.cell(row=row_cursor, column=col).fill = example_fill
            row_cursor += 1

    dict_sheet = wb.create_sheet("field_dictionary")
    dict_sheet["A1"] = "field"
    dict_sheet["B1"] = "kind"
    dict_sheet["C1"] = "description"
    dict_sheet["D1"] = "allowed_values"
    for c in ("A1", "B1", "C1", "D1"):
        dict_sheet[c].font = Font(bold=True)
    for idx, row in enumerate(field_rows, start=2):
        dict_sheet.cell(row=idx, column=1, value=row["field"])
        dict_sheet.cell(row=idx, column=2, value=row["kind"])
        dict_sheet.cell(row=idx, column=3, value=row["description"])
        dict_sheet.cell(row=idx, column=4, value=row["allowed_values"])

    # Pretty widths.
    for ws_name in ["instructions", "field_dictionary"]:
        w = wb[ws_name]
        for cell in w[1]:
            cell.fill = header_fill
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        widths_map = {"A": 28, "B": 60, "C": 60, "D": 60, "E": 18, "F": 18, "G": 18, "H": 18, "I": 18, "J": 18, "K": 18, "L": 20, "M": 18, "N": 20, "O": 18}
        for letter, width in widths_map.items():
            w.column_dimensions[letter].width = width
        for row in w.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

    wb.save(path)


def build_guides() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    direction_instructions = [
        {"item": "purpose", "description": "Direction round 1 annotation pack for non-official-test task4 rows. Human fields are blank and must be filled by annotators only."},
        {"item": "leakage", "description": "Do not use official test labels, model predictions, probabilities, or rule outputs. Only the article text, local evidence, and visible context are shown."},
        {"item": "direction_rule", "description": "Direction judges target-relative impact polarity, not article sentiment. Positive text can still mean negative impact, and vice versa."},
        {"item": "review_rule", "description": "If direction cannot be judged reliably from the visible evidence, mark review instead of forcing a label."},
    ]
    direction_fields = [
        {"field": "human_direction", "kind": "label", "description": "Target-relative direction label for the news impact.", "allowed_values": "positive / negative / neutral / uncertain"},
        {"field": "impact_horizon", "kind": "label", "description": "Time horizon used in the direction judgment.", "allowed_values": "short / medium / long / not_evaluable"},
        {"field": "target_role", "kind": "label", "description": "How the target participates in the event.", "allowed_values": "direct_beneficiary / direct_exposed / indirect_chain / industry_level / unclear"},
        {"field": "direction_confidence", "kind": "label", "description": "Annotator confidence for the direction judgment.", "allowed_values": "high / medium / low"},
        {"field": "direction_evidence", "kind": "free_text", "description": "Short evidence note supporting the direction label.", "allowed_values": "blank during distribution"},
        {"field": "direction_reason", "kind": "free_text", "description": "Short reason for the final direction judgment.", "allowed_values": "blank during distribution"},
        {"field": "valid_for_direction", "kind": "label", "description": "Whether the row is valid for direction judgment.", "allowed_values": "yes / no / review"},
    ]
    direction_examples = [
        {
            "example_id": "DIR_EX_1",
            "target_name": "示例股份A",
            "event_name": "订单增长",
            "evidence": "2026年一季度订单和交付同步增长，营收明显改善。",
            "human_direction": "positive",
            "impact_horizon": "short",
            "target_role": "direct_beneficiary",
            "direction_confidence": "high",
            "direction_evidence": "订单、交付和营收都直接改善。",
            "direction_reason": "事件直接抬升目标经营结果。",
            "valid_for_direction": "yes",
        },
        {
            "example_id": "DIR_EX_2",
            "target_name": "示例股份B",
            "event_name": "监管处罚",
            "evidence": "公司因违规被立案调查并收到行政处罚决定。",
            "human_direction": "negative",
            "impact_horizon": "short",
            "target_role": "direct_exposed",
            "direction_confidence": "high",
            "direction_evidence": "处罚直接作用于目标主体。",
            "direction_reason": "监管结果通常对目标形成直接负向冲击。",
            "valid_for_direction": "yes",
        },
    ]

    impact_instructions = [
        {"item": "purpose", "description": "Five-axis round 1 annotation pack for non-official-test task4 rows. Human fields are blank and must be filled by annotators only."},
        {"item": "leakage", "description": "Do not use model scores, fixed thresholds, prefilled severity, or weak labels as if they were gold."},
        {"item": "severity_rule", "description": "Risk intensity is the primary 1-5 score. The five axes are Severity, Scope, Urgency, Novelty, and Evolution."},
        {"item": "review_rule", "description": "If the evidence does not support a stable severity judgment, use not_evaluable or review rather than forcing a score."},
    ]
    impact_fields = [
        {"field": "risk_intensity", "kind": "label", "description": "Primary risk intensity score for the event-target pair.", "allowed_values": "1 / 2 / 3 / 4 / 5 / not_evaluable"},
        {"field": "severity_axis", "kind": "label", "description": "Severity axis score.", "allowed_values": "1 / 2 / 3 / 4 / 5 / not_evaluable"},
        {"field": "scope_axis", "kind": "label", "description": "Scope axis score.", "allowed_values": "1 / 2 / 3 / 4 / 5 / not_evaluable"},
        {"field": "urgency_axis", "kind": "label", "description": "Urgency axis score.", "allowed_values": "1 / 2 / 3 / 4 / 5 / not_evaluable"},
        {"field": "novelty_axis", "kind": "label", "description": "Novelty axis score.", "allowed_values": "1 / 2 / 3 / 4 / 5 / not_evaluable"},
        {"field": "evolution_axis", "kind": "label", "description": "Evolution axis score.", "allowed_values": "1 / 2 / 3 / 4 / 5 / not_evaluable"},
        {"field": "annotation_confidence", "kind": "label", "description": "Annotator confidence for the five-axis judgment.", "allowed_values": "high / medium / low"},
        {"field": "annotation_basis", "kind": "free_text", "description": "Short rationale for the five-axis score.", "allowed_values": "blank during distribution"},
        {"field": "evidence_ids", "kind": "free_text", "description": "Article-local evidence ids or short evidence references.", "allowed_values": "blank during distribution"},
        {"field": "exclusion_reason", "kind": "free_text", "description": "Why the row is not evaluable or should be excluded.", "allowed_values": "blank during distribution"},
        {"field": "human_review_status", "kind": "label", "description": "Final review status for the row.", "allowed_values": "pass / review / abstain"},
    ]
    impact_examples = [
        {
            "example_id": "IMPACT_EX_1",
            "target_name": "示例股份C",
            "event_name": "债务展期失败",
            "evidence": "债券展期方案未通过，短期偿债压力显著上升。",
            "risk_intensity": "5",
            "severity_axis": "5",
            "scope_axis": "3",
            "urgency_axis": "5",
            "novelty_axis": "4",
            "evolution_axis": "4",
            "annotation_confidence": "high",
            "annotation_basis": "直接债务冲击且时间压力强。",
            "evidence_ids": "blank during distribution",
            "exclusion_reason": "",
            "human_review_status": "pass",
        },
        {
            "example_id": "IMPACT_EX_2",
            "target_name": "示例行业D",
            "event_name": "政策利好发布",
            "evidence": "新政策扩大行业支持范围，但对单个主体的直接冲击有限。",
            "risk_intensity": "2",
            "severity_axis": "2",
            "scope_axis": "4",
            "urgency_axis": "2",
            "novelty_axis": "2",
            "evolution_axis": "1",
            "annotation_confidence": "medium",
            "annotation_basis": "行业面更广，单体严重度较低。",
            "evidence_ids": "blank during distribution",
            "exclusion_reason": "",
            "human_review_status": "pass",
        },
    ]
    return direction_instructions, direction_fields, direction_examples, impact_instructions, impact_fields, impact_examples


def write_reports(
    stage2: pd.DataFrame,
    impact_frame: pd.DataFrame,
    workload: pd.DataFrame,
    selected_direction: pd.DataFrame,
    selected_impact: pd.DataFrame,
    selection_detail: dict[str, Any],
    inventory: dict[str, Any],
    leakage: dict[str, Any],
    test_info: dict[str, Any],
    gold_inventory: dict[str, Any],
    sidecar_inventory: dict[str, Any],
    source_summary: dict[str, Any],
) -> None:
    stage2_train = stage2[stage2["official_split"].eq("train")].copy()
    direction_counts = {
        "total": int(len(selected_direction)),
        "seed_rows": int(selection_detail["seed_rows"]),
        "extra_nonpositive_rows": int(selection_detail["extra_nonpositive_selected"]),
        "extra_positive_rows": int(selection_detail["direction_positive_selected"]),
        "target_type_distribution": {str(k): int(v) for k, v in selected_direction["target_type"].value_counts().to_dict().items()},
        "emotion_label_distribution": {str(k): int(v) for k, v in selected_direction["emotion_label"].value_counts().to_dict().items()},
        "month_distribution": {str(k): int(v) for k, v in selected_direction["month"].value_counts().sort_index().to_dict().items()},
        "source_top15_distribution": {str(k): int(v) for k, v in selected_direction["source"].value_counts().head(15).to_dict().items()},
        "risk_or_opportunity_distribution": {str(k): int(v) for k, v in selected_direction["risk_or_opportunity"].value_counts().to_dict().items()},
        "frame_confidence_distribution": {str(k): int(v) for k, v in pd.to_numeric(selected_direction["frame_confidence"], errors="coerce").value_counts(dropna=False).sort_index().to_dict().items()},
        "annotatability_bucket_distribution": {str(k): int(v) for k, v in selected_direction["annotatability_bucket"].fillna("").value_counts(dropna=False).to_dict().items()},
        "train_coverage_ratio": float(len(selected_direction) / len(stage2_train)) if len(stage2_train) else 0.0,
    }
    impact_counts = {
        "total": int(len(selected_impact)),
        "seed_rows": int(selection_detail["seed_rows"]),
        "extra_nonpositive_rows": int(selection_detail["extra_nonpositive_selected"]),
        "extra_positive_rows": int(selection_detail["impact_positive_selected"]),
        "target_type_distribution": {str(k): int(v) for k, v in selected_impact["target_type"].value_counts().to_dict().items()},
        "emotion_label_distribution": {str(k): int(v) for k, v in selected_impact["emotion_label"].value_counts().to_dict().items()},
        "month_distribution": {str(k): int(v) for k, v in selected_impact["month"].value_counts().sort_index().to_dict().items()},
        "source_top15_distribution": {str(k): int(v) for k, v in selected_impact["source"].value_counts().head(15).to_dict().items()},
        "risk_or_opportunity_distribution": {str(k): int(v) for k, v in selected_impact["risk_or_opportunity"].value_counts().to_dict().items()},
        "frame_confidence_distribution": {str(k): int(v) for k, v in pd.to_numeric(selected_impact["frame_confidence"], errors="coerce").value_counts(dropna=False).sort_index().to_dict().items()},
        "annotatability_bucket_distribution": {str(k): int(v) for k, v in selected_impact["annotatability_bucket"].fillna("").value_counts(dropna=False).to_dict().items()},
        "subset_of_direction": bool(set(selected_impact["task4_sample_id"]).issubset(set(selected_direction["task4_sample_id"]))),
    }
    write_json(OUT / "sampling_report.json", {
        "status": "sampling_complete",
        "seed_source": str(WORKLOAD),
        "selection_detail": selection_detail,
        "direction": direction_counts,
        "impact": impact_counts,
        "notes": [
            "Direction uses the full workload seed plus all remaining non-positive rows and the highest-priority positive rows.",
            "Impact is a strict subset of Direction and prioritizes the same hard positives.",
            "No model predictions, weak labels, or official test labels are shown in the annotation workbooks.",
        ],
    })

    write_json(OUT / "input_manifest.json", {
        "status": "inputs_registered",
        "seed": SEED,
        "inputs": {
            "stage2_pool": table_manifest(STAGE2),
            "impact_frame": table_manifest(IMPACT_FRAME),
            "workload_list": table_manifest(WORKLOAD),
            "gold_master": gold_inventory,
            "highrisk_audit_pack": table_manifest(HIGHRISK),
            "weak_sidecar": sidecar_inventory,
            "source_summary": {
                "path": source_summary["path"],
                "sha256": source_summary["sha256"],
                "keys": sorted(source_summary["payload"].keys()),
            },
            "official_test_raw": test_info,
        },
        "selection_plan": {
            "direction_total": 1200,
            "impact_total": 800,
            "direction_subset_of_impact": False,
            "impact_subset_of_direction": True,
        },
    })

    write_json(OUT / "non_test_data_inventory.json", inventory)

    write_json(OUT / "leakage_report.json", leakage)


def make_leakage_report(selected_direction: pd.DataFrame, selected_impact: pd.DataFrame, test_info: dict[str, Any]) -> dict[str, Any]:
    direction_article_overlap = sorted(set(selected_direction["article_id"]).intersection(test_info["article_ids"]))
    impact_article_overlap = sorted(set(selected_impact["article_id"]).intersection(test_info["article_ids"]))
    direction_hash_overlap = sorted(set(selected_direction["clean_content_hash"]).intersection(test_info["content_hashes"]))
    impact_hash_overlap = sorted(set(selected_impact["clean_content_hash"]).intersection(test_info["content_hashes"]))
    direction_dup_values = set(selected_direction["duplication_id"].dropna().map(clean_text)) - {""}
    impact_dup_values = set(selected_impact["duplication_id"].dropna().map(clean_text)) - {""}
    direction_dup_overlap = sorted(direction_dup_values.intersection(test_info["duplication_ids"]))
    impact_dup_overlap = sorted(impact_dup_values.intersection(test_info["duplication_ids"]))
    return {
        "status": "leakage_checked",
        "official_test_path": test_info["path"],
        "official_test_sha256": test_info["sha256"],
        "official_test_row_count": test_info["row_count"],
        "direction_article_id_overlap_count": int(len(direction_article_overlap)),
        "impact_article_id_overlap_count": int(len(impact_article_overlap)),
        "direction_content_hash_overlap_count": int(len(direction_hash_overlap)),
        "impact_content_hash_overlap_count": int(len(impact_hash_overlap)),
        "direction_duplication_id_overlap_count": int(len(direction_dup_overlap)),
        "impact_duplication_id_overlap_count": int(len(impact_dup_overlap)),
        "direction_overlap_examples": direction_article_overlap[:20],
        "impact_overlap_examples": impact_article_overlap[:20],
        "direction_hash_examples": direction_hash_overlap[:10],
        "impact_hash_examples": impact_hash_overlap[:10],
        "gate_pass": bool(
            len(direction_article_overlap) == 0
            and len(impact_article_overlap) == 0
            and len(direction_hash_overlap) == 0
            and len(impact_hash_overlap) == 0
            and len(direction_dup_overlap) == 0
            and len(impact_dup_overlap) == 0
        ),
    }


def build_base_report(direction_df: pd.DataFrame, impact_df: pd.DataFrame, selection_detail: dict[str, Any], inventory: dict[str, Any], leakage: dict[str, Any]) -> None:
    write_json(OUT / "task4_direction_round1.parquet_manifest.json", {
        "rows": int(len(direction_df)),
        "columns": list(direction_df.columns),
        "sha256": sha256_bytes(direction_df.to_json(orient="records", force_ascii=False).encode("utf-8")),
    })
    write_json(OUT / "task4_impact_axes_round1.parquet_manifest.json", {
        "rows": int(len(impact_df)),
        "columns": list(impact_df.columns),
        "sha256": sha256_bytes(impact_df.to_json(orient="records", force_ascii=False).encode("utf-8")),
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    test_info = load_test_fingerprints()
    stage2 = read_stage2()
    impact_frame = read_impact_features()
    workload = read_workload()
    gold_inventory = load_gold_inventory()
    sidecar_inventory = read_sidecar_inventory()
    source_summary = load_source_summary()
    highrisk = pd.read_parquet(HIGHRISK)
    inventory = build_non_test_inventory(stage2, impact_frame, workload, gold_inventory, sidecar_inventory, source_summary, highrisk, test_info)

    selected_direction_raw, selected_impact_raw, selection_detail = select_samples(stage2, workload, direction_total=1200, impact_total=800)
    selected_direction = selected_direction_raw.copy().reset_index(drop=True)
    selected_impact = selected_impact_raw.copy().reset_index(drop=True)

    leakage = make_leakage_report(selected_direction, selected_impact, test_info)
    write_reports(stage2, impact_frame, workload, selected_direction, selected_impact, selection_detail, inventory, leakage, test_info, gold_inventory, sidecar_inventory, source_summary)

    direction_pack = build_direction_pack(selected_direction)
    impact_pack = build_impact_pack(selected_impact)

    direction_pack_a = sort_for_version(direction_pack, SEED)
    direction_pack_b = sort_for_version(direction_pack, SEED + 1)
    impact_pack_a = sort_for_version(impact_pack, SEED + 2)
    impact_pack_b = sort_for_version(impact_pack, SEED + 3)

    # guard rails before writing outputs
    assert len(direction_pack_a) == 1200
    assert len(direction_pack_b) == 1200
    assert len(impact_pack_a) == 800
    assert len(impact_pack_b) == 800
    assert set(direction_pack_a["sample_id"]) == set(direction_pack_b["sample_id"])
    assert set(impact_pack_a["sample_id"]) == set(impact_pack_b["sample_id"])
    assert set(impact_pack_a["sample_id"]).issubset(set(direction_pack_a["sample_id"]))
    assert direction_pack_a["sample_id"].duplicated().sum() == 0
    assert impact_pack_a["sample_id"].duplicated().sum() == 0
    assert direction_pack_a[DIRECTION_FIELDS].fillna("").eq("").all().all()
    assert direction_pack_b[DIRECTION_FIELDS].fillna("").eq("").all().all()
    assert impact_pack_a[IMPACT_FIELDS].fillna("").eq("").all().all()
    assert impact_pack_b[IMPACT_FIELDS].fillna("").eq("").all().all()
    assert leakage["gate_pass"] is True

    # write json/parquet versions for reproducibility
    direction_pack_a.to_json(OUT / "task4_direction_round1_A.jsonl", orient="records", lines=True, force_ascii=False)
    direction_pack_b.to_json(OUT / "task4_direction_round1_B.jsonl", orient="records", lines=True, force_ascii=False)
    impact_pack_a.to_json(OUT / "task4_impact_axes_round1_A.jsonl", orient="records", lines=True, force_ascii=False)
    impact_pack_b.to_json(OUT / "task4_impact_axes_round1_B.jsonl", orient="records", lines=True, force_ascii=False)
    direction_pack_a.to_parquet(OUT / "task4_direction_round1.parquet", index=False)
    impact_pack_a.to_parquet(OUT / "task4_impact_axes_round1.parquet", index=False)

    direction_instructions, direction_fields, direction_examples, impact_instructions, impact_fields, impact_examples = build_guides()
    write_main_workbook(OUT / "task4_direction_round1_A.xlsx", direction_pack_a, "direction_round1", direction_instructions, direction_fields, direction_examples, "direction")
    write_main_workbook(OUT / "task4_direction_round1_B.xlsx", direction_pack_b, "direction_round1", direction_instructions, direction_fields, direction_examples, "direction")
    write_main_workbook(OUT / "task4_impact_axes_round1_A.xlsx", impact_pack_a, "impact_axes_round1", impact_instructions, impact_fields, impact_examples, "impact")
    write_main_workbook(OUT / "task4_impact_axes_round1_B.xlsx", impact_pack_b, "impact_axes_round1", impact_instructions, impact_fields, impact_examples, "impact")

    build_base_report(direction_pack_a, impact_pack_a, selection_detail, inventory, leakage)

    write_json(OUT / "protected_sha256_before.json", {
        "stage2_pool": sha256_file(STAGE2),
        "impact_frame": sha256_file(IMPACT_FRAME),
        "workload_list": sha256_file(WORKLOAD),
        "gold_master": sha256_file(GOLD_MASTER),
        "highrisk_audit_pack": sha256_file(HIGHRISK),
        "weak_sidecar": sha256_file(SIDE_CAR),
        "source_summary": sha256_file(SOURCE_SUMMARY),
        "official_test_raw": test_info["sha256"],
    })
    write_json(OUT / "protected_sha256_after.json", {
        "stage2_pool": sha256_file(STAGE2),
        "impact_frame": sha256_file(IMPACT_FRAME),
        "workload_list": sha256_file(WORKLOAD),
        "gold_master": sha256_file(GOLD_MASTER),
        "highrisk_audit_pack": sha256_file(HIGHRISK),
        "weak_sidecar": sha256_file(SIDE_CAR),
        "source_summary": sha256_file(SOURCE_SUMMARY),
        "official_test_raw": test_info["sha256"],
    })

    guide_md = OUT / "annotation_guide.md"
    guide_md.write_text(
        "\n".join(
            [
                "# Task4 Impact v2 Annotation Guide",
                "",
                "This package is for non-official-test Direction and five-axis annotation only.",
                "",
                "## Shared rules",
                "- Do not use model predictions, probabilities, thresholds, or weak labels as gold.",
                "- Keep annotation grounded in the visible article text and the local evidence window.",
                "- Direction is target-relative polarity; severity is intensity, not polarity.",
                "",
                "## Direction fields",
                "- human_direction: positive / negative / neutral / uncertain",
                "- impact_horizon: short / medium / long / not_evaluable",
                "- target_role: direct_beneficiary / direct_exposed / indirect_chain / industry_level / unclear",
                "- direction_confidence: high / medium / low",
                "- direction_evidence: short supporting evidence note",
                "- direction_reason: one-sentence final judgment",
                "- valid_for_direction: yes / no / review",
                "",
                "## Five-axis fields",
                "- risk_intensity: 1 / 2 / 3 / 4 / 5 / not_evaluable",
                "- severity_axis, scope_axis, urgency_axis, novelty_axis, evolution_axis: 1 / 2 / 3 / 4 / 5 / not_evaluable",
                "- annotation_confidence: high / medium / low",
                "- annotation_basis: one-sentence rationale",
                "- evidence_ids: article-local evidence references",
                "- exclusion_reason: why a row is not evaluable",
                "- human_review_status: pass / review / abstain",
                "",
                "## Examples",
                "Direction example 1: a target wins orders or sales and the evidence shows direct operational improvement -> positive.",
                "Direction example 2: a target receives a penalty, investigation, or downgrade -> negative.",
                "Five-axis example 1: debt rollover failure with immediate repayment pressure -> high intensity and high urgency.",
                "Five-axis example 2: a broad policy benefit with limited single-name impact -> low intensity but larger scope.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# task4_impact_v2_annotation_round1",
                "",
                "Outputs:",
                "- `task4_direction_round1_A.xlsx` / `task4_direction_round1_B.xlsx`",
                "- `task4_impact_axes_round1_A.xlsx` / `task4_impact_axes_round1_B.xlsx`",
                "- `task4_direction_round1.parquet` / `task4_impact_axes_round1.parquet`",
                "- `input_manifest.json`, `non_test_data_inventory.json`, `leakage_report.json`, `sampling_report.json`",
                "- `annotation_guide.md`, `protected_sha256_before.json`, `protected_sha256_after.json`",
                "",
                "Rebuild:",
                "```bash",
                "cd /opt/data/private/JinRong/kechuang/financial_event_project",
                "conda run -n task4-impact-v3 python scripts/build_task4_impact_v2_annotation_round1.py",
                "```",
                "",
                "Notes:",
                "- Direction and impact workbooks use the same sample set inside each A/B pair, but different row order.",
                "- Human annotation columns are blank by construction.",
                "- Official test rows from `data/raw/news_with_tags_test.xlsx` are excluded by article id and content hash.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    wrapper = OUT / "rebuild_task4_impact_v2_annotation_round1.sh"
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /opt/data/private/JinRong/kechuang/financial_event_project",
                "conda run -n task4-impact-v3 python scripts/build_task4_impact_v2_annotation_round1.py",
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    print(
        json.dumps(
            {
                "status": "task4_impact_v2_annotation_round1_complete",
                "direction_rows": int(len(direction_pack_a)),
                "impact_rows": int(len(impact_pack_a)),
                "impact_subset_of_direction": bool(set(impact_pack_a["sample_id"]).issubset(set(direction_pack_a["sample_id"]))),
                "output_dir": str(OUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
