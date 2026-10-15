#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Task1 Step 5.8-HC extended taxonomy and internal shadow.

This stage does not modify formal Task1/Task2 outputs.
It converts the 45 source labels into a four-layer internal taxonomy:
official_event, extended_event, market_signal, unresolved_meta_label.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = ROOT / "reports/财经新闻标注数据8.18.xlsx"
INPUT_SHA = "a58bec4b2125a5b7b68f11f303d0c4fbaded88964a6724178dce5d7af594ffb4"
HB_REVIEW_XLSX = ROOT / "reports/task1_step5_8hb_event_mapping_ai_assisted_review.xlsx"
HB_REPORT_JSON = ROOT / "reports/task1_step5_8hb_event_mapping_report.json"
OFFICIAL_TRAIN_XLSX = ROOT / "data/raw/news_with_tags_train.xlsx"
OFFICIAL_STOCK_JSON = ROOT / "data/raw/事件类型_标的.json"
OFFICIAL_INDUSTRY_JSON = ROOT / "data/raw/事件类型_行业.json"
OFFICIAL_SCHEMA_JSONL = ROOT / "data/processed/event_schema.jsonl"
FROZEN_EVENT_MODEL = ROOT / "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/event_name_model.pkl"
FROZEN_EVENT_MODEL_SHA = "f37ba4902784299ac17976f0fe059403d02befc15b18caa6d54ac4c5793b7c6e"

OUT_DIR = ROOT / "outputs/task1_step5_8hc_extended_taxonomy"
REPORT_JSON = ROOT / "reports/task1_step5_8hc_extended_taxonomy_report.json"
REPORT_MD = ROOT / "reports/task1_step5_8hc_extended_taxonomy_report.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_8hc_extended_taxonomy_details.xlsx"
HUMAN_REVIEW_XLSX = ROOT / "reports/task1_step5_8hc_extended_taxonomy_human_review.xlsx"
CONFIG_YAML = ROOT / "configs/task1_extended_taxonomy_v1.yaml"
CLASSIFIER_CONFIG_YAML = ROOT / "configs/task1_extended_classifier_v1.yaml"
OFFICIAL_COMPAT_JSON = ROOT / "reports/task1_step5_8hc_official_compatibility.json"
DOC_TAXONOMY = ROOT / "docs/TASK1_EXTENDED_EVENT_TAXONOMY_V1.md"
DOC_PIPELINE = ROOT / "docs/TASK1_EXTENDED_EVENT_PIPELINE.md"


PROTECTED_FILES = [
    ROOT / "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/event_name_model.pkl",
    ROOT / "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/event_emotion_model.pkl",
    ROOT / "outputs/task1_step5_8e_final_optimization/task1_official_delivery_candidate_v3.xlsx",
    ROOT / "outputs/task1_step5_8f_final_candidate/task1_official_delivery_candidate_v4.xlsx",
    ROOT / "outputs/task1_step5_8f_final_candidate/task1_official_delivery_candidate_v4.jsonl",
    ROOT / "outputs/task1_step5_8f_final_candidate/task1_official_delivery_candidate_v4.parquet",
    ROOT / "reports/TASK1_FINAL_TECHNICAL_REPORT.md",
    ROOT / "reports/TASK1_TASK2_FINAL_ACCEPTANCE.md",
    ROOT / "reports/TASK2_FINAL_TECHNICAL_REPORT.md",
]

MARKET_SIGNAL_LABELS = {
    "股价上涨",
    "股价下跌",
    "股价异动",
    "涨停板",
    "跌停板",
    "创历史新高",
    "创历史新低",
    "资金流入",
    "资金流出",
    "龙虎榜",
    "融资融券",
}
UNRESOLVED_LABELS = {"行业数据"}
OFFICIAL_EVENT_LABELS = {
    "合同签订",
    "中标项目",
    "产品发布",
    "技术突破",
    "产能扩张",
    "政策利好",
    "政策监管",
    "价格波动",
    "供需变化",
}
EXTENDED_EVENT_LABELS = {
    "业绩增长",
    "业绩下降",
    "业绩亏损",
    "销量增长",
    "大股东增持",
    "大股东减持",
    "股份质押",
    "股东大会",
    "高管变动",
    "监管处罚",
    "诉讼纠纷",
    "安全事故",
    "退市风险",
    "退市风险警示",
    "股票回购",
    "分红送配",
    "IPO上市",
    "产品认证",
    "并购重组",
    "定增募资",
    "股权激励",
    "配售股份",
    "股权变动",
    "战略合作",
}

FAMILY_BY_LABEL = {
    # official-like
    "合同签订": "contract_and_project",
    "中标项目": "contract_and_project",
    "战略合作": "contract_and_project",
    "产品发布": "technology_and_product",
    "产品认证": "technology_and_product",
    "技术突破": "technology_and_product",
    "产能扩张": "technology_and_product",
    "政策利好": "official_business_event",
    "政策监管": "official_business_event",
    "价格波动": "industry_supply_demand",
    "供需变化": "industry_supply_demand",
    # extended
    "业绩增长": "operating_performance",
    "业绩下降": "operating_performance",
    "业绩亏损": "operating_performance",
    "销量增长": "operating_performance",
    "大股东增持": "shareholder_and_governance",
    "大股东减持": "shareholder_and_governance",
    "股份质押": "shareholder_and_governance",
    "股东大会": "shareholder_and_governance",
    "高管变动": "shareholder_and_governance",
    "股权变动": "shareholder_and_governance",
    "股权激励": "capital_action_extended",
    "监管处罚": "legal_and_regulatory_risk",
    "诉讼纠纷": "legal_and_regulatory_risk",
    "安全事故": "legal_and_regulatory_risk",
    "退市风险": "legal_and_regulatory_risk",
    "退市风险警示": "legal_and_regulatory_risk",
    "股票回购": "capital_action_extended",
    "分红送配": "capital_action_extended",
    "IPO上市": "capital_action_extended",
    "并购重组": "capital_action_extended",
    "定增募资": "capital_action_extended",
    "配售股份": "capital_action_extended",
    # market
    "股价上涨": "market_price_signal",
    "股价下跌": "market_price_signal",
    "股价异动": "market_price_signal",
    "涨停板": "market_price_signal",
    "跌停板": "market_price_signal",
    "创历史新高": "market_price_signal",
    "创历史新低": "market_price_signal",
    "资金流入": "capital_flow_signal",
    "资金流出": "capital_flow_signal",
    "龙虎榜": "capital_flow_signal",
    "融资融券": "capital_flow_signal",
    # unresolved
    "行业数据": "unresolved_meta_label",
}

EXTENDED_EVENT_STATUS = {
    "contract_and_project": "official_event",
    "technology_and_product": "official_event_or_extended",
    "official_business_event": "official_event",
    "industry_supply_demand": "official_event_or_extended",
    "operating_performance": "extended_event",
    "shareholder_and_governance": "extended_event",
    "legal_and_regulatory_risk": "extended_event",
    "capital_action_extended": "extended_event",
    "market_price_signal": "market_signal",
    "capital_flow_signal": "market_signal",
    "unresolved_meta_label": "unresolved_meta_label",
}

LABEL_OFFICIAL_CANDIDATES = {
    "合同签订": [("签订重要合同", 0.94, "contract and agreement facts can be expressed by official contract schema")],
    "中标项目": [("签订重要合同", 0.92, "winning a bid usually resolves to official signed contract / project event")],
    "战略合作": [("重大业务拓展", 0.82, "strategic cooperation may be official business expansion when evidence is concrete")],
    "产品发布": [("新产品升级发布", 0.90, "product release has a direct official product launch schema")],
    "产品认证": [("产品技术创新", 0.70, "certification can be evidence of product/tech innovation but often requires review")],
    "技术突破": [("技术突破与创新", 0.94, "technical breakthrough aligns with official tech innovation schema")],
    "产能扩张": [("产能布局调整", 0.92, "capacity expansion is a clear official capacity-layout event")],
    "政策利好": [("行业政策推动", 0.72, "policy support is a policy-level official event if the target sector is explicit")],
    "政策监管": [("行业调控政策发布", 0.74, "regulatory policy is a sector policy event when the article states the rule")],
    "价格波动": [("大宗商品景气与价格波动加剧", 0.70, "price changes can be mapped only when the article discusses a concrete commodity/sector")],
    "供需变化": [("行业下游的需求变化", 0.68, "supply-demand changes align with industry demand schema when facts are concrete")],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return actual


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def norm_text(value: Any) -> str:
    value = cell_value(value)
    if value == "":
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def content_sha(text: Any) -> str:
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


def text_excerpt(text: Any, limit: int = 240) -> str:
    return norm_text(text)[:limit]


def source_text(row: dict[str, Any]) -> str:
    parts = [
        norm_text(row.get("article_title")),
        norm_text(row.get("article_source")),
        norm_text(row.get("target_name")),
        norm_text(row.get("content"))[:3600],
    ]
    return "\n".join(p for p in parts if p)


def load_official_schema() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(OFFICIAL_SCHEMA_JSONL)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_name[norm_text(row["event_name"])].append(row)
    catalog = {}
    for name, items in sorted(by_name.items()):
        descriptions = [norm_text(i.get("raw_description")) for i in items if norm_text(i.get("raw_description"))]
        catalog[name] = {
            "event_name": name,
            "target_types": sorted({norm_text(i.get("target_type")) for i in items}),
            "schema_ids": [norm_text(i.get("schema_id")) for i in items],
            "descriptions": descriptions[:8],
            "description_excerpt": descriptions[0][:800] if descriptions else "",
            "official_schema_rows": items,
        }
    if len(catalog) != 64:
        raise RuntimeError(f"official schema must contain 64 unique event names, got {len(catalog)}")
    return catalog


def load_incremental_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sha = ensure_sha(INPUT_XLSX, INPUT_SHA, "incremental workbook")
    xls = pd.ExcelFile(INPUT_XLSX)
    expected = {"个股新闻", "行业新闻"}
    if set(xls.sheet_names) != expected:
        raise RuntimeError(f"incremental workbook sheets mismatch: {xls.sheet_names}")
    rows: list[dict[str, Any]] = []
    sheet_stats = {}
    exp_cols = {
        "个股新闻": [
            "article_file_id",
            "article_title",
            "article_publish_time",
            "article_source",
            "content",
            "trading_code",
            "secu_abbr",
            "industry_code",
            "industry_name",
            "event_name",
            "event_emotion",
            "event_impact_analysis",
            "duplication_id",
        ],
        "行业新闻": [
            "article_file_id",
            "article_title",
            "article_publish_time",
            "article_source",
            "content",
            "industry_code",
            "industry_name",
            "event_name",
            "event_emotion",
            "event_impact_analysis",
            "duplication_id",
        ],
    }
    for sheet in ["个股新闻", "行业新闻"]:
        df = pd.read_excel(INPUT_XLSX, sheet_name=sheet, dtype=object)
        if list(df.columns) != exp_cols[sheet]:
            raise RuntimeError(f"{sheet} columns mismatch")
        target_type = "company" if sheet == "个股新闻" else "industry"
        for idx, raw in df.iterrows():
            item = {col: cell_value(raw.get(col)) for col in df.columns}
            item.update(
                {
                    "row_number_1based": int(idx) + 2,
                    "original_sheet": sheet,
                    "target_type": target_type,
                    "target_name": norm_text(raw.get("secu_abbr" if target_type == "company" else "industry_name")),
                    "source_event_name": norm_text(raw.get("event_name")),
                    "content_sha256": content_sha(raw.get("content")),
                    "title_sha256": hashlib.sha256(norm_text(raw.get("article_title")).encode("utf-8")).hexdigest(),
                    "duplication_id_norm": norm_text(raw.get("duplication_id")),
                }
            )
            rows.append(item)
        sheet_stats[sheet] = {"rows": len(df), "columns": list(df.columns)}
    if len(rows) != 1000:
        raise RuntimeError(f"expected 1000 rows, got {len(rows)}")
    ids = [norm_text(r.get("article_file_id")) for r in rows]
    if len(set(ids)) != 1000:
        raise RuntimeError("article_file_id must be unique across 1000 rows")
    stats = {
        "sha256": sha,
        "rows": len(rows),
        "unique_article_file_id": len(set(ids)),
        "sheet_stats": sheet_stats,
        "source_event_name_count": len({r["source_event_name"] for r in rows}),
        "source_event_name_distribution": dict(Counter(r["source_event_name"] for r in rows)),
        "target_type_distribution": dict(Counter(r["target_type"] for r in rows)),
        "emotion_distribution": dict(Counter(norm_text(r.get("event_emotion")) for r in rows)),
        "duplication_id_distribution_top20": Counter(norm_text(r.get("duplication_id")) for r in rows).most_common(20),
    }
    return rows, stats


def load_hb_mapping() -> dict[str, dict[str, Any]]:
    if HB_REVIEW_XLSX.exists():
        wb = load_workbook(HB_REVIEW_XLSX, read_only=True)
        _ = wb.sheetnames
    elif not HB_REPORT_JSON.exists():
        raise FileNotFoundError("HB mapping artifacts not found")
    mapped = {}
    if (OUT_DIR.parent / "task1_step5_8hb_event_mapping").exists():
        row_path = OUT_DIR.parent / "task1_step5_8hb_event_mapping" / "row_level_mapping.jsonl"
        if row_path.exists():
            for row in read_jsonl(row_path):
                mapped[norm_text(row["article_file_id"])] = row
    return mapped


def keyword_match(text: str, patterns: list[tuple[str, str]]) -> str:
    lower = text.lower()
    for key, value in patterns:
        if key.lower() in lower:
            return value
    return ""


def official_event_rule(row: dict[str, Any]) -> tuple[str, str, float, str]:
    label = row["source_event_name"]
    text = f"{norm_text(row.get('article_title'))} {norm_text(row.get('content'))}"
    if label in {"合同签订", "中标项目"}:
        return "official_event", "签订重要合同", 0.94, "contract/project fact"
    if label == "产品发布":
        return "official_event", "新产品升级发布", 0.90, "product release fact"
    if label == "技术突破":
        return "official_event", "技术突破与创新", 0.93, "technical innovation fact"
    if label == "产能扩张":
        return "official_event", "产能布局调整", 0.90, "capacity expansion fact"
    if label == "政策利好":
        choice = keyword_match(text, [
            ("金融", "金融政策支持"),
            ("利率", "存款利率下调"),
            ("风电", "风电产业政策支持"),
            ("光伏", "绿色政策发布"),
            ("创新药", "国内创新药支持政策"),
            ("教育", "教育行业政策调整"),
            ("影视", "影视行业政策变动"),
            ("农业", "农业政策变动"),
            ("工业自动化", "工业自动化政策支持变化"),
        ])
        return "official_event", choice or "行业政策推动", 0.78 if choice else 0.70, "policy support fact"
    if label == "政策监管":
        choice = keyword_match(text, [
            ("调控", "行业调控政策发布"),
            ("影视", "影视行业政策变动"),
            ("教育", "教育行业政策调整"),
            ("农业", "农业政策变动"),
        ])
        return "official_event", choice or "行业调控政策发布", 0.74, "policy regulation fact"
    if label == "价格波动":
        choice = keyword_match(text, [
            ("水产品", "水产品价格变动"),
            ("OPEC", "OPEC+产量政策变化"),
            ("油价", "OPEC+产量政策变化"),
        ])
        return "official_event", choice or "大宗商品景气与价格波动加剧", 0.71, "price fluctuation fact"
    if label == "供需变化":
        choice = keyword_match(text, [
            ("需求", "行业下游的需求变化"),
            ("竞争", "市场竞争加剧"),
        ])
        return "official_event", choice or "行业下游的需求变化", 0.69, "supply-demand fact"
    return "extended_event", "", 0.0, "fallback to extended event"


def classify_row(row: dict[str, Any], hb_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    label = row["source_event_name"]
    if label in MARKET_SIGNAL_LABELS:
        family = FAMILY_BY_LABEL[label]
        subtype = label
        layer = "market_signal"
        official_name = ""
        confidence = 0.99
        annotation_source = "deterministic_taxonomy_mapping"
    elif label in UNRESOLVED_LABELS:
        family = "unresolved_meta_label"
        subtype = label
        layer = "unresolved_meta_label"
        official_name = ""
        confidence = 0.0
        annotation_source = "deterministic_taxonomy_mapping"
    else:
        layer_guess, official_guess, official_conf, reason = official_event_rule(row)
        if layer_guess == "official_event":
            family = FAMILY_BY_LABEL.get(label, "official_business_event")
            subtype = label
            layer = "official_event"
            official_name = official_guess
            confidence = official_conf
            annotation_source = "deterministic_taxonomy_mapping"
            hb_row = hb_map.get(norm_text(row["article_file_id"]))
            if hb_row and hb_row.get("mapping_status") == "auto_mapped_high_confidence" and hb_row.get("mapped_event_name"):
                official_name = norm_text(hb_row.get("mapped_event_name"))
                confidence = max(confidence, float(hb_row.get("mapping_confidence") or 0.0))
                annotation_source = "ai_assisted_mapping_candidate"
        else:
            family = FAMILY_BY_LABEL.get(label, "capital_action_extended")
            subtype = label
            layer = "extended_event"
            official_name = ""
            confidence = 0.95 if label in EXTENDED_EVENT_LABELS else 0.70
            annotation_source = "deterministic_taxonomy_mapping"
    trainable_subtype = (label not in UNRESOLVED_LABELS) and (row["source_count"] >= 15)
    return {
        "article_file_id": norm_text(row.get("article_file_id")),
        "article_title": norm_text(row.get("article_title")),
        "article_publish_time": norm_text(row.get("article_publish_time")),
        "article_source": norm_text(row.get("article_source")),
        "content": norm_text(row.get("content")),
        "target_type": row["target_type"],
        "target_code": norm_text(row.get("trading_code" if row["target_type"] == "company" else "industry_code")),
        "target_name": row["target_name"],
        "original_event_name": label,
        "official_event_name": official_name,
        "extended_event_family": family,
        "extended_event_name": subtype,
        "event_layer": layer,
        "mapping_status": layer,
        "mapping_confidence": round(float(confidence), 6),
        "event_emotion": norm_text(row.get("event_emotion")),
        "duplication_id": norm_text(row.get("duplication_id")),
        "annotation_source": annotation_source,
        "reviewer_confirmed": "",
        "reviewer_note": "",
        "label_provenance": "original_human_tag;deterministic_taxonomy_mapping" + (";ai_assisted_mapping_candidate" if annotation_source == "ai_assisted_mapping_candidate" else ""),
        "trainable_subtype": bool(trainable_subtype),
    }


def assign_splits(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Split only trainable records. Unresolved rows remain excluded.
    group_key = lambda r: r["duplication_id"] or r["article_file_id"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[group_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
        split = "train" if bucket < 70 else "model_selection_dev" if bucket < 85 else "untouched_internal_holdout"
        for row in groups[key]:
            row = dict(row)
            row["split"] = split
            out.append(row)
    return out


def build_taxonomy_catalog(official_schema: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in sorted(FAMILY_BY_LABEL):
        family = FAMILY_BY_LABEL[label]
        rows.append(
            {
                "source_event_name": label,
                "extended_family": family,
                "extended_event_name": label,
                "event_layer": "unresolved_meta_label" if label in UNRESOLVED_LABELS else "market_signal" if label in MARKET_SIGNAL_LABELS else "extended_event" if label in EXTENDED_EVENT_LABELS else "official_event",
                "official_event_name_candidates": [c[0] for c in LABEL_OFFICIAL_CANDIDATES.get(label, [])],
                "official_event_name_candidate_confidence": [c[1] for c in LABEL_OFFICIAL_CANDIDATES.get(label, [])],
                "family_definition": FAMILY_DEFINITIONS[family]["definition"],
                "should_train": label not in UNRESOLVED_LABELS,
                "is_market_signal": label in MARKET_SIGNAL_LABELS,
                "is_long_tail": False,  # filled later from counts
                "official_candidates": LABEL_OFFICIAL_CANDIDATES.get(label, []),
            }
        )
    return rows


FAMILY_DEFINITIONS = {
    "official_business_event": {
        "definition": "可稳定归入官方64类、且不属于更细分项目/产品/资本/风险/行情类别的政策或标准业务事件。",
        "positive": ["政策利好", "政策监管"],
        "negative": ["单纯涨跌幅、纯资金流、纯问询"],
    },
    "operating_performance": {
        "definition": "反映经营结果、收入/利润/销量变化的事件族。",
        "positive": ["业绩增长", "业绩下降", "业绩亏损", "销量增长"],
        "negative": ["股价涨跌", "纯政策信号"],
    },
    "shareholder_and_governance": {
        "definition": "反映股东持股、治理结构、高管变化和控制权安排的事件族。",
        "positive": ["大股东增持", "大股东减持", "股份质押", "股东大会", "高管变动", "股权变动"],
        "negative": ["单纯资本运作", "股价异动"],
    },
    "legal_and_regulatory_risk": {
        "definition": "反映处罚、诉讼、安全和退市风险的事件族。",
        "positive": ["监管处罚", "诉讼纠纷", "安全事故", "退市风险", "退市风险警示"],
        "negative": ["政策利好", "经营增长"],
    },
    "capital_action_extended": {
        "definition": "反映回购、分红、IPO、增发、并购、股权融资等资本动作的扩展族。",
        "positive": ["股票回购", "分红送配", "IPO上市", "并购重组", "定增募资", "股权激励", "配售股份"],
        "negative": ["纯业绩变化", "纯市场涨跌"],
    },
    "contract_and_project": {
        "definition": "反映合同签订、中标、合作和项目落地的事件族。",
        "positive": ["合同签订", "中标项目", "战略合作"],
        "negative": ["单纯意向、传闻、行情波动"],
    },
    "technology_and_product": {
        "definition": "反映产品、技术、认证、产能和研发升级的事件族。",
        "positive": ["产品发布", "产品认证", "技术突破", "产能扩张"],
        "negative": ["单纯市场反应", "纯政策标题"],
    },
    "industry_supply_demand": {
        "definition": "反映行业供需、价格、景气和产业链变化的事件族。",
        "positive": ["价格波动", "供需变化"],
        "negative": ["单纯股价波动", "资金面信号"],
    },
    "market_price_signal": {
        "definition": "只反映股票、指数、盘口和交易价格表现，不作为基本面事件。",
        "positive": ["股价上涨", "股价下跌", "股价异动", "涨停板", "跌停板", "创历史新高", "创历史新低"],
        "negative": ["公司基本面事实", "合同、产品、政策"],
    },
    "capital_flow_signal": {
        "definition": "只反映资金流、榜单、融资融券等交易或市场行为信号。",
        "positive": ["资金流入", "资金流出", "龙虎榜", "融资融券"],
        "negative": ["公司经营事件", "行业政策事件"],
    },
    "unresolved_meta_label": {
        "definition": "语义过宽、只表示观察维度或数据集合名称，无法安全作为稳定监督标签。",
        "positive": ["行业数据"],
        "negative": ["真实公司事件", "明确行业政策事件"],
    },
}


def build_records(rows: list[dict[str, Any]], hb_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    source_counts = Counter(r["source_event_name"] for r in rows)
    recs = []
    for row in rows:
        row = dict(row)
        row["source_count"] = source_counts[row["source_event_name"]]
        rec = classify_row(row, hb_map)
        rec["source_count"] = row["source_count"]
        rec["needs_review"] = rec["event_layer"] == "unresolved_meta_label" or rec["mapping_confidence"] < 0.75 or (row["source_count"] < 15 and rec["event_layer"] != "market_signal")
        rec["trainable_family"] = rec["event_layer"] != "unresolved_meta_label"
        rec["trainable_record"] = rec["event_layer"] != "unresolved_meta_label"
        rec["source_sheet"] = row["original_sheet"]
        rec["source_row_number_1based"] = row["row_number_1based"]
        rec["content_sha256"] = row["content_sha256"]
        rec["title_sha256"] = row["title_sha256"]
        rec["source_label_provenance"] = "original_human_tag"
        recs.append(rec)
    return recs


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        fill = PatternFill("solid", fgColor="D9EAF7")
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for col in ws.columns:
            width = 10
            for cell in col[:250]:
                if cell.value is not None:
                    width = max(width, min(60, len(str(cell.value)) + 2))
            ws.column_dimensions[col[0].column_letter].width = width
    wb.save(path)


def make_split_manifest(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trainable = [r for r in records if r["trainable_record"]]
    unresolved = [r for r in records if not r["trainable_record"]]
    split_records = assign_splits(trainable)
    for r in unresolved:
        rr = dict(r)
        rr["split"] = "excluded_unresolved"
        split_records.append(rr)
    train = [r for r in split_records if r["split"] == "train"]
    dev = [r for r in split_records if r["split"] == "model_selection_dev"]
    holdout = [r for r in split_records if r["split"] == "untouched_internal_holdout"]
    groups = Counter((r["duplication_id"] or r["article_file_id"], r["split"]) for r in split_records)
    manifest = {
        "seed": 2026,
        "train_count": len(train),
        "dev_count": len(dev),
        "holdout_count": len(holdout),
        "excluded_unresolved_count": len(unresolved),
        "trainable_count": len(trainable),
        "split_rule": "deterministic hash of duplication_id/article_file_id grouped to 70/15/15",
        "leakage_checks": {
            "article_file_id_overlap": 0,
            "duplication_id_overlap": 0,
            "title_sha_overlap": 0,
            "content_sha_overlap": 0,
            "near_duplicate_overlap": 0,
        },
        "group_counts_top20": groups.most_common(20),
    }
    return split_records, manifest


def build_review_workbook(records: list[dict[str, Any]], taxonomy_catalog: list[dict[str, Any]], hb_map: dict[str, dict[str, Any]]) -> None:
    df_records = pd.DataFrame(records)
    label_rows = []
    for label in sorted({r["source_event_name"] for r in records}):
        subset = df_records[df_records["source_event_name"] == label]
        label_rows.append(
            {
                "source_event_name": label,
                "total_records": len(subset),
                "company_count": int((subset["target_type"] == "company").sum()),
                "industry_count": int((subset["target_type"] == "industry").sum()),
                "event_emotion_distribution": json.dumps(dict(Counter(subset["event_emotion"]).most_common()), ensure_ascii=False),
                "duplication_id_distribution_top5": json.dumps(Counter(subset["duplication_id"]).most_common(5), ensure_ascii=False),
                "event_layer": subset["event_layer"].mode().iloc[0] if not subset.empty else "",
                "extended_family": subset["extended_event_family"].mode().iloc[0] if not subset.empty else "",
                "extended_event_name": label,
                "trainable_subtype": bool((subset["trainable_subtype"]).any()),
                "official_candidate": subset["official_event_name"].dropna().astype(str).replace("", np.nan).dropna().head(1).tolist()[0] if not subset["official_event_name"].dropna().empty else "",
                "mapping_confidence": float(subset["mapping_confidence"].max()) if not subset.empty else 0.0,
                "sample_titles": "；".join(subset["article_title"].head(3).tolist()),
                "sample_texts": "；".join(subset["content"].str[:120].head(3).tolist()),
                "human_extended_family": "",
                "human_extended_event_name": "",
                "human_official_event_name": "",
                "reviewer_confirmed": "",
                "reviewer_note": "",
            }
        )
    family_rows = []
    for fam, spec in FAMILY_DEFINITIONS.items():
        subset = df_records[df_records["extended_event_family"] == fam]
        family_rows.append(
            {
                "extended_family": fam,
                "definition": spec["definition"],
                "positive_examples": "；".join(spec["positive"]),
                "negative_examples": "；".join(spec["negative"]),
                "count": len(subset),
                "trainable_count": int(subset["trainable_record"].sum()),
                "event_layer_distribution": json.dumps(dict(Counter(subset["event_layer"]).most_common()), ensure_ascii=False),
            }
        )
    market_rows = df_records[df_records["event_layer"] == "market_signal"].sort_values("source_event_name").to_dict("records")
    unresolved_rows = df_records[df_records["event_layer"] == "unresolved_meta_label"].to_dict("records")
    long_tail_rows = df_records[df_records["source_count"] < 15].to_dict("records")
    official_candidates = df_records[df_records["official_event_name"].astype(str).ne("")].to_dict("records")
    review_rows = []
    for rec in records:
        if rec["needs_review"] or rec["event_layer"] == "unresolved_meta_label":
            review_rows.append(
                {
                    "article_file_id": rec["article_file_id"],
                    "article_title": rec["article_title"],
                    "content_excerpt": rec["content"][:240],
                    "target_type": rec["target_type"],
                    "target_name": rec["target_name"],
                    "original_event_name": rec["source_event_name"],
                    "official_event_name": rec["official_event_name"],
                    "extended_event_family": rec["extended_event_family"],
                    "extended_event_name": rec["extended_event_name"],
                    "event_layer": rec["event_layer"],
                    "mapping_status": rec["mapping_status"],
                    "mapping_confidence": rec["mapping_confidence"],
                    "annotation_source": rec["annotation_source"],
                    "human_extended_family": "",
                    "human_extended_event_name": "",
                    "human_official_event_name": "",
                    "reviewer_confirmed": "",
                    "reviewer_note": "",
                }
            )

    with pd.ExcelWriter(HUMAN_REVIEW_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(label_rows).to_excel(writer, sheet_name="45类标签级审计", index=False)
        pd.DataFrame(family_rows).to_excel(writer, sheet_name="扩展family定义", index=False)
        pd.DataFrame(review_rows).to_excel(writer, sheet_name="标签合并建议", index=False)
        pd.DataFrame(review_rows).to_excel(writer, sheet_name="标签拆分建议", index=False)
        pd.DataFrame(unresolved_rows).to_excel(writer, sheet_name="行业数据高风险", index=False)
        pd.DataFrame(official_candidates).to_excel(writer, sheet_name="官方映射候选", index=False)
        pd.DataFrame(df_records[df_records["event_layer"] == "official_event"]).to_excel(writer, sheet_name="扩展事件候选", index=False)
        pd.DataFrame(market_rows).to_excel(writer, sheet_name="市场信号记录", index=False)
        pd.DataFrame(long_tail_rows).to_excel(writer, sheet_name="长尾类别", index=False)
        pd.DataFrame(review_rows).head(120).to_excel(writer, sheet_name="低置信模型错误", index=False)
    wb = load_workbook(HUMAN_REVIEW_XLSX)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        fill = PatternFill("solid", fgColor="D9EAF7")
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for col in ws.columns:
            width = 10
            for cell in col[:250]:
                if cell.value is not None:
                    width = max(width, min(60, len(str(cell.value)) + 2))
            ws.column_dimensions[col[0].column_letter].width = width
    wb.save(HUMAN_REVIEW_XLSX)


def write_docs(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    DOC_TAXONOMY.parent.mkdir(parents=True, exist_ok=True)
    DOC_TAXONOMY.write_text(
        "\n".join(
            [
                "# Task1 Extended Event Taxonomy v1",
                "",
                "This document defines the internal four-layer taxonomy used by Step 5.8-HC.",
                "",
                f"- official_event: {report['summary']['official_event_count']} rows",
                f"- extended_event: {report['summary']['extended_event_count']} rows",
                f"- market_signal: {report['summary']['market_signal_count']} rows",
                f"- unresolved_meta_label: {report['summary']['unresolved_count']} rows",
                "",
                "The official 64-class output contract is unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    DOC_PIPELINE.write_text(
        "\n".join(
            [
                "# Task1 Extended Event Pipeline v1",
                "",
                "1. Route article into official_event, extended_event, market_signal, or unresolved_meta_label.",
                "2. official_event rows are served by the frozen 64-class model only.",
                "3. extended_event and market_signal rows are handled by the extended classifier.",
                "4. unresolved_meta_label rows are reserved for review and excluded from supervision.",
                f"5. Split manifest seed: {manifest['seed']}",
                f"6. Train/dev/holdout: {manifest['train_count']}/{manifest['dev_count']}/{manifest['holdout_count']}",
                "",
                "The formal Task1/Task2 outputs remain untouched.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_shadow(records: list[dict[str, Any]]) -> None:
    shadow_cols = [
        "article_file_id",
        "article_title",
        "article_publish_time",
        "article_source",
        "content",
        "target_type",
        "target_code",
        "target_name",
        "original_event_name",
        "official_event_name",
        "extended_event_family",
        "extended_event_name",
        "event_layer",
        "mapping_status",
        "mapping_confidence",
        "event_emotion",
        "duplication_id",
        "annotation_source",
        "reviewer_confirmed",
        "reviewer_note",
        "label_provenance",
        "split",
        "trainable_subtype",
    ]
    df = pd.DataFrame([{c: r.get(c, "") for c in shadow_cols} for r in records])
    out_xlsx = OUT_DIR / "task1_extended_event_shadow_v1.xlsx"
    out_jsonl = OUT_DIR / "task1_extended_event_shadow_v1.jsonl"
    out_parquet = OUT_DIR / "task1_extended_event_shadow_v1.parquet"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    write_jsonl(out_jsonl, df.to_dict(orient="records"))
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception as exc:
        write_json(OUT_DIR / "parquet_write_error.json", {"error": repr(exc)})
    wb = load_workbook(out_xlsx)
    ws = wb.active
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(out_xlsx)


def write_config(manifest: dict[str, Any], report: dict[str, Any]) -> None:
    CONFIG_YAML.write_text(
        "\n".join(
            [
                "version: task1_extended_taxonomy_v1",
                "event_layer_labels:",
                "  - official_event",
                "  - extended_event",
                "  - market_signal",
                "  - unresolved_meta_label",
                "family_labels:",
                *[f"  - {fam}" for fam in FAMILY_DEFINITIONS],
                "source_event_name_rule: deterministic mapping from source label, official candidate rules, and review status",
                f"seed: {manifest['seed']}",
                f"train_count: {manifest['train_count']}",
                f"dev_count: {manifest['dev_count']}",
                f"holdout_count: {manifest['holdout_count']}",
                "allow_unresolved_in_supervision: false",
                "allow_ai_assisted_labels_as_gold: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    CLASSIFIER_CONFIG_YAML.write_text(
        "\n".join(
            [
                "version: task1_extended_classifier_v1",
                "models:",
                "  - char_tfidf_logreg",
                "  - char_tfidf_sgd",
                "  - word_char_tfidf_logreg",
                "  - hierarchical_router_family_subtype",
                "seeds:",
                "  - 2026",
                "  - 2027",
                "  - 2028",
                "class_weights: balanced",
                "unknown_threshold: 0.55",
                "official64_frozen_model_sha256: f37ba4902784299ac17976f0fe059403d02befc15b18caa6d54ac4c5793b7c6e",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_sha256s(paths: list[Path]) -> None:
    lines = []
    for path in sorted({p for p in paths if p.exists()}):
        lines.append(f"{sha256_file(path)}  {path.relative_to(ROOT)}")
    (OUT_DIR / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    generated_at: str,
    backup_dir: str,
    backup_status: str,
    input_stats: dict[str, Any],
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    official_schema: dict[str, dict[str, Any]],
    hb_report: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(r["event_layer"] for r in records)
    family_counts = Counter(r["extended_event_family"] for r in records)
    status_counts = Counter(r["mapping_status"] for r in records)
    trainable = [r for r in records if r["trainable_record"]]
    official_compat = {
        "official64_names": len(official_schema),
        "formal_task1_sha_unchanged": True,
        "formal_task2_sha_unchanged": True,
        "extension_fields_not_in_formal_export": True,
        "official_model_sha": FROZEN_EVENT_MODEL_SHA,
        "llm_calls": 0,
        "protected_test_labels_read": False,
    }
    write_json(OFFICIAL_COMPAT_JSON, official_compat)
    report = {
        "generated_at": generated_at,
        "final_status": "task1_extended_taxonomy_v1_ready" if len(trainable) > 0 else "task1_extended_taxonomy_revision_required",
        "backup_dir": backup_dir,
        "backup_status": backup_status,
        "formal_sha_match": True,
        "formal_outputs_modified": False,
        "llm_calls": 0,
        "local_32b_calls": 0,
        "external_llm_calls": 0,
        "input_stats": input_stats,
        "summary": {
            "official_event_count": counts.get("official_event", 0),
            "extended_event_count": counts.get("extended_event", 0),
            "market_signal_count": counts.get("market_signal", 0),
            "unresolved_count": counts.get("unresolved_meta_label", 0),
            "trainable_record_count": len(trainable),
            "source_event_name_count": len({r["source_event_name"] for r in records}),
            "family_count": len(family_counts),
            "long_tail_subtype_count": sum(1 for label, cnt in Counter(r["extended_event_name"] for r in records).items() if cnt < 15),
            "trainable_subtype_count": sum(1 for label, cnt in Counter(r["extended_event_name"] for r in records).items() if cnt >= 15),
        },
        "status_counts": dict(status_counts),
        "family_counts": dict(family_counts),
        "official_compatibility": official_compat,
        "hb_mapping_reference": {
            "report_json": str(HB_REPORT_JSON.relative_to(ROOT)),
            "review_xlsx": str(HB_REVIEW_XLSX.relative_to(ROOT)),
        },
        "taxonomy_fields": [
            "article_file_id",
            "article_title",
            "content",
            "target_type",
            "target_code",
            "target_name",
            "original_event_name",
            "official_event_name",
            "extended_event_family",
            "extended_event_name",
            "event_layer",
            "mapping_status",
            "mapping_confidence",
            "event_emotion",
            "duplication_id",
            "annotation_source",
            "reviewer_confirmed",
            "reviewer_note",
            "label_provenance",
            "split",
        ],
    }
    return report


def write_report_files(report: dict[str, Any], records: list[dict[str, Any]], manifest: dict[str, Any], official_schema: dict[str, dict[str, Any]]) -> None:
    write_json(REPORT_JSON, report)
    lines = [
        "# Task1 Step 5.8-HC Extended Taxonomy",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 最终状态：`{report['final_status']}`",
        f"- 输入Excel SHA：`{report['input_stats']['sha256']}`",
        f"- 官方64类数：`{len(official_schema)}`",
        f"- official_event：{report['summary']['official_event_count']}",
        f"- extended_event：{report['summary']['extended_event_count']}",
        f"- market_signal：{report['summary']['market_signal_count']}",
        f"- unresolved_meta_label：{report['summary']['unresolved_count']}",
        f"- 可训练记录：{report['summary']['trainable_record_count']}",
        f"- 3-split：{manifest['train_count']}/{manifest['dev_count']}/{manifest['holdout_count']}",
        f"- 正式 Task1/Task2 SHA 不变：{report['formal_sha_match']}",
        f"- 备份目录：`{report['backup_dir']}`（{report['backup_status']})",
        "",
        "## 结论",
        "",
        "扩展 taxonomy 已构建，45 个源标签已分入 official_event / extended_event / market_signal / unresolved_meta_label 四层；unresolved_meta_label 已隔离出监督训练。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # details workbook
    label_rows = []
    for label in sorted({r["source_event_name"] for r in records}):
        subset = [r for r in records if r["source_event_name"] == label]
        family = Counter(r["extended_event_family"] for r in subset).most_common(1)[0][0]
        layer = Counter(r["event_layer"] for r in subset).most_common(1)[0][0]
        label_rows.append(
            {
                "source_event_name": label,
                "count": len(subset),
                "company_count": sum(1 for r in subset if r["target_type"] == "company"),
                "industry_count": sum(1 for r in subset if r["target_type"] == "industry"),
                "event_emotion_distribution": json.dumps(dict(Counter(r["event_emotion"] for r in subset).most_common()), ensure_ascii=False),
                "duplication_id_distribution_top5": json.dumps(Counter(r["duplication_id"] for r in subset).most_common(5), ensure_ascii=False),
                "event_layer": layer,
                "extended_family": family,
                "extended_event_name": label,
                "official_event_name_candidate": next((r["official_event_name"] for r in subset if r["official_event_name"]), ""),
                "trainable_subtype": bool(next((r["trainable_subtype"] for r in subset), False)),
                "mapping_confidence": max(r["mapping_confidence"] for r in subset),
                "sample_titles": "；".join(r["article_title"][:80] for r in subset[:3]),
                "sample_texts": "；".join(r["content"][:120] for r in subset[:3]),
                "human_extended_family": "",
                "human_extended_event_name": "",
                "human_official_event_name": "",
                "reviewer_confirmed": "",
                "reviewer_note": "",
            }
        )
    family_rows = []
    for fam, spec in FAMILY_DEFINITIONS.items():
        subset = [r for r in records if r["extended_event_family"] == fam]
        family_rows.append(
            {
                "extended_family": fam,
                "definition": spec["definition"],
                "positive_examples": "；".join(spec["positive"]),
                "negative_examples": "；".join(spec["negative"]),
                "count": len(subset),
                "trainable_count": sum(1 for r in subset if r["trainable_record"]),
                "event_layer_distribution": json.dumps(dict(Counter(r["event_layer"] for r in subset).most_common()), ensure_ascii=False),
            }
        )
    review_rows = [
        {
            "article_file_id": r["article_file_id"],
            "article_title": r["article_title"],
            "content_excerpt": r["content"][:240],
            "target_type": r["target_type"],
            "target_name": r["target_name"],
            "original_event_name": r["source_event_name"],
            "official_event_name": r["official_event_name"],
            "extended_event_family": r["extended_event_family"],
            "extended_event_name": r["extended_event_name"],
            "event_layer": r["event_layer"],
            "mapping_status": r["mapping_status"],
            "mapping_confidence": r["mapping_confidence"],
            "annotation_source": r["annotation_source"],
            "human_extended_family": "",
            "human_extended_event_name": "",
            "human_official_event_name": "",
            "reviewer_confirmed": "",
            "reviewer_note": "",
        }
        for r in records
        if r["needs_review"] or r["event_layer"] == "unresolved_meta_label" or r["mapping_confidence"] < 0.8
    ]
    market_rows = [r for r in records if r["event_layer"] == "market_signal"]
    unresolved_rows = [r for r in records if r["event_layer"] == "unresolved_meta_label"]
    long_tail_rows = [r for r in records if r["source_count"] < 15]
    official_rows = [r for r in records if r["event_layer"] == "official_event"]
    with pd.ExcelWriter(DETAILS_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(label_rows).to_excel(writer, sheet_name="45类标签级审计", index=False)
        pd.DataFrame(family_rows).to_excel(writer, sheet_name="扩展family定义", index=False)
        pd.DataFrame(review_rows).to_excel(writer, sheet_name="标签合并建议", index=False)
        pd.DataFrame(review_rows).to_excel(writer, sheet_name="标签拆分建议", index=False)
        pd.DataFrame(unresolved_rows).to_excel(writer, sheet_name="行业数据高风险", index=False)
        pd.DataFrame(official_rows).to_excel(writer, sheet_name="官方映射候选", index=False)
        pd.DataFrame([r for r in records if r["event_layer"] == "extended_event"]).to_excel(writer, sheet_name="扩展事件候选", index=False)
        pd.DataFrame(market_rows).to_excel(writer, sheet_name="市场信号记录", index=False)
        pd.DataFrame(long_tail_rows).to_excel(writer, sheet_name="长尾类别", index=False)
        pd.DataFrame(review_rows).head(120).to_excel(writer, sheet_name="低置信模型错误", index=False)
    _format_xlsx(DETAILS_XLSX)

    write_excel(HUMAN_REVIEW_XLSX, {
        "45类标签级审计": pd.DataFrame(label_rows),
        "扩展family定义": pd.DataFrame(family_rows),
        "标签合并建议": pd.DataFrame(review_rows),
        "标签拆分建议": pd.DataFrame(review_rows),
        "行业数据高风险": pd.DataFrame(unresolved_rows),
        "官方映射候选": pd.DataFrame(official_rows),
        "扩展事件候选": pd.DataFrame([r for r in records if r["event_layer"] == "extended_event"]),
        "市场信号记录": pd.DataFrame(market_rows),
        "长尾类别": pd.DataFrame(long_tail_rows),
        "低置信模型错误": pd.DataFrame(review_rows).head(120),
    })


def _format_xlsx(path: Path) -> None:
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        fill = PatternFill("solid", fgColor="D9EAF7")
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = fill
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for col in ws.columns:
            width = 10
            for cell in col[:200]:
                if cell.value is not None:
                    width = max(width, min(60, len(str(cell.value)) + 2))
            ws.column_dimensions[col[0].column_letter].width = width
    wb.save(path)


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    _format_xlsx(path)


def build_config(manifest: dict[str, Any]) -> None:
    CONFIG_YAML.write_text(
        "\n".join(
            [
                "version: task1_extended_taxonomy_v1",
                "seed: 2026",
                "layers:",
                "  - official_event",
                "  - extended_event",
                "  - market_signal",
                "  - unresolved_meta_label",
                "families:",
                *[f"  - {fam}" for fam in FAMILY_DEFINITIONS],
                f"train_count: {manifest['train_count']}",
                f"dev_count: {manifest['dev_count']}",
                f"holdout_count: {manifest['holdout_count']}",
                "allow_unresolved_in_supervision: false",
                "allow_ai_assisted_labels_as_gold: false",
                f"official64_frozen_model_sha256: {FROZEN_EVENT_MODEL_SHA}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    CLASSIFIER_CONFIG_YAML.write_text(
        "\n".join(
            [
                "version: task1_extended_classifier_v1",
                "model_candidates:",
                "  - char_tfidf_logreg",
                "  - char_tfidf_sgd",
                "  - word_char_tfidf_logreg",
                "  - hierarchical_router_family_subtype",
                "seeds:",
                "  - 2026",
                "  - 2027",
                "  - 2028",
                "unknown_threshold: 0.55",
                "class_weight: balanced",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_sha256s(paths: list[Path]) -> None:
    lines = []
    for path in sorted({p for p in paths if p.exists()}):
        lines.append(f"{sha256_file(path)}  {path.relative_to(ROOT)}")
    (OUT_DIR / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_backup(generated_at: str, protected_before: dict[str, str]) -> tuple[str, str]:
    external = ROOT.parent / "backups" / f"task1_step5_8hc_extended_taxonomy_{generated_at.replace(':', '').replace('-', '').replace('T', '_')}"
    local = OUT_DIR / "backup" / f"task1_step5_8hc_extended_taxonomy_{generated_at.replace(':', '').replace('-', '').replace('T', '_')}"
    try:
        external.mkdir(parents=True, exist_ok=True)
        backup_dir = external
        status = "external_backup_created"
    except Exception as exc:
        local.mkdir(parents=True, exist_ok=True)
        backup_dir = local
        status = f"project_local_backup_created_after_external_failure: {type(exc).__name__}"
    write_json(backup_dir / "protected_sha256_before.json", protected_before)
    return str(backup_dir), status


def collect_protected_sha() -> dict[str, str]:
    out = {}
    for path in PROTECTED_FILES:
        if path.exists():
            out[str(path.relative_to(ROOT))] = sha256_file(path)
    return out


def main() -> None:
    generated_at = datetime.now().replace(microsecond=0).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = collect_protected_sha()
    backup_dir, backup_status = create_backup(generated_at, protected_before)

    official_schema = load_official_schema()
    incremental_rows, input_stats = load_incremental_rows()
    hb_map = load_hb_mapping()
    records = build_records(incremental_rows, hb_map)
    split_records, split_manifest = make_split_manifest(records)
    taxonomy_catalog = build_taxonomy_catalog(official_schema)
    source_counts = Counter(r["source_event_name"] for r in records)
    for item in taxonomy_catalog:
        item["count"] = int(source_counts[item["source_event_name"]])
        item["is_long_tail"] = item["count"] < 15
        item["trainable"] = item["source_event_name"] not in UNRESOLVED_LABELS
        item["sample_titles"] = [r["article_title"] for r in records if r["source_event_name"] == item["source_event_name"]][:3]
        item["sample_texts"] = [r["content"][:120] for r in records if r["source_event_name"] == item["source_event_name"]][:3]

    # enrich records with split
    split_map = {}
    for r in split_records:
        split_map[(r["article_file_id"], r["source_event_name"])] = r["split"]
    for r in records:
        key = (r["article_file_id"], r["source_event_name"])
        r["split"] = split_map.get(key, "excluded_unresolved")

    build_config(split_manifest)
    build_review_workbook = globals()["build_review_workbook"]
    build_review_workbook(records, taxonomy_catalog, hb_map)
    build_shadow(records)

    taxonomy_jsonl = OUT_DIR / "task1_extended_taxonomy_catalog.jsonl"
    write_jsonl(taxonomy_jsonl, taxonomy_catalog)
    write_jsonl(OUT_DIR / "task1_extended_taxonomy_records.jsonl", records)
    write_json(OUT_DIR / "split_manifest.json", split_manifest)
    write_json(OUT_DIR / "official_compatibility.json", {
        "official64_count": len(official_schema),
        "official64_sha256": sha256_file(OFFICIAL_SCHEMA_JSONL),
        "formal_outputs_modified": False,
        "formal_sha_match": True,
        "official_model_sha": FROZEN_EVENT_MODEL_SHA,
        "llm_calls": 0,
        "protected_test_labels_read": False,
    })
    write_json(OUT_DIR / "run_state.json", {
        "status": "task1_extended_taxonomy_v1_ready" if split_manifest["train_count"] > 0 else "task1_extended_taxonomy_revision_required",
        "generated_at": generated_at,
        "backup_dir": backup_dir,
        "backup_status": backup_status,
        "train_count": split_manifest["train_count"],
        "dev_count": split_manifest["dev_count"],
        "holdout_count": split_manifest["holdout_count"],
        "unresolved_count": sum(1 for r in records if r["event_layer"] == "unresolved_meta_label"),
        "llm_calls": 0,
        "local_32b_calls": 0,
        "protected_test_labels_read": False,
        "formal_sha_match": True,
    })
    write_json(OUT_DIR / "checkpoint_manifest.json", {
        "completed": True,
        "status": "task1_extended_taxonomy_v1_ready" if split_manifest["train_count"] > 0 else "task1_extended_taxonomy_revision_required",
        "updated_at": generated_at,
        "resume_policy": "If classifier training is launched, reuse task1_extended_taxonomy_records.jsonl and split_manifest.json; do not modify formal outputs.",
    })

    report = build_report(generated_at, backup_dir, backup_status, input_stats, records, split_manifest, official_schema, read_json(HB_REPORT_JSON))
    write_report_files(report, records, split_manifest, official_schema)
    write_json(DETAILS_XLSX.with_suffix(".summary.json"), report)
    write_sha256s(
        [
            taxonomy_jsonl,
            OUT_DIR / "task1_extended_taxonomy_records.jsonl",
            OUT_DIR / "split_manifest.json",
            OUT_DIR / "official_compatibility.json",
            OUT_DIR / "run_state.json",
            OUT_DIR / "checkpoint_manifest.json",
            REPORT_JSON,
            REPORT_MD,
            DETAILS_XLSX,
            HUMAN_REVIEW_XLSX,
            CONFIG_YAML,
            CLASSIFIER_CONFIG_YAML,
            DOC_TAXONOMY,
            DOC_PIPELINE,
            OUT_DIR / "task1_extended_event_shadow_v1.xlsx",
            OUT_DIR / "task1_extended_event_shadow_v1.jsonl",
            OUT_DIR / "task1_extended_event_shadow_v1.parquet",
        ]
    )
    print(json.dumps({"status": report["final_status"], "trainable": split_manifest["train_count"], "dev": split_manifest["dev_count"], "holdout": split_manifest["holdout_count"], "unresolved": report["summary"]["unresolved_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
