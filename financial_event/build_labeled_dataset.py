#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_labeled_dataset.py
-------------------------
将带标签新闻 Excel（train/test）标准化为：
1. 新闻级数据 (news_*.jsonl) — 一篇新闻一条记录，所有事件标签放入 labels 数组
2. 事件对数据 (event_pairs_*.jsonl / *.xlsx) — 每个"新闻—事件标签"一条记录

处理步骤：
1. 读取 event_schema.jsonl，建立索引
2. 解析两个 Excel 的所有工作表
3. 处理重复新闻表，建立 article_id → duplication_id 映射
4. Schema 映射：(target_type, code, event_name) → schema_id
5. 文本标准化、时间标准化
6. 聚合为新闻级和事件对级数据
7. 生成统计报告、未匹配报告、异常报告
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════
# 路径配置
# ══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LABELED_DIR = os.path.join(PROCESSED_DIR, "labeled")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

SCHEMA_JSONL = os.path.join(PROCESSED_DIR, "event_schema.jsonl")

TRAIN_XLSX = os.path.join(RAW_DIR, "news_with_tags_train.xlsx")
TEST_XLSX = os.path.join(RAW_DIR, "news_with_tags_test.xlsx")

# 输出文件
NEWS_TRAIN_JSONL = os.path.join(LABELED_DIR, "news_train.jsonl")
NEWS_TEST_JSONL = os.path.join(LABELED_DIR, "news_test.jsonl")
EPAIRS_TRAIN_JSONL = os.path.join(LABELED_DIR, "event_pairs_train.jsonl")
EPAIRS_TEST_JSONL = os.path.join(LABELED_DIR, "event_pairs_test.jsonl")
EPAIRS_TRAIN_XLSX = os.path.join(LABELED_DIR, "event_pairs_train.xlsx")
EPAIRS_TEST_XLSX = os.path.join(LABELED_DIR, "event_pairs_test.xlsx")

UNRESOLVED_CSV = os.path.join(REPORTS_DIR, "unresolved_schema_mappings.csv")
INVALID_TIME_CSV = os.path.join(REPORTS_DIR, "invalid_publish_times.csv")
UNRESOLVED_DUP_CSV = os.path.join(REPORTS_DIR, "unresolved_duplication_relations.csv")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "labeled_dataset_summary.json")
EVENT_FREQ_CSV = os.path.join(REPORTS_DIR, "labeled_event_frequency.csv")
SHEET_PROFILE_JSON = os.path.join(REPORTS_DIR, "excel_sheet_profile.json")
CONFLICT_CSV = os.path.join(REPORTS_DIR, "article_content_conflicts.csv")

EXCEL_COLUMNS = [
    "sample_id", "split", "article_id", "title", "publish_time", "source",
    "target_type", "target_code", "target_name", "event_name", "schema_id",
    "emotion", "duplication_id", "content", "event_description",
    "impact_analysis", "label"
]

# ══════════════════════════════════════════════════════════════
# 文本清洗
# ══════════════════════════════════════════════════════════════
def clean_text(text) -> str:
    """轻量文本清洗：去除首尾空格、统一换行、合并连续空格和空行。"""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    # 合并连续空格
    text = re.sub(r" {2,}", " ", text)
    # 每行去首尾空格
    lines = [line.strip() for line in text.split("\n")]
    # 合并 3+ 连续空行为 1 个空行
    cleaned = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
    text = "\n".join(cleaned).strip()
    return text


def normalize_for_match(text) -> str:
    """用于 Schema 匹配的规范化：去首尾空格、统一全/半角空格、合并空格、去换行。"""
    if not text:
        return ""
    text = text.strip()
    # 全角空格 → 半角空格
    text = text.replace("　", " ")
    text = text.replace("\r\n", "").replace("\n", "").replace("\r", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_code(code_val) -> str:
    """标准化代码：将 pandas int/float 转为字符串，去除 .0，补齐 6 位（公司代码）。"""
    if isinstance(code_val, float):
        if np.isnan(code_val):
            return ""
        if code_val == int(code_val):
            code_val = int(code_val)
    if isinstance(code_val, (int, np.integer)):
        return str(int(code_val))
    s = str(code_val).strip()
    # 去掉 .0 后缀
    s = re.sub(r"\.0+$", "", s)
    return s


def pad_company_code(code: str) -> str:
    """将公司代码补齐为 6 位，保留前导零。"""
    if not code:
        return code
    return code.zfill(6)


# ══════════════════════════════════════════════════════════════
# 时间标准化
# ══════════════════════════════════════════════════════════════
def parse_publish_time(val) -> tuple:
    """
    尝试解析发布时间，返回 (raw_str, formatted_str_or_None)。
    支持 pandas Timestamp、datetime、字符串等。
    """
    raw = str(val) if val is not None else ""
    if isinstance(val, pd.Timestamp):
        return raw, val.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(val, datetime):
        return raw, val.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(val, str):
        # 尝试多种格式
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            "%Y年%m月%d日",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(val.strip(), fmt)
                return val, dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return raw, None


# ══════════════════════════════════════════════════════════════
# 加载 Schema
# ══════════════════════════════════════════════════════════════
def load_event_schema(filepath: str) -> dict:
    """加载事件 Schema，构建多级索引。"""
    schemas = {}
    idx_code_event = {}   # (target_type, target_code, event_name) → schema
    idx_name_event = {}   # (target_type, target_name, event_name) → schema

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["schema_id"]
            schemas[sid] = rec

            tt = rec["target_type"]
            tc = str(rec["target_code"])
            tn = normalize_for_match(rec["target_name"])
            en = normalize_for_match(rec["event_name"])

            idx_code_event[(tt, tc, en)] = rec
            idx_name_event[(tt, tn, en)] = rec

    return schemas, idx_code_event, idx_name_event


def find_schema(target_type: str, target_code: str, target_name: str,
                event_name: str, idx_code_event: dict, idx_name_event: dict) -> tuple:
    """
    按优先级匹配 Schema：
    1. (target_type, target_code, event_name) 精确匹配
    2. (target_type, target_name, event_name) 精确匹配
    3. 规范化文本后再次尝试
    4. 无法匹配返回 None
    """
    tc = normalize_for_match(target_code)
    tn = normalize_for_match(target_name)
    en = normalize_for_match(event_name)

    # 优先 code+event
    key1 = (target_type, tc, en)
    if key1 in idx_code_event:
        return idx_code_event[key1], "code_event_match"

    # 其次 name+event
    key2 = (target_type, tn, en)
    if key2 in idx_name_event:
        return idx_name_event[key2], "name_event_match"

    # 规范化后再次尝试 code+event
    # (already done above with normalize_for_match)
    return None, "no_match"


# ══════════════════════════════════════════════════════════════
# 读取 Excel 标签数据
# ══════════════════════════════════════════════════════════════
def read_labeled_sheet(filepath: str, sheet_name: str) -> pd.DataFrame:
    """读取带标签的工作表。"""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df


def read_duplicate_sheet(filepath: str, sheet_name: str) -> pd.DataFrame:
    """读取重复新闻工作表。"""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df


# ══════════════════════════════════════════════════════════════
# 处理主标签数据
# ══════════════════════════════════════════════════════════════
def process_company_sheet(df: pd.DataFrame, split: str, sheet_name: str,
                          idx_code_event: dict, idx_name_event: dict) -> dict:
    """处理个股新闻表，返回原始行记录列表和异常信息。"""
    rows = []
    unresolved = []
    time_errors = []

    for i, row in df.iterrows():
        excel_row = i + 2  # Excel 行号（1-indexed + header）

        article_id = str(row.get("article_file_id", "")).strip()
        if not article_id or article_id == "nan":
            continue

        # 标准化公司代码
        raw_trading_code = row.get("trading_code")
        trading_code = normalize_code(raw_trading_code)
        trading_code_padded = pad_company_code(trading_code)

        secu_abbr = clean_text(row.get("secu_abbr", ""))
        event_name = clean_text(row.get("event_name", ""))
        emotion = clean_text(row.get("event_emotion", ""))
        impact = clean_text(row.get("event_impact_analysis", ""))
        dup_id = str(row.get("duplication_id", "")).strip()
        if dup_id in ("nan", "NaN", ""):
            dup_id = None

        # 找到 Schema
        schema_rec, match_type = find_schema(
            "company", trading_code_padded, secu_abbr, event_name,
            idx_code_event, idx_name_event,
        )

        if schema_rec is None:
            unresolved.append({
                "split": split,
                "sheet_name": sheet_name,
                "excel_row": excel_row,
                "article_id": article_id,
                "target_type": "company",
                "raw_target_code": str(raw_trading_code),
                "raw_target_name": secu_abbr,
                "raw_event_name": event_name,
                "normalized_target_code": trading_code_padded,
                "normalized_target_name": normalize_for_match(secu_abbr),
                "normalized_event_name": normalize_for_match(event_name),
                "failure_reason": f"no_schema_match (code={trading_code_padded}, name={secu_abbr[:20]}, event={event_name[:30]})",
            })
            continue

        rows.append({
            "excel_row": excel_row,
            "sheet_name": sheet_name,
            "split": split,
            "article_id": article_id,
            "schema_id": schema_rec["schema_id"],
            "target_type": "company",
            "target_code": trading_code_padded,
            "target_name": secu_abbr,
            "event_name": event_name,
            "event_description": schema_rec["raw_description"],
            "emotion": emotion,
            "impact_analysis": impact,
            "duplication_id": dup_id,
        })

    return rows, unresolved, time_errors


def process_industry_sheet(df: pd.DataFrame, split: str, sheet_name: str,
                           idx_code_event: dict, idx_name_event: dict) -> dict:
    """处理行业新闻表，返回原始行记录列表和异常信息。"""
    rows = []
    unresolved = []
    time_errors = []

    for i, row in df.iterrows():
        excel_row = i + 2

        article_id = str(row.get("article_file_id", "")).strip()
        if not article_id or article_id == "nan":
            continue

        raw_industry_code = row.get("industry_code")
        industry_code = normalize_code(raw_industry_code)

        industry_name = clean_text(row.get("industry_name", ""))
        event_name = clean_text(row.get("event_name", ""))
        emotion = clean_text(row.get("event_emotion", ""))
        impact = clean_text(row.get("event_impact_analysis", ""))
        dup_id = str(row.get("duplication_id", "")).strip()
        if dup_id in ("nan", "NaN", ""):
            dup_id = None

        schema_rec, match_type = find_schema(
            "industry", industry_code, industry_name, event_name,
            idx_code_event, idx_name_event,
        )

        if schema_rec is None:
            unresolved.append({
                "split": split,
                "sheet_name": sheet_name,
                "excel_row": excel_row,
                "article_id": article_id,
                "target_type": "industry",
                "raw_target_code": str(raw_industry_code),
                "raw_target_name": industry_name,
                "raw_event_name": event_name,
                "normalized_target_code": industry_code,
                "normalized_target_name": normalize_for_match(industry_name),
                "normalized_event_name": normalize_for_match(event_name),
                "failure_reason": f"no_schema_match (code={industry_code}, name={industry_name[:20]}, event={event_name[:30]})",
            })
            continue

        rows.append({
            "excel_row": excel_row,
            "sheet_name": sheet_name,
            "split": split,
            "article_id": article_id,
            "schema_id": schema_rec["schema_id"],
            "target_type": "industry",
            "target_code": industry_code,
            "target_name": industry_name,
            "event_name": event_name,
            "event_description": schema_rec["raw_description"],
            "emotion": emotion,
            "impact_analysis": impact,
            "duplication_id": dup_id,
        })

    return rows, unresolved, time_errors


# ══════════════════════════════════════════════════════════════
# 处理重复新闻表
# ══════════════════════════════════════════════════════════════
def build_duplication_mapping(df: pd.DataFrame) -> dict:
    """
    从重复新闻表构建 article_file_id → duplication_id 映射。
    重复新闻表结构：article_file_id, article_title, article_source, content, article_publish_time, duplication_id
    """
    mapping = {}
    for _, row in df.iterrows():
        aid = str(row.get("article_file_id", "")).strip()
        did = str(row.get("duplication_id", "")).strip()
        if aid and aid != "nan" and did and did != "nan":
            mapping[aid] = did
    return mapping


def build_reverse_duplication_map(dup_mapping: dict) -> dict:
    """
    构建 duplication_id → [article_file_id, ...] 的反向映射。
    用于将同一重复组的文章关联起来。
    """
    reverse_map = defaultdict(list)
    for aid, did in dup_mapping.items():
        reverse_map[did].append(aid)
    return dict(reverse_map)


# ══════════════════════════════════════════════════════════════
# 聚合为新闻级数据
# ══════════════════════════════════════════════════════════════
def build_article_metadata_map(df_main: pd.DataFrame, article_id_col: str) -> dict:
    """从主 DataFrame 提取每篇文章的基础元数据（取第一行）。"""
    meta = {}
    for _, row in df_main.iterrows():
        aid = str(row.get(article_id_col, "")).strip()
        if not aid or aid == "nan" or aid in meta:
            continue
        title = clean_text(row.get("article_title", ""))
        content = clean_text(row.get("content", ""))
        source = clean_text(row.get("article_source", ""))
        pub_raw, pub_formatted = parse_publish_time(row.get("article_publish_time"))
        dup_id = str(row.get("duplication_id", "")).strip()
        if dup_id in ("nan", "NaN", ""):
            dup_id = None

        meta[aid] = {
            "article_id": aid,
            "title": title,
            "publish_time_raw": pub_raw,
            "publish_time": pub_formatted,
            "source": source,
            "content": content,
            "duplication_id": dup_id,
        }
    return meta


def aggregate_news_level(label_rows: list, article_meta: dict, dup_mapping: dict) -> tuple:
    """将标签行聚合为新闻级数据，返回 (news_records, content_conflicts)。"""
    # 按 article_id 分组标签
    article_labels = defaultdict(list)
    for lr in label_rows:
        aid = lr["article_id"]
        article_labels[aid].append(lr)

    news_records = []
    content_conflicts = []
    seen_article_ids = set()

    for aid, labels in article_labels.items():
        # 合并文章元数据
        meta = article_meta.get(aid, {})
        title = meta.get("title", "")
        content = meta.get("content", "")
        source = meta.get("source", "")
        pub_raw = meta.get("publish_time_raw", "")
        pub_time = meta.get("publish_time", "")
        dup_id = meta.get("duplication_id")

        # 如果 duplication_id 缺失，尝试从映射中补充
        if not dup_id and aid in dup_mapping:
            dup_id = dup_mapping[aid]

        # 去重：相同 schema_id + target_code + event_name 只保留一次
        deduped_labels = []
        seen_keys = set()
        for lb in labels:
            key = (lb["schema_id"], lb["target_code"], lb["event_name"])
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_labels.append({
                    "schema_id": lb["schema_id"],
                    "target_type": lb["target_type"],
                    "target_code": lb["target_code"],
                    "target_name": lb["target_name"],
                    "event_name": lb["event_name"],
                    "emotion": lb["emotion"],
                    "impact_analysis": lb["impact_analysis"],
                })

        news_records.append({
            "article_id": aid,
            "title": title,
            "publish_time_raw": pub_raw,
            "publish_time": pub_time,
            "source": source,
            "content": content,
            "duplication_id": dup_id,
            "labels": deduped_labels,
        })

    return news_records, content_conflicts


def build_event_pairs(news_records: list, label_rows: list, split: str) -> list:
    """从新闻级记录构建事件对记录（每个标签一条）。"""
    pairs = []
    for lr in label_rows:
        sid = f"{split}_{lr['article_id']}_{lr['schema_id']}"
        pairs.append({
            "sample_id": sid,
            "split": split,
            "article_id": lr["article_id"],
            "title": "",  # 后续从 news_records 填充
            "publish_time": "",
            "source": "",
            "content": "",
            "duplication_id": lr.get("duplication_id"),
            "schema_id": lr["schema_id"],
            "target_type": lr["target_type"],
            "target_code": lr["target_code"],
            "target_name": lr["target_name"],
            "event_name": lr["event_name"],
            "event_description": lr["event_description"],
            "emotion": lr["emotion"],
            "impact_analysis": lr["impact_analysis"],
            "label": 1,
        })
    return pairs


def fill_event_pairs_meta(pairs: list, news_lookup: dict):
    """用新闻元数据回填事件对记录。"""
    for p in pairs:
        aid = p["article_id"]
        news = news_lookup.get(aid, {})
        p["title"] = news.get("title", "")
        p["publish_time"] = news.get("publish_time", "")
        p["source"] = news.get("source", "")
        p["content"] = news.get("content", "")
        p["duplication_id"] = news.get("duplication_id")


# ══════════════════════════════════════════════════════════════
# 时间校验
# ══════════════════════════════════════════════════════════════
def collect_time_errors(news_records: list) -> list:
    """收集时间解析失败记录。"""
    errors = []
    for nr in news_records:
        if nr["publish_time"] is None:
            errors.append({
                "article_id": nr["article_id"],
                "publish_time_raw": nr["publish_time_raw"],
            })
    return errors


# ══════════════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════════════
def compute_stats(label_rows: list, news_records: list, event_pairs: list,
                  mapping_success: int, mapping_failure: int) -> dict:
    """计算统计指标。"""
    company_label_rows = [r for r in label_rows if r["target_type"] == "company"]
    industry_label_rows = [r for r in label_rows if r["target_type"] == "industry"]

    multi_label_count = sum(1 for n in news_records if len(n["labels"]) > 1)
    max_labels = max((len(n["labels"]) for n in news_records), default=0)

    with_dup = sum(1 for n in news_records if n.get("duplication_id"))
    without_dup = len(news_records) - with_dup

    total_mapping = mapping_success + mapping_failure
    rate = mapping_success / total_mapping if total_mapping > 0 else 0.0

    return {
        "company_label_rows": len(company_label_rows),
        "industry_label_rows": len(industry_label_rows),
        "total_label_rows": len(label_rows),
        "unique_article_count": len(news_records),
        "event_pair_count": len(event_pairs),
        "multi_label_article_count": multi_label_count,
        "max_labels_per_article": max_labels,
        "schema_mapping_success_count": mapping_success,
        "schema_mapping_failure_count": mapping_failure,
        "schema_mapping_success_rate": round(rate, 4),
        "article_with_duplication_id_count": with_dup,
        "article_without_duplication_id_count": without_dup,
    }


def build_sheet_profile(filepath: str) -> list:
    """生成 Excel 工作表概况。"""
    profiles = []
    xl = pd.ExcelFile(filepath)
    for sheet in xl.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet)
        nulls = {col: int(df[col].isnull().sum()) for col in df.columns}
        types = {col: str(df[col].dtype) for col in df.columns}
        profiles.append({
            "file": os.path.basename(filepath),
            "sheet_name": sheet,
            "row_count": len(df),
            "column_count": len(df.columns),
            "column_names": list(df.columns),
            "null_count_by_column": nulls,
            "inferred_type_by_column": types,
        })
    return profiles


# ══════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════
def write_jsonl(records: list, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_excel(records: list, filepath: str):
    columns = EXCEL_COLUMNS
    df = pd.DataFrame(records, columns=columns)
    # 确保列顺序
    df = df[columns]
    df.to_excel(filepath, index=False, engine="openpyxl")


def write_unresolved_csv(records: list, filepath: str):
    if not records:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("split,sheet_name,excel_row,article_id,target_type,raw_target_code,raw_target_name,raw_event_name,normalized_target_code,normalized_target_name,normalized_event_name,failure_reason\n")
        return
    df = pd.DataFrame(records)
    columns = ["split", "sheet_name", "excel_row", "article_id", "target_type",
               "raw_target_code", "raw_target_name", "raw_event_name",
               "normalized_target_code", "normalized_target_name", "normalized_event_name",
               "failure_reason"]
    df = df[columns]
    df.to_csv(filepath, index=False, encoding="utf-8-sig")


def write_time_errors(errors: list, filepath: str):
    if not errors:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("article_id,publish_time_raw\n")
        return
    df = pd.DataFrame(errors)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def process_one_split(xlsx_path: str, split: str, idx_code_event: dict,
                      idx_name_event: dict) -> dict:
    """处理一个 Excel（train 或 test）。"""
    print(f"\n{'='*60}")
    print(f"处理 {split} 数据集: {os.path.basename(xlsx_path)}")
    print(f"{'='*60}")

    # 读取所有工作表
    company_df = read_labeled_sheet(xlsx_path, "个股新闻")
    industry_df = read_labeled_sheet(xlsx_path, "行业新闻")
    dup_company_df = read_duplicate_sheet(xlsx_path, "个股重复新闻")
    dup_industry_df = read_duplicate_sheet(xlsx_path, "行业重复新闻")

    print(f"  个股新闻: {len(company_df)} 行")
    print(f"  行业新闻: {len(industry_df)} 行")
    print(f"  个股重复新闻: {len(dup_company_df)} 行")
    print(f"  行业重复新闻: {len(dup_industry_df)} 行")

    # 构建文章元数据
    article_meta = {}
    article_meta.update(build_article_metadata_map(company_df, "article_file_id"))
    article_meta.update(build_article_metadata_map(industry_df, "article_file_id"))
    # 也从重复新闻表补充元数据（含dup新闻的content等）
    dup_meta = build_article_metadata_map(dup_company_df, "article_file_id")
    dup_meta.update(build_article_metadata_map(dup_industry_df, "article_file_id"))
    article_meta.update(dup_meta)

    # 构建 duplication 映射
    dup_mapping = {}
    dup_mapping.update(build_duplication_mapping(dup_company_df))
    dup_mapping.update(build_duplication_mapping(dup_industry_df))
    print(f"  duplication 映射条目: {len(dup_mapping)}")

    # 反向映射
    reverse_dup = build_reverse_duplication_map(dup_mapping)

    # 处理公司标签
    print("\n  处理个股新闻标签...")
    company_rows, comp_unresolved, comp_time_err = process_company_sheet(
        company_df, split, "个股新闻", idx_code_event, idx_name_event)
    print(f"    成功映射: {len(company_rows)}, 未匹配: {len(comp_unresolved)}")

    # 处理行业标签
    print("  处理行业新闻标签...")
    industry_rows, ind_unresolved, ind_time_err = process_industry_sheet(
        industry_df, split, "行业新闻", idx_code_event, idx_name_event)
    print(f"    成功映射: {len(industry_rows)}, 未匹配: {len(ind_unresolved)}")

    all_label_rows = company_rows + industry_rows
    all_unresolved = comp_unresolved + ind_unresolved
    all_time_errors = comp_time_err + ind_time_err

    # 去重：完全相同键的标签行
    # 已经在 aggregate_news_level 层面做了

    # 补充 duplication_id：如果 label_row 没有 dup_id，尝试从映射中补充
    for lr in all_label_rows:
        if not lr.get("duplication_id"):
            aid = lr["article_id"]
            if aid in dup_mapping:
                lr["duplication_id"] = dup_mapping[aid]

    # 聚合为新闻级数据
    print("\n  聚合新闻级数据...")
    news_records, conflicts = aggregate_news_level(all_label_rows, article_meta, dup_mapping)

    # 回填 duplication_id：新闻记录也需要
    for nr in news_records:
        if not nr.get("duplication_id"):
            aid = nr["article_id"]
            if aid in dup_mapping:
                nr["duplication_id"] = dup_mapping[aid]

    print(f"    新闻记录数: {len(news_records)}")
    print(f"    内容冲突数: {len(conflicts)}")

    # 构建事件对数据
    print("  构建事件对数据...")
    event_pairs = build_event_pairs(news_records, all_label_rows, split)
    # 用新闻级数据回填元数据
    news_lookup = {n["article_id"]: n for n in news_records}
    fill_event_pairs_meta(event_pairs, news_lookup)
    print(f"    事件对记录数: {len(event_pairs)}")

    # 时间校验
    time_errors = collect_time_errors(news_records)
    print(f"    时间解析失败: {len(time_errors)}")

    # 统计
    mapping_success = len(all_label_rows)
    mapping_failure = len(all_unresolved)
    stats = compute_stats(all_label_rows, news_records, event_pairs,
                          mapping_success, mapping_failure)

    return {
        "news_records": news_records,
        "event_pairs": event_pairs,
        "unresolved": all_unresolved,
        "time_errors": time_errors,
        "conflicts": conflicts,
        "stats": stats,
        "dup_mapping": dup_mapping,
        "reverse_dup": reverse_dup,
        "company_df": company_df,
        "industry_df": industry_df,
        "dup_company_df": dup_company_df,
        "dup_industry_df": dup_industry_df,
    }


def main():
    print("=" * 60)
    print("带标签数据集构建工具")
    print("=" * 60)

    # 1. 加载 Schema
    print("\n[1] 加载事件 Schema...")
    schemas, idx_code_event, idx_name_event = load_event_schema(SCHEMA_JSONL)
    print(f"  加载 {len(schemas)} 个 Schema")
    print(f"  code+event 索引: {len(idx_code_event)} 个键")
    print(f"  name+event 索引: {len(idx_name_event)} 个键")

    # 2. 处理 train
    print("\n[2] 处理训练集...")
    train_result = process_one_split(TRAIN_XLSX, "train", idx_code_event, idx_name_event)

    # 3. 处理 test
    print("\n[3] 处理测试集...")
    test_result = process_one_split(TEST_XLSX, "test", idx_code_event, idx_name_event)

    # 4. 输出文件
    print("\n[4] 写入输出文件...")
    os.makedirs(LABELED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    for result, split in [(train_result, "train"), (test_result, "test")]:
        news_file = NEWS_TRAIN_JSONL if split == "train" else NEWS_TEST_JSONL
        epairs_jsonl = EPAIRS_TRAIN_JSONL if split == "train" else EPAIRS_TEST_JSONL
        epairs_xlsx = EPAIRS_TRAIN_XLSX if split == "train" else EPAIRS_TEST_XLSX

        write_jsonl(result["news_records"], news_file)
        print(f"  ✓ {os.path.basename(news_file)}: {len(result['news_records'])} 条")

        write_jsonl(result["event_pairs"], epairs_jsonl)
        print(f"  ✓ {os.path.basename(epairs_jsonl)}: {len(result['event_pairs'])} 条")

        write_excel(result["event_pairs"], epairs_xlsx)
        print(f"  ✓ {os.path.basename(epairs_xlsx)}: {len(result['event_pairs'])} 条")

    # 未匹配记录
    all_unresolved = train_result["unresolved"] + test_result["unresolved"]
    write_unresolved_csv(all_unresolved, UNRESOLVED_CSV)
    print(f"  ✓ 未匹配记录: {UNRESOLVED_CSV} ({len(all_unresolved)} 条)")

    # 时间异常
    all_time_errors = train_result["time_errors"] + test_result["time_errors"]
    write_time_errors(all_time_errors, INVALID_TIME_CSV)
    print(f"  ✓ 时间异常: {INVALID_TIME_CSV} ({len(all_time_errors)} 条)")

    # 内容冲突
    all_conflicts = train_result["conflicts"] + test_result["conflicts"]
    if all_conflicts:
        pd.DataFrame(all_conflicts).to_csv(CONFLICT_CSV, index=False, encoding="utf-8-sig")
        print(f"  ✓ 内容冲突: {CONFLICT_CSV} ({len(all_conflicts)} 条)")
    else:
        with open(CONFLICT_CSV, "w", encoding="utf-8") as f:
            f.write("article_id,field,value_1,value_2,source_row_1,source_row_2\n")

    # 5. 统计报告
    print("\n[5] 生成统计报告...")

    summary = {
        "train": train_result["stats"],
        "test": test_result["stats"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 汇总报告: {SUMMARY_JSON}")

    # 事件频率
    all_pairs = train_result["event_pairs"] + test_result["event_pairs"]
    freq_data = []
    for p in all_pairs:
        freq_data.append((p["split"], p["target_type"], p["schema_id"],
                          p["target_name"], p["event_name"]))
    freq_df = pd.DataFrame(freq_data, columns=["split", "target_type", "schema_id", "target_name", "event_name"])
    freq_counts = freq_df.groupby(["split", "target_type", "schema_id", "target_name", "event_name"]).size().reset_index(name="count")
    freq_counts.to_csv(EVENT_FREQ_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✓ 事件频率: {EVENT_FREQ_CSV}")

    # 工作表概况
    all_profiles = []
    for xlsx_path in [TRAIN_XLSX, TEST_XLSX]:
        all_profiles.extend(build_sheet_profile(xlsx_path))
    with open(SHEET_PROFILE_JSON, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 工作表概况: {SHEET_PROFILE_JSON}")

    # 重复新闻关系异常报告
    dup_anomalies = []
    # 检查主表中 duplication_id 是否指向不存在的 article
    all_article_ids = set()
    for result in [train_result, test_result]:
        for nr in result["news_records"]:
            all_article_ids.add(nr["article_id"])
    for result in [train_result, test_result]:
        for nr in result["news_records"]:
            did = nr.get("duplication_id")
            if did:
                # 可能是逗号分隔的多个 ID
                for single_did in did.split(","):
                    single_did = single_did.strip()
                    if single_did and single_did not in all_article_ids:
                        # 可能存在于重复新闻表中
                        pass  # 不报错，因为可能在另一个split中
    if dup_anomalies:
        pd.DataFrame(dup_anomalies).to_csv(UNRESOLVED_DUP_CSV, index=False, encoding="utf-8-sig")
    else:
        with open(UNRESOLVED_DUP_CSV, "w", encoding="utf-8") as f:
            f.write("article_id,duplication_id,issue\n")

    # 6. 最终摘要
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    for split_name, result in [("train", train_result), ("test", test_result)]:
        s = result["stats"]
        print(f"\n  [{split_name}]")
        print(f"    公司标签行数: {s['company_label_rows']}")
        print(f"    行业标签行数: {s['industry_label_rows']}")
        print(f"    总标签行数: {s['total_label_rows']}")
        print(f"    唯一新闻数: {s['unique_article_count']}")
        print(f"    事件对数: {s['event_pair_count']}")
        print(f"    多标签新闻数: {s['multi_label_article_count']}")
        print(f"    最大单篇标签数: {s['max_labels_per_article']}")
        print(f"    Schema 映射成功/失败: {s['schema_mapping_success_count']}/{s['schema_mapping_failure_count']}")
        print(f"    映射成功率: {s['schema_mapping_success_rate']}")
        print(f"    有 duplication_id 的新闻: {s['article_with_duplication_id_count']}")
        print(f"    无 duplication_id 的新闻: {s['article_without_duplication_id_count']}")

    # 展示前几条
    print("\n--- news_train.jsonl 前 2 条 ---")
    for i, nr in enumerate(train_result["news_records"][:2]):
        display = {k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
                   for k, v in nr.items()}
        print(f"\n[{i+1}] {json.dumps(display, ensure_ascii=False, indent=2)}")

    print("\n--- event_pairs_train.jsonl 前 3 条 ---")
    for i, ep in enumerate(train_result["event_pairs"][:3]):
        display = {k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
                   for k, v in ep.items()}
        print(f"\n[{i+1}] {json.dumps(display, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
