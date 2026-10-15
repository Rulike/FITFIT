#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_duplicate_news.py
-------------------------
第三步：根据重复新闻表的 duplication_id，将主新闻事件标签传播到转载新闻，
构建"重复新闻增强正样本集"。

逻辑：
  duplicate_article.duplication_id → main_article.article_id → 继承全部 labels
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LABELED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "labeled")
AUGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "augmented")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

SCHEMA_JSONL = os.path.join(PROJECT_ROOT, "data", "processed", "event_schema.jsonl")

TRAIN_XLSX = os.path.join(RAW_DIR, "news_with_tags_train.xlsx")
TEST_XLSX = os.path.join(RAW_DIR, "news_with_tags_test.xlsx")

# ── 输出路径 ──────────────────────────────────────────────
DUP_NEWS_TRAIN = os.path.join(AUGMENTED_DIR, "duplicate_news_train.jsonl")
DUP_NEWS_TEST = os.path.join(AUGMENTED_DIR, "duplicate_news_test.jsonl")
DUP_EPAIRS_TRAIN = os.path.join(AUGMENTED_DIR, "duplicate_event_pairs_train.jsonl")
DUP_EPAIRS_TEST = os.path.join(AUGMENTED_DIR, "duplicate_event_pairs_test.jsonl")
DUP_EPAIRS_TRAIN_XLSX = os.path.join(AUGMENTED_DIR, "duplicate_event_pairs_train.xlsx")
DUP_EPAIRS_TEST_XLSX = os.path.join(AUGMENTED_DIR, "duplicate_event_pairs_test.xlsx")

NEWS_AUG_TRAIN = os.path.join(AUGMENTED_DIR, "news_train_augmented.jsonl")
NEWS_AUG_TEST = os.path.join(AUGMENTED_DIR, "news_test_augmented.jsonl")
EPAIRS_AUG_TRAIN = os.path.join(AUGMENTED_DIR, "event_pairs_train_augmented.jsonl")
EPAIRS_AUG_TEST = os.path.join(AUGMENTED_DIR, "event_pairs_test_augmented.jsonl")

UNRESOLVED_CSV = os.path.join(REPORTS_DIR, "unresolved_duplicate_label_propagation.csv")
CONFLICT_CSV = os.path.join(REPORTS_DIR, "duplicate_article_conflicts.csv")
SIMILARITY_CSV = os.path.join(REPORTS_DIR, "duplicate_text_similarity.csv")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "duplicate_augmentation_summary.json")
AUG_FREQ_CSV = os.path.join(REPORTS_DIR, "duplicate_augmented_event_frequency.csv")

EXCEL_COLUMNS = [
    "sample_id", "split", "article_id", "canonical_article_id", "title",
    "publish_time", "source", "target_type", "target_code", "target_name",
    "event_name", "schema_id", "emotion", "duplication_id", "content",
    "event_description", "impact_analysis", "label",
    "is_duplicate_news", "label_source"
]


# ══════════════════════════════════════════════════════════════
def clean_text(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    lines = [line.strip() for line in text.split("\n")]
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
    return "\n".join(cleaned).strip()


def parse_publish_time(val):
    raw = str(val) if val is not None else ""
    if isinstance(val, pd.Timestamp):
        return raw, val.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(val, str):
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"]:
            try:
                from datetime import datetime
                dt = datetime.strptime(val.strip(), fmt)
                return val, dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return raw, None


# ══════════════════════════════════════════════════════════════
def read_duplicate_sheets(xlsx_path: str, split: str) -> list:
    """读取重复新闻表，返回标准化记录列表。"""
    records = []
    for sheet_name in ["个股重复新闻", "行业重复新闻"]:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        for i, row in df.iterrows():
            aid = str(row.get("article_file_id", "")).strip()
            did = str(row.get("duplication_id", "")).strip()
            if aid in ("nan", "", "NaN"):
                continue
            pub_raw, pub_time = parse_publish_time(row.get("article_publish_time"))
            records.append({
                "excel_row": i + 2,
                "sheet_name": sheet_name,
                "split": split,
                "article_id": aid,
                "title": clean_text(row.get("article_title", "")),
                "source": clean_text(row.get("article_source", "")),
                "content": clean_text(row.get("content", "")),
                "publish_time_raw": pub_raw,
                "publish_time": pub_time,
                "duplication_id": did if did not in ("nan", "", "NaN") else None,
            })
    return records


def load_main_news_index(split: str) -> dict:
    """加载主新闻级数据，建立 article_id → news_record 索引。"""
    filepath = os.path.join(LABELED_DIR, f"news_{split}.jsonl")
    index = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                index[rec["article_id"]] = rec
    return index


def load_schemas() -> dict:
    """加载事件 Schema。"""
    schemas = {}
    with open(SCHEMA_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                schemas[rec["schema_id"]] = rec
    return schemas


# ══════════════════════════════════════════════════════════════
def compute_jaccard(s1: str, s2: str, n: int = 1) -> float:
    """计算两个字符串的 n-gram Jaccard 相似度。"""
    if not s1 or not s2:
        return 0.0
    ngrams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    ngrams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union) if union else 0.0


def compute_similarities(dup_records: list, main_index: dict) -> list:
    """计算每组主新闻与转载新闻之间的文本相似度。"""
    results = []
    for dup in dup_records:
        did = dup.get("duplication_id")
        if not did or did not in main_index:
            continue
        main = main_index[did]
        title_j = compute_jaccard(dup["title"], main.get("title", ""))
        content_3g = compute_jaccard(dup["content"], main.get("content", ""), n=3)
        len_dup = len(dup["content"])
        len_main = len(main.get("content", ""))
        len_ratio = min(len_dup, len_main) / max(len_dup, len_main) if max(len_dup, len_main) > 0 else 0.0
        results.append({
            "split": dup["split"],
            "duplicate_article_id": dup["article_id"],
            "canonical_article_id": did,
            "title_jaccard": round(title_j, 4),
            "content_3gram_jaccard": round(content_3g, 4),
            "content_length_ratio": round(len_ratio, 4),
            "canonical_content_length": len_main,
            "duplicate_content_length": len_dup,
        })
    return results


# ══════════════════════════════════════════════════════════════
def propagate_labels(dup_records: list, main_index: dict, schemas: dict) -> dict:
    """
    核心逻辑：将主新闻标签传播到重复新闻。
    返回 (dup_news, dup_event_pairs, unresolved, conflicts)
    """
    dup_news = []
    dup_event_pairs = []
    unresolved = []
    conflicts = []
    seen_aids = {}  # article_id → first occurrence index

    # 去重：同一 article_id 出现多次的处理
    deduped = []
    for i, rec in enumerate(dup_records):
        aid = rec["article_id"]
        if aid in seen_aids:
            prev_idx = seen_aids[aid]
            prev = deduped[prev_idx]
            # 检查是否完全相同
            if (prev["title"] == rec["title"] and
                prev["duplication_id"] == rec["duplication_id"] and
                prev["content"] == rec["content"]):
                continue  # 完全相同，跳过
            else:
                # 内容冲突，保留更长的
                conflicts.append({
                    "article_id": aid,
                    "row_1": prev["excel_row"],
                    "row_2": rec["excel_row"],
                    "content_len_1": len(prev["content"]),
                    "content_len_2": len(rec["content"]),
                    "action": "kept_longer",
                })
                if len(rec["content"]) > len(prev["content"]):
                    deduped[prev_idx] = rec
        else:
            seen_aids[aid] = len(deduped)
            deduped.append(rec)

    for rec in deduped:
        aid = rec["article_id"]
        did = rec["duplication_id"]

        # 检查是否已存在于主新闻
        if aid in main_index:
            unresolved.append({
                "split": rec["split"],
                "sheet_name": rec["sheet_name"],
                "excel_row": rec["excel_row"],
                "duplicate_article_id": aid,
                "duplication_id": did or "",
                "failure_reason": "duplicate_article_id_conflict",
            })
            continue

        # 检查 duplication_id
        if not did:
            unresolved.append({
                "split": rec["split"],
                "sheet_name": rec["sheet_name"],
                "excel_row": rec["excel_row"],
                "duplicate_article_id": aid,
                "duplication_id": "",
                "failure_reason": "missing_duplication_id",
            })
            continue

        # 查找主新闻
        if did not in main_index:
            unresolved.append({
                "split": rec["split"],
                "sheet_name": rec["sheet_name"],
                "excel_row": rec["excel_row"],
                "duplicate_article_id": aid,
                "duplication_id": did,
                "failure_reason": "canonical_article_not_found",
            })
            continue

        main = main_index[did]
        if not main.get("labels"):
            unresolved.append({
                "split": rec["split"],
                "sheet_name": rec["sheet_name"],
                "excel_row": rec["excel_row"],
                "duplicate_article_id": aid,
                "duplication_id": did,
                "failure_reason": "canonical_article_has_no_labels",
            })
            continue

        # 传播标签（去重）
        seen_keys = set()
        propagated_labels = []
        for lb in main["labels"]:
            key = (lb["schema_id"], lb.get("target_code", ""), lb.get("event_name", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                propagated_labels.append({
                    "schema_id": lb["schema_id"],
                    "target_type": lb["target_type"],
                    "target_code": lb["target_code"],
                    "target_name": lb["target_name"],
                    "event_name": lb["event_name"],
                    "emotion": lb.get("emotion", ""),
                    "impact_analysis": lb.get("impact_analysis", ""),
                })

        # 构建重复新闻级记录
        news_rec = {
            "article_id": aid,
            "title": rec["title"],
            "publish_time_raw": rec["publish_time_raw"],
            "publish_time": rec["publish_time"],
            "source": rec["source"],
            "content": rec["content"],
            "duplication_id": did,
            "canonical_article_id": did,
            "is_duplicate_news": True,
            "label_source": "inherited_from_canonical_article",
            "labels": propagated_labels,
        }
        dup_news.append(news_rec)

        # 展开为事件对
        for lb in propagated_labels:
            sid = rec["schema_id"] if "schema_id" in rec else lb["schema_id"]
            event_desc = ""
            schema_rec = schemas.get(lb["schema_id"])
            if schema_rec:
                event_desc = schema_rec.get("raw_description", "")

            ep = {
                "sample_id": f"{rec['split']}_dup_{aid}_{lb['schema_id']}",
                "split": rec["split"],
                "article_id": aid,
                "canonical_article_id": did,
                "title": rec["title"],
                "publish_time": rec["publish_time"],
                "source": rec["source"],
                "content": rec["content"],
                "duplication_id": did,
                "is_duplicate_news": True,
                "label_source": "inherited_from_canonical_article",
                "schema_id": lb["schema_id"],
                "target_type": lb["target_type"],
                "target_code": lb["target_code"],
                "target_name": lb["target_name"],
                "event_name": lb["event_name"],
                "event_description": event_desc,
                "emotion": lb["emotion"],
                "impact_analysis": lb["impact_analysis"],
                "label": 1,
            }
            dup_event_pairs.append(ep)

    return {
        "dup_news": dup_news,
        "dup_event_pairs": dup_event_pairs,
        "unresolved": unresolved,
        "conflicts": conflicts,
    }


# ══════════════════════════════════════════════════════════════
def build_augmented(orig_news_path: str, orig_epairs_path: str,
                    dup_news: list, dup_epairs: list) -> tuple:
    """合并原始数据和重复新闻增强数据。"""
    orig_news = []
    with open(orig_news_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                orig_news.append(json.loads(line))

    orig_epairs = []
    with open(orig_epairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                orig_epairs.append(json.loads(line))

    # 去重：确保重复新闻不在原始数据中
    orig_ids = {n["article_id"] for n in orig_news}
    filtered_dup_news = [d for d in dup_news if d["article_id"] not in orig_ids]
    filtered_dup_epairs = [d for d in dup_epairs if d["article_id"] not in orig_ids]

    aug_news = orig_news + filtered_dup_news
    aug_epairs = orig_epairs + filtered_dup_epairs

    return aug_news, aug_epairs


# ══════════════════════════════════════════════════════════════
def compute_stats(dup_records: list, dup_news: list, dup_epairs: list,
                  unresolved: list, aug_news_count: int, aug_epairs_count: int,
                  orig_news_count: int, orig_epairs_count: int) -> dict:
    """生成统计。"""
    unique_dup = len({r["article_id"] for r in dup_records})
    matched = len(dup_news)
    unmatched = len(unresolved)
    multi = sum(1 for n in dup_news if len(n["labels"]) > 1)
    max_l = max((len(n["labels"]) for n in dup_news), default=0)

    return {
        "raw_duplicate_row_count": len(dup_records),
        "unique_duplicate_article_count": unique_dup,
        "matched_canonical_article_count": matched,
        "unmatched_canonical_article_count": unmatched,
        "generated_duplicate_news_count": len(dup_news),
        "generated_duplicate_event_pair_count": len(dup_epairs),
        "multi_label_duplicate_news_count": multi,
        "max_labels_per_duplicate_article": max_l,
        "original_news_count": orig_news_count,
        "augmented_news_count": aug_news_count,
        "original_event_pair_count": orig_epairs_count,
        "augmented_event_pair_count": aug_epairs_count,
    }


def build_frequency(dup_epairs: list, orig_epairs_path: str, split: str) -> list:
    """构建增强事件频率表。"""
    # 原始事件对频率
    orig_counter = Counter()
    with open(orig_epairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                orig_counter[rec["schema_id"]] += 1

    # 重复事件对频率
    dup_counter = Counter()
    for ep in dup_epairs:
        dup_counter[ep["schema_id"]] += 1

    all_sids = set(orig_counter.keys()) | set(dup_counter.keys())
    rows = []
    for sid in sorted(all_sids):
        # 从第一个事件对获取元数据
        target_type = ""
        target_name = ""
        event_name = ""
        for ep in dup_epairs:
            if ep["schema_id"] == sid:
                target_type = ep["target_type"]
                target_name = ep["target_name"]
                event_name = ep["event_name"]
                break
        orig_c = orig_counter.get(sid, 0)
        dup_c = dup_counter.get(sid, 0)
        rows.append({
            "split": split,
            "schema_id": sid,
            "target_type": target_type,
            "target_name": target_name,
            "event_name": event_name,
            "original_count": orig_c,
            "duplicate_augmented_count": dup_c,
            "total_count": orig_c + dup_c,
        })
    return rows


# ══════════════════════════════════════════════════════════════
def write_jsonl(records: list, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_excel(records: list, filepath: str, columns: list):
    df = pd.DataFrame(records, columns=columns)
    df = df[columns]
    df.to_excel(filepath, index=False, engine="openpyxl")


# ══════════════════════════════════════════════════════════════
def process_one_split(xlsx_path: str, split: str, schemas: dict) -> dict:
    """处理一个 split 的重复新闻标签传播。"""
    print(f"\n{'='*60}")
    print(f"处理 {split} 重复新闻标签传播")
    print(f"{'='*60}")

    # 读取重复新闻表
    dup_records = read_duplicate_sheets(xlsx_path, split)
    print(f"  读取重复新闻: {len(dup_records)} 条")

    # 加载主新闻索引
    main_index = load_main_news_index(split)
    print(f"  主新闻索引: {len(main_index)} 条")

    # 传播标签
    result = propagate_labels(dup_records, main_index, schemas)
    print(f"  成功传播: {len(result['dup_news'])} 条新闻, {len(result['dup_event_pairs'])} 个事件对")
    print(f"  未匹配: {len(result['unresolved'])} 条")
    print(f"  冲突: {len(result['conflicts'])} 条")

    # 文本相似度
    similarities = compute_similarities(dup_records, main_index)

    # 加载原始数据计数
    orig_news_path = os.path.join(LABELED_DIR, f"news_{split}.jsonl")
    orig_epairs_path = os.path.join(LABELED_DIR, f"event_pairs_{split}.jsonl")
    orig_news_count = sum(1 for _ in open(orig_news_path))
    orig_epairs_count = sum(1 for _ in open(orig_epairs_path))

    # 构建增强数据
    aug_news, aug_epairs = build_augmented(
        orig_news_path, orig_epairs_path,
        result["dup_news"], result["dup_event_pairs"])

    # 统计
    stats = compute_stats(dup_records, result["dup_news"], result["dup_event_pairs"],
                          result["unresolved"], len(aug_news), len(aug_epairs),
                          orig_news_count, orig_epairs_count)

    # 事件频率
    freq_rows = build_frequency(result["dup_event_pairs"], orig_epairs_path, split)

    return {
        "dup_news": result["dup_news"],
        "dup_event_pairs": result["dup_event_pairs"],
        "unresolved": result["unresolved"],
        "conflicts": result["conflicts"],
        "similarities": similarities,
        "stats": stats,
        "freq_rows": freq_rows,
        "aug_news": aug_news,
        "aug_epairs": aug_epairs,
    }


def main():
    print("=" * 60)
    print("重复新闻标签传播增强工具")
    print("=" * 60)

    # 加载 Schema
    print("\n[1] 加载事件 Schema...")
    schemas = load_schemas()
    print(f"  加载 {len(schemas)} 个 Schema")

    # 处理 train 和 test
    train_result = process_one_split(TRAIN_XLSX, "train", schemas)
    test_result = process_one_split(TEST_XLSX, "test", schemas)

    # 写入输出文件
    print(f"\n{'='*60}")
    print("写入输出文件...")
    print(f"{'='*60}")

    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    for result, split in [(train_result, "train"), (test_result, "test")]:
        # 仅重复新闻
        dup_news_path = DUP_NEWS_TRAIN if split == "train" else DUP_NEWS_TEST
        dup_epairs_path = DUP_EPAIRS_TRAIN if split == "train" else DUP_EPAIRS_TEST
        dup_xlsx_path = DUP_EPAIRS_TRAIN_XLSX if split == "train" else DUP_EPAIRS_TEST_XLSX

        write_jsonl(result["dup_news"], dup_news_path)
        print(f"  ✓ {os.path.basename(dup_news_path)}: {len(result['dup_news'])} 条")

        write_jsonl(result["dup_event_pairs"], dup_epairs_path)
        print(f"  ✓ {os.path.basename(dup_epairs_path)}: {len(result['dup_event_pairs'])} 条")

        write_excel(result["dup_event_pairs"], dup_xlsx_path, EXCEL_COLUMNS)
        print(f"  ✓ {os.path.basename(dup_xlsx_path)}: {len(result['dup_event_pairs'])} 条")

        # 增强合并数据
        aug_news_path = NEWS_AUG_TRAIN if split == "train" else NEWS_AUG_TEST
        aug_epairs_path = EPAIRS_AUG_TRAIN if split == "train" else EPAIRS_AUG_TEST

        write_jsonl(result["aug_news"], aug_news_path)
        print(f"  ✓ {os.path.basename(aug_news_path)}: {len(result['aug_news'])} 条")

        write_jsonl(result["aug_epairs"], aug_epairs_path)
        print(f"  ✓ {os.path.basename(aug_epairs_path)}: {len(result['aug_epairs'])} 条")

    # 汇总统计
    summary = {
        "train": train_result["stats"],
        "test": test_result["stats"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 汇总: {SUMMARY_JSON}")

    # 未匹配记录
    all_unresolved = train_result["unresolved"] + test_result["unresolved"]
    if all_unresolved:
        pd.DataFrame(all_unresolved).to_csv(UNRESOLVED_CSV, index=False, encoding="utf-8-sig")
    else:
        with open(UNRESOLVED_CSV, "w", encoding="utf-8") as f:
            f.write("split,sheet_name,excel_row,duplicate_article_id,duplication_id,failure_reason\n")
    print(f"  ✓ 未匹配: {UNRESOLVED_CSV} ({len(all_unresolved)} 条)")

    # 内容冲突
    all_conflicts = train_result["conflicts"] + test_result["conflicts"]
    if all_conflicts:
        pd.DataFrame(all_conflicts).to_csv(CONFLICT_CSV, index=False, encoding="utf-8-sig")
    else:
        with open(CONFLICT_CSV, "w", encoding="utf-8") as f:
            f.write("article_id,row_1,row_2,content_len_1,content_len_2,action\n")
    print(f"  ✓ 冲突: {CONFLICT_CSV} ({len(all_conflicts)} 条)")

    # 相似度
    all_sim = train_result["similarities"] + test_result["similarities"]
    if all_sim:
        pd.DataFrame(all_sim).to_csv(SIMILARITY_CSV, index=False, encoding="utf-8-sig")
    else:
        with open(SIMILARITY_CSV, "w", encoding="utf-8") as f:
            f.write("split,duplicate_article_id,canonical_article_id,title_jaccard,content_3gram_jaccard,content_length_ratio,canonical_content_length,duplicate_content_length\n")
    print(f"  ✓ 相似度: {SIMILARITY_CSV} ({len(all_sim)} 条")

    # 频率
    all_freq = train_result["freq_rows"] + test_result["freq_rows"]
    pd.DataFrame(all_freq).to_csv(AUG_FREQ_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✓ 频率: {AUG_FREQ_CSV} ({len(all_freq)} 行)")

    # 输出统计摘要
    print(f"\n{'='*60}")
    print("统计摘要")
    print(f"{'='*60}")
    for split_name, result in [("train", train_result), ("test", test_result)]:
        s = result["stats"]
        print(f"\n[{split_name}]")
        for k, v in s.items():
            print(f"  {k}: {v}")

    # 相似度统计
    print(f"\n[文本相似度]")
    for split_name in ["train", "test"]:
        sims = [r for r in all_sim if r["split"] == split_name]
        if sims:
            titles = [s["title_jaccard"] for s in sims]
            c3gs = [s["content_3gram_jaccard"] for s in sims]
            lens = [s["content_length_ratio"] for s in sims]
            print(f"  [{split_name}] 标题 Jaccard: mean={np.mean(titles):.4f}, median={np.median(titles):.4f}, min={np.min(titles):.4f}")
            print(f"  [{split_name}] 正文3-gram Jaccard: mean={np.mean(c3gs):.4f}, median={np.median(c3gs):.4f}, min={np.min(c3gs):.4f}")
            print(f"  [{split_name}] 长度比: mean={np.mean(lens):.4f}, median={np.median(lens):.4f}, min={np.min(lens):.4f}")

    # 展示样本
    print(f"\n{'='*60}")
    print("样本展示")
    print(f"{'='*60}")
    print("\n--- 重复新闻级样本 (train 前2条) ---")
    for i, n in enumerate(train_result["dup_news"][:2]):
        display = json.dumps({
            "article_id": n["article_id"],
            "title": n["title"][:80],
            "canonical_article_id": n["canonical_article_id"],
            "is_duplicate_news": n["is_duplicate_news"],
            "label_source": n["label_source"],
            "labels_count": len(n["labels"]),
            "labels": [{"schema_id": l["schema_id"], "event_name": l["event_name"], "emotion": l["emotion"]} for l in n["labels"]],
        }, ensure_ascii=False, indent=2)
        print(f"\n[{i+1}] {display}")

    print("\n--- 重复事件对样本 (train 前3条) ---")
    for i, ep in enumerate(train_result["dup_event_pairs"][:3]):
        display = json.dumps({
            "sample_id": ep["sample_id"],
            "article_id": ep["article_id"],
            "canonical_article_id": ep["canonical_article_id"],
            "schema_id": ep["schema_id"],
            "target_type": ep["target_type"],
            "target_code": ep["target_code"],
            "target_name": ep["target_name"],
            "event_name": ep["event_name"],
            "emotion": ep["emotion"],
            "label": ep["label"],
            "is_duplicate_news": ep["is_duplicate_news"],
            "label_source": ep["label_source"],
        }, ensure_ascii=False, indent=2)
        print(f"\n[{i+1}] {display}")


if __name__ == "__main__":
    main()
