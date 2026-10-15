#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_train_validation.py
-------------------------
第五步：按 canonical_group_id 分组划分内部训练集（~80%）和验证集（~20%），
杜绝近重复泄漏。同时生成 model_input_text 和 minimal 格式。
"""

import json
import os
import hashlib
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════
RANDOM_SEED = 2026
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_JSONL = os.path.join(PROJECT_ROOT, "data", "processed", "event_schema.jsonl")

BALANCED_TRAIN = os.path.join(PROJECT_ROOT, "data", "processed", "training_pairs", "train_pairs_balanced.jsonl")
OFFICIAL_TEST = os.path.join(PROJECT_ROOT, "data", "processed", "training_pairs", "test_pairs_balanced.jsonl")

MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "model_data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ── 输出 ──────────────────────────────────────────────────
TRAIN_JSONL = os.path.join(MODEL_DIR, "event_match_train.jsonl")
VAL_JSONL = os.path.join(MODEL_DIR, "event_match_validation.jsonl")
TRAIN_XLSX = os.path.join(MODEL_DIR, "event_match_train.xlsx")
VAL_XLSX = os.path.join(MODEL_DIR, "event_match_validation.xlsx")
OFFICIAL_TEST_JSONL = os.path.join(MODEL_DIR, "event_match_official_test.jsonl")
TRAIN_MIN = os.path.join(MODEL_DIR, "event_match_train_minimal.jsonl")
VAL_MIN = os.path.join(MODEL_DIR, "event_match_validation_minimal.jsonl")

SUMMARY_JSON = os.path.join(REPORTS_DIR, "train_validation_split_summary.json")
SCHEMA_DIST_CSV = os.path.join(REPORTS_DIR, "train_validation_schema_distribution.csv")
NEG_TYPE_DIST_CSV = os.path.join(REPORTS_DIR, "train_validation_negative_type_distribution.csv")
MISSING_SCHEMA_CSV = os.path.join(REPORTS_DIR, "schemas_missing_in_validation.csv")
LEAKAGE_CSV = os.path.join(REPORTS_DIR, "train_validation_leakage.csv")

EXCEL_COLUMNS = [
    "sample_id", "internal_split", "article_id", "canonical_group_id", "title", "source",
    "target_type", "target_code", "target_name", "event_name", "schema_id",
    "label", "negative_type", "negative_confidence", "schema_similarity",
    "is_duplicate_news", "content", "event_description", "source_positive_sample_id"
]

TARGET_VAL_RATIO = 0.2


# ══════════════════════════════════════════════════════════════
def load_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_schemas():
    schemas = {}
    with open(SCHEMA_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                schemas[r["schema_id"]] = r
    return schemas


def text_hash(title, content):
    """Normalize and hash title + content."""
    text = (title + "\n" + content).strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = " ".join(text.split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_model_input(rec):
    return (
        f"目标类型：{rec.get('target_type', '')}\n"
        f"目标名称：{rec.get('target_name', '')}\n"
        f"目标代码：{rec.get('target_code', '')}\n"
        f"候选事件：{rec.get('event_name', '')}\n"
        f"事件定义：{rec.get('event_description', '')}\n"
        f"新闻标题：{rec.get('title', '')}\n"
        f"新闻正文：{rec.get('content', '')}"
    )


# ══════════════════════════════════════════════════════════════
def aggregate_groups(records):
    """按 canonical_group_id 聚合事件对，并预计算每组统计。"""
    groups = defaultdict(list)
    for r in records:
        cgid = r.get("canonical_group_id") or r["article_id"]
        groups[cgid].append(r)

    group_stats = {}
    for cgid, items in groups.items():
        pos_count = sum(1 for i in items if i["label"] == 1)
        neg_count = sum(1 for i in items if i["label"] == 0)
        pos_schemas = set(i["schema_id"] for i in items if i["label"] == 1)
        neg_types = Counter(i.get("negative_type", "") for i in items if i["label"] == 0)
        is_dup = any(i.get("is_duplicate_news") for i in items)
        has_main = any(not i.get("is_duplicate_news") for i in items)
        group_stats[cgid] = {
            "total": len(items),
            "pos_count": pos_count,
            "neg_count": neg_count,
            "pos_schemas": pos_schemas,
            "neg_types": neg_types,
            "is_dup": is_dup,
            "has_main": has_main,
            "items": items,
        }
    return dict(groups), group_stats


def deterministic_group_split(group_stats, schema_groups, target_val_ratio=0.2):
    """
    贪心算法：优先将稀有Schema组分配到验证集，再逐组分配接近目标比例。
    """
    all_groups = list(group_stats.keys())
    random.shuffle(all_groups)

    val_groups = set()
    train_groups = set()

    # Phase 1: 稀有Schema（<5组）优先进入验证集以确保覆盖
    for sid, schema_group_set in schema_groups.items():
        if len(schema_group_set) < 5:
            for cgid in schema_group_set:
                if cgid not in val_groups and cgid not in train_groups:
                    val_groups.add(cgid)

    # Phase 2: 对于>=5组的Schema，分配约20%的组到验证集
    for sid, schema_group_set in schema_groups.items():
        if len(schema_group_set) >= 5:
            schema_grps = [cg for cg in schema_group_set if cg not in val_groups and cg not in train_groups]
            random.shuffle(schema_grps)
            n_val = max(1, int(len(schema_group_set) * target_val_ratio))
            for cgid in schema_grps[:n_val]:
                if cgid not in train_groups:
                    val_groups.add(cgid)

    # Phase 3: 剩余未分配组
    remaining = [cg for cg in all_groups if cg not in val_groups and cg not in train_groups]
    random.shuffle(remaining)

    # 计算当前验证集比例
    total_groups = len(all_groups)
    current_val = len(val_groups)
    target_val_count = int(total_groups * target_val_ratio)

    # 如果验证集还不够，从剩余中添加
    needed = max(0, target_val_count - current_val)
    for cgid in remaining[:needed]:
        val_groups.add(cgid)

    # 其余全部进训练集
    for cgid in all_groups:
        if cgid not in val_groups:
            train_groups.add(cgid)

    return train_groups, val_groups


# ══════════════════════════════════════════════════════════════
def check_leakage(train_records, val_records):
    """全面泄漏检查。"""
    train_cg = {r.get("canonical_group_id", r["article_id"]) for r in train_records}
    val_cg = {r.get("canonical_group_id", r["article_id"]) for r in val_records}

    train_aids = {r["article_id"] for r in train_records}
    val_aids = {r["article_id"] for r in val_records}

    train_sids = {r["sample_id"] for r in train_records}
    val_sids = {r["sample_id"] for r in val_records}

    train_hashes = {text_hash(r["title"], r["content"]) for r in train_records}
    val_hashes = {text_hash(r["title"], r["content"]) for r in val_records}

    leakage = {
        "canonical_group_overlap_count": len(train_cg & val_cg),
        "article_id_overlap_count": len(train_aids & val_aids),
        "sample_id_overlap_count": len(train_sids & val_sids),
        "exact_text_hash_overlap_count": len(train_hashes & val_hashes),
    }

    # 主新闻-转载新闻跨split检查
    train_aid_set = train_aids
    val_aid_set = val_aids
    cross_dup = []
    for r in train_records:
        cid = r.get("canonical_article_id")
        if cid and cid in val_aid_set and cid != r["article_id"]:
            cross_dup.append(r["sample_id"])
    for r in val_records:
        cid = r.get("canonical_article_id")
        if cid and cid in train_aid_set and cid != r["article_id"]:
            cross_dup.append(r["sample_id"])
    leakage["cross_split_canonical_refs"] = len(cross_dup)

    return leakage


# ══════════════════════════════════════════════════════════════
def compute_statistics(records, label):
    """计算 split 统计。"""
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    articles = set(r["article_id"] for r in records)
    cg = set(r.get("canonical_group_id", r["article_id"]) for r in records)
    main_count = sum(1 for r in records if not r.get("is_duplicate_news"))
    dup_count = sum(1 for r in records if r.get("is_duplicate_news"))

    ratio = f"1:{len(neg)/len(pos):.2f}" if pos else "N/A"
    return {
        "pair_count": len(records),
        "positive_count": len(pos),
        "negative_count": len(neg),
        "positive_negative_ratio": ratio,
        "article_count": len(articles),
        "canonical_group_count": len(cg),
        "main_article_count": main_count,
        "duplicate_article_count": dup_count,
    }


def build_schema_distribution(train_records, val_records):
    """各Schema在train/val中的正负分布。"""
    schema_stats = defaultdict(lambda: {
        "target_type": "", "target_name": "", "event_name": "",
        "total_positive_count": 0, "total_negative_count": 0,
        "train_positive_count": 0, "val_positive_count": 0,
        "train_negative_count": 0, "val_negative_count": 0,
    })
    for r in train_records + val_records:
        sid = r["schema_id"]
        s = schema_stats[sid]
        s["target_type"] = r.get("target_type", "")
        s["target_name"] = r.get("target_name", "")
        s["event_name"] = r.get("event_name", "")
        if r["label"] == 1:
            s["total_positive_count"] += 1
        else:
            s["total_negative_count"] += 1

    train_sids = set(r["schema_id"] for r in train_records)
    val_sids = set(r["schema_id"] for r in val_records)

    for r in train_records:
        sid = r["schema_id"]
        if r["label"] == 1:
            schema_stats[sid]["train_positive_count"] += 1
        else:
            schema_stats[sid]["train_negative_count"] += 1

    for r in val_records:
        sid = r["schema_id"]
        if r["label"] == 1:
            schema_stats[sid]["val_positive_count"] += 1
        else:
            schema_stats[sid]["val_negative_count"] += 1

    rows = []
    for sid, s in sorted(schema_stats.items()):
        rows.append({"schema_id": sid, **s})
    return rows


def build_negative_type_distribution(train_records, val_records):
    """三类负样本分布。"""
    all_neg = [r for r in train_records + val_records if r["label"] == 0]
    train_neg = [r for r in train_records if r["label"] == 0]
    val_neg = [r for r in val_records if r["label"] == 0]

    total_counts = Counter(n.get("negative_type", "unknown") for n in all_neg)
    train_counts = Counter(n.get("negative_type", "unknown") for n in train_neg)
    val_counts = Counter(n.get("negative_type", "unknown") for n in val_neg)

    rows = []
    total_all = sum(total_counts.values())
    for nt in sorted(total_counts.keys()):
        tc = total_counts[nt]
        trc = train_counts.get(nt, 0)
        vc = val_counts.get(nt, 0)
        rows.append({
            "negative_type": nt,
            "total_count": tc,
            "train_count": trc,
            "validation_count": vc,
            "train_ratio": round(trc / tc, 4) if tc > 0 else 0,
            "validation_ratio": round(vc / tc, 4) if tc > 0 else 0,
        })
    return rows


def build_missing_schema_report(val_records, schema_groups):
    """验证集未覆盖的Schema报告。"""
    all_sids = set(schema_groups.keys())
    val_sids = set(r["schema_id"] for r in val_records)
    missing = all_sids - val_sids

    rows = []
    for sid in sorted(missing):
        groups_set = schema_groups.get(sid, set())
        total_pos_groups = len(groups_set)
        reason = "too_few_positive_groups" if total_pos_groups < 5 else "group_constraint_prevents_coverage"
        rows.append({
            "schema_id": sid,
            "target_type": "",
            "target_name": "",
            "event_name": "",
            "total_positive_pair_count": 0,
            "total_positive_group_count": total_pos_groups,
            "reason": reason,
        })
    return rows


# ══════════════════════════════════════════════════════════════
def write_jsonl(records, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_excel(records, filepath, columns):
    df = pd.DataFrame(records, columns=columns)
    df = df[columns]
    df.to_excel(filepath, index=False, engine="openpyxl")


def write_minimal(records, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            minimal = {
                "sample_id": rec["sample_id"],
                "canonical_group_id": rec.get("canonical_group_id", rec["article_id"]),
                "text": rec["model_input_text"],
                "label": rec["label"],
            }
            f.write(json.dumps(minimal, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("内部训练/验证集分组划分工具")
    print("=" * 60)

    # 加载
    print("\n[1] 加载数据...")
    records = load_jsonl(BALANCED_TRAIN)
    print(f"  balanced train: {len(records)} 条")

    official_test = load_jsonl(OFFICIAL_TEST)
    print(f"  official test: {len(official_test)} 条（仅复制，不参与划分）")

    # 分组聚合
    print("\n[2] 按 canonical_group_id 聚合...")
    groups, group_stats = aggregate_groups(records)
    print(f"  canonical groups: {len(groups)}")

    # Schema→groups索引
    schema_groups = defaultdict(set)
    for r in records:
        if r["label"] == 1:
            cgid = r.get("canonical_group_id", r["article_id"])
            schema_groups[r["schema_id"]].add(cgid)
    print(f"  Schema数: {len(schema_groups)}")

    # 划分
    print("\n[3] 贪心分组划分...")
    train_groups, val_groups = deterministic_group_split(group_stats, schema_groups, TARGET_VAL_RATIO)
    print(f"  train groups: {len(train_groups)}")
    print(f"  validation groups: {len(val_groups)}")
    print(f"  val ratio: {len(val_groups) / len(groups):.4f}")

    # 收集记录
    train_records = []
    val_records = []
    for cgid in train_groups:
        for r in group_stats[cgid]["items"]:
            r = dict(r)
            r["internal_split"] = "train"
            r["split_seed"] = RANDOM_SEED
            r["model_input_text"] = build_model_input(r)
            train_records.append(r)

    for cgid in val_groups:
        for r in group_stats[cgid]["items"]:
            r = dict(r)
            r["internal_split"] = "validation"
            r["split_seed"] = RANDOM_SEED
            r["model_input_text"] = build_model_input(r)
            val_records.append(r)

    print(f"  train records: {len(train_records)}")
    print(f"  validation records: {len(val_records)}")
    print(f"  total: {len(train_records) + len(val_records)} (vs input {len(records)})")

    # 泄漏检查
    print("\n[4] 数据泄漏检查...")
    leakage = check_leakage(train_records, val_records)
    for k, v in leakage.items():
        status = "✓" if v == 0 else "✗ LEAK!"
        print(f"  {status} {k}: {v}")

    if any(v > 0 for v in leakage.values()):
        print("\n  ❌ 检测到数据泄漏！请检查划分逻辑。")
        sys.exit(1)

    # 统计
    print("\n[5] 生成统计...")
    train_stats = compute_statistics(train_records, "train")
    val_stats = compute_statistics(val_records, "validation")
    schema_dist = build_schema_distribution(train_records, val_records)
    neg_type_dist = build_negative_type_distribution(train_records, val_records)
    missing_schemas = build_missing_schema_report(val_records, schema_groups)

    for name, s in [("train", train_stats), ("validation", val_stats)]:
        print(f"  [{name}] pairs={s['pair_count']}, pos={s['positive_count']}, "
              f"neg={s['negative_count']}, ratio={s['positive_negative_ratio']}, "
              f"groups={s['canonical_group_count']}")

    print(f"  验证集未覆盖Schema: {len(missing_schemas)}")

    # 泄漏详细记录
    leakage_rows = []
    if leakage["exact_text_hash_overlap_count"] > 0:
        train_hashes = {text_hash(r["title"], r["content"]): r["sample_id"] for r in train_records}
        val_hashes = {text_hash(r["title"], r["content"]): r["sample_id"] for r in val_records}
        for h in set(train_hashes) & set(val_hashes):
            leakage_rows.append({
                "hash": h[:16],
                "train_sample_id": train_hashes[h],
                "validation_sample_id": val_hashes[h],
            })

    # 输出
    print("\n[6] 写入输出文件...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    write_jsonl(train_records, TRAIN_JSONL)
    print(f"  ✓ {os.path.basename(TRAIN_JSONL)}: {len(train_records)} 条")

    write_jsonl(val_records, VAL_JSONL)
    print(f"  ✓ {os.path.basename(VAL_JSONL)}: {len(val_records)} 条")

    write_excel(train_records, TRAIN_XLSX, EXCEL_COLUMNS)
    write_excel(val_records, VAL_XLSX, EXCEL_COLUMNS)
    print(f"  ✓ Excel文件已生成")

    write_minimal(train_records, TRAIN_MIN)
    write_minimal(val_records, VAL_MIN)
    print(f"  ✓ Minimal文件已生成")

    # 官方test原样复制
    test_out = []
    for r in official_test:
        r = dict(r)
        r["internal_split"] = "official_test"
        r["split_seed"] = RANDOM_SEED
        r["model_input_text"] = build_model_input(r)
        test_out.append(r)
    write_jsonl(test_out, OFFICIAL_TEST_JSONL)
    print(f"  ✓ {os.path.basename(OFFICIAL_TEST_JSONL)}: {len(test_out)} 条（原样复制）")

    # 报告
    summary = {
        "seed": RANDOM_SEED,
        "target_validation_group_ratio": TARGET_VAL_RATIO,
        "train": train_stats,
        "validation": val_stats,
        "leakage": leakage,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {SUMMARY_JSON}")

    pd.DataFrame(schema_dist).to_csv(SCHEMA_DIST_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(neg_type_dist).to_csv(NEG_TYPE_DIST_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_schemas).to_csv(MISSING_SCHEMA_CSV, index=False, encoding="utf-8-sig")

    if leakage_rows:
        pd.DataFrame(leakage_rows).to_csv(LEAKAGE_CSV, index=False, encoding="utf-8-sig")
    else:
        with open(LEAKAGE_CSV, "w", encoding="utf-8") as f:
            f.write("hash,train_sample_id,validation_sample_id\n")

    print(f"  ✓ 所有报告已生成")

    # 摘要
    print(f"\n{'='*60}")
    print("划分完成")
    print(f"{'='*60}")
    print(f"  Train: {train_stats['pair_count']} pairs, {train_stats['canonical_group_count']} groups")
    print(f"  Validation: {val_stats['pair_count']} pairs, {val_stats['canonical_group_count']} groups")
    print(f"  Val group ratio: {val_stats['canonical_group_count'] / (train_stats['canonical_group_count'] + val_stats['canonical_group_count']):.4f}")
    print(f"  Leakage: all zero ✓" if all(v == 0 for v in leakage.values()) else f"  Leakage: {leakage}")

    # 展示 minimal 样本
    print(f"\n--- Minimal Train 前2条 ---")
    mt = load_jsonl(TRAIN_MIN)
    for i, r in enumerate(mt[:2]):
        display = {"sample_id": r["sample_id"], "canonical_group_id": r["canonical_group_id"],
                   "text": r["text"][:150] + "...", "label": r["label"]}
        print(f"[{i+1}] {json.dumps(display, ensure_ascii=False, indent=2)}")

    print(f"\n--- Minimal Validation 前2条 ---")
    mv = load_jsonl(VAL_MIN)
    for i, r in enumerate(mv[:2]):
        display = {"sample_id": r["sample_id"], "canonical_group_id": r["canonical_group_id"],
                   "text": r["text"][:150] + "...", "label": r["label"]}
        print(f"[{i+1}] {json.dumps(display, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
