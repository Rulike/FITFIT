#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_negative_pairs.py
------------------------
第四步：基于增强正样本构造三种类型的事件识别负样本，并生成正负训练对数据。

负样本类型：
A. same_target_other_event   — 同实体其他事件
B. same_event_wrong_target   — 同事件名、错实体
C. similar_schema_wrong_target — 相似Schema、错实体
"""

import json
import os
import re
import sys
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════
RANDOM_SEED = 2026
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_JSONL = os.path.join(PROJECT_ROOT, "data", "processed", "event_schema.jsonl")
AUGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "augmented")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "training_pairs")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
SIMILARITY_CSV = os.path.join(REPORTS_DIR, "duplicate_text_similarity.csv")

# ── 输出路径 ──────────────────────────────────────────────
NEG_TRAIN = os.path.join(TRAINING_DIR, "negative_pairs_train.jsonl")
NEG_TEST = os.path.join(TRAINING_DIR, "negative_pairs_test.jsonl")
ALL_TRAIN = os.path.join(TRAINING_DIR, "all_pairs_train.jsonl")
ALL_TEST = os.path.join(TRAINING_DIR, "all_pairs_test.jsonl")
BAL_TRAIN_JSONL = os.path.join(TRAINING_DIR, "train_pairs_balanced.jsonl")
BAL_TEST_JSONL = os.path.join(TRAINING_DIR, "test_pairs_balanced.jsonl")
BAL_TRAIN_XLSX = os.path.join(TRAINING_DIR, "train_pairs_balanced.xlsx")
BAL_TEST_XLSX = os.path.join(TRAINING_DIR, "test_pairs_balanced.xlsx")

SUMMARY_JSON = os.path.join(REPORTS_DIR, "negative_sampling_summary.json")
NEG_FREQ_CSV = os.path.join(REPORTS_DIR, "negative_type_frequency.csv")
SCHEMA_NEG_FREQ_CSV = os.path.join(REPORTS_DIR, "schema_negative_frequency.csv")
NO_CAND_CSV = os.path.join(REPORTS_DIR, "articles_without_negative_candidates.csv")
GROUP_SAMPLE_CSV = os.path.join(REPORTS_DIR, "canonical_group_sampling.csv")
SUSPICIOUS_CSV = os.path.join(REPORTS_DIR, "suspicious_negative_pairs.csv")

EXCEL_COLUMNS = [
    "sample_id", "split", "article_id", "canonical_group_id", "title", "source",
    "target_type", "target_code", "target_name", "event_name", "schema_id",
    "label", "negative_type", "negative_confidence", "schema_similarity",
    "is_duplicate_news", "content", "event_description", "source_positive_sample_id"
]


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


def compute_jaccard(s1, s2, n=1):
    if not s1 or not s2:
        return 0.0
    ngrams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    ngrams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    if not ngrams1 or not ngrams2:
        return 0.0
    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)


def build_schema_similarity(schemas):
    """计算所有Schema之间的相似度矩阵（轻量Jaccard）。"""
    schema_texts = {}
    for sid, s in schemas.items():
        schema_texts[sid] = s["target_name"] + " " + s["event_name"] + " " + s.get("raw_description", "")

    sid_list = list(schemas.keys())
    n = len(sid_list)
    sim_matrix = defaultdict(dict)

    for i in range(n):
        si = sid_list[i]
        si_name = schemas[si]["event_name"]
        si_desc = schema_texts[si]
        for j in range(i+1, n):
            sj = sid_list[j]
            sj_name = schemas[sj]["event_name"]
            sj_desc = schema_texts[sj]
            name_j = compute_jaccard(si_name, sj_name, n=2)
            desc_j = compute_jaccard(si_desc, sj_desc, n=3)
            score = 0.7 * name_j + 0.3 * desc_j
            if score > 0:
                sim_matrix[si][sj] = score
                sim_matrix[sj][si] = score

    return dict(sim_matrix)


# ══════════════════════════════════════════════════════════════
def load_positive_data(split):
    """加载增强正样本并补充 canonical_group_id。"""
    pos_path = os.path.join(AUGMENTED_DIR, f"event_pairs_{split}_augmented.jsonl")
    dup_path = os.path.join(AUGMENTED_DIR, f"duplicate_event_pairs_{split}.jsonl")

    pos_pairs = load_jsonl(pos_path)

    # Build lookup for duplicate info
    dup_info = {}
    for dp in load_jsonl(dup_path):
        aid = dp["article_id"]
        dup_info[aid] = {
            "canonical_article_id": dp.get("canonical_article_id", aid),
            "is_duplicate_news": True,
        }

    # Add canonical_group_id
    for p in pos_pairs:
        aid = p["article_id"]
        if aid in dup_info:
            p["canonical_article_id"] = dup_info[aid]["canonical_article_id"]
            p["is_duplicate_news"] = True
        else:
            p["canonical_article_id"] = aid
            p["is_duplicate_news"] = False
        p["canonical_group_id"] = p["canonical_article_id"]
        # Add null fields for consistency
        p["negative_type"] = None
        p["negative_confidence"] = None
        p["schema_similarity"] = None
        p["source_positive_sample_id"] = None

    return pos_pairs


def build_indexes(pos_pairs):
    """构建正样本索引。"""
    pos_schemas_by_article = defaultdict(set)     # article_id → set of schema_id
    pos_targets_by_article = defaultdict(set)     # article_id → set of (type, code)
    pos_events_by_article = defaultdict(set)      # article_id → set of event_name
    pos_target_names_by_article = defaultdict(set)
    article_target_schemas = defaultdict(lambda: defaultdict(set))  # article → target → set of schemas
    article_info = {}                              # article_id → article metadata

    for p in pos_pairs:
        aid = p["article_id"]
        sid = p["schema_id"]
        tt = p["target_type"]
        tc = p["target_code"]
        en = p["event_name"]
        tn = p["target_name"]

        pos_schemas_by_article[aid].add(sid)
        pos_targets_by_article[aid].add((tt, tc))
        pos_events_by_article[aid].add(en)
        pos_target_names_by_article[aid].add(tn.lower())
        article_target_schemas[aid][(tt, tc)].add(sid)

        if aid not in article_info:
            article_info[aid] = {
                "article_id": aid,
                "title": p.get("title", ""),
                "content": p.get("content", ""),
                "publish_time": p.get("publish_time", ""),
                "source": p.get("source", ""),
                "canonical_group_id": p.get("canonical_group_id", aid),
                "canonical_article_id": p.get("canonical_article_id", aid),
                "is_duplicate_news": p.get("is_duplicate_news", False),
                "positive_pairs": [],
                "positive_targets": set(),
                "positive_schemas": set(),
            }
        article_info[aid]["positive_pairs"].append(p)
        article_info[aid]["positive_targets"].add((tt, tc))
        article_info[aid]["positive_schemas"].add(sid)

    return {
        "pos_schemas_by_article": dict(pos_schemas_by_article),
        "pos_targets_by_article": dict(pos_targets_by_article),
        "pos_events_by_article": dict(pos_events_by_article),
        "pos_target_names_by_article": dict(pos_target_names_by_article),
        "article_target_schemas": dict(article_target_schemas),
        "article_info": dict(article_info),
    }


# ══════════════════════════════════════════════════════════════
def target_name_in_text(target_name, title, content):
    """检查 target_name 是否精确出现在文本中。"""
    text = (title + " " + content).lower()
    return target_name.lower() in text


def event_name_in_title(event_name, title):
    """检查 event_name 是否完整出现在标题中。"""
    return event_name.lower() in title.lower()


# ══════════════════════════════════════════════════════════════
def sample_type_a(article_id, indexes, schemas, max_per_article=2):
    """
    Type A: same_target_other_event
    同实体其他事件 — 同公司/行业，不同事件Schema。
    """
    negs = []
    pos_schemas = indexes["pos_schemas_by_article"].get(article_id, set())
    pos_targets = indexes["pos_targets_by_article"].get(article_id, set())
    info = indexes["article_info"].get(article_id)

    if not info:
        return negs

    candidates = []
    for (tt, tc) in pos_targets:
        # 找到该目标在Schema中的所有事件
        for sid, s in schemas.items():
            if s["target_type"] != tt or str(s["target_code"]) != str(tc):
                continue
            if sid in pos_schemas:
                continue
            candidates.append(sid)

    if not candidates:
        return negs

    random.shuffle(candidates)
    selected = candidates[:max_per_article]

    for sid in selected:
        s = schemas[sid]
        sp_id = info["positive_pairs"][0]["sample_id"] if info["positive_pairs"] else ""
        negs.append({
            "article_id": article_id,
            "schema_id": sid,
            "negative_type": "same_target_other_event",
            "negative_confidence": "medium",
            "schema_similarity": None,
            "source_positive_sample_id": sp_id,
        })
    return negs


def sample_type_b(article_id, indexes, schemas, max_per_article=1):
    """
    Type B: same_event_wrong_target
    同事件名、错目标实体。
    """
    negs = []
    pos_schemas = indexes["pos_schemas_by_article"].get(article_id, set())
    pos_targets = indexes["pos_targets_by_article"].get(article_id, set())
    pos_events = indexes["pos_events_by_article"].get(article_id, set())
    info = indexes["article_info"].get(article_id)
    if not info or not pos_events:
        return negs

    candidates = []
    for en in pos_events:
        for sid, s in schemas.items():
            if s["event_name"] != en:
                continue
            if sid in pos_schemas:
                continue
            tt = s["target_type"]
            tc = str(s["target_code"])
            if (tt, tc) in pos_targets:
                continue
            # 目标名称不能在文本中精确出现
            if target_name_in_text(s["target_name"], info["title"], info["content"]):
                continue
            # target_type 必须与某个正标签一致
            pos_types = {t[0] for t in pos_targets}
            if tt not in pos_types:
                continue
            candidates.append(sid)

    if not candidates:
        return negs

    random.shuffle(candidates)
    selected = candidates[:max_per_article]

    for sid in selected:
        s = schemas[sid]
        sp_id = info["positive_pairs"][0]["sample_id"] if info["positive_pairs"] else ""
        negs.append({
            "article_id": article_id,
            "schema_id": sid,
            "negative_type": "same_event_wrong_target",
            "negative_confidence": "high",
            "schema_similarity": None,
            "source_positive_sample_id": sp_id,
        })
    return negs


def sample_type_c(article_id, indexes, schemas, schema_sim, max_per_article=1):
    """
    Type C: similar_schema_wrong_target
    相似Schema、错误目标实体。
    """
    negs = []
    pos_schemas = indexes["pos_schemas_by_article"].get(article_id, set())
    pos_targets = indexes["pos_targets_by_article"].get(article_id, set())
    info = indexes["article_info"].get(article_id)
    if not info:
        return negs

    # 找到与正Schema最相似的非正Schema（且目标不同）
    candidates = []  # (schema_id, similarity)
    for pos_sid in pos_schemas:
        pos_s = schemas.get(pos_sid)
        if not pos_s:
            continue
        pos_tt = pos_s["target_type"]
        for neg_sid, sim in schema_sim.get(pos_sid, {}).items():
            neg_s = schemas.get(neg_sid)
            if not neg_s:
                continue
            if neg_sid in pos_schemas:
                continue
            tt = neg_s["target_type"]
            tc = str(neg_s["target_code"])
            if (tt, tc) in pos_targets:
                continue
            if target_name_in_text(neg_s["target_name"], info["title"], info["content"]):
                continue
            if tt != pos_tt:
                continue
            candidates.append((neg_sid, sim))

    if not candidates:
        return negs

    # 按相似度降序
    candidates.sort(key=lambda x: -x[1])
    selected = candidates[:max_per_article]

    for sid, sim in selected:
        s = schemas[sid]
        sp_id = info["positive_pairs"][0]["sample_id"] if info["positive_pairs"] else ""
        negs.append({
            "article_id": article_id,
            "schema_id": sid,
            "negative_type": "similar_schema_wrong_target",
            "negative_confidence": "high",
            "schema_similarity": round(float(sim), 4),
            "source_positive_sample_id": sp_id,
        })
    return negs


# ══════════════════════════════════════════════════════════════
def deduplicate_negatives(negatives):
    """按 (article_id, schema_id) 去重，优先级 A > B > C。"""
    type_priority = {"same_target_other_event": 0, "same_event_wrong_target": 1, "similar_schema_wrong_target": 2}
    best = {}
    all_types = defaultdict(list)
    for n in negatives:
        key = (n["article_id"], n["schema_id"])
        all_types[key].append(n["negative_type"])
        if key not in best:
            best[key] = n
        else:
            cur_pri = type_priority.get(best[key]["negative_type"], 99)
            new_pri = type_priority.get(n["negative_type"], 99)
            if new_pri < cur_pri:
                best[key] = n
    # Add all_negative_types
    for key, n in best.items():
        types_list = list(set(all_types[key]))
        if len(types_list) > 1:
            n["all_negative_types"] = types_list
    return list(best.values())


def build_negative_pairs(pos_pairs, indexes, schemas, schema_sim, split):
    """为每篇文章生成三种类型的负样本。"""
    all_negs = []
    article_ids = list(indexes["article_info"].keys())

    for aid in article_ids:
        negs_a = sample_type_a(aid, indexes, schemas, max_per_article=2)
        negs_b = sample_type_b(aid, indexes, schemas, max_per_article=1)
        negs_c = sample_type_c(aid, indexes, schemas, schema_sim, max_per_article=1)
        all_negs.extend(negs_a + negs_b + negs_c)

    # 去重
    all_negs = deduplicate_negatives(all_negs)

    # 构建完整记录
    info_map = indexes["article_info"]
    negative_pairs = []
    for n in all_negs:
        aid = n["article_id"]
        sid = n["schema_id"]
        info = info_map.get(aid, {})
        s = schemas.get(sid, {})

        pair = {
            "sample_id": f"{split}_neg_{aid}_{sid}",
            "split": split,
            "article_id": aid,
            "canonical_article_id": info.get("canonical_article_id", aid),
            "canonical_group_id": info.get("canonical_group_id", aid),
            "title": info.get("title", ""),
            "publish_time": info.get("publish_time", ""),
            "source": info.get("source", ""),
            "content": info.get("content", ""),
            "is_duplicate_news": info.get("is_duplicate_news", False),
            "schema_id": sid,
            "target_type": s.get("target_type", ""),
            "target_code": str(s.get("target_code", "")),
            "target_name": s.get("target_name", ""),
            "event_name": s.get("event_name", ""),
            "event_description": s.get("raw_description", ""),
            "label": 0,
            "negative_type": n["negative_type"],
            "negative_confidence": n["negative_confidence"],
            "schema_similarity": n.get("schema_similarity"),
            "source_positive_sample_id": n.get("source_positive_sample_id", ""),
        }
        # all_negative_types
        if "all_negative_types" in n:
            pair["all_negative_types"] = n["all_negative_types"]
        negative_pairs.append(pair)

    return negative_pairs


# ══════════════════════════════════════════════════════════════
def check_suspicious(neg_pairs, pos_pairs, schemas):
    """检查可疑负样本。"""
    suspicious = []
    clean_negs = []

    pos_keys = {(p["article_id"], p["schema_id"]) for p in pos_pairs}

    for n in neg_pairs:
        aid = n["article_id"]
        sid = n["schema_id"]
        reasons = []

        # 1. target_name 出现在标题或正文
        if target_name_in_text(n["target_name"], n["title"], n["content"]):
            reasons.append("target_name_in_text")

        # 2. event_name 完整出现在标题
        if event_name_in_title(n["event_name"], n["title"]):
            reasons.append("event_name_in_title")

        # 3. 负Schema与正Schema相同（article_id, schema_id冲突）
        if (aid, sid) in pos_keys:
            reasons.append("positive_negative_conflict")

        # 4. 候选目标是该文章真实正目标（由code+type确认）
        # Already checked during generation, but double-check
        pos_info = None
        for p in pos_pairs:
            if p["article_id"] == aid:
                pos_info = p
                break

        # 5. same_target 中的 target_code 应与正目标一致
        if n["negative_type"] == "same_target_other_event":
            pos_targets_for_article = set()
            for p in pos_pairs:
                if p["article_id"] == aid:
                    pos_targets_for_article.add((p["target_type"], str(p["target_code"])))
            if (n["target_type"], str(n["target_code"])) not in pos_targets_for_article:
                reasons.append("same_target_mismatch")

        if reasons:
            suspicious.append({**n, "suspicious_reasons": ";".join(reasons)})
        else:
            clean_negs.append(n)

    return clean_negs, suspicious


# ══════════════════════════════════════════════════════════════
def build_balanced(positive_pairs, negative_pairs, target_ratio=3.0, max_dup_per_group=2):
    """
    构建平衡训练集：
    - 正负比例约 1:3
    - 每个 canonical_group_id 最多 1 主新闻 + max_dup_per_group 转载新闻（正负合并计算）
    """
    # 加载相似度数据用于优先选择差异大的转载
    sim_data = {}
    if os.path.exists(SIMILARITY_CSV):
        df_sim = pd.read_csv(SIMILARITY_CSV)
        for _, row in df_sim.iterrows():
            sim_data[row["duplicate_article_id"]] = row.get("content_3gram_jaccard", 1.0)

    # 合并正负样本按 canonical_group_id 分组
    all_items = list(positive_pairs) + list(negative_pairs)

    groups = defaultdict(lambda: {"main": [], "dup": []})
    for item in all_items:
        cgid = item.get("canonical_group_id", item["article_id"])
        if item.get("is_duplicate_news"):
            groups[cgid]["dup"].append(item)
        else:
            groups[cgid]["main"].append(item)

    # 对每个 group，限制总共最多 1 main + max_dup_per_group dups
    selected_all = []
    for cgid, g in groups.items():
        selected_all.extend(g["main"])
        dups = g["dup"]
        if len(dups) > max_dup_per_group:
            dups_sorted = sorted(dups, key=lambda d: sim_data.get(d["article_id"], 1.0))
            dups = [d for d in dups_sorted if sim_data.get(d["article_id"], 1.0) > 0.05][:max_dup_per_group]
        selected_all.extend(dups)

    # 分离正负
    selected_pos = [p for p in selected_all if p.get("label") == 1]
    selected_neg = [p for p in selected_all if p.get("label") == 0]

    # 按 1:3 比例采样负样本
    target_neg_count = min(len(selected_pos) * target_ratio, len(selected_neg))
    target_neg_count = int(target_neg_count)

    # 按优先级排列负样本
    neg_priority = {"same_target_other_event": 0, "same_event_wrong_target": 1, "similar_schema_wrong_target": 2}
    selected_neg_sorted = sorted(
        selected_neg,
        key=lambda n: (neg_priority.get(n.get("negative_type", ""), 9), random.random())
    )

    balanced_neg = selected_neg_sorted[:target_neg_count]
    balanced = selected_pos + balanced_neg

    # Canonical group stats for balanced set
    cg_stats = defaultdict(lambda: {"main": 0, "dup": 0})
    for p in balanced:
        cgid = p.get("canonical_group_id", p["article_id"])
        if p.get("is_duplicate_news"):
            cg_stats[cgid]["dup"] += 1
        else:
            cg_stats[cgid]["main"] += 1

    return balanced, dict(cg_stats), len(selected_pos), len(balanced_neg)


# ══════════════════════════════════════════════════════════════
def compute_statistics(pos_pairs, neg_pairs, balanced, split, cg_stats):
    neg_types = Counter(n["negative_type"] for n in neg_pairs)
    articles_no_neg = set()
    # Find articles with no negative candidates
    article_ids_with_negs = set(n["article_id"] for n in neg_pairs)
    all_article_ids = set(p["article_id"] for p in pos_pairs)
    articles_no_neg = all_article_ids - article_ids_with_negs

    pos_count = len(pos_pairs)
    neg_count = len(neg_pairs)
    ratio = f"1:{neg_count/pos_count:.2f}" if pos_count > 0 else "N/A"

    bal_pos = sum(1 for p in balanced if p["label"] == 1)
    bal_neg = sum(1 for p in balanced if p["label"] == 0)
    bal_ratio = f"1:{bal_neg/bal_pos:.2f}" if bal_pos > 0 else "N/A"

    return {
        "positive_pair_count": pos_count,
        "negative_pair_count": neg_count,
        "same_target_other_event_count": neg_types.get("same_target_other_event", 0),
        "same_event_wrong_target_count": neg_types.get("same_event_wrong_target", 0),
        "similar_schema_wrong_target_count": neg_types.get("similar_schema_wrong_target", 0),
        "positive_negative_ratio": ratio,
        "unique_article_count": len(all_article_ids),
        "unique_canonical_group_count": len(cg_stats),
        "articles_without_negative_candidates": len(articles_no_neg),
        "balanced_positive_count": bal_pos,
        "balanced_negative_count": bal_neg,
        "balanced_positive_negative_ratio": bal_ratio,
    }


def compute_schema_frequency(pos_pairs, neg_pairs, split):
    """每类Schema的正负样本分布。"""
    pos_counter = Counter(p["schema_id"] for p in pos_pairs)
    neg_counter = Counter(n["schema_id"] for n in neg_pairs)
    all_sids = set(pos_counter.keys()) | set(neg_counter.keys())
    rows = []
    for sid in sorted(all_sids):
        s_info = pos_pairs[0] if pos_pairs else {}
        for p in pos_pairs:
            if p["schema_id"] == sid:
                s_info = p
                break
        if not s_info.get("target_name"):
            for n in neg_pairs:
                if n["schema_id"] == sid:
                    s_info = n
                    break
        rows.append({
            "split": split,
            "schema_id": sid,
            "target_name": s_info.get("target_name", ""),
            "event_name": s_info.get("event_name", ""),
            "positive_count": pos_counter.get(sid, 0),
            "negative_count": neg_counter.get(sid, 0),
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


# ══════════════════════════════════════════════════════════════
def process_split(split, schemas, schema_sim):
    print(f"\n{'='*60}")
    print(f"处理 {split} 负样本构造")
    print(f"{'='*60}")

    pos_pairs = load_positive_data(split)
    print(f"  正样本: {len(pos_pairs)} 条")

    indexes = build_indexes(pos_pairs)
    print(f"  唯一文章数: {len(indexes['article_info'])}")
    print(f"  唯一 canonical group: {len(set(a['canonical_group_id'] for a in indexes['article_info'].values()))}")

    # 生成负样本
    neg_pairs = build_negative_pairs(pos_pairs, indexes, schemas, schema_sim, split)
    print(f"  生成负样本: {len(neg_pairs)} 条")

    # 去重
    neg_pairs_clean, suspicious = check_suspicious(neg_pairs, pos_pairs, schemas)
    print(f"  清除可疑: {len(suspicious)} 条 → 剩余: {len(neg_pairs_clean)} 条")

    # 全量合并
    all_pairs = pos_pairs + neg_pairs_clean

    # 平衡
    balanced, cg_stats, bal_pos, bal_neg = build_balanced(pos_pairs, neg_pairs_clean)
    print(f"  平衡集: {len(balanced)} 条 (正:{bal_pos}, 负:{bal_neg})")

    # 统计
    stats = compute_statistics(pos_pairs, neg_pairs_clean, balanced, split, cg_stats)
    schema_freq = compute_schema_frequency(pos_pairs, neg_pairs_clean, split)

    return {
        "pos_pairs": pos_pairs,
        "neg_pairs": neg_pairs_clean,
        "all_pairs": all_pairs,
        "balanced": balanced,
        "suspicious": suspicious,
        "stats": stats,
        "schema_freq": schema_freq,
        "cg_stats": cg_stats,
        "indexes": indexes,
    }


def main():
    print("=" * 60)
    print("事件识别负样本构造工具")
    print("=" * 60)

    # 加载 Schema
    print("\n[1] 加载 Schema 并计算相似度...")
    schemas = load_schemas()
    print(f"  加载 {len(schemas)} 个 Schema")
    schema_sim = build_schema_similarity(schemas)
    sim_entries = sum(len(v) for v in schema_sim.values())
    print(f"  Schema相似度条目: {sim_entries}")

    # 处理 train 和 test
    train_res = process_split("train", schemas, schema_sim)
    test_res = process_split("test", schemas, schema_sim)

    # 输出
    print(f"\n{'='*60}")
    print("写入输出文件...")
    print(f"{'='*60}")
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    for result, split in [(train_res, "train"), (test_res, "test")]:
        neg_path = NEG_TRAIN if split == "train" else NEG_TEST
        all_path = ALL_TRAIN if split == "train" else ALL_TEST
        bal_jsonl = BAL_TRAIN_JSONL if split == "train" else BAL_TEST_JSONL
        bal_xlsx = BAL_TRAIN_XLSX if split == "train" else BAL_TEST_XLSX

        write_jsonl(result["neg_pairs"], neg_path)
        print(f"  ✓ {os.path.basename(neg_path)}: {len(result['neg_pairs'])} 条")

        write_jsonl(result["all_pairs"], all_path)
        print(f"  ✓ {os.path.basename(all_path)}: {len(result['all_pairs'])} 条")

        write_jsonl(result["balanced"], bal_jsonl)
        print(f"  ✓ {os.path.basename(bal_jsonl)}: {len(result['balanced'])} 条")

        write_excel(result["balanced"], bal_xlsx, EXCEL_COLUMNS)
        print(f"  ✓ {os.path.basename(bal_xlsx)}: {len(result['balanced'])} 条")

    # 汇总统计
    summary = {
        "train": train_res["stats"],
        "test": test_res["stats"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {SUMMARY_JSON}")

    # 负样本类型频率
    freq_rows = []
    for split, result in [("train", train_res), ("test", test_res)]:
        type_counts = Counter(n["negative_type"] for n in result["neg_pairs"])
        for nt, cnt in type_counts.items():
            freq_rows.append({"split": split, "negative_type": nt, "count": cnt})
    pd.DataFrame(freq_rows).to_csv(NEG_FREQ_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✓ {NEG_FREQ_CSV}")

    # Schema频率
    all_schema_freq = train_res["schema_freq"] + test_res["schema_freq"]
    pd.DataFrame(all_schema_freq).to_csv(SCHEMA_NEG_FREQ_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✓ {SCHEMA_NEG_FREQ_CSV}")

    # 无候选文章
    no_cand_rows = []
    for split, result in [("train", train_res), ("test", test_res)]:
        idx = result["indexes"]
        all_aids = set(idx["article_info"].keys())
        neg_aids = set(n["article_id"] for n in result["neg_pairs"])
        for aid in sorted(all_aids - neg_aids):
            info = idx["article_info"][aid]
            no_cand_rows.append({
                "split": split,
                "article_id": aid,
                "title": info["title"][:100],
                "positive_label_count": len(info["positive_pairs"]),
            })
    if no_cand_rows:
        pd.DataFrame(no_cand_rows).to_csv(NO_CAND_CSV, index=False, encoding="utf-8-sig")
    else:
        with open(NO_CAND_CSV, "w", encoding="utf-8") as f:
            f.write("split,article_id,title,positive_label_count\n")
    print(f"  ✓ {NO_CAND_CSV} ({len(no_cand_rows)} 条)")

    # Canonical group 采样
    cg_rows = []
    for split, result in [("train", train_res), ("test", test_res)]:
        all_cg = set()
        for p in result["pos_pairs"]:
            all_cg.add(p.get("canonical_group_id", p["article_id"]))
        for cgid in sorted(all_cg):
            cg_stat = result["cg_stats"].get(cgid, {"main": 0, "dup": 0})
            # Count original
            orig_main = sum(1 for p in result["pos_pairs"] if p.get("canonical_group_id", p["article_id"]) == cgid and not p.get("is_duplicate_news"))
            orig_dup = sum(1 for p in result["pos_pairs"] if p.get("canonical_group_id", p["article_id"]) == cgid and p.get("is_duplicate_news"))
            cg_rows.append({
                "split": split,
                "canonical_group_id": cgid,
                "main_article_count": orig_main,
                "duplicate_article_count": orig_dup,
                "selected_main_count": cg_stat.get("main", 0) if cg_stat.get("main", 0) > 0 else min(orig_main, 1),
                "selected_duplicate_count": cg_stat.get("dup", 0),
                "total_selected_count": min(orig_main, 1) + cg_stat.get("dup", 0),
            })
    pd.DataFrame(cg_rows).to_csv(GROUP_SAMPLE_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✓ {GROUP_SAMPLE_CSV}")

    # 可疑负样本
    all_suspicious = train_res["suspicious"] + test_res["suspicious"]
    if all_suspicious:
        sus_df = pd.DataFrame(all_suspicious)
        # Remove content field to keep size manageable
        if "content" in sus_df.columns:
            sus_df["content"] = sus_df["content"].apply(lambda x: (str(x)[:200] if x else ""))
        sus_df.to_csv(SUSPICIOUS_CSV, index=False, encoding="utf-8-sig")
    else:
        with open(SUSPICIOUS_CSV, "w", encoding="utf-8") as f:
            f.write("split,sample_id,article_id,schema_id,negative_type,suspicious_reasons\n")
    print(f"  ✓ {SUSPICIOUS_CSV} ({len(all_suspicious)} 条)")

    # 摘要
    print(f"\n{'='*60}")
    print("统计摘要")
    print(f"{'='*60}")
    for split, s_data in [("train", summary["train"]), ("test", summary["test"])]:
        print(f"\n[{split}]")
        for k, v in s_data.items():
            print(f"  {k}: {v}")

    # 展示样本
    print(f"\n{'='*60}")
    print("负样本展示")
    print(f"{'='*60}")

    for neg_type, label in [("same_target_other_event", "同实体其他事件"),
                             ("same_event_wrong_target", "同事件错误实体"),
                             ("similar_schema_wrong_target", "相似Schema错误实体")]:
        samples = [n for n in train_res["neg_pairs"] if n["negative_type"] == neg_type][:2]
        print(f"\n--- {label} ({neg_type}) ---")
        for i, s in enumerate(samples):
            display = json.dumps({
                "sample_id": s["sample_id"],
                "article_id": s["article_id"],
                "schema_id": s["schema_id"],
                "target_code": s["target_code"],
                "target_name": s["target_name"],
                "event_name": s["event_name"],
                "negative_type": s["negative_type"],
                "negative_confidence": s["negative_confidence"],
                "schema_similarity": s.get("schema_similarity"),
            }, ensure_ascii=False, indent=2)
            print(f"\n[{i+1}] {display}")


if __name__ == "__main__":
    main()
