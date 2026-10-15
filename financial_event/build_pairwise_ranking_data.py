#!/usr/bin/env python3
"""build_pairwise_ranking_data.py — 构造安全的Pairwise训练/验证数据。只使用 same_target_other_event 作为排序负例。"""

import json, os, sys, random
from collections import defaultdict
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 2026
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TRAIN_FILE = os.path.join(PROJECT_ROOT, "data/processed/model_data/event_match_train.jsonl")
VAL_FILE = os.path.join(PROJECT_ROOT, "data/processed/model_data/event_match_validation.jsonl")
PAIRWISE_DIR = os.path.join(PROJECT_ROOT, "data/processed/pairwise")
SIM_CSV = os.path.join(PROJECT_ROOT, "reports/duplicate_text_similarity.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def load_jsonl(fp):
    r = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line: r.append(json.loads(line))
    return r


def build_pairs(records, split_name):
    """从事件对记录构造Pairwise排序数据。"""
    # Group by (article_id, target_type, target_code)
    groups = defaultdict(lambda: {"pos": [], "neg": [], "meta": {}})
    for r in records:
        gk = (r["article_id"], r["target_type"], str(r["target_code"]))
        if r["label"] == 1:
            groups[gk]["pos"].append(r)
            groups[gk]["meta"] = {
                "article_id": r["article_id"],
                "canonical_group_id": r.get("canonical_group_id", r["article_id"]),
                "target_type": r["target_type"],
                "target_code": str(r["target_code"]),
                "target_name": r.get("target_name", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "source": r.get("source", ""),
                "is_duplicate_news": r.get("is_duplicate_news", False),
            }
        elif r["label"] == 0 and r.get("negative_type") == "same_target_other_event":
            groups[gk]["neg"].append(r)

    pairs = []
    for gk, g in groups.items():
        pos = g["pos"]
        neg = g["neg"]
        meta = g["meta"]
        if not pos or not neg or not meta:
            continue

        for p in pos:
            for n in neg:
                pairs.append({
                    "pair_id": f"pair_{p['article_id']}_{p['target_type']}_{p.get('target_code','')}_{p['schema_id']}_{n['schema_id']}",
                    "article_id": meta["article_id"],
                    "canonical_group_id": meta["canonical_group_id"],
                    "target_type": meta["target_type"],
                    "target_code": meta["target_code"],
                    "target_name": meta["target_name"],
                    "positive_sample_id": p["sample_id"],
                    "positive_schema_id": p["schema_id"],
                    "positive_event_name": p.get("event_name", ""),
                    "positive_event_description": p.get("event_description", ""),
                    "negative_sample_id": n["sample_id"],
                    "negative_schema_id": n["schema_id"],
                    "negative_event_name": n.get("event_name", ""),
                    "negative_event_description": n.get("event_description", ""),
                    "title": meta["title"],
                    "content": meta["content"],
                    "source": meta["source"],
                    "is_duplicate_news": meta["is_duplicate_news"],
                    "pair_source": "observed_same_target_negative",
                })

    return pairs, groups


def deduplicate_by_canonical_group(pairs):
    """同一个 canonical_group + target_code + pos_sid + neg_sid 最多保留 2 个article版本。
    优先主新闻，其次与主新闻差异更大的1条转载。"""
    sim_data = {}
    if os.path.exists(SIM_CSV):
        df = pd.read_csv(SIM_CSV)
        for _, row in df.iterrows():
            sim_data[row["duplicate_article_id"]] = row.get("content_3gram_jaccard", 1.0)

    key_groups = defaultdict(list)
    for p in pairs:
        key = (p["canonical_group_id"], p["target_code"],
               p["positive_schema_id"], p["negative_schema_id"])
        key_groups[key].append(p)

    kept = []
    removed = 0
    for key, items in key_groups.items():
        mains = [p for p in items if not p["is_duplicate_news"]]
        dups = [p for p in items if p["is_duplicate_news"]]
        # Sort dups by similarity (lower = more different, prefer > 0.05)
        dups_sorted = sorted(dups, key=lambda p: sim_data.get(p["article_id"], 1.0))
        dups_filtered = [d for d in dups_sorted if sim_data.get(d["article_id"], 1.0) > 0.05]
        # Keep all mains + up to 1 dup
        kept.extend(mains)
        kept.extend(dups_filtered[:1] if mains else dups_filtered[:2])
        removed += max(0, len(items) - len(kept) + len(mains) + min(1, len(dups_filtered)))

    return kept, removed


def main():
    os.makedirs(PAIRWISE_DIR, exist_ok=True)
    print("构建Pairwise Ranking数据...")

    train_recs = load_jsonl(TRAIN_FILE)
    val_recs = load_jsonl(VAL_FILE)
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}")

    # Build pairs
    train_pairs, train_groups = build_pairs(train_recs, "train")
    val_pairs, val_groups = build_pairs(val_recs, "validation")
    print(f"Raw train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

    # Dedup
    train_pairs, train_removed = deduplicate_by_canonical_group(train_pairs)
    val_pairs, val_removed = deduplicate_by_canonical_group(val_pairs)
    print(f"After dedup: train={len(train_pairs)} (removed {train_removed}), val={len(val_pairs)} (removed {val_removed})")

    # Save
    for split, pairs, fname in [("train", train_pairs, "pairwise_train.jsonl"),
                                  ("validation", val_pairs, "pairwise_validation.jsonl")]:
        path = os.path.join(PAIRWISE_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        xlsx_path = path.replace(".jsonl", ".xlsx")
        pd.DataFrame(pairs).to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"Saved: {path} ({len(pairs)})")

    # Cross-split check
    train_aids = set(p["article_id"] for p in train_pairs)
    val_aids = set(p["article_id"] for p in val_pairs)
    overlap = train_aids & val_aids
    print(f"Train/Val article overlap: {len(overlap)}")

    # Summary
    train_cg = set(p["canonical_group_id"] for p in train_pairs)
    val_cg = set(p["canonical_group_id"] for p in val_pairs)
    train_targets = set((p["target_type"], p["target_code"]) for p in train_pairs)
    schema_pairs_train = set((p["positive_schema_id"], p["negative_schema_id"]) for p in train_pairs)

    multi_pos = sum(1 for g in train_groups.values() if len(g["pos"]) > 1)

    summary = {
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(val_pairs),
        "train_article_count": len(train_aids),
        "validation_article_count": len(val_aids),
        "train_canonical_group_count": len(train_cg),
        "validation_canonical_group_count": len(val_cg),
        "train_target_count": len(train_targets),
        "validation_target_count": len(set((p["target_type"], p["target_code"]) for p in val_pairs)),
        "train_schema_pair_count": len(schema_pairs_train),
        "validation_schema_pair_count": len(set((p["positive_schema_id"], p["negative_schema_id"]) for p in val_pairs)),
        "duplicate_pairs_removed": train_removed + val_removed,
        "groups_with_multiple_positive_events": multi_pos,
        "cross_split_conflict_count": len(overlap),
    }
    with open(os.path.join(REPORTS_DIR, "pairwise_dataset_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")

    # By target
    target_rows = []
    for (tt, tc) in sorted(set((p["target_type"], p["target_code"]) for p in train_pairs)):
        tp = [p for p in train_pairs if p["target_type"] == tt and p["target_code"] == tc]
        vp = [p for p in val_pairs if p["target_type"] == tt and p["target_code"] == tc]
        target_rows.append({
            "target_type": tt, "target_code": tc, "target_name": tp[0]["target_name"] if tp else "",
            "train_pair_count": len(tp), "validation_pair_count": len(vp),
        })
    pd.DataFrame(target_rows).to_csv(os.path.join(REPORTS_DIR, "pairwise_dataset_by_target.csv"), index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
