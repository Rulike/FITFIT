#!/usr/bin/env python3
"""Build deterministic, stratified, label-empty task3 pilot files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from task3_common import ROOT, read_csv_rows, read_jsonl, stable_hash


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select(rows: list[dict[str, Any]], limit: int, seed: int, label: str, strata_field: str, quota_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    if not rows:
        return []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(strata_field) or "unknown")].append(row)
    ranked_strata = sorted(buckets, key=lambda key: (-len(buckets[key]), key))
    selected: list[dict[str, Any]] = []
    remaining = limit
    for strata in ranked_strata:
        bucket = buckets[strata]
        bucket_sorted = sorted(bucket, key=lambda row: ("|".join(str(row.get(field) or "") for field in quota_fields), stable_hash(seed, label, row.get("pair_id", row.get("event_mention_id", row.get("provisional_cluster_id", ""))), strata)))
        take = min(len(bucket_sorted), max(1, round(limit * len(bucket_sorted) / len(rows))))
        selected.extend(bucket_sorted[:take])
        remaining = limit - len(selected)
        if remaining <= 0:
            break
    if len(selected) < limit:
        leftovers = [row for strata in ranked_strata for row in buckets[strata] if row not in selected]
        leftovers_sorted = sorted(leftovers, key=lambda row: stable_hash(seed, label, row.get("pair_id", row.get("event_mention_id", row.get("provisional_cluster_id", "")))))
        selected.extend(leftovers_sorted[: limit - len(selected)])
    return selected[:limit]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["pair_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--near", default="outputs/task3/near_duplicate_pairs.csv")
    parser.add_argument("--same", default="outputs/task3/same_event_pairs.csv")
    parser.add_argument("--cluster", default="outputs/task3/cluster_annotation_seed.jsonl")
    parser.add_argument("--timeline", default="outputs/task3/timeline_annotation_seed.jsonl")
    parser.add_argument("--output-dir", default="outputs/task3/annotation_pilot")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    output_dir = resolve(args.output_dir)
    near = read_csv_rows(args.near)
    same = read_csv_rows(args.same)
    clusters = read_jsonl(args.cluster)
    timelines = read_jsonl(args.timeline)

    near_rows = select(near, 50, args.seed, "near", "candidate_reason", ("candidate_reason", "hard_negative", "lexical_similarity"))
    same_rows = select(same, 50, args.seed, "same", "candidate_reason", ("candidate_reason", "same_document", "time_gap_days"))
    for row in near_rows:
        row["human_is_near_duplicate"] = ""
        row["annotator_note"] = ""
        row["pilot_selection"] = "deterministic_stratified_candidate_sample"
    for row in same_rows:
        row["human_same_event"] = ""
        row["human_relation"] = ""
        row["annotator_note"] = ""
        row["pilot_selection"] = "deterministic_stratified_candidate_sample"
    write_csv(output_dir / "near_duplicate_pilot.csv", near_rows)
    write_csv(output_dir / "same_event_pilot.csv", same_rows)

    cluster_rows = select(clusters, 20, args.seed, "cluster", "provisional_event_type", ("provisional_event_type", "trust_score", "document_id"))
    timeline_rows = select(timelines, 20, args.seed, "timeline", "provisional_cluster_id", ("provisional_cluster_id", "trust_score"))
    for row in cluster_rows:
        row["human_cluster_id"] = ""
        row["annotator_note"] = ""
        row["sampling_stratum"] = row.get("provisional_event_type") or "unknown"
    for row in timeline_rows:
        for field in ("human_key_nodes", "human_stage", "human_relation", "human_reversal", "annotator_note"):
            row[field] = ""
        row["sampling_stratum"] = row.get("provisional_cluster_id") or "unknown"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cluster_pilot.jsonl").open("w", encoding="utf-8") as handle:
        for row in cluster_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "timeline_pilot.jsonl").open("w", encoding="utf-8") as handle:
        for row in timeline_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dictionary = """# 任务三试标包字段说明

## 来源字段

- `document_id` / `article_id`：任务三文档层唯一标识，来源于任务一文章。
- `event_mention_id`：任务三根据文档和事件要素生成的稳定事件提及标识。
- `title`、`content`、`source_name`、`publish_time`：来自任务一；发布时间可能由 metadata index 修复。
- `schema_id`、`provisional_event_type`：任务一字段。当前没有正式 `event_type` 时仅作为临时类型。
- `event_type`、`event_fingerprint`：只有任务一明确提供时才填写；缺失时保持空，不用 `schema_id` 冒充。
- `linked_stocks`、`linked_industries`：任务一原生字段优先；由主体类型和代码推导时带有临时 provenance。

## 任务二字段

- `trust_score` / `credibility_score`：任务二可信度分数。当前为 cluster 范围，不是 article-level 原始评分。
- `trust_score_scope`：必须为 `cluster`，表示该分数来自事件簇。
- `verdict`：任务二的验证状态或 review 结果，仅作为辅助证据。

## 候选和人工字段

- `candidate_reason`：程序生成候选的原因，不是人工标签。
- `lexical_similarity`、`title_similarity`、`content_similarity`：程序相似度，不是人工判断。
- `human_is_near_duplicate`：人工填写 `true`、`false` 或 `uncertain`。
- `human_same_event`：人工填写是否同一具体事件。
- `human_relation`：人工填写 `same_event`、`related_update`、`clarification_or_refutation`、`reversal` 或 `uncertain`。
- `human_cluster_id`：人工为事件簇分配的稳定编号。
- `human_stage`：人工填写事件阶段。
- `human_reversal`：人工判断是否发生方向反转。
- `annotator_note`：人工记录证据、分歧和缺失信息。所有这些人工字段初始必须为空。

## 关系区别

- `near_duplicate`：同一内容的转载、改写或摘要，事实没有重要新增。
- `same_event`：不同报道指向同一具体事实。
- `related_update`：不是同一篇事实，但属于同一事件的后续发展。
- `clarification_or_refutation`：澄清、否认或纠正前一说法。
- `reversal`：后续证据使事件结论方向发生反转。

## 标注原则

先看主体、对象、动作、关键时间和项目，再看标题相似度。信息不足时使用 `uncertain`，不要猜测。
"""
    (output_dir / "field_dictionary.md").write_text(dictionary, encoding="utf-8")
    report = {
        "seed": args.seed,
        "near_duplicate_pilot_count": len(near_rows),
        "same_event_pilot_count": len(same_rows),
        "cluster_pilot_count": len(cluster_rows),
        "timeline_pilot_count": len(timeline_rows),
        "sampling_strata": {
            "cluster": sorted(Counter(row.get("sampling_stratum") for row in cluster_rows).items()),
            "timeline": sorted(Counter(row.get("sampling_stratum") for row in timeline_rows).items()),
        },
        "human_labels_generated": False,
        "output_dir": str(output_dir.relative_to(ROOT)),
    }
    report_path = ROOT / "reports/task3_pilot_sampling_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from collections import Counter
    main()
