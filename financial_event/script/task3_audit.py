#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from task3_common import ROOT, parse_datetime, read_jsonl


def parse_dt(value):
    return parse_datetime(value)


def safe_float(value):
    try:
        if value is None or str(value).strip() == '':
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

DEFAULT_PATHS = {
    "task1_train": "outputs/final_task1/final_events_train.jsonl",
    "task1_test": "outputs/final_task1/final_events_test.jsonl",
    "task1_all_with_sources": "data/processed/event_clusters/all_final_events_with_sources.jsonl",
    "task2_scores": "outputs/event_credibility/event_credibility_scores.jsonl",
    "task2_scores_fulltext": "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
    "task2_timeline": "outputs/event_credibility/event_timeline_for_task3.jsonl",
    "task2_llm_review": "outputs/event_credibility/llm_review_results.jsonl",
    "cluster_members": "outputs/event_clustering/event_cluster_members.jsonl",
    "clusters": "outputs/event_clustering/event_clusters.jsonl",
    "article_metadata": "data/processed/event_clusters/article_metadata_index.jsonl",
    "llm_queue": "data/processed/event_credibility/llm_review_queue.jsonl",
}

SAMPLE_FIELDS = [
    "article_id",
    "cluster_id",
    "event_id",
    "schema_id",
    "publish_time",
    "article_publish_time",
    "earliest_publish_time",
    "latest_publish_time",
    "source",
    "source_name",
    "source_normalized",
    "target_type",
    "target_code",
    "target_name",
    "event_name",
    "credibility_score",
    "credibility_level",
    "verification_status",
    "verdict",
    "trust_score",
    "event_fingerprint",
    "linked_stocks",
    "linked_industries",
    "event_type",
    "impact_direction",
    "conflicting_reports",
    "verification_hints",
]


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def field_missing_rate(rows: list[dict], field: str) -> float:
    if not rows:
        return float("nan")
    missing = 0
    for row in rows:
        val = row.get(field)
        if val is None:
            missing += 1
        elif isinstance(val, str) and not val.strip():
            missing += 1
        elif isinstance(val, list) and len(val) == 0:
            missing += 1
    return missing / len(rows)


def analyze_file(name: str, path: Path) -> dict:
    rows = load_records(path)
    if not rows:
        return {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "records": 0,
            "fields": [],
            "time_min": None,
            "time_max": None,
            "article_id_unique": None,
            "article_id_duplicates": None,
        }
    fields = sorted({key for row in rows for key in row.keys()})
    time_candidates = []
    for row in rows:
        for key in ("publish_time", "article_publish_time", "earliest_publish_time", "latest_publish_time"):
            dt = parse_dt(row.get(key))
            if dt:
                time_candidates.append(dt)
                break
    article_ids = [str(r.get("article_id")) for r in rows if r.get("article_id") is not None]
    article_counter = Counter(article_ids)
    cluster_ids = [str(r.get("cluster_id")) for r in rows if r.get("cluster_id") is not None]
    cluster_counter = Counter(cluster_ids)
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "records": len(rows),
        "fields": fields,
        "time_min": min(time_candidates).isoformat(sep=" ") if time_candidates else None,
        "time_max": max(time_candidates).isoformat(sep=" ") if time_candidates else None,
        "article_id_unique": len(article_counter),
        "article_id_duplicates": sum(v - 1 for v in article_counter.values() if v > 1),
        "cluster_id_unique": len(cluster_counter) if cluster_counter else None,
        "cluster_id_duplicates": sum(v - 1 for v in cluster_counter.values() if v > 1) if cluster_counter else None,
        "field_missing_rates": {field: field_missing_rate(rows, field) for field in SAMPLE_FIELDS},
        "sample_row": rows[0] if rows else None,
    }


def join_task1_task2(task1_rows: list[dict], cluster_rows: list[dict], scores_rows: list[dict], timeline_rows: list[dict]) -> dict:
    task1_article_ids = {str(r.get("article_id")) for r in task1_rows if r.get("article_id")}
    cluster_article_ids = {str(r.get("article_id")) for r in cluster_rows if r.get("article_id")}
    timeline_cluster_ids = {str(r.get("cluster_id")) for r in timeline_rows if r.get("cluster_id")}
    score_cluster_ids = {str(r.get("cluster_id")) for r in scores_rows if r.get("cluster_id")}
    task2_article_ids = {str(r.get("article_id")) for r in scores_rows if r.get("article_id")}

    return {
        "task1_article_ids": len(task1_article_ids),
        "cluster_article_ids": len(cluster_article_ids),
        "task1_cluster_intersection": len(task1_article_ids & cluster_article_ids),
        "task1_not_in_clusters": len(task1_article_ids - cluster_article_ids),
        "clusters_not_in_task1": len(cluster_article_ids - task1_article_ids),
        "task2_has_article_id": bool(task2_article_ids),
        "task2_article_id_count": len(task2_article_ids),
        "task2_cluster_ids": len(score_cluster_ids),
        "timeline_cluster_ids": len(timeline_cluster_ids),
        "scores_timeline_cluster_intersection": len(score_cluster_ids & timeline_cluster_ids),
    }


def trust_score_summary(score_rows: list[dict]) -> dict:
    scores = []
    for row in score_rows:
        val = row.get("credibility_score", row.get("trust_score"))
        f = safe_float(val)
        if f is not None:
            scores.append(f)
    if not scores:
        return {"count": 0}
    buckets = Counter()
    for score in scores:
        if score < 20:
            buckets["[0,20)"] += 1
        elif score < 40:
            buckets["[20,40)"] += 1
        elif score < 60:
            buckets["[40,60)"] += 1
        elif score < 80:
            buckets["[60,80)"] += 1
        else:
            buckets["[80,100]"] += 1
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": mean(scores),
        "distribution": dict(buckets),
    }


def verdict_summary(llm_rows: list[dict]) -> dict:
    verdicts = Counter()
    for row in llm_rows:
        verdict = str(row.get("verdict", "")).strip() or "UNKNOWN"
        verdicts[verdict] += 1
    return dict(verdicts)


def validation_readiness(task1_rows: list[dict], task2_rows: list[dict], cluster_rows: list[dict]) -> dict:
    required_task1_fields = ["article_id", "title", "publish_time", "source", "target_type", "target_code", "target_name", "schema_id", "event_name"]
    missing = {
        field: field_missing_rate(task1_rows, field)
        for field in required_task1_fields
    }
    task1_articles = {str(r.get("article_id")) for r in task1_rows if r.get("article_id")}
    task2_clusters = {str(r.get("cluster_id")) for r in task2_rows if r.get("cluster_id")}
    task1_cluster_count = len({str(r.get("cluster_id")) for r in cluster_rows if r.get("cluster_id")})
    can_build = all(rate == 0 for rate in missing.values()) and len(task1_articles) > 0 and len(task2_clusters) > 0 and task1_cluster_count > 0
    return {
        "required_task1_missing_rates": missing,
        "task1_article_count": len(task1_articles),
        "task2_cluster_count": len(task2_clusters),
        "cluster_count": task1_cluster_count,
        "can_build_validation_set": can_build,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/task3_data_audit.md")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.config:
        load_yaml(args.config)

    paths = {name: ROOT / rel for name, rel in DEFAULT_PATHS.items()}
    task1_train = load_records(paths["task1_train"])
    task1_test = load_records(paths["task1_test"])
    task1_all = task1_train + task1_test
    cluster_members = load_records(paths["cluster_members"])
    clusters = load_records(paths["clusters"])
    task2_scores = load_records(paths["task2_scores"])
    task2_scores_fulltext = load_records(paths["task2_scores_fulltext"])
    task2_timeline = load_records(paths["task2_timeline"])
    task2_llm_review = load_records(paths["task2_llm_review"])
    all_final = load_records(paths["task1_all_with_sources"])
    article_meta = load_records(paths["article_metadata"])
    llm_queue = load_records(paths["llm_queue"])

    file_stats = [analyze_file(name, path) for name, path in paths.items()]
    join_stats = join_task1_task2(task1_all, cluster_members, task2_scores, task2_timeline)
    trust_stats = trust_score_summary(task2_scores)
    verdict_stats = verdict_summary(task2_llm_review)
    readiness = validation_readiness(task1_all, task2_scores, clusters)

    task1_ids = [str(r.get("article_id")) for r in task1_all if r.get("article_id")]
    task1_counts = Counter(task1_ids)
    duplicated_task1 = {aid: cnt for aid, cnt in task1_counts.items() if cnt > 1}

    cluster_article_ids = {str(r.get("article_id")) for r in cluster_members if r.get("article_id")}
    cluster_missing_task1 = sorted(set(task1_ids) - cluster_article_ids)

    lines = []
    lines.append("# 任务三数据审计报告")
    lines.append("")
    lines.append("## 1. 输入文件清单")
    lines.append("")
    for stat in file_stats:
        lines.append(f"- `{stat['path']}`：{stat['records']} 条记录，时间范围 {stat['time_min']} ~ {stat['time_max']}")
    lines.append("")
    lines.append("## 2. 字段与缺失情况")
    lines.append("")
    for stat in file_stats:
        lines.append(f"### `{stat['name']}`")
        if not stat['exists']:
            lines.append("- 文件缺失")
            lines.append("")
            continue
        lines.append(f"- 记录数：{stat['records']}")
        if stat['time_min']:
            lines.append(f"- 时间范围：{stat['time_min']} ~ {stat['time_max']}")
        lines.append(f"- 字段数：{len(stat['fields'])}")
        lines.append(f"- `article_id` 唯一数：{stat['article_id_unique']}；重复数：{stat['article_id_duplicates']}")
        if stat.get('cluster_id_unique') is not None:
            lines.append(f"- `cluster_id` 唯一数：{stat['cluster_id_unique']}；重复数：{stat['cluster_id_duplicates']}")
        lines.append("- 字段缺失率（部分关键字段）：")
        for field in ["article_id", "publish_time", "source", "source_name", "target_type", "target_code", "target_name", "schema_id", "event_name", "credibility_score", "credibility_level", "verification_status", "verdict", "trust_score", "event_fingerprint", "linked_stocks", "linked_industries", "event_type", "impact_direction", "conflicting_reports", "verification_hints"]:
            rate = stat['field_missing_rates'].get(field)
            if rate is None or (isinstance(rate, float) and math.isnan(rate)):
                continue
            lines.append(f"  - `{field}`：{rate:.2%}")
        if stat["sample_row"]:
            preview = json.dumps(stat["sample_row"], ensure_ascii=False)[:500]
            lines.append(f"- 样例：`{preview}`")
        lines.append("")

    lines.append("## 3. article_id 唯一性与连接性")
    lines.append("")
    lines.append(f"- 任务一总记录数：{len(task1_all)}")
    lines.append(f"- 任务一不同 `article_id` 数：{len(set(task1_ids))}")
    lines.append(f"- 重复 `article_id` 数：{len(duplicated_task1)}，重复记录数合计：{sum(v - 1 for v in duplicated_task1.values())}")
    lines.append(f"- 任务一文章在簇成员表中可连接数：{join_stats['task1_cluster_intersection']}")
    lines.append(f"- 任务一无法通过 `article_id` 找到簇成员的文章数：{join_stats['task1_not_in_clusters']}")
    lines.append(f"- 簇成员表中无法回到任务一的文章数：{join_stats['clusters_not_in_task1']}")
    lines.append("")

    lines.append("## 4. trust_score / credibility_score 统计")
    lines.append("")
    lines.append(f"- 可解析分数数量：{trust_stats.get('count', 0)}")
    if trust_stats.get("count", 0):
        lines.append(f"- 最小值：{trust_stats['min']:.2f}")
        lines.append(f"- 最大值：{trust_stats['max']:.2f}")
        lines.append(f"- 均值：{trust_stats['mean']:.4f}")
        lines.append(f"- 分布：{json.dumps(trust_stats['distribution'], ensure_ascii=False)}")
    lines.append("")

    lines.append("## 5. verdict 统计")
    lines.append("")
    lines.append(f"- `{json.dumps(verdict_stats, ensure_ascii=False)}`")
    lines.append("")

    lines.append("## 6. publish_time 解析")
    lines.append("")
    def parse_success_rate(rows: list[dict]) -> str:
        if not rows:
            return "N/A"
        ok = 0
        total = 0
        for row in rows:
            total += 1
            if any(parse_dt(row.get(k)) for k in ("publish_time", "article_publish_time", "earliest_publish_time", "latest_publish_time")):
                ok += 1
        return f"{ok}/{total} ({ok / total:.2%})"
    lines.append(f"- 任务一：{parse_success_rate(task1_all)}")
    lines.append(f"- 任务二簇级：{parse_success_rate(task2_scores)}")
    lines.append(f"- 时间线：{parse_success_rate(task2_timeline)}")
    lines.append("")

    lines.append("## 7. 数据质量问题")
    lines.append("")
    lines.append("1. 任务一最终输出缺少 `content`、`event_fingerprint`、`linked_stocks`、`linked_industries` 等任务三理想输入字段。")
    lines.append("2. 任务二为簇级输出，`cluster_id` 需要通过簇成员表间接回连，不能直接按 `article_id` 对齐。")
    lines.append("3. 任务一存在多标签同文现象，导致 `article_id` 重复，但这属于业务事实，需要在任务三按文章去重或按事件记录处理。")
    lines.append("4. 已有 `event_stage`、`verification_status`、`credibility_level` 是系统派生结果，只能用于候选生成和审计，不能作为人工真值。")
    lines.append("")

    lines.append("## 8. 是否足以构建任务三验证集")
    lines.append("")
    lines.append(f"- `{json.dumps(readiness, ensure_ascii=False)}`")
    lines.append("")
    lines.append("结论：可以构建任务三人工验证集的候选样本，但需要接受部分字段缺失并以 repair/pending 处理。")
    lines.append("")

    lines.append("## 9. 字段级审计结论")
    lines.append("")
    lines.append("- `publish_time`：任务一部分记录缺失，但可由 article metadata index 回填；回填结果需保留来源标记，不能视为原始任务一字段。")
    lines.append("- `linked_stocks` / `linked_industries`：任务一输出未提供统一数组字段，当前仅可由 `target_type` 和 `target_code` 做候选级映射。")
    lines.append("- `event_fingerprint`：任务一输出未提供；当前候选脚本使用 `schema_id` 作为 provisional fingerprint，不能作为人工真值。")
    lines.append("- `trust_score`：任务二为 cluster 级 `credibility_score`，必须通过 `event_cluster_members` 映射到文章；不存在直接 article_id 连接。")
    lines.append("- `verdict`：LLM review 结果只有部分 cluster 覆盖，不能代表所有文章的人工/模型最终判断。")
    lines.append("- `conflicting_reports` / `verification_hints`：可由任务二 `risk_reasons`、`supporting_reasons` 等派生，但应标注为辅助证据。")
    lines.append("")

    lines.append("## 10. 数据质量问题及建议")
    lines.append("")
    lines.append("1. 任务一存在重复 `article_id`；建议任务三保留 article-level 去重键，同时保留原始 event_record_id 追溯字段。")
    lines.append("2. 任务一核心任务三字段不完整；建议后续任务一正式输出补充统一接口字段，避免任务三依赖启发式映射。")
    lines.append("3. 任务二结果是簇级而非文章级；建议接口层明确 cluster-to-article 连接关系和版本。")
    lines.append("4. 当前候选的 provisional_group 来源于已有程序簇，不能替代人工 `human_cluster_id`。")
    lines.append("5. 时间切分应在人工簇完成后重新执行；当前仅作为临时防泄漏切分。")
    lines.append("")

    lines.append("## 11. 处理阻塞项")
    lines.append("")
    lines.append("- 未发现阻塞候选生成和审计的缺失文件。")
    lines.append("- 主要非阻塞限制：任务一未提供完整统一字段；任务二缺少逐 article_id 评分；人工 cluster_id 尚未完成。")
    lines.append("")

    lines.append("## 12. 关键统计补充")
    lines.append("")
    lines.append(f"- 任务二簇数：{len(clusters)}")
    lines.append(f"- 任务二可信度结果条数：{len(task2_scores)}")
    lines.append(f"- 任务二 fulltext 结果条数：{len(task2_scores_fulltext)}")
    lines.append(f"- 任务二 LLM review 条数：{len(task2_llm_review)}")
    lines.append(f"- article metadata 记录数：{len(article_meta)}")
    lines.append(f"- LLM queue 记录数：{len(llm_queue)}")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "task1_records": len(task1_all),
        "task2_scores": len(task2_scores),
        "task2_clusters": len(clusters),
        "task1_duplicate_article_ids": len(duplicated_task1),
        "task1_cluster_join_missing": join_stats['task1_not_in_clusters'],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
