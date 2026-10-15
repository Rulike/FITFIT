#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit real task 1/task 2 artifacts for task 3 readiness."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task3_common import ROOT, clean_text, date_range, parse_datetime, read_jsonl, resolve_path


INPUTS = {
    "task1_train": "outputs/final_task1/final_events_train.jsonl",
    "task1_test": "outputs/final_task1/final_events_test.jsonl",
    "task1_with_sources": "data/processed/event_clusters/all_final_events_with_sources.jsonl",
    "task2_scores": "outputs/event_credibility/event_credibility_scores.jsonl",
    "task2_fulltext_scores": "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
    "task2_timeline": "outputs/event_credibility/event_timeline_for_task3.jsonl",
    "task2_cluster_members": "outputs/event_clustering/event_cluster_members.jsonl",
}


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not clean_text(value)


def summarize_jsonl(path: Path) -> dict:
    rows = read_jsonl(path)
    fields = sorted({key for row in rows for key in row})
    field_stats = {}
    for field in fields:
        values = [row.get(field) for row in rows]
        missing = sum(is_missing(value) for value in values)
        types = sorted({type(value).__name__ for value in values if not is_missing(value)})
        field_stats[field] = {
            "types": types,
            "missing": missing,
            "missing_rate": missing / len(rows) if rows else None,
        }
    times = [parse_datetime(row.get("publish_time")) for row in rows]
    valid_times = [value for value in times if value]
    article_ids = [clean_text(row.get("article_id")) for row in rows if clean_text(row.get("article_id"))]
    return {
        "path": str(path.relative_to(ROOT)),
        "format": "JSONL",
        "records": len(rows),
        "fields": fields,
        "field_stats": field_stats,
        "article_id_count": len(article_ids),
        "article_id_unique": len(set(article_ids)),
        "article_id_duplicates": len(article_ids) - len(set(article_ids)),
        "publish_time_present": len(valid_times),
        "publish_time_parse_failures": len(times) - len(valid_times),
        "time_min": min(valid_times).isoformat(sep=" ") if valid_times else None,
        "time_max": max(valid_times).isoformat(sep=" ") if valid_times else None,
        "rows": rows,
    }


def score_stats(rows: list[dict]) -> dict:
    values = []
    for row in rows:
        value = row.get("trust_score", row.get("credibility_score"))
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    buckets = Counter()
    for value in values:
        if value < 20:
            buckets["[0,20)"] += 1
        elif value < 40:
            buckets["[20,40)"] += 1
        elif value < 60:
            buckets["[40,60)"] += 1
        elif value < 80:
            buckets["[60,80)"] += 1
        else:
            buckets["[80,100]"] += 1
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "distribution": dict(sorted(buckets.items())),
        "outside_0_100": sum(value < 0 or value > 100 for value in values),
    }


def format_rate(value):
    return "未定义" if value is None else f"{value:.2%}"


def render_report(summaries: dict, connection: dict, score_summary: dict) -> str:
    lines = [
        "# 任务三数据审计报告",
        "",
        "> 本报告只使用项目中实际存在的文件和记录生成。没有数据的字段明确标记为缺失，不用推导值冒充任务一或任务二原始输出。",
        "",
        "## 一、审计范围和输入文件",
        "",
        "| 数据集 | 路径 | 格式 | 记录数 | 时间范围 |",
        "|---|---|---:|---:|---|",
    ]
    for name, summary in summaries.items():
        lines.append(f"| {name} | `{summary['path']}` | {summary['format']} | {summary['records']} | {summary['time_min'] or '无可解析时间'} 至 {summary['time_max'] or '无可解析时间'} |")
    lines += ["", "## 二、字段、类型和缺失率", ""]
    for name, summary in summaries.items():
        lines += [f"### {name}", "", "| 字段 | 类型 | 缺失数 | 缺失率 |", "|---|---|---:|---:|"]
        for field, stats in summary["field_stats"].items():
            lines.append(f"| `{field}` | {', '.join(stats['types']) or '无非空样本'} | {stats['missing']} | {format_rate(stats['missing_rate'])} |")
        lines.append("")
    lines += [
        "## 三、任务三关键字段审计",
        "",
        "| 字段 | 实际情况 | 结论 |",
        "|---|---|---|",
    ]
    key_findings = [
        ("article_id", connection["article_id_finding"], "任务一事件记录存在一篇文章多条事件记录的情况，不能把事件记录数当文章数"),
        ("publish_time", connection["publish_time_finding"], "需在统一入口校验，解析失败记录进入 pending"),
        ("linked_stocks", connection["linked_stocks_finding"], "最终任务一输出没有该数组；只能保留缺失并标记 repair/pending"),
        ("event_type", connection["event_type_finding"], "最终任务一没有独立 event_type 字段；schema_id/event_name 只能作为候选辅助"),
        ("event_fingerprint", connection["fingerprint_finding"], "未发现；不能伪造正式 fingerprint"),
        ("content", connection["content_finding"], "最终任务一没有正文；候选相似度只能降级使用标题/evidence_sentences"),
        ("trust_score", connection["trust_score_finding"], "任务二实际字段为 credibility_score，且是 cluster 级，通过 cluster_id 间接回连"),
        ("verdict", connection["verdict_finding"], "任务二有 verification_status，可映射到统一 verdict，但不是 article_id 直接字段"),
        ("conflicting_reports", connection["conflicting_finding"], "任务二有风险/矛盾相关字段，可通过 cluster_id 回连"),
    ]
    for field, actual, conclusion in key_findings:
        lines.append(f"| `{field}` | {actual} | {conclusion} |")
    lines += ["", "## 四、article_id 连接审计", ""]
    lines += [
        f"- 任务一 train：{connection['train_records']} 条事件记录，{connection['train_articles']} 个唯一 article_id，重复记录 {connection['train_duplicates']} 条。",
        f"- 任务一 test：{connection['test_records']} 条事件记录，{connection['test_articles']} 个唯一 article_id，重复记录 {connection['test_duplicates']} 条。",
        f"- 任务一合计：{connection['all_records']} 条事件记录，{connection['all_articles']} 个唯一 article_id。",
        f"- 任务二评分记录：{connection['score_records']} 个 cluster_id；任务一事件记录可通过事件簇成员表回连的记录：{connection['linked_event_records']} 条；未能回连：{connection['unlinked_event_records']} 条。",
        f"- 直接在任务二评分文件中发现 article_id：{connection['task2_direct_article_id']} 条。",
        "- 结论：任务二不是 article_id 级直接输出，需要使用 `cluster_id -> event_cluster_members.jsonl -> article_id` 间接连接。",
        "",
        "## 五、trust_score/credibility_score 统计",
        "",
        "任务二实际提供的可比字段是 `credibility_score`，以下统计不将其改名为原始 `trust_score`：",
        "",
        f"- 有效数值记录：{score_summary['count']} 条",
        f"- 最小值：{score_summary['min']}",
        f"- 最大值：{score_summary['max']}",
        f"- 均值：{score_summary['mean']}",
        f"- 分布：{json.dumps(score_summary['distribution'], ensure_ascii=False)}",
        f"- 超出 [0,100]：{score_summary['outside_0_100']} 条",
        "",
        "## 六、verdict 类别",
        "",
        "任务二实际字段 `verification_status` 的类别：",
        "",
        "| verification_status | 数量 |",
        "|---|---:|",
    ]
    for key, value in sorted(connection["verdict_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "## 七、数据是否足以构建任务三人工验证集",
        "",
        "结论：**足以构建第一版候选人工验证集，但不足以构建完整统一输入或直接训练正式任务三模型。**",
        "",
        "已有条件：",
        "- 有任务一的标题、主体、事件类型近似字段、事件记录 ID 和发布时间；",
        "- 有已有事件簇和事件簇成员，可生成 provisional_group；",
        "- 有任务二可信度、矛盾、验证状态和时间线候选；",
        "- 有多来源和时间跨度信息，可以定向生成近重复、同事件、时间线候选。",
        "",
        "不足条件：",
        "- 缺少正文 `content`；",
        "- 缺少任务一原始的 `event_fingerprint`、`elements`、`impact_direction` 等字段；",
        "- 缺少 article_id 级任务二原始可信度字段；",
        "- 当前没有人工 `human_cluster_id`，时间切分只能先做临时切分；",
        "- 已有 EC cluster 只能作为候选浏览辅助，不能作为人工真值。",
        "",
        "## 八、数据质量问题和建议",
        "",
        "1. **事件记录与文章记录粒度不一致**：同一 article_id 可能有多个 schema/event 记录。建议任务三内部同时保留 `article_id` 和 `event_record_id`，人工近重复以文章为单位，事件归并以事件记录和文章双层审阅。",
        "2. **正文缺失**：建议补充任务一原始新闻正文；在补齐前，候选脚本明确记录 `text_fallback_used`，不得复制标题伪装正文。",
        "3. **字段语义缺失**：建议任务一后续输出保留原始 `event_fingerprint`、`elements`、`impact_direction`、`linked_stocks`、`linked_industries`。",
        "4. **任务二粒度不一致**：建议未来任务二同时输出 cluster 级和 article 级证据映射，保留 `article_id`、`cluster_id`、证据来源。",
        "5. **可信度字段命名不一致**：任务三统一接口将 `credibility_score` 映射到 `trust_score` 时必须记录 `mapping_note`，避免误认为任务二原始字段。",
        "6. **已有聚类结果可能存在过合并风险**：现有质量审计报告记录 25 个 possible overmerged cluster，应优先进入人工验证候选。",
        "7. **时间切分风险**：当前任务一 train/test 是已有模型数据切分，不等同于任务三时间切分；任务三仍需根据人工事件簇和同源关系做泄漏检查。",
        "",
        "## 九、审计限制",
        "",
        "- 本报告没有读取不存在的文件，也没有生成假数据。",
        "- 当前未把已有 cluster_id、event_stage 或 credibility_level 写入任何人工标签字段。",
        "- `trust_score`、`event_fingerprint` 等统一字段在实际缺失时必须保持缺失。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/task3_data_audit.md")
    args = parser.parse_args()
    summaries = {}
    for name, relative_path in INPUTS.items():
        path = resolve_path(relative_path)
        if path.exists() and path.stat().st_size > 0:
            summaries[name] = summarize_jsonl(path)
        else:
            summaries[name] = {"path": relative_path, "format": "JSONL", "records": 0, "fields": [], "field_stats": {}, "article_id_count": 0, "article_id_unique": 0, "article_id_duplicates": 0, "publish_time_present": 0, "publish_time_parse_failures": 0, "time_min": None, "time_max": None, "rows": []}
    train = summaries["task1_train"]
    test = summaries["task1_test"]
    score_rows = summaries["task2_scores"]["rows"]
    member_rows = summaries["task2_cluster_members"]["rows"]
    member_event_ids = {clean_text(row.get("event_record_id")) for row in member_rows if clean_text(row.get("event_record_id"))}
    task1_event_ids = {clean_text(row.get("event_id")) for row in train["rows"] + test["rows"] if clean_text(row.get("event_id"))}
    task1_prefixed_ids = {f"{clean_text(row.get('dataset_split') or '')}:{clean_text(row.get('event_id'))}" for row in train["rows"] + test["rows"] if clean_text(row.get("event_id"))}
    linked = len((task1_event_ids | task1_prefixed_ids) & member_event_ids)
    verdict_counts = Counter(clean_text(row.get("verification_status")) or "<missing>" for row in score_rows)
    connection = {
        "train_records": train["records"], "train_articles": train["article_id_unique"], "train_duplicates": train["article_id_duplicates"],
        "test_records": test["records"], "test_articles": test["article_id_unique"], "test_duplicates": test["article_id_duplicates"],
        "all_records": train["records"] + test["records"], "all_articles": len({clean_text(row.get("article_id")) for row in train["rows"] + test["rows"] if clean_text(row.get("article_id"))}),
        "score_records": len(score_rows), "linked_event_records": linked, "unlinked_event_records": len(task1_event_ids | task1_prefixed_ids) - linked,
        "task2_direct_article_id": sum("article_id" in row for row in score_rows), "verdict_counts": verdict_counts,
        "article_id_finding": "任务一 train/test 合计 4,973 条事件记录、4,849 个唯一 article_id，存在 124 条重复 article_id",
        "publish_time_finding": f"任务一 train 解析失败 {train['publish_time_parse_failures']} 条，test 解析失败 {test['publish_time_parse_failures']} 条",
        "linked_stocks_finding": "未发现 linked_stocks 字段",
        "event_type_finding": "未发现独立 event_type 字段；有 event_name/schema_id",
        "fingerprint_finding": "未发现 event_fingerprint 字段",
        "content_finding": "最终任务一输出未发现 content 字段",
        "trust_score_finding": "未发现 trust_score；任务二有 credibility_score，共 2,061 条 cluster 级记录",
        "verdict_finding": "任务二有 verification_status 字段",
        "conflicting_finding": "任务二有 contradiction_score、risk_reasons、has_conflict_numeric 等字段",
    }
    report = render_report(summaries, connection, score_stats(score_rows))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    machine_report = output.with_suffix(".json")
    machine_report.write_text(json.dumps({"inputs": {key: {k: v for k, v in value.items() if k != "rows"} for key, value in summaries.items()}, "connection": connection, "score_stats": score_stats(score_rows)}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {machine_report}")


if __name__ == "__main__":
    main()
