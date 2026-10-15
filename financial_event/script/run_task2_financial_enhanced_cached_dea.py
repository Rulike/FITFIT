#!/usr/bin/env python3
"""Dry-run financial enhancement audit for Task2 Cached DEA.

The repository currently has no local CSMAR bundle and the task-2 model server
is not guaranteed to be available. This script therefore focuses on the
deterministic parts of the enhancement layer:

- load the 500-cluster audit package
- extract financial claims from the available news text proxies
- run the rule-based financial consistency module with an empty or caller-
  supplied CSMAR adapter
- emit an audit manifest, JSONL trace, and markdown report

When the user later provides CSMAR data and a live model server, the same
financial_context payload can be passed into the Cached DEA run path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility import financial_consistency  # noqa: E402
from src.event_credibility.csmar_adapter import (  # noqa: E402
    build_mapping_template,
    inventory_sources,
    load_csmar_records,
    missing_expected_files,
)


DEFAULT_INPUT_JSONL = ROOT / "outputs/event_credibility/task2_independent500_v1/independent_samples.jsonl"
DEFAULT_GOLD_WORKBOOK = ROOT / "reports/task2_independent500_human_audit.xlsx"
DEFAULT_CONFIG = ROOT / "configs/task2_financial_consistency_v1.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/event_credibility/task2_financial_enhanced_cached_dea_v1"
DEFAULT_REPORT_JSON = ROOT / "reports/task2_financial_enhancement_report.json"
DEFAULT_REPORT_MD = ROOT / "reports/task2_financial_enhancement_report.md"

DEFAULT_CSMAR_HINTS = [
    ROOT / "data/external/csmar",
    ROOT / "data/csmar",
    ROOT / "data/raw/csmar",
    ROOT / "data/processed/csmar",
    ROOT / "inputs/csmar",
]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def first_evidence_text(row: dict[str, Any]) -> str:
    items = row.get("evidence_items") or []
    if isinstance(items, dict):
        items = [items]
    snippets: list[str] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        snippets.append(" ".join(
            str(item.get(field, ""))
            for field in ("title", "sentence", "content", "evidence_text")
            if str(item.get(field, "")).strip()
        ))
    return " || ".join(snippets)


def build_input_text(row: dict[str, Any]) -> tuple[str, str]:
    title = row.get("title_samples") or row.get("event_name") or row.get("claim") or row.get("event_claim") or ""
    body = first_evidence_text(row)
    if not body:
        body = row.get("event_claim") or ""
    return str(title), str(body)


def enrich_row(row: dict[str, Any], csmar_records: list[dict[str, Any]]) -> dict[str, Any]:
    title, body = build_input_text(row)
    company = row.get("target_name") or row.get("subject/company") or row.get("主体") or ""
    stock_code = row.get("target_code") or row.get("stock_code") or ""
    publish_time = row.get("event_time") or row.get("publish_time") or row.get("latest_publish_time") or row.get("earliest_publish_time") or ""
    report_period = row.get("event_time") or row.get("report_period") or row.get("event_date") or ""
    context = financial_consistency.analyze_financial_consistency(
        news_id=row.get("cluster_id") or row.get("event_id"),
        title=title,
        body=body,
        company_name=company,
        stock_code=stock_code,
        publish_time=publish_time,
        financial_claim_text=row.get("event_claim") or row.get("claim") or title,
        report_period=report_period,
        csmar_records=csmar_records,
    )
    merged = dict(row)
    merged["financial_context"] = context
    merged["financial_prompt_block"] = financial_consistency.format_financial_prompt_block(context.get("prompt_summary") or {})
    merged.update(financial_consistency.merge_with_original_scores(row, context))
    merged["news_title_proxy"] = title
    merged["news_body_proxy"] = body
    return merged


def _count_claim_types(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for claim in (row.get("financial_context") or {}).get("financial_claims") or []:
            counter[str(claim.get("claim_type") or "unknown")] += 1
    return dict(counter)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    context_rows = [row.get("financial_context") or {} for row in rows]
    verdicts = Counter(str(ctx.get("financial_consistency_verdict") or "missing") for ctx in context_rows)
    availability = Counter("has_csmar_records" if ctx.get("data_availability", {}).get("has_csmar_records") else "missing_csmar" for ctx in context_rows)
    prompt_rows = sum(1 for row in rows if str(row.get("financial_prompt_block") or "").strip())
    matched_rows = sum(1 for ctx in context_rows if ctx.get("financial_evidence_ids"))
    anomaly_rows = sum(1 for ctx in context_rows if ctx.get("financial_anomaly_flags"))
    score_values = [float(row.get("financial_consistency_score") or 0.0) for row in rows]
    adjustment_values = [float(row.get("financial_score_adjustment") or 0.0) for row in rows]
    enhanced_scores = [float(row.get("enhanced_trust_score") or 0.0) for row in rows]
    original_scores = [float(row.get("original_trust_score") or 0.0) for row in rows]
    return {
        "row_count": len(rows),
        "financial_consistency_verdict_distribution": dict(verdicts),
        "availability_distribution": dict(availability),
        "rows_with_prompt_block": prompt_rows,
        "rows_with_matched_financial_evidence": matched_rows,
        "rows_with_anomaly_flags": anomaly_rows,
        "mean_financial_consistency_score": round(sum(score_values) / max(1, len(score_values)), 6),
        "mean_financial_score_adjustment": round(sum(adjustment_values) / max(1, len(adjustment_values)), 6),
        "mean_original_trust_score": round(sum(original_scores) / max(1, len(original_scores)), 6),
        "mean_enhanced_trust_score": round(sum(enhanced_scores) / max(1, len(enhanced_scores)), 6),
        "claim_type_distribution": _count_claim_types(rows),
    }


def load_human_audit_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    sheets = pd.read_excel(path, sheet_name=None)
    summary: dict[str, Any] = {"available": True, "sheets": list(sheets)}
    if "Primary人工审核" in sheets and "Stress人工审核" in sheets:
        combined = pd.concat([sheets["Primary人工审核"], sheets["Stress人工审核"]], ignore_index=True)
        summary["human_row_count"] = int(len(combined))
        for column in ("human_verdict", "human_confidence", "claim_clarity", "subject_match", "time_match"):
            if column in combined.columns:
                summary[f"{column}_distribution"] = combined[column].fillna("").astype(str).value_counts().to_dict()
    return summary


def build_report(
    input_rows: list[dict[str, Any]],
    enriched_rows: list[dict[str, Any]],
    csmar_sources: list[str],
    missing_csmar_locations: list[str],
    workbook_summary: dict[str, Any],
    config: dict[str, Any],
    input_path: Path,
) -> dict[str, Any]:
    summary = summarize_rows(enriched_rows)
    prompt_examples = [
        {
            "cluster_id": row.get("cluster_id"),
            "financial_consistency_verdict": (row.get("financial_context") or {}).get("financial_consistency_verdict"),
            "financial_score_adjustment": row.get("financial_score_adjustment"),
            "enhanced_trust_score": row.get("enhanced_trust_score"),
            "financial_evidence_ids": (row.get("financial_context") or {}).get("financial_evidence_ids"),
            "financial_claim_count": len((row.get("financial_context") or {}).get("financial_claims") or []),
        }
        for row in enriched_rows[:8]
    ]
    repo_inventory = {
        "csmar_source_files": csmar_sources,
        "missing_csmar_locations": missing_csmar_locations,
        "mapping_template": build_mapping_template(),
    }
    report = {
        "status": "task2_financial_enhancement_dry_run",
        "input_path": str(input_path),
        "input_count": len(input_rows),
        "summary": summary,
        "prompt_examples": prompt_examples,
        "workbook_summary": workbook_summary,
        "repo_inventory": repo_inventory,
        "config": {
            "config_path": str(DEFAULT_CONFIG),
            "config_keys": sorted(config.keys()),
        },
    }
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    repo = report["repo_inventory"]
    workbook = report["workbook_summary"]
    lines = [
        "# Task2 Financial Enhancement Audit",
        "",
        "本报告是干运行审计：仓库内没有本地 CSMAR 数据，也没有可用的 Task2 语言模型服务，因此只验证财务增强层的抽取、适配和审计输出。",
        "",
        f"- 输入样本数: {report['input_count']}",
        f"- 财务一致性 verdict 分布: {json.dumps(summary['financial_consistency_verdict_distribution'], ensure_ascii=False)}",
        f"- 财务覆盖: {summary['rows_with_matched_financial_evidence']} / {summary['row_count']}",
        f"- prompt 财务块覆盖: {summary['rows_with_prompt_block']} / {summary['row_count']}",
        f"- 异常触发行数: {summary['rows_with_anomaly_flags']}",
        f"- 平均财务一致性分: {summary['mean_financial_consistency_score']}",
        f"- 平均财务调整: {summary['mean_financial_score_adjustment']}",
        f"- 平均 original trust score: {summary['mean_original_trust_score']}",
        f"- 平均 enhanced trust score: {summary['mean_enhanced_trust_score']}",
        "",
        "## Workbook",
        f"- workbook available: {workbook.get('available', False)}",
        f"- sheets: {json.dumps(workbook.get('sheets', []), ensure_ascii=False)}",
        f"- human row count: {workbook.get('human_row_count', 0)}",
        "",
        "## Repository Inventory",
        f"- csmar source files: {json.dumps(repo['csmar_source_files'], ensure_ascii=False)}",
        f"- missing CSMAR locations: {json.dumps(repo['missing_csmar_locations'], ensure_ascii=False)}",
        "",
        "## Prompt Examples",
    ]
    for item in report["prompt_examples"]:
        lines.append(
            f"- {item['cluster_id']}: verdict={item['financial_consistency_verdict']}, "
            f"adjustment={item['financial_score_adjustment']}, enhanced={item['enhanced_trust_score']}, "
            f"evidence={json.dumps(item['financial_evidence_ids'], ensure_ascii=False)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_manifest(paths: list[Path]) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in paths:
        if path.exists():
            manifest[str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)] = sha256(path)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = resolve(args.input_jsonl)
    workbook_path = resolve(args.gold_workbook)
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    config = load_config(config_path)
    input_rows = read_jsonl(input_path)
    if args.limit is not None and args.limit > 0:
        input_rows = input_rows[: args.limit]
    csmar_paths = []
    if args.csmar_source:
        for item in args.csmar_source:
            path = resolve(item)
            if path.exists():
                csmar_paths.append(path)
    else:
        for item in DEFAULT_CSMAR_HINTS:
            if item.exists():
                csmar_paths.append(item)
    csmar_sources = inventory_sources(csmar_paths[0]) if len(csmar_paths) == 1 and csmar_paths[0].is_dir() else [str(path) for path in csmar_paths]
    missing_locations = [str(path) for path in DEFAULT_CSMAR_HINTS if not path.exists()]
    missing_locations.extend(missing_expected_files(ROOT, ["data/processed/csmar", "data/external/csmar", "inputs/csmar"]))
    csmar_records = load_csmar_records(csmar_paths) if csmar_paths else []
    enriched_rows = [enrich_row(row, csmar_records) for row in input_rows]
    report = build_report(
        input_rows=input_rows,
        enriched_rows=enriched_rows,
        csmar_sources=csmar_sources,
        missing_csmar_locations=sorted(dict.fromkeys(missing_locations)),
        workbook_summary=load_human_audit_summary(workbook_path),
        config=config,
        input_path=input_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "financial_audit_rows.jsonl", enriched_rows)
    write_json(output_dir / "financial_audit_summary.json", report["summary"])
    write_json(output_dir / "financial_audit_report.json", report)
    write_json(output_dir / "manifest.json", {"status": report["status"], "output_dir": str(output_dir), "input_path": str(input_path), "workbook_path": str(workbook_path), "config_path": str(config_path), "sha256": sha256_manifest([input_path, workbook_path, config_path, output_dir / "financial_audit_rows.jsonl", output_dir / "financial_audit_summary.json", output_dir / "financial_audit_report.json"])})
    write_markdown(output_dir / "financial_audit_report.md", report)
    write_json(DEFAULT_REPORT_JSON, report)
    write_markdown(DEFAULT_REPORT_MD, report)
    report["output_dir"] = str(output_dir)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--gold-workbook", type=Path, default=DEFAULT_GOLD_WORKBOOK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--csmar-source", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps({
        "output_dir": report.get("output_dir", ""),
        "status": report.get("status", ""),
        "row_count": report.get("summary", {}).get("row_count", 0),
        "matched_rows": report.get("summary", {}).get("rows_with_matched_financial_evidence", 0),
        "prompt_rows": report.get("summary", {}).get("rows_with_prompt_block", 0),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

