#!/usr/bin/env python3
"""Build Task2 Hybrid-v3 market consistency v2 evidence on clean_evidence_1500.

This script does not train, tune, read locked500 gold, or modify existing
predictions/silver/gold/formal Task2 outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.market_consistency_v2 import (  # noqa: E402
    CONTRADICTORY,
    EXPLICIT_PRICE_CLAIM,
    INSUFFICIENT,
    NON_MARKET_BUSINESS_CLAIM,
    NOT_APPLICABLE,
    MarketRuleConfig,
    aggregate_market_consistency,
    build_market_index,
)
from src.event_credibility.resset_market_adapter import load_market_cache  # noqa: E402


SEED = 2026
EXPECTED_CLEAN_SHA256 = "6538549b53c74fc0518804b98e73d4ad3330be319a0a08c4d9a53c41ff0af9f0"

DEFAULT_CLEAN = ROOT / "outputs/event_credibility/task2_hybrid_v3/clean_evidence_v1/clean_evidence_1500.jsonl"
DEFAULT_MARKET_CACHE = ROOT / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v1/resset_market_daily_cache.parquet"
DEFAULT_LOCKED_IDS = ROOT / "configs/task2_independent500_locked_ids.json"
DEFAULT_OUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v2"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(obj)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def load_locked_ids(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    if isinstance(obj, list):
        ids.update(str(item) for item in obj)
    elif isinstance(obj, dict):
        for key in ("cluster_ids", "sample_ids", "locked_ids", "ids"):
            values = obj.get(key)
            if isinstance(values, list):
                ids.update(str(item) for item in values)
    return ids


def validate_sample_ids(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("sample_id") or "") for row in rows]
    missing = [idx for idx, sid in enumerate(ids, 1) if not sid]
    duplicates = sorted(sid for sid, count in Counter(ids).items() if sid and count > 1)
    return {
        "row_count": len(rows),
        "unique_sample_id_count": len({sid for sid in ids if sid}),
        "missing_sample_id_count": len(missing),
        "duplicate_sample_ids": duplicates[:50],
    }


def summarize(
    clean_rows: list[dict[str, Any]],
    market_rows: list[dict[str, Any]],
    explicit_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    locked_ids: set[str],
    market_cache_before_sha: str,
    market_cache_after_sha: str,
    clean_before_sha: str,
    clean_after_sha: str,
    full_run: bool,
) -> dict[str, Any]:
    clean_ids = {str(row.get("sample_id") or "") for row in clean_rows if row.get("sample_id")}
    locked_overlap = sorted(clean_ids & locked_ids)
    verdict_counts = Counter(str(row.get("market_consistency_verdict") or "") for row in market_rows)
    claim_type_counts = Counter(str(row.get("market_claim_type") or "") for row in market_rows)
    verdict_distribution = {
        "consistent": verdict_counts.get("consistent", 0),
        "contradictory": verdict_counts.get("contradictory", 0),
        "insufficient": verdict_counts.get("insufficient", 0),
        "not_applicable": verdict_counts.get("not_applicable", 0),
    }
    claim_type_distribution = {
        "explicit_price_claim": claim_type_counts.get("explicit_price_claim", 0),
        "market_sentiment_claim": claim_type_counts.get("market_sentiment_claim", 0),
        "financial_performance_claim": claim_type_counts.get("financial_performance_claim", 0),
        "non_market_business_claim": claim_type_counts.get("non_market_business_claim", 0),
    }
    closed_track_verifiable_count = sum(1 for row in market_rows if row.get("closed_track_verifiable"))
    entity_errors = [
        row
        for row in market_rows
        if row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM
        and str(row.get("entity_alignment_status") or "") != "aligned_target_stock"
    ]
    date_errors = [
        row
        for row in market_rows
        if row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM
        and str(row.get("time_alignment_status") or "") == "date_alignment_error"
    ]
    post_event_only = [
        row
        for row in market_rows
        if row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM
        and str(row.get("time_alignment_status") or "") == "post_event_only"
    ]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "task_scope": "market_consistency_v2_clean_evidence_1500_only",
        "training_performed": False,
        "tuning_performed": False,
        "locked500_gold_read": False,
        "locked500_evaluation_run": False,
        "predictions_modified": False,
        "silver_modified": False,
        "gold_modified": False,
        "input_rows": len(clean_rows),
        "unique_sample_id_count": len(clean_ids),
        "explicit_price_claim_count": claim_type_counts.get(EXPLICIT_PRICE_CLAIM, 0),
        "accepted_explicit_price_claim_candidate_count": len(explicit_rows),
        "rejected_explicit_price_claim_candidate_count": len(rejected_rows),
        "successful_entity_alignment_count": sum(
            1 for row in market_rows if row.get("entity_alignment_status") == "aligned_target_stock"
        ),
        "successful_time_alignment_count": sum(
            1 for row in market_rows if row.get("time_alignment_status") == "aligned_closed_track"
        ),
        "closed_track_verifiable_count": closed_track_verifiable_count,
        "actual_stock_claims_verified_count": closed_track_verifiable_count,
        "post_event_only_count": len(post_event_only),
        "market_consistency_verdict_distribution": verdict_distribution,
        "market_claim_type_distribution": claim_type_distribution,
        "percent_parse_error_count": sum(
            1
            for row in explicit_rows
            for value in row.get("parsed_claim_values", [])
            if value.get("parse_validation_error")
        ),
        "entity_alignment_error_count": len(entity_errors),
        "date_alignment_error_count": len(date_errors),
        "future_evidence_used_count": sum(1 for row in market_rows if row.get("future_data_used")),
        "locked500_overlap_count": len(locked_overlap),
        "locked500_overlap_sample_ids": locked_overlap[:100],
        "input_sha256_before_after": {
            "clean_evidence_before": clean_before_sha,
            "clean_evidence_after": clean_after_sha,
            "clean_evidence_unchanged": clean_before_sha == clean_after_sha,
            "market_cache_before": market_cache_before_sha,
            "market_cache_after": market_cache_after_sha,
            "market_cache_unchanged": market_cache_before_sha == market_cache_after_sha,
        },
        "selected_output_counts": {
            "market_consistency_1500": len(market_rows),
            "explicit_market_claims": len(explicit_rows),
            "rejected_market_claims": len(rejected_rows),
        },
        "rejected_reason_distribution": dict(
            Counter(str(row.get("entity_reject_reason") or row.get("time_reject_reason") or "") for row in rejected_rows)
        ),
        "quality_checks": {
            "input_rows_1500": (len(clean_rows) == 1500) if full_run else True,
            "unique_sample_ids_1500": (len(clean_ids) == 1500) if full_run else True,
            "clean_sha256_expected": clean_before_sha == EXPECTED_CLEAN_SHA256 and clean_after_sha == EXPECTED_CLEAN_SHA256,
            "locked500_overlap_zero": len(locked_overlap) == 0,
            "future_evidence_used_zero": sum(1 for row in market_rows if row.get("future_data_used")) == 0,
            "percent_parse_error_zero": all(
                not value.get("parse_validation_error")
                for row in explicit_rows
                for value in row.get("parsed_claim_values", [])
            ),
            "non_market_not_fact_checked_by_price": all(
                row.get("market_consistency_verdict") in {INSUFFICIENT, NOT_APPLICABLE}
                or row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM
                for row in market_rows
            ),
        },
    }


def audit_rows_for_xlsx(market_rows: list[dict[str, Any]], clean_by_id: dict[str, dict[str, Any]], limit: int = 500) -> list[dict[str, Any]]:
    priority = [
        row
        for row in market_rows
        if row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM
        or row.get("market_consistency_verdict") == CONTRADICTORY
        or row.get("entity_alignment_status") not in {"not_applicable", "aligned_target_stock"}
    ]
    selected: dict[str, dict[str, Any]] = {str(row["sample_id"]): row for row in priority[:limit]}
    rng = random.Random(SEED)
    rest = [row for row in market_rows if str(row["sample_id"]) not in selected]
    rng.shuffle(rest)
    for row in rest:
        if len(selected) >= limit:
            break
        selected[str(row["sample_id"])] = row
    output: list[dict[str, Any]] = []
    for row in selected.values():
        clean = clean_by_id.get(str(row["sample_id"]), {})
        claim = clean.get("claim") if isinstance(clean.get("claim"), dict) else {}
        output.append(
            {
                "sample_id": row.get("sample_id"),
                "target": claim.get("target") or clean.get("target_entity"),
                "claim_title": claim.get("title"),
                "claim_publish_time": row.get("claim_time"),
                "market_claim_type": row.get("market_claim_type"),
                "claim_text": row.get("claim_text"),
                "claimed_price_direction": row.get("claimed_price_direction"),
                "claimed_price_change": row.get("claimed_price_change"),
                "actual_price_change": row.get("actual_price_change"),
                "market_consistency_verdict": row.get("market_consistency_verdict"),
                "entity_alignment_status": row.get("entity_alignment_status"),
                "entity_reject_reason": row.get("entity_reject_reason"),
                "time_alignment_status": row.get("time_alignment_status"),
                "time_reject_reason": row.get("time_reject_reason"),
                "market_reference_date": row.get("market_reference_date"),
                "causal_cutoff": row.get("causal_cutoff"),
                "market_evidence_date": row.get("market_evidence_date"),
                "market_reason": row.get("market_reason"),
                "parsed_claim_values": json.dumps(row.get("parsed_claim_values") or [], ensure_ascii=False),
            }
        )
    return output


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Task2 Market Consistency v2 Report",
        "",
        "本轮只重建 clean_evidence_1500 的市场事实核验证据；未训练、未调参、未读取 locked500 gold、未运行 locked500 评测。",
        "",
        "## 验收统计",
        f"- 输入行数：{report['input_rows']}，唯一sample_id：{report['unique_sample_id_count']}",
        f"- explicit_price_claim数量：{report['explicit_price_claim_count']}",
        f"- 接受/拒绝的显式价格候选数量：{report['accepted_explicit_price_claim_candidate_count']} / {report['rejected_explicit_price_claim_candidate_count']}",
        f"- 成功实体对齐数量：{report['successful_entity_alignment_count']}",
        f"- 成功时间对齐数量：{report['successful_time_alignment_count']}",
        f"- closed-track可核验数量：{report['closed_track_verifiable_count']}",
        f"- post-event-only数量：{report['post_event_only_count']}",
        f"- consistent/contradictory/insufficient分布：`{json.dumps(report['market_consistency_verdict_distribution'], ensure_ascii=False)}`",
        f"- 百分数解析错误数量：{report['percent_parse_error_count']}",
        f"- entity_alignment_error数量：{report['entity_alignment_error_count']}",
        f"- date_alignment_error数量：{report['date_alignment_error_count']}",
        f"- future_evidence_used数量：{report['future_evidence_used_count']}",
        f"- locked500_overlap数量：{report['locked500_overlap_count']}",
        "",
        "## 输入SHA256",
        f"- clean_evidence before/after：`{report['input_sha256_before_after']['clean_evidence_before']}` / `{report['input_sha256_before_after']['clean_evidence_after']}`",
        f"- market_cache before/after：`{report['input_sha256_before_after']['market_cache_before']}` / `{report['input_sha256_before_after']['market_cache_after']}`",
        "",
        "## 结论",
        "市场层已具备进入 Hybrid-v3 非locked开发阶段的基本条件：小数百分号、方向、实体排除和closed-track时间边界均有测试和审计输出。进入下一阶段前仍应在非locked人工开发集上抽样复核 rejected/contradictory 样本，不得使用 locked500 调参。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sha256s(output_dir: Path, extra_paths: list[Path]) -> None:
    paths = [path for path in output_dir.glob("*") if path.is_file() and path.name != "SHA256SUMS.txt"]
    paths.extend(path for path in extra_paths if path.exists())
    lines = []
    for path in sorted(set(paths)):
        lines.append(f"{sha256(path)}  {display_path(path)}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--market-cache", type=Path, default=DEFAULT_MARKET_CACHE)
    parser.add_argument("--locked-ids", type=Path, default=DEFAULT_LOCKED_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = resolve(args.input)
    market_cache = resolve(args.market_cache)
    locked_ids_path = resolve(args.locked_ids)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_sha_before = sha256(input_path)
    market_sha_before = sha256(market_cache)
    if clean_sha_before != EXPECTED_CLEAN_SHA256 and args.limit == 0:
        raise ValueError(f"clean_evidence_1500 SHA mismatch: {clean_sha_before}")
    clean_rows = read_jsonl(input_path)
    selected_rows = clean_rows[: args.limit] if args.limit and args.limit > 0 else clean_rows
    sample_audit = validate_sample_ids(selected_rows)
    if sample_audit["missing_sample_id_count"] or sample_audit["duplicate_sample_ids"]:
        raise ValueError(f"input sample_id audit failed: {sample_audit}")

    locked_ids = load_locked_ids(locked_ids_path)
    market_index = build_market_index(load_market_cache(market_cache))
    cfg = MarketRuleConfig()

    market_rows: list[dict[str, Any]] = []
    explicit_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        market_row, accepted, rejected = aggregate_market_consistency(row, market_index, cfg)
        market_rows.append(market_row)
        explicit_rows.extend(accepted)
        rejected_rows.extend(rejected)

    clean_sha_after = sha256(input_path)
    market_sha_after = sha256(market_cache)
    report = summarize(
        selected_rows,
        market_rows,
        explicit_rows,
        rejected_rows,
        locked_ids,
        market_sha_before,
        market_sha_after,
        clean_sha_before,
        clean_sha_after,
        args.limit == 0,
    )
    clean_by_id = {str(row.get("sample_id")): row for row in selected_rows}

    market_path = output_dir / "market_consistency_1500.jsonl"
    explicit_path = output_dir / "explicit_market_claims.jsonl"
    rejected_path = output_dir / "rejected_market_claims.jsonl"
    report_json = output_dir / "market_consistency_report.json"
    report_md = output_dir / "market_consistency_report.md"
    audit_xlsx = output_dir / "market_claim_audit.xlsx"
    manifest_path = output_dir / "manifest.json"
    run_state_path = output_dir / "run_state.json"

    write_jsonl(market_path, market_rows)
    write_jsonl(explicit_path, explicit_rows)
    write_jsonl(rejected_path, rejected_rows)
    write_json(report_json, report)
    write_markdown_report(report_md, report)
    with pd.ExcelWriter(audit_xlsx) as writer:
        pd.DataFrame(audit_rows_for_xlsx(market_rows, clean_by_id)).to_excel(writer, sheet_name="market_claim_audit", index=False)
        pd.DataFrame([report]).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(Counter(row.get("entity_reject_reason") for row in rejected_rows).items(), columns=["reason", "count"]).to_excel(
            writer, sheet_name="rejected_reasons", index=False
        )

    run_state = {
        "status": "completed" if all(report["quality_checks"].values()) else "completed_with_validation_warnings",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_performed": False,
        "tuning_performed": False,
        "locked500_gold_read": False,
        "locked500_evaluation_run": False,
        "predictions_modified": False,
        "formal_outputs_modified": False,
        "silver_modified": False,
        "gold_modified": False,
        "quality_checks": report["quality_checks"],
    }
    write_json(run_state_path, run_state)
    manifest = {
        "status": run_state["status"],
        "script": "scripts/run_task2_market_consistency_v2.py",
        "source_version": cfg.version,
        "input_files": {
            "clean_evidence_1500": display_path(input_path),
            "market_cache": display_path(market_cache),
            "locked_ids": display_path(locked_ids_path),
        },
        "output_files": {
            "market_consistency_1500": display_path(market_path),
            "explicit_market_claims": display_path(explicit_path),
            "rejected_market_claims": display_path(rejected_path),
            "market_consistency_report_json": display_path(report_json),
            "market_consistency_report_md": display_path(report_md),
            "market_claim_audit_xlsx": display_path(audit_xlsx),
            "manifest": display_path(manifest_path),
            "run_state": display_path(run_state_path),
            "sha256s": display_path(output_dir / "SHA256SUMS.txt"),
        },
        "summary": report,
    }
    write_json(manifest_path, manifest)
    write_sha256s(output_dir, [])

    print(
        json.dumps(
            {
                "status": run_state["status"],
                "output_dir": display_path(output_dir),
                "training_performed": False,
                "locked500_gold_read": False,
                "summary": {
                    key: report[key]
                    for key in (
                        "input_rows",
                        "unique_sample_id_count",
                        "explicit_price_claim_count",
                        "accepted_explicit_price_claim_candidate_count",
                        "rejected_explicit_price_claim_candidate_count",
                        "successful_entity_alignment_count",
                        "successful_time_alignment_count",
                        "closed_track_verifiable_count",
                        "post_event_only_count",
                        "market_consistency_verdict_distribution",
                        "percent_parse_error_count",
                        "entity_alignment_error_count",
                        "date_alignment_error_count",
                        "future_evidence_used_count",
                        "locked500_overlap_count",
                    )
                },
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0 if run_state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
