#!/usr/bin/env python3
"""Run Task2 Hybrid-v3 market-consistency dry-run on 1500 silver samples.

This script reads clean evidence and AI-silver labels only to attach market
features and coverage/audit summaries.  It never reads locked-500 gold labels,
trains models, tunes thresholds, or modifies existing Task2 outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.market_consistency import (  # noqa: E402
    EXPLICIT_PRICE_CLAIM,
    MARKET_SENTIMENT_CLAIM,
    NON_MARKET_BUSINESS_CLAIM,
    MarketConsistencyConfig,
    build_market_lookup,
    compute_market_consistency,
)
from src.event_credibility.resset_market_adapter import load_market_cache, sha256_file  # noqa: E402


SEED = 2026
EXPECTED_CLEAN_SHA256 = "6538549b53c74fc0518804b98e73d4ad3330be319a0a08c4d9a53c41ff0af9f0"

OUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v1"
CACHE_FILE = OUT_DIR / "resset_market_daily_cache.parquet"
CACHE_MANIFEST = OUT_DIR / "resset_market_cache_manifest.json"
CLEAN_EVIDENCE = ROOT / "outputs/event_credibility/task2_hybrid_v3/clean_evidence_v1/clean_evidence_1500.jsonl"
SILVER_FINAL = ROOT / "outputs/event_credibility/task2_hybrid_v3/heterogeneous_silver_v2/silver_final_1500.jsonl"
SILVER_ELIGIBLE = ROOT / "outputs/event_credibility/task2_hybrid_v3/heterogeneous_silver_v2/silver_training_eligible.jsonl"
LOCKED_IDS = ROOT / "configs/task2_independent500_locked_ids.json"
CONFIG_FILE = ROOT / "configs/task2_market_consistency_v1.yaml"

MARKET_JSONL = OUT_DIR / "market_consistency_1500.jsonl"
CANDIDATES_JSONL = OUT_DIR / "market_consistency_candidates.jsonl"
REPORT_JSON = OUT_DIR / "market_consistency_report.json"
REPORT_MD = OUT_DIR / "market_consistency_report.md"
AUDIT_XLSX = OUT_DIR / "market_consistency_audit_200.xlsx"
RUN_STATE = OUT_DIR / "run_state.json"
SHA_FILE = OUT_DIR / "SHA256SUMS.txt"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def index_by_sample_id(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    ids = [str(row.get("sample_id") or "") for row in rows]
    missing = [idx for idx, sid in enumerate(ids, 1) if not sid]
    duplicates = sorted(sid for sid, count in Counter(ids).items() if sid and count > 1)
    if missing or duplicates:
        raise ValueError(f"{name} invalid sample_id: missing={len(missing)}, duplicates={len(duplicates)}")
    return {str(row["sample_id"]): row for row in rows}


def _count_if(rows: list[dict[str, Any]], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def build_report(
    market_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    silver_rows: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    locked_ids: set[str],
    cache_manifest: dict[str, Any],
    config: MarketConsistencyConfig,
) -> dict[str, Any]:
    sample_ids = [str(row.get("sample_id") or "") for row in market_rows]
    claim_type_counts = Counter(row.get("market_claim_type") for row in market_rows)
    verdict_counts = Counter(row.get("market_verdict") for row in market_rows)
    original_label_counts = Counter(row.get("original_silver_label") for row in market_rows)
    proposed_counts = Counter(row.get("proposed_label_change") or "none" for row in market_rows)
    missing_counter: Counter[str] = Counter()
    for row in market_rows:
        for reason in str(row.get("missing_reason") or "").split(";"):
            if reason:
                missing_counter[reason] += 1

    insufficient_direct = [
        row
        for row in market_rows
        if row.get("original_silver_label") == "insufficient"
        and row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM
        and row.get("post_event_signal_available")
    ]
    possible_support = [row for row in insufficient_direct if row.get("market_verdict") == "consistent"]
    possible_contradict = [row for row in insufficient_direct if row.get("market_verdict") == "contradictory"]

    locked_overlap = sorted(set(sample_ids) & locked_ids)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "task_scope": "market_consistency_feature_audit_only",
        "training_performed": False,
        "tuning_performed": False,
        "locked500_gold_read": False,
        "locked500_used_for_threshold_selection": False,
        "formal_task2_outputs_modified": False,
        "original_silver_modified": False,
        "input_counts": {
            "clean_evidence_rows": len(clean_rows),
            "clean_evidence_unique_sample_ids": len({str(row.get("sample_id") or "") for row in clean_rows}),
            "silver_final_rows": len(silver_rows),
            "silver_training_eligible_rows": len(eligible_rows),
            "market_consistency_rows": len(market_rows),
            "market_consistency_unique_sample_ids": len(set(sample_ids)),
        },
        "input_sha256": {
            "clean_evidence": sha256_file(CLEAN_EVIDENCE),
            "silver_final": sha256_file(SILVER_FINAL),
            "silver_training_eligible": sha256_file(SILVER_ELIGIBLE),
            "locked_ids": sha256_file(LOCKED_IDS),
            "market_cache": sha256_file(CACHE_FILE) if CACHE_FILE.exists() else "",
        },
        "resset_cache": cache_manifest,
        "thresholds": {
            "direction_unclear_abs_return": config.direction_unclear_abs_return,
            "explicit_contradict_abs_return": config.explicit_contradict_abs_return,
            "explicit_adjustment_cap": config.explicit_adjustment_cap,
            "sentiment_adjustment_cap": config.sentiment_adjustment_cap,
            "benchmark_type": cache_manifest.get("benchmark_type"),
        },
        "coverage": {
            "stock_code_identified": _count_if(market_rows, lambda r: bool(r.get("trading_code"))),
            "a_share_code": _count_if(market_rows, lambda r: "non_a_share_or_invalid_code" not in str(r.get("missing_reason") or "") and bool(r.get("trading_code"))),
            "date_in_market_coverage": _count_if(market_rows, lambda r: "publish_date_outside_market_coverage" not in str(r.get("missing_reason") or "") and bool(r.get("article_publish_time"))),
            "matched_market_rows": _count_if(market_rows, lambda r: bool(r.get("matched_trading_date"))),
            "causal_no_future_available": _count_if(market_rows, lambda r: bool(r.get("causal_signal_available"))),
            "post_event_audit_available": _count_if(market_rows, lambda r: bool(r.get("post_event_signal_available"))),
        },
        "claim_type_distribution": dict(claim_type_counts),
        "market_verdict_distribution": dict(verdict_counts),
        "original_silver_label_distribution": dict(original_label_counts),
        "proposed_label_change_distribution": dict(proposed_counts),
        "missing_reason_counts": dict(missing_counter),
        "explicit_price_claim_count": claim_type_counts.get(EXPLICIT_PRICE_CLAIM, 0),
        "market_sentiment_claim_count": claim_type_counts.get(MARKET_SENTIMENT_CLAIM, 0),
        "non_market_business_claim_count": claim_type_counts.get(NON_MARKET_BUSINESS_CLAIM, 0),
        "consistent_count": verdict_counts.get("consistent", 0) + verdict_counts.get("weak_consistent", 0),
        "contradictory_count": verdict_counts.get("contradictory", 0) + verdict_counts.get("weak_contradictory", 0),
        "insufficient_count": verdict_counts.get("insufficient", 0),
        "original_insufficient_direct_market_check_count": len(insufficient_direct),
        "market_layer_possible_support_count": len(possible_support),
        "market_layer_possible_contradict_count": len(possible_contradict),
        "possible_label_change_candidate_count": len([row for row in market_rows if row.get("proposed_label_change")]),
        "locked500_overlap_count": len(locked_overlap),
        "locked500_overlap_sample_ids": locked_overlap[:50],
        "future_relative_to_publication_count": _count_if(market_rows, lambda r: bool(r.get("future_relative_to_publication"))),
        "forbidden_in_closed_track_count": _count_if(market_rows, lambda r: bool(r.get("forbidden_in_closed_track"))),
        "quality_checks": {
            "input_rows_1500": len(clean_rows) == 1500,
            "output_rows_1500": len(market_rows) == 1500,
            "output_sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
            "clean_evidence_sha256_matches_expected": sha256_file(CLEAN_EVIDENCE) == EXPECTED_CLEAN_SHA256,
            "locked500_overlap_zero": len(locked_overlap) == 0,
            "market_score_all_in_0_1": all(0.0 <= float(row.get("market_consistency_score", -1)) <= 1.0 for row in market_rows),
            "non_market_adjustment_zero": all(
                abs(float(row.get("suggested_market_adjustment") or 0.0)) == 0.0
                for row in market_rows
                if row.get("market_claim_type") == NON_MARKET_BUSINESS_CLAIM
            ),
        },
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    cache = report["resset_cache"]
    coverage = report["coverage"]
    lines = [
        "# Task2 Market Consistency v1 Dry-run Report",
        "",
        "本报告只记录市场反应一致性特征和覆盖率审计，不训练、不调参、不修改原银标或正式Task2输出。",
        "",
        "## RESSET行情缓存",
        f"- source_file: `{cache.get('source_file')}`",
        f"- source_sha256: `{cache.get('source_sha256')}`",
        f"- worksheet_dimension_ref: `{cache.get('worksheet_dimension_ref')}`",
        f"- excel_dimension_recovered: `{cache.get('excel_dimension_recovered')}`",
        f"- rows: `{cache.get('cache_rows')}`",
        f"- unique_trading_codes: `{cache.get('unique_trading_codes')}`",
        f"- date_range: `{cache.get('date_min')}` to `{cache.get('date_max')}`",
        f"- missing_close_rows: `{cache.get('missing_close_rows')}`",
        f"- benchmark_type: `{cache.get('benchmark_type')}`",
        "",
        "## 1500条覆盖率",
        f"- stock_code_identified: `{coverage.get('stock_code_identified')}`",
        f"- a_share_code: `{coverage.get('a_share_code')}`",
        f"- date_in_market_coverage: `{coverage.get('date_in_market_coverage')}`",
        f"- matched_market_rows: `{coverage.get('matched_market_rows')}`",
        f"- causal_no_future_available: `{coverage.get('causal_no_future_available')}`",
        f"- post_event_audit_available: `{coverage.get('post_event_audit_available')}`",
        "",
        "## 主张与判断分布",
        f"- claim_type_distribution: `{report.get('claim_type_distribution')}`",
        f"- market_verdict_distribution: `{report.get('market_verdict_distribution')}`",
        f"- possible_label_change_candidate_count: `{report.get('possible_label_change_candidate_count')}`",
        "",
        "## 隔离检查",
        f"- locked500_overlap_count: `{report.get('locked500_overlap_count')}`",
        f"- locked500_gold_read: `{report.get('locked500_gold_read')}`",
        f"- training_performed: `{report.get('training_performed')}`",
        f"- tuning_performed: `{report.get('tuning_performed')}`",
        f"- original_silver_modified: `{report.get('original_silver_modified')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_sample_rows(market_rows: list[dict[str, Any]], clean_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    priority = [
        row
        for row in market_rows
        if row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM
        or row.get("market_verdict") in {"contradictory", "weak_contradictory"}
        or row.get("proposed_label_change")
        or "stock_code" in str(row.get("missing_reason") or "")
    ]
    selected: dict[str, dict[str, Any]] = {str(row["sample_id"]): row for row in priority[:200]}
    remaining = [row for row in market_rows if str(row["sample_id"]) not in selected]
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= 200:
            break
        selected[str(row["sample_id"])] = row

    audit_rows: list[dict[str, Any]] = []
    for row in selected.values():
        clean = clean_by_id[str(row["sample_id"])]
        claim = clean.get("claim") if isinstance(clean.get("claim"), dict) else {}
        audit_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "新闻标题": claim.get("title"),
                "新闻发布时间": row.get("article_publish_time"),
                "股票代码": row.get("trading_code"),
                "公司名称": row.get("company_name"),
                "原始核心主张": claim.get("event_claim"),
                "市场主张类型": row.get("market_claim_type"),
                "声称方向": row.get("claimed_direction"),
                "匹配交易日": row.get("matched_trading_date"),
                "前收盘价": row.get("previous_close"),
                "当前收盘价": row.get("close"),
                "原始收益率": row.get("daily_return"),
                "市场中位收益率": row.get("market_median_return"),
                "市场调整收益率": row.get("market_adjusted_return"),
                "换手率异常": row.get("turnover_abnormal_ratio"),
                "市场一致性判断": row.get("market_verdict"),
                "是否使用未来行情": row.get("future_relative_to_publication"),
                "原银标": row.get("original_silver_label"),
                "建议变化": row.get("proposed_label_change"),
                "解释": row.get("verdict_reason") or row.get("missing_reason"),
            }
        )
    return audit_rows


def write_sha256s(paths: list[Path]) -> None:
    rows = []
    for path in paths:
        if path.exists() and path != SHA_FILE:
            rows.append(f"{sha256_file(path)}  {path.relative_to(ROOT)}")
    SHA_FILE.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists() or not CACHE_MANIFEST.exists():
        RUN_STATE.write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "blocked_stage": "market_cache_missing",
                    "blocker_reason": "Run scripts/build_task2_market_consistency_cache.py first.",
                    "training_performed": False,
                    "tuning_performed": False,
                    "formal_outputs_modified": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return 2

    clean_sha = sha256_file(CLEAN_EVIDENCE)
    if clean_sha != EXPECTED_CLEAN_SHA256:
        raise ValueError(f"clean evidence SHA mismatch: {clean_sha} != {EXPECTED_CLEAN_SHA256}")

    clean_rows = read_jsonl(CLEAN_EVIDENCE)
    silver_rows = read_jsonl(SILVER_FINAL)
    eligible_rows = read_jsonl(SILVER_ELIGIBLE)
    clean_by_id = index_by_sample_id(clean_rows, "clean_evidence")
    silver_by_id = index_by_sample_id(silver_rows, "silver_final")
    locked_ids = load_locked_ids(LOCKED_IDS)
    if set(clean_by_id) != set(silver_by_id):
        missing = sorted(set(clean_by_id) ^ set(silver_by_id))[:20]
        raise ValueError(f"clean/silver sample_id mismatch examples={missing}")

    market_df = load_market_cache(CACHE_FILE)
    market_lookup = build_market_lookup(market_df)
    config = MarketConsistencyConfig()

    market_rows: list[dict[str, Any]] = []
    for sid in sorted(clean_by_id):
        silver = silver_by_id[sid]
        original_label = str(silver.get("final_label") or "")
        market_rows.append(compute_market_consistency(clean_by_id[sid], original_label, market_lookup, config))

    write_jsonl(MARKET_JSONL, market_rows)
    candidate_rows = [
        row
        for row in market_rows
        if row.get("market_claim_type") != NON_MARKET_BUSINESS_CLAIM
        or row.get("market_verdict") in {"consistent", "contradictory", "weak_consistent", "weak_contradictory"}
        or row.get("proposed_label_change")
    ]
    write_jsonl(CANDIDATES_JSONL, candidate_rows)

    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    report = build_report(market_rows, clean_rows, silver_rows, eligible_rows, locked_ids, cache_manifest, config)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report, REPORT_MD)

    audit_rows = audit_sample_rows(market_rows, clean_by_id)
    pd.DataFrame(audit_rows).to_excel(AUDIT_XLSX, index=False)

    run_state = {
        "status": "completed" if all(report["quality_checks"].values()) else "completed_with_validation_warnings",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_stage": "market_consistency_dryrun_1500",
        "training_performed": False,
        "tuning_performed": False,
        "locked500_gold_read": False,
        "formal_outputs_modified": False,
        "original_silver_modified": False,
        "quality_checks": report["quality_checks"],
        "output_files": {
            "market_consistency_1500": str(MARKET_JSONL),
            "market_consistency_candidates": str(CANDIDATES_JSONL),
            "market_consistency_report_json": str(REPORT_JSON),
            "market_consistency_report_md": str(REPORT_MD),
            "market_consistency_audit_200": str(AUDIT_XLSX),
        },
    }
    RUN_STATE.write_text(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    source_market_file = Path(str(cache_manifest.get("source_file") or ""))
    write_sha256s(
        [
            source_market_file,
            CACHE_FILE,
            CACHE_MANIFEST,
            CLEAN_EVIDENCE,
            SILVER_FINAL,
            SILVER_ELIGIBLE,
            LOCKED_IDS,
            CONFIG_FILE,
            MARKET_JSONL,
            CANDIDATES_JSONL,
            REPORT_JSON,
            REPORT_MD,
            AUDIT_XLSX,
            RUN_STATE,
        ]
    )

    print(
        json.dumps(
            {
                "market_rows": len(market_rows),
                "candidate_rows": len(candidate_rows),
                "claim_type_distribution": report["claim_type_distribution"],
                "market_verdict_distribution": report["market_verdict_distribution"],
                "quality_checks": report["quality_checks"],
                "report": str(REPORT_JSON),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(report["quality_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
