#!/usr/bin/env python3
"""Build Task2 Hybrid-v3 market consistency v2.1 evidence on clean_evidence_1500.

This runner does not train, tune, read locked500 gold, run locked500
evaluation, or modify previous market v2 outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_task2_market_consistency_v2 as base_runner  # noqa: E402
from src.event_credibility.market_consistency_v21 import (  # noqa: E402
    VERSION,
    MarketRuleConfig,
    aggregate_market_consistency,
    build_market_index,
    target_directly_bound_to_price_expression,
)
from src.event_credibility.market_consistency_v2 import CONSISTENT, EXPLICIT_PRICE_CLAIM  # noqa: E402
from src.event_credibility.resset_market_adapter import load_market_cache  # noqa: E402


SEED = 2026
EXPECTED_CLEAN_SHA256 = base_runner.EXPECTED_CLEAN_SHA256
DEFAULT_CLEAN = base_runner.DEFAULT_CLEAN
DEFAULT_MARKET_CACHE = base_runner.DEFAULT_MARKET_CACHE
DEFAULT_LOCKED_IDS = base_runner.DEFAULT_LOCKED_IDS
DEFAULT_OUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v21"
WARNING_SAMPLE_IDS = {"EC-000034", "EC-000153__v2"}


def norm(value: Any) -> str:
    return base_runner.norm(value) if hasattr(base_runner, "norm") else str(value or "").strip()


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except Exception:
        return default
    return number if math.isfinite(number) else default


def parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return pd.to_datetime(text).to_pydatetime()
    except Exception:
        return None


def percent_parse_ok(row: dict[str, Any]) -> bool:
    values = row.get("parsed_claim_values")
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = []
    if not isinstance(values, list):
        values = []
    if not values and "raw_value" in row:
        values = [row]
    for value in values:
        if not isinstance(value, dict):
            continue
        if value.get("parse_validation_error"):
            return False
        unit = norm(value.get("unit"))
        raw_value = value.get("raw_value")
        normalized = value.get("normalized_value")
        if unit in {"%", "个百分点", "个点", "pp", "PP"} and raw_value is not None and normalized is not None:
            if not math.isclose(abs(safe_float(normalized, 0.0) or 0.0), abs(safe_float(raw_value, 0.0) or 0.0) / 100.0, rel_tol=0.0, abs_tol=1e-12):
                return False
    return True


def direction_parse_ok(row: dict[str, Any]) -> bool:
    text = norm(row.get("raw_text") or row.get("claim_text"))
    direction = norm(row.get("direction") or row.get("claimed_price_direction"))
    if any(term in text for term in ("跌", "下跌", "收跌", "跌停", "下挫", "走弱")):
        return direction in {"down", ""}
    if any(term in text for term in ("涨", "上涨", "收涨", "涨停", "上扬", "走强")):
        return direction in {"up", ""}
    return True


def market_time_ok(row: dict[str, Any]) -> bool:
    if row.get("future_data_used"):
        return False
    status = norm(row.get("time_alignment_status"))
    if status in {"", "not_applicable"}:
        return True
    if status == "aligned_closed_track":
        evidence_dt = parse_dt(row.get("market_evidence_date"))
        cutoff_dt = parse_dt(row.get("causal_cutoff"))
        return bool(evidence_dt and cutoff_dt and evidence_dt.date() <= cutoff_dt.date())
    if status == "post_event_only":
        return not bool(row.get("closed_track_verifiable"))
    return status == "date_alignment_error"


def closed_verdict_supported(row: dict[str, Any]) -> bool:
    verdict = norm(row.get("market_consistency_verdict"))
    if not row.get("closed_track_verifiable"):
        return True
    actual = safe_float(row.get("actual_price_change"), 0.0) or 0.0
    claimed = row.get("claimed_price_change")
    claimed_direction = norm(row.get("claimed_price_direction"))
    actual_direction = "up" if actual > 0 else ("down" if actual < 0 else "flat")
    if verdict == "consistent":
        if claimed_direction in {"up", "down"} and claimed_direction != actual_direction:
            return False
        if claimed is not None:
            denominator = abs(safe_float(claimed, 0.0) or 0.0)
            if denominator > 0:
                return abs(abs(actual) - denominator) / denominator <= 0.20 + 1e-9
        return actual_direction in {"up", "down"} or "limit_up_down" in norm(row.get("market_reason"))
    if verdict == "contradictory":
        return claimed_direction in {"up", "down"} and claimed_direction != actual_direction or "over_20pct" in norm(row.get("market_reason"))
    return False


def market_entity_ok(row: dict[str, Any], audit_kind: str) -> bool:
    status = norm(row.get("entity_alignment_status"))
    reason = norm(row.get("entity_reject_reason"))
    claim_type = norm(row.get("market_claim_type"))
    if audit_kind == "closed_track_consistent":
        return status == "aligned_target_stock" and claim_type == EXPLICIT_PRICE_CLAIM
    if status == "aligned_target_stock":
        return True
    if status in {"not_applicable", ""}:
        return True
    text = norm(row.get("claim_text") or row.get("text"))
    raw = norm(row.get("raw_text"))
    if raw and text:
        pos = text.find(raw)
        bound, _subject = target_directly_bound_to_price_expression(
            text,
            norm(row.get("target_name")),
            norm(row.get("target_code")),
            pos,
            pos + len(raw),
        ) if pos >= 0 else (False, "")
    else:
        bound = False
    if "excluded_instrument" in reason or "non_target" in reason or "price_expression_bound_to_non_target" in reason:
        return not bound
    return False


def run_quality_gate(
    market_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    audit_xlsx_path: Path,
    quality_json_path: Path,
    seed: int,
) -> dict[str, Any]:
    closed = [
        row
        for row in market_rows
        if row.get("closed_track_verifiable") and row.get("market_consistency_verdict") == CONSISTENT
    ]
    insufficient = [row for row in market_rows if row.get("market_consistency_verdict") == "insufficient"]
    sampled_insufficient = random.Random(seed).sample(insufficient, min(30, len(insufficient)))
    audit_rows: list[dict[str, Any]] = []
    for kind, rows in (
        ("closed_track_consistent", closed),
        ("rejected", rejected_rows),
        ("insufficient_seed2026_sample", sampled_insufficient),
    ):
        for row in rows:
            audit_rows.append(
                {
                    "audit_kind": kind,
                    "sample_id": row.get("sample_id"),
                    "target_name": row.get("target_name"),
                    "target_code": row.get("target_code"),
                    "claim_text": row.get("claim_text") or row.get("text"),
                    "raw_text": row.get("raw_text"),
                    "market_claim_type": row.get("market_claim_type"),
                    "market_consistency_verdict": row.get("market_consistency_verdict"),
                    "market_reason": row.get("market_reason"),
                    "entity_alignment_status": row.get("entity_alignment_status"),
                    "entity_reject_reason": row.get("entity_reject_reason"),
                    "time_alignment_status": row.get("time_alignment_status"),
                    "time_reject_reason": row.get("time_reject_reason"),
                    "market_evidence_date": row.get("market_evidence_date"),
                    "causal_cutoff": row.get("causal_cutoff"),
                    "claimed_price_direction": row.get("claimed_price_direction") or row.get("direction"),
                    "claimed_price_change": row.get("claimed_price_change"),
                    "actual_price_change": row.get("actual_price_change"),
                    "percent_parse_ok": percent_parse_ok(row),
                    "direction_parse_ok": direction_parse_ok(row),
                    "entity_alignment_ok": market_entity_ok(row, kind),
                    "time_alignment_ok": market_time_ok(row),
                    "closed_track_verdict_supported": closed_verdict_supported(row),
                    "future_data_used": bool(row.get("future_data_used")),
                    "audit_note": "",
                }
            )
    parsed_rows = [
        row
        for row in audit_rows
        if row["raw_text"] or row["claimed_price_change"] is not None or row["market_claim_type"] == EXPLICIT_PRICE_CLAIM
    ]
    percent_accuracy = sum(row["percent_parse_ok"] for row in parsed_rows) / max(1, len(parsed_rows))
    entity_accuracy = sum(row["entity_alignment_ok"] for row in audit_rows) / max(1, len(audit_rows))
    time_accuracy = sum(row["time_alignment_ok"] for row in audit_rows) / max(1, len(audit_rows))
    closed_accuracy = sum(row["closed_track_verdict_supported"] for row in audit_rows if row["audit_kind"] == "closed_track_consistent") / max(1, len(closed))
    future_count = sum(1 for row in market_rows if row.get("future_data_used"))
    warning_rows = [row for row in market_rows if norm(row.get("sample_id")) in WARNING_SAMPLE_IDS]
    warning_fix = {
        row["sample_id"]: {
            "entity_alignment_status": row.get("entity_alignment_status"),
            "time_alignment_status": row.get("time_alignment_status"),
            "market_consistency_verdict": row.get("market_consistency_verdict"),
            "accepted_market_claim_count": row.get("accepted_market_claim_count"),
            "rejected_market_claim_count": row.get("rejected_market_claim_count"),
            "fixed": row.get("entity_alignment_status") == "aligned_target_stock" and row.get("accepted_market_claim_count", 0) >= 1,
        }
        for row in warning_rows
    }
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "source_version": VERSION,
        "locked500_gold_read": False,
        "locked500_evaluation_run": False,
        "training_performed": False,
        "tuning_performed": False,
        "audit_method": "deterministic audit over nonlocked market_consistency_v21 outputs; no gold labels used",
        "closed_track_consistent_audited": len(closed),
        "rejected_audited": len(rejected_rows),
        "insufficient_sample_audited": len(sampled_insufficient),
        "total_audited": len(audit_rows),
        "percent_parse_accuracy": round(percent_accuracy, 6),
        "entity_alignment_accuracy": round(entity_accuracy, 6),
        "time_alignment_accuracy": round(time_accuracy, 6),
        "closed_track_verdict_accuracy": round(closed_accuracy, 6),
        "future_evidence_used_count": future_count,
        "warning_fix": warning_fix,
        "all_known_warnings_fixed": all(item.get("fixed") for item in warning_fix.values()) and set(warning_fix) == WARNING_SAMPLE_IDS,
        "quality_gate_passed": bool(
            percent_accuracy >= 1.0
            and entity_accuracy >= 0.95
            and time_accuracy >= 0.95
            and closed_accuracy >= 0.90
            and future_count == 0
            and all(item.get("fixed") for item in warning_fix.values())
            and set(warning_fix) == WARNING_SAMPLE_IDS
        ),
        "thresholds": {
            "percent_parse_accuracy": 1.0,
            "entity_alignment_accuracy": 0.95,
            "time_alignment_accuracy": 0.95,
            "closed_track_verdict_accuracy": 0.90,
            "future_evidence_used_count": 0,
        },
    }
    with pd.ExcelWriter(audit_xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="audit_rows", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(list(warning_fix.items()), columns=["sample_id", "fix_detail"]).to_excel(writer, sheet_name="warning_fix", index=False)
    base_runner.write_json(quality_json_path, {"summary": summary, "audit_rows": audit_rows})
    return summary


def write_markdown_report(path: Path, report: dict[str, Any], quality: dict[str, Any]) -> None:
    lines = [
        "# Task2 Market Consistency v2.1 Report",
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
        "## v2.1实体修复",
        f"- EC-000034 / EC-000153__v2 通用规则修复：`{quality['all_known_warnings_fixed']}`",
        f"- 质量闸门通过：`{quality['quality_gate_passed']}`",
        "",
        "## 输入SHA256",
        f"- clean_evidence before/after：`{report['input_sha256_before_after']['clean_evidence_before']}` / `{report['input_sha256_before_after']['clean_evidence_after']}`",
        f"- market_cache before/after：`{report['input_sha256_before_after']['market_cache_before']}` / `{report['input_sha256_before_after']['market_cache_after']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--market-cache", type=Path, default=DEFAULT_MARKET_CACHE)
    parser.add_argument("--locked-ids", type=Path, default=DEFAULT_LOCKED_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = base_runner.resolve(args.input)
    market_cache = base_runner.resolve(args.market_cache)
    locked_ids_path = base_runner.resolve(args.locked_ids)
    output_dir = base_runner.resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_sha_before = base_runner.sha256(input_path)
    market_sha_before = base_runner.sha256(market_cache)
    if clean_sha_before != EXPECTED_CLEAN_SHA256 and args.limit == 0:
        raise ValueError(f"clean_evidence_1500 SHA mismatch: {clean_sha_before}")
    clean_rows = base_runner.read_jsonl(input_path)
    selected_rows = clean_rows[: args.limit] if args.limit and args.limit > 0 else clean_rows
    sample_audit = base_runner.validate_sample_ids(selected_rows)
    if sample_audit["missing_sample_id_count"] or sample_audit["duplicate_sample_ids"]:
        raise ValueError(f"input sample_id audit failed: {sample_audit}")

    locked_ids = base_runner.load_locked_ids(locked_ids_path)
    market_index = build_market_index(load_market_cache(market_cache))
    cfg = MarketRuleConfig(version=VERSION)

    market_rows: list[dict[str, Any]] = []
    explicit_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        market_row, accepted, rejected = aggregate_market_consistency(row, market_index, cfg)
        market_rows.append(market_row)
        explicit_rows.extend(accepted)
        rejected_rows.extend(rejected)

    clean_sha_after = base_runner.sha256(input_path)
    market_sha_after = base_runner.sha256(market_cache)
    report = base_runner.summarize(
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
    report["task_scope"] = "market_consistency_v21_clean_evidence_1500_only"
    report["source_version"] = VERSION
    clean_by_id = {str(row.get("sample_id")): row for row in selected_rows}

    market_path = output_dir / "market_consistency_1500.jsonl"
    explicit_path = output_dir / "explicit_market_claims.jsonl"
    rejected_path = output_dir / "rejected_market_claims.jsonl"
    report_json = output_dir / "market_consistency_report.json"
    report_md = output_dir / "market_consistency_report.md"
    audit_xlsx = output_dir / "market_claim_audit.xlsx"
    quality_xlsx = output_dir / "quality_audit.xlsx"
    quality_json = output_dir / "quality_gate.json"
    manifest_path = output_dir / "manifest.json"
    run_state_path = output_dir / "run_state.json"

    base_runner.write_jsonl(market_path, market_rows)
    base_runner.write_jsonl(explicit_path, explicit_rows)
    base_runner.write_jsonl(rejected_path, rejected_rows)
    base_runner.write_json(report_json, report)
    quality = run_quality_gate(market_rows, rejected_rows, quality_xlsx, quality_json, SEED)
    write_markdown_report(report_md, report, quality)
    with pd.ExcelWriter(audit_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(base_runner.audit_rows_for_xlsx(market_rows, clean_by_id)).to_excel(writer, sheet_name="market_claim_audit", index=False)
        pd.DataFrame([report]).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(Counter(row.get("entity_reject_reason") for row in rejected_rows).items(), columns=["reason", "count"]).to_excel(
            writer,
            sheet_name="rejected_reasons",
            index=False,
        )

    run_state = {
        "status": "completed" if all(report["quality_checks"].values()) and quality["quality_gate_passed"] else "completed_with_validation_warnings",
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
        "quality_gate": quality,
    }
    base_runner.write_json(run_state_path, run_state)
    manifest = {
        "status": run_state["status"],
        "script": "scripts/run_task2_market_consistency_v21.py",
        "source_version": VERSION,
        "input_files": {
            "clean_evidence_1500": base_runner.display_path(input_path),
            "market_cache": base_runner.display_path(market_cache),
            "locked_ids": base_runner.display_path(locked_ids_path),
        },
        "output_files": {
            "market_consistency_1500": base_runner.display_path(market_path),
            "explicit_market_claims": base_runner.display_path(explicit_path),
            "rejected_market_claims": base_runner.display_path(rejected_path),
            "market_consistency_report_json": base_runner.display_path(report_json),
            "market_consistency_report_md": base_runner.display_path(report_md),
            "market_claim_audit_xlsx": base_runner.display_path(audit_xlsx),
            "quality_audit_xlsx": base_runner.display_path(quality_xlsx),
            "quality_gate": base_runner.display_path(quality_json),
            "manifest": base_runner.display_path(manifest_path),
            "run_state": base_runner.display_path(run_state_path),
            "sha256s": base_runner.display_path(output_dir / "SHA256SUMS.txt"),
        },
        "summary": report,
        "quality_gate": quality,
    }
    base_runner.write_json(manifest_path, manifest)
    base_runner.write_sha256s(output_dir, [])

    print(
        json.dumps(
            {
                "status": run_state["status"],
                "output_dir": base_runner.display_path(output_dir),
                "training_performed": False,
                "tuning_performed": False,
                "locked500_gold_read": False,
                "locked500_evaluation_run": False,
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
                "quality_gate": {
                    "quality_gate_passed": quality["quality_gate_passed"],
                    "all_known_warnings_fixed": quality["all_known_warnings_fixed"],
                    "warning_fix": quality["warning_fix"],
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
