#!/usr/bin/env python3
"""Run Step 4.5-C shadow scoring with the 260-row final Cached DEA results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_task2_final_cached_dea_shadow_llm_results import build_shadow_file
from scripts.task2_final_cached_dea_score_shadow_common import (
    FAILED_RECORDS,
    FORMAL_BASE_SCORES,
    FORMAL_FULLTEXT_PARQUET,
    FORMAL_LLM_RESULTS,
    FORMAL_OUTPUTS,
    FUSION_MANIFEST,
    FULLTEXT_EVIDENCE,
    FREEZE_FILES,
    MODEL_NAME,
    PIPELINE_VERSION,
    SHADOW_FULLTEXT_JSONL,
    SHADOW_FULLTEXT_PARQUET,
    SHADOW_FULLTEXT_XLSX,
    SHADOW_LLM_RESULTS,
    SHADOW_OUTPUT_DIR,
    exact_row_equal,
    formal_schema_columns,
    group_level,
    hash_paths,
    llm_adjustment,
    load_base_scores,
    load_fulltext_evidence,
    load_formal_llm_rows,
    load_formal_scores,
    norm,
    pctl,
    resolve,
    sha256,
    write_json,
    write_jsonl,
)
from src.event_credibility import fulltext


def ensure_shadow_llm_file(path: Path, formal_llm_results_path: Path, final_results_path: Path, rebuild: bool = False) -> dict[str, Any]:
    if rebuild or not path.exists():
        return build_shadow_file(formal_llm_results_path, final_results_path, path)
    return {"status": "reused", "output_path": str(path)}


def llm_formula_audit() -> dict[str, Any]:
    return {
        "confidence_semantics": "confidence 表示模型对自身 verdict 的确定程度，不是事件为真的概率",
        "support_direction": "support 以 +12 * confidence 进入融合，正向影响评分",
        "contradict_direction": "contradict 以 -35 * confidence 进入融合，负向影响评分",
        "insufficient_direction": "insufficient 以 -5 * confidence 进入融合，按现有生产公式处理",
        "unclear_handling": "当前生产 LLM 结果合法 verdict 集不包含 unclear；不属于本次融合输入",
        "failed_isolated": "failed 结果在 read_llm_results 中会被隔离，不参与融合",
        "formula_changed": False,
        "llm_adjustment_logic": {
            "support": "+12 * confidence",
            "contradict": "-35 * confidence",
            "insufficient": "-5 * confidence",
            "other": "0",
        },
    }


def validate_shadow_outputs(
    formal_scores: pd.DataFrame,
    shadow_scores: pd.DataFrame,
    formal_schema: list[str],
) -> dict[str, Any]:
    formal_ids = [norm(cid) for cid in formal_scores["cluster_id"].tolist()]
    shadow_ids = [norm(cid) for cid in shadow_scores["cluster_id"].tolist()]
    duplicate_shadow = len(shadow_ids) - len(set(shadow_ids))
    duplicate_formal = len(formal_ids) - len(set(formal_ids))
    missing_ids = sorted(set(formal_ids) - set(shadow_ids))
    extra_ids = sorted(set(shadow_ids) - set(formal_ids))

    score_col = "fulltext_credibility_score"
    level_col = "fulltext_credibility_level"
    status_col = "fulltext_verification_status"
    llm_cols = ["llm_verdict", "llm_confidence", "llm_reason", "llm_error"]
    allowed_statuses = {
        "reversed_or_contradicted",
        "llm_contradicted",
        "official_fulltext_supported",
        "llm_supported",
        "fulltext_supported",
        "needs_manual_review",
        "insufficient_evidence",
    }
    allowed_levels = {
        "高可信",
        "较可信",
        "待核验",
        "疑似低可信",
        "疑似虚假或已反转",
    }
    score_values = shadow_scores[score_col].tolist()
    invalid_score_count = 0
    for value in score_values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            invalid_score_count += 1
        else:
            try:
                numeric = float(value)
            except Exception:
                invalid_score_count += 1
                continue
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
                invalid_score_count += 1

    invalid_level_count = sum(1 for level in shadow_scores[level_col].tolist() if norm(level) not in allowed_levels)
    invalid_status_count = sum(1 for status in shadow_scores[status_col].tolist() if norm(status) not in allowed_statuses)

    row_diffs = []
    exact_same_count = 0
    merged = formal_scores.merge(shadow_scores, on="cluster_id", suffixes=("_formal", "_shadow"))
    if len(merged) != len(formal_scores) or len(merged) != len(shadow_scores):
        raise ValueError("shadow score merge failed to preserve row count")
    compare_columns = [column for column in formal_schema if column != "cluster_id"]
    for _, row in merged.iterrows():
        left = {column: row[f"{column}_formal"] for column in compare_columns}
        right = {column: row[f"{column}_shadow"] for column in compare_columns}
        if exact_row_equal(left, right, compare_columns):
            exact_same_count += 1
        else:
            row_diffs.append(
                {
                    "cluster_id": row["cluster_id"],
                    "score_formal": row[f"{score_col}_formal"],
                    "score_shadow": row[f"{score_col}_shadow"],
                    "level_formal": row[f"{level_col}_formal"],
                    "level_shadow": row[f"{level_col}_shadow"],
                    "status_formal": row[f"{status_col}_formal"],
                    "status_shadow": row[f"{status_col}_shadow"],
                    "llm_verdict_formal": row["llm_verdict_formal"],
                    "llm_verdict_shadow": row["llm_verdict_shadow"],
                    "llm_confidence_formal": row["llm_confidence_formal"],
                    "llm_confidence_shadow": row["llm_confidence_shadow"],
                }
            )

    changed_count = len(formal_scores) - exact_same_count
    score_diffs = [abs(float(item["score_shadow"]) - float(item["score_formal"])) for item in row_diffs if item["score_formal"] is not None and item["score_shadow"] is not None]
    delta_signed = [float(item["score_shadow"]) - float(item["score_formal"]) for item in row_diffs if item["score_formal"] is not None and item["score_shadow"] is not None]
    level_changes = sum(1 for item in row_diffs if norm(item["level_formal"]) != norm(item["level_shadow"]))
    status_changes = sum(1 for item in row_diffs if norm(item["status_formal"]) != norm(item["status_shadow"]))
    score_up = sum(1 for delta in delta_signed if delta > 0)
    score_down = sum(1 for delta in delta_signed if delta < 0)
    exact_level_count = len(formal_scores) - level_changes

    group_counts_formal = Counter(group_level(level) for level in formal_scores[level_col].tolist())
    group_counts_shadow = Counter(group_level(level) for level in shadow_scores[level_col].tolist())
    group_share_delta = {
        key: round((group_counts_shadow.get(key, 0) / max(1, len(shadow_scores)) - group_counts_formal.get(key, 0) / max(1, len(formal_scores))) * 100, 4)
        for key in sorted(set(group_counts_formal) | set(group_counts_shadow))
    }
    level_dist_formal = Counter(norm(level) for level in formal_scores[level_col].tolist())
    level_dist_shadow = Counter(norm(level) for level in shadow_scores[level_col].tolist())
    verification_dist_formal = Counter(norm(status) for status in formal_scores[status_col].tolist())
    verification_dist_shadow = Counter(norm(status) for status in shadow_scores[status_col].tolist())

    alerts = []
    for key in sorted(set(group_counts_formal) | set(group_counts_shadow)):
        delta_pp = abs((group_counts_shadow.get(key, 0) / max(1, len(shadow_scores))) - (group_counts_formal.get(key, 0) / max(1, len(formal_scores)))) * 100
        if delta_pp > 5:
            alerts.append(f"group_share_shift>{key}:{delta_pp:.2f}pp")
    formal_mean = float(formal_scores[score_col].mean())
    shadow_mean = float(shadow_scores[score_col].mean())
    if abs(shadow_mean - formal_mean) > 5:
        alerts.append(f"mean_shift>{abs(shadow_mean - formal_mean):.2f}")
    low_count_formal = int((formal_scores[level_col].map(group_level) == "低可信").sum())
    low_count_shadow = int((shadow_scores[level_col].map(group_level) == "低可信").sum())
    if abs(low_count_shadow - low_count_formal) / max(1, len(formal_scores)) * 100 > 10:
        alerts.append("low_level_share_shift>10pp")
    if shadow_scores[level_col].map(group_level).tolist().count("低可信") > 0 and any(
        not math.isfinite(float(v)) or float(v) < 0.0 or float(v) > 100.0 for v in shadow_scores[score_col].dropna().tolist()
    ):
        alerts.append("score_out_of_bounds")

    return {
        "input_count": len(formal_scores),
        "shadow_count": len(shadow_scores),
        "duplicate_formal_cluster_id_count": duplicate_formal,
        "duplicate_shadow_cluster_id_count": duplicate_shadow,
        "missing_cluster_id_count": len(missing_ids),
        "extra_cluster_id_count": len(extra_ids),
        "missing_cluster_ids": missing_ids,
        "extra_cluster_ids": extra_ids,
        "exact_same_count": exact_same_count,
        "changed_count": changed_count,
        "score_changed_count": len(row_diffs),
        "level_changed_count": level_changes,
        "status_changed_count": status_changes,
        "score_mean_abs_diff": round(sum(score_diffs) / max(1, len(score_diffs)), 6),
        "score_median_abs_diff": pctl(score_diffs, 0.5),
        "score_p95_abs_diff": pctl(score_diffs, 0.95),
        "score_max_abs_diff": round(max(score_diffs), 6) if score_diffs else 0.0,
        "score_increase_count": score_up,
        "score_decrease_count": score_down,
        "formal_level_distribution": dict(level_dist_formal),
        "shadow_level_distribution": dict(level_dist_shadow),
        "formal_status_distribution": dict(verification_dist_formal),
        "shadow_status_distribution": dict(verification_dist_shadow),
        "formal_group_distribution": dict(group_counts_formal),
        "shadow_group_distribution": dict(group_counts_shadow),
        "group_share_delta_pp": group_share_delta,
        "alerts": alerts,
        "invalid_score_count": invalid_score_count,
        "invalid_level_count": invalid_level_count,
        "invalid_status_count": invalid_status_count,
        "shadow_columns_match_formal_schema": shadow_scores.columns.tolist() == formal_schema,
        "row_diffs": row_diffs,
    }


def score_shadow(
    formal_scores_path: Path = FORMAL_FULLTEXT_PARQUET,
    base_scores_path: Path = FORMAL_BASE_SCORES,
    evidence_path: Path = FULLTEXT_EVIDENCE,
    shadow_llm_results_path: Path = SHADOW_LLM_RESULTS,
    output_dir: Path = SHADOW_OUTPUT_DIR,
    formal_llm_results_path: Path = FORMAL_LLM_RESULTS,
    final_results_path: Path | None = None,
    rebuild_shadow_llm: bool = False,
) -> dict[str, Any]:
    output_dir = resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shadow_llm_results_path = resolve(shadow_llm_results_path)
    shadow_jsonl_path = output_dir / "event_credibility_scores_fulltext_shadow.jsonl"
    shadow_parquet_path = output_dir / "event_credibility_scores_fulltext_shadow.parquet"
    shadow_xlsx_path = output_dir / "event_credibility_scores_fulltext_shadow.xlsx"
    failed_records_path = output_dir / "failed_records.jsonl"
    fusion_manifest_path = output_dir / "fusion_manifest.json"
    formal_hash_before = hash_paths(FORMAL_OUTPUTS)
    freeze_hash_before = hash_paths(FREEZE_FILES)

    llm_build_summary = ensure_shadow_llm_file(
        shadow_llm_results_path,
        resolve(formal_llm_results_path),
        resolve(final_results_path) if final_results_path is not None else resolve("outputs/event_credibility/final_cached_dea_v1_260/final_results.jsonl"),
        rebuild=rebuild_shadow_llm,
    )
    shadow_llm_rows = load_formal_llm_rows(shadow_llm_results_path)
    llm_rows_summary = {
        "shadow_llm_rows": len(shadow_llm_rows),
        "shadow_llm_path": str(resolve(shadow_llm_results_path)),
    }

    base_scores = load_base_scores(resolve(base_scores_path))
    evidence = load_fulltext_evidence(resolve(evidence_path))
    llm_results = fulltext.read_llm_results(shadow_llm_results_path)
    shadow_scores = fulltext.fuse_fulltext_scores(base_scores, evidence, llm_results)
    formal_scores = load_formal_scores(resolve(formal_scores_path))
    formal_schema = formal_scores.columns.tolist()
    missing_schema = [col for col in formal_schema if col not in shadow_scores.columns]
    if missing_schema:
        raise ValueError(f"shadow score output missing columns: {missing_schema}")
    shadow_scores = shadow_scores.reindex(columns=formal_schema)

    validation = validate_shadow_outputs(formal_scores, shadow_scores, formal_schema)
    if validation["missing_cluster_id_count"] or validation["extra_cluster_id_count"]:
        raise ValueError("shadow score cluster_id set does not match formal output")
    if validation["invalid_score_count"] or validation["invalid_level_count"] or validation["invalid_status_count"]:
        raise ValueError("shadow score validation failed")
    if validation["alerts"]:
        # Alerts are reported, not rejected.
        pass

    shadow_scores.to_parquet(shadow_parquet_path, index=False)
    shadow_scores.to_excel(shadow_xlsx_path, index=False)
    write_jsonl(shadow_jsonl_path, shadow_scores.to_dict("records"))
    write_jsonl(failed_records_path, [])

    formal_hash_after = hash_paths(FORMAL_OUTPUTS)
    freeze_hash_after = hash_paths(FREEZE_FILES)
    if formal_hash_before != formal_hash_after:
        raise RuntimeError("formal output hashes changed during shadow scoring")
    if freeze_hash_before != freeze_hash_after:
        raise RuntimeError("freeze file hashes changed during shadow scoring")

    shadow_hashes = hash_paths(
        [
            shadow_jsonl_path,
            shadow_parquet_path,
            shadow_xlsx_path,
            shadow_llm_results_path,
            failed_records_path,
        ]
    )
    manifest = {
        "step": "4.5-C",
        "status": "completed",
        "pipeline_version": PIPELINE_VERSION,
        "model_name": MODEL_NAME,
        "formal_score_schema_columns": formal_schema,
        "shadow_score_schema_columns": shadow_scores.columns.tolist(),
        "formal_outputs_before": formal_hash_before,
        "formal_outputs_after": formal_hash_after,
        "freeze_files_before": freeze_hash_before,
        "freeze_files_after": freeze_hash_after,
        "formal_outputs_unchanged": formal_hash_before == formal_hash_after,
        "freeze_files_unchanged": freeze_hash_before == freeze_hash_after,
        "shadow_llm_build_summary": llm_build_summary,
        "shadow_llm_rows": len(shadow_llm_rows),
        "shadow_score_summary": {
            "rows": len(shadow_scores),
            "score_mean": float(shadow_scores["fulltext_credibility_score"].mean()),
            "score_std": float(shadow_scores["fulltext_credibility_score"].std(ddof=0)),
            "score_min": float(shadow_scores["fulltext_credibility_score"].min()),
            "score_max": float(shadow_scores["fulltext_credibility_score"].max()),
            "level_distribution": dict(Counter(norm(level) for level in shadow_scores["fulltext_credibility_level"].tolist())),
            "group_distribution": dict(Counter(group_level(level) for level in shadow_scores["fulltext_credibility_level"].tolist())),
            "verification_status_distribution": dict(Counter(norm(status) for status in shadow_scores["fulltext_verification_status"].tolist())),
        },
        "validation": {k: v for k, v in validation.items() if k != "row_diffs"},
        "shadow_hashes": shadow_hashes,
        "alert_count": len(validation["alerts"]),
        "alerts": validation["alerts"],
        "failed_records_path": str(failed_records_path),
        "output_dir": str(output_dir),
        "shadow_llm_results_path": str(shadow_llm_results_path),
        "formal_scores_path": str(resolve(formal_scores_path)),
        "base_scores_path": str(resolve(base_scores_path)),
        "evidence_path": str(resolve(evidence_path)),
        "llm_results_summary": llm_rows_summary,
        "formula_audit": llm_formula_audit(),
        "shadow_score_changed_rows": len(validation["row_diffs"]),
        "shadow_score_row_diffs": validation["row_diffs"],
    }
    write_json(fusion_manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-scores", type=Path, default=FORMAL_FULLTEXT_PARQUET)
    parser.add_argument("--base-scores", type=Path, default=FORMAL_BASE_SCORES)
    parser.add_argument("--evidence", type=Path, default=FULLTEXT_EVIDENCE)
    parser.add_argument("--shadow-llm-results", type=Path, default=SHADOW_LLM_RESULTS)
    parser.add_argument("--formal-llm-results", type=Path, default=FORMAL_LLM_RESULTS)
    parser.add_argument("--final-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=SHADOW_OUTPUT_DIR)
    parser.add_argument("--rebuild-shadow-llm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = score_shadow(
        formal_scores_path=resolve(args.formal_scores),
        base_scores_path=resolve(args.base_scores),
        evidence_path=resolve(args.evidence),
        shadow_llm_results_path=resolve(args.shadow_llm_results),
        output_dir=resolve(args.output_dir),
        formal_llm_results_path=resolve(args.formal_llm_results),
        final_results_path=resolve(args.final_results) if args.final_results is not None else None,
        rebuild_shadow_llm=bool(args.rebuild_shadow_llm),
    )
    print(json.dumps({k: manifest[k] for k in ("status", "formal_outputs_unchanged", "freeze_files_unchanged", "shadow_llm_rows", "alert_count")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
