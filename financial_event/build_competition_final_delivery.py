#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "competition_final_delivery"
RUNTIME = OUT / "runtime"
NATIONAL = OUT / "national_submission"
COMPUTE = OUT / "compute_platform_delivery"

FORBIDDEN_NAME_PARTS = [
    "golddata",
    "filled",
    "final_holdout",
    "official1799_final_gold",
    "task3_final_gold",
    "task4_final_gold",
    "csmar",
    "secret",
    "api_key",
    ".env",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_meta(path: Path, include_sha: bool = True) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "path": rel(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }
    if include_sha:
        meta["sha256"] = sha256_file(path)
    return meta


def sha256_tree(paths: list[Path]) -> dict[str, str | None]:
    return {rel(p): sha256_file(p) for p in paths}


def forbidden(path: Path) -> bool:
    s = path.as_posix().lower()
    return any(part in s for part in FORBIDDEN_NAME_PARTS)


def safe_copy(src: Path, dst: Path) -> dict[str, Any]:
    if forbidden(src) or forbidden(dst):
        return {
            "source": rel(src),
            "target": rel(dst),
            "copied": False,
            "reason": "excluded_by_forbidden_name_policy",
        }
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": rel(src),
        "target": rel(dst),
        "copied": True,
        "sha256": sha256_file(dst),
        "size_bytes": dst.stat().st_size,
    }


def rewrite_packaged_string(value: str) -> str:
    root_prefix = str(ROOT.resolve()) + "/"
    if value.startswith(root_prefix):
        local = value[len(root_prefix) :]
        if forbidden(Path(local)):
            return "restricted_data_reference_omitted"
        return local
    if forbidden(Path(value)):
        return "restricted_data_reference_omitted"
    return value


def sanitize_json_value(obj: Any, drop_keys: set = None) -> Any:
    drop_keys = drop_keys or set()
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in drop_keys:
                continue
            out[key] = sanitize_json_value(value, drop_keys)
        return out
    if isinstance(obj, list):
        return [sanitize_json_value(item, drop_keys) for item in obj]
    if isinstance(obj, str):
        return rewrite_packaged_string(obj)
    return obj


def write_sanitized_json_copy(src: Path, dst: Path, mode: str = "generic") -> dict[str, Any]:
    obj = read_json(src)
    if mode == "task1_taxonomy":
        keep_keys = [
            "taxonomy_version",
            "generated_at",
            "official64_count",
            "official64_preserved",
            "leaf_count",
            "l1_count",
            "l2_count",
            "event_layers",
            "l1_nodes",
            "l2_nodes",
            "leaf_nodes",
            "backward_compatibility_map",
            "source_label_forward_map",
        ]
        obj = {key: obj[key] for key in keep_keys if key in obj}
        obj["delivery_sanitization"] = {
            "sample_titles_removed": True,
            "sample_content_excerpts_removed": True,
            "positive_and_hard_negative_examples_removed": True,
            "source_artifact_paths_removed": True,
        }
        drop_keys = {"positive_examples", "hard_negative_examples", "sample_titles", "sample_content_excerpts"}
    elif mode == "structured_event_contract":
        if isinstance(obj, dict) and "benchmark_source" in obj:
            obj["benchmark_source"] = {
                "path": "restricted_benchmark_not_packaged",
                "sha256_redacted": True,
                "packaged": False,
            }
        drop_keys = set()
    else:
        drop_keys = set()
    sanitized = sanitize_json_value(obj, drop_keys)
    write_json(dst, sanitized)
    return {
        "source": rel(src),
        "target": rel(dst),
        "copied": True,
        "sanitized": True,
        "sha256": sha256_file(dst),
        "size_bytes": dst.stat().st_size,
    }


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 120) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.perf_counter() - t0, 6),
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": 124,
            "elapsed_seconds": round(time.perf_counter() - t0, 6),
            "stdout": (e.stdout or "")[-8000:] if isinstance(e.stdout, str) else "",
            "stderr": "timeout",
        }


def ensure_clean_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for child in OUT.iterdir():
        if child.name in {
            "runtime",
            "national_submission",
            "compute_platform_delivery",
            "national_submission.zip",
            "compute_platform_delivery.tar.gz",
        }:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def build_inventories() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task1_hi = ROOT / "outputs/task1_step5_8hi_tag_optimization"
    task1_hc = ROOT / "outputs/task1_step5_8hc_hierarchical_event"
    task1_8e = ROOT / "outputs/task1_step5_8e_final_optimization"
    task1_report = ROOT / "reports/task1_step5_8hi_final_report.json"
    task1_hc_report = ROOT / "reports/task1_step5_8hc_hierarchical_event_report.json"
    task1_8e_manifest = task1_8e / "checkpoint_manifest.json"
    task1_hc_manifest = task1_hc / "checkpoint_manifest.json"
    task1_hi_manifest = task1_hi / "checkpoint_manifest.json"
    t1_hi = read_json(task1_report)
    t1_hc = read_json(task1_hc_report)
    t1_8e = read_json(task1_8e_manifest)
    t1_hcm = read_json(task1_hc_manifest)

    task1_inventory = {
        "task": "Task1",
        "selected_version": {
            "core": "task1_step5_8e_final_optimization",
            "hierarchical_core": "task1_step5_8hc_hierarchical_event",
            "best_safe_event_recognition": "task1_step5_8hi_tag_optimization",
        },
        "model_path": {
            "core_event_name": rel(task1_8e / "frozen_best_candidate/event_name_model.pkl"),
            "core_event_emotion": rel(task1_8e / "frozen_best_candidate/event_emotion_model.pkl"),
            "hierarchical": rel(task1_hc / "models/best_hierarchical_model.pkl"),
            "safe_event_recognition_prediction": rel(task1_hi / "official_labeled_test_predictions.jsonl"),
        },
        "inference_path": [
            "scripts/run_task1_step5_8e_final_optimization.py",
            "scripts/run_task1_step5_8hi_tag_optimization.py",
        ],
        "input_schema": {
            "required": ["article_id", "title", "content", "publish_time"],
            "optional": ["source", "target_code", "target_name", "industry_code", "industry_name"],
        },
        "output_schema": {
            "fields": [
                "event_mention_id",
                "target_type",
                "target_code",
                "target_name",
                "event_name",
                "event_emotion",
                "event_time",
                "evidence",
                "duplication_id",
            ]
        },
        "metrics": {
            "core_delivery_rows": t1_8e["outputs"]["row_count"],
            "core_company_rows": t1_8e["outputs"]["company_rows"],
            "core_industry_rows": t1_8e["outputs"]["industry_rows"],
            "core_illegal_event_name": t1_8e["outputs"]["illegal_event_name"],
            "safe_event_name_micro_f1": t1_hi["event_name"]["micro_f1"],
            "safe_event_name_fixed64_macro_f1": t1_hi["optimized_metrics"]["event_name_fixed64_macro_f1"],
            "safe_target_event_joint_micro_f1": t1_hi["target_event_joint"]["micro_f1"],
            "safe_event_emotion_macro_f1": t1_hi["event_emotion"]["macro_f1"],
            "safe_duplication_pairwise_f1": t1_hi["duplication"]["pairwise_f1"],
            "safe_duplication_bcubed_f1": t1_hi["duplication"]["bcubed_f1"],
            "safe_corrected_full_quality_pass": t1_hi["corrected_exact"]["full_quality_pass"],
            "hierarchical_holdout_official64_macro_f1": t1_hc["selected"]["best_holdout"]["official64_macro_f1"],
            "hierarchical_holdout_hierarchical_f1": t1_hc["selected"]["best_holdout"]["hierarchical_f1"],
        },
        "metric_source": [
            rel(ROOT / "reports/TASK1_FINAL_TECHNICAL_REPORT.md"),
            rel(task1_report),
            rel(task1_hc_report),
            rel(task1_8e_manifest),
        ],
        "sha256": sha256_tree(
            [
                task1_8e / "frozen_best_candidate/event_name_model.pkl",
                task1_8e / "frozen_best_candidate/event_emotion_model.pkl",
                task1_8e / "frozen_best_candidate/frozen_model_manifest.json",
                task1_hc / "models/best_hierarchical_model.pkl",
                task1_hc / "models/best_hierarchical_model_manifest.json",
                task1_hi / "official_labeled_test_predictions.jsonl",
                task1_hi / "frozen_model_manifest.json",
                task1_report,
            ]
        ),
        "dependencies": ["python>=3.8", "pandas", "scikit-learn", "joblib"],
        "runnable": {
            "demo_adapter": True,
            "full_historical_inference": "requires original Task1 runtime dependencies and selected frozen models",
        },
        "controls": {
            "model_frozen": True,
            "thresholds_frozen": True,
            "taxonomy_frozen": True,
            "no_retraining_this_delivery": True,
            "selected_hierarchical_scheme": t1_hcm.get("selected_scheme"),
        },
    }

    task2_prod = ROOT / "outputs/event_credibility/final_cached_dea_v1_260"
    task2_v2 = ROOT / "outputs/event_credibility/task2_sixdim_ai_fusion_v2"
    task2_promo = ROOT / "reports/task2_final_cached_dea_v1_promotion_manifest.json"
    task2_report = ROOT / "reports/task2_final_cached_dea_v1_260_report.json"
    task2_shadow = ROOT / "reports/task2_final_cached_dea_score_shadow_report.json"
    task2_latest_report = ROOT / "reports/task2_sixdim_ai_fusion_v2_report.md"
    t2_promo = read_json(task2_promo)
    t2_run = read_json(task2_prod / "run_manifest.json")
    t2_v2_config = read_json(task2_v2 / "frozen_inference_config.json")
    t2_v2_freeze = read_json(task2_v2 / "freeze_manifest.json")
    t2_v2_model = read_json(task2_v2 / "best_model_manifest.json")

    task2_inventory = {
        "task": "Task2",
        "selected_version": "final_cached_dea_v1_260",
        "latest_freeze_report_checked": {
            "version": "task2_sixdim_ai_fusion_v2",
            "report": rel(task2_latest_report),
            "frozen_at": t2_v2_freeze.get("frozen_at"),
            "selected_model": t2_v2_freeze.get("selected_model"),
            "advanced_to_main_system": t2_v2_config.get("advanced_to_main_system"),
            "development_gate_passed": t2_v2_config.get("development_gate_passed"),
            "production_decision": "not_selected_for_production_because_not_advanced_to_main_system",
        },
        "model_path": {
            "production_llm_model_name": "DianJin-R1-32B",
            "production_frozen_outputs": [
                "outputs/event_credibility/llm_review_results.jsonl",
                "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
                "outputs/event_credibility/event_credibility_scores_fulltext.parquet",
                "outputs/event_credibility/event_credibility_scores_fulltext.xlsx",
            ],
            "latest_candidate_model": rel(task2_v2 / "best_model.pkl"),
        },
        "inference_path": ["scripts/run_task2_final_cached_dea_v1.py"],
        "input_schema": {
            "required": ["cluster_id", "event_mention_id", "claim", "evidence_items"],
            "optional": ["source_count", "source_names", "article_ids"],
        },
        "output_schema": {
            "fields": [
                "cluster_id",
                "credibility_score",
                "credibility_level",
                "final_verdict",
                "confidence",
                "evidence_chain",
                "explanation",
            ]
        },
        "metrics": {
            "independent50_cached_dea_accuracy": 0.74,
            "independent50_cached_dea_fixed_macro_f1": 0.6327,
            "independent50_label_distribution": {
                "support": 25,
                "insufficient": 20,
                "unclear": 4,
                "contradict": 1,
            },
            "production_260_completed_count": t2_run["summary"]["completed_count"],
            "production_260_dea_trigger_count": t2_run["summary"]["dea_trigger_count"],
            "production_260_dea_trigger_ratio": t2_run["summary"]["dea_trigger_ratio"],
            "production_260_final_verdict_distribution": t2_run["summary"]["final_verdict_distribution"],
            "production_260_round1_final_verdict_change_count": t2_run["summary"]["round1_final_verdict_change_count"],
            "production_260_new_evidence_total_count": t2_run["summary"]["new_evidence_total_count"],
            "production_260_total_avg_seconds": t2_run["summary"]["latency"]["total_avg"],
            "production_260_total_p95_seconds": t2_run["summary"]["latency"]["total_p95"],
            "production_260_timeout_5xx_parse_oom": {
                "timeout": t2_run["summary"]["errors"]["timeout_count"],
                "http_5xx": t2_run["summary"]["errors"]["http_5xx_count"],
                "parse_failure": t2_run["summary"]["errors"]["parse_failure_count"],
                "cuda_oom": t2_run["summary"]["errors"]["cuda_oom_count"],
            },
            "formal_score_rows": t2_promo["row_counts"]["formal_scores_jsonl"],
            "formal_score_changed_count": t2_promo["score_change_summary"]["all_2061"]["score_changed_count"],
            "formal_level_changed_count": t2_promo["score_change_summary"]["all_2061"]["level_changed_count"],
        },
        "metric_source": [
            rel(ROOT / "reports/TASK2_FINAL_TECHNICAL_REPORT.md"),
            rel(task2_report),
            rel(task2_shadow),
            rel(task2_promo),
            rel(task2_latest_report),
        ],
        "sha256": {
            **{k: v for k, v in t2_promo["new_formal_sha256"].items()},
            rel(task2_prod / "final_results.jsonl"): sha256_file(task2_prod / "final_results.jsonl"),
            rel(task2_v2 / "best_model.pkl"): t2_v2_model.get("artifact_sha256"),
            rel(task2_v2 / "freeze_manifest.json"): sha256_file(task2_v2 / "freeze_manifest.json"),
        },
        "dependencies": ["python>=3.8", "pandas", "pyarrow", "sqlite3/FTS5", "OpenAI-compatible chat API for DianJin-R1-32B"],
        "runnable": {
            "demo_adapter": True,
            "full_historical_inference": "requires DianJin-R1-32B service and SQLite retrieval index mount",
        },
        "controls": {
            "promotion_committed": t2_promo["promotion_committed"],
            "model_calls_added_after_promotion": t2_promo["model_calls_added"],
            "production_training_performed_this_delivery": False,
            "production_tuning_performed_this_delivery": False,
        },
    }

    task3_legacy_metrics = ROOT / "outputs/task3_resolver_v3_official1799_multiroute_evaluation/legacy_official_metrics.json"
    task3_multiroute_metrics = ROOT / "outputs/task3_resolver_v3_official1799_multiroute_evaluation/multiroute_official_metrics.json"
    task3_final_report = ROOT / "reports/task3_kg_v3_final_evaluation.json"
    task3_release_manifest = ROOT / "outputs/task3_kg_v3_release/release_manifest.json"
    t3_legacy = read_json(task3_legacy_metrics)
    t3_release = read_json(task3_release_manifest)
    task3_inventory = {
        "task": "Task3",
        "selected_version": "legacy_official_pipeline",
        "selected_production_version": "legacy_official_pipeline",
        "model_path": {
            "resolver_v2_model": "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_model.joblib",
            "release_graph_dir": "outputs/task3_kg_v3_release",
            "official_legacy_prediction_dir": "outputs/task3_kg_v4_official1799_inference_v1",
        },
        "inference_path": [
            "scripts/build_task3_kg_v4_official1799_inference_v1.py",
            "scripts/task3_kg_v3_release.py",
        ],
        "input_schema": {
            "required": ["event_mention_id", "target_key", "event_name", "event_time", "evidence"],
            "optional": ["credibility_score", "source_name", "article_id"],
        },
        "output_schema": {
            "fields": [
                "canonical_event_id",
                "event_thread_id",
                "current_state_type",
                "resolver_decision",
                "graph_layer",
                "trust_score",
            ]
        },
        "metrics": {
            "official1799_candidate_pair_n": t3_legacy["official_candidate_pairwise"]["n"],
            "official1799_candidate_pair_precision": t3_legacy["official_candidate_pairwise"]["same_event_precision"],
            "official1799_candidate_pair_recall": t3_legacy["official_candidate_pairwise"]["same_event_recall"],
            "official1799_candidate_pair_f1": t3_legacy["official_candidate_pairwise"]["same_event_f1"],
            "official1799_candidate_pair_macro_f1_sklearn": t3_legacy["official_candidate_pairwise"]["sklearn_macro_f1"],
            "official1799_b3_n": t3_legacy["b3"]["n"],
            "official1799_b3_precision": t3_legacy["b3"]["precision"],
            "official1799_b3_recall": t3_legacy["b3"]["recall"],
            "official1799_b3_f1": t3_legacy["b3"]["f1"],
            "official1799_full_cluster_pair_n": t3_legacy["same_denominator_full_cluster_pairwise"]["n"],
            "official1799_full_cluster_pair_f1": t3_legacy["same_denominator_full_cluster_pairwise"]["same_event_f1"],
            "release_event_mention_coverage": "4973/4973 mapped",
            "release_selected_canonical_event_version": t3_release["selected_canonical_event_version"],
        },
        "metric_source": [rel(task3_legacy_metrics), rel(task3_final_report), rel(task3_release_manifest)],
        "sha256": {
            **{f"outputs/task3_kg_v3_release/{k}": v for k, v in t3_release["hashes"].items()},
            "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_model.joblib": sha256_file(
                ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_model.joblib"
            ),
            rel(task3_legacy_metrics): sha256_file(task3_legacy_metrics),
            rel(task3_multiroute_metrics): sha256_file(task3_multiroute_metrics),
        },
        "dependencies": ["python>=3.8", "pandas", "joblib", "networkx-like JSONL graph consumers"],
        "runnable": {
            "demo_adapter": True,
            "full_historical_inference": "requires Task1 frozen models and original official-test inference inputs; official test data is not packaged",
        },
        "controls": {
            "hybrid_v3_1_status": "rejected_by_single_use_final_holdout",
            "multiroute_status": "research_ablation_only",
            "this_delivery_gold_read": False,
            "this_delivery_final_holdout_read": False,
        },
    }

    task4_closure = ROOT / "outputs/task4_final_contract_separated_closure"
    task4_manifest = read_json(task4_closure / "final_model_manifest.json")
    task4_formal = read_json(task4_closure / "official1799_formal_metrics.json")
    task4_legacy_ind = read_json(ROOT / "outputs/task34_official1799_final_evaluation_v1/task4_independent_metrics.json")
    task4_legacy_e2e = read_json(ROOT / "outputs/task34_official1799_final_evaluation_v1/task4_end_to_end_metrics.json")
    task4_inventory = {
        "task": "Task4",
        "selected_version": "modular_final_combination",
        "model_path": {
            "stable_high_risk_head": "legacy_official_pipeline/frozen_task4_predictions",
            "applicability_gate": "legacy_official_pipeline/auto_score_eligibility_gate",
            "alert_head": "legacy_official_pipeline/alert_binary",
            "direction": "legacy_official_pipeline/task4_independent_metrics.direction",
            "ordinal_severity": rel(task4_closure / "models/task4_final_contract_separated_closure_bundle.joblib"),
            "trace_explanation": rel(ROOT / "outputs/task4_trace_refine_v3_2_student/models/task4_trace_refine_v3_2_student_bundle.joblib"),
        },
        "inference_path": [
            "outputs/task4_final_contract_separated_closure/inference_pipeline.py",
            "outputs/task4_trace_refine_v3_2_student/inference_pipeline.py",
        ],
        "input_schema": {
            "required": ["canonical_event_id", "event_thread_id", "event_mention_id", "target_key", "event_name", "evidence"],
            "optional": ["credibility_score", "direction_hint", "source_count"],
        },
        "output_schema": {
            "fields": [
                "direction",
                "severity",
                "official_high_risk",
                "applicability_gate",
                "alert",
                "trace",
                "rag_review_basis",
            ]
        },
        "metrics": {
            "important_note": "Metrics below are module-level results from existing one-time official1799 evaluations of different frozen modules, not a newly rerun unified end-to-end evaluation of the modular combination.",
            "stable_legacy_warning_module": {
                "dataset": "official1799 existing one-time evaluation",
                "high_severity_precision": task4_legacy_ind["severity"]["high_severity_precision"],
                "high_severity_recall": task4_legacy_ind["severity"]["high_severity_recall"],
                "high_severity_f1": task4_legacy_ind["severity"]["high_severity_f1"],
                "gate_accuracy": task4_legacy_e2e["auto_score_eligibility_gate"]["accuracy"],
                "gate_f1": task4_legacy_e2e["auto_score_eligibility_gate"]["f1"],
                "alert_recall": task4_legacy_e2e["alert_binary"]["recall"],
                "alert_f1": task4_legacy_e2e["alert_binary"]["f1"],
                "automatic_coverage": task4_legacy_e2e["coverage_rate"],
                "direction_accuracy": task4_legacy_ind["direction"]["accuracy"],
                "direction_macro_f1": task4_legacy_ind["direction"]["macro_f1"],
            },
            "gold_only_ordinal_module": {
                "dataset": "official1799 severity eligible rows from existing closure formal metrics",
                "n": task4_formal["official_risk"]["n"],
                "qwk": task4_formal["official_risk"]["qwk"],
                "mae": task4_formal["official_risk"]["mae"],
                "spearman": task4_formal["official_risk"]["spearman"],
            },
            "closure_oof_reference_not_display_head": {
                "dataset": "legacy human risk gold OOF 657 rows",
                "qwk": read_json(task4_closure / "official_risk_metrics.json")["selected_oof"]["qwk"],
                "mae": read_json(task4_closure / "official_risk_metrics.json")["selected_oof"]["mae"],
                "highrisk_oof_f1": read_json(task4_closure / "highrisk_metrics.json")["selected_oof"]["f1"],
            },
        },
        "metric_source": [
            "outputs/task34_official1799_final_evaluation_v1/task4_independent_metrics.json",
            "outputs/task34_official1799_final_evaluation_v1/task4_end_to_end_metrics.json",
            rel(task4_closure / "official1799_formal_metrics.json"),
            rel(task4_closure / "final_model_manifest.json"),
        ],
        "sha256": {
            rel(task4_closure / "models/task4_final_contract_separated_closure_bundle.joblib"): task4_manifest["model_bundle"][
                "bundle_sha256"
            ],
            task4_manifest["model_bundle"]["v3_2_bundle_path"].replace(str(ROOT) + "/", ""): task4_manifest["model_bundle"][
                "v3_2_bundle_sha256"
            ],
            rel(task4_closure / "final_model_manifest.json"): sha256_file(task4_closure / "final_model_manifest.json"),
            rel(task4_closure / "official1799_formal_metrics.json"): sha256_file(task4_closure / "official1799_formal_metrics.json"),
        },
        "dependencies": ["python>=3.8", "joblib", "scikit-learn", "lightgbm optional for full bundle replay"],
        "runnable": {
            "demo_adapter": True,
            "full_historical_inference": "requires original feature tables and model dependencies; official test data is not packaged",
        },
        "controls": {
            "trace_and_rag_official_risk_influence": "none",
            "v3_4_status": "rejected_ablation",
            "task4_combination_official1799_reexecuted_this_delivery": False,
            "final_holdout_read": task4_manifest["final_holdout_read"],
        },
    }

    inventories = {
        "task1": task1_inventory,
        "task2": task2_inventory,
        "task3": task3_inventory,
        "task4": task4_inventory,
    }

    final_asset_selection = {
        "generated_at": now_iso(),
        "constraints": {
            "training_executed": False,
            "tuning_executed": False,
            "pseudo_label_generated": False,
            "official1799_filled_gold_read": False,
            "final_holdout_read": False,
            "frozen_assets_modified": False,
            "official_test_or_gold_packaged": False,
        },
        "selected_versions": {
            "task1": task1_inventory["selected_version"],
            "task2": task2_inventory["selected_version"],
            "task3": task3_inventory["selected_version"],
            "task4": task4_inventory["selected_version"],
        },
        "task4_heads": [
            "Stable High-Risk Head: legacy official high-risk module",
            "Applicability Gate: legacy official auto-score eligibility gate",
            "Alert Head: legacy official alert binary module",
            "Direction: legacy official direction module",
            "Ordinal Severity: Gold-only ordinal model",
            "TRACE/RAG: explanation and review basis only, no Official Risk influence",
        ],
    }

    metric_source_registry = {
        "generated_at": now_iso(),
        "rule": "Every metric records dataset, denominator and frozen model/version. No metric in this delivery was recomputed against official1799 filled Gold or final_holdout.",
        "task1": {
            "dataset": "retrospective_user_supplied_labeled_test / historical official-compatible 1799 rows plus model-selection holdout for HC",
            "denominator": {"official_labeled_test_rows": 1799, "hc_holdout_rows": 312},
            "model_version": task1_inventory["selected_version"],
            "sources": task1_inventory["metric_source"],
        },
        "task2": {
            "dataset": "independent50 user-confirmed evaluation, production 260 DEA run, formal 2061 score shadow",
            "denominator": {"independent": 50, "production_run": 260, "formal_scores": 2061},
            "model_version": task2_inventory["selected_version"],
            "sources": task2_inventory["metric_source"],
            "latest_frozen_candidate_checked": task2_inventory["latest_freeze_report_checked"],
        },
        "task3": {
            "dataset": "official1799 existing one-time evaluation and Task3 KG v3 release inventory",
            "denominator": {
                "candidate_pairs": 3377,
                "b3_mentions": 1799,
                "same_denominator_full_cluster_pairs": 1617301,
            },
            "model_version": "legacy_official_pipeline / resolver_v2",
            "sources": task3_inventory["metric_source"],
        },
        "task4": {
            "dataset": "official1799 existing one-time module-level results; closure formal metrics for Gold-only ordinal module",
            "denominator": {
                "severity_eligible": 741,
                "alert_evaluated": 1784,
                "gate_full_rows": 1799,
                "closure_oof_reference": 657,
            },
            "model_version": "modular_final_combination",
            "not_unified_e2e_retest": True,
            "sources": task4_inventory["metric_source"],
        },
    }

    excluded_assets = {
        "generated_at": now_iso(),
        "excluded_from_production": [
            {
                "asset": "Task1 Step 5.8-F final_candidate",
                "path": "outputs/task1_step5_8f_final_candidate",
                "reason": "duplication validation insufficient; no formal output overwrite; event_name remains frozen from 5.8-E",
                "status": "not_selected",
            },
            {
                "asset": "Task2 six-dimensional AI fusion v2",
                "path": "outputs/event_credibility/task2_sixdim_ai_fusion_v2",
                "reason": "latest frozen candidate checked, but advanced_to_main_system=false and development_gate_passed=false",
                "status": "frozen_candidate_not_production",
            },
            {
                "asset": "Task2 ISR",
                "path": "outputs/event_credibility/isr_diagnostic_v1",
                "reason": "diagnostic only; corrected 0/6 candidates and added request/token cost",
                "status": "disabled",
            },
            {
                "asset": "Task2 Safety Gate",
                "path": "outputs/event_credibility/dea_safety_gate_v1_2_independent",
                "reason": "independent test zero-trigger / no production gain",
                "status": "disabled",
            },
            {
                "asset": "Task2 Claim Contract Adapter",
                "path": "outputs/event_credibility/claim_contract_adapter_v1_gray",
                "reason": "diagnostic accuracy below Cached DEA and higher latency",
                "status": "disabled",
            },
            {
                "asset": "Task3 Hybrid v3.1",
                "path": "outputs/task3_resolver_v3_1_hybrid_final",
                "reason": "rejected_by_single_use_final_holdout",
                "status": "rejected_by_single_use_final_holdout",
            },
            {
                "asset": "Task3 Multiroute",
                "path": "outputs/task3_resolver_v3_official1799_multiroute_evaluation",
                "reason": "official one-time acceptance status fail; retained as research ablation only",
                "status": "research_ablation_only",
            },
            {
                "asset": "Task4 TRACE refine v3.4",
                "path": "outputs/task4_trace_refine_v3_4_final",
                "reason": "admission failed / selected_ablation_did_not_improve_enough; not in production inference",
                "status": "rejected_ablation",
            },
        ],
        "excluded_from_archives": [
            "official1799 filled Gold files",
            "final_holdout files",
            "official test raw workbook/data",
            "CSMAR original database and cache",
            "keys, credentials, .env and secret-like files",
            "training records and pseudo-label data not required for inference",
        ],
    }

    registry_with_exclusions = dict(metric_source_registry)
    registry_with_exclusions["excluded_assets"] = excluded_assets
    return inventories, final_asset_selection, registry_with_exclusions


def create_runtime() -> None:
    for sub in ["adapters", "configs", "demo_data", "schema", "outputs", "logs"]:
        (RUNTIME / sub).mkdir(parents=True, exist_ok=True)

    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Competition Unified Pipeline Input",
        "type": "object",
        "required": ["article_id", "title", "content", "publish_time"],
        "properties": {
            "article_id": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "publish_time": {"type": "string"},
            "source": {"type": "string"},
            "target_code": {"type": "string"},
            "target_name": {"type": "string"},
            "industry_code": {"type": "string"},
            "industry_name": {"type": "string"},
        },
        "additionalProperties": True,
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Competition Unified Pipeline Output",
        "type": "object",
        "required": ["article_id", "task1", "task2", "task3", "task4", "meta"],
        "properties": {
            "article_id": {"type": "string"},
            "task1": {"type": "object"},
            "task2": {"type": "object"},
            "task3": {"type": "object"},
            "task4": {"type": "object"},
            "meta": {"type": "object"},
        },
    }
    write_json(RUNTIME / "schema/input_schema.json", input_schema)
    write_json(RUNTIME / "schema/output_schema.json", output_schema)

    sample_rows = [
        {
            "article_id": "DEMO_0001",
            "title": "示例新能源公告：基地项目正式立项并启动建设",
            "source": "交易所公告",
            "publish_time": "2026-08-01T09:30:00+08:00",
            "target_code": "600000",
            "target_name": "示例新能源",
            "content": "示例新能源公告称，新能源基地项目已经正式立项并启动一期建设，预计提升长期产能。",
        },
        {
            "article_id": "DEMO_0002",
            "title": "论坛传闻称示例科技可能出现大额裁员",
            "source": "论坛热帖",
            "publish_time": "2026-08-02T10:00:00+08:00",
            "target_code": "000001",
            "target_name": "示例科技",
            "content": "论坛传闻称示例科技或将进行大额裁员，但尚未看到正式公告或权威证据。",
        },
        {
            "article_id": "DEMO_0003",
            "title": "示例装备获得核心客户订单",
            "source": "主流财经媒体",
            "publish_time": "2026-08-03T11:20:00+08:00",
            "target_code": "300000",
            "target_name": "示例装备",
            "content": "采访中公司确认已获得核心客户订单，另一报道补充称订单金额预计超过十亿元。",
        },
    ]
    write_text(
        RUNTIME / "demo_data/sample.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in sample_rows) + "\n",
    )

    runtime_config = {
        "runtime_version": "competition_final_delivery_demo_v1",
        "mode": "schema_adapter_demo",
        "formal_metrics_not_recomputed": True,
        "frozen_model_mount_policy": {
            "demo": "uses deterministic adapters for end-to-end smoke testing",
            "full_replay": "mount frozen models and restricted data separately according to deployment_guide.md",
        },
    }
    write_json(RUNTIME / "configs/runtime_config.json", runtime_config)

    adapter_code = r'''from __future__ import annotations

import hashlib
import re
from typing import Any


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "||".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def pick_event_name(text: str) -> str:
    rules = [
        ("裁员", "人员调整传闻"),
        ("订单", "重大订单获取"),
        ("投产", "重大项目投资建设"),
        ("建设", "重大项目投资建设"),
        ("立项", "重大项目投资建设"),
        ("处罚", "监管处罚"),
        ("诉讼", "法律纠纷"),
        ("技术", "关键技术突破"),
    ]
    for key, value in rules:
        if key in text:
            return value
    return "业务动态"


def pick_emotion(text: str) -> str:
    if any(k in text for k in ["裁员", "处罚", "诉讼", "亏损", "违约", "下滑"]):
        return "负面"
    if any(k in text for k in ["订单", "投产", "提升", "突破", "增长", "中标"]):
        return "正面"
    return "中性"


def task1_extract(row: dict[str, Any]) -> dict[str, Any]:
    text = f"{row.get('title', '')} {row.get('content', '')}"
    target_name = row.get("target_name") or row.get("secu_abbr") or "未识别主体"
    target_code = row.get("target_code") or row.get("trading_code") or ""
    event_name = pick_event_name(text)
    event_emotion = pick_emotion(text)
    evidence = str(row.get("content", ""))[:120]
    return {
        "event_mention_id": stable_id("EM", row.get("article_id"), target_name, event_name),
        "target_type": "company" if target_code else "industry",
        "target_code": target_code,
        "target_name": target_name,
        "event_name": event_name,
        "event_emotion": event_emotion,
        "event_time": row.get("publish_time", ""),
        "evidence": evidence,
        "duplication_id": stable_id("DUP", target_name, event_name, re.sub(r"\s+", "", evidence)[:48]),
    }


def task2_score(row: dict[str, Any], task1: dict[str, Any]) -> dict[str, Any]:
    text = f"{row.get('title', '')} {row.get('content', '')}"
    source = str(row.get("source", ""))
    if "论坛" in source or "传闻" in text:
        score, verdict, level = 42.0, "insufficient", "needs_review"
    elif "公告" in source or "确认" in text or "交易所" in source:
        score, verdict, level = 82.0, "support", "high"
    else:
        score, verdict, level = 68.0, "support", "medium"
    if task1.get("event_emotion") == "负面":
        score = min(score, 55.0)
    return {
        "cluster_id": stable_id("CL", task1.get("duplication_id")),
        "credibility_score": score,
        "credibility_level": level,
        "final_verdict": verdict,
        "confidence": round(score / 100.0, 4),
        "evidence_chain": [
            {
                "source": source or "input",
                "evidence": task1.get("evidence", ""),
                "support_type": verdict,
            }
        ],
        "explanation": "Demo adapter output; formal Task2 metrics come from frozen reports.",
    }


def task3_resolve(row: dict[str, Any], task1: dict[str, Any], task2: dict[str, Any]) -> dict[str, Any]:
    target = task1.get("target_code") or task1.get("target_name")
    event = task1.get("event_name")
    text = f"{row.get('title', '')} {row.get('content', '')}"
    if any(k in text for k in ["立项", "传闻"]):
        state = "announcement"
    elif any(k in text for k in ["投产", "完成", "下线"]):
        state = "completion"
    elif "否认" in text or "澄清" in text:
        state = "clarification"
    else:
        state = "confirmation"
    canonical_event_id = stable_id("CE", target, event)
    return {
        "canonical_event_id": canonical_event_id,
        "event_thread_id": stable_id("TH", target, event),
        "current_state_type": state,
        "event_mention_id": task1.get("event_mention_id"),
        "resolver_decision": "same_event_or_new_by_legacy_adapter",
        "graph_layer": "legacy_official_pipeline_demo_adapter",
        "trust_score": task2.get("confidence"),
    }


def task4_assess(row: dict[str, Any], task1: dict[str, Any], task2: dict[str, Any], task3: dict[str, Any]) -> dict[str, Any]:
    emotion = task1.get("event_emotion")
    score = float(task2.get("credibility_score", 0.0))
    if emotion == "负面":
        direction = "negative"
    elif emotion == "正面":
        direction = "positive"
    else:
        direction = "neutral"
    if score < 50:
        severity = 2
        gate = "review"
        alert = False
    elif direction == "negative":
        severity = 4
        gate = "covered"
        alert = True
    elif "重大" in task1.get("event_name", "") or "订单" in task1.get("event_name", ""):
        severity = 3
        gate = "covered"
        alert = False
    else:
        severity = 2
        gate = "covered"
        alert = False
    high_risk = bool(alert or severity >= 4)
    return {
        "direction": direction,
        "severity": severity,
        "official_high_risk": high_risk,
        "applicability_gate": gate,
        "alert": alert,
        "trace": {
            "target_impact_severity": severity,
            "credibility_basis": task2.get("final_verdict"),
            "event_state": task3.get("current_state_type"),
            "evidence": task1.get("evidence"),
        },
        "rag_review_basis": [
            "TRACE/RAG is explanation and review basis only; it does not modify Official Risk in this delivery."
        ],
    }
'''
    write_text(RUNTIME / "adapters/common.py", adapter_code)
    write_text(RUNTIME / "adapters/__init__.py", "")

    run_pipeline = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from adapters.common import task1_extract, task2_score, task3_resolve, task4_assess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Task1-Task4 competition demo pipeline")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"input_not_found: {input_path}", file=sys.stderr)
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in ("article_id", "title", "content", "publish_time") if k not in row]
            if missing:
                raise ValueError(f"line {line_no} missing required fields: {missing}")
            task1 = task1_extract(row)
            task2 = task2_score(row, task1)
            task3 = task3_resolve(row, task1, task2)
            task4 = task4_assess(row, task1, task2, task3)
            rows.append(
                {
                    "article_id": row["article_id"],
                    "task1": task1,
                    "task2": task2,
                    "task3": task3,
                    "task4": task4,
                    "meta": {
                        "runtime_version": "competition_final_delivery_demo_v1",
                        "formal_metrics_recomputed": False,
                        "demo_adapter": True,
                    },
                }
            )
    with output_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elapsed = time.perf_counter() - t0
    print(json.dumps({"status": "ok", "rows": len(rows), "output": str(output_path), "elapsed_seconds": round(elapsed, 6)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write_text(RUNTIME / "run_pipeline.py", run_pipeline)
    os.chmod(RUNTIME / "run_pipeline.py", 0o755)

    health = r'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 --version >/dev/null
test -f schema/input_schema.json
test -f schema/output_schema.json
test -f demo_data/sample.jsonl
python3 - <<'PY'
import json
for p in ["schema/input_schema.json", "schema/output_schema.json"]:
    json.load(open(p, encoding="utf-8"))
print("healthcheck_ok")
PY
'''
    smoke = r'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs logs
python3 run_pipeline.py --input demo_data/sample.jsonl --output outputs/result.jsonl | tee logs/smoke_test.log
python3 - <<'PY'
import json
from pathlib import Path
path = Path("outputs/result.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
assert rows, "empty output"
for row in rows:
    assert row["task1"]["event_mention_id"]
    assert row["task2"]["credibility_level"]
    assert row["task3"]["canonical_event_id"]
    assert "severity" in row["task4"]
print(json.dumps({"smoke_test": "passed", "rows": len(rows), "output": str(path)}, ensure_ascii=False))
PY
'''
    start = r'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
printf '%s\n' $$ > logs/runtime.pid
echo "runtime_ready: demo pipeline is CLI-based; use python3 run_pipeline.py --input demo_data/sample.jsonl --output outputs/result.jsonl"
'''
    stop = r'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rm -f logs/runtime.pid
echo "runtime_stopped"
'''
    for name, text in [
        ("healthcheck.sh", health),
        ("smoke_test.sh", smoke),
        ("start.sh", start),
        ("stop.sh", stop),
    ]:
        write_text(RUNTIME / name, text)
        os.chmod(RUNTIME / name, 0o755)

    write_text(
        RUNTIME / "README.md",
        """# Competition Final Delivery Runtime

Run:

```bash
python3 run_pipeline.py --input demo_data/sample.jsonl --output outputs/result.jsonl
```

This runtime is a unified schema demo adapter for Task1-Task4 integration. It does not retrain, tune thresholds, read official1799 filled Gold, read final_holdout, or recompute official metrics.
""",
    )


def create_docs_and_packages(inventories: dict[str, Any], selection: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    excluded = registry["excluded_assets"]
    for name, obj in [
        ("task1_final_inventory.json", inventories["task1"]),
        ("task2_final_inventory.json", inventories["task2"]),
        ("task3_final_inventory.json", inventories["task3"]),
        ("task4_final_inventory.json", inventories["task4"]),
        ("final_asset_selection.json", selection),
        ("metric_source_registry.json", {k: v for k, v in registry.items() if k != "excluded_assets"}),
        ("excluded_assets.json", excluded),
    ]:
        write_json(OUT / name, obj)

    summary_md = f"""# Task1-Task4 Results Summary

All metrics are copied from frozen reports or manifests. This delivery did not train, tune, generate pseudo labels, read official1799 filled Gold, or read final_holdout.

| Task | Final production version | Dataset / denominator | Official metrics and source |
|---|---|---|---|
| Task1 | Core: `task1_step5_8e_final_optimization`; Safe recognition: `task1_step5_8hi_tag_optimization` | 1799 retrospective user-supplied labeled test; HC holdout 312 rows | Event-name micro-F1 0.767686, fixed-64 macro-F1 0.669357, target+event joint micro-F1 0.746654, corrected full-quality pass 0.527725. Source: `reports/task1_step5_8hi_final_report.json`. |
| Task2 | `final_cached_dea_v1_260` | independent50, production 260 run, formal 2061 score shadow | independent50 Cached DEA accuracy 74.00%, fixed macro-F1 0.6327; 260 run completed 260/260, DEA trigger 155, failures 0; formal score rows 2061, level changes 0. Source: Task2 final cached DEA reports. |
| Task3 | `legacy_official_pipeline` | official1799 one-time existing evaluation: 3377 candidate pairs, 1799 mentions | Candidate-pair P/R/F1 0.981707/0.589744/0.736842; B3 F1 0.842650; full cluster same-event F1 0.190339. Source: `legacy_official_metrics.json`. |
| Task4 | `modular_final_combination` | official1799 existing module-level results; ordinal eligible rows 741 | Stable legacy warning: high-severity P/R/F1 48.15%/72.22%/57.78%, Gate Acc/F1 77.60%/70.56%, Alert Recall/F1 64.00%/42.11%, auto coverage 34.91%, Direction Acc/Macro-F1 49.66%/20.35%. Gold-only ordinal: QWK 0.3170, MAE 0.4507, Spearman 0.3411. |

Task4 note: these are module-level results from existing one-time official1799 evaluations of different frozen modules, not a newly rerun unified end-to-end score for the modular combination.
"""
    write_text(OUT / "task1_task4_results_summary.md", summary_md)

    final_report = {
        "generated_at": now_iso(),
        "selected_versions": selection["selected_versions"],
        "constraints_observed": selection["constraints"],
        "task_metrics": {
            "task1": inventories["task1"]["metrics"],
            "task2": inventories["task2"]["metrics"],
            "task3": inventories["task3"]["metrics"],
            "task4": inventories["task4"]["metrics"],
        },
        "task4_heads": selection["task4_heads"],
        "excluded_assets": excluded["excluded_from_production"],
        "archives": {},
    }
    write_json(OUT / "final_delivery_report.json", final_report)

    final_report_md = f"""# Final Delivery Report

Generated at: `{final_report['generated_at']}`

## Selected Versions

- Task1: Core `task1_step5_8e_final_optimization`; safe event recognition `task1_step5_8hi_tag_optimization`; hierarchical model `task1_step5_8hc_hierarchical_event`.
- Task2: `final_cached_dea_v1_260`. The newer `task2_sixdim_ai_fusion_v2` freeze report was checked but it is not production because `advanced_to_main_system=false` and `development_gate_passed=false`.
- Task3: `legacy_official_pipeline`.
- Task4: `modular_final_combination`.

## Task4 Modular Heads

- Stable High-Risk Head: legacy official high-risk module.
- Applicability Gate: legacy official auto-score eligibility gate.
- Alert Head: legacy official alert binary module.
- Direction: legacy official direction module.
- Ordinal Severity: Gold-only ordinal module.
- TRACE five-axis and RAG: explanation/review basis only; no Official Risk influence.

## Controls

- No training, tuning, pseudo-label generation, official1799 filled Gold read, or final_holdout read in this delivery build.
- Frozen models, predictions, configs and historical evaluations were not modified.
- Failed or non-admitted versions are listed in `excluded_assets.json`.
- Official test data, Gold files, CSMAR raw data, credentials and restricted data are excluded from archives.
"""
    write_text(OUT / "final_delivery_report.md", final_report_md)

    dependency_audit = {
        "generated_at": now_iso(),
        "runtime_minimal": ["python>=3.8"],
        "full_replay_optional": [
            "pandas>=2.0,<3.0",
            "pyarrow>=14,<23",
            "scikit-learn",
            "joblib",
            "lightgbm",
            "sqlite3 FTS5",
            "OpenAI-compatible DianJin-R1-32B service for Task2 full historical replay",
        ],
        "current_environment_probe": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "notes": "Smoke runtime intentionally uses standard-library deterministic adapters; full historical replay requires offline model/data mounts.",
    }
    write_json(OUT / "dependency_audit.json", dependency_audit)

    data_license_audit = {
        "generated_at": now_iso(),
        "included_data": [
            "demo_data/sample.jsonl synthetic demo rows",
            "model manifests, metrics registries and checksums",
            "selected frozen model binaries required for platform deployment where license permits local package inclusion",
            "sanitized Task1 taxonomy/config copies with raw examples and restricted paths removed",
        ],
        "excluded_data": [
            "official1799 filled Gold and derived raw Gold workbooks",
            "official test raw workbook/data",
            "CSMAR raw database and source zips",
            "final_holdout data",
            "secret files and credentials",
        ],
        "license_status": "competition internal delivery; restricted upstream data must be mounted separately under its original license.",
    }
    write_json(OUT / "data_license_audit.json", data_license_audit)

    missing_files = {
        "generated_at": now_iso(),
        "missing_in_repository": [
            {
                "item": "official blank originality declaration template",
                "required_for": "national_submission/04_原创性声明",
                "status": "not_found_by_local_search",
                "delivery_action": "placeholder file generated; member signatures are not substituted",
            }
        ],
        "external_mounts_or_services_required_for_full_replay": [
            {
                "item": "DianJin-R1-32B OpenAI-compatible service",
                "required_for": "Task2 full historical replay",
                "packaged": False,
                "reason": "external model service, not a local frozen binary in repository",
            },
            {
                "item": "Task2 SQLite retrieval index",
                "required_for": "Task2 full DEA retrieval replay",
                "packaged": False,
                "reason": "restricted runtime mount; demo adapter does not require it",
            },
            {
                "item": "official test workbook / Gold / final_holdout / CSMAR raw data",
                "required_for": "authorized historical official evaluation replay only",
                "packaged": False,
                "reason": "explicitly excluded by delivery constraints",
            },
        ],
    }
    write_json(OUT / "missing_files.json", missing_files)
    final_report["missing_or_external_assets"] = missing_files

    create_national_submission(summary_md)
    model_manifest = create_compute_delivery(inventories)
    write_json(OUT / "model_manifest.json", model_manifest)
    secret_scan = scan_secrets()
    write_json(OUT / "secret_scan_report.json", secret_scan)

    return final_report


def scan_secrets() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    patterns = [
        ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b", re.IGNORECASE)),
        (
            "secret_assignment",
            re.compile(
                r"\b(api[_-]?key|secret[_-]?key|access[_-]?key|password|passwd)\b\s*[:=]\s*['\"]?[^'\"\s]{8,}",
                re.IGNORECASE,
            ),
        ),
        ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ]
    text_suffixes = {".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".sh", ".py", ".cfg", ".ini"}
    roots = [RUNTIME, NATIONAL, COMPUTE]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.stat().st_size > 1024 * 1024:
                continue
            if path.suffix.lower() not in text_suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for name, pattern in patterns:
                if pattern.search(text):
                    findings.append({"path": rel(path), "pattern": name, "note": "manual_review_required"})
    return {
        "generated_at": now_iso(),
        "scanned_roots": [rel(r) for r in roots],
        "finding_count": len(findings),
        "findings": findings,
        "archive_policy": "No secret-like file names are copied into archives; textual hits are reported for review.",
    }


def create_national_submission(summary_md: str) -> None:
    for sub in ["01_精益画布", "02_可运行项目成果", "03_技术文档", "04_原创性声明"]:
        (NATIONAL / sub).mkdir(parents=True, exist_ok=True)
    write_text(
        NATIONAL / "01_精益画布/精益画布_骨架.md",
        """# 精益画布

## 问题
- 金融事件抽取、可信度评估、事件归并与风险预警在跨源传播场景下难以统一。

## 解决方案
- Task1-Task4 模块化管线：结构化事件、可信评估、CanonicalEvent/Event Thread、风险方向/分级/预警解释。

## 关键指标
- 详见 `../03_技术文档/task1_task4_results_summary.md`。

## 交付形态
- 可运行统一 Demo、冻结资产清单、工程包、模型清单、部署指南。
""",
    )
    write_text(
        NATIONAL / "02_可运行项目成果/README.md",
        """# 可运行项目成果

运行入口位于 `../../runtime/`：

```bash
cd runtime
python3 run_pipeline.py --input demo_data/sample.jsonl --output outputs/result.jsonl
```

该 Demo 只验证统一 schema 与 Task1-Task4 串联，不作为官方评测。
""",
    )
    write_text(NATIONAL / "03_技术文档/task1_task4_results_summary.md", summary_md)
    write_text(
        NATIONAL / "03_技术文档/技术文档_骨架.md",
        """# 技术文档

## 版本冻结
- 详见 `final_asset_selection.json`、四个 `task*_final_inventory.json`。

## 指标来源
- 详见 `metric_source_registry.json`。

## 部署运行
- 详见智算平台交付包中的 `deployment_guide.md`。

## 风险与限制
- Task4 组合版没有重新执行统一官方端到端评测。
- 受限数据、官方测试数据和 Gold 不随包交付。
""",
    )
    write_text(
        NATIONAL / "04_原创性声明/原创性声明_官方空模板待签署.md",
        """# 原创性声明

本文件为待签署空模板占位。请使用赛事官方空模板替换或补充，并由参赛成员本人签字/盖章。

声明人：

日期：
""",
    )


def create_compute_delivery(inventories: dict[str, Any]) -> dict[str, Any]:
    for sub in ["source_code", "models", "configs", "demo_data", "deployment"]:
        (COMPUTE / sub).mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUNTIME, COMPUTE / "source_code/runtime", dirs_exist_ok=True)
    shutil.copy2(RUNTIME / "demo_data/sample.jsonl", COMPUTE / "demo_data/sample.jsonl")

    selected_copies = [
        (ROOT / "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/event_name_model.pkl", COMPUTE / "models/task1/event_name_model.pkl"),
        (ROOT / "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/event_emotion_model.pkl", COMPUTE / "models/task1/event_emotion_model.pkl"),
        (ROOT / "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/frozen_model_manifest.json", COMPUTE / "models/task1/frozen_model_manifest.json"),
        (ROOT / "outputs/task1_step5_8hc_hierarchical_event/models/best_hierarchical_model.pkl", COMPUTE / "models/task1/best_hierarchical_model.pkl"),
        (ROOT / "outputs/task1_step5_8hc_hierarchical_event/models/best_hierarchical_model_manifest.json", COMPUTE / "models/task1/best_hierarchical_model_manifest.json"),
        (ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_model.joblib", COMPUTE / "models/task3/resolver_v2_model.joblib"),
        (ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_selection.json", COMPUTE / "models/task3/resolver_v2_selection.json"),
        (ROOT / "outputs/task3_kg_v3_release/release_manifest.json", COMPUTE / "models/task3/release_manifest.json"),
        (ROOT / "outputs/task4_final_contract_separated_closure/models/task4_final_contract_separated_closure_bundle.joblib", COMPUTE / "models/task4/task4_final_contract_separated_closure_bundle.joblib"),
        (ROOT / "outputs/task4_trace_refine_v3_2_student/models/task4_trace_refine_v3_2_student_bundle.joblib", COMPUTE / "models/task4/task4_trace_refine_v3_2_student_bundle.joblib"),
    ]
    copy_results = [safe_copy(src, dst) for src, dst in selected_copies if src.exists()]
    task4_manifest_src = ROOT / "outputs/task4_final_contract_separated_closure/final_model_manifest.json"
    if task4_manifest_src.exists():
        copy_results.append(
            write_sanitized_json_copy(task4_manifest_src, COMPUTE / "models/task4/final_model_manifest.json")
        )

    config_files = [
        "configs/task2_final_cached_dea_v1.yaml",
        "configs/task3_kg_v3_final.yaml",
        "configs/task4_metric_contract.yaml",
        "configs/task4_label_contract.yaml",
    ]
    for cfg in config_files:
        src = ROOT / cfg
        if src.exists():
            safe_copy(src, COMPUTE / "configs" / Path(cfg).name)
    structured_contract = ROOT / "configs/structured_event_contract_v1.json"
    if structured_contract.exists():
        write_sanitized_json_copy(
            structured_contract,
            COMPUTE / "configs/structured_event_contract_v1.json",
            mode="structured_event_contract",
        )
    task1_taxonomy = ROOT / "configs/task1_event_taxonomy_hierarchical_v1.json"
    if task1_taxonomy.exists():
        write_sanitized_json_copy(
            task1_taxonomy,
            COMPUTE / "configs/task1_event_taxonomy_hierarchical_v1_sanitized.json",
            mode="task1_taxonomy",
        )

    write_text(
        COMPUTE / "requirements.txt",
        """# Minimal demo runtime

# Full historical replay optional dependencies:
pandas>=2.0,<3.0
pyarrow>=14,<23
scikit-learn
joblib
lightgbm
numpy
""",
    )
    write_text(
        COMPUTE / "environment.yml",
        """name: competition-final-delivery
channels:
  - conda-forge
dependencies:
  - python>=3.8
  - pip
  - pip:
      - pandas>=2.0,<3.0
      - pyarrow>=14,<23
      - scikit-learn
      - joblib
      - lightgbm
      - numpy
""",
    )

    deployment_guide = """# Deployment Guide

## Quick Smoke Demo

```bash
cd source_code/runtime
bash healthcheck.sh
bash smoke_test.sh
```

## Full Offline Replay Mounts

The archive intentionally excludes official test data, filled Gold, final_holdout, CSMAR raw data, credentials and restricted datasets.

Mount required restricted assets externally when full historical replay is needed:

- DianJin-R1-32B service exposed as OpenAI-compatible API at `http://127.0.0.1:8000/v1`.
- Task2 SQLite retrieval index at `cache/event_credibility/dea_retrieval_index`.
- Official test workbook only in the authorized evaluation environment; it is not included in this package.
- CSMAR/financial source data only under its original license; not included.

If a full replay script cannot find one of these mounts, it must fail with an explicit missing-mount message. The bundled smoke demo does not silently substitute those restricted assets for official evaluation.

## Production Version Selection

See `model_manifest.json`, `final_asset_selection.json`, and `metric_source_registry.json`.
"""
    write_text(COMPUTE / "deployment_guide.md", deployment_guide)
    write_text(COMPUTE / "deployment/README.md", deployment_guide)
    write_text(OUT / "deployment_guide.md", deployment_guide)

    model_manifest = {
        "generated_at": now_iso(),
        "model_policy": "Only selected production model artifacts are copied. Non-production frozen candidates are referenced in inventory only. Restricted data and Gold are excluded.",
        "copied_models": copy_results,
        "selected_versions": {
            "task1": inventories["task1"]["selected_version"],
            "task2": inventories["task2"]["selected_version"],
            "task3": inventories["task3"]["selected_version"],
            "task4": inventories["task4"]["selected_version"],
        },
        "excluded_large_or_restricted": [
            "Task2 six-dimensional AI fusion v2 model is checked in inventory but not copied because it is not production.",
            "Official test predictions and Gold-derived raw files are not copied.",
            "CSMAR raw and caches are not copied.",
        ],
    }
    write_json(COMPUTE / "model_manifest.json", model_manifest)
    write_json(COMPUTE / "final_asset_selection.json", read_json(OUT / "final_asset_selection.json"))
    write_json(COMPUTE / "metric_source_registry.json", read_json(OUT / "metric_source_registry.json"))
    return model_manifest


def file_checksums(root: Path) -> dict[str, str]:
    checks: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            checks[path.relative_to(root).as_posix()] = sha256_file(path) or ""
    return checks


def write_checksums(root: Path) -> None:
    checks = file_checksums(root)
    lines = [f"{sha}  {name}" for name, sha in checks.items() if name != "checksums.sha256"]
    write_text(root / "checksums.sha256", "\n".join(lines) + "\n")


def create_archives() -> dict[str, Any]:
    write_checksums(COMPUTE)
    write_checksums(OUT)

    national_zip = OUT / "national_submission.zip"
    with zipfile.ZipFile(national_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(NATIONAL.rglob("*")):
            if path.is_file() and not forbidden(path):
                zf.write(path, path.relative_to(OUT).as_posix())

    compute_tar = OUT / "compute_platform_delivery.tar.gz"
    with tarfile.open(compute_tar, "w:gz") as tf:
        for path in sorted(COMPUTE.rglob("*")):
            if path.is_file() and not forbidden(path):
                tf.add(path, arcname=path.relative_to(OUT).as_posix())

    archive_meta = {
        "national_submission.zip": {
            "path": rel(national_zip),
            "size_bytes": national_zip.stat().st_size,
            "sha256": sha256_file(national_zip),
        },
        "compute_platform_delivery.tar.gz": {
            "path": rel(compute_tar),
            "size_bytes": compute_tar.stat().st_size,
            "sha256": sha256_file(compute_tar),
        },
    }
    write_json(OUT / "archive_manifest.json", archive_meta)
    write_checksums(OUT)
    return archive_meta


def validate_archive(archive_meta: dict[str, Any]) -> dict[str, Any]:
    tmp_root = Path("/tmp") / f"competition_final_delivery_validate_{int(time.time())}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    install_start = time.perf_counter()
    tar_path = ROOT / archive_meta["compute_platform_delivery.tar.gz"]["path"]
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if not forbidden(Path(m.name))]
        tf.extractall(tmp_root, members=members)
    install_elapsed = time.perf_counter() - install_start

    runtime_dir = tmp_root / "compute_platform_delivery/source_code/runtime"
    start = run_cmd(["bash", "start.sh"], runtime_dir, timeout=30)
    health = run_cmd(["bash", "healthcheck.sh"], runtime_dir, timeout=30)
    smoke_t0 = time.perf_counter()
    smoke = run_cmd(["bash", "smoke_test.sh"], runtime_dir, timeout=60)
    inference_elapsed = time.perf_counter() - smoke_t0
    e2e = run_cmd(
        ["python3", "run_pipeline.py", "--input", "demo_data/sample.jsonl", "--output", "outputs/e2e_result.jsonl"],
        runtime_dir,
        timeout=60,
    )
    stop = run_cmd(["bash", "stop.sh"], runtime_dir, timeout=30)

    output_path = runtime_dir / "outputs/result.jsonl"
    e2e_path = runtime_dir / "outputs/e2e_result.jsonl"
    rows = 0
    if output_path.exists():
        rows = len([line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()])

    report = {
        "generated_at": now_iso(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(runtime_dir),
        },
        "install_check_dependency": {
            "archive_extracted_to": str(tmp_root),
            "install_time_seconds": round(install_elapsed, 6),
            "dependency_check": "standard-library demo runtime; full replay dependencies audited separately",
        },
        "start_time_seconds": start["elapsed_seconds"],
        "healthcheck_time_seconds": health["elapsed_seconds"],
        "smoke_test_time_seconds": smoke["elapsed_seconds"],
        "inference_time_seconds": round(inference_elapsed, 6),
        "end_to_end_demo_time_seconds": e2e["elapsed_seconds"],
        "peak_memory": {
            "rss_mb": None,
            "gpu_mb": None,
            "note": "resource peak not available from standard subprocess probe; full replay requires platform monitor.",
        },
        "task_status": {
            "task1": smoke["returncode"] == 0,
            "task2": smoke["returncode"] == 0,
            "task3": smoke["returncode"] == 0,
            "task4": smoke["returncode"] == 0,
        },
        "input_output_files": {
            "input": str(runtime_dir / "demo_data/sample.jsonl"),
            "smoke_output": str(output_path),
            "end_to_end_output": str(e2e_path),
            "rows": rows,
        },
        "commands": {
            "start": start,
            "healthcheck": health,
            "smoke_test": smoke,
            "end_to_end_demo": e2e,
            "stop": stop,
        },
        "success": all(x["returncode"] == 0 for x in [start, health, smoke, e2e, stop]),
        "restricted_dependency_policy": {
            "full_replay_missing_mounts_fail_explicitly": True,
            "no_silent_skip": True,
        },
    }
    write_json(OUT / "smoke_test_report.json", report)
    write_text(
        OUT / "validation_full.log",
        "\n\n".join(
            [
                "START\n" + json.dumps(start, ensure_ascii=False, indent=2),
                "HEALTHCHECK\n" + json.dumps(health, ensure_ascii=False, indent=2),
                "SMOKE\n" + json.dumps(smoke, ensure_ascii=False, indent=2),
                "E2E\n" + json.dumps(e2e, ensure_ascii=False, indent=2),
                "STOP\n" + json.dumps(stop, ensure_ascii=False, indent=2),
            ]
        ),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    ensure_clean_out()
    inventories, selection, registry = build_inventories()
    create_runtime()
    final_report = create_docs_and_packages(inventories, selection, registry)
    archive_meta = create_archives()
    if not args.skip_validation:
        smoke_report = validate_archive(archive_meta)
    else:
        smoke_report = {"success": None, "skipped": True}
        write_json(OUT / "smoke_test_report.json", smoke_report)

    final_report["archives"] = archive_meta
    final_report["smoke_test_success"] = smoke_report.get("success")
    write_json(OUT / "final_delivery_report.json", final_report)
    write_checksums(COMPUTE)
    write_checksums(OUT)
    archive_meta = create_archives()
    final_report["archives"] = archive_meta
    write_json(OUT / "final_delivery_report.json", final_report)
    write_checksums(OUT)
    print(json.dumps({"status": "ok", "output_dir": rel(OUT), "archives": archive_meta, "smoke_success": smoke_report.get("success")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
