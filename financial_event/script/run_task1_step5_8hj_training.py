#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main orchestrator for Task1 Step 5.8-HJ training and freezing."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import socket
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.task1_step5_8hj_common import (
    BACKUP_ROOT,
    HIER_MODEL_PKL,
    HIST_HF_REPORT,
    LEGAL_EMOTIONS,
    LOCAL_MACBERT,
    MODEL_DIR,
    OFFICIAL_TEST_XLSX,
    OUT_DIR,
    REPORT_DIR,
    SEEDS,
    UNLABELED_TEST_XLSX,
    allowed_official_events_by_type,
    article_group_key,
    build_emotion_examples,
    build_event_oof_predictions,
    build_training_rows,
    code_metrics,
    event_metrics_from_predictions,
    load_candidate_generator,
    load_official_schemas,
    norm_code,
    norm_text,
    predict_linear_event,
    predict_target_ranker,
    read_json,
    read_jsonl,
    target_full_metrics,
    train_linear_event_ranker,
    train_linear_target_ranker,
    write_json,
    write_jsonl,
    write_sha_manifest,
)

EVENT_BASE_CONFIG = ROOT / "configs/task1_step5_8hj_event_name.yaml"
EMOTION_BASE_CONFIG = ROOT / "configs/task1_step5_8hj_emotion.yaml"
EVENT_TRAIN_SCRIPT = ROOT / "scripts/train_task1_step5_8hj_event_name.py"
EMOTION_TRAIN_SCRIPT = ROOT / "scripts/train_task1_step5_8hj_emotion.py"
EVENT_RUN_DIR = MODEL_DIR / "event_name"
EMOTION_RUN_DIR = MODEL_DIR / "emotion"
TARGET_RUN_DIR = MODEL_DIR / "target"
REPORT_JSON = REPORT_DIR / "task1_step5_8hj_training_report.json"
REPORT_MD = REPORT_DIR / "task1_step5_8hj_training_report.md"
DETAILS_XLSX = REPORT_DIR / "task1_step5_8hj_training_details.xlsx"
COMPARISON_XLSX = REPORT_DIR / "task1_step5_8hj_experiment_comparison.xlsx"
BASELINE_XLSX = REPORT_DIR / "task1_step5_8hj_baseline_vs_optimized.xlsx"
EVENT64_XLSX = REPORT_DIR / "task1_step5_8hj_event_name_64_classes.xlsx"
EMOTION3_XLSX = REPORT_DIR / "task1_step5_8hj_emotion_3_classes.xlsx"
ERROR_XLSX = REPORT_DIR / "task1_step5_8hj_error_analysis.xlsx"
TARGET_OOF_JSON = OUT_DIR / "target_oof_event_map.json"
EVENT_OOF_JSON = OUT_DIR / "event_name_oof_map.json"
FROZEN_MANIFEST = OUT_DIR / "frozen_model_manifest.json"
RUN_STATE = OUT_DIR / "run_state.json"
CHECKPOINT_MANIFEST = OUT_DIR / "checkpoint_manifest.json"
EXPERIMENT_REGISTRY = OUT_DIR / "experiment_registry.json"
SHA_SUMS = OUT_DIR / "SHA256SUMS.txt"

GPU_ENV = ["CUDA_VISIBLE_DEVICES=0,1,2,3"]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def finite_metric(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if math.isfinite(number) else default


def latest_backup_dir() -> Path:
    candidates = sorted(BACKUP_ROOT.glob("task1_step5_8hj_training_*"))
    if candidates:
        return candidates[-1]
    backup = BACKUP_ROOT / f"task1_step5_8hj_training_{time.strftime('%Y%m%d_%H%M%S')}"
    backup.mkdir(parents=True, exist_ok=True)
    return backup


def update_state(stage: str, **extra: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = {}
    if RUN_STATE.exists():
        state = read_json(RUN_STATE)
    state.update({"stage": stage, "updated_at": now_iso(), **extra})
    write_json(RUN_STATE, state)


def write_config(path: Path, base_path: Path, overrides: dict[str, Any]) -> Path:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base, dict):
        base = {}
    base.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def torchrun_cmd() -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run"]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_torch_experiment(script: Path, config: Path, seed: int, output_dir: Path, extra_args: list[str] | None = None, resume: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    master_port = int(os.environ.get("TASK1_HJ_MASTER_PORT") or find_free_port())
    cmd = [
        *torchrun_cmd(),
        "--nnodes=1",
        "--nproc_per_node=4",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=127.0.0.1:%d" % master_port,
        str(script),
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if resume:
        cmd.append("--resume")
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    env["GLOO_SOCKET_IFNAME"] = env.get("GLOO_SOCKET_IFNAME", "lo")
    env["NCCL_SOCKET_IFNAME"] = env.get("NCCL_SOCKET_IFNAME", "lo")
    env["NCCL_IB_DISABLE"] = env.get("NCCL_IB_DISABLE", "1")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"experiment failed: {script.name} seed={seed}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if not summary_path.exists():
        raise RuntimeError(f"missing summary after experiment: {summary_path}")
    return read_json(summary_path)


def convert_target_preds(rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row, cand in zip(rows, candidate_rows):
        if str(cand.get("target_type") or "company") == "company":
            out.append(
                {
                    **row,
                    "target_type": "company",
                    "trading_code": norm_code(cand.get("target_code")),
                    "secu_abbr": norm_text(cand.get("target_name")),
                    "industry_code": "",
                    "industry_name": "",
                }
            )
        else:
            out.append(
                {
                    **row,
                    "target_type": "industry",
                    "trading_code": "",
                    "secu_abbr": "",
                    "industry_code": norm_code(cand.get("target_code")),
                    "industry_name": norm_text(cand.get("target_name")),
                }
            )
    return out


def build_target_candidate_rows(train_rows: list[dict[str, Any]], analyzer: str, seed: int) -> dict[str, Any]:
    model = train_linear_target_ranker(train_rows, seed=seed, analyzer=analyzer)
    generator = load_candidate_generator(train_rows)
    return {"model": model, "generator": generator}


def train_target_suite(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], experiment_registry: list[dict[str, Any]]) -> dict[str, Any]:
    TARGET_RUN_DIR.mkdir(parents=True, exist_ok=True)
    generator = load_candidate_generator(train_rows)
    variants = [
        {"name": "char", "analyzer": "char", "ngram_range": (2, 4)},
        {"name": "word", "analyzer": "word", "ngram_range": (1, 2)},
        {"name": "hash_char", "analyzer": "hash_char", "ngram_range": (2, 4)},
    ]
    best = None
    best_score = None
    no_improve_streak = 0
    for variant in variants:
        for seed in SEEDS:
            exp_id = f"target_{variant['name']}_seed_{seed}"
            exp_dir = TARGET_RUN_DIR / exp_id
            summary_path = exp_dir / "target_summary.json"
            if summary_path.exists():
                summary = read_json(summary_path)
            else:
                model = train_linear_target_ranker(train_rows, seed=seed, analyzer=variant["analyzer"], ngram_range=variant["ngram_range"])
                dev_preds = convert_target_preds(dev_rows, predict_target_ranker(model, dev_rows, generator))
                hold_preds = convert_target_preds(hold_rows, predict_target_ranker(model, hold_rows, generator))
                dev_metrics = target_full_metrics(dev_preds, dev_rows)
                hold_metrics = target_full_metrics(hold_preds, hold_rows)
                exp_dir.mkdir(parents=True, exist_ok=True)
                with (exp_dir / "target_model.pkl").open("wb") as handle:
                    pickle.dump(model, handle)
                write_json(
                    summary_path,
                    {
                        "task": "target",
                        "variant": variant,
                        "seed": seed,
                        "dev_metrics": dev_metrics,
                        "holdout_metrics": hold_metrics,
                        "model_path": str(exp_dir / "target_model.pkl"),
                        "model_sha256": __import__("hashlib").sha256((exp_dir / "target_model.pkl").read_bytes()).hexdigest(),
                    },
                )
                summary = read_json(summary_path)
            dev = summary["dev_metrics"]
            score = (
                finite_metric(dev.get("main_target_exact")),
                finite_metric(dev["target_type"].get("macro_f1")),
                finite_metric(dev["company"]["trading_code"].get("exact_accuracy")),
                finite_metric(dev["industry"]["industry_code"].get("exact_accuracy")),
            )
            experiment_registry.append({"task": "target", "variant": variant, "seed": seed, "summary_path": str(summary_path), "score": score})
            if best_score is None or score > best_score:
                best_score = score
                best = summary
                no_improve_streak = 0
            else:
                no_improve_streak += 1
            if no_improve_streak >= 3:
                return best
    if best is None:
        raise RuntimeError("no target model selected")
    return best


def train_event_linear_suite(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], experiment_registry: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    EVENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    allowed_by_type = allowed_official_events_by_type()
    best = None
    best_score = None
    selected_seed = SEEDS[0]
    no_improve_streak = 0
    for seed in SEEDS:
        exp_id = f"linear_char_seed_{seed}"
        exp_dir = EVENT_RUN_DIR / exp_id
        summary_path = exp_dir / "event_linear_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
        else:
            model = train_linear_event_ranker(train_rows, seed=seed, analyzer="char", ngram_range=(2, 4))
            dev_pred, dev_top, dev_conf = predict_linear_event(model, dev_rows, allowed_by_type)
            hold_pred, hold_top, hold_conf = predict_linear_event(model, hold_rows, allowed_by_type)
            dev_metrics = event_metrics_from_predictions(
                dev_rows,
                dev_pred,
                dev_top,
                dev_conf,
                sorted({r["event_name"] for r in train_rows + dev_rows + hold_rows}),
                allowed_by_type["company"],
                allowed_by_type["industry"],
            )
            hold_metrics = event_metrics_from_predictions(
                hold_rows,
                hold_pred,
                hold_top,
                hold_conf,
                sorted({r["event_name"] for r in train_rows + dev_rows + hold_rows}),
                allowed_by_type["company"],
                allowed_by_type["industry"],
            )
            exp_dir.mkdir(parents=True, exist_ok=True)
            with (exp_dir / "event_linear_model.pkl").open("wb") as handle:
                pickle.dump(model, handle)
            write_json(
                summary_path,
                {
                    "task": "event_linear",
                    "seed": seed,
                    "dev_metrics": dev_metrics,
                    "holdout_metrics": hold_metrics,
                    "model_path": str(exp_dir / "event_linear_model.pkl"),
                    "model_sha256": __import__("hashlib").sha256((exp_dir / "event_linear_model.pkl").read_bytes()).hexdigest(),
                },
            )
            summary = read_json(summary_path)
        score = (
            summary["dev_metrics"]["micro_f1"],
            summary["dev_metrics"]["fixed_64_macro_f1"],
            summary["dev_metrics"]["company44_macro_f1"],
            summary["dev_metrics"]["industry20_macro_f1"],
        )
        experiment_registry.append({"task": "event_linear", "seed": seed, "summary_path": str(summary_path), "score": score})
        if best_score is None or score > best_score:
            best_score = score
            best = summary
            selected_seed = seed
            no_improve_streak = 0
        else:
            no_improve_streak += 1
        if no_improve_streak >= 3:
            break
    if best is None:
        raise RuntimeError("no event linear model selected")
    oof_map, oof_summary = build_event_oof_predictions(train_rows, allowed_by_type, seed=selected_seed, n_splits=3)
    write_json(EVENT_OOF_JSON, oof_map)
    best["oof_summary"] = oof_summary
    return best, oof_map


def build_event_map_for_emotion(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], selected_event_predictions: dict[str, str], oof_map: dict[str, str]) -> dict[str, str]:
    mapping = dict(oof_map)
    for row in dev_rows + hold_rows:
        aid = str(row.get("article_file_id") or row.get("article_id") or "")
        if aid in selected_event_predictions:
            mapping[aid] = selected_event_predictions[aid]
    for row in train_rows:
        aid = str(row.get("article_file_id") or row.get("article_id") or "")
        mapping.setdefault(aid, str(row.get("event_name") or ""))
    return mapping


def run_event_neural_suite(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], experiment_registry: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    EVENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    variants = [
        ("pair_ce_f4", {"loss_type": "ce", "freeze_last_n_layers": 4, "negatives_per_row": 4, "dev_negatives_per_row": 3, "epochs": 1.0}),
        ("pair_focal_f6", {"loss_type": "focal", "freeze_last_n_layers": 6, "negatives_per_row": 6, "dev_negatives_per_row": 4, "epochs": 1.0}),
        ("pair_ce_f8", {"loss_type": "ce", "freeze_last_n_layers": 8, "negatives_per_row": 8, "dev_negatives_per_row": 4, "epochs": 1.0}),
    ]
    best = None
    best_score = None
    selected_seed = SEEDS[0]
    selected_event_preds = {}
    selected_dev_pred_rows = []
    selected_hold_pred_rows = []
    no_improve_streak = 0
    for variant_name, overrides in variants:
        for seed in SEEDS:
            exp_id = f"{variant_name}_seed_{seed}"
            exp_dir = EVENT_RUN_DIR / exp_id
            summary_path = exp_dir / "training_summary.json"
            config_path = OUT_DIR / "configs" / f"{exp_id}.yaml"
            write_config(config_path, EVENT_BASE_CONFIG, overrides)
            summary = run_torch_experiment(EVENT_TRAIN_SCRIPT, config_path, seed, exp_dir, resume=True)
            experiment_registry.append({"task": "event_neural", "variant": variant_name, "seed": seed, "summary_path": str(summary_path), "score": (
                summary["dev_article_metrics"]["micro_f1"],
                summary["dev_article_metrics"]["fixed_64_macro_f1"],
                summary["dev_article_metrics"]["company44_macro_f1"],
                summary["dev_article_metrics"]["industry20_macro_f1"],
            )})
            score = (
                summary["dev_article_metrics"]["micro_f1"],
                summary["dev_article_metrics"]["fixed_64_macro_f1"],
                summary["dev_article_metrics"]["company44_macro_f1"],
                summary["dev_article_metrics"]["industry20_macro_f1"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best = summary
                selected_seed = seed
                pred_dev_path = Path(exp_dir / "dev_predictions.jsonl")
                pred_hold_path = Path(exp_dir / "holdout_predictions.jsonl")
                selected_dev_pred_rows = read_jsonl(pred_dev_path) if pred_dev_path.exists() else []
                selected_hold_pred_rows = read_jsonl(pred_hold_path) if pred_hold_path.exists() else []
                selected_event_preds = {
                    str(row["article_file_id"]): str(row["pred_event_name"])
                    for row in selected_dev_pred_rows + selected_hold_pred_rows
                }
                no_improve_streak = 0
            else:
                no_improve_streak += 1
            if no_improve_streak >= 3:
                break
        if no_improve_streak >= 3:
            break
    if best is None:
        raise RuntimeError("no event neural model selected")
    train_oof_map, _ = build_event_oof_predictions(train_rows, allowed_official_events_by_type(), seed=selected_seed, n_splits=3)
    event_map = build_event_map_for_emotion(train_rows, dev_rows, hold_rows, selected_event_preds, train_oof_map)
    write_json(OUT_DIR / "event_name_feature_map.json", event_map)
    return best, event_map, selected_event_preds


def run_emotion_neural_suite(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], event_map: dict[str, str], experiment_registry: list[dict[str, Any]]) -> dict[str, Any]:
    EMOTION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    variants = [
        ("text_only_ce", {"loss_type": "ce", "freeze_last_n_layers": 4, "epochs": 2.0, "use_empty_event_feature": True}),
        ("event_cb", {"loss_type": "ce", "freeze_last_n_layers": 4, "epochs": 2.0}),
        ("event_focal", {"loss_type": "focal", "freeze_last_n_layers": 6, "epochs": 2.0}),
    ]
    best = None
    best_score = None
    no_improve_streak = 0
    for variant_name, overrides in variants:
        for seed in SEEDS:
            exp_id = f"{variant_name}_seed_{seed}"
            exp_dir = EMOTION_RUN_DIR / exp_id
            summary_path = exp_dir / "training_summary.json"
            config_path = OUT_DIR / "configs" / f"{exp_id}.yaml"
            write_config(config_path, EMOTION_BASE_CONFIG, overrides)
            event_map_file = OUT_DIR / "event_name_feature_map_empty.json" if overrides.get("use_empty_event_feature") else OUT_DIR / "event_name_feature_map.json"
            if overrides.get("use_empty_event_feature"):
                write_json(event_map_file, {str(row.get("article_file_id") or row.get("article_id") or ""): "" for row in train_rows + dev_rows + hold_rows})
            summary = run_torch_experiment(
                EMOTION_TRAIN_SCRIPT,
                config_path,
                seed,
                exp_dir,
                extra_args=["--event-map-file", str(event_map_file)],
                resume=True,
            )
            score = (
                summary["dev_metrics"]["macro_f1"],
                summary["dev_metrics"]["balanced_accuracy"],
                summary["dev_metrics"]["per_class"][1]["f1"] if len(summary["dev_metrics"]["per_class"]) > 1 else 0.0,
                summary["dev_metrics"]["per_class"][2]["f1"] if len(summary["dev_metrics"]["per_class"]) > 2 else 0.0,
            )
            experiment_registry.append({"task": "emotion_neural", "variant": variant_name, "seed": seed, "summary_path": str(summary_path), "score": score})
            if best_score is None or score > best_score:
                best_score = score
                best = summary
                no_improve_streak = 0
            else:
                no_improve_streak += 1
            if no_improve_streak >= 3:
                break
        if no_improve_streak >= 3:
            break
    if best is None:
        raise RuntimeError("no emotion model selected")
    return best


def build_final_manifest(target_best: dict[str, Any], event_best: dict[str, Any], emotion_best: dict[str, Any], backup_dir: Path, experiment_registry: list[dict[str, Any]], source_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "task1_target_event_emotion_optimized_and_frozen",
        "training_status": "performed",
        "generated_at": now_iso(),
        "model_frozen": True,
        "thresholds_frozen": True,
        "taxonomy_frozen": True,
        "official_test_labels_read": False,
        "backup_dir": str(backup_dir),
        "source_meta": source_meta,
        "selected": {
            "target": target_best,
            "event": event_best,
            "emotion": emotion_best,
        },
        "experiment_registry": str(EXPERIMENT_REGISTRY),
        "checkpoint_manifest": str(CHECKPOINT_MANIFEST),
        "sha_manifest": str(SHA_SUMS),
        "model_dirs": {
            "target": str(TARGET_RUN_DIR),
            "event": str(EVENT_RUN_DIR),
            "emotion": str(EMOTION_RUN_DIR),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    EMOTION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_RUN_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = latest_backup_dir()

    update_state("initialized", backup_dir=str(backup_dir), model_frozen=False, thresholds_frozen=False, taxonomy_frozen=True, official_test_labels_read=False)
    formal_sha_before = write_sha_manifest(
        OUT_DIR / "formal_sha_before.txt",
        [p for p in [HIER_MODEL_PKL, HIST_HF_REPORT, OFFICIAL_TEST_XLSX, UNLABELED_TEST_XLSX] if p.exists()],
    )

    rows, rows_by_split, source_rows, meta = build_training_rows()
    train_rows = rows_by_split["train"]
    dev_rows = rows_by_split["model_selection_dev"]
    hold_rows = rows_by_split["untouched_internal_holdout"]
    update_state(
        "data_loaded",
        split_counts={k: len(v) for k, v in rows_by_split.items()},
        source_stats=meta["source_stats"],
        manual5000_stats=meta["manual5000_stats"],
        incremental_stats=meta["incremental_stats"],
    )

    experiment_registry: list[dict[str, Any]] = []
    target_best = train_target_suite(train_rows, dev_rows, hold_rows, experiment_registry)
    event_linear_best, oof_map = train_event_linear_suite(train_rows, dev_rows, hold_rows, experiment_registry)
    event_best, event_map, selected_event_preds = run_event_neural_suite(train_rows, dev_rows, hold_rows, experiment_registry)
    emotion_best = run_emotion_neural_suite(train_rows, dev_rows, hold_rows, event_map, experiment_registry)

    write_json(TARGET_OOF_JSON, oof_map)
    write_json(EXPERIMENT_REGISTRY, {"experiments": experiment_registry})
    selected_bundle = build_final_manifest(target_best, event_best, emotion_best, backup_dir, experiment_registry, meta)
    write_json(FROZEN_MANIFEST, selected_bundle)
    write_json(
        CHECKPOINT_MANIFEST,
        {
            "status": selected_bundle["status"],
            "training_status": selected_bundle["training_status"],
            "selected_target": target_best.get("model_path"),
            "selected_event": event_best.get("model_dir"),
            "selected_emotion": emotion_best.get("model_dir"),
            "backup_dir": str(backup_dir),
            "updated_at": now_iso(),
        },
    )

    sha_targets = [
        OUT_DIR / "run_state.json",
        CHECKPOINT_MANIFEST,
        EXPERIMENT_REGISTRY,
        FROZEN_MANIFEST,
        TARGET_OOF_JSON,
        EVENT_OOF_JSON,
    ]
    write_sha_manifest(SHA_SUMS, sha_targets)
    formal_sha_after = write_sha_manifest(
        OUT_DIR / "formal_sha_after.txt",
        [p for p in [HIER_MODEL_PKL, HIST_HF_REPORT, OFFICIAL_TEST_XLSX, UNLABELED_TEST_XLSX] if p.exists()],
    )
    update_state(
        "frozen",
        selected_status=selected_bundle["status"],
        target_model=target_best.get("model_path"),
        event_model=event_best.get("model_dir"),
        emotion_model=emotion_best.get("model_dir"),
        formal_sha_match=formal_sha_before == formal_sha_after,
    )
    print(json.dumps(selected_bundle, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
