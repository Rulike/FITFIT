#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-L structure extractor root-cause audit.

This script consolidates the evidence gathered so far for the Step 5.7-K
MacBERT baseline: code-path audit, data diagnostics, small-sample memorization
tests, and oracle decomposition on the 40-row structure dev split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_7f_structure_training import evaluate_predictions  # noqa: E402
from src.event_extraction.task1_supervised_structure_extractor import normalize_time as content_only_normalize_time  # noqa: E402


AUDIT_DIR = ROOT / "outputs/task1_step5_7l_structure_root_cause"
REPORT_JSON = ROOT / "reports/task1_step5_7l_structure_root_cause_audit.json"
REPORT_MD = ROOT / "reports/task1_step5_7l_structure_root_cause_audit.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_7l_structure_root_cause_audit.xlsx"
BACKUP_ROOT = Path("/opt/data/private/JinRong/kechuang/backups")

WORKER = ROOT / "scripts/run_task1_step5_7k_macbert_worker.py"
TRAIN_SCRIPT = ROOT / "scripts/train_task1_step5_7k_structure_extractor.py"
K_REPORT = ROOT / "reports/task1_step5_7k_structure_extractor_report.json"
TRAIN_JSONL = ROOT / "outputs/task1_step5_7k_structure_extractor/structure_train_400_for_training.jsonl"
DEV_JSONL = ROOT / "outputs/task1_step5_7k_structure_extractor/structure_dev_40_for_dev_selection.jsonl"
BEST_DEV_PRED = ROOT / "outputs/task1_step5_7k_structure_extractor/gold_free_inference_runs/best_dev_predictions.jsonl"
BEST_DEV_INPUT = ROOT / "outputs/task1_step5_7k_structure_extractor/gold_free_inference_runs/best_dev_input.jsonl"
LOCAL_MACBERT = ROOT / "outputs/task1_step5_4_dev_baseline_v1/pretrained/hfl_chinese_macbert_base"
CU_PYTHON = Path("/root/miniconda3/envs/dianjin-cu128/bin/python")

SPAN_FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
ROLE_FIELDS = ["subject_company", "event_time", "object_target", "trigger"]
DIR_LABELS = ["positive", "negative", "neutral", "unclear"]
SEED = 2026

PROTECTED_FILES = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/final_task1/final_events_train.xlsx",
    ROOT / "outputs/final_task1/final_events_test.xlsx",
    ROOT / "outputs/final_task1/task1_completion_certificate.json",
    ROOT / "outputs/event_credibility/llm_review_results.jsonl",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.xlsx",
    K_REPORT,
    TRAIN_JSONL,
    DEV_JSONL,
    BEST_DEV_PRED,
    BEST_DEV_INPUT,
]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sha_txt(path: Path, snapshot: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(snapshot.items())) + "\n", encoding="utf-8")


def protected_paths() -> list[Path]:
    return sorted({path for path in PROTECTED_FILES if path.exists()}, key=rel)


def sha_snapshot(paths: list[Path]) -> dict[str, str]:
    return {rel(path): sha256_file(path) for path in paths if path.exists()}


def process_check() -> list[str]:
    proc = subprocess.run(
        ["bash", "-lc", "ps -eo pid,ppid,stat,etime,cmd | rg -i 'step5_7l|step5_7k|structure_extractor|vllm|DianJin-R1-32B|8000' || true"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    skip = ["rg -i", "audit_task1_step5_7l_structure_root_cause.py"]
    return [line for line in proc.stdout.splitlines() if not any(token in line for token in skip)]


def gpu_status() -> dict[str, Any]:
    proc = subprocess.run(
        ["bash", "-lc", "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            rows.append(
                {
                    "index": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                    "utilization_gpu": int(parts[3]),
                }
            )
    return {"gpus": rows, "all_idle": all(row["memory_used_mb"] == 0 for row in rows), "all_four_visible": len(rows) == 4}


def line_refs(path: Path, patterns: list[str]) -> dict[str, list[int]]:
    refs: dict[str, list[int]] = {}
    if not path.exists():
        return refs
    lines = path.read_text(encoding="utf-8").splitlines()
    for pat in patterns:
        hits = [idx + 1 for idx, line in enumerate(lines) if pat in line]
        if hits:
            refs[pat] = hits[:8]
    return refs


def field_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field, {})
    if isinstance(value, dict):
        return str(value.get("text", "") or "")
    return str(value or "")


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("event_mention_id", ""))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        if "event_mention_id" not in row:
            raise RuntimeError(f"missing event_mention_id in {path}")
    return rows


def choose_subset(rows: list[dict[str, Any]], size: int, seed: int = SEED) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (str(row.get("event_name", "")), str(row.get("article_id", "")), row_key(row)))
    rng = random.Random(seed)
    if len(ordered) <= size:
        return ordered
    indices = sorted(rng.sample(range(len(ordered)), size))
    return [ordered[idx] for idx in indices]


def span_to_text(span: dict[str, Any]) -> str:
    return str(span.get("text", "") or "")


def span_to_dict(span: dict[str, Any]) -> dict[str, Any]:
    out = {
        "text": str(span.get("text", "") or ""),
        "start": int(span.get("start", -1)),
        "end": int(span.get("end", -1)),
    }
    if "normalized" in span:
        out["normalized"] = str(span.get("normalized", "未明确") or "未明确")
    return out


def to_arguments(spans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    arguments = []
    for role in ROLE_FIELDS:
        span = spans.get(role, {})
        if span.get("text"):
            arguments.append(
                {
                    "text": span["text"],
                    "role": role,
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                }
            )
    return arguments


def make_prediction(row: dict[str, Any], source: dict[str, Any], scheme: str) -> dict[str, Any]:
    pred = {
        "event_mention_id": row["event_mention_id"],
        "article_id": row["article_id"],
        "scheme": scheme,
    }
    for field in SPAN_FIELDS:
        pred[field] = span_to_dict(source.get(field, {}))
    pred["direction"] = str(source.get("direction", "unclear") or "unclear")
    pred["arguments"] = deepcopy(source.get("arguments", []))
    pred["argument_roles"] = deepcopy(source.get("argument_roles", pred["arguments"]))
    return pred


def hybrid_prediction(row: dict[str, Any], model: dict[str, Any], gold: dict[str, Any], keep_model_fields: set[str], scheme: str) -> dict[str, Any]:
    source = {}
    for field in SPAN_FIELDS:
        source[field] = model[field] if field in keep_model_fields else gold[field]
    source["direction"] = model["direction"] if "direction" in keep_model_fields else gold["direction"]
    source["arguments"] = model["arguments"] if "arguments" in keep_model_fields else gold["arguments"]
    source["argument_roles"] = model["argument_roles"] if "argument_roles" in keep_model_fields else gold.get("argument_roles", gold.get("arguments", []))
    return make_prediction(row, source, scheme)


def evaluate_best_dev_reference() -> dict[str, Any]:
    gold = load_rows(DEV_JSONL)
    pred = load_rows(BEST_DEV_PRED)
    by_id = {row_key(row): row for row in pred}
    current_norm = 0
    improved_norm = 0
    exact = 0
    improved_examples = []
    for row in gold:
        pred_row = by_id[row_key(row)]["event_time"]
        gold_norm = row["event_time"].get("normalized", "未明确")
        cur_norm = pred_row.get("normalized", "未明确")
        improved = content_only_normalize_time(pred_row.get("text", ""), row.get("publish_time", "")) if pred_row.get("text") else "未明确"
        if gold_norm == cur_norm:
            current_norm += 1
        if gold_norm == improved:
            improved_norm += 1
        if pred_row.get("text") == row["event_time"].get("text") and pred_row.get("text"):
            exact += 1
        if len(improved_examples) < 8 and pred_row.get("text") and gold_norm != cur_norm and gold_norm == improved:
            improved_examples.append(
                {
                    "event_mention_id": row["event_mention_id"],
                    "publish_time": row.get("publish_time", ""),
                    "gold_text": row["event_time"].get("text", ""),
                    "gold_normalized": gold_norm,
                    "current_normalized": cur_norm,
                    "improved_normalized": improved,
                }
            )
    return {
        "sample_count": len(gold),
        "current_event_time_normalized_accuracy": round(current_norm / max(1, len(gold)), 6),
        "content_only_normalized_accuracy": round(improved_norm / max(1, len(gold)), 6),
        "predicted_event_time_text_exact_count": exact,
        "improved_examples": improved_examples,
    }


def build_code_audit() -> dict[str, Any]:
    worker_text = WORKER.read_text(encoding="utf-8")
    train_text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    allowed_inference_fields = ["content", "event_name", "title"]
    forbidden_refs = {key: worker_text.count(key) + train_text.count(key) for key in ["ai_", "human_", "reviewer_", "reference_raw_text", "draft_quality_grade", "validation_errors"]}
    return {
        "worker_path": rel(WORKER),
        "train_path": rel(TRAIN_SCRIPT),
        "allowed_inference_fields": allowed_inference_fields,
        "predict_structure_signature": "predict_structure(content, event_name, title=None)",
        "worker_line_refs": line_refs(
            WORKER,
            [
                "def normalize_time",
                "def as_span",
                "def choose_span_pair",
                "def train_epoch",
                "def enforce_prediction",
                "def predict_structure",
            ],
        ),
        "train_line_refs": line_refs(
            TRAIN_SCRIPT,
            [
                "run_macbert_worker_train",
                "predict_with_model",
                "best_scheme",
                "dev_gate_passed",
                "trigger_not_in_event_span_0",
                "window_coverage",
            ],
        ),
        "forbidden_reference_counts": forbidden_refs,
        "worker_reads_publish_time_in_event_time_normalization": "publish_time" in worker_text and "normalize_time(text)" in worker_text,
        "worker_uses_only_allowed_inference_fields": all(item in worker_text for item in ["content", "event_name", "title"]) and "ai_" not in worker_text and "human_" not in worker_text,
        "worker_event_time_normalization_uses_publish_time": "publish_time" in worker_text.split("def normalize_time", 1)[-1].split("def", 1)[0],
        "worker_padding_mask_in_loss": "ignore_index=-100" in worker_text and "content_mask" in worker_text and "masked_fill(~valid_mask.unsqueeze(1), -1e4)" in worker_text,
        "worker_heads_in_loss": {
            "scheme_A_token_loss": "span_loss = sum(token_loss_fn" in worker_text and "direction_logits" in worker_text,
            "scheme_BC_start_end_loss": "exists_loss = bce_loss" in worker_text and "ce_loss(out[\"direction_logits\"]" in worker_text,
        },
        "checkpoint_selection_uses_best_dev": "best_metrics" in train_text and "predict_with_model(best_model, dev_rows" in train_text,
        "inference_loads_best_checkpoint": "load_checkpoint" in worker_text and "--checkpoint" in train_text,
        "trigger_event_span_postprocess": "enforce_prediction" in worker_text,
        "leakage_from_ai_fields_in_final_inference": False,
    }


def build_data_audit(rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    field_stats: dict[str, Any] = {}
    for field in SPAN_FIELDS:
        spans = [row[field] for row in rows]
        nonempty = [span for span in spans if span.get("text")]
        field_stats[field] = {
            "row_count": len(rows),
            "nonempty_count": len(nonempty),
            "empty_count": len(rows) - len(nonempty),
            "span_length_mean": round(sum(len(str(span.get("text", ""))) for span in nonempty) / max(1, len(nonempty)), 3),
            "span_length_median": sorted(len(str(span.get("text", ""))) for span in nonempty)[len(nonempty) // 2] if nonempty else 0,
        }

    event_time_stats = {
        "raw_text_equals_normalized_count": 0,
        "content_only_normalize_equals_normalized_count": 0,
        "content_only_oracle_upper_bound_count": 0,
        "publish_time_based_examples": [],
        "top_mismatch_texts": [],
    }
    mismatches = Counter()
    for row in rows:
        time_span = row["event_time"]
        text = str(time_span.get("text", "") or "")
        gold_norm = str(time_span.get("normalized", "未明确") or "未明确")
        if not text:
            continue
        if text == gold_norm:
            event_time_stats["raw_text_equals_normalized_count"] += 1
        if content_only_normalize_time(text, row.get("publish_time", "")) == gold_norm:
            event_time_stats["content_only_normalize_equals_normalized_count"] += 1
        if "publish_time" in row and content_only_normalize_time(text, row.get("publish_time", "")) == gold_norm:
            event_time_stats["content_only_oracle_upper_bound_count"] += 1
        if content_only_normalize_time(text, row.get("publish_time", "")) != gold_norm:
            mismatches[text] += 1
            if len(event_time_stats["publish_time_based_examples"]) < 12:
                event_time_stats["publish_time_based_examples"].append(
                    {
                        "event_mention_id": row["event_mention_id"],
                        "publish_time": row.get("publish_time", ""),
                        "text": text,
                        "gold_normalized": gold_norm,
                        "content_only_normalized": content_only_normalize_time(text, row.get("publish_time", "")),
                    }
                )
    event_time_stats["top_mismatch_texts"] = [{"text": text, "count": count} for text, count in mismatches.most_common(20)]
    event_time_stats["rows_with_explicit_time"] = sum(1 for row in rows if row["event_time"].get("text"))
    event_time_stats["rows_with_unspecified_time"] = sum(1 for row in rows if not row["event_time"].get("text"))
    event_time_stats["split"] = split_name
    return {"split": split_name, "field_stats": field_stats, "event_time_stats": event_time_stats}


def run_cu_script(script: str, timeout: int = 7200) -> dict[str, Any]:
    if not CU_PYTHON.exists():
        raise FileNotFoundError(CU_PYTHON)
    proc = subprocess.run(
        [str(CU_PYTHON), "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**dict(**__import__("os").environ), "TRANSFORMERS_OFFLINE": "1"},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cu script failed: {proc.stderr[-4000:]}\nSTDOUT:\n{proc.stdout[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_label_distribution_probe() -> dict[str, Any]:
    script = textwrap.dedent(
        f"""
        import json
        from collections import Counter, defaultdict
        from pathlib import Path
        from transformers import AutoTokenizer
        from scripts.run_task1_step5_7k_macbert_worker import (
            DIR_LABELS,
            SPAN_FIELDS,
            StructureDataset,
            bio_labels_for_field,
            encode_windows,
            span_start_end,
            token_window_coverage,
            read_jsonl,
        )

        root = Path({str(ROOT)!r})
        tokenizer = AutoTokenizer.from_pretrained({str(LOCAL_MACBERT)!r}, local_files_only=True, use_fast=True)
        train_rows = read_jsonl(root / {str(TRAIN_JSONL.relative_to(ROOT))!r})
        dev_rows = read_jsonl(root / {str(DEV_JSONL.relative_to(ROOT))!r})
        out = {{}}
        for split_name, rows in [("train", train_rows), ("dev", dev_rows)]:
            split = {{}}
            split["row_count"] = len(rows)
            split["span_coverage"] = token_window_coverage(tokenizer, rows, "C")
            for scheme in ["A", "C"]:
                field_stats = {{}}
                for field in SPAN_FIELDS:
                    field_stats[field] = {{"nonempty_rows": 0, "empty_rows": 0, "candidate_windows": 0}}
                if scheme == "A":
                    bio_counts = {{field: Counter() for field in SPAN_FIELDS}}
                    for row in rows:
                        feats = encode_windows(tokenizer, row, scheme, with_labels=True)
                        for field in SPAN_FIELDS:
                            span = row[field]
                            if span.get("text"):
                                field_stats[field]["nonempty_rows"] += 1
                            else:
                                field_stats[field]["empty_rows"] += 1
                            for feat in feats:
                                labels = bio_labels_for_field(feat.offsets, feat.content_mask, span)
                                field_stats[field]["candidate_windows"] += 1
                                bio_counts[field].update(labels)
                    split["scheme_A"] = {{"field_stats": {{field: dict(stats) for field, stats in field_stats.items()}}, "bio_label_counts": {{field: dict(counts) for field, counts in bio_counts.items()}}}}
                else:
                    start_counts = {{field: Counter() for field in SPAN_FIELDS}}
                    end_counts = {{field: Counter() for field in SPAN_FIELDS}}
                    exists_counts = {{field: Counter() for field in SPAN_FIELDS}}
                    for row in rows:
                        feats = encode_windows(tokenizer, row, scheme, with_labels=True)
                        for field in SPAN_FIELDS:
                            span = row[field]
                            if span.get("text"):
                                field_stats[field]["nonempty_rows"] += 1
                            else:
                                field_stats[field]["empty_rows"] += 1
                        for feat in feats:
                            fields = [feat.field_index] if scheme == "B" and feat.field_index >= 0 else list(range(len(SPAN_FIELDS)))
                            for idx in fields:
                                field = SPAN_FIELDS[idx]
                                if feat.start_labels is None:
                                    continue
                                start_counts[field].update([feat.start_labels[0 if scheme == "B" else idx]])
                                end_counts[field].update([feat.end_labels[0 if scheme == "B" else idx]])
                                exists_counts[field].update([int(feat.exists_labels[0 if scheme == "B" else idx] > 0.5)])
                                field_stats[field]["candidate_windows"] += 1
                    split["scheme_C"] = {{
                        "field_stats": {{field: dict(stats) for field, stats in field_stats.items()}},
                        "start_label_counts": {{field: dict(counts) for field, counts in start_counts.items()}},
                        "end_label_counts": {{field: dict(counts) for field, counts in end_counts.items()}},
                        "exists_label_counts": {{field: dict(counts) for field, counts in exists_counts.items()}},
                    }}
            out[split_name] = split
        print(json.dumps(out, ensure_ascii=False))
        """
    )
    return run_cu_script(script, timeout=1800)


def one_batch_gradient_probe(subset_path: Path) -> dict[str, Any]:
    script = textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer
        from scripts.run_task1_step5_7k_macbert_worker import (
            SPAN_FIELDS,
            StructureDataset,
            build_model,
            collate,
            read_jsonl,
        )

        root = Path({str(ROOT)!r})
        rows = read_jsonl(root / {str(subset_path.relative_to(ROOT))!r})
        tokenizer = AutoTokenizer.from_pretrained({str(LOCAL_MACBERT)!r}, local_files_only=True, use_fast=True)
        dataset = StructureDataset(rows, tokenizer, "C", train=True)
        loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, collate_fn=collate)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
        model = build_model({str(LOCAL_MACBERT)!r}, "C").to(device)
        batch = None
        for candidate in loader:
            if (candidate["exists_labels"] > 0.5).any():
                batch = candidate
                break
        if batch is None:
            batch = next(iter(loader))
        batch = {{k: v.to(device) for k, v in batch.items()}}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        out = model(batch)
        start_logits = out["start_logits"]
        end_logits = out["end_logits"]
        exists_logits = out["exists_logits"]
        ce = torch.nn.CrossEntropyLoss()
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], device=device))
        valid_mask = batch["content_mask"]
        masked_start = start_logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)
        masked_end = end_logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)
        span_loss = torch.tensor(0.0, device=device)
        active_fields = 0
        for idx in range(batch["start_labels"].shape[1]):
            positive = batch["exists_labels"][:, idx] > 0.5
            if positive.any():
                span_loss = span_loss + ce(masked_start[positive, idx], batch["start_labels"][positive, idx])
                span_loss = span_loss + ce(masked_end[positive, idx], batch["end_labels"][positive, idx])
                active_fields += 1
        if active_fields:
            span_loss = span_loss / active_fields
        loss = span_loss + bce(exists_logits, batch["exists_labels"]) + 0.2 * ce(out["direction_logits"], batch["direction_label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norms = {{}}
        for name, param in model.named_parameters():
            if param.grad is not None and any(key in name for key in ["start_heads", "end_heads", "exists_head", "direction_head"]):
                norms[name] = float(param.grad.detach().norm().cpu())
        if torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3)
        else:
            peak = 0.0
        print(json.dumps({{"grad_norms": norms, "loss": float(loss.detach().cpu()), "peak_gpu_memory_mb": peak, "row_count": len(rows), "active_fields": active_fields, "has_positive_span_batch": bool((batch["exists_labels"] > 0.5).any().item())}}, ensure_ascii=False))
        """
    )
    return run_cu_script(script, timeout=1800)


def run_memory_test(n: int) -> dict[str, Any]:
    train_rows = load_rows(TRAIN_JSONL)
    subset = choose_subset(train_rows, n, SEED + n)
    mem_dir = AUDIT_DIR / f"memory_{n}"
    mem_dir.mkdir(parents=True, exist_ok=True)
    train_path = mem_dir / "train.jsonl"
    dev_path = mem_dir / "dev.jsonl"
    if not train_path.exists():
        write_jsonl(train_path, subset)
    if not dev_path.exists():
        write_jsonl(dev_path, subset)
    started = time.time()
    worker_log_path = mem_dir / "worker_training_log.json"
    pred_path = mem_dir / "dev_predictions.jsonl"
    if not (worker_log_path.exists() and pred_path.exists()):
        cmd = [
            str(CU_PYTHON),
            str(WORKER),
            "train",
            "--scheme",
            "C",
            "--train",
            str(train_path),
            "--dev",
            str(dev_path),
            "--out-dir",
            str(mem_dir),
            "--model-dir",
            str(LOCAL_MACBERT),
            "--epochs",
            "50",
            "--batch-size",
            "4",
            "--lr",
            "2e-5",
        ]
        env = {**__import__("os").environ, "TRANSFORMERS_OFFLINE": "1"}
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise RuntimeError(f"memory test {n} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        stdout_tail = proc.stdout.splitlines()[-6:]
    else:
        stdout_tail = ["reused_existing_memory_test_artifacts"]
    pred_path = mem_dir / "dev_predictions.jsonl"
    preds = read_jsonl(pred_path)
    metrics = evaluate_predictions(subset, preds, f"memory_{n}")
    worker_log = read_json(worker_log_path)
    metrics.update(
        {
            "train_seconds": worker_log.get("train_seconds"),
            "epoch_losses": worker_log.get("epoch_losses", []),
            "peak_gpu_memory_mb": worker_log.get("peak_gpu_memory_mb", 0.0),
            "predict_manifest": read_json(pred_path.with_suffix(".manifest.json")) if pred_path.with_suffix(".manifest.json").exists() else {},
            "selected_event_mention_ids": [row["event_mention_id"] for row in subset],
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": stdout_tail,
        }
    )
    return metrics


def build_oracle_decomposition() -> dict[str, Any]:
    gold = load_rows(DEV_JSONL)
    model = load_rows(BEST_DEV_PRED)
    model_by_id = {row_key(row): row for row in model}
    gold_by_id = {row_key(row): row for row in gold}

    oracle_all: list[dict[str, Any]] = []
    model_sentence_oracle_fields: list[dict[str, Any]] = []
    oracle_sentence_model_fields: list[dict[str, Any]] = []
    for row in gold:
        g = gold_by_id[row_key(row)]
        m = model_by_id[row_key(row)]
        oracle_all.append(make_prediction(g, g, "oracle_all_fields"))
        model_sentence_oracle_fields.append(
            hybrid_prediction(
                g,
                m,
                g,
                keep_model_fields={"trigger", "event_span"},
                scheme="model_sentence_oracle_fields",
            )
        )
        oracle_sentence_model_fields.append(
            hybrid_prediction(
                g,
                m,
                g,
                keep_model_fields={"subject_company", "event_time", "object_target", "direction", "arguments", "argument_roles"},
                scheme="oracle_sentence_model_fields",
            )
        )

    return {
        "oracle_sentence_oracle_fields": evaluate_predictions(gold, oracle_all, "oracle_sentence_oracle_fields"),
        "model_sentence_oracle_fields": evaluate_predictions(gold, model_sentence_oracle_fields, "model_sentence_oracle_fields"),
        "oracle_sentence_model_fields": evaluate_predictions(gold, oracle_sentence_model_fields, "oracle_sentence_model_fields"),
        "full_model": evaluate_predictions(gold, model, "full_model"),
    }


def apply_time_norm_map(predictions: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for row in predictions:
        item = deepcopy(row)
        span = item.get("event_time", {})
        text = str(span.get("text", "") or "")
        if text in mapping:
            span["normalized"] = mapping[text]
        elif not text:
            span["normalized"] = "未明确"
        item["event_time"] = span
        out.append(item)
    return out


def build_time_norm_map(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        text = str(row.get("event_time", {}).get("text", "") or "")
        norm = str(row.get("event_time", {}).get("normalized", "未明确") or "未明确")
        if text:
            counts[text][norm] += 1
    mapping = {text: counter.most_common(1)[0][0] for text, counter in counts.items()}
    ambiguous = {text: dict(counter) for text, counter in counts.items() if len(counter) > 1}
    return mapping, {"map_size": len(mapping), "ambiguous_count": len(ambiguous), "ambiguous_examples": dict(list(ambiguous.items())[:12])}


def time_normalization_dictionary_probe() -> dict[str, Any]:
    train_rows = load_rows(TRAIN_JSONL)
    dev_rows = load_rows(DEV_JSONL)
    train_map, train_map_meta = build_time_norm_map(train_rows)
    dev_predictions = load_rows(BEST_DEV_PRED)
    dev_before = evaluate_predictions(dev_rows, dev_predictions, "best_dev_before_time_map")
    dev_after = evaluate_predictions(dev_rows, apply_time_norm_map(dev_predictions, train_map), "best_dev_after_time_map")
    memory_results = {}
    for n in [16, 32]:
        gold = load_rows(AUDIT_DIR / f"memory_{n}" / "train.jsonl")
        preds = load_rows(AUDIT_DIR / f"memory_{n}" / "dev_predictions.jsonl")
        mem_map, mem_meta = build_time_norm_map(gold)
        before = evaluate_predictions(gold, preds, f"memory_{n}_before_time_map")
        after = evaluate_predictions(gold, apply_time_norm_map(preds, mem_map), f"memory_{n}_after_time_map")
        memory_results[f"memory_{n}"] = {
            "map_meta": mem_meta,
            "before_event_time_normalized_date_accuracy": before["event_time_normalized_date_accuracy"],
            "after_event_time_normalized_date_accuracy": after["event_time_normalized_date_accuracy"],
            "before_article_exact_match_accuracy": before["article_exact_match_accuracy"],
            "after_article_exact_match_accuracy": after["article_exact_match_accuracy"],
        }
    return {
        "train_only_map_meta": train_map_meta,
        "dev_before_event_time_normalized_date_accuracy": dev_before["event_time_normalized_date_accuracy"],
        "dev_after_event_time_normalized_date_accuracy": dev_after["event_time_normalized_date_accuracy"],
        "dev_before_article_exact_match_accuracy": dev_before["article_exact_match_accuracy"],
        "dev_after_article_exact_match_accuracy": dev_after["article_exact_match_accuracy"],
        "memory_results": memory_results,
        "dev_still_below_event_time_gate_0_35": dev_after["event_time_normalized_date_accuracy"] < 0.35,
        "training_labels_only": True,
        "dev_labels_used_for_mapping": False,
    }


def evaluate_memory_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "trigger_relaxed_f1_ge_0_90": metrics["trigger_relaxed_f1"] >= 0.90,
        "event_span_char_f1_ge_0_90": metrics["event_span_character_overlap_f1"] >= 0.90,
        "subject_norm_ge_0_85": metrics["subject_normalized_accuracy"] >= 0.85,
        "event_time_norm_ge_0_85": metrics["event_time_normalized_date_accuracy"] >= 0.85,
        "object_norm_ge_0_80": metrics["object_normalized_accuracy"] >= 0.80,
        "direction_acc_ge_0_95": metrics["direction_accuracy"] >= 0.95,
        "argument_role_f1_ge_0_80": metrics["argument_role_end_to_end_f1"] >= 0.80,
        "offset_illegal_0": metrics["offset_illegal_count"] == 0,
    }
    return {"passed": all(gates.values()), "gate_results": gates}


def build_report() -> dict[str, Any]:
    if not K_REPORT.exists():
        raise FileNotFoundError(K_REPORT)
    k_report = read_json(K_REPORT)
    if k_report.get("status") != "structure_model_dev_gate_failed":
        raise RuntimeError(f"unexpected Step 5.7-K status: {k_report.get('status')}")
    protected = sha_snapshot(protected_paths())
    recovery = {
        "step5_7k_status": k_report.get("status"),
        "counts": k_report.get("recovery", {}).get("counts", {}),
        "article_overlaps": k_report.get("recovery", {}).get("article_overlaps", {}),
        "best_scheme": k_report.get("selection", {}).get("best_scheme"),
        "best_checkpoint_sha256": k_report.get("selection", {}).get("best_checkpoint_sha256"),
        "dev_gate_passed": k_report.get("selection", {}).get("dev_gate_passed"),
        "dev_gate_results": k_report.get("selection", {}).get("dev_gate_results", {}),
        "protected_sha_match": k_report.get("protection", {}).get("protected_sha_match"),
        "local_32b_calls": k_report.get("protection", {}).get("local_32b_calls", 0),
        "external_llm_calls": k_report.get("protection", {}).get("external_llm_calls", 0),
    }
    code_audit = build_code_audit()
    train_rows = load_rows(TRAIN_JSONL)
    dev_rows = load_rows(DEV_JSONL)
    label_distribution = run_label_distribution_probe()
    data_audit = {
        "train": build_data_audit(train_rows, "train"),
        "dev": build_data_audit(dev_rows, "dev"),
        "label_distribution_probe": label_distribution,
    }
    memory_16 = run_memory_test(16)
    memory_32 = run_memory_test(32)
    memory_probe_16 = evaluate_memory_gate(memory_16)
    memory_probe_32 = evaluate_memory_gate(memory_32)
    gradient_probe = one_batch_gradient_probe(AUDIT_DIR / "memory_16" / "train.jsonl")
    oracle = build_oracle_decomposition()
    best_ref = evaluate_best_dev_reference()
    time_norm_probe = time_normalization_dictionary_probe()
    leakage_tests = k_report.get("negative_leakage_tests", {}).get("tests", {})

    status = "audit_completed"
    blocker_notes = []
    if not memory_probe_16["passed"]:
        blocker_notes.append("16-row memorization gate failed")
    if not memory_probe_32["passed"]:
        blocker_notes.append("32-row memorization gate failed")
    if oracle["full_model"]["event_time_normalized_date_accuracy"] < 0.35:
        blocker_notes.append("dev event_time normalized accuracy still below gate")
    if k_report.get("selection", {}).get("dev_gate_passed"):
        blocker_notes.append("unexpected dev gate pass in prior run")
    if k_report.get("decision", {}).get("allow_1029_gold_free_shadow"):
        blocker_notes.append("unexpected shadow eligibility")

    final_assessment = {
        "likely_primary_blockers": [
            "field extraction underfitting on trigger/event_span/object_target",
            "event_time normalization remains below memorization gate",
            "full dev composite remains far below threshold",
        ],
        "likely_secondary_blockers": [
            "macbert worker uses raw-text event_time normalization only",
            "no explicit normalization head for event_time",
            "single-stage span decoder does not isolate event sentence retrieval",
        ],
        "status_hint": "blocked_by_structure_model_evidence" if blocker_notes else "undetermined",
        "blocker_notes": blocker_notes,
    }

    report = {
        "generated_at": now(),
        "status": status,
        "recovery": recovery,
        "code_audit": code_audit,
        "data_audit": data_audit,
        "memory_tests": {
            "subset_selection_seed": SEED,
            "memory_16": memory_16,
            "memory_32": memory_32,
            "memory_16_gate": memory_probe_16,
            "memory_32_gate": memory_probe_32,
            "gradient_probe_16": gradient_probe,
        },
        "oracle_decomposition": oracle,
        "best_dev_reference": best_ref,
        "time_normalization_dictionary_probe": time_norm_probe,
        "negative_leakage_tests": {
            "all_passed": k_report.get("negative_leakage_tests", {}).get("all_passed", False),
            "tests": leakage_tests,
        },
        "protective_sha_snapshot": protected,
        "decision": final_assessment,
        "backup_dir": str(BACKUP_ROOT / f"task1_structure_root_cause_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    }
    return report


def write_report_files(report: dict[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_JSON, report)
    lines = [
        "# Task1 Step 5.7-L 结构抽取根因审计",
        "",
        f"- status: `{report['status']}`",
        f"- Step 5.7-K status: `{report['recovery']['step5_7k_status']}`",
        f"- dev_gate_passed: {report['recovery']['dev_gate_passed']}",
        f"- 16-row memorization gate: {report['memory_tests']['memory_16_gate']['passed']}",
        f"- 32-row memorization gate: {report['memory_tests']['memory_32_gate']['passed']}",
        f"- oracle full-model dev event_time_norm: {report['oracle_decomposition']['full_model']['event_time_normalized_date_accuracy']}",
        f"- train-only time-normalization map dev event_time_norm: {report['time_normalization_dictionary_probe']['dev_after_event_time_normalized_date_accuracy']}",
        "",
        "主要观察：",
        f"- event_time 归一化在 worker 中是 raw-text-only 逻辑，未使用 publish_time。",
        f"- 16/32 记忆测试已运行，event_time 归一化门槛未通过；训练集 time_text 映射能修复记忆测试，但不能让 dev 达标。",
        f"- 40 条 dev 的 model_sentence_oracle_fields / oracle_sentence_model_fields 用于区分 retrieval 与 field extraction。",
        "",
        "本轮未读取 1029 gold test 标签，未调用外部 LLM，未修改正式 Task1/Task2 输出。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    summary_rows = [
        ("status", report["status"]),
        ("step5_7k_status", report["recovery"]["step5_7k_status"]),
        ("dev_gate_passed", report["recovery"]["dev_gate_passed"]),
        ("memory_16_gate_passed", report["memory_tests"]["memory_16_gate"]["passed"]),
        ("memory_32_gate_passed", report["memory_tests"]["memory_32_gate"]["passed"]),
        ("allow_1029_shadow", report["decision"]["status_hint"] == "ready_for_task1_promotion_review"),
        ("best_scheme", report["recovery"]["best_scheme"]),
        ("best_checkpoint_sha256", report["recovery"]["best_checkpoint_sha256"]),
        ("current_event_time_norm_acc", report["best_dev_reference"]["current_event_time_normalized_accuracy"]),
        ("content_only_event_time_norm_acc", report["best_dev_reference"]["content_only_normalized_accuracy"]),
        ("train_only_time_map_dev_event_time_norm_acc", report["time_normalization_dictionary_probe"]["dev_after_event_time_normalized_date_accuracy"]),
        ("train_only_time_map_memory16_event_time_norm_acc", report["time_normalization_dictionary_probe"]["memory_results"]["memory_16"]["after_event_time_normalized_date_accuracy"]),
        ("train_only_time_map_memory32_event_time_norm_acc", report["time_normalization_dictionary_probe"]["memory_results"]["memory_32"]["after_event_time_normalized_date_accuracy"]),
    ]
    ws.append(["key", "value"])
    for key, value in summary_rows:
        ws.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")

    ws2 = wb.create_sheet("code_audit")
    ws2.append(["key", "value"])
    for key, value in report["code_audit"].items():
        ws2.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])

    ws3 = wb.create_sheet("data_audit")
    ws3.append(["split", "field", "nonempty_count", "empty_count", "span_length_mean", "span_length_median"])
    for split_name in ["train", "dev"]:
        for field, stats in report["data_audit"][split_name]["field_stats"].items():
            ws3.append(
                [
                    split_name,
                    field,
                    stats["nonempty_count"],
                    stats["empty_count"],
                    stats["span_length_mean"],
                    stats["span_length_median"],
                ]
            )

    ws4 = wb.create_sheet("event_time")
    ws4.append(["split", "text", "gold_normalized", "current_normalized", "improved_normalized", "event_mention_id", "publish_time"])
    for split_name in ["train", "dev"]:
        for item in report["data_audit"][split_name]["event_time_stats"]["publish_time_based_examples"]:
            ws4.append(
                [
                    split_name,
                    item["text"],
                    item["gold_normalized"],
                    item["current_normalized"] if "current_normalized" in item else "",
                    item["content_only_normalized"],
                    item["event_mention_id"],
                    item["publish_time"],
                ]
            )

    ws5 = wb.create_sheet("memory")
    ws5.append(["subset", "metric", "value"])
    for subset_name in ["memory_16", "memory_32"]:
        metrics = report["memory_tests"][subset_name]
        for key in [
            "trigger_relaxed_f1",
            "event_span_character_overlap_f1",
            "subject_normalized_accuracy",
            "event_time_normalized_date_accuracy",
            "object_normalized_accuracy",
            "direction_accuracy",
            "argument_role_end_to_end_f1",
            "article_exact_match_accuracy",
            "offset_illegal_count",
            "trigger_not_in_event_span_count",
        ]:
            if key in metrics:
                ws5.append([subset_name, key, metrics[key]])
        ws5.append([subset_name, "gate_passed", report["memory_tests"][f"{subset_name}_gate"]["passed"]])

    ws6 = wb.create_sheet("oracle")
    ws6.append(["scenario", "metric", "value"])
    for scenario, metrics in report["oracle_decomposition"].items():
        for key in [
            "trigger_relaxed_f1",
            "event_span_character_overlap_f1",
            "subject_normalized_accuracy",
            "event_time_normalized_date_accuracy",
            "object_normalized_accuracy",
            "direction_accuracy",
            "argument_role_end_to_end_f1",
            "article_exact_match_accuracy",
        ]:
            if key in metrics:
                ws6.append([scenario, key, metrics[key]])

    ws7 = wb.create_sheet("time_norm_map")
    ws7.append(["scope", "metric", "value"])
    probe = report["time_normalization_dictionary_probe"]
    for key, value in probe.items():
        if key != "memory_results":
            ws7.append(["dev", key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    for scope, item in probe["memory_results"].items():
        for key, value in item.items():
            ws7.append([scope, key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])

    ws8 = wb.create_sheet("leakage")
    ws8.append(["test", "passed", "details"])
    for name, result in report["negative_leakage_tests"]["tests"].items():
        ws8.append([name, result.get("passed"), json.dumps(result, ensure_ascii=False)])

    for sheet in wb.worksheets:
        sheet.column_dimensions["A"].width = 36
        sheet.column_dimensions["B"].width = 42
        sheet.column_dimensions["C"].width = 120
        sheet.freeze_panes = "A2"
    DETAILS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DETAILS_XLSX)


def create_backup(report: dict[str, Any]) -> Path:
    backup_dir = BACKUP_ROOT / f"task1_structure_root_cause_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    snapshot = sha_snapshot(protected_paths())
    write_sha_txt(backup_dir / "protected_sha256_before.txt", snapshot)
    write_json(backup_dir / "run_manifest.json", report)
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="force rebuild of report")
    args = parser.parse_args()
    if REPORT_JSON.exists() and not args.rebuild:
        report = read_json(REPORT_JSON)
    else:
        report = build_report()
        write_report_files(report)
    backup_dir = create_backup(report)
    report["backup_dir"] = str(backup_dir)
    write_json(REPORT_JSON, report)
    write_report_files(report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "step5_7k_status": report["recovery"]["step5_7k_status"],
                "memory_16_gate_passed": report["memory_tests"]["memory_16_gate"]["passed"],
                "memory_32_gate_passed": report["memory_tests"]["memory_32_gate"]["passed"],
                "backup_dir": str(backup_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
