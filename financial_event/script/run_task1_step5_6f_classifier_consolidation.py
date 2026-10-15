#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.6-F classifier consolidation.

This step does not train models, does not call LLMs, and does not read the
1029-row gold-test labels. It consolidates existing Candidate-v2 and classifier
scores under the Step 5.6-D dev/holdout split, freezes one dev-selected
baseline, and only then optionally runs the 80-row confirmed diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_5_candidate_v2 import (  # noqa: E402
    protected_file_list,
    read_jsonl,
    rel,
    sha256_file,
    write_json,
    write_jsonl,
    write_xlsx,
)
from scripts.run_task1_step5_6c_dev_calibration import select_rows  # noqa: E402
from scripts.train_task1_step5_6d_classifier import build_step56d_input  # noqa: E402
from src.data.event_match_text import build_input_text  # noqa: E402
from src.evaluation.task1_task_metrics import candidate_recall_at_k, evaluate_event_predictions  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/task1_step5_6f_classifier_consolidation"
REPORT_JSON = ROOT / "reports/task1_step5_6f_classifier_consolidation_report.json"
REPORT_MD = ROOT / "reports/task1_step5_6f_classifier_consolidation_report.md"
REPORT_XLSX = ROOT / "reports/task1_step5_6f_classifier_consolidation_details.xlsx"

STEP56C_DIR = ROOT / "outputs/task1_step5_6c_classifier_calibration"
STEP56D_DIR = ROOT / "outputs/task1_step5_6d_classifier_retraining"
STEP56E_DIR = ROOT / "outputs/task1_step5_6e_32b_teacher_pilot"
STEP56B_DIR = ROOT / "outputs/task1_step5_6b_latest_classifier_shadow"
MODEL_EPOCH2_DIR = ROOT / "outputs/task1_limited_training_round1/model_epoch_2"
EXPERIMENT_C_DIR = STEP56D_DIR / "experiment_C/best_model"
CONFIRMED_80 = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"

FUSION_WEIGHTS = [
    (0.50, 0.25, 0.25),
    (0.30, 0.35, 0.35),
    (0.20, 0.40, 0.40),
    (0.40, 0.40, 0.20),
    (0.40, 0.20, 0.40),
]
CANDIDATE_SCORE_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
FUSION_THRESHOLDS = [0.0, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80]
MARGIN_RULES = [
    {"margin": 0.00, "fallback": "higher_top_probability_then_candidate_top1"},
    {"margin": 0.05, "fallback": "higher_top_probability_then_candidate_top1"},
    {"margin": 0.10, "fallback": "higher_top_probability_then_candidate_top1"},
    {"margin": 0.20, "fallback": "higher_top_probability_then_candidate_top1"},
]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_text(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def event_names() -> list[str]:
    return sorted({str(row["event_name"]) for row in read_jsonl(ROOT / "data/processed/event_schema.jsonl") if row.get("event_name")})


def ids_from_articles(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["article_id"]) for row in rows]


def ensure_required_inputs() -> dict[str, Any]:
    required = [
        STEP56C_DIR / "frozen_selection_manifest.json",
        STEP56C_DIR / "evaluation_80.json",
        STEP56D_DIR / "frozen_selection_manifest.json",
        STEP56D_DIR / "dev_articles.jsonl",
        STEP56D_DIR / "dev_gold_pairs.jsonl",
        STEP56D_DIR / "dev_candidate_v2_top20.jsonl",
        STEP56D_DIR / "holdout_articles.jsonl",
        STEP56D_DIR / "holdout_gold_pairs.jsonl",
        STEP56D_DIR / "holdout_candidate_v2_top20.jsonl",
        STEP56D_DIR / "experiment_C/dev_scores_best.jsonl",
        STEP56D_DIR / "holdout_pair_predictions.jsonl",
        STEP56E_DIR / "evaluation.json",
        STEP56E_DIR / "teacher_predictions.jsonl",
        STEP56B_DIR / "gold_free_inputs.jsonl",
        STEP56B_DIR / "candidates_top20.jsonl",
        STEP56B_DIR / "pair_predictions.jsonl",
        MODEL_EPOCH2_DIR / "model.safetensors",
        EXPERIMENT_C_DIR / "model.safetensors",
        ROOT / "src/routing/task1_candidate_v2.py",
        ROOT / "configs/task1_candidate_v2.yaml",
        CONFIRMED_80,
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required Step 5.6-F inputs missing: {missing}")
    teacher_report = read_json(STEP56E_DIR / "evaluation.json")
    return {
        "missing": missing,
        "step5_6e_status": teacher_report.get("status"),
        "step5_6e_teacher_micro_f1": teacher_report.get("method_evaluations", {}).get("DianJin_R1_32B_teacher", {}).get("metrics", {}).get("micro_f1"),
        "step5_6e_expanded_to_5000": False,
    }


def process_check() -> dict[str, Any]:
    proc = subprocess.run(
        ["bash", "-lc", "ps -eo pid,ppid,stat,etime,cmd | rg -i 'vllm|DianJin|train_task1|run_task1|evaluate_task1|python.*task1' || true\nss -ltnp | rg ':8000' || true"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    current = os.getpid()
    suspicious = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 4)
        if parts and parts[0].isdigit() and int(parts[0]) == current:
            continue
        lower = line.lower()
        if "run_task1_step5_6f_classifier_consolidation.py" in lower:
            continue
        if any(token in lower for token in ("vllm", "dianjin", "train_task1", "run_task1", "evaluate_task1", "python")):
            if any(token in lower for token in (" rg ", " ps ", " ss ", "bash -lc")):
                continue
            suspicious.append(line.strip())
    return {"status": "ok" if not suspicious else "suspicious_processes_found", "suspicious": suspicious, "raw": proc.stdout.splitlines()}


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([item for item in path.rglob("*") if item.is_file()])
    return []


def protected_and_model_files() -> list[Path]:
    files = list(protected_file_list())
    for path in [
        MODEL_EPOCH2_DIR,
        EXPERIMENT_C_DIR,
        ROOT / "outputs/event_match_baseline/best_model",
        ROOT / "outputs/event_match_hard_negative/best_model",
        STEP56C_DIR / "frozen_selection_manifest.json",
        STEP56D_DIR / "frozen_selection_manifest.json",
        STEP56E_DIR / "evaluation.json",
        ROOT / "src/routing/task1_candidate_v2.py",
        ROOT / "configs/task1_candidate_v2.yaml",
        CONFIRMED_80,
    ]:
        files.extend(files_under(path))
    return sorted(set(files), key=lambda p: rel(p))


def write_sha_snapshot(path: Path, files: list[Path]) -> dict[str, str]:
    rows = {rel(item): sha256_file(item) for item in files if item.exists()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(rows.items())), encoding="utf-8")
    return rows


def load_sha_snapshot(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        out[name] = digest
    return out


def init_safety(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = output_dir / "recovery_safety_manifest.json"
    if existing.exists():
        manifest = read_json(existing)
        before = load_sha_snapshot(output_dir / "protected_and_model_sha256_before.txt")
        return {**manifest, "reused_existing_safety_manifest": True, "before_sha_count": len(before)}
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_6f_classifier_consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    files = protected_and_model_files()
    before = write_sha_snapshot(output_dir / "protected_and_model_sha256_before.txt", files)
    shutil.copy2(output_dir / "protected_and_model_sha256_before.txt", backup_dir / "protected_and_model_sha256_before.txt")
    manifest = {
        "generated_at": now(),
        "backup_dir": str(backup_dir),
        "output_dir": rel(output_dir),
        "before_sha_count": len(before),
        "process_check": process_check(),
        "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
        "gold_test_labels_read": False,
    }
    write_json_text(output_dir / "recovery_safety_manifest.json", manifest)
    write_json_text(backup_dir / "recovery_safety_manifest.json", manifest)
    return manifest


def article_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["article_id"]): row for row in rows}


def enrich_limited(row: dict[str, Any], articles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = dict(row)
    article = articles.get(str(row["article_id"]), {})
    content = str(article.get("content", item.get("content", "")))
    if len(content) > 2600:
        content = content[:1900] + "\n[正文中间省略]\n" + content[-600:]
    item["title"] = article.get("title", item.get("title", ""))
    item["content"] = content
    return item


def score_softmax_model(
    model_dir: Path,
    candidate_rows: list[dict[str, Any]],
    articles: dict[str, dict[str, Any]],
    output_path: Path,
    source: str,
    batch_size: int,
) -> dict[str, Any]:
    if output_path.exists():
        return {"status": "reused", "path": rel(output_path), "record_count": len(read_jsonl(output_path))}
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    class PairDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = enrich_limited(self.rows[idx], articles)
            return {"text": build_input_text(row), "row": self.rows[idx]}

    def collate(batch: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        enc = tokenizer([item["text"] for item in batch], truncation=True, max_length=512, padding=True, return_tensors="pt")
        return enc, [item["row"] for item in batch]

    start = time.time()
    peak = 0
    out = []
    loader = DataLoader(PairDataset(candidate_rows), batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for enc, batch_rows in loader:
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits.detach().cpu()
            probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
            for row, logit_row, prob in zip(batch_rows, logits.tolist(), probs):
                item = dict(row)
                item["model_source"] = source
                item["raw_score_negative_logit"] = float(logit_row[0])
                item["raw_score_positive_logit"] = float(logit_row[1])
                item["probability_positive"] = float(prob)
                out.append(item)
            if device.type == "cuda":
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
    write_jsonl(output_path, out)
    return {
        "status": "completed",
        "path": rel(output_path),
        "record_count": len(out),
        "runtime_seconds": round(time.time() - start, 3),
        "device": str(device),
        "peak_gpu_memory_bytes": peak,
    }


def score_sigmoid_model(
    model_dir: Path,
    candidate_rows: list[dict[str, Any]],
    articles: dict[str, dict[str, Any]],
    output_path: Path,
    source: str,
    batch_size: int,
) -> dict[str, Any]:
    if output_path.exists():
        return {"status": "reused", "path": rel(output_path), "record_count": len(read_jsonl(output_path))}
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    start = time.time()
    peak = 0
    out = []
    with torch.no_grad():
        for offset in range(0, len(candidate_rows), batch_size):
            batch = candidate_rows[offset : offset + batch_size]
            texts = []
            for row in batch:
                article = articles.get(str(row["article_id"]), {})
                item = dict(row)
                item["title"] = article.get("title", item.get("title", ""))
                item["content"] = article.get("content", item.get("content", ""))
                texts.append(build_step56d_input(item))
            enc = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits.view(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            for row, logit, prob in zip(batch, logits.detach().cpu().tolist(), probs):
                item = dict(row)
                item["model_source"] = source
                item["raw_score_logit"] = float(logit)
                item["probability_positive"] = float(prob)
                out.append(item)
            if device.type == "cuda":
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
    write_jsonl(output_path, out)
    return {
        "status": "completed",
        "path": rel(output_path),
        "record_count": len(out),
        "runtime_seconds": round(time.time() - start, 3),
        "device": str(device),
        "peak_gpu_memory_bytes": peak,
    }


def group_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["article_id"])].append(row)
    return grouped


def prediction_sets(rows: list[dict[str, Any]], ids: list[str]) -> dict[str, set[str]]:
    out = {aid: set() for aid in ids}
    for row in rows:
        out.setdefault(str(row["article_id"]), set()).add(str(row["event_name"]))
    return out


def top1_rows(candidate_rows: list[dict[str, Any]], ids: list[str], min_candidate_score: float | None = None) -> list[dict[str, Any]]:
    grouped = group_rows(candidate_rows)
    out = []
    for aid in ids:
        rows = sorted(
            grouped.get(aid, []),
            key=lambda row: (int(row.get("candidate_rank", 10**9) or 10**9), -float(row.get("candidate_score", 0.0) or 0.0), str(row.get("event_name", ""))),
        )
        if not rows:
            continue
        top = rows[0]
        if min_candidate_score is not None and float(top.get("candidate_score", 0.0) or 0.0) < min_candidate_score:
            continue
        item = dict(top)
        item["probability_positive"] = 1.0
        item["selection_score"] = float(item.get("candidate_score", 0.0) or 0.0)
        item["decode_rule"] = "candidate_v2_top1" if min_candidate_score is None else "candidate_v2_top1_min_candidate_score"
        item["decision"] = True
        out.append(item)
    return out


def normalize_rank_score(rank: Any) -> float:
    try:
        value = int(rank)
    except Exception:
        value = 20
    value = max(1, min(20, value))
    return 1.0 - (value - 1) / 19.0


def score_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["article_id"]), str(row["event_name"])): row for row in rows}


def event_set_rows(ids: list[str], event_sets: dict[str, set[str]], score_source: dict[tuple[str, str], dict[str, Any]] | None, rule: str) -> list[dict[str, Any]]:
    out = []
    for aid in ids:
        for event_name in sorted(event_sets.get(aid, set())):
            base = dict((score_source or {}).get((aid, event_name), {"article_id": aid, "event_name": event_name}))
            base["article_id"] = aid
            base["event_name"] = event_name
            base["probability_positive"] = float(base.get("probability_positive", base.get("selection_score", 1.0)) or 1.0)
            base["selection_score"] = float(base.get("selection_score", base.get("probability_positive", 1.0)) or 1.0)
            base["decision"] = True
            base["decode_rule"] = rule
            out.append(base)
    return out


def intersection_rows(ids: list[str], rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sets_a = prediction_sets(rows_a, ids)
    sets_b = prediction_sets(rows_b, ids)
    idx = score_index(rows_a + rows_b)
    return event_set_rows(ids, {aid: sets_a[aid] & sets_b[aid] for aid in ids}, idx, "model_epoch2_experiment_C_intersection")


def union_rows(ids: list[str], rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sets_a = prediction_sets(rows_a, ids)
    sets_b = prediction_sets(rows_b, ids)
    idx = score_index(rows_a + rows_b)
    return event_set_rows(ids, {aid: sets_a[aid] | sets_b[aid] for aid in ids}, idx, "model_epoch2_experiment_C_union")


def majority_vote_rows(ids: list[str], candidate_top1: list[dict[str, Any]], rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = [prediction_sets(candidate_top1, ids), prediction_sets(rows_a, ids), prediction_sets(rows_b, ids)]
    idx = score_index(candidate_top1 + rows_a + rows_b)
    by_article: dict[str, set[str]] = {}
    for aid in ids:
        counts = Counter(event for source in sources for event in source[aid])
        by_article[aid] = {event for event, count in counts.items() if count >= 2}
    return event_set_rows(ids, by_article, idx, "three_way_majority_vote")


def top_prob(rows: list[dict[str, Any]], aid: str) -> tuple[str, float]:
    article_rows = [row for row in rows if str(row["article_id"]) == aid]
    if not article_rows:
        return "", -1.0
    top = max(article_rows, key=lambda row: (float(row.get("probability_positive", 0.0) or 0.0), -int(row.get("candidate_rank", 10**9) or 10**9)))
    return str(top.get("event_name") or ""), float(top.get("probability_positive", 0.0) or 0.0)


def agreement_fallback_rows(
    ids: list[str],
    candidate_top1: list[dict[str, Any]],
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    all_scores_a: list[dict[str, Any]],
    all_scores_b: list[dict[str, Any]],
    margin: float,
) -> list[dict[str, Any]]:
    sets_a = prediction_sets(rows_a, ids)
    sets_b = prediction_sets(rows_b, ids)
    cand = prediction_sets(candidate_top1, ids)
    idx = score_index(candidate_top1 + rows_a + rows_b + all_scores_a + all_scores_b)
    out_sets: dict[str, set[str]] = {}
    grouped_a = group_rows(all_scores_a)
    grouped_b = group_rows(all_scores_b)
    for aid in ids:
        if sets_a[aid] == sets_b[aid] and sets_a[aid]:
            out_sets[aid] = set(sets_a[aid])
        elif not sets_a[aid] and not sets_b[aid]:
            out_sets[aid] = set(cand[aid])
        else:
            _, score_a = top_prob(grouped_a.get(aid, []), aid)
            _, score_b = top_prob(grouped_b.get(aid, []), aid)
            if abs(score_a - score_b) >= margin:
                out_sets[aid] = set(sets_a[aid] if score_a > score_b else sets_b[aid])
                if not out_sets[aid]:
                    out_sets[aid] = set(cand[aid])
            else:
                out_sets[aid] = set(cand[aid])
    return event_set_rows(ids, out_sets, idx, f"agreement_first_candidate_fallback_margin_{margin:.2f}")


def fusion_rows(
    ids: list[str],
    candidate_rows: list[dict[str, Any]],
    scores_a: list[dict[str, Any]],
    scores_b: list[dict[str, Any]],
    weights: tuple[float, float, float],
    min_score: float,
) -> list[dict[str, Any]]:
    w_rank, w_a, w_b = weights
    idx_a = score_index(scores_a)
    idx_b = score_index(scores_b)
    fused = []
    for row in candidate_rows:
        key = (str(row["article_id"]), str(row["event_name"]))
        a = float(idx_a.get(key, {}).get("probability_positive", 0.0) or 0.0)
        b = float(idx_b.get(key, {}).get("probability_positive", 0.0) or 0.0)
        rank_score = normalize_rank_score(row.get("candidate_rank"))
        item = dict(row)
        item["rank_score_norm"] = rank_score
        item["model_epoch2_probability"] = a
        item["experiment_C_probability"] = b
        item["fusion_score"] = w_rank * rank_score + w_a * a + w_b * b
        item["probability_positive"] = item["fusion_score"]
        fused.append(item)
    return select_rows(
        fused,
        ids,
        "fusion_top1_min_score",
        {"score_key": "fusion_score", "min_score": min_score, "weights": weights},
    )


def confirmed_80_gold_pairs() -> list[dict[str, Any]]:
    pairs = []
    for row in read_jsonl(CONFIRMED_80):
        names = str(row.get("gold_event_names_for_context_after_classification_eval") or "").split("|")
        for name in sorted({item.strip() for item in names if item.strip()}):
            pairs.append({"article_id": str(row["article_id"]), "event_name": name})
    return pairs


def add_metric_extras(metrics: dict[str, Any]) -> dict[str, Any]:
    counts = [int(row["pred_count"]) for row in metrics.get("article_rows", [])]
    metrics["avg_predicted_event_count"] = sum(counts) / len(counts) if counts else 0.0
    metrics["complete_miss_count"] = sum(1 for row in metrics["article_rows"] if int(row["gold_count"]) > 0 and int(row["pred_count"]) == 0)
    metrics["overpredicted_article_count"] = sum(1 for row in metrics["article_rows"] if int(row["fp"]) > 0)
    metrics["false_positive_rate"] = metrics["fp"] / len(metrics["article_rows"]) if metrics.get("article_rows") else 0.0
    metrics["pred_article_count_0"] = sum(1 for count in counts if count == 0)
    metrics["pred_article_count_1"] = sum(1 for count in counts if count == 1)
    metrics["pred_article_count_2"] = sum(1 for count in counts if count == 2)
    metrics["pred_article_count_3plus"] = sum(1 for count in counts if count >= 3)
    return metrics


def error_breakdown(ids: list[str], gold_pairs: list[dict[str, Any]], candidates: list[dict[str, Any]], selected: list[dict[str, Any]], fixed: list[str]) -> dict[str, Any]:
    gold = defaultdict(set)
    cand = defaultdict(set)
    pred = defaultdict(set)
    for row in gold_pairs:
        gold[str(row["article_id"])].add(str(row["event_name"]))
    for row in candidates:
        cand[str(row["article_id"])].add(str(row.get("event_name") or ""))
    for row in selected:
        pred[str(row["article_id"])].add(str(row.get("event_name") or ""))
    fixed_set = set(fixed)
    buckets = {
        "candidate_missing": [],
        "candidate_present_classifier_false_negative": [],
        "false_positive": [],
        "label_mapping_error": [],
        "join_error": [],
    }
    for aid in ids:
        if aid not in cand:
            buckets["join_error"].append({"article_id": aid, "reason": "missing_candidate_rows"})
        for name in gold[aid]:
            if name not in fixed_set:
                buckets["label_mapping_error"].append({"article_id": aid, "event_name": name, "reason": "gold_not_in_fixed64"})
            elif name not in cand[aid]:
                buckets["candidate_missing"].append({"article_id": aid, "event_name": name})
            elif name not in pred[aid]:
                buckets["candidate_present_classifier_false_negative"].append({"article_id": aid, "event_name": name})
        for name in pred[aid] - gold[aid]:
            if name not in fixed_set:
                buckets["label_mapping_error"].append({"article_id": aid, "event_name": name, "reason": "prediction_not_in_fixed64"})
            else:
                buckets["false_positive"].append({"article_id": aid, "event_name": name})
    return {key: {"count": len(rows), "rows": rows} for key, rows in buckets.items()}


def evaluate_selected(strategy_id: str, split_name: str, ids: list[str], gold_pairs: list[dict[str, Any]], selected: list[dict[str, Any]], candidates: list[dict[str, Any]], fixed: list[str], params: dict[str, Any]) -> dict[str, Any]:
    metrics = add_metric_extras(evaluate_event_predictions(ids, gold_pairs, selected, fixed, threshold=0.0))
    breakdown = error_breakdown(ids, gold_pairs, candidates, selected, fixed)
    summary = {
        "strategy_id": strategy_id,
        "split": split_name,
        "strategy_group": params.get("strategy_group", strategy_id),
        "params": params,
        "micro_precision": metrics["micro_precision"],
        "micro_recall": metrics["micro_recall"],
        "micro_f1": metrics["micro_f1"],
        "macro_f1_fixed_64": metrics["macro_f1_fixed_64"],
        "macro_f1_observed": metrics["macro_f1_observed"],
        "weighted_f1": metrics["weighted_f1"],
        "article_exact_accuracy": metrics["article_exact_accuracy"],
        "at_least_one_correct_article_ratio": metrics["at_least_one_correct_article_ratio"],
        "avg_predicted_event_count": metrics["avg_predicted_event_count"],
        "complete_miss_count": metrics["complete_miss_count"],
        "overpredicted_article_count": metrics["overpredicted_article_count"],
        "candidate_present_fn": breakdown["candidate_present_classifier_false_negative"]["count"],
        "candidate_missing": breakdown["candidate_missing"]["count"],
        "false_positive": breakdown["false_positive"]["count"],
        "false_positive_rate": metrics["false_positive_rate"],
        "label_mapping_error": breakdown["label_mapping_error"]["count"],
        "join_error": breakdown["join_error"]["count"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "predicted_event_count": metrics["predicted_event_count"],
        "gold_event_count": metrics["gold_event_count"],
        "pred_article_count_0": metrics["pred_article_count_0"],
        "pred_article_count_1": metrics["pred_article_count_1"],
        "pred_article_count_2": metrics["pred_article_count_2"],
        "pred_article_count_3plus": metrics["pred_article_count_3plus"],
        "selection_score": 0.45 * metrics["micro_f1"] + 0.25 * metrics["macro_f1_fixed_64"] + 0.20 * metrics["article_exact_accuracy"] - 0.10 * metrics["false_positive_rate"],
    }
    return {"summary": summary, "metrics": metrics, "error_breakdown": breakdown, "selected_rows": selected}


def load_strategy_params() -> tuple[dict[str, Any], dict[str, Any]]:
    c_manifest = read_json(STEP56C_DIR / "frozen_selection_manifest.json")
    d_manifest = read_json(STEP56D_DIR / "frozen_selection_manifest.json")
    c = c_manifest["selected_model_strategy"]
    d = d_manifest["selected"]
    c["strategy_params_dict"] = json.loads(c["strategy_params"]) if isinstance(c.get("strategy_params"), str) else c.get("strategy_params", {})
    d["strategy_params_dict"] = json.loads(d["strategy_params"]) if isinstance(d.get("strategy_params"), str) else d.get("strategy_params", {})
    return c, d


def build_strategy_outputs(
    split_name: str,
    ids: list[str],
    candidates: list[dict[str, Any]],
    model_epoch2_scores: list[dict[str, Any]],
    experiment_c_scores: list[dict[str, Any]],
    c_params: dict[str, Any],
    d_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    model_epoch2_selected = select_rows(model_epoch2_scores, ids, c_params["strategy"], c_params["strategy_params_dict"])
    experiment_c_selected = select_rows(experiment_c_scores, ids, d_params["strategy"], d_params["strategy_params_dict"])
    top1 = top1_rows(candidates, ids)
    outputs: dict[str, dict[str, Any]] = {
        "A_candidate_v2_top1": {
            "rows": top1,
            "params": {"strategy_group": "A", "rule": "candidate_v2_top1"},
        },
        "C_model_epoch2_frozen_per_class_threshold": {
            "rows": model_epoch2_selected,
            "params": {"strategy_group": "C", "rule": c_params["strategy"], "params": c_params["strategy_params_dict"]},
        },
        "D_experiment_C_frozen_decoder": {
            "rows": experiment_c_selected,
            "params": {"strategy_group": "D", "rule": d_params["strategy"], "params": d_params["strategy_params_dict"]},
        },
        "E_model_intersection": {
            "rows": intersection_rows(ids, model_epoch2_selected, experiment_c_selected),
            "params": {"strategy_group": "E", "rule": "intersection"},
        },
        "F_model_union": {
            "rows": union_rows(ids, model_epoch2_selected, experiment_c_selected),
            "params": {"strategy_group": "F", "rule": "union"},
        },
        "G_three_way_majority": {
            "rows": majority_vote_rows(ids, top1, model_epoch2_selected, experiment_c_selected),
            "params": {"strategy_group": "G", "rule": "candidate_top1_model_epoch2_experiment_C_majority"},
        },
    }
    for threshold in CANDIDATE_SCORE_THRESHOLDS:
        outputs[f"B_candidate_top1_min_candidate_score_{threshold:g}"] = {
            "rows": top1_rows(candidates, ids, threshold),
            "params": {"strategy_group": "B", "rule": "candidate_v2_top1_min_candidate_score", "min_candidate_score": threshold},
        }
    for rule in MARGIN_RULES:
        margin = float(rule["margin"])
        outputs[f"H_agreement_fallback_margin_{margin:.2f}"] = {
            "rows": agreement_fallback_rows(ids, top1, model_epoch2_selected, experiment_c_selected, model_epoch2_scores, experiment_c_scores, margin),
            "params": {"strategy_group": "H", **rule},
        }
    for weights in FUSION_WEIGHTS:
        for threshold in FUSION_THRESHOLDS:
            outputs[f"I_score_fusion_{weights[0]:.2f}_{weights[1]:.2f}_{weights[2]:.2f}_min_{threshold:.2f}"] = {
                "rows": fusion_rows(ids, candidates, model_epoch2_scores, experiment_c_scores, weights, threshold),
                "params": {
                    "strategy_group": "I",
                    "rule": "linear_fusion_top1_min_score",
                    "weights": {"candidate_rank_score": weights[0], "model_epoch2_probability": weights[1], "experiment_C_probability": weights[2]},
                    "min_fusion_score": threshold,
                },
            }
    return outputs


def choose_dev_strategy(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in dev_rows if float(row["micro_recall"]) >= 0.55]
    if not eligible:
        return max(dev_rows, key=lambda row: row["selection_score"])
    return max(
        eligible,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["micro_f1"]),
            float(row["macro_f1_fixed_64"]),
            float(row["article_exact_accuracy"]),
            -float(row["avg_predicted_event_count"]),
            row["strategy_id"],
        ),
    )


def candidate_recall(ids: list[str], gold_pairs: list[dict[str, Any]], candidates: list[dict[str, Any]], fixed: list[str]) -> dict[str, Any]:
    raw = candidate_recall_at_k(ids, gold_pairs, candidates, fixed, [1, 3, 5, 10, 20])
    return {
        "candidate_recall_at_1": raw.get("candidate_recall_at_1", 0.0),
        "candidate_recall_at_3": raw.get("candidate_recall_at_3", 0.0),
        "candidate_recall_at_5": raw.get("candidate_recall_at_5", 0.0),
        "candidate_recall_at_10": raw.get("candidate_recall_at_10", 0.0),
        "candidate_recall_at_20": raw.get("candidate_recall_at_20", 0.0),
        "candidate_missing_at_20": raw.get("candidate_missing_at_max_k", 0),
        "gold_event_count": raw.get("gold_event_count", 0),
        "class_rows": raw.get("class_rows", []),
    }


def leakage_summary() -> dict[str, Any]:
    split = read_json(STEP56D_DIR / "split_summary.json")
    return {
        "train_dev_overlap_count": split.get("article_id_overlap", {}).get("train_dev", 0),
        "train_holdout_overlap_count": split.get("article_id_overlap", {}).get("train_holdout", 0),
        "dev_holdout_overlap_count": split.get("article_id_overlap", {}).get("dev_holdout", 0),
        "duplicate_group_leakage_count": split.get("duplicate_group_leakage_count", 0),
        "confirmed_80_overlap_count": split.get("excluded_80_overlap_count", 0),
        "gold_free_test_overlap_count": split.get("excluded_gold_free_test_overlap_count", 0),
    }


def holdout_gate(metrics: dict[str, Any], recall: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "candidate_recall_at_20_ge_0_95": float(recall["candidate_recall_at_20"]) >= 0.95,
        "micro_f1_ge_0_50": float(metrics["micro_f1"]) >= 0.50,
        "recall_ge_0_55": float(metrics["micro_recall"]) >= 0.55,
        "article_exact_ge_0_40": float(metrics["article_exact_accuracy"]) >= 0.40,
        "illegal_category_zero": int(metrics.get("label_mapping_error", 0)) == 0,
        "join_error_zero": int(metrics.get("join_error", 0)) == 0,
        "no_split_or_gold_leakage": all(int(leakage.get(key, 0)) == 0 for key in leakage),
    }
    return {"passed": all(checks.values()), "checks": checks}


def article_set_delta(ids: list[str], gold_pairs: list[dict[str, Any]], new_rows: list[dict[str, Any]], old_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = prediction_sets(gold_pairs, ids)
    new = prediction_sets(new_rows, ids)
    old = prediction_sets(old_rows, ids)
    fixed = []
    improved = worsened = unchanged = 0
    detail = []
    for aid in ids:
        new_exact = new[aid] == gold[aid]
        old_exact = old[aid] == gold[aid]
        if new_exact and not old_exact:
            improved += 1
            status = "improved_to_exact"
        elif old_exact and not new_exact:
            worsened += 1
            status = "worsened_from_exact"
        else:
            unchanged += 1
            status = "unchanged"
        fixed.append(status)
        detail.append({"article_id": aid, "gold": "|".join(sorted(gold[aid])), "old": "|".join(sorted(old[aid])), "new": "|".join(sorted(new[aid])), "status": status})
    return {"improved_article_count": improved, "worsened_article_count": worsened, "unchanged_article_count": unchanged, "rows": detail}


def write_predictions(path: Path, split: str, strategy_rows: dict[str, dict[str, Any]]) -> None:
    rows = []
    for strategy_id, payload in sorted(strategy_rows.items()):
        for row in payload["selected_rows"]:
            item = dict(row)
            item["split"] = split
            item["strategy_id"] = strategy_id
            item["strategy_group"] = payload["summary"]["strategy_group"]
            rows.append(item)
    write_jsonl(path, rows)


def write_sha_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append((rel(path), sha256_file(path)))
    (output_dir / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in rows), encoding="utf-8")


def final_protection(output_dir: Path, before_path: Path) -> dict[str, Any]:
    before = load_sha_snapshot(before_path)
    after = write_sha_snapshot(output_dir / "protected_and_model_sha256_after.txt", protected_and_model_files())
    return {
        "before_sha_file": rel(before_path),
        "after_sha_file": rel(output_dir / "protected_and_model_sha256_after.txt"),
        "sha_count_before": len(before),
        "sha_count_after": len(after),
        "sha_unchanged": before == after,
        "changed": [{"path": key, "before": before.get(key), "after": after.get(key)} for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)],
    }


def report_markdown(report: dict[str, Any]) -> str:
    selected = report["selection"]["dev_selected"]
    holdout = report.get("holdout", {})
    eval80 = report.get("evaluation_80", {})
    failed = [key for key, value in report.get("holdout_gate", {}).get("checks", {}).items() if not value]
    lines = [
        "# Task1 Step 5.6-F Classifier Consolidation",
        "",
        f"Generated at: {report['generated_at']}",
        f"Status: `{report['status']}`",
        "",
        "## Dev Selection",
        f"- selected strategy: `{selected['strategy_id']}`",
        f"- selection score: {selected['selection_score']:.6f}",
        f"- P/R/F1: {selected['micro_precision']:.6f} / {selected['micro_recall']:.6f} / {selected['micro_f1']:.6f}",
        f"- fixed64 macro-F1: {selected['macro_f1_fixed_64']:.6f}",
        f"- article exact: {selected['article_exact_accuracy']:.6f}",
        "",
        "## Holdout",
        f"- executed: {bool(holdout)}",
        f"- P/R/F1: {holdout.get('summary', {}).get('micro_precision', '')} / {holdout.get('summary', {}).get('micro_recall', '')} / {holdout.get('summary', {}).get('micro_f1', '')}",
        f"- article exact: {holdout.get('summary', {}).get('article_exact_accuracy', '')}",
        f"- candidate recall@20: {report.get('holdout_candidate_recall', {}).get('candidate_recall_at_20', '')}",
        f"- gate passed: {report.get('holdout_gate', {}).get('passed', False)}",
        f"- failed gate checks: {', '.join(failed) if failed else 'none'}",
        "",
        "## Confirmed 80 Diagnostic",
        f"- executed: {eval80.get('executed', False)}",
        f"- P/R/F1: {eval80.get('summary', {}).get('micro_precision', '')} / {eval80.get('summary', {}).get('micro_recall', '')} / {eval80.get('summary', {}).get('micro_f1', '')}",
        f"- article exact: {eval80.get('summary', {}).get('article_exact_accuracy', '')}",
        "",
        "## 32B Decision",
        "- DianJin-R1-32B classifier teacher is permanently disabled for this classification pipeline.",
        "- No 5000-row Silver expansion; no 32B Silver labels enter training.",
        "- Future 32B use is limited to a separate given-correct-event-category structure-extraction test.",
        "",
        "## Protection",
        f"- protected/model SHA unchanged: {report.get('protection', {}).get('sha_unchanged', False)}",
        f"- 32B/external LLM calls: {report['llm_calls']['dianjin_r1_32b_calls']} / {report['llm_calls']['external_llm_calls']}",
        f"- training executed: {report['training_executed']}",
        f"- gold test labels read: {report['gold_test_labels_read']}",
    ]
    return "\n".join(lines) + "\n"


def compact_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "strategy_id",
        "split",
        "strategy_group",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_f1_fixed_64",
        "macro_f1_observed",
        "weighted_f1",
        "article_exact_accuracy",
        "at_least_one_correct_article_ratio",
        "avg_predicted_event_count",
        "complete_miss_count",
        "overpredicted_article_count",
        "candidate_present_fn",
        "candidate_missing",
        "false_positive",
        "false_positive_rate",
        "label_mapping_error",
        "join_error",
        "selection_score",
        "tp",
        "fp",
        "fn",
        "predicted_event_count",
        "gold_event_count",
    ]
    return {key: row.get(key) for key in keys}


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    safety = init_safety(output_dir)
    input_status = ensure_required_inputs()
    proc = process_check()
    if proc["status"] != "ok":
        raise RuntimeError(f"unsafe process state: {proc['suspicious']}")

    fixed = event_names()
    c_params, d_params = load_strategy_params()
    leakage = leakage_summary()

    dev_articles = read_jsonl(STEP56D_DIR / "dev_articles.jsonl")
    holdout_articles = read_jsonl(STEP56D_DIR / "holdout_articles.jsonl")
    dev_ids = ids_from_articles(dev_articles)
    holdout_ids = ids_from_articles(holdout_articles)
    dev_gold = read_jsonl(STEP56D_DIR / "dev_gold_pairs.jsonl")
    holdout_gold = read_jsonl(STEP56D_DIR / "holdout_gold_pairs.jsonl")
    dev_candidates = read_jsonl(STEP56D_DIR / "dev_candidate_v2_top20.jsonl")
    holdout_candidates = read_jsonl(STEP56D_DIR / "holdout_candidate_v2_top20.jsonl")

    score_info: dict[str, Any] = {
        "model_epoch2_dev": {"status": "reused", "path": "outputs/task1_limited_training_round1/round1_epoch_2_dev_scores.jsonl"},
        "experiment_C_dev": {"status": "reused", "path": "outputs/task1_step5_6d_classifier_retraining/experiment_C/dev_scores_best.jsonl"},
        "experiment_C_holdout": {"status": "reused", "path": "outputs/task1_step5_6d_classifier_retraining/holdout_pair_predictions.jsonl"},
    }
    dev_model_epoch2_scores = read_jsonl(ROOT / "outputs/task1_limited_training_round1/round1_epoch_2_dev_scores.jsonl")
    dev_experiment_c_scores = read_jsonl(STEP56D_DIR / "experiment_C/dev_scores_best.jsonl")
    holdout_experiment_c_scores = read_jsonl(STEP56D_DIR / "holdout_pair_predictions.jsonl")
    score_info["model_epoch2_holdout"] = score_softmax_model(
        MODEL_EPOCH2_DIR,
        holdout_candidates,
        article_map(holdout_articles),
        output_dir / "model_epoch2_holdout_scores.jsonl",
        "model_epoch2",
        args.batch_size,
    )
    holdout_model_epoch2_scores = read_jsonl(output_dir / "model_epoch2_holdout_scores.jsonl")

    dev_outputs_raw = build_strategy_outputs("dev", dev_ids, dev_candidates, dev_model_epoch2_scores, dev_experiment_c_scores, c_params, d_params)
    dev_evals = {
        sid: evaluate_selected(sid, "dev", dev_ids, dev_gold, payload["rows"], dev_candidates, fixed, payload["params"])
        for sid, payload in dev_outputs_raw.items()
    }
    dev_summaries = [payload["summary"] for payload in dev_evals.values()]
    selected_dev = choose_dev_strategy(dev_summaries)
    selected_id = selected_dev["strategy_id"]

    holdout_outputs_raw = build_strategy_outputs("holdout", holdout_ids, holdout_candidates, holdout_model_epoch2_scores, holdout_experiment_c_scores, c_params, d_params)
    holdout_eval = evaluate_selected(
        selected_id,
        "holdout",
        holdout_ids,
        holdout_gold,
        holdout_outputs_raw[selected_id]["rows"],
        holdout_candidates,
        fixed,
        holdout_outputs_raw[selected_id]["params"],
    )
    holdout_recall = candidate_recall(holdout_ids, holdout_gold, holdout_candidates, fixed)
    gate = holdout_gate(holdout_eval["summary"], holdout_recall, leakage)

    status = "classification_baseline_frozen" if gate["passed"] else "classification_consolidation_failed"
    frozen_strategy_id = selected_id if gate["passed"] else "A_candidate_v2_top1"
    frozen_note = "dev-selected strategy passed holdout gate" if gate["passed"] else "holdout gate failed; Candidate-v2 Top1 frozen only as retrieval baseline"
    frozen_manifest = {
        "generated_at": now(),
        "status": status,
        "frozen_strategy_id": frozen_strategy_id,
        "dev_selected_strategy_id": selected_id,
        "frozen_note": frozen_note,
        "candidate_v2": {
            "code": "src/routing/task1_candidate_v2.py",
            "code_sha256": sha256_file(ROOT / "src/routing/task1_candidate_v2.py"),
            "config": "configs/task1_candidate_v2.yaml",
            "config_sha256": sha256_file(ROOT / "configs/task1_candidate_v2.yaml"),
            "top_k": 20,
        },
        "models": {
            "model_epoch2": {"path": rel(MODEL_EPOCH2_DIR), "sha256": sha256_file(MODEL_EPOCH2_DIR / "model.safetensors"), "probability": "softmax(logits)[:,1]", "input_contract": "src.data.event_match_text.build_input_text"},
            "experiment_C": {"path": rel(EXPERIMENT_C_DIR), "sha256": sha256_file(EXPERIMENT_C_DIR / "model.safetensors"), "probability": "sigmoid(logit)", "input_contract": "scripts.train_task1_step5_6d_classifier.build_step56d_input"},
        },
        "dev_selection_rule": "exclude recall < 0.55; maximize 0.45*micro_F1 + 0.25*fixed64_macro_F1 + 0.20*article_exact - 0.10*(FP/article_count)",
        "dev_selected": selected_dev,
        "holdout_gate": gate,
        "gold_test_labels_read": False,
        "training_executed": False,
        "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
    }
    write_json_text(output_dir / "frozen_classifier_manifest.json", frozen_manifest)
    frozen_manifest["sha256"] = sha256_file(output_dir / "frozen_classifier_manifest.json")
    write_json_text(output_dir / "frozen_classifier_manifest.json", frozen_manifest)

    write_predictions(output_dir / "unified_dev_predictions.jsonl", "dev", dev_evals)
    write_predictions(output_dir / "unified_holdout_predictions.jsonl", "holdout", {selected_id: holdout_eval})
    write_json_text(output_dir / "holdout_evaluation.json", {"summary": holdout_eval["summary"], "metrics": holdout_eval["metrics"], "error_breakdown": holdout_eval["error_breakdown"], "candidate_recall": holdout_recall, "gate": gate})

    evaluation_80: dict[str, Any] = {"executed": False, "status": "skipped_holdout_gate_failed"}
    comparison_80: dict[str, Any] = {}
    if gate["passed"]:
        gold_free_80 = read_jsonl(STEP56B_DIR / "gold_free_inputs.jsonl")
        ids80 = ids_from_articles(gold_free_80)
        candidates80 = read_jsonl(STEP56B_DIR / "candidates_top20.jsonl")
        gold80 = confirmed_80_gold_pairs()
        score_info["model_epoch2_80"] = {"status": "reused", "path": "outputs/task1_step5_6b_latest_classifier_shadow/pair_predictions.jsonl"}
        model80 = read_jsonl(STEP56B_DIR / "pair_predictions.jsonl")
        score_info["experiment_C_80"] = score_sigmoid_model(
            EXPERIMENT_C_DIR,
            candidates80,
            article_map(gold_free_80),
            output_dir / "experiment_C_80_scores.jsonl",
            "experiment_C",
            args.batch_size,
        )
        exp80 = read_jsonl(output_dir / "experiment_C_80_scores.jsonl")
        outputs80 = build_strategy_outputs("confirmed_80", ids80, candidates80, model80, exp80, c_params, d_params)
        frozen80_rows = outputs80[frozen_strategy_id]["rows"]
        write_jsonl(output_dir / "frozen_80_predictions.jsonl", frozen80_rows)
        frozen80_sha_before_gold = sha256_file(output_dir / "frozen_80_predictions.jsonl")
        evaluation = evaluate_selected(frozen_strategy_id, "confirmed_80", ids80, gold80, frozen80_rows, candidates80, fixed, outputs80[frozen_strategy_id]["params"])
        baseline_c = select_rows(model80, ids80, c_params["strategy"], c_params["strategy_params_dict"])
        baseline_top1 = top1_rows(candidates80, ids80)
        baseline_exp = select_rows(exp80, ids80, d_params["strategy"], d_params["strategy_params_dict"])
        comparison_80 = {
            "vs_step5_6c_model_epoch2": article_set_delta(ids80, gold80, frozen80_rows, baseline_c),
            "vs_candidate_v2_top1": article_set_delta(ids80, gold80, frozen80_rows, baseline_top1),
            "vs_experiment_C": article_set_delta(ids80, gold80, frozen80_rows, baseline_exp),
        }
        evaluation_80 = {
            "executed": True,
            "status": "frozen_80_evaluated",
            "frozen_prediction_sha256_before_gold_eval": frozen80_sha_before_gold,
            "frozen_prediction_sha256_after_gold_eval": sha256_file(output_dir / "frozen_80_predictions.jsonl"),
            "summary": evaluation["summary"],
            "metrics": evaluation["metrics"],
            "error_breakdown": evaluation["error_breakdown"],
            "candidate_recall": candidate_recall(ids80, gold80, candidates80, fixed),
            "comparison": comparison_80,
            "gold_labels_read_after_prediction_freeze": True,
        }
        write_json_text(output_dir / "evaluation_80.json", evaluation_80)
    else:
        # The stop rule forbids 80-row diagnostic when holdout fails.
        write_json_text(output_dir / "evaluation_80.json", evaluation_80)

    strategy_comparison = {
        "generated_at": now(),
        "dev_selection_rule": frozen_manifest["dev_selection_rule"],
        "dev_candidate_recall": candidate_recall(dev_ids, dev_gold, dev_candidates, fixed),
        "holdout_candidate_recall": holdout_recall,
        "dev_strategies": sorted(dev_summaries, key=lambda row: row["selection_score"], reverse=True),
        "holdout_selected_strategy": holdout_eval["summary"],
        "selected_strategy_id": selected_id,
        "frozen_strategy_id": frozen_strategy_id,
        "score_info": score_info,
    }
    write_json_text(output_dir / "strategy_comparison.json", strategy_comparison)

    protection = final_protection(output_dir, output_dir / "protected_and_model_sha256_before.txt")
    report = {
        "generated_at": now(),
        "status": status,
        "input_status": input_status,
        "safety": safety,
        "process_check": proc,
        "data_scope": {
            "dev_article_count": len(dev_ids),
            "dev_gold_event_count": len(dev_gold),
            "holdout_article_count": len(holdout_ids),
            "holdout_gold_event_count": len(holdout_gold),
            "fixed_event_class_count": len(fixed),
            "uses_step5_6d_split": True,
        },
        "selection": {
            "dev_selected": selected_dev,
            "frozen_strategy_id": frozen_strategy_id,
            "frozen_note": frozen_note,
            "eligible_dev_strategy_count": sum(1 for row in dev_summaries if row["micro_recall"] >= 0.55),
            "total_dev_strategy_count": len(dev_summaries),
        },
        "strategy_comparison": strategy_comparison,
        "holdout": {"summary": holdout_eval["summary"], "metrics": holdout_eval["metrics"], "error_breakdown": holdout_eval["error_breakdown"]},
        "holdout_candidate_recall": holdout_recall,
        "holdout_gate": gate,
        "evaluation_80": evaluation_80,
        "classification_module_limitations": [
            "Only 2046 labeled official training articles are available for supervised event classification.",
            "Candidate-v2 recall is high, but classification calibration remains sensitive to split and decoder.",
            "DianJin-R1-32B teacher pilot failed and is not used for Silver-label expansion.",
            "classification_baseline_frozen is a best available baseline, not formal production promotion.",
        ],
        "step5_6e_32b_followup_decision": {
            "teacher_status": input_status["step5_6e_status"],
            "permanently_disabled_for_classification_teacher": True,
            "expand_to_5000": False,
            "silver_labels_enter_training": False,
            "future_allowed_scope": "given-correct-event-category structure extraction only, with separate small-scale validation",
        },
        "allow_next_step": {
            "conditional_structure_extraction_baseline": True,
            "full_end_to_end_task1_claim": False,
            "reason": "classification baseline is frozen/stopped; structure extraction may be developed only as a conditional module.",
        },
        "protection": protection,
        "gold_test_labels_read": False,
        "training_executed": False,
        "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
        "runtime_seconds": round(time.time() - start, 3),
    }
    write_json_text(output_dir / "evaluation_summary.json", report)
    write_json_text(REPORT_JSON, report)
    REPORT_MD.write_text(report_markdown(report), encoding="utf-8")
    write_xlsx(
        REPORT_XLSX,
        {
            "dev_strategies": [compact_summary(row) for row in strategy_comparison["dev_strategies"]],
            "basic_scheme_dev": [compact_summary(row) for row in strategy_comparison["dev_strategies"] if row["strategy_group"] in list("ABCDEFGHI")],
            "holdout_selected": [compact_summary(holdout_eval["summary"])],
            "holdout_class_metrics": holdout_eval["metrics"].get("class_rows", []),
            "holdout_article_rows": holdout_eval["metrics"].get("article_rows", []),
            "candidate_recall_holdout": holdout_recall.get("class_rows", []),
            "evaluation_80_summary": [compact_summary(evaluation_80.get("summary", {}))] if evaluation_80.get("summary") else [],
            "evaluation_80_class": evaluation_80.get("metrics", {}).get("class_rows", []),
            "evaluation_80_article": evaluation_80.get("metrics", {}).get("article_rows", []),
            "error_counts": [
                {"split": "holdout", "reason": key, "count": value.get("count", 0)}
                for key, value in holdout_eval["error_breakdown"].items()
            ]
            + [
                {"split": "confirmed_80", "reason": key, "count": value.get("count", 0)}
                for key, value in evaluation_80.get("error_breakdown", {}).items()
            ],
            "sha_protection": [{"sha_unchanged": protection["sha_unchanged"], "changed_count": len(protection["changed"])}],
        },
    )
    write_sha_manifest(output_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "selected": report["selection"]["dev_selected"]["strategy_id"], "holdout_micro_f1": report["holdout"]["summary"]["micro_f1"], "evaluation_80_executed": report["evaluation_80"].get("executed", False), "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
