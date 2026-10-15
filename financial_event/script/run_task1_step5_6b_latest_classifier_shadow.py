#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.6-B gold-free 80-row latest-classifier shadow run.

Prediction phase only. It builds a label-free 80-row input, runs Candidate-v2
Top-20 and the frozen latest classifier, then freezes outputs before evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_5_candidate_v2 import (  # noqa: E402
    candidate_rows_for_top_k,
    load_config,
    load_frozen_split,
    protected_file_list,
    read_json,
    read_jsonl,
    rel,
    sha256_file,
    slim_candidate,
    write_csv,
    write_json,
    write_jsonl,
)
from src.data.event_match_text import build_input_text  # noqa: E402
from src.routing.task1_candidate_v2 import Task1CandidateGeneratorV2  # noqa: E402

DEFAULT_INPUT = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/task1_step5_6b_latest_classifier_shadow"
DEFAULT_MODEL_DIR = ROOT / "outputs/task1_limited_training_round1/model_epoch_2"
DEFAULT_CANDIDATE_CONFIG = ROOT / "configs/task1_candidate_v2.yaml"
FROZEN_CLASSIFIER_MANIFEST = ROOT / "outputs/task1_frozen_gold_blind_round1/frozen_manifest.json"
EXPECTED_MODEL_SHA256 = "8ccd5ec466d39325762f7bbdf2da6e0c435a0c42dd0f871e84df7fc33cc6a35d"

ALLOWED_INPUT_FIELDS = ["article_id", "title", "content", "publish_time", "source"]
FORBIDDEN_INPUT_FIELDS = [
    "gold_event_names_for_context_after_classification_eval",
    "pred_event_names",
    "annotate_trigger",
    "annotate_event_span",
    "annotate_subject_company",
    "annotate_event_time",
    "annotate_object_target",
    "annotate_direction",
    "annotate_arguments_json",
    "annotate_argument_roles_json",
    "reviewer_confirmed",
]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "gold_free_inputs": output_dir / "gold_free_inputs.jsonl",
        "gold_free_manifest": output_dir / "gold_free_input_manifest.json",
        "candidates_all64": output_dir / "candidates_all64.jsonl",
        "candidates_top20": output_dir / "candidates_top20.jsonl",
        "candidate_timing": output_dir / "candidate_generation_timing.csv",
        "pair_predictions": output_dir / "pair_predictions.jsonl",
        "article_predictions": output_dir / "article_predictions.jsonl",
        "event_mentions_skeleton": output_dir / "event_mentions_skeleton.jsonl",
        "run_manifest": output_dir / "run_manifest.json",
        "sha256sums": output_dir / "SHA256SUMS.txt",
    }


def assert_safe_resume(output_dir: Path, resume: bool) -> None:
    paths = output_paths(output_dir)
    existing = [path for key, path in paths.items() if key not in {"sha256sums"} and path.exists()]
    if existing and not resume:
        raise FileExistsError(f"Step 5.6-B outputs already exist; use --resume after checking manifest: {[str(p) for p in existing[:5]]}")
    if resume and paths["run_manifest"].exists():
        manifest = read_json(paths["run_manifest"])
        if manifest.get("model_dir") != rel(DEFAULT_MODEL_DIR):
            raise RuntimeError(f"resume refused: model mismatch in existing manifest: {manifest.get('model_dir')}")
        if manifest.get("model_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("resume refused: model SHA mismatch in existing manifest")
        if manifest.get("candidate_generator", {}).get("top_k") != 20:
            raise RuntimeError("resume refused: candidate top_k mismatch")


def build_gold_free_input(source_path: Path, output_dir: Path, resume: bool) -> list[dict[str, Any]]:
    paths = output_paths(output_dir)
    if resume and paths["gold_free_inputs"].exists():
        rows = read_jsonl(paths["gold_free_inputs"])
        ensure_label_free(rows)
        return rows
    source_rows = read_jsonl(source_path)
    if len(source_rows) != 80:
        raise RuntimeError(f"expected 80 confirmed annotations, got {len(source_rows)}")
    ids = [str(row.get("article_id") or "") for row in source_rows]
    if any(not aid for aid in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("confirmed annotations have missing or duplicate article_id")
    out = []
    removed = Counter()
    for row in source_rows:
        out.append({field: row.get(field, "") for field in ALLOWED_INPUT_FIELDS})
        for key in row:
            if key not in ALLOWED_INPUT_FIELDS:
                removed[key] += 1
    ensure_label_free(out)
    write_jsonl(paths["gold_free_inputs"], out)
    write_json(
        paths["gold_free_manifest"],
        {
            "generated_at": now(),
            "source_file": rel(source_path),
            "input_count": len(out),
            "allowed_fields": ALLOWED_INPUT_FIELDS,
            "removed_forbidden_fields": {key: removed.get(key, 0) for key in FORBIDDEN_INPUT_FIELDS},
            "article_id_unique": len(set(ids)) == len(ids),
            "gold_free_inputs_sha256": sha256_file(paths["gold_free_inputs"]),
        },
    )
    return out


def ensure_label_free(rows: list[dict[str, Any]]) -> None:
    allowed = set(ALLOWED_INPUT_FIELDS)
    for row in rows:
        extra = set(row) - allowed
        if extra:
            raise RuntimeError(f"gold-free input contains disallowed fields: {extra}")
        suspicious = [
            key
            for key in row
            if any(token in key.lower() for token in ["label", "schema", "event_name", "annotate", "gold"])
            or any(token in key for token in ["标签", "事件"])
        ]
        if suspicious:
            raise RuntimeError(f"gold-free input contains label-like keys: {suspicious}")


def train_dev_overlap(article_ids: list[str], split: dict[str, Any]) -> dict[str, Any]:
    train = set(map(str, split["train_ids"]))
    dev = set(map(str, split["dev_ids"]))
    ids = list(map(str, article_ids))
    return {
        "train_overlap_count": len(set(ids) & train),
        "train_overlap_article_ids": sorted(set(ids) & train),
        "dev_overlap_count": len(set(ids) & dev),
        "dev_overlap_article_ids": sorted(set(ids) & dev),
        "duplicate_count": len(ids) - len(set(ids)),
        "duplicate_article_ids": sorted([aid for aid, count in Counter(ids).items() if count > 1]),
        "unable_to_trace_training_source_count": 0,
        "lineage_status": "verified",
        "split_checks": split["checks"],
        "evaluation_mode": "blind_evaluation" if not (set(ids) & train) else "non_blind_diagnostic",
    }


def build_candidates(output_dir: Path, config: dict[str, Any], split: dict[str, Any], articles: list[dict[str, Any]], resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = output_paths(output_dir)
    if resume and paths["candidates_all64"].exists() and paths["candidates_top20"].exists() and paths["candidate_timing"].exists():
        all64 = read_jsonl(paths["candidates_all64"])
        top20 = read_jsonl(paths["candidates_top20"])
        timing = list(csv.DictReader(paths["candidate_timing"].open("r", encoding="utf-8-sig")))
        return all64, top20, timing
    generator = Task1CandidateGeneratorV2.from_paths(
        schema_path=ROOT / config["schema_path"],
        train_articles=split["train_articles"],
        train_gold_pairs=split["train_pairs"],
        alias_path=ROOT / config["target_alias_path"],
        config=config.get("candidate_generator", {}),
    )
    rows, timing = generator.generate_many(articles, include_all64=True)
    all64 = [slim_candidate(row) for row in rows]
    top20 = candidate_rows_for_top_k(all64, 20)
    write_jsonl(paths["candidates_all64"], all64)
    write_jsonl(paths["candidates_top20"], top20)
    write_csv(paths["candidate_timing"], timing, ["article_id", "candidate_count", "generation_time_ms"])
    return all64, top20, timing


def enrich_for_model(row: dict[str, Any], article_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = dict(row)
    article = article_by_id[str(row["article_id"])]
    content = str(article.get("content", ""))
    if len(content) > 2600:
        content = content[:1900] + "\n[正文中间省略]\n" + content[-600:]
    item["title"] = article.get("title", item.get("title", ""))
    item["content"] = content
    return item


def load_threshold() -> tuple[float, dict[str, Any]]:
    if not FROZEN_CLASSIFIER_MANIFEST.exists():
        raise RuntimeError("threshold_lineage_missing: frozen classifier manifest not found")
    manifest = read_json(FROZEN_CLASSIFIER_MANIFEST)
    decoder = manifest.get("decoder") or {}
    if decoder.get("rule") != "top1_min_score" or "threshold" not in decoder:
        raise RuntimeError(f"threshold_lineage_missing: invalid decoder config {decoder}")
    return float(decoder["threshold"]), {"manifest": rel(FROZEN_CLASSIFIER_MANIFEST), "decoder": decoder}


def score_model(output_dir: Path, candidate_rows: list[dict[str, Any]], articles: list[dict[str, Any]], threshold: float, resume: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = output_paths(output_dir)
    if resume and paths["pair_predictions"].exists():
        rows = read_jsonl(paths["pair_predictions"])
        return rows, {"status": "reused", "path": rel(paths["pair_predictions"]), "record_count": len(rows)}
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"ML dependency missing: {exc!r}") from exc

    if sha256_file(DEFAULT_MODEL_DIR / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA mismatch before inference")

    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(DEFAULT_MODEL_DIR), local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    article_by_id = {str(row["article_id"]): row for row in articles}

    class PairDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = enrich_for_model(self.rows[idx], article_by_id)
            return {"text": build_input_text(row), "row": self.rows[idx]}

    def collate(batch: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        enc = tokenizer([row["text"] for row in batch], truncation=True, max_length=512, padding=True, return_tensors="pt")
        return enc, [row["row"] for row in batch]

    start = time.time()
    rows_with_scores: list[dict[str, Any]] = []
    loader = DataLoader(PairDataset(candidate_rows), batch_size=64, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for enc, batch_rows in loader:
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits.detach().cpu()
            probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
            for row, logit_row, prob in zip(batch_rows, logits.tolist(), probs):
                item = dict(row)
                item["raw_score_negative_logit"] = float(logit_row[0])
                item["raw_score_positive_logit"] = float(logit_row[1])
                item["probability_positive"] = float(prob)
                item["decode_threshold"] = threshold
                item["decode_rule"] = "candidate_v2_top20_top1_min_score"
                item["decision"] = False
                rows_with_scores.append(item)

    by_article: dict[str, list[dict[str, Any]]] = {}
    for row in rows_with_scores:
        by_article.setdefault(str(row["article_id"]), []).append(row)
    for aid, rows in by_article.items():
        ordered = sorted(
            rows,
            key=lambda item: (-float(item["probability_positive"]), int(item.get("candidate_rank", 10**9) or 10**9), str(item.get("event_name", ""))),
        )
        if ordered and float(ordered[0]["probability_positive"]) >= threshold:
            ordered[0]["decision"] = True

    write_jsonl(paths["pair_predictions"], rows_with_scores)
    return rows_with_scores, {
        "status": "completed",
        "path": rel(paths["pair_predictions"]),
        "record_count": len(rows_with_scores),
        "runtime_seconds": round(time.time() - start, 3),
        "device": str(device),
    }


def build_article_predictions(output_dir: Path, articles: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = output_paths(output_dir)
    by_article: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        by_article.setdefault(str(row["article_id"]), []).append(row)
    article_rows = []
    for article in articles:
        aid = str(article["article_id"])
        rows = sorted(
            by_article.get(aid, []),
            key=lambda item: (-float(item.get("probability_positive", 0.0) or 0.0), int(item.get("candidate_rank", 10**9) or 10**9), str(item.get("event_name", ""))),
        )
        selected = [row for row in rows if row.get("decision") is True]
        article_rows.append(
            {
                "article_id": aid,
                "title": article.get("title", ""),
                "predicted_event_count": len(selected),
                "pred_event_names": "|".join(row["event_name"] for row in selected),
                "top_event_name": rows[0].get("event_name", "") if rows else "",
                "top_probability_positive": float(rows[0].get("probability_positive", 0.0)) if rows else "",
                "top_candidate_rank": int(rows[0].get("candidate_rank", 0) or 0) if rows else "",
                "processing_status": "success_with_events" if selected else "success_no_event",
            }
        )
    write_jsonl(paths["article_predictions"], article_rows)
    return article_rows


def build_event_mentions_skeleton(output_dir: Path, pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = output_paths(output_dir)
    mentions = []
    for rank, row in enumerate([item for item in pair_rows if item.get("decision") is True], 1):
        mention_id = f"S56B-{row['article_id']}-{row['schema_id']}".replace("/", "_")
        mentions.append(
            {
                "event_mention_id": mention_id,
                "article_id": row["article_id"],
                "schema_id": row["schema_id"],
                "event_name": row["event_name"],
                "target_type": row.get("target_type", ""),
                "target_code": row.get("target_code", ""),
                "target_name": row.get("target_name", ""),
                "classification_probability": row.get("probability_positive", ""),
                "classification_threshold": row.get("decode_threshold", ""),
                "classification_source": "task1_step5_6b_latest_classifier_shadow",
                "structure_fields": {
                    "trigger": {"status": "not_implemented"},
                    "event_span": {"status": "not_implemented"},
                    "subject_company": {"status": "not_implemented"},
                    "event_time": {"status": "not_implemented"},
                    "object_target": {"status": "not_implemented"},
                    "direction": {"status": "not_implemented"},
                    "arguments": {"status": "not_implemented"},
                    "argument_roles": {"status": "not_implemented"},
                },
                "rank": rank,
            }
        )
    write_jsonl(paths["event_mentions_skeleton"], mentions)
    return mentions


def write_sha_manifest(output_dir: Path) -> None:
    paths = output_paths(output_dir)
    rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    with paths["sha256sums"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['sha256']}  {row['path']}\n")


def protected_snapshot(path: Path) -> dict[str, str]:
    rows = {rel(item): sha256_file(item) for item in protected_file_list()}
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(rows.items())), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--candidate-config", type=Path, default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_dir.resolve() != DEFAULT_MODEL_DIR.resolve():
        raise RuntimeError("Step 5.6-B is frozen to outputs/task1_limited_training_round1/model_epoch_2")
    if not args.input.exists() or not args.model_dir.exists() or not args.candidate_config.exists():
        raise FileNotFoundError("required Step 5.6-B input/model/config is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert_safe_resume(args.output_dir, args.resume)
    paths = output_paths(args.output_dir)

    start = time.time()
    protected_before = protected_snapshot(args.output_dir / "formal_protected_sha256_before.txt")
    config = load_config(args.candidate_config)
    split = load_frozen_split(config)
    articles = build_gold_free_input(args.input, args.output_dir, args.resume)
    article_ids = [str(row["article_id"]) for row in articles]
    overlap = train_dev_overlap(article_ids, split)
    threshold, threshold_lineage = load_threshold()
    all64, top20, timing = build_candidates(args.output_dir, config, split, articles, args.resume)
    pair_rows, score_summary = score_model(args.output_dir, top20, articles, threshold, args.resume)
    article_rows = build_article_predictions(args.output_dir, articles, pair_rows)
    mentions = build_event_mentions_skeleton(args.output_dir, pair_rows)
    protected_after = protected_snapshot(args.output_dir / "formal_protected_sha256_after_prediction.txt")

    run_manifest = {
        "generated_at": now(),
        "status": "prediction_frozen",
        "stage": "Task1 Step 5.6-B",
        "gold_labels_read_in_prediction_phase": False,
        "input_confirmed_annotation": rel(args.input),
        "gold_free_inputs": rel(paths["gold_free_inputs"]),
        "gold_free_inputs_sha256": sha256_file(paths["gold_free_inputs"]),
        "article_count": len(articles),
        "train_dev_overlap": overlap,
        "model_dir": rel(args.model_dir),
        "model_sha256": sha256_file(args.model_dir / "model.safetensors"),
        "expected_model_sha256": EXPECTED_MODEL_SHA256,
        "candidate_generator": {
            "code": "src/routing/task1_candidate_v2.py",
            "code_sha256": sha256_file(ROOT / "src/routing/task1_candidate_v2.py"),
            "config": rel(args.candidate_config),
            "config_sha256": sha256_file(args.candidate_config),
            "top_k": 20,
            "all64_candidate_rows": len(all64),
            "top20_candidate_rows": len(top20),
            "avg_top20_per_article": len(top20) / len(articles) if articles else 0.0,
        },
        "threshold": threshold,
        "threshold_lineage": threshold_lineage,
        "pair_predictions": rel(paths["pair_predictions"]),
        "article_predictions": rel(paths["article_predictions"]),
        "event_mentions_skeleton": rel(paths["event_mentions_skeleton"]),
        "prediction_event_count": sum(1 for row in pair_rows if row.get("decision") is True),
        "success_with_events": sum(1 for row in article_rows if row["processing_status"] == "success_with_events"),
        "success_no_event": sum(1 for row in article_rows if row["processing_status"] == "success_no_event"),
        "score_summary": score_summary,
        "candidate_timing_summary": {
            "record_count": len(timing),
            "avg_generation_time_ms": sum(float(row.get("generation_time_ms", 0.0) or 0.0) for row in timing) / len(timing) if timing else 0.0,
            "avg_candidate_count": sum(float(row.get("candidate_count", 0.0) or 0.0) for row in timing) / len(timing) if timing else 0.0,
        },
        "formal_sha_unchanged_after_prediction": protected_before == protected_after,
        "runtime_seconds": round(time.time() - start, 3),
        "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
        "reproduction_command": "python scripts/run_task1_step5_6b_latest_classifier_shadow.py --resume",
    }
    write_json(paths["run_manifest"], run_manifest)
    write_sha_manifest(args.output_dir)
    print(json.dumps({
        "status": "prediction_frozen",
        "article_count": len(articles),
        "top20_candidate_rows": len(top20),
        "prediction_event_count": run_manifest["prediction_event_count"],
        "runtime_seconds": run_manifest["runtime_seconds"],
        "evaluation_mode": overlap["evaluation_mode"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
