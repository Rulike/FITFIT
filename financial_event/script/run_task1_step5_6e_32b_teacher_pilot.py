#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Task1 Step 5.6-E local DianJin-R1-32B teacher pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_5_candidate_v2 import rel, sha256_file, write_json, write_jsonl  # noqa: E402
from src.event_classification.task1_teacher_32b import (  # noqa: E402
    API_BASE,
    MODEL_NAME,
    build_user_message,
    chat_completion,
    correction_user_message,
    evidence_sentences,
    strict_json_loads,
    teacher_prompt_template,
    validate_teacher_output,
)

OUTPUT_DIR = ROOT / "outputs/task1_step5_6e_32b_teacher_pilot"
CONFIRMED_80 = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"
GOLD_FREE_TEST = ROOT / "outputs/task1_gold_shadow_eval_v1/blind_input.jsonl"
FORBIDDEN_GOLD_TEST_LABEL_PATHS = {
    "data/processed/labeled/news_test.jsonl",
    "data/processed/labeled/event_pairs_test.jsonl",
    "data/processed/model_data/event_match_official_test.jsonl",
    "data/raw/news_with_tags_test.xlsx",
}


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_sha_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append((rel(path), sha256_file(path)))
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in rows),
        encoding="utf-8",
    )


def assert_no_gold_test_label_paths() -> None:
    for rel_path in FORBIDDEN_GOLD_TEST_LABEL_PATHS:
        path = ROOT / rel_path
        if path.exists():
            continue


def article_ids_from_jsonl(path: Path) -> set[str]:
    return {str(row.get("article_id") or "") for row in read_jsonl(path) if row.get("article_id")}


def labels_by_article(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        out[str(row["article_id"])].add(str(row["event_name"]))
    return out


def select_preflight(train_articles: list[dict[str, Any]], rng: random.Random) -> list[str]:
    ids = sorted(str(row["article_id"]) for row in train_articles)
    return rng.sample(ids, 20)


def select_pilot_200(dev_articles: list[dict[str, Any]], dev_gold: list[dict[str, Any]], forbidden_ids: set[str], rng: random.Random) -> tuple[list[str], dict[str, Any]]:
    article_by_id = {str(row["article_id"]): row for row in dev_articles}
    gold = labels_by_article(dev_gold)
    by_event: dict[str, list[str]] = defaultdict(list)
    for aid, names in gold.items():
        if aid not in article_by_id or aid in forbidden_ids:
            continue
        for name in names:
            by_event[name].append(aid)
    selected: list[str] = []
    selected_set: set[str] = set()
    used_dup: set[str] = set()

    def can_add(aid: str) -> bool:
        if aid in selected_set or aid in forbidden_ids:
            return False
        dup = str(article_by_id[aid].get("duplication_id") or "")
        return not dup or dup not in used_dup

    def add(aid: str) -> bool:
        if not can_add(aid):
            return False
        selected.append(aid)
        selected_set.add(aid)
        dup = str(article_by_id[aid].get("duplication_id") or "")
        if dup:
            used_dup.add(dup)
        return True

    for event_name in sorted(by_event, key=lambda name: (len(by_event[name]), name)):
        aids = list(dict.fromkeys(by_event[event_name]))
        rng.shuffle(aids)
        for aid in aids:
            if add(aid):
                break
    multi = [aid for aid, names in gold.items() if len(names) > 1 and aid in article_by_id]
    rng.shuffle(multi)
    for aid in multi:
        if len(selected) >= 200:
            break
        add(aid)
    remaining = [aid for aid in sorted(article_by_id) if can_add(aid)]
    rng.shuffle(remaining)
    for aid in remaining:
        if len(selected) >= 200:
            break
        add(aid)
    if len(selected) != 200:
        raise RuntimeError(f"could not select exactly 200 pilot articles, got {len(selected)}")
    covered_events = sorted({name for aid in selected for name in gold.get(aid, set())})
    return selected, {
        "selected_count": len(selected),
        "covered_event_class_count": len(covered_events),
        "covered_event_classes": covered_events,
        "multi_event_article_count": sum(1 for aid in selected if len(gold.get(aid, set())) > 1),
        "duplicate_id_count": len([aid for aid in selected if article_by_id[aid].get("duplication_id")]),
    }


def candidates_by_article(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("candidate_rank", 10**9) or 10**9) <= 20:
            out[str(row["article_id"])].append(row)
    for aid in out:
        out[aid].sort(key=lambda row: (int(row.get("candidate_rank", 10**9) or 10**9), str(row.get("event_name", ""))))
    return out


def slim_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_id": str(row["article_id"]),
        "event_name": str(row["event_name"]),
        "schema_id": str(row.get("schema_id", "")),
        "target_type": str(row.get("target_type", "")),
        "target_code": str(row.get("target_code", "")),
        "target_name": str(row.get("target_name", "")),
        "candidate_rank": int(row.get("candidate_rank", 0) or 0),
        "candidate_score": float(row.get("candidate_score", 0.0) or 0.0),
        "candidate_source": str(row.get("candidate_source", "")),
        "event_description": str(row.get("event_description", "")),
    }


def build_gold_free_records(article_ids: list[str], articles: dict[str, dict[str, Any]], candidates: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    candidate_rows = []
    evidence_rows = []
    for idx, aid in enumerate(article_ids):
        article = articles[aid]
        cand_rows = [slim_candidate(row) for row in candidates.get(aid, [])[:20]]
        if len(cand_rows) != 20:
            raise RuntimeError(f"article {aid} does not have 20 candidates")
        evid = evidence_sentences(article, cand_rows)
        inputs.append(
            {
                "input_index": idx,
                "article_id": aid,
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "publish_time": article.get("publish_time", ""),
                "source": article.get("source", ""),
                "candidates": cand_rows,
                "evidence_sentences": evid,
            }
        )
        for row in cand_rows:
            candidate_rows.append(row)
        evidence_rows.append({"input_index": idx, "article_id": aid, "evidence_sentences": evid})
    return inputs, candidate_rows, evidence_rows


def prompt_manifest(output_dir: Path, prompt_path: Path) -> None:
    write_json(
        output_dir / "frozen_teacher_prompt_manifest.json",
        {
            "generated_at": now(),
            "prompt_path": rel(prompt_path),
            "prompt_sha256": sha256_file(prompt_path),
            "label": "local_32b_teacher_silver_label_pilot",
            "model": MODEL_NAME,
            "temperature": 0,
            "prompt_frozen_after_preflight_technical_checks": True,
            "preflight_accuracy_used_for_prompt_design": False,
            "gold_test_labels_read": False,
            "external_llm_calls": 0,
        },
    )


def request_one(item: dict[str, Any], system_prompt: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_names = {row["event_name"] for row in item["candidates"]}
    evidence_ids = {row["evidence_id"] for row in item["evidence_sentences"]}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(item)},
    ]
    raw_outputs = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    errors: list[str] = []
    started = time.perf_counter()
    http_5xx = 0
    timeout = 0
    cuda_oom = 0
    for attempt in range(2):
        try:
            content, full_payload, elapsed = chat_completion(messages, api_base=args.api_base, model=args.model, timeout=args.timeout, max_tokens=args.max_tokens)
            usage = full_payload.get("usage") or {}
            for key in usage_total:
                usage_total[key] += int(usage.get(key, 0) or 0)
            raw_outputs.append({"attempt": attempt + 1, "content": content, "elapsed_seconds": elapsed, "usage": usage})
            try:
                payload = strict_json_loads(content)
                valid, validation_errors, parsed = validate_teacher_output(payload, candidate_names, evidence_ids)
            except Exception as exc:
                valid, validation_errors, parsed = False, [f"json_parse_error:{exc}"], None
            if valid and parsed is not None:
                prediction = {
                    "input_index": item["input_index"],
                    "article_id": item["article_id"],
                    "status": "success",
                    "selected_events": parsed["selected_events"],
                    "no_event": parsed["no_event"],
                    "retry_count": attempt,
                }
                metric = {
                    "input_index": item["input_index"],
                    "article_id": item["article_id"],
                    "status": "success",
                    "request_count": attempt + 1,
                    "retry_count": attempt,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "prompt_tokens": usage_total["prompt_tokens"],
                    "completion_tokens": usage_total["completion_tokens"],
                    "total_tokens": usage_total["total_tokens"],
                    "json_parse_failed": False,
                    "illegal_event_name_count": 0,
                    "illegal_confidence_count": 0,
                    "illegal_evidence_id_count": 0,
                    "http_5xx_count": http_5xx,
                    "timeout_count": timeout,
                    "cuda_oom_count": cuda_oom,
                }
                raw = {"input_index": item["input_index"], "article_id": item["article_id"], "status": "success", "raw_outputs": raw_outputs}
                return raw, prediction, metric
            errors = validation_errors
            if attempt == 0:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": correction_user_message(content, validation_errors)},
                ]
        except RuntimeError as exc:
            msg = str(exc)
            errors = [msg]
            if "http_error:5" in msg:
                http_5xx += 1
            if "timed out" in msg.lower() or "timeout" in msg.lower():
                timeout += 1
            if "cuda" in msg.lower() and "oom" in msg.lower():
                cuda_oom += 1
            if attempt == 0:
                continue
    raw = {"input_index": item["input_index"], "article_id": item["article_id"], "status": "failed", "raw_outputs": raw_outputs, "errors": errors}
    prediction = {"input_index": item["input_index"], "article_id": item["article_id"], "status": "failed", "selected_events": [], "no_event": True, "errors": errors}
    metric = {
        "input_index": item["input_index"],
        "article_id": item["article_id"],
        "status": "failed",
        "request_count": len(raw_outputs) or 2,
        "retry_count": max(0, len(raw_outputs) - 1),
        "latency_seconds": round(time.perf_counter() - started, 6),
        "prompt_tokens": usage_total["prompt_tokens"],
        "completion_tokens": usage_total["completion_tokens"],
        "total_tokens": usage_total["total_tokens"],
        "json_parse_failed": True,
        "illegal_event_name_count": -1,
        "illegal_confidence_count": -1,
        "illegal_evidence_id_count": -1,
        "http_5xx_count": http_5xx,
        "timeout_count": timeout,
        "cuda_oom_count": cuda_oom,
        "errors": "|".join(errors),
    }
    return raw, prediction, metric


def run_requests(items: list[dict[str, Any]], system_prompt: str, prefix: str, args: argparse.Namespace) -> dict[str, Any]:
    raw_tmp = args.output_dir / f"{prefix}_raw_responses.jsonl.tmp"
    pred_tmp = args.output_dir / f"{prefix}_teacher_predictions.jsonl.tmp"
    metrics_tmp = args.output_dir / f"{prefix}_request_metrics.jsonl.tmp"
    raw_final = args.output_dir / ("raw_responses.jsonl" if prefix == "pilot" else f"{prefix}_raw_responses.jsonl")
    pred_final = args.output_dir / ("teacher_predictions.jsonl" if prefix == "pilot" else f"{prefix}_teacher_predictions.jsonl")
    metrics_final = args.output_dir / ("request_metrics.jsonl" if prefix == "pilot" else f"{prefix}_request_metrics.jsonl")
    if raw_final.exists() and pred_final.exists() and metrics_final.exists():
        return summarize_request_metrics(read_jsonl(metrics_final), reused=True)
    done = set()
    if pred_tmp.exists():
        for row in read_jsonl(pred_tmp):
            if row.get("status") == "success":
                done.add(int(row["input_index"]))
    remaining = [item for item in items if int(item["input_index"]) not in done]
    raw_handle = raw_tmp.open("a", encoding="utf-8")
    pred_handle = pred_tmp.open("a", encoding="utf-8")
    metrics_handle = metrics_tmp.open("a", encoding="utf-8")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(request_one, item, system_prompt, args): item for item in remaining}
            for future in concurrent.futures.as_completed(future_map):
                raw, prediction, metric = future.result()
                raw_handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
                pred_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                metrics_handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
                raw_handle.flush()
                pred_handle.flush()
                metrics_handle.flush()
    finally:
        raw_handle.close()
        pred_handle.close()
        metrics_handle.close()
    raw_rows = sorted(read_jsonl(raw_tmp), key=lambda row: int(row["input_index"]))
    pred_rows_by_idx = {}
    for row in read_jsonl(pred_tmp):
        pred_rows_by_idx[int(row["input_index"])] = row
    metric_rows_by_idx = {}
    for row in read_jsonl(metrics_tmp):
        metric_rows_by_idx[int(row["input_index"])] = row
    pred_rows = [pred_rows_by_idx[idx] for idx in sorted(pred_rows_by_idx)]
    metric_rows = [metric_rows_by_idx[idx] for idx in sorted(metric_rows_by_idx)]
    if len(pred_rows) == len(items):
        write_jsonl(raw_final, raw_rows)
        write_jsonl(pred_final, pred_rows)
        write_jsonl(metrics_final, metric_rows)
    return summarize_request_metrics(metric_rows, reused=False)


def summarize_request_metrics(rows: list[dict[str, Any]], reused: bool = False) -> dict[str, Any]:
    latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in rows]
    totals = [int(row.get("total_tokens", 0) or 0) for row in rows]
    success = [row for row in rows if row.get("status") == "success"]

    def p95(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(statistics.quantiles(values, n=20, method="inclusive")[18]) if len(values) > 1 else float(values[0])

    return {
        "reused": reused,
        "record_count": len(rows),
        "success_count": len(success),
        "failed_count": len(rows) - len(success),
        "first_success_count": sum(1 for row in rows if row.get("status") == "success" and int(row.get("retry_count", 0) or 0) == 0),
        "retry_count": sum(int(row.get("retry_count", 0) or 0) for row in rows),
        "request_count": sum(int(row.get("request_count", 0) or 0) for row in rows),
        "json_parse_failed_count": sum(1 for row in rows if row.get("json_parse_failed")),
        "illegal_event_name_count": sum(max(0, int(row.get("illegal_event_name_count", 0) or 0)) for row in rows),
        "illegal_confidence_count": sum(max(0, int(row.get("illegal_confidence_count", 0) or 0)) for row in rows),
        "illegal_evidence_id_count": sum(max(0, int(row.get("illegal_evidence_id_count", 0) or 0)) for row in rows),
        "http_5xx_count": sum(int(row.get("http_5xx_count", 0) or 0) for row in rows),
        "timeout_count": sum(int(row.get("timeout_count", 0) or 0) for row in rows),
        "cuda_oom_count": sum(int(row.get("cuda_oom_count", 0) or 0) for row in rows),
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_seconds": p95(latencies),
        "avg_total_tokens": sum(totals) / len(totals) if totals else 0.0,
        "p95_total_tokens": p95([float(value) for value in totals]),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens", 0) or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in rows),
    }


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    train_articles = read_jsonl(ROOT / "outputs/task1_step5_6d_classifier_retraining/train_articles.jsonl")
    dev_articles = read_jsonl(ROOT / "outputs/task1_step5_6d_classifier_retraining/dev_articles.jsonl")
    holdout_articles = read_jsonl(ROOT / "outputs/task1_step5_6d_classifier_retraining/holdout_articles.jsonl")
    dev_gold = read_jsonl(ROOT / "outputs/task1_step5_6d_classifier_retraining/dev_gold_pairs.jsonl")
    train_candidates = candidates_by_article(read_jsonl(ROOT / "outputs/task1_step5_6d_classifier_retraining/train_candidate_v2_top20.jsonl"))
    dev_candidates = candidates_by_article(read_jsonl(ROOT / "outputs/task1_step5_6d_classifier_retraining/dev_candidate_v2_top20.jsonl"))
    train_map = {str(row["article_id"]): row for row in train_articles}
    dev_map = {str(row["article_id"]): row for row in dev_articles}
    holdout_ids = {str(row["article_id"]) for row in holdout_articles}
    confirmed_80_ids = article_ids_from_jsonl(CONFIRMED_80)
    gold_free_test_ids = article_ids_from_jsonl(GOLD_FREE_TEST)
    preflight_ids = select_preflight(train_articles, rng)
    forbidden_for_pilot = set(preflight_ids) | set(train_map) | holdout_ids | confirmed_80_ids | gold_free_test_ids
    pilot_ids, pilot_meta = select_pilot_200(dev_articles, dev_gold, forbidden_for_pilot, rng)
    preflight_inputs, preflight_candidate_rows, preflight_evidence_rows = build_gold_free_records(preflight_ids, train_map, train_candidates)
    pilot_inputs, pilot_candidate_rows, pilot_evidence_rows = build_gold_free_records(pilot_ids, dev_map, dev_candidates)
    write_json(args.output_dir / "preflight_20_ids.json", {"seed": args.seed, "count": len(preflight_ids), "article_ids": preflight_ids})
    write_json(args.output_dir / "pilot_200_ids.json", {"seed": args.seed, "article_ids": pilot_ids, **pilot_meta})
    write_jsonl(args.output_dir / "gold_free_preflight_inputs.jsonl", preflight_inputs)
    write_jsonl(args.output_dir / "preflight_candidates_top20.jsonl", preflight_candidate_rows)
    write_jsonl(args.output_dir / "preflight_evidence_sentences.jsonl", preflight_evidence_rows)
    write_jsonl(args.output_dir / "gold_free_pilot_inputs.jsonl", pilot_inputs)
    write_jsonl(args.output_dir / "candidates_top20.jsonl", pilot_candidate_rows)
    write_jsonl(args.output_dir / "evidence_sentences.jsonl", pilot_evidence_rows)
    split_manifest = {
        "generated_at": now(),
        "seed": args.seed,
        "preflight_count": len(preflight_ids),
        "pilot_count": len(pilot_ids),
        "pilot_source": "Step 5.6-D frozen dev split",
        "preflight_source": "Step 5.6-D train split",
        "train_overlap_count": len(set(pilot_ids) & set(train_map)),
        "holdout_overlap_count": len(set(pilot_ids) & holdout_ids),
        "preflight_overlap_count": len(set(pilot_ids) & set(preflight_ids)),
        "confirmed_80_overlap_count": len(set(pilot_ids) & confirmed_80_ids),
        "gold_free_test_overlap_count": len(set(pilot_ids) & gold_free_test_ids),
        "gold_test_labels_read": False,
        "gold_free_inputs_fields": ["input_index", "article_id", "title", "content", "publish_time", "source", "candidates", "evidence_sentences"],
    }
    write_json(args.output_dir / "sampling_manifest.json", split_manifest)
    return {
        "preflight_inputs": preflight_inputs,
        "pilot_inputs": pilot_inputs,
        "sampling_manifest": split_manifest,
    }


def preflight_passed(summary: dict[str, Any]) -> bool:
    return (
        summary["record_count"] == 20
        and summary["success_count"] == 20
        and summary["failed_count"] == 0
        and summary["illegal_event_name_count"] == 0
        and summary["illegal_confidence_count"] == 0
        and summary["illegal_evidence_id_count"] == 0
        and summary["cuda_oom_count"] == 0
        and summary["json_parse_failed_count"] == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-pilot", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert_no_gold_test_label_paths()
    data = build_inputs(args)
    draft_prompt = teacher_prompt_template()
    preflight_summary = run_requests(data["preflight_inputs"], draft_prompt, "preflight", args)
    if not preflight_passed(preflight_summary):
        run_manifest = {
            "generated_at": now(),
            "status": "preflight_failed",
            "label": "local_32b_teacher_silver_label_pilot",
            "preflight_summary": preflight_summary,
            "pilot_executed": False,
            "gold_test_labels_read": False,
            "external_llm_calls": 0,
        }
        write_json(args.output_dir / "run_manifest.json", run_manifest)
        write_sha_manifest(args.output_dir)
        print(json.dumps({"status": "preflight_failed", "preflight_summary": preflight_summary}, ensure_ascii=False))
        return
    prompt_path = args.output_dir / "frozen_teacher_prompt.txt"
    prompt_path.write_text(draft_prompt, encoding="utf-8")
    prompt_manifest(args.output_dir, prompt_path)
    if args.skip_pilot:
        run_manifest = {
            "generated_at": now(),
            "status": "preflight_passed_skip_pilot",
            "label": "local_32b_teacher_silver_label_pilot",
            "preflight_summary": preflight_summary,
            "pilot_executed": False,
            "prompt_sha256": sha256_file(prompt_path),
            "gold_test_labels_read": False,
            "external_llm_calls": 0,
        }
        write_json(args.output_dir / "run_manifest.json", run_manifest)
        write_sha_manifest(args.output_dir)
        print(json.dumps({"status": "preflight_passed_skip_pilot", "preflight_summary": preflight_summary}, ensure_ascii=False))
        return
    pilot_summary = run_requests(data["pilot_inputs"], draft_prompt, "pilot", args)
    status = "pilot_completed" if pilot_summary["record_count"] == 200 and pilot_summary["success_count"] == 200 else "pilot_failed"
    run_manifest = {
        "generated_at": now(),
        "status": status,
        "label": "local_32b_teacher_silver_label_pilot",
        "api_base": args.api_base,
        "model": args.model,
        "temperature": 0,
        "workers": args.workers,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "preflight_summary": preflight_summary,
        "pilot_summary": pilot_summary,
        "prompt_sha256": sha256_file(prompt_path),
        "sampling_manifest": data["sampling_manifest"],
        "manual_startup_health_chat_completion_calls_before_script": 1,
        "gold_test_labels_read": False,
        "gold_test_labels_used_for_prompt_or_selection": False,
        "80_confirmed_used_for_prompt_or_selection": False,
        "train_or_finetune_executed": False,
        "structure_extraction_executed": False,
        "external_llm_calls": 0,
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)
    write_sha_manifest(args.output_dir)
    print(json.dumps({"status": status, "preflight": preflight_summary, "pilot": pilot_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
