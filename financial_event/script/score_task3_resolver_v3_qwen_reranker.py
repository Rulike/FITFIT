#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/task3_resolver_v3_candidate"
R3_DEV_INPUT = OUT / "r3_ann_dev_input.json"
DEV_CANDIDATES = OUT / "r_channel_candidate_pairs_dev.parquet"
MODEL_PATH = ROOT.parent / "downloads/Qwen/Qwen3-Reranker-0.6B"
SCORES_PARQUET = OUT / "qwen_reranker_dev_scores.parquet"
SCORES_CSV = OUT / "qwen_reranker_dev_scores.csv"
SCORES_REPORT = OUT / "qwen_reranker_dev_scores_report.json"


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    raw = str(value).strip()
    return "" if raw.lower() in {"none", "nan", "null", "nat"} else raw


def pair_key(a: Any, b: Any) -> tuple[str, str]:
    return tuple(sorted((text(a), text(b))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_pairs(limit: int | None = None) -> tuple[dict[str, str], list[tuple[str, str]]]:
    payload = json.loads(R3_DEV_INPUT.read_text(encoding="utf-8"))
    event_text = {text(row["event_uid"]): text(row["text"]) for row in payload["events"]}
    candidate_df = pd.read_parquet(DEV_CANDIDATES)
    pairs = [pair_key(row.sample_id_a, row.sample_id_b) for row in candidate_df.itertuples(index=False)]
    pairs = sorted(set(pairs))
    if limit is not None:
        pairs = pairs[:limit]
    missing = [key for key in pairs if key[0] not in event_text or key[1] not in event_text]
    if missing:
        raise RuntimeError(f"missing event text for {len(missing)} candidate pairs")
    return event_text, pairs


def build_prompt(left_text: str, right_text: str, max_chars: int) -> str:
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the two structured financial event mentions describe the same real-world event. "
        "They must match on target, time, amount, project/product/object, and event state. "
        "Answer only yes or no.<|im_end|>\n"
        "<|im_start|>user\n"
    )
    body = f"Event A:\n{left_text[:max_chars]}\n\nEvent B:\n{right_text[:max_chars]}"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prefix + body + suffix


@torch.no_grad()
def score_batch(model: Any, tokenizer: Any, prompts: list[str], yes_id: int, no_id: int, max_length: int) -> list[float]:
    encoded = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    logits = model(**encoded).logits[:, -1, :]
    yes = logits[:, yes_id]
    no = logits[:, no_id]
    stacked = torch.stack([no, yes], dim=1)
    return torch.softmax(stacked, dim=1)[:, 1].detach().cpu().float().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-chars-per-event", type=int, default=1100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if SCORES_PARQUET.exists() and SCORES_REPORT.exists() and not args.force:
        report = json.loads(SCORES_REPORT.read_text(encoding="utf-8"))
        print(json.dumps({"status": "cache_hit", **report}, ensure_ascii=False, indent=2))
        return

    started = time.perf_counter()
    event_text, pairs = load_pairs(args.limit or None)
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), padding_side="left", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=dtype,
        local_files_only=True,
    ).eval()
    if cuda:
        model = model.to("cuda")
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")

    rows: list[dict[str, Any]] = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        prompts = [build_prompt(event_text[a], event_text[b], args.max_chars_per_event) for a, b in batch]
        scores = score_batch(model, tokenizer, prompts, yes_id, no_id, args.max_length)
        for (a, b), score in zip(batch, scores):
            rows.append(
                {
                    "sample_id_a": a,
                    "sample_id_b": b,
                    "qwen_reranker_same_score": round(float(score), 8),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(SCORES_PARQUET, index=False)
    frame.to_csv(SCORES_CSV, index=False, encoding="utf-8-sig")
    report = {
        "status": "qwen_reranker_dev_scores_complete",
        "candidate_pairs_scored": len(rows),
        "model_path": str(MODEL_PATH),
        "model_safetensors_sha256": sha256_file(MODEL_PATH / "model.safetensors"),
        "device": "cuda" if cuda else "cpu",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_chars_per_event": args.max_chars_per_event,
        "official1799_read": 0,
        "final_holdout_read": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "score_summary": {
            "min": round(float(frame["qwen_reranker_same_score"].min()), 8) if len(frame) else None,
            "mean": round(float(frame["qwen_reranker_same_score"].mean()), 8) if len(frame) else None,
            "max": round(float(frame["qwen_reranker_same_score"].max()), 8) if len(frame) else None,
        },
    }
    write_json(SCORES_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
