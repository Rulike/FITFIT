#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train/evaluate a lightweight MacBERT sentence selector for Task1 Step 5.7-V.

Run in an environment with torch and transformers.  It only reads the 850-row
training file and holds out the fixed 32 rows used by Step 5.7-S/T/U.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
TRAIN850 = ROOT / "outputs/task1_step5_7n_chatgpt500_confirmation/merged_structure_train_850.jsonl"
LOCAL_MACBERT = ROOT / "outputs/task1_step5_4_dev_baseline_v1/pretrained/hfl_chinese_macbert_base"
OUT_DIR = ROOT / "outputs/task1_step5_7v_sentence_selector"
MODEL_DIR = OUT_DIR / "models/macbert_sentence_selector"
RANKINGS = OUT_DIR / "macbert_selector_rankings.jsonl"
REPORT = OUT_DIR / "macbert_selector_report.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_sentences(content: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    sid = 1
    pattern = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", re.M)
    for m in pattern.finditer(content or ""):
        text = m.group(0)
        if not text:
            continue
        raw_start, raw_end = m.start(), m.end()
        stripped = text.strip()
        if not stripped:
            continue
        leading = len(text) - len(text.lstrip())
        trailing = len(text.rstrip())
        start = raw_start + leading
        end = raw_start + trailing
        sentences.append({"sentence_id": f"S{sid:04d}", "text": content[start:end], "start": start, "end": end})
        sid += 1
    if not sentences and content:
        sentences.append({"sentence_id": "S0001", "text": content, "start": 0, "end": len(content)})
    return sentences


def pick_memory(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if row.get("event_exists", "yes") == "yes"
        and row.get("event_name")
        and row.get("trigger", {}).get("text")
        and row.get("event_span", {}).get("text")
        and len(row.get("content", "")) <= 2500
    ]
    usable.sort(key=lambda row: (len(row.get("content", "")), row.get("event_mention_id", "")))
    if len(usable) < n:
        raise RuntimeError(f"insufficient fixed memory rows: {len(usable)}")
    return usable[:n]


def make_windows(row: dict[str, Any], max_size: int = 3) -> list[dict[str, Any]]:
    content = row.get("content", "")
    sentences = split_sentences(content)
    windows: list[dict[str, Any]] = []
    seen = set()
    for i in range(len(sentences)):
        for size in range(1, max_size + 1):
            j = i + size - 1
            if j >= len(sentences):
                continue
            start, end = sentences[i]["start"], sentences[j]["end"]
            if (start, end) in seen:
                continue
            seen.add((start, end))
            windows.append(
                {
                    "candidate_id": f"W{i + 1:04d}_{size}",
                    "sentence_ids": [sentences[k]["sentence_id"] for k in range(i, j + 1)],
                    "window_size": size,
                    "text": content[start:end],
                    "start": start,
                    "end": end,
                    "sentence_index": i,
                }
            )
    return windows


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def char_f1(gold: dict[str, Any], pred: dict[str, Any]) -> float:
    gs, ge = int(gold.get("start", -1)), int(gold.get("end", -1))
    ps, pe = int(pred.get("start", -1)), int(pred.get("end", -1))
    if gs < 0 or ge <= gs or ps < 0 or pe <= ps:
        return 0.0
    ov = overlap_len(gs, ge, ps, pe)
    if ov <= 0:
        return 0.0
    p = ov / max(1, pe - ps)
    r = ov / max(1, ge - gs)
    return 2 * p * r / max(1e-12, p + r)


def gold_window(row: dict[str, Any]) -> dict[str, Any]:
    gold = row.get("event_span", {})
    gs, ge = int(gold.get("start", -1)), int(gold.get("end", -1))
    best, best_key = None, (-1.0, -999999)
    for win in make_windows(row):
        contains = win["start"] <= gs and ge <= win["end"]
        f1 = char_f1(gold, win)
        key = (1.0 if contains else f1, -len(win["text"]))
        if key > best_key:
            best, best_key = dict(win), key
            best["contains_gold_span"] = contains
            best["gold_char_f1"] = round(f1, 6)
    return best or {"candidate_id": "", "start": -1, "end": -1}


def event_terms(event_name: str) -> list[str]:
    raw = str(event_name or "")
    terms = [raw]
    for sep in ["与", "及", "和", "、", "/", "+", "（", "）", "(", ")", "，", ","]:
        parts = []
        for item in terms:
            parts.extend(item.split(sep))
        terms.extend(parts)
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", raw))
    return list(dict.fromkeys(t.strip() for t in terms if len(t.strip()) >= 2))


ACTION_WORDS = ["发布", "推出", "启动", "签约", "中标", "获批", "增长", "下降", "突破", "创新", "收购", "投资", "融资", "上市", "建设", "升级", "处罚", "亏损", "盈利", "供货", "宣布", "完成", "提升", "进入", "计划", "预计"]


def lexical_prefilter_score(row: dict[str, Any], win: dict[str, Any]) -> float:
    text = win["text"]
    score = 0.0
    for term in event_terms(row.get("event_name", "")):
        if term in text:
            score += len(term) * 0.1
    score += 0.2 * sum(1 for word in ACTION_WORDS if word in text)
    if re.search(r"\d{4}年|\d{1,2}月\d{1,2}日|近日|今日|昨日|当日|上半年|季度", text):
        score += 0.2
    score += max(0, 0.3 - 0.025 * win.get("sentence_index", 0))
    return score


def pair_text(row: dict[str, Any], win: dict[str, Any]) -> str:
    return f"事件名称：{row.get('event_name','')} [SEP] 标题：{row.get('title','')} [SEP] 候选句：{win.get('text','')}"


class PairDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        enc = self.tokenizer(item["text"], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": torch.tensor(item["label"], dtype=torch.long)}


def build_train_items(rows: list[dict[str, Any]], negative_per_row: int) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        wins = make_windows(row)
        if not wins:
            continue
        gw = gold_window(row)
        pos_id = gw.get("candidate_id")
        for win in wins:
            if win["candidate_id"] == pos_id:
                items.append({"text": pair_text(row, win), "label": 1})
                break
        negatives = [win for win in wins if win["candidate_id"] != pos_id]
        negatives.sort(key=lambda w: (-lexical_prefilter_score(row, w), abs(w["start"] - gw.get("start", 0))))
        for win in negatives[:negative_per_row]:
            items.append({"text": pair_text(row, win), "label": 0})
    random.Random(2026).shuffle(items)
    return items


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def train_model(train_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MACBERT), local_files_only=True)
    if (MODEL_DIR / "pytorch_model.bin").exists() or (MODEL_DIR / "model.safetensors").exists():
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True).to(args.device)
        return tokenizer, model, {"trained": False, "model_path": str(MODEL_DIR), "model_sha256": sha256_file(next(MODEL_DIR.glob("pytorch_model.bin")) if (MODEL_DIR / "pytorch_model.bin").exists() else next(MODEL_DIR.glob("model.safetensors")))}
    items = build_train_items(train_rows, args.negative_per_row)
    ds = PairDataset(items, tokenizer, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = AutoModelForSequenceClassification.from_pretrained(str(LOCAL_MACBERT), num_labels=2, local_files_only=True).to(args.device)
    # Keep this intentionally lightweight.
    for name, param in model.named_parameters():
        if "classifier" not in name and "pooler" not in name and not re.search(r"encoder\\.layer\\.(10|11)\\.", name):
            param.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    started = time.time()
    losses = []
    model.train()
    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach().cpu()))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    weights = MODEL_DIR / "model.safetensors"
    if not weights.exists():
        weights = MODEL_DIR / "pytorch_model.bin"
    return tokenizer, model, {
        "trained": True,
        "model_path": str(MODEL_DIR),
        "model_sha256": sha256_file(weights),
        "train_item_count": len(items),
        "positive_count": sum(1 for item in items if item["label"] == 1),
        "negative_count": sum(1 for item in items if item["label"] == 0),
        "epochs": args.epochs,
        "elapsed_seconds": round(time.time() - started, 3),
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def rank_row(row: dict[str, Any], tokenizer: Any, model: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    wins = make_windows(row)
    started = time.time()
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(wins), args.eval_batch_size):
            chunk = wins[start : start + args.eval_batch_size]
            enc = tokenizer([pair_text(row, win) for win in chunk], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
            enc = {k: v.to(args.device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            scores.extend(probs)
    ranked = []
    for win, score in zip(wins, scores):
        item = dict(win)
        item["score"] = float(score)
        item["selector"] = "B_macbert_lightweight_cross_encoder"
        ranked.append(item)
    ranked.sort(key=lambda x: (-x["score"], len(x["text"]), x["start"]))
    return ranked, time.time() - started


def metrics(rows: list[dict[str, Any]], rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    top1 = top3 = top5 = exact = contain = illegal = 0
    f1_sum = 0.0
    lat = []
    for row in rows:
        ranked = rankings[row["event_mention_id"]]
        gold = row.get("event_span", {})
        trig = row.get("trigger", {})
        def hit(win: dict[str, Any]) -> bool:
            return win.get("start", 10**9) <= gold.get("start", -1) and gold.get("end", -1) <= win.get("end", -1)
        if ranked and hit(ranked[0]):
            top1 += 1
        if any(hit(win) for win in ranked[:3]):
            top3 += 1
        if any(hit(win) for win in ranked[:5]):
            top5 += 1
        chosen = ranked[0]
        exact += int(chosen.get("text") == gold.get("text"))
        f1_sum += char_f1(gold, chosen)
        contain += int(chosen.get("start", 10**9) <= trig.get("start", -1) and trig.get("end", -1) <= chosen.get("end", -1))
        illegal += int(row.get("content", "")[chosen["start"] : chosen["end"]] != chosen["text"])
        lat.append(chosen.get("latency_seconds", 0.0))
    n = max(1, len(rows))
    return {
        "sample_count": len(rows),
        "sentence_top1_accuracy": round(top1 / n, 6),
        "sentence_top3_recall": round(top3 / n, 6),
        "sentence_top5_recall": round(top5 / n, 6),
        "event_span_exact": round(exact / n, 6),
        "event_span_char_f1": round(f1_sum / n, 6),
        "trigger_containment": round(contain / n, 6),
        "illegal_offset_count": illegal,
        "average_latency_seconds": round(sum(lat) / n, 6),
        "p95_latency_seconds": round(sorted(lat)[min(n - 1, max(0, math.ceil(n * 0.95) - 1))], 6) if lat else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--negative-per-row", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    rows = read_jsonl(TRAIN850)
    memory32 = pick_memory(rows, 32)
    fixed_ids = {row["event_mention_id"] for row in memory32}
    train_rows = [row for row in rows if row.get("event_mention_id") not in fixed_ids and row.get("event_span", {}).get("text") and row.get("event_exists", "yes") == "yes"]
    tokenizer, model, train_report = train_model(train_rows, args)
    rankings: dict[str, list[dict[str, Any]]] = {}
    for row in memory32:
        ranked, latency = rank_row(row, tokenizer, model, args)
        for item in ranked:
            item["latency_seconds"] = round(latency / max(1, len(ranked)), 6)
            item["contains_gold"] = item["start"] <= row.get("event_span", {}).get("start", -1) and row.get("event_span", {}).get("end", -1) <= item["end"]
            item["gold_char_f1"] = round(char_f1(row.get("event_span", {}), item), 6)
        rankings[row["event_mention_id"]] = ranked
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with RANKINGS.open("w", encoding="utf-8") as f:
        for row in memory32:
            f.write(json.dumps({"event_mention_id": row["event_mention_id"], "ranked": rankings[row["event_mention_id"]]}, ensure_ascii=False) + "\n")
    result = {
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "selector": "B_macbert_lightweight_cross_encoder",
        "model_dir": str(MODEL_DIR),
        "base_model_dir": str(LOCAL_MACBERT),
        "base_model_sha256": sha256_file(LOCAL_MACBERT / "pytorch_model.bin"),
        "train_rows": len(train_rows),
        "fixed32_count": len(memory32),
        "training": train_report,
        "metrics": metrics(memory32, rankings),
        "rankings_path": str(RANKINGS),
        "protection": {"dev90_labels_read": False, "frozen60_labels_read": False, "gold1029_labels_read": False, "full850_structure_training": False},
        "gpu": {
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else 0.0,
        },
    }
    write_json(REPORT, result)
    print(json.dumps({"selector": result["selector"], "metrics": result["metrics"], "model_sha": train_report.get("model_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
