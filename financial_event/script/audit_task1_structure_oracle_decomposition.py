#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 structure oracle decomposition and root-cause audit.

This script performs no training. It only reads the confirmed 850/90 structure
data, the frozen short-field PP-UIE checkpoint, and the existing Dev90
predictions/report artifacts to decompose the failure into:

1. actual sentence selector + actual short-field model,
2. oracle sentence + actual short-field model,
3. oracle sentence + oracle short fields + real post-processing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_7f_structure_training import evaluate_predictions  # noqa: E402
from scripts.run_task1_step5_7s_ppuie_multisample_memory import (  # noqa: E402
    DIRS,
    FIELDS,
    MODEL_DIR,
    NO_ENTITY,
    ROLES,
    TRAIN850,
    first_generated_line,
    generate_answer,
    normalize_time_for_memory,
    render_official_prompt,
    sha256_file,
)
from scripts.run_task1_step5_7u_event_span_hybrid import normalize_time_u, split_sentences  # noqa: E402
from scripts.run_task1_step5_7v_sentence_selector import (  # noqa: E402
    build_lexicons,
    gold_window,
    make_windows,
    rank_rule,
)
from scripts.train_task1_step5_7w_structure_850 import (  # noqa: E402
    constrained_locate,
    field_empty,
    load_ppuie_checkpoint,
    predict_direction,
    trim_event_span,
)


DEV90 = ROOT / "outputs/task1_step5_7n_chatgpt500_confirmation/merged_structure_dev_90.jsonl"
REPORT_JSON = ROOT / "reports/task1_structure_oracle_decomposition_report.json"
REPORT_MD = ROOT / "reports/task1_structure_oracle_decomposition_report.md"
DETAILS_XLSX = ROOT / "reports/task1_structure_oracle_decomposition_details.xlsx"
OUT_DIR = ROOT / "outputs/task1_structure_oracle_decomposition_v1"
FIXED_PRED = ROOT / "outputs/task1_step5_7w_structure_850/dev_predictions.jsonl"
FIXED_DETAILS = ROOT / "outputs/task1_step5_7w_structure_850/dev_boundary_details.jsonl"
FIXED_REPORT = ROOT / "reports/task1_step5_7w_structure_850_dev_report.json"
SHORT_CKPT = ROOT / "outputs/task1_step5_7w_structure_850/frozen_best_model/ppuie_short_fields_850"
DIRECTION_MODEL = ROOT / "outputs/task1_step5_7w_structure_850/models/direction_classifier.joblib"
FROZEN60 = ROOT / "outputs/task1_step5_7j_structure_confirmation/independent_frozen_test_confirmed.jsonl"
FROZEN60_EXPECTED_SHA = "76efd5a27057dd177ee8688f4769bfb6e57e8539795cad1727ea29418f7c3579"
TASK1234 = Path("/opt/data/private/JinRong/kechuang/task1234_delivery_20260811.tar.gz")

FORMAL_PROTECTED = [
    ROOT / "outputs/final_task1/final_events_train.jsonl",
    ROOT / "outputs/final_task1/final_events_test.jsonl",
    ROOT / "outputs/final_task1/final_events_train.xlsx",
    ROOT / "outputs/final_task1/final_events_test.xlsx",
    ROOT / "outputs/final_task1/task1_completion_certificate.json",
    ROOT / "outputs/event_credibility/llm_review_results.jsonl",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet",
    ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.xlsx",
    TASK1234,
]

SHORT_FIELD_MAX_NEW = {"trigger": 32, "subject_company": 48, "event_time": 32, "object_target": 64}
SHORT_FIELD_NAMES = ["trigger", "subject_company", "event_time", "object_target"]
EVENT_NAME_BUCKETS = [(3, "head_gt30"), (10, "mid_4_to_10"), (30, "tail_11_to_30"), (10**9, "long_tail_le_3")]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha_snapshot(paths: list[Path]) -> dict[str, str]:
    out = {}
    for path in paths:
        if path.exists() and path.is_file():
            out[rel(path)] = sha256_file(path)
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sha_txt(path: Path, snapshot: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(snapshot.items())), encoding="utf-8")


def content_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("content", "")).encode("utf-8")).hexdigest()


def count_sentences(content: str) -> int:
    return len(split_sentences(content or ""))


def sentence_bucket(n: int) -> str:
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11+"


def length_bucket(n: int) -> str:
    if n <= 500:
        return "0-500"
    if n <= 1000:
        return "501-1000"
    if n <= 2000:
        return "1001-2000"
    return "2001+"


def event_tail_bucket(freq: int) -> str:
    if freq <= 3:
        return "long_tail_le_3"
    if freq <= 10:
        return "tail_4_to_10"
    if freq <= 30:
        return "mid_11_to_30"
    return "head_gt30"


def fallback_rank_windows(row: dict[str, Any]) -> list[dict[str, Any]]:
    wins = [dict(win) for win in make_windows(row)]
    for win in wins:
        win["score"] = -0.01 * int(win.get("sentence_index", 0)) - 0.001 * int(win.get("window_size", 1))
        win["selector"] = "fallback_no_event"
    wins.sort(key=lambda x: (-x["score"], len(x["text"]), x["start"]))
    return wins


def process_check() -> dict[str, Any]:
    proc = subprocess.run(
        "ps -eo pid,ppid,stat,etime,cmd | rg -i 'task1_structure_oracle_decomposition|task1_step5_7w_structure_850|DianJin-R1-32B|vllm|python scripts/' | rg -v 'rg -i|codex-linux-sandbox|bwrap' || true",
        shell=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return {"returncode": proc.returncode, "raw": proc.stdout.splitlines()}


def article_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("article_id", "")).strip() for row in rows if str(row.get("article_id", "")).strip()}


def event_name_frequencies(rows: list[dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        name = str(row.get("event_name", "")).strip()
        if name:
            counter[name] += 1
    return counter


def gold_sentence_info(row: dict[str, Any], max_size: int = 3) -> dict[str, Any]:
    span = row.get("event_span", {}) or {}
    gs, ge = int(span.get("start", -1)), int(span.get("end", -1))
    sentences = split_sentences(row.get("content", ""))
    if not sentences:
        return {
            "candidate_id": "W0001_1",
            "sentence_ids": ["S0001"],
            "window_size": 1,
            "text": row.get("content", ""),
            "start": 0,
            "end": len(row.get("content", "")),
            "contains_gold_span": False,
            "gold_char_f1": 0.0,
            "sentence_count": 0,
        }
    best = None
    best_key = (-1.0, -999999)
    for i in range(len(sentences)):
        for size in range(1, max_size + 1):
            j = i + size - 1
            if j >= len(sentences):
                continue
            start, end = sentences[i]["start"], sentences[j]["end"]
            text = row.get("content", "")[start:end]
            contains = start <= gs and ge <= end
            ov = max(0, min(ge, end) - max(gs, start))
            f1 = 0.0
            if gs >= 0 and ge > gs and ov > 0:
                p = ov / max(1, end - start)
                r = ov / max(1, ge - gs)
                f1 = 2 * p * r / max(1e-12, p + r)
            key = (1.0 if contains else f1, -len(text))
            if key > best_key:
                best_key = key
                best = {
                    "candidate_id": f"W{i + 1:04d}_{size}",
                    "sentence_ids": [sentences[k]["sentence_id"] for k in range(i, j + 1)],
                    "window_size": size,
                    "text": text,
                    "start": start,
                    "end": end,
                    "contains_gold_span": contains,
                    "gold_char_f1": round(f1, 6),
                    "sentence_count": len(sentences),
                    "window_sentence_count": size,
                }
    return best or {
        "candidate_id": "",
        "sentence_ids": [],
        "window_size": 1,
        "text": "",
        "start": -1,
        "end": -1,
        "contains_gold_span": False,
        "gold_char_f1": 0.0,
        "sentence_count": len(sentences),
        "window_sentence_count": 0,
    }


def window_contains_gold_span(row: dict[str, Any], win: dict[str, Any]) -> bool:
    span = row.get("event_span", {}) or {}
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    return bool(win) and start >= 0 and end > start and int(win.get("start", 10**9)) <= start and end <= int(win.get("end", -1))


def event_term_hits(event_name: str, text: str) -> int:
    raw = str(event_name or "")
    terms = [raw]
    for sep in ["与", "及", "和", "、", "/", "+", "（", "）", "(", ")", "，", ","]:
        expanded = []
        for item in terms:
            expanded.extend(item.split(sep))
        terms.extend(expanded)
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", raw))
    unique_terms = list(dict.fromkeys(t.strip() for t in terms if len(t.strip()) >= 2))
    return sum(1 for term in unique_terms if term in text)


def locate_text(content: str, text: str, preferred_start: int | None = None, preferred_end: int | None = None) -> dict[str, Any]:
    value = "" if text is None else str(text)
    if not value or value == NO_ENTITY:
        return {"text": "", "start": -1, "end": -1}
    positions: list[int] = []
    start = 0
    while True:
        pos = content.find(value, start)
        if pos < 0:
            break
        positions.append(pos)
        start = pos + 1
    if not positions:
        return {"text": value, "start": -1, "end": -1}
    if preferred_start is not None and preferred_end is not None:
        inside = [p for p in positions if preferred_start <= p and p + len(value) <= preferred_end]
        if len(inside) == 1:
            p = inside[0]
            return {"text": value, "start": p, "end": p + len(value), "resolution": "unique_in_selected_window"}
        if inside:
            p = min(inside, key=lambda x: abs(x - preferred_start))
            return {"text": value, "start": p, "end": p + len(value), "resolution": "multiple_in_selected_window_nearest_start", "ambiguous_count": len(inside)}
    if len(positions) == 1:
        p = positions[0]
        return {"text": value, "start": p, "end": p + len(value), "resolution": "unique_in_full_content"}
    p = min(positions)
    return {"text": value, "start": p, "end": p + len(value), "resolution": "multiple_in_full_content_first", "ambiguous_count": len(positions)}


def sentence_selector_audit(rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    train_yes = [row for row in train_rows if row.get("event_exists", "yes") == "yes"]
    lex = build_lexicons(train_yes)
    # rank_rule expects every seen event_name to have Counter-valued buckets.
    for row in train_yes:
        ev = str(row.get("event_name", "")).strip()
        if not ev:
            continue
        _ = lex["trigger_by_event"][ev]
        _ = lex["subject_by_event"][ev]
        _ = lex["object_by_event"][ev]
    article_counts = Counter(str(row.get("article_id", "")) for row in rows)
    rankings: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("event_name", "")).strip() and row.get("event_exists", "yes") != "no":
            ranked = [dict(win) for win in rank_rule(row, lex)]
        else:
            ranked = fallback_rank_windows(row)
        rankings[row["event_mention_id"]] = ranked
        gold = gold_sentence_info(row)
        top1 = ranked[0] if ranked else {}
        top1_contains = window_contains_gold_span(row, top1)
        top3_hit = any(window_contains_gold_span(row, win) for win in ranked[:3])
        top5_hit = any(window_contains_gold_span(row, win) for win in ranked[:5])
        gold_contains = bool(gold.get("contains_gold_span"))
        gold_hit = top1_contains
        flags: list[str] = []
        primary = "ranking_error"
        if not gold_contains:
            flags.append("gold_span_not_in_candidates")
            if gold.get("window_sentence_count", 0) > 1:
                flags.append("gold_span_cross_sentence")
                primary = "gold_span_cross_sentence"
            else:
                flags.append("sentence_split_error")
                primary = "sentence_split_error"
        elif not gold_hit:
            flags.append("ranking_error")
            primary = "ranking_error"
        else:
            flags.append("top1_correct")
            primary = "top1_correct"

        if article_counts[str(row.get("article_id", ""))] > 1:
            flags.append("multi_event_article")
            if top1 and gold_contains:
                top1_start, top1_end = int(top1.get("start", -1)), int(top1.get("end", -1))
                current_overlap = max(0, min(top1_end, int(row.get("event_span", {}).get("end", -1))) - max(top1_start, int(row.get("event_span", {}).get("start", -1))))
                other_overlap = 0
                for other in rows:
                    if other["event_mention_id"] == row["event_mention_id"] or other.get("article_id") != row.get("article_id"):
                        continue
                    other_span = other.get("event_span", {})
                    ov = max(0, min(top1_end, int(other_span.get("end", -1))) - max(top1_start, int(other_span.get("start", -1))))
                    other_overlap = max(other_overlap, ov)
                if other_overlap > current_overlap:
                    flags.append("multi_event_alignment_error")
                    if primary == "top1_correct":
                        primary = "multi_event_alignment_error"

        gold_text = gold.get("text", "")
        top1_text = str(top1.get("text", ""))
        if gold_contains and top1_text and event_term_hits(str(row.get("event_name", "")), top1_text) > event_term_hits(str(row.get("event_name", "")), gold_text):
            flags.append("event_name_structure_misalignment")
            if primary == "ranking_error":
                primary = "event_name_structure_misalignment"

        details.append(
            {
                "event_mention_id": row.get("event_mention_id", ""),
                "article_id": row.get("article_id", ""),
                "event_name": row.get("event_name", ""),
                "content_length": len(row.get("content", "")),
                "sentence_count": len(split_sentences(row.get("content", ""))),
                "article_event_mention_count": article_counts[str(row.get("article_id", ""))],
                "gold_candidate_id": gold.get("candidate_id", ""),
                "gold_contains_gold_span": gold_contains,
                "gold_window_sentence_ids": ",".join(gold.get("sentence_ids", [])),
                "gold_window_size": gold.get("window_size", 0),
                "gold_window_text": gold.get("text", ""),
                "selected_candidate_id": top1.get("candidate_id", ""),
                "selected_contains_gold_span": top1_contains,
                "selected_window_text": top1.get("text", ""),
                "top1_hit": int(gold_hit),
                "top3_hit": int(top3_hit),
                "top5_hit": int(top5_hit),
                "primary_error": primary,
                "error_flags": "|".join(dict.fromkeys(flags)),
                "event_span_cross_sentence": int(gold.get("window_sentence_count", 0) > 1),
                "gold_char_f1": gold.get("gold_char_f1", 0.0),
                "selector_score_gap": round(float(top1.get("score", 0.0)) - float(gold.get("score", 0.0)), 6) if gold.get("candidate_id") and top1.get("candidate_id") else None,
                "article_len_bucket": length_bucket(len(row.get("content", ""))),
                "sentence_count_bucket": sentence_bucket(len(split_sentences(row.get("content", "")))),
                "event_name_train_freq": 0,
                "event_name_freq_bucket": "",
                "multi_event_article": int(article_counts[str(row.get("article_id", ""))] > 1),
            }
        )
        details[-1]["event_name_train_freq"] = 0  # filled below
    freq = event_name_frequencies(train_yes)
    for item in details:
        item["event_name_train_freq"] = int(freq.get(item["event_name"], 0))
        item["event_name_freq_bucket"] = event_tail_bucket(int(freq.get(item["event_name"], 0)))

    n = max(1, len(rows))
    summary = {
        "sample_count": len(rows),
        "selected_sentence_correct_count": sum(1 for item in details if item["top1_hit"]),
        "sentence_top1_accuracy": round(sum(item["top1_hit"] for item in details) / n, 6),
        "sentence_top3_recall": round(sum(item["top3_hit"] for item in details) / n, 6),
        "sentence_top5_recall": round(sum(item["top5_hit"] for item in details) / n, 6),
        "gold_candidate_missing_count": sum(1 for item in details if "gold_span_not_in_candidates" in item["error_flags"]),
        "gold_span_cross_sentence_count": sum(1 for item in details if "gold_span_cross_sentence" in item["error_flags"]),
        "ranking_error_count": sum(1 for item in details if item["primary_error"] == "ranking_error"),
        "event_name_structure_misalignment_count": sum(1 for item in details if "event_name_structure_misalignment" in item["error_flags"]),
        "multi_event_alignment_error_count": sum(1 for item in details if "multi_event_alignment_error" in item["error_flags"]),
        "sentence_split_error_count": sum(1 for item in details if item["primary_error"] == "sentence_split_error"),
        "single_sentence_gold_count": sum(1 for item in details if item["gold_window_size"] == 1),
        "multi_sentence_gold_count": sum(1 for item in details if item["gold_window_size"] > 1),
        "multi_event_article_count": sum(1 for item in details if item["multi_event_article"]),
        "average_candidate_count": round(sum(len(rankings[item["event_mention_id"]]) for item in details) / n, 3),
        "by_event_name": {},
        "by_article_len_bucket": {},
        "by_sentence_count_bucket": {},
        "by_tail_bucket": {},
        "error_flag_counts": dict(Counter(flag for item in details for flag in item["error_flags"].split("|") if flag)),
        "primary_error_counts": dict(Counter(item["primary_error"] for item in details)),
    }
    for key, group_field in [
        ("by_event_name", "event_name"),
        ("by_article_len_bucket", "article_len_bucket"),
        ("by_sentence_count_bucket", "sentence_count_bucket"),
        ("by_tail_bucket", "event_name_freq_bucket"),
    ]:
        grouped = defaultdict(list)
        for item in details:
            grouped[item[group_field]].append(item)
        result = {}
        for grp, items in grouped.items():
            result[grp] = {
                "count": len(items),
                "top1_accuracy": round(sum(i["top1_hit"] for i in items) / max(1, len(items)), 6),
                "top3_recall": round(sum(i["top3_hit"] for i in items) / max(1, len(items)), 6),
                "top5_recall": round(sum(i["top5_hit"] for i in items) / max(1, len(items)), 6),
                "gold_candidate_missing": sum(1 for i in items if "gold_span_not_in_candidates" in i["error_flags"]),
                "ranking_error": sum(1 for i in items if i["primary_error"] == "ranking_error"),
                "cross_sentence": sum(1 for i in items if "gold_span_cross_sentence" in i["error_flags"]),
                "multi_event_alignment": sum(1 for i in items if "multi_event_alignment_error" in i["error_flags"]),
            }
        summary[key] = result
    summary["rankings"] = rankings
    return summary, details, rankings


def load_base_checkpoint(dtype: str = "float16") -> tuple[Any, Any]:
    return load_ppuie_checkpoint(SHORT_CKPT, dtype)


def predict_short_fields_with_window(
    row: dict[str, Any],
    window: dict[str, Any],
    model: Any,
    tokenizer: Any,
    direction_model: Any | None,
    use_oracle_fields: bool = False,
) -> dict[str, Any]:
    content = row.get("content", "")
    pred: dict[str, Any] = {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "scheme": "oracle_sentence_real_short_fields" if not use_oracle_fields else "oracle_sentence_oracle_short_fields",
        "event_span": {"text": window.get("text", ""), "start": window.get("start", -1), "end": window.get("end", -1)},
    }
    parser_total = 0
    parser_ok = 0
    if use_oracle_fields:
        for field in SHORT_FIELD_NAMES:
            span = row.get(field, {}) or {}
            text = str(span.get("text", "") or "")
            if field == "event_time":
                pred[field] = {
                    "text": text if text and text != NO_ENTITY else "",
                    "start": int(span.get("start", -1)) if text else -1,
                    "end": int(span.get("end", -1)) if text else -1,
                    "normalized": str(span.get("normalized", "未明确") or "未明确"),
                }
            else:
                pred[field] = {
                    "text": text if text and text != NO_ENTITY else "",
                    "start": int(span.get("start", -1)) if text else -1,
                    "end": int(span.get("end", -1)) if text else -1,
                }
    else:
        for field in SHORT_FIELD_NAMES:
            prompt_row = dict(row)
            prompt_row["content"] = window.get("text", "")
            out = generate_answer(
                model,
                tokenizer,
                render_official_prompt(tokenizer, prompt_row, field),
                max_new_tokens=SHORT_FIELD_MAX_NEW[field],
            )
            raw = first_generated_line(out["text"])
            loc = constrained_locate(content, window, raw, field)
            parser_total += 1
            if not raw or raw == NO_ENTITY or (
                loc.get("text")
                and 0 <= int(loc.get("start", -1)) <= int(loc.get("end", -1)) <= len(content)
                and content[int(loc["start"]) : int(loc["end"])] == loc.get("text")
            ):
                parser_ok += 1
            if field == "event_time":
                loc["normalized"] = normalize_time_u(loc.get("text", ""), row.get("publish_time", ""))
            pred[field] = loc if loc.get("text") else field_empty(field)
    if parser_total:
        pred["_parser_counts"] = {"total": parser_total, "ok": parser_ok}
    pred["direction"] = predict_direction(row, window, pred, direction_model)
    args = [
        {"text": pred[role]["text"], "role": role, "start": pred[role]["start"], "end": pred[role]["end"]}
        for role in ROLES
        if pred.get(role, {}).get("text")
    ]
    pred["arguments"] = args
    pred["argument_roles"] = args
    return pred


def build_c_oracle_pred(row: dict[str, Any], window: dict[str, Any], direction_model: Any | None) -> dict[str, Any]:
    pred = predict_short_fields_with_window(row, window, model=None, tokenizer=None, direction_model=direction_model, use_oracle_fields=True)  # type: ignore[arg-type]
    # Deterministic event span construction based on oracle short fields.
    pred["event_span"] = trim_event_span(row, window, pred)
    return pred


def evaluate_with_prediction_rows(rows: list[dict[str, Any]], preds: list[dict[str, Any]], scheme: str) -> dict[str, Any]:
    metrics = evaluate_predictions(rows, preds, scheme)
    text_ok = norm_ok = 0
    for row, pred in zip(rows, preds):
        text_ok += int((row.get("event_time", {}).get("text", "") or "") == (pred.get("event_time", {}).get("text", "") or ""))
        norm_ok += int((row.get("event_time", {}).get("normalized", "未明确") or "未明确") == (pred.get("event_time", {}).get("normalized", "未明确") or "未明确"))
    labels = ["positive", "negative", "neutral", "unclear"]
    y_true = [row.get("direction", "neutral") if row.get("direction", "neutral") in labels else "neutral" for row in rows]
    y_pred = [pred.get("direction", "neutral") if pred.get("direction", "neutral") in labels else "neutral" for pred in preds]
    metrics.update(
        {
            "event_time_text_accuracy": round(text_ok / max(1, len(rows)), 6),
            "event_time_normalized_accuracy": round(norm_ok / max(1, len(rows)), 6),
        }
    )
    parser_total = sum(int(pred.get("_parser_counts", {}).get("total", 0)) for pred in preds)
    parser_ok = sum(int(pred.get("_parser_counts", {}).get("ok", 0)) for pred in preds)
    if parser_total:
        metrics["constrained_parser_preservation"] = round(parser_ok / max(1, parser_total), 6)
    else:
        metrics["constrained_parser_preservation"] = 1.0 if metrics.get("offset_illegal_count", 0) == 0 else 0.0
    return metrics


def build_c_oracle_direction_pred(row: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    pred = {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "scheme": "oracle_sentence_oracle_short_fields_oracle_direction",
        "event_span": trim_event_span(row, window, {
            "trigger": row.get("trigger", {}),
            "subject_company": row.get("subject_company", {}),
            "event_time": row.get("event_time", {}),
            "object_target": row.get("object_target", {}),
        }),
        "trigger": {"text": row.get("trigger", {}).get("text", ""), "start": row.get("trigger", {}).get("start", -1), "end": row.get("trigger", {}).get("end", -1)},
        "subject_company": {"text": row.get("subject_company", {}).get("text", ""), "start": row.get("subject_company", {}).get("start", -1), "end": row.get("subject_company", {}).get("end", -1)},
        "event_time": {
            "text": row.get("event_time", {}).get("text", ""),
            "start": row.get("event_time", {}).get("start", -1),
            "end": row.get("event_time", {}).get("end", -1),
            "normalized": str(row.get("event_time", {}).get("normalized", "未明确") or "未明确"),
        },
        "object_target": {"text": row.get("object_target", {}).get("text", ""), "start": row.get("object_target", {}).get("start", -1), "end": row.get("object_target", {}).get("end", -1)},
    }
    pred["direction"] = row.get("direction", "neutral")
    args = [
        {"text": pred[role]["text"], "role": role, "start": pred[role]["start"], "end": pred[role]["end"]}
        for role in ROLES
        if pred.get(role, {}).get("text")
    ]
    pred["arguments"] = args
    pred["argument_roles"] = args
    return pred


def build_bucket_stats(items: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped = defaultdict(list)
    for item in items:
        grouped[item[field]].append(item)
    return {
        key: {
            "count": len(vals),
            "top1_accuracy": round(sum(v["top1_hit"] for v in vals) / max(1, len(vals)), 6),
            "top3_recall": round(sum(v["top3_hit"] for v in vals) / max(1, len(vals)), 6),
            "top5_recall": round(sum(v["top5_hit"] for v in vals) / max(1, len(vals)), 6),
            "gold_candidate_missing": sum(1 for v in vals if "gold_span_not_in_candidates" in v["error_flags"]),
            "ranking_error": sum(1 for v in vals if v["primary_error"] == "ranking_error"),
            "cross_sentence": sum(1 for v in vals if "gold_span_cross_sentence" in v["error_flags"]),
            "multi_event_alignment": sum(1 for v in vals if "multi_event_alignment_error" in v["error_flags"]),
        }
        for key, vals in grouped.items()
    }


def write_reports(report: dict[str, Any], selector_details: list[dict[str, Any]], c_details: list[dict[str, Any]]) -> None:
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Task1 结构抽取 Oracle 分解与根因审计",
                "",
                f"- status: `{report['status']}`",
                f"- next module: `{report['decision']['next_module']}`",
                f"- A/B/C event_span char F1: `{report['comparison']['A']['event_span_character_overlap_f1']}` / `{report['comparison']['B']['event_span_character_overlap_f1']}` / `{report['comparison']['C']['event_span_character_overlap_f1']}`",
                f"- A/B/C trigger relaxed F1: `{report['comparison']['A']['trigger_relaxed_f1']}` / `{report['comparison']['B']['trigger_relaxed_f1']}` / `{report['comparison']['C']['trigger_relaxed_f1']}`",
                f"- A/B/C subject normalized: `{report['comparison']['A']['subject_normalized_accuracy']}` / `{report['comparison']['B']['subject_normalized_accuracy']}` / `{report['comparison']['C']['subject_normalized_accuracy']}`",
                f"- A/B/C time normalized: `{report['comparison']['A']['event_time_normalized_accuracy']}` / `{report['comparison']['B']['event_time_normalized_accuracy']}` / `{report['comparison']['C']['event_time_normalized_accuracy']}`",
                f"- A/B/C object normalized: `{report['comparison']['A']['object_normalized_accuracy']}` / `{report['comparison']['B']['object_normalized_accuracy']}` / `{report['comparison']['C']['object_normalized_accuracy']}`",
                f"- A/B/C direction accuracy: `{report['comparison']['A']['direction_accuracy']}` / `{report['comparison']['B']['direction_accuracy']}` / `{report['comparison']['C']['direction_accuracy']}`",
                f"- A/B/C argument+role F1: `{report['comparison']['A']['argument_role_end_to_end_f1']}` / `{report['comparison']['B']['argument_role_end_to_end_f1']}` / `{report['comparison']['C']['argument_role_end_to_end_f1']}`",
                f"- A/B/C parser preservation: `{report['comparison']['A']['constrained_parser_preservation']}` / `{report['comparison']['B']['constrained_parser_preservation']}` / `{report['comparison']['C']['constrained_parser_preservation']}`",
                f"- oracle sentence top1/top3/top5: `{report['selector_audit']['sentence_top1_accuracy']}` / `{report['selector_audit']['sentence_top3_recall']}` / `{report['selector_audit']['sentence_top5_recall']}`",
                f"- selected sentence correct count: `{report['selector_audit']['selected_sentence_correct_count']}`",
                f"- gold candidate missing: `{report['selector_audit']['gold_candidate_missing_count']}`",
                f"- ranking error: `{report['selector_audit']['ranking_error_count']}`",
                f"- multi-event alignment error: `{report['selector_audit']['multi_event_alignment_error_count']}`",
                "",
                "本轮未训练、未读取冻结60条、未读取1029 gold，未调用32B/GPT/外部LLM，未修改正式Task1/Task2输出。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(["key", "value"])
    for key in ["status", "decision", "selector_audit", "comparison", "protection"]:
        ws.append([key, json.dumps(report.get(key, {}), ensure_ascii=False)[:32000]])

    ws2 = wb.create_sheet("selector_errors")
    selector_headers = list(selector_details[0].keys()) if selector_details else []
    ws2.append(selector_headers)
    for row in selector_details:
        ws2.append([row.get(h, "") for h in selector_headers])

    ws3 = wb.create_sheet("oracle_comparison")
    ws3.append(["event_mention_id", "A_trigger_f1", "A_span_char", "B_trigger_f1", "B_span_char", "C_trigger_f1", "C_span_char", "A_article_exact", "B_article_exact", "C_article_exact"])
    rows_by_id = {row["event_mention_id"]: row for row in report["comparison_rows"]}
    for row in report["comparison_rows"]:
        ws3.append([
            row["event_mention_id"],
            row["A"]["trigger_relaxed_f1"],
            row["A"]["event_span_character_overlap_f1"],
            row["B"]["trigger_relaxed_f1"],
            row["B"]["event_span_character_overlap_f1"],
            row["C"]["trigger_relaxed_f1"],
            row["C"]["event_span_character_overlap_f1"],
            row["A"]["article_exact_match_accuracy"],
            row["B"]["article_exact_match_accuracy"],
            row["C"]["article_exact_match_accuracy"],
        ])

    ws4 = wb.create_sheet("c_details")
    ws4.append(["event_mention_id", "article_id", "event_name", "trigger", "subject", "time_text", "time_norm", "object", "direction", "event_span", "parser_status"])
    for row in c_details:
        pred = row["pred"]
        ws4.append([
            row["event_mention_id"],
            row["article_id"],
            row["event_name"],
            pred.get("trigger", {}).get("text", ""),
            pred.get("subject_company", {}).get("text", ""),
            pred.get("event_time", {}).get("text", ""),
            pred.get("event_time", {}).get("normalized", ""),
            pred.get("object_target", {}).get("text", ""),
            pred.get("direction", ""),
            pred.get("event_span", {}).get("text", ""),
            row.get("parser_status", ""),
        ])

    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
        sheet.column_dimensions["A"].width = 28
        sheet.column_dimensions["B"].width = 28
        sheet.column_dimensions["C"].width = 28
        sheet.column_dimensions["D"].width = 45
        sheet.column_dimensions["E"].width = 45
    DETAILS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DETAILS_XLSX)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--backup-dir", default="")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(args.backup_dir) if args.backup_dir else Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_structure_oracle_decomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    required_inputs = [TRAIN850, DEV90, FIXED_PRED, FIXED_DETAILS, FIXED_REPORT, SHORT_CKPT / "model_state.pdparams", SHORT_CKPT / "config.json", DIRECTION_MODEL, TASK1234]
    before = sha_snapshot([*FORMAL_PROTECTED, TRAIN850, DEV90, FIXED_PRED, FIXED_DETAILS, FIXED_REPORT, SHORT_CKPT / "model_state.pdparams", SHORT_CKPT / "config.json", SHORT_CKPT / "generation_config.json", DIRECTION_MODEL, FROZEN60])
    write_sha_txt(OUT_DIR / "protected_sha256_before.txt", before)
    write_sha_txt(backup_dir / "protected_sha256_before.txt", before)

    if before.get(rel(FROZEN60)) != FROZEN60_EXPECTED_SHA:
        write_json(REPORT_JSON, {"status": "blocked_by_data_integrity_failure", "reason": "frozen60 SHA mismatch", "actual": before.get(rel(FROZEN60)), "expected": FROZEN60_EXPECTED_SHA})
        return 2
    missing = [path for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {[str(p) for p in missing]}")

    proc = process_check()
    train_rows = read_jsonl(TRAIN850)
    dev_rows = read_jsonl(DEV90)
    if len(dev_rows) != 90:
        raise RuntimeError(f"unexpected Dev90 count: {len(dev_rows)}")
    if len(train_rows) != 850:
        raise RuntimeError(f"unexpected train850 count: {len(train_rows)}")

    train_ids = article_ids(train_rows)
    dev_ids = article_ids(dev_rows)
    overlap = sorted(train_ids & dev_ids)
    content_hash_overlap = sorted({content_hash(row) for row in train_rows} & {content_hash(row) for row in dev_rows})
    if overlap or content_hash_overlap:
        raise RuntimeError(f"data leakage between train and dev: article_overlap={len(overlap)} content_hash_overlap={len(content_hash_overlap)}")

    selector_audit, selector_details, rankings = sentence_selector_audit(dev_rows, train_rows)

    a_preds = read_jsonl(FIXED_PRED)
    if len(a_preds) != 90:
        raise RuntimeError(f"unexpected A prediction count: {len(a_preds)}")
    a_metrics = evaluate_with_prediction_rows(dev_rows, a_preds, "A_actual_pipeline")
    fixed_report = read_json(FIXED_REPORT)
    a_metrics["constrained_parser_preservation"] = fixed_report.get("dev_metrics_trimmed", {}).get(
        "constrained_parser_preservation",
        fixed_report.get("training", {}).get("epoch_summaries", [{}])[-1].get("dev_metrics", {}).get("constrained_parser_preservation", a_metrics.get("constrained_parser_preservation", 0.0)),
    )
    a_metrics["raw_parser_preservation"] = fixed_report.get("dev_metrics_trimmed", {}).get("raw_parser_preservation", 0.0)

    tokenizer, model = load_base_checkpoint(args.dtype)
    direction_model = joblib.load(DIRECTION_MODEL)
    train_yes = [row for row in train_rows if row.get("event_exists", "yes") == "yes"]

    b_preds = []
    b_details = []
    c_preds = []
    c_oracle_direction_preds = []
    for row in dev_rows:
        gw = gold_window(row)
        b_pred = predict_short_fields_with_window(row, gw, model, tokenizer, direction_model, use_oracle_fields=False)
        b_preds.append(b_pred)
        b_details.append({"event_mention_id": row["event_mention_id"], "article_id": row["article_id"], "event_name": row["event_name"], "window": gw, "pred": b_pred})

        c_pred = predict_short_fields_with_window(row, gw, model, tokenizer, direction_model, use_oracle_fields=True)
        c_pred["event_span"] = trim_event_span(row, gw, c_pred)
        c_preds.append(c_pred)
        c_oracle_direction = build_c_oracle_direction_pred(row, gw)
        c_oracle_direction_preds.append(c_oracle_direction)

    b_metrics = evaluate_with_prediction_rows(dev_rows, b_preds, "B_oracle_sentence_real_short_fields")
    c_metrics = evaluate_with_prediction_rows(dev_rows, c_preds, "C_oracle_sentence_oracle_short_fields_real_postprocess")
    c_oracle_dir_metrics = evaluate_with_prediction_rows(dev_rows, c_oracle_direction_preds, "C_oracle_sentence_oracle_short_fields_oracle_direction")

    a_score = a_metrics.get("trigger_relaxed_f1", 0.0) + a_metrics.get("event_span_character_overlap_f1", 0.0)
    b_score = b_metrics.get("trigger_relaxed_f1", 0.0) + b_metrics.get("event_span_character_overlap_f1", 0.0)
    c_score = c_metrics.get("trigger_relaxed_f1", 0.0) + c_metrics.get("event_span_character_overlap_f1", 0.0)

    if b_metrics.get("trigger_relaxed_f1", 0.0) >= 0.70 and b_metrics.get("event_span_character_overlap_f1", 0.0) >= 0.65 and b_metrics.get("subject_normalized_accuracy", 0.0) >= 0.60 and b_metrics.get("event_time_normalized_accuracy", 0.0) >= 0.55 and b_metrics.get("object_normalized_accuracy", 0.0) >= 0.50 and b_metrics.get("direction_accuracy", 0.0) >= 0.80 and b_metrics.get("argument_role_end_to_end_f1", 0.0) >= 0.40 and b_metrics.get("article_exact_match_accuracy", 0.0) >= 0.10 and b_metrics.get("offset_illegal_count", 0) == 0:
        next_module = "event_sentence_selector_primary_bottleneck"
    elif b_score >= a_score + 0.20:
        next_module = "short_field_model_or_training_contract_primary_bottleneck"
    else:
        next_module = "short_field_model_or_training_contract_primary_bottleneck"

    if (
        c_metrics.get("event_span_character_overlap_f1", 0.0) < 0.90
        or c_metrics.get("direction_accuracy", 0.0) < 0.90
        or c_metrics.get("constrained_parser_preservation", 0.0) < 0.99
    ):
        postprocess_note = "postprocess_offset_direction_or_assembly_loss_remains"
    else:
        postprocess_note = "postprocess_gap_small"

    comparison_rows = []
    by_id_a = {row["event_mention_id"]: row for row in a_preds}
    by_id_b = {row["event_mention_id"]: row for row in b_preds}
    by_id_c = {row["event_mention_id"]: row for row in c_preds}
    for row in dev_rows:
        mid = row["event_mention_id"]
        top1_correct = 0
        gold = gold_sentence_info(row)
        if selector_audit["rankings"].get(mid):
            top1_correct = int(window_contains_gold_span(row, selector_audit["rankings"][mid][0]))
        comparison_rows.append(
            {
                "event_mention_id": mid,
                "article_id": row.get("article_id", ""),
                "event_name": row.get("event_name", ""),
                "selector_top1_correct": top1_correct,
                "A": {
                    "trigger_relaxed_f1": evaluate_predictions([row], [by_id_a[mid]], "A_row").get("trigger_relaxed_f1", 0.0),
                    "event_span_character_overlap_f1": evaluate_predictions([row], [by_id_a[mid]], "A_row").get("event_span_character_overlap_f1", 0.0),
                    "subject_normalized_accuracy": evaluate_predictions([row], [by_id_a[mid]], "A_row").get("subject_normalized_accuracy", 0.0),
                    "event_time_normalized_accuracy": evaluate_predictions([row], [by_id_a[mid]], "A_row").get("event_time_normalized_date_accuracy", 0.0),
                    "object_normalized_accuracy": evaluate_predictions([row], [by_id_a[mid]], "A_row").get("object_normalized_accuracy", 0.0),
                    "direction_accuracy": evaluate_predictions([row], [by_id_a[mid]], "A_row").get("direction_accuracy", 0.0),
                    "argument_role_end_to_end_f1": evaluate_predictions([row], [by_id_a[mid]], "A_row").get("argument_role_end_to_end_f1", 0.0),
                    "article_exact_match_accuracy": evaluate_predictions([row], [by_id_a[mid]], "A_row").get("article_exact_match_accuracy", 0.0),
                },
                "B": {
                    "trigger_relaxed_f1": evaluate_predictions([row], [by_id_b[mid]], "B_row").get("trigger_relaxed_f1", 0.0),
                    "event_span_character_overlap_f1": evaluate_predictions([row], [by_id_b[mid]], "B_row").get("event_span_character_overlap_f1", 0.0),
                    "subject_normalized_accuracy": evaluate_predictions([row], [by_id_b[mid]], "B_row").get("subject_normalized_accuracy", 0.0),
                    "event_time_normalized_accuracy": evaluate_predictions([row], [by_id_b[mid]], "B_row").get("event_time_normalized_date_accuracy", 0.0),
                    "object_normalized_accuracy": evaluate_predictions([row], [by_id_b[mid]], "B_row").get("object_normalized_accuracy", 0.0),
                    "direction_accuracy": evaluate_predictions([row], [by_id_b[mid]], "B_row").get("direction_accuracy", 0.0),
                    "argument_role_end_to_end_f1": evaluate_predictions([row], [by_id_b[mid]], "B_row").get("argument_role_end_to_end_f1", 0.0),
                    "article_exact_match_accuracy": evaluate_predictions([row], [by_id_b[mid]], "B_row").get("article_exact_match_accuracy", 0.0),
                },
                "C": {
                    "trigger_relaxed_f1": evaluate_predictions([row], [by_id_c[mid]], "C_row").get("trigger_relaxed_f1", 0.0),
                    "event_span_character_overlap_f1": evaluate_predictions([row], [by_id_c[mid]], "C_row").get("event_span_character_overlap_f1", 0.0),
                    "subject_normalized_accuracy": evaluate_predictions([row], [by_id_c[mid]], "C_row").get("subject_normalized_accuracy", 0.0),
                    "event_time_normalized_accuracy": evaluate_predictions([row], [by_id_c[mid]], "C_row").get("event_time_normalized_date_accuracy", 0.0),
                    "object_normalized_accuracy": evaluate_predictions([row], [by_id_c[mid]], "C_row").get("object_normalized_accuracy", 0.0),
                    "direction_accuracy": evaluate_predictions([row], [by_id_c[mid]], "C_row").get("direction_accuracy", 0.0),
                    "argument_role_end_to_end_f1": evaluate_predictions([row], [by_id_c[mid]], "C_row").get("argument_role_end_to_end_f1", 0.0),
                    "article_exact_match_accuracy": evaluate_predictions([row], [by_id_c[mid]], "C_row").get("article_exact_match_accuracy", 0.0),
                },
            }
        )

    protection = {
        "protected_sha_match": False,
        "independent_frozen60_labels_read": False,
        "gold1029_labels_read": False,
        "local_32b_calls": 0,
        "gpt_or_external_llm_calls": 0,
        "task1234_sha_before": before.get(rel(TASK1234)),
        "task1234_sha_after": None,
        "frozen60_sha_before": before.get(rel(FROZEN60)),
        "frozen60_sha_after": None,
    }

    report = {
        "generated_at": now(),
        "status": "oracle_decomposition_complete",
        "backup_dir": str(backup_dir),
        "data_integrity": {
            "train_count": len(train_rows),
            "dev_count": len(dev_rows),
            "train_unique_articles": len(train_ids),
            "dev_unique_articles": len(dev_ids),
            "article_overlap": len(overlap),
            "content_hash_overlap": len(content_hash_overlap),
            "frozen60_sha_before": before.get(rel(FROZEN60)),
            "frozen60_sha_expected": FROZEN60_EXPECTED_SHA,
        },
        "selector_audit": {k: v for k, v in selector_audit.items() if k != "rankings"},
        "comparison": {
            "A": a_metrics,
            "B": b_metrics,
            "C": c_metrics,
            "C_oracle_direction": c_oracle_dir_metrics,
            "delta_B_minus_A": {
                "trigger_relaxed_f1": round(b_metrics.get("trigger_relaxed_f1", 0.0) - a_metrics.get("trigger_relaxed_f1", 0.0), 6),
                "event_span_character_overlap_f1": round(b_metrics.get("event_span_character_overlap_f1", 0.0) - a_metrics.get("event_span_character_overlap_f1", 0.0), 6),
                "subject_normalized_accuracy": round(b_metrics.get("subject_normalized_accuracy", 0.0) - a_metrics.get("subject_normalized_accuracy", 0.0), 6),
                "event_time_normalized_accuracy": round(b_metrics.get("event_time_normalized_accuracy", 0.0) - a_metrics.get("event_time_normalized_accuracy", 0.0), 6),
                "object_normalized_accuracy": round(b_metrics.get("object_normalized_accuracy", 0.0) - a_metrics.get("object_normalized_accuracy", 0.0), 6),
                "direction_accuracy": round(b_metrics.get("direction_accuracy", 0.0) - a_metrics.get("direction_accuracy", 0.0), 6),
                "argument_role_end_to_end_f1": round(b_metrics.get("argument_role_end_to_end_f1", 0.0) - a_metrics.get("argument_role_end_to_end_f1", 0.0), 6),
                "article_exact_match_accuracy": round(b_metrics.get("article_exact_match_accuracy", 0.0) - a_metrics.get("article_exact_match_accuracy", 0.0), 6),
            },
            "delta_C_minus_B": {
                "trigger_relaxed_f1": round(c_metrics.get("trigger_relaxed_f1", 0.0) - b_metrics.get("trigger_relaxed_f1", 0.0), 6),
                "event_span_character_overlap_f1": round(c_metrics.get("event_span_character_overlap_f1", 0.0) - b_metrics.get("event_span_character_overlap_f1", 0.0), 6),
                "subject_normalized_accuracy": round(c_metrics.get("subject_normalized_accuracy", 0.0) - b_metrics.get("subject_normalized_accuracy", 0.0), 6),
                "event_time_normalized_accuracy": round(c_metrics.get("event_time_normalized_accuracy", 0.0) - b_metrics.get("event_time_normalized_accuracy", 0.0), 6),
                "object_normalized_accuracy": round(c_metrics.get("object_normalized_accuracy", 0.0) - b_metrics.get("object_normalized_accuracy", 0.0), 6),
                "direction_accuracy": round(c_metrics.get("direction_accuracy", 0.0) - b_metrics.get("direction_accuracy", 0.0), 6),
                "argument_role_end_to_end_f1": round(c_metrics.get("argument_role_end_to_end_f1", 0.0) - b_metrics.get("argument_role_end_to_end_f1", 0.0), 6),
                "article_exact_match_accuracy": round(c_metrics.get("article_exact_match_accuracy", 0.0) - b_metrics.get("article_exact_match_accuracy", 0.0), 6),
            },
        },
        "comparison_rows": comparison_rows,
        "decision": {
            "next_module": next_module,
            "postprocess_note": postprocess_note,
            "rationale": {
                "selector_improves_sentence_layer": b_metrics.get("trigger_relaxed_f1", 0.0) > a_metrics.get("trigger_relaxed_f1", 0.0) + 0.10 or b_metrics.get("event_span_character_overlap_f1", 0.0) > a_metrics.get("event_span_character_overlap_f1", 0.0) + 0.10,
                "short_field_model_still_below_gate": b_metrics.get("trigger_relaxed_f1", 0.0) < 0.70 or b_metrics.get("subject_normalized_accuracy", 0.0) < 0.60 or b_metrics.get("object_normalized_accuracy", 0.0) < 0.50,
                "postprocess_still_losses": c_metrics.get("constrained_parser_preservation", 0.0) < 0.99 or c_metrics.get("article_exact_match_accuracy", 0.0) < 0.10,
            },
        },
        "protection": protection,
    }

    write_jsonl(OUT_DIR / "selector_audit_rows.jsonl", selector_details)
    write_jsonl(OUT_DIR / "A_actual_pipeline_predictions.jsonl", a_preds)
    write_jsonl(OUT_DIR / "B_oracle_sentence_real_short_fields_predictions.jsonl", b_preds)
    write_jsonl(OUT_DIR / "C_oracle_sentence_oracle_short_fields_real_postprocess_predictions.jsonl", c_preds)
    write_jsonl(OUT_DIR / "selector_rankings.jsonl", [
        {
            "event_mention_id": row["event_mention_id"],
            "article_id": row["article_id"],
            "event_name": row["event_name"],
            "ranked": selector_audit["rankings"].get(row["event_mention_id"], [])[:8],
        }
        for row in dev_rows
    ])

    write_reports(report, selector_details, [{"event_mention_id": row["event_mention_id"], "article_id": row["article_id"], "event_name": row["event_name"], "pred": pred, "parser_status": ("ok" if pred.get("event_span", {}).get("text", "") and pred.get("event_span", {}).get("start", -1) >= 0 else "empty")} for row, pred in zip(dev_rows, c_preds)])

    after = sha_snapshot([*FORMAL_PROTECTED, TRAIN850, DEV90, FIXED_PRED, FIXED_DETAILS, FIXED_REPORT, SHORT_CKPT / "model_state.pdparams", SHORT_CKPT / "config.json", SHORT_CKPT / "generation_config.json", DIRECTION_MODEL, FROZEN60])
    protection["protected_sha_match"] = before == after
    protection["task1234_sha_after"] = after.get(rel(TASK1234))
    protection["frozen60_sha_after"] = after.get(rel(FROZEN60))
    write_json(REPORT_JSON, report)
    write_json(backup_dir / "run_manifest.json", report)
    write_sha_txt(OUT_DIR / "protected_sha256_after.txt", after)
    write_sha_txt(backup_dir / "protected_sha256_after.txt", after)
    sha_targets = [
        REPORT_JSON,
        REPORT_MD,
        DETAILS_XLSX,
        OUT_DIR / "selector_audit_rows.jsonl",
        OUT_DIR / "selector_rankings.jsonl",
        OUT_DIR / "A_actual_pipeline_predictions.jsonl",
        OUT_DIR / "B_oracle_sentence_real_short_fields_predictions.jsonl",
        OUT_DIR / "C_oracle_sentence_oracle_short_fields_real_postprocess_predictions.jsonl",
        OUT_DIR / "protected_sha256_before.txt",
        OUT_DIR / "protected_sha256_after.txt",
    ]
    write_sha_txt(OUT_DIR / "SHA256SUMS.txt", sha_snapshot(sha_targets))
    print(json.dumps({"status": report["status"], "next_module": report["decision"]["next_module"], "sentence_top1": report["selector_audit"]["sentence_top1_accuracy"], "B_trigger_relaxed": report["comparison"]["B"]["trigger_relaxed_f1"], "C_trigger_relaxed": report["comparison"]["C"]["trigger_relaxed_f1"], "protected_sha_match": report["protection"]["protected_sha_match"], "backup_dir": str(backup_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
