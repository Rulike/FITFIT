#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Listwise/rule sentence ranking utilities for Task1 Step 5.7-X."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from src.event_extraction.task1_sentence_window_candidates import make_sentence_windows, contains_gold_span, char_f1


ACTION_WORDS = [
    "发布", "推出", "启动", "签约", "合作", "中标", "获批", "增长", "下降", "突破", "收购",
    "投资", "融资", "上市", "建设", "升级", "处罚", "亏损", "盈利", "供货", "宣布", "完成",
    "提升", "进入", "计划", "预计", "恢复", "扩产", "投产", "减产", "停产", "涨价", "降价",
]


def event_terms(event_name: str) -> list[str]:
    raw = str(event_name or "")
    terms = [raw]
    for sep in ["与", "及", "和", "、", "/", "+", "（", "）", "(", ")", "，", ","]:
        expanded = []
        for item in terms:
            expanded.extend(item.split(sep))
        terms.extend(expanded)
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", raw))
    return list(dict.fromkeys(term.strip() for term in terms if len(term.strip()) >= 2))


def build_lexicons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lex = {
        "trigger_by_event": defaultdict(Counter),
        "subject_by_event": defaultdict(Counter),
        "object_by_event": defaultdict(Counter),
        "event_support": Counter(),
    }
    for row in rows:
        event_name = str(row.get("event_name", "") or "")
        if not event_name:
            continue
        lex["event_support"][event_name] += 1
        for field, bucket in [("trigger", "trigger_by_event"), ("subject_company", "subject_by_event"), ("object_target", "object_by_event")]:
            text = str((row.get(field) or {}).get("text", "") or "")
            if text:
                lex[bucket][event_name][text] += 1
    return lex


def title_overlap(title: str, text: str) -> float:
    chars = {ch for ch in str(title or "") if "\u4e00" <= ch <= "\u9fff"}
    if not chars:
        return 0.0
    return sum(1 for ch in chars if ch in text) / max(1, len(chars))


def rule_score(row: dict[str, Any], candidate: dict[str, Any], lex: dict[str, Any]) -> float:
    text = str(candidate.get("text", "") or "")
    event_name = str(row.get("event_name", "") or "")
    score = 0.0
    for term in event_terms(event_name):
        if term in text:
            score += min(2.0, len(term) * 0.18)
    for term, count in lex["trigger_by_event"].get(event_name, Counter()).most_common(30):
        if term and term in text:
            score += 3.0 + math.log1p(count)
    for term, count in lex["subject_by_event"].get(event_name, Counter()).most_common(30):
        if term and term in text:
            score += 1.0 + 0.2 * math.log1p(count)
    for term, count in lex["object_by_event"].get(event_name, Counter()).most_common(30):
        if term and term in text:
            score += 0.8 + 0.2 * math.log1p(count)
    score += 0.22 * sum(1 for word in ACTION_WORDS if word in text)
    if re.search(r"\d{4}年|\d{1,2}月\d{1,2}日|近日|今日|昨日|当日|上半年|季度", text):
        score += 0.35
    score += 0.7 * title_overlap(str(row.get("title", "") or ""), text)
    score += max(0.0, 0.35 - 0.025 * int(candidate.get("start", 0)) / 80.0)
    if any(bad in text for bad in ["免责声明", "风险提示", "仅供参考", "文章来源", "责任编辑"]):
        score -= 2.0
    if len(text) < 8:
        score -= 1.0
    if candidate.get("source_type") == "paragraph":
        score -= 0.4
    return score


def rank_rule(row: dict[str, Any], lex: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [dict(c) for c in make_sentence_windows(row)]
    for candidate in candidates:
        candidate["rule_score"] = rule_score(row, candidate, lex)
        candidate["score"] = candidate["rule_score"]
        candidate["selector"] = "rule"
    candidates.sort(key=lambda x: (-float(x["score"]), len(x.get("text", "")), int(x.get("start", 0))))
    return candidates


def selector_metrics(rows: list[dict[str, Any]], rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    top1 = top3 = top5 = missing = ranking_error = illegal = multi_alignment = 0
    f1_total = 0.0
    for row in rows:
        ranked = rankings.get(row["event_mention_id"], [])
        if not ranked:
            missing += 1
            continue
        gold_in_pool = any(contains_gold_span(row, cand) for cand in ranked)
        missing += int((not gold_in_pool) and bool(row.get("event_span", {}).get("text")))
        top1_hit = contains_gold_span(row, ranked[0])
        top1 += int(top1_hit)
        top3 += int(any(contains_gold_span(row, cand) for cand in ranked[:3]))
        top5 += int(any(contains_gold_span(row, cand) for cand in ranked[:5]))
        ranking_error += int(gold_in_pool and not top1_hit)
        f1_total += char_f1(row.get("event_span", {}), ranked[0])
        illegal += int(row.get("content", "")[ranked[0]["start"] : ranked[0]["end"]] != ranked[0]["text"])
        multi_alignment += int(bool(row.get("multi_event_article")) and gold_in_pool and not top1_hit)
    n = max(1, len(rows))
    return {
        "sample_count": len(rows),
        "top1": round(top1 / n, 6),
        "top3": round(top3 / n, 6),
        "top5": round(top5 / n, 6),
        "gold_candidate_missing": missing,
        "ranking_error": ranking_error,
        "multi_event_alignment_error": multi_alignment,
        "event_span_char_f1": round(f1_total / n, 6),
        "illegal_offset_count": illegal,
    }
