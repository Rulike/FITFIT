#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gold-free candidate constrained structure extraction utilities for Task1."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ALLOWED_INPUT_FIELDS = ["event_mention_id", "article_id", "title", "content", "publish_time", "event_name"]
DIR_LABELS = ["positive", "negative", "neutral", "unclear"]
ROLE_LABELS = ["subject_company", "event_time", "object_target", "trigger", "other"]
SPAN_FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
LIMITS = {"sentence": 8, "span": 8, "trigger": 12, "subject": 10, "time": 8, "object": 12}

BANNED_PREFIXES = ("ai_", "human_", "reviewer_")
BANNED_EXACT = {
    "ai_fields",
    "reference_raw_text",
    "validation_errors",
    "quality_grade",
    "original_quality_grade",
    "selected_quality_grade",
    "annotation_source",
    "annotation_status",
    "confirmed_at",
    "confirmation_note",
    "source_workbook",
    "trigger",
    "event_span",
    "subject_company",
    "event_time",
    "object_target",
    "direction",
    "arguments",
}

GENERIC_TRIGGERS = [
    "发布",
    "推出",
    "上线",
    "启动",
    "落地",
    "签署",
    "合作",
    "中标",
    "获批",
    "获准",
    "增长",
    "下降",
    "上涨",
    "下跌",
    "增产",
    "减产",
    "扩产",
    "投产",
    "停产",
    "涨价",
    "降价",
    "突破",
    "创新",
    "并购",
    "收购",
    "投资",
    "融资",
    "上市",
    "开工",
    "竣工",
    "交付",
    "成立",
    "建设",
    "升级",
    "调整",
    "召回",
    "处罚",
    "亏损",
    "盈利",
    "扭亏",
    "预增",
    "预减",
    "签约",
    "供货",
    "采购",
    "恢复",
    "维持",
    "宣布",
    "完成",
]

EVENT_TRIGGER_HINTS = {
    "OPEC+产量政策变化": ["增产", "减产", "维持产量", "恢复增产", "讨论", "产量"],
    "关键技术突破": ["突破", "攻克", "实现", "首创", "打破", "发布"],
    "产品技术创新": ["创新", "发布", "推出", "实现", "升级"],
    "产能布局调整": ["投产", "扩产", "建设", "基地", "布局", "调整", "启动"],
    "新业务拓展": ["推出", "上线", "拓展", "布局", "启动", "成立"],
    "新车发布定价": ["上市", "发布", "定价", "售价", "推出"],
}

POSITIVE_WORDS = ["增长", "上涨", "突破", "获批", "中标", "盈利", "扭亏", "成功", "提升", "利好", "创新", "扩产", "投产", "签约"]
NEGATIVE_WORDS = ["下降", "下跌", "亏损", "处罚", "召回", "减产", "停产", "风险", "利空", "下滑", "取消", "终止"]
NEUTRAL_WORDS = ["发布", "宣布", "维持", "召开", "讨论", "计划", "拟", "称", "表示"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", "", text.lower())
    text = re.sub(r"[，。；：、,.!！?？（）()《》“”\"'`·\-—_<>【】\[\]{}]", "", text)
    for suffix in ["股份有限公司", "有限责任公司", "有限公司", "集团股份", "集团", "公司"]:
        text = text.replace(suffix, "")
    return text


def banned_field_hits(row: dict[str, Any]) -> list[str]:
    return sorted(key for key in row if key in BANNED_EXACT or key.startswith(BANNED_PREFIXES))


def project_allowed(row: dict[str, Any]) -> dict[str, Any]:
    return {field: str(row.get(field, "") or "") for field in ALLOWED_INPUT_FIELDS}


def stable_json_sha(rows: Any) -> str:
    return sha256_text(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def span_empty() -> dict[str, Any]:
    return {"text": "", "start": -1, "end": -1}


def time_empty() -> dict[str, Any]:
    return {"text": "", "normalized": "未明确", "start": -1, "end": -1}


def strict_span(content: str, text: str, start: int, end: int) -> dict[str, Any] | None:
    if text and 0 <= start <= end <= len(content) and content[start:end] == text:
        return {"text": text, "start": start, "end": end}
    return None


def iter_occurrences(content: str, phrase: str) -> Iterable[tuple[int, int]]:
    phrase = str(phrase or "").strip()
    if not phrase:
        return
    start = 0
    while True:
        idx = content.find(phrase, start)
        if idx < 0:
            break
        yield idx, idx + len(phrase)
        start = idx + max(1, len(phrase))


def split_sentences(content: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    start = 0
    for idx, ch in enumerate(content):
        if ch in "。！？!?；;\n":
            end = idx + 1
            text = content[start:end].strip()
            if text:
                lead = len(content[start:end]) - len(content[start:end].lstrip())
                trail = len(content[start + lead : end]) - len(content[start + lead : end].rstrip())
                real_start = start + lead
                real_end = end - trail
                if real_start < real_end:
                    sentences.append({"all_id": len(sentences), "text": content[real_start:real_end], "start": real_start, "end": real_end})
            start = end
    if start < len(content):
        text = content[start:].strip()
        if text:
            lead = len(content[start:]) - len(content[start:].lstrip())
            real_start = start + lead
            real_end = len(content)
            if real_start < real_end:
                sentences.append({"all_id": len(sentences), "text": content[real_start:real_end], "start": real_start, "end": real_end})
    if not sentences and content:
        sentences.append({"all_id": 0, "text": content, "start": 0, "end": len(content)})
    return sentences


def normalize_time(text: str, publish_time: str = "") -> str:
    value = str(text or "")
    if not value:
        return "未明确"
    year = ""
    m_pub = re.search(r"(20\d{2})", str(publish_time or ""))
    if m_pub:
        year = m_pub.group(1)
    m = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?", value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})月(\d{1,2})日", value)
    if m and year:
        return f"{int(year):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.search(r"(20\d{2})年(第一|第二|第三|第四|一|二|三|四)?季度", value)
    if m:
        qmap = {"第一": "01", "一": "01", "第二": "04", "二": "04", "第三": "07", "三": "07", "第四": "10", "四": "10", None: "01"}
        return f"{m.group(1)}-{qmap.get(m.group(2), '01')}-01"
    if any(word in value for word in ["近日", "日前", "当日", "今日", "当天", "近期", "今年", "未来"]):
        return value
    return value


def event_keywords(event_name: str) -> list[str]:
    out = [event_name]
    for sep in ["与", "及", "和", "、", "/", "+"]:
        for part in event_name.split(sep):
            if 2 <= len(part) < len(event_name):
                out.append(part)
    for key, values in EVENT_TRIGGER_HINTS.items():
        if key == event_name:
            out.extend(values)
    return list(dict.fromkeys(out))


def build_candidate_rules(rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggers: dict[str, Counter[str]] = defaultdict(Counter)
    subjects: dict[str, Counter[str]] = defaultdict(Counter)
    objects: dict[str, Counter[str]] = defaultdict(Counter)
    directions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        event_name = str(row.get("event_name", ""))
        if (row.get("trigger") or {}).get("text"):
            triggers[event_name][row["trigger"]["text"]] += 1
        if (row.get("subject_company") or {}).get("text"):
            subjects[event_name][row["subject_company"]["text"]] += 1
        if (row.get("object_target") or {}).get("text"):
            objects[event_name][row["object_target"]["text"]] += 1
        if row.get("direction") in DIR_LABELS:
            directions[event_name][row["direction"]] += 1
    return {
        "version": "task1_candidate_constrained_structure_v1",
        "limits": LIMITS,
        "allowed_input_fields": ALLOWED_INPUT_FIELDS,
        "event_trigger_lexicon": {k: [w for w, _ in c.most_common(32)] for k, c in triggers.items()},
        "event_subject_lexicon": {k: [w for w, _ in c.most_common(32)] for k, c in subjects.items()},
        "event_object_lexicon": {k: [w for w, _ in c.most_common(32)] for k, c in objects.items()},
        "event_direction_prior": {k: (c.most_common(1)[0][0] if c else "unclear") for k, c in directions.items()},
        "generic_triggers": GENERIC_TRIGGERS,
        "event_trigger_hints": EVENT_TRIGGER_HINTS,
        "direction_words": {"positive": POSITIVE_WORDS, "negative": NEGATIVE_WORDS, "neutral": NEUTRAL_WORDS},
    }


def find_sentence_id(sentences: list[dict[str, Any]], start: int, end: int) -> int | None:
    for sent in sentences:
        if sent["start"] <= start < end <= sent["end"]:
            return int(sent["all_id"])
    return None


def make_candidate_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:02d}"


def score_sentence(sent: dict[str, Any], title: str, event_name: str, phrases: list[str]) -> float:
    text = sent["text"]
    score = 0.0
    if sent["start"] < 400:
        score += 0.3
    title_chars = set(norm_text(title))
    if title_chars:
        score += min(0.8, len(title_chars & set(norm_text(text))) / max(1, len(title_chars)))
    for phrase in event_keywords(event_name) + phrases:
        if phrase and phrase in text:
            score += 1.2
    for word in POSITIVE_WORDS + NEGATIVE_WORDS + NEUTRAL_WORDS:
        if word in text:
            score += 0.1
    return score


def select_sentences(content: str, title: str, event_name: str, phrase_hits: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
    all_sentences = split_sentences(content)
    phrases = [hit[2] for hit in phrase_hits]
    required_ids = {find_sentence_id(all_sentences, s, e) for s, e, _ in phrase_hits[: LIMITS["trigger"]]}
    required_ids.discard(None)
    scored = []
    for sent in all_sentences:
        score = score_sentence(sent, title, event_name, phrases)
        if sent["all_id"] in required_ids:
            score += 3.0
        scored.append((score, -abs(sent["start"] - 300), sent))
    selected = [item[2] for item in sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[: LIMITS["sentence"]]]
    selected = sorted(selected, key=lambda s: s["start"])
    out = []
    for idx, sent in enumerate(selected):
        out.append({"sentence_id": make_candidate_id("SENT", idx), "text": sent["text"], "start": sent["start"], "end": sent["end"], "score": round(score_sentence(sent, title, event_name, phrases), 6), "all_id": sent["all_id"]})
    return out


def phrase_hits(content: str, phrases: list[str]) -> list[tuple[int, int, str]]:
    seen: set[tuple[int, int, str]] = set()
    hits: list[tuple[int, int, str]] = []
    for phrase in sorted(set(p.strip() for p in phrases if p and len(p.strip()) >= 2), key=len, reverse=True):
        for start, end in iter_occurrences(content, phrase):
            key = (start, end, phrase)
            if key not in seen:
                seen.add(key)
                hits.append(key)
    return sorted(hits, key=lambda x: (x[0], -len(x[2])))


def selected_sentence_for(sentences: list[dict[str, Any]], start: int, end: int) -> str:
    for sent in sentences:
        if sent["start"] <= start < end <= sent["end"]:
            return sent["sentence_id"]
    return ""


def add_candidate(candidates: list[dict[str, Any]], prefix: str, text: str, start: int, end: int, sentence_id: str, source: str, score: float, extra: dict[str, Any] | None = None) -> None:
    if not text:
        return
    if any(c["text"] == text and c["start"] == start and c["end"] == end for c in candidates):
        return
    item = {"candidate_id": make_candidate_id(prefix, len(candidates)), "text": text, "start": start, "end": end, "sentence_id": sentence_id, "source": source, "score": round(float(score), 6)}
    if extra:
        item.update(extra)
    candidates.append(item)


def generate_trigger_candidates(row: dict[str, Any], rules: dict[str, Any], selected_sentences: list[dict[str, Any]], hits: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
    content = row["content"]
    event_name = row["event_name"]
    event_lex = rules.get("event_trigger_lexicon", {}).get(event_name, [])
    hints = rules.get("event_trigger_hints", {}).get(event_name, []) + rules.get("generic_triggers", GENERIC_TRIGGERS)
    phrases = list(dict.fromkeys(event_lex + hints))
    candidates: list[dict[str, Any]] = []
    for start, end, phrase in hits + phrase_hits(content, phrases):
        sid = selected_sentence_for(selected_sentences, start, end)
        if not sid:
            continue
        score = 2.0 if phrase in event_lex else 1.0
        if phrase in EVENT_TRIGGER_HINTS.get(event_name, []):
            score += 0.7
        add_candidate(candidates, "TRIGGER", content[start:end], start, end, sid, "event_trigger_lexicon_or_hint", score)
        if len(candidates) >= LIMITS["trigger"]:
            break
    return candidates[: LIMITS["trigger"]]


def generate_subject_candidates(row: dict[str, Any], rules: dict[str, Any], selected_sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content = row["content"]
    event_name = row["event_name"]
    candidates: list[dict[str, Any]] = []
    phrases = rules.get("event_subject_lexicon", {}).get(event_name, []) + ["欧佩克+", "欧佩克", "中国移动", "中国联通", "公司", "机构", "政府", "工信部"]
    for phrase in list(dict.fromkeys(phrases)):
        for start, end in iter_occurrences(content, phrase):
            sid = selected_sentence_for(selected_sentences, start, end)
            if sid:
                add_candidate(candidates, "SUBJECT", content[start:end], start, end, sid, "subject_lexicon", 2.0)
                break
        if len(candidates) >= LIMITS["subject"]:
            return candidates
    patterns = [
        r"[\u4e00-\u9fa5A-Za-z0-9]{2,24}(?:股份有限公司|有限责任公司|有限公司|集团|公司)",
        r"(?:欧佩克\+?|OPEC\+?|工信部|发改委|国家能源局|商务部|证监会|上交所|深交所)",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, content, re.IGNORECASE):
            sid = selected_sentence_for(selected_sentences, m.start(), m.end())
            if sid:
                add_candidate(candidates, "SUBJECT", content[m.start():m.end()], m.start(), m.end(), sid, "entity_regex", 1.0)
            if len(candidates) >= LIMITS["subject"]:
                return candidates
    return candidates[: LIMITS["subject"]]


def generate_time_candidates(row: dict[str, Any], selected_sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content = row["content"]
    patterns = [
        r"20\d{2}年(?:第一|第二|第三|第四|一|二|三|四)?季度",
        r"20\d{2}[年/-]\d{1,2}[月/-]\d{1,2}日?",
        r"\d{1,2}月\d{1,2}日",
        r"20\d{2}年\d{1,2}月",
        r"(?:近日|日前|当日|当天|今日|近期|今年|未来|上半年|下半年)",
    ]
    candidates: list[dict[str, Any]] = [{"candidate_id": "TIME_NONE", "text": "", "start": -1, "end": -1, "sentence_id": "", "source": "fixed_none", "score": 0.5, "normalized": "未明确"}]
    for pattern in patterns:
        for m in re.finditer(pattern, content):
            sid = selected_sentence_for(selected_sentences, m.start(), m.end())
            if sid:
                add_candidate(candidates, "TIME", content[m.start():m.end()], m.start(), m.end(), sid, "time_regex", 1.0, {"normalized": normalize_time(content[m.start():m.end()], row.get("publish_time", ""))})
            if len(candidates) >= LIMITS["time"]:
                return candidates[: LIMITS["time"]]
    return candidates[: LIMITS["time"]]


def generate_object_candidates(row: dict[str, Any], rules: dict[str, Any], selected_sentences: list[dict[str, Any]], trigger_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content = row["content"]
    event_name = row["event_name"]
    candidates: list[dict[str, Any]] = [{"candidate_id": "OBJECT_NONE", "text": "", "start": -1, "end": -1, "sentence_id": "", "source": "fixed_none", "score": 0.5}]
    for phrase in rules.get("event_object_lexicon", {}).get(event_name, []):
        for start, end in iter_occurrences(content, phrase):
            sid = selected_sentence_for(selected_sentences, start, end)
            if sid:
                add_candidate(candidates, "OBJECT", content[start:end], start, end, sid, "object_lexicon", 2.0)
                break
    for trig in trigger_candidates[:6]:
        for sent in selected_sentences:
            if sent["sentence_id"] != trig["sentence_id"]:
                continue
            text = sent["text"]
            base = sent["start"]
            rel = max(0, trig["end"] - base)
            for m in re.finditer(r"[\u4e00-\u9fa5A-Za-z0-9《》“”\"'·\-]{3,30}(?:项目|产品|技术|基地|市场|政策|方案|白皮书|订单|产能|价格|业务|平台|车型|股份|股权)", text[rel:]):
                start = base + rel + m.start()
                end = base + rel + m.end()
                add_candidate(candidates, "OBJECT", content[start:end], start, end, sent["sentence_id"], "trigger_neighbor_np", 1.0)
            break
        if len(candidates) >= LIMITS["object"]:
            break
    return candidates[: LIMITS["object"]]


def generate_span_candidates(selected_sentences: list[dict[str, Any]], trigger_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    trigger_sids = {c["sentence_id"] for c in trigger_candidates}
    for sent in selected_sentences:
        score = sent.get("score", 0.0) + (2.0 if sent["sentence_id"] in trigger_sids else 0.0)
        spans.append({"candidate_id": make_candidate_id("SPAN", len(spans)), "text": sent["text"], "start": sent["start"], "end": sent["end"], "sentence_id": sent["sentence_id"], "source": "single_sentence", "score": round(score, 6)})
    by_all = {s["all_id"]: s for s in selected_sentences}
    for sent in selected_sentences:
        nxt = by_all.get(sent["all_id"] + 1)
        if not nxt:
            continue
        text = sent["text"] + "" + nxt["text"] if sent["end"] == nxt["start"] else None
        if text is None:
            continue
        score = max(sent.get("score", 0.0), nxt.get("score", 0.0)) + 0.2
        spans.append({"candidate_id": make_candidate_id("SPAN", len(spans)), "text": text, "start": sent["start"], "end": nxt["end"], "sentence_id": f"{sent['sentence_id']}+{nxt['sentence_id']}", "source": "adjacent_two_sentences", "score": round(score, 6)})
    spans = sorted(spans, key=lambda c: (c["score"], -len(c["text"])), reverse=True)[: LIMITS["span"]]
    for idx, cand in enumerate(spans):
        cand["candidate_id"] = make_candidate_id("SPAN", idx)
    return spans


def generate_candidates(row: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    row = project_allowed(row)
    content = row["content"]
    event_name = row["event_name"]
    phrases = list(dict.fromkeys(rules.get("event_trigger_lexicon", {}).get(event_name, []) + rules.get("event_trigger_hints", {}).get(event_name, []) + rules.get("generic_triggers", GENERIC_TRIGGERS)))
    hits = phrase_hits(content, phrases)
    sentences = select_sentences(content, row["title"], event_name, hits)
    trigger_candidates = generate_trigger_candidates(row, rules, sentences, hits)
    subject_candidates = generate_subject_candidates(row, rules, sentences)
    time_candidates = generate_time_candidates(row, sentences)
    object_candidates = generate_object_candidates(row, rules, sentences, trigger_candidates)
    span_candidates = generate_span_candidates(sentences, trigger_candidates)
    direction_prior = rules.get("event_direction_prior", {}).get(event_name, "unclear")
    direction_candidates = [{"candidate_id": f"DIRECTION_{label.upper()}", "direction": label, "score": 1.0 if label == direction_prior else 0.2} for label in DIR_LABELS]
    payload = {
        "event_mention_id": row["event_mention_id"],
        "article_id": row["article_id"],
        "event_name": event_name,
        "title": row["title"],
        "publish_time": row["publish_time"],
        "sentence_candidates": sentences,
        "trigger_candidates": trigger_candidates,
        "subject_candidates": [{"candidate_id": "SUBJECT_NONE", "text": "", "start": -1, "end": -1, "sentence_id": "", "source": "fixed_none", "score": 0.1}] + subject_candidates[: max(0, LIMITS["subject"] - 1)],
        "event_time_candidates": time_candidates,
        "object_candidates": object_candidates,
        "event_span_candidates": span_candidates,
        "direction_candidates": direction_candidates,
    }
    return payload


def candidate_index(candidates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for key in ["trigger_candidates", "subject_candidates", "event_time_candidates", "object_candidates", "event_span_candidates"]:
        for cand in candidates.get(key, []):
            idx[cand["candidate_id"]] = cand
    return idx


def deterministic_selection(candidates: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    spans = candidates.get("event_span_candidates", [])
    triggers = candidates.get("trigger_candidates", [])
    subjects = [c for c in candidates.get("subject_candidates", []) if c["candidate_id"] != "SUBJECT_NONE"]
    times = candidates.get("event_time_candidates", [])
    objects = candidates.get("object_candidates", [])
    span = max(spans, key=lambda c: c.get("score", 0.0), default=None)
    trigger_pool = [t for t in triggers if span and span["start"] <= t["start"] < t["end"] <= span["end"]]
    trigger = max(trigger_pool or triggers, key=lambda c: c.get("score", 0.0), default=None)
    subject = max(subjects, key=lambda c: c.get("score", 0.0), default={"candidate_id": "SUBJECT_NONE"})
    explicit_times = [t for t in times if t["candidate_id"] != "TIME_NONE"]
    event_time = max(explicit_times, key=lambda c: c.get("score", 0.0), default={"candidate_id": "TIME_NONE"})
    explicit_objects = [o for o in objects if o["candidate_id"] != "OBJECT_NONE"]
    obj = max(explicit_objects, key=lambda c: c.get("score", 0.0), default={"candidate_id": "OBJECT_NONE"})
    direction = rules.get("event_direction_prior", {}).get(candidates.get("event_name", ""), "unclear")
    context = " ".join([span.get("text", "") if span else "", trigger.get("text", "") if trigger else ""])
    if any(w in context for w in POSITIVE_WORDS):
        direction = "positive"
    elif any(w in context for w in NEGATIVE_WORDS):
        direction = "negative"
    elif any(w in context for w in NEUTRAL_WORDS):
        direction = "neutral"
    args = []
    for cand, role in [(subject, "subject_company"), (event_time, "event_time"), (obj, "object_target"), (trigger, "trigger")]:
        if cand and cand.get("candidate_id") and not cand["candidate_id"].endswith("NONE"):
            args.append({"candidate_id": cand["candidate_id"], "role": role})
    return {
        "event_span_candidate_id": span["candidate_id"] if span else "",
        "trigger_candidate_id": trigger["candidate_id"] if trigger else "",
        "subject_candidate_id": subject.get("candidate_id", "SUBJECT_NONE"),
        "event_time_candidate_id": event_time.get("candidate_id", "TIME_NONE"),
        "object_candidate_id": obj.get("candidate_id", "OBJECT_NONE"),
        "direction": direction if direction in DIR_LABELS else "unclear",
        "argument_candidate_ids": args,
        "confidence": 0.5,
    }


def validate_selection(selection: dict[str, Any], candidates: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    idx = candidate_index(candidates)
    allowed_keys = {"event_span_candidate_id", "trigger_candidate_id", "subject_candidate_id", "event_time_candidate_id", "object_candidate_id", "direction", "argument_candidate_ids", "confidence"}
    extra = sorted(set(selection) - allowed_keys)
    if extra:
        errors.append(f"extra_keys:{','.join(extra)}")
    checks = {
        "event_span_candidate_id": "SPAN",
        "trigger_candidate_id": "TRIGGER",
        "subject_candidate_id": "SUBJECT",
        "event_time_candidate_id": "TIME",
        "object_candidate_id": "OBJECT",
    }
    for field, prefix in checks.items():
        cid = str(selection.get(field, ""))
        if cid in {"SUBJECT_NONE", "TIME_NONE", "OBJECT_NONE"}:
            continue
        if not cid.startswith(prefix + "_") or cid not in idx:
            errors.append(f"invalid_candidate_id:{field}:{cid}")
    if selection.get("direction") not in DIR_LABELS:
        errors.append("invalid_direction")
    try:
        conf = float(selection.get("confidence"))
        if not math.isfinite(conf) or not (0 <= conf <= 1):
            errors.append("invalid_confidence")
    except Exception:
        errors.append("invalid_confidence")
    span = idx.get(str(selection.get("event_span_candidate_id", "")))
    trigger = idx.get(str(selection.get("trigger_candidate_id", "")))
    if span and trigger and not (span["start"] <= trigger["start"] < trigger["end"] <= span["end"]):
        errors.append("trigger_not_inside_event_span")
    args = selection.get("argument_candidate_ids", [])
    if not isinstance(args, list):
        errors.append("argument_candidate_ids_not_list")
    else:
        for item in args:
            if not isinstance(item, dict):
                errors.append("argument_item_not_object")
                continue
            cid = str(item.get("candidate_id", ""))
            role = str(item.get("role", ""))
            if cid not in idx and cid not in {"SUBJECT_NONE", "TIME_NONE", "OBJECT_NONE"}:
                errors.append(f"argument_invalid_candidate_id:{cid}")
            if role not in ROLE_LABELS:
                errors.append(f"argument_invalid_role:{role}")
    return not errors, errors


def selection_to_prediction(row: dict[str, Any], candidates: dict[str, Any], selection: dict[str, Any], scheme: str, status: str = "completed", errors: list[str] | None = None) -> dict[str, Any]:
    idx = candidate_index(candidates)
    def span_for(cid: str, is_time: bool = False) -> dict[str, Any]:
        if cid in {"", "SUBJECT_NONE", "OBJECT_NONE"}:
            return span_empty()
        if cid == "TIME_NONE":
            return time_empty()
        cand = idx.get(cid)
        if not cand:
            return time_empty() if is_time else span_empty()
        out = {"text": cand.get("text", ""), "start": cand.get("start", -1), "end": cand.get("end", -1)}
        if is_time:
            out["normalized"] = cand.get("normalized", normalize_time(cand.get("text", ""), row.get("publish_time", "")))
        return out

    trigger = span_for(str(selection.get("trigger_candidate_id", "")))
    event_span = span_for(str(selection.get("event_span_candidate_id", "")))
    subject = span_for(str(selection.get("subject_candidate_id", "SUBJECT_NONE")))
    event_time = span_for(str(selection.get("event_time_candidate_id", "TIME_NONE")), True)
    obj = span_for(str(selection.get("object_candidate_id", "OBJECT_NONE")))
    arguments = []
    for cid, role in [
        (selection.get("subject_candidate_id", "SUBJECT_NONE"), "subject_company"),
        (selection.get("event_time_candidate_id", "TIME_NONE"), "event_time"),
        (selection.get("object_candidate_id", "OBJECT_NONE"), "object_target"),
        (selection.get("trigger_candidate_id", ""), "trigger"),
    ]:
        span = span_for(str(cid), role == "event_time")
        if span.get("text"):
            arguments.append({"text": span["text"], "role": role, "start": span["start"], "end": span["end"]})
    return {
        "event_mention_id": row["event_mention_id"],
        "article_id": row["article_id"],
        "scheme": scheme,
        "status": status,
        "selection": selection,
        "validation_errors": errors or [],
        "trigger": trigger,
        "event_span": event_span,
        "subject_company": subject,
        "event_time": event_time,
        "object_target": obj,
        "direction": selection.get("direction", "unclear") if selection.get("direction") in DIR_LABELS else "unclear",
        "arguments": arguments,
    }


def prompt_text(prompt_name: str, row: dict[str, Any], candidates: dict[str, Any]) -> list[dict[str, str]]:
    def compact(items: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
        return [{field: item.get(field, "") for field in fields if field in item} for item in items]
    candidate_payload = {
        "event_span_candidates": compact(candidates["event_span_candidates"], ["candidate_id", "text", "start", "end", "score"]),
        "trigger_candidates": compact(candidates["trigger_candidates"], ["candidate_id", "text", "start", "end", "sentence_id", "score"]),
        "subject_candidates": compact(candidates["subject_candidates"], ["candidate_id", "text", "start", "end", "sentence_id", "score"]),
        "event_time_candidates": compact(candidates["event_time_candidates"], ["candidate_id", "text", "normalized", "start", "end", "sentence_id", "score"]),
        "object_candidates": compact(candidates["object_candidates"], ["candidate_id", "text", "start", "end", "sentence_id", "score"]),
        "direction_candidates": ["positive", "negative", "neutral", "unclear"],
    }
    schema = {
        "event_span_candidate_id": "SPAN_x",
        "trigger_candidate_id": "TRIGGER_x",
        "subject_candidate_id": "SUBJECT_x或SUBJECT_NONE",
        "event_time_candidate_id": "TIME_x或TIME_NONE",
        "object_candidate_id": "OBJECT_x或OBJECT_NONE",
        "direction": "positive|negative|neutral|unclear",
        "argument_candidate_ids": [{"candidate_id": "候选ID", "role": "subject_company|event_time|object_target|trigger|other"}],
        "confidence": 0.0,
    }
    extra = ""
    if prompt_name == "strict_candidate_selection_with_self_check":
        extra = "返回前自检：所有ID必须来自候选；trigger必须位于event_span内；不要输出候选外文本。"
    system = "你是金融事件结构抽取候选选择器。只能选择候选ID，禁止生成任何自由文本、offset或新候选。只返回一个JSON对象。"
    user = (
        f"事件类型：{row['event_name']}\n标题：{row['title']}\n发布时间：{row.get('publish_time','')}\n"
        f"请选择最匹配该事件的候选ID。{extra}\n"
        f"输出JSON格式：{json.dumps(schema, ensure_ascii=False)}\n"
        f"候选：{json.dumps(candidate_payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value).strip()
        value = re.sub(r"```$", "", value).strip()
    if value.startswith("{") and value.endswith("}"):
        return json.loads(value)
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        return json.loads(value[start : end + 1])
    raise ValueError("no_json_object")

