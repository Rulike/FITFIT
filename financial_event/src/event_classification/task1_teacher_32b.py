#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for the Task1 Step 5.6-E local 32B teacher pilot."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from src.routing.task1_candidate_v2 import extract_terms


API_BASE = "http://127.0.0.1:8000/v1"
MODEL_NAME = "DianJin-R1-32B"
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def sentence_split(text: str) -> list[str]:
    rows = []
    seen = set()
    for part in SENTENCE_RE.split(str(text or "")):
        value = re.sub(r"\s+", " ", part).strip()
        if len(value) < 8:
            continue
        if value in seen:
            continue
        seen.add(value)
        rows.append(value)
    return rows


def evidence_sentences(article: dict[str, Any], candidates: list[dict[str, Any]], max_sentences: int = 24, max_chars: int = 5200) -> list[dict[str, Any]]:
    """Select evidence sentences without using gold labels or model scores."""
    title = str(article.get("title") or "").strip()
    content = str(article.get("content") or "")
    raw_sentences = []
    if title:
        raw_sentences.append(("title", title))
    for idx, sent in enumerate(sentence_split(content)):
        raw_sentences.append((f"body_{idx + 1}", sent))

    event_terms = Counter()
    for row in candidates:
        text = f"{row.get('event_name', '')} {row.get('event_description', '')}"
        for term in extract_terms(text):
            if len(term) >= 2:
                event_terms[term] += 1
    top_terms = {term for term, _ in event_terms.most_common(120)}

    scored = []
    for pos, (source, sent) in enumerate(raw_sentences):
        terms = set(extract_terms(sent))
        overlap = len(terms & top_terms)
        action = sum(1 for kw in ["公告", "发布", "签订", "投产", "建设", "研发", "获批", "纳入", "调整", "扩产", "投资", "合作", "下调", "上涨", "政策"] if kw in sent)
        time_hit = 1 if re.search(r"20\d{2}年|\d{1,2}月\d{1,2}日|近日|日前|目前|未来|计划|拟", sent) else 0
        first_bonus = max(0.0, 3.0 - pos * 0.03) if pos < 25 else 0.0
        title_bonus = 8.0 if source == "title" else 0.0
        score = title_bonus + 1.5 * overlap + 1.2 * action + 0.8 * time_hit + first_bonus
        scored.append((score, pos, source, sent))

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    selected.sort(key=lambda item: item[1])
    out = []
    total_chars = 0
    for _, _, source, sent in selected:
        if total_chars + len(sent) > max_chars and out:
            continue
        out.append({"evidence_id": f"S{len(out) + 1}", "source": source, "text": sent})
        total_chars += len(sent)
    return out


def teacher_prompt_template() -> str:
    return """你是金融新闻事件分类教师模型。请仅根据输入新闻、候选事件类别和证据句，判断这篇新闻是否明确发生了候选列表中的一个或多个事件。

硬性规则：
1. 只能从给出的Candidate-v2候选事件中选择，event_name必须逐字匹配候选名称。
2. 可以选择0个、1个或多个事件；不要强制每篇都有事件。
3. 每个选中事件必须引用至少一个合法证据句ID，例如S1、S2。
4. 新闻仅讨论行业观点、可能性、投资者提问、行情涨跌、分析师预测或背景材料时，不要自动判定为公司/行业事件已经发生。
5. 区分已发生事实、正式公告或明确计划、评论/推测、无充分证据；只有事实、公告或明确计划可选。
6. 不输出思维过程，不输出Markdown代码块，不在JSON前后添加说明文字。
7. confidence表示对该候选判断的确定程度，必须是0到1之间的有限数。

只输出以下严格JSON格式：
{
  "selected_events": [
    {
      "event_name": "必须来自候选列表",
      "confidence": 0.0,
      "evidence_ids": ["S1"],
      "evidence_summary": "不超过40字的事实摘要"
    }
  ],
  "no_event": false
}

一致性要求：
- selected_events为空时，no_event必须为true。
- selected_events非空时，no_event必须为false。
- evidence_ids必须存在于当前文章证据集合。
"""


def build_user_message(item: dict[str, Any]) -> str:
    candidate_lines = []
    for row in item["candidates"]:
        candidate_lines.append(
            f"{int(row['candidate_rank'])}. {row['event_name']} | 定义：{row.get('event_description', '')[:280]}"
        )
    evidence_lines = [f"{row['evidence_id']}: {row['text']}" for row in item["evidence_sentences"]]
    return (
        f"article_id: {item['article_id']}\n"
        f"标题：{item.get('title', '')}\n"
        f"来源：{item.get('source', '')}\n"
        f"发布时间：{item.get('publish_time', '')}\n\n"
        "Candidate-v2候选事件：\n"
        + "\n".join(candidate_lines)
        + "\n\n证据句：\n"
        + "\n".join(evidence_lines)
        + "\n\n请只输出严格JSON。"
    )


def strict_json_loads(text: str) -> Any:
    value = str(text or "").strip()
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError("output is not a single JSON object")
    return json.loads(value)


def validate_teacher_output(payload: Any, candidate_names: set[str], evidence_ids: set[str]) -> tuple[bool, list[str], dict[str, Any] | None]:
    errors = []
    if not isinstance(payload, dict):
        return False, ["top-level output is not object"], None
    selected = payload.get("selected_events")
    no_event = payload.get("no_event")
    if not isinstance(selected, list):
        errors.append("selected_events is not a list")
        selected = []
    if not isinstance(no_event, bool):
        errors.append("no_event is not boolean")
        no_event = False
    parsed_events = []
    for idx, event in enumerate(selected):
        if not isinstance(event, dict):
            errors.append(f"selected_events[{idx}] is not object")
            continue
        name = str(event.get("event_name") or "")
        if name not in candidate_names:
            errors.append(f"illegal event_name: {name}")
        conf = event.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not math.isfinite(float(conf)) or not (0.0 <= float(conf) <= 1.0):
            errors.append(f"illegal confidence for {name}")
            conf = 0.0
        eids = event.get("evidence_ids")
        if not isinstance(eids, list) or not eids:
            errors.append(f"missing evidence_ids for {name}")
            eids = []
        clean_eids = []
        for eid in eids:
            eid_value = str(eid)
            if eid_value not in evidence_ids:
                errors.append(f"illegal evidence_id for {name}: {eid_value}")
            else:
                clean_eids.append(eid_value)
        summary = str(event.get("evidence_summary") or "")[:80]
        parsed_events.append(
            {
                "event_name": name,
                "confidence": float(conf),
                "evidence_ids": clean_eids,
                "evidence_summary": summary,
            }
        )
    if selected and no_event:
        errors.append("no_event true with non-empty selected_events")
    if not selected and no_event is not True:
        errors.append("no_event false with empty selected_events")
    if errors:
        return False, errors, None
    return True, [], {"selected_events": parsed_events, "no_event": bool(no_event)}


def correction_user_message(raw_output: str, errors: list[str]) -> str:
    schema = {
        "selected_events": [
            {
                "event_name": "必须逐字来自候选列表",
                "confidence": 0.0,
                "evidence_ids": ["S1"],
                "evidence_summary": "不超过40字的事实摘要",
            }
        ],
        "no_event": False,
    }
    return (
        "你的上一次输出无法通过JSON格式校验。请只修正格式，不添加解释，不改变判断依据。\n"
        f"错误原因：{'; '.join(errors)}\n"
        f"JSON Schema示例：{json.dumps(schema, ensure_ascii=False)}\n"
        f"原始模型输出：{raw_output}"
    )


def chat_completion(
    messages: list[dict[str, str]],
    api_base: str = API_BASE,
    model: str = MODEL_NAME,
    timeout: int = 120,
    max_tokens: int = 512,
) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.perf_counter()
    try:
        with opener.open(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            elapsed = time.perf_counter() - start
            payload = json.loads(body)
            content = payload["choices"][0]["message"]["content"]
            return str(content), payload, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"http_error:{exc.code}:{body[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"request_error:{type(exc).__name__}:{exc}") from exc

