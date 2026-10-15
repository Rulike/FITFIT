#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conditional Task1 structure extraction with a local DianJin-R1-32B service.

The utilities here are intentionally independent of event classification.  They
assume the event category is already known, build gold-free model inputs, parse
strict JSON structure predictions, and evaluate them against the 80-row
AI-assisted, human-confirmed diagnostic annotations.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

API_BASE = "http://127.0.0.1:8000/v1"
MODEL_NAME = "DianJin-R1-32B"
DIR_LABELS = ["positive", "negative", "neutral", "unclear"]
ARG_ROLES = ["subject_company", "event_time", "object_target", "trigger"]
REQUIRED_STRUCTURE_FIELDS = [
    "annotate_trigger",
    "annotate_event_span",
    "annotate_subject_company",
    "annotate_event_time",
    "annotate_object_target",
    "annotate_direction",
    "annotate_arguments_json",
    "annotate_argument_roles_json",
]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strict_json_loads(text: str) -> Any:
    value = str(text or "").strip()
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError("output is not a single JSON object")
    return json.loads(value)


def norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。；：、,.!！?？（）()《》“”\"'`·\-—_<>【】\[\]{}]", "", text)
    for suffix in ["股份有限公司", "有限责任公司", "有限公司", "集团股份", "集团", "公司"]:
        text = text.replace(suffix, "")
    return text


def loose_text_equal(pred: Any, gold: Any) -> bool:
    pred_norm = norm_text(pred)
    gold_norm = norm_text(gold)
    if not pred_norm or not gold_norm:
        return pred_norm == gold_norm
    return pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm


def normalize_direction(value: Any) -> str:
    text = norm_text(value)
    if text in {"正面", "积极", "利好", "positive", "pos"}:
        return "positive"
    if text in {"负面", "消极", "利空", "negative", "neg"}:
        return "negative"
    if text in {"中性", "neutral"}:
        return "neutral"
    if text in {"不清楚", "未明确", "不明确", "unknown", "none", "null", ""}:
        return "unclear"
    return text if text in DIR_LABELS else "unclear"


def normalize_date(value: Any) -> str:
    text = "" if value is None else str(value)
    if "未明确" in text or "不明确" in text:
        return "未明确"
    m = re.search(r"(20\d{2})[-年/\.](\d{1,2})[-月/\.](\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})月(\d{1,2})日", text)
    if m:
        return f"XXXX-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return norm_text(text)


def char_overlap_f1(pred: str, gold: str) -> float:
    pred_value = str(pred or "")
    gold_value = str(gold or "")
    if not pred_value and not gold_value:
        return 1.0
    if not pred_value or not gold_value:
        return 0.0
    pred_chars = Counter(pred_value)
    gold_chars = Counter(gold_value)
    overlap = sum((pred_chars & gold_chars).values())
    precision = overlap / len(pred_value) if pred_value else 0.0
    recall = overlap / len(gold_value) if gold_value else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def set_f1(pred: set[Any], gold: set[Any]) -> dict[str, float | int]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def role_macro_f1(pred: set[tuple[str, str, str]], gold: set[tuple[str, str, str]]) -> float:
    roles = sorted({role for _, _, role in pred | gold})
    if not roles:
        return 0.0
    values = []
    for role in roles:
        pred_role = {(aid, text) for aid, text, r in pred if r == role}
        gold_role = {(aid, text) for aid, text, r in gold if r == role}
        values.append(float(set_f1(pred_role, gold_role)["f1"]))
    return sum(values) / len(values)


def parse_json_field(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def parse_gold_args(value: Any) -> list[dict[str, str]]:
    parsed = parse_json_field(value)
    if isinstance(parsed, dict):
        return [{"text": str(k), "role": str(v), "span": ""} for k, v in parsed.items()]
    out = []
    for item in parsed or []:
        if isinstance(item, dict):
            out.append(
                {
                    "text": str(item.get("text", "")),
                    "role": str(item.get("role", "")),
                    "span": str(item.get("span", "")),
                }
            )
    return out


def parse_gold_roles(value: Any) -> dict[str, str]:
    parsed = parse_json_field(value)
    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, list):
        return {str(item.get("text", "")): str(item.get("role", "")) for item in parsed if isinstance(item, dict)}
    return {}


def event_names_from_row(row: dict[str, Any]) -> list[str]:
    names = [part.strip() for part in str(row.get("gold_event_names_for_context_after_classification_eval", "")).split("|") if part.strip()]
    return names


def target_event_name(row: dict[str, Any]) -> str:
    names = event_names_from_row(row)
    return names[0] if names else ""


def validate_confirmed_annotations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("article_id", "")).strip() for row in rows]
    duplicates = sorted([aid for aid, count in Counter(ids).items() if aid and count > 1])
    missing_ids = sum(1 for aid in ids if not aid)
    field_missing = Counter()
    json_failures = []
    span_failures = []
    trigger_failures = []
    event_name_failures = []
    for row in rows:
        aid = str(row.get("article_id", ""))
        for field in REQUIRED_STRUCTURE_FIELDS:
            if not str(row.get(field, "")).strip():
                field_missing[field] += 1
        for field in ["annotate_arguments_json", "annotate_argument_roles_json"]:
            try:
                parse_json_field(row.get(field))
            except Exception as exc:  # pragma: no cover - error path is tested by fixture mutation.
                json_failures.append({"article_id": aid, "field": field, "error": str(exc)})
        content = str(row.get("content", ""))
        span = str(row.get("annotate_event_span", ""))
        trigger = str(row.get("annotate_trigger", ""))
        if span and span not in content:
            span_failures.append(aid)
        if trigger and trigger not in span:
            trigger_failures.append(aid)
        if not event_names_from_row(row):
            event_name_failures.append(aid)
    total = len(rows)
    return {
        "row_count": total,
        "unique_article_id_count": len(set(ids)),
        "duplicate_article_ids": duplicates,
        "missing_article_id_count": missing_ids,
        "field_coverage": {field: 1.0 - (field_missing[field] / total if total else 0.0) for field in REQUIRED_STRUCTURE_FIELDS},
        "json_parse_failure_count": len(json_failures),
        "json_parse_failures": json_failures,
        "event_span_not_in_content_count": len(span_failures),
        "event_span_not_in_content_article_ids": span_failures,
        "trigger_not_in_event_span_count": len(trigger_failures),
        "trigger_not_in_event_span_article_ids": trigger_failures,
        "missing_event_name_count": len(event_name_failures),
        "missing_event_name_article_ids": event_name_failures,
        "multi_event_name_rows": [
            {"article_id": str(row.get("article_id")), "event_names": event_names_from_row(row), "target_event_name": target_event_name(row)}
            for row in rows
            if len(event_names_from_row(row)) > 1
        ],
        "status": "valid"
        if total == 80
        and not duplicates
        and not missing_ids
        and not json_failures
        and not span_failures
        and not trigger_failures
        and not event_name_failures
        and all(value == 1.0 for value in (1.0 - (field_missing[field] / total if total else 0.0) for field in REQUIRED_STRUCTURE_FIELDS))
        else "invalid",
    }


def stratified_prompt_eval_split(rows: list[dict[str, Any]], seed: int = 2026, prompt_size: int = 20) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[target_event_name(row)].append(row)
    for event_name in groups:
        groups[event_name].sort(key=lambda item: str(item["article_id"]))
        rng.shuffle(groups[event_name])

    selected_ids: list[str] = []
    selected_set: set[str] = set()
    multi_count_classes = [name for name, items in groups.items() if len(items) >= 2]
    multi_count_classes.sort(key=lambda name: (-len(groups[name]), name))
    for name in multi_count_classes:
        if len(selected_ids) >= prompt_size:
            break
        row = groups[name][0]
        selected_ids.append(str(row["article_id"]))
        selected_set.add(str(row["article_id"]))

    while len(selected_ids) < prompt_size:
        candidates = []
        for name, items in groups.items():
            remaining = [row for row in items if str(row["article_id"]) not in selected_set]
            if remaining:
                selected_from_class = sum(1 for row in items if str(row["article_id"]) in selected_set)
                candidates.append((selected_from_class / len(items), -len(items), name, remaining[0]))
        if not candidates:
            break
        _, _, _, row = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0]
        selected_ids.append(str(row["article_id"]))
        selected_set.add(str(row["article_id"]))

    prompt_ids = selected_ids[:prompt_size]
    eval_ids = [str(row["article_id"]) for row in rows if str(row["article_id"]) not in set(prompt_ids)]
    by_id = {str(row["article_id"]): row for row in rows}
    split = {
        "seed": seed,
        "prompt_dev_count": len(prompt_ids),
        "frozen_eval_count": len(eval_ids),
        "prompt_dev_article_ids": prompt_ids,
        "frozen_eval_article_ids": eval_ids,
        "intersection_count": len(set(prompt_ids) & set(eval_ids)),
        "prompt_dev_event_distribution": dict(Counter(target_event_name(by_id[aid]) for aid in prompt_ids)),
        "frozen_eval_event_distribution": dict(Counter(target_event_name(by_id[aid]) for aid in eval_ids)),
        "multi_event_rows": [
            {"article_id": str(row.get("article_id")), "event_names": event_names_from_row(row), "target_event_name": target_event_name(row)}
            for row in rows
            if len(event_names_from_row(row)) > 1
        ],
        "reference_annotation": "AI-assisted, human-confirmed diagnostic set; not an independent manual gold corpus.",
    }
    if len(prompt_ids) != prompt_size or len(eval_ids) != len(rows) - prompt_size or split["intersection_count"] != 0:
        raise RuntimeError(f"unsafe split generated: {split}")
    return split


def content_windows(content: str, max_chars: int = 7200) -> list[dict[str, Any]]:
    text = str(content or "")
    if len(text) <= max_chars:
        return [{"start": 0, "end": len(text), "text": text}]
    spans = [(0, min(4200, len(text))), (max(0, len(text) - 1600), len(text))]
    merged = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    out = []
    total = 0
    for start, end in merged:
        if total >= max_chars:
            break
        take_end = min(end, start + max_chars - total)
        out.append({"start": start, "end": take_end, "text": text[start:take_end]})
        total += take_end - start
    return out


def output_schema() -> dict[str, Any]:
    return {
        "event_name": "必须与输入event_name完全一致",
        "trigger": {"text": "", "start": 0, "end": 0},
        "event_span": {"text": "", "start": 0, "end": 0},
        "subject_company": {"text": "", "start": 0, "end": 0},
        "event_time": {"text": "", "normalized": "YYYY-MM-DD/未明确", "start": 0, "end": 0},
        "object_target": {"text": "", "start": 0, "end": 0},
        "direction": "positive|negative|neutral|unclear",
        "arguments": [{"text": "", "role": "subject_company|event_time|object_target|trigger", "start": 0, "end": 0}],
        "confidence": 0.0,
    }


def prompt_template(version: str) -> str:
    base = """你是金融新闻事件结构抽取器。已知输入文章中需要抽取的事件类别event_name是正确类别，请只基于输入正文片段抽取该事件的一条结构化事件记录。

硬性规则：
1. 只输出一个JSON对象，不输出Markdown代码块、解释或额外字段。
2. event_name必须与输入event_name逐字一致。
3. 所有非空text必须来自输入正文片段原文；start为content全局包含式起点，end为不包含式终点，content[start:end]必须严格等于text。
4. trigger必须位于event_span内部，event_span应覆盖触发词和核心事实，不要整篇新闻。
5. event_time正文未明确时，event_time必须为：text=""，normalized="未明确"，start=-1，end=-1。
6. object_target正文没有明确对象时，object_target必须为：text=""，start=-1，end=-1。
7. direction只能是positive、negative、neutral、unclear之一；无法判断时用unclear。
8. arguments只列出实际抽到的subject_company、event_time、object_target、trigger，role必须使用四个合法枚举。
9. confidence必须是0到1之间的有限数。

字段定义：
- trigger：最能表达事件发生的触发词或短语。
- event_span：包含事件事实的最小连续文本片段。
- subject_company：事件主体公司、机构、产品或行业主体。
- event_time：事件发生或公告时间；没有明确时间时标为未明确。
- object_target：事件影响对象、产品、项目、交易标的或主题对象；没有明确对象时留空。
- direction：事件对主体或行业的影响方向。
- arguments：上述核心论元及角色。
"""
    if version == "strict_self_check":
        base += """
输出前请在内部自检但不要写出自检过程：
- JSON是否能直接解析。
- event_name、direction、confidence是否合法。
- 每个非空text的offset是否能在content中精确切片得到。
- trigger是否在event_span范围内。
- 未明确时间和无明确对象是否使用空值与-1偏移。
"""
    return base


def build_gold_free_input(row: dict[str, Any], input_index: int, split: str) -> dict[str, Any]:
    content = str(row.get("content", ""))
    windows = content_windows(content)
    return {
        "input_index": input_index,
        "split": split,
        "article_id": str(row.get("article_id", "")),
        "event_name": target_event_name(row),
        "title": str(row.get("title", "")),
        "content": content,
        "content_length": len(content),
        "content_windows": windows,
        "publish_time": str(row.get("publish_time", "")),
        "source": str(row.get("source", "")),
    }


def build_user_message(item: dict[str, Any]) -> str:
    window_text = []
    for idx, window in enumerate(item["content_windows"], 1):
        window_text.append(f"[W{idx} offset {window['start']}..{window['end']})\n{window['text']}")
    return (
        f"article_id: {item['article_id']}\n"
        f"event_name: {item['event_name']}\n"
        f"title: {item.get('title', '')}\n"
        f"publish_time: {item.get('publish_time', '')}\n"
        f"source: {item.get('source', '')}\n"
        f"content_length: {item['content_length']}\n\n"
        "正文片段（offset均为原始content全局字符偏移）：\n"
        + "\n\n".join(window_text)
        + "\n\n只输出符合以下Schema的严格JSON：\n"
        + json.dumps(output_schema(), ensure_ascii=False, indent=2)
    )


def correction_user_message(raw_output: str, errors: list[str]) -> str:
    return (
        "你的上一次输出没有通过格式校验。请只修正JSON格式、枚举、offset一致性和Schema，不添加解释，不输出Markdown，不改变依据。\n"
        f"错误原因：{'; '.join(errors[:12])}\n"
        f"JSON Schema：{json.dumps(output_schema(), ensure_ascii=False)}\n"
        f"原始模型输出：{raw_output}"
    )


def chat_completion(
    messages: list[dict[str, str]],
    api_base: str = API_BASE,
    model: str = MODEL_NAME,
    timeout: int = 120,
    max_tokens: int = 1024,
) -> tuple[str, dict[str, Any], float]:
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
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


def check_model_service(api_base: str = API_BASE, model: str = MODEL_NAME, timeout: int = 5) -> dict[str, Any]:
    req = urllib.request.Request(f"{api_base.rstrip('/')}/models", method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"available": False, "model_found": False, "error": f"{type(exc).__name__}:{exc}"}
    model_ids = [str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)]
    return {"available": True, "model_found": model in model_ids, "model_ids": model_ids}


def _coerce_span(value: Any, field: str) -> tuple[dict[str, Any], list[str]]:
    errors = []
    if not isinstance(value, dict):
        return {"text": "", "start": -1, "end": -1}, [f"{field} is not object"]
    text = str(value.get("text", "") or "")
    try:
        start = int(value.get("start", -1))
        end = int(value.get("end", -1))
    except Exception:
        start, end = -1, -1
        errors.append(f"{field} start/end not int")
    out = {"text": text, "start": start, "end": end}
    if field == "event_time":
        out["normalized"] = str(value.get("normalized", "") or "")
    return out, errors


def _repair_span(span: dict[str, Any], content: str, field: str) -> tuple[dict[str, Any], list[str], int]:
    text = str(span.get("text", "") or "")
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    errors: list[str] = []
    repaired = 0
    if not text:
        span["start"] = -1
        span["end"] = -1
        return span, errors, repaired
    if 0 <= start <= end <= len(content) and content[start:end] == text:
        return span, errors, repaired
    positions = [m.start() for m in re.finditer(re.escape(text), content)]
    if positions:
        if start >= 0:
            chosen = min(positions, key=lambda pos: abs(pos - start))
        else:
            chosen = positions[0]
        span["start"] = chosen
        span["end"] = chosen + len(text)
        repaired = 1
        return span, errors, repaired
    errors.append(f"{field} offset invalid and text not found")
    return span, errors, repaired


def validate_and_normalize_output(payload: Any, item: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any] | None, dict[str, Any]]:
    errors: list[str] = []
    repair_count = 0
    extra_fields: list[str] = []
    if not isinstance(payload, dict):
        return False, ["top-level output is not object"], None, {"offset_repair_count": 0, "extra_fields": []}
    expected = {"event_name", "trigger", "event_span", "subject_company", "event_time", "object_target", "direction", "arguments", "confidence"}
    extra_fields = sorted(set(payload) - expected)
    if extra_fields:
        errors.append(f"extra fields: {extra_fields}")
    if str(payload.get("event_name", "")) != str(item["event_name"]):
        errors.append("event_name mismatch")
    try:
        confidence = float(payload.get("confidence"))
        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
            errors.append("confidence out of range")
    except Exception:
        confidence = 0.0
        errors.append("confidence invalid")
    direction = str(payload.get("direction", ""))
    if direction not in DIR_LABELS:
        errors.append("direction invalid")
        direction = normalize_direction(direction)
    content = str(item["content"])
    normalized: dict[str, Any] = {
        "event_name": str(payload.get("event_name", "")),
        "direction": direction if direction in DIR_LABELS else "unclear",
        "confidence": confidence,
    }
    for field in ["trigger", "event_span", "subject_company", "event_time", "object_target"]:
        span, span_errors = _coerce_span(payload.get(field), field)
        errors.extend(span_errors)
        if field == "event_time" and (str(span.get("normalized", "")) in {"未明确", "不明确"} or str(span.get("text", "")) in {"未明确", "不明确"}):
            span["text"] = ""
            span["start"] = -1
            span["end"] = -1
            span["normalized"] = "未明确"
        span, offset_errors, repaired = _repair_span(span, content, field)
        errors.extend(offset_errors)
        repair_count += repaired
        normalized[field] = span
    if normalized["trigger"]["text"] and normalized["event_span"]["text"]:
        trig_start = normalized["trigger"]["start"]
        trig_end = normalized["trigger"]["end"]
        span_start = normalized["event_span"]["start"]
        span_end = normalized["event_span"]["end"]
        if not (span_start <= trig_start < trig_end <= span_end):
            errors.append("trigger not inside event_span")
    args = payload.get("arguments")
    if not isinstance(args, list):
        errors.append("arguments is not list")
        args = []
    parsed_args = []
    for idx, arg in enumerate(args):
        if not isinstance(arg, dict):
            errors.append(f"arguments[{idx}] is not object")
            continue
        role = str(arg.get("role", ""))
        if role not in ARG_ROLES:
            errors.append(f"arguments[{idx}] role invalid")
        span, span_errors = _coerce_span(arg, f"arguments[{idx}]")
        errors.extend(span_errors)
        span, offset_errors, repaired = _repair_span(span, content, f"arguments[{idx}]")
        errors.extend(offset_errors)
        repair_count += repaired
        parsed_args.append({"text": span["text"], "role": role, "start": span["start"], "end": span["end"]})
    normalized["arguments"] = parsed_args
    meta = {"offset_repair_count": repair_count, "extra_fields": extra_fields, "final_illegal_offset_count": sum(1 for err in errors if "offset invalid" in err)}
    return not errors, errors, normalized if not errors else None, meta


def request_structure_one(item: dict[str, Any], prompt: str, args: Any, prompt_version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": build_user_message(item)}]
    raw_outputs = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    errors: list[str] = []
    started = time.perf_counter()
    http_5xx = timeout_count = cuda_oom = 0
    offset_repairs = 0
    final_illegal_offsets = 0
    strict_json_parse_failed = False
    for attempt in range(2):
        try:
            content, full_payload, elapsed = chat_completion(messages, api_base=args.api_base, model=args.model, timeout=args.timeout, max_tokens=args.max_tokens)
            usage = full_payload.get("usage") or {}
            for key in usage_total:
                usage_total[key] += int(usage.get(key, 0) or 0)
            raw_outputs.append({"attempt": attempt + 1, "content": content, "elapsed_seconds": elapsed, "usage": usage})
            try:
                payload = strict_json_loads(content)
                valid, validation_errors, parsed, meta = validate_and_normalize_output(payload, item)
                offset_repairs += int(meta.get("offset_repair_count", 0) or 0)
                final_illegal_offsets = int(meta.get("final_illegal_offset_count", 0) or 0)
            except Exception as exc:
                valid, validation_errors, parsed = False, [f"json_parse_error:{exc}"], None
                strict_json_parse_failed = True
            if valid and parsed is not None:
                pred = {
                    "input_index": item["input_index"],
                    "split": item["split"],
                    "prompt_version": prompt_version,
                    "article_id": item["article_id"],
                    "status": "success",
                    "prediction": parsed,
                    "retry_count": attempt,
                    "offset_repair_count": offset_repairs,
                }
                metric = {
                    "input_index": item["input_index"],
                    "split": item["split"],
                    "prompt_version": prompt_version,
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
                    "illegal_direction_count": 0,
                    "illegal_confidence_count": 0,
                    "final_illegal_offset_count": 0,
                    "offset_repair_count": offset_repairs,
                    "http_5xx_count": http_5xx,
                    "timeout_count": timeout_count,
                    "cuda_oom_count": cuda_oom,
                }
                raw = {
                    "input_index": item["input_index"],
                    "split": item["split"],
                    "prompt_version": prompt_version,
                    "article_id": item["article_id"],
                    "status": "success",
                    "raw_outputs": raw_outputs,
                }
                return raw, pred, metric
            errors = validation_errors
            final_illegal_offsets = sum(1 for err in errors if "offset invalid" in err)
            if attempt == 0:
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": correction_user_message(content, validation_errors)}]
        except RuntimeError as exc:
            msg = str(exc)
            errors = [msg]
            lower = msg.lower()
            if "http_error:5" in lower:
                http_5xx += 1
            if "timed out" in lower or "timeout" in lower:
                timeout_count += 1
            if "cuda" in lower and "oom" in lower:
                cuda_oom += 1
            if attempt == 0:
                continue
    metric = {
        "input_index": item["input_index"],
        "split": item["split"],
        "prompt_version": prompt_version,
        "article_id": item["article_id"],
        "status": "failed",
        "request_count": max(1, len(raw_outputs)),
        "retry_count": max(0, len(raw_outputs) - 1),
        "latency_seconds": round(time.perf_counter() - started, 6),
        "prompt_tokens": usage_total["prompt_tokens"],
        "completion_tokens": usage_total["completion_tokens"],
        "total_tokens": usage_total["total_tokens"],
        "json_parse_failed": strict_json_parse_failed,
        "format_validation_failed": True,
        "illegal_event_name_count": 1 if any("event_name" in err for err in errors) else 0,
        "illegal_direction_count": 1 if any("direction" in err for err in errors) else 0,
        "illegal_confidence_count": 1 if any("confidence" in err for err in errors) else 0,
        "final_illegal_offset_count": final_illegal_offsets,
        "offset_repair_count": offset_repairs,
        "http_5xx_count": http_5xx,
        "timeout_count": timeout_count,
        "cuda_oom_count": cuda_oom,
        "errors": "|".join(errors[:20]),
    }
    pred = {
        "input_index": item["input_index"],
        "split": item["split"],
        "prompt_version": prompt_version,
        "article_id": item["article_id"],
        "status": "failed",
        "prediction": None,
        "errors": errors,
        "retry_count": metric["retry_count"],
    }
    raw = {
        "input_index": item["input_index"],
        "split": item["split"],
        "prompt_version": prompt_version,
        "article_id": item["article_id"],
        "status": "failed",
        "raw_outputs": raw_outputs,
        "errors": errors,
    }
    return raw, pred, metric


def summarize_request_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in rows]
    tokens = [float(row.get("total_tokens", 0) or 0) for row in rows]

    def p95(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(statistics.quantiles(values, n=20, method="inclusive")[18]) if len(values) > 1 else float(values[0])

    success = [row for row in rows if row.get("status") == "success"]
    return {
        "record_count": len(rows),
        "success_count": len(success),
        "failed_count": len(rows) - len(success),
        "request_count": sum(int(row.get("request_count", 0) or 0) for row in rows),
        "retry_count": sum(int(row.get("retry_count", 0) or 0) for row in rows),
        "json_parse_failed_count": sum(1 for row in rows if row.get("json_parse_failed")),
        "format_validation_failed_count": sum(1 for row in rows if row.get("format_validation_failed")),
        "illegal_event_name_count": sum(int(row.get("illegal_event_name_count", 0) or 0) for row in rows),
        "illegal_direction_count": sum(int(row.get("illegal_direction_count", 0) or 0) for row in rows),
        "illegal_confidence_count": sum(int(row.get("illegal_confidence_count", 0) or 0) for row in rows),
        "final_illegal_offset_count": sum(int(row.get("final_illegal_offset_count", 0) or 0) for row in rows),
        "offset_repair_count": sum(int(row.get("offset_repair_count", 0) or 0) for row in rows),
        "http_5xx_count": sum(int(row.get("http_5xx_count", 0) or 0) for row in rows),
        "timeout_count": sum(int(row.get("timeout_count", 0) or 0) for row in rows),
        "cuda_oom_count": sum(int(row.get("cuda_oom_count", 0) or 0) for row in rows),
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_seconds": p95(latencies),
        "avg_total_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
        "p95_total_tokens": p95(tokens),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens", 0) or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in rows),
    }


def _prediction_fields(prediction: dict[str, Any] | None) -> dict[str, Any]:
    if not prediction:
        return {
            "trigger": "",
            "event_span": "",
            "subject_company": "",
            "event_time_text": "",
            "event_time_normalized": "",
            "object_target": "",
            "direction": "unclear",
            "arguments": [],
        }
    return {
        "trigger": prediction.get("trigger", {}).get("text", ""),
        "event_span": prediction.get("event_span", {}).get("text", ""),
        "subject_company": prediction.get("subject_company", {}).get("text", ""),
        "event_time_text": prediction.get("event_time", {}).get("text", ""),
        "event_time_normalized": prediction.get("event_time", {}).get("normalized", ""),
        "object_target": prediction.get("object_target", {}).get("text", ""),
        "direction": prediction.get("direction", "unclear"),
        "arguments": prediction.get("arguments", []),
    }


def evaluate_structure_predictions(
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gold_by_id = {str(row["article_id"]): row for row in annotations}
    pred_by_id = {str(row["article_id"]): row for row in predictions}
    detail_rows: list[dict[str, Any]] = []
    direction_true: list[str] = []
    direction_pred: list[str] = []
    arg_pred_all: set[tuple[str, str]] = set()
    arg_gold_all: set[tuple[str, str]] = set()
    argrole_pred_all: set[tuple[str, str, str]] = set()
    argrole_gold_all: set[tuple[str, str, str]] = set()
    field_errors: dict[str, Counter[str]] = defaultdict(Counter)
    for aid, gold in gold_by_id.items():
        pred_row = pred_by_id.get(aid, {})
        prediction = pred_row.get("prediction") if pred_row.get("status") == "success" else None
        fields = _prediction_fields(prediction)
        gold_args = parse_gold_args(gold["annotate_arguments_json"])
        gold_roles = parse_gold_roles(gold["annotate_argument_roles_json"])
        pred_args = fields["arguments"]
        gold_arg_texts = {norm_text(arg.get("text", "")) for arg in gold_args if norm_text(arg.get("text", ""))}
        pred_arg_texts = {norm_text(arg.get("text", "")) for arg in pred_args if norm_text(arg.get("text", ""))}
        gold_arg_roles = {(norm_text(text), str(role)) for text, role in gold_roles.items() if norm_text(text)}
        pred_arg_roles = {(norm_text(arg.get("text", "")), str(arg.get("role", ""))) for arg in pred_args if norm_text(arg.get("text", ""))}
        for text in gold_arg_texts:
            arg_gold_all.add((aid, text))
        for text in pred_arg_texts:
            arg_pred_all.add((aid, text))
        for text, role in gold_arg_roles:
            argrole_gold_all.add((aid, text, role))
        for text, role in pred_arg_roles:
            argrole_pred_all.add((aid, text, role))
        dir_gold = normalize_direction(gold["annotate_direction"])
        dir_pred = normalize_direction(fields["direction"])
        direction_true.append(dir_gold)
        direction_pred.append(dir_pred)
        pred_time = fields["event_time_normalized"] or fields["event_time_text"]
        flags = {
            "trigger": bool(fields["trigger"]) and norm_text(fields["trigger"]) == norm_text(gold["annotate_trigger"]),
            "event_span": bool(fields["event_span"]) and norm_text(fields["event_span"]) == norm_text(gold["annotate_event_span"]),
            "subject_company": bool(fields["subject_company"]) and loose_text_equal(fields["subject_company"], gold["annotate_subject_company"]),
            "event_time": normalize_date(pred_time) == normalize_date(gold["annotate_event_time"]),
            "object_target": (not norm_text(gold["annotate_object_target"]) and not norm_text(fields["object_target"]))
            or loose_text_equal(fields["object_target"], gold["annotate_object_target"]),
            "direction": dir_gold == dir_pred,
            "arguments": not (gold_arg_texts - pred_arg_texts) and not (pred_arg_texts - gold_arg_texts),
            "argument_roles": not (gold_arg_roles - pred_arg_roles) and not (pred_arg_roles - gold_arg_roles),
        }
        for field, ok in flags.items():
            if ok:
                continue
            if field in {"arguments", "argument_roles"}:
                missing = not pred_arg_texts if field == "arguments" else not pred_arg_roles
                if not missing and (pred_arg_texts - gold_arg_texts if field == "arguments" else pred_arg_roles - gold_arg_roles):
                    field_errors[field]["extra"] += 1
            elif field == "event_time":
                missing = normalize_date(pred_time) == ""
            else:
                key_map = {"trigger": "trigger", "event_span": "event_span", "subject_company": "subject_company", "object_target": "object_target", "direction": "direction"}
                missing = not str(fields.get(key_map.get(field, field), "") or "").strip()
            field_errors[field]["missing" if missing else "wrong"] += 1
        trigger_overlap = char_overlap_f1(norm_text(fields["trigger"]), norm_text(gold["annotate_trigger"]))
        span_overlap = char_overlap_f1(norm_text(fields["event_span"]), norm_text(gold["annotate_event_span"]))
        detail_rows.append(
            {
                "scope": scope,
                "article_id": aid,
                "event_name": target_event_name(gold),
                "all_gold_event_names": "|".join(event_names_from_row(gold)),
                "prediction_status": pred_row.get("status", "missing"),
                "gold_trigger": gold["annotate_trigger"],
                "pred_trigger": fields["trigger"],
                "trigger_exact": flags["trigger"],
                "trigger_overlap_f1": trigger_overlap,
                "gold_event_span": gold["annotate_event_span"],
                "pred_event_span": fields["event_span"],
                "event_span_exact": flags["event_span"],
                "event_span_overlap_f1": span_overlap,
                "gold_subject_company": gold["annotate_subject_company"],
                "pred_subject_company": fields["subject_company"],
                "subject_exact": str(fields["subject_company"]) == str(gold["annotate_subject_company"]),
                "subject_normalized": flags["subject_company"],
                "gold_event_time": gold["annotate_event_time"],
                "pred_event_time": pred_time,
                "event_time_exact": str(pred_time) == str(gold["annotate_event_time"]),
                "event_time_normalized": flags["event_time"],
                "unspecified_time_correct": (normalize_date(pred_time) == "未明确") == (normalize_date(gold["annotate_event_time"]) == "未明确"),
                "gold_object_target": gold["annotate_object_target"],
                "pred_object_target": fields["object_target"],
                "object_exact": str(fields["object_target"]) == str(gold["annotate_object_target"]),
                "object_normalized": flags["object_target"],
                "gold_direction": dir_gold,
                "pred_direction": dir_pred,
                "direction_correct": flags["direction"],
                "gold_arg_texts": "|".join(sorted(gold_arg_texts)),
                "pred_arg_texts": "|".join(sorted(pred_arg_texts)),
                "gold_arg_roles": "|".join(f"{text}:{role}" for text, role in sorted(gold_arg_roles)),
                "pred_arg_roles": "|".join(f"{text}:{role}" for text, role in sorted(pred_arg_roles)),
                "argument_exact_set": flags["arguments"],
                "argument_role_exact_set": flags["argument_roles"],
                "full_structure_exact": all(flags.values()),
                "at_least_one_structure_field_correct": any(flags.values()),
            }
        )
    n = len(detail_rows)
    trigger_pred_count = sum(1 for row in detail_rows if row["pred_trigger"])
    trigger_tp = sum(1 for row in detail_rows if row["trigger_exact"])
    trigger_p = trigger_tp / trigger_pred_count if trigger_pred_count else 0.0
    trigger_r = trigger_tp / n if n else 0.0
    trigger_f1 = 2 * trigger_p * trigger_r / (trigger_p + trigger_r) if trigger_p + trigger_r else 0.0
    direction_rows = []
    direction_f1s = []
    for label in DIR_LABELS:
        tp = sum(1 for g, p in zip(direction_true, direction_pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(direction_true, direction_pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(direction_true, direction_pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        direction_f1s.append(f1)
        direction_rows.append({"scope": scope, "label": label, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})
    arg_metrics = set_f1(arg_pred_all, arg_gold_all)
    argrole_metrics = set_f1(argrole_pred_all, argrole_gold_all)
    report = {
        "scope": scope,
        "sample_count": n,
        "success_count": sum(1 for row in detail_rows if row["prediction_status"] == "success"),
        "trigger": {
            "exact_precision": trigger_p,
            "exact_recall": trigger_r,
            "exact_f1": trigger_f1,
            "relaxed_overlap_f1": sum(row["trigger_overlap_f1"] for row in detail_rows) / n if n else 0.0,
        },
        "event_span": {
            "exact_match_accuracy": sum(1 for row in detail_rows if row["event_span_exact"]) / n if n else 0.0,
            "character_overlap_f1": sum(row["event_span_overlap_f1"] for row in detail_rows) / n if n else 0.0,
        },
        "subject_company": {
            "exact_accuracy": sum(1 for row in detail_rows if row["subject_exact"]) / n if n else 0.0,
            "normalized_accuracy": sum(1 for row in detail_rows if row["subject_normalized"]) / n if n else 0.0,
        },
        "event_time": {
            "exact_accuracy": sum(1 for row in detail_rows if row["event_time_exact"]) / n if n else 0.0,
            "normalized_date_accuracy": sum(1 for row in detail_rows if row["event_time_normalized"]) / n if n else 0.0,
            "unspecified_recognition_accuracy": sum(1 for row in detail_rows if row["unspecified_time_correct"]) / n if n else 0.0,
        },
        "object_target": {
            "exact_accuracy": sum(1 for row in detail_rows if row["object_exact"]) / n if n else 0.0,
            "normalized_accuracy": sum(1 for row in detail_rows if row["object_normalized"]) / n if n else 0.0,
        },
        "direction": {
            "accuracy": sum(1 for row in detail_rows if row["direction_correct"]) / n if n else 0.0,
            "fixed_label_macro_f1": sum(direction_f1s) / len(direction_f1s) if direction_f1s else 0.0,
            "confusion_matrix": direction_rows,
        },
        "arguments": {
            "argument_identification_precision": arg_metrics["precision"],
            "argument_identification_recall": arg_metrics["recall"],
            "argument_identification_f1": arg_metrics["f1"],
            "argument_role_classification_macro_f1": role_macro_f1(argrole_pred_all, argrole_gold_all),
            "end_to_end_argument_role_f1": argrole_metrics["f1"],
            "argument_tp": arg_metrics["tp"],
            "argument_fp": arg_metrics["fp"],
            "argument_fn": arg_metrics["fn"],
            "argument_role_tp": argrole_metrics["tp"],
            "argument_role_fp": argrole_metrics["fp"],
            "argument_role_fn": argrole_metrics["fn"],
        },
        "complete_structure": {
            "article_exact_match_accuracy": sum(1 for row in detail_rows if row["full_structure_exact"]) / n if n else 0.0,
            "at_least_one_structure_field_correct_ratio": sum(1 for row in detail_rows if row["at_least_one_structure_field_correct"]) / n if n else 0.0,
            "all_fields_correct_article_count": sum(1 for row in detail_rows if row["full_structure_exact"]),
            "field_error_counts": {field: dict(counts) for field, counts in sorted(field_errors.items())},
        },
    }
    return report, detail_rows, direction_rows


def prompt_selection_score(metrics: dict[str, Any]) -> float:
    return (
        0.20 * metrics["trigger"]["relaxed_overlap_f1"]
        + 0.20 * metrics["event_span"]["character_overlap_f1"]
        + 0.15 * metrics["subject_company"]["normalized_accuracy"]
        + 0.15 * metrics["event_time"]["normalized_date_accuracy"]
        + 0.10 * metrics["direction"]["accuracy"]
        + 0.10 * metrics["arguments"]["end_to_end_argument_role_f1"]
        + 0.10 * metrics["complete_structure"]["at_least_one_structure_field_correct_ratio"]
    )


def acceptance_status(eval_metrics: dict[str, Any], request_summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    gates = {
        "success_60_of_60": request_summary["success_count"] == 60 and eval_metrics["sample_count"] == 60,
        "illegal_json_zero": request_summary["json_parse_failed_count"] == 0,
        "illegal_event_name_zero": request_summary["illegal_event_name_count"] == 0,
        "final_illegal_offset_zero": request_summary["final_illegal_offset_count"] == 0,
        "trigger_relaxed_f1_ge_0_60": eval_metrics["trigger"]["relaxed_overlap_f1"] >= 0.60,
        "event_span_char_f1_ge_0_60": eval_metrics["event_span"]["character_overlap_f1"] >= 0.60,
        "subject_norm_acc_ge_0_65": eval_metrics["subject_company"]["normalized_accuracy"] >= 0.65,
        "event_time_norm_acc_ge_0_55": eval_metrics["event_time"]["normalized_date_accuracy"] >= 0.55,
        "direction_acc_ge_0_65": eval_metrics["direction"]["accuracy"] >= 0.65,
        "argument_role_f1_ge_0_45": eval_metrics["arguments"]["end_to_end_argument_role_f1"] >= 0.45,
        "at_least_one_field_ratio_ge_0_85": eval_metrics["complete_structure"]["at_least_one_structure_field_correct_ratio"] >= 0.85,
        "cuda_oom_zero": request_summary["cuda_oom_count"] == 0,
    }
    if all(gates.values()):
        return "conditional_structure_32b_passed", gates
    good_core = (
        gates["success_60_of_60"]
        and gates["illegal_json_zero"]
        and gates["illegal_event_name_zero"]
        and gates["final_illegal_offset_zero"]
        and gates["cuda_oom_zero"]
        and eval_metrics["complete_structure"]["at_least_one_structure_field_correct_ratio"] >= 0.85
    )
    if good_core and (
        eval_metrics["trigger"]["relaxed_overlap_f1"] >= 0.60
        or eval_metrics["event_span"]["character_overlap_f1"] >= 0.60
        or eval_metrics["subject_company"]["normalized_accuracy"] >= 0.65
    ):
        return "hybrid_structure_baseline_recommended", gates
    if request_summary["cuda_oom_count"] or request_summary["timeout_count"] or request_summary["http_5xx_count"]:
        return "blocked", gates
    return "structure_training_data_expansion_required", gates
