from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.event_credibility import financial_consistency
from src.event_credibility.scoring import (
    CLARIFICATION_PATTERNS,
    OFFICIAL_PATTERNS,
    REVERSAL_PATTERNS,
    RUMOR_PATTERNS,
    ROOT,
    clamp,
    credibility_level,
    source_profile,
    write_jsonl,
)


ARTICLE_COLUMNS = [
    "dataset_split",
    "article_id",
    "article_title",
    "article_source",
    "article_publish_time",
    "content",
    "content_length",
    "source_type",
    "source_reliability",
]

GENERIC_TITLE_TERMS = {
    "NEWS",
    "MULTISOURCE",
    "http",
    "https",
    "股市必读",
    "异动提醒",
    "公司公告",
    "公告",
}

VALID_LLM_VERDICTS = {"support", "contradict", "insufficient"}
MAX_LLM_REASON_CHARS = 400
MAX_LLM_RAW_RESPONSE_CHARS = 2000
DEFAULT_LLM_MAX_TOKENS = 512
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_TIMEOUT = 180.0
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_FORMAT_RETRIES = 1
DEFAULT_LLM_WORKERS = 1
DEFAULT_LLM_CHECKPOINT_EVERY = 10
ALLOWED_LLM_WORKERS = {1, 2, 4, 8}

DISCLAIMER_PATTERNS = (
    "不存在任何虚假记载",
    "没有虚假记载",
    "虚假记载、误导性陈述",
    "信息披露的内容真实、准确、完整",
    "真实性、准确性和完整性承担法律责任",
)

STRICT_CONTRADICTION_PATTERNS = (
    "澄清",
    "辟谣",
    "否认",
    "不属实",
    "失实",
    "假消息",
    "传闻不实",
    "消息不实",
    "与公司无关",
    "与其无关",
)

NON_EVENT_CONTRADICTION_PATTERNS = (
    "辟谣作品",
    "辟谣论坛",
    "金融辟谣",
    "网络辟谣",
    "防非反诈",
    "反诈宣传",
    "不予澄清",
    "并未否认",
    "并未直接否认",
    "未直接否认",
    "没有否认",
    "不否认",
)


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_boilerplate_disclaimer(sentence: str) -> bool:
    value = normalize_text(sentence)
    if not value:
        return False
    if "重大遗漏" in value and any(pattern in value for pattern in DISCLAIMER_PATTERNS):
        return True
    return "本公司董事会" in value and "保证" in value and any(pattern in value for pattern in DISCLAIMER_PATTERNS)


def is_informative_event_term(term: str, target_name: str = "") -> bool:
    value = normalize_text(term)
    if not value or value == normalize_text(target_name):
        return False
    if value in GENERIC_TITLE_TERMS:
        return False
    if re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        return False
    if re.fullmatch(r"\d{6}", value):
        return False
    return len(value) >= 3


def has_true_contradiction_signal(sentence: str) -> bool:
    value = normalize_text(sentence)
    if not value or is_boilerplate_disclaimer(value):
        return False
    if any(pattern in value for pattern in NON_EVENT_CONTRADICTION_PATTERNS):
        return False
    if any(pattern in value for pattern in ("辟谣", "不属实", "失实", "假消息", "传闻不实", "消息不实")):
        return True
    if "否认" in value:
        return True
    if "澄清" in value and any(pattern in value for pattern in RUMOR_PATTERNS + CLARIFICATION_PATTERNS + ("无关", "未参与")):
        return True
    if "虚假" in value and any(pattern in value for pattern in RUMOR_PATTERNS + ("传闻不实", "消息不实")):
        return True
    if "撤回" in value and any(pattern in value for pattern in ("此前报道", "相关报道", "传闻", "网传", "申报材料")):
        return True
    if "更正" in value and any(pattern in value for pattern in ("此前报道", "原公告", "误写", "错误", "不准确")):
        return True
    return False


def read_raw_news_excel(path: Path, split: str, labeled: bool) -> pd.DataFrame:
    df = pd.read_excel(path)
    rename = {
        "article_file_id": "article_id",
        "article_title": "article_title",
        "article_source": "article_source",
        "article_publish_time": "article_publish_time",
    }
    df = df.rename(columns=rename)
    keep = ["article_id", "article_title", "article_source", "article_publish_time", "content"]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[keep].copy()
    out["dataset_split"] = split
    out["is_labeled_source"] = bool(labeled)
    return out


def build_fulltext_article_index(raw_dir: Path, output_path: Path, force: bool = False) -> pd.DataFrame:
    if output_path.exists() and not force:
        return pd.read_parquet(output_path)

    specs = [
        ("news_without_tags_train.xlsx", "train", False),
        ("news_without_tags_test.xlsx", "test", False),
        ("news_with_tags_train.xlsx", "labeled_train", True),
        ("news_with_tags_test.xlsx", "labeled_test", True),
    ]
    frames = []
    for filename, split, labeled in specs:
        path = raw_dir / filename
        if not path.exists():
            continue
        frames.append(read_raw_news_excel(path, split=split, labeled=labeled))
    if not frames:
        raise FileNotFoundError(f"no raw news Excel files found under {raw_dir}")

    articles = pd.concat(frames, ignore_index=True)
    for col in ("article_id", "article_title", "article_source", "content"):
        articles[col] = articles[col].map(normalize_text)
    articles["article_publish_time"] = articles["article_publish_time"].astype(str).map(normalize_text)
    articles["content_length"] = articles["content"].str.len().astype(int)
    profiles = articles["article_source"].map(source_profile)
    articles["source_type"] = [p[0] for p in profiles]
    articles["source_reliability"] = [p[1] for p in profiles]
    articles = articles.sort_values(
        ["article_id", "content_length", "source_reliability", "is_labeled_source"],
        ascending=[True, False, False, False],
    ).drop_duplicates("article_id", keep="first").reset_index(drop=True)
    articles = articles[ARTICLE_COLUMNS + ["is_labeled_source"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(output_path, index=False)
    return articles


def split_sentences(text: str, max_sentences: int = 80) -> list[str]:
    value = normalize_text(text)
    if not value:
        return []
    pieces = re.split(r"(?<=[。！？!?；;])|\n+", value)
    sentences = []
    for piece in pieces:
        sentence = normalize_text(piece)
        if len(sentence) < 8:
            continue
        sentences.append(sentence[:320])
        if len(sentences) >= max_sentences:
            break
    if not sentences and value:
        sentences.append(value[:320])
    return sentences


def keyword_terms(row: dict) -> list[str]:
    terms = []
    for field in ("target_name", "event_name"):
        value = normalize_text(row.get(field, ""))
        if value and value not in terms:
            terms.append(value)
    schema = normalize_text(row.get("schema_id", ""))
    if "_" in schema:
        tail = schema.split("_")[-1]
        if tail and tail not in terms:
            terms.append(tail)
    title_samples = normalize_text(row.get("title_samples", ""))
    for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{3,}", title_samples):
        if token in terms:
            continue
        if token in GENERIC_TITLE_TERMS:
            continue
        if len(terms) >= 12:
            break
        terms.append(token)
    return terms


def sentence_score(sentence: str, terms: list[str], source_type: str, source_reliability: float) -> float:
    score = 0.0
    for idx, term in enumerate(terms):
        if term and term in sentence:
            score += 3.0 if idx == 0 else 1.3
    if any(word in sentence for word in OFFICIAL_PATTERNS):
        score += 1.5
    if has_true_contradiction_signal(sentence):
        score += 2.5
    if any(word in sentence for word in RUMOR_PATTERNS):
        score += 1.0
    if source_type == "official":
        score += 1.2
    elif source_type == "mainstream":
        score += 0.8
    score += source_reliability
    return score


def extract_evidence_sentences(article: dict, terms: list[str], max_sentences: int = 3) -> list[dict]:
    sentences = split_sentences(article.get("content", ""))
    ranked = []
    for sentence in sentences:
        if is_boilerplate_disclaimer(sentence):
            continue
        score = sentence_score(
            sentence,
            terms,
            str(article.get("source_type", "unknown")),
            float(article.get("source_reliability", 0.25) or 0.25),
        )
        if score <= 0:
            continue
        ranked.append((score, sentence))
    ranked.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for score, sentence in ranked:
        key = sentence[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "article_id": article.get("article_id", ""),
                "title": article.get("article_title", ""),
                "source": article.get("article_source", ""),
                "source_type": article.get("source_type", ""),
                "publish_time": article.get("article_publish_time", ""),
                "sentence": sentence,
                "sentence_score": round(float(score), 4),
            }
        )
        if len(out) >= max_sentences:
            break
    return out


def classify_evidence(evidence_items: list[dict], terms: list[str]) -> dict:
    support_count = 0
    contradict_count = 0
    official_support_count = 0
    evidence_source_counter = Counter()
    support_source_counter = Counter()
    reasons = []
    target_name = terms[0] if terms else ""
    event_terms = [term for term in terms[1:] if is_informative_event_term(term, target_name)]
    for item in evidence_items:
        sentence = item.get("sentence", "")
        title = item.get("title", "")
        combined_text = f"{title} {sentence}"
        source_type = item.get("source_type", "")
        source = item.get("source", "unknown")
        evidence_source_counter[source] += 1
        has_target = bool(target_name and target_name in combined_text)
        has_event = any(term and term in combined_text for term in event_terms)
        sentence_has_target = bool(target_name and target_name in sentence)
        sentence_has_event = any(term and term in sentence for term in event_terms)
        title_has_target_reversal = bool(target_name and target_name in title and has_true_contradiction_signal(title))
        has_reversal = (
            has_true_contradiction_signal(sentence) and (sentence_has_target or sentence_has_event)
        ) or (
            title_has_target_reversal and (has_event or sentence_has_target)
        )
        has_rumor = any(word in sentence for word in RUMOR_PATTERNS)
        if has_reversal and (has_target or has_event or has_rumor):
            contradict_count += 1
            reasons.append("正文存在否认/澄清/不属实/虚假等反证句")
        elif has_target and has_event:
            support_count += 1
            support_source_counter[source] += 1
            if source_type == "official":
                official_support_count += 1
    source_count = len(support_source_counter)
    support_score = clamp(
        0.35 * min(official_support_count, 2) / 2
        + 0.25 * min(support_count, 4) / 4
        + 0.25 * min(source_count, 5) / 5
        + 0.15 * int(support_count >= 2 and source_count >= 2)
    )
    contradiction_score = clamp(0.55 * min(contradict_count, 2) / 2 + 0.20 * int(contradict_count > 0))
    if support_count == 0 and contradict_count == 0:
        status = "insufficient"
    elif contradict_count > 0:
        status = "contradict"
    else:
        status = "support"
    return {
        "fulltext_evidence_status": status,
        "fulltext_support_count": support_count,
        "fulltext_contradict_count": contradict_count,
        "fulltext_official_support_count": official_support_count,
        "fulltext_distinct_source_count": source_count,
        "fulltext_support_score": support_score,
        "fulltext_contradiction_score": contradiction_score,
        "fulltext_reason": "；".join(dict.fromkeys(reasons)),
    }


def build_target_article_candidates(articles: pd.DataFrame, target_names: list[str]) -> dict[str, list[int]]:
    escaped = [re.escape(x) for x in sorted({x for x in target_names if x}, key=len, reverse=True)]
    if not escaped:
        return {}
    pattern = re.compile("|".join(escaped))
    candidates: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(articles.to_dict("records")):
        text = f"{row.get('article_title', '')} {str(row.get('content', ''))[:1200]}"
        matches = set(pattern.findall(text))
        for match in matches:
            candidates[match].append(position)
    return candidates


def build_fulltext_evidence(
    scores: pd.DataFrame,
    members: pd.DataFrame,
    articles: pd.DataFrame,
    max_extra_articles: int = 8,
    max_member_articles: int = 6,
) -> pd.DataFrame:
    article_by_id = {str(row.article_id): row._asdict() for row in articles.itertuples(index=False)}
    member_by_cluster: dict[str, list[str]] = defaultdict(list)
    for row in members.to_dict("records"):
        member_by_cluster[str(row.get("cluster_id"))].append(str(row.get("article_id")))

    target_candidates = build_target_article_candidates(articles, sorted(scores["target_name"].dropna().astype(str).unique()))
    output = []
    for row in scores.to_dict("records"):
        cluster_id = str(row["cluster_id"])
        terms = keyword_terms(row)
        article_ids = []
        for article_id in member_by_cluster.get(cluster_id, [])[:max_member_articles]:
            if article_id in article_by_id:
                article_ids.append(article_id)
        extra_rows = []
        for idx in target_candidates.get(str(row.get("target_name", "")), []):
            article = articles.iloc[idx].to_dict()
            if str(article.get("article_id")) in article_ids:
                continue
            search_text = f"{article.get('article_title', '')} {str(article.get('content', ''))[:1800]}"
            score = sentence_score(search_text, terms, str(article.get("source_type", "")), float(article.get("source_reliability", 0.25)))
            if score <= 0:
                continue
            extra_rows.append((score, article))
        extra_rows.sort(key=lambda x: x[0], reverse=True)
        for _, article in extra_rows[:max_extra_articles]:
            article_ids.append(str(article.get("article_id")))

        evidence_items = []
        for article_id in article_ids:
            article = article_by_id.get(article_id)
            if not article:
                continue
            evidence_items.extend(extract_evidence_sentences(article, terms, max_sentences=2))
        evidence_items = sorted(evidence_items, key=lambda x: x.get("sentence_score", 0), reverse=True)[:12]
        for idx, item in enumerate(evidence_items, 1):
            item["evidence_id"] = f"E{idx}"
        classified = classify_evidence(evidence_items, terms)
        output.append(
            {
                "cluster_id": cluster_id,
                "target_code": row.get("target_code", ""),
                "target_name": row.get("target_name", ""),
                "schema_id": row.get("schema_id", ""),
                "event_name": row.get("event_name", ""),
                "retrieved_article_count": len(set(article_ids)),
                "evidence_sentence_count": len(evidence_items),
                "query_terms": json.dumps(terms, ensure_ascii=False),
                "evidence_items": json.dumps(evidence_items, ensure_ascii=False),
                **classified,
            }
        )
    return pd.DataFrame(output)


@dataclass
class LLMParseResult:
    parsed: dict[str, Any] | None
    format: str
    error: str


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_json_constant)


def _find_first_json_value(text: str) -> Any | None:
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    for match in re.finditer(r"[{\[]", text):
        candidate = text[match.start() :]
        try:
            obj, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, (dict, list)):
            return obj
    return None


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", "", str(text or ""), flags=re.S | re.I)


def _extract_tag_content(text: str, tag: str) -> str | None:
    match = re.search(fr"<{tag}\b[^>]*>(.*?)</{tag}>", str(text or ""), re.S | re.I)
    if not match:
        return None
    return match.group(1)


def _extract_markdown_json_block(text: str) -> str | None:
    match = re.search(r"```(?:json|JSON)?\s*(.*?)\s*```", str(text or ""), re.S)
    if not match:
        return None
    return match.group(1)


def _parse_json_candidate(text: str) -> Any | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        return _strict_json_loads(stripped)
    except Exception:
        return _find_first_json_value(stripped)


def _candidate_llm_texts(text: str) -> list[tuple[str, str]]:
    stripped = str(text or "").strip()
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(format_name: str, candidate: str | None) -> None:
        value = str(candidate or "").strip()
        if not value:
            return
        key = (format_name, value)
        if key in seen:
            return
        seen.add(key)
        candidates.append(key)

    add("pure_json", stripped)
    add("answer_wrapped_json", _extract_tag_content(stripped, "answer"))
    add("markdown_json_block", _extract_markdown_json_block(stripped))
    without_think = _strip_think_blocks(stripped).strip()
    if without_think and without_think != stripped:
        add("think_then_json", without_think)
    add("json_with_extra_text", stripped)
    return candidates


def parse_llm_review_content(content: str) -> LLMParseResult:
    text = str(content or "").strip()
    if not text:
        return LLMParseResult(None, "empty", "empty content")

    errors = []
    for format_name, candidate in _candidate_llm_texts(text):
        if format_name == "pure_json":
            try:
                parsed = _strict_json_loads(candidate)
            except Exception:
                parsed = None
        else:
            parsed = _parse_json_candidate(candidate)
        if isinstance(parsed, dict):
            return LLMParseResult(parsed, format_name, "")
        if parsed is None:
            errors.append(f"{format_name}: no JSON object found")
        else:
            errors.append(f"{format_name}: parsed JSON is not an object")

    return LLMParseResult(None, "unparseable", "; ".join(dict.fromkeys(errors)) or "no JSON object found")


def safe_json_loads(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    parsed = _parse_json_candidate(str(text or "").strip())
    return parsed if parsed is not None else {}


def validate_llm_review_object(obj: Any, evidence_ids: Iterable[str] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return {"valid": False, "errors": ["parsed result is not an object"]}

    verdict = obj.get("verdict")
    if isinstance(verdict, str):
        verdict = verdict.strip()
    if verdict not in VALID_LLM_VERDICTS:
        errors.append(f"invalid verdict: {verdict!r}")

    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append(f"confidence is not numeric: {confidence!r}")
    else:
        confidence_value = float(confidence)
        if not math.isfinite(confidence_value):
            errors.append(f"confidence is not finite: {confidence!r}")
        elif not 0.0 <= confidence_value <= 1.0:
            errors.append(f"confidence out of range [0, 1]: {confidence!r}")

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason is empty or not a string")
    elif len(reason.strip()) > MAX_LLM_REASON_CHARS:
        errors.append(f"reason exceeds max length {MAX_LLM_REASON_CHARS}")

    raw_evidence_ids = obj.get("evidence_ids")
    normalized_evidence_ids: list[str] = []
    if not isinstance(raw_evidence_ids, list) or not all(isinstance(x, str) for x in raw_evidence_ids):
        errors.append("evidence_ids is not a string list")
    else:
        normalized_evidence_ids = [x.strip() for x in raw_evidence_ids if x.strip()]
        if len(normalized_evidence_ids) != len(raw_evidence_ids):
            errors.append("evidence_ids contains empty strings")
        if evidence_ids is not None:
            allowed_ids = {str(x).strip() for x in evidence_ids if str(x).strip()}
            missing = [x for x in normalized_evidence_ids if x not in allowed_ids]
            if missing:
                errors.append(f"unknown evidence_ids: {missing}")

    if verdict in {"support", "contradict"} and not normalized_evidence_ids:
        errors.append("evidence_ids must be non-empty for support/contradict verdicts")

    return {"valid": not errors, "errors": errors}


def canonicalize_llm_review_object(obj: dict[str, Any]) -> dict[str, Any]:
    verdict = str(obj.get("verdict", "")).strip()
    confidence_value = obj.get("confidence")
    confidence = float(confidence_value) if confidence_value is not None else 0.0
    reason = str(obj.get("reason", "")).strip()
    evidence_ids = [str(x).strip() for x in obj.get("evidence_ids", []) if str(x).strip()]
    if verdict == "support":
        supporting_evidence_ids = evidence_ids
        contradicting_evidence_ids: list[str] = []
    elif verdict == "contradict":
        supporting_evidence_ids = []
        contradicting_evidence_ids = evidence_ids
    else:
        supporting_evidence_ids = []
        contradicting_evidence_ids = []
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
        "evidence_ids": evidence_ids,
        "supporting_evidence_ids": supporting_evidence_ids,
        "contradicting_evidence_ids": contradicting_evidence_ids,
    }


def sanitize_llm_raw_response(content: str) -> str:
    return _strip_think_blocks(str(content or "")).strip()[:MAX_LLM_RAW_RESPONSE_CHARS]


def build_event_claim(row: dict) -> str:
    claim = normalize_text(row.get("event_claim") or row.get("claim_text") or row.get("claim"))
    if claim:
        return claim
    target_name = normalize_text(row.get("target_name", ""))
    event_name = normalize_text(row.get("event_name", "") or row.get("schema_id", ""))
    if target_name and event_name:
        return f"关于{target_name}的{event_name}相关主张"
    if target_name:
        return f"关于{target_name}的事件主张"
    if event_name:
        return f"关于{event_name}的事件主张"
    return "待验证事件主张"


def build_event_time_value(row: dict) -> str:
    earliest = normalize_text(row.get("earliest_publish_time") or row.get("event_time") or row.get("publish_time"))
    latest = normalize_text(row.get("latest_publish_time"))
    if earliest and latest and earliest != latest:
        return f"{earliest} ~ {latest}"
    return earliest or latest


def normalize_llm_evidence_items(evidence_items: Any) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(evidence_items, list):
        return normalized
    for idx, item in enumerate(evidence_items, 1):
        if not isinstance(item, dict):
            continue
        evidence_id = normalize_text(item.get("evidence_id") or f"E{idx}")
        normalized.append({**item, "evidence_id": evidence_id})
    return normalized


def build_llm_prompt(row: dict, evidence_items: list[dict]) -> str:
    normalized_items = normalize_llm_evidence_items(evidence_items)
    target_name = normalize_text(row.get("target_name", ""))
    event_name = normalize_text(row.get("event_name", "") or row.get("schema_id", ""))
    event_claim = build_event_claim(row)
    event_time = build_event_time_value(row) or "未提供"
    financial_context = row.get("financial_context")
    financial_prompt_context = None
    if isinstance(financial_context, dict):
        financial_prompt_context = financial_context.get("prompt_summary") or financial_context
    financial_block = financial_consistency.format_financial_prompt_block(financial_prompt_context)
    lines = [
        "你是金融事件证据盲审员。",
        "只根据下面给出的主张和证据判断，不使用模型记忆补充外部事实，不输出思维过程。",
        "只输出一个 JSON 对象，不要输出任何其他文本。",
        "JSON 字段必须为 verdict、confidence、reason、evidence_ids。",
        "verdict 只能是 support、contradict、insufficient。",
        "verdict 定义：support 表示证据明确支持事件核心主张；contradict 表示证据明确否定、撤回、澄清或反转核心主张；insufficient 表示现有证据无法充分支持或反驳，不等于事件为假。",
        "confidence 表示你对自己所选 verdict 的确定程度，不是事件为真的概率；contradict 的高 confidence 表示高度确定证据反驳主张。",
        "confidence 必须是 0 到 1 之间的有限数字。",
        "reason 必须简洁。",
        "evidence_ids 必须引用输入证据编号，且只能使用下方编号。",
        "",
        f"待验证主张：{event_claim}",
        f"主体：{target_name}",
        f"事件类型：{event_name}",
        f"事件时间：{event_time}",
    ]
    if financial_block:
        lines.extend(["", financial_block])
    lines.extend(["", "证据："])
    for idx, item in enumerate(normalized_items, 1):
        evidence_id = normalize_text(item.get("evidence_id") or f"E{idx}")
        source_type = normalize_text(item.get("source_type", ""))
        publish_time = normalize_text(item.get("publish_time", ""))
        source = normalize_text(item.get("source", ""))
        title = normalize_text(item.get("title", ""))
        sentence = normalize_text(item.get("sentence", ""))
        lines.append(
            f"{evidence_id}. 来源类型：{source_type}；发布时间：{publish_time}；来源：{source}；标题：{title}；证据文本：{sentence}"
        )
    lines.extend(
        [
            "",
            "请只输出一个 JSON 对象，格式如下：",
            '{"verdict":"support","confidence":0.85,"reason":"简短理由","evidence_ids":["E1"]}',
        ]
    )
    return "\n".join(lines)


def build_llm_format_correction_prompt(row: dict, evidence_items: list[dict], previous_error: str) -> str:
    base_prompt = build_llm_prompt(row, evidence_items)
    return (
        "上一轮输出格式不合法。请只输出一个符合要求的 JSON 对象，不要输出其他文本，不要输出思维过程。\n"
        f"{base_prompt}"
    )


def build_llm_review_output(
    row: dict,
    model: str,
    raw_response: str,
    parse_result: LLMParseResult,
    canonical: dict[str, Any] | None,
    attempt_count: int,
    format_retry_count: int,
    validation_errors: list[str] | None = None,
    status: str = "ok",
) -> dict:
    output = {
        "cluster_id": row["cluster_id"],
        "model": model,
        "status": status,
        "llm_valid": status == "ok" and canonical is not None,
        "attempt_count": int(attempt_count),
        "attempts": int(attempt_count),
        "format_retry_count": int(format_retry_count),
        "response_format": parse_result.format,
        "raw_response": sanitize_llm_raw_response(raw_response),
        "raw_response_preview": sanitize_llm_raw_response(raw_response),
        "error_type": "",
        "error_message": "",
        "error": "",
        "message": "",
        "verdict": "",
        "confidence": 0.0,
        "reason": "",
        "evidence_ids": [],
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
    }
    if status == "ok" and canonical:
        output.update(canonical)
    else:
        error_type = "format_validation_failed" if validation_errors else "llm_request_failed"
        error_message = "; ".join(validation_errors or [parse_result.error])[:800]
        output.update(
            {
                "error_type": error_type,
                "error_message": error_message,
                "error": f"{error_type}: {error_message}",
                "message": error_message,
            }
        )
    return output


def build_llm_review_queue(scores: pd.DataFrame, evidence: pd.DataFrame, output_path: Path, max_items: int = 260) -> pd.DataFrame:
    merged = scores.merge(evidence, on=["cluster_id", "target_code", "target_name", "schema_id", "event_name"], how="left")
    candidates = pd.concat(
        [
            merged[merged["verification_status"].isin(["needs_manual_review", "reversed_or_contradicted"])],
            merged.assign(_distance=(merged["credibility_score"] - 50).abs()).sort_values("_distance").head(max_items),
            merged.sort_values("credibility_score").head(80),
            merged.sort_values("credibility_score", ascending=False).head(40),
        ],
        ignore_index=True,
    ).drop_duplicates("cluster_id")
    candidates = candidates.head(max_items)
    rows = []
    for row in candidates.to_dict("records"):
        evidence_items = safe_json_loads(row.get("evidence_items", "[]"))
        if isinstance(evidence_items, dict):
            evidence_items = [evidence_items]
        normalized_items = normalize_llm_evidence_items(evidence_items[:8])
        prompt_row = dict(row)
        prompt_row["event_claim"] = build_event_claim(row)
        prompt_row["event_time"] = build_event_time_value(row)
        rows.append(
            {
                "cluster_id": row["cluster_id"],
                "target_name": row.get("target_name", ""),
                "event_name": row.get("event_name", ""),
                "event_claim": prompt_row["event_claim"],
                "event_time": prompt_row["event_time"],
                "credibility_score": row.get("credibility_score", 0),
                "credibility_level": row.get("credibility_level", ""),
                "risk_reasons": row.get("risk_reasons", ""),
                "supporting_reasons": row.get("supporting_reasons", ""),
                "fulltext_evidence_status": row.get("fulltext_evidence_status", ""),
                "evidence_ids": [item["evidence_id"] for item in normalized_items],
                "evidence_items": normalized_items,
                "llm_prompt": build_llm_prompt(prompt_row, normalized_items),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    return pd.DataFrame(rows)


def build_endpoint_url(base_url: str, endpoint: str, endpoint_style: str = "openai_v1") -> str:
    url = base_url.rstrip("/")
    endpoint = endpoint.strip("/")
    if url.endswith("/" + endpoint):
        return url
    if endpoint_style in {"direct", "no_v1"}:
        return f"{url}/{endpoint}"
    if url.endswith("/v1"):
        return f"{url}/{endpoint}"
    return f"{url}/v1/{endpoint}"


def llm_endpoint_requires_api_key(base_url: str) -> bool:
    value = str(base_url or "").lower()
    return not re.match(r"^https?://(127\.0\.0\.1|localhost|0\.0\.0\.0)([:/]|$)", value)


def llm_request_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 Task2CredibilityReview/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_response_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    if isinstance(data.get("choices"), list) and data["choices"]:
        message = data["choices"][0].get("message", {})
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    chunks = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def call_chat_completions_compatible(
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = DEFAULT_LLM_TIMEOUT,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    endpoint_style: str = "openai_v1",
) -> dict:
    url = build_endpoint_url(base_url, "chat/completions", endpoint_style=endpoint_style)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "你只输出严格 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=llm_request_headers(api_key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = extract_response_text(data)
    parsed = safe_json_loads(content)
    output = {"raw_response": content, "usage": data.get("usage", {}) if isinstance(data, dict) else {}}
    if isinstance(parsed, dict):
        output.update(parsed)
    return output


def call_responses_compatible(
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = DEFAULT_LLM_TIMEOUT,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    endpoint_style: str = "openai_v1",
) -> dict:
    url = build_endpoint_url(base_url, "responses", endpoint_style=endpoint_style)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "instructions": "你只输出严格 JSON。",
        "input": prompt,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=llm_request_headers(api_key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = extract_response_text(data)
    parsed = safe_json_loads(content)
    output = {"raw_response": content, "usage": data.get("usage", {}) if isinstance(data, dict) else {}}
    if isinstance(parsed, dict):
        output.update(parsed)
    return output


def llm_exception_payload(cluster_id: str, exc: Exception) -> dict:
    message = str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
            if body:
                message = f"{message} | body: {body[:500]}"
        except Exception:
            pass
    return {"cluster_id": cluster_id, "error": type(exc).__name__, "message": message[:800]}


def is_retryable_llm_exception(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {429, 500, 502, 503, 504}:
            return True
    message = str(exc).lower()
    retry_markers = (
        "temporarily",
        "temporary",
        "overloaded",
        "timeout",
        "timed out",
        "upstream",
        "service unavailable",
        "connection reset",
    )
    return any(marker in message for marker in retry_markers)


def _env_int(name: str, default: int, low: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if low is not None:
        parsed = max(low, parsed)
    return parsed


def _env_float(name: str, default: float, low: float | None = None) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = float(value)
    if low is not None:
        parsed = max(low, parsed)
    return parsed


def llm_exception_message(exc: Exception) -> str:
    payload = llm_exception_payload("", exc)
    return str(payload.get("message") or payload.get("error") or exc)[:800]


def llm_runtime_config() -> dict[str, Any]:
    base_url = os.environ.get("TASK2_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("TASK2_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("TASK2_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    wire_api = (os.environ.get("TASK2_LLM_WIRE_API") or os.environ.get("OPENAI_WIRE_API") or "chat_completions").lower()
    endpoint_style = (os.environ.get("TASK2_LLM_ENDPOINT_STYLE") or os.environ.get("OPENAI_ENDPOINT_STYLE") or "openai_v1").lower()
    return {
        "base_url": base_url,
        "api_key": api_key or "",
        "model": model,
        "wire_api": wire_api,
        "endpoint_style": endpoint_style,
        "max_retries": _env_int("TASK2_LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES, low=0),
        "format_retries": _env_int("TASK2_LLM_FORMAT_RETRIES", DEFAULT_LLM_FORMAT_RETRIES, low=0),
        "retry_backoff_seconds": _env_float("TASK2_LLM_RETRY_BACKOFF_SECONDS", 1.0, low=0.0),
        "max_tokens": _env_int("TASK2_LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS, low=1),
        "temperature": _env_float("TASK2_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE, low=0.0),
        "timeout": _env_float("TASK2_LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, low=1.0),
        "workers": _env_int("TASK2_LLM_WORKERS", DEFAULT_LLM_WORKERS, low=1),
    }


def prepare_llm_review_row(row: dict) -> tuple[dict, list[dict], set[str]]:
    prepared = dict(row)
    evidence_items = safe_json_loads(prepared.get("evidence_items", []))
    if isinstance(evidence_items, dict):
        evidence_items = [evidence_items]
    normalized_items = normalize_llm_evidence_items(evidence_items)
    prepared["evidence_items"] = normalized_items
    prepared["evidence_ids"] = [item["evidence_id"] for item in normalized_items]
    prepared["event_claim"] = prepared.get("event_claim") or build_event_claim(prepared)
    prepared["event_time"] = prepared.get("event_time") or build_event_time_value(prepared)
    return prepared, normalized_items, set(prepared["evidence_ids"])


def call_llm_once(prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    call_kwargs = {
        "base_url": config["base_url"],
        "api_key": config["api_key"],
        "model": config["model"],
        "timeout": config["timeout"],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "endpoint_style": config["endpoint_style"],
    }
    if config["wire_api"] in {"responses", "response"}:
        result = call_responses_compatible(prompt, **call_kwargs)
    else:
        result = call_chat_completions_compatible(prompt, **call_kwargs)
    return result


def unpack_llm_call_result(result: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(result, dict):
        usage = result.get("usage", {})
        return str(result.get("raw_response", "")), usage if isinstance(usage, dict) else {}
    return str(result or ""), {}


def llm_token_usage_fields(usage: dict[str, Any] | None) -> dict[str, int]:
    value = usage if isinstance(usage, dict) else {}

    def as_int(*names: str) -> int:
        for name in names:
            raw = value.get(name)
            if raw is not None:
                try:
                    return int(raw)
                except Exception:
                    return 0
        return 0

    prompt_tokens = as_int("prompt_tokens", "input_tokens")
    completion_tokens = as_int("completion_tokens", "output_tokens")
    total_tokens = as_int("total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def llm_review_event_id(row: dict[str, Any]) -> str:
    return normalize_text(row.get("cluster_id") or row.get("event_id") or row.get("event_cluster_id"))


def resolve_llm_workers(workers: int | None) -> int:
    value = DEFAULT_LLM_WORKERS if workers is None else int(workers)
    if value not in ALLOWED_LLM_WORKERS:
        raise ValueError(f"workers must be one of {sorted(ALLOWED_LLM_WORKERS)}, got {value}")
    return value


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} has invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path} line {line_no} is not a JSON object")
            rows.append(obj)
    return rows


def is_successful_llm_record(obj: dict[str, Any]) -> bool:
    status = normalize_text(obj.get("status") or obj.get("llm_status") or "")
    if status == "failed":
        return False
    if obj.get("llm_valid") is False:
        return False
    verdict = normalize_text(obj.get("verdict") or obj.get("llm_verdict") or "")
    confidence = obj.get("confidence", obj.get("llm_confidence"))
    reason = normalize_text(obj.get("reason") or obj.get("llm_reason") or "")
    evidence_ids = _as_evidence_id_list(obj.get("evidence_ids"))
    if not evidence_ids:
        evidence_ids = _as_evidence_id_list(obj.get("supporting_evidence_ids"))
    if not evidence_ids:
        evidence_ids = _as_evidence_id_list(obj.get("contradicting_evidence_ids"))
    validation = validate_llm_review_object(
        {"verdict": verdict, "confidence": confidence, "reason": reason, "evidence_ids": evidence_ids}
    )
    explicit_failure = bool(obj.get("error_type")) or (
        bool(obj.get("error")) and verdict not in VALID_LLM_VERDICTS
    )
    return validation["valid"] and not explicit_failure


def inspect_existing_llm_output(output_path: Path) -> dict[str, Any]:
    records = read_jsonl_records(output_path)
    successful_ids: set[str] = set()
    failed_ids: set[str] = set()
    duplicate_success_ids: set[str] = set()
    for obj in records:
        event_id = llm_review_event_id(obj)
        if not event_id:
            continue
        if is_successful_llm_record(obj):
            if event_id in successful_ids:
                duplicate_success_ids.add(event_id)
            successful_ids.add(event_id)
        else:
            failed_ids.add(event_id)
    return {
        "records": records,
        "successful_ids": successful_ids,
        "failed_ids": failed_ids,
        "duplicate_success_ids": duplicate_success_ids,
    }


def load_llm_queue_rows(queue_path: Path, max_reviews: int = 0) -> list[tuple[int, dict[str, Any]]]:
    rows = read_jsonl_records(queue_path) if queue_path.exists() else []
    indexed_rows = list(enumerate(rows))
    if max_reviews > 0:
        indexed_rows = indexed_rows[:max_reviews]
    return indexed_rows


def ensure_unique_llm_input_ids(indexed_rows: list[tuple[int, dict[str, Any]]]) -> None:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for input_index, row in indexed_rows:
        event_id = llm_review_event_id(row)
        if not event_id:
            raise ValueError(f"missing cluster_id/event_id at input_index={input_index}")
        if event_id in seen:
            duplicates.append(event_id)
        seen[event_id] = input_index
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"duplicate event IDs in LLM queue: {sample}")


def write_llm_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    queue_path: Path,
    progress: dict[str, Any],
    successful_ids: set[str],
    failed_ids: set[str],
    skipped_ids: set[str],
    completed_ids: set[str],
    workers: int,
    resume: bool,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "queue_path": str(queue_path),
        "output_path": str(output_path),
        "workers": workers,
        "resume": resume,
        "progress": progress,
        "completed_ids": sorted(completed_ids),
        "successful_ids": sorted(successful_ids),
        "failed_ids": sorted(failed_ids),
        "skipped_ids": sorted(skipped_ids),
        "resume_command": (
            f"python scripts/run_task2_llm_review.py --queue {queue_path} "
            f"--output {output_path} --resume --workers {workers}"
        ),
    }
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(checkpoint_path.parent),
        prefix=checkpoint_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, checkpoint_path)


def append_llm_result(out, payload: dict[str, Any]) -> None:
    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
    out.flush()


def build_unhandled_llm_failure(row: dict[str, Any], input_index: int, model: str, exc: BaseException, started: float) -> dict:
    prepared = dict(row)
    event_id = llm_review_event_id(prepared)
    message = f"{type(exc).__name__}: {str(exc)}"[:800]
    payload = build_llm_review_output(
        prepared,
        model=model,
        raw_response="",
        parse_result=LLMParseResult(None, "unexpected_error", message),
        canonical=None,
        attempt_count=0,
        format_retry_count=0,
        validation_errors=None,
        status="failed",
    )
    latency = round(time.perf_counter() - started, 3)
    payload.update(
        {
            "cluster_id": event_id,
            "event_id": event_id,
            "input_index": int(input_index),
            "latency_seconds": latency,
            "elapsed_seconds": latency,
            "error_type": "unexpected_error",
            "error_message": message,
            "error": f"unexpected_error: {message}",
            "message": message,
            **llm_token_usage_fields({}),
        }
    )
    return payload


def review_llm_queue_row(row: dict[str, Any], input_index: int, config: dict[str, Any]) -> dict:
    row_started = time.perf_counter()
    model = config["model"]
    max_retries = config["max_retries"]
    format_retries = config["format_retries"]
    retry_backoff_seconds = config["retry_backoff_seconds"]
    prepared, evidence_items, evidence_ids = prepare_llm_review_row(row)
    prompt = build_llm_prompt(prepared, evidence_items)
    payload: dict[str, Any] | None = None
    attempt_count = 0
    format_retry_count = 0
    last_raw_response = ""
    last_usage: dict[str, Any] = {}
    last_parse_result = LLMParseResult(None, "not_called", "LLM was not called")
    last_validation_errors: list[str] = []
    request_failed = False

    for format_attempt in range(format_retries + 1):
        if format_attempt > 0:
            prompt = build_llm_format_correction_prompt(
                prepared,
                evidence_items,
                "; ".join(last_validation_errors) or last_parse_result.error,
            )
        request_failed = False
        for http_attempt in range(max_retries + 1):
            try:
                attempt_count += 1
                call_result = call_llm_once(prompt, config)
                last_raw_response, last_usage = unpack_llm_call_result(call_result)
                request_failed = False
                last_parse_result = parse_llm_review_content(last_raw_response)
                if last_parse_result.parsed is None:
                    last_validation_errors = [last_parse_result.error or "no JSON object found"]
                else:
                    validation = validate_llm_review_object(last_parse_result.parsed, evidence_ids)
                    last_validation_errors = validation["errors"]
                if last_parse_result.parsed is not None and not last_validation_errors:
                    canonical = canonicalize_llm_review_object(last_parse_result.parsed)
                    payload = build_llm_review_output(
                        prepared,
                        model=model,
                        raw_response=last_raw_response,
                        parse_result=last_parse_result,
                        canonical=canonical,
                        attempt_count=attempt_count,
                        format_retry_count=format_retry_count,
                        status="ok",
                    )
                break
            except Exception as exc:
                request_failed = True
                last_parse_result = LLMParseResult(None, "request_error", llm_exception_message(exc))
                last_validation_errors = []
                if http_attempt >= max_retries or not is_retryable_llm_exception(exc):
                    break
                time.sleep(retry_backoff_seconds * (2 ** http_attempt))

        if payload:
            break
        if request_failed:
            break
        if format_attempt < format_retries:
            format_retry_count += 1
            continue
        break

    if not payload:
        payload = build_llm_review_output(
            prepared,
            model=model,
            raw_response=last_raw_response,
            parse_result=last_parse_result,
            canonical=None,
            attempt_count=attempt_count,
            format_retry_count=format_retry_count,
            validation_errors=None if request_failed else last_validation_errors,
            status="failed",
        )
    latency = round(time.perf_counter() - row_started, 3)
    event_id = llm_review_event_id(prepared)
    payload.update(
        {
            "event_id": event_id,
            "input_index": int(input_index),
            "latency_seconds": latency,
            "elapsed_seconds": latency,
            **llm_token_usage_fields(last_usage),
        }
    )
    return payload


def run_llm_review(
    queue_path: Path,
    output_path: Path,
    max_reviews: int = 0,
    sleep_seconds: float = 0.2,
    workers: int | None = None,
    resume: bool = False,
    checkpoint_every: int = DEFAULT_LLM_CHECKPOINT_EVERY,
    retry_failed: bool = True,
) -> dict:
    config = llm_runtime_config()
    base_url = config["base_url"]
    api_key = config["api_key"]
    model = config["model"]
    wire_api = config["wire_api"]
    endpoint_style = config["endpoint_style"]
    max_retries = config["max_retries"]
    format_retries = config["format_retries"]
    retry_backoff_seconds = config["retry_backoff_seconds"]
    max_tokens = config["max_tokens"]
    temperature = config["temperature"]
    timeout = config["timeout"]
    workers = resolve_llm_workers(workers if workers is not None else config.get("workers", DEFAULT_LLM_WORKERS))
    checkpoint_every = max(0, int(checkpoint_every))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    started = time.perf_counter()

    existing_state = {"records": [], "successful_ids": set(), "failed_ids": set(), "duplicate_success_ids": set()}
    if output_path.exists():
        if not resume:
            raise FileExistsError(f"{output_path} already exists; pass resume=True to continue without overwriting")
        if output_path.stat().st_size > 0:
            existing_state = inspect_existing_llm_output(output_path)
            if existing_state["duplicate_success_ids"]:
                sample = ", ".join(sorted(existing_state["duplicate_success_ids"])[:10])
                raise ValueError(f"{output_path} already contains duplicate successful records: {sample}")

    indexed_rows = load_llm_queue_rows(queue_path, max_reviews=max_reviews)
    ensure_unique_llm_input_ids(indexed_rows)
    successful_ids: set[str] = set(existing_state["successful_ids"])
    failed_ids: set[str] = set(existing_state["failed_ids"])
    skipped_ids: set[str] = set()
    completed_ids: set[str] = set(successful_ids)

    pending_rows: list[tuple[int, dict[str, Any]]] = []
    for input_index, row in indexed_rows:
        event_id = llm_review_event_id(row)
        if event_id in successful_ids:
            skipped_ids.add(event_id)
            continue
        if not retry_failed and event_id in failed_ids:
            skipped_ids.add(event_id)
            completed_ids.add(event_id)
            continue
        pending_rows.append((input_index, row))

    total = len(indexed_rows)
    processed = 0
    succeeded = 0
    failures = 0
    retried = 0
    interrupted = False

    def progress_payload(status: str) -> dict[str, Any]:
        elapsed = time.perf_counter() - started
        throughput = processed / elapsed if elapsed > 0 else 0.0
        return {
            "status": status,
            "total": total,
            "input_count": total,
            "pending_count": len(pending_rows),
            "completed": len(skipped_ids) + processed,
            "processed": processed,
            "succeeded": succeeded,
            "failed": failures,
            "skipped": len(skipped_ids),
            "retried": retried,
            "elapsed": round(elapsed, 3),
            "throughput": round(throughput, 6),
            "throughput_items_per_second": round(throughput, 6),
        }

    if not base_url:
        summary = {"status": "skipped_missing_relay_base_url", "reviewed_count": 0}
        if not output_path.exists():
            output_path.write_text("", encoding="utf-8")
        return summary
    if not api_key and llm_endpoint_requires_api_key(base_url):
        summary = {"status": "skipped_missing_api_key", "reviewed_count": 0}
        if not output_path.exists():
            output_path.write_text("", encoding="utf-8")
        return summary

    def record_result(out, payload: dict[str, Any]) -> None:
        nonlocal processed, succeeded, failures, retried
        append_llm_result(out, payload)
        processed += 1
        event_id = llm_review_event_id(payload)
        completed_ids.add(event_id)
        if payload.get("status") == "ok" and payload.get("llm_valid"):
            succeeded += 1
            successful_ids.add(event_id)
            failed_ids.discard(event_id)
        else:
            failures += 1
            failed_ids.add(event_id)
        if int(payload.get("attempt_count") or payload.get("attempts") or 0) > 1 or int(payload.get("format_retry_count") or 0) > 0:
            retried += 1
        if checkpoint_every and processed % checkpoint_every == 0:
            write_llm_checkpoint(
                checkpoint_path,
                output_path,
                queue_path,
                progress_payload("running"),
                successful_ids,
                failed_ids,
                skipped_ids,
                completed_ids,
                workers,
                resume,
            )

    mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as out:
        try:
            if workers == 1:
                for input_index, row in pending_rows:
                    payload = review_llm_queue_row(row, input_index, config)
                    record_result(out, payload)
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
            else:
                pending_iter = iter(pending_rows)
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
                futures: dict[concurrent.futures.Future, tuple[int, dict[str, Any], float]] = {}

                def submit_next() -> bool:
                    try:
                        input_index, row = next(pending_iter)
                    except StopIteration:
                        return False
                    futures[executor.submit(review_llm_queue_row, row, input_index, config)] = (
                        input_index,
                        row,
                        time.perf_counter(),
                    )
                    return True

                try:
                    for _ in range(min(workers, len(pending_rows))):
                        submit_next()
                    while futures:
                        done, _ = concurrent.futures.wait(
                            futures,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            input_index, row, task_started = futures.pop(future)
                            try:
                                payload = future.result()
                            except KeyboardInterrupt:
                                raise
                            except BaseException as exc:
                                payload = build_unhandled_llm_failure(row, input_index, model, exc, task_started)
                            record_result(out, payload)
                            submit_next()
                except KeyboardInterrupt:
                    interrupted = True
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
                else:
                    executor.shutdown(wait=True, cancel_futures=True)
        except KeyboardInterrupt:
            interrupted = True

    final_status = "interrupted" if interrupted else "completed"
    if not interrupted and failures and succeeded:
        final_status = "completed_with_failures"
    elif not interrupted and failures and not succeeded and pending_rows:
        final_status = "failed_all"
    final_progress = progress_payload(final_status)
    write_llm_checkpoint(
        checkpoint_path,
        output_path,
        queue_path,
        final_progress,
        successful_ids,
        failed_ids,
        skipped_ids,
        completed_ids,
        workers,
        resume,
    )
    status = "completed"
    if interrupted:
        status = "interrupted"
    elif failures and succeeded:
        status = "completed_with_failures"
    elif failures and not succeeded and pending_rows:
        status = "failed_all"
    return {
        "status": status,
        "reviewed_count": succeeded,
        "failure_count": failures,
        "total": total,
        "completed": final_progress["completed"],
        "succeeded": succeeded,
        "failed": failures,
        "skipped": len(skipped_ids),
        "retried": retried,
        "throughput": final_progress["throughput"],
        "throughput_items_per_second": final_progress["throughput_items_per_second"],
        "elapsed": final_progress["elapsed"],
        "workers": workers,
        "resume": resume,
        "retry_failed": retry_failed,
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "resume_command": final_progress.get("resume_command", "")
        or f"python scripts/run_task2_llm_review.py --queue {queue_path} --output {output_path} --resume --workers {workers}",
        "model": model,
        "base_url": base_url,
        "wire_api": wire_api,
        "endpoint_style": endpoint_style,
        "max_retries": max_retries,
        "format_retries": format_retries,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
    }


LLM_RESULT_COLUMNS = [
    "cluster_id",
    "llm_input_index",
    "llm_status",
    "llm_valid",
    "llm_verdict",
    "llm_confidence",
    "llm_reason",
    "llm_evidence_ids",
    "llm_supporting_evidence_ids",
    "llm_contradicting_evidence_ids",
    "llm_error",
    "llm_error_type",
    "llm_error_message",
    "llm_attempt_count",
    "llm_format_retry_count",
    "llm_latency_seconds",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
    "llm_raw_response",
]


def _as_evidence_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _legacy_evidence_ids(obj: dict[str, Any], verdict: str) -> list[str]:
    evidence_ids = _as_evidence_id_list(obj.get("evidence_ids"))
    if evidence_ids:
        return evidence_ids
    supporting = _as_evidence_id_list(obj.get("supporting_evidence_ids"))
    contradicting = _as_evidence_id_list(obj.get("contradicting_evidence_ids"))
    if verdict == "support":
        return supporting
    if verdict == "contradict":
        return contradicting
    return supporting + contradicting


def read_llm_results(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=LLM_RESULT_COLUMNS)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            verdict = normalize_text(obj.get("verdict") or obj.get("llm_verdict") or "")
            reason = normalize_text(obj.get("reason") or obj.get("llm_reason") or "")
            confidence_raw = obj.get("confidence", obj.get("llm_confidence"))
            evidence_ids = _legacy_evidence_ids(obj, verdict)
            status = normalize_text(obj.get("status") or "")
            explicit_failure = status == "failed" or bool(obj.get("error_type")) or (
                bool(obj.get("error")) and verdict not in VALID_LLM_VERDICTS
            )
            validation = validate_llm_review_object(
                {
                    "verdict": verdict,
                    "confidence": confidence_raw,
                    "reason": reason,
                    "evidence_ids": evidence_ids,
                }
            )
            valid = not explicit_failure and validation["valid"]
            confidence = float(confidence_raw) if valid else 0.0
            normalized_status = "ok" if valid else "failed"
            validation_message = "; ".join(validation["errors"])
            error_message = normalize_text(obj.get("error_message") or obj.get("message") or obj.get("error") or validation_message)
            error_type = normalize_text(obj.get("error_type") or ("" if valid else "field_validation_failed"))
            rows.append(
                {
                    "cluster_id": obj.get("cluster_id"),
                    "llm_input_index": int(obj.get("input_index", -1)) if obj.get("input_index") is not None else -1,
                    "llm_status": normalized_status,
                    "llm_valid": bool(valid),
                    "llm_verdict": verdict if valid else "",
                    "llm_confidence": confidence,
                    "llm_reason": reason if valid else "",
                    "llm_evidence_ids": evidence_ids if valid else [],
                    "llm_supporting_evidence_ids": _as_evidence_id_list(obj.get("supporting_evidence_ids")) if valid else [],
                    "llm_contradicting_evidence_ids": _as_evidence_id_list(obj.get("contradicting_evidence_ids")) if valid else [],
                    "llm_error": "" if valid else error_message,
                    "llm_error_type": "" if valid else error_type,
                    "llm_error_message": "" if valid else error_message,
                    "llm_attempt_count": int(obj.get("attempt_count") or obj.get("attempts") or 0),
                    "llm_format_retry_count": int(obj.get("format_retry_count") or 0),
                    "llm_latency_seconds": float(obj.get("latency_seconds") or obj.get("elapsed_seconds") or 0.0),
                    "llm_prompt_tokens": int(obj.get("prompt_tokens") or 0),
                    "llm_completion_tokens": int(obj.get("completion_tokens") or 0),
                    "llm_total_tokens": int(obj.get("total_tokens") or 0),
                    "llm_raw_response": obj.get("raw_response", ""),
                }
            )
    return pd.DataFrame(rows)


def fuse_fulltext_scores(scores: pd.DataFrame, evidence: pd.DataFrame, llm_results: pd.DataFrame) -> pd.DataFrame:
    out = scores.merge(evidence, on=["cluster_id", "target_code", "target_name", "schema_id", "event_name"], how="left")
    if llm_results.empty and "cluster_id" not in llm_results.columns:
        llm_results = pd.DataFrame(columns=LLM_RESULT_COLUMNS)
    if "llm_valid" in llm_results.columns:
        llm_results = llm_results[llm_results["llm_valid"] == True].copy()  # noqa: E712
    out = out.merge(llm_results, on="cluster_id", how="left")
    out["fulltext_support_score"] = out["fulltext_support_score"].fillna(0.0)
    out["fulltext_contradiction_score"] = out["fulltext_contradiction_score"].fillna(0.0)
    if "llm_verdict" not in out.columns:
        out["llm_verdict"] = ""
    if "llm_confidence" not in out.columns:
        out["llm_confidence"] = 0.0
    out["llm_confidence"] = out["llm_confidence"].fillna(0.0)
    llm_adjust = np.select(
        [
            out["llm_verdict"] == "support",
            out["llm_verdict"] == "contradict",
            out["llm_verdict"] == "insufficient",
        ],
        [
            12.0 * out["llm_confidence"],
            -35.0 * out["llm_confidence"],
            -5.0 * out["llm_confidence"],
        ],
        default=0.0,
    )
    out["base_credibility_score"] = out["credibility_score"]
    out["fulltext_credibility_score"] = (
        out["base_credibility_score"]
        + 18.0 * out["fulltext_support_score"]
        - 42.0 * out["fulltext_contradiction_score"]
        + llm_adjust
    ).clip(0, 100).round(2)
    contradiction = np.maximum(out["contradiction_score"].fillna(0), out["fulltext_contradiction_score"].fillna(0))
    out["fulltext_credibility_level"] = [
        credibility_level(score, contra)
        for score, contra in zip(out["fulltext_credibility_score"], contradiction)
    ]
    out["fulltext_verification_status"] = np.select(
        [
            contradiction >= 0.75,
            out["llm_verdict"] == "contradict",
            out["fulltext_official_support_count"].fillna(0) > 0,
            out["llm_verdict"] == "support",
            out["fulltext_support_score"] >= 0.55,
            out["fulltext_credibility_score"] < 50,
        ],
        [
            "reversed_or_contradicted",
            "llm_contradicted",
            "official_fulltext_supported",
            "llm_supported",
            "fulltext_supported",
            "needs_manual_review",
        ],
        default="insufficient_evidence",
    )
    return out


def summarize_fulltext(final_scores: pd.DataFrame, llm_summary: dict) -> dict:
    return {
        "cluster_count": int(len(final_scores)),
        "fulltext_level_distribution": final_scores["fulltext_credibility_level"].value_counts().to_dict(),
        "fulltext_verification_status_distribution": final_scores["fulltext_verification_status"].value_counts().to_dict(),
        "fulltext_score_mean": float(final_scores["fulltext_credibility_score"].mean()),
        "fulltext_score_min": float(final_scores["fulltext_credibility_score"].min()),
        "fulltext_score_max": float(final_scores["fulltext_credibility_score"].max()),
        "clusters_with_evidence": int((final_scores["evidence_sentence_count"].fillna(0) > 0).sum()),
        "clusters_with_support": int((final_scores["fulltext_support_count"].fillna(0) > 0).sum()),
        "clusters_with_contradiction": int((final_scores["fulltext_contradict_count"].fillna(0) > 0).sum()),
        "llm_review": llm_summary,
    }


def run_fulltext_pipeline(args: argparse.Namespace) -> dict:
    article_index_path = ROOT / args.article_index
    articles = build_fulltext_article_index(ROOT / args.raw_dir, article_index_path, force=args.rebuild_article_index)
    scores = pd.read_parquet(ROOT / args.base_scores)
    members = pd.read_parquet(ROOT / args.members)
    evidence = build_fulltext_evidence(scores, members, articles, max_extra_articles=args.max_extra_articles)

    out_dir = ROOT / args.output_dir
    report_dir = ROOT / args.report_dir
    processed_dir = ROOT / args.processed_dir
    for directory in (out_dir, report_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    evidence_path = out_dir / "fulltext_evidence.parquet"
    evidence.to_parquet(evidence_path, index=False)
    write_jsonl(out_dir / "fulltext_evidence.jsonl", evidence.to_dict("records"))
    queue_path = processed_dir / "llm_review_queue.jsonl"
    build_llm_review_queue(scores, evidence, queue_path, max_items=args.llm_queue_size)
    llm_results_path = out_dir / "llm_review_results.jsonl"
    llm_summary = {"status": "disabled", "reviewed_count": 0}
    for attr, env_name in (
        ("llm_max_retries", "TASK2_LLM_MAX_RETRIES"),
        ("llm_format_retries", "TASK2_LLM_FORMAT_RETRIES"),
        ("llm_retry_backoff_seconds", "TASK2_LLM_RETRY_BACKOFF_SECONDS"),
        ("llm_max_tokens", "TASK2_LLM_MAX_TOKENS"),
        ("llm_temperature", "TASK2_LLM_TEMPERATURE"),
        ("llm_timeout", "TASK2_LLM_TIMEOUT"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env_name] = str(value)
    if args.run_llm_review:
        llm_summary = run_llm_review(queue_path, llm_results_path, max_reviews=args.max_llm_reviews, sleep_seconds=args.llm_sleep_seconds)
    else:
        llm_results_path.parent.mkdir(parents=True, exist_ok=True)
        if not llm_results_path.exists():
            llm_results_path.write_text("", encoding="utf-8")

    llm_results = read_llm_results(llm_results_path)
    final_scores = fuse_fulltext_scores(scores, evidence, llm_results)
    final_scores.to_parquet(out_dir / "event_credibility_scores_fulltext.parquet", index=False)
    final_scores.to_excel(out_dir / "event_credibility_scores_fulltext.xlsx", index=False)
    write_jsonl(out_dir / "event_credibility_scores_fulltext.jsonl", final_scores.to_dict("records"))

    summary = summarize_fulltext(final_scores, llm_summary)
    (report_dir / "fulltext_credibility_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_fulltext_report(report_dir, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def write_fulltext_report(report_dir: Path, summary: dict) -> None:
    lines = [
        "# 任务二全文证据增强实验报告",
        "",
        "## 实验范围",
        "",
        "本轮在事件簇级可信评分基础上接入原始新闻全文，构建全文证据索引、抽取证据句，并融合全文 support/contradict/insufficient 信号。",
        "",
        "## 运行结果",
        "",
        f"- 事件簇数量：{summary['cluster_count']}",
        f"- 有全文证据句的事件簇：{summary['clusters_with_evidence']}",
        f"- 有正文支撑信号的事件簇：{summary['clusters_with_support']}",
        f"- 有正文矛盾/反转信号的事件簇：{summary['clusters_with_contradiction']}",
        f"- 全文增强平均分：{summary['fulltext_score_mean']:.2f}",
        f"- 全文增强最低分：{summary['fulltext_score_min']:.2f}",
        f"- 全文增强最高分：{summary['fulltext_score_max']:.2f}",
        "",
        "## 全文增强可信等级分布",
        "",
    ]
    for level, count in summary["fulltext_level_distribution"].items():
        lines.append(f"- {level}: {count}")
    lines.extend(["", "## 全文验证状态分布", ""])
    for status, count in summary["fulltext_verification_status_distribution"].items():
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## 在线大模型复核",
            "",
            f"- 状态：{summary['llm_review'].get('status')}",
            f"- 已复核数量：{summary['llm_review'].get('reviewed_count', 0)}",
            f"- 失败数量：{summary['llm_review'].get('failure_count', 0)}",
            f"- 接口格式：{summary['llm_review'].get('wire_api', '')}",
            f"- 中转站：{summary['llm_review'].get('base_url', '')}",
            "",
            "## 主要产物",
            "",
            "- `data/processed/event_credibility/fulltext_article_index.parquet`: 30万篇全文索引",
            "- `outputs/event_credibility/fulltext_evidence.parquet/jsonl`: 事件簇全文证据句",
            "- `data/processed/event_credibility/llm_review_queue.jsonl`: 中转站大模型复核队列",
            "- `outputs/event_credibility/llm_review_results.jsonl`: 大模型复核结果",
            "- `outputs/event_credibility/event_credibility_scores_fulltext.parquet/jsonl/xlsx`: 全文增强最终可信分",
        ]
    )
    (report_dir / "task2_fulltext_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--article_index", default="data/processed/event_credibility/fulltext_article_index.parquet")
    parser.add_argument("--base_scores", default="outputs/event_credibility/event_credibility_scores.parquet")
    parser.add_argument("--members", default="outputs/event_clustering/event_cluster_members.parquet")
    parser.add_argument("--processed_dir", default="data/processed/event_credibility")
    parser.add_argument("--output_dir", default="outputs/event_credibility")
    parser.add_argument("--report_dir", default="reports/event_credibility")
    parser.add_argument("--max_extra_articles", type=int, default=8)
    parser.add_argument("--llm_queue_size", type=int, default=260)
    parser.add_argument("--run_llm_review", action="store_true")
    parser.add_argument("--max_llm_reviews", type=int, default=0)
    parser.add_argument("--llm_sleep_seconds", type=float, default=0.2)
    parser.add_argument("--llm_max_retries", type=int, default=None)
    parser.add_argument("--llm_format_retries", type=int, default=None)
    parser.add_argument("--llm_retry_backoff_seconds", type=float, default=None)
    parser.add_argument("--llm_max_tokens", type=int, default=None)
    parser.add_argument("--llm_temperature", type=float, default=None)
    parser.add_argument("--llm_timeout", type=float, default=None)
    parser.add_argument("--rebuild_article_index", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict:
    return run_fulltext_pipeline(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
