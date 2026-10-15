from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import os
import random
import re
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from src.event_credibility import fulltext
from src.event_credibility.retrieval_index import CACHE_DIR as RETRIEVAL_CACHE_DIR
from src.event_credibility.retrieval_index import load_task2_retrieval_index
from src.event_credibility.scoring import REVERSAL_PATTERNS, RUMOR_PATTERNS, source_profile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_INITIAL_EVIDENCE_LIMIT = 4
DEFAULT_ADDITIONAL_EVIDENCE_LIMIT = 4
DEFAULT_MAX_ROUNDS = 2
DEFAULT_GRAY_SEED = 2026
DEFAULT_GRAY_SAMPLE_COUNT = 60
CONFLICT_TERMS = tuple(dict.fromkeys(REVERSAL_PATTERNS + ("反转", "撤稿", "澄清", "不属实", "否认")))
FORMAL_OUTPUT_NAMES = {
    "llm_review_results.jsonl",
    "event_credibility_scores_fulltext.jsonl",
    "event_credibility_scores_fulltext.parquet",
    "event_credibility_scores_fulltext.xlsx",
}


@dataclass
class DEAConfig:
    initial_evidence_limit: int = DEFAULT_INITIAL_EVIDENCE_LIMIT
    additional_evidence_limit: int = DEFAULT_ADDITIONAL_EVIDENCE_LIMIT
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    low_confidence_threshold: float = 0.65
    max_rounds: int = DEFAULT_MAX_ROUNDS
    trigger_policy: str = "v2"
    compact_prompt: bool = True


def normalize_text(value: Any) -> str:
    return fulltext.normalize_text(value)


def safe_json_loads(value: Any) -> Any:
    return fulltext.safe_json_loads(value)


def evidence_text(item: dict[str, Any]) -> str:
    return normalize_text(" ".join(str(item.get(field, "")) for field in ("title", "sentence", "content")))


def stable_text_key(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
    return text[:220]


def sequence_similarity(a: Any, b: Any) -> float:
    a_text = stable_text_key(a)
    b_text = stable_text_key(b)
    if not a_text or not b_text:
        return 0.0
    return SequenceMatcher(None, a_text, b_text).ratio()


def sequence_similarity_at_least(a: Any, b: Any, threshold: float) -> bool:
    a_text = stable_text_key(a)
    b_text = stable_text_key(b)
    if not a_text or not b_text:
        return False
    if a_text == b_text:
        return True
    max_possible = (2.0 * min(len(a_text), len(b_text))) / (len(a_text) + len(b_text))
    if max_possible < threshold:
        return False
    matcher = SequenceMatcher(None, a_text, b_text)
    if matcher.real_quick_ratio() < threshold:
        return False
    if matcher.quick_ratio() < threshold:
        return False
    return matcher.ratio() >= threshold


def stable_text_profile(value: Any) -> tuple[str, int, Counter[str]]:
    text = stable_text_key(value)
    return text, len(text), Counter(text)


def profile_similarity_at_least(
    current: tuple[str, int, Counter[str]],
    previous: tuple[str, int, Counter[str]],
    threshold: float,
) -> bool:
    current_text, current_len, current_counts = current
    previous_text, previous_len, previous_counts = previous
    if not current_text or not previous_text:
        return False
    if current_text == previous_text:
        return True
    max_possible = (2.0 * min(current_len, previous_len)) / (current_len + previous_len)
    if max_possible < threshold:
        return False
    overlap = sum(min(count, previous_counts.get(char, 0)) for char, count in current_counts.items())
    if (2.0 * overlap) / (current_len + previous_len) < threshold:
        return False
    matcher = SequenceMatcher(None, current_text, previous_text)
    if matcher.real_quick_ratio() < threshold:
        return False
    if matcher.quick_ratio() < threshold:
        return False
    return matcher.ratio() >= threshold


def tokenize_keywords(value: Any) -> list[str]:
    text = normalize_text(value)
    pieces = re.findall(r"[A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}", text)
    out: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        if piece in seen:
            continue
        if piece in {"关于", "事件", "主张", "相关", "公司", "股份", "有限", "集团"}:
            continue
        seen.add(piece)
        out.append(piece)
    return out


def claim_keywords(row: dict[str, Any], reason: str = "") -> list[str]:
    keys: list[str] = []
    for field in ("target_name", "event_name", "schema_id", "event_claim", "claim_text", "title_samples"):
        keys.extend(tokenize_keywords(row.get(field, "")))
    keys.extend(re.findall(r"\d+(?:\.\d+)?%?|\d+(?:\.\d+)?[亿元万股吨片台]+", normalize_text(row.get("event_claim", ""))))
    if reason:
        keys.extend(tokenize_keywords(reason))
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        value = key.strip("_ -")
        if len(value) < 2 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out[:18]


def target_aliases(row: dict[str, Any]) -> list[str]:
    target = normalize_text(row.get("target_name"))
    code = normalize_text(row.get("target_code"))
    schema_id = normalize_text(row.get("schema_id"))
    aliases: list[str] = []
    if target:
        aliases.append(target)
        short = target
        for suffix in ("股份有限公司", "有限责任公司", "有限公司", "集团股份", "控股集团", "股份", "集团"):
            if short.endswith(suffix) and len(short) > len(suffix) + 1:
                aliases.append(short[: -len(suffix)])
        aliases.append(re.sub(r"(股份有限|有限责任|有限|股份|集团|控股|公司)$", "", target))
    if code:
        aliases.extend(re.findall(r"\b\d{6}\b", code))
    aliases.extend(re.findall(r"\b\d{6}\b", schema_id))
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        value = normalize_text(alias).strip(" -_（）()")
        if len(value) < 2 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out[:5]


def evidence_matches_target(item: dict[str, Any], row: dict[str, Any]) -> bool:
    text = evidence_text(item)
    return any(alias and alias in text for alias in target_aliases(row))


def subject_match_ratio(items: list[dict[str, Any]], row: dict[str, Any] | None) -> float:
    if not row or not items:
        return 1.0
    aliases = target_aliases(row)
    if not aliases:
        return 1.0
    matched = sum(1 for item in items if evidence_matches_target(item, row))
    return matched / max(1, len(items))


def compact_text(value: Any, limit: int) -> str:
    text = normalize_text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def compact_prompt_evidence_items(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in evidence_items:
        compacted.append(
            {
                "evidence_id": normalize_text(item.get("evidence_id")),
                "source_type": normalize_text(item.get("source_type")),
                "publish_time": normalize_text(item.get("publish_time") or item.get("article_publish_time")),
                "source": compact_text(item.get("source") or item.get("article_source"), 32),
                "title": compact_text(item.get("title") or item.get("article_title"), 72),
                "sentence": compact_text(item.get("sentence") or item.get("content"), 220),
            }
        )
    return compacted


def parse_dt(value: Any):
    text = normalize_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            from datetime import datetime

            return datetime.strptime(text, fmt)
        except Exception:
            continue
    m = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", text)
    if not m:
        return None
    try:
        from datetime import datetime

        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def time_proximity_score(item: dict[str, Any], row: dict[str, Any]) -> float:
    event_dt = parse_dt(row.get("event_time") or row.get("earliest_publish_time") or row.get("latest_publish_time"))
    item_dt = parse_dt(item.get("publish_time") or item.get("article_publish_time"))
    if not event_dt or not item_dt:
        return 0.35
    days = abs((item_dt - event_dt).days)
    if days <= 1:
        return 1.0
    if days <= 7:
        return 0.85
    if days <= 30:
        return 0.65
    if days <= 90:
        return 0.35
    return 0.1


def source_authority_score(item: dict[str, Any]) -> float:
    source_type = normalize_text(item.get("source_type"))
    source = normalize_text(item.get("source") or item.get("article_source"))
    if source_type:
        profile = {
            "official": 1.0,
            "regulator": 1.0,
            "exchange": 1.0,
            "mainstream": 0.82,
            "portal": 0.58,
            "general_media": 0.5,
            "self_media": 0.25,
            "unknown": 0.25,
        }
        if source_type in profile:
            return profile[source_type]
    _, reliability = source_profile(source)
    return float(reliability)


def official_boost(item: dict[str, Any]) -> float:
    source_type = normalize_text(item.get("source_type"))
    text = normalize_text(" ".join(str(item.get(field, "")) for field in ("source", "title", "sentence")))
    if source_type == "official":
        return 1.0
    if any(term in text for term in ("公告", "监管", "交易所", "上交所", "深交所", "证监会", "国家药监局")):
        return 0.75
    return 0.0


def relevance_score(item: dict[str, Any], row: dict[str, Any]) -> dict[str, float]:
    text = evidence_text(item)
    event_name = normalize_text(row.get("event_name"))
    keys = claim_keywords(row)
    overlap = sum(1 for key in keys if key and key in text)
    relevance = min(1.0, overlap / max(3, min(8, len(keys) or 3)))
    has_target = 1.0 if evidence_matches_target(item, row) else 0.0
    has_event = 1.0 if event_name and event_name in text else 0.0
    has_conflict = 1.0 if any(term in text for term in CONFLICT_TERMS + RUMOR_PATTERNS) else 0.0
    return {
        "text_relevance": round(relevance, 4),
        "has_target": has_target,
        "has_event_term": has_event,
        "has_conflict_signal": has_conflict,
    }


def score_evidence_item(item: dict[str, Any], row: dict[str, Any], duplicate_penalty: float = 0.0) -> dict[str, Any]:
    rel = relevance_score(item, row)
    authority = source_authority_score(item)
    time_score = time_proximity_score(item, row)
    official = official_boost(item)
    total = (
        0.38 * rel["text_relevance"]
        + 0.18 * authority
        + 0.12 * time_score
        + 0.12 * rel["has_target"]
        + 0.08 * rel["has_event_term"]
        + 0.08 * official
        + 0.04 * rel["has_conflict_signal"]
        - duplicate_penalty
    )
    sub = {
        **rel,
        "source_authority": round(authority, 4),
        "time_proximity": round(time_score, 4),
        "official_boost": round(official, 4),
        "duplicate_penalty": round(duplicate_penalty, 4),
        "total": round(max(0.0, total), 4),
    }
    reason = []
    if rel["has_target"]:
        reason.append("包含主体")
    if rel["has_event_term"]:
        reason.append("包含事件词")
    if official:
        reason.append("正式来源或公告信号")
    if authority >= 0.8:
        reason.append("来源权威性较高")
    if duplicate_penalty:
        reason.append("重复内容降权")
    return {**item, "selection_scores": sub, "selection_reason": "；".join(reason) or "文本相关性排序入选"}


def normalize_evidence_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_items = safe_json_loads(row.get("evidence_items", []))
    if isinstance(evidence_items, dict):
        evidence_items = [evidence_items]
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(evidence_items or [], 1):
        if not isinstance(item, dict):
            continue
        evidence_id = normalize_text(item.get("evidence_id") or f"E{idx}")
        normalized.append(
            {
                **item,
                "evidence_id": evidence_id,
                "trace_id": normalize_text(item.get("trace_id") or item.get("article_id") or evidence_id),
                "evidence_source": normalize_text(item.get("evidence_source") or "cluster_pool"),
            }
        )
    return normalized


def dedupe_evidence(items: list[dict[str, Any]], similarity_threshold: float = 0.92) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    seen_texts: list[str] = []
    seen_text_profiles: list[tuple[str, int, Counter[str]]] = []
    for item in items:
        article_id = normalize_text(item.get("article_id"))
        text = evidence_text(item)
        text_key = stable_text_key(text)
        text_profile = stable_text_profile(text_key)
        reason = ""
        if article_id and article_id in seen_docs:
            reason = "duplicate_document"
        elif text_key and text_key in seen_texts:
            reason = "duplicate_sentence"
        elif any(profile_similarity_at_least(text_profile, old, similarity_threshold) for old in seen_text_profiles):
            reason = "highly_similar_sentence"
        if reason:
            rejected.append({**item, "reject_reason": reason})
            continue
        selected.append(item)
        if article_id:
            seen_docs.add(article_id)
        if text_key:
            seen_texts.append(text_key)
            seen_text_profiles.append(text_profile)
    return selected, rejected


def select_ranked_evidence(row: dict[str, Any], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = normalize_evidence_pool(row)
    scored = [score_evidence_item(item, row) for item in pool]
    scored.sort(key=lambda item: item.get("selection_scores", {}).get("total", 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    while scored and len(selected) < limit:
        best_idx = 0
        if used_sources:
            diverse = [
                (idx, item)
                for idx, item in enumerate(scored[: min(len(scored), 8)])
                if normalize_text(item.get("source") or item.get("article_source")) not in used_sources
            ]
            if diverse:
                best_idx = diverse[0][0]
        item = scored.pop(best_idx)
        deduped, duplicate_rejects = dedupe_evidence(selected + [item])
        if len(deduped) == len(selected):
            rejected.extend(duplicate_rejects)
            continue
        selected = deduped
        source = normalize_text(item.get("source") or item.get("article_source"))
        if source:
            used_sources.add(source)
    rejected.extend({**item, "reject_reason": "not_in_top_limit"} for item in scored)
    return selected[:limit], rejected


def valid_referenced_evidence_ids(result: dict[str, Any], allowed_ids: set[str]) -> list[str]:
    if result.get("status") != "ok" or not result.get("llm_valid"):
        return []
    ids = result.get("evidence_ids") or []
    if not isinstance(ids, list):
        return []
    return [normalize_text(item) for item in ids if normalize_text(item) in allowed_ids]


def should_trigger_dea(
    round1: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    low_confidence_threshold: float = 0.65,
    trigger_policy: str = "v2",
    row: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    allowed = {normalize_text(item.get("evidence_id")) for item in selected_evidence}
    valid_ids = valid_referenced_evidence_ids(round1, allowed)
    sources = {normalize_text(item.get("source") or item.get("article_source")) for item in selected_evidence if normalize_text(item.get("source") or item.get("article_source"))}
    docs = {normalize_text(item.get("article_id")) for item in selected_evidence if normalize_text(item.get("article_id"))}
    confidence = float(round1.get("confidence") or 0.0)
    verdict = round1.get("verdict")
    if trigger_policy == "v1":
        if (
            round1.get("status") == "ok"
            and round1.get("llm_valid")
            and verdict in {"support", "contradict"}
            and confidence >= confidence_threshold
            and valid_ids
        ):
            return False, []
        reasons: list[str] = []
        if round1.get("status") != "ok" or not round1.get("llm_valid"):
            reasons.append("round1_failed")
        if verdict == "insufficient":
            reasons.append("verdict_insufficient")
        if confidence < confidence_threshold:
            reasons.append("low_confidence")
        if not valid_ids:
            reasons.append("no_valid_evidence_ids")
        if int(round1.get("format_retry_count") or 0) > 0 and round1.get("status") == "ok":
            reasons.append("round1_format_retry_used")
        if len(selected_evidence) < 2 or len(sources) < 2 or len(docs) < 2:
            reasons.append("evidence_coverage_or_source_diversity_insufficient")
        return True, list(dict.fromkeys(reasons))

    reasons = []
    subject_ratio = subject_match_ratio(selected_evidence, row)
    subject_mismatch = subject_ratio < 0.5
    repeated_sources = bool(selected_evidence) and len(sources) <= 1 and len(selected_evidence) >= 2
    if round1.get("status") != "ok" or not round1.get("llm_valid"):
        reasons.append("round1_failed")
    if verdict == "insufficient":
        reasons.append("verdict_insufficient")
    if not valid_ids:
        reasons.append("no_valid_evidence_ids")
    if int(round1.get("format_retry_count") or 0) > 0 and round1.get("status") == "ok" and confidence < low_confidence_threshold:
        reasons.append("round1_format_retry_used_with_low_confidence")
    if subject_mismatch:
        reasons.append("subject_mismatch")
    if confidence < low_confidence_threshold and verdict in {"support", "contradict"}:
        reasons.append("low_confidence_below_0_65")
    elif (
        verdict in {"support", "contradict"}
        and low_confidence_threshold <= confidence < confidence_threshold
        and valid_ids
        and repeated_sources
    ):
        reasons.append("conditional_repeated_source_low_margin")
    if not reasons:
        return False, []
    return True, list(dict.fromkeys(reasons))


def build_dea_queries(row: dict[str, Any], reason: str = "") -> list[str]:
    aliases = target_aliases(row)
    target = aliases[0] if aliases else normalize_text(row.get("target_name"))
    event = normalize_text(row.get("event_name"))
    keys = claim_keywords(row, reason=reason)
    noisy_terms = {"需要", "更多", "证据", "显示", "明确", "无法", "充分", "支持", "反驳", "主张", "信息", "披露", "相关"}
    fact_keys = [key for key in keys if key not in noisy_terms and key not in aliases and key != event]
    queries: list[str] = []
    if target:
        queries.append(target)
    if target and event:
        queries.append(f"{target} {event}")
    if target and fact_keys:
        queries.append(" ".join([target] + fact_keys[:3]))
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        value = normalize_text(query)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out[:3]


def next_evidence_id(existing: list[dict[str, Any]]) -> str:
    nums: list[int] = []
    for item in existing:
        match = re.fullmatch(r"E(\d+)", normalize_text(item.get("evidence_id")))
        if match:
            nums.append(int(match.group(1)))
    return f"E{(max(nums) if nums else 0) + 1}"


def assign_new_ids(existing: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    current = list(existing)
    for item in candidates:
        evidence_id = normalize_text(item.get("evidence_id"))
        if not evidence_id or evidence_id in {normalize_text(x.get("evidence_id")) for x in current}:
            evidence_id = next_evidence_id(current)
        new_item = {**item, "evidence_id": evidence_id}
        current.append(new_item)
        result.append(new_item)
    return result


class FulltextArticleIndex:
    def __init__(self, path: Path, members_path: Path | None = None, cache_dir: Path | None = None):
        self.path = path
        self.members_path = members_path
        canonical_source = (ROOT / "data/processed/event_credibility/fulltext_article_index.parquet").resolve()
        if cache_dir is not None:
            self.cache_dir = cache_dir
        elif path.resolve() == canonical_source:
            self.cache_dir = RETRIEVAL_CACHE_DIR
        else:
            self.cache_dir = path.parent / ".dea_retrieval_index_cache"
        self.available = False
        self.error = ""
        self.df: pd.DataFrame | None = None
        self.backend = None
        self.member_article_ids: dict[str, list[str]] = {}

    def load(self) -> None:
        try:
            self.backend = load_task2_retrieval_index(self.path, index_dir=self.cache_dir, auto_build=True)
            self.available = bool(self.backend.available)
            self.error = self.backend.error
        except Exception as exc:
            self.available = False
            self.error = f"{type(exc).__name__}: {exc}"[:800]
            return
        if not self.available:
            return
        if self.members_path and self.members_path.exists():
            try:
                members = pd.read_parquet(self.members_path, columns=["cluster_id", "article_id"]).fillna("")
                grouped: dict[str, list[str]] = defaultdict(list)
                for row in members.to_dict("records"):
                    grouped[normalize_text(row.get("cluster_id"))].append(normalize_text(row.get("article_id")))
                self.member_article_ids = dict(grouped)
            except Exception as exc:
                self.error = f"member index unavailable: {type(exc).__name__}: {exc}"[:800]

    def article_rows(self, article_ids: list[str]) -> pd.DataFrame:
        if not self.available or self.backend is None or not article_ids:
            return pd.DataFrame()
        return self.backend.fetch_articles_by_ids(article_ids)

    def extract_from_articles(self, rows: pd.DataFrame, row: dict[str, Any], source_label: str, limit: int = 12) -> list[dict[str, Any]]:
        if rows.empty:
            return []
        terms = claim_keywords(row)
        candidates: list[dict[str, Any]] = []
        for article in rows.to_dict("records"):
            article_dict = {
                "article_id": article.get("article_id", ""),
                "article_title": article.get("article_title", ""),
                "article_source": article.get("article_source", ""),
                "article_publish_time": article.get("article_publish_time", ""),
                "content": article.get("content", ""),
                "source_type": article.get("source_type", ""),
                "source_reliability": article.get("source_reliability", 0.25),
            }
            for item in fulltext.extract_evidence_sentences(article_dict, terms, max_sentences=2):
                candidates.append(
                    {
                        **item,
                        "evidence_source": source_label,
                        "trace_id": normalize_text(article.get("article_id")),
                        "retrieval_score": float(article.get("_retrieval_score") or 0.0),
                        "retrieval_reason": normalize_text(article.get("_retrieval_reason")),
                        "retrieval_query": normalize_text(article.get("_retrieval_query")),
                    }
                )
        scored = [score_evidence_item(item, row) for item in candidates]
        scored.sort(key=lambda item: item.get("selection_scores", {}).get("total", 0.0), reverse=True)
        return scored[:limit]

    def retrieve_linked_article_evidence(self, row: dict[str, Any], used_ids: set[str], limit: int = 12) -> list[dict[str, Any]]:
        cluster_id = normalize_text(row.get("cluster_id"))
        article_ids = [item for item in self.member_article_ids.get(cluster_id, []) if item and item not in used_ids]
        return self.extract_from_articles(self.article_rows(article_ids), row, "cluster_linked_article", limit=limit)

    def document_relevance_score(self, article: dict[str, Any], row: dict[str, Any], query_terms: list[str]) -> tuple[float, list[str]]:
        text = normalize_text(
            " ".join(str(article.get(field, "")) for field in ("article_title", "article_source", "content"))
        )
        title = normalize_text(article.get("article_title"))
        aliases = target_aliases(row)
        keys = claim_keywords(row)
        matched_aliases = [alias for alias in aliases if alias and alias in text]
        matched_query_terms = [term for term in query_terms if term and term in text]
        matched_claim_terms = [term for term in keys if term and term in text]
        authority = source_authority_score(
            {
                "source_type": article.get("source_type", ""),
                "article_source": article.get("article_source", ""),
                "source": article.get("article_source", ""),
            }
        )
        official = official_boost(
            {
                "source_type": article.get("source_type", ""),
                "source": article.get("article_source", ""),
                "title": article.get("article_title", ""),
                "sentence": text[:300],
            }
        )
        event_dt = parse_dt(row.get("event_time") or row.get("earliest_publish_time") or row.get("latest_publish_time"))
        pub_dt = parse_dt(article.get("article_publish_time"))
        if event_dt and pub_dt:
            days = abs((pub_dt - event_dt).days)
            time_score = 1.0 if days <= 7 else (0.75 if days <= 60 else (0.45 if days <= 365 else 0.12))
        else:
            time_score = 0.45
        score = (
            0.34 * min(1.0, len(matched_aliases))
            + 0.24 * min(1.0, len(matched_query_terms) / 2)
            + 0.16 * min(1.0, len(matched_claim_terms) / 4)
            + 0.12 * authority
            + 0.08 * time_score
            + 0.06 * official
        )
        if any(term and term in title for term in matched_query_terms):
            score += 0.06
        reasons = []
        if matched_aliases:
            reasons.append("主体别名命中")
        if matched_query_terms:
            reasons.append("查询关键词命中")
        if matched_claim_terms:
            reasons.append("主张关键词命中")
        if authority >= 0.8 or official:
            reasons.append("正式或高权威来源")
        if pub_dt is None:
            reasons.append("发布时间缺失但保留为时间降权")
        return round(min(1.0, score), 4), reasons

    def query_global(self, row: dict[str, Any], queries: list[str], used_article_ids: set[str], limit: int = 16) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.available or self.backend is None:
            return [], []
        selected_articles: list[dict[str, Any]] = []
        query_reports: list[dict[str, Any]] = []
        aliases = target_aliases(row)
        used_ids = {normalize_text(item) for item in used_article_ids if normalize_text(item)}
        for query in queries:
            terms = [term for term in tokenize_keywords(query) if term]
            raw_hits, base_report = self.backend.query_candidates(row, query, limit=400)
            if not raw_hits.empty and used_ids:
                raw_hits = raw_hits[~raw_hits["article_id"].astype(str).isin(used_ids)].copy()
            excluded = Counter()
            scored_rows: list[dict[str, Any]] = []
            for article in raw_hits.to_dict("records"):
                text = normalize_text(
                    " ".join(
                        str(article.get(field, ""))
                        for field in ("article_title", "article_source", "content")
                    )
                )
                has_subject = any(alias and alias in text for alias in aliases) if aliases else True
                if not has_subject:
                    excluded["subject_not_matched"] += 1
                    continue
                pub_dt = parse_dt(article.get("article_publish_time"))
                event_dt = parse_dt(row.get("event_time") or row.get("earliest_publish_time") or row.get("latest_publish_time"))
                if event_dt and pub_dt and abs((pub_dt - event_dt).days) > 730:
                    excluded["time_far_penalized"] += 1
                score, reasons = self.document_relevance_score(article, row, terms)
                if score < 0.38:
                    excluded["low_relevance"] += 1
                    continue
                scored_rows.append({**article, "_retrieval_score": score, "_retrieval_reason": "；".join(reasons), "_retrieval_query": query})
            scored_rows.sort(key=lambda item: float(item.get("_retrieval_score") or 0.0), reverse=True)
            seen_docs: set[str] = set()
            seen_texts: list[str] = []
            seen_text_profiles: list[tuple[str, int, Counter[str]]] = []
            deduped_rows: list[dict[str, Any]] = []
            for pos, article in enumerate(scored_rows):
                article_id = normalize_text(article.get("article_id"))
                key = stable_text_key(normalize_text(article.get("article_title")) + " " + normalize_text(article.get("content"))[:500])
                profile = stable_text_profile(key)
                if article_id in seen_docs:
                    excluded["duplicate_document"] += 1
                    continue
                if any(profile_similarity_at_least(profile, old, 0.92) for old in seen_text_profiles):
                    excluded["duplicate_or_reprint"] += 1
                    continue
                seen_docs.add(article_id)
                if key:
                    seen_texts.append(key)
                    seen_text_profiles.append(profile)
                deduped_rows.append(article)
                if len(deduped_rows) >= 80:
                    skipped_tail = max(0, len(scored_rows) - pos - 1)
                    if skipped_tail:
                        excluded["dedupe_tail_not_needed"] += skipped_tail
                    break
            selected_articles.extend(deduped_rows[:80])
            top_df = pd.DataFrame(deduped_rows[:80])
            candidate_count = len(self.extract_from_articles(top_df, row, "global_fulltext_index", limit=limit)) if not top_df.empty else 0
            report = dict(base_report)
            report.update(
                {
                    "raw_document_count": int(len(raw_hits)),
                    "hit_document_count": int(len(raw_hits)),
                    "subject_filter_count": int(len(scored_rows) + excluded.get("low_relevance", 0)),
                    "time_filter_count": int(len(scored_rows) + excluded.get("low_relevance", 0)),
                    "relevance_filter_count": int(len(scored_rows)),
                    "dedupe_count": int(len(deduped_rows)),
                    "candidate_evidence_count": int(candidate_count),
                    "excluded_reason_counts": dict(excluded),
                    "top_document_ids": [normalize_text(item.get("article_id")) for item in deduped_rows[:5]],
                }
            )
            query_reports.append(report)
        if not selected_articles:
            return [], query_reports
        rows = pd.DataFrame(selected_articles).drop_duplicates("article_id").sort_values("_retrieval_score", ascending=False).head(120)
        candidates = self.extract_from_articles(rows, row, "global_fulltext_index", limit=limit)
        score_by_id = {
            normalize_text(article.get("article_id")): (float(article.get("_retrieval_score") or 0.0), normalize_text(article.get("_retrieval_reason")), normalize_text(article.get("_retrieval_query")))
            for article in selected_articles
        }
        for item in candidates:
            article_id = normalize_text(item.get("article_id"))
            score, reason, query = score_by_id.get(article_id, (0.0, "", ""))
            item["retrieval_score"] = score
            item["retrieval_reason"] = reason
            item["retrieval_query"] = query
        return candidates, query_reports


def acquire_additional_evidence(
    row: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
    round1: dict[str, Any],
    article_index: FulltextArticleIndex | None,
    additional_limit: int,
) -> dict[str, Any]:
    used_article_ids = {normalize_text(item.get("article_id")) for item in selected_evidence if normalize_text(item.get("article_id"))}
    used_evidence_ids = {normalize_text(item.get("evidence_id")) for item in selected_evidence}
    queries = build_dea_queries(row, reason=round1.get("reason", ""))
    unused_pool = [item for item in normalize_evidence_pool(row) if normalize_text(item.get("evidence_id")) not in used_evidence_ids]
    pool_scored = [score_evidence_item(item, row) for item in unused_pool]
    pool_scored.sort(key=lambda item: item.get("selection_scores", {}).get("total", 0.0), reverse=True)
    candidates: list[dict[str, Any]] = []
    query_reports = [{"query": query, "hit_document_count": 0, "candidate_evidence_count": 0} for query in queries]
    retrieval_mode = "cluster_pool_only"
    retrieval_started = time.perf_counter()
    retrieval_cache_stats: dict[str, Any] = {}
    if pool_scored:
        candidates.extend(pool_scored)
    if article_index and article_index.available:
        linked = article_index.retrieve_linked_article_evidence(row, used_article_ids, limit=additional_limit * 3)
        candidates.extend(linked)
        global_candidates, query_reports = article_index.query_global(row, queries, used_article_ids, limit=additional_limit * 4)
        candidates.extend(global_candidates)
        retrieval_mode = "global_fulltext_index"
        if getattr(article_index, "backend", None) is not None and hasattr(article_index.backend, "cache_stats"):
            retrieval_cache_stats = article_index.backend.cache_stats()
    elif article_index and article_index.error:
        retrieval_mode = "cluster_pool_only"
    retrieval_latency_seconds = round(time.perf_counter() - retrieval_started, 6)
    deduped, rejected = dedupe_evidence(selected_evidence + candidates)
    new_candidates = [item for item in deduped if normalize_text(item.get("evidence_id")) not in used_evidence_ids]
    scored = [score_evidence_item(item, row) for item in new_candidates]
    scored.sort(key=lambda item: item.get("selection_scores", {}).get("total", 0.0), reverse=True)
    chosen = assign_new_ids(selected_evidence, scored[:additional_limit])
    global_selected = [item for item in chosen if item.get("evidence_source") == "global_fulltext_index"]
    for query in query_reports:
        query["selected_evidence_ids"] = [
            item.get("evidence_id")
            for item in global_selected
            if item.get("evidence_source") == "global_fulltext_index"
            and (not item.get("retrieval_query") or item.get("retrieval_query") == query.get("query"))
        ]
        query["final_selected_count"] = len(query["selected_evidence_ids"])
    source_count = len({normalize_text(item.get("source") or item.get("article_source")) for item in chosen if normalize_text(item.get("source") or item.get("article_source"))})
    return {
        "queries": query_reports[:3],
        "new_evidence": chosen,
        "new_evidence_ids": [item.get("evidence_id") for item in chosen],
        "new_source_count": source_count,
        "retrieval_mode": retrieval_mode,
        "retrieval_latency_seconds": retrieval_latency_seconds,
        "retrieval_cache_stats": retrieval_cache_stats,
        "global_hit_document_count": sum(int(q.get("hit_document_count", 0)) for q in query_reports),
        "global_raw_hit_document_count": sum(int(q.get("raw_document_count", q.get("hit_document_count", 0))) for q in query_reports),
        "global_selected_evidence_count": len(global_selected),
        "unselected_reasons": [{"evidence_id": item.get("evidence_id"), "reason": item.get("reject_reason", "")} for item in rejected[:20]],
        "index_error": article_index.error if article_index else "",
    }


def review_with_evidence(
    row: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    input_index: int,
    llm_config: dict[str, Any],
    compact_prompt: bool = True,
) -> dict[str, Any]:
    prompt_evidence = compact_prompt_evidence_items(evidence_items) if compact_prompt else evidence_items
    review_row = {
        "cluster_id": row.get("cluster_id"),
        "target_name": row.get("target_name"),
        "event_name": row.get("event_name"),
        "event_claim": row.get("event_claim") or fulltext.build_event_claim(row),
        "event_time": row.get("event_time") or fulltext.build_event_time_value(row),
        "evidence_items": prompt_evidence,
    }
    if row.get("financial_context") is not None:
        review_row["financial_context"] = row.get("financial_context")
    result = fulltext.review_llm_queue_row(review_row, input_index, llm_config)
    result["prompt_tokens_estimated"] = estimate_prompt_tokens(review_row, prompt_evidence)
    result["effective_prompt_tokens"] = int(result.get("prompt_tokens") or result.get("prompt_tokens_estimated") or 0)
    result["effective_total_tokens"] = int(result.get("total_tokens") or result.get("effective_prompt_tokens") or 0)
    return result


def estimate_prompt_tokens(row: dict[str, Any], evidence_items: list[dict[str, Any]]) -> int:
    prompt = fulltext.build_llm_prompt(row, evidence_items)
    return max(1, math.ceil(len(prompt) / 2.1))


def summarize_round(result: dict[str, Any], evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evidence_ids": [item.get("evidence_id") for item in evidence_items],
        "verdict": result.get("verdict", ""),
        "confidence": float(result.get("confidence") or 0.0),
        "reason": result.get("reason", ""),
        "latency_seconds": float(result.get("latency_seconds") or result.get("elapsed_seconds") or 0.0),
        "status": result.get("status", ""),
        "llm_valid": bool(result.get("llm_valid")),
        "attempt_count": int(result.get("attempt_count") or 0),
        "format_retry_count": int(result.get("format_retry_count") or 0),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
        "effective_prompt_tokens": int(result.get("effective_prompt_tokens") or 0),
        "effective_total_tokens": int(result.get("effective_total_tokens") or 0),
    }


def run_dea_for_row(
    row: dict[str, Any],
    input_index: int,
    llm_config: dict[str, Any],
    dea_config: DEAConfig,
    article_index: FulltextArticleIndex | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    initial_evidence, rejected_initial = select_ranked_evidence(row, dea_config.initial_evidence_limit)
    round1_result = review_with_evidence(row, initial_evidence, input_index, llm_config, compact_prompt=dea_config.compact_prompt)
    trigger, trigger_reasons = should_trigger_dea(
        round1_result,
        initial_evidence,
        dea_config.confidence_threshold,
        low_confidence_threshold=dea_config.low_confidence_threshold,
        trigger_policy=dea_config.trigger_policy,
        row=row,
    )
    acquisition = {
        "queries": [],
        "new_evidence_ids": [],
        "new_source_count": 0,
        "retrieval_mode": "not_triggered",
        "global_hit_document_count": 0,
        "global_raw_hit_document_count": 0,
        "global_selected_evidence_count": 0,
        "unselected_reasons": [],
    }
    round2_result: dict[str, Any] | None = None
    final_result = round1_result
    final_source = "round1"
    combined_evidence = list(initial_evidence)
    if trigger and dea_config.max_rounds >= 2:
        acquisition = acquire_additional_evidence(
            row,
            initial_evidence,
            round1_result,
            article_index,
            dea_config.additional_evidence_limit,
        )
        combined_evidence = (initial_evidence + acquisition["new_evidence"])[: dea_config.initial_evidence_limit + dea_config.additional_evidence_limit]
        if acquisition["new_evidence"]:
            round2_result = review_with_evidence(row, combined_evidence, input_index, llm_config, compact_prompt=dea_config.compact_prompt)
            if round2_result.get("status") == "ok" and round2_result.get("llm_valid"):
                final_result = round2_result
                final_source = "round2"
            else:
                final_source = "round1_fallback_after_round2_failure"
        else:
            trigger_reasons.append("no_new_evidence_available")
            final_source = "round1_no_new_evidence"
    status = "success"
    if final_result.get("status") != "ok" or not final_result.get("llm_valid"):
        status = "failed"
    elif round2_result and (round2_result.get("status") != "ok" or not round2_result.get("llm_valid")):
        status = "round2_failed"
    output = {
        "cluster_id": row.get("cluster_id"),
        "event_id": row.get("cluster_id"),
        "input_index": input_index,
        "dea_triggered": bool(trigger and dea_config.max_rounds >= 2),
        "trigger_reasons": list(dict.fromkeys(trigger_reasons)),
        "retrieval_mode": acquisition.get("retrieval_mode", "not_triggered"),
        "round1": summarize_round(round1_result, initial_evidence),
        "acquisition": {
            "queries": acquisition.get("queries", []),
            "new_evidence_ids": acquisition.get("new_evidence_ids", []),
            "new_source_count": acquisition.get("new_source_count", 0),
            "global_hit_document_count": acquisition.get("global_hit_document_count", 0),
            "global_raw_hit_document_count": acquisition.get("global_raw_hit_document_count", 0),
            "global_selected_evidence_count": acquisition.get("global_selected_evidence_count", 0),
            "retrieval_latency_seconds": acquisition.get("retrieval_latency_seconds", 0.0),
            "retrieval_cache_stats": acquisition.get("retrieval_cache_stats", {}),
            "unselected_reasons": acquisition.get("unselected_reasons", []),
            "index_error": acquisition.get("index_error", ""),
        },
        "round2": summarize_round(round2_result, combined_evidence) if round2_result else {},
        "final_verdict": final_result.get("verdict", ""),
        "final_confidence": float(final_result.get("confidence") or 0.0),
        "final_reason": final_result.get("reason", ""),
        "final_source": final_source,
        "status": status,
        "error_type": final_result.get("error_type", "") if status != "success" else "",
        "error_message": final_result.get("error_message", "") if status != "success" else "",
        "initial_evidence": initial_evidence,
        "additional_evidence": acquisition.get("new_evidence", []),
        "final_evidence": combined_evidence if final_source.startswith("round2") else initial_evidence,
        "initial_rejected": rejected_initial[:20],
        "latency_seconds": round(time.perf_counter() - started, 3),
        "llm_call_count": 1 + int(round2_result is not None),
        "effective_prompt_tokens": int(summarize_round(round1_result, initial_evidence)["effective_prompt_tokens"])
        + (int(summarize_round(round2_result, combined_evidence)["effective_prompt_tokens"]) if round2_result else 0),
        "effective_total_tokens": int(summarize_round(round1_result, initial_evidence)["effective_total_tokens"])
        + (int(summarize_round(round2_result, combined_evidence)["effective_total_tokens"]) if round2_result else 0),
    }
    if row.get("financial_context") is not None:
        output["financial_context"] = row.get("financial_context")
    if not all(normalize_text(item.get("evidence_id")) and normalize_text(item.get("trace_id") or item.get("article_id")) for item in output["final_evidence"]):
        output["status"] = "failed"
        output["error_type"] = "evidence_traceability_failed"
        output["error_message"] = "one or more final evidence items cannot be traced to article_id/trace_id"
    return output


def run_baseline_for_row(row: dict[str, Any], input_index: int, llm_config: dict[str, Any], evidence_limit: int = 8) -> dict[str, Any]:
    started = time.perf_counter()
    evidence_items = normalize_evidence_pool(row)[:evidence_limit]
    result = review_with_evidence(row, evidence_items, input_index, llm_config)
    return {
        "cluster_id": row.get("cluster_id"),
        "event_id": row.get("cluster_id"),
        "input_index": input_index,
        "status": "success" if result.get("status") == "ok" and result.get("llm_valid") else "failed",
        "verdict": result.get("verdict", ""),
        "confidence": float(result.get("confidence") or 0.0),
        "reason": result.get("reason", ""),
        "evidence_ids": [item.get("evidence_id") for item in evidence_items],
        "evidence": evidence_items,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "llm_latency_seconds": float(result.get("latency_seconds") or result.get("elapsed_seconds") or 0.0),
        "llm_call_count": 1,
        "effective_prompt_tokens": int(result.get("effective_prompt_tokens") or 0),
        "effective_total_tokens": int(result.get("effective_total_tokens") or 0),
        "attempt_count": int(result.get("attempt_count") or 0),
        "format_retry_count": int(result.get("format_retry_count") or 0),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
        "raw_response_preview": result.get("raw_response_preview", ""),
    }


def is_success_record(row: dict[str, Any], mode: str) -> bool:
    if mode == "baseline":
        return row.get("status") == "success" and row.get("verdict") in fulltext.VALID_LLM_VERDICTS
    return row.get("status") in {"success", "round2_failed"} and row.get("final_verdict") in fulltext.VALID_LLM_VERDICTS


def output_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".checkpoint.json")


def inspect_existing_output(path: Path, mode: str) -> dict[str, Any]:
    rows = fulltext.read_jsonl_records(path) if path.exists() and path.stat().st_size else []
    success_ids: set[str] = set()
    duplicate_success_ids: set[str] = set()
    for row in rows:
        event_id = normalize_text(row.get("cluster_id") or row.get("event_id"))
        if not event_id:
            continue
        if is_success_record(row, mode):
            if event_id in success_ids:
                duplicate_success_ids.add(event_id)
            success_ids.add(event_id)
    return {"rows": rows, "success_ids": success_ids, "duplicate_success_ids": duplicate_success_ids}


def write_checkpoint(path: Path, output_path: Path, progress: dict[str, Any], completed_ids: set[str], success_ids: set[str], workers: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_path": str(output_path),
        "progress": progress,
        "completed_ids": sorted(completed_ids),
        "success_ids": sorted(success_ids),
        "workers": workers,
        "resume_command": f"python scripts/run_task2_dea_review.py --output {output_path} --resume --workers {workers}",
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), prefix=path.name + ".", suffix=".tmp", delete=False) as tmp:
        tmp.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def run_dea_batch(
    rows: list[dict[str, Any]],
    output_path: Path,
    llm_config: dict[str, Any],
    dea_config: DEAConfig,
    mode: str = "dea",
    workers: int = 4,
    resume: bool = False,
    checkpoint_every: int = 10,
    article_index: FulltextArticleIndex | None = None,
) -> dict[str, Any]:
    workers = fulltext.resolve_llm_workers(workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not resume:
        raise FileExistsError(f"{output_path} already exists; pass --resume to continue")
    existing = inspect_existing_output(output_path, mode) if resume else {"success_ids": set(), "duplicate_success_ids": set()}
    if existing["duplicate_success_ids"]:
        raise ValueError(f"duplicate successful records exist: {sorted(existing['duplicate_success_ids'])[:10]}")
    success_ids: set[str] = set(existing["success_ids"])
    skipped_ids: set[str] = set(success_ids)
    pending = [(idx, row) for idx, row in enumerate(rows) if normalize_text(row.get("cluster_id")) not in success_ids]
    completed_ids: set[str] = set(success_ids)
    processed = 0
    succeeded = 0
    failed = 0
    safety_stop = ""
    consecutive_http_failures = 0
    started = time.perf_counter()
    checkpoint_path = output_checkpoint_path(output_path)

    def progress(status: str) -> dict[str, Any]:
        elapsed = time.perf_counter() - started
        return {
            "status": status,
            "total": len(rows),
            "completed": len(skipped_ids) + processed,
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": len(skipped_ids),
            "elapsed": round(elapsed, 3),
            "throughput_items_per_second": round(processed / elapsed, 6) if elapsed > 0 else 0.0,
        }

    def work(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, row = item
        if mode == "baseline":
            return run_baseline_for_row(row, idx, llm_config, evidence_limit=8)
        return run_dea_for_row(row, idx, llm_config, dea_config, article_index=article_index)

    def is_http_failure(result: dict[str, Any]) -> bool:
        text = " ".join(str(result.get(field, "")) for field in ("error_type", "error_message"))
        return bool(re.search(r"HTTP|timeout|timed out|connection|request", text, re.I))

    out_mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(out_mode, encoding="utf-8") as out:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures: dict[concurrent.futures.Future, tuple[int, dict[str, Any]]] = {}
        pending_iter = iter(pending)

        def submit_next() -> bool:
            try:
                item = next(pending_iter)
            except StopIteration:
                return False
            futures[executor.submit(work, item)] = item
            return True

        for _ in range(min(workers, len(pending))):
            submit_next()
        try:
            while futures:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    idx, row = futures.pop(future)
                    try:
                        result = future.result()
                    except BaseException as exc:
                        result = {
                            "cluster_id": row.get("cluster_id"),
                            "event_id": row.get("cluster_id"),
                            "input_index": idx,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:800],
                        }
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    processed += 1
                    event_id = normalize_text(result.get("cluster_id") or row.get("cluster_id"))
                    completed_ids.add(event_id)
                    if is_success_record(result, mode):
                        succeeded += 1
                        success_ids.add(event_id)
                        consecutive_http_failures = 0
                    else:
                        failed += 1
                        consecutive_http_failures = consecutive_http_failures + 1 if is_http_failure(result) else 0
                    text = json.dumps(result, ensure_ascii=False)
                    if re.search(r"CUDA out of memory|out of memory|OOM", text, re.I):
                        safety_stop = "CUDA OOM detected"
                    elif consecutive_http_failures >= 3:
                        safety_stop = "three consecutive HTTP failures"
                    elif failed / max(1, len(rows)) >= 0.05:
                        safety_stop = "failure rate reached 5%"
                    if checkpoint_every and processed % checkpoint_every == 0:
                        write_checkpoint(checkpoint_path, output_path, progress("running"), completed_ids, success_ids, workers)
                    if safety_stop:
                        for fut in futures:
                            fut.cancel()
                        break
                    submit_next()
                if safety_stop:
                    break
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    status = "safety_stopped" if safety_stop else ("completed_with_failures" if failed else "completed")
    final_progress = progress(status)
    final_progress["safety_stop_reason"] = safety_stop
    write_checkpoint(checkpoint_path, output_path, final_progress, completed_ids, success_ids, workers)
    return {
        **final_progress,
        "output_path": str(output_path),
        "checkpoint_path": str(checkpoint_path),
        "mode": mode,
        "workers": workers,
    }


def build_review_pool(scores_path: Path, evidence_path: Path) -> list[dict[str, Any]]:
    scores = pd.read_parquet(scores_path)
    evidence = pd.read_parquet(evidence_path)
    merged = scores.merge(evidence, on=["cluster_id", "target_code", "target_name", "schema_id", "event_name"], how="left")
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        evidence_items = safe_json_loads(row.get("evidence_items", []))
        if isinstance(evidence_items, dict):
            evidence_items = [evidence_items]
        item = {
            "cluster_id": row.get("cluster_id"),
            "target_code": row.get("target_code", ""),
            "target_name": row.get("target_name"),
            "event_name": row.get("event_name"),
            "schema_id": row.get("schema_id"),
            "event_claim": fulltext.build_event_claim(row),
            "event_time": fulltext.build_event_time_value(row),
            "earliest_publish_time": row.get("earliest_publish_time", ""),
            "latest_publish_time": row.get("latest_publish_time", ""),
            "title_samples": row.get("title_samples", ""),
            "evidence_items": fulltext.normalize_llm_evidence_items(evidence_items),
            "fulltext_evidence_status": row.get("fulltext_evidence_status", ""),
            "fulltext_support_score": row.get("fulltext_support_score", 0.0),
            "fulltext_contradiction_score": row.get("fulltext_contradiction_score", 0.0),
            "risk_reasons": row.get("risk_reasons", ""),
            "supporting_reasons": row.get("supporting_reasons", ""),
            "credibility_score": row.get("credibility_score", 0.0),
            "credibility_level": row.get("credibility_level", ""),
            "verification_status": row.get("verification_status", ""),
        }
        rows.append(item)
    return rows


def sample_gray_rows(rows: list[dict[str, Any]], sample_count: int = DEFAULT_GRAY_SAMPLE_COUNT, seed: int = DEFAULT_GRAY_SEED) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_id = {normalize_text(row.get("cluster_id")): row for row in rows if normalize_text(row.get("cluster_id"))}

    def is_conflict(row: dict[str, Any]) -> bool:
        text = normalize_text(" ".join(str(row.get(field, "")) for field in ("fulltext_evidence_status", "risk_reasons", "event_name", "evidence_items")))
        return row.get("fulltext_evidence_status") == "contradict" or any(term in text for term in CONFLICT_TERMS)

    support = [row for row in by_id.values() if row.get("fulltext_evidence_status") == "support" and not is_conflict(row)]
    conflict = [row for row in by_id.values() if is_conflict(row)]
    insufficient = [row for row in by_id.values() if row.get("fulltext_evidence_status") == "insufficient"]
    groups = [("support_like", support), ("conflict_like", conflict), ("insufficient_like", insufficient)]
    per_group = sample_count // 3
    selected: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    used: set[str] = set()
    for label, candidates in groups:
        candidates = [row for row in candidates if normalize_text(row.get("cluster_id")) not in used]
        rng.shuffle(candidates)
        for row in candidates[:per_group]:
            cid = normalize_text(row.get("cluster_id"))
            selected.append(row)
            used.add(cid)
            meta.append({"cluster_id": cid, "sample_bucket": label})
    if len(selected) < sample_count:
        remaining = [row for row in by_id.values() if normalize_text(row.get("cluster_id")) not in used]
        rng.shuffle(remaining)
        for row in remaining[: sample_count - len(selected)]:
            cid = normalize_text(row.get("cluster_id"))
            selected.append(row)
            used.add(cid)
            meta.append({"cluster_id": cid, "sample_bucket": "fallback"})
    selected.sort(key=lambda row: normalize_text(row.get("cluster_id")))
    bucket = {item["cluster_id"]: item["sample_bucket"] for item in meta}
    meta = [{"cluster_id": normalize_text(row.get("cluster_id")), "sample_bucket": bucket.get(normalize_text(row.get("cluster_id")), "")} for row in selected]
    return selected[:sample_count], meta


def check_vllm_models(base_url: str, endpoint_style: str, api_key: str = "", timeout: float = 30.0) -> dict[str, Any]:
    try:
        url = fulltext.build_endpoint_url(base_url, "models", endpoint_style=endpoint_style)
        req = urllib.request.Request(url, headers=fulltext.llm_request_headers(api_key), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return {"ok": True, "status": resp.status, "data": data}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:800]}


def guard_output_path(path: Path) -> None:
    resolved = path.resolve()
    formal_dir = (ROOT / "outputs/event_credibility").resolve()
    if resolved.name in FORMAL_OUTPUT_NAMES and resolved.parent == formal_dir:
        raise ValueError(f"refusing to write formal output path: {resolved}")


def load_records(path: Path) -> list[dict[str, Any]]:
    return fulltext.read_jsonl_records(path)


def verdict_distribution(records: list[dict[str, Any]], mode: str) -> dict[str, int]:
    if mode == "baseline":
        values = [row.get("verdict", "") for row in records if is_success_record(row, mode)]
    else:
        values = [row.get("final_verdict", "") for row in records if is_success_record(row, mode)]
    return dict(Counter(values))


def summarize_gray(
    baseline_path: Path,
    dea_path: Path,
    output_json: Path,
    output_md: Path,
    audit_jsonl: Path,
    audit_csv: Path,
    previous_dea_path: Path | None = None,
) -> dict[str, Any]:
    baseline = load_records(baseline_path)
    dea = load_records(dea_path)
    previous_dea = load_records(previous_dea_path) if previous_dea_path and previous_dea_path.exists() else []
    baseline_by_id = {normalize_text(row.get("cluster_id")): row for row in baseline}
    dea_by_id = {normalize_text(row.get("cluster_id")): row for row in dea}
    previous_dea_by_id = {normalize_text(row.get("cluster_id")): row for row in previous_dea}
    ids = sorted(set(baseline_by_id) & set(dea_by_id))
    triggered = [dea_by_id[cid] for cid in ids if dea_by_id[cid].get("dea_triggered")]
    insufficient_solved = []
    changed = []
    confidence_diffs: list[float] = []
    audit_rows: list[dict[str, Any]] = []
    for cid in ids:
        base = baseline_by_id[cid]
        item = dea_by_id[cid]
        base_verdict = base.get("verdict", "")
        dea_verdict = item.get("final_verdict", "")
        if base_verdict != dea_verdict:
            changed.append(cid)
        if base_verdict == "insufficient" and dea_verdict in {"support", "contradict"}:
            insufficient_solved.append(cid)
        confidence_diffs.append(abs(float(base.get("confidence") or 0.0) - float(item.get("final_confidence") or 0.0)))
        new_has_contradict = any(
            any(term in evidence_text(ev) for term in CONFLICT_TERMS)
            for ev in item.get("additional_evidence", [])
        )
        global_selected = any(ev.get("evidence_source") == "global_fulltext_index" for ev in item.get("additional_evidence", []))
        previous_trigger_changed = (
            cid in previous_dea_by_id
            and bool(previous_dea_by_id[cid].get("dea_triggered")) != bool(item.get("dea_triggered"))
        )
        high_conf_low_evidence = float(item.get("final_confidence") or 0.0) >= 0.85 and len(item.get("final_evidence", [])) <= 2
        high_conf_not_triggered = (
            not item.get("dea_triggered")
            and float((item.get("round1") or {}).get("confidence") or 0.0) >= 0.85
            and (item.get("round1") or {}).get("verdict") in {"support", "contradict"}
        )
        include = (
            previous_trigger_changed
            or base_verdict != dea_verdict
            or cid in insufficient_solved
            or new_has_contradict
            or global_selected
            or high_conf_low_evidence
            or high_conf_not_triggered
        )
        if include:
            audit_rows.append(
                {
                    "human_verdict": "",
                    "round1_evidence_sufficient": "",
                    "new_evidence_relevant": "",
                    "dea_final_correct": "",
                    "human_notes": "",
                    "cluster_id": cid,
                    "baseline_verdict": base_verdict,
                    "baseline_confidence": base.get("confidence", 0.0),
                    "baseline_reason": base.get("reason", ""),
                    "baseline_evidence_ids": base.get("evidence_ids", []),
                    "dea_triggered": item.get("dea_triggered", False),
                    "trigger_reasons": item.get("trigger_reasons", []),
                    "dea_final_verdict": dea_verdict,
                    "dea_final_confidence": item.get("final_confidence", 0.0),
                    "dea_final_reason": item.get("final_reason", ""),
                    "round1": item.get("round1", {}),
                    "round2": item.get("round2", {}),
                    "initial_evidence": item.get("initial_evidence", []),
                    "additional_evidence": item.get("additional_evidence", []),
                    "final_evidence": item.get("final_evidence", []),
                    "audit_reason": ",".join(
                        reason
                        for reason, flag in (
                            ("verdict_changed", base_verdict != dea_verdict),
                            ("insufficient_solved", cid in insufficient_solved),
                            ("new_contradict_evidence", new_has_contradict),
                            ("global_fulltext_evidence_selected", global_selected),
                            ("trigger_decision_changed_from_previous_dea", previous_trigger_changed),
                            ("high_confidence_low_evidence", high_conf_low_evidence),
                            ("high_confidence_not_triggered", high_conf_not_triggered),
                        )
                        if flag
                    ),
                }
            )
    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        pos = (len(ordered) - 1) * p / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return round(ordered[lo], 4)
        return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 4)

    baseline_success = [row for row in baseline if is_success_record(row, "baseline")]
    dea_success = [row for row in dea if is_success_record(row, "dea")]
    retrieval_latencies = [
        float(row.get("acquisition", {}).get("retrieval_latency_seconds") or 0.0)
        for row in dea
        if row.get("dea_triggered")
    ]
    previous_comparison: dict[str, Any] = {}
    if previous_dea_by_id:
        previous_ids = sorted(set(previous_dea_by_id) & set(dea_by_id))

        def evidence_identity(item: dict[str, Any]) -> str:
            article_id = normalize_text(item.get("trace_id") or item.get("article_id"))
            text = normalize_text(item.get("sentence") or item.get("content") or item.get("title"))
            return f"{article_id}|{text[:160]}"

        def evidence_key_set(item: dict[str, Any], field: str, source: str = "") -> set[str]:
            values: set[str] = set()
            for ev in item.get(field, []):
                if source and ev.get("evidence_source") != source:
                    continue
                key = evidence_identity(ev)
                if key.strip("|"):
                    values.add(key)
            return values

        def evidence_overlap(field: str, source: str = "") -> float:
            ref_total = 0
            common = 0
            for cid in previous_ids:
                old_keys = evidence_key_set(previous_dea_by_id[cid], field, source=source)
                new_keys = evidence_key_set(dea_by_id[cid], field, source=source)
                ref_total += len(old_keys)
                common += len(old_keys & new_keys)
            return round(common / ref_total, 6) if ref_total else 1.0

        previous_confidence_diffs = [
            abs(float(previous_dea_by_id[cid].get("final_confidence") or 0.0) - float(dea_by_id[cid].get("final_confidence") or 0.0))
            for cid in previous_ids
        ]
        previous_verdict_changes = [
            cid
            for cid in previous_ids
            if previous_dea_by_id[cid].get("final_verdict") != dea_by_id[cid].get("final_verdict")
        ]
        previous_comparison = {
            "sample_count": len(previous_ids),
            "verdict_consistency_rate": round((len(previous_ids) - len(previous_verdict_changes)) / max(1, len(previous_ids)), 6),
            "verdict_changed_count": len(previous_verdict_changes),
            "verdict_changed_ids": previous_verdict_changes,
            "confidence_mean_abs_diff": avg(previous_confidence_diffs),
            "final_evidence_overlap_rate": evidence_overlap("final_evidence"),
            "additional_evidence_overlap_rate": evidence_overlap("additional_evidence"),
            "global_evidence_overlap_rate": evidence_overlap("additional_evidence", source="global_fulltext_index"),
        }

    def failure_texts(rows: list[dict[str, Any]]) -> list[str]:
        texts: list[str] = []
        for row in rows:
            if row.get("status") == "failed":
                texts.append(" ".join(str(row.get(field, "")) for field in ("error_type", "error_message")))
            for key in ("round1", "round2"):
                nested = row.get(key) or {}
                if isinstance(nested, dict) and nested.get("status") == "failed":
                    texts.append(" ".join(str(nested.get(field, "")) for field in ("error_type", "error_message")))
        return texts

    failure_text_pool = failure_texts(baseline + dea)
    payload = {
        "sample_count": len(ids),
        "dea_trigger_count": len(triggered),
        "dea_trigger_rate": round(len(triggered) / max(1, len(ids)), 4),
        "baseline": {
            "success_count": len(baseline_success),
            "failed_count": len(baseline) - len(baseline_success),
            "average_evidence_count": avg([len(row.get("evidence", [])) for row in baseline]),
            "average_prompt_tokens": avg([float(row.get("effective_prompt_tokens") or 0) for row in baseline]),
            "total_tokens": int(sum(int(row.get("effective_total_tokens") or 0) for row in baseline)),
            "average_latency": avg([float(row.get("latency_seconds") or 0) for row in baseline]),
            "total_latency": round(sum(float(row.get("latency_seconds") or 0) for row in baseline), 3),
            "average_llm_call_count": avg([float(row.get("llm_call_count") or 0) for row in baseline]),
            "verdict_distribution": verdict_distribution(baseline, "baseline"),
        },
        "dea": {
            "success_count": len(dea_success),
            "failed_count": len(dea) - len(dea_success),
            "average_evidence_count": avg([len(row.get("final_evidence", [])) for row in dea]),
            "average_prompt_tokens": avg([float(row.get("effective_prompt_tokens") or 0) for row in dea]),
            "total_tokens": int(sum(int(row.get("effective_total_tokens") or 0) for row in dea)),
            "average_latency": avg([float(row.get("latency_seconds") or 0) for row in dea]),
            "total_latency": round(sum(float(row.get("latency_seconds") or 0) for row in dea), 3),
            "average_llm_call_count": avg([float(row.get("llm_call_count") or 0) for row in dea]),
            "verdict_distribution": verdict_distribution(dea, "dea"),
            "retrieval_mode_distribution": dict(Counter(row.get("retrieval_mode", "") for row in dea)),
            "new_independent_source_count": int(sum(int(row.get("acquisition", {}).get("new_source_count") or 0) for row in dea)),
            "global_hit_document_count": int(sum(int(row.get("acquisition", {}).get("global_hit_document_count") or 0) for row in dea)),
            "global_raw_hit_document_count": int(sum(int(row.get("acquisition", {}).get("global_raw_hit_document_count") or row.get("acquisition", {}).get("global_hit_document_count") or 0) for row in dea)),
            "global_selected_evidence_count": int(sum(int(row.get("acquisition", {}).get("global_selected_evidence_count") or 0) for row in dea)),
            "average_retrieval_latency": avg(retrieval_latencies),
            "p95_retrieval_latency": percentile(retrieval_latencies, 95),
            "total_retrieval_latency": round(sum(retrieval_latencies), 4),
        },
        "comparison": {
            "insufficient_solved_count": len(insufficient_solved),
            "insufficient_solved_ids": insufficient_solved,
            "verdict_changed_count": len(changed),
            "verdict_changed_ids": changed,
            "confidence_mean_abs_diff": avg(confidence_diffs),
            "token_delta": int(sum(int(row.get("effective_total_tokens") or 0) for row in dea) - sum(int(row.get("effective_total_tokens") or 0) for row in baseline)),
            "latency_delta": round(sum(float(row.get("latency_seconds") or 0) for row in dea) - sum(float(row.get("latency_seconds") or 0) for row in baseline), 3),
            "llm_call_delta": int(sum(int(row.get("llm_call_count") or 0) for row in dea) - sum(int(row.get("llm_call_count") or 0) for row in baseline)),
        },
        "previous_dea_comparison": previous_comparison,
        "failure_counts": {
            "timeout": sum(1 for text in failure_text_pool if re.search(r"timeout|timed out|超时", text, re.I)),
            "http_5xx": sum(1 for text in failure_text_pool if re.search(r"HTTP.*5\d\d|\b5\d\d\b", text, re.I)),
            "parse_failure": sum(1 for text in failure_text_pool if re.search(r"format_validation_failed|parse|JSON|unparseable", text, re.I)),
            "cuda_oom": sum(1 for text in failure_text_pool if re.search(r"CUDA out of memory|out of memory|OOM", text, re.I)),
        },
        "manual_audit_count": len(audit_rows),
        "manual_audit_jsonl": str(audit_jsonl),
        "manual_audit_csv": str(audit_csv),
        "note": "No gold labels are used; verdict changes and insufficient reductions are evidence-coverage signals, not accuracy improvements.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(output_md, payload)
    write_audit_files(audit_jsonl, audit_csv, audit_rows)
    return payload


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Task 2 DEA Gray Report",
        "",
        "本报告没有人工金标，不把 verdict 变化、insufficient 减少或 confidence 提高称为准确率提升。",
        "",
        f"- 样本数: {payload['sample_count']}",
        f"- DEA 触发: {payload['dea_trigger_count']} ({payload['dea_trigger_rate']:.2%})",
        f"- insufficient 被补证解决: {payload['comparison']['insufficient_solved_count']}",
        f"- verdict 变化: {payload['comparison']['verdict_changed_count']}",
        f"- 新增独立来源数: {payload['dea']['new_independent_source_count']}",
        f"- 全局索引原始命中文档数: {payload['dea'].get('global_raw_hit_document_count', payload['dea']['global_hit_document_count'])}",
        f"- 全局证据最终入选数: {payload['dea'].get('global_selected_evidence_count', 0)}",
        f"- 平均检索耗时: {payload['dea'].get('average_retrieval_latency', 0.0)}",
        f"- p95 检索耗时: {payload['dea'].get('p95_retrieval_latency', 0.0)}",
        f"- 检索模式分布: {json.dumps(payload['dea'].get('retrieval_mode_distribution', {}), ensure_ascii=False)}",
        "",
        "| mode | avg evidence | avg prompt tokens | total tokens | avg latency | total latency | avg calls | verdicts |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in ("baseline", "dea"):
        item = payload[mode]
        verdicts = ", ".join(f"{k}:{v}" for k, v in item["verdict_distribution"].items())
        lines.append(
            f"| {mode} | {item['average_evidence_count']} | {item['average_prompt_tokens']} | {item['total_tokens']} | "
            f"{item['average_latency']} | {item['total_latency']} | {item['average_llm_call_count']} | {verdicts} |"
        )
    lines.extend(
        [
            "",
            f"- timeout: {payload['failure_counts']['timeout']}",
            f"- HTTP 5xx: {payload['failure_counts']['http_5xx']}",
            f"- parse failure: {payload['failure_counts']['parse_failure']}",
            f"- CUDA OOM: {payload['failure_counts']['cuda_oom']}",
            f"- manual audit rows: {payload['manual_audit_count']}",
        ]
    )
    previous = payload.get("previous_dea_comparison") or {}
    if previous:
        lines.extend(
            [
                "",
                "## Previous DEA Comparison",
                "",
                f"- verdict 一致率: {previous.get('verdict_consistency_rate', 0.0)}",
                f"- confidence 平均绝对差: {previous.get('confidence_mean_abs_diff', 0.0)}",
                f"- final evidence 重合率: {previous.get('final_evidence_overlap_rate', 0.0)}",
                f"- additional evidence 重合率: {previous.get('additional_evidence_overlap_rate', 0.0)}",
                f"- global evidence 重合率: {previous.get('global_evidence_overlap_rate', 0.0)}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit_files(jsonl_path: Path, csv_path: Path, rows: list[dict[str, Any]]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    fields = [
        "cluster_id",
        "audit_reason",
        "baseline_verdict",
        "baseline_confidence",
        "baseline_reason",
        "dea_triggered",
        "dea_final_verdict",
        "dea_final_confidence",
        "dea_final_reason",
        "trigger_reasons",
        "baseline_evidence_ids",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field), ensure_ascii=False) if isinstance(row.get(field), (list, dict)) else row.get(field, "") for field in fields})
