"""Evidence isolation utilities for Task2 Hybrid-v3 clean evidence packs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any


GENERIC_TERMS = {
    "关于",
    "相关",
    "主张",
    "公司",
    "股份",
    "有限",
    "集团",
    "控股",
    "业务",
    "行业",
    "变化",
    "调整",
    "发展",
    "影响",
    "情况",
    "事件",
    "市场",
    "投资",
    "经营",
    "公告",
    "新闻",
}


@dataclass(frozen=True)
class EvidenceIsolationPolicy:
    title_near_duplicate_threshold: float = 0.88
    content_near_duplicate_threshold: float = 0.92
    evidence_reprint_threshold: float = 0.90
    max_evidence_items: int = 8
    require_non_future: bool = True
    same_source_limit: int = 1
    version: str = "task2_clean_evidence_v1"


@dataclass
class ClaimIdentity:
    article_file_id: str
    document_id: str
    cluster_id: str
    normalized_title_sha256: str
    normalized_content_sha256: str
    source: str
    publish_time: str
    target_entity: str
    target_code: str
    event_type: str
    title: str
    content: str


@dataclass
class EvidenceItem:
    evidence_id: str
    article_id: str
    document_id: str
    source: str
    source_type: str
    publish_time: str
    title: str
    sentence: str
    content: str
    score: float
    raw: dict[str, Any]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    text = str(value).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。、“”‘’；：:,.!?！？（）()\[\]【】《》<>|｜\\-—_·]", "", text)
    return text


def normalized_sha256(value: Any) -> str:
    text = normalize_text(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def parse_time(value: Any) -> datetime | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def char_shingles(value: Any, n: int = 2, limit: int = 3000) -> set[str]:
    text = normalize_text(value)[:limit]
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(a: Any, b: Any, n: int = 2, limit: int = 3000) -> float:
    left = char_shingles(a, n=n, limit=limit)
    right = char_shingles(b, n=n, limit=limit)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_legacy_evidence_list(value: Any) -> list[dict[str, Any]]:
    text = "" if value is None else str(value)
    if not text.strip():
        return []
    rows: list[dict[str, Any]] = []
    for idx, part in enumerate(p.strip() for p in re.split(r"[；;]\s*", text) if p.strip()):
        fields = [x.strip() for x in part.split("|")]
        if len(fields) >= 4:
            rows.append(
                {
                    "evidence_id": fields[0],
                    "article_id": fields[0],
                    "source": fields[1],
                    "publish_time": fields[2],
                    "title": "|".join(fields[3:]).strip(),
                    "sentence": "",
                    "score": 0.0,
                }
            )
        else:
            rows.append(
                {
                    "evidence_id": f"legacy_E{idx + 1}",
                    "article_id": "",
                    "source": "",
                    "publish_time": "",
                    "title": part,
                    "sentence": "",
                    "score": 0.0,
                }
            )
    return rows


def parse_evidence_items(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(value)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def extract_target_entity(target: Any) -> tuple[str, str]:
    text = "" if target is None else str(target).strip()
    code_match = re.search(r"\b(\d{6})\b", text)
    code = code_match.group(1) if code_match else ""
    entity = text.split("/", 1)[0].strip() if "/" in text else text
    entity = re.sub(r"\s+\d{6}.*$", "", entity).strip()
    return entity, code


def extract_event_type(claim: Any) -> str:
    text = "" if claim is None else str(claim).strip()
    match = re.search(r"的(.+?)相关主张", text)
    if match:
        return match.group(1).strip()
    return text[:40] or "unknown"


def target_aliases(entity: str, code: str) -> list[str]:
    aliases = []
    if entity:
        aliases.append(entity)
        short = entity
        for suffix in ("股份有限公司", "有限责任公司", "有限公司", "集团股份", "控股集团", "股份", "集团", "公司"):
            if short.endswith(suffix) and len(short) > len(suffix) + 1:
                aliases.append(short[: -len(suffix)])
        aliases.append(re.sub(r"(股份有限|有限责任|有限|股份|集团|控股|公司)$", "", entity))
    if code:
        aliases.append(code)
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        value = normalize_text(alias)
        if len(value) >= 2 and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def event_terms(event_type: str, claim: str, title: str, target_entity: str) -> list[str]:
    text = normalize_text(event_type or claim or title)
    target_norm = normalize_text(target_entity)
    if target_norm:
        text = text.replace(target_norm, "")
    for term in GENERIC_TERMS:
        text = text.replace(term, " ")
    raw_terms = re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{2,}", text)
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        if term in GENERIC_TERMS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms[:8]


def build_claim_identity(candidate: dict[str, Any], article_lookup: dict[str, dict[str, Any]] | None = None) -> ClaimIdentity:
    article_lookup = article_lookup or {}
    claim_article_id = str(candidate.get("selected_article_id") or candidate.get("article_file_id") or "").strip()
    article = article_lookup.get(claim_article_id, {})
    title = str(candidate.get("article_title") or article.get("article_title") or "").strip()
    content = str(candidate.get("article_content") or article.get("content") or "").strip()
    source = str(article.get("article_source") or candidate.get("source") or "").strip()
    publish_time = str(candidate.get("publish_time") or article.get("article_publish_time") or "").strip()
    target_entity, target_code = extract_target_entity(candidate.get("target"))
    return ClaimIdentity(
        article_file_id=claim_article_id,
        document_id=str(candidate.get("document_id") or claim_article_id).strip(),
        cluster_id=str(candidate.get("cluster_id") or candidate.get("sample_id") or "").strip(),
        normalized_title_sha256=normalized_sha256(title),
        normalized_content_sha256=normalized_sha256(content),
        source=source,
        publish_time=publish_time,
        target_entity=target_entity,
        target_code=target_code,
        event_type=extract_event_type(candidate.get("event_claim")),
        title=title,
        content=content,
    )


def build_evidence_item(raw: dict[str, Any], article_lookup: dict[str, dict[str, Any]] | None = None) -> EvidenceItem:
    article_lookup = article_lookup or {}
    article_id = str(raw.get("article_id") or raw.get("trace_id") or raw.get("evidence_id") or "").strip()
    article = article_lookup.get(article_id, {})
    evidence_id = str(raw.get("evidence_id") or article_id or "").strip()
    source = str(raw.get("source") or raw.get("article_source") or article.get("article_source") or "").strip()
    publish_time = str(raw.get("publish_time") or raw.get("article_publish_time") or article.get("article_publish_time") or "").strip()
    title = str(raw.get("title") or raw.get("article_title") or article.get("article_title") or "").strip()
    sentence = str(raw.get("sentence") or raw.get("evidence_text") or "").strip()
    content = str(raw.get("content") or article.get("content") or "").strip()
    score = raw.get("sentence_score", raw.get("retrieval_score", raw.get("score", 0.0)))
    try:
        score_float = float(score or 0.0)
    except Exception:
        score_float = 0.0
    return EvidenceItem(
        evidence_id=evidence_id,
        article_id=article_id,
        document_id=str(raw.get("document_id") or article_id).strip(),
        source=source,
        source_type=str(raw.get("source_type") or article.get("source_type") or "").strip(),
        publish_time=publish_time,
        title=title,
        sentence=sentence,
        content=content,
        score=score_float,
        raw=dict(raw),
    )


def self_evidence_reasons(claim: ClaimIdentity, item: EvidenceItem, policy: EvidenceIsolationPolicy | None = None) -> list[str]:
    policy = policy or EvidenceIsolationPolicy()
    reasons: list[str] = []
    if claim.article_file_id and item.article_id and normalize_text(claim.article_file_id) == normalize_text(item.article_id):
        reasons.append("same_article_file_id")
    if claim.document_id and item.document_id and normalize_text(claim.document_id) == normalize_text(item.document_id):
        reasons.append("same_document_id")
    ev_title_sha = normalized_sha256(item.title)
    ev_content_sha = normalized_sha256(item.content or item.sentence)
    if claim.normalized_title_sha256 and ev_title_sha and claim.normalized_title_sha256 == ev_title_sha:
        reasons.append("same_title_hash")
    if claim.normalized_content_sha256 and ev_content_sha and claim.normalized_content_sha256 == ev_content_sha:
        reasons.append("same_content_hash")
    if item.title and claim.title and jaccard_similarity(claim.title, item.title) >= policy.title_near_duplicate_threshold:
        reasons.append("near_duplicate_title")
    if item.content and claim.content and jaccard_similarity(claim.content, item.content) >= policy.content_near_duplicate_threshold:
        reasons.append("near_duplicate_content")
    return sorted(set(reasons))


def target_match(claim: ClaimIdentity, item: EvidenceItem) -> tuple[bool, str]:
    evidence_text = normalize_text(" ".join([item.title, item.sentence, item.content, item.source]))
    aliases = target_aliases(claim.target_entity, claim.target_code)
    hits = [alias for alias in aliases if alias and alias in evidence_text]
    if hits:
        return True, "matched_alias:" + ",".join(hits[:3])
    return False, "no_target_alias_match"


def event_match(claim: ClaimIdentity, candidate: dict[str, Any], item: EvidenceItem) -> tuple[bool, str]:
    evidence_text = normalize_text(" ".join([item.title, item.sentence, item.content]))
    claim_text = str(candidate.get("event_claim") or "")
    terms = event_terms(claim.event_type, claim_text, claim.title, claim.target_entity)
    hits = [term for term in terms if term and term in evidence_text]
    if hits:
        return True, "matched_event_terms:" + ",".join(hits[:4])
    title_sim = jaccard_similarity(claim.title, " ".join([item.title, item.sentence]), n=2, limit=800)
    if title_sim >= 0.28:
        return True, f"title_sentence_similarity:{title_sim:.4f}"
    return False, "core_event_not_matched"


def is_future_evidence(claim: ClaimIdentity, item: EvidenceItem) -> bool:
    event_dt = parse_time(claim.publish_time)
    ev_dt = parse_time(item.publish_time)
    if event_dt is None or ev_dt is None:
        return False
    return ev_dt > event_dt


def financial_context_temporal_status(financial_context: Any, event_time: Any) -> dict[str, Any]:
    context = parse_json_maybe(financial_context)
    if not context:
        return {"available": False, "temporal_valid": False, "status": "not_available", "reason": "no_financial_context"}
    if isinstance(context, list):
        items = [item for item in context if isinstance(item, dict)]
    elif isinstance(context, dict):
        items = [context]
    else:
        items = []
    if not items:
        return {"available": False, "temporal_valid": False, "status": "not_available", "reason": "unsupported_financial_context"}
    event_dt = parse_time(event_time)
    valid_count = 0
    uncertain_count = 0
    for item in items:
        disclosure = item.get("disclosure_date") or item.get("announce_date") or item.get("public_date") or item.get("data_available_date")
        disclosure_dt = parse_time(disclosure)
        if disclosure_dt and event_dt and disclosure_dt <= event_dt:
            valid_count += 1
        elif disclosure_dt and event_dt and disclosure_dt > event_dt:
            continue
        else:
            uncertain_count += 1
    if valid_count:
        return {"available": True, "temporal_valid": True, "status": "available_before_event", "valid_count": valid_count}
    if uncertain_count:
        return {
            "available": True,
            "temporal_valid": False,
            "status": "temporal_availability_uncertain",
            "uncertain_count": uncertain_count,
        }
    return {"available": True, "temporal_valid": False, "status": "future_only_or_invalid"}


def isolate_evidence(
    candidate: dict[str, Any],
    raw_items: list[dict[str, Any]],
    article_lookup: dict[str, dict[str, Any]] | None = None,
    policy: EvidenceIsolationPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or EvidenceIsolationPolicy()
    article_lookup = article_lookup or {}
    claim = build_claim_identity(candidate, article_lookup)
    evidence_items = [build_evidence_item(item, article_lookup) for item in raw_items]
    evidence_items.sort(key=lambda item: (-item.score, item.publish_time, item.article_id, item.evidence_id))

    valid_items: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_story_titles: list[str] = []
    reason_counts: Counter[str] = Counter()

    for item in evidence_items:
        reasons = self_evidence_reasons(claim, item, policy)
        primary_reason = ""
        detail = ""
        if reasons:
            primary_reason = "self_evidence"
            detail = ",".join(reasons)
        elif policy.require_non_future and is_future_evidence(claim, item):
            primary_reason = "future_evidence"
            detail = "evidence_publish_time_after_claim_time"
        else:
            target_ok, target_reason = target_match(claim, item)
            if not target_ok:
                primary_reason = "target_mismatch"
                detail = target_reason
            else:
                event_ok, event_reason = event_match(claim, candidate, item)
                if not event_ok:
                    primary_reason = "event_mismatch"
                    detail = event_reason
                else:
                    source_key = normalize_text(item.source) or "unknown_source"
                    title_norm = normalize_text(item.title)
                    if policy.same_source_limit == 1 and source_key in seen_sources:
                        primary_reason = "same_source_duplicate"
                        detail = f"source={item.source}"
                    elif any(jaccard_similarity(title_norm, old, n=2, limit=500) >= policy.evidence_reprint_threshold for old in seen_story_titles):
                        primary_reason = "duplicate_or_reprint"
                        detail = "near_duplicate_evidence_title"
                    else:
                        seen_sources.add(source_key)
                        if title_norm:
                            seen_story_titles.append(title_norm)
                        valid_items.append(
                            {
                                "evidence_id": f"CE{len(valid_items) + 1}",
                                "source_evidence_id": item.evidence_id,
                                "article_id": item.article_id,
                                "document_id": item.document_id,
                                "source": item.source,
                                "source_type": item.source_type,
                                "publish_time": item.publish_time,
                                "title": item.title,
                                "sentence": item.sentence,
                                "title_sha256": normalized_sha256(item.title),
                                "content_sha256": normalized_sha256(item.content or item.sentence),
                                "score": round(item.score, 6),
                                "target_match_valid": True,
                                "target_match_reason": target_reason,
                                "event_match_valid": True,
                                "event_match_reason": event_reason,
                                "temporal_relation": "before_or_at_claim_time",
                            }
                        )
                        if len(valid_items) >= policy.max_evidence_items:
                            break
        if primary_reason:
            reason_counts[primary_reason] += 1
            excluded.append(
                {
                    "sample_id": str(candidate.get("sample_id") or ""),
                    "claim_article_id": claim.article_file_id,
                    "evidence_article_id": item.article_id,
                    "evidence_id": item.evidence_id,
                    "source": item.source,
                    "publish_time": item.publish_time,
                    "title": item.title,
                    "title_sha256": normalized_sha256(item.title),
                    "content_sha256": normalized_sha256(item.content or item.sentence),
                    "exclusion_reason": primary_reason,
                    "exclusion_detail": detail,
                }
            )

    independent_sources = {normalize_text(item["source"]) for item in valid_items if normalize_text(item.get("source"))}
    if valid_items and independent_sources:
        status = "evidence_ready"
    elif evidence_items:
        status = "insufficient_independent_evidence"
    else:
        status = "no_valid_evidence"
    if not evidence_items:
        status = "no_valid_evidence"

    financial_status = financial_context_temporal_status(candidate.get("financial_context"), claim.publish_time)
    clean_record = {
        "sample_id": str(candidate.get("sample_id") or ""),
        "claim": {
            "article_file_id": claim.article_file_id,
            "document_id": claim.document_id,
            "cluster_id": claim.cluster_id,
            "normalized_title_sha256": claim.normalized_title_sha256,
            "normalized_content_sha256": claim.normalized_content_sha256,
            "source": claim.source,
            "publish_time": claim.publish_time,
            "title": claim.title,
            "target": candidate.get("target"),
            "event_claim": candidate.get("event_claim"),
        },
        "target_entity": claim.target_entity,
        "event_type": claim.event_type,
        "event_time": claim.publish_time,
        "claim_article_id": claim.article_file_id,
        "evidence_items": valid_items,
        "independent_source_count": len(independent_sources),
        "valid_evidence_count": len(valid_items),
        "excluded_evidence_count": len(excluded),
        "exclusion_reasons": dict(reason_counts),
        "self_evidence_count": reason_counts.get("self_evidence", 0),
        "duplicate_evidence_count": reason_counts.get("duplicate_or_reprint", 0) + reason_counts.get("same_source_duplicate", 0),
        "future_evidence_count": reason_counts.get("future_evidence", 0),
        "target_mismatch_count": reason_counts.get("target_mismatch", 0),
        "event_mismatch_count": reason_counts.get("event_mismatch", 0),
        "financial_context": {
            "status": financial_status.get("status"),
            "temporal_valid": financial_status.get("temporal_valid"),
            "available": financial_status.get("available"),
        },
        "evidence_status": status,
        "isolation_policy": {
            "version": policy.version,
            "title_near_duplicate_threshold": policy.title_near_duplicate_threshold,
            "content_near_duplicate_threshold": policy.content_near_duplicate_threshold,
            "evidence_reprint_threshold": policy.evidence_reprint_threshold,
            "same_source_limit": policy.same_source_limit,
            "max_evidence_items": policy.max_evidence_items,
        },
    }
    return {"clean_record": clean_record, "excluded": excluded}
