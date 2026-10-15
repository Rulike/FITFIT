"""Input adapter for replaying DEA safety-gate decisions on project artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


HUMAN_ONLY_FIELDS = {
    "human_verdict",
    "human_notes",
    "human_confidence",
    "evidence_relevance",
    "evidence_stance",
    "evidence_independent",
    "evidence_notes",
    "baseline_correct",
    "cached_final_correct",
    "preferred_result",
}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def first_value(record: Mapping[str, Any] | None, fields: Sequence[str]) -> tuple[str, str]:
    if not isinstance(record, Mapping):
        return "", ""
    for field in fields:
        value = norm(record.get(field))
        if value:
            return value, field
    return "", ""


def parse_schema_id(schema_id: Any) -> dict[str, Any]:
    raw = norm(schema_id)
    match = re.fullmatch(r"(company|industry)_(\d+)_(.+)", raw)
    if not raw:
        return {
            "raw_schema_id": "",
            "entity_type": "",
            "numeric_code": "",
            "event_type": "",
            "status": "missing",
        }
    if not match:
        return {
            "raw_schema_id": raw,
            "entity_type": "",
            "numeric_code": "",
            "event_type": "",
            "status": "parse_failed",
        }
    return {
        "raw_schema_id": raw,
        "entity_type": match.group(1),
        "numeric_code": match.group(2),
        "event_type": match.group(3),
        "status": "ok",
    }


def sanitize_claim_text(text: Any, schema_parse: Mapping[str, Any]) -> str:
    value = norm(text)
    raw_schema = norm(schema_parse.get("raw_schema_id"))
    parsed_event = norm(schema_parse.get("event_type"))
    if raw_schema and parsed_event:
        value = value.replace(raw_schema, parsed_event)
    value = re.sub(r"(?i)(?:company|industry)_\d+_", "", value)
    value = re.sub(r"(?i)(?:company|industry)_\d+", "", value)
    return norm(value)


def build_safety_gate_claim(
    sample_row: Mapping[str, Any] | None,
    cluster_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sample = sample_row if isinstance(sample_row, Mapping) else {}
    cluster = cluster_record if isinstance(cluster_record, Mapping) else {}
    field_sources: dict[str, str] = {}

    subject, source = first_value(cluster, ("主体", "subject", "target_name", "company", "entity", "target"))
    if not subject:
        subject, source = first_value(sample, ("target_name", "subject", "company", "entity", "target"))
    if source:
        field_sources["subject"] = source

    raw_schema, schema_source = first_value(sample, ("schema_id", "event_type"))
    schema_parse = parse_schema_id(raw_schema)
    if schema_source:
        field_sources["raw_schema_id"] = f"sample_row.{schema_source}"

    explicit_event_type, event_type_source = first_value(cluster, ("事件类型", "event_type", "event_name"))
    sample_event_name, sample_event_source = first_value(sample, ("event_name", "event_type"))
    if explicit_event_type:
        event_type = explicit_event_type
        field_sources["event_type"] = f"cluster_record.{event_type_source}"
    elif sample_event_name:
        event_type = sample_event_name
        field_sources["event_type"] = f"sample_row.{sample_event_source}"
    elif schema_parse["status"] == "ok":
        event_type = norm(schema_parse["event_type"])
        field_sources["event_type"] = "sample_row.schema_id.parsed_event_type"
    else:
        event_type = ""
        field_sources["event_type"] = "missing"

    event_time, event_time_source = first_value(cluster, ("事件时间", "event_time", "event_date", "time"))
    if not event_time:
        event_time, event_time_source = first_value(sample, ("event_time", "event_date", "earliest_publish_time", "latest_publish_time", "time"))
        if event_time_source:
            field_sources["event_time"] = f"sample_row.{event_time_source}"
    elif event_time_source:
        field_sources["event_time"] = f"cluster_record.{event_time_source}"

    raw_claim_text, claim_source = first_value(cluster, ("待验证主张", "claim_text", "event_claim"))
    if not raw_claim_text:
        raw_claim_text, claim_source = first_value(sample, ("event_claim", "claim_text", "title_samples"))
        if claim_source:
            field_sources["raw_claim_text"] = f"sample_row.{claim_source}"
    elif claim_source:
        field_sources["raw_claim_text"] = f"cluster_record.{claim_source}"

    claim_text = sanitize_claim_text(raw_claim_text, schema_parse)
    if not claim_text and event_type and subject:
        claim_text = f"关于{subject}的{event_type}相关主张"
        field_sources["claim_text"] = "constructed_from_subject_event_type"
    else:
        field_sources["claim_text"] = field_sources.get("raw_claim_text", "missing")

    event_name = explicit_event_type or sample_event_name or event_type
    claim = {
        "subject": subject,
        "target_name": subject,
        "event_type": event_type,
        "event_name": event_name,
        "event_time": event_time,
        "claim_text": claim_text,
        "event_claim": claim_text,
        "schema_id": event_type,
        "raw_schema_id": raw_schema,
        "raw_claim_text": raw_claim_text,
        "schema_parse": schema_parse,
        "field_sources": field_sources,
    }
    if norm(sample.get("target_code")):
        claim["target_code"] = norm(sample.get("target_code"))
        claim["field_sources"]["target_code"] = "sample_row.target_code"
    return claim


def parse_evidence_id_tokens(value: Any) -> list[str]:
    ids: list[str] = []
    if value is None:
        return ids
    if isinstance(value, Mapping):
        for key in ("evidence_id", "id"):
            ids.extend(parse_evidence_id_tokens(value.get(key)))
        return unique_ids(ids)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            ids.extend(parse_evidence_id_tokens(item))
        return unique_ids(ids)
    text = norm(value)
    for number in re.findall(r"(?<![A-Za-z0-9])E\s*[-_]?\s*(\d+)(?![A-Za-z0-9])", text):
        ids.append(f"E{int(number)}")
    return unique_ids(ids)


def unique_ids(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = norm(value)
        if not item:
            continue
        match = re.fullmatch(r"E0*(\d+)", item)
        if match:
            item = f"E{int(match.group(1))}"
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_cited_evidence_ids(
    cached_result: Mapping[str, Any] | None,
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    result = cached_result if isinstance(cached_result, Mapping) else {}
    allowed = {norm(row.get("evidence_id")) for row in evidence_rows if norm(row.get("evidence_id"))}
    ids: list[str] = []
    for field in ("evidence_ids", "cited_evidence_ids", "direct_evidence_ids", "final_evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids"):
        ids.extend(parse_evidence_id_tokens(result.get(field)))
    for field in ("reason", "final_reason", "raw_response", "raw_response_preview"):
        ids.extend(parse_evidence_id_tokens(result.get(field)))
    for nested_key in ("round2", "round1", "final_result"):
        nested = result.get(nested_key)
        if isinstance(nested, Mapping):
            for field in ("evidence_ids", "cited_evidence_ids", "direct_evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids", "reason"):
                ids.extend(parse_evidence_id_tokens(nested.get(field)))
    return [item for item in unique_ids(ids) if item in allowed]


def unmatched_cited_evidence_ids(
    cached_result: Mapping[str, Any] | None,
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    result = cached_result if isinstance(cached_result, Mapping) else {}
    allowed = {norm(row.get("evidence_id")) for row in evidence_rows if norm(row.get("evidence_id"))}
    ids: list[str] = []
    for field in ("evidence_ids", "cited_evidence_ids", "direct_evidence_ids", "final_evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids"):
        ids.extend(parse_evidence_id_tokens(result.get(field)))
    for field in ("reason", "final_reason", "raw_response", "raw_response_preview"):
        ids.extend(parse_evidence_id_tokens(result.get(field)))
    for nested_key in ("round2", "round1", "final_result"):
        nested = result.get(nested_key)
        if isinstance(nested, Mapping):
            for field in ("evidence_ids", "cited_evidence_ids", "direct_evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids", "reason"):
                ids.extend(parse_evidence_id_tokens(nested.get(field)))
    return [item for item in unique_ids(ids) if item not in allowed]


def normalize_bool(value: Any) -> bool:
    text = norm(value).lower()
    return text in {"yes", "true", "1", "y", "是"}


def normalize_optional_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or norm(value) == "":
        return default
    text = norm(value).lower()
    if text in {"yes", "true", "1", "y", "是"}:
        return True
    if text in {"no", "false", "0", "n", "否"}:
        return False
    return default


def evidence_field(record: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = norm(record.get(field))
        if value:
            return value
    return ""


def strip_human_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in record.items() if str(key) not in HUMAN_ONLY_FIELDS}


def normalize_evidence_record(
    record: Mapping[str, Any],
    cached_ids: set[str],
    round1_ids: set[str],
) -> dict[str, Any]:
    raw_record = strip_human_fields(record)
    text = evidence_field(record, "text", "evidence_text", "证据文本", "sentence", "content", "article_title", "title")
    source = evidence_field(record, "source", "来源", "article_source", "source_name", "publisher")
    publish_time = evidence_field(record, "publish_time", "发布时间", "article_publish_time", "event_time", "time", "date")
    evidence_id = evidence_field(record, "evidence_id", "id")
    stage = evidence_field(record, "evidence_stage", "stage")
    retrieval_source = evidence_field(record, "检索来源", "evidence_source", "retrieval_source")
    is_global = normalize_bool(record.get("是否全局索引检索")) or retrieval_source == "global_fulltext_index"
    trace_status = evidence_field(record, "可追溯状态", "trace_status")
    traceable = (trace_status == "ok") if trace_status else bool(evidence_id and source)
    document_id = evidence_field(record, "document_id", "article_id", "trace_id")
    normalized_stage = stage.lower()
    normalized = {
        "evidence_id": evidence_id,
        "document_id": document_id,
        "article_id": document_id,
        "source": source,
        "article_source": source,
        "publish_time": publish_time,
        "article_publish_time": publish_time,
        "text": text,
        "sentence": text,
        "content": text,
        "title": evidence_field(record, "title", "article_title") or text[:80],
        "evidence_stage": stage,
        "is_round2_new": normalized_stage == "round2_new" or normalize_bool(record.get("是否第二轮新增")),
        "is_global_retrieval": is_global,
        "evidence_source": retrieval_source or ("global_fulltext_index" if is_global else evidence_field(record, "evidence_source")),
        "traceable": traceable,
        "cited_by_cached": evidence_id in cached_ids,
        "cited_by_round1": evidence_id in round1_ids,
        "raw_record": raw_record,
    }
    return normalized


def dedupe_evidence(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        key = norm(record.get("evidence_id")) or f"NO_ID_{len(order)}"
        if key not in merged:
            merged[key] = dict(record)
            merged[key]["evidence_stage_members"] = [norm(record.get("evidence_stage"))]
            order.append(key)
            continue
        current = merged[key]
        current["cited_by_cached"] = bool(current.get("cited_by_cached") or record.get("cited_by_cached"))
        current["cited_by_round1"] = bool(current.get("cited_by_round1") or record.get("cited_by_round1"))
        current["is_round2_new"] = bool(current.get("is_round2_new") or record.get("is_round2_new"))
        current["is_global_retrieval"] = bool(current.get("is_global_retrieval") or record.get("is_global_retrieval"))
        current["traceable"] = bool(current.get("traceable") or record.get("traceable"))
        if norm(record.get("evidence_stage")) and norm(record.get("evidence_stage")) not in current["evidence_stage_members"]:
            current["evidence_stage_members"].append(norm(record.get("evidence_stage")))
        for field in ("text", "sentence", "content", "source", "publish_time", "document_id", "article_id"):
            if not norm(current.get(field)) and norm(record.get(field)):
                current[field] = record[field]
    return [merged[key] for key in order]


def build_safety_gate_evidence_records(
    cluster_id: str,
    evidence_rows: Sequence[Mapping[str, Any]],
    cached_result: Mapping[str, Any] | None,
    round1_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    cluster_rows = [
        row for row in evidence_rows
        if not norm(row.get("cluster_id")) or norm(row.get("cluster_id")) == norm(cluster_id)
    ]
    cached = cached_result if isinstance(cached_result, Mapping) else {}
    round1 = round1_result if isinstance(round1_result, Mapping) else {}
    if not cluster_rows:
        cluster_rows = [
            item for item in (cached.get("final_evidence") or [])
            if isinstance(item, Mapping)
        ]
    preliminary = [
        {
            **dict(row),
            "evidence_id": evidence_field(row, "evidence_id", "id"),
        }
        for row in cluster_rows
    ]
    cached_ids = set(normalize_cited_evidence_ids(cached, preliminary))
    if not cached_ids and cached.get("final_evidence"):
        cached_ids = {evidence_field(item, "evidence_id") for item in cached.get("final_evidence") or [] if isinstance(item, Mapping)}
    round1_ids = set(normalize_cited_evidence_ids(round1, preliminary))
    if not round1_ids:
        round1_ids = set(parse_evidence_id_tokens(round1.get("evidence_ids")))
    normalized = [normalize_evidence_record(row, cached_ids, round1_ids) for row in cluster_rows]
    final_evidence_ids = {
        evidence_field(item, "evidence_id")
        for item in cached.get("final_evidence") or []
        if isinstance(item, Mapping)
    }
    for record in normalized:
        record["in_cached_final_evidence"] = norm(record.get("evidence_id")) in final_evidence_ids if final_evidence_ids else True
    return dedupe_evidence(normalized)


def canonical_result_for_gate(
    result: Mapping[str, Any] | None,
    evidence_records: Sequence[Mapping[str, Any]],
    role: str = "cached",
) -> dict[str, Any]:
    source = result if isinstance(result, Mapping) else {}
    cited = normalize_cited_evidence_ids(source, evidence_records)
    if role == "cached":
        return {
            "final_verdict": norm(source.get("final_verdict") or source.get("verdict")),
            "final_confidence": source.get("final_confidence", source.get("confidence")),
            "status": norm(source.get("status")) or "success",
            "llm_valid": normalize_optional_bool(source.get("llm_valid"), default=True),
            "evidence_ids": cited,
            "reason": norm(source.get("final_reason") or source.get("reason")),
            "unmatched_cited_evidence_ids": unmatched_cited_evidence_ids(source, evidence_records),
        }
    return {
        "verdict": norm(source.get("verdict") or source.get("final_verdict")),
        "confidence": source.get("confidence", source.get("final_confidence")),
        "status": norm(source.get("status")) or "ok",
        "llm_valid": normalize_optional_bool(source.get("llm_valid"), default=True),
        "evidence_ids": cited,
        "reason": norm(source.get("reason") or source.get("final_reason")),
        "unmatched_cited_evidence_ids": unmatched_cited_evidence_ids(source, evidence_records),
    }
