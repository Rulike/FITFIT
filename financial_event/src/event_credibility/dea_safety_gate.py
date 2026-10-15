"""Deterministic safety gate for cached DEA verdicts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "task2_dea_safety_gate_v1.yaml"


def load_dea_safety_gate_config(config: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Load a gate configuration mapping or the versioned default YAML file."""
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config) if config is not None else DEFAULT_CONFIG_PATH
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load DEA safety-gate YAML configuration") from exc
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"DEA safety-gate config must be a mapping: {path}")
    return dict(loaded)


def apply_dea_safety_gate(
    claim: Mapping[str, Any] | str | None,
    baseline_result: Mapping[str, Any] | None,
    round1_result: Mapping[str, Any] | None,
    cached_result: Mapping[str, Any] | None,
    evidence_records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    config: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    """Return a deterministic gated verdict without calling retrieval or any model."""
    cfg = load_dea_safety_gate_config(config)
    valid_verdicts = set(_config_list(cfg, "valid_verdicts"))
    if not valid_verdicts:
        raise ValueError("DEA safety-gate config requires non-empty valid_verdicts")

    baseline = _result_view(baseline_result, "baseline", cfg, valid_verdicts)
    round1 = _result_view(round1_result, "round1", cfg, valid_verdicts)
    cached = _result_view(cached_result, "cached", cfg, valid_verdicts)
    fallback_verdict, fallback_source = _fallback_verdict(round1, baseline, valid_verdicts)

    evidence_items = _evidence_items(evidence_records)
    evidence_by_id = {
        evidence_id: item
        for item in evidence_items
        if (evidence_id := _evidence_id(item))
    }
    claim_times = _record_dates(claim, _config_list(cfg, "claim_time_fields"))
    aliases = _claim_aliases(claim, cfg)
    event_terms = _claim_event_terms(claim, cfg)

    subject_items = [item for item in evidence_items if _subject_matches(item, aliases, cfg)]
    time_matched_items = [
        item for item in subject_items if _time_matches(claim_times, _record_dates(item, _config_list(cfg, "evidence_time_fields")), cfg)
    ]
    direct_items = [item for item in time_matched_items if _event_matches(item, event_terms, cfg)]
    direct_evidence_ids = _unique_ids(direct_items)

    subject_match = bool(subject_items)
    time_match = bool(time_matched_items)
    evidence_can_supply_time = bool(subject_items and any(_record_dates(item, _config_list(cfg, "evidence_time_fields")) for item in subject_items))
    claim_unclear, unclear_reasons = _claim_unclear(claim, evidence_can_supply_time, cfg)

    legal_cached_ids = [evidence_id for evidence_id in cached["evidence_ids"] if evidence_id in evidence_by_id]
    cached_direct_ids = [evidence_id for evidence_id in legal_cached_ids if evidence_id in set(direct_evidence_ids)]
    referenced_items = [evidence_by_id[evidence_id] for evidence_id in legal_cached_ids]
    evidence_scope = referenced_items or evidence_items
    only_investor_questions = _only_investor_questions(evidence_scope, cfg)
    only_market_signals = _only_market_signals(evidence_scope, cfg)

    contradiction_items = [
        item
        for item in time_matched_items
        if _is_official(item, cfg) and _has_explicit_contradiction(item, cfg) and _event_matches(item, event_terms, cfg)
    ]
    contradiction_ids = _unique_ids(contradiction_items)
    cached_contradiction_ids = [evidence_id for evidence_id in legal_cached_ids if evidence_id in set(contradiction_ids)]
    official_contradiction = bool(contradiction_items)

    traceable_source = any(_source_value(item, cfg) for item in referenced_items or direct_items)
    support_reasons = _support_block_reasons(
        cached=cached,
        legal_cached_ids=legal_cached_ids,
        cached_direct_ids=cached_direct_ids,
        subject_match=subject_match,
        time_match=time_match,
        only_investor_questions=only_investor_questions,
        only_market_signals=only_market_signals,
        traceable_source=traceable_source,
        claim_unclear=claim_unclear,
        cfg=cfg,
    )
    contradiction_reasons = _contradiction_block_reasons(
        cached=cached,
        official_contradiction=official_contradiction,
        cached_contradiction_ids=cached_contradiction_ids,
    )
    support_eligible = not support_reasons
    contradiction_eligible = not contradiction_reasons

    gate_reasons: list[str] = []
    original_cached_verdict = cached["verdict"]
    gated_verdict = original_cached_verdict if cached["usable"] else fallback_verdict
    if not cached["usable"]:
        gate_reasons.append("cached_result_invalid")

    if claim_unclear:
        gated_verdict = "unclear"
        gate_reasons.extend(["claim_unclear", *unclear_reasons])
    elif only_investor_questions:
        gated_verdict = fallback_verdict
        gate_reasons.append("only_investor_questions")
    elif only_market_signals:
        gated_verdict = fallback_verdict
        gate_reasons.append("only_market_signals")
    elif original_cached_verdict == "support" and original_cached_verdict != fallback_verdict:
        if support_eligible:
            gated_verdict = original_cached_verdict
        else:
            gated_verdict = fallback_verdict
            gate_reasons.extend(support_reasons)
    elif original_cached_verdict == "contradict" and original_cached_verdict != fallback_verdict:
        if contradiction_eligible:
            gated_verdict = original_cached_verdict
        else:
            gated_verdict = fallback_verdict
            gate_reasons.extend(contradiction_reasons)

    if gated_verdict not in valid_verdicts:
        gated_verdict = fallback_verdict
        gate_reasons.append("gated_verdict_invalid")
    gate_reasons = _unique_text(gate_reasons)
    return {
        "original_cached_verdict": original_cached_verdict,
        "gated_verdict": gated_verdict,
        "fallback_verdict": fallback_verdict,
        "fallback_source": fallback_source,
        "gate_blocked": gated_verdict != original_cached_verdict,
        "gate_reasons": gate_reasons,
        "claim_unclear": claim_unclear,
        "direct_evidence_ids": direct_evidence_ids,
        "subject_match": subject_match,
        "time_match": time_match,
        "only_investor_questions": only_investor_questions,
        "only_market_signals": only_market_signals,
        "official_contradiction": official_contradiction,
        "safety_gate_version": _text(cfg.get("version")),
        "support_eligible": support_eligible,
        "contradiction_eligible": contradiction_eligible,
        "cached_evidence_ids": list(cached["evidence_ids"]),
        "round1_verdict": round1["verdict"],
        "baseline_verdict": baseline["verdict"],
    }


def _result_view(
    result: Mapping[str, Any] | None,
    role: str,
    cfg: Mapping[str, Any],
    valid_verdicts: set[str],
) -> dict[str, Any]:
    outer = dict(result) if isinstance(result, Mapping) else {}
    nested_name = "round1" if role == "round1" else "baseline"
    payload = outer.get(nested_name) if isinstance(outer.get(nested_name), Mapping) else outer
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    verdict_keys = ("final_verdict", "verdict") if role == "cached" else ("verdict", "final_verdict")
    verdict = _first_text(payload, verdict_keys)
    if not verdict and payload is not outer:
        verdict = _first_text(outer, verdict_keys)
    confidence, confidence_valid = _confidence(_first_value(payload, ("final_confidence", "confidence")))
    if confidence is None and confidence_valid:
        confidence, confidence_valid = _confidence(_first_value(outer, ("final_confidence", "confidence")))
    status = _first_text(payload, ("status",)) or _first_text(outer, ("status",))
    llm_valid = _first_value(payload, ("llm_valid",))
    if llm_valid is None:
        llm_valid = _first_value(outer, ("llm_valid",))
    evidence_ids = _result_evidence_ids(payload)
    if not evidence_ids and payload is not outer:
        evidence_ids = _result_evidence_ids(outer)

    parse_failed = _has_truthy_field(payload, cfg, "parse_failure_fields") or _has_truthy_field(outer, cfg, "parse_failure_fields")
    format_degraded = _has_format_degradation(payload, cfg) or _has_format_degradation(outer, cfg)
    error_text = _first_text(payload, ("error_type", "error_message")) or _first_text(outer, ("error_type", "error_message"))
    valid_statuses = set(_config_list(cfg, "valid_statuses"))
    status_usable = not status or status in valid_statuses
    usable = bool(
        verdict in valid_verdicts
        and status_usable
        and llm_valid is not False
        and not parse_failed
        and not error_text
        and confidence_valid
    )
    return {
        "verdict": verdict,
        "confidence": confidence,
        "confidence_valid": confidence_valid,
        "evidence_ids": tuple(evidence_ids),
        "parse_failed": parse_failed,
        "format_degraded": format_degraded,
        "usable": usable,
    }


def _fallback_verdict(
    round1: Mapping[str, Any],
    baseline: Mapping[str, Any],
    valid_verdicts: set[str],
) -> tuple[str, str]:
    if round1.get("usable"):
        return str(round1["verdict"]), "round1"
    if baseline.get("usable"):
        return str(baseline["verdict"]), "baseline"
    return ("insufficient" if "insufficient" in valid_verdicts else sorted(valid_verdicts)[0]), "default_insufficient"


def _claim_unclear(
    claim: Mapping[str, Any] | str | None,
    evidence_can_supply_time: bool,
    cfg: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    claim_map = claim if isinstance(claim, Mapping) else {"claim_text": claim}
    event_descriptors = [_text(claim_map.get(field)) for field in _config_list(cfg, "event_type_fields")]
    event_descriptors = [value for value in event_descriptors if value]
    claim_text = _record_text(claim_map, _config_list(cfg, "claim_text_fields"))
    reasons: list[str] = []
    if not event_descriptors:
        reasons.append("event_type_missing")
    if _matches_any(claim_text, _config_list(cfg, "claim_code_patterns"), regex=True):
        reasons.append("encoded_event_type")
    broad_only = bool(event_descriptors) and all(_is_broad_descriptor(value, cfg) for value in event_descriptors)
    has_specific_descriptor = any(
        not _is_broad_descriptor(value, cfg) and _text(value) not in set(_config_list(cfg, "generic_event_terms"))
        for value in event_descriptors
    )
    has_concrete_fact = has_specific_descriptor or _matches_any(claim_text, _config_list(cfg, "concrete_fact_patterns"))
    if not has_concrete_fact:
        reasons.append("missing_concrete_fact")
    if broad_only and not _matches_any(claim_text, _config_list(cfg, "concrete_fact_patterns")):
        reasons.append("broad_claim_only")
    claim_times = _record_dates(claim_map, _config_list(cfg, "claim_time_fields"))
    if not claim_times and not evidence_can_supply_time:
        reasons.append("missing_event_time")
    return bool(reasons), reasons


def _claim_aliases(claim: Mapping[str, Any] | str | None, cfg: Mapping[str, Any]) -> list[str]:
    claim_map = claim if isinstance(claim, Mapping) else {}
    aliases: list[str] = []
    for field in _config_list(cfg, "subject_fields"):
        value = _text(claim_map.get(field))
        normalized = re.sub(r"\s+", "", value)
        if len(normalized) >= 2 and (not normalized.isdigit() or len(normalized) >= 6):
            aliases.append(normalized)
    return _unique_text(aliases)


def _claim_event_terms(claim: Mapping[str, Any] | str | None, cfg: Mapping[str, Any]) -> list[str]:
    claim_map = claim if isinstance(claim, Mapping) else {"claim_text": claim}
    terms: list[str] = []
    ignored = set(_config_list(cfg, "generic_event_terms"))
    for field in _config_list(cfg, "event_type_fields"):
        value = _text(claim_map.get(field))
        if not value or _is_broad_descriptor(value, cfg) or value in ignored:
            continue
        value = re.sub(r"(?i)^(?:company|industry)_\d+_", "", value)
        value = re.sub(r"\s+", "", value)
        if len(value) >= 2:
            terms.append(value)
    for field in ("action", "behavior", "product", "policy", "event_object", "object"):
        value = re.sub(r"\s+", "", _text(claim_map.get(field)))
        if len(value) >= 2 and value not in ignored:
            terms.append(value)
    return _unique_text(terms)


def _evidence_items(
    evidence_records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if isinstance(evidence_records, Mapping):
        for key in ("evidence_records", "evidence", "final_evidence", "items"):
            value = evidence_records.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        return []
    if isinstance(evidence_records, Sequence) and not isinstance(evidence_records, (str, bytes)):
        return [dict(item) for item in evidence_records if isinstance(item, Mapping)]
    return []


def _evidence_id(item: Mapping[str, Any]) -> str:
    return _text(item.get("evidence_id"))


def _unique_ids(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return _unique_text([_evidence_id(item) for item in items])


def _subject_matches(item: Mapping[str, Any], aliases: Sequence[str], cfg: Mapping[str, Any]) -> bool:
    if not aliases:
        return False
    text = re.sub(r"\s+", "", _record_text(item, _config_list(cfg, "evidence_text_fields")))
    return any(alias in text for alias in aliases)


def _event_matches(item: Mapping[str, Any], event_terms: Sequence[str], cfg: Mapping[str, Any]) -> bool:
    if not event_terms:
        return False
    text = re.sub(r"\s+", "", _record_text(item, _config_list(cfg, "evidence_text_fields")))
    return any(term in text for term in event_terms)


def _time_matches(claim_dates: Sequence[datetime], evidence_dates: Sequence[datetime], cfg: Mapping[str, Any]) -> bool:
    if not evidence_dates:
        return False
    if not claim_dates:
        return True
    window_days = int(cfg.get("allowed_time_window_days", 0))
    return any(abs((evidence_date - claim_date).days) <= window_days for claim_date in claim_dates for evidence_date in evidence_dates)


def _only_investor_questions(items: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    if not items:
        return False
    return all(_is_investor_question(item, cfg) for item in items) and not any(_is_official_response(item, cfg) for item in items)


def _only_market_signals(items: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    return bool(items) and all(
        _matches_any(_record_text(item, _config_list(cfg, "evidence_text_fields")), _config_list(cfg, "market_signal_patterns"))
        for item in items
    )


def _is_investor_question(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    source = _source_value(item, cfg)
    text = _record_text(item, _config_list(cfg, "evidence_text_fields"))
    return _matches_any(source, _config_list(cfg, "investor_question_sources")) and _matches_any(
        text, _config_list(cfg, "investor_question_patterns")
    )


def _is_official_response(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    return _matches_any(
        _record_text(item, _config_list(cfg, "evidence_text_fields")),
        _config_list(cfg, "official_response_patterns"),
    )


def _is_official(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    return _matches_any(_source_value(item, cfg), _config_list(cfg, "official_source_patterns"))


def _has_explicit_contradiction(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    text = _record_text(item, _config_list(cfg, "evidence_text_fields"))
    if _matches_any(text, _config_list(cfg, "contradiction_exclusion_patterns")):
        return False
    return _matches_any(text, _config_list(cfg, "contradiction_patterns"))


def _support_block_reasons(
    cached: Mapping[str, Any],
    legal_cached_ids: Sequence[str],
    cached_direct_ids: Sequence[str],
    subject_match: bool,
    time_match: bool,
    only_investor_questions: bool,
    only_market_signals: bool,
    traceable_source: bool,
    claim_unclear: bool,
    cfg: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    threshold = float(cfg.get("support_confidence_threshold", 1.0))
    if claim_unclear:
        reasons.append("claim_unclear")
    if not legal_cached_ids:
        reasons.append("cached_support_missing_evidence_id")
    if not cached_direct_ids:
        reasons.append("cached_support_no_direct_evidence")
    if not subject_match:
        reasons.append("subject_mismatch")
    if not time_match:
        reasons.append("time_mismatch")
    if only_investor_questions:
        reasons.append("only_investor_questions")
    if only_market_signals:
        reasons.append("only_market_signals")
    if not cached.get("confidence_valid") or cached.get("confidence") is None:
        reasons.append("cached_support_confidence_invalid")
    elif float(cached["confidence"]) < threshold:
        reasons.append("cached_support_confidence_below_threshold")
    if not traceable_source:
        reasons.append("cached_support_no_traceable_source")
    if cached.get("parse_failed"):
        reasons.append("cached_support_parse_failure")
    if cached.get("format_degraded"):
        reasons.append("cached_support_format_correction")
    return reasons


def _contradiction_block_reasons(
    cached: Mapping[str, Any],
    official_contradiction: bool,
    cached_contradiction_ids: Sequence[str],
) -> list[str]:
    reasons: list[str] = []
    if not official_contradiction:
        reasons.append("official_contradiction_missing")
    if not cached_contradiction_ids:
        reasons.append("cached_contradiction_missing_evidence_id")
    if not cached.get("confidence_valid"):
        reasons.append("cached_contradiction_confidence_invalid")
    if cached.get("parse_failed"):
        reasons.append("cached_contradiction_parse_failure")
    if cached.get("format_degraded"):
        reasons.append("cached_contradiction_format_correction")
    return reasons


def _result_evidence_ids(result: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("evidence_ids", "final_evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids"):
        value = result.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(value)
    final_evidence = result.get("final_evidence")
    if isinstance(final_evidence, Sequence) and not isinstance(final_evidence, (str, bytes)):
        values.extend(item.get("evidence_id") for item in final_evidence if isinstance(item, Mapping))
    return _unique_text([_text(value) for value in values])


def _has_truthy_field(result: Mapping[str, Any], cfg: Mapping[str, Any], config_key: str) -> bool:
    for field in _config_list(cfg, config_key):
        value = result.get(field)
        if isinstance(value, str):
            if value.strip().lower() not in {"", "0", "false", "none", "null"}:
                return True
        elif bool(value):
            return True
    return False


def _has_format_degradation(result: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    for field in _config_list(cfg, "format_correction_fields"):
        value = result.get(field)
        if field == "format_retry_count":
            try:
                if int(value or 0) > 0:
                    return True
            except (TypeError, ValueError):
                return True
        elif isinstance(value, str):
            if value.strip().lower() not in {"", "0", "false", "none", "null"}:
                return True
        elif bool(value):
            return True
    return False


def _record_dates(record: Mapping[str, Any] | str | None, fields: Sequence[str]) -> list[datetime]:
    if not isinstance(record, Mapping):
        return _dates_from_text(_text(record))
    dates: list[datetime] = []
    for field in fields:
        dates.extend(_dates_from_text(_text(record.get(field))))
    return dates


def _dates_from_text(value: str) -> list[datetime]:
    text = _text(value)
    if not text:
        return []
    dates: list[datetime] = []
    for year, month, day in re.findall(r"(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})", text):
        try:
            dates.append(datetime(int(year), int(month), int(day)))
        except ValueError:
            continue
    return dates


def _record_text(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    return " ".join(_text(record.get(field)) for field in fields if _text(record.get(field)))


def _source_value(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    return " ".join(_text(item.get(field)) for field in _config_list(cfg, "source_fields") if _text(item.get(field)))


def _is_broad_descriptor(value: str, cfg: Mapping[str, Any]) -> bool:
    normalized = _text(value)
    normalized = re.sub(r"(?i)^(?:company|industry)_\d+_", "", normalized)
    return normalized in set(_config_list(cfg, "broad_claim_terms"))


def _matches_any(value: str, patterns: Sequence[str], regex: bool = False) -> bool:
    text = _text(value)
    if not text:
        return False
    if regex:
        return any(re.search(pattern, text) for pattern in patterns)
    return any(pattern and pattern in text for pattern in patterns)


def _confidence(value: Any) -> tuple[float | None, bool]:
    if value is None or _text(value) == "":
        return None, True
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None, False
    return (confidence, math.isfinite(confidence) and 0.0 <= confidence <= 1.0)


def _first_value(record: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in record and record.get(field) is not None:
            return record.get(field)
    return None


def _first_text(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    return _text(_first_value(record, fields))


def _config_list(config: Mapping[str, Any], key: str) -> list[str]:
    value = config.get(key, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [_text(item) for item in value if _text(item)]
    return []


def _unique_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = _text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
