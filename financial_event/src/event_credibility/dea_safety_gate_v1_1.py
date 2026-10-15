"""Version 1.1 deterministic safety gate for cached DEA verdicts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "task2_dea_safety_gate_v1_1.yaml"

HUMAN_ONLY_FIELDS = {
    "human_verdict",
    "human_notes",
    "human_confidence",
    "claim_is_clear",
    "preferred_result",
    "baseline_correct",
    "cached_final_correct",
    "evidence_relevance",
    "evidence_stance",
    "evidence_independent",
    "evidence_notes",
}


def load_dea_safety_gate_v1_1_config(config: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
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


def apply_dea_safety_gate_v1_1(
    claim: Mapping[str, Any] | str | None,
    baseline_result: Mapping[str, Any] | None,
    round1_result: Mapping[str, Any] | None,
    cached_result: Mapping[str, Any] | None,
    evidence_records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    config: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    cfg = load_dea_safety_gate_v1_1_config(config)
    valid_verdicts = set(_config_list(cfg, "valid_verdicts"))
    if not valid_verdicts:
        raise ValueError("DEA safety-gate config requires non-empty valid_verdicts")

    baseline = _result_view(baseline_result, "baseline", cfg, valid_verdicts)
    round1 = _result_view(round1_result, "round1", cfg, valid_verdicts)
    cached = _result_view(cached_result, "cached", cfg, valid_verdicts)
    fallback_verdict, fallback_source = _fallback_verdict(round1, baseline, valid_verdicts)

    evidence_items = [_strip_human_fields(item) for item in _evidence_items(evidence_records)]
    evidence_by_id = {
        evidence_id: item
        for item in evidence_items
        if (evidence_id := _evidence_id(item))
    }
    legal_cached_ids = [evidence_id for evidence_id in cached["evidence_ids"] if evidence_id in evidence_by_id]
    referenced_items = [evidence_by_id[evidence_id] for evidence_id in legal_cached_ids]

    scores = [score_direct_evidence(claim, item, cfg) for item in evidence_items]
    score_by_id = {
        _text(score.get("evidence_id")): score
        for score in scores
        if _text(score.get("evidence_id"))
    }
    direct_evidence_ids = [
        score["evidence_id"]
        for score in scores
        if score.get("is_direct_evidence") and score.get("evidence_id")
    ]
    cached_direct_ids = [evidence_id for evidence_id in legal_cached_ids if evidence_id in set(direct_evidence_ids)]
    cached_direct_items = [evidence_by_id[evidence_id] for evidence_id in cached_direct_ids]

    subject_match = any(float(score.get("subject_score", 0.0)) > 0.0 for score in scores)
    min_time = float(cfg.get("direct_time_score_threshold", 0.6))
    time_match = any(
        float(score.get("subject_score", 0.0)) > 0.0 and float(score.get("time_score", 0.0)) >= min_time
        for score in scores
    )

    raw_claim_incomplete, incomplete_reasons = _raw_claim_incomplete(claim, cfg)
    evidence_resolves_claim = bool(cached_direct_ids or direct_evidence_ids)
    claim_truly_unclear = bool(raw_claim_incomplete and not evidence_resolves_claim)

    model_used_scope = cached_direct_items or referenced_items
    only_investor_questions = _only_investor_questions(model_used_scope, cfg)
    only_market_signals = _only_market_signals(model_used_scope, cfg)

    contradiction_items = [
        item
        for item in referenced_items
        if _is_official(item, cfg)
        and _has_explicit_contradiction(item, cfg)
        and _subject_matches(item, _claim_aliases(claim, cfg), cfg)
        and _time_reasonable(claim, item, cfg)
    ]
    contradiction_ids = _unique_text([_evidence_id(item) for item in contradiction_items])
    official_contradiction = bool(contradiction_items)

    support_reasons = _support_block_reasons(
        cached=cached,
        legal_cached_ids=legal_cached_ids,
        cached_direct_ids=cached_direct_ids,
        subject_match=subject_match,
        time_match=time_match,
        only_investor_questions=only_investor_questions,
        only_market_signals=only_market_signals,
        traceable_source=any(_traceable(item, cfg) for item in cached_direct_items or referenced_items),
        claim_truly_unclear=claim_truly_unclear,
        cfg=cfg,
    )
    contradiction_reasons = _contradiction_block_reasons(
        cached=cached,
        official_contradiction=official_contradiction,
        cached_contradiction_ids=[evidence_id for evidence_id in legal_cached_ids if evidence_id in set(contradiction_ids)],
    )
    support_eligible = not support_reasons
    contradiction_eligible = not contradiction_reasons

    gate_reasons: list[str] = []
    original_cached_verdict = cached["verdict"]
    gated_verdict = original_cached_verdict if cached["usable"] else fallback_verdict
    if not cached["usable"]:
        gate_reasons.append("cached_result_invalid")

    if claim_truly_unclear:
        gated_verdict = "unclear"
        gate_reasons.extend(["claim_truly_unclear", *incomplete_reasons])
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

    return {
        "original_cached_verdict": original_cached_verdict,
        "gated_verdict": gated_verdict,
        "fallback_verdict": fallback_verdict,
        "fallback_source": fallback_source,
        "gate_blocked": gated_verdict != original_cached_verdict,
        "gate_reasons": _unique_text(gate_reasons),
        "claim_unclear": claim_truly_unclear,
        "raw_claim_incomplete": raw_claim_incomplete,
        "raw_claim_incomplete_reasons": incomplete_reasons,
        "evidence_resolves_claim": evidence_resolves_claim,
        "claim_truly_unclear": claim_truly_unclear,
        "direct_evidence_ids": direct_evidence_ids,
        "cached_direct_evidence_ids": cached_direct_ids,
        "direct_evidence_scores": {evidence_id: score_by_id[evidence_id] for evidence_id in direct_evidence_ids if evidence_id in score_by_id},
        "all_evidence_scores": score_by_id,
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


def apply_dea_safety_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return apply_dea_safety_gate_v1_1(*args, **kwargs)


def score_direct_evidence(
    claim: Mapping[str, Any] | str | None,
    evidence: Mapping[str, Any] | None,
    config: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    cfg = load_dea_safety_gate_v1_1_config(config)
    item = _strip_human_fields(evidence if isinstance(evidence, Mapping) else {})
    evidence_id = _evidence_id(item)
    text = _record_text(item, _config_list(cfg, "evidence_text_fields"))
    body_text_fields = _config_list(cfg, "evidence_body_text_fields") or _config_list(cfg, "evidence_text_fields")
    body_text = _record_text(item, body_text_fields)
    source = _source_value(item, cfg)
    aliases = _claim_aliases(claim, cfg)
    subject_score = 1.0 if _subject_matches(item, aliases, cfg) else 0.0

    semantic_matches = _event_semantic_matches(claim, body_text or text, cfg)
    semantic_score = min(1.0, len(semantic_matches) / max(1, int(cfg.get("min_event_semantic_hits", 1))))
    concrete_matches = _concrete_fact_matches(body_text or text, source, cfg)
    concrete_score = min(1.0, len(concrete_matches) / max(1, int(cfg.get("min_concrete_fact_hits", 1))))
    time_score, time_reason = _time_score(claim, item, cfg)
    source_score = _source_score(item, cfg)
    traceability_score = 1.0 if evidence_id and _traceable(item, cfg) else 0.0
    question_penalty = 1.0 if _is_investor_question(item, cfg) else 0.0
    market_signal_penalty = 1.0 if _is_market_signal(item, cfg) else 0.0
    duplicate_penalty = _duplicate_penalty(item)

    weights = cfg.get("direct_evidence_score_weights") or {}
    total = (
        subject_score * float(weights.get("subject", 0.25))
        + semantic_score * float(weights.get("event_semantic", 0.25))
        + concrete_score * float(weights.get("concrete_fact", 0.2))
        + time_score * float(weights.get("time", 0.12))
        + source_score * float(weights.get("source", 0.08))
        + traceability_score * float(weights.get("traceability", 0.1))
        - question_penalty * float(weights.get("question_penalty", 0.4))
        - market_signal_penalty * float(weights.get("market_signal_penalty", 0.4))
        - duplicate_penalty * float(weights.get("duplicate_penalty", 0.1))
    )
    total = max(0.0, min(1.0, total))
    match_reasons: list[str] = []
    if subject_score > 0:
        match_reasons.append("subject_match")
    match_reasons.extend(f"event:{term}" for term in semantic_matches[:5])
    match_reasons.extend(f"fact:{term}" for term in concrete_matches[:5])
    if time_reason:
        match_reasons.append(time_reason)
    if source_score > 0:
        match_reasons.append("source_present")
    if traceability_score > 0:
        match_reasons.append("traceable_evidence_id")
    if question_penalty:
        match_reasons.append("investor_question_penalty")
    if market_signal_penalty:
        match_reasons.append("market_signal_penalty")

    is_direct = bool(
        evidence_id
        and subject_score >= float(cfg.get("direct_subject_score_threshold", 1.0))
        and semantic_score >= float(cfg.get("direct_event_semantic_score_threshold", 1.0))
        and concrete_score >= float(cfg.get("direct_concrete_fact_score_threshold", 1.0))
        and time_score >= float(cfg.get("direct_time_score_threshold", 0.6))
        and traceability_score >= float(cfg.get("direct_traceability_score_threshold", 1.0))
        and not question_penalty
        and not market_signal_penalty
        and total >= float(cfg.get("direct_evidence_total_threshold", 0.72))
    )
    return {
        "evidence_id": evidence_id,
        "subject_score": round(subject_score, 6),
        "event_semantic_score": round(semantic_score, 6),
        "concrete_fact_score": round(concrete_score, 6),
        "time_score": round(time_score, 6),
        "source_score": round(source_score, 6),
        "traceability_score": round(traceability_score, 6),
        "question_penalty": round(question_penalty, 6),
        "market_signal_penalty": round(market_signal_penalty, 6),
        "duplicate_penalty": round(duplicate_penalty, 6),
        "total_score": round(total, 6),
        "is_direct_evidence": is_direct,
        "match_reasons": _unique_text(match_reasons),
    }


def _raw_claim_incomplete(claim: Mapping[str, Any] | str | None, cfg: Mapping[str, Any]) -> tuple[bool, list[str]]:
    claim_map = claim if isinstance(claim, Mapping) else {"claim_text": claim}
    reasons: list[str] = []
    event_descriptors = [_text(claim_map.get(field)) for field in _config_list(cfg, "event_type_fields")]
    event_descriptors = [value for value in event_descriptors if value]
    raw_text = _claim_text_for_completeness(claim_map, cfg)
    if not event_descriptors:
        reasons.append("event_type_missing")
    if _matches_any(raw_text, _config_list(cfg, "claim_code_patterns"), regex=True):
        reasons.append("encoded_event_type")
    if not _claim_has_concrete_fact(raw_text, event_descriptors, cfg):
        reasons.append("missing_concrete_fact")
    if event_descriptors and all(_is_broad_descriptor(value, cfg) for value in event_descriptors):
        reasons.append("broad_claim_only")
    if not _record_dates(claim_map, _config_list(cfg, "claim_time_fields")):
        reasons.append("missing_event_time")
    return bool(reasons), _unique_text(reasons)


def _claim_has_concrete_fact(text: str, event_descriptors: Sequence[str], cfg: Mapping[str, Any]) -> bool:
    if _concrete_fact_matches(text, "", cfg):
        return True
    ignored = set(_config_list(cfg, "generic_event_terms"))
    for value in event_descriptors:
        if value and not _is_broad_descriptor(value, cfg) and value not in ignored:
            return True
    return False


def _claim_text_for_completeness(claim_map: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    fields = _config_list(cfg, "claim_text_fields") + ["raw_claim_text", "raw_schema_id"]
    return _record_text(claim_map, fields)


def _event_semantic_matches(claim: Mapping[str, Any] | str | None, evidence_text: str, cfg: Mapping[str, Any]) -> list[str]:
    text = re.sub(r"\s+", "", _text(evidence_text))
    terms = _event_semantic_terms(claim, cfg)
    return _unique_text([term for term in terms if term and re.sub(r"\s+", "", term) in text])


def _event_semantic_terms(claim: Mapping[str, Any] | str | None, cfg: Mapping[str, Any]) -> list[str]:
    claim_map = claim if isinstance(claim, Mapping) else {"claim_text": claim}
    descriptors: list[str] = []
    for field in _config_list(cfg, "event_type_fields"):
        value = _text(claim_map.get(field))
        if value:
            descriptors.append(value)
    normalized_descriptors = [re.sub(r"\s+", "", re.sub(r"(?i)^(?:company|industry)_\d+_", "", item)) for item in descriptors]
    terms: list[str] = []
    for group in cfg.get("event_semantic_groups") or []:
        if not isinstance(group, Mapping):
            continue
        patterns = [re.sub(r"\s+", "", _text(item)) for item in group.get("event_type_patterns") or [] if _text(item)]
        if any(pattern and any(pattern in descriptor for descriptor in normalized_descriptors) for pattern in patterns):
            terms.extend(_text(item) for item in group.get("action_terms") or [] if _text(item))
    ignored = set(_config_list(cfg, "generic_event_terms"))
    for value in normalized_descriptors:
        if value and value not in ignored and not _is_broad_descriptor(value, cfg):
            terms.append(value)
    return _unique_text(terms)


def _concrete_fact_matches(text: str, source: str, cfg: Mapping[str, Any]) -> list[str]:
    value = _text(text)
    matches: list[str] = []
    for pattern in _config_list(cfg, "concrete_fact_patterns"):
        if pattern and pattern in value:
            matches.append(pattern)
    regex_matches: list[str] = []
    for pattern in _config_list(cfg, "concrete_fact_regex_patterns"):
        if pattern and re.search(pattern, value):
            regex_matches.append(pattern)
    substantive_regex = [pattern for pattern in regex_matches if "20\\d{2}" not in pattern]
    matches.extend(substantive_regex)
    if matches and any("20\\d{2}" in pattern for pattern in regex_matches):
        matches.append("event_date_anchor")
    if matches and _matches_any(source, _config_list(cfg, "official_source_patterns")) and _matches_any(value, _config_list(cfg, "official_action_patterns")):
        matches.append("official_action")
    return _unique_text(matches)


def _time_score(claim: Mapping[str, Any] | str | None, item: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[float, str]:
    claim_dates = _record_dates(claim, _config_list(cfg, "claim_time_fields"))
    evidence_dates = _record_dates(item, _config_list(cfg, "evidence_time_fields"))
    if not evidence_dates:
        return 0.0, "evidence_time_missing"
    if not claim_dates:
        return 0.8, "evidence_time_supplies_missing_claim_time"
    window_days = int(cfg.get("allowed_time_window_days", 0))
    background_days = int(cfg.get("background_time_window_days", window_days))
    min_delta = min(abs((evidence_date - claim_date).days) for claim_date in claim_dates for evidence_date in evidence_dates)
    if min_delta <= window_days:
        return 1.0, "time_match"
    if min_delta <= background_days:
        return 0.3, "background_time_only"
    return 0.0, "time_mismatch"


def _time_reasonable(claim: Mapping[str, Any] | str | None, item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    score, _ = _time_score(claim, item, cfg)
    return score >= float(cfg.get("direct_time_score_threshold", 0.6))


def _source_score(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    source = _source_value(item, cfg)
    if not source:
        return 0.0
    if _is_official(item, cfg):
        return 1.0
    if _matches_any(source, _config_list(cfg, "mainstream_source_patterns")):
        return 0.8
    return 0.6


def _duplicate_penalty(item: Mapping[str, Any]) -> float:
    try:
        duplicate_count = int(_first_value(item, ("duplicate_count", "duplicate_group_size")) or 0)
    except (TypeError, ValueError):
        duplicate_count = 0
    return 0.5 if duplicate_count > 3 else 0.0


def _support_block_reasons(
    cached: Mapping[str, Any],
    legal_cached_ids: Sequence[str],
    cached_direct_ids: Sequence[str],
    subject_match: bool,
    time_match: bool,
    only_investor_questions: bool,
    only_market_signals: bool,
    traceable_source: bool,
    claim_truly_unclear: bool,
    cfg: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    threshold = float(cfg.get("support_confidence_threshold", 1.0))
    if claim_truly_unclear:
        reasons.append("claim_truly_unclear")
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
    return _unique_text(reasons)


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
    return _unique_text(reasons)


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
    llm_valid = _normalize_optional_bool(llm_valid, default=True)
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


def _fallback_verdict(round1: Mapping[str, Any], baseline: Mapping[str, Any], valid_verdicts: set[str]) -> tuple[str, str]:
    if round1.get("usable"):
        return str(round1["verdict"]), "round1"
    if baseline.get("usable"):
        return str(baseline["verdict"]), "baseline"
    return ("insufficient" if "insufficient" in valid_verdicts else sorted(valid_verdicts)[0]), "default_insufficient"


def _evidence_items(evidence_records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if isinstance(evidence_records, Mapping):
        for key in ("evidence_records", "evidence", "final_evidence", "items"):
            value = evidence_records.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        return []
    if isinstance(evidence_records, Sequence) and not isinstance(evidence_records, (str, bytes)):
        return [dict(item) for item in evidence_records if isinstance(item, Mapping)]
    return []


def _strip_human_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in record.items() if str(key) not in HUMAN_ONLY_FIELDS}


def _claim_aliases(claim: Mapping[str, Any] | str | None, cfg: Mapping[str, Any]) -> list[str]:
    claim_map = claim if isinstance(claim, Mapping) else {}
    aliases: list[str] = []
    for field in _config_list(cfg, "subject_fields"):
        value = _text(claim_map.get(field))
        normalized = re.sub(r"\s+", "", value)
        if len(normalized) >= 2 and (not normalized.isdigit() or len(normalized) >= 6):
            aliases.append(normalized)
    return _unique_text(aliases)


def _subject_matches(item: Mapping[str, Any], aliases: Sequence[str], cfg: Mapping[str, Any]) -> bool:
    if not aliases:
        return False
    text = re.sub(r"\s+", "", _record_text(item, _config_list(cfg, "evidence_text_fields")))
    return any(alias in text for alias in aliases)


def _only_investor_questions(items: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    return bool(items) and all(_is_investor_question(item, cfg) for item in items)


def _only_market_signals(items: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    return bool(items) and all(_is_market_signal(item, cfg) for item in items)


def _is_investor_question(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    source = _source_value(item, cfg)
    text = _record_text(item, _config_list(cfg, "evidence_text_fields"))
    return _matches_any(source, _config_list(cfg, "investor_question_sources")) and _matches_any(
        text, _config_list(cfg, "investor_question_patterns")
    ) and not _matches_any(text, _config_list(cfg, "official_response_patterns"))


def _is_market_signal(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    text = _record_text(item, _config_list(cfg, "evidence_text_fields"))
    return _matches_any(text, _config_list(cfg, "market_signal_patterns"))


def _is_official(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    return _matches_any(_source_value(item, cfg), _config_list(cfg, "official_source_patterns"))


def _has_explicit_contradiction(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    text = _record_text(item, _config_list(cfg, "evidence_text_fields"))
    if _matches_any(text, _config_list(cfg, "contradiction_exclusion_patterns")):
        return False
    return _matches_any(text, _config_list(cfg, "contradiction_patterns"))


def _traceable(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    trace_status = _first_text(item, ("traceable", "可追溯状态", "trace_status"))
    if trace_status.lower() in {"false", "no", "0", "否", "failed", "missing"}:
        return False
    if trace_status.lower() in {"true", "yes", "1", "是", "ok"}:
        return True
    return bool(_evidence_id(item) and _source_value(item, cfg))


def _result_evidence_ids(result: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("evidence_ids", "cited_evidence_ids", "direct_evidence_ids", "final_evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids"):
        value = result.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(value)
        else:
            values.extend(_evidence_ids_from_text(_text(value)))
    final_evidence = result.get("final_evidence")
    if isinstance(final_evidence, Sequence) and not isinstance(final_evidence, (str, bytes)):
        values.extend(item.get("evidence_id") for item in final_evidence if isinstance(item, Mapping))
    for key in ("reason", "final_reason", "raw_response", "raw_response_preview"):
        values.extend(_evidence_ids_from_text(_text(result.get(key))))
    return _unique_text([_normalize_evidence_id(value) for value in values])


def _evidence_ids_from_text(text: str) -> list[str]:
    return [f"E{int(number)}" for number in re.findall(r"(?<![A-Za-z0-9])E\s*[-_]?\s*(\d+)(?![A-Za-z0-9])", text)]


def _normalize_evidence_id(value: Any) -> str:
    text = _text(value)
    match = re.fullmatch(r"E0*(\d+)", text)
    return f"E{int(match.group(1))}" if match else text


def _evidence_id(item: Mapping[str, Any]) -> str:
    return _normalize_evidence_id(_first_text(item, ("evidence_id", "id")))


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


def _first_value(record: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in record and record.get(field) is not None:
            return record.get(field)
    return None


def _first_text(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    return _text(_first_value(record, fields))


def _normalize_optional_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or _text(value) == "":
        return default
    text = _text(value).lower()
    if text in {"yes", "true", "1", "y", "是"}:
        return True
    if text in {"no", "false", "0", "n", "否"}:
        return False
    return default


def _config_list(config: Mapping[str, Any], key: str) -> list[str]:
    value = config.get(key, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [_text(item) for item in value if _text(item)]
    return []


def _unique_text(values: Sequence[Any]) -> list[str]:
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
