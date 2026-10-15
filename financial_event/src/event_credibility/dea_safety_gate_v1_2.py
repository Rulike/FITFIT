"""Version 1.2 conservative selector for cached DEA verdicts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.event_credibility import dea_safety_gate_v1_1 as v11


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "task2_dea_safety_gate_v1_2.yaml"
SAFETY_GATE_VERSION = "1.2"

HUMAN_ONLY_FIELDS = set(v11.HUMAN_ONLY_FIELDS) | {
    "gold_verdict",
    "preferred_result",
    "audit_gate_should_block",
    "audit_claim_unclear",
    "audit_direct_evidence_valid",
    "audit_mapping_bug",
    "audit_rule_too_strict",
    "audit_notes",
}


def load_dea_safety_gate_v1_2_config(config: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
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


def apply_dea_safety_gate_v1_2(
    claim: Mapping[str, Any] | str | None,
    baseline_result: Mapping[str, Any] | None,
    round1_result: Mapping[str, Any] | None,
    cached_result: Mapping[str, Any] | None,
    evidence_records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    config: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    cfg = load_dea_safety_gate_v1_2_config(config)
    valid_verdicts = set(_config_list(cfg, "valid_verdicts"))
    if not valid_verdicts:
        raise ValueError("DEA safety-gate config requires non-empty valid_verdicts")

    clean_claim = _strip_human_fields(claim if isinstance(claim, Mapping) else {"claim_text": claim})
    evidence_items = [_strip_human_fields(item) for item in _evidence_items(evidence_records)]
    evidence_by_id = {
        evidence_id: item
        for item in evidence_items
        if (evidence_id := _evidence_id(item))
    }

    baseline = _result_view_v1_2(baseline_result, "baseline", cfg, valid_verdicts, evidence_by_id)
    round1 = _result_view_v1_2(round1_result, "round1", cfg, valid_verdicts, evidence_by_id)
    cached = _result_view_v1_2(cached_result, "cached", cfg, valid_verdicts, evidence_by_id)
    fallback_verdict, fallback_source = _fallback_verdict(round1, baseline, valid_verdicts)

    scores = [v11.score_direct_evidence(clean_claim, item, cfg) for item in evidence_items]
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

    cited_evidence_ids = list(cached["evidence_ids"])
    valid_cited_evidence_ids = [evidence_id for evidence_id in cited_evidence_ids if evidence_id in evidence_by_id]
    cited_items = [evidence_by_id[evidence_id] for evidence_id in valid_cited_evidence_ids]
    direct_items = [evidence_by_id[evidence_id] for evidence_id in direct_evidence_ids if evidence_id in evidence_by_id]
    audit_items = cited_items or direct_items

    cited_subject_match = _scope_subject_match(clean_claim, audit_items, cfg)
    cited_time_match, cited_time_conflict = _scope_time_status(clean_claim, audit_items, cfg)
    cited_contains_concrete_fact = _scope_contains_concrete_fact(clean_claim, audit_items, cfg)
    cited_traceable = any(_traceable(item, cfg) for item in audit_items)
    cited_only_investor_questions = _only_investor_questions(audit_items, cfg)
    cited_only_market_signals = _only_market_signals(audit_items, cfg)

    raw_claim_incomplete, raw_claim_reasons = _raw_claim_incomplete(clean_claim, cfg)
    evidence_resolves_claim = _evidence_resolves_claim(clean_claim, audit_items, cfg)
    structurally_unverifiable, structural_reasons = _structurally_unverifiable(
        clean_claim,
        audit_items,
        evidence_resolves_claim,
        cfg,
    )

    original_cached_verdict = cached["verdict"]
    gate_reasons: list[str] = []
    selected_result_source = "cached"
    fallback_reason = "cached_retained"
    gated_verdict = original_cached_verdict if cached["usable"] else fallback_verdict

    if not cached["usable"]:
        gate_reasons.extend(["cached_result_invalid", *cached["invalid_reasons"]])
        gated_verdict, selected_result_source = _blocked_fallback(
            structurally_unverifiable,
            fallback_verdict,
            fallback_source,
        )
        fallback_reason = "cached_result_invalid"
    elif original_cached_verdict == "insufficient":
        gated_verdict = original_cached_verdict
        selected_result_source = "cached"
        fallback_reason = "cached_insufficient_retained"
    elif original_cached_verdict == "unclear":
        if structurally_unverifiable:
            gated_verdict = "unclear"
            selected_result_source = "cached"
            fallback_reason = "cached_unclear_structural"
        else:
            gated_verdict, selected_result_source = _blocked_fallback(False, fallback_verdict, fallback_source)
            fallback_reason = "cached_unclear_but_claim_verifiable"
            gate_reasons.append("cached_unclear_but_claim_verifiable")
    elif structurally_unverifiable:
        gated_verdict = "unclear"
        selected_result_source = "safety_unclear"
        fallback_reason = "structurally_unverifiable"
        gate_reasons.extend(["structurally_unverifiable", *structural_reasons])
    elif original_cached_verdict == "support":
        support_reasons = _cached_support_block_reasons(
            clean_claim,
            cached,
            audit_items,
            cited_subject_match,
            cited_time_conflict,
            cited_contains_concrete_fact,
            cited_traceable,
            cited_only_investor_questions,
            cited_only_market_signals,
            evidence_resolves_claim,
            cfg,
        )
        if support_reasons:
            gated_verdict, selected_result_source = _blocked_fallback(False, fallback_verdict, fallback_source)
            fallback_reason = "cached_support_high_risk"
            gate_reasons.extend(support_reasons)
    elif original_cached_verdict == "contradict":
        contradiction_reasons = _cached_contradict_block_reasons(
            clean_claim,
            audit_items,
            cited_subject_match,
            cited_time_conflict,
            cited_traceable,
            cfg,
        )
        if contradiction_reasons:
            gated_verdict, selected_result_source = _blocked_fallback(False, fallback_verdict, fallback_source)
            fallback_reason = "cached_contradict_not_verified"
            gate_reasons.extend(contradiction_reasons)
    if gated_verdict not in valid_verdicts:
        gated_verdict = "insufficient" if "insufficient" in valid_verdicts else sorted(valid_verdicts)[0]
        selected_result_source = "default_insufficient"
        fallback_reason = "gated_verdict_invalid"
        gate_reasons.append("gated_verdict_invalid")

    return {
        "original_cached_verdict": original_cached_verdict,
        "gated_verdict": gated_verdict,
        "fallback_verdict": fallback_verdict,
        "selected_result_source": selected_result_source,
        "gate_blocked": selected_result_source != "cached",
        "gate_reasons": _unique_text(gate_reasons),
        "structurally_unverifiable": bool(structurally_unverifiable),
        "raw_claim_incomplete": bool(raw_claim_incomplete),
        "raw_claim_incomplete_reasons": raw_claim_reasons,
        "evidence_resolves_claim": bool(evidence_resolves_claim),
        "cited_evidence_ids": cited_evidence_ids,
        "valid_cited_evidence_ids": valid_cited_evidence_ids,
        "unmatched_cited_evidence_ids": list(cached["unmatched_evidence_ids"]),
        "cited_evidence_scope_size": len(audit_items),
        "cited_subject_match": bool(cited_subject_match),
        "cited_time_match": bool(cited_time_match),
        "cited_contains_concrete_fact": bool(cited_contains_concrete_fact),
        "cited_only_investor_questions": bool(cited_only_investor_questions),
        "cited_only_market_signals": bool(cited_only_market_signals),
        "cited_traceable": bool(cited_traceable),
        "direct_evidence_ids": direct_evidence_ids,
        "direct_evidence_scores": {
            evidence_id: score_by_id[evidence_id]
            for evidence_id in direct_evidence_ids
            if evidence_id in score_by_id
        },
        "all_evidence_scores": score_by_id,
        "direct_evidence_diagnostic_only": True,
        "fallback_reason": fallback_reason,
        "safety_gate_version": SAFETY_GATE_VERSION,
    }


def apply_dea_safety_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return apply_dea_safety_gate_v1_2(*args, **kwargs)


def _result_view_v1_2(
    result: Mapping[str, Any] | None,
    role: str,
    cfg: Mapping[str, Any],
    valid_verdicts: set[str],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    view = dict(v11._result_view(result, role, cfg, valid_verdicts))
    outer = dict(result) if isinstance(result, Mapping) else {}
    unmatched = [_text(item) for item in outer.get("unmatched_cited_evidence_ids") or [] if _text(item)]
    explicit_ids = list(view.get("evidence_ids") or [])
    invalid_reasons: list[str] = []
    if view.get("verdict") not in valid_verdicts:
        invalid_reasons.append(f"{role}_verdict_invalid")
    if view.get("confidence") is None or not view.get("confidence_valid"):
        invalid_reasons.append(f"{role}_confidence_invalid")
    if unmatched:
        invalid_reasons.append(f"{role}_unmatched_evidence_id")
    missing_explicit = [evidence_id for evidence_id in explicit_ids if evidence_id and evidence_id not in evidence_by_id]
    if missing_explicit:
        invalid_reasons.append(f"{role}_untraceable_evidence_id")
        unmatched.extend(missing_explicit)
    view["unmatched_evidence_ids"] = _unique_text(unmatched)
    view["invalid_reasons"] = _unique_text(invalid_reasons)
    view["usable"] = bool(view.get("usable") and not invalid_reasons)
    return view


def _fallback_verdict(round1: Mapping[str, Any], baseline: Mapping[str, Any], valid_verdicts: set[str]) -> tuple[str, str]:
    if round1.get("usable"):
        return _text(round1.get("verdict")), "round1"
    if baseline.get("usable"):
        return _text(baseline.get("verdict")), "baseline"
    return ("insufficient" if "insufficient" in valid_verdicts else sorted(valid_verdicts)[0]), "default_insufficient"


def _blocked_fallback(structural: bool, fallback_verdict: str, fallback_source: str) -> tuple[str, str]:
    if structural:
        return "unclear", "safety_unclear"
    return fallback_verdict, fallback_source


def _cached_support_block_reasons(
    claim: Mapping[str, Any],
    cached: Mapping[str, Any],
    scope: Sequence[Mapping[str, Any]],
    subject_match: bool,
    time_conflict: bool,
    contains_concrete_fact: bool,
    traceable: bool,
    only_investor_questions: bool,
    only_market_signals: bool,
    evidence_resolves_claim: bool,
    cfg: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not scope or not traceable:
        reasons.append("cached_support_all_cited_untraceable")
    if scope and _claim_has_subject(claim, cfg) and not subject_match:
        reasons.append("cached_support_subject_mismatch")
    if time_conflict:
        reasons.append("cached_support_time_conflict")
    if only_investor_questions:
        reasons.append("cited_only_investor_questions")
    if only_market_signals:
        reasons.append("cited_only_market_signals")
    if not contains_concrete_fact and not evidence_resolves_claim:
        reasons.append("cached_support_no_event_fact")
    if not cached.get("confidence_valid") or cached.get("confidence") is None:
        reasons.append("cached_support_confidence_invalid")
    return _unique_text(reasons)


def _cached_contradict_block_reasons(
    claim: Mapping[str, Any],
    scope: Sequence[Mapping[str, Any]],
    subject_match: bool,
    time_conflict: bool,
    traceable: bool,
    cfg: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    contradiction_items = [
        item
        for item in scope
        if _traceable(item, cfg)
        and v11._has_explicit_contradiction(item, cfg)
        and _scope_subject_match(claim, [item], cfg)
        and not _scope_time_status(claim, [item], cfg)[1]
    ]
    if not scope or not traceable:
        reasons.append("cached_contradict_missing_traceable_evidence")
    if scope and _claim_has_subject(claim, cfg) and not subject_match:
        reasons.append("cached_contradict_subject_mismatch")
    if time_conflict:
        reasons.append("cached_contradict_time_conflict")
    if not contradiction_items:
        reasons.append("cached_contradict_explicit_reversal_missing")
    return _unique_text(reasons)


def _structurally_unverifiable(
    claim: Mapping[str, Any],
    scope: Sequence[Mapping[str, Any]],
    evidence_resolves_claim: bool,
    cfg: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    if evidence_resolves_claim:
        return False, []
    claim_text = _claim_text(claim, cfg)
    reasons: list[str] = []
    has_subject = _claim_has_subject(claim, cfg)
    has_event_fact = _text_has_event_or_fact(claim_text, cfg) or _claim_event_descriptors(claim, cfg)
    has_declarative_fact = any(_item_has_declarative_fact(item, cfg) for item in scope)
    if not has_subject and not has_event_fact:
        reasons.append("no_subject_and_no_event_fact")
    if _is_question_or_opinion(claim_text, cfg) and not has_declarative_fact:
        reasons.append("question_or_opinion_without_declarative_fact")
    if _encoded_placeholder_only(claim, cfg) and not has_declarative_fact:
        reasons.append("unresolved_encoded_placeholder")
    if scope and not any(_forms_subject_event_fact_relation(claim, item, cfg) for item in scope):
        if not has_subject and not has_event_fact:
            reasons.append("no_subject_event_fact_relation")
    if not scope and not has_subject and not has_event_fact:
        reasons.append("no_cited_evidence_to_resolve_claim")
    return bool(reasons), _unique_text(reasons)


def _evidence_resolves_claim(claim: Mapping[str, Any], scope: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    return any(_forms_subject_event_fact_relation(claim, item, cfg) for item in scope if _traceable(item, cfg))


def _forms_subject_event_fact_relation(claim: Mapping[str, Any], item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    return bool(
        _scope_subject_match(claim, [item], cfg)
        and _item_has_event_action_or_state(claim, item, cfg)
        and _item_contains_concrete_fact(item, cfg)
    )


def _scope_subject_match(claim: Mapping[str, Any], scope: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    aliases = v11._claim_aliases(claim, cfg)
    return any(v11._subject_matches(item, aliases, cfg) for item in scope)


def _scope_time_status(claim: Mapping[str, Any], scope: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> tuple[bool, bool]:
    if not scope:
        return False, False
    claim_dates = v11._record_dates(claim, _config_list(cfg, "claim_time_fields"))
    scored = [(item, v11._time_score(claim, item, cfg)) for item in scope]
    threshold = float(cfg.get("direct_time_score_threshold", 0.6))
    has_match = any(score >= threshold for _, (score, _) in scored)
    if not claim_dates:
        return True, False
    evidence_dates_present = [bool(v11._record_dates(item, _config_list(cfg, "evidence_time_fields"))) for item, _ in scored]
    conflict = bool(scored) and all(evidence_dates_present) and not has_match
    return has_match, conflict


def _scope_contains_concrete_fact(claim: Mapping[str, Any], scope: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    return any(_item_contains_concrete_fact(item, cfg) and _item_has_event_action_or_state(claim, item, cfg) for item in scope)


def _item_has_declarative_fact(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    text = _record_text(item, _config_list(cfg, "evidence_body_text_fields") or _config_list(cfg, "evidence_text_fields"))
    return bool(_item_contains_concrete_fact(item, cfg) or _matches_any(text, _config_list(cfg, "declarative_fact_patterns")))


def _item_contains_concrete_fact(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    text = _record_text(item, _config_list(cfg, "evidence_body_text_fields") or _config_list(cfg, "evidence_text_fields"))
    source = v11._source_value(item, cfg)
    return bool(v11._concrete_fact_matches(text, source, cfg))


def _item_has_event_action_or_state(claim: Mapping[str, Any], item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    text = _record_text(item, _config_list(cfg, "evidence_body_text_fields") or _config_list(cfg, "evidence_text_fields"))
    return bool(
        v11._event_semantic_matches(claim, text, cfg)
        or _matches_any(text, _config_list(cfg, "event_action_terms"))
        or _matches_any(text, _config_list(cfg, "event_state_terms"))
        or _matches_any(text, _config_list(cfg, "official_action_patterns"))
    )


def _only_investor_questions(scope: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    if not scope:
        return False
    return all(v11._is_investor_question(item, cfg) for item in scope)


def _only_market_signals(scope: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    if not scope:
        return False
    return all(_is_market_signal_only(item, cfg) for item in scope)


def _is_market_signal_only(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    text = _record_text(item, _config_list(cfg, "evidence_text_fields"))
    if not _matches_any(text, _config_list(cfg, "market_signal_patterns")):
        return False
    return not _item_has_declarative_fact(item, cfg) or not _matches_any(text, _config_list(cfg, "non_market_fact_patterns"))


def _raw_claim_incomplete(claim: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, list[str]]:
    return v11._raw_claim_incomplete(claim, cfg)


def _claim_has_subject(claim: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    return bool(v11._claim_aliases(claim, cfg))


def _claim_event_descriptors(claim: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[str]:
    return [_text(claim.get(field)) for field in _config_list(cfg, "event_type_fields") if _text(claim.get(field))]


def _claim_text(claim: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    fields = _config_list(cfg, "claim_text_fields") + ["raw_claim_text", "raw_schema_id"]
    return _record_text(claim, fields)


def _text_has_event_or_fact(text: str, cfg: Mapping[str, Any]) -> bool:
    return bool(
        _matches_any(text, _config_list(cfg, "event_action_terms"))
        or _matches_any(text, _config_list(cfg, "event_state_terms"))
        or _matches_any(text, _config_list(cfg, "concrete_fact_patterns"))
        or _matches_any(text, _config_list(cfg, "declarative_fact_patterns"))
    )


def _is_question_or_opinion(text: str, cfg: Mapping[str, Any]) -> bool:
    return _matches_any(text, _config_list(cfg, "question_or_opinion_patterns"))


def _encoded_placeholder_only(claim: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    raw = _text(claim.get("raw_claim_text")) or _text(claim.get("claim_text")) or _text(claim.get("event_claim"))
    if not _matches_any(raw, _config_list(cfg, "claim_code_patterns"), regex=True):
        return False
    descriptors = _claim_event_descriptors(claim, cfg)
    has_time = bool(v11._record_dates(claim, _config_list(cfg, "claim_time_fields")))
    return not descriptors and not has_time


def _evidence_items(evidence_records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return v11._evidence_items(evidence_records)


def _strip_human_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in record.items() if str(key) not in HUMAN_ONLY_FIELDS}


def _evidence_id(item: Mapping[str, Any]) -> str:
    return v11._evidence_id(item)


def _traceable(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    return v11._traceable(item, cfg)


def _record_text(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    return v11._record_text(record, fields)


def _matches_any(value: str, patterns: Sequence[str], regex: bool = False) -> bool:
    return v11._matches_any(value, patterns, regex=regex)


def _config_list(config: Mapping[str, Any], key: str) -> list[str]:
    return v11._config_list(config, key)


def _unique_text(values: Sequence[Any]) -> list[str]:
    return v11._unique_text(values)


def _text(value: Any) -> str:
    return v11._text(value)
