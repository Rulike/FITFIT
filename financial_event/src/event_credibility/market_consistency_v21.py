"""Task2 market consistency v2.1 entity-alignment patch.

The v2.1 layer keeps the deterministic v2 market rules but fixes a generic
over-rejection pattern: when the target stock name or code is directly bound to
an explicit price direction expression inside a broader sector/index sentence,
the candidate should be accepted as a target-stock claim.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.event_credibility import market_consistency_v2 as _v2
from src.event_credibility.market_consistency_v2 import *  # noqa: F401,F403


VERSION = "task2_market_consistency_v21"


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", _v2.norm(value)))


def _target_aliases(target_name: str, target_code: str) -> list[str]:
    aliases: list[str] = []
    for value in (target_name, target_code):
        item = _compact(value).strip("()（）")
        if item and item not in aliases:
            aliases.append(item)
    if target_code:
        for suffix in ("SZ", "SH", ".SZ", ".SH"):
            item = f"{target_code}{suffix}"
            if item not in aliases:
                aliases.append(item)
    return aliases


def target_directly_bound_to_price_expression(
    text: str,
    target_name: str,
    target_code: str,
    start: int,
    end: int,
    radius: int = 42,
) -> tuple[bool, str]:
    """Return True when target name/code is the local subject of a price verb.

    The check uses only the local clause immediately before the price expression.
    This prevents a sector-level expression such as "房地产板块收涨0.27%" from
    being rescued merely because the target appears later in the same title,
    while accepting "房地产板块... 万科A(000002.SZ)涨停".
    """

    normalized = unicodedata.normalize("NFKC", _v2.norm(text))
    left = _compact(normalized[max(0, start - radius) : start])
    if not left:
        return False, ""
    clause = re.split(r"[|｜,，;；。！？!?]", left)[-1]
    if not clause:
        return False, ""
    aliases = _target_aliases(target_name, target_code)
    best_tail = ""
    for alias in aliases:
        pos = clause.rfind(alias)
        if pos < 0:
            continue
        tail = clause[pos:]
        if len(tail) <= 28:
            best_tail = tail
            break
    if not best_tail:
        return False, ""
    lower_tail = best_tail.lower()
    if any(term.lower() in lower_tail for term in _v2.EXCLUDED_INSTRUMENT_TERMS if term not in {"A股", "a股"}):
        return False, best_tail
    return True, best_tail


def align_candidate_entity(candidate: dict[str, Any], target_name: str, target_code: str) -> dict[str, Any]:
    text = candidate["text"]
    start = int(candidate["start"])
    end = int(candidate["end"])
    token = _v2.left_token(text, start)
    excluded, excluded_term = _v2.has_excluded_instrument_context(text, start, end)
    token_excluded, token_excluded_term = _v2.token_has_excluded_instrument(token)
    non_price_metric, non_price_metric_term = _v2.has_non_price_metric_context(text, start, end)
    near = _v2.target_near(text, target_name, target_code, start, end)
    token_has_target = bool((target_name and target_name in token) or (target_code and target_code in token))
    token_is_pronoun = token in {"该股", "公司", "股票"}
    token_has_price_subject = any(term in token for term in ("股价", "股票", "该股", "收盘价", "开盘价"))
    direct_target_binding, direct_bound_subject = target_directly_bound_to_price_expression(
        text,
        target_name,
        target_code,
        start,
        end,
    )

    if non_price_metric and not _v2.has_price_subject_context(text, start, end):
        return {
            **candidate,
            "entity_alignment_status": "rejected_non_price_metric",
            "entity_reject_reason": f"non_price_metric_context:{non_price_metric_term}",
            "bound_subject": token,
        }
    if _v2.has_lhb_seat_context(text, target_name, start, end):
        return {
            **candidate,
            "entity_alignment_status": "rejected_lhb_seat_not_target_stock",
            "entity_reject_reason": "non_target_lhb_seat_or_broker_subject",
            "bound_subject": token,
        }
    if direct_target_binding:
        return {
            **candidate,
            "entity_alignment_status": "aligned_target_stock",
            "entity_reject_reason": "",
            "bound_subject": direct_bound_subject or token,
        }
    if token_excluded:
        return {
            **candidate,
            "entity_alignment_status": "rejected_excluded_instrument",
            "entity_reject_reason": f"excluded_instrument_bound_subject:{token_excluded_term}",
            "bound_subject": token,
        }
    if excluded and not token_has_target:
        return {
            **candidate,
            "entity_alignment_status": "rejected_excluded_instrument",
            "entity_reject_reason": f"excluded_instrument_context:{excluded_term}",
            "bound_subject": token,
        }
    if token_has_target or (token_is_pronoun and target_name in text) or (near and not token and target_name in text):
        return {
            **candidate,
            "entity_alignment_status": "aligned_target_stock",
            "entity_reject_reason": "",
            "bound_subject": token,
        }
    if token_has_price_subject and target_name in text:
        return {
            **candidate,
            "entity_alignment_status": "aligned_target_stock",
            "entity_reject_reason": "",
            "bound_subject": token,
        }
    if near and any(term in text for term in ("股价", "股票", "该股")):
        return {
            **candidate,
            "entity_alignment_status": "aligned_target_stock",
            "entity_reject_reason": "",
            "bound_subject": token,
        }
    if token:
        return {
            **candidate,
            "entity_alignment_status": "rejected_other_or_unbound_entity",
            "entity_reject_reason": f"price_expression_bound_to_non_target_or_unverified_subject:{token}",
            "bound_subject": token,
        }
    return {
        **candidate,
        "entity_alignment_status": "rejected_no_target_binding",
        "entity_reject_reason": "target_name_or_code_not_near_price_expression",
        "bound_subject": token,
    }


def explicit_price_candidates(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _v2.align_candidate_entity = align_candidate_entity
    return _v2.explicit_price_candidates(row)


def aggregate_market_consistency(
    row: dict[str, Any],
    index: _v2.MarketIndex,
    cfg: _v2.MarketRuleConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _v2.align_candidate_entity = align_candidate_entity
    cfg = cfg or _v2.MarketRuleConfig(version=VERSION)
    result, accepted, rejected = _v2.aggregate_market_consistency(row, index, cfg)
    result["source_version"] = cfg.version
    for item in accepted:
        item["source_version"] = cfg.version
    for item in rejected:
        item["source_version"] = cfg.version
    return result, accepted, rejected


MarketRuleConfig = _v2.MarketRuleConfig
build_market_index = _v2.build_market_index
parse_market_claim_values = _v2.parse_market_claim_values
