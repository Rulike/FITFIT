"""Market reaction consistency features for Task2 Hybrid-v3.

This module only computes auditable market-consistency features.  It does not
change Task2 labels, train models, tune thresholds, or read locked test gold.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

import pandas as pd


LABELS = {"support", "contradict", "insufficient", "unclear"}
EXPLICIT_PRICE_CLAIM = "explicit_price_claim"
MARKET_SENTIMENT_CLAIM = "market_sentiment_claim"
NON_MARKET_BUSINESS_CLAIM = "non_market_business_claim"


@dataclass(frozen=True)
class MarketConsistencyConfig:
    close_time: str = "15:00"
    direction_unclear_abs_return: float = 0.005
    explicit_contradict_abs_return: float = 0.01
    explicit_adjustment_cap: float = 0.10
    sentiment_adjustment_cap: float = 0.03
    seed: int = 2026
    version: str = "task2_market_consistency_v1"


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[，。、“”‘’；：:,.!?！？（）()\[\]【】《》<>|｜\\-—_·]", "", text)


def parse_datetime(value: Any) -> datetime | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(text).to_pydatetime()
    except Exception:
        return None


def parse_close_time(value: str) -> time:
    hour, minute = [int(part) for part in value.split(":", 1)]
    return time(hour, minute)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def extract_stock_code(record: dict[str, Any]) -> str:
    claim = record.get("claim") if isinstance(record.get("claim"), dict) else {}
    candidates = [
        claim.get("target_code"),
        record.get("target_code"),
        claim.get("target"),
        record.get("target_entity"),
        claim.get("title"),
        claim.get("event_claim"),
    ]
    for value in candidates:
        text = "" if value is None else str(value)
        match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
        if match:
            return match.group(1)
    return ""


def extract_company_name(record: dict[str, Any]) -> str:
    claim = record.get("claim") if isinstance(record.get("claim"), dict) else {}
    target = str(claim.get("target") or record.get("target_entity") or "").strip()
    if "/" in target:
        return target.split("/", 1)[0].strip()
    return target


def claim_text_parts(record: dict[str, Any]) -> tuple[str, str, str, str]:
    claim = record.get("claim") if isinstance(record.get("claim"), dict) else {}
    title = str(claim.get("title") or record.get("title") or "")
    event_claim = str(claim.get("event_claim") or record.get("event_claim") or "")
    event_type = str(record.get("event_type") or claim.get("event_type") or "")
    target = str(claim.get("target") or record.get("target_entity") or "")
    return title, event_claim, event_type, target


def extract_claimed_return(text: str) -> float | None:
    patterns = [
        r"(?:涨幅|上涨|大涨|拉升|走强|涨超|上涨约|上涨近|上涨逾|涨逾|涨近)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"(?:跌幅|下跌|大跌|跳水|走弱|跌超|下跌约|下跌近|下跌逾|跌逾|跌近)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:涨幅|跌幅|上涨|下跌)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)) / 100.0
    return None


def _direction_from_text(text: str) -> str:
    up_terms = ("上涨", "上扬", "走强", "大涨", "拉升", "涨停", "创新高", "新高", "涨超", "涨近", "涨逾", "强势", "冲高")
    down_terms = ("下跌", "走弱", "大跌", "跳水", "跌停", "创新低", "新低", "跌超", "跌近", "跌逾", "承压", "重挫", "杀跌")
    up = any(term in text for term in up_terms)
    down = any(term in text for term in down_terms)
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    if up and down:
        return "mixed"
    return ""


def classify_market_claim(record: dict[str, Any]) -> dict[str, Any]:
    title, event_claim, event_type, target = claim_text_parts(record)
    combined = f"{title} {event_claim} {event_type}"
    normalized = normalize_text(combined)

    explicit_terms = (
        "股价",
        "股票",
        "涨停",
        "跌停",
        "涨幅",
        "跌幅",
        "下跌",
        "上涨",
        "大涨",
        "大跌",
        "走强",
        "走弱",
        "创新高",
        "创新低",
        "新高",
        "新低",
        "拉升",
        "跳水",
        "龙虎榜",
        "异动",
        "二级市场",
        "资金推动",
        "资金流入",
        "资金流出",
    )
    sentiment_terms = (
        "市场不看好",
        "投资者信心下降",
        "投资者信心不足",
        "市场反应冷淡",
        "资本市场承压",
        "市场担忧",
        "看空",
        "抛压",
        "信心下降",
        "市场热捧",
        "市场看好",
        "反应积极",
    )

    direction = _direction_from_text(normalized)
    claimed_return = extract_claimed_return(normalized)
    if any(term in normalized for term in explicit_terms):
        reason = "text_contains_explicit_price_or_trading_terms"
        return {
            "market_claim_type": EXPLICIT_PRICE_CLAIM,
            "claimed_direction": direction or "unspecified",
            "claimed_return": claimed_return,
            "claim_time_scope": "intraday_or_daily" if any(term in normalized for term in ("今日", "当天", "当日", "盘中", "截至")) else "unspecified",
            "market_signal_applicable": direction in {"up", "down", "mixed"} or claimed_return is not None,
            "applicability_reason": reason if direction or claimed_return is not None else "explicit_market_terms_without_direction",
            "claim_text": combined.strip(),
            "target": target,
        }

    if any(term in normalized for term in sentiment_terms):
        direction = direction or ("down" if any(term in normalized for term in ("不看好", "下降", "不足", "冷淡", "承压", "担忧", "看空", "抛压")) else "up")
        return {
            "market_claim_type": MARKET_SENTIMENT_CLAIM,
            "claimed_direction": direction,
            "claimed_return": claimed_return,
            "claim_time_scope": "sentiment_window_unspecified",
            "market_signal_applicable": True,
            "applicability_reason": "text_contains_market_sentiment_terms",
            "claim_text": combined.strip(),
            "target": target,
        }

    return {
        "market_claim_type": NON_MARKET_BUSINESS_CLAIM,
        "claimed_direction": "",
        "claimed_return": None,
        "claim_time_scope": "not_market_claim",
        "market_signal_applicable": False,
        "applicability_reason": "non_market_business_claim_not_applicable",
        "claim_text": combined.strip(),
        "target": target,
    }


def sorted_trading_dates(market_df: pd.DataFrame) -> list[date]:
    return sorted(pd.to_datetime(market_df["trading_date"]).dt.date.unique().tolist())


def align_causal_trading_date(publish_dt: datetime, trading_dates: list[date], close_time: str = "15:00") -> date | None:
    close_clock = parse_close_time(close_time)
    pub_date = publish_dt.date()
    if pub_date in set(trading_dates) and publish_dt.time() >= close_clock:
        return pub_date
    pos = bisect_left(trading_dates, pub_date)
    if pos == 0:
        return None
    return trading_dates[pos - 1]


def align_post_event_trading_date(publish_dt: datetime, trading_dates: list[date]) -> date | None:
    pub_date = publish_dt.date()
    pos = bisect_left(trading_dates, pub_date)
    if pos >= len(trading_dates):
        return None
    return trading_dates[pos]


def next_trading_date(current: date, trading_dates: list[date], offset: int = 1) -> date | None:
    pos = bisect_right(trading_dates, current)
    idx = pos + offset - 1
    if idx < 0 or idx >= len(trading_dates):
        return None
    return trading_dates[idx]


def _row_lookup(market_df: pd.DataFrame) -> dict[tuple[str, date], dict[str, Any]]:
    return {
        (str(row.trading_code).zfill(6), row.trading_date): row._asdict()
        for row in market_df.itertuples(index=False)
    }


def build_market_lookup(market_df: pd.DataFrame) -> dict[str, Any]:
    dates = sorted_trading_dates(market_df)
    return {
        "dates": dates,
        "rows": _row_lookup(market_df),
        "codes": set(str(code).zfill(6) for code in market_df["trading_code"].dropna().astype(str).unique()),
    }


def is_a_share_code(code: str, available_codes: set[str]) -> bool:
    if not re.fullmatch(r"\d{6}", code or ""):
        return False
    if code in available_codes:
        return True
    return code.startswith(("0", "3", "6", "4", "8"))


def _metric_row(lookup: dict[str, Any], code: str, trading_date: date | None) -> dict[str, Any] | None:
    if trading_date is None:
        return None
    return lookup["rows"].get((code, trading_date))


def _row_valid(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("valid_price") and finite_or_none(row.get("daily_return")) is not None)


def _market_reaction_verdict(
    claim_type: str,
    claimed_direction: str,
    adjusted_return: float | None,
    config: MarketConsistencyConfig,
) -> tuple[str, float, float, str]:
    if claim_type == NON_MARKET_BUSINESS_CLAIM:
        return "insufficient", 0.5, 0.0, "market_signal_not_applicable_to_business_claim"
    if adjusted_return is None:
        return "insufficient", 0.5, 0.0, "market_adjusted_return_missing"
    if claimed_direction not in {"up", "down"}:
        return "insufficient", 0.5, 0.0, "claimed_direction_missing_or_mixed"

    abs_ret = abs(adjusted_return)
    if abs_ret < config.direction_unclear_abs_return:
        return "insufficient", 0.5, 0.0, "market_adjusted_direction_too_small"

    actual_direction = "up" if adjusted_return > 0 else "down"
    same_direction = actual_direction == claimed_direction

    if claim_type == EXPLICIT_PRICE_CLAIM:
        if same_direction:
            adjustment = min(config.explicit_adjustment_cap, abs_ret * 2.0)
            return "consistent", clamp(0.5 + adjustment, 0.0, 1.0), adjustment, "explicit_price_direction_matches_market_adjusted_return"
        if abs_ret >= config.explicit_contradict_abs_return:
            adjustment = -min(config.explicit_adjustment_cap, abs_ret * 2.0)
            return "contradictory", clamp(0.5 + adjustment, 0.0, 1.0), adjustment, "explicit_price_direction_opposes_market_adjusted_return"
        return "insufficient", 0.5, 0.0, "opposite_direction_but_below_contradiction_threshold"

    if claim_type == MARKET_SENTIMENT_CLAIM:
        adjustment = min(config.sentiment_adjustment_cap, abs_ret)
        if same_direction:
            return "weak_consistent", clamp(0.5 + adjustment, 0.0, 1.0), adjustment, "market_sentiment_direction_weakly_matches"
        return "weak_contradictory", clamp(0.5 - adjustment, 0.0, 1.0), -adjustment, "market_sentiment_direction_weakly_opposes"

    return "insufficient", 0.5, 0.0, "unknown_claim_type"


def _cum_return(rows: list[dict[str, Any] | None]) -> float | None:
    value = 1.0
    used = 0
    for row in rows:
        ret = finite_or_none(row.get("daily_return") if row else None)
        if ret is None:
            continue
        value *= 1.0 + ret
        used += 1
    if used == 0:
        return None
    return value - 1.0


def compute_market_consistency(
    record: dict[str, Any],
    original_silver_label: str,
    market_lookup: dict[str, Any],
    config: MarketConsistencyConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MarketConsistencyConfig()
    claim = record.get("claim") if isinstance(record.get("claim"), dict) else {}
    sample_id = str(record.get("sample_id") or "")
    publish_dt = parse_datetime(claim.get("publish_time") or record.get("event_time"))
    code = extract_stock_code(record)
    company_name = extract_company_name(record)
    classification = classify_market_claim(record)
    dates: list[date] = market_lookup["dates"]
    codes: set[str] = market_lookup["codes"]

    causal_date = align_causal_trading_date(publish_dt, dates, cfg.close_time) if publish_dt and dates else None
    post_date = align_post_event_trading_date(publish_dt, dates) if publish_dt and dates else None
    r1_date = next_trading_date(post_date, dates, 1) if post_date else None
    r2_date = next_trading_date(post_date, dates, 2) if post_date else None

    causal_row = _metric_row(market_lookup, code, causal_date) if code else None
    post_row = _metric_row(market_lookup, code, post_date) if code else None
    r1_row = _metric_row(market_lookup, code, r1_date) if code else None
    r2_row = _metric_row(market_lookup, code, r2_date) if code else None

    causal_available = _row_valid(causal_row)
    post_available = _row_valid(post_row)
    primary_row = post_row if post_row is not None else causal_row
    matched_date = post_date if post_row is not None else (causal_date if causal_row is not None else None)
    post_event_only = bool(post_available and classification["market_claim_type"] != NON_MARKET_BUSINESS_CLAIM)
    future_relative_to_publication = post_event_only
    forbidden_in_closed_track = post_event_only

    missing_reasons: list[str] = []
    if not code:
        missing_reasons.append("stock_code_missing")
    elif not is_a_share_code(code, codes):
        missing_reasons.append("non_a_share_or_invalid_code")
    elif code not in codes:
        missing_reasons.append("stock_code_not_in_market_cache")
    if publish_dt is None:
        missing_reasons.append("publish_time_missing_or_invalid")
    elif dates and (publish_dt.date() < dates[0] or publish_dt.date() > dates[-1]):
        missing_reasons.append("publish_date_outside_market_coverage")
    if code and code in codes and publish_dt and not post_row and classification["market_claim_type"] != NON_MARKET_BUSINESS_CLAIM:
        missing_reasons.append("matched_market_row_missing")
    if primary_row and not primary_row.get("valid_price"):
        reason = str(primary_row.get("missing_reason") or "invalid_price")
        if reason:
            missing_reasons.append(reason)
    if classification["market_claim_type"] == NON_MARKET_BUSINESS_CLAIM:
        missing_reasons.append("market_signal_not_applicable")

    adjusted_return = finite_or_none(primary_row.get("market_adjusted_return") if primary_row else None)
    verdict, score, adjustment, verdict_reason = _market_reaction_verdict(
        classification["market_claim_type"],
        classification["claimed_direction"],
        adjusted_return if post_available else None,
        cfg,
    )
    if not post_available and classification["market_claim_type"] != NON_MARKET_BUSINESS_CLAIM:
        verdict, score, adjustment, verdict_reason = "insufficient", 0.5, 0.0, "post_event_market_signal_unavailable"

    proposed_change = None
    proposed_reason = ""
    if (
        original_silver_label == "insufficient"
        and classification["market_claim_type"] == EXPLICIT_PRICE_CLAIM
        and verdict in {"consistent", "contradictory"}
        and post_available
    ):
        proposed_change = "insufficient_to_support_candidate" if verdict == "consistent" else "insufficient_to_contradict_candidate"
        proposed_reason = "explicit price claim has direct post-event market audit signal; do not auto-apply without downstream policy"

    result = {
        "sample_id": sample_id,
        "trading_code": code,
        "company_name": company_name,
        "article_publish_time": publish_dt.isoformat(sep=" ") if publish_dt else "",
        "market_claim_type": classification["market_claim_type"],
        "claimed_direction": classification["claimed_direction"],
        "claimed_return": classification["claimed_return"],
        "claim_time_scope": classification["claim_time_scope"],
        "market_signal_applicable": bool(classification["market_signal_applicable"]),
        "applicability_reason": classification["applicability_reason"],
        "matched_trading_date": str(matched_date) if matched_date else "",
        "previous_close": finite_or_none(primary_row.get("previous_close") if primary_row else None),
        "close": finite_or_none(primary_row.get("close") if primary_row else None),
        "daily_return": finite_or_none(primary_row.get("daily_return") if primary_row else None),
        "market_median_return": finite_or_none(primary_row.get("market_median_return") if primary_row else None),
        "market_adjusted_return": finite_or_none(primary_row.get("market_adjusted_return") if primary_row else None),
        "turnover_abnormal_ratio": finite_or_none(primary_row.get("turnover_abnormal_ratio") if primary_row else None),
        "causal_signal_available": bool(causal_available),
        "post_event_signal_available": bool(post_available),
        "market_verdict": verdict,
        "market_consistency_score": clamp(score, 0.0, 1.0),
        "suggested_market_adjustment": clamp(adjustment, -cfg.explicit_adjustment_cap, cfg.explicit_adjustment_cap),
        "evidence_dates": {
            "causal_no_future_date": str(causal_date) if causal_date else "",
            "post_event_date": str(post_date) if post_date else "",
            "next_trading_date": str(r1_date) if r1_date else "",
            "r0_3_dates": [str(d) for d in (post_date, r1_date, r2_date) if d],
            "r0_return": finite_or_none(post_row.get("daily_return") if post_row else None),
            "r1_return": finite_or_none(r1_row.get("daily_return") if r1_row else None),
            "r0_3_return": _cum_return([post_row, r1_row, r2_row]),
            "r0_market_adjusted_return": finite_or_none(post_row.get("market_adjusted_return") if post_row else None),
        },
        "post_event_only": post_event_only,
        "future_relative_to_publication": future_relative_to_publication,
        "forbidden_in_closed_track": forbidden_in_closed_track,
        "missing_reason": ";".join(dict.fromkeys(reason for reason in missing_reasons if reason)),
        "original_silver_label": original_silver_label,
        "proposed_label_change": proposed_change,
        "proposed_label_change_reason": proposed_reason,
        "verdict_reason": verdict_reason,
        "claim_text": classification["claim_text"],
        "source_version": cfg.version,
    }
    return sanitize_json_record(result)


def sanitize_json_record(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json_record(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_record(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json_record(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value
