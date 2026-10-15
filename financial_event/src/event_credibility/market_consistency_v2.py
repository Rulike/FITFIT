"""Deterministic market-claim consistency layer for Task2 Hybrid-v3.

The v2 layer fixes market claim parsing, entity alignment, and closed-track
time alignment. It is feature/audit generation only: no training, no threshold
tuning, no label rewriting, and no locked-gold access.
"""

from __future__ import annotations

import math
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable

import pandas as pd


EXPLICIT_PRICE_CLAIM = "explicit_price_claim"
MARKET_SENTIMENT_CLAIM = "market_sentiment_claim"
FINANCIAL_PERFORMANCE_CLAIM = "financial_performance_claim"
NON_MARKET_BUSINESS_CLAIM = "non_market_business_claim"

CONSISTENT = "consistent"
CONTRADICTORY = "contradictory"
INSUFFICIENT = "insufficient"
NOT_APPLICABLE = "not_applicable"

MARKET_CLOSE = time(15, 0, 0)

PRICE_DIRECTION_TERMS = (
    "涨停",
    "跌停",
    "收涨",
    "收跌",
    "上涨",
    "下跌",
    "上扬",
    "下挫",
    "大涨",
    "大跌",
    "走强",
    "走弱",
    "涨幅",
    "跌幅",
    "涨超",
    "跌超",
    "涨逾",
    "跌逾",
    "涨近",
    "跌近",
    "涨",
    "跌",
)
PRICE_CONTEXT_TERMS = (
    "股价",
    "股票",
    "该股",
    "a股",
    "A股",
    "收盘价",
    "报",
    "元/股",
    "每股",
    "成交额",
    "换手率",
    "总市值",
    "盘中",
    "收盘",
    "开盘",
    "尾盘",
)
NON_PRICE_METRIC_TERMS = (
    "成交额",
    "成交量",
    "主力资金",
    "净流入",
    "净流出",
    "换手率",
    "振幅",
    "市值",
    "总市值",
    "流通市值",
    "融资",
    "融券",
    "余额",
    "持股",
    "持仓",
    "股东户数",
    "买入",
    "卖出",
)
PRICE_SUBJECT_TERMS = (
    "股价",
    "股票",
    "该股",
    "收盘价",
    "开盘价",
    "报",
    "元/股",
    "每股",
    "收盘",
    "开盘",
    "盘中",
    "尾盘",
)
MARKET_SENTIMENT_TERMS = (
    "市场表现低迷",
    "市场低迷",
    "投资者信心不足",
    "投资者信心下降",
    "投资者信心增强",
    "投资者信心提振",
    "市场反应积极",
    "市场表现积极",
    "市场反应冷淡",
    "市场不看好",
    "市场担忧",
    "市场看好",
    "资本市场承压",
)
FINANCIAL_TERMS = (
    "盈利能力",
    "现金流",
    "营收",
    "营业收入",
    "净利润",
    "归母净利润",
    "扣非净利润",
    "偿债压力",
    "流动性压力",
    "资产负债率",
    "负债率",
    "毛利率",
    "roe",
    "roa",
    "财报",
    "年报",
    "季报",
    "一季报",
    "半年报",
    "三季报",
    "业绩预告",
    "业绩快报",
    "业绩预亏",
)
EXCLUDED_INSTRUMENT_TERMS = (
    "板块",
    "概念",
    "指数",
    "行业",
    "etf",
    "ETF",
    "基金",
    "净值",
    "混合发起",
    "联接",
    "lof",
    "LOF",
    "港股",
    "H股",
    "美股",
    "债券",
    "转债",
    "期货",
)
LHB_SEAT_TERMS = ("席位", "营业部", "分公司", "证券营业部", "龙虎榜")
INTRADAY_TERMS = ("盘中", "截至发稿", "截至", "午评", "早盘", "开盘", "尾盘", "临近午盘", "快讯")
FULL_DAY_TERMS = ("收盘", "收涨", "收跌", "收盘价")
FORECAST_TERMS = ("预计", "预期", "有望", "或将", "将", "拟", "计划", "后市", "明日", "未来", "目标价")

VALUE_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
NUMERIC_PRICE_PATTERN = re.compile(
    r"(?P<verb>涨停|跌停|收涨|收跌|上涨|下跌|上扬|下挫|大涨|大跌|涨幅|跌幅|涨超|跌超|涨逾|跌逾|涨近|跌近|涨|跌)"
    r"\s*(?:约|近|超|逾|超过|达到|达|为)?\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>个百分点|个点|pp|PP|%)"
)
POSTFIX_PRICE_PATTERN = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>个百分点|个点|pp|PP|%)\s*"
    r"(?:的)?\s*"
    r"(?P<verb>涨幅|跌幅|上涨|下跌|上扬|下挫|收涨|收跌)"
)
LIMIT_PRICE_PATTERN = re.compile(r"(?P<verb>涨停|跌停|收涨|收跌|上涨|下跌|上扬|下挫|大涨|大跌|走强|走弱)")
DATE_PATTERNS = (
    re.compile(r"(?P<year>20\d{2})[./-](?P<month>\d{1,2})[./-](?P<day>\d{1,2})"),
    re.compile(r"(?P<year>20\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"),
    re.compile(r"(?P<month>\d{1,2})月(?P<day>\d{1,2})日"),
)
RANGE_WINDOW_PATTERN = re.compile(r"(?:连续|近)(?P<num>\d+|[一二两三四五六七八九十]+)(?:个)?(?:交易日|日|天)")
STREAK_WINDOW_PATTERN = re.compile(r"(?P<num>\d+|[一二两三四五六七八九十]+)连[涨跌]")
BROAD_RANGE_PATTERN = re.compile(r"年内|今年|本年|本月|月内|本周|周内|近半年|近一月|近一个月|历史")
CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


@dataclass(frozen=True)
class MarketRuleConfig:
    numeric_relative_error_threshold: float = 0.20
    close_time: time = MARKET_CLOSE
    limit_up_down_floor: float = 0.045
    max_range_window: int = 60
    version: str = "task2_market_consistency_v2"


@dataclass(frozen=True)
class MarketIndex:
    rows: dict[tuple[str, date], dict[str, Any]]
    dates_by_code: dict[str, list[date]]
    global_dates: list[date]
    global_date_set: set[date]
    codes: set[str]
    rows_count: int
    date_min: str
    date_max: str


def norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    """Normalize text for matching while preserving decimal points and %."""
    text = unicodedata.normalize("NFKC", norm(value))
    text = re.sub(r"\s+", "", text)
    return text


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def parse_datetime(value: Any) -> datetime | None:
    text = norm(value)
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(text).to_pydatetime()
    except Exception:
        return None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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


def extract_target_code(row: dict[str, Any]) -> str:
    claim = row.get("claim") if isinstance(row.get("claim"), dict) else {}
    for value in (
        row.get("target_code"),
        claim.get("target"),
        row.get("target_entity"),
        claim.get("title"),
        claim.get("event_claim"),
    ):
        match = re.search(r"(?<!\d)(\d{6})(?!\d)", norm(value))
        if match:
            return match.group(1)
    return ""


def extract_target_name(row: dict[str, Any]) -> str:
    claim = row.get("claim") if isinstance(row.get("claim"), dict) else {}
    target = norm(row.get("target_entity") or claim.get("target"))
    if "/" in target:
        target = target.split("/", 1)[0]
    return target.strip()


def claim_publish_time(row: dict[str, Any]) -> str:
    claim = row.get("claim") if isinstance(row.get("claim"), dict) else {}
    return norm(claim.get("publish_time") or row.get("event_time"))


def claim_title(row: dict[str, Any]) -> str:
    claim = row.get("claim") if isinstance(row.get("claim"), dict) else {}
    return norm(claim.get("title"))


def claim_event_text(row: dict[str, Any]) -> str:
    claim = row.get("claim") if isinstance(row.get("claim"), dict) else {}
    return " ".join(
        part
        for part in (
            norm(claim.get("event_claim")),
            norm(row.get("event_type")),
            norm(claim.get("target") or row.get("target_entity")),
        )
        if part
    )


def build_market_index(market_df: pd.DataFrame) -> MarketIndex:
    df = market_df.copy()
    df["trading_code"] = df["trading_code"].astype(str).str.zfill(6)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df.sort_values(["trading_code", "trading_date"], inplace=True, kind="mergesort")
    rows: dict[tuple[str, date], dict[str, Any]] = {}
    dates_by_code: dict[str, list[date]] = defaultdict(list)
    for row in df.itertuples(index=False):
        payload = row._asdict()
        code = str(payload["trading_code"]).zfill(6)
        trading_day = payload["trading_date"]
        rows[(code, trading_day)] = payload
        dates_by_code[code].append(trading_day)
    global_dates = sorted(set(df["trading_date"].tolist()))
    return MarketIndex(
        rows=rows,
        dates_by_code={code: sorted(set(days)) for code, days in dates_by_code.items()},
        global_dates=global_dates,
        global_date_set=set(global_dates),
        codes=set(dates_by_code),
        rows_count=int(len(df)),
        date_min=str(global_dates[0]) if global_dates else "",
        date_max=str(global_dates[-1]) if global_dates else "",
    )


def market_row(index: MarketIndex, code: str, trading_day: date | None) -> dict[str, Any] | None:
    if not code or trading_day is None:
        return None
    return index.rows.get((code, trading_day))


def row_valid(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("valid_price") and safe_float(row.get("daily_return")) is not None)


def previous_global_trading_date(index: MarketIndex, current: date) -> date | None:
    pos = bisect_left(index.global_dates, current)
    if pos == 0:
        return None
    return index.global_dates[pos - 1]


def next_global_trading_date(index: MarketIndex, current: date, offset: int = 1) -> date | None:
    pos = bisect_right(index.global_dates, current)
    target = pos + offset - 1
    if target < 0 or target >= len(index.global_dates):
        return None
    return index.global_dates[target]


def latest_completed_trading_date(publish_dt: datetime | None, index: MarketIndex, cfg: MarketRuleConfig) -> date | None:
    if publish_dt is None or not index.global_dates:
        return None
    pub_day = publish_dt.date()
    if pub_day in index.global_date_set and publish_dt.time() >= cfg.close_time:
        return pub_day
    pos = bisect_left(index.global_dates, pub_day)
    if pos == 0:
        return None
    return index.global_dates[pos - 1]


def first_post_event_trading_date(publish_dt: datetime | None, index: MarketIndex) -> date | None:
    if publish_dt is None or not index.global_dates:
        return None
    pub_day = publish_dt.date()
    if pub_day in index.global_date_set:
        return pub_day
    pos = bisect_right(index.global_dates, pub_day)
    if pos >= len(index.global_dates):
        return None
    return index.global_dates[pos]


def previous_code_dates(index: MarketIndex, code: str, end_day: date, window: int) -> list[date]:
    days = index.dates_by_code.get(code, [])
    pos = bisect_right(days, end_day)
    return days[max(0, pos - window) : pos]


def split_fragments(row: dict[str, Any]) -> list[dict[str, Any]]:
    title = claim_title(row)
    publish = claim_publish_time(row)
    fragments: list[dict[str, Any]] = []
    if title:
        parts = re.split(r"\s*\|\|\s*|[。；;]\s*", title)
        for idx, part in enumerate(parts):
            if norm(part):
                fragments.append(
                    {
                        "source_kind": "claim_title",
                        "source_index": idx,
                        "text": norm(part),
                        "publish_time": publish,
                    }
                )
    return fragments


def direction_from_verb(verb: str) -> str:
    if any(term in verb for term in ("跌", "下挫", "走弱")):
        return "down"
    if any(term in verb for term in ("涨", "上涨", "上扬", "走强")):
        return "up"
    return ""


def value_type_from_unit(unit: str) -> str:
    if unit in {"%", "pp", "PP", "个百分点", "个点"}:
        return "percentage_point" if unit in {"pp", "PP", "个百分点", "个点"} else "percent"
    if unit in {"元", "元/股", "亿元", "万元", "万", "亿"}:
        return "money_or_price"
    if unit in {"股", "万股", "手"}:
        return "quantity"
    return "unknown"


def parse_market_claim_values(text: str) -> list[dict[str, Any]]:
    normalized = unicodedata.normalize("NFKC", text)
    values: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for pattern in (NUMERIC_PRICE_PATTERN, POSTFIX_PRICE_PATTERN):
        for match in pattern.finditer(normalized):
            if (match.start(), match.end()) in seen:
                continue
            seen.add((match.start(), match.end()))
            raw_value = float(match.group("value"))
            unit = match.group("unit")
            value_type = value_type_from_unit(unit)
            normalized_value = raw_value / 100.0 if value_type in {"percent", "percentage_point"} else raw_value
            direction = direction_from_verb(match.group("verb"))
            if raw_value < 0:
                direction = "down" if direction == "up" else direction
            values.append(
                {
                    "raw_text": match.group(0),
                    "raw_value": raw_value,
                    "normalized_value": normalized_value,
                    "unit": unit,
                    "value_type": value_type,
                    "direction": direction,
                    "start": match.start(),
                    "end": match.end(),
                    "parse_validation_error": value_type in {"percent", "percentage_point"}
                    and not math.isclose(normalized_value, raw_value / 100.0, rel_tol=0.0, abs_tol=1e-12),
                }
            )
    for match in LIMIT_PRICE_PATTERN.finditer(normalized):
        if any(start <= match.start() < end for start, end in seen):
            continue
        verb = match.group("verb")
        if verb in {"上涨", "下跌", "上扬", "下挫", "大涨", "大跌", "走强", "走弱"}:
            context = normalized[max(0, match.start() - 12) : min(len(normalized), match.end() + 12)]
            if not any(term.lower() in context.lower() for term in PRICE_CONTEXT_TERMS):
                continue
        values.append(
            {
                "raw_text": verb,
                "raw_value": None,
                "normalized_value": None,
                "unit": "",
                "value_type": "direction",
                "direction": direction_from_verb(verb),
                "start": match.start(),
                "end": match.end(),
                "parse_validation_error": False,
            }
        )
    values.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return values


def left_token(text: str, start: int) -> str:
    left = text[:start]
    match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]+)$", left)
    token = match.group(1) if match else ""
    return token.strip("的于在为和与及、")


def code_near(text: str, code: str, start: int, end: int, radius: int = 24) -> bool:
    if not code:
        return False
    context = text[max(0, start - radius) : min(len(text), end + radius)]
    return code in context


def target_near(text: str, target_name: str, code: str, start: int, end: int, radius: int = 24) -> bool:
    context = text[max(0, start - radius) : min(len(text), end + radius)]
    return bool((target_name and target_name in context) or code_near(text, code, start, end, radius))


def has_excluded_instrument_context(text: str, start: int, end: int) -> tuple[bool, str]:
    context = text[max(0, start - 28) : min(len(text), end + 28)]
    lower = context.lower()
    for term in EXCLUDED_INSTRUMENT_TERMS:
        if term.lower() in lower:
            return True, term
    return False, ""


def token_has_excluded_instrument(token: str) -> tuple[bool, str]:
    lower = token.lower()
    for term in EXCLUDED_INSTRUMENT_TERMS:
        if term.lower() in lower:
            return True, term
    return False, ""


def has_non_price_metric_context(text: str, start: int, end: int) -> tuple[bool, str]:
    context = text[max(0, start - 16) : min(len(text), end + 16)]
    for term in NON_PRICE_METRIC_TERMS:
        term_pos = context.find(term)
        if term_pos < 0:
            continue
        absolute_pos = max(0, start - 16) + term_pos
        if absolute_pos <= end:
            return True, term
    return False, ""


def has_price_subject_context(text: str, start: int, end: int) -> bool:
    context = text[max(0, start - 20) : min(len(text), end + 12)]
    return any(term in context for term in PRICE_SUBJECT_TERMS)


def has_lhb_seat_context(text: str, target_name: str, start: int, end: int) -> bool:
    context = text[max(0, start - 12) : min(len(text), end + 42)]
    return any(term in context for term in LHB_SEAT_TERMS) and bool(target_name and target_name in context and target_name not in left_token(text, start))


def align_candidate_entity(candidate: dict[str, Any], target_name: str, target_code: str) -> dict[str, Any]:
    text = candidate["text"]
    start = int(candidate["start"])
    end = int(candidate["end"])
    token = left_token(text, start)
    excluded, excluded_term = has_excluded_instrument_context(text, start, end)
    token_excluded, token_excluded_term = token_has_excluded_instrument(token)
    non_price_metric, non_price_metric_term = has_non_price_metric_context(text, start, end)
    near = target_near(text, target_name, target_code, start, end)
    token_has_target = bool((target_name and target_name in token) or (target_code and target_code in token))
    token_is_pronoun = token in {"该股", "公司", "股票"}
    token_has_price_subject = any(term in token for term in ("股价", "股票", "该股", "收盘价", "开盘价"))

    if token_excluded:
        return {
            **candidate,
            "entity_alignment_status": "rejected_excluded_instrument",
            "entity_reject_reason": f"excluded_instrument_bound_subject:{token_excluded_term}",
            "bound_subject": token,
        }
    if non_price_metric and not has_price_subject_context(text, start, end):
        return {
            **candidate,
            "entity_alignment_status": "rejected_non_price_metric",
            "entity_reject_reason": f"non_price_metric_context:{non_price_metric_term}",
            "bound_subject": token,
        }
    if excluded and not token_has_target:
        return {
            **candidate,
            "entity_alignment_status": "rejected_excluded_instrument",
            "entity_reject_reason": f"excluded_instrument_context:{excluded_term}",
            "bound_subject": token,
        }
    if has_lhb_seat_context(text, target_name, start, end):
        return {
            **candidate,
            "entity_alignment_status": "rejected_lhb_seat_not_target_stock",
            "entity_reject_reason": "non_target_lhb_seat_or_broker_subject",
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
    target_name = extract_target_name(row)
    target_code = extract_target_code(row)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for fragment in split_fragments(row):
        text = unicodedata.normalize("NFKC", fragment["text"])
        values = parse_market_claim_values(text)
        if not values:
            continue
        for value in values:
            candidate = {
                "sample_id": norm(row.get("sample_id")),
                "target_name": target_name,
                "target_code": target_code,
                "source_kind": fragment["source_kind"],
                "source_index": fragment["source_index"],
                "claim_text": fragment["text"],
                "text": text,
                "claim_publish_time": fragment["publish_time"],
                **value,
            }
            aligned = align_candidate_entity(candidate, target_name, target_code)
            if aligned["entity_alignment_status"] == "aligned_target_stock":
                accepted.append(aligned)
            else:
                rejected.append(aligned)
    if len(accepted) > 1:
        date_direction_pairs = {
            (
                norm(extract_reference_date(item["text"], parse_datetime(item["claim_publish_time"]))),
                item.get("direction"),
                item.get("normalized_value"),
            )
            for item in accepted
        }
        if len(date_direction_pairs) > 1:
            for item in accepted:
                rejected.append(
                    {
                        **item,
                        "entity_alignment_status": "rejected_ambiguous_multiple_target_price_claims",
                        "entity_reject_reason": "multiple_target_price_claims_with_different_date_direction_or_value",
                    }
                )
            accepted = []
    return accepted, rejected


def classify_non_explicit(row: dict[str, Any]) -> str:
    text = normalize_text(" ".join(part for part in (claim_title(row), claim_event_text(row)) if part))
    lower = text.lower()
    if any(term.lower() in lower for term in FINANCIAL_TERMS):
        return FINANCIAL_PERFORMANCE_CLAIM
    if any(term.lower() in lower for term in MARKET_SENTIMENT_TERMS):
        return MARKET_SENTIMENT_CLAIM
    return NON_MARKET_BUSINESS_CLAIM


def extract_reference_date(text: str, publish_dt: datetime | None) -> date | None:
    normalized = unicodedata.normalize("NFKC", text)
    for pattern in DATE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        year = int(match.groupdict().get("year") or (publish_dt.year if publish_dt else 0))
        month = int(match.group("month"))
        day = int(match.group("day"))
        if not year:
            return None
        try:
            return date(year, month, day)
        except ValueError:
            return None
    if publish_dt and any(term in normalized for term in ("昨日", "昨天")):
        return publish_dt.date() - timedelta(days=1)
    if publish_dt and any(term in normalized for term in ("今日", "今天", "当日", "当天")):
        return publish_dt.date()
    return publish_dt.date() if publish_dt else None


def parse_small_int(text: str) -> int | None:
    normalized = normalize_text(text)
    if normalized.isdigit():
        value = int(normalized)
        return value if value > 0 else None
    if normalized == "十":
        return 10
    if "十" in normalized:
        left, right = normalized.split("十", 1)
        tens = CHINESE_DIGITS.get(left, 1 if left == "" else 0)
        ones = CHINESE_DIGITS.get(right, 0) if right else 0
        value = tens * 10 + ones
        return value if value > 0 else None
    if len(normalized) == 1:
        return CHINESE_DIGITS.get(normalized)
    return None


def local_scope_context(text: str, start: int | None = None, end: int | None = None) -> str:
    normalized = normalize_text(text)
    if start is None or end is None:
        return normalized
    return normalized[max(0, start - 18) : min(len(normalized), end + 10)]


def extract_range_window(text: str, start: int | None = None, end: int | None = None) -> int | None:
    context = local_scope_context(text, start, end)
    for pattern in (RANGE_WINDOW_PATTERN, STREAK_WINDOW_PATTERN):
        match = pattern.search(context)
        if match:
            return parse_small_int(match.group("num"))
    return None


def is_consecutive_range_claim(text: str, start: int | None = None, end: int | None = None) -> bool:
    context = local_scope_context(text, start, end)
    return bool(re.search(r"连续|连[涨跌]", context))


def time_scope(text: str, start: int | None = None, end: int | None = None) -> str:
    context = local_scope_context(text, start, end)
    if any(term in context for term in INTRADAY_TERMS) and not any(term in context for term in FULL_DAY_TERMS):
        return "intraday"
    if RANGE_WINDOW_PATTERN.search(context) or STREAK_WINDOW_PATTERN.search(context) or BROAD_RANGE_PATTERN.search(context):
        return "historical_range"
    if any(term in context for term in FULL_DAY_TERMS):
        return "close_or_full_day"
    return "daily_unspecified"


def is_forecast_or_future_claim(text: str) -> bool:
    normalized = normalize_text(text)
    return any(term in normalized for term in FORECAST_TERMS)


def align_claim_time(
    candidate: dict[str, Any],
    index: MarketIndex,
    cfg: MarketRuleConfig,
) -> dict[str, Any]:
    publish_dt = parse_datetime(candidate.get("claim_publish_time"))
    cutoff = latest_completed_trading_date(publish_dt, index, cfg)
    ref_date = extract_reference_date(candidate["text"], publish_dt)
    start = int(candidate.get("start") or 0)
    end = int(candidate.get("end") or start)
    scope = time_scope(candidate["text"], start, end)
    range_window = extract_range_window(candidate["text"], start, end)
    status = "aligned_closed_track"
    reason = ""
    post_event_only = False
    if publish_dt is None:
        status = "date_alignment_error"
        reason = "claim_publish_time_missing_or_invalid"
    elif cutoff is None:
        status = "date_alignment_error"
        reason = "no_completed_trading_day_before_claim_time"
    elif ref_date is None:
        status = "date_alignment_error"
        reason = "market_reference_date_missing"
    elif ref_date not in index.global_date_set:
        status = "date_alignment_error"
        reason = "market_reference_date_not_trading_day"
    elif is_forecast_or_future_claim(candidate["text"]):
        status = "post_event_only"
        reason = "forecast_or_future_price_claim_not_closed_track_verifiable"
        post_event_only = True
    elif scope == "intraday":
        status = "post_event_only"
        reason = "intraday_claim_cannot_be_verified_with_daily_close_in_closed_track"
        post_event_only = True
    elif ref_date > cutoff:
        status = "post_event_only"
        reason = "referenced_market_date_after_causal_cutoff"
        post_event_only = True
    return {
        **candidate,
        "claim_time": publish_dt.isoformat(sep=" ") if publish_dt else "",
        "claim_time_scope": scope,
        "market_range_window": range_window,
        "market_range_is_consecutive": is_consecutive_range_claim(candidate["text"], start, end),
        "market_reference_date": ref_date.isoformat() if ref_date else "",
        "causal_cutoff": cutoff.isoformat() if cutoff else "",
        "time_alignment_status": status,
        "time_reject_reason": reason,
        "post_event_only": post_event_only,
        "future_data_used": False,
    }


def actual_direction(value: float | None) -> str:
    if value is None:
        return ""
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def claimed_direction(candidate: dict[str, Any]) -> str:
    return norm(candidate.get("direction") or "")


def claimed_price_change(candidate: dict[str, Any]) -> float | None:
    value_type = candidate.get("value_type")
    if value_type in {"percent", "percentage_point"}:
        value = safe_float(candidate.get("normalized_value"))
        if value is None:
            return None
        direction = claimed_direction(candidate)
        if direction == "down":
            return -abs(value)
        if direction == "up":
            return abs(value)
        return value
    return None


def is_limit_claim(candidate: dict[str, Any]) -> bool:
    raw = norm(candidate.get("raw_text"))
    return "涨停" in raw or "跌停" in raw


def evaluate_range_candidate(candidate: dict[str, Any], index: MarketIndex, cfg: MarketRuleConfig) -> dict[str, Any]:
    code = norm(candidate.get("target_code"))
    ref = norm(candidate.get("market_reference_date"))
    ref_date = datetime.strptime(ref, "%Y-%m-%d").date() if ref else None
    window = candidate.get("market_range_window")
    if not isinstance(window, int) or window <= 1 or window > cfg.max_range_window:
        return {
            **candidate,
            "actual_price_change": None,
            "price_direction": "",
            "market_consistency_verdict": INSUFFICIENT,
            "market_consistency_score": 0.5,
            "market_evidence_strength": "none",
            "market_evidence_date": ref,
            "market_evidence_ids": [],
            "causal_market_available": False,
            "market_reason": "unsupported_or_missing_historical_range_window",
            "closed_track_verifiable": False,
        }
    days = previous_code_dates(index, code, ref_date, window) if ref_date is not None else []
    rows = [market_row(index, code, day) for day in days]
    if len(days) < window or any(not row_valid(item) for item in rows):
        return {
            **candidate,
            "actual_price_change": None,
            "price_direction": "",
            "market_consistency_verdict": INSUFFICIENT,
            "market_consistency_score": 0.5,
            "market_evidence_strength": "none",
            "market_evidence_date": ref,
            "market_evidence_ids": [],
            "causal_market_available": False,
            "market_reason": "historical_range_market_window_missing_or_invalid",
            "closed_track_verifiable": False,
        }

    first_previous_close = safe_float(rows[0].get("previous_close") if rows[0] else None)
    last_close = safe_float(rows[-1].get("close") if rows[-1] else None)
    if first_previous_close is None or first_previous_close <= 0 or last_close is None:
        return {
            **candidate,
            "actual_price_change": None,
            "price_direction": "",
            "market_consistency_verdict": INSUFFICIENT,
            "market_consistency_score": 0.5,
            "market_evidence_strength": "none",
            "market_evidence_date": ref,
            "market_evidence_ids": [],
            "causal_market_available": False,
            "market_reason": "historical_range_baseline_close_missing_or_invalid",
            "closed_track_verifiable": False,
        }

    actual = last_close / first_previous_close - 1.0
    actual_dir = actual_direction(actual)
    expected_dir = claimed_direction(candidate)
    expected_change = claimed_price_change(candidate)
    if expected_dir not in {"up", "down"} or actual_dir not in {"up", "down"}:
        verdict = INSUFFICIENT
        reason = "historical_range_direction_missing_or_actual_flat"
    elif candidate.get("market_range_is_consecutive"):
        daily_dirs = [actual_direction(safe_float(item.get("daily_return") if item else None)) for item in rows]
        direction_matches = all(day_dir == expected_dir for day_dir in daily_dirs)
        limit_matches = (
            all(abs(safe_float(item.get("daily_return") if item else None) or 0.0) >= cfg.limit_up_down_floor for item in rows)
            if is_limit_claim(candidate)
            else True
        )
        verdict = CONSISTENT if direction_matches and limit_matches else CONTRADICTORY
        reason = "consecutive_range_direction_matches" if verdict == CONSISTENT else "consecutive_range_direction_or_limit_mismatch"
    elif expected_dir != actual_dir:
        verdict = CONTRADICTORY
        reason = "historical_range_direction_opposes_range_return"
    elif expected_change is not None:
        denominator = abs(expected_change)
        relative_error = abs(abs(actual) - denominator) / denominator if denominator > 0 else 0.0
        if relative_error > cfg.numeric_relative_error_threshold:
            verdict = CONTRADICTORY
            reason = f"historical_range_relative_error_{relative_error:.6f}_over_20pct"
        else:
            verdict = CONSISTENT
            reason = f"historical_range_relative_error_{relative_error:.6f}_within_20pct"
    else:
        verdict = CONSISTENT
        reason = "historical_range_direction_matches_range_return"

    return {
        **candidate,
        "actual_price_change": actual,
        "price_direction": actual_dir,
        "market_consistency_verdict": verdict,
        "market_consistency_score": 0.8 if verdict == CONSISTENT else (0.2 if verdict == CONTRADICTORY else 0.5),
        "market_evidence_strength": "strong" if verdict in {CONSISTENT, CONTRADICTORY} else "none",
        "market_evidence_date": ref,
        "market_evidence_ids": [
            evidence_id
            for day in days
            for evidence_id in evidence_ids(code, day, ("close", "previous_close", "daily_return"))
        ],
        "causal_market_available": True,
        "market_reason": reason,
        "closed_track_verifiable": verdict in {CONSISTENT, CONTRADICTORY},
    }


def evaluate_candidate(candidate: dict[str, Any], index: MarketIndex, cfg: MarketRuleConfig) -> dict[str, Any]:
    code = norm(candidate.get("target_code"))
    ref = norm(candidate.get("market_reference_date"))
    ref_date = datetime.strptime(ref, "%Y-%m-%d").date() if ref else None
    row = market_row(index, code, ref_date)
    if candidate.get("time_alignment_status") != "aligned_closed_track":
        return {
            **candidate,
            "actual_price_change": None,
            "price_direction": "",
            "market_consistency_verdict": INSUFFICIENT,
            "market_consistency_score": 0.5,
            "market_evidence_strength": "none",
            "market_evidence_date": "",
            "market_evidence_ids": [],
            "causal_market_available": False,
            "market_reason": candidate.get("time_reject_reason") or candidate.get("time_alignment_status"),
            "closed_track_verifiable": False,
        }
    if candidate.get("claim_time_scope") == "historical_range":
        return evaluate_range_candidate(candidate, index, cfg)
    if not row_valid(row):
        return {
            **candidate,
            "actual_price_change": None,
            "price_direction": "",
            "market_consistency_verdict": INSUFFICIENT,
            "market_consistency_score": 0.5,
            "market_evidence_strength": "none",
            "market_evidence_date": ref,
            "market_evidence_ids": [],
            "causal_market_available": False,
            "market_reason": norm((row or {}).get("missing_reason")) or "market_row_missing_or_invalid",
            "closed_track_verifiable": False,
        }
    actual = safe_float(row.get("daily_return"))
    actual_dir = actual_direction(actual)
    expected_dir = claimed_direction(candidate)
    expected_change = claimed_price_change(candidate)
    verdict = INSUFFICIENT
    reason = "claimed_direction_missing_or_actual_flat"
    if expected_dir in {"up", "down"} and actual_dir in {"up", "down"}:
        if expected_dir != actual_dir:
            verdict = CONTRADICTORY
            reason = "price_direction_opposes_daily_close_return"
        elif expected_change is not None:
            denominator = abs(expected_change)
            relative_error = abs(abs(actual or 0.0) - denominator) / denominator if denominator > 0 else 0.0
            if relative_error > cfg.numeric_relative_error_threshold:
                verdict = CONTRADICTORY
                reason = f"price_change_relative_error_{relative_error:.6f}_over_20pct"
            else:
                verdict = CONSISTENT
                reason = f"price_change_relative_error_{relative_error:.6f}_within_20pct"
        elif is_limit_claim(candidate):
            if abs(actual or 0.0) >= cfg.limit_up_down_floor:
                verdict = CONSISTENT
                reason = "limit_up_down_direction_matches_and_return_exceeds_floor"
            else:
                verdict = CONTRADICTORY
                reason = "limit_up_down_claim_but_daily_return_below_floor"
        else:
            verdict = CONSISTENT
            reason = "price_direction_matches_daily_close_return"
    return {
        **candidate,
        "actual_price_change": actual,
        "price_direction": actual_dir,
        "market_consistency_verdict": verdict,
        "market_consistency_score": 0.8 if verdict == CONSISTENT else (0.2 if verdict == CONTRADICTORY else 0.5),
        "market_evidence_strength": "strong" if verdict in {CONSISTENT, CONTRADICTORY} else "none",
        "market_evidence_date": ref,
        "market_evidence_ids": evidence_ids(code, ref_date, ("close", "previous_close", "daily_return")),
        "causal_market_available": True,
        "market_reason": reason,
        "closed_track_verifiable": verdict in {CONSISTENT, CONTRADICTORY},
    }


def evidence_ids(code: str, trading_day: date | None, fields: Iterable[str]) -> list[str]:
    if not code or trading_day is None:
        return []
    return [f"RESSET_DRESSTK_2026:{code}:{trading_day.isoformat()}:{field}" for field in fields]


def post_event_audit(row: dict[str, Any], index: MarketIndex) -> dict[str, Any]:
    code = extract_target_code(row)
    publish_dt = parse_datetime(claim_publish_time(row))
    first = first_post_event_trading_date(publish_dt, index)
    dates: list[date] = []
    if first is not None:
        dates.append(first)
        current = first
        for _ in range(4):
            nxt = next_global_trading_date(index, current)
            if nxt is None:
                break
            dates.append(nxt)
            current = nxt
    rows = [market_row(index, code, day) for day in dates]
    return {
        "post_event_market_available": any(row_valid(item) for item in rows),
        "post_event_only": bool(dates),
        "post_event_audit_only": True,
        "forbidden_in_closed_track": True,
        "used_in_closed_track": False,
        "post_event_dates": [item.isoformat() for item in dates],
        "post_event_returns": [safe_float(item.get("daily_return") if item else None) for item in rows],
    }


def insufficient_row(
    row: dict[str, Any],
    claim_type: str,
    reason: str,
    post_audit: dict[str, Any],
    accepted: list[dict[str, Any]] | None = None,
    rejected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target_code = extract_target_code(row)
    target_name = extract_target_name(row)
    return {
        "sample_id": norm(row.get("sample_id")),
        "target_code": target_code,
        "target_name": target_name,
        "market_claim_type": claim_type,
        "claim_text": claim_title(row),
        "claim_time": claim_publish_time(row),
        "claim_time_scope": "",
        "market_range_window": None,
        "market_range_is_consecutive": False,
        "claimed_price_direction": "",
        "claimed_price_change": None,
        "actual_price_change": None,
        "price_direction": "",
        "market_consistency_verdict": INSUFFICIENT if claim_type != NON_MARKET_BUSINESS_CLAIM else NOT_APPLICABLE,
        "market_consistency_score": 0.5,
        "market_evidence_strength": "none",
        "market_evidence_date": "",
        "market_evidence_ids": [],
        "market_reference_date": "",
        "causal_cutoff": "",
        "causal_market_available": False,
        "post_event_market_available": bool(post_audit["post_event_market_available"]),
        "post_event_only": False,
        "post_event_audit_only": True,
        "closed_track_verifiable": False,
        "future_data_used": False,
        "entity_alignment_status": "not_applicable" if claim_type != EXPLICIT_PRICE_CLAIM else "failed",
        "entity_reject_reason": reason if claim_type == EXPLICIT_PRICE_CLAIM else "",
        "time_alignment_status": "not_applicable",
        "time_reject_reason": "",
        "market_reason": reason,
        "accepted_market_claim_count": len(accepted or []),
        "rejected_market_claim_count": len(rejected or []),
        "parsed_claim_values": [],
        "post_event_audit": post_audit,
        "source_version": MarketRuleConfig().version,
    }


def aggregate_market_consistency(
    row: dict[str, Any],
    index: MarketIndex,
    cfg: MarketRuleConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = cfg or MarketRuleConfig()
    post_audit = post_event_audit(row, index)
    accepted, rejected = explicit_price_candidates(row)
    aligned_candidates = [align_claim_time(item, index, cfg) for item in accepted]
    evaluated = [evaluate_candidate(item, index, cfg) for item in aligned_candidates]
    if evaluated:
        verifiable = [item for item in evaluated if item["market_consistency_verdict"] in {CONSISTENT, CONTRADICTORY}]
        selected = verifiable[0] if verifiable else evaluated[0]
        result = {
            "sample_id": norm(row.get("sample_id")),
            "target_code": extract_target_code(row),
            "target_name": extract_target_name(row),
            "market_claim_type": EXPLICIT_PRICE_CLAIM,
            "claim_text": selected.get("claim_text") or claim_title(row),
            "claim_time": selected.get("claim_time"),
            "claim_time_scope": selected.get("claim_time_scope"),
            "market_range_window": selected.get("market_range_window"),
            "market_range_is_consecutive": bool(selected.get("market_range_is_consecutive")),
            "claimed_price_direction": selected.get("direction") or "",
            "claimed_price_change": claimed_price_change(selected),
            "actual_price_change": selected.get("actual_price_change"),
            "price_direction": selected.get("price_direction") or "",
            "market_consistency_verdict": selected.get("market_consistency_verdict"),
            "market_consistency_score": selected.get("market_consistency_score"),
            "market_evidence_strength": selected.get("market_evidence_strength"),
            "market_evidence_date": selected.get("market_evidence_date") or "",
            "market_evidence_ids": selected.get("market_evidence_ids") or [],
            "market_reference_date": selected.get("market_reference_date") or "",
            "causal_cutoff": selected.get("causal_cutoff") or "",
            "causal_market_available": bool(selected.get("causal_market_available")),
            "post_event_market_available": bool(post_audit["post_event_market_available"]),
            "post_event_only": bool(selected.get("post_event_only")),
            "post_event_audit_only": True,
            "closed_track_verifiable": bool(selected.get("closed_track_verifiable")),
            "future_data_used": False,
            "entity_alignment_status": selected.get("entity_alignment_status"),
            "entity_reject_reason": selected.get("entity_reject_reason") or "",
            "time_alignment_status": selected.get("time_alignment_status"),
            "time_reject_reason": selected.get("time_reject_reason") or "",
            "market_reason": selected.get("market_reason"),
            "accepted_market_claim_count": len(accepted),
            "rejected_market_claim_count": len(rejected),
            "parsed_claim_values": [
                {
                    key: item.get(key)
                    for key in (
                        "raw_text",
                        "raw_value",
                        "normalized_value",
                        "unit",
                        "value_type",
                        "direction",
                        "parse_validation_error",
                    )
                }
                for item in evaluated
            ],
            "post_event_audit": post_audit,
            "source_version": cfg.version,
        }
    elif rejected:
        result = insufficient_row(
            row,
            EXPLICIT_PRICE_CLAIM,
            ";".join(dict.fromkeys(norm(item.get("entity_reject_reason")) for item in rejected if norm(item.get("entity_reject_reason")))),
            post_audit,
            accepted,
            rejected,
        )
        result["entity_alignment_status"] = "failed"
    else:
        claim_type = classify_non_explicit(row)
        reason = "non_price_market_claim_not_verified_by_stock_price" if claim_type == MARKET_SENTIMENT_CLAIM else (
            "financial_or_business_claim_not_verified_by_stock_price"
            if claim_type == FINANCIAL_PERFORMANCE_CLAIM
            else "no_explicit_price_claim"
        )
        result = insufficient_row(row, claim_type, reason, post_audit, accepted, rejected)
    return jsonable(result), [jsonable(item) for item in evaluated], [jsonable(item) for item in rejected]
