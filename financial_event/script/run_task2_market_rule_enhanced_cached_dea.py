#!/usr/bin/env python3
"""Deterministic market-rule enhancement for Task2 CachedDEA.

This script does not train a model. It writes locked500 predictions and their
SHA256 before loading the locked gold workbook for one-time evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.resset_market_adapter import load_market_cache  # noqa: E402


DEFAULT_INPUT = ROOT / "outputs/event_credibility/task2_independent500_v1/independent_samples.jsonl"
DEFAULT_FINANCIAL_OUTPUT_DIR = ROOT / "outputs/event_credibility/task2_financial_enhanced_cached_dea_v2"
DEFAULT_MARKET_CACHE = (
    ROOT
    / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v1/resset_market_daily_cache.parquet"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/event_credibility/task2_market_rule_enhanced_cached_dea_v1"
DEFAULT_GOLD = ROOT / "reports/task2_independent500_auto_assisted_human_confirmed_gold.xlsx"
DEFAULT_REPORT_MD = ROOT / "reports/task2_market_rule_enhanced_cached_dea_report.md"
DEFAULT_DETAILS_XLSX = ROOT / "reports/task2_market_rule_enhanced_cached_dea_details.xlsx"

EXPECTED_LOCKED500_SHA256 = "80a3666317a23e5a3c79539a6c000a11293d08fa24efdf41af06dfffc2a5472c"
LABELS = ["support", "contradict", "insufficient", "unclear"]

EXPLICIT_PRICE_CLAIM = "explicit_price_claim"
MARKET_SENTIMENT_CLAIM = "market_sentiment_claim"
FINANCIAL_PERFORMANCE_CLAIM = "financial_performance_claim"
NON_MARKET_BUSINESS_CLAIM = "non_market_business_claim"

MARKET_CLOSE = time(15, 0, 0)
SEED = 2026


@dataclass(frozen=True)
class RuleConfig:
    explicit_direction_contradiction_adjustment: float = -0.10
    explicit_numeric_contradiction_adjustment: float = -0.15
    explicit_consistent_adjustment: float = 0.05
    weak_sentiment_adjustment: float = 0.03
    numeric_relative_error_threshold: float = 0.20
    sentiment_flat_threshold: float = 1e-12


@dataclass
class MarketIndex:
    rows: dict[tuple[str, date], dict[str, Any]]
    dates_by_code: dict[str, list[date]]
    global_dates: list[date]
    global_date_set: set[date]
    codes: set[str]
    rows_count: int
    date_min: str
    date_max: str


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(obj)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")


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


def norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def compact_text(value: Any) -> str:
    text = norm(value).lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[，。、“”‘’；：:,.!?！？（）()\[\]【】《》<>|｜\\_·]", "", text)


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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


def extract_stock_code(row: dict[str, Any]) -> str:
    for key in ("target_code", "stock_code", "target_entity", "title_samples", "event_claim", "event_name"):
        text = norm(row.get(key))
        match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
        if match:
            return match.group(1)
    return ""


def claim_surface_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title_samples", "event_claim", "event_name", "event_type", "schema_id"):
        value = row.get(key)
        if isinstance(value, list):
            parts.extend(norm(item) for item in value if norm(item))
        elif norm(value):
            parts.append(norm(value))
    return " ".join(dict.fromkeys(parts))


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
    date_min = str(global_dates[0]) if global_dates else ""
    date_max = str(global_dates[-1]) if global_dates else ""
    return MarketIndex(
        rows=rows,
        dates_by_code={code: sorted(set(days)) for code, days in dates_by_code.items()},
        global_dates=global_dates,
        global_date_set=set(global_dates),
        codes=set(dates_by_code),
        rows_count=int(len(df)),
        date_min=date_min,
        date_max=date_max,
    )


def row_valid(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("valid_price") and safe_float(row.get("daily_return")) is not None)


def market_row(index: MarketIndex, code: str, trading_day: date | None) -> dict[str, Any] | None:
    if not code or trading_day is None:
        return None
    return index.rows.get((code, trading_day))


def latest_completed_trading_date(publish_dt: datetime | None, index: MarketIndex) -> date | None:
    if publish_dt is None or not index.global_dates:
        return None
    pub_day = publish_dt.date()
    if pub_day in index.global_date_set and publish_dt.time() >= MARKET_CLOSE:
        return pub_day
    pos = bisect_left(index.global_dates, pub_day)
    if pos == 0:
        return None
    return index.global_dates[pos - 1]


def first_post_event_trading_date(publish_dt: datetime | None, index: MarketIndex) -> date | None:
    if publish_dt is None or not index.global_dates:
        return None
    pub_day = publish_dt.date()
    if pub_day in index.global_date_set and publish_dt.time() < MARKET_CLOSE:
        return pub_day
    pos = bisect_right(index.global_dates, pub_day)
    if pos >= len(index.global_dates):
        return None
    return index.global_dates[pos]


def next_global_trading_date(index: MarketIndex, current: date, offset: int = 1) -> date | None:
    pos = bisect_right(index.global_dates, current)
    target = pos + offset - 1
    if target < 0 or target >= len(index.global_dates):
        return None
    return index.global_dates[target]


def previous_global_trading_date(index: MarketIndex, current: date) -> date | None:
    pos = bisect_left(index.global_dates, current)
    if pos == 0:
        return None
    return index.global_dates[pos - 1]


def previous_code_dates(index: MarketIndex, code: str, end_day: date, window: int) -> list[date]:
    days = index.dates_by_code.get(code, [])
    pos = bisect_right(days, end_day)
    return days[max(0, pos - window) : pos]


CN_NUMBERS = {
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
    "十": 10,
}


def parse_cn_int(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if text in CN_NUMBERS:
        return CN_NUMBERS[text]
    return None


UP_TERMS = (
    "上涨",
    "上扬",
    "走强",
    "大涨",
    "拉升",
    "涨停",
    "收涨",
    "反弹",
    "冲高",
    "创新高",
    "创历史新高",
    "新高",
    "涨超",
    "涨逾",
    "涨近",
    "市场反应积极",
    "市场表现积极",
    "市场看好",
    "投资者信心增强",
    "投资者信心提振",
    "资金净流入",
    "净流入",
)
DOWN_TERMS = (
    "下跌",
    "下挫",
    "走弱",
    "大跌",
    "跳水",
    "跌停",
    "收跌",
    "重挫",
    "承压",
    "杀跌",
    "创新低",
    "创历史新低",
    "新低",
    "跌超",
    "跌逾",
    "跌近",
    "市场表现低迷",
    "市场低迷",
    "投资者信心不足",
    "投资者信心下降",
    "市场不看好",
    "市场担忧",
    "市场反应冷淡",
    "看空",
    "抛压",
    "资金净流出",
    "净流出",
)
PRICE_CONTEXT_TERMS = (
    "股价",
    "股票",
    "a股",
    "A股",
    "涨停",
    "跌停",
    "涨幅",
    "跌幅",
    "收盘价",
    "开盘",
    "盘中",
    "收涨",
    "收跌",
    "股市",
    "二级市场",
    "龙虎榜",
    "元/股",
    "每股",
)
SENTIMENT_TERMS = (
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
    "抛压",
    "主力资金",
    "资金净流入",
    "资金净流出",
    "净流入",
    "净流出",
)
FINANCIAL_TERMS = (
    "盈利能力",
    "现金流",
    "营收",
    "营业收入",
    "收入增长",
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


def direction_from_text(text: str) -> str:
    up = any(term.lower() in text for term in UP_TERMS)
    down = any(term.lower() in text for term in DOWN_TERMS)
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    if up and down:
        return "mixed"
    return ""


def claimed_change_from_text(text: str, direction: str) -> float | None:
    patterns = [
        r"(?:上涨|上扬|大涨|涨幅|涨超|涨逾|涨近|拉升|收涨)\s*(?:约|近|超|逾)?\s*([-+]?\d+(?:\.\d+)?)\s*%",
        r"(?:下跌|下挫|大跌|跌幅|跌超|跌逾|跌近|跳水|收跌)\s*(?:约|近|超|逾)?\s*([-+]?\d+(?:\.\d+)?)\s*%",
        r"(?<!同比)(?<!环比)(?:涨|跌)\s*(?:停|约|近|超|逾)?\s*([-+]?\d+(?:\.\d+)?)\s*%",
        r"([-+]?\d+(?:\.\d+)?)\s*%\s*(?:的)?\s*(?:涨幅|跌幅|上涨|下跌|收涨|收跌)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            magnitude = abs(float(match.group(1))) / 100.0
            if direction == "down":
                return -magnitude
            if direction == "up":
                return magnitude
            return magnitude
    return None


def extract_streak_claim(text: str) -> dict[str, Any]:
    match = re.search(r"连续([一二两三四五六七八九十\d]+)个?交易日?(上涨|上扬|下跌|下挫|走强|走弱)", text)
    if match:
        days = parse_cn_int(match.group(1))
        direction = "up" if match.group(2) in {"上涨", "上扬", "走强"} else "down"
        return {"days": days, "direction": direction}
    match = re.search(r"([一二两三四五六七八九十\d]+)连(涨|跌)", text)
    if match:
        days = parse_cn_int(match.group(1))
        direction = "up" if match.group(2) == "涨" else "down"
        return {"days": days, "direction": direction}
    return {"days": None, "direction": ""}


def extract_high_low_claim(text: str) -> str:
    if "历史新高" in text or "创新高" in text or "新高" in text:
        return "new_high"
    if "历史新低" in text or "创新低" in text or "新低" in text:
        return "new_low"
    return ""


def has_standalone_price_move(text: str) -> bool:
    match = re.search(r"(?<!同比)(?<!环比)(涨|跌)\s*(?:停|约|近|超|逾)?\s*\d+(?:\.\d+)?\s*%", text)
    if not match:
        return False
    left = text[max(0, match.start() - 10) : match.start()]
    return not any(term in left for term in ("营收", "收入", "净利润", "毛利率", "现金流", "roe", "roa"))


def classify_market_claim(row: dict[str, Any], financial_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    text = claim_surface_text(row)
    compact = compact_text(text)
    direction = direction_from_text(compact)
    claimed_change = claimed_change_from_text(compact, direction)
    streak = extract_streak_claim(compact)
    high_low = extract_high_low_claim(compact)
    has_price_context = any(term.lower() in compact for term in PRICE_CONTEXT_TERMS)
    has_price_move = has_standalone_price_move(compact)
    financial_claim_count = int((financial_audit or {}).get("financial_claim_count") or 0)
    has_financial = financial_claim_count > 0 or any(term.lower() in compact for term in FINANCIAL_TERMS)

    if has_price_context or has_price_move:
        return {
            "market_claim_type": EXPLICIT_PRICE_CLAIM,
            "claimed_price_direction": direction or "unspecified",
            "claimed_price_change": claimed_change,
            "streak_days": streak["days"],
            "streak_direction": streak["direction"],
            "high_low_claim": high_low,
            "claim_text": text,
            "classification_reason": "explicit_price_or_trading_terms",
            "financial_claim_count": financial_claim_count,
        }

    if any(term.lower() in compact for term in SENTIMENT_TERMS):
        if direction not in {"up", "down"}:
            if any(term in compact for term in ("低迷", "不足", "下降", "不看好", "担忧", "冷淡", "承压", "抛压", "净流出")):
                direction = "down"
            elif any(term in compact for term in ("积极", "看好", "增强", "提振", "净流入")):
                direction = "up"
        return {
            "market_claim_type": MARKET_SENTIMENT_CLAIM,
            "claimed_price_direction": direction or "unspecified",
            "claimed_price_change": None,
            "streak_days": None,
            "streak_direction": "",
            "high_low_claim": "",
            "claim_text": text,
            "classification_reason": "market_sentiment_terms",
            "financial_claim_count": financial_claim_count,
        }

    if has_financial:
        return {
            "market_claim_type": FINANCIAL_PERFORMANCE_CLAIM,
            "claimed_price_direction": "",
            "claimed_price_change": None,
            "streak_days": None,
            "streak_direction": "",
            "high_low_claim": "",
            "claim_text": text,
            "classification_reason": "financial_performance_terms_or_csmar_claims",
            "financial_claim_count": financial_claim_count,
        }

    return {
        "market_claim_type": NON_MARKET_BUSINESS_CLAIM,
        "claimed_price_direction": "",
        "claimed_price_change": None,
        "streak_days": None,
        "streak_direction": "",
        "high_low_claim": "",
        "claim_text": text,
        "classification_reason": "non_market_business_claim",
        "financial_claim_count": financial_claim_count,
    }


def extract_referenced_date(text: str, publish_dt: datetime | None) -> date | None:
    if publish_dt is None:
        return None
    for pattern in (r"(20\d{2})年(\d{1,2})月(\d{1,2})日", r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})"):
        match = re.search(pattern, text)
        if match:
            year, month, day = (int(match.group(i)) for i in (1, 2, 3))
            try:
                return date(year, month, day)
            except ValueError:
                return None
    match = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})日", text)
    if match:
        month, day = int(match.group(1)), int(match.group(2))
        try:
            return date(publish_dt.year, month, day)
        except ValueError:
            return None
    if any(term in text for term in ("昨日", "昨天")):
        return publish_dt.date() - timedelta(days=1)
    return None


def choose_closed_track_date(
    classification: dict[str, Any],
    publish_dt: datetime | None,
    index: MarketIndex,
) -> tuple[date | None, str]:
    if publish_dt is None:
        return None, "publish_time_missing_or_invalid"
    latest = latest_completed_trading_date(publish_dt, index)
    if latest is None:
        return None, "no_completed_trading_day_before_publication"
    compact = compact_text(classification.get("claim_text"))
    if any(term in compact for term in ("今日", "今天", "当日", "当天")) and publish_dt.date() in index.global_date_set and publish_dt.time() < MARKET_CLOSE:
        return None, "same_day_close_not_completed_before_publication"
    if "前一交易日" in compact or "上一交易日" in compact:
        previous = previous_global_trading_date(index, publish_dt.date())
        if previous and previous <= latest:
            return previous, "previous_trading_day_reference"
    referenced = extract_referenced_date(compact, publish_dt)
    if referenced is not None:
        if referenced > publish_dt.date() or referenced > latest:
            return None, "referenced_trading_day_not_completed_before_publication"
        if referenced not in index.global_date_set:
            return None, "referenced_date_not_a_trading_day"
        return referenced, "explicit_date_reference"
    return latest, "latest_completed_trading_day"


def evidence_ids(code: str, trading_day: date | None, fields: Iterable[str]) -> list[str]:
    if not code or trading_day is None:
        return []
    return [f"RESSET_DRESSTK_2026:{code}:{trading_day.isoformat()}:{field}" for field in fields]


def actual_direction(actual_change: float | None) -> str:
    if actual_change is None:
        return ""
    if actual_change > 0:
        return "up"
    if actual_change < 0:
        return "down"
    return "flat"


def cumulative_return(rows: list[dict[str, Any] | None]) -> float | None:
    value = 1.0
    used = 0
    for row in rows:
        daily = safe_float(row.get("daily_return") if row else None)
        if daily is None:
            return None
        value *= 1.0 + daily
        used += 1
    return value - 1.0 if used else None


def trailing_returns(index: MarketIndex, code: str, trading_day: date | None) -> dict[str, Any]:
    if trading_day is None or not code:
        return {"previous_1d_return": None, "previous_3d_return": None, "previous_5d_return": None}
    result: dict[str, Any] = {}
    for window in (1, 3, 5):
        days = previous_code_dates(index, code, trading_day, window)
        rows = [market_row(index, code, day) for day in days]
        result[f"previous_{window}d_return"] = cumulative_return(rows) if len(rows) == window else None
    return result


def post_event_audit(index: MarketIndex, code: str, publish_dt: datetime | None) -> dict[str, Any]:
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
        "post_event_market_available": any(row_valid(row) for row in rows),
        "post_event_audit_only": bool(dates),
        "forbidden_in_closed_track": True,
        "used_in_closed_track": False,
        "post_event_dates": [day.isoformat() for day in dates],
        "post_event_returns": [safe_float(row.get("daily_return") if row else None) for row in rows],
        "post_event_market_evidence_ids": [
            item
            for day in dates
            for item in evidence_ids(code, day, ("close", "previous_close", "daily_return"))
            if market_row(index, code, day) is not None
        ],
    }


def base_market_record(
    row: dict[str, Any],
    financial_audit: dict[str, Any] | None,
    index: MarketIndex,
) -> tuple[dict[str, Any], dict[str, Any], str, datetime | None, date | None, dict[str, Any] | None, dict[str, Any]]:
    code = extract_stock_code(row)
    publish_dt = parse_datetime(row.get("earliest_publish_time") or row.get("event_time") or row.get("latest_publish_time"))
    classification = classify_market_claim(row, financial_audit)
    selected_day, date_reason = choose_closed_track_date(classification, publish_dt, index)
    selected_row = market_row(index, code, selected_day)
    post_audit = post_event_audit(index, code, publish_dt)
    return classification, post_audit, code, publish_dt, selected_day, selected_row, {"date_reason": date_reason}


def insufficient_market_result(
    row: dict[str, Any],
    classification: dict[str, Any],
    post_audit: dict[str, Any],
    code: str,
    selected_day: date | None,
    reason: str,
    financial_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cluster_id": norm(row.get("cluster_id")),
        "trading_code": code,
        "market_claim_type": classification["market_claim_type"],
        "claimed_price_direction": classification.get("claimed_price_direction") or "",
        "claimed_price_change": classification.get("claimed_price_change"),
        "actual_price_change": None,
        "price_direction": "",
        "market_consistency_verdict": "insufficient",
        "market_consistency_score": 0.5,
        "market_score_adjustment": 0.0,
        "market_evidence_strength": "none",
        "market_evidence_date": selected_day.isoformat() if selected_day else "",
        "market_evidence_ids": [],
        "causal_market_available": False,
        "post_event_market_available": bool(post_audit["post_event_market_available"]),
        "post_event_audit_only": bool(post_audit["post_event_audit_only"]),
        "market_reason": reason,
        "future_market_data_used_in_closed_track": False,
        "secondary_market_signal": {},
        "financial_claim_count": int((financial_audit or {}).get("financial_claim_count") or 0),
        "classification_reason": classification.get("classification_reason") or "",
        "claim_text": classification.get("claim_text") or "",
        "post_event_audit": post_audit,
    }


def evaluate_explicit_price_claim(
    row: dict[str, Any],
    classification: dict[str, Any],
    post_audit: dict[str, Any],
    code: str,
    selected_day: date | None,
    selected_row: dict[str, Any] | None,
    reason_bits: list[str],
    cfg: RuleConfig,
    index: MarketIndex,
) -> dict[str, Any]:
    if not code:
        return insufficient_market_result(row, classification, post_audit, code, selected_day, "stock_code_missing")
    if selected_day is None:
        return insufficient_market_result(row, classification, post_audit, code, selected_day, ";".join(reason_bits))
    if selected_row is None:
        return insufficient_market_result(
            row,
            classification,
            post_audit,
            code,
            selected_day,
            ";".join(reason_bits + ["market_row_missing_or_suspended"]),
        )
    if not row_valid(selected_row):
        missing_reason = norm(selected_row.get("missing_reason")) or "daily_return_missing_or_invalid"
        return insufficient_market_result(row, classification, post_audit, code, selected_day, ";".join(reason_bits + [missing_reason]))

    actual_change = safe_float(selected_row.get("daily_return"))
    direction = actual_direction(actual_change)
    claimed_direction = classification.get("claimed_price_direction") or ""
    claimed_change = safe_float(classification.get("claimed_price_change"))
    verdict = "insufficient"
    adjustment = 0.0
    market_reason = "claimed_price_direction_missing"

    streak_days = classification.get("streak_days")
    streak_direction = classification.get("streak_direction") or ""
    if streak_days and streak_direction in {"up", "down"}:
        market_reason = f"streak_{streak_days}_trading_days_checked"
        days = previous_code_dates(index, code, selected_day, int(streak_days))
        rows = [market_row(index, code, day) for day in days]
        changes = [safe_float(item.get("daily_return") if item else None) for item in rows]
        if len(changes) != int(streak_days) or any(change is None or change == 0 for change in changes):
            verdict = "insufficient"
            adjustment = 0.0
            market_reason = "streak_market_data_insufficient"
        else:
            directions = [actual_direction(change) for change in changes]
            if all(item == streak_direction for item in directions):
                verdict = "consistent"
                adjustment = cfg.explicit_consistent_adjustment
            else:
                verdict = "contradictory"
                adjustment = cfg.explicit_direction_contradiction_adjustment

    high_low_claim = classification.get("high_low_claim") or ""
    if verdict == "insufficient" and high_low_claim:
        days = previous_code_dates(index, code, selected_day, 10_000)
        rows = [market_row(index, code, day) for day in days]
        closes = [safe_float(item.get("close") if item else None) for item in rows]
        if len([value for value in closes if value is not None]) < 2:
            market_reason = "high_low_claim_market_history_insufficient"
        else:
            current_close = closes[-1]
            prior = [value for value in closes[:-1] if value is not None]
            if current_close is None or not prior:
                market_reason = "high_low_claim_market_history_insufficient"
            elif high_low_claim == "new_high":
                verdict = "consistent" if current_close >= max(prior) else "contradictory"
                adjustment = cfg.explicit_consistent_adjustment if verdict == "consistent" else cfg.explicit_direction_contradiction_adjustment
                market_reason = "new_high_checked_against_available_prior_resset_history"
            elif high_low_claim == "new_low":
                verdict = "consistent" if current_close <= min(prior) else "contradictory"
                adjustment = cfg.explicit_consistent_adjustment if verdict == "consistent" else cfg.explicit_direction_contradiction_adjustment
                market_reason = "new_low_checked_against_available_prior_resset_history"

    if not high_low_claim and not streak_days:
        if claimed_change is not None and actual_change is not None:
            denominator = abs(claimed_change)
            relative_error = abs(abs(actual_change) - denominator) / denominator if denominator > 0 else 0.0
            if relative_error > cfg.numeric_relative_error_threshold:
                verdict = "contradictory"
                adjustment = cfg.explicit_numeric_contradiction_adjustment
                market_reason = f"explicit_price_numeric_relative_error_{relative_error:.6f}_over_20pct"
            elif claimed_direction in {"up", "down"} and direction in {"up", "down"} and claimed_direction != direction:
                verdict = "contradictory"
                adjustment = cfg.explicit_direction_contradiction_adjustment
                market_reason = "explicit_price_direction_opposes_closed_track_return"
            elif claimed_direction in {"up", "down"} and direction in {"up", "down"} and claimed_direction == direction:
                verdict = "consistent"
                adjustment = cfg.explicit_consistent_adjustment
                market_reason = f"explicit_price_numeric_relative_error_{relative_error:.6f}_within_20pct"
            else:
                verdict = "consistent"
                adjustment = cfg.explicit_consistent_adjustment
                market_reason = f"explicit_price_numeric_relative_error_{relative_error:.6f}_within_20pct_direction_unspecified"
        elif claimed_direction in {"up", "down"} and direction in {"up", "down"} and claimed_direction == direction:
            verdict = "consistent"
            adjustment = cfg.explicit_consistent_adjustment
            market_reason = "explicit_price_direction_matches_closed_track_return"
        elif claimed_direction in {"up", "down"} and direction == "flat":
            verdict = "insufficient"
            adjustment = 0.0
            market_reason = "actual_return_flat_no_direction"

    return {
        "cluster_id": norm(row.get("cluster_id")),
        "trading_code": code,
        "market_claim_type": EXPLICIT_PRICE_CLAIM,
        "claimed_price_direction": claimed_direction,
        "claimed_price_change": claimed_change,
        "actual_price_change": actual_change,
        "price_direction": direction,
        "market_consistency_verdict": verdict,
        "market_consistency_score": clamp(0.5 + adjustment),
        "market_score_adjustment": adjustment,
        "market_evidence_strength": "strong" if verdict in {"consistent", "contradictory"} else "none",
        "market_evidence_date": selected_day.isoformat(),
        "market_evidence_ids": evidence_ids(code, selected_day, ("close", "previous_close", "daily_return")),
        "causal_market_available": True,
        "post_event_market_available": bool(post_audit["post_event_market_available"]),
        "post_event_audit_only": bool(post_audit["post_event_audit_only"]),
        "market_reason": ";".join(reason_bits + [market_reason]),
        "future_market_data_used_in_closed_track": False,
        "secondary_market_signal": {},
        "financial_claim_count": 0,
        "classification_reason": classification.get("classification_reason") or "",
        "claim_text": classification.get("claim_text") or "",
        "post_event_audit": post_audit,
    }


def evaluate_sentiment_claim(
    row: dict[str, Any],
    classification: dict[str, Any],
    post_audit: dict[str, Any],
    code: str,
    selected_day: date | None,
    selected_row: dict[str, Any] | None,
    reason_bits: list[str],
    cfg: RuleConfig,
    index: MarketIndex,
) -> dict[str, Any]:
    if not code:
        return insufficient_market_result(row, classification, post_audit, code, selected_day, "stock_code_missing")
    if selected_day is None:
        return insufficient_market_result(row, classification, post_audit, code, selected_day, ";".join(reason_bits))
    if selected_row is None or not row_valid(selected_row):
        return insufficient_market_result(row, classification, post_audit, code, selected_day, ";".join(reason_bits + ["market_row_missing_or_invalid"]))

    trailing = trailing_returns(index, code, selected_day)
    weighted_parts: list[tuple[float, float]] = []
    for key, weight in (("previous_1d_return", 0.5), ("previous_3d_return", 0.3), ("previous_5d_return", 0.2)):
        value = safe_float(trailing.get(key))
        if value is not None:
            weighted_parts.append((value, weight))
    if not weighted_parts:
        return insufficient_market_result(row, classification, post_audit, code, selected_day, ";".join(reason_bits + ["sentiment_return_windows_missing"]))
    signal = sum(value * weight for value, weight in weighted_parts) / sum(weight for _, weight in weighted_parts)
    if abs(signal) <= cfg.sentiment_flat_threshold:
        return insufficient_market_result(row, classification, post_audit, code, selected_day, ";".join(reason_bits + ["sentiment_return_signal_flat"]))

    claimed_direction = classification.get("claimed_price_direction") or "unspecified"
    signal_direction = "up" if signal > 0 else "down"
    if claimed_direction not in {"up", "down"}:
        return insufficient_market_result(row, classification, post_audit, code, selected_day, ";".join(reason_bits + ["sentiment_direction_missing"]))
    same = claimed_direction == signal_direction
    verdict = "weak_consistent" if same else "weak_contradictory"
    adjustment = cfg.weak_sentiment_adjustment if same else -cfg.weak_sentiment_adjustment
    turnover_ratio = safe_float(selected_row.get("turnover_abnormal_ratio"))
    turnover_change = turnover_ratio - 1.0 if turnover_ratio is not None else None
    actual_change = safe_float(selected_row.get("daily_return"))
    return {
        "cluster_id": norm(row.get("cluster_id")),
        "trading_code": code,
        "market_claim_type": MARKET_SENTIMENT_CLAIM,
        "claimed_price_direction": claimed_direction,
        "claimed_price_change": None,
        "actual_price_change": actual_change,
        "price_direction": actual_direction(actual_change),
        "market_consistency_verdict": verdict,
        "market_consistency_score": clamp(0.5 + adjustment),
        "market_score_adjustment": adjustment,
        "market_evidence_strength": "weak",
        "market_evidence_date": selected_day.isoformat(),
        "market_evidence_ids": evidence_ids(code, selected_day, ("close", "previous_close", "daily_return", "turnover_abnormal_ratio")),
        "causal_market_available": True,
        "post_event_market_available": bool(post_audit["post_event_market_available"]),
        "post_event_audit_only": bool(post_audit["post_event_audit_only"]),
        "market_reason": ";".join(reason_bits + [f"sentiment_{signal_direction}_signal_from_closed_track_1d_3d_5d_returns"]),
        "future_market_data_used_in_closed_track": False,
        "secondary_market_signal": {
            **trailing,
            "turnover_change": turnover_change,
            "weighted_return_signal": signal,
        },
        "financial_claim_count": 0,
        "classification_reason": classification.get("classification_reason") or "",
        "claim_text": classification.get("claim_text") or "",
        "post_event_audit": post_audit,
    }


def evaluate_financial_claim(
    row: dict[str, Any],
    classification: dict[str, Any],
    post_audit: dict[str, Any],
    code: str,
    selected_day: date | None,
    selected_row: dict[str, Any] | None,
    reason_bits: list[str],
    financial_audit: dict[str, Any] | None,
    index: MarketIndex,
) -> dict[str, Any]:
    actual_change = safe_float(selected_row.get("daily_return")) if selected_row and row_valid(selected_row) else None
    strength = "secondary_market_signal" if actual_change is not None else "none"
    market_ids = evidence_ids(code, selected_day, ("close", "previous_close", "daily_return")) if actual_change is not None else []
    financial_verdict = norm((financial_audit or {}).get("financial_consistency_verdict"))
    return {
        "cluster_id": norm(row.get("cluster_id")),
        "trading_code": code,
        "market_claim_type": FINANCIAL_PERFORMANCE_CLAIM,
        "claimed_price_direction": "",
        "claimed_price_change": None,
        "actual_price_change": actual_change,
        "price_direction": actual_direction(actual_change),
        "market_consistency_verdict": "secondary_market_signal_only" if actual_change is not None else "insufficient",
        "market_consistency_score": 0.5,
        "market_score_adjustment": 0.0,
        "market_evidence_strength": strength,
        "market_evidence_date": selected_day.isoformat() if actual_change is not None and selected_day else "",
        "market_evidence_ids": market_ids,
        "causal_market_available": actual_change is not None,
        "post_event_market_available": bool(post_audit["post_event_market_available"]),
        "post_event_audit_only": bool(post_audit["post_event_audit_only"]),
        "market_reason": ";".join(
            reason_bits
            + [
                "financial_performance_claim_uses_csmar_primary_market_secondary_only",
                f"csmar_financial_verdict={financial_verdict or 'missing'}",
            ]
        ),
        "future_market_data_used_in_closed_track": False,
        "secondary_market_signal": {
            **trailing_returns(index, code, selected_day),
            "financial_consistency_verdict": financial_verdict,
            "financial_claim_count": int((financial_audit or {}).get("financial_claim_count") or classification.get("financial_claim_count") or 0),
        },
        "financial_claim_count": int((financial_audit or {}).get("financial_claim_count") or classification.get("financial_claim_count") or 0),
        "classification_reason": classification.get("classification_reason") or "",
        "claim_text": classification.get("claim_text") or "",
        "post_event_audit": post_audit,
    }


def evaluate_non_market_claim(
    row: dict[str, Any],
    classification: dict[str, Any],
    post_audit: dict[str, Any],
    code: str,
    selected_day: date | None,
    selected_row: dict[str, Any] | None,
    reason_bits: list[str],
) -> dict[str, Any]:
    return {
        "cluster_id": norm(row.get("cluster_id")),
        "trading_code": code,
        "market_claim_type": NON_MARKET_BUSINESS_CLAIM,
        "claimed_price_direction": "",
        "claimed_price_change": None,
        "actual_price_change": None,
        "price_direction": "",
        "market_consistency_verdict": "not_applicable",
        "market_consistency_score": 0.5,
        "market_score_adjustment": 0.0,
        "market_evidence_strength": "none",
        "market_evidence_date": "",
        "market_evidence_ids": [],
        "causal_market_available": bool(selected_row and row_valid(selected_row)),
        "post_event_market_available": bool(post_audit["post_event_market_available"]),
        "post_event_audit_only": bool(post_audit["post_event_audit_only"]),
        "market_reason": ";".join(reason_bits + ["non_market_business_claim_market_neutral"]),
        "future_market_data_used_in_closed_track": False,
        "secondary_market_signal": {},
        "financial_claim_count": 0,
        "classification_reason": classification.get("classification_reason") or "",
        "claim_text": classification.get("claim_text") or "",
        "post_event_audit": post_audit,
    }


def evaluate_market_rule(
    row: dict[str, Any],
    financial_audit: dict[str, Any] | None,
    index: MarketIndex,
    cfg: RuleConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or RuleConfig()
    classification, post_audit, code, _publish_dt, selected_day, selected_row, reason = base_market_record(row, financial_audit, index)
    reason_bits = [reason["date_reason"]]
    claim_type = classification["market_claim_type"]
    if claim_type == EXPLICIT_PRICE_CLAIM:
        return evaluate_explicit_price_claim(row, classification, post_audit, code, selected_day, selected_row, reason_bits, cfg, index)
    if claim_type == MARKET_SENTIMENT_CLAIM:
        return evaluate_sentiment_claim(row, classification, post_audit, code, selected_day, selected_row, reason_bits, cfg, index)
    if claim_type == FINANCIAL_PERFORMANCE_CLAIM:
        return evaluate_financial_claim(row, classification, post_audit, code, selected_day, selected_row, reason_bits, financial_audit, index)
    return evaluate_non_market_claim(row, classification, post_audit, code, selected_day, selected_row, reason_bits)


def result_label(result: dict[str, Any] | None) -> str:
    label = norm((result or {}).get("verdict"))
    return label if label in LABELS else ""


def result_confidence(result: dict[str, Any] | None) -> float:
    value = safe_float((result or {}).get("confidence"))
    return clamp(value if value is not None else 0.0)


def fuse_market_rule(base_result: dict[str, Any], market_context: dict[str, Any]) -> dict[str, Any]:
    base_label = result_label(base_result) or "unclear"
    base_confidence = result_confidence(base_result)
    adjustment = safe_float(market_context.get("market_score_adjustment")) or 0.0
    new_confidence = clamp(base_confidence + adjustment)
    return {
        "verdict": base_label,
        "confidence": round(new_confidence, 6),
        "source_status": norm(base_result.get("source_status")) or "financial_enhanced_cached_dea",
        "source_score": round(new_confidence, 6),
        "base_confidence": round(base_confidence, 6),
        "market_score_adjustment": round(adjustment, 6),
        "market_rule_effective": abs(adjustment) > 0,
        "label_locked_from_base": True,
        "label_lock_reason": "market_rule_adjusts_confidence_only_and_does_not_override_cached_dea_verdict",
    }


def validate_unique_ids(rows: list[dict[str, Any]], path: Path) -> list[str]:
    ids = [norm(row.get("cluster_id")) for row in rows]
    if any(not item for item in ids):
        raise ValueError(f"{path} contains rows without cluster_id")
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{path} contains duplicate cluster_id values: {duplicates[:10]}")
    return ids


def build_prediction_rows(
    samples: list[dict[str, Any]],
    frozen_rows: list[dict[str, Any]],
    financial_audit_rows: list[dict[str, Any]],
    market_index: MarketIndex,
    limit: int = 0,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    frozen_by_id = {norm(row.get("cluster_id")): row for row in frozen_rows}
    audit_by_id = {norm(row.get("cluster_id")): row for row in financial_audit_rows}
    selected_samples = samples[:limit] if limit and limit > 0 else samples
    baseline_rows: list[dict[str, Any]] = []
    cached_rows: list[dict[str, Any]] = []
    financial_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    market_audit_rows: list[dict[str, Any]] = []
    missing_frozen: list[str] = []
    for idx, sample in enumerate(selected_samples, 1):
        cid = norm(sample.get("cluster_id"))
        frozen = frozen_by_id.get(cid)
        if not frozen:
            missing_frozen.append(cid)
            continue
        financial_audit = audit_by_id.get(cid)
        market_context = evaluate_market_rule(sample, financial_audit, market_index)
        base_financial_result = frozen.get("financial_enhanced_result") or frozen.get("cached_dea_result") or {}
        market_result = fuse_market_rule(base_financial_result, market_context)
        common = {
            "cluster_id": cid,
            "sample_index": sample.get("sample_index", frozen.get("sample_index", idx)),
            "sample_group": sample.get("sample_group", frozen.get("sample_group", "")),
            "target_code": sample.get("target_code"),
            "target_name": sample.get("target_name"),
            "event_time": sample.get("event_time"),
            "earliest_publish_time": sample.get("earliest_publish_time"),
        }
        baseline = frozen.get("baseline_result") or {}
        cached = frozen.get("cached_dea_result") or {}
        financial = frozen.get("financial_enhanced_result") or {}
        baseline_rows.append({**common, "verdict": result_label(baseline), "confidence": result_confidence(baseline), "baseline_result": baseline})
        cached_rows.append({**common, "verdict": result_label(cached), "confidence": result_confidence(cached), "cached_dea_result": cached})
        financial_rows.append(
            {
                **common,
                "verdict": result_label(financial),
                "confidence": result_confidence(financial),
                "financial_enhanced_result": financial,
                "financial_context": frozen.get("financial_context") or {},
            }
        )
        market_row_payload = {
            **common,
            "verdict": market_result["verdict"],
            "confidence": market_result["confidence"],
            "financial_market_rule_enhanced_result": market_result,
            "financial_enhanced_result": financial,
            "cached_dea_result": cached,
            "market_rule_context": market_context,
        }
        for field in (
            "market_claim_type",
            "claimed_price_direction",
            "claimed_price_change",
            "actual_price_change",
            "market_consistency_verdict",
            "market_consistency_score",
            "market_evidence_strength",
            "market_evidence_date",
            "market_evidence_ids",
            "causal_market_available",
            "post_event_market_available",
            "post_event_audit_only",
            "market_reason",
        ):
            market_row_payload[field] = market_context.get(field)
        market_rows.append(market_row_payload)
        market_audit_rows.append({**common, **market_context, "market_rule_enhanced_result": market_result})
    if missing_frozen:
        raise ValueError(f"missing frozen predictions for cluster ids: {missing_frozen[:10]}")
    outputs = {
        "baseline_predictions_500.jsonl": baseline_rows,
        "cached_dea_predictions_500.jsonl": cached_rows,
        "financial_enhanced_predictions_500.jsonl": financial_rows,
        "market_rule_enhanced_predictions_500.jsonl": market_rows,
        "predictions_frozen_before_gold.jsonl": market_rows,
    }
    summary = summarize_market_audit(market_audit_rows)
    return outputs, market_audit_rows, summary


def summarize_market_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_counts = Counter(norm(row.get("market_claim_type")) for row in rows)
    verdict_counts = Counter(norm(row.get("market_consistency_verdict")) for row in rows)
    by_type_verdict: dict[str, dict[str, int]] = {}
    for row in rows:
        claim_type = norm(row.get("market_claim_type"))
        by_type_verdict.setdefault(claim_type, Counter())
        by_type_verdict[claim_type][norm(row.get("market_consistency_verdict"))] += 1
    explicit_rows = [row for row in rows if row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM]
    financial_rows = [row for row in rows if row.get("market_claim_type") == FINANCIAL_PERFORMANCE_CLAIM]
    effective_rows = [row for row in rows if abs(safe_float(row.get("market_score_adjustment")) or 0.0) > 0]
    return {
        "rows": len(rows),
        "market_claim_type_distribution": dict(claim_counts),
        "market_consistency_verdict_distribution": dict(verdict_counts),
        "market_consistency_by_type": {key: dict(value) for key, value in by_type_verdict.items()},
        "explicit_price_claim_count": len(explicit_rows),
        "explicit_price_claim_verified_count": sum(
            1 for row in explicit_rows if row.get("market_consistency_verdict") in {"consistent", "contradictory"}
        ),
        "explicit_price_claim_insufficient_count": sum(1 for row in explicit_rows if row.get("market_consistency_verdict") == "insufficient"),
        "financial_performance_claim_count": len(financial_rows),
        "financial_claim_secondary_market_signal_only_count": sum(
            1 for row in financial_rows if row.get("market_consistency_verdict") == "secondary_market_signal_only"
        ),
        "market_rule_effective_count": len(effective_rows),
        "causal_market_available_count": sum(1 for row in rows if row.get("causal_market_available")),
        "post_event_market_available_count": sum(1 for row in rows if row.get("post_event_market_available")),
        "post_event_audit_only_count": sum(1 for row in rows if row.get("post_event_audit_only")),
        "future_market_data_used_in_closed_track_count": sum(1 for row in rows if row.get("future_market_data_used_in_closed_track")),
    }


def freeze_predictions(
    output_dir: Path,
    outputs: dict[str, list[dict[str, Any]]],
    audit_rows: list[dict[str, Any]],
    inputs: dict[str, str],
    data_audit: dict[str, Any],
    limit: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for filename, rows in outputs.items():
        path = output_dir / filename
        write_jsonl(path, rows)
        written_paths.append(path)
    audit_path = output_dir / "market_rule_audit_500.jsonl"
    write_jsonl(audit_path, audit_rows)
    written_paths.append(audit_path)
    prediction_hashes = {
        display_path(path): sha256(path)
        for path in sorted(written_paths)
        if path.name.endswith(".jsonl")
    }
    freeze_manifest = {
        "status": "prediction_frozen_before_gold",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "limit": limit,
        "training_performed": False,
        "tuning_performed": False,
        "market_rules_fixed_before_gold": True,
        "gold_loaded_before_prediction_sha256": False,
        "gold_loaded_at_freeze_stage": False,
        "locked500_gold_used_for_rule_or_threshold_selection": False,
        "input_files": inputs,
        "data_audit": data_audit,
        "prediction_files": prediction_hashes,
    }
    freeze_path = output_dir / "prediction_freeze_manifest.json"
    write_json(freeze_path, freeze_manifest)
    return freeze_manifest


def load_gold(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=["Primary人工审核", "Stress人工审核"], dtype=str)
    frame = pd.concat(sheets.values(), ignore_index=True)
    frame["cluster_id"] = frame["cluster_id"].map(norm)
    frame["human_verdict"] = frame["human_verdict"].map(norm)
    frame["reviewer_confirmed"] = frame["reviewer_confirmed"].map(norm)
    if len(frame) != 500 or frame["cluster_id"].nunique() != 500:
        raise ValueError("gold workbook must contain 500 unique rows")
    if int((frame["human_verdict"] == "").sum()) != 0:
        raise ValueError("gold workbook contains empty human_verdict")
    illegal = sorted(set(frame["human_verdict"]) - set(LABELS))
    if illegal:
        raise ValueError(f"illegal labels in gold workbook: {illegal}")
    if int((frame["reviewer_confirmed"] != "yes").sum()) != 0:
        raise ValueError("all reviewer_confirmed values must be yes")
    return frame


def prediction_label(row: dict[str, Any]) -> str:
    direct = norm(row.get("verdict"))
    if direct in LABELS:
        return direct
    for key in ("financial_market_rule_enhanced_result", "financial_enhanced_result", "cached_dea_result", "baseline_result"):
        label = result_label(row.get(key) or {})
        if label:
            return label
    return ""


def metric_summary(gold: list[str], pred: list[str]) -> dict[str, Any]:
    total = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    per_class: dict[str, dict[str, Any]] = {}
    fixed_f1s: list[float] = []
    weighted_f1 = 0.0
    matrix = {label: {col: 0 for col in LABELS} for label in LABELS}
    for g, p in zip(gold, pred):
        if g in LABELS and p in LABELS:
            matrix[g][p] += 1
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        support = sum(g == label for g in gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        fixed_f1s.append(f1)
        weighted_f1 += f1 * support
    observed_labels = [label for label in LABELS if sum(g == label for g in gold) > 0]
    observed_macro = sum(per_class[label]["f1"] for label in observed_labels) / len(observed_labels) if observed_labels else 0.0
    return {
        "sample_count": total,
        "accuracy": round(correct / total if total else 0.0, 6),
        "macro_f1": round(sum(fixed_f1s) / len(LABELS), 6),
        "fixed_4_class_macro_f1": round(sum(fixed_f1s) / len(LABELS), 6),
        "observed_class_macro_f1": round(observed_macro, 6),
        "micro_f1": round(correct / total if total else 0.0, 6),
        "weighted_f1": round(weighted_f1 / total if total else 0.0, 6),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "gold_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(pred)),
        "coverage": 1.0 if total else 0.0,
        "failed_count": sum(item not in LABELS for item in pred),
    }


def bootstrap_ci(gold: list[str], pred: list[str], iterations: int = 2000, seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    size = len(gold)
    acc: list[float] = []
    macro: list[float] = []
    weighted: list[float] = []
    if not size:
        return {"seed": seed, "iterations": iterations, "accuracy_ci95": [0.0, 0.0], "macro_f1_ci95": [0.0, 0.0], "weighted_f1_ci95": [0.0, 0.0]}
    for _ in range(iterations):
        idxs = [rng.randrange(size) for _ in range(size)]
        g = [gold[i] for i in idxs]
        p = [pred[i] for i in idxs]
        metrics = metric_summary(g, p)
        acc.append(metrics["accuracy"])
        macro.append(metrics["macro_f1"])
        weighted.append(metrics["weighted_f1"])

    def ci(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]

    return {
        "seed": seed,
        "iterations": iterations,
        "accuracy_ci95": ci(acc),
        "macro_f1_ci95": ci(macro),
        "weighted_f1_ci95": ci(weighted),
    }


def compare_predictions(gold_by_id: dict[str, str], base_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], subset_ids: set[str] | None = None) -> dict[str, Any]:
    base_by_id = {norm(row.get("cluster_id")): prediction_label(row) for row in base_rows}
    new_by_id = {norm(row.get("cluster_id")): prediction_label(row) for row in new_rows}
    ids = [cid for cid in gold_by_id if cid in base_by_id and cid in new_by_id and (subset_ids is None or cid in subset_ids)]
    improved = 0
    regressed = 0
    changed = 0
    for cid in ids:
        gold = gold_by_id[cid]
        base = base_by_id[cid]
        new = new_by_id[cid]
        if base != new:
            changed += 1
        if base != gold and new == gold:
            improved += 1
        if base == gold and new != gold:
            regressed += 1
    return {
        "sample_count": len(ids),
        "label_changed_count": changed,
        "improved_count": improved,
        "regressed_count": regressed,
    }


def evaluate_frozen_predictions(output_dir: Path, gold_path: Path, freeze_manifest: dict[str, Any]) -> dict[str, Any]:
    gold = load_gold(gold_path)
    gold_by_id = dict(zip(gold["cluster_id"], gold["human_verdict"]))
    system_files = {
        "Baseline": output_dir / "baseline_predictions_500.jsonl",
        "CachedDEA": output_dir / "cached_dea_predictions_500.jsonl",
        "Financial-enhanced CachedDEA": output_dir / "financial_enhanced_predictions_500.jsonl",
        "Financial + market-rule Enhanced CachedDEA": output_dir / "market_rule_enhanced_predictions_500.jsonl",
    }
    systems: dict[str, Any] = {}
    predictions_by_system: dict[str, list[dict[str, Any]]] = {}
    for name, path in system_files.items():
        rows = read_jsonl(path)
        predictions_by_system[name] = rows
        pred_by_id = {norm(row.get("cluster_id")): prediction_label(row) for row in rows}
        joined_ids = [cid for cid in gold["cluster_id"] if cid in pred_by_id]
        y_true = [gold_by_id[cid] for cid in joined_ids]
        y_pred_raw = [pred_by_id[cid] for cid in joined_ids]
        y_pred = [item if item in LABELS else "unclear" for item in y_pred_raw]
        systems[name] = {
            "prediction_file": display_path(path),
            "prediction_sha256": sha256(path),
            **metric_summary(y_true, y_pred),
            "bootstrap_ci95": bootstrap_ci(y_true, y_pred),
        }
    market_rows = predictions_by_system["Financial + market-rule Enhanced CachedDEA"]
    audit_rows = read_jsonl(output_dir / "market_rule_audit_500.jsonl")
    effective_ids = {norm(row.get("cluster_id")) for row in audit_rows if abs(safe_float(row.get("market_score_adjustment")) or 0.0) > 0}
    explicit_ids = {norm(row.get("cluster_id")) for row in audit_rows if row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM}
    explicit_verified_ids = {
        norm(row.get("cluster_id"))
        for row in audit_rows
        if row.get("market_claim_type") == EXPLICIT_PRICE_CLAIM and row.get("market_consistency_verdict") in {"consistent", "contradictory"}
    }
    comparisons = {
        "baseline_vs_cached_dea": compare_predictions(gold_by_id, predictions_by_system["Baseline"], predictions_by_system["CachedDEA"]),
        "cached_dea_vs_financial_enhanced": compare_predictions(
            gold_by_id,
            predictions_by_system["CachedDEA"],
            predictions_by_system["Financial-enhanced CachedDEA"],
        ),
        "financial_enhanced_vs_market_rule": compare_predictions(
            gold_by_id,
            predictions_by_system["Financial-enhanced CachedDEA"],
            market_rows,
        ),
        "financial_enhanced_vs_market_rule_effective_subset": compare_predictions(
            gold_by_id,
            predictions_by_system["Financial-enhanced CachedDEA"],
            market_rows,
            subset_ids=effective_ids,
        ),
        "financial_enhanced_vs_market_rule_explicit_price_claim_subset": compare_predictions(
            gold_by_id,
            predictions_by_system["Financial-enhanced CachedDEA"],
            market_rows,
            subset_ids=explicit_ids,
        ),
        "financial_enhanced_vs_market_rule_explicit_verified_subset": compare_predictions(
            gold_by_id,
            predictions_by_system["Financial-enhanced CachedDEA"],
            market_rows,
            subset_ids=explicit_verified_ids,
        ),
    }
    market_effective_metrics: dict[str, Any] = {}
    if effective_ids:
        pred_by_id = {norm(row.get("cluster_id")): prediction_label(row) for row in market_rows}
        joined = [cid for cid in gold["cluster_id"] if cid in effective_ids and cid in pred_by_id]
        market_effective_metrics = metric_summary([gold_by_id[cid] for cid in joined], [pred_by_id[cid] for cid in joined])
    evaluation = {
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "predictions_frozen_before_gold": True,
        "prediction_freeze_manifest_sha256": sha256(output_dir / "prediction_freeze_manifest.json"),
        "freeze_manifest_status": freeze_manifest.get("status"),
        "gold_loaded_after_prediction_sha256": True,
        "training_performed": False,
        "tuning_performed": False,
        "locked500_gold_used_for_rule_or_threshold_selection": False,
        "future_market_data_used_in_closed_track": False,
        "gold": {
            "path": display_path(gold_path),
            "sha256": sha256(gold_path),
            "distribution": dict(Counter(gold["human_verdict"])),
        },
        "systems": systems,
        "comparisons": comparisons,
        "market_effective_subset_metrics": market_effective_metrics,
    }
    write_json(output_dir / "locked500_evaluation.json", evaluation)
    write_json(output_dir / "bootstrap_results.json", {name: payload["bootstrap_ci95"] for name, payload in systems.items()})
    return evaluation


def build_data_audit(
    samples: list[dict[str, Any]],
    frozen_rows: list[dict[str, Any]],
    financial_audit_rows: list[dict[str, Any]],
    market_index: MarketIndex,
    inputs: dict[str, str],
) -> dict[str, Any]:
    sample_ids = validate_unique_ids(samples, DEFAULT_INPUT)
    frozen_ids = validate_unique_ids(frozen_rows, DEFAULT_FINANCIAL_OUTPUT_DIR / "predictions_frozen_before_gold.jsonl")
    financial_ids = validate_unique_ids(financial_audit_rows, DEFAULT_FINANCIAL_OUTPUT_DIR / "financial_audit_500.jsonl")
    return {
        "locked500_rows": len(samples),
        "locked500_unique_ids": len(set(sample_ids)),
        "locked500_sha256_ok": inputs.get("locked500_sha256") == EXPECTED_LOCKED500_SHA256,
        "frozen_financial_rows": len(frozen_rows),
        "financial_audit_rows": len(financial_audit_rows),
        "sample_vs_frozen_id_match": set(sample_ids) == set(frozen_ids),
        "sample_vs_financial_audit_id_match": set(sample_ids) == set(financial_ids),
        "missing_in_frozen": sorted(set(sample_ids) - set(frozen_ids))[:20],
        "missing_in_financial_audit": sorted(set(sample_ids) - set(financial_ids))[:20],
        "market_cache_rows": market_index.rows_count,
        "market_cache_codes": len(market_index.codes),
        "market_cache_date_min": market_index.date_min,
        "market_cache_date_max": market_index.date_max,
        "training_performed": False,
        "silver_labels_modified": False,
        "gold_read_for_rule_or_threshold_selection": False,
    }


def write_manifest_and_sha(
    output_dir: Path,
    report_md: Path,
    details_xlsx: Path,
    freeze_manifest: dict[str, Any],
    evaluation: dict[str, Any] | None,
    market_summary: dict[str, Any],
    data_audit: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    run_state = {
        "status": "complete" if evaluation is not None else "predictions_frozen_no_gold_evaluation_skipped",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_performed": False,
        "model_trained": False,
        "silver_labels_modified": False,
        "heterogeneous_silver_v2_modified": False,
        "locked500_gold_used_for_rule_or_threshold_selection": False,
        "predictions_frozen_before_gold": True,
        "future_market_data_used_in_closed_track": False,
        "post_event_market_used_in_closed_track": False,
        "market_rules_fixed_before_gold": True,
    }
    write_json(output_dir / "run_state.json", run_state)
    manifest = {
        "status": run_state["status"],
        "script": "scripts/run_task2_market_rule_enhanced_cached_dea.py",
        "output_dir": display_path(output_dir),
        "report_md": display_path(report_md),
        "details_xlsx": display_path(details_xlsx),
        "inputs": inputs,
        "rule_config": RuleConfig().__dict__,
        "market_summary": market_summary,
        "data_audit": data_audit,
        "freeze_manifest_sha256": sha256(output_dir / "prediction_freeze_manifest.json"),
        "evaluation_file": display_path(output_dir / "locked500_evaluation.json") if evaluation is not None else "",
        "run_state": run_state,
    }
    write_json(output_dir / "manifest.json", manifest)
    files = [path for path in output_dir.glob("*") if path.is_file() and path.name != "SHA256SUMS.txt"]
    for extra in (report_md, details_xlsx):
        if extra.exists():
            files.append(extra)
    lines = []
    for path in sorted(set(files)):
        lines.append(f"{sha256(path)}  {display_path(path)}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def rows_for_excel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(jsonable(value), ensure_ascii=False)
            else:
                flat[key] = value
        flat_rows.append(flat)
    return flat_rows


def write_details_xlsx(path: Path, audit_rows: list[dict[str, Any]], evaluation: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    comparison_rows = []
    if evaluation is not None:
        for name, payload in evaluation.get("systems", {}).items():
            metrics_rows.append(
                {
                    "system": name,
                    "accuracy": payload.get("accuracy"),
                    "macro_f1": payload.get("macro_f1"),
                    "weighted_f1": payload.get("weighted_f1"),
                    "support_f1": (payload.get("per_class") or {}).get("support", {}).get("f1"),
                    "contradict_f1": (payload.get("per_class") or {}).get("contradict", {}).get("f1"),
                    "insufficient_f1": (payload.get("per_class") or {}).get("insufficient", {}).get("f1"),
                    "unclear_f1": (payload.get("per_class") or {}).get("unclear", {}).get("f1"),
                }
            )
        for name, payload in evaluation.get("comparisons", {}).items():
            comparison_rows.append({"comparison": name, **payload})
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(rows_for_excel(audit_rows)).to_excel(writer, sheet_name="market_rule_audit", index=False)
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame(comparison_rows).to_excel(writer, sheet_name="comparisons", index=False)


def write_report_md(
    path: Path,
    market_summary: dict[str, Any],
    data_audit: dict[str, Any],
    evaluation: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# Task2 Market-rule Enhanced CachedDEA v1",
        "",
        "本轮只实现确定性的市场事实核验规则；未训练任何使用股价特征的模型，未修改银标或人工gold。",
        "",
        "## 数据与安全门",
        f"- locked500行数：{data_audit['locked500_rows']}，SHA匹配：{data_audit['locked500_sha256_ok']}",
        f"- 已冻结financial预测行数：{data_audit['frozen_financial_rows']}，ID匹配：{data_audit['sample_vs_frozen_id_match']}",
        f"- RESSET缓存：{data_audit['market_cache_rows']}行，{data_audit['market_cache_date_min']} 至 {data_audit['market_cache_date_max']}",
        "- 训练模型：否",
        "- 使用未来行情进入封闭轨：否",
        "- locked500 gold参与规则/阈值选择：否",
        "",
        "## 市场规则覆盖",
        f"- explicit_price_claim：{market_summary['explicit_price_claim_count']}，实际完成核验：{market_summary['explicit_price_claim_verified_count']}",
        f"- financial_performance_claim：{market_summary['financial_performance_claim_count']}，股价仅作辅助证据：{market_summary['financial_claim_secondary_market_signal_only_count']}",
        f"- 市场规则实际生效：{market_summary['market_rule_effective_count']}",
        f"- claim type分布：`{json.dumps(market_summary['market_claim_type_distribution'], ensure_ascii=False)}`",
        f"- verdict分布：`{json.dumps(market_summary['market_consistency_verdict_distribution'], ensure_ascii=False)}`",
    ]
    if evaluation is not None:
        lines.extend(["", "## Locked500一次性评测"])
        for name, payload in evaluation["systems"].items():
            lines.append(
                f"- {name}: Accuracy {payload['accuracy']:.6f}, Macro-F1 {payload['macro_f1']:.6f}, "
                f"support/contradict/insufficient/unclear F1 "
                f"{payload['per_class']['support']['f1']:.6f}/"
                f"{payload['per_class']['contradict']['f1']:.6f}/"
                f"{payload['per_class']['insufficient']['f1']:.6f}/"
                f"{payload['per_class']['unclear']['f1']:.6f}"
            )
        cmp_payload = evaluation["comparisons"]["financial_enhanced_vs_market_rule"]
        explicit_cmp = evaluation["comparisons"]["financial_enhanced_vs_market_rule_explicit_verified_subset"]
        lines.extend(
            [
                "",
                "## 改动效果",
                f"- Financial vs Market-rule标签变化：{cmp_payload['label_changed_count']}，改对：{cmp_payload['improved_count']}，改错：{cmp_payload['regressed_count']}",
                f"- explicit_price_claim已核验子集改对：{explicit_cmp['improved_count']}，改错：{explicit_cmp['regressed_count']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 输出",
            f"- 输出目录：`{manifest['output_dir']}`",
            f"- 明细Excel：`{manifest['details_xlsx']}`",
            f"- SHA256：`{manifest['output_dir']}/SHA256SUMS.txt`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--financial-output-dir", type=Path, default=DEFAULT_FINANCIAL_OUTPUT_DIR)
    parser.add_argument("--market-cache", type=Path, default=DEFAULT_MARKET_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--details-xlsx", type=Path, default=DEFAULT_DETAILS_XLSX)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    input_path = resolve(args.input)
    financial_dir = resolve(args.financial_output_dir)
    market_cache_path = resolve(args.market_cache)
    output_dir = resolve(args.output_dir)
    report_md = resolve(args.report_md)
    details_xlsx = resolve(args.details_xlsx)
    gold_path = resolve(args.gold)

    inputs = {
        "locked500": display_path(input_path),
        "locked500_sha256": sha256(input_path),
        "financial_predictions_frozen_before_gold": str((financial_dir / "predictions_frozen_before_gold.jsonl").relative_to(ROOT)),
        "financial_predictions_frozen_before_gold_sha256": sha256(financial_dir / "predictions_frozen_before_gold.jsonl"),
        "financial_audit": str((financial_dir / "financial_audit_500.jsonl").relative_to(ROOT)),
        "financial_audit_sha256": sha256(financial_dir / "financial_audit_500.jsonl"),
        "market_cache": display_path(market_cache_path),
        "market_cache_sha256": sha256(market_cache_path),
    }
    if inputs["locked500_sha256"] != EXPECTED_LOCKED500_SHA256:
        raise ValueError("locked500 SHA256 mismatch; stop before prediction")

    samples = read_jsonl(input_path)
    frozen_rows = read_jsonl(financial_dir / "predictions_frozen_before_gold.jsonl")
    financial_audit_rows = read_jsonl(financial_dir / "financial_audit_500.jsonl")
    if len(samples) != 500 and args.limit == 0:
        raise ValueError(f"locked500 input must contain 500 rows, got {len(samples)}")
    market_index = build_market_index(load_market_cache(market_cache_path))
    data_audit = build_data_audit(samples, frozen_rows, financial_audit_rows, market_index, inputs)
    if not data_audit["locked500_sha256_ok"] or not data_audit["sample_vs_frozen_id_match"] or not data_audit["sample_vs_financial_audit_id_match"]:
        raise ValueError("input alignment audit failed; stop before prediction")

    outputs, market_audit_rows, market_summary = build_prediction_rows(
        samples,
        frozen_rows,
        financial_audit_rows,
        market_index,
        limit=args.limit,
    )
    if market_summary["future_market_data_used_in_closed_track_count"] != 0:
        raise ValueError("future market data entered closed track; stop before evaluation")

    write_json(output_dir / "data_audit.json", data_audit)
    write_json(output_dir / "leakage_audit.json", {
        "training_performed": False,
        "silver_labels_modified": False,
        "locked500_gold_used_for_rule_or_threshold_selection": False,
        "future_market_data_used_in_closed_track": False,
        "post_event_market_used_in_closed_track": False,
        "prediction_freeze_before_gold_required": True,
    })
    write_json(output_dir / "market_rule_summary.json", market_summary)
    freeze_manifest = freeze_predictions(output_dir, outputs, market_audit_rows, inputs, data_audit, limit=args.limit)

    evaluation = None
    if not args.no_eval and args.limit == 0:
        evaluation = evaluate_frozen_predictions(output_dir, gold_path, freeze_manifest)

    write_details_xlsx(details_xlsx, market_audit_rows, evaluation)
    manifest = write_manifest_and_sha(output_dir, report_md, details_xlsx, freeze_manifest, evaluation, market_summary, data_audit, inputs)
    write_report_md(report_md, market_summary, data_audit, evaluation, manifest)
    manifest = write_manifest_and_sha(output_dir, report_md, details_xlsx, freeze_manifest, evaluation, market_summary, data_audit, inputs)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": manifest["output_dir"],
                "training_performed": False,
                "future_market_data_used_in_closed_track": False,
                "market_summary": market_summary,
                "evaluation_available": evaluation is not None,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
