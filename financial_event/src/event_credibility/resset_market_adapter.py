"""RESSET daily stock market adapter for Task2 market consistency features.

The RESSET xlsx used by this project has an invalid worksheet dimension
(`A1`) while the sheet XML contains the full table.  This module intentionally
streams worksheet XML directly and does not modify the source workbook.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd


DEFAULT_MARKET_FILE = Path("data/external/market/RESSET_DRESSTK_2026__1.xlsx")
BENCHMARK_TYPE = "market_cross_sectional_median_proxy"
EXPECTED_COLUMNS = (
    "观测序号_nobs",
    "公司代码_CompanyCode",
    "证券代码_R_SecuCode",
    "上市公司代码_Comcd",
    "股票代码_Stkcd",
    "最新股票名称_Lstknm",
    "交易时股票代码_Stkcdotrd",
    "日期_Date",
    "收盘价(元)_Clpr",
    "总股数日换手率()_DFulTurnR",
    "流通股日换手率()_DTrdTurnR",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_resset_market_file(root: Path) -> Path | None:
    preferred = root / DEFAULT_MARKET_FILE
    if preferred.exists():
        return preferred
    matches = sorted((root / "data").rglob("*DRESSTK_2026*.xlsx")) if (root / "data").exists() else []
    return matches[0] if matches else None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()) or "A"
    value = 0
    for char in letters.upper():
        value = value * 26 + ord(char) - 64
    return value - 1


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    strings: list[str] = []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    for si in root.iter():
        if _local_name(si.tag) != "si":
            continue
        parts = [node.text or "" for node in si.iter() if _local_name(node.tag) == "t"]
        strings.append("".join(parts))
    return strings


def _first_sheet_path(zf: zipfile.ZipFile) -> str:
    if "xl/worksheets/sheet1.xml" in zf.namelist():
        return "xl/worksheets/sheet1.xml"
    sheets = sorted(name for name in zf.namelist() if name.startswith("xl/worksheets/") and name.endswith(".xml"))
    if not sheets:
        raise ValueError("xlsx workbook has no worksheet XML")
    return sheets[0]


def worksheet_dimension_ref(path: Path) -> str | None:
    with zipfile.ZipFile(path) as zf:
        sheet_path = _first_sheet_path(zf)
        prefix = zf.read(sheet_path)[:2048].decode("utf-8", errors="ignore")
    match = re.search(r'<dimension\s+ref="([^"]+)"', prefix)
    return match.group(1) if match else None


def iter_xlsx_rows_stream(path: Path) -> Iterator[list[Any]]:
    """Yield rows from the first worksheet without trusting worksheet dimension."""
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_path = _first_sheet_path(zf)
        with zf.open(sheet_path) as fh:
            context = ET.iterparse(fh, events=("end",))
            for _, elem in context:
                if _local_name(elem.tag) != "row":
                    continue
                values: list[Any] = []
                for cell in elem:
                    if _local_name(cell.tag) != "c":
                        continue
                    idx = _column_index(cell.attrib.get("r", "A1"))
                    while len(values) <= idx:
                        values.append(None)
                    cell_type = cell.attrib.get("t")
                    raw_value: str | None = None
                    for child in cell:
                        child_name = _local_name(child.tag)
                        if child_name == "v":
                            raw_value = child.text
                        elif child_name == "is":
                            raw_value = "".join(node.text or "" for node in child.iter() if _local_name(node.tag) == "t")
                    if cell_type == "s" and raw_value is not None:
                        value: Any = shared_strings[int(raw_value)]
                    else:
                        value = raw_value
                    values[idx] = value
                yield values
                elem.clear()


def read_resset_xlsx(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    rows = iter_xlsx_rows_stream(path)
    try:
        header = ["" if value is None else str(value).strip() for value in next(rows)]
    except StopIteration as exc:
        raise ValueError(f"empty xlsx worksheet: {path}") from exc
    if len(header) <= 1:
        raise ValueError(f"xlsx parsing failed; header has only {len(header)} column(s)")

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        if max_rows is not None and idx > max_rows:
            break
        if not any(value not in (None, "") for value in row):
            continue
        padded = list(row) + [None] * max(0, len(header) - len(row))
        records.append({header[i]: padded[i] if i < len(padded) else None for i in range(len(header))})
    return pd.DataFrame.from_records(records, columns=header)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        serial = float(text)
    except ValueError:
        return None
    if 20000 <= serial <= 60000:
        return (datetime(1899, 12, 30) + timedelta(days=serial)).date()
    return None


def _pick_column(df: pd.DataFrame, exact: str, contains: str) -> str:
    if exact in df.columns:
        return exact
    matches = [col for col in df.columns if contains in col]
    if not matches:
        raise KeyError(f"missing RESSET column containing {contains!r}")
    return matches[0]


def _normalize_code(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    return digits.zfill(6)[-6:]


def normalize_market_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    stk_col = _pick_column(raw_df, "股票代码_Stkcd", "Stkcd")
    name_col = _pick_column(raw_df, "最新股票名称_Lstknm", "Lstknm")
    date_col = _pick_column(raw_df, "日期_Date", "Date")
    close_col = _pick_column(raw_df, "收盘价(元)_Clpr", "Clpr")
    full_turnover_col = _pick_column(raw_df, "总股数日换手率()_DFulTurnR", "DFulTurnR")
    float_turnover_col = _pick_column(raw_df, "流通股日换手率()_DTrdTurnR", "DTrdTurnR")

    df = pd.DataFrame(
        {
            "trading_code": raw_df[stk_col].map(_normalize_code),
            "trading_date": raw_df[date_col].map(_coerce_date),
            "stock_name": raw_df[name_col].fillna("").astype(str),
            "close": _coerce_numeric(raw_df[close_col]),
            "turnover_full": _coerce_numeric(raw_df[full_turnover_col]),
            "turnover_float": _coerce_numeric(raw_df[float_turnover_col]),
        }
    )
    df = df[df["trading_code"].ne("") & df["trading_date"].notna()].copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df.sort_values(["trading_code", "trading_date"], inplace=True, kind="mergesort")

    duplicate_key_rows = int(df.duplicated(["trading_code", "trading_date"]).sum())
    if duplicate_key_rows:
        df = df.drop_duplicates(["trading_code", "trading_date"], keep="first").copy()

    df["valid_price"] = df["close"].notna() & (df["close"] > 0)
    df["missing_reason"] = ""
    df.loc[df["close"].isna(), "missing_reason"] = "missing_close"
    df.loc[df["close"].notna() & (df["close"] <= 0), "missing_reason"] = "invalid_close"

    valid_close = df["close"].where(df["valid_price"])
    df["previous_close"] = valid_close.groupby(df["trading_code"], sort=False).ffill().groupby(df["trading_code"], sort=False).shift(1)
    df["daily_return"] = pd.NA
    can_return = df["valid_price"] & df["previous_close"].notna() & (df["previous_close"] > 0)
    df.loc[can_return, "daily_return"] = df.loc[can_return, "close"] / df.loc[can_return, "previous_close"] - 1.0
    df["daily_return"] = pd.to_numeric(df["daily_return"], errors="coerce")

    df["market_median_return"] = df.groupby("trading_date")["daily_return"].transform("median")
    df["market_adjusted_return"] = df["daily_return"] - df["market_median_return"]

    turnover = df["turnover_float"].where(df["turnover_float"].notna(), df["turnover_full"])
    df["turnover_for_baseline"] = turnover
    df["turnover_baseline"] = (
        df.groupby("trading_code", sort=False)["turnover_for_baseline"]
        .transform(lambda s: s.shift(1).rolling(window=20, min_periods=5).median())
        .astype("float64")
    )
    df["turnover_signal_available"] = turnover.notna() & df["turnover_baseline"].notna() & (df["turnover_baseline"] > 0)
    df["turnover_abnormal_ratio"] = pd.NA
    df.loc[df["turnover_signal_available"], "turnover_abnormal_ratio"] = (
        turnover[df["turnover_signal_available"]] / df.loc[df["turnover_signal_available"], "turnover_baseline"]
    )
    df["turnover_abnormal_ratio"] = pd.to_numeric(df["turnover_abnormal_ratio"], errors="coerce")
    df["benchmark_type"] = BENCHMARK_TYPE

    keep_cols = [
        "trading_code",
        "trading_date",
        "stock_name",
        "close",
        "previous_close",
        "daily_return",
        "market_median_return",
        "market_adjusted_return",
        "turnover_full",
        "turnover_float",
        "turnover_baseline",
        "turnover_abnormal_ratio",
        "turnover_signal_available",
        "valid_price",
        "missing_reason",
        "benchmark_type",
    ]
    df = df[keep_cols].copy()
    stats = {
        "source_rows": int(len(raw_df)),
        "cache_rows": int(len(df)),
        "duplicate_trading_code_date_rows_removed": duplicate_key_rows,
        "unique_trading_codes": int(df["trading_code"].nunique()),
        "date_min": str(df["trading_date"].min()) if len(df) else "",
        "date_max": str(df["trading_date"].max()) if len(df) else "",
        "missing_close_rows": int((df["missing_reason"] == "missing_close").sum()),
        "invalid_close_rows": int((df["missing_reason"] == "invalid_close").sum()),
        "valid_price_rows": int(df["valid_price"].sum()),
        "fields": keep_cols,
        "benchmark_type": BENCHMARK_TYPE,
    }
    return df, stats


def build_resset_market_cache(source_xlsx: Path, cache_parquet: Path, manifest_path: Path) -> dict[str, Any]:
    source_sha_before = sha256_file(source_xlsx)
    dimension_ref = worksheet_dimension_ref(source_xlsx)
    raw_df = read_resset_xlsx(source_xlsx)
    if raw_df.shape[0] <= 1 or raw_df.shape[1] <= 1:
        raise ValueError(f"RESSET xlsx parse failed: shape={raw_df.shape}")
    cache_df, stats = normalize_market_frame(raw_df)
    cache_parquet.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(cache_parquet, index=False)
    source_sha_after = sha256_file(source_xlsx)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(source_xlsx),
        "source_sha256": source_sha_before,
        "source_sha256_after_read": source_sha_after,
        "source_file_modified": source_sha_before != source_sha_after,
        "worksheet_dimension_ref": dimension_ref,
        "excel_dimension_recovered": dimension_ref == "A1" and stats["source_rows"] > 1 and len(raw_df.columns) > 1,
        "cache_file": str(cache_parquet),
        "cache_sha256": sha256_file(cache_parquet),
        "expected_columns_present": [col for col in EXPECTED_COLUMNS if col in raw_df.columns],
        **stats,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_market_cache(cache_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache_parquet)
    df["trading_code"] = df["trading_code"].astype(str).str.zfill(6)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    return df
