#!/usr/bin/env python3
"""Build normalized RESSET market cache for Task2 market consistency v1."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.resset_market_adapter import (  # noqa: E402
    build_resset_market_cache,
    find_resset_market_file,
    sha256_file,
)


OUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3/market_consistency_v1"
CACHE_FILE = OUT_DIR / "resset_market_daily_cache.parquet"
MANIFEST_FILE = OUT_DIR / "resset_market_cache_manifest.json"
RUN_STATE_FILE = OUT_DIR / "run_state.json"


def write_run_state(state: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = find_resset_market_file(ROOT)
    if source is None:
        write_run_state(
            {
                "status": "blocked",
                "blocked_stage": "market_file_discovery",
                "blocker_reason": "RESSET_DRESSTK_2026 xlsx not found under data/",
                "training_performed": False,
                "tuning_performed": False,
                "formal_outputs_modified": False,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        print("RESSET market xlsx not found under data/")
        return 2

    try:
        manifest = build_resset_market_cache(source, CACHE_FILE, MANIFEST_FILE)
    except Exception as exc:
        write_run_state(
            {
                "status": "blocked",
                "blocked_stage": "market_cache_build",
                "blocker_reason": str(exc),
                "source_file": str(source),
                "source_sha256": sha256_file(source) if source.exists() else "",
                "training_performed": False,
                "tuning_performed": False,
                "formal_outputs_modified": False,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        raise

    validations = {
        "real_market_rows_expected_299860": manifest.get("source_rows") == 299860,
        "unique_stock_codes_expected_5003": manifest.get("unique_trading_codes") == 5003,
        "date_min_expected_2026_04_01": manifest.get("date_min") == "2026-04-01",
        "date_max_expected_2026_06_30": manifest.get("date_max") == "2026-06-30",
        "cache_key_unique": manifest.get("duplicate_trading_code_date_rows_removed") == 0,
        "source_file_unmodified": not manifest.get("source_file_modified"),
        "excel_dimension_recovered": bool(manifest.get("excel_dimension_recovered")),
    }
    state = {
        "status": "completed" if all(validations.values()) else "completed_with_validation_warnings",
        "stage": "market_cache_build",
        "validations": validations,
        "source_file": str(source),
        "source_sha256": manifest.get("source_sha256"),
        "cache_file": str(CACHE_FILE),
        "cache_sha256": manifest.get("cache_sha256"),
        "training_performed": False,
        "tuning_performed": False,
        "formal_outputs_modified": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_run_state(state)
    print(json.dumps({"cache_manifest": str(MANIFEST_FILE), "run_state": str(RUN_STATE_FILE), **validations}, ensure_ascii=False, indent=2))
    return 0 if all(validations.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
