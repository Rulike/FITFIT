#!/usr/bin/env python3
"""Build the Task2 CSMAR financial cache v2 from the fixed FI_T*.xlsx files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.csmar_adapter import build_csmar_financial_cache_v2  # noqa: E402


DEFAULT_SOURCE_DIR = ROOT / "data/external/csmar/财务指标"
DEFAULT_CACHE_DIR = ROOT / "data/processed/event_credibility/csmar_financial_cache_v2"
DEFAULT_OUTPUT_MANIFEST = ROOT / "outputs/event_credibility/task2_financial_enhanced_cached_dea_v2/csmar_cache_manifest.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = build_csmar_financial_cache_v2(resolve(args.source_dir), resolve(args.cache_dir), force=args.force)
    write_json(resolve(args.output_manifest), manifest)
    print(json.dumps({
        "status": "ok",
        "cache_dir": str(resolve(args.cache_dir)),
        "rows": manifest.get("outputs", {}).get("financial_records_rows"),
        "source_sha_match": manifest.get("source_sha_match"),
        "excel_dimension_recovered": manifest.get("excel_dimension_recovered"),
        "output_manifest": str(resolve(args.output_manifest)),
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
