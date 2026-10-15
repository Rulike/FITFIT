#!/usr/bin/env python3
"""Build the shadow LLM review file for Step 4.5-C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.task2_final_cached_dea_score_shadow_common import (
    FINAL_RESULTS_260,
    FORMAL_LLM_RESULTS,
    SHADOW_LLM_RESULTS,
    build_shadow_llm_rows,
    load_final_rows,
    load_formal_llm_rows,
    resolve,
    write_json,
    write_jsonl,
)


def build_shadow_file(
    formal_llm_results_path: Path = FORMAL_LLM_RESULTS,
    final_results_path: Path = FINAL_RESULTS_260,
    output_path: Path = SHADOW_LLM_RESULTS,
) -> dict[str, Any]:
    formal_rows = load_formal_llm_rows(formal_llm_results_path)
    final_rows = load_final_rows(final_results_path)
    shadow_rows, summary = build_shadow_llm_rows(formal_rows, final_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, shadow_rows)
    summary.update(
        {
            "formal_llm_results_path": str(formal_llm_results_path),
            "final_results_path": str(final_results_path),
            "output_path": str(output_path),
            "shadow_rows_written": len(shadow_rows),
            "shadow_verdict_distribution": summary.get("verdict_distribution", {}),
        }
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-llm-results", type=Path, default=FORMAL_LLM_RESULTS)
    parser.add_argument("--final-results", type=Path, default=FINAL_RESULTS_260)
    parser.add_argument("--output", type=Path, default=SHADOW_LLM_RESULTS)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_shadow_file(
        resolve(args.formal_llm_results),
        resolve(args.final_results),
        resolve(args.output),
    )
    if args.manifest is not None:
        write_json(resolve(args.manifest), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
