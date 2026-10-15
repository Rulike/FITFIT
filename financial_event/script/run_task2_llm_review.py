#!/usr/bin/env python3
"""Run task 2 LLM review against the existing fulltext review queue."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.fulltext import (
    fuse_fulltext_scores,
    llm_runtime_config,
    read_llm_results,
    run_llm_review,
    summarize_fulltext,
    write_fulltext_report,
)
from src.event_credibility.scoring import write_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_scores", default="outputs/event_credibility/event_credibility_scores.parquet")
    parser.add_argument("--evidence", default="outputs/event_credibility/fulltext_evidence.parquet")
    parser.add_argument("--queue", default="data/processed/event_credibility/llm_review_queue.jsonl")
    parser.add_argument("--output_dir", default="outputs/event_credibility")
    parser.add_argument("--output", default="")
    parser.add_argument("--report_dir", default="reports/event_credibility")
    parser.add_argument("--max_llm_reviews", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm_sleep_seconds", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("TASK2_LLM_WORKERS", "1")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=-1)
    parser.add_argument("--format_retries", type=int, default=None)
    parser.add_argument("--retry_backoff_seconds", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--base_url", default="")
    parser.add_argument("--wire_api", choices=["chat_completions", "responses"], default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint_style", choices=["openai_v1", "direct"], default="")
    parser.add_argument("--try_shell_keys", action="store_true")
    parser.add_argument("--api_key_stdin", action="store_true")
    parser.add_argument("--api_key_getpass", action="store_true")
    return parser


def collect_shell_key_candidates() -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen = set()
    for key_name in ("TASK2_LLM_API_KEY", "MDKJ_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(key_name)
        if value and value not in seen:
            candidates.append((f"env:{key_name}", value))
            seen.add(value)
    shell_files = [Path.home() / ".bashrc", Path.home() / ".profile", Path.home() / ".zshrc"]
    pattern = re.compile(r"""(?:export\s+)?(TASK2_LLM_API_KEY|MDKJ_API_KEY|OPENAI_API_KEY)\s*=\s*['"]?([^'"\s]+)""")
    for path in shell_files:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(2)
            if value and value not in seen:
                candidates.append((f"{path.name}:{line_no}:{match.group(1)}", value))
                seen.add(value)
    return candidates


def select_working_key(queue_path: Path, model: str) -> str:
    original_key = os.environ.get("TASK2_LLM_API_KEY")
    probe_path = Path("/tmp/task2_llm_key_probe.jsonl")
    for label, key in collect_shell_key_candidates():
        os.environ["TASK2_LLM_API_KEY"] = key
        summary = run_llm_review(queue_path, probe_path, max_reviews=1, sleep_seconds=0)
        if summary.get("reviewed_count", 0) > 0:
            print(json.dumps({"api_key_source": label, "probe_status": "accepted", "model": model}, ensure_ascii=False))
            return label
    if original_key:
        os.environ["TASK2_LLM_API_KEY"] = original_key
    else:
        os.environ.pop("TASK2_LLM_API_KEY", None)
    return ""


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    out_dir = ROOT / args.output_dir
    report_dir = ROOT / args.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.base_url:
        os.environ["TASK2_LLM_BASE_URL"] = args.base_url
    if args.wire_api:
        os.environ["TASK2_LLM_WIRE_API"] = args.wire_api
    if args.model:
        os.environ["TASK2_LLM_MODEL"] = args.model
    if args.endpoint_style:
        os.environ["TASK2_LLM_ENDPOINT_STYLE"] = args.endpoint_style
    if args.max_retries >= 0:
        os.environ["TASK2_LLM_MAX_RETRIES"] = str(args.max_retries)
    if args.format_retries is not None:
        os.environ["TASK2_LLM_FORMAT_RETRIES"] = str(args.format_retries)
    if args.retry_backoff_seconds is not None:
        os.environ["TASK2_LLM_RETRY_BACKOFF_SECONDS"] = str(args.retry_backoff_seconds)
    if args.max_tokens is not None:
        os.environ["TASK2_LLM_MAX_TOKENS"] = str(args.max_tokens)
    if args.temperature is not None:
        os.environ["TASK2_LLM_TEMPERATURE"] = str(args.temperature)
    if args.timeout is not None:
        os.environ["TASK2_LLM_TIMEOUT"] = str(args.timeout)
    if args.api_key_stdin:
        api_key = sys.stdin.read().strip()
        if api_key:
            os.environ["TASK2_LLM_API_KEY"] = api_key
    if args.api_key_getpass and not os.environ.get("TASK2_LLM_API_KEY"):
        os.environ["TASK2_LLM_API_KEY"] = getpass.getpass("TASK2_LLM_API_KEY: ")
    if args.try_shell_keys:
        select_working_key(ROOT / args.queue, os.environ.get("TASK2_LLM_MODEL", args.model or "gpt-4o-mini"))

    review_limit = args.limit if args.limit is not None else args.max_llm_reviews
    llm_results_path = Path(args.output) if args.output else out_dir / "llm_review_results.jsonl"
    llm_results_path = llm_results_path if llm_results_path.is_absolute() else ROOT / llm_results_path
    queue_path = ROOT / args.queue
    with queue_path.open(encoding="utf-8") as f:
        queue_count = sum(1 for line in f if line.strip())
    input_count = min(review_limit, queue_count) if review_limit > 0 else queue_count
    runtime = llm_runtime_config()
    print(
        json.dumps(
            {
                "endpoint": runtime.get("base_url", ""),
                "model": runtime.get("model", ""),
                "workers": args.workers,
                "input_count": input_count,
                "output_path": str(llm_results_path),
                "resume": bool(args.resume),
            },
            ensure_ascii=False,
        )
    )
    llm_summary = run_llm_review(
        queue_path,
        llm_results_path,
        max_reviews=review_limit,
        sleep_seconds=args.llm_sleep_seconds,
        workers=args.workers,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    review_only = bool(args.output)
    if review_only:
        summary = llm_summary
        (report_dir / "llm_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        scores = pd.read_parquet(ROOT / args.base_scores)
        evidence = pd.read_parquet(ROOT / args.evidence)
        llm_results = read_llm_results(llm_results_path)
        final_scores = fuse_fulltext_scores(scores, evidence, llm_results)
        final_scores.to_parquet(out_dir / "event_credibility_scores_fulltext.parquet", index=False)
        final_scores.to_excel(out_dir / "event_credibility_scores_fulltext.xlsx", index=False)
        write_jsonl(out_dir / "event_credibility_scores_fulltext.jsonl", final_scores.to_dict("records"))

        summary = summarize_fulltext(final_scores, llm_summary)
        (report_dir / "fulltext_credibility_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_fulltext_report(report_dir, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
