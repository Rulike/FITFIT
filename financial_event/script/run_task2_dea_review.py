#!/usr/bin/env python3
"""Run task 2 dynamic evidence acquisition gray review."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.dea import (  # noqa: E402
    DEAConfig,
    FulltextArticleIndex,
    build_review_pool,
    check_vllm_models,
    guard_output_path,
    run_dea_batch,
    sample_gray_rows,
    summarize_gray,
)
from src.event_credibility.fulltext import llm_runtime_config  # noqa: E402


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "DianJin-R1-32B"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="outputs/event_credibility/event_credibility_scores.parquet")
    parser.add_argument("--evidence", default="outputs/event_credibility/fulltext_evidence.parquet")
    parser.add_argument("--article-index", default="data/processed/event_credibility/fulltext_article_index.parquet")
    parser.add_argument("--members", default="outputs/event_clustering/event_cluster_members.parquet")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--output", default="outputs/event_credibility/dea_v1_gray")
    parser.add_argument("--sample-rows", default="")
    parser.add_argument("--baseline-results", default="")
    parser.add_argument("--previous-dea-results", default="")
    parser.add_argument("--previous-report-json", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--initial-evidence-limit", type=int, default=4)
    parser.add_argument("--additional-evidence-limit", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--trigger-policy", choices=("v1", "v2"), default="v1")
    parser.add_argument("--no-compact-prompt", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--base-url", default=os.environ.get("TASK2_LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("TASK2_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--wire-api", default=os.environ.get("TASK2_LLM_WIRE_API", "chat_completions"))
    parser.add_argument("--endpoint-style", default=os.environ.get("TASK2_LLM_ENDPOINT_STYLE", "openai_v1"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("TASK2_LLM_TIMEOUT", "180")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("TASK2_LLM_MAX_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TASK2_LLM_TEMPERATURE", "0")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("TASK2_LLM_MAX_RETRIES", "1")))
    parser.add_argument("--format-retries", type=int, default=int(os.environ.get("TASK2_LLM_FORMAT_RETRIES", "1")))
    parser.add_argument("--report-json", default="reports/task2_dea_gray_report.json")
    parser.add_argument("--report-md", default="reports/task2_dea_gray_report.md")
    parser.add_argument("--audit-jsonl", default="reports/task2_dea_manual_audit.jsonl")
    parser.add_argument("--audit-csv", default="reports/task2_dea_manual_audit.csv")
    return parser


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def resolve_outputs(output_value: str) -> tuple[Path, Path, Path]:
    path = resolve(output_value)
    if path.suffix == ".jsonl":
        out_dir = path.parent
        dea_output = path
        baseline_output = out_dir / f"{path.stem}_baseline.jsonl"
    else:
        out_dir = path
        baseline_output = out_dir / "baseline_results.jsonl"
        dea_output = out_dir / "dea_results.jsonl"
    return out_dir, baseline_output, dea_output


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    if args.max_rounds != 2:
        raise ValueError("--max-rounds is fixed at 2 for DEA")
    os.environ["TASK2_LLM_BASE_URL"] = args.base_url
    os.environ["TASK2_LLM_MODEL"] = args.model
    os.environ["TASK2_LLM_WIRE_API"] = args.wire_api
    os.environ["TASK2_LLM_ENDPOINT_STYLE"] = args.endpoint_style
    os.environ["TASK2_LLM_TIMEOUT"] = str(args.timeout)
    os.environ["TASK2_LLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["TASK2_LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["TASK2_LLM_MAX_RETRIES"] = str(args.max_retries)
    os.environ["TASK2_LLM_FORMAT_RETRIES"] = str(args.format_retries)
    os.environ["TASK2_LLM_WORKERS"] = str(args.workers)

    out_dir, baseline_output, dea_output = resolve_outputs(args.output)
    baseline_input = resolve(args.baseline_results) if args.baseline_results else baseline_output
    guarded_paths = [dea_output] + ([] if args.baseline_results else [baseline_output])
    for path in guarded_paths:
        guard_output_path(path)
        if path.exists() and not args.resume:
            raise FileExistsError(f"{path} already exists; pass --resume to continue")
    if args.baseline_results and not baseline_input.exists():
        raise FileNotFoundError(f"baseline results not found: {baseline_input}")
    model_check = check_vllm_models(args.base_url, args.endpoint_style, api_key=os.environ.get("TASK2_LLM_API_KEY", ""), timeout=min(args.timeout, 30))
    if not model_check.get("ok"):
        raise RuntimeError(f"vLLM service is not reachable: {model_check.get('error')}")

    if args.sample_rows:
        rows = read_jsonl(resolve(args.sample_rows))
        metadata = [{"cluster_id": row.get("cluster_id", ""), "sample_bucket": "from_sample_rows"} for row in rows]
        if args.limit:
            rows = rows[: args.limit]
            metadata = metadata[: args.limit]
    else:
        pool = build_review_pool(resolve(args.scores), resolve(args.evidence))
        rows, metadata = sample_gray_rows(pool, sample_count=args.limit, seed=args.seed)
    if len(rows) != args.limit:
        raise RuntimeError(f"expected {args.limit} gray samples, got {len(rows)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample_rows.jsonl"
    sample_meta_path = out_dir / "sample_metadata.json"
    sample_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(sample_meta_path, metadata)

    article_index = FulltextArticleIndex(resolve(args.article_index), members_path=resolve(args.members))
    article_index.load()
    retrieval_mode = "global_fulltext_index" if article_index.available else "cluster_pool_only"
    if article_index.error:
        print(json.dumps({"article_index_warning": article_index.error, "retrieval_mode": retrieval_mode}, ensure_ascii=False))

    llm_config = llm_runtime_config()
    dea_config = DEAConfig(
        initial_evidence_limit=args.initial_evidence_limit,
        additional_evidence_limit=args.additional_evidence_limit,
        confidence_threshold=args.confidence_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        max_rounds=args.max_rounds,
        trigger_policy=args.trigger_policy,
        compact_prompt=not args.no_compact_prompt,
    )
    startup = {
        "endpoint": llm_config.get("base_url"),
        "model": llm_config.get("model"),
        "workers": args.workers,
        "sample_count": len(rows),
        "baseline_output": str(baseline_output),
        "baseline_input": str(baseline_input),
        "dea_output": str(dea_output),
        "resume": bool(args.resume),
        "retrieval_mode": retrieval_mode,
        "trigger_policy": args.trigger_policy,
        "confidence_threshold": args.confidence_threshold,
        "low_confidence_threshold": args.low_confidence_threshold,
        "compact_prompt": not args.no_compact_prompt,
    }
    print(json.dumps(startup, ensure_ascii=False), flush=True)

    if args.baseline_results:
        baseline_summary = {
            "status": "reused",
            "total": len(rows),
            "completed": len(rows),
            "processed": len(rows),
            "succeeded": len(rows),
            "failed": 0,
            "skipped": 0,
            "elapsed": 0.0,
            "throughput_items_per_second": 0.0,
            "safety_stop_reason": "",
            "output_path": str(baseline_input),
            "checkpoint_path": "",
            "mode": "baseline",
            "workers": args.workers,
        }
        print(json.dumps({"baseline_summary": baseline_summary}, ensure_ascii=False), flush=True)
    else:
        baseline_summary = run_dea_batch(
            rows,
            baseline_output,
            llm_config,
            dea_config,
            mode="baseline",
            workers=args.workers,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            article_index=None,
        )
        print(json.dumps({"baseline_summary": baseline_summary}, ensure_ascii=False), flush=True)
        if baseline_summary.get("safety_stop_reason"):
            raise RuntimeError(f"baseline safety stopped: {baseline_summary['safety_stop_reason']}")

    dea_summary = run_dea_batch(
        rows,
        dea_output,
        llm_config,
        dea_config,
        mode="dea",
        workers=args.workers,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        article_index=article_index,
    )
    print(json.dumps({"dea_summary": dea_summary}, ensure_ascii=False), flush=True)
    if dea_summary.get("safety_stop_reason"):
        raise RuntimeError(f"DEA safety stopped: {dea_summary['safety_stop_reason']}")

    report = summarize_gray(
        baseline_input,
        dea_output,
        resolve(args.report_json),
        resolve(args.report_md),
        resolve(args.audit_jsonl),
        resolve(args.audit_csv),
        previous_dea_path=resolve(args.previous_dea_results) if args.previous_dea_results else None,
    )
    report["startup"] = startup
    report["baseline_summary"] = baseline_summary
    report["dea_summary"] = dea_summary
    report["sample_rows"] = str(sample_path)
    report["sample_metadata"] = str(sample_meta_path)
    report["article_index_available"] = bool(article_index.available)
    report["article_index_error"] = article_index.error
    if args.previous_report_json:
        previous_report_path = resolve(args.previous_report_json)
        if previous_report_path.exists():
            previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
            report["baseline_v1_v2_comparison"] = {
                "baseline": previous_report.get("baseline", report.get("baseline", {})),
                "dea_v1": previous_report.get("dea", {}),
                "dea_v2": report.get("dea", {}),
                "dea_v1_trigger_rate": previous_report.get("dea_trigger_rate", 0.0),
                "dea_v2_trigger_rate": report.get("dea_trigger_rate", 0.0),
            }
    write_json(resolve(args.report_json), report)
    print(json.dumps({"report_json": str(resolve(args.report_json)), "report_md": str(resolve(args.report_md)), "audit_jsonl": str(resolve(args.audit_jsonl)), "audit_csv": str(resolve(args.audit_csv))}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
