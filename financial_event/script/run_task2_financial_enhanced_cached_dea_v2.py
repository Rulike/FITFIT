#!/usr/bin/env python3
"""Run the deterministic Task2 financial consistency enhancement v2.

This script does not read independent500 gold labels. It builds/loads the real
CSMAR cache, attaches financial_context to the unlabeled independent500 rows,
and writes an audit trail. If the frozen Task2 LLM service is unavailable, it
stops before prediction and records a blocked run_state instead of fabricating
model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility import financial_consistency, fulltext  # noqa: E402
from src.event_credibility.csmar_adapter import (  # noqa: E402
    build_csmar_financial_cache_v2,
    load_csmar_financial_records_v2,
)


DEFAULT_CONFIG = ROOT / "configs/task2_financial_consistency_v2.yaml"
DEFAULT_FINAL_CACHED_CONFIG = ROOT / "configs/task2_final_cached_dea_v1.yaml"
DEFAULT_INPUT = ROOT / "outputs/event_credibility/task2_independent500_v1/independent_samples.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/event_credibility/task2_financial_enhanced_cached_dea_v2"
DEFAULT_REPORT_MD = ROOT / "reports/task2_financial_enhanced_independent500_report.md"
DEFAULT_DETAILS_XLSX = ROOT / "reports/task2_financial_enhanced_independent500_details.xlsx"
MODEL_START_COMMAND = (
    "CUDA_VISIBLE_DEVICES=0,1,2,3 "
    "/root/miniconda3/envs/dianjin-cu128/bin/vllm serve "
    "/opt/data/private/JinRong/models/DianJin-R1-32B "
    "--served-model-name DianJin-R1-32B --tensor-parallel-size 4 --dtype bfloat16 "
    "--max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.90 "
    "--disable-custom-all-reduce --host 127.0.0.1 --port 8000"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(obj)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def evidence_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in row.get("evidence_items") or []:
        if not isinstance(item, dict):
            continue
        parts.append(" ".join(str(item.get(field, "")) for field in ("title", "evidence_text", "sentence", "content") if str(item.get(field, "")).strip()))
    return " || ".join(parts)


def publish_time(row: dict[str, Any]) -> str:
    return fulltext.normalize_text(row.get("earliest_publish_time") or row.get("latest_publish_time") or row.get("event_time") or "")


def report_period_hint(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(field, "")) for field in ("event_claim", "title_samples", "event_name"))
    extracted = financial_consistency.extract_report_period(text)
    return extracted if financial_consistency._is_report_period(extracted) else ""


def analyze_rows(rows: list[dict[str, Any]], csmar_records: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    selected = rows[:limit] if limit and limit > 0 else rows
    stock_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in csmar_records:
        code = fulltext.normalize_text(record.get("stock_code"))
        if code:
            stock_index[code].append(record)
    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        title = str(row.get("title_samples") or row.get("event_name") or "")
        body = evidence_text(row)
        claim_text = " ".join(part for part in [str(row.get("event_claim") or ""), title, body] if part)
        row_code = fulltext.normalize_text(row.get("target_code") or "")
        row_records = stock_index.get(row_code) or csmar_records
        context = financial_consistency.analyze_financial_consistency(
            news_id=row.get("cluster_id"),
            title=title,
            body=body,
            company_name=row.get("target_name") or "",
            stock_code=row.get("target_code") or "",
            publish_time=publish_time(row),
            financial_claim_text=claim_text,
            report_period=report_period_hint(row),
            csmar_records=row_records,
        )
        merged = {
            "input_index": idx,
            "cluster_id": row.get("cluster_id"),
            "target_code": row.get("target_code"),
            "target_name": row.get("target_name"),
            "event_claim": row.get("event_claim"),
            "publish_time": publish_time(row),
            "financial_context": context,
            "financial_prompt_block": financial_consistency.format_financial_prompt_block(context.get("prompt_summary") or {}),
            **financial_consistency.merge_with_original_scores(row, context),
        }
        enriched.append(merged)
    return enriched


def summarize(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = [row.get("financial_context") or {} for row in enriched]
    claims = [claim for ctx in contexts for claim in (ctx.get("financial_claims") or [])]
    facts = [fact for ctx in contexts for fact in (ctx.get("matched_financial_facts") or [])]
    rule_hits = Counter(str(claim.get("rule_key") or claim.get("metric") or "unknown") for claim in claims)
    table_hits = Counter(str(fact.get("csmar_table") or fact.get("matched_table_name") or "unknown") for fact in facts if str(fact.get("claim_verdict") or "") != "insufficient_data")
    temporal_uncertain = sum(
        1
        for fact in facts
        if (fact.get("temporal_availability") or {}).get("status") == "temporal_availability_uncertain"
    )
    evidence_complete = 0
    for fact in facts:
        if all(fact.get(field) for field in ("claim_text", "metric", "csmar_table", "field_code", "current_period", "source_file", "source_sha256")):
            evidence_complete += 1
    return {
        "input_samples": len(enriched),
        "successful_stock_code_rows": sum(1 for row in enriched if fulltext.normalize_text(row.get("target_code"))),
        "successful_csmar_company_match_rows": sum(1 for ctx in contexts if (ctx.get("data_availability") or {}).get("has_candidate_records")),
        "financial_narrative_rows": sum(1 for ctx in contexts if int(ctx.get("financial_claim_count") or 0) > 0),
        "financial_claim_count": len(claims),
        "financial_evidence_claim_count": sum(1 for ctx in contexts for _ in range(int(ctx.get("financial_matched_claim_count") or 0))),
        "financial_consistency_verdict_distribution": dict(Counter(str(ctx.get("financial_consistency_verdict") or "missing") for ctx in contexts)),
        "consistent_count": sum(1 for ctx in contexts if ctx.get("financial_consistency_verdict") == "consistent"),
        "partially_consistent_count": sum(1 for ctx in contexts if ctx.get("financial_consistency_verdict") == "partially_consistent"),
        "contradictory_count": sum(1 for ctx in contexts if ctx.get("financial_consistency_verdict") == "contradictory"),
        "insufficient_data_count": sum(1 for ctx in contexts if ctx.get("financial_consistency_verdict") == "insufficient_data"),
        "temporal_availability_uncertain_count": temporal_uncertain,
        "future_financial_data_used_count": sum(1 for ctx in contexts if ctx.get("future_financial_data_used")),
        "rule_hit_counts": dict(rule_hits),
        "csmar_table_usage_counts": dict(table_hits),
        "evidence_chain_complete_count": evidence_complete,
        "evidence_chain_total_count": len(facts),
        "financial_consistency_score_min": min([float(ctx.get("financial_consistency_score") or 0.5) for ctx in contexts] or [0.5]),
        "financial_consistency_score_max": max([float(ctx.get("financial_consistency_score") or 0.5) for ctx in contexts] or [0.5]),
        "neutral_no_data_rows": sum(1 for ctx in contexts if ctx.get("financial_consistency_verdict") == "insufficient_data" and float(ctx.get("financial_consistency_score") or 0.0) == 0.5 and float(ctx.get("financial_score_adjustment") or 0.0) == 0.0),
    }


def check_model_service(final_config: Path) -> dict[str, Any]:
    config = load_yaml(final_config)
    api_base = str(config.get("api_base") or "http://127.0.0.1:8000/v1")
    model_name = str(config.get("model_name") or "DianJin-R1-32B")
    url = fulltext.build_endpoint_url(api_base, "models", endpoint_style=str(config.get("endpoint_style", "openai_v1")))
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        return {
            "available": False,
            "api_base": api_base,
            "model_name": model_name,
            "error": f"{type(exc).__name__}: {exc}",
            "pending_command": MODEL_START_COMMAND,
        }
    names = [str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)]
    return {
        "available": model_name in names,
        "api_base": api_base,
        "model_name": model_name,
        "model_names": names,
        "pending_command": "" if model_name in names else MODEL_START_COMMAND,
    }


def write_sha256s(output_dir: Path, extra_paths: list[Path]) -> None:
    paths = [path for path in output_dir.glob("*") if path.is_file()] + [path for path in extra_paths if path.exists()]
    lines = []
    for path in sorted(set(paths)):
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        lines.append(f"{sha256(path)}  {rel}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any], service: dict[str, Any]) -> None:
    s = report["summary"]
    lines = [
        "# Task2 Financial Enhanced CachedDEA v2 Independent500 Dry Run",
        "",
        "评测集口径：500条auto-assisted、human-confirmed、independent-evaluation-only评测集；本脚本未读取gold标签。",
        "",
        f"- CSMAR真实读取：{report['csmar']['source_sha_match']}",
        f"- Excel dimension恢复：{report['csmar']['excel_dimension_recovered']}",
        f"- 缓存行数：{report['csmar']['financial_records_rows']}",
        f"- 输入样本数：{s['input_samples']}",
        f"- 股票代码覆盖：{s['successful_stock_code_rows']} / {s['input_samples']}",
        f"- CSMAR公司匹配：{s['successful_csmar_company_match_rows']} / {s['input_samples']}",
        f"- 财务claim覆盖样本：{s['financial_narrative_rows']} / {s['input_samples']}",
        f"- claim数量：{s['financial_claim_count']}",
        f"- 有证据claim数量：{s['financial_evidence_claim_count']}",
        f"- verdict分布：{json.dumps(s['financial_consistency_verdict_distribution'], ensure_ascii=False)}",
        f"- 未来数据使用数：{s['future_financial_data_used_count']}",
        f"- 模型服务可用：{service.get('available')}",
        "",
        "## Blocker",
        f"- blocked_stage：{report['run_state'].get('blocked_stage', '')}",
        f"- blocker_reason：{report['run_state'].get('blocker_reason', '')}",
        f"- pending_command：`{report['run_state'].get('pending_command', '')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--final-cached-config", type=Path, default=DEFAULT_FINAL_CACHED_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    config = load_yaml(resolve(args.config))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = config.get("paths") or {}
    cache_dir = resolve(paths.get("csmar_cache_dir", "data/processed/event_credibility/csmar_financial_cache_v2"))
    source_dir = resolve(paths.get("csmar_source_dir", "data/external/csmar/财务指标"))
    cache_manifest = build_csmar_financial_cache_v2(source_dir, cache_dir, force=args.force_cache)
    write_json(output_dir / "csmar_cache_manifest.json", cache_manifest)

    rows = read_jsonl(resolve(args.input))
    if len(rows) != 500 and args.limit == 0:
        raise ValueError(f"independent500 input must contain 500 rows, got {len(rows)}")
    csmar_records = load_csmar_financial_records_v2(cache_dir)
    enriched = analyze_rows(rows, csmar_records, limit=args.limit)
    write_jsonl(output_dir / "financial_audit_500.jsonl", enriched)
    summary = summarize(enriched)
    service = check_model_service(resolve(args.final_cached_config))
    blocked = not service.get("available")
    run_state = {
        "status": "blocked" if blocked else "financial_dry_run_complete_prediction_pending",
        "blocked_stage": "frozen_cached_dea_prediction" if blocked else "",
        "blocker_reason": "local Task2 model service unavailable" if blocked else "",
        "completed_stages": ["csmar_cache", "financial_consistency_dry_run"],
        "pending_command": service.get("pending_command", ""),
        "no_fake_metrics": True,
        "seed": 2026,
        "gold_used_for_development": False,
        "prediction_frozen_before_gold_join": False,
        "formal_task2_outputs_modified": False,
        "baseline_predictions_modified": False,
        "csmar_real_data_loaded": True,
        "csmar_source_sha_match": bool(cache_manifest.get("source_sha_match")),
        "excel_dimension_recovered": bool(cache_manifest.get("excel_dimension_recovered")),
        "stock_code_leading_zero_preserved": bool(cache_manifest.get("stock_code_leading_zero_preserved")),
        "future_financial_data_used": bool(summary.get("future_financial_data_used_count")),
    }
    report = {
        "status": "financial_dry_run_complete_model_blocked" if blocked else "financial_dry_run_complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(resolve(args.config)),
        "input": str(resolve(args.input)),
        "output_dir": str(output_dir),
        "csmar": {
            "source_sha_match": cache_manifest.get("source_sha_match"),
            "excel_dimension_recovered": cache_manifest.get("excel_dimension_recovered"),
            "financial_records_rows": cache_manifest.get("outputs", {}).get("financial_records_rows"),
            "financial_records_columns": cache_manifest.get("outputs", {}).get("financial_records_columns"),
            "field_mapping": cache_manifest.get("field_mapping"),
            "missing_raw_fields": cache_manifest.get("missing_raw_fields"),
        },
        "summary": summary,
        "model_service": service,
        "run_state": run_state,
    }
    write_json(output_dir / "financial_audit_report.json", report)
    write_json(output_dir / "run_state.json", run_state)
    if args.limit <= 0:
        pd.DataFrame(enriched).to_excel(DEFAULT_DETAILS_XLSX, index=False)
    write_markdown(DEFAULT_REPORT_MD, report, service)
    write_sha256s(output_dir, [DEFAULT_REPORT_MD, DEFAULT_DETAILS_XLSX])
    print(json.dumps({"status": report["status"], "summary": summary, "run_state": run_state}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
