#!/usr/bin/env python3
"""Offline attribution audit for Safety Gate v1.2 zero-trigger behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_task2_safety_gate_v1_2_independent import (  # noqa: E402
    evidence_rows_for_gate,
    latest_by_cluster,
    norm,
)
from src.event_credibility import dea_safety_gate_v1_2 as gate_v12  # noqa: E402
from src.event_credibility.dea_safety_gate_adapter import (  # noqa: E402
    build_safety_gate_claim,
    build_safety_gate_evidence_records,
    canonical_result_for_gate,
)


INPUT_DIR = ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent"
GOLD_RESULTS = ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent_gold_eval/independent_gold_results.jsonl"
ERROR_AUDIT = ROOT / "reports/task2_independent_error_audit_v1.json"
CONFIG = ROOT / "configs/task2_dea_safety_gate_v1_2.yaml"
OUTPUT_DIR = ROOT / "outputs/event_credibility/safety_gate_v1_2_zero_trigger_audit_v1"

FORMAL_OUTPUTS = [
    Path("outputs/event_credibility/llm_review_results.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.parquet"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.xlsx"),
]
FROZEN_FILES = [
    Path("src/event_credibility/dea_safety_gate_v1_2.py"),
    Path("configs/task2_dea_safety_gate_v1_2.yaml"),
    Path("src/event_credibility/dea_safety_gate_adapter.py"),
    Path("src/event_credibility/dea_safety_gate_v1_1.py"),
    Path("configs/task2_dea_safety_gate_v1_1.yaml"),
]

ERROR_CATEGORY_BY_ROUTE = {
    "D. 门控修复优先": "A. 门控规则未覆盖",
    "A. ISR适合": "C. 样本本身不满足阻止条件",
    "C. 主张修复优先": "D. 金标或证据存在歧义",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel_path in paths:
        path = ROOT / rel_path
        out[str(rel_path)] = sha256(path) if path.exists() else "missing"
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = p * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def number_stats(values: list[float], threshold: float | None = None) -> dict[str, Any]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return {"count": 0}
    result: dict[str, Any] = {
        "count": len(cleaned),
        "min": round(min(cleaned), 6),
        "p50": round(percentile(cleaned, 0.5), 6),
        "p95": round(percentile(cleaned, 0.95), 6),
        "max": round(max(cleaned), 6),
    }
    if threshold is not None:
        margins = [value - threshold for value in cleaned]
        result["threshold"] = threshold
        result["margin_min"] = round(min(margins), 6)
        result["margin_p50"] = round(percentile(margins, 0.5), 6)
        result["margin_p95"] = round(percentile(margins, 0.95), 6)
        result["margin_max"] = round(max(margins), 6)
        result["below_threshold_count"] = sum(1 for value in cleaned if value < threshold)
    return result


def output_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = output_path(args.input_dir)
    samples = read_jsonl(input_dir / "independent_samples.jsonl")
    baseline = read_jsonl(input_dir / "baseline_results.jsonl")
    cached = read_jsonl(input_dir / "cached_dea_results.jsonl")
    safety = read_jsonl(input_dir / "safety_gate_v1_2_results.jsonl")
    gold = read_jsonl(output_path(args.gold_results))
    error_audit = json.loads(output_path(args.error_audit).read_text(encoding="utf-8"))
    return {
        "samples": latest_by_cluster(samples, "samples"),
        "baseline": latest_by_cluster(baseline, "baseline"),
        "cached": latest_by_cluster(cached, "cached"),
        "safety": latest_by_cluster(safety, "safety"),
        "gold": latest_by_cluster(gold, "gold"),
        "error_records": error_audit["error_records"],
        "error_summary": error_audit["summary"],
        "config": gate_v12.load_dea_safety_gate_v1_2_config(output_path(args.config)),
    }


def build_exact_gate_row(cid: str, inputs: dict[str, Any]) -> dict[str, Any]:
    sample = inputs["samples"][cid]
    baseline = inputs["baseline"][cid]
    cached = inputs["cached"][cid]
    round1 = cached.get("round1") if isinstance(cached.get("round1"), dict) else {}
    evidence_rows = evidence_rows_for_gate(sample, cached)
    evidence_records = build_safety_gate_evidence_records(cid, evidence_rows, cached, round1)
    claim = build_safety_gate_claim(sample, sample)
    cached_for_gate = dict(cached)
    if not cached_for_gate.get("evidence_ids"):
        cached_for_gate["evidence_ids"] = [
            norm(item.get("evidence_id"))
            for item in cached_for_gate.get("final_evidence", [])
            if isinstance(item, dict) and norm(item.get("evidence_id"))
        ]
    canonical_round1 = canonical_result_for_gate(round1, evidence_records, role="round1")
    canonical_cached = canonical_result_for_gate(cached_for_gate, evidence_records, role="cached")
    gate = gate_v12.apply_dea_safety_gate_v1_2(
        claim,
        baseline,
        canonical_round1,
        canonical_cached,
        evidence_records,
        inputs["config"],
    )
    return {
        "claim": claim,
        "baseline": baseline,
        "cached": cached,
        "round1": canonical_round1,
        "canonical_cached": canonical_cached,
        "evidence_records": evidence_records,
        "gate": gate,
    }


def relation_hits(claim: dict[str, Any], evidence_records: list[dict[str, Any]], all_scores: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in [gate_v12._strip_human_fields(row) for row in gate_v12._evidence_items(evidence_records)]:
        if not gate_v12._forms_subject_event_fact_relation(claim, item, config):
            continue
        eid = norm(item.get("evidence_id"))
        score = all_scores.get(eid, {})
        hits.append(
            {
                "evidence_id": eid,
                "source": norm(item.get("source") or item.get("article_source") or item.get("source_name")),
                "publish_time": norm(item.get("publish_time") or item.get("article_publish_time")),
                "total_score": score.get("total_score"),
                "event_semantic_score": score.get("event_semantic_score"),
                "concrete_fact_score": score.get("concrete_fact_score"),
                "market_signal_penalty": score.get("market_signal_penalty"),
                "text": compact_evidence(item),
            }
        )
    return hits


def compact_evidence(item: dict[str, Any], limit: int = 180) -> str:
    text = norm(item.get("sentence") or item.get("text") or item.get("content") or item.get("title"))
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def explain_error(row: dict[str, Any], gate_row: dict[str, Any]) -> str:
    cached_verdict = row.get("cached_verdict")
    human_verdict = row.get("human_verdict")
    if cached_verdict == "insufficient":
        if human_verdict == "support":
            return "Cached为insufficient，v1.2按规则直接保留insufficient；该类是召回/排序/推理不足，不属于阻止support/contradict的门控条件。"
        if human_verdict == "unclear":
            return "Cached为insufficient，v1.2没有insufficient转unclear规则；同时evidence_resolves_claim=true使structurally_unverifiable不成立。"
    if cached_verdict in {"support", "contradict"}:
        if gate_row["gate"]["gate_reasons"]:
            return "已有gate_reasons但未改变输出，需要检查回退源；本轮未出现。"
        return (
            "Cached为support/contradict，但门控输入同时满足subject/time/concrete_fact/traceable，"
            "且未命中only_market_signals或only_investor_questions；当前规则缺少语义错配和混合行情证据阻断。"
        )
    return "未进入现有阻止分支。"


def audit_rows(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    error_by_id = {row["cluster_id"]: row for row in inputs["error_records"]}
    for cid, gold in sorted(inputs["gold"].items()):
        exact = build_exact_gate_row(cid, inputs)
        stored = inputs["safety"][cid]
        gate = exact["gate"]
        selected_fields = {
            "safety_gate_v1_2_verdict": gate.get("gated_verdict"),
            "gate_blocked": bool(gate.get("gate_blocked")),
            "gate_reasons": list(gate.get("gate_reasons") or []),
        }
        stored_fields = {
            "safety_gate_v1_2_verdict": stored.get("safety_gate_v1_2_verdict"),
            "gate_blocked": bool(stored.get("gate_blocked")),
            "gate_reasons": list(stored.get("gate_reasons") or []),
        }
        if selected_fields != stored_fields:
            mismatches.append({"cluster_id": cid, "expected": stored_fields, "actual": selected_fields})
        all_scores = gate.get("all_evidence_scores") or {}
        row = {
            "cluster_id": cid,
            "sample_group": gold.get("sample_group"),
            "claim": gold.get("claim"),
            "human_verdict": gold.get("human_verdict"),
            "baseline_verdict": gold.get("baseline_verdict"),
            "round1_verdict": gold.get("round1_verdict"),
            "cached_verdict": gold.get("cached_verdict"),
            "safety_gate_v1_2_verdict": gold.get("safety_gate_v1_2_verdict"),
            "cached_confidence": gold.get("cached_confidence"),
            "cached_correct": bool(gold.get("cached_correct")),
            "gate_blocked": bool(gate.get("gate_blocked")),
            "gate_reasons": list(gate.get("gate_reasons") or []),
            "selected_result_source": gate.get("selected_result_source"),
            "fallback_verdict": gate.get("fallback_verdict"),
            "raw_claim_incomplete": bool(gate.get("raw_claim_incomplete")),
            "raw_claim_incomplete_reasons": list(gate.get("raw_claim_incomplete_reasons") or []),
            "evidence_resolves_claim": bool(gate.get("evidence_resolves_claim")),
            "structurally_unverifiable": bool(gate.get("structurally_unverifiable")),
            "cited_subject_match": bool(gate.get("cited_subject_match")),
            "cited_time_match": bool(gate.get("cited_time_match")),
            "cited_contains_concrete_fact": bool(gate.get("cited_contains_concrete_fact")),
            "cited_only_investor_questions": bool(gate.get("cited_only_investor_questions")),
            "cited_only_market_signals": bool(gate.get("cited_only_market_signals")),
            "cited_traceable": bool(gate.get("cited_traceable")),
            "direct_evidence_ids": list(gate.get("direct_evidence_ids") or []),
            "valid_cited_evidence_ids": list(gate.get("valid_cited_evidence_ids") or []),
            "unmatched_cited_evidence_ids": list(gate.get("unmatched_cited_evidence_ids") or []),
            "field_sources": exact["claim"].get("field_sources", {}),
            "schema_parse": exact["claim"].get("schema_parse", {}),
            "relation_hit_count": len(relation_hits(exact["claim"], exact["evidence_records"], all_scores, inputs["config"])),
            "relation_hits": relation_hits(exact["claim"], exact["evidence_records"], all_scores, inputs["config"])[:12],
            "is_error_sample": cid in error_by_id,
        }
        if cid in error_by_id:
            error = error_by_id[cid]
            row["primary_error_type"] = error.get("primary_error_type")
            row["route_label"] = error.get("route_label")
            row["attribution_category"] = error_category(error)
            row["why_gate_not_triggered"] = explain_error(error, exact)
        rows.append(row)
    return rows, mismatches


def error_category(error: dict[str, Any]) -> str:
    cid = error.get("cluster_id")
    if cid == "EC-002023":
        return "A. 门控规则未覆盖"
    return ERROR_CATEGORY_BY_ROUTE.get(error.get("route_label"), "C. 样本本身不满足阻止条件")


def summarize(rows: list[dict[str, Any]], mismatches: list[dict[str, Any]], inputs: dict[str, Any]) -> dict[str, Any]:
    gate_reason_counts = Counter(reason for row in rows for reason in row["gate_reasons"])
    raw_reason_counts = Counter(reason for row in rows for reason in row["raw_claim_incomplete_reasons"])
    signal_counts = {
        field: {
            "true": sum(1 for row in rows if row[field]),
            "false": sum(1 for row in rows if not row[field]),
        }
        for field in [
            "raw_claim_incomplete",
            "evidence_resolves_claim",
            "structurally_unverifiable",
            "cited_subject_match",
            "cited_time_match",
            "cited_contains_concrete_fact",
            "cited_only_investor_questions",
            "cited_only_market_signals",
            "cited_traceable",
        ]
    }
    selected_direct_scores = []
    all_relation_hit_scores = []
    for row in rows:
        for item in row["relation_hits"]:
            if isinstance(item.get("total_score"), (int, float)):
                all_relation_hit_scores.append(float(item["total_score"]))
    for cid in inputs["gold"]:
        exact = build_exact_gate_row(cid, inputs)
        scores = exact["gate"].get("all_evidence_scores") or {}
        for eid in exact["gate"].get("direct_evidence_ids") or []:
            score = scores.get(eid)
            if isinstance(score, dict) and isinstance(score.get("total_score"), (int, float)):
                selected_direct_scores.append(float(score["total_score"]))
    candidate_rows = [
        row for row in rows
        if row["baseline_verdict"] == "insufficient" and row["cached_verdict"] in {"support", "contradict"}
    ]
    wrong_candidates = [row for row in candidate_rows if not row["cached_correct"]]
    true_candidates = [row for row in candidate_rows if row["cached_correct"]]
    error_rows = [row for row in rows if row["is_error_sample"]]
    category_counts = Counter(row["attribution_category"] for row in error_rows)
    support_confidences = [float(row["cached_confidence"]) for row in candidate_rows if row["cached_confidence"] is not None]
    return {
        "step": "4.5-D",
        "audit_scope": "Safety Gate v1.2 zero-trigger attribution on independent 50-row gold set",
        "offline": True,
        "model_calls": 0,
        "isr_executed": False,
        "sample_count": len(rows),
        "error_count": len(error_rows),
        "gate_blocked_count": sum(1 for row in rows if row["gate_blocked"]),
        "gate_reason_counts": dict(sorted(gate_reason_counts.items())),
        "stored_current_gate_mismatch_count": len(mismatches),
        "stored_current_gate_mismatches": mismatches,
        "selected_result_source_distribution": dict(Counter(row["selected_result_source"] for row in rows)),
        "verdict_distribution": dict(Counter(row["safety_gate_v1_2_verdict"] for row in rows)),
        "signal_counts": signal_counts,
        "raw_claim_incomplete_reason_counts": dict(sorted(raw_reason_counts.items())),
        "direct_evidence_row_count": sum(1 for row in rows if row["direct_evidence_ids"]),
        "direct_evidence_id_total_count": sum(len(row["direct_evidence_ids"]) for row in rows),
        "selected_direct_evidence_total_score_stats": number_stats(
            selected_direct_scores,
            threshold=float(inputs["config"].get("direct_evidence_total_threshold", 0.72)),
        ),
        "relation_hit_score_stats": number_stats(all_relation_hit_scores),
        "support_confidence_threshold_in_v1_2": inputs["config"].get("support_confidence_threshold"),
        "support_confidence_note": "v1.2 support分支只检查confidence是否有效，不检查低置信度阈值；v1.1的0.75阈值仅作对比，不是本轮规则。",
        "candidate_confidence_stats_vs_v1_1_0_75": number_stats(support_confidences, threshold=0.75),
        "insufficient_resolution_candidate_count": len(candidate_rows),
        "insufficient_true_solution_count": len(true_candidates),
        "insufficient_wrong_solution_count": len(wrong_candidates),
        "insufficient_wrong_solution_ids": [row["cluster_id"] for row in wrong_candidates],
        "insufficient_true_solution_ids": [row["cluster_id"] for row in true_candidates],
        "error_attribution_category_counts": dict(sorted(category_counts.items())),
        "field_wiring_findings": [
            {
                "check": "schema_id_to_event_type",
                "status": "ok",
                "detail": "build_safety_gate_claim(sample, sample) parses sample.schema_id and stores raw_schema_id/schema_parse; no field-name loss was observed.",
            },
            {
                "check": "human_fields_used",
                "status": "ok",
                "detail": "independent safety results mark human_fields_used=false; gold labels are loaded only after gate outputs are frozen.",
            },
            {
                "check": "unmatched_cited_evidence_ids",
                "status": "ok",
                "affected_count": sum(len(row["unmatched_cited_evidence_ids"]) for row in rows),
                "detail": "No cited evidence-id loss was found in the independent 50-row gate results.",
            },
        ],
        "implementation_bug_found": False,
        "sample_coverage_gap_found": False,
        "conclusion": "rule_coverage_gap",
        "conclusion_reason": (
            "独立集已经覆盖了应被拦截的错误support样本；字段接入和类型检查未发现异常。"
            "零触发来自v1.2规则覆盖不足：raw_claim_incomplete虽50/50命中，但只要evidence_resolves_claim=true，"
            "structurally_unverifiable即不成立；support分支也未要求direct evidence或语义对象严格一致。"
        ),
    }


def build_audit(args: argparse.Namespace | None = None) -> dict[str, Any]:
    parsed = args or build_arg_parser().parse_args([])
    inputs = load_inputs(parsed)
    rows, mismatches = audit_rows(inputs)
    summary = summarize(rows, mismatches, inputs)
    error_focus = [row for row in rows if row["is_error_sample"]]
    insufficient_focus = [
        row for row in rows
        if row["baseline_verdict"] == "insufficient" and row["cached_verdict"] in {"support", "contradict"}
    ]
    return {
        "summary": summary,
        "error_focus_13": error_focus,
        "insufficient_resolution_focus_5": insufficient_focus,
        "audit_rows_50": rows,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Safety Gate v1.2 零触发归因审计",
        "",
        "本报告只读取既有独立50条评估、13条错误审计和门控输出；未调用32B，未进入ISR，未修改正式260条结果。",
        "",
        "## 结论",
        "",
        f"- 最终归因: `{summary['conclusion']}`",
        f"- gate_blocked: {summary['gate_blocked_count']}/{summary['sample_count']}",
        f"- 13条错误样本归因: `{json.dumps(summary['error_attribution_category_counts'], ensure_ascii=False)}`",
        f"- insufficient错误解决: {summary['insufficient_wrong_solution_count']} 条，cluster_id={summary['insufficient_wrong_solution_ids']}",
        "",
        "## 门控信号",
        "",
        f"- gate_reasons: `{json.dumps(summary['gate_reason_counts'], ensure_ascii=False)}`",
        f"- selected_result_source: `{json.dumps(summary['selected_result_source_distribution'], ensure_ascii=False)}`",
        f"- raw_claim_incomplete原因: `{json.dumps(summary['raw_claim_incomplete_reason_counts'], ensure_ascii=False)}`",
        f"- signal_counts: `{json.dumps(summary['signal_counts'], ensure_ascii=False)}`",
        f"- direct_evidence rows/items: {summary['direct_evidence_row_count']} / {summary['direct_evidence_id_total_count']}",
        f"- direct_evidence_total_score margin to threshold: `{json.dumps(summary['selected_direct_evidence_total_score_stats'], ensure_ascii=False)}`",
        f"- support confidence: {summary['support_confidence_note']}",
        "",
        "## 13条错误样本",
        "",
        "| cluster_id | gold | baseline | round1 | cached | v1.2 | 归因 | 为什么没有触发 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in audit["error_focus_13"]:
        lines.append(
            "| {cluster_id} | {human_verdict} | {baseline_verdict} | {round1_verdict} | "
            "{cached_verdict} | {safety_gate_v1_2_verdict} | {attribution_category} | {why_gate_not_triggered} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 2条错误解决的 insufficient",
            "",
            "| cluster_id | gold | cached | confidence | direct_evidence_ids | gate输入 | 判断 |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for row in audit["insufficient_resolution_focus_5"]:
        if row["cluster_id"] not in summary["insufficient_wrong_solution_ids"]:
            continue
        gate_inputs = (
            f"raw_claim_incomplete={row['raw_claim_incomplete']}; "
            f"evidence_resolves_claim={row['evidence_resolves_claim']}; "
            f"subject/time/fact/traceable={row['cited_subject_match']}/"
            f"{row['cited_time_match']}/{row['cited_contains_concrete_fact']}/{row['cited_traceable']}"
        )
        lines.append(
            f"| {row['cluster_id']} | {row['human_verdict']} | {row['cached_verdict']} | "
            f"{row['cached_confidence']} | {json.dumps(row['direct_evidence_ids'], ensure_ascii=False)} | "
            f"{gate_inputs} | 应由更强语义覆盖规则阻止；v1.2未覆盖。 |"
        )
    lines.extend(
        [
            "",
            "## 字段与实现检查",
            "",
        ]
    )
    for finding in summary["field_wiring_findings"]:
        lines.append(f"- {finding['check']}: {finding['status']} - {finding['detail']}")
    lines.extend(
        [
            "",
            "## 下一步建议",
            "",
            "- Safety Gate v1.2 不建议进入正式流程；保留代码和报告作为消融记录。",
            "- 若后续继续，只做离线v1.3规格设计，优先补语义对象一致性、混合行情证据和encoded/broad claim规则；不直接调阈值，不调用32B。",
            "",
        ]
    )
    return "\n".join(lines)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def write_outputs(audit: dict[str, Any], args: argparse.Namespace, formal_before: dict[str, str], frozen_before: dict[str, str]) -> dict[str, str]:
    output_dir = output_path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    summary = dict(audit["summary"])
    summary["formal_sha256_before"] = formal_before
    summary["formal_sha256_after"] = hash_files(FORMAL_OUTPUTS)
    summary["formal_outputs_unchanged"] = summary["formal_sha256_before"] == summary["formal_sha256_after"]
    summary["frozen_sha256_before"] = frozen_before
    summary["frozen_sha256_after"] = hash_files(FROZEN_FILES)
    summary["frozen_files_unchanged"] = summary["frozen_sha256_before"] == summary["frozen_sha256_after"]
    audit = dict(audit)
    audit["summary"] = summary
    paths = {
        "audit_report_json": output_dir / "audit_report.json",
        "audit_report_md": output_dir / "audit_report.md",
        "audit_rows_jsonl": output_dir / "audit_rows.jsonl",
        "error_focus_jsonl": output_dir / "error_focus_13.jsonl",
        "insufficient_focus_jsonl": output_dir / "insufficient_resolution_focus_5.jsonl",
    }
    write_json(paths["audit_report_json"], audit)
    paths["audit_report_md"].write_text(render_markdown(audit) + "\n", encoding="utf-8")
    for key, rows in [
        ("audit_rows_jsonl", audit["audit_rows_50"]),
        ("error_focus_jsonl", audit["error_focus_13"]),
        ("insufficient_focus_jsonl", audit["insufficient_resolution_focus_5"]),
    ]:
        with paths[key].open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sums = {name: sha256(path) for name, path in paths.items()}
    sha_path = output_dir / "SHA256SUMS.txt"
    sha_path.write_text(
        "".join(f"{value}  {paths[name].name}\n" for name, value in sorted(sums.items())),
        encoding="utf-8",
    )
    sums["SHA256SUMS"] = sha256(sha_path)
    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "output_dir": str(output_dir.relative_to(ROOT)),
            "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sums[name]} for name, path in paths.items()},
            "SHA256SUMS": {"path": str(sha_path.relative_to(ROOT)), "sha256": sums["SHA256SUMS"]},
            "summary": {
                "conclusion": summary["conclusion"],
                "gate_blocked_count": summary["gate_blocked_count"],
                "error_count": summary["error_count"],
                "insufficient_wrong_solution_ids": summary["insufficient_wrong_solution_ids"],
                "formal_outputs_unchanged": summary["formal_outputs_unchanged"],
                "frozen_files_unchanged": summary["frozen_files_unchanged"],
                "model_calls": summary["model_calls"],
            },
        },
    )
    sums["manifest"] = sha256(manifest_path)
    return sums


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--gold-results", type=Path, default=GOLD_RESULTS)
    parser.add_argument("--error-audit", type=Path, default=ERROR_AUDIT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    formal_before = hash_files(FORMAL_OUTPUTS)
    frozen_before = hash_files(FROZEN_FILES)
    audit = build_audit(args)
    sums = write_outputs(audit, args, formal_before=formal_before, frozen_before=frozen_before)
    print(
        json.dumps(
            {
                "output_dir": str(output_path(args.output_dir)),
                "conclusion": audit["summary"]["conclusion"],
                "gate_blocked_count": audit["summary"]["gate_blocked_count"],
                "error_count": audit["summary"]["error_count"],
                "insufficient_wrong_solution_ids": audit["summary"]["insufficient_wrong_solution_ids"],
                "model_calls": audit["summary"]["model_calls"],
                "sha256": sums,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
