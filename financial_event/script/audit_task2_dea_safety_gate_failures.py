#!/usr/bin/env python3
"""Build a human audit pack for DEA safety-gate false blocks and zero-trigger rules."""

from __future__ import annotations

import argparse
import html
import json
import math
import posixpath
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_task2_dea_human_audit import table_records  # noqa: E402
from src.event_credibility.dea_safety_gate import load_dea_safety_gate_config  # noqa: E402


DEFAULT_GOLD_FILE = ROOT / "reports/task2_dea_cached_human_audit_gold_filled.xlsx"
DEFAULT_SAFETY_RESULTS = ROOT / "outputs/event_credibility/dea_safety_gate_v1/safety_gate_results.jsonl"
DEFAULT_BASELINE_RESULTS = ROOT / "outputs/event_credibility/dea_v1_gray/baseline_results.jsonl"
DEFAULT_CACHED_RESULTS = ROOT / "outputs/event_credibility/dea_v2_cached_gray/dea_results.jsonl"
DEFAULT_SAMPLE_ROWS = ROOT / "outputs/event_credibility/dea_v2_cached_gray/sample_rows.jsonl"
DEFAULT_CONFIG = ROOT / "configs/task2_dea_safety_gate_v1.yaml"
DEFAULT_REPORT_JSON = ROOT / "reports/task2_dea_safety_gate_failure_audit.json"
DEFAULT_REPORT_MD = ROOT / "reports/task2_dea_safety_gate_failure_audit.md"
DEFAULT_REPORT_XLSX = ROOT / "reports/task2_dea_safety_gate_failure_audit.xlsx"

FROZEN_FILES = [
    Path("src/event_credibility/dea_safety_gate.py"),
    Path("configs/task2_dea_safety_gate_v1.yaml"),
    Path("scripts/test_task2_dea_safety_gate.py"),
]
EXPECTED_FROZEN_SHA256 = {
    "src/event_credibility/dea_safety_gate.py": "ed2d0ebde6ecf65636990c8917f002cbc53b2e766344ce318ca0dca0ca2ffaae",
    "configs/task2_dea_safety_gate_v1.yaml": "819a5e1b77de3584cf5e082144c021bda77e810b2804bb2dfa944b62a83ffa12",
    "scripts/test_task2_dea_safety_gate.py": "6659f93eb61ec227a74066c813f93ab94fbc1e11d14c6ef91044b691460690f9",
}
FORMAL_OUTPUTS = [
    Path("outputs/event_credibility/llm_review_results.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.parquet"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.xlsx"),
]
AUDIT_FIELDS = [
    "audit_gate_should_block",
    "audit_claim_unclear",
    "audit_direct_evidence_valid",
    "audit_mapping_bug",
    "audit_rule_too_strict",
    "audit_notes",
]
ZERO_RULES = ["investor_question", "market_signal", "subject_mismatch", "time_mismatch"]


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def compact(value: Any, limit: int = 320) -> str:
    text = norm(value)
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(ROOT / path) for path in paths}


def ensure_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"audit outputs already exist; pass --overwrite to replace them: {existing}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def unique_by_cluster(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    ids = [norm(row.get("cluster_id")) for row in rows]
    duplicates = sorted({cid for cid in ids if cid and ids.count(cid) > 1})
    missing = [idx + 1 for idx, cid in enumerate(ids) if not cid]
    if duplicates or missing:
        parts = []
        if duplicates:
            parts.append(f"{name} duplicate cluster_id: {duplicates[:20]}")
        if missing:
            parts.append(f"{name} missing cluster_id rows: {missing[:20]}")
        raise ValueError("; ".join(parts))
    return {norm(row.get("cluster_id")): row for row in rows}


def load_audit_inputs(args: argparse.Namespace) -> dict[str, Any]:
    gold_rows = table_records(args.gold_file, "审核主表")
    evidence_detail_rows = table_records(args.gold_file, "证据明细")
    return {
        "gold": unique_by_cluster(gold_rows, "gold"),
        "evidence_detail": evidence_detail_rows,
        "evidence_detail_by_cluster": group_by_cluster(evidence_detail_rows),
        "safety": unique_by_cluster(read_jsonl(args.safety_results), "safety_results"),
        "baseline": unique_by_cluster(read_jsonl(args.baseline_results), "baseline_results"),
        "cached": unique_by_cluster(read_jsonl(args.cached_results), "cached_results"),
        "sample_rows": unique_by_cluster(read_jsonl(args.sample_rows), "sample_rows"),
        "config": load_dea_safety_gate_config(args.config),
    }


def group_by_cluster(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[norm(row.get("cluster_id"))].append(row)
    return dict(by_cluster)


def map_chinese_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Map human-audit evidence detail columns to gate-compatible field names."""
    return {
        "cluster_id": norm(row.get("cluster_id")),
        "evidence_stage": norm(row.get("evidence_stage")),
        "evidence_id": norm(row.get("evidence_id")),
        "article_id": norm(row.get("document_id")),
        "source": norm(row.get("来源")),
        "publish_time": norm(row.get("发布时间")),
        "sentence": norm(row.get("证据文本")),
        "title": compact(row.get("证据文本"), 80),
        "evidence_source": norm(row.get("检索来源")),
        "is_global_retrieval": norm(row.get("是否全局索引检索")).lower() == "yes",
        "trace_status": norm(row.get("可追溯状态")),
        "retrieval_query": norm(row.get("检索查询")),
        "retrieval_rank": norm(row.get("排名")),
    }


def evidence_id(item: dict[str, Any]) -> str:
    return norm(item.get("evidence_id"))


def evidence_text(item: dict[str, Any]) -> str:
    fields = ["title", "article_title", "sentence", "content", "source", "article_source", "source_name", "publish_time", "article_publish_time"]
    return norm(" ".join(norm(item.get(field)) for field in fields))


def evidence_source(item: dict[str, Any]) -> str:
    return norm(" ".join(norm(item.get(field)) for field in ["source", "article_source", "source_name", "publisher", "source_type"]))


def evidence_time(item: dict[str, Any]) -> str:
    return norm(" ".join(norm(item.get(field)) for field in ["publish_time", "article_publish_time", "event_time", "event_date", "time", "date"]))


def get_cached_evidence(cached_row: dict[str, Any]) -> list[dict[str, Any]]:
    final_evidence = cached_row.get("final_evidence")
    if isinstance(final_evidence, list):
        return [dict(item) for item in final_evidence if isinstance(item, dict)]
    evidence: list[dict[str, Any]] = []
    for key in ("initial_evidence", "additional_evidence", "evidence"):
        value = cached_row.get(key)
        if isinstance(value, list):
            evidence.extend(dict(item) for item in value if isinstance(item, dict))
    return evidence


def result_is_wrong_block(row: dict[str, Any]) -> bool:
    return bool(row.get("gate_blocked") and row.get("cached_correct") and not row.get("safety_gated_correct"))


def result_is_correct_block(row: dict[str, Any]) -> bool:
    return bool(row.get("gate_blocked") and not row.get("cached_correct") and row.get("safety_gated_correct"))


def result_is_true_insufficient_solution_blocked(row: dict[str, Any]) -> bool:
    return bool(
        row.get("baseline_verdict") == "insufficient"
        and row.get("cached_verdict") in {"support", "contradict"}
        and row.get("cached_correct")
        and row.get("gate_blocked")
        and not row.get("safety_gated_correct")
    )


def cited_evidence_ids(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z0-9])E\d+(?![A-Za-z0-9])", norm(text)))


def dates_from_text(value: Any) -> list[datetime]:
    text = norm(value)
    dates: list[datetime] = []
    for year, month, day in re.findall(r"(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})", text):
        try:
            dates.append(datetime(int(year), int(month), int(day)))
        except ValueError:
            continue
    return dates


def time_relation(claim_time: str, item_time: str, allowed_days: int = 7) -> str:
    claim_dates = dates_from_text(claim_time)
    item_dates = dates_from_text(item_time)
    if not item_dates:
        return "evidence_time_missing_or_unparsed"
    if not claim_dates:
        return "claim_time_missing_gate_accepts_evidence_time"
    min_days = min(abs((left - right).days) for left in claim_dates for right in item_dates)
    return "within_window" if min_days <= allowed_days else f"outside_window_{min_days}_days"


def subject_relation(target_name: str, item: dict[str, Any]) -> str:
    target = norm(target_name)
    if not target:
        return "claim_subject_missing"
    text = evidence_text(item).replace(" ", "")
    return "matched" if target.replace(" ", "") in text else "not_matched"


def classify_wrong_block_reason(
    safety_row: dict[str, Any],
    gold_row: dict[str, Any],
    cached_row: dict[str, Any],
    sample_row: dict[str, Any],
) -> tuple[str, str, str]:
    reasons = set(safety_row.get("gate_reasons") or [])
    cached_evidence = get_cached_evidence(cached_row)
    input_ids = {evidence_id(item) for item in cached_evidence if evidence_id(item)}
    round2_ids = set(cached_row.get("round2", {}).get("evidence_ids") or [])
    has_id_mapping = bool(round2_ids <= input_ids) if round2_ids else bool(input_ids)
    sample_schema = norm(sample_row.get("schema_id"))
    claim_event_type = norm(gold_row.get("事件类型"))
    claim_is_clear = norm(safety_row.get("claim_is_clear"))

    if not has_id_mapping:
        return "evidence_id映射失败", "数据映射问题", "Cached/Round2 evidence_ids are not all present in gate evidence_records."
    if "event_type_missing" in reasons and sample_schema and not claim_event_type:
        return "event_type_missing但证据可补足", "数据映射问题", "Gold主表事件类型为空，但 sample_rows 中存在 schema_id；Step 4.4-B claim适配未把schema_id传入门控。"
    if "encoded_event_type" in reasons and claim_is_clear == "yes":
        return "encoded_event_type误判", "门控规则过严", "编码式schema在证据可补足时被直接视为模糊。"
    if "cached_support_no_direct_evidence" in reasons:
        return "direct_evidence识别失败", "门控规则过严", "主体和时间已匹配，但事件词项/宽泛类别到证据文本的direct evidence识别未连上。"
    if "broad_claim_only" in reasons:
        return "broad_claim_only过严", "门控规则过严", "人工判定证据可支持该宽泛事件类别，但门控将宽泛类别一律降级。"
    if "missing_concrete_fact" in reasons:
        return "missing_concrete_fact误判", "门控规则过严", "证据包含可核验事实，但主张文本本身未写出具体动作，门控未用证据补足具体事实。"
    if not safety_row.get("subject_match"):
        return "主体识别失败", "数据映射问题", "门控没有在证据文本中识别主体。"
    if not safety_row.get("time_match"):
        return "时间解析失败", "数据映射问题", "门控没有完成事件时间与证据时间匹配。"
    return "其他", "需要人工确认", "未命中预定义诊断规则。"


def referenced_evidence_rows(cached_row: dict[str, Any]) -> list[dict[str, Any]]:
    cached_evidence = get_cached_evidence(cached_row)
    by_id = {evidence_id(item): item for item in cached_evidence if evidence_id(item)}
    ids: list[str] = []
    for key in ("round2", "round1"):
        result = cached_row.get(key)
        if isinstance(result, dict):
            ids.extend(norm(item) for item in result.get("evidence_ids") or [])
    if not ids:
        ids = list(by_id)
    seen: set[str] = set()
    rows = []
    for eid in ids:
        if eid in seen or eid not in by_id:
            continue
        seen.add(eid)
        rows.append(by_id[eid])
    return rows


def compact_evidence_bundle(items: list[dict[str, Any]], target_name: str, claim_time: str, allowed_days: int) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        rows.append(
            {
                "evidence_id": evidence_id(item),
                "被引用证据全文": evidence_text(item),
                "证据来源": evidence_source(item),
                "证据发布时间": evidence_time(item),
                "是否全局检索证据": norm(item.get("evidence_source")) == "global_fulltext_index",
                "主体匹配诊断": subject_relation(target_name, item),
                "时间匹配诊断": time_relation(claim_time, evidence_time(item), allowed_days=allowed_days),
            }
        )
    return rows


def wrong_block_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    allowed_days = int(inputs["config"].get("allowed_time_window_days", 7))
    for cid, safety in sorted(inputs["safety"].items()):
        if not result_is_wrong_block(safety):
            continue
        gold = inputs["gold"][cid]
        cached = inputs["cached"][cid]
        sample = inputs["sample_rows"].get(cid, {})
        primary, category, explanation = classify_wrong_block_reason(safety, gold, cached, sample)
        evidence_rows = compact_evidence_bundle(referenced_evidence_rows(cached), norm(gold.get("主体")), norm(gold.get("事件时间")), allowed_days)
        row = {
            "cluster_id": cid,
            "主体": norm(gold.get("主体")),
            "事件类型": norm(gold.get("事件类型")),
            "事件时间": norm(gold.get("事件时间")),
            "待验证主张": norm(gold.get("待验证主张")),
            "human_verdict": norm(safety.get("human_verdict")),
            "Round1 verdict": norm(safety.get("round1_verdict")),
            "Cached verdict": norm(safety.get("cached_verdict")),
            "Safety-gated verdict": norm(safety.get("safety_gated_verdict")),
            "cached confidence": safety.get("original_cached_confidence", ""),
            "gate_reasons": ";".join(safety.get("gate_reasons") or []),
            "claim_unclear": bool(safety.get("claim_unclear")),
            "direct_evidence_ids": ";".join(safety.get("direct_evidence_ids") or []),
            "direct_evidence_count": len(safety.get("direct_evidence_ids") or []),
            "subject_match": bool(safety.get("subject_match")),
            "time_match": bool(safety.get("time_match")),
            "only_investor_questions": bool(safety.get("only_investor_questions")),
            "only_market_signals": bool(safety.get("only_market_signals")),
            "official_contradiction": bool(safety.get("official_contradiction")),
            "被引用证据全文": "\n---\n".join(item["被引用证据全文"] for item in evidence_rows),
            "证据来源": "; ".join(item["证据来源"] for item in evidence_rows),
            "证据发布时间": "; ".join(item["证据发布时间"] for item in evidence_rows),
            "是否全局检索证据": "yes" if any(item["是否全局检索证据"] for item in evidence_rows) else "no",
            "human_notes": norm(safety.get("human_notes")),
            "误伤的直接原因": primary,
            "问题归类": category,
            "诊断说明": explanation,
        }
        row.update({field: "" for field in AUDIT_FIELDS})
        rows.append(row)
    return rows


def true_solution_blocked_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    allowed_days = int(inputs["config"].get("allowed_time_window_days", 7))
    for cid, safety in sorted(inputs["safety"].items()):
        if not result_is_true_insufficient_solution_blocked(safety):
            continue
        gold = inputs["gold"][cid]
        cached = inputs["cached"][cid]
        round1_ids = set(cached.get("round1", {}).get("evidence_ids") or [])
        round2 = cached.get("round2") or {}
        round2_ids = set(round2.get("evidence_ids") or [])
        added_ids = {evidence_id(item) for item in cached.get("additional_evidence") or [] if evidence_id(item)}
        cited_new_ids = cited_evidence_ids(norm(cached.get("final_reason")) + " " + norm(round2.get("reason"))) & (added_ids or round2_ids - round1_ids)
        if not cited_new_ids:
            cited_new_ids = (round2_ids - round1_ids) or added_ids
        by_id = {evidence_id(item): item for item in get_cached_evidence(cached) if evidence_id(item)}
        decisive = [by_id[eid] for eid in sorted(cited_new_ids) if eid in by_id]
        row = {
            "cluster_id": cid,
            "主体": norm(gold.get("主体")),
            "事件类型": norm(gold.get("事件类型")),
            "事件时间": norm(gold.get("事件时间")),
            "待验证主张": norm(gold.get("待验证主张")),
            "human_verdict": norm(safety.get("human_verdict")),
            "Round1 verdict": norm(safety.get("round1_verdict")),
            "Cached verdict": norm(safety.get("cached_verdict")),
            "Safety-gated verdict": norm(safety.get("safety_gated_verdict")),
            "cached confidence": safety.get("original_cached_confidence", ""),
            "使insufficient改变的新增证据ID": ";".join(sorted(cited_new_ids)),
            "新增证据全文": "\n---\n".join(evidence_text(item) for item in decisive),
            "新增证据来源": "; ".join(evidence_source(item) for item in decisive),
            "新增证据发布时间": "; ".join(evidence_time(item) for item in decisive),
            "证据是否直接相关": "mentioned_in_cached_reason" if decisive else "not_resolved_by_script",
            "主体是否匹配": "; ".join(subject_relation(norm(gold.get("主体")), item) for item in decisive),
            "时间是否匹配": "; ".join(time_relation(norm(gold.get("事件时间")), evidence_time(item), allowed_days) for item in decisive),
            "来源是否可追溯": "yes" if decisive and all(evidence_source(item) for item in decisive) else "no",
            "为什么现有门控没有放行": ";".join(safety.get("gate_reasons") or []),
            "问题类型": "输入字段缺失" if "event_type_missing" in (safety.get("gate_reasons") or []) else "规则过严",
            "human_notes": norm(safety.get("human_notes")),
        }
        row.update({field: "" for field in AUDIT_FIELDS})
        rows.append(row)
    return rows


def coverage_rate(rows: list[dict[str, Any]], checker) -> dict[str, Any]:
    total = len(rows)
    nonempty = sum(1 for row in rows if checker(row))
    return {"nonempty": nonempty, "total": total, "rate": round(nonempty / total, 6) if total else 0.0}


def field_coverage(inputs: dict[str, Any]) -> dict[str, Any]:
    cached_evidence = [item for row in inputs["cached"].values() for item in get_cached_evidence(row)]
    original_evidence = [map_chinese_evidence(row) for row in inputs["evidence_detail"]]
    claim_rows = list(inputs["gold"].values())
    return {
        "actual_gate_input_final_evidence": {
            "evidence_records": len(cached_evidence),
            "source": coverage_rate(cached_evidence, lambda row: bool(evidence_source(row))),
            "evidence_text": coverage_rate(cached_evidence, lambda row: bool(evidence_text(row))),
            "publish_time": coverage_rate(cached_evidence, lambda row: bool(evidence_time(row))),
            "evidence_id": coverage_rate(cached_evidence, lambda row: bool(evidence_id(row))),
            "evidence_subject_fields": coverage_rate(cached_evidence, lambda row: any(norm(row.get(field)) for field in ["subject", "company", "entity", "target"])),
            "claim_subject": coverage_rate(claim_rows, lambda row: bool(norm(row.get("主体")))),
        },
        "original_evidence_detail_sheet_mapped": {
            "evidence_records": len(original_evidence),
            "source": coverage_rate(original_evidence, lambda row: bool(evidence_source(row))),
            "evidence_text": coverage_rate(original_evidence, lambda row: bool(evidence_text(row))),
            "publish_time": coverage_rate(original_evidence, lambda row: bool(evidence_time(row))),
            "evidence_id": coverage_rate(original_evidence, lambda row: bool(evidence_id(row))),
            "evidence_subject_fields": coverage_rate(original_evidence, lambda row: any(norm(row.get(field)) for field in ["subject", "company", "entity", "target"])),
            "claim_subject": coverage_rate(claim_rows, lambda row: bool(norm(row.get("主体")))),
        },
    }


def any_pattern(text: str, patterns: list[str], regex: bool = False) -> tuple[bool, str]:
    for pattern in patterns:
        if regex:
            if re.search(pattern, text):
                return True, pattern
        elif pattern and pattern in text:
            return True, pattern
    return False, ""


def rule_signal_counts(inputs: dict[str, Any]) -> dict[str, Any]:
    cfg = inputs["config"]
    cached_evidence_by_cluster = {
        cid: get_cached_evidence(inputs["cached"][cid])
        for cid in inputs["safety"]
        if cid in inputs["cached"]
    }
    out = {}
    investor_clusters = 0
    market_clusters = 0
    subject_false = 0
    time_false = 0
    source_hits = 0
    question_hits = 0
    market_hits = 0
    for cid, evidence_rows in cached_evidence_by_cluster.items():
        if any(any_pattern(evidence_source(item), cfg.get("investor_question_sources", []))[0] for item in evidence_rows):
            source_hits += 1
        if any(any_pattern(evidence_text(item), cfg.get("investor_question_patterns", []))[0] for item in evidence_rows):
            question_hits += 1
        if any(any_pattern(evidence_text(item), cfg.get("market_signal_patterns", []))[0] for item in evidence_rows):
            market_hits += 1
        investor_item_flags = [
            any_pattern(evidence_source(item), cfg.get("investor_question_sources", []))[0]
            and any_pattern(evidence_text(item), cfg.get("investor_question_patterns", []))[0]
            for item in evidence_rows
        ]
        market_item_flags = [any_pattern(evidence_text(item), cfg.get("market_signal_patterns", []))[0] for item in evidence_rows]
        if evidence_rows and all(investor_item_flags):
            investor_clusters += 1
        if evidence_rows and all(market_item_flags):
            market_clusters += 1
        safety = inputs["safety"][cid]
        if not safety.get("subject_match"):
            subject_false += 1
        if not safety.get("time_match"):
            time_false += 1
    out["investor_question"] = {
        "cluster_all_items_match_count": investor_clusters,
        "cluster_any_source_pattern_count": source_hits,
        "cluster_any_question_pattern_count": question_hits,
        "gate_trigger_count": sum(1 for row in inputs["safety"].values() if row.get("only_investor_questions")),
    }
    out["market_signal"] = {
        "cluster_all_items_match_count": market_clusters,
        "cluster_any_market_pattern_count": market_hits,
        "gate_trigger_count": sum(1 for row in inputs["safety"].values() if row.get("only_market_signals")),
    }
    out["subject_mismatch"] = {
        "subject_match_false_count": subject_false,
        "gate_reason_count": sum(1 for row in inputs["safety"].values() if "subject_mismatch" in (row.get("gate_reasons") or [])),
    }
    out["time_mismatch"] = {
        "time_match_false_count": time_false,
        "gate_reason_count": sum(1 for row in inputs["safety"].values() if "time_mismatch" in (row.get("gate_reasons") or [])),
    }
    return out


def candidate_clusters_from_notes(inputs: dict[str, Any], rule: str) -> list[str]:
    patterns = {
        "investor_question": r"投资者|互动|提问|问答|股东提问",
        "market_signal": r"行情|股价|资金|融资|龙虎榜|成交|板块指数|券商推荐",
        "subject_mismatch": r"主体|对象|不匹配|关联|子公司|母公司",
        "time_mismatch": r"时间|日期|早于|晚于|旧闻|过期|指定时间",
    }
    pattern = patterns[rule]
    ids = []
    for cid, gold in sorted(inputs["gold"].items()):
        text = " ".join([norm(gold.get("human_notes")), norm(gold.get("待验证主张")), norm(gold.get("事件类型"))])
        if re.search(pattern, text):
            ids.append(cid)
    return ids


def first_trace_for_rule(inputs: dict[str, Any], rule: str) -> dict[str, Any]:
    cfg = inputs["config"]
    candidates = candidate_clusters_from_notes(inputs, rule)
    for cid in candidates:
        gold = inputs["gold"][cid]
        for item in get_cached_evidence(inputs["cached"][cid]):
            raw = evidence_text(item)
            normalized = norm(raw)
            if rule == "investor_question":
                source_hit, source_pattern = any_pattern(evidence_source(item), cfg.get("investor_question_sources", []))
                text_hit, text_pattern = any_pattern(normalized, cfg.get("investor_question_patterns", []))
                if source_hit or text_hit:
                    return trace_payload(rule, cid, gold, item, normalized, [source_pattern, text_pattern], source_hit and text_hit, source_hit, text_hit)
            elif rule == "market_signal":
                hit, pattern = any_pattern(normalized, cfg.get("market_signal_patterns", []))
                if hit:
                    return trace_payload(rule, cid, gold, item, normalized, [pattern], hit)
            elif rule == "subject_mismatch":
                relation = subject_relation(norm(gold.get("主体")), item)
                return trace_payload(rule, cid, gold, item, normalized, [norm(gold.get("主体"))], relation == "not_matched", relation)
            elif rule == "time_mismatch":
                relation = time_relation(norm(gold.get("事件时间")), evidence_time(item), int(cfg.get("allowed_time_window_days", 7)))
                return trace_payload(rule, cid, gold, item, normalized, [norm(gold.get("事件时间"))], relation.startswith("outside_window"), relation)
    return {
        "rule": rule,
        "candidate_source": "human_notes_or_error_type",
        "candidate_cluster_count": len(candidates),
        "cluster_id": "",
        "raw_evidence_text": "",
        "normalized_text": "",
        "patterns_used": [],
        "pattern_hit": False,
        "miss_reason": "No candidate evidence from human notes produced a concrete trace.",
    }


def trace_payload(
    rule: str,
    cid: str,
    gold: dict[str, Any],
    item: dict[str, Any],
    normalized: str,
    patterns: list[Any],
    hit: bool,
    *details: Any,
) -> dict[str, Any]:
    miss_reason = ""
    if rule == "investor_question":
        source_hit, text_hit = details[0], details[1]
        if not hit:
            miss_reason = f"source_hit={source_hit}, question_text_hit={text_hit}; gate requires both on every direct evidence item and no official response."
        else:
            miss_reason = "Pattern matched this item, but zero-trigger can still occur because not all evidence items are investor questions."
    elif rule == "market_signal":
        miss_reason = "Pattern matched this item, but zero-trigger can still occur because not all evidence items are market signals."
    elif rule == "subject_mismatch":
        miss_reason = f"subject_relation={details[0]}; gate uses claim target_name against evidence text."
    elif rule == "time_mismatch":
        miss_reason = f"time_relation={details[0]}; gate accepts at least one subject-matched evidence within configured window."
    return {
        "rule": rule,
        "candidate_source": "human_notes_or_error_type",
        "cluster_id": cid,
        "主体": norm(gold.get("主体")),
        "事件类型": norm(gold.get("事件类型")),
        "raw_evidence_text": evidence_text(item),
        "normalized_text": normalized,
        "source": evidence_source(item),
        "publish_time": evidence_time(item),
        "patterns_used": [pattern for pattern in patterns if pattern],
        "pattern_hit": bool(hit),
        "miss_reason": miss_reason,
    }


def zero_trigger_diagnostics(inputs: dict[str, Any]) -> dict[str, Any]:
    coverage = field_coverage(inputs)
    return {
        "field_coverage": coverage,
        "gate_read_field_names": {
            "source_fields": inputs["config"].get("source_fields", []),
            "evidence_text_fields": inputs["config"].get("evidence_text_fields", []),
            "evidence_time_fields": inputs["config"].get("evidence_time_fields", []),
            "subject_fields": inputs["config"].get("subject_fields", []),
        },
        "original_data_field_names": {
            "source": ["来源"],
            "evidence_text": ["证据文本"],
            "publish_time": ["发布时间"],
            "evidence_id": ["evidence_id"],
            "global_retrieval": ["是否全局索引检索", "检索来源"],
            "claim_subject": ["主体"],
        },
        "rule_signal_counts": rule_signal_counts(inputs),
        "pattern_traces": [first_trace_for_rule(inputs, rule) for rule in ZERO_RULES],
        "schema_mapping_findings": schema_mapping_findings(inputs),
    }


def schema_mapping_findings(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    final_counts = {cid: len(get_cached_evidence(row)) for cid, row in inputs["cached"].items()}
    detail_counts = {cid: len(rows) for cid, rows in inputs["evidence_detail_by_cluster"].items()}
    if any(detail_counts.get(cid, 0) > final_counts.get(cid, 0) for cid in inputs["gold"]):
        findings.append(
            {
                "check": "only_selected_final_evidence_passed",
                "status": "observed",
                "detail": "Step 4.4-B passed cached final_evidence to the gate, not every original evidence-detail candidate row.",
            }
        )
    missing_schema = []
    for cid, safety in inputs["safety"].items():
        sample = inputs["sample_rows"].get(cid, {})
        gold = inputs["gold"].get(cid, {})
        if "event_type_missing" in (safety.get("gate_reasons") or []) and norm(sample.get("schema_id")) and not norm(gold.get("事件类型")):
            missing_schema.append(cid)
    findings.append(
        {
            "check": "schema_id_not_passed_to_claim",
            "status": "observed" if missing_schema else "not_observed",
            "affected_count": len(missing_schema),
            "detail": "Gold主表事件类型为空时，sample_rows.schema_id 没有进入 Step 4.4-B claim 映射，导致 event_type_missing 诊断。生产规则未因此改变。",
        }
    )
    round2_missing = []
    for cid, cached in inputs["cached"].items():
        final_ids = {evidence_id(item) for item in get_cached_evidence(cached)}
        new_ids = {evidence_id(item) for item in cached.get("additional_evidence") or [] if evidence_id(item)}
        if new_ids and not new_ids <= final_ids:
            round2_missing.append(cid)
    findings.append(
        {
            "check": "round2_new_evidence_passed",
            "status": "ok" if not round2_missing else "missing",
            "affected_count": len(round2_missing),
            "detail": "All cached additional_evidence IDs are present in final_evidence." if not round2_missing else "Some round2_new IDs are absent from final_evidence.",
        }
    )
    bool_issues = []
    for row in inputs["evidence_detail"]:
        for field in ["是否全局索引检索", "是否第一轮已存在", "是否第二轮新增"]:
            value = norm(row.get(field)).lower()
            if value and value not in {"yes", "no"}:
                bool_issues.append((norm(row.get("cluster_id")), field, value))
    findings.append(
        {
            "check": "yes_no_boolean_fields",
            "status": "ok" if not bool_issues else "non_standard_values",
            "affected_count": len(bool_issues),
            "detail": "Workbook yes/no fields are not passed as Python booleans to the gate; audit maps them explicitly when used.",
        }
    )
    return findings


def mapping_flow_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "原始证据记录",
            "field_mapping": "证据明细: evidence_id, 来源, 发布时间, 证据文本, 是否全局索引检索, 检索来源",
            "diagnosis": "中文字段完整；若直接作为门控输入，需要映射为 English gate schema。",
        },
        {
            "stage": "评估脚本字段映射",
            "field_mapping": "Step 4.4-B evidence_for_cached(cache) -> cached.final_evidence",
            "diagnosis": "实际门控输入来自 Cached DEA final_evidence，不是 workbook 证据明细全量候选。",
        },
        {
            "stage": "apply_dea_safety_gate输入",
            "field_mapping": "claim: 主体/事件类型/待验证主张/事件时间; evidence_records: final_evidence",
            "diagnosis": "claim 未合并 sample_rows.schema_id；事件类型为空的编码式样本会出现 event_type_missing。",
        },
        {
            "stage": "门控内部归一化",
            "field_mapping": "source_fields/evidence_text_fields/evidence_time_fields/subject_fields from config",
            "diagnosis": "final_evidence 的 source/sentence/title/publish_time/evidence_id 字段覆盖率高；证据级 subject 字段基本不存在，主体通过 claim 主体在证据文本中匹配。",
        },
        {
            "stage": "各规则判断",
            "field_mapping": "claim_unclear, only_investor_questions, only_market_signals, support/contradict eligibility",
            "diagnosis": "零触发规则不等于 pattern 没命中；投资者/行情要求 evidence_scope 全部满足，而实际样本多为混合证据。",
        },
    ]


def sheet_rows_from_field_coverage(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scope, data in coverage.items():
        for field, value in data.items():
            if field == "evidence_records":
                rows.append({"scope": scope, "field": field, "nonempty": value, "total": value, "rate": 1.0})
            else:
                rows.append({"scope": scope, "field": field, **value})
    return rows


def zero_trigger_sheet_rows(diag: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    signal_counts = diag["rule_signal_counts"]
    for rule in ZERO_RULES:
        row = {"rule": rule}
        row.update(signal_counts.get(rule, {}))
        row["gate_read_source_fields"] = ";".join(diag["gate_read_field_names"].get("source_fields", []))
        row["gate_read_text_fields"] = ";".join(diag["gate_read_field_names"].get("evidence_text_fields", []))
        row["original_source_fields"] = ";".join(diag["original_data_field_names"].get("source", []))
        row["original_text_fields"] = ";".join(diag["original_data_field_names"].get("evidence_text", []))
        rows.append(row)
    return rows


def build_audit_report(inputs: dict[str, Any]) -> dict[str, Any]:
    wrong_rows = wrong_block_rows(inputs)
    true_blocked = true_solution_blocked_rows(inputs)
    zero_diag = zero_trigger_diagnostics(inputs)
    correct_count = sum(1 for row in inputs["safety"].values() if result_is_correct_block(row))
    wrong_count = sum(1 for row in inputs["safety"].values() if result_is_wrong_block(row))
    if any(result_is_correct_block(row) and result_is_wrong_block(row) for row in inputs["safety"].values()):
        raise ValueError("correct_block and wrong_block classifications overlap")
    category_counts = Counter(row["问题归类"] for row in wrong_rows)
    direct_reason_counts = Counter(row["误伤的直接原因"] for row in wrong_rows)
    return {
        "summary": {
            "sample_count": len(inputs["safety"]),
            "wrong_block_count": wrong_count,
            "correct_block_count": correct_count,
            "true_insufficient_solution_blocked_count": len(true_blocked),
            "wrong_block_direct_reason_counts": dict(sorted(direct_reason_counts.items())),
            "wrong_block_category_counts": dict(sorted(category_counts.items())),
            "field_coverage": zero_diag["field_coverage"],
            "schema_mapping_findings": zero_diag["schema_mapping_findings"],
            "zero_trigger_rule_signal_counts": zero_diag["rule_signal_counts"],
            "model_calls": 0,
        },
        "wrong_blocks": wrong_rows,
        "true_solution_blocked": true_blocked,
        "zero_trigger_diagnostics": zero_diag,
        "mapping_flow": mapping_flow_rows(inputs),
    }


def col_letter(n: int) -> str:
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def xesc(value: Any) -> str:
    return html.escape(norm(value), quote=True)


def write_multi_sheet_xlsx(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets)
    workbook_sheets = []
    workbook_rels = []
    overrides = []
    sheet_xmls = {}
    for idx, sheet_name in enumerate(sheet_names, 1):
        rows = sheets[sheet_name]
        headers = list(rows[0].keys()) if rows else ["empty"]
        max_row = len(rows) + 1
        max_col = len(headers)
        cols = "".join(
            f'<col min="{col}" max="{col}" width="{min(max(len(headers[col - 1]), 12) + 2, 70)}" customWidth="1"/>'
            for col in range(1, max_col + 1)
        )
        xml_rows = []
        header_cells = "".join(
            f'<c r="{col_letter(col)}1" t="inlineStr" s="1"><is><t>{xesc(header)}</t></is></c>'
            for col, header in enumerate(headers, 1)
        )
        xml_rows.append(f'<row r="1">{header_cells}</row>')
        for row_idx, row in enumerate(rows, 2):
            cells = []
            for col_idx, header in enumerate(headers, 1):
                cells.append(f'<c r="{col_letter(col_idx)}{row_idx}" t="inlineStr" s="0"><is><t>{xesc(row.get(header, ""))}</t></is></c>')
            xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
        sheet_xmls[f"xl/worksheets/sheet{idx}.xml"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:{col_letter(max_col)}{max_row}"/>'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
            f"<cols>{cols}</cols><sheetData>{''.join(xml_rows)}</sheetData><autoFilter ref=\"A1:{col_letter(max_col)}{max_row}\"/></worksheet>"
        )
        workbook_sheets.append(f'<sheet name="{xesc(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    overrides.append('<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>')
    overrides.append('<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>')
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(workbook_sheets)}</sheets></workbook>"
    )
    workbook_rels_text = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(workbook_rels)}"
        f'<Relationship Id="rId{len(sheet_names) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Times New Roman"/></font><font><b/><sz val="11"/><name val="Times New Roman"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_text)
        zf.writestr("xl/styles.xml", styles)
        for sheet_path, sheet_xml in sheet_xmls.items():
            zf.writestr(sheet_path, sheet_xml)


def write_reports(report: dict[str, Any], args: argparse.Namespace, formal_before: dict[str, str], frozen_before: dict[str, str]) -> None:
    frozen_after = hash_files(FROZEN_FILES)
    formal_after = hash_files(FORMAL_OUTPUTS)
    report["summary"]["frozen_sha256_before"] = frozen_before
    report["summary"]["frozen_sha256_after"] = frozen_after
    report["summary"]["frozen_files_unchanged"] = frozen_before == frozen_after == EXPECTED_FROZEN_SHA256
    report["summary"]["formal_output_sha256_before"] = formal_before
    report["summary"]["formal_output_sha256_after"] = formal_after
    report["summary"]["formal_outputs_unchanged"] = formal_before == formal_after
    if not report["summary"]["frozen_files_unchanged"]:
        raise RuntimeError("frozen safety-gate artifacts changed")
    if not report["summary"]["formal_outputs_unchanged"]:
        raise RuntimeError("formal outputs changed")
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    sheets = {
        "错误阻止11条": report["wrong_blocks"],
        "真实解决被拦2条": report["true_solution_blocked"],
        "零触发规则诊断": zero_trigger_sheet_rows(report["zero_trigger_diagnostics"]),
        "字段覆盖统计": sheet_rows_from_field_coverage(report["zero_trigger_diagnostics"]["field_coverage"]),
        "证据匹配轨迹": report["zero_trigger_diagnostics"]["pattern_traces"],
    }
    write_multi_sheet_xlsx(args.report_xlsx, sheets)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Task 2 DEA Safety Gate Failure Audit",
        "",
        "本报告只做误伤与零触发规则诊断；未修改门控规则、配置或正式输出，未调用模型。",
        "",
        f"- 样本数: {summary['sample_count']}",
        f"- 错误阻止: {summary['wrong_block_count']}",
        f"- 正确阻止: {summary['correct_block_count']}",
        f"- 真实 insufficient 解决被拦: {summary['true_insufficient_solution_blocked_count']}",
        f"- 误伤原因分布: {json.dumps(summary['wrong_block_direct_reason_counts'], ensure_ascii=False)}",
        f"- 问题归类分布: {json.dumps(summary['wrong_block_category_counts'], ensure_ascii=False)}",
        "",
        "## 零触发规则",
        "",
        json.dumps(summary["zero_trigger_rule_signal_counts"], ensure_ascii=False, indent=2),
        "",
        "## 字段覆盖",
        "",
        json.dumps(summary["field_coverage"], ensure_ascii=False, indent=2),
        "",
        "## Schema 映射发现",
        "",
    ]
    for item in summary["schema_mapping_findings"]:
        lines.append(f"- {item['check']}: {item['status']} ({item.get('affected_count', '')}) {item['detail']}")
    lines.extend(["", "## 冻结与输出保护", ""])
    lines.append(f"- frozen_files_unchanged: {summary.get('frozen_files_unchanged')}")
    lines.append(f"- formal_outputs_unchanged: {summary.get('formal_outputs_unchanged')}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_GOLD_FILE)
    parser.add_argument("--safety-results", type=Path, default=DEFAULT_SAFETY_RESULTS)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--cached-results", type=Path, default=DEFAULT_CACHED_RESULTS)
    parser.add_argument("--sample-rows", type=Path, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-xlsx", type=Path, default=DEFAULT_REPORT_XLSX)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ensure_outputs_available([args.report_json, args.report_md, args.report_xlsx], args.overwrite)
    frozen_before = hash_files(FROZEN_FILES)
    formal_before = hash_files(FORMAL_OUTPUTS)
    inputs = load_audit_inputs(args)
    report = build_audit_report(inputs)
    write_reports(report, args, formal_before=formal_before, frozen_before=frozen_before)
    print(
        json.dumps(
            {
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "report_xlsx": str(args.report_xlsx),
                "wrong_block_count": report["summary"]["wrong_block_count"],
                "true_insufficient_solution_blocked_count": report["summary"]["true_insufficient_solution_blocked_count"],
                "frozen_files_unchanged": report["summary"]["frozen_files_unchanged"],
                "formal_outputs_unchanged": report["summary"]["formal_outputs_unchanged"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
