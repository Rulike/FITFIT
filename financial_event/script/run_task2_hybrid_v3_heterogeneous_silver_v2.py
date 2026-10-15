#!/usr/bin/env python3
"""Build Task2 Hybrid-v3 heterogeneous silver labels from clean evidence.

The pipeline reads only clean evidence artifacts and locked IDs. It does not
train, tune, read locked-500 gold labels, read retired silver labels, or run
locked-500 evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SEED = 2026
EXPECTED_CLEAN_SHA256 = "6538549b53c74fc0518804b98e73d4ad3330be319a0a08c4d9a53c41ff0af9f0"

CLEAN_EVIDENCE = ROOT / "outputs/event_credibility/task2_hybrid_v3/clean_evidence_v1/clean_evidence_1500.jsonl"
CLEAN_MANIFEST = ROOT / "outputs/event_credibility/task2_hybrid_v3/clean_evidence_v1/evidence_manifest.json"
CLEAN_REPORT = ROOT / "outputs/event_credibility/task2_hybrid_v3/clean_evidence_v1/evidence_retrieval_report.json"
LOCKED_IDS = ROOT / "configs/task2_independent500_locked_ids.json"

OUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3/heterogeneous_silver_v2"
REPORT_XLSX = ROOT / "reports/task2_hybrid_v3_heterogeneous_silver_v2.xlsx"
REPORT_MD = ROOT / "reports/task2_hybrid_v3_heterogeneous_silver_v2_report.md"

MODEL_DIR = Path("/opt/data/private/JinRong/models/DianJin-R1-32B")
DIANJIN_BASE_URL = "http://127.0.0.1:8000/v1"
DIANJIN_MODEL = "DianJin-R1-32B"
DIANJIN_TIMEOUT_SECONDS = 120

LABELS = ("support", "contradict", "insufficient", "unclear")
EVIDENCE_READY = "evidence_ready"
INSUFFICIENT_EVIDENCE = "insufficient_independent_evidence"

CONTRADICTION_TERMS = (
    "否认",
    "澄清",
    "不属实",
    "并非",
    "不存在",
    "暂无",
    "暂不",
    "没有",
    "无",
    "终止",
    "取消",
    "撤回",
    "辟谣",
    "反驳",
    "更正",
    "存疑",
)

CONTRADICTION_PATTERNS = (
    r"未(?:披露|收到|开展|涉及|签订|达成|确认|获得|通过|考虑|回复|发生|存在|构成|进入|合作|参与|持有|生产|销售|供货)",
    r"不(?:存在|属实|考虑|涉及|构成|确认|会|应当)",
)

DIRECT_SUPPORT_TERMS = (
    "公告",
    "披露",
    "表示",
    "称",
    "确认",
    "通过",
    "签订",
    "中标",
    "获得",
    "获批",
    "完成",
    "发布",
    "推出",
    "增资",
    "收购",
    "合作",
    "回购",
    "分红",
    "增长",
    "转正",
    "下降",
    "提升",
    "拟",
    "将",
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[，。、“”‘’；：:,.!?！？（）()\[\]【】《》<>|｜\\-—_·]", "", text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(public_row(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_locked_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    if isinstance(payload, list):
        ids.update(str(item) for item in payload)
    elif isinstance(payload, dict):
        for key in ("cluster_ids", "sample_ids", "locked_ids", "ids"):
            values = payload.get(key)
            if isinstance(values, list):
                ids.update(str(item) for item in values)
    return ids


def validate_clean_inputs(rows: list[dict[str, Any]], locked_ids: set[str]) -> dict[str, Any]:
    ids = [str(row.get("sample_id") or "") for row in rows]
    unique_ids = set(ids)
    duplicate_count = len(ids) - len(unique_ids)
    locked_overlap = [
        row["sample_id"]
        for row in rows
        if str(row.get("sample_id") or "") in locked_ids
        or str(row.get("claim", {}).get("cluster_id") or "") in locked_ids
    ]
    retained_self = 0
    retained_future = 0
    retained_duplicate = 0
    illegal_evidence_ids = 0
    status_counts = Counter(str(row.get("evidence_status") or "") for row in rows)
    for row in rows:
        claim_article_id = normalize_text(row.get("claim_article_id"))
        title_hash = str(row.get("claim", {}).get("normalized_title_sha256") or "")
        content_hash = str(row.get("claim", {}).get("normalized_content_sha256") or "")
        seen_evidence_ids: set[str] = set()
        seen_source_title: set[tuple[str, str]] = set()
        for item in row.get("evidence_items") or []:
            evidence_id = str(item.get("evidence_id") or "")
            if not evidence_id:
                illegal_evidence_ids += 1
            if evidence_id in seen_evidence_ids:
                illegal_evidence_ids += 1
            seen_evidence_ids.add(evidence_id)
            if claim_article_id and normalize_text(item.get("article_id")) == claim_article_id:
                retained_self += 1
            if title_hash and str(item.get("title_sha256") or "") == title_hash:
                retained_self += 1
            if content_hash and str(item.get("content_sha256") or "") == content_hash:
                retained_self += 1
            if str(item.get("temporal_relation") or "") != "before_or_at_claim_time":
                retained_future += 1
            key = (normalize_text(item.get("source")), normalize_text(item.get("title")))
            if key in seen_source_title:
                retained_duplicate += 1
            seen_source_title.add(key)
    passed = (
        len(rows) == 1500
        and len(unique_ids) == 1500
        and duplicate_count == 0
        and not locked_overlap
        and retained_self == 0
        and retained_future == 0
        and retained_duplicate == 0
        and illegal_evidence_ids == 0
        and status_counts.get(EVIDENCE_READY, 0) == 354
        and status_counts.get(INSUFFICIENT_EVIDENCE, 0) == 1146
    )
    return {
        "passed": passed,
        "input_rows": len(rows),
        "unique_sample_ids": len(unique_ids),
        "duplicate_sample_id_count": duplicate_count,
        "locked500_overlap_count": len(locked_overlap),
        "locked500_overlap_sample_ids": locked_overlap[:100],
        "retained_self_evidence_count": retained_self,
        "retained_future_evidence_count": retained_future,
        "retained_duplicate_evidence_count": retained_duplicate,
        "illegal_clean_evidence_id_count": illegal_evidence_ids,
        "evidence_status_distribution": dict(status_counts),
    }


def valid_evidence_ids(record: dict[str, Any]) -> set[str]:
    return {str(item.get("evidence_id") or "") for item in record.get("evidence_items") or [] if item.get("evidence_id")}


def evidence_text(item: dict[str, Any]) -> str:
    return " ".join(str(item.get(key) or "") for key in ("source", "source_type", "title", "sentence"))


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    norm = normalize_text(text)
    return any(term in norm for term in terms)


def has_contradiction_signal(text: str) -> bool:
    norm = normalize_text(text)
    if any(term in norm for term in CONTRADICTION_TERMS):
        return True
    return any(re.search(pattern, norm) for pattern in CONTRADICTION_PATTERNS)


def direct_support_ids(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in record.get("evidence_items") or []:
        text = evidence_text(item)
        if has_any(text, DIRECT_SUPPORT_TERMS) and not has_contradiction_signal(text):
            out.append(str(item["evidence_id"]))
    return out


def contradiction_ids(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in record.get("evidence_items") or []:
        if has_contradiction_signal(evidence_text(item)):
            out.append(str(item["evidence_id"]))
    return out


def confidence_for(record: dict[str, Any], label: str, evidence_ids: list[str]) -> float:
    count_bonus = min(0.10, 0.035 * max(0, len(evidence_ids) - 1))
    source_count = int(record.get("independent_source_count") or 0)
    source_bonus = min(0.06, 0.02 * max(0, source_count - 1))
    if label == "support":
        base = 0.82
    elif label == "contradict":
        base = 0.84
    elif label == "unclear":
        base = 0.68
    else:
        base = 0.72
    return round(min(0.95, base + count_bonus + source_bonus), 4)


def codex_annotate_record(record: dict[str, Any]) -> dict[str, Any]:
    ids = valid_evidence_ids(record)
    temporal_match = all(
        str(item.get("temporal_relation") or "") == "before_or_at_claim_time"
        for item in record.get("evidence_items") or []
    )
    target_match = bool(record.get("evidence_items")) and all(
        bool(item.get("target_match_valid")) for item in record.get("evidence_items") or []
    )
    event_match_value = bool(record.get("evidence_items")) and all(
        bool(item.get("event_match_valid")) for item in record.get("evidence_items") or []
    )

    contrad = contradiction_ids(record)
    supp = direct_support_ids(record)
    if contrad and supp:
        label = "unclear"
        final_ids = sorted(set(contrad + supp))
        uncertainty = "清洗证据中同时存在支持性表述和否认/澄清/冲突信号"
        reasoning = "存在合法历史证据，但证据立场冲突，不能稳定归为support或contradict。"
    elif contrad:
        label = "contradict"
        final_ids = contrad
        uncertainty = ""
        reasoning = "至少一条合法独立证据包含明确否认、澄清、撤回或冲突信号。"
    elif supp:
        label = "support"
        final_ids = supp
        uncertainty = ""
        reasoning = "至少一条合法独立证据直接对应主体、事件和时间，并包含披露/确认/公告等支持性表述。"
    else:
        label = "insufficient"
        final_ids = []
        uncertainty = "清洗后虽有相关证据，但未形成直接支持或反驳核心主张的明确文本。"
        reasoning = "不能因为没有反证或原新闻语气肯定而判定support。"

    if label in {"support", "contradict"}:
        final_ids = [eid for eid in final_ids if eid in ids]
        if not final_ids:
            label = "insufficient"
            uncertainty = "support/contradict缺少合法clean evidence引用，降为insufficient。"
            reasoning = "严格证据门控禁止引用不存在的evidence_id。"

    return {
        "sample_id": record["sample_id"],
        "label": label,
        "confidence": confidence_for(record, label, final_ids),
        "atomic_claim": str(record.get("claim", {}).get("event_claim") or ""),
        "target_match": target_match,
        "event_match": event_match_value,
        "temporal_match": temporal_match,
        "supporting_evidence_ids": final_ids if label == "support" else [],
        "contradicting_evidence_ids": final_ids if label == "contradict" else [],
        "evidence_reasoning": reasoning,
        "uncertainty_reason": uncertainty,
        "annotator": "codex_clean_evidence_pass_a",
        "annotation_source": "ai_silver",
        "future_evidence_used": False,
    }


def validate_annotation(record: dict[str, Any], annotation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = str(annotation.get("label") or "")
    ids = valid_evidence_ids(record)
    if label not in LABELS:
        errors.append("illegal_label")
    try:
        conf = float(annotation.get("confidence"))
    except Exception:
        conf = -1.0
    if conf < 0.0 or conf > 1.0:
        errors.append("confidence_out_of_range")
    referenced = set(str(eid) for eid in annotation.get("supporting_evidence_ids") or [])
    referenced.update(str(eid) for eid in annotation.get("contradicting_evidence_ids") or [])
    illegal = sorted(eid for eid in referenced if eid not in ids)
    if illegal:
        errors.append("illegal_evidence_id:" + ",".join(illegal[:5]))
    if label == "support" and not annotation.get("supporting_evidence_ids"):
        errors.append("support_without_evidence")
    if label == "contradict" and not annotation.get("contradicting_evidence_ids"):
        errors.append("contradict_without_evidence")
    if annotation.get("future_evidence_used") is not False:
        errors.append("future_evidence_used_not_false")
    return errors


def deterministic_insufficient_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "label": "insufficient",
        "confidence": 0.95,
        "label_source": "deterministic_clean_evidence_gate",
        "annotation_source": "ai_silver",
        "reason": "清洗后缺少足够的独立外部证据",
        "evidence_status": record.get("evidence_status"),
        "final_evidence_ids": [],
        "original_evidence_stats": {
            "independent_source_count": record.get("independent_source_count"),
            "valid_evidence_count": record.get("valid_evidence_count"),
            "excluded_evidence_count": record.get("excluded_evidence_count"),
            "self_evidence_count": record.get("self_evidence_count"),
            "duplicate_evidence_count": record.get("duplicate_evidence_count"),
            "future_evidence_count": record.get("future_evidence_count"),
        },
        "usable_for_training": True,
        "needs_human_review": False,
        "locked500_overlap": False,
        "future_evidence_used": False,
        "human_label": "",
        "human_verdict": "",
    }


def load_existing_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows = read_jsonl(path)
    return {str(row.get("sample_id") or ""): public_row(row) for row in rows if row.get("sample_id")}


def write_checkpoint(path: Path, completed_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {"updated_at": now_iso(), "completed_count": len(completed_ids), "completed_sample_ids": completed_ids},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run_codex_pass_a(records: list[dict[str, Any]], output_path: Path, checkpoint_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ready_records = [row for row in records if row.get("evidence_status") == EVIDENCE_READY]
    ready_by_id = {row["sample_id"]: row for row in ready_records}
    existing = load_existing_jsonl_by_id(output_path)
    valid_existing: dict[str, dict[str, Any]] = {}
    invalid_existing = 0
    for sid, annotation in existing.items():
        if sid not in ready_by_id:
            invalid_existing += 1
            continue
        if validate_annotation(ready_by_id[sid], annotation):
            invalid_existing += 1
            continue
        valid_existing[sid] = annotation

    outputs = dict(valid_existing)
    sorted_records = sorted(ready_records, key=lambda row: row["sample_id"])
    written_since_checkpoint = 0
    for record in sorted_records:
        sid = record["sample_id"]
        if sid in outputs:
            continue
        annotation = codex_annotate_record(record)
        errors = validate_annotation(record, annotation)
        if errors:
            annotation["label"] = "insufficient"
            annotation["confidence"] = 0.70
            annotation["supporting_evidence_ids"] = []
            annotation["contradicting_evidence_ids"] = []
            annotation["uncertainty_reason"] = "annotation validation failed: " + ";".join(errors)
            annotation["evidence_reasoning"] = "严格证据约束失败，降为insufficient并等待后续复核。"
        outputs[sid] = annotation
        written_since_checkpoint += 1
        if written_since_checkpoint >= 25:
            ordered = [outputs[row["sample_id"]] for row in sorted_records if row["sample_id"] in outputs]
            write_jsonl(output_path, ordered)
            write_checkpoint(checkpoint_path, [row["sample_id"] for row in ordered])
            written_since_checkpoint = 0

    ordered = [outputs[row["sample_id"]] for row in sorted_records if row["sample_id"] in outputs]
    write_jsonl(output_path, ordered)
    write_checkpoint(checkpoint_path, [row["sample_id"] for row in ordered])
    return ordered, {
        "expected_rows": len(ready_records),
        "completed_rows": len(ordered),
        "resumed_rows": len(valid_existing),
        "invalid_existing_rows_ignored": invalid_existing,
        "label_distribution": dict(Counter(row["label"] for row in ordered)),
    }


def check_dianjin_service() -> dict[str, Any]:
    url = DIANJIN_BASE_URL.rstrip("/") + "/models"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        model_names = [str(item.get("id") or item.get("name") or "") for item in payload.get("data", [])] if isinstance(payload, dict) else []
        model_available = DIANJIN_MODEL in model_names or not model_names
        return {
            "available": model_available,
            "http_ok": True,
            "base_url": DIANJIN_BASE_URL,
            "model": DIANJIN_MODEL,
            "model_dir": str(MODEL_DIR),
            "model_dir_exists": MODEL_DIR.exists(),
            "model_names": model_names,
            "error": "" if model_available else "model_name_not_listed",
        }
    except Exception as exc:
        return {
            "available": False,
            "http_ok": False,
            "base_url": DIANJIN_BASE_URL,
            "model": DIANJIN_MODEL,
            "model_dir": str(MODEL_DIR),
            "model_dir_exists": MODEL_DIR.exists(),
            "model_names": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def dianjin_prompt(record: dict[str, Any]) -> str:
    evidence_lines = []
    for item in record.get("evidence_items") or []:
        evidence_lines.append(
            "\n".join(
                [
                    f"- evidence_id: {item.get('evidence_id')}",
                    f"  source: {item.get('source')}",
                    f"  publish_time: {item.get('publish_time')}",
                    f"  title: {item.get('title')}",
                    f"  sentence: {item.get('sentence')}",
                    f"  target_match_reason: {item.get('target_match_reason')}",
                    f"  event_match_reason: {item.get('event_match_reason')}",
                ]
            )
        )
    return "\n".join(
        [
            "你是Task2可信度银标复核器。只能依据下列clean evidence判断，不得使用常识或外部记忆。",
            "标签只能为support、contradict、insufficient、unclear。",
            "support/contradict必须引用存在的evidence_id；没有足够独立证据时输出insufficient。",
            "只输出JSON对象。",
            "",
            f"sample_id: {record.get('sample_id')}",
            f"target_entity: {record.get('target_entity')}",
            f"event_type: {record.get('event_type')}",
            f"event_time: {record.get('event_time')}",
            f"claim: {record.get('claim', {}).get('event_claim')}",
            f"financial_context: {json.dumps(record.get('financial_context') or {}, ensure_ascii=False)}",
            "",
            "clean evidence:",
            "\n".join(evidence_lines),
            "",
            'JSON格式：{"label":"support","confidence":0.85,"atomic_claim":"","target_match":true,"event_match":true,"temporal_match":true,"supporting_evidence_ids":["CE1"],"contradicting_evidence_ids":[],"evidence_reasoning":"","uncertainty_reason":"","future_evidence_used":false}',
        ]
    )


def call_dianjin(prompt: str) -> dict[str, Any]:
    url = DIANJIN_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": DIANJIN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": 1024,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=DIANJIN_TIMEOUT_SECONDS) as response:
        body = response.read().decode("utf-8", errors="replace")
    payload = json.loads(body)
    content = payload["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        raise ValueError("no_json_object_in_dianjin_response")
    return json.loads(match.group(0))


def normalize_dianjin_annotation(record: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    annotation = {
        "sample_id": record["sample_id"],
        "label": str(raw.get("label") or "").strip(),
        "confidence": float(raw.get("confidence") or 0.0),
        "atomic_claim": str(raw.get("atomic_claim") or record.get("claim", {}).get("event_claim") or ""),
        "target_match": bool(raw.get("target_match")),
        "event_match": bool(raw.get("event_match")),
        "temporal_match": bool(raw.get("temporal_match")),
        "supporting_evidence_ids": [str(x) for x in raw.get("supporting_evidence_ids") or []],
        "contradicting_evidence_ids": [str(x) for x in raw.get("contradicting_evidence_ids") or []],
        "evidence_reasoning": str(raw.get("evidence_reasoning") or ""),
        "uncertainty_reason": str(raw.get("uncertainty_reason") or ""),
        "annotator": "dianjin_r1_32b_clean_evidence_pass_b",
        "annotation_source": "ai_silver",
        "future_evidence_used": bool(raw.get("future_evidence_used")),
    }
    errors = validate_annotation(record, annotation)
    if errors:
        annotation["label"] = "insufficient"
        annotation["confidence"] = min(annotation["confidence"], 0.70)
        annotation["supporting_evidence_ids"] = []
        annotation["contradicting_evidence_ids"] = []
        annotation["uncertainty_reason"] = "DianJin annotation validation failed: " + ";".join(errors)
    return annotation


def run_dianjin_pass_b(records: list[dict[str, Any]], output_path: Path, checkpoint_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ready_records = sorted([row for row in records if row.get("evidence_status") == EVIDENCE_READY], key=lambda row: row["sample_id"])
    ready_by_id = {row["sample_id"]: row for row in ready_records}
    existing = load_existing_jsonl_by_id(output_path)
    outputs: dict[str, dict[str, Any]] = {}
    invalid_existing = 0
    for sid, annotation in existing.items():
        if sid not in ready_by_id or validate_annotation(ready_by_id[sid], annotation):
            invalid_existing += 1
            continue
        outputs[sid] = annotation
    written_since_checkpoint = 0
    failures = []
    for record in ready_records:
        sid = record["sample_id"]
        if sid in outputs:
            continue
        try:
            raw = call_dianjin(dianjin_prompt(record))
        except Exception as first_exc:
            correction_prompt = dianjin_prompt(record) + "\n上一轮不是合法JSON，请重新只输出JSON对象。"
            try:
                raw = call_dianjin(correction_prompt)
            except Exception as second_exc:
                failures.append({"sample_id": sid, "error": f"{type(second_exc).__name__}: {second_exc}", "first_error": f"{type(first_exc).__name__}: {first_exc}"})
                continue
        outputs[sid] = normalize_dianjin_annotation(record, raw)
        written_since_checkpoint += 1
        if written_since_checkpoint >= 25:
            ordered = [outputs[row["sample_id"]] for row in ready_records if row["sample_id"] in outputs]
            write_jsonl(output_path, ordered)
            write_checkpoint(checkpoint_path, [row["sample_id"] for row in ordered])
            written_since_checkpoint = 0
    ordered = [outputs[row["sample_id"]] for row in ready_records if row["sample_id"] in outputs]
    write_jsonl(output_path, ordered)
    write_checkpoint(checkpoint_path, [row["sample_id"] for row in ordered])
    return ordered, {
        "expected_rows": len(ready_records),
        "completed_rows": len(ordered),
        "resumed_rows": len(existing) - invalid_existing,
        "invalid_existing_rows_ignored": invalid_existing,
        "failure_count": len(failures),
        "failures": failures[:20],
        "label_distribution": dict(Counter(row["label"] for row in ordered)),
    }


def evidence_refs(annotation: dict[str, Any]) -> set[str]:
    refs = set(str(x) for x in annotation.get("supporting_evidence_ids") or [])
    refs.update(str(x) for x in annotation.get("contradicting_evidence_ids") or [])
    return refs


def high_confident_usable(record: dict[str, Any], annotation: dict[str, Any]) -> bool:
    label = annotation.get("label")
    if float(annotation.get("confidence") or 0.0) < 0.80:
        return False
    if annotation.get("future_evidence_used") is not False:
        return False
    if not annotation.get("target_match") or not annotation.get("event_match") or not annotation.get("temporal_match"):
        return False
    if label == "support" and not annotation.get("supporting_evidence_ids"):
        return False
    if label == "contradict" and not annotation.get("contradicting_evidence_ids"):
        return False
    return not validate_annotation(record, annotation)


def anonymized_adjudicate(record: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        {"candidate": "Candidate 1", "label": a.get("label"), "confidence": a.get("confidence"), "evidence_ids": sorted(evidence_refs(a))},
        {"candidate": "Candidate 2", "label": b.get("label"), "confidence": b.get("confidence"), "evidence_ids": sorted(evidence_refs(b))},
    ]
    rng = random.Random(f"{SEED}:{record['sample_id']}")
    rng.shuffle(candidates)
    reread = codex_annotate_record(record)
    final_evidence_ids = sorted(evidence_refs(reread))
    needs_review = float(reread.get("confidence") or 0.0) < 0.75
    usable = high_confident_usable(record, reread) and not needs_review
    return {
        "sample_id": record["sample_id"],
        "anonymous_candidates": candidates,
        "final_label": reread["label"],
        "final_confidence": reread["confidence"],
        "final_evidence_ids": final_evidence_ids,
        "adjudication_reason": "匿名裁决重新阅读clean evidence后给出判定；未按模型名或置信度投票。",
        "needs_human_review": needs_review,
        "usable_for_training": usable,
    }


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[label] / n) * (right[label] / n) for label in LABELS)
    if expected >= 1.0:
        return 1.0
    return round((agree - expected) / (1.0 - expected), 6)


def build_agreement_and_final(
    clean_records: list[dict[str, Any]],
    deterministic_rows: list[dict[str, Any]],
    codex_rows: list[dict[str, Any]],
    dianjin_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    record_by_id = {row["sample_id"]: row for row in clean_records}
    codex_by_id = {row["sample_id"]: row for row in codex_rows}
    dianjin_by_id = {row["sample_id"]: row for row in dianjin_rows}
    ready_ids = sorted(sid for sid, row in record_by_id.items() if row.get("evidence_status") == EVIDENCE_READY)
    missing = [sid for sid in ready_ids if sid not in codex_by_id or sid not in dianjin_by_id]
    if missing:
        raise RuntimeError(f"missing heterogeneous pass rows: {len(missing)}")

    disagreements: list[dict[str, Any]] = []
    adjudicated: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []
    ref_agree = 0
    illegal_ref_count = 0
    confusion = Counter()
    confidence_diffs: list[float] = []
    by_label_total = Counter()
    by_label_agree = Counter()

    for sid in ready_ids:
        record = record_by_id[sid]
        a = codex_by_id[sid]
        b = dianjin_by_id[sid]
        pairs.append((a["label"], b["label"]))
        confusion[(a["label"], b["label"])] += 1
        by_label_total[a["label"]] += 1
        if a["label"] == b["label"]:
            by_label_agree[a["label"]] += 1
        if evidence_refs(a) == evidence_refs(b):
            ref_agree += 1
        confidence_diffs.append(abs(float(a.get("confidence") or 0.0) - float(b.get("confidence") or 0.0)))
        errors = validate_annotation(record, a) + validate_annotation(record, b)
        illegal_ref_count += sum(1 for err in errors if err.startswith("illegal_evidence_id"))
        agreement_usable = (
            a["label"] == b["label"]
            and float(a.get("confidence") or 0.0) >= 0.80
            and float(b.get("confidence") or 0.0) >= 0.80
            and high_confident_usable(record, a)
            and high_confident_usable(record, b)
        )
        adjudication_needed = (
            not agreement_usable
            or a["label"] != b["label"]
            or evidence_refs(a) != evidence_refs(b)
            or bool(validate_annotation(record, a))
            or bool(validate_annotation(record, b))
            or a.get("target_match") != b.get("target_match")
            or a.get("event_match") != b.get("event_match")
            or a.get("temporal_match") != b.get("temporal_match")
        )
        if adjudication_needed:
            disagreements.append({"sample_id": sid, "codex": a, "dianjin": b, "record_summary": clean_record_summary(record)})
            decision = anonymized_adjudicate(record, a, b)
            adjudicated.append(decision)
            final_label = decision["final_label"]
            final_conf = decision["final_confidence"]
            final_eids = decision["final_evidence_ids"]
            usable = decision["usable_for_training"]
            needs_review = decision["needs_human_review"]
            reason = decision["adjudication_reason"]
            label_source = "anonymous_adjudication"
        else:
            final_label = a["label"]
            final_conf = min(float(a.get("confidence") or 0.0), float(b.get("confidence") or 0.0))
            final_eids = sorted(evidence_refs(a))
            usable = True
            needs_review = False
            reason = "异构双模型高置信一致，证据引用合法。"
            label_source = "heterogeneous_high_confidence_agreement"
        final_rows.append(
            {
                "sample_id": sid,
                "final_label": final_label,
                "final_confidence": final_conf,
                "label_source": label_source,
                "annotation_source": "ai_silver",
                "evidence_status": record.get("evidence_status"),
                "final_evidence_ids": final_eids,
                "evidence_reasoning": reason,
                "codex_label": a["label"],
                "dianjin_label": b["label"],
                "heterogeneous_agreement": a["label"] == b["label"],
                "adjudicated": adjudication_needed,
                "usable_for_training": usable,
                "needs_human_review": needs_review,
                "locked500_overlap": False,
                "future_evidence_used": False,
                "human_label": "",
                "human_verdict": "",
            }
        )

    det_final = [
        {
            "sample_id": row["sample_id"],
            "final_label": row["label"],
            "final_confidence": row["confidence"],
            "label_source": row["label_source"],
            "annotation_source": "ai_silver",
            "evidence_status": row["evidence_status"],
            "final_evidence_ids": [],
            "evidence_reasoning": row["reason"],
            "codex_label": "",
            "dianjin_label": "",
            "heterogeneous_agreement": False,
            "adjudicated": False,
            "usable_for_training": True,
            "needs_human_review": False,
            "locked500_overlap": False,
            "future_evidence_used": False,
            "human_label": "",
            "human_verdict": "",
        }
        for row in deterministic_rows
    ]
    all_final = sorted(det_final + final_rows, key=lambda row: row["sample_id"])
    agreement_count = sum(1 for a, b in pairs if a == b)
    return {
        "disagreements": disagreements,
        "adjudicated": adjudicated,
        "final_rows": all_final,
        "agreement_report": {
            "ready_rows": len(ready_ids),
            "label_agreement_count": agreement_count,
            "label_agreement_rate": agreement_count / len(ready_ids) if ready_ids else 0.0,
            "cohen_kappa": cohen_kappa(pairs),
            "per_label_agreement_rate": {
                label: by_label_agree[label] / by_label_total[label] if by_label_total[label] else 0.0
                for label in LABELS
            },
            "support_agreement_count": by_label_agree["support"],
            "contradict_agreement_count": by_label_agree["contradict"],
            "insufficient_agreement_count": by_label_agree["insufficient"],
            "unclear_agreement_count": by_label_agree["unclear"],
            "average_confidence_difference": round(sum(confidence_diffs) / len(confidence_diffs), 6) if confidence_diffs else 0.0,
            "evidence_id_reference_agreement_rate": ref_agree / len(ready_ids) if ready_ids else 0.0,
            "illegal_evidence_id_count": illegal_ref_count,
            "missing_record_count": len(missing),
            "confusion": {f"{a}->{b}": count for (a, b), count in sorted(confusion.items())},
        },
    }


def clean_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "target_entity": record.get("target_entity"),
        "event_type": record.get("event_type"),
        "event_time": record.get("event_time"),
        "evidence_status": record.get("evidence_status"),
        "valid_evidence_count": record.get("valid_evidence_count"),
        "independent_source_count": record.get("independent_source_count"),
    }


def write_agreement_csv(path: Path, confusion: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["codex_label", "dianjin_label", "count"])
        writer.writeheader()
        for key, count in sorted(confusion.items()):
            left, right = key.split("->", 1)
            writer.writerow({"codex_label": left, "dianjin_label": right, "count": count})


def write_excel_report(path: Path, final_rows: list[dict[str, Any]], disagreements: list[dict[str, Any]], agreement: dict[str, Any]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(final_rows).to_excel(writer, sheet_name="silver_final", index=False)
        pd.DataFrame(disagreements).to_excel(writer, sheet_name="disagreements", index=False)
        pd.DataFrame([agreement]).to_excel(writer, sheet_name="agreement_report", index=False)


def startup_command() -> str:
    return (
        "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server "
        "--model /opt/data/private/JinRong/models/DianJin-R1-32B "
        "--served-model-name DianJin-R1-32B "
        "--host 127.0.0.1 --port 8000 "
        "--tensor-parallel-size 4 --trust-remote-code"
    )


def write_manifest_and_sha(
    input_sha: dict[str, str],
    output_paths: list[Path],
    source_paths: list[Path],
    extra: dict[str, Any],
) -> None:
    source_sha = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths if path.exists()}
    output_sha = {str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths if path.exists()}
    manifest_path = OUT_DIR / "annotation_manifest.json"
    manifest = {
        "generated_at": now_iso(),
        "seed": SEED,
        "inputs": input_sha,
        "source_files": source_sha,
        "outputs": output_sha,
        "sha256_policy": "SHA256SUMS.txt records inputs, new source files, primary outputs, and manifest; it omits itself to avoid self-reference.",
        **extra,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_sha[str(manifest_path.relative_to(ROOT))] = sha256_file(manifest_path)
    sha_entries = {**input_sha, **source_sha, **output_sha}
    (OUT_DIR / "SHA256SUMS.txt").write_text(
        "\n".join(f"{digest}  {path}" for path, digest in sorted(sha_entries.items())) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = [CLEAN_EVIDENCE, CLEAN_MANIFEST, CLEAN_REPORT, LOCKED_IDS]
    input_sha = {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths}
    if input_sha[str(CLEAN_EVIDENCE.relative_to(ROOT))] != EXPECTED_CLEAN_SHA256:
        raise RuntimeError(
            "clean_evidence_1500.jsonl SHA256 mismatch: "
            f"{input_sha[str(CLEAN_EVIDENCE.relative_to(ROOT))]} != {EXPECTED_CLEAN_SHA256}"
        )

    rows = read_jsonl(CLEAN_EVIDENCE)
    locked_ids = load_locked_ids(LOCKED_IDS)
    clean_manifest = json.loads(CLEAN_MANIFEST.read_text(encoding="utf-8"))
    clean_report = json.loads(CLEAN_REPORT.read_text(encoding="utf-8"))
    validation = validate_clean_inputs(rows, locked_ids)
    if not validation["passed"]:
        raise RuntimeError("clean evidence validation failed: " + json.dumps(validation, ensure_ascii=False))

    insufficient_rows = [row for row in rows if row.get("evidence_status") == INSUFFICIENT_EVIDENCE]
    deterministic_rows = [deterministic_insufficient_record(row) for row in sorted(insufficient_rows, key=lambda r: r["sample_id"])]
    deterministic_path = OUT_DIR / "deterministic_insufficient_1146.jsonl"
    write_jsonl(deterministic_path, deterministic_rows)

    codex_path = OUT_DIR / "codex_pass_a_354.jsonl"
    codex_checkpoint = OUT_DIR / "codex_pass_a_354.checkpoint.json"
    codex_rows, codex_summary = run_codex_pass_a(rows, codex_path, codex_checkpoint)

    service = check_dianjin_service()
    common_extra = {
        "input_validation": validation,
        "clean_manifest_sha256": input_sha[str(CLEAN_MANIFEST.relative_to(ROOT))],
        "clean_report_sha256": input_sha[str(CLEAN_REPORT.relative_to(ROOT))],
        "clean_report_summary": {
            "allowed_for_next_independent_silver_labeling": clean_report.get("allowed_for_next_independent_silver_labeling"),
            "evidence_status_distribution": clean_report.get("clean_evidence", {}).get("evidence_status_distribution"),
        },
        "deterministic_gate": {
            "expected_rows": 1146,
            "completed_rows": len(deterministic_rows),
            "label": "insufficient",
            "confidence": 0.95,
            "label_source": "deterministic_clean_evidence_gate",
            "usable_for_training": True,
        },
        "codex_pass_a": codex_summary,
        "dianjin_service": service,
        "protections": {
            "locked500_gold_read": False,
            "retired_silver_read": False,
            "old_model_predictions_read": False,
            "training_performed": False,
            "tuning_performed": False,
            "locked500_evaluation_run": False,
            "clean_evidence_modified": False,
            "formal_task2_outputs_modified": False,
        },
    }

    output_paths = [deterministic_path, codex_path, codex_checkpoint]
    source_paths = [
        ROOT / "scripts/run_task2_hybrid_v3_heterogeneous_silver_v2.py",
        ROOT / "scripts/test_task2_hybrid_v3_heterogeneous_silver_v2.py",
    ]

    if not service["available"]:
        run_state = {
            "status": "blocked",
            "blocked_stage": "dianjin_pass_b_service_check",
            "blocker_reason": service["error"],
            "completed_stages": ["input_sha_validation", "clean_evidence_validation", "deterministic_gate_1146", "codex_pass_a_354"],
            "deterministic_insufficient_completed_rows": len(deterministic_rows),
            "codex_pass_a_completed_rows": len(codex_rows),
            "dianjin_pass_b_completed_rows": 0,
            "pending_command": startup_command(),
            "no_fake_pass_b": True,
            "no_final_silver_generated": True,
            "no_training_performed": True,
            "no_locked500_gold_read": True,
        }
        run_state_path = OUT_DIR / "run_state.json"
        run_state_path.write_text(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        output_paths.append(run_state_path)
        write_manifest_and_sha(input_sha, output_paths, source_paths, {**common_extra, "run_state": run_state})
        print(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    dianjin_path = OUT_DIR / "dianjin_pass_b_354.jsonl"
    dianjin_checkpoint = OUT_DIR / "dianjin_pass_b_354.checkpoint.json"
    dianjin_rows, dianjin_summary = run_dianjin_pass_b(rows, dianjin_path, dianjin_checkpoint)
    output_paths.extend([dianjin_path, dianjin_checkpoint])
    if len(dianjin_rows) != 354:
        run_state = {
            "status": "blocked",
            "blocked_stage": "dianjin_pass_b_completion",
            "blocker_reason": "dianjin_pass_b_incomplete",
            "completed_stages": ["input_sha_validation", "clean_evidence_validation", "deterministic_gate_1146", "codex_pass_a_354"],
            "dianjin_summary": dianjin_summary,
            "no_fake_pass_b": True,
            "no_final_silver_generated": True,
            "no_training_performed": True,
            "no_locked500_gold_read": True,
        }
        run_state_path = OUT_DIR / "run_state.json"
        run_state_path.write_text(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        output_paths.append(run_state_path)
        write_manifest_and_sha(input_sha, output_paths, source_paths, {**common_extra, "dianjin_pass_b": dianjin_summary, "run_state": run_state})
        print(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True))
        return 3

    final_payload = build_agreement_and_final(rows, deterministic_rows, codex_rows, dianjin_rows)
    disagreements_path = OUT_DIR / "heterogeneous_disagreements.jsonl"
    adjudicated_path = OUT_DIR / "adjudicated_disagreements.jsonl"
    final_path = OUT_DIR / "silver_final_1500.jsonl"
    eligible_path = OUT_DIR / "silver_training_eligible.jsonl"
    review_path = OUT_DIR / "silver_needs_review.jsonl"
    agreement_path = OUT_DIR / "agreement_report.json"
    confusion_path = OUT_DIR / "agreement_confusion.csv"
    write_jsonl(disagreements_path, final_payload["disagreements"])
    write_jsonl(adjudicated_path, final_payload["adjudicated"])
    write_jsonl(final_path, final_payload["final_rows"])
    write_jsonl(eligible_path, [row for row in final_payload["final_rows"] if row["usable_for_training"]])
    write_jsonl(review_path, [row for row in final_payload["final_rows"] if row["needs_human_review"]])
    agreement_report = final_payload["agreement_report"]
    agreement_path.write_text(json.dumps(agreement_report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_agreement_csv(confusion_path, agreement_report["confusion"])
    write_excel_report(REPORT_XLSX, final_payload["final_rows"], final_payload["disagreements"], agreement_report)
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Task2 Hybrid-v3 Heterogeneous Silver v2",
                "",
                f"- deterministic_insufficient: {len(deterministic_rows)}",
                f"- codex_pass_a: {len(codex_rows)}",
                f"- dianjin_pass_b: {len(dianjin_rows)}",
                f"- agreement_rate: {agreement_report['label_agreement_rate']:.6f}",
                f"- cohen_kappa: {agreement_report['cohen_kappa']}",
                f"- final_distribution: {dict(Counter(row['final_label'] for row in final_payload['final_rows']))}",
                f"- training_eligible: {sum(1 for row in final_payload['final_rows'] if row['usable_for_training'])}",
                f"- needs_human_review: {sum(1 for row in final_payload['final_rows'] if row['needs_human_review'])}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_state = {
        "status": "completed",
        "completed_stages": [
            "input_sha_validation",
            "clean_evidence_validation",
            "deterministic_gate_1146",
            "codex_pass_a_354",
            "dianjin_pass_b_354",
            "heterogeneous_agreement",
            "anonymous_adjudication",
            "silver_final_1500",
        ],
        "no_training_performed": True,
        "no_locked500_gold_read": True,
    }
    run_state_path = OUT_DIR / "run_state.json"
    run_state_path.write_text(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_paths.extend(
        [
            disagreements_path,
            adjudicated_path,
            final_path,
            eligible_path,
            review_path,
            agreement_path,
            confusion_path,
            run_state_path,
            REPORT_XLSX,
            REPORT_MD,
        ]
    )
    write_manifest_and_sha(
        input_sha,
        output_paths,
        source_paths,
        {
            **common_extra,
            "dianjin_pass_b": dianjin_summary,
            "agreement_report": agreement_report,
            "final_distribution": dict(Counter(row["final_label"] for row in final_payload["final_rows"])),
            "training_eligible_count": sum(1 for row in final_payload["final_rows"] if row["usable_for_training"]),
            "needs_human_review_count": sum(1 for row in final_payload["final_rows"] if row["needs_human_review"]),
            "run_state": run_state,
        },
    )
    print(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
