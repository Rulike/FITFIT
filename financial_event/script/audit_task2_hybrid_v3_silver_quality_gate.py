#!/usr/bin/env python3
"""Strict quality gate for Task2 Hybrid-v3 Codex dual-pass silver labels.

This script is intentionally read-only for all silver and locked-id inputs. It
does not train, tune, evaluate the locked 500 set, or inspect any gold label.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SEED = 2026
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3/silver_quality_gate"

PASS_A = ROOT / "outputs/event_credibility/task2_hybrid_v3/codex_silver_pass_a.jsonl"
PASS_B = ROOT / "outputs/event_credibility/task2_hybrid_v3/codex_silver_pass_b.jsonl"
FINAL = ROOT / "outputs/event_credibility/task2_hybrid_v3/codex_silver_final.jsonl"
CANDIDATES = ROOT / "outputs/event_credibility/task2_hybrid_v3/annotation_candidates.jsonl"
LOCKED_IDS = ROOT / "configs/task2_independent500_locked_ids.json"

LEGAL_LABELS = ("support", "contradict", "insufficient", "unclear")
SUPPORT_SAMPLE_SIZE = 150

DENIAL_PATTERNS = (
    "否认",
    "澄清",
    "不属实",
    "并不属实",
    "并非",
    "没有",
    "未",
    "暂无",
    "暂不",
    "不存在",
    "无",
    "终止",
    "取消",
    "撤回",
    "更正",
    "辟谣",
    "反驳",
    "未收到",
    "未开展",
    "未涉及",
    "不考虑",
    "别瞎说",
    "存疑",
)

GENERIC_EVENT_TERMS = {
    "关于",
    "相关",
    "主张",
    "公司",
    "股份",
    "有限",
    "集团",
    "业务",
    "行业",
    "变化",
    "调整",
    "发展",
    "影响",
    "情况",
    "事件",
    "市场",
    "投资",
    "经营",
}


@dataclass
class EvidenceItem:
    evidence_id: str
    source: str
    publish_time: str
    title: str
    raw: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。、“”‘’；：:,.!?！？（）()\[\]【】《》<>|｜\-—_]", "", text)
    return text.lower()


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(v) for v in decoded if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [p.strip() for p in re.split(r"[;,，；]\s*", text) if p.strip()]


def evidence_id_set(row: dict[str, Any], suffix: str) -> tuple[str, ...]:
    ids = as_list(row.get(f"supporting_evidence_ids_{suffix}"))
    ids.extend(as_list(row.get(f"contradicting_evidence_ids_{suffix}")))
    return tuple(sorted(set(ids)))


def parse_evidence_list(value: Any) -> list[EvidenceItem]:
    text = "" if value is None else str(value)
    if not text.strip():
        return []
    parts = [p.strip() for p in re.split(r"[；;]\s*", text) if p.strip()]
    items: list[EvidenceItem] = []
    for idx, part in enumerate(parts):
        fields = [x.strip() for x in part.split("|")]
        if len(fields) >= 4:
            evidence_id = fields[0]
            source = fields[1]
            publish_time = fields[2]
            title = "|".join(fields[3:]).strip()
        else:
            evidence_id = f"evidence_{idx + 1}"
            source = ""
            publish_time = ""
            title = part
        items.append(EvidenceItem(evidence_id, source, publish_time, title, part))
    return items


def load_locked_ids(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    if isinstance(obj, list):
        ids.update(str(v) for v in obj)
    elif isinstance(obj, dict):
        for key in ("cluster_ids", "sample_ids", "locked_ids", "ids"):
            values = obj.get(key)
            if isinstance(values, list):
                ids.update(str(v) for v in values)
        for key in ("cluster_id", "sample_id"):
            value = obj.get(key)
            if value:
                ids.add(str(value))
    return ids


def require_unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    counter = Counter(str(row.get("sample_id", "")) for row in rows)
    missing = counter.get("", 0)
    duplicates = sorted(sid for sid, count in counter.items() if sid and count > 1)
    if missing or duplicates:
        raise ValueError(f"{name} has missing sample_id={missing}, duplicate sample_id={len(duplicates)}")
    return {str(row["sample_id"]): row for row in rows}


def label_from(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, "")).strip()


def extract_target_entity(target: Any) -> str:
    text = "" if target is None else str(target).strip()
    if "/" in text:
        return text.split("/", 1)[0].strip()
    return re.sub(r"\s+\d{6}.*$", "", text).strip()


def extract_event_type(claim: Any) -> str:
    text = "" if claim is None else str(claim)
    match = re.search(r"的(.+?)相关主张", text)
    if match:
        return match.group(1)
    return text[:24] or "unknown"


def claim_keywords(claim: str, title: str) -> set[str]:
    text = normalize_text(claim + title)
    chunks = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    result: set[str] = set()
    for chunk in chunks:
        for term in GENERIC_EVENT_TERMS:
            chunk = chunk.replace(term, "")
        if len(chunk) >= 2:
            result.add(chunk)
    # Add overlapping two-character tokens to keep short event names usable.
    for i in range(max(len(text) - 1, 0)):
        token = text[i : i + 2]
        if token and token not in GENERIC_EVENT_TERMS:
            result.add(token)
    return result


def char_jaccard(a: str, b: str) -> float:
    sa = set(normalize_text(a))
    sb = set(normalize_text(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_self_evidence(item: EvidenceItem, article_title: str, article_time: datetime | None) -> bool:
    ev_title = normalize_text(item.title)
    art_title = normalize_text(article_title)
    if ev_title and art_title and (ev_title == art_title or ev_title in art_title or art_title in ev_title):
        return True
    ev_time = parse_time(item.publish_time)
    if article_time and ev_time and ev_time == article_time and ev_title and char_jaccard(ev_title, art_title) >= 0.75:
        return True
    return False


def is_future(item: EvidenceItem, article_time: datetime | None) -> bool:
    if not article_time:
        return False
    ev_time = parse_time(item.publish_time)
    if not ev_time:
        return False
    return ev_time > article_time


def has_denial_signal(text: str) -> bool:
    normalized = normalize_text(text)
    return any(token in normalized for token in DENIAL_PATTERNS)


def valid_event_match(item: EvidenceItem, candidate: dict[str, Any], target_entity: str) -> bool:
    claim = str(candidate.get("event_claim", ""))
    article_title = str(candidate.get("article_title", ""))
    evidence_text = f"{item.title} {item.raw}"
    if target_entity and target_entity not in evidence_text:
        return False
    if has_denial_signal(evidence_text):
        return True
    if char_jaccard(article_title, item.title) >= 0.50:
        return True
    keywords = claim_keywords(claim, article_title)
    if not keywords:
        return False
    hit_count = sum(1 for kw in keywords if kw in normalize_text(evidence_text))
    return hit_count >= 2 or (hit_count >= 1 and char_jaccard(claim, evidence_text) >= 0.20)


def source_cluster_key(item: EvidenceItem) -> tuple[str, str]:
    return (normalize_text(item.source), normalize_text(item.title))


def story_cluster_key(item: EvidenceItem) -> str:
    title = normalize_text(item.title)
    # Treat identical headlines across portals as one story, not multiple sources.
    return title or normalize_text(item.raw)


def choose_support_sample(final_rows: dict[str, dict[str, Any]], candidates: dict[str, dict[str, Any]]) -> set[str]:
    rng = random.Random(SEED)
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for sid, row in final_rows.items():
        if label_from(row, "ai_final_verdict") != "support":
            continue
        cand = candidates[sid]
        items = parse_evidence_list(cand.get("evidence列表"))
        event_type = extract_event_type(cand.get("event_claim"))
        first_source = items[0].source if items else "no_source"
        evidence_bucket = "0"
        if len(items) == 1:
            evidence_bucket = "1"
        elif 2 <= len(items) <= 3:
            evidence_bucket = "2-3"
        elif len(items) > 3:
            evidence_bucket = "4+"
        groups[(event_type, first_source, evidence_bucket)].append(sid)

    for values in groups.values():
        rng.shuffle(values)
    keys = list(groups)
    rng.shuffle(keys)

    selected: set[str] = set()
    while keys and len(selected) < SUPPORT_SAMPLE_SIZE:
        next_keys: list[tuple[str, str, str]] = []
        for key in keys:
            bucket = groups[key]
            if bucket and len(selected) < SUPPORT_SAMPLE_SIZE:
                selected.add(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def audit_one(
    sid: str,
    final_row: dict[str, Any],
    pass_a: dict[str, Any],
    pass_b: dict[str, Any],
    candidate: dict[str, Any],
    locked_ids: set[str],
    audit_set_reason: list[str],
) -> dict[str, Any]:
    original_label = label_from(final_row, "ai_final_verdict")
    target_entity = extract_target_entity(candidate.get("target"))
    core_claim = str(candidate.get("event_claim", ""))
    event_time_text = str(candidate.get("publish_time", ""))
    article_time = parse_time(event_time_text)
    article_title = str(candidate.get("article_title", ""))
    article_content = str(candidate.get("article_content", ""))
    items = parse_evidence_list(candidate.get("evidence列表"))

    locked_overlap = sid in locked_ids or str(candidate.get("cluster_id", "")) in locked_ids
    future_available = [item.evidence_id for item in items if is_future(item, article_time)]
    non_future_items = [item for item in items if not is_future(item, article_time)]
    self_ids = [item.evidence_id for item in non_future_items if is_self_evidence(item, article_title, article_time)]
    non_self_items = [item for item in non_future_items if item.evidence_id not in set(self_ids)]

    source_keys = [source_cluster_key(item) for item in non_self_items]
    story_keys = [story_cluster_key(item) for item in non_self_items]
    duplicate_source_detected = len(source_keys) != len(set(source_keys)) or len(story_keys) != len(set(story_keys))
    independent_story_keys = {story_cluster_key(item) for item in non_self_items if story_cluster_key(item)}
    independent_source_count = len(independent_story_keys)
    source_independence_valid = independent_source_count > 0

    target_match_in_article = bool(target_entity and target_entity in f"{article_title} {article_content}")
    target_match_in_evidence = any(target_entity and target_entity in f"{item.title} {item.raw}" for item in non_self_items)
    target_match_valid = target_match_in_article or target_match_in_evidence

    matched_items = [item for item in non_self_items if valid_event_match(item, candidate, target_entity)]
    event_match_valid = bool(matched_items)

    contradict_items = [item for item in matched_items if has_denial_signal(f"{item.title} {item.raw}")]
    article_denial_signal = has_denial_signal(f"{article_title} {article_content}")
    explicit_contradiction_found = bool(contradict_items)

    future_evidence_used = False
    temporal_match_valid = True
    financial_evidence_used = any(
        key in candidate
        for key in (
            "financial_evidence",
            "financial_evidence_chain",
            "financial_consistency_score",
            "financial_audit_trace",
        )
    )

    if locked_overlap:
        audited_label = "unclear"
        audit_confidence = 0.50
        reason = "样本ID或cluster_id与locked500重叠，禁止进入训练。"
    elif not target_match_valid:
        audited_label = "unclear"
        audit_confidence = 0.60
        reason = "主体在原文或合法非自身证据中无法稳定匹配。"
    elif explicit_contradiction_found:
        audited_label = "contradict"
        audit_confidence = 0.88
        reason = "合法非自身证据中存在明确否认、澄清或反证信号。"
    elif original_label == "contradict" and article_denial_signal and not source_independence_valid:
        audited_label = "insufficient"
        audit_confidence = 0.78
        reason = "原文含否认或澄清信号，但没有可追溯的独立外部反证，不能仅用待核查新闻自身判定contradict。"
    elif original_label == "support" and source_independence_valid and event_match_valid:
        audited_label = "support"
        audit_confidence = 0.84
        reason = "存在合法非未来、非自身证据，且主体和事件文本均可匹配。"
    elif original_label == "insufficient" and not source_independence_valid:
        audited_label = "insufficient"
        audit_confidence = 0.84
        reason = "没有足够的独立证据，符合insufficient。"
    elif original_label == "insufficient" and not event_match_valid:
        audited_label = "insufficient"
        audit_confidence = 0.82
        reason = "存在证据条目但不能可靠对应核心事件。"
    elif source_independence_valid and not event_match_valid:
        audited_label = "insufficient"
        audit_confidence = 0.80
        reason = "证据主体或主题相近，但未直接对应核心主张。"
    else:
        audited_label = "insufficient"
        audit_confidence = 0.80
        reason = "仅有待核查新闻自身、重复转载或证据不足，不能判定support。"

    if future_available and not matched_items:
        reason += " 可见未来证据已排除，未用于裁决。"

    needs_human_review = (
        audit_confidence < 0.75
        or locked_overlap
        or (explicit_contradiction_found and original_label == "support")
        or (source_independence_valid and event_match_valid and article_denial_signal and not explicit_contradiction_found)
        or (label_from(pass_a, "ai_verdict_a") != label_from(pass_b, "ai_verdict_b") and audit_confidence < 0.85)
    )

    usable_for_training = (
        audit_confidence >= 0.80
        and not future_evidence_used
        and not self_ids
        and not locked_overlap
        and target_match_valid
        and temporal_match_valid
        and not needs_human_review
        and (
            (audited_label == "support" and source_independence_valid and event_match_valid)
            or (audited_label == "contradict" and explicit_contradiction_found)
            or (audited_label == "insufficient" and (not source_independence_valid or not event_match_valid))
        )
    )

    # Contradict labels require explicit independent反证, not just a lack of support.
    if audited_label == "contradict" and not explicit_contradiction_found:
        usable_for_training = False
    # Support labels require at least one independent source.
    if audited_label == "support" and not (source_independence_valid and event_match_valid):
        usable_for_training = False

    return {
        "sample_id": sid,
        "audit_set_reason": sorted(set(audit_set_reason)),
        "original_silver_label": original_label,
        "audited_label": audited_label,
        "label_changed": audited_label != original_label,
        "target_entity": target_entity,
        "core_claim": core_claim,
        "event_time": event_time_text,
        "evidence_before_event": [item.evidence_id for item in non_future_items],
        "evidence_source_ids": [item.evidence_id for item in items],
        "independent_source_count": independent_source_count,
        "self_evidence_detected": bool(self_ids),
        "self_evidence_ids": self_ids,
        "duplicate_source_detected": duplicate_source_detected,
        "source_independence_valid": source_independence_valid,
        "target_match_valid": target_match_valid,
        "event_match_valid": event_match_valid,
        "temporal_match_valid": temporal_match_valid,
        "future_evidence_used": future_evidence_used,
        "future_evidence_available_ids": future_available,
        "financial_evidence_used": financial_evidence_used,
        "explicit_contradiction_found": explicit_contradiction_found,
        "supporting_evidence_ids": [item.evidence_id for item in matched_items if item.evidence_id not in {x.evidence_id for x in contradict_items}],
        "contradicting_evidence_ids": [item.evidence_id for item in contradict_items],
        "audit_reason": reason,
        "audit_confidence": round(float(audit_confidence), 4),
        "usable_for_training": bool(usable_for_training),
        "needs_human_review": bool(needs_human_review),
        "locked500_overlap": bool(locked_overlap),
        "candidate_cluster_id": candidate.get("cluster_id"),
    }


def compute_ab_stats(a_rows: dict[str, dict[str, Any]], b_rows: dict[str, dict[str, Any]], candidate_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample_ids = sorted(a_rows)
    label_agree = 0
    reason_exact = 0
    evidence_exact = 0
    core_claim_exact = 0
    a_reason_nonempty = 0
    b_reason_nonempty = 0
    a_evidence_nonempty = 0
    b_evidence_nonempty = 0
    for sid in sample_ids:
        a = a_rows[sid]
        b = b_rows[sid]
        if label_from(a, "ai_verdict_a") == label_from(b, "ai_verdict_b"):
            label_agree += 1
        a_reason = normalize_text(a.get("ai_reason_a", ""))
        b_reason = normalize_text(b.get("ai_reason_b", ""))
        if a_reason:
            a_reason_nonempty += 1
        if b_reason:
            b_reason_nonempty += 1
        if a_reason == b_reason:
            reason_exact += 1
        a_eids = evidence_id_set(a, "a")
        b_eids = evidence_id_set(b, "b")
        if a_eids:
            a_evidence_nonempty += 1
        if b_eids:
            b_evidence_nonempty += 1
        if a_eids == b_eids:
            evidence_exact += 1
        if str(candidate_rows[sid].get("event_claim", "")) == str(candidate_rows[sid].get("event_claim", "")):
            core_claim_exact += 1

    total = len(sample_ids)
    label_agreement_rate = label_agree / total if total else 0.0
    reason_exact_ratio = reason_exact / total if total else 0.0
    evidence_exact_ratio = evidence_exact / total if total else 0.0
    core_claim_exact_ratio = core_claim_exact / total if total else 0.0
    independence_insufficient = (
        label_agreement_rate >= 0.995
        or evidence_exact_ratio >= 0.90
        or reason_exact_ratio >= 0.50
    )
    reasons = []
    if label_agreement_rate >= 0.995:
        reasons.append("A/B标签一致率>=99.5%，不能视作有效独立人工一致性。")
    if evidence_exact_ratio >= 0.90:
        reasons.append("A/B证据ID集合完全相同比例>=90%，证据选择独立性不足。")
    if reason_exact_ratio >= 0.50:
        reasons.append("A/B理由文本完全相同比例>=50%，理由生成独立性不足。")
    if core_claim_exact_ratio == 1.0:
        reasons.append("A/B使用同一候选核心主张输入，核心主张文本相同属于流程设计。")

    return {
        "total": total,
        "label_agree_count": label_agree,
        "label_agreement_rate": label_agreement_rate,
        "reason_exact_count": reason_exact,
        "reason_exact_ratio": reason_exact_ratio,
        "evidence_id_set_exact_count": evidence_exact,
        "evidence_id_set_exact_ratio": evidence_exact_ratio,
        "core_claim_exact_count": core_claim_exact,
        "core_claim_exact_ratio": core_claim_exact_ratio,
        "core_claim_note": "A/B输出不携带独立core_claim字段；两轮均使用同一annotation_candidates.event_claim输入。",
        "a_reason_nonempty_count": a_reason_nonempty,
        "b_reason_nonempty_count": b_reason_nonempty,
        "a_evidence_nonempty_count": a_evidence_nonempty,
        "b_evidence_nonempty_count": b_evidence_nonempty,
        "effective_independence": not independence_insufficient,
        "independence_conclusion": "insufficient" if independence_insufficient else "acceptable",
        "independence_reasons": reasons,
    }


def validate_inputs(
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    final_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sets = {
        "pass_a": set(a_rows),
        "pass_b": set(b_rows),
        "final": set(final_rows),
        "annotation_candidates": set(candidate_rows),
    }
    reference = sets["pass_a"]
    set_match = {name: sorted(reference ^ ids)[:50] for name, ids in sets.items() if ids != reference}

    illegal = {
        "pass_a": sorted({label_from(row, "ai_verdict_a") for row in a_rows.values()} - set(LEGAL_LABELS)),
        "pass_b": sorted({label_from(row, "ai_verdict_b") for row in b_rows.values()} - set(LEGAL_LABELS)),
        "final": sorted({label_from(row, "ai_final_verdict") for row in final_rows.values()} - set(LEGAL_LABELS)),
    }

    if set_match:
        raise ValueError(f"sample_id sets are not identical: {set_match}")
    if any(values for values in illegal.values()):
        raise ValueError(f"illegal labels found: {illegal}")

    return {
        "sample_id_sets_identical": not set_match,
        "sample_id_set_differences": set_match,
        "legal_labels": list(LEGAL_LABELS),
        "illegal_labels": illegal,
        "rows": {
            "pass_a": len(a_rows),
            "pass_b": len(b_rows),
            "final": len(final_rows),
            "annotation_candidates": len(candidate_rows),
        },
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = [PASS_A, PASS_B, FINAL, CANDIDATES, LOCKED_IDS]
    input_sha_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths}

    a_rows_list = read_jsonl(PASS_A)
    b_rows_list = read_jsonl(PASS_B)
    final_rows_list = read_jsonl(FINAL)
    candidates_list = read_jsonl(CANDIDATES)
    a_rows = require_unique(a_rows_list, "pass_a")
    b_rows = require_unique(b_rows_list, "pass_b")
    final_rows = require_unique(final_rows_list, "final")
    candidate_rows = require_unique(candidates_list, "annotation_candidates")
    locked_ids = load_locked_ids(LOCKED_IDS)

    validation = validate_inputs(a_rows, b_rows, final_rows, candidate_rows)
    ab_stats = compute_ab_stats(a_rows, b_rows, candidate_rows)

    audit_reasons: dict[str, list[str]] = defaultdict(list)
    for sid, row in final_rows.items():
        label = label_from(row, "ai_final_verdict")
        if label == "contradict":
            audit_reasons[sid].append("all_contradict")
        if label == "insufficient":
            audit_reasons[sid].append("all_insufficient")
        if bool(row.get("needs_human_review")):
            audit_reasons[sid].append("all_needs_human_review")
        if label_from(a_rows[sid], "ai_verdict_a") != label_from(b_rows[sid], "ai_verdict_b"):
            audit_reasons[sid].append("all_ab_disagreement")

    support_sample = choose_support_sample(final_rows, candidate_rows)
    for sid in support_sample:
        audit_reasons[sid].append("stratified_support_sample")

    audit_ids = sorted(audit_reasons)
    audit_rows = [
        audit_one(
            sid,
            final_rows[sid],
            a_rows[sid],
            b_rows[sid],
            candidate_rows[sid],
            locked_ids,
            audit_reasons[sid],
        )
        for sid in audit_ids
    ]

    label_distribution = Counter(row["audited_label"] for row in audit_rows)
    original_distribution = Counter(row["original_silver_label"] for row in audit_rows)
    label_changes = [row for row in audit_rows if row["label_changed"]]
    confusion = Counter((row["original_silver_label"], row["audited_label"]) for row in audit_rows)
    eligible_rows = [row for row in audit_rows if row["usable_for_training"]]
    excluded_rows = [row for row in audit_rows if not row["usable_for_training"] or row["needs_human_review"]]

    original_contradict = [row for row in audit_rows if row["original_silver_label"] == "contradict"]
    original_support_sample = [
        row
        for row in audit_rows
        if row["original_silver_label"] == "support" and "stratified_support_sample" in row["audit_set_reason"]
    ]
    contradict_pass_count = sum(
        1 for row in original_contradict if row["audited_label"] == "contradict" and row["usable_for_training"]
    )
    support_pass_count = sum(
        1 for row in original_support_sample if row["audited_label"] == "support" and row["usable_for_training"]
    )
    contradict_pass_rate = contradict_pass_count / len(original_contradict) if original_contradict else 0.0
    support_pass_rate = support_pass_count / len(original_support_sample) if original_support_sample else 0.0

    locked_overlap_rows = [row for row in audit_rows if row["locked500_overlap"]]
    future_used_count = sum(1 for row in audit_rows if row["future_evidence_used"])
    self_evidence_count = sum(1 for row in audit_rows if row["self_evidence_detected"])
    self_evidence_eligible_count = sum(
        1 for row in eligible_rows if row["self_evidence_detected"] and row["usable_for_training"]
    )
    duplicate_source_count = sum(1 for row in audit_rows if row["duplicate_source_detected"])
    low_confidence_count = sum(1 for row in audit_rows if row["audit_confidence"] < 0.80)
    future_available_count = sum(1 for row in audit_rows if row["future_evidence_available_ids"])

    output_unique = len({row["sample_id"] for row in audit_rows}) == len(audit_rows)
    eligible_constraints_ok = all(
        row["audit_confidence"] >= 0.80
        and not row["future_evidence_used"]
        and not row["self_evidence_detected"]
        and not row["locked500_overlap"]
        and row["target_match_valid"]
        and row["event_match_valid"] if row["audited_label"] in ("support", "contradict") else True
        for row in eligible_rows
    )
    # Insufficient records are allowed to be event_match_valid=false when evidence is absent or off-topic.
    eligible_constraints_ok = eligible_constraints_ok and all(
        (row["audited_label"] != "support" or row["source_independence_valid"])
        and (row["audited_label"] != "contradict" or row["explicit_contradiction_found"])
        and not row["needs_human_review"]
        for row in eligible_rows
    )

    audit_change_rate = len(label_changes) / len(audit_rows) if audit_rows else 0.0
    gate_failures: list[str] = []
    if future_used_count != 0:
        gate_failures.append("future_evidence_used_count_nonzero")
    if locked_overlap_rows:
        gate_failures.append("locked500_overlap_nonzero")
    if self_evidence_eligible_count != 0:
        gate_failures.append("self_evidence_entered_training")
    if audit_change_rate > 0.15:
        gate_failures.append("audit_label_change_rate_above_15_percent")
    if contradict_pass_rate < 0.80:
        gate_failures.append("contradict_audit_pass_rate_below_80_percent")
    if support_pass_rate < 0.80:
        gate_failures.append("support_sample_pass_rate_below_80_percent")
    if not output_unique:
        gate_failures.append("output_sample_id_not_unique")
    if not eligible_constraints_ok:
        gate_failures.append("eligible_records_do_not_satisfy_constraints")
    if not ab_stats["effective_independence"]:
        gate_failures.append("ab_independence_insufficient")

    quality_gate_passed = not gate_failures

    report = {
        "seed": SEED,
        "quality_gate_passed": quality_gate_passed,
        "gate_failures": gate_failures,
        "input_validation": validation,
        "ab_independence": ab_stats,
        "silver_final_distribution": dict(Counter(label_from(row, "ai_final_verdict") for row in final_rows.values())),
        "audit": {
            "audit_count": len(audit_rows),
            "audit_unique_sample_ids": len({row["sample_id"] for row in audit_rows}),
            "audit_set_reason_counts": dict(Counter(reason for reasons in audit_reasons.values() for reason in reasons)),
            "original_label_distribution_in_audit": dict(original_distribution),
            "audited_label_distribution": dict(label_distribution),
            "label_changed_count": len(label_changes),
            "label_changed_rate": audit_change_rate,
            "label_change_directions": {f"{src}->{dst}": count for (src, dst), count in sorted(confusion.items()) if src != dst},
            "confusion": {f"{src}->{dst}": count for (src, dst), count in sorted(confusion.items())},
            "training_eligible_count": len(eligible_rows),
            "excluded_or_review_count": len(excluded_rows),
            "training_eligible_distribution": dict(Counter(row["audited_label"] for row in eligible_rows)),
            "self_evidence_count": self_evidence_count,
            "self_evidence_entered_training_count": self_evidence_eligible_count,
            "duplicate_or_same_source_evidence_count": duplicate_source_count,
            "future_evidence_used_count": future_used_count,
            "future_evidence_available_count": future_available_count,
            "locked500_overlap_count": len(locked_overlap_rows),
            "low_confidence_count": low_confidence_count,
            "contradict_audit_total": len(original_contradict),
            "contradict_audit_pass_count": contradict_pass_count,
            "contradict_audit_pass_rate": contradict_pass_rate,
            "support_sample_total": len(original_support_sample),
            "support_sample_pass_count": support_pass_count,
            "support_sample_pass_rate": support_pass_rate,
            "needs_reannotation_sample_ids": [row["sample_id"] for row in audit_rows if not row["usable_for_training"]],
        },
        "protections": {
            "training_performed": False,
            "tuning_performed": False,
            "locked500_gold_read": False,
            "final500_evaluation_run": False,
            "original_silver_files_modified": False,
            "formal_outputs_modified": False,
        },
    }

    audit_path = OUT_DIR / "silver_quality_audit.jsonl"
    eligible_path = OUT_DIR / "silver_training_eligible.jsonl"
    excluded_path = OUT_DIR / "silver_excluded_or_review.jsonl"
    report_json_path = OUT_DIR / "silver_quality_report.json"
    report_md_path = OUT_DIR / "silver_quality_report.md"
    confusion_path = OUT_DIR / "silver_label_confusion.csv"
    manifest_path = OUT_DIR / "manifest.json"
    run_state_path = OUT_DIR / "run_state.json"
    sha_path = OUT_DIR / "SHA256SUMS.txt"

    write_jsonl(audit_path, audit_rows)
    write_jsonl(eligible_path, eligible_rows)
    write_jsonl(excluded_path, excluded_rows)
    write_csv(
        confusion_path,
        ["original_silver_label", "audited_label", "count"],
        [
            {"original_silver_label": src, "audited_label": dst, "count": count}
            for (src, dst), count in sorted(confusion.items())
        ],
    )
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Task2 Hybrid-v3 Codex Silver Quality Gate",
        "",
        f"- quality_gate_passed: {str(quality_gate_passed).lower()}",
        f"- gate_failures: {', '.join(gate_failures) if gate_failures else 'none'}",
        f"- A/B标签一致率: {ab_stats['label_agreement_rate']:.6f} ({ab_stats['label_agree_count']}/{ab_stats['total']})",
        f"- A/B理由完全相同比例: {ab_stats['reason_exact_ratio']:.6f}",
        f"- A/B证据集合完全相同比例: {ab_stats['evidence_id_set_exact_ratio']:.6f}",
        f"- A/B核心主张完全相同比例: {ab_stats['core_claim_exact_ratio']:.6f}",
        f"- 双轮有效独立性: {ab_stats['independence_conclusion']}",
        "",
        "## Audit Summary",
        "",
        f"- 实际审计数量: {len(audit_rows)}",
        f"- 标签修改数量: {len(label_changes)}",
        f"- 标签修改率: {audit_change_rate:.6f}",
        f"- 审计后标签分布: {dict(label_distribution)}",
        f"- 训练可用数量: {len(eligible_rows)}",
        f"- 排除或待复核数量: {len(excluded_rows)}",
        f"- self-evidence数量: {self_evidence_count}",
        f"- 重复或同源证据数量: {duplicate_source_count}",
        f"- future evidence used数量: {future_used_count}",
        f"- locked500重叠数量: {len(locked_overlap_rows)}",
        f"- 低置信度数量: {low_confidence_count}",
        f"- contradict审计通过率: {contradict_pass_rate:.6f}",
        f"- support抽查通过率: {support_pass_rate:.6f}",
        "",
        "## Independence Notes",
        "",
    ]
    md.extend(f"- {item}" for item in ab_stats["independence_reasons"])
    md.extend(
        [
            "",
            "## Reannotation Required",
            "",
            "完整列表见 `silver_excluded_or_review.jsonl` 和 `silver_quality_report.json` 的 `needs_reannotation_sample_ids`。",
        ]
    )
    report_md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    input_sha_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths}
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "inputs": {str(path.relative_to(ROOT)): input_sha_before[str(path.relative_to(ROOT))] for path in input_paths},
        "input_sha256_unchanged": input_sha_before == input_sha_after,
        "outputs": {},
        "quality_gate_passed": quality_gate_passed,
        "locked500_overlap_count": len(locked_overlap_rows),
        "audit_count": len(audit_rows),
        "audit_unique_sample_ids": len({row["sample_id"] for row in audit_rows}),
        "training_eligible_count": len(eligible_rows),
        "excluded_or_review_count": len(excluded_rows),
        "legal_labels": list(LEGAL_LABELS),
        "no_training_performed": True,
        "no_tuning_performed": True,
        "locked500_gold_read": False,
        "existing_silver_files_modified": False,
    }
    run_state = {
        "status": "completed",
        "quality_gate_passed": quality_gate_passed,
        "blocked_next_stage": not quality_gate_passed,
        "completed_stages": [
            "input_validation",
            "ab_independence_audit",
            "forced_audit_set_construction",
            "silver_quality_audit",
            "quality_gate_decision",
        ],
        "training_performed": False,
        "tuning_performed": False,
        "final500_evaluation_run": False,
        "locked500_gold_read": False,
        "failure_reasons": gate_failures,
    }
    run_state_path.write_text(json.dumps(run_state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    output_paths = [
        audit_path,
        report_json_path,
        report_md_path,
        eligible_path,
        excluded_path,
        confusion_path,
        run_state_path,
    ]
    output_sha = {str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths if path.exists()}
    manifest["outputs"] = output_sha
    manifest["sha256_policy"] = "SHA256SUMS.txt records inputs, primary outputs, and manifest; it omits itself to avoid a self-referential checksum."
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_sha[str(manifest_path.relative_to(ROOT))] = sha256_file(manifest_path)
    sha_lines = []
    for path, digest in sorted({**input_sha_before, **output_sha}.items()):
        sha_lines.append(f"{digest}  {path}")
    sha_path.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "quality_gate_passed": quality_gate_passed,
        "audit_count": len(audit_rows),
        "label_changed_count": len(label_changes),
        "audited_label_distribution": dict(label_distribution),
        "training_eligible_count": len(eligible_rows),
        "excluded_or_review_count": len(excluded_rows),
        "gate_failures": gate_failures,
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
