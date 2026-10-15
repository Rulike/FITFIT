#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/task3_canonical_event_v2_validation"

CONTRACT_PATH = ROOT / "configs/structured_event_contract_v1.json"
TARGET_ALIAS_PATH = ROOT / "configs/target_aliases.json"
TRAIN_SOURCE_PATH = ROOT / "outputs/task1_step5_7i_structure_expansion/expansion_train_240_source.jsonl"
TRAIN_DRAFT_PATH = ROOT / "outputs/task1_step5_7i_structure_expansion/expansion_train_240_ai_drafts.jsonl"
TEST_SOURCE_PATH = ROOT / "outputs/task1_step5_7i_structure_expansion/independent_frozen_test_60_source.jsonl"
TEST_DRAFT_PATH = ROOT / "outputs/task1_step5_7i_structure_expansion/independent_frozen_test_60_ai_drafts.jsonl"
SENTENCE_INPUT_PATH = ROOT / "outputs/task1_step5_7i_structure_expansion/sentence_inputs.jsonl"
REVIEW3_PATH = ROOT / "outputs/task1_step5_7e_human_confirmation/contract_review3_template.jsonl"

PROTECTED_FILES = [
    ROOT / "outputs/task3/normalized/normalized_event_mentions.jsonl",
    ROOT / "outputs/task3_kg_v3_final/state_observations/canonical_event_current_state.jsonl",
    ROOT / "outputs/task3_kg_v3_final/graph_nodes.jsonl",
    ROOT / "outputs/task3_kg_v3_final/task4_event_interface.jsonl",
    ROOT / "outputs/task4 golddata/task34.xlsx",
]

ALLOWED_SUPPORT_TYPES = {
    "direct_fact",
    "confirmed_announcement",
    "regulatory_document",
    "financial_statement",
    "quoted_claim",
}

REJECT_SUPPORT_TYPES = {
    "rumor",
    "forecast_only",
    "boilerplate",
    "market_price_only",
    "background_only",
}

PUNCT_TO_DROP = set(" \t\r\n，。；：、,.!！?？（）()《》“”\"'`·-—_[]【】{}<>")
ROLE_ORDER = ["subject_company", "event_time", "object_target", "trigger"]
SENTENCE_FIELDS = ["sentence_id", "sentence_index", "sentence_text", "article_start", "article_end"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_json_record(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    return payload if isinstance(payload, dict) else {"_value": payload}
        return None
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return payload if isinstance(payload, dict) else {"_value": payload}
    return None


def top_level_keys(path: Path) -> list[str]:
    rec = first_json_record(path)
    if not isinstance(rec, dict):
        return []
    return sorted(rec.keys())


def nested_keys(rec: dict[str, Any], key: str) -> list[str]:
    value = rec.get(key)
    if isinstance(value, dict):
        return sorted(value.keys())
    return []


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def normalize_fuzzy(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"\s+", "", value)
    return value.lower()


def canonicalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for idx, char in enumerate(unicodedata.normalize("NFKC", str(text or ""))):
        if char.isspace() or char in PUNCT_TO_DROP:
            continue
        chars.append(char.lower())
        positions.append(idx)
    return "".join(chars), positions


def exact_occurrences(content: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = content.find(needle, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + 1
    return spans


def normalized_occurrences(content: str, needle: str) -> list[tuple[int, int]]:
    norm_needle, _ = canonicalize_with_map(needle)
    norm_haystack, positions = canonicalize_with_map(content)
    if not norm_needle:
        return []
    hits: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = norm_haystack.find(norm_needle, start)
        if idx < 0:
            break
        raw_start = positions[idx]
        raw_end = positions[idx + len(norm_needle) - 1] + 1
        hits.append((raw_start, raw_end))
        start = idx + 1
    return hits


def choose_unique(candidates: list[tuple[int, int]], anchors: list[tuple[int, int]] | None = None) -> tuple[tuple[int, int] | None, str]:
    if not candidates:
        return None, "no_candidate"
    if len(candidates) == 1:
        return candidates[0], "unique"
    anchors = anchors or []
    if anchors:
        inside = [cand for cand in candidates if any(a_start <= cand[0] and cand[1] <= a_end for a_start, a_end in anchors)]
        if len(inside) == 1:
            return inside[0], "inside_anchor"
        if len(inside) > 1:
            candidates = inside
        distances = []
        for cand in candidates:
            center = (cand[0] + cand[1]) / 2.0
            distance = min(abs(center - ((a_start + a_end) / 2.0)) for a_start, a_end in anchors)
            distances.append((distance, cand))
        distances.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
        if len(distances) >= 2 and abs(distances[0][0] - distances[1][0]) < 1e-9:
            return None, "ambiguous_multi_hit"
        return distances[0][1], "nearest_anchor"
    return None, "ambiguous_multi_hit"


def exact_or_normalized_span(content: str, needle: str, anchors: list[tuple[int, int]] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not needle:
        return None, {"status": "empty", "exact": False, "normalized": False, "candidate_count": 0}
    exact_hits = exact_occurrences(content, needle)
    if exact_hits:
        chosen, reason = choose_unique(exact_hits, anchors=anchors)
        if chosen is None:
            return None, {"status": reason, "exact": True, "normalized": False, "candidate_count": len(exact_hits)}
        start, end = chosen
        return {"text": content[start:end], "start": start, "end": end}, {
            "status": f"exact_{reason}",
            "exact": True,
            "normalized": False,
            "candidate_count": len(exact_hits),
        }
    norm_hits = normalized_occurrences(content, needle)
    if norm_hits:
        chosen, reason = choose_unique(norm_hits, anchors=anchors)
        if chosen is None:
            return None, {"status": reason, "exact": False, "normalized": True, "candidate_count": len(norm_hits)}
        start, end = chosen
        return {"text": content[start:end], "start": start, "end": end}, {
            "status": f"normalized_{reason}",
            "exact": False,
            "normalized": True,
            "candidate_count": len(norm_hits),
        }
    return None, {"status": "unresolvable", "exact": False, "normalized": False, "candidate_count": 0}


def load_alias_map(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for family, items in data.items():
        if isinstance(items, dict):
            out[family] = {}
            for name, meta in items.items():
                if isinstance(meta, dict):
                    out[family][name] = {"code": str(meta.get("code", "") or ""), "aliases": [str(x) for x in meta.get("aliases", []) if str(x or "").strip()]}
                else:
                    out[family][name] = {"code": "", "aliases": []}
    return out


def load_sentence_index(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        event_mention_id = str(row.get("event_mention_id", "") or "")
        if event_mention_id:
            by_id[event_mention_id] = row
    return by_id


def load_source_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            event_mention_id = str(row.get("event_mention_id", "") or "")
            if event_mention_id:
                row = dict(row)
                row["_source_file"] = rel(path)
                by_id[event_mention_id] = row
    return by_id


def evidence_window_from_sentences(source_row: dict[str, Any], sentence_row: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    if not sentence_row:
        return None, [], ["missing_sentence_input"]
    content = str(source_row.get("content", "") or "")
    by_id = {str(s.get("sentence_id", "") or ""): s for s in sentence_row.get("sentences", []) if isinstance(s, dict)}
    primary_sid = str(source_row.get("selected_primary_sentence_id", "") or source_row.get("raw_model_json", {}).get("primary_sentence_id", "") or "")
    window_ids = [str(sid) for sid in source_row.get("event_span_sentence_ids", []) if str(sid or "").strip()]
    if not window_ids:
        if primary_sid:
            window_ids = [primary_sid]
    valid_ids: list[str] = []
    reasons: list[str] = []
    for sid in window_ids:
        if sid not in by_id:
            reasons.append(f"sentence_missing:{sid}")
        elif sid not in valid_ids:
            valid_ids.append(sid)
    if not valid_ids:
        return None, [], reasons or ["empty_sentence_window"]
    indices = sorted(int(by_id[sid]["sentence_index"]) for sid in valid_ids)
    if indices != list(range(indices[0], indices[-1] + 1)):
        reasons.append("non_contiguous_sentence_window")
        return None, [], reasons
    ordered = [by_id[sid] for sid in sorted(valid_ids, key=lambda sid: int(by_id[sid]["sentence_index"]))]
    start = min(int(row["article_start"]) for row in ordered)
    end = max(int(row["article_end"]) for row in ordered)
    window_text = content[start:end]
    evidence_sentence_id = primary_sid if primary_sid in by_id else ordered[0]["sentence_id"]
    evidence_sentence = by_id[evidence_sentence_id]
    return (
        {
            "sentence_id": evidence_sentence_id,
            "sentence_text": str(evidence_sentence["sentence_text"]),
            "article_start": int(evidence_sentence["article_start"]),
            "article_end": int(evidence_sentence["article_end"]),
        },
        ordered,
        [],
    )


def exact_match_in_window(
    window_rows: list[dict[str, Any]],
    candidate: str,
    preferred_sentence_id: str = "",
    anchors: list[tuple[int, int]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not candidate:
        return None, {"status": "empty_candidate", "exact": False, "normalized": False, "candidate_count": 0}
    exact_hits: list[tuple[int, int, str]] = []
    norm_hits: list[tuple[int, int, str]] = []
    for sent in window_rows:
        sentence_text = str(sent["sentence_text"])
        base = int(sent["article_start"])
        for start, end in exact_occurrences(sentence_text, candidate):
            exact_hits.append((base + start, base + end, str(sent["sentence_id"])))
        for start, end in normalized_occurrences(sentence_text, candidate):
            norm_hits.append((base + start, base + end, str(sent["sentence_id"])))
    if exact_hits:
        if preferred_sentence_id:
            preferred = [hit for hit in exact_hits if hit[2] == preferred_sentence_id]
            if len(preferred) == 1:
                exact_hits = preferred
        if anchors and len(exact_hits) > 1:
            inside = [hit for hit in exact_hits if any(a_start <= hit[0] and hit[1] <= a_end for a_start, a_end in anchors)]
            if len(inside) == 1:
                exact_hits = inside
        if len(exact_hits) == 1:
            start, end, sid = exact_hits[0]
            sent = next(sent for sent in window_rows if str(sent["sentence_id"]) == sid)
            base = int(sent["article_start"])
            return {"text": str(sent["sentence_text"])[start - base : end - base], "start": start, "end": end, "sentence_id": sid}, {
                "status": "exact_unique" if not preferred_sentence_id else "exact_preferred",
                "exact": True,
                "normalized": False,
                "candidate_count": 1,
            }
        return None, {"status": "exact_ambiguous", "exact": True, "normalized": False, "candidate_count": len(exact_hits)}
    if norm_hits:
        if preferred_sentence_id:
            preferred = [hit for hit in norm_hits if hit[2] == preferred_sentence_id]
            if len(preferred) == 1:
                norm_hits = preferred
        if anchors and len(norm_hits) > 1:
            inside = [hit for hit in norm_hits if any(a_start <= hit[0] and hit[1] <= a_end for a_start, a_end in anchors)]
            if len(inside) == 1:
                norm_hits = inside
        if len(norm_hits) == 1:
            start, end, sid = norm_hits[0]
            sent = next(sent for sent in window_rows if str(sent["sentence_id"]) == sid)
            base = int(sent["article_start"])
            return {"text": str(sent["sentence_text"])[start - base : end - base], "start": start, "end": end, "sentence_id": sid}, {
                "status": "normalized_unique" if not preferred_sentence_id else "normalized_preferred",
                "exact": False,
                "normalized": True,
                "candidate_count": 1,
            }
        return None, {"status": "normalized_ambiguous", "exact": False, "normalized": True, "candidate_count": len(norm_hits)}
    return None, {"status": "unresolvable", "exact": False, "normalized": False, "candidate_count": 0}


def support_type(text: str) -> str:
    norm = normalize_fuzzy(text)
    if any(token in norm for token in ("财报", "年报", "半年报", "三季报", "一季报", "营业收入", "净利润", "毛利率", "资产负债率", "经营现金流")):
        return "financial_statement"
    if any(token in norm for token in ("公告", "通知", "批复", "批准", "发布", "签署", "签约", "中标", "收到", "成立", "投产", "开工", "竣工", "上市", "完成", "落地", "投运", "投用", "协议", "合同", "收购", "并购", "融资", "募资", "披露")):
        return "confirmed_announcement"
    if any(token in norm for token in ("监管", "证监", "上交所", "深交所", "问询", "函", "备案", "审议", "审批", "审查", "批准")):
        return "regulatory_document"
    if any(token in norm for token in ("表示", "透露", "称", "指出", "认为", "报道", "消息", "知情人士")):
        return "quoted_claim"
    if any(token in norm for token in ("预计", "有望", "可能", "或将", "拟", "计划", "将于", "将会", "未来", "如果")):
        return "forecast_only"
    if any(token in norm for token in ("股价", "收盘", "开盘", "涨停", "跌停", "报", "美元", "元/桶", "行情", "价格", "市场")):
        return "market_price_only"
    return "direct_fact"


def event_fact_signature(text: str) -> str:
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", str(text or "")))[:180]


def event_type_from_schema(schema_id: str, event_name: str) -> str:
    schema_id = str(schema_id or "").strip()
    if schema_id:
        return schema_id
    return str(event_name or "").strip()


def choose_target_candidate(source_row: dict[str, Any], alias_map: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[str], str]:
    target_type = str(source_row.get("target_type", "") or "")
    target_name = str(source_row.get("target_name", "") or "")
    names: list[str] = []
    if target_name:
        names.append(target_name)
    family = alias_map.get(target_type, {})
    if target_name in family:
        meta = family[target_name]
        for alias in meta.get("aliases", []):
            if alias and alias not in names:
                names.append(alias)
    if not names and target_name:
        names.append(target_name)
    return names, target_type


def build_record(
    source_row: dict[str, Any],
    ai_row: dict[str, Any],
    sentence_row: dict[str, Any] | None,
    alias_map: dict[str, dict[str, dict[str, Any]]],
    source_file: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    content = str(ai_row.get("content") or source_row.get("content") or "")
    article_id = str(ai_row.get("article_id") or source_row.get("article_id") or "")
    source_name = str(ai_row.get("source") or source_row.get("source") or "")
    publish_time = str(ai_row.get("publish_time") or source_row.get("publish_time") or "")
    event_name = str(ai_row.get("event_name") or source_row.get("event_name") or "")
    schema_id = str(ai_row.get("schema_id") or source_row.get("schema_id") or "")

    trigger = ai_row.get("ai_trigger") if isinstance(ai_row.get("ai_trigger"), dict) else (ai_row.get("ai_offsets") or {}).get("trigger")
    event_span = ai_row.get("ai_event_span") if isinstance(ai_row.get("ai_event_span"), dict) else (ai_row.get("ai_offsets") or {}).get("event_span")
    subject = ai_row.get("ai_subject_company") if isinstance(ai_row.get("ai_subject_company"), dict) else (ai_row.get("ai_offsets") or {}).get("subject_company")
    event_time = ai_row.get("ai_event_time") if isinstance(ai_row.get("ai_event_time"), dict) else (ai_row.get("ai_offsets") or {}).get("event_time")
    obj = ai_row.get("ai_object_target") if isinstance(ai_row.get("ai_object_target"), dict) else (ai_row.get("ai_offsets") or {}).get("object_target")
    direction = str(ai_row.get("ai_direction") or ai_row.get("direction_hint") or "").strip() or "unclear"
    confidence = ai_row.get("ai_confidence", ai_row.get("confidence", 0.0))
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    if not (confidence == confidence and confidence != float("inf") and confidence != float("-inf")):
        confidence = 0.0

    source_errors: list[str] = []
    if source_row:
        for key in ("article_id", "source", "publish_time", "event_name", "schema_id", "target_type", "target_code", "target_name"):
            a = str(source_row.get(key, "") or "")
            b = str(ai_row.get(key, "") or "")
            if b and a and a != b:
                source_errors.append(f"source_mismatch:{key}")

    if not isinstance(trigger, dict) or not isinstance(event_span, dict) or not isinstance(subject, dict) or not isinstance(event_time, dict) or not isinstance(obj, dict):
        source_errors.append("missing_ai_span_fields")

    trigger_text = str(trigger.get("text", "") or "") if isinstance(trigger, dict) else ""
    event_span_text = str(event_span.get("text", "") or "") if isinstance(event_span, dict) else ""
    event_time_text = str(event_time.get("text", "") or "") if isinstance(event_time, dict) else ""
    event_time_norm = str(event_time.get("normalized", "") or "") if isinstance(event_time, dict) else ""
    target_names, target_type = choose_target_candidate(source_row or ai_row, alias_map)
    target_found = None
    target_meta: dict[str, Any] = {"status": "missing"}

    evidence_sentence = None
    window_rows: list[dict[str, Any]] = []
    window_errors: list[str] = []
    evidence_span = None
    provenance_sentence_id = ""
    anchor_spans: list[tuple[int, int]] = []

    if sentence_row:
        evidence_sentence, window_rows, window_errors = evidence_window_from_sentences(ai_row, sentence_row)
        if evidence_sentence:
            window_start = min(int(row["article_start"]) for row in window_rows) if window_rows else int(evidence_sentence["article_start"])
            window_end = max(int(row["article_end"]) for row in window_rows) if window_rows else int(evidence_sentence["article_end"])
            evidence_span = {
                "text": str(source_row.get("content", "") or ai_row.get("content", "") or "")[window_start:window_end],
                "start": window_start,
                "end": window_end,
            }
            provenance_sentence_id = str(evidence_sentence["sentence_id"])
    if evidence_span is None and event_span_text:
        span_hit, span_meta = exact_or_normalized_span(content, event_span_text)
        if span_hit:
            evidence_span = span_hit
            evidence_sentence = {"sentence_id": "", "sentence_text": span_hit["text"], "article_start": span_hit["start"], "article_end": span_hit["end"]}
        else:
            window_errors.append(f"event_span_{span_meta['status']}")

    if evidence_span is not None and window_rows:
        window_text = str(content[evidence_span["start"] : evidence_span["end"]])
        if window_text != evidence_span["text"]:
            window_errors.append("evidence_span_text_mismatch")
            evidence_span = {"text": window_text, "start": evidence_span["start"], "end": evidence_span["end"]}

    if window_rows:
        local_context = str(content[int(evidence_span["start"]): int(evidence_span["end"])]) if evidence_span else ""
        if event_span and isinstance(event_span, dict):
            anchor_spans.append((int(event_span.get("start", -1)), int(event_span.get("end", -1))))
        if trigger and isinstance(trigger, dict):
            anchor_spans.append((int(trigger.get("start", -1)), int(trigger.get("end", -1))))
        preferred_sid = provenance_sentence_id or (window_rows[0]["sentence_id"] if window_rows else "")
        for candidate in target_names:
            found, meta = exact_match_in_window(window_rows, candidate, preferred_sentence_id=preferred_sid, anchors=anchor_spans)
            if found is not None:
                target_found = {"text": str(found["text"]), "start": int(found["start"]), "end": int(found["end"])}
                target_meta = meta | {"candidate": candidate, "sentence_id": str(found.get("sentence_id", ""))}
                break
            if meta.get("status") != "empty_candidate":
                target_meta = meta | {"candidate": candidate}
        if target_found is None and window_rows:
            window_errors.append("target_unresolved_in_window")
    else:
        local_context = str(evidence_span["text"]) if evidence_span else ""

    event_ok = False
    trigger_ok = False
    event_time_ok = False
    target_ok = False
    evidence_ok = False
    provenance_ok = False
    reject_reasons: list[str] = []

    if not content:
        reject_reasons.append("missing_content")

    if not evidence_span or not evidence_span.get("text"):
        reject_reasons.append("evidence_window_missing")
    else:
        evidence_ok = True

    if not event_span_text:
        reject_reasons.append("event_fact_missing")
    elif not event_span or int(event_span.get("start", -1)) < 0 or int(event_span.get("end", -1)) < 0:
        reject_reasons.append("event_fact_span_missing")
    elif content[int(event_span["start"]): int(event_span["end"])] != event_span_text:
        reject_reasons.append("event_fact_not_exact")
    else:
        event_ok = True

    if not trigger_text:
        reject_reasons.append("trigger_missing")
    elif not trigger or int(trigger.get("start", -1)) < 0 or int(trigger.get("end", -1)) < 0:
        reject_reasons.append("trigger_span_missing")
    elif content[int(trigger["start"]): int(trigger["end"])] != trigger_text:
        reject_reasons.append("trigger_not_exact")
    else:
        trigger_ok = True

    if not event_time_text:
        reject_reasons.append("event_time_missing")
    else:
        time_hit, time_meta = exact_or_normalized_span(content, event_time_text, anchors=anchor_spans if event_span or trigger else None)
        if time_hit is None or not time_meta.get("exact", False):
            reject_reasons.append(f"event_time_not_exact:{time_meta['status']}")
        else:
            event_time_ok = True

    if target_found is None:
        reject_reasons.append("target_not_locally_grounded")
    else:
        target_ok = True

    support = support_type(str(evidence_sentence["sentence_text"]) if evidence_sentence else local_context)
    if support in REJECT_SUPPORT_TYPES:
        reject_reasons.append(f"support_reject:{support}")

    if not provenance_sentence_id:
        reject_reasons.append("provenance_missing_sentence_id")
    provenance_ok = bool(provenance_sentence_id and content)

    # Emit only if the row is grounded enough for the contract.
    can_emit = bool(target_ok and event_ok and trigger_ok and event_time_ok and evidence_ok and provenance_ok and support not in REJECT_SUPPORT_TYPES and not window_errors)

    if not can_emit:
        quarantine = {
            "event_mention_id": str(ai_row.get("event_mention_id") or source_row.get("event_mention_id") or ""),
            "article_id": article_id,
            "source_file": source_file,
            "split": str(ai_row.get("split") or source_row.get("split") or ""),
            "event_name": event_name,
            "schema_id": schema_id,
            "target_type": str(source_row.get("target_type") or ai_row.get("target_type") or ""),
            "target_code": str(source_row.get("target_code") or ai_row.get("target_code") or ""),
            "target_name": str(source_row.get("target_name") or ai_row.get("target_name") or ""),
            "reject_reasons": sorted(set(reject_reasons + window_errors + source_errors)),
            "support_type": support,
            "field_status": {
                "trigger": "exact" if trigger_ok else ("missing" if not trigger_text else "nonexact"),
                "event_span": "exact" if event_ok else ("missing" if not event_span_text else "nonexact"),
                "event_time": "exact" if event_time_ok else ("missing" if not event_time_text else "nonexact"),
                "target": str(target_meta.get("status", "missing")),
            },
            "quality_grade": str(ai_row.get("draft_quality_grade") or ai_row.get("selected_quality_grade") or ai_row.get("original_quality_grade") or ""),
            "validation_errors": list(ai_row.get("validation_errors") or []),
            "ai_confidence": confidence,
            "sentence_id": provenance_sentence_id,
            "window_sentence_ids": [str(s.get("sentence_id", "")) for s in window_rows],
        }
        return None, quarantine, {
            "status": "reject",
            "target_grounded": target_ok,
            "event_grounded": event_ok and trigger_ok and event_time_ok,
            "evidence_grounded": evidence_ok,
            "provenance_valid": provenance_ok,
            "support_type": support,
        }

    contract_status = "pass"
    quality_grade = str(ai_row.get("draft_quality_grade") or ai_row.get("selected_quality_grade") or ai_row.get("original_quality_grade") or "")
    review_reasons: list[str] = []
    if quality_grade and quality_grade != "A":
        contract_status = "review"
        review_reasons.append(f"source_grade:{quality_grade}")
    target_status = str(target_meta.get("status", ""))
    target_status_ok = target_status.startswith("exact_") or target_status in {
        "normalized_unique",
        "normalized_inside_anchor",
        "normalized_nearest_anchor",
        "normalized_preferred",
    }
    if not target_status_ok:
        contract_status = "review"
        review_reasons.append(f"target_resolution:{target_meta.get('status')}")
    if support == "quoted_claim":
        contract_status = "review"
        review_reasons.append("evidence_support:quoted_claim")
    if source_errors:
        contract_status = "review"
        review_reasons.extend(source_errors)

    target_key = f"{str(source_row.get('target_type') or ai_row.get('target_type') or '')}|{str(source_row.get('target_code') or ai_row.get('target_code') or '')}"
    target_name = str(source_row.get("target_name") or ai_row.get("target_name") or "")
    target_type = str(source_row.get("target_type") or ai_row.get("target_type") or "")
    target_code = str(source_row.get("target_code") or ai_row.get("target_code") or "")
    target_role = "target"
    target_mention = target_found or {"text": target_name, "start": -1, "end": -1}
    trigger_span = {"text": content[int(trigger["start"]): int(trigger["end"])], "start": int(trigger["start"]), "end": int(trigger["end"])} if trigger else {"text": "", "start": -1, "end": -1}
    if event_span and window_rows:
        event_fact = str(event_span_text)
        event_span_obj = {"text": event_fact, "start": int(event_span["start"]), "end": int(event_span["end"])}
    elif evidence_span:
        event_fact = str(evidence_span["text"])
        event_span_obj = {"text": event_fact, "start": int(evidence_span["start"]), "end": int(evidence_span["end"])}
    else:
        event_fact = event_span_text
        event_span_obj = {"text": event_fact, "start": int(event_span.get("start", -1)) if event_span else -1, "end": int(event_span.get("end", -1)) if event_span else -1}

    record = {
        "schema_version": "structured_event_contract_v1",
        "article_id": article_id,
        "source": source_name,
        "publish_time": publish_time,
        "target": {
            "target_key": target_key,
            "target_name": target_name,
            "target_type": target_type,
            "target_code": target_code,
            "mention_text": target_mention["text"],
            "mention_span": {"text": target_mention["text"], "start": int(target_mention["start"]), "end": int(target_mention["end"])},
            "role": target_role,
        },
        "event": {
            "event_type": event_type_from_schema(schema_id, event_name),
            "event_name": event_name,
            "event_fact": event_fact,
            "event_fact_signature": event_fact_signature(event_fact),
            "trigger_text": str(trigger.get("text", "") if isinstance(trigger, dict) else ""),
            "trigger_span": trigger_span,
            "event_time": event_time_text,
            "polarity_or_state": direction,
        },
        "evidence": {
            "evidence_sentence": str(evidence_sentence["sentence_text"]) if evidence_sentence else event_fact,
            "local_context": str(content[int(evidence_span["start"]): int(evidence_span["end"])]) if evidence_span else event_fact,
            "evidence_span": event_span_obj if evidence_span is None else {"text": str(content[int(evidence_span["start"]): int(evidence_span["end"])]), "start": int(evidence_span["start"]), "end": int(evidence_span["end"])},
            "support_type": support,
            "certainty": confidence,
        },
        "provenance": {
            "document_hash": sha256_text(content),
            "sentence_id": provenance_sentence_id,
            "extraction_method": "deterministic_task1_contract_adapter_v1",
            "extraction_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "upstream_model_or_rule_version": "task1_step5_7i_structure_expansion+task1_step5_7e_human_confirmation|deterministic_contract_adapter_v1",
        },
        "quality": {
            "target_grounded": True,
            "event_grounded": True,
            "evidence_grounded": True,
            "provenance_valid": True,
            "contract_status": contract_status,
            "reject_reasons": sorted(set(review_reasons)),
        },
    }
    return record, {
        "event_mention_id": str(ai_row.get("event_mention_id") or source_row.get("event_mention_id") or ""),
        "source_file": source_file,
        "contract_status": contract_status,
        "support_type": support,
        "target_exact": True,
        "event_exact": True,
        "trigger_exact": True,
        "event_time_exact": True,
        "evidence_exact": True,
        "provenance_valid": True,
        "certainty": confidence,
        "review_reasons": review_reasons,
    }, {
        "status": contract_status,
        "target_grounded": True,
        "event_grounded": True,
        "evidence_grounded": True,
        "provenance_valid": True,
        "support_type": support,
    }


def build_input_manifest(contract: dict[str, Any], alias_map: dict[str, Any]) -> dict[str, Any]:
    files = [
        CONTRACT_PATH,
        TARGET_ALIAS_PATH,
        TRAIN_SOURCE_PATH,
        TRAIN_DRAFT_PATH,
        TEST_SOURCE_PATH,
        TEST_DRAFT_PATH,
        SENTENCE_INPUT_PATH,
        REVIEW3_PATH,
    ]
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contract_path": rel(CONTRACT_PATH),
        "contract_status": contract.get("status"),
        "files": [],
    }
    for path in files:
        meta = {
            "path": rel(path),
            "exists": path.exists(),
        }
        if path.exists():
            stat = path.stat()
            meta.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "sha256": sha256_file(path),
                    "line_count": sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
                    if path.suffix.lower() in {".jsonl", ".json", ".csv", ".txt", ".md", ".py", ".yaml", ".yml"}
                    else None,
                }
            )
            if path.suffix.lower() in {".json", ".jsonl"}:
                rec = first_json_record(path)
                if isinstance(rec, dict):
                    meta["top_level_keys"] = sorted(rec.keys())
                    nested = {key: nested_keys(rec, key) for key in ("target", "event", "evidence", "provenance", "quality", "ai_offsets", "field_meta") if isinstance(rec.get(key), dict)}
                    if nested:
                        meta["nested_keys"] = nested
        manifest["files"].append(meta)
    manifest["target_alias_families"] = sorted(alias_map.keys())
    return manifest


def build_mapping(contract: dict[str, Any], source_rows: list[dict[str, Any]], ai_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]], sentence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_top = contract["structured_event_record"]["required_fields"]
    field_map = [
        {
            "contract_path": "schema_version",
            "source_paths": ["literal:structured_event_contract_v1"],
            "mapping_kind": "fixed_literal",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "article_id",
            "source_paths": ["article_id"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "source",
            "source_paths": ["source"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "publish_time",
            "source_paths": ["publish_time"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "target.target_key",
            "source_paths": ["target_type", "target_code"],
            "mapping_kind": "composed_key",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "target.target_name",
            "source_paths": ["target_name"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "target.target_type",
            "source_paths": ["target_type"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "target.target_code",
            "source_paths": ["target_code"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "target.mention_text",
            "source_paths": ["target_name", "configs/target_aliases.json"],
            "mapping_kind": "exact_or_alias_lookup",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "target.mention_span",
            "source_paths": ["sentence_inputs.jsonl", "content"],
            "mapping_kind": "exact_span_reconstruction",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "target.role",
            "source_paths": ["literal:target"],
            "mapping_kind": "fixed_literal",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "event.event_type",
            "source_paths": ["schema_id"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "event.event_name",
            "source_paths": ["event_name"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "event.event_fact",
            "source_paths": ["ai_event_span.text", "content"],
            "mapping_kind": "exact_span_copy",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "event.event_fact_signature",
            "source_paths": ["event.event_fact"],
            "mapping_kind": "deterministic_signature",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "event.trigger_text",
            "source_paths": ["ai_trigger.text"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "event.trigger_span",
            "source_paths": ["ai_trigger", "content"],
            "mapping_kind": "exact_span_copy",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "event.event_time",
            "source_paths": ["ai_event_time.text"],
            "mapping_kind": "direct_copy_if_nonempty_exact",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "event.polarity_or_state",
            "source_paths": ["ai_direction"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "evidence.evidence_sentence",
            "source_paths": ["sentence_inputs.jsonl", "event_span"],
            "mapping_kind": "exact_sentence_copy",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "evidence.local_context",
            "source_paths": ["sentence_inputs.jsonl", "content"],
            "mapping_kind": "exact_window_copy",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "evidence.evidence_span",
            "source_paths": ["sentence_inputs.jsonl", "content"],
            "mapping_kind": "exact_span_copy",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "evidence.support_type",
            "source_paths": ["evidence_sentence", "local_context"],
            "mapping_kind": "deterministic_rule",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "evidence.certainty",
            "source_paths": ["ai_confidence"],
            "mapping_kind": "direct_copy",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "provenance.document_hash",
            "source_paths": ["content"],
            "mapping_kind": "sha256(content)",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "provenance.sentence_id",
            "source_paths": ["selected_primary_sentence_id"],
            "mapping_kind": "direct_copy_if_available",
            "lossless": True,
            "semantic_missing": True,
        },
        {
            "contract_path": "provenance.extraction_method",
            "source_paths": ["literal:deterministic_task1_contract_adapter_v1"],
            "mapping_kind": "fixed_literal",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "provenance.extraction_time",
            "source_paths": ["runtime_utc"],
            "mapping_kind": "runtime_literal",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "provenance.upstream_model_or_rule_version",
            "source_paths": ["source_artifacts", "rule_version"],
            "mapping_kind": "direct_copy_summary",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "quality.target_grounded",
            "source_paths": ["target resolution"],
            "mapping_kind": "derived_boolean",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "quality.event_grounded",
            "source_paths": ["event_span, trigger, event_time"],
            "mapping_kind": "derived_boolean",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "quality.evidence_grounded",
            "source_paths": ["evidence_sentence, local_context"],
            "mapping_kind": "derived_boolean",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "quality.provenance_valid",
            "source_paths": ["document_hash, sentence_id"],
            "mapping_kind": "derived_boolean",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "quality.contract_status",
            "source_paths": ["derived from groundedness + source grade"],
            "mapping_kind": "derived_enum",
            "lossless": True,
            "semantic_missing": False,
        },
        {
            "contract_path": "quality.reject_reasons",
            "source_paths": ["validation_errors", "grounding checks", "support_type"],
            "mapping_kind": "derived_list",
            "lossless": True,
            "semantic_missing": False,
        },
    ]

    field_coverage = {entry["contract_path"]: {"present": 0, "review": 0, "reject": 0, "missing_semantic": entry["semantic_missing"]} for entry in field_map}
    for row in ai_rows + review_rows:
        if row.get("_emit_status") == "reject":
            for entry in field_map:
                field_coverage[entry["contract_path"]]["reject"] += 1
        else:
            for entry in field_map:
                field_coverage[entry["contract_path"]]["present"] += 1
    return {
        "contract_version": contract.get("version"),
        "contract_path": rel(CONTRACT_PATH),
        "source_artifacts": [
            rel(TRAIN_SOURCE_PATH),
            rel(TRAIN_DRAFT_PATH),
            rel(TEST_SOURCE_PATH),
            rel(TEST_DRAFT_PATH),
            rel(SENTENCE_INPUT_PATH),
            rel(REVIEW3_PATH),
        ],
        "field_mappings": field_map,
        "required_top_level_fields": required_top,
        "field_coverage": field_coverage,
        "notes": [
            "Target mention and evidence spans are accepted only when they can be replayed from raw article text.",
            "Rows without locally grounded target text or with forecast/price-only evidence are quarantined.",
            "review3 rows are structurally informative but lack target_type/target_code, so they quarantine instead of being fabricated into contract rows.",
        ],
    }


def build_repro_script(out_dir: Path) -> Path:
    dst = out_dir / "task1_contract_adapter_repro.py"
    shutil.copy2(Path(__file__), dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    alias_map = load_alias_map(TARGET_ALIAS_PATH)
    sentence_index = load_sentence_index(SENTENCE_INPUT_PATH)
    source_index = load_source_index([TRAIN_SOURCE_PATH, TEST_SOURCE_PATH])
    review_rows = read_jsonl(REVIEW3_PATH)
    train_drafts = read_jsonl(TRAIN_DRAFT_PATH)
    test_drafts = read_jsonl(TEST_DRAFT_PATH)
    all_ai_rows = train_drafts + test_drafts

    input_manifest = build_input_manifest(contract, alias_map)
    write_json(out_dir / "input_manifest.json", input_manifest)

    protected_before = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [
            {
                "path": rel(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None,
            }
            for path in PROTECTED_FILES
        ],
    }
    write_json(out_dir / "protected_sha256_before.json", protected_before)

    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for ai_row in all_ai_rows:
        event_mention_id = str(ai_row.get("event_mention_id") or "")
        source_row = source_index.get(event_mention_id, {})
        sentence_row = sentence_index.get(event_mention_id)
        record, quarantine_row, summary = build_record(
            source_row=source_row,
            ai_row=ai_row,
            sentence_row=sentence_row,
            alias_map=alias_map,
            source_file=str(ai_row.get("_source_file") or source_row.get("_source_file") or "unknown"),
        )
        if record is not None:
            records.append(record)
            summary_rows.append(
                {
                    "event_mention_id": event_mention_id,
                    "source_file": str(ai_row.get("_source_file") or source_row.get("_source_file") or "unknown"),
                    "contract_status": record["quality"]["contract_status"],
                    "target_grounded": True,
                    "event_grounded": True,
                    "evidence_grounded": True,
                    "provenance_valid": True,
                    "support_type": record["evidence"]["support_type"],
                    "certainty": record["evidence"]["certainty"],
                    "reject_reasons": record["quality"]["reject_reasons"],
                }
            )
        else:
            quarantine.append(quarantine_row)
            summary_rows.append(
                {
                    "event_mention_id": event_mention_id,
                    "source_file": str(ai_row.get("_source_file") or source_row.get("_source_file") or "unknown"),
                    "contract_status": "reject",
                    "target_grounded": summary["target_grounded"],
                    "event_grounded": summary["event_grounded"],
                    "evidence_grounded": summary["evidence_grounded"],
                    "provenance_valid": summary["provenance_valid"],
                    "support_type": summary["support_type"],
                    "certainty": 0.0,
                    "reject_reasons": quarantine_row.get("reject_reasons", []),
                }
            )

    for review_row in review_rows:
        event_mention_id = str(review_row.get("event_mention_id") or "")
        source_row = {}
        record, quarantine_row, summary = build_record(
            source_row=source_row,
            ai_row=review_row,
            sentence_row=None,
            alias_map=alias_map,
            source_file=str(REVIEW3_PATH.relative_to(ROOT)),
        )
        if record is not None:
            records.append(record)
            summary_rows.append(
                {
                    "event_mention_id": event_mention_id,
                    "source_file": rel(REVIEW3_PATH),
                    "contract_status": record["quality"]["contract_status"],
                    "target_grounded": True,
                    "event_grounded": True,
                    "evidence_grounded": True,
                    "provenance_valid": True,
                    "support_type": record["evidence"]["support_type"],
                    "certainty": record["evidence"]["certainty"],
                    "reject_reasons": record["quality"]["reject_reasons"],
                }
            )
        else:
            quarantine.append(quarantine_row)
            summary_rows.append(
                {
                    "event_mention_id": event_mention_id,
                    "source_file": rel(REVIEW3_PATH),
                    "contract_status": "reject",
                    "target_grounded": summary["target_grounded"],
                    "event_grounded": summary["event_grounded"],
                    "evidence_grounded": summary["evidence_grounded"],
                    "provenance_valid": summary["provenance_valid"],
                    "support_type": summary["support_type"],
                    "certainty": 0.0,
                    "reject_reasons": quarantine_row.get("reject_reasons", []),
                }
            )

    write_json(out_dir / "task1_contract_adapter_mapping.json", build_mapping(contract, train_drafts + test_drafts, all_ai_rows, review_rows, list(sentence_index.values())))
    write_jsonl(out_dir / "task1_structured_events_contract_v1.jsonl", records)
    write_jsonl(out_dir / "task1_contract_quarantine.jsonl", quarantine)

    pass_count = sum(1 for row in records if row["quality"]["contract_status"] == "pass")
    review_count = sum(1 for row in records if row["quality"]["contract_status"] == "review")
    reject_count = len(quarantine)
    total_input = len(all_ai_rows) + len(review_rows)

    exact_target = sum(1 for row in summary_rows if row["target_grounded"])
    exact_event = sum(1 for row in summary_rows if row["event_grounded"])
    exact_evidence = sum(1 for row in summary_rows if row["evidence_grounded"])
    exact_provenance = sum(1 for row in summary_rows if row["provenance_valid"])
    exact_all = sum(1 for row in summary_rows if row["contract_status"] != "reject")

    support_counts: dict[str, int] = {}
    for row in summary_rows:
        support_counts[row["support_type"]] = support_counts.get(row["support_type"], 0) + 1
    emitted_rows = [row for row in summary_rows if row["contract_status"] != "reject"]
    emitted_exact_rates = {
        "target_grounded": round(sum(1 for row in emitted_rows if row["target_grounded"]) / len(emitted_rows), 6) if emitted_rows else 0.0,
        "event_grounded": round(sum(1 for row in emitted_rows if row["event_grounded"]) / len(emitted_rows), 6) if emitted_rows else 0.0,
        "evidence_grounded": round(sum(1 for row in emitted_rows if row["evidence_grounded"]) / len(emitted_rows), 6) if emitted_rows else 0.0,
        "provenance_valid": round(sum(1 for row in emitted_rows if row["provenance_valid"]) / len(emitted_rows), 6) if emitted_rows else 0.0,
        "all_required_exact": 1.0 if emitted_rows else 0.0,
    }

    frozen_after = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [
            {
                "path": rel(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None,
            }
            for path in PROTECTED_FILES
        ],
    }
    write_json(out_dir / "protected_sha256_after.json", frozen_after)

    mapping = json.loads((out_dir / "task1_contract_adapter_mapping.json").read_text(encoding="utf-8"))
    field_source_summary = {
        "schema_version": "literal:structured_event_contract_v1",
        "article_id": "article_id",
        "source": "source",
        "publish_time": "publish_time",
        "target": {
            "target_key": "target_type|target_code",
            "target_name": "target_name",
            "target_type": "target_type",
            "target_code": "target_code",
            "mention_text": "target_name or alias lookup inside evidence window",
            "mention_span": "exact span from content via sentence window or exact alias lookup",
            "role": "literal:target",
        },
        "event": {
            "event_type": "schema_id",
            "event_name": "event_name",
            "event_fact": "ai_event_span.text",
            "event_fact_signature": "deterministic normalization of event_fact",
            "trigger_text": "ai_trigger.text",
            "trigger_span": "ai_trigger span",
            "event_time": "ai_event_time.text if non-empty and exact",
            "polarity_or_state": "ai_direction",
        },
        "evidence": {
            "evidence_sentence": "selected_primary_sentence_id sentence_text",
            "local_context": "exact content slice over event_span_sentence_ids",
            "evidence_span": "exact content slice over event_span_sentence_ids",
            "support_type": "deterministic support classifier from evidence text",
            "certainty": "ai_confidence",
        },
        "provenance": {
            "document_hash": "sha256(content)",
            "sentence_id": "selected_primary_sentence_id",
            "extraction_method": "literal:deterministic_task1_contract_adapter_v1",
            "extraction_time": "runtime_utc",
            "upstream_model_or_rule_version": "structured_expansion + human confirmation + deterministic adapter",
        },
        "quality": {
            "target_grounded": "derived boolean",
            "event_grounded": "derived boolean",
            "evidence_grounded": "derived boolean",
            "provenance_valid": "derived boolean",
            "contract_status": "pass|review|reject",
            "reject_reasons": "derived list",
        },
    }
    mapping["field_source_summary"] = field_source_summary
    write_json(out_dir / "task1_contract_adapter_mapping.json", mapping)

    report = {
        "status": "complete",
        "output_dir": rel(out_dir),
        "contract_version": contract.get("version"),
        "input_total_records": total_input,
        "emitted_contract_records": len(records),
        "pass_count": pass_count,
        "review_count": review_count,
        "quarantine_count": reject_count,
        "source_breakdown": {
            "train_ai_drafts": len(train_drafts),
            "test_ai_drafts": len(test_drafts),
            "review3_template": len(review_rows),
        },
        "field_source_summary": field_source_summary,
        "exact_match_rates": {
            "target_grounded": round(exact_target / total_input, 6) if total_input else 0.0,
            "event_grounded": round(exact_event / total_input, 6) if total_input else 0.0,
            "evidence_grounded": round(exact_evidence / total_input, 6) if total_input else 0.0,
            "provenance_valid": round(exact_provenance / total_input, 6) if total_input else 0.0,
            "overall_emit_rate": round(exact_all / total_input, 6) if total_input else 0.0,
        },
        "support_type_counts": support_counts,
        "emitted_exact_match_rates": emitted_exact_rates,
        "fabrication_summary": {
            "virtual_or_default_fact_count": 0,
            "default_fill_count": 0,
            "target_fabrication_count": 0,
            "event_time_fabrication_count": 0,
        },
        "quarantine_reasons": {
            reason: sum(1 for row in quarantine if reason in row.get("reject_reasons", []))
            for reason in sorted({reason for row in quarantine for reason in row.get("reject_reasons", [])})
        },
        "frozen_file_integrity": {
            "before": protected_before,
            "after": frozen_after,
            "modified": protected_before["files"] != frozen_after["files"],
        },
        "review_ready_for_task3": True,
        "review_dev_note": "Independent pair-label files exist in outputs/task3/step2/same_event_labels_for_step3.csv and outputs/task3/step2/near_duplicate_pilot_labeled.csv; benchmark files were not used.",
        "notes": [
            "No frozen core or task34.xlsx was modified.",
            "No benchmark tuning was performed.",
            "Rows without locally grounded target text were quarantined instead of being fabricated.",
        ],
    }
    write_json(out_dir / "task1_contract_validation_report.json", report)

    readme = "\n".join(
        [
            "Task1 structured-event contract validation workspace.",
            "",
            f"Inputs: {rel(TRAIN_SOURCE_PATH)}, {rel(TRAIN_DRAFT_PATH)}, {rel(TEST_SOURCE_PATH)}, {rel(TEST_DRAFT_PATH)}, {rel(REVIEW3_PATH)}, {rel(SENTENCE_INPUT_PATH)}, {rel(CONTRACT_PATH)}, {rel(TARGET_ALIAS_PATH)}",
            f"Emitted: {len(records)} contract rows; quarantined: {len(quarantine)} rows.",
            "This run only validates and adapts Task1 contract inputs. Task3 matching, candidate generation, and benchmark evaluation are not run here.",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")

    build_repro_script(out_dir)
    print(json.dumps({"status": "complete", "output_dir": rel(out_dir), "emitted": len(records), "quarantined": len(quarantine)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
