"""Deterministic claim semantic normalization for Step 4.5-A.

The adapter never reads evidence and never invents facts. It only uses the
claim and structured metadata supplied by the caller.
"""

from __future__ import annotations

import re
from typing import Any


ADAPTER_VERSION = "claim_contract_v1"
VALID_MODES = {
    "specific_fact",
    "category_existence",
    "structurally_unverifiable",
}

GENERIC_CLAIM_PATTERNS = (
    re.compile(r"^关于.+的.+相关主张$"),
    re.compile(r"^关于.+的.+相关事项$"),
    re.compile(r"^关于.+的.+相关信息$"),
)
SCHEMA_PATTERN = re.compile(r"(?i)^(company|industry)_(\d{6})_(.+)$")
ACTION_TERMS = (
    "收购",
    "并购",
    "重组",
    "发行",
    "回购",
    "签署",
    "签订",
    "发布",
    "推出",
    "量产",
    "投产",
    "开工",
    "竣工",
    "获批",
    "申请专利",
    "获得专利",
    "研发",
    "技术突破",
    "战略合作",
    "扩产",
    "停产",
    "定价",
    "交付",
)
FACT_MARKERS = (
    "%",
    "亿元",
    "万元",
    "万股",
    "万吨",
    "股份",
    "股权",
    "金额",
    "比例",
    "首款",
    "首台",
    "合同",
    "协议",
)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def parse_schema_id(schema_id: Any) -> dict[str, str]:
    raw = norm(schema_id)
    match = SCHEMA_PATTERN.fullmatch(raw)
    if not match:
        return {
            "raw_schema_id": raw,
            "entity_type": "",
            "entity_code": "",
            "event_type": "",
            "parse_status": "unparseable" if raw else "missing",
        }
    return {
        "raw_schema_id": raw,
        "entity_type": match.group(1).lower(),
        "entity_code": match.group(2),
        "event_type": norm(match.group(3)),
        "parse_status": "parsed",
    }


def resolve_event_type(event_name: Any, schema_id: Any) -> tuple[str, str]:
    event = norm(event_name)
    if event:
        return event, "event_name"
    parsed = parse_schema_id(schema_id)
    if parsed["event_type"]:
        return parsed["event_type"], "schema_id"
    return "", "unresolved"


def looks_like_specific_fact(claim: str) -> bool:
    value = norm(claim)
    if not value or any(pattern.fullmatch(value) for pattern in GENERIC_CLAIM_PATTERNS):
        return False
    if re.search(r"(?i)(?:company|industry)_\d{6}_", value):
        return False
    has_action = any(term in value for term in ACTION_TERMS)
    has_fact = any(term in value for term in FACT_MARKERS) or bool(re.search(r"\d", value))
    return has_action and (has_fact or len(value) >= 18)


def build_claim_contract(
    claim: Any,
    company_or_subject: Any,
    schema_id: Any,
    event_name: Any,
    event_time: Any,
) -> dict[str, str]:
    raw_claim = norm(claim)
    subject = norm(company_or_subject)
    raw_schema = norm(schema_id)
    event_type, event_source = resolve_event_type(event_name, raw_schema)
    time_scope = norm(event_time) or "未提供事件时间"

    if looks_like_specific_fact(raw_claim):
        mode = "specific_fact"
        normalized = raw_claim
        policy = (
            "核验原始主张中的具体动作、对象和事实；support或contradict必须有直接证据，"
            "否则输出insufficient；只有主张本身无法理解时输出unclear。"
        )
    elif subject and event_type:
        mode = "category_existence"
        normalized = (
            f"在给定时间范围内，是否存在至少一项主体为{subject}、"
            f"事件类型属于{event_type}的具体事件？"
        )
        policy = (
            "核验主体在给定时间范围内是否发生属于该事件类别的具体事实；"
            "只有相关讨论、传闻、行情或投资者提问不能support；"
            "权威明确否认可contradict；证据不足则insufficient。"
        )
    else:
        mode = "structurally_unverifiable"
        normalized = "无法从当前主张和结构化字段形成可验证的主体—事件核验问题。"
        policy = "主张结构无法恢复时输出unclear，不得根据证据猜测主体或事件类型。"

    return {
        "raw_claim": raw_claim,
        "claim_mode": mode,
        "normalized_claim": normalized,
        "subject": subject,
        "event_type": event_type,
        "time_scope": time_scope,
        "verification_policy": policy,
        "adapter_version": ADAPTER_VERSION,
        "event_type_source": event_source,
        "schema_parse_status": parse_schema_id(raw_schema)["parse_status"],
    }

