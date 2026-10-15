#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official Task1 field contract.

The contract is intentionally field-level, not DocFEE/argument-role style.
External event types are never treated as official event_name labels unless a
manual mapping marks them as exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_EMOTIONS = {"正面", "负面", "中性", "positive", "negative", "neutral"}


@dataclass(frozen=True)
class EventTypeMapping:
    source_dataset: str
    source_event_type: str
    official_event_name: str | None
    mapping_status: str
    confidence: float
    rationale: str


def load_official_event_names(company_json: Path, industry_json: Path) -> set[str]:
    names: set[str] = set()
    for path in [company_json, industry_json]:
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data:
            for item in record.get("event_schema", []):
                name = str(item.get("event_name") or "").strip()
                if name:
                    names.add(name)
    return names


def default_external_event_mappings() -> list[EventTypeMapping]:
    """Conservative selected mapping table.

    Detailed candidate mappings live in
    ``configs/task1_external_to_official64_mapping_v1.yaml`` after Step 5.8-B
    mapping audit.  This function returns the selected hard-label mapping only.
    Near matches remain unselected and are marked partial_overlap/none, so they
    can be used for shared extraction pretraining without entering official
    64-class classification loss.
    """
    docfee = {
        "破产清算": ("partial_overlap", "Near 现金流风险, but official definition is public debt repayment pressure, not bankruptcy liquidation."),
        "股东减持": ("partial_overlap", "Shareholder trading overlaps capital-market behavior but official class is target-specific and broader/different."),
        "股东增持": ("partial_overlap", "Shareholder trading overlaps capital-market behavior but official class is target-specific and broader/different."),
        "股权冻结": ("partial_overlap", "Equity legal status overlaps financial risk/capital-market signals but no exact official class."),
        "股权质押": ("partial_overlap", "Equity pledge overlaps financial risk/capital-market signals but no exact official class."),
        "重大安全事故": ("none", "Safety incident; no exact official 64-class equivalent."),
        "重大对外赔付": ("none", "External compensation event; no exact official 64-class equivalent."),
        "重大资产损失": ("none", "Asset loss event; no exact official 64-class equivalent."),
        "高层死亡": ("none", "Executive death event; no exact official 64-class equivalent."),
    }
    chfinann = {
        "EquityOverweight": ("partial_overlap", "Shareholder increase overlaps capital-market behavior but is not exact official 64-class event."),
        "EquityUnderweight": ("partial_overlap", "Shareholder reduction overlaps capital-market behavior but is not exact official 64-class event."),
        "EquityFreeze": ("partial_overlap", "Equity freeze overlaps risk/capital-market signals but no exact official class."),
        "EquityPledge": ("partial_overlap", "Equity pledge overlaps risk/capital-market signals but no exact official class."),
        "EquityRepurchase": ("partial_overlap", "Share repurchase overlaps capital-market behavior but no exact official class."),
    }
    rows: list[EventTypeMapping] = []
    for source, mapping in [("docfee", docfee), ("chfinann", chfinann)]:
        for event_type, (status, rationale) in mapping.items():
            rows.append(
                EventTypeMapping(
                    source_dataset=source,
                    source_event_type=event_type,
                    official_event_name=None,
                    mapping_status=status,
                    confidence=0.0,
                    rationale=rationale,
                )
            )
    return rows


def source_role_mapping() -> dict[str, dict[str, str]]:
    return {
        "docfee": {
            "Company Name": "target_name",
            "Industry": "impact_target",
            "Announcement Date": "event_time",
            "Judgment Date": "event_time",
            "Pledge Start Date": "event_time",
            "Pledge End Date": "event_time",
            "Reduction Start Date": "event_time",
            "Increase Start Date": "event_time",
            "Court of Acceptance": "impact_target",
            "Shareholder": "impact_target",
            "Pledgor": "impact_target",
            "Receiver": "impact_target",
            "Frozen Shareholder": "impact_target",
            "Other Impacts": "impact_target",
            "Loss Amount": "impact_amount",
            "Reduction Amount": "impact_amount",
            "Increase Amount": "impact_amount",
            "Pledge Amount": "impact_amount",
            "Freeze Amount": "impact_amount",
        },
        "chfinann": {
            "CompanyName": "target_name",
            "EquityHolder": "impact_target",
            "Pledger": "impact_target",
            "Pledgee": "impact_target",
            "LegalInstitution": "impact_target",
            "StartDate": "event_time",
            "EndDate": "event_time",
            "ReleasedDate": "event_time",
            "UnfrozeDate": "event_time",
            "ClosingDate": "event_time",
            "TradedShares": "impact_amount",
            "PledgedShares": "impact_amount",
            "FrozeShares": "impact_amount",
            "RepurchasedShares": "impact_amount",
            "RepurchaseAmount": "impact_amount",
            "AveragePrice": "impact_amount",
            "HighestTradingPrice": "impact_amount",
            "LowestTradingPrice": "impact_amount",
            "TotalHoldingShares": "impact_amount",
            "TotalHoldingRatio": "impact_amount",
            "TotalPledgedShares": "impact_amount",
            "LaterHoldingShares": "impact_amount",
        },
    }


def validate_official_record(record: dict[str, Any], official_event_names: set[str]) -> list[str]:
    errors: list[str] = []
    for field in ["target_type", "target_code", "target_name", "schema_id", "event_name"]:
        if not str(record.get(field, "")).strip():
            errors.append(f"missing_{field}")
    if record.get("target_type") not in {"company", "industry"}:
        errors.append("illegal_target_type")
    if record.get("event_name") not in official_event_names:
        errors.append("illegal_event_name")
    if str(record.get("event_emotion", "")).strip() and record.get("event_emotion") not in OFFICIAL_EMOTIONS:
        errors.append("illegal_event_emotion")
    for ev in record.get("evidence", []) or []:
        quote = str(ev.get("quote", "") or "")
        content = str(ev.get("content", "") or "")
        if quote and content and quote not in content:
            errors.append("evidence_quote_not_in_content")
    return errors
