#!/usr/bin/env python3
"""build_event_quality_gate.py — 对全量新闻级预测进行质量分层"""

import json, os, sys
from collections import defaultdict
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs/unlabeled_event_inference/full")
CSV_DIR = os.path.join(PROJECT_ROOT, "reports")
OUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/event_extraction")
os.makedirs(OUT_DIR, exist_ok=True)

# Load suspicious predictions
sus_set = set()
sus_path = os.path.join(CSV_DIR, "unlabeled_full_suspicious_predictions.csv")
if os.path.exists(sus_path):
    df_sus = pd.read_csv(sus_path)
    for _, row in df_sus.iterrows():
        sus_set.add((str(row["article_id"]), str(row["target_code"]), str(row["target_type"])))

stats = {}
for ds in ["train", "test"]:
    news = []
    with open(os.path.join(PRED_DIR, f"news_predictions_{ds}.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line: news.append(json.loads(line))

    auto_accept = []; joint_adj = []; manual_review = []; rejected = []

    for n in news:
        aid = n["article_id"]; tc = n.get("target_code",""); tt = n.get("target_type","")
        pec = n["predicted_event_count"]
        events = n.get("events", [])
        routing = n.get("routing", {})
        route_conf = routing.get("route_confidence", 0)
        route_method = routing.get("route_method", "")
        matched_alias = routing.get("matched_alias", "")

        # Determine suspicious reasons
        is_sus = (aid, tc, tt) in sus_set
        sus_reasons = []
        if pec >= 3: sus_reasons.append("too_many_events_for_target")
        high_events = [e for e in events if e.get("confidence_level") == "high"]
        if len(high_events) >= 3: sus_reasons.append("multiple_high_confidence_events")
        if len(events) >= 2:
            probs = [e["probability"] for e in events]
            if probs[0] - probs[1] < 0.03: sus_reasons.append("ambiguous_top_events")
        near_threshold = [e for e in events if 0.38 <= e.get("probability", 0) < 0.45]
        if near_threshold: sus_reasons.append("near_threshold")

        # Build quality record for each event
        for e in events:
            base = {
                "article_id": aid, "title": n.get("title", ""),
                "content": n.get("content", "") if "content" in n else "",
                "publish_time": n.get("publish_time", ""), "source": n.get("source", ""),
                "target_type": tt, "target_code": tc, "target_name": n["target_name"],
                "route_method": route_method, "route_confidence": route_conf,
                "matched_alias": matched_alias,
                "schema_id": e["schema_id"], "event_name": e["event_name"],
                "event_description": e.get("event_description", ""),
                "macbert_probability": e["probability"],
                "confidence_level": e.get("confidence_level", ""),
                "predicted_event_count": pec,
                "suspicious_reasons": sus_reasons,
            }

            prob = e["probability"]
            has_sus = len(sus_reasons) > 0

            # Quality gating (strict: auto_accept requires ALL conditions met)
            if pec == 1 and prob >= 0.80 and route_conf >= 0.95 and not has_sus:
                base["quality_status"] = "auto_accept"
                base["quality_score"] = 1.0
                auto_accept.append(base)
            elif pec >= 2 or any(r in str(sus_reasons) for r in ["multiple_high_confidence", "ambiguous_top_events", "too_many_events"]):
                # Goes to joint adjudication group level (handled separately)
                pass
            else:
                # Everything else goes to manual review
                base["quality_status"] = "manual_review"
                base["quality_score"] = round(0.3 + 0.3 * prob + 0.4 * route_conf, 2)
                manual_review.append(base)

        # Handle no-event groups
        if not events:
            manual_review.append({
                "article_id": aid, "title": n.get("title", ""),
                "content": n.get("content", "") if "content" in n else "",
                "target_type": tt, "target_code": tc, "target_name": n["target_name"],
                "route_method": route_method, "route_confidence": route_conf,
                "quality_status": "reject", "quality_score": 0.0,
                "predicted_event_count": 0,
            })

    # Now handle joint adjudication groups (separate from individual events)
    # Group by (article_id, target_code) for multi-event cases
    j_groups = defaultdict(list)
    for n in news:
        pec = n["predicted_event_count"]
        events = n.get("events", [])
        if pec >= 2:
            gk = (n["article_id"], n["target_code"])
            j_groups[gk].append(n)

    jg_records = []
    for gk, items in j_groups.items():
        n = items[0]  # Take first occurrence
        jg_records.append({
            "article_id": n["article_id"], "title": n.get("title", ""),
            "content": n.get("content", "") if "content" in n else "",
            "publish_time": n.get("publish_time", ""), "source": n.get("source", ""),
            "target_type": n["target_type"], "target_code": n["target_code"],
            "target_name": n["target_name"],
            "route_method": n.get("routing", {}).get("route_method", ""),
            "route_confidence": n.get("routing", {}).get("route_confidence", 0),
            "events": n["events"],
            "predicted_event_count": n["predicted_event_count"],
            "quality_status": "joint_adjudication",
            "suspicious_reasons": [],  # populated below
        })
    # Add suspicious reasons at group level
    for jg in jg_records:
        aid, tc = jg["article_id"], jg["target_code"]
        if (aid, tc, jg["target_type"]) in sus_set:
            jg["suspicious_reasons"] = ["too_many_events_for_target", "multiple_high_confidence_events"]

    # Save
    with open(os.path.join(OUT_DIR, f"auto_accept_events_{ds}.jsonl"), "w", encoding="utf-8") as f:
        for a in auto_accept: f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, f"joint_adjudication_groups_{ds}.jsonl"), "w", encoding="utf-8") as f:
        for j in jg_records: f.write(json.dumps(j, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, f"manual_review_events_{ds}.jsonl"), "w", encoding="utf-8") as f:
        for m in manual_review: f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Combined quality-gated
    combined = auto_accept + manual_review + jg_records
    with open(os.path.join(OUT_DIR, f"quality_gated_{ds}.jsonl"), "w", encoding="utf-8") as f:
        for c in combined: f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats[ds] = {
        "total_news_groups": len(news),
        "auto_accept_count": len(auto_accept),
        "joint_adjudication_groups": len(jg_records),
        "manual_review_count": len(manual_review),
        "rejected_count": sum(1 for m in manual_review if m.get("quality_status") == "reject"),
    }
    print(f"[{ds}] auto_accept={len(auto_accept)}, joint_adj_groups={len(jg_records)}, manual_review={len(manual_review)}, total={len(news)}")

with open(os.path.join(PROJECT_ROOT, "reports/quality_gate_summary.json"), "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("\nQuality gate complete!")
