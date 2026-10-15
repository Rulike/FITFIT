#!/usr/bin/env python3
"""aggregate_final_task1_results.py — 汇总任务一最终结果，生成统计报告和交付文件"""

import json, os, sys
from collections import Counter, defaultdict
import pandas as pd; import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_DIR = os.path.join(PROJECT_ROOT, "data/processed/event_extraction")
ADJ_DIR = os.path.join(PROJECT_ROOT, "outputs/final_task1")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/final_task1")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs/unlabeled_event_inference/full")

os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(REPORTS_DIR, exist_ok=True)


def load_jsonl(fp):
    r = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line: r.append(json.loads(line))
    return r


def main():
    all_final = {"train": [], "test": []}
    all_rejected = []
    all_manual = []
    agg_stats = {}

    for ds in ["train", "test"]:
        print(f"\n=== Aggregating {ds} ===")

        # 1. Auto-accept events
        auto = load_jsonl(os.path.join(QA_DIR, f"auto_accept_events_{ds}.jsonl"))
        for a in auto:
            all_final[ds].append({
                "event_id": f"final_{ds}_{a['article_id']}_{a['schema_id']}",
                "article_id": a["article_id"], "title": a.get("title", ""),
                "publish_time": a.get("publish_time", ""), "source": a.get("source", ""),
                "target_type": a["target_type"], "target_code": a["target_code"],
                "target_name": a["target_name"],
                "schema_id": a["schema_id"], "event_name": a["event_name"],
                "event_match_probability": a.get("macbert_probability", 0),
                "route_confidence": a.get("route_confidence", 0),
                "quality_status": "auto_accept",
                "final_confidence_score": round(0.30 * a.get("route_confidence", 0) + 0.30 * a.get("macbert_probability", 0) + 0.25 * 1.0 + 0.15 * 0.9, 2),
                "final_confidence_level": "high" if (0.30 * a.get("route_confidence", 0) + 0.30 * a.get("macbert_probability", 0) + 0.25 * 1.0 + 0.15 * 0.9) >= 0.85 else "medium",
                "final_status": "auto_accept",
                "event_subject_name": "",
                "event_subject_type": "company" if a["target_type"] == "company" else "industry",
                "event_time_raw": "", "event_time_normalized": "", "event_time_precision": "unknown",
                "event_action": "", "event_object": "", "event_status": "unknown",
                "location": "", "impact_direction": "uncertain",
                "evidence_sentences": [], "extraction_confidence": 0.0,
            })

        # 2. Joint adjudication results
        adj_file = os.path.join(ADJ_DIR, f"joint_adjudication_{ds}.jsonl")
        if os.path.exists(adj_file):
            adj_results = load_jsonl(adj_file)
            for r in adj_results:
                for s in r.get("selected_events", []):
                    all_final[ds].append({
                        "event_id": f"final_{ds}_{r['article_id']}_{s['schema_id']}",
                        "article_id": r["article_id"], "title": r.get("title", ""),
                        "target_type": r["target_type"], "target_code": r["target_code"],
                        "target_name": r["target_name"],
                        "schema_id": s["schema_id"], "event_name": s.get("event_name", ""),
                        "event_match_probability": 0,
                        "route_confidence": 0,
                        "quality_status": "llm_accepted",
                        "final_confidence_score": s.get("decision_confidence", 0.8),
                        "final_confidence_level": "high" if s.get("decision_confidence", 0) >= 0.85 else "medium",
                        "final_status": "llm_accepted",
                        "event_subject_name": "", "event_subject_type": "",
                        "event_time_raw": "", "event_action": "", "event_object": "",
                        "event_status": "unknown", "impact_direction": "uncertain",
                        "evidence_sentences": s.get("evidence_sentences", []),
                        "extraction_confidence": 0.0,
                    })
                for rj in r.get("rejected_events", []):
                    all_rejected.append({
                        "dataset": ds, "article_id": r["article_id"],
                        "target_name": r["target_name"],
                        "schema_id": rj["schema_id"], "event_name": rj.get("event_name", ""),
                        "reason": rj.get("reason", ""),
                    })
                if len(r.get("selected_events", [])) == 0:
                    all_rejected.append({
                        "dataset": ds, "article_id": r["article_id"],
                        "target_name": r["target_name"],
                        "schema_id": "ALL", "event_name": "no_matching_event",
                        "reason": "no_candidate_confirmed",
                    })

        # 3. Manual review events
        man_file = os.path.join(QA_DIR, f"manual_review_events_{ds}.jsonl")
        if os.path.exists(man_file):
            for m in load_jsonl(man_file):
                if m.get("quality_status") == "reject":
                    continue
                all_manual.append({
                    "event_id": f"review_{ds}_{m['article_id']}_{m.get('schema_id','NONE')}",
                    "article_id": m["article_id"], "title": m.get("title", "")[:200],
                    "target_type": m.get("target_type", ""), "target_code": m.get("target_code", ""),
                    "target_name": m.get("target_name", ""),
                    "quality_status": "manual_review",
                    "reason": m.get("suspicious_reasons", "low_confidence"),
                })

        stats = {
            "auto_accept_count": len(auto),
            "joint_adj_groups_processed": len(load_jsonl(adj_file) if os.path.exists(adj_file) else []),
            "joint_adj_accepted": sum(len(r.get("selected_events", [])) for r in (load_jsonl(adj_file) if os.path.exists(adj_file) else [])),
            "joint_adj_rejected": sum(len(r.get("rejected_events", [])) for r in (load_jsonl(adj_file) if os.path.exists(adj_file) else [])),
            "manual_review_count": len(all_manual),
            "final_accepted_events": len(all_final[ds]),
        }
        agg_stats[ds] = stats
        print(f"  auto_accept={stats['auto_accept_count']}, accepted={stats['final_accepted_events']}")

    # Save final outputs
    for ds in ["train", "test"]:
        events = all_final[ds]
        with open(os.path.join(OUT_DIR, f"final_events_{ds}.jsonl"), "w", encoding="utf-8") as f:
            for e in events: f.write(json.dumps(e, ensure_ascii=False) + "\n")

        # Excel (no full content)
        xlsx_rows = [{k: v for k, v in e.items() if k != "content"}
                     for e in events]
        df = pd.DataFrame(xlsx_rows)
        df.to_excel(os.path.join(OUT_DIR, f"final_events_{ds}.xlsx"), index=False, engine="openpyxl")
        print(f"Saved {ds}: {len(events)} final events, {len(xlsx_rows)} xlsx rows")

    # Manual review queue
    if all_manual:
        with open(os.path.join(OUT_DIR, "manual_review_queue.jsonl"), "w", encoding="utf-8") as f:
            for m in all_manual: f.write(json.dumps(m, ensure_ascii=False) + "\n")
        pd.DataFrame(all_manual).to_excel(os.path.join(OUT_DIR, "manual_review_queue.xlsx"), index=False, engine="openpyxl")

    # Rejected events
    if all_rejected:
        with open(os.path.join(OUT_DIR, "rejected_event_predictions.jsonl"), "w", encoding="utf-8") as f:
            for r in all_rejected: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Frequency
    event_counter = Counter()
    for ds in ["train", "test"]:
        for e in all_final[ds]:
            event_counter[(ds, e["target_type"], e["schema_id"], e["target_name"], e["event_name"])] += 1
    freq_rows = [{"dataset": ds, "target_type": tt, "schema_id": sid, "target_name": tn,
                   "event_name": en, "count": c}
                  for (ds, tt, sid, tn, en), c in event_counter.most_common()]
    pd.DataFrame(freq_rows).to_csv(os.path.join(REPORTS_DIR, "task1_final_event_frequency.csv"), index=False)

    # Final summary
    total_auto = sum(agg_stats[ds]["auto_accept_count"] for ds in ["train", "test"])
    total_accepted = sum(len(all_final[ds]) for ds in ["train", "test"])
    total_manual = len(all_manual)
    total_rejected = len(all_rejected)

    final_summary = {
        "input_macbert_positive_predictions": 12841,
        "auto_accept_count": total_auto,
        "joint_adjudication_groups": sum(agg_stats[ds].get("joint_adj_groups_processed", 0) for ds in ["train", "test"]),
        "joint_adjudication_accepted": sum(agg_stats[ds].get("joint_adj_accepted", 0) for ds in ["train", "test"]),
        "joint_adjudication_rejected": sum(agg_stats[ds].get("joint_adj_rejected", 0) for ds in ["train", "test"]),
        "manual_review_count": total_manual,
        "rejected_count": total_rejected,
        "final_accepted_events": total_accepted,
        "company_events": sum(1 for ds in ["train", "test"] for e in all_final[ds] if e["target_type"] == "company"),
        "industry_events": sum(1 for ds in ["train", "test"] for e in all_final[ds] if e["target_type"] == "industry"),
        "high_confidence_count": sum(1 for ds in ["train", "test"] for e in all_final[ds] if e.get("final_confidence_level") == "high"),
        "medium_confidence_count": sum(1 for ds in ["train", "test"] for e in all_final[ds] if e.get("final_confidence_level") == "medium"),
    }
    with open(os.path.join(REPORTS_DIR, "task1_final_summary.json"), "w") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Task 1 Final Summary:")
    for k, v in final_summary.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
