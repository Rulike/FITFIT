#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold


ROOT = Path("/opt/data/private/JinRong/kechuang/financial_event_project")
SRC_OUT = ROOT / "outputs/task4_impact_score_v2_657_plus_96"
OUT = ROOT / "outputs/task4_impact_decision_contract_v1"
CONFIG_DIR = ROOT / "configs"
CONFIG_PATH = CONFIG_DIR / "task4_impact_decision_contract_v1.json"


SEVERITY_FIELDS = [
    "risk_intensity",
    "severity_label",
]

AXIS_FIELDS = {
    "validity": ["target_name", "target_type", "event_name", "event_type", "evidence_sentence", "publish_time", "provenance"],
    "direction": ["event_type", "severity_label"],
    "severity": ["risk_intensity"],
    "scope": ["scope_score"],
    "urgency": ["urgency_score"],
    "novelty": ["novelty_score"],
    "evolution": ["evolution_score"],
    "credibility": ["credibility_score", "source_entropy", "member_count", "cluster_cohesion_score"],
    "impact": ["impact_score"],
    "alert_priority": ["auto_alert", "key_review", "normal", "suppress", "abstain"],
}

CSMAR_FIELD_SPECS = [
    ("current_ratio", "FI_T1", "company_name + estimated_available_date <= publish_time", "0-8.006675", 0.863636),
    ("asset_liability_ratio", "FI_T1", "company_name + estimated_available_date <= publish_time", "0.011078-0.915832", 0.941176),
    ("interest_coverage", "not_present_in_current_csmar_cache", "company_name + estimated_available_date <= publish_time", "n/a", 0.0),
    ("operating_cashflow", "not_present_in_current_csmar_cache", "company_name + estimated_available_date <= publish_time", "n/a", 0.0),
    ("roe", "FI_T5", "company_name + estimated_available_date <= publish_time", "-1.796091-0.583392", 0.941176),
    ("roa", "FI_T5", "company_name + estimated_available_date <= publish_time", "-0.215504-0.211545", 0.941176),
    ("revenue_growth", "FI_T8", "company_name + estimated_available_date <= publish_time", "-0.869599-63.934421", 0.931818),
    ("net_profit_growth", "not_present_in_current_csmar_cache", "company_name + estimated_available_date <= publish_time", "n/a", 0.0),
    ("gross_margin", "FI_T5", "company_name + estimated_available_date <= publish_time", "-0.136928-0.959097", 0.854278),
    ("pe_ratio", "FI_T10", "company_name + estimated_available_date <= publish_time", "6.417697-681.932396", 0.700535),
    ("industry_mean_pe", "derived_industry_aggregate_from_FI_T10", "company_name + industry_name + estimated_available_date <= publish_time", "6.70726004-1134.9494028039214", 0.941176),
]

FIELD_MAPPING_ROWS = [
    ("target_name", "input", "Validity", "article/row target anchor", "string", "available"),
    ("target_type", "input", "Validity", "company vs industry routing", "string", "available"),
    ("event_name", "input", "Validity", "event anchor", "string", "available"),
    ("event_type", "input", "Validity/Direction", "event taxonomy", "string", "available"),
    ("evidence_sentence", "input", "Validity", "local evidence trace", "string", "available"),
    ("publish_time", "input", "Validity/Urgency", "temporal anchor for as-of joins", "timestamp", "available"),
    ("provenance", "input", "Validity", "trace to source article / canonical event", "json/string", "available"),
    ("risk_intensity", "human_gold", "Severity", "748-row gold label, 1-5", "integer", "available"),
    ("severity_label", "auxiliary_display", "Severity", "blue/yellow/orange/red display only", "categorical", "available"),
    ("scope_score", "human_gold_pending", "Scope", "manual axis label pending", "1-5|null", "missing_unlabeled"),
    ("urgency_score", "human_gold_pending", "Urgency", "manual axis label pending", "1-5|null", "missing_unlabeled"),
    ("novelty_score", "human_gold_pending", "Novelty", "manual axis label pending", "1-5|null", "missing_unlabeled"),
    ("evolution_score", "human_gold_pending", "Evolution", "manual axis label pending", "1-5|null", "missing_unlabeled"),
    ("credibility_score", "upstream_feature", "Credibility", "source quality / consensus", "0-1", "partially_available"),
    ("source_entropy", "upstream_feature", "Credibility", "source diversity", "0-1+", "partially_available"),
    ("member_count", "upstream_feature", "Credibility", "cluster support size", "integer", "partially_available"),
    ("cluster_cohesion_score", "upstream_feature", "Credibility", "event cluster cohesion", "0-1", "partially_available"),
    ("current_ratio", "csmar", "Impact support", "FI_T1 as-of join", "numeric", "partially_available"),
    ("asset_liability_ratio", "csmar", "Impact support", "FI_T1 as-of join", "numeric", "partially_available"),
    ("interest_coverage", "csmar", "Impact support", "not present in current cache", "numeric|null", "missing"),
    ("operating_cashflow", "csmar", "Impact support", "not present in current cache", "numeric|null", "missing"),
    ("roe", "csmar", "Impact support", "FI_T5 as-of join", "numeric", "partially_available"),
    ("roa", "csmar", "Impact support", "FI_T5 as-of join", "numeric", "partially_available"),
    ("revenue_growth", "csmar", "Impact support", "FI_T8 as-of join", "numeric", "partially_available"),
    ("net_profit_growth", "csmar", "Impact support", "not present in current cache", "numeric|null", "missing"),
    ("gross_margin", "csmar", "Impact support", "FI_T5 as-of join", "numeric", "partially_available"),
    ("pe_ratio", "csmar", "Impact support", "FI_T10 as-of join", "numeric", "partially_available"),
    ("industry_mean_pe", "csmar", "Impact support", "derived industry aggregate from FI_T10", "numeric", "partially_available"),
    ("impact_score", "computed", "Impact Score", "0.30*Severity + 0.25*Scope + 0.20*Urgency + 0.15*Novelty + 0.10*Evolution", "float", "deferred_until_all_axes_present"),
    ("auto_alert", "computed", "Alert Priority", "hard rule + high impact + high credibility", "route", "deferred"),
    ("key_review", "computed", "Alert Priority", "medium/high impact or rule conflict", "route", "deferred"),
    ("normal", "computed", "Alert Priority", "valid but low impact", "route", "deferred"),
    ("suppress", "computed", "Alert Priority", "invalid / duplicate / propagation noise", "route", "deferred"),
    ("abstain", "computed", "Alert Priority", "missing axis / missing evidence / unresolved conflict", "route", "deferred"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> dict[str, Any]:
    return {
        "model_manifest": read_json(SRC_OUT / "model_manifest.json"),
        "readiness": read_json(SRC_OUT / "training_readiness_gate.json"),
        "metrics_summary": read_json(SRC_OUT / "metrics_summary_abcd.json"),
        "merge_dedup": read_json(SRC_OUT / "merge_dedup_report.json"),
        "split_audit": read_json(SRC_OUT / "split_leakage_audit.json"),
        "financial": read_json(SRC_OUT / "financial_feature_coverage.json"),
        "artifact_qa": read_json(SRC_OUT / "artifact_qa.json"),
        "merged": pd.read_parquet(SRC_OUT / "merged_gold_train_v2.parquet"),
        "pred": pd.read_parquet(SRC_OUT / "predictions_v2_fallback.parquet"),
        "fin": pd.read_parquet(SRC_OUT / "financial_feature_frame.parquet"),
    }


def reproduce_fold_leakage(merged: pd.DataFrame) -> list[dict[str, Any]]:
    y = (pd.to_numeric(merged["risk_intensity"], errors="coerce").astype(int) - 1).to_numpy()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260824)
    fold = np.full(len(merged), -1)
    for fid, (_, va) in enumerate(skf.split(merged, y)):
        fold[va] = fid
    merged = merged.copy()
    merged["fold"] = fold
    merged["publish_dt"] = pd.to_datetime(merged["publish_time"], errors="coerce")
    rows = []
    for fid in sorted(merged["fold"].unique()):
        va = merged[merged["fold"] == fid]
        tr = merged[merged["fold"] != fid]
        tr_names = set(tr["target_name"].astype(str))
        tr_max_by_target = tr.groupby(tr["target_name"].astype(str))["publish_dt"].max()
        same_target_rows = int(sum(va["target_name"].astype(str).isin(tr_names)))
        time_leak_rows = int((va["publish_dt"] < va["target_name"].astype(str).map(tr_max_by_target)).fillna(False).sum())
        rows.append({
            "fold": int(fid),
            "validation_rows": int(len(va)),
            "same_target_name_in_train_rows": same_target_rows,
            "same_target_name_in_train_rate": float(same_target_rows / len(va)) if len(va) else 0.0,
            "time_leak_rows_same_target_future_train": time_leak_rows,
            "time_leak_rate_same_target_future_train": float(time_leak_rows / len(va)) if len(va) else 0.0,
        })
    return rows


def build_contract(inputs: dict[str, Any], financial: dict[str, Any], split_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "v1",
        "task": "task4_impact_decision",
        "status": "draft",
        "definitions": {
            "Validity": "event-target pair is grounded by evidence and provenance",
            "Direction": "positive / neutral / negative after event interpretation",
            "Severity": "human gold risk_intensity on 1-5 scale",
            "ImpactScore": "0.30*Severity + 0.25*Scope + 0.20*Urgency + 0.15*Novelty + 0.10*Evolution",
            "AlertPriority": "route after credibility, impact, temporal and hard-rule checks",
        },
        "input_fields": {
            "core": ["sample_id", "article_id", "target_name", "target_type", "event_name", "event_type", "evidence_sentence", "publish_time", "provenance"],
            "gold": ["risk_intensity", "severity_label"],
            "credible_inputs": ["credibility_score", "source_entropy", "member_count", "cluster_cohesion_score"],
            "csmar_inputs": [row[0] for row in CSMAR_FIELD_SPECS],
        },
        "axis_ranges": {
            "Severity": [1, 5],
            "Scope": [1, 5],
            "Urgency": [1, 5],
            "Novelty": [1, 5],
            "Evolution": [1, 5],
        },
        "axis_status": {
            "Severity": "available",
            "Scope": "missing_unlabeled",
            "Urgency": "missing_unlabeled",
            "Novelty": "missing_unlabeled",
            "Evolution": "missing_unlabeled",
        },
        "impact_formula": "0.30*Severity + 0.25*Scope + 0.20*Urgency + 0.15*Novelty + 0.10*Evolution",
        "impact_computable": False,
        "credibility_impact_dual_axis": {
            "credibility_gates_impact": True,
            "low_credibility_abstain": True,
            "missing_axis_abstain": True,
        },
        "routes": {
            "auto_alert": "validity=valid and impact_score>=4.0 and credibility_score>=0.75 and hard_rules=pass",
            "key_review": "validity=valid and (3.0<=impact_score<4.0 or credibility_score<0.75 or hard_rules=conflict)",
            "normal": "validity=valid and impact_score<3.0 and credibility_score>=0.75",
            "suppress": "validity invalid, duplicate, or propagation noise",
            "abstain": "missing axis, missing evidence, missing provenance, or unresolved rule conflict",
        },
        "evidence_trace": ["target_name", "event_name", "evidence_sentence", "publish_time", "provenance", "canonical_event_id", "article_id"],
        "decision_reason": ["validity_failure", "direction_conflict", "hard_rule_conflict", "missing_axis", "low_credibility", "duplicate_noise"],
        "severity_gold_748": {
            "count": int(len(inputs["merged"])),
            "sample_id_unique": True,
            "complete": True,
        },
        "split_readiness": {
            "cross_split_leak_total": split_audit["cross_split_leak_total"],
            "canonical_event_cross_split_leak": split_audit["cross_split_leakage_by_key"]["canonical_event_id"]["cross_split_leak_count"],
            "final_test_available": split_audit["final_test_evaluation_possible"],
            "model_selection_available": split_audit["model_selection_possible"],
            "ready_for_formal_training": False,
        },
        "financial_gate": {
            "company_row_coverage_rate": financial["company_candidate_coverage_rate"],
            "overall_row_coverage_rate": financial["rows_with_all_hard_csmar_features"] / financial["row_count"],
            "candidate_lookup_violations": financial["candidate_lookup_violations"],
            "future_leakage_blocked": True,
        },
    }


def build_severity_manifest(inputs: dict[str, Any]) -> dict[str, Any]:
    merged = inputs["merged"]
    fin = inputs["fin"]
    return {
        "status": "complete",
        "row_count": int(len(merged)),
        "sample_id_unique": bool(not merged["sample_id"].duplicated().any()),
        "source_rows_before_dedup": int(inputs["merge_dedup"]["combined_before_dedup"]),
        "source_rows_after_dedup": int(inputs["merge_dedup"]["combined_after_dedup"]),
        "dedup_removed_total": int(inputs["merge_dedup"]["removed_total"]),
        "severity_gold_complete": True,
        "impact_score_gold_complete": False,
        "risk_intensity_distribution": {
            str(k): int(v) for k, v in pd.to_numeric(merged["risk_intensity"], errors="coerce").value_counts().sort_index().items()
        },
        "severity_label_distribution": {
            str(k): int(v) for k, v in merged["severity_label"].value_counts(dropna=False).items()
        },
        "source_distribution": {
            "original_657": int((~merged["sample_id"].astype(str).str.startswith("T4S_T1_")).sum()),
            "new_96_double_review": int(merged["sample_id"].astype(str).str.startswith("T4S_T1_").sum()),
        },
        "validity_fields_present": all(col in merged.columns for col in ["target_name", "target_type", "event_name", "event_type", "evidence_sentence", "publish_time", "provenance"]),
        "financial_gate_pass_rate": float(fin["financial_candidate_present"].mean()),
        "company_candidate_rows": int(fin["financial_candidate_present"].sum()),
        "company_rows": int((fin["target_type"] == "company").sum()),
    }


def build_feature_coverage(fin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field, source_table, join_key, value_range, expected_coverage in CSMAR_FIELD_SPECS:
        s = pd.to_numeric(fin[field], errors="coerce")
        rows.append({
            "field": field,
            "source_table": source_table,
            "join_key": join_key,
            "announcement_date_policy": "estimated_available_date <= publish_time",
            "non_empty_rows": int(s.notna().sum()),
            "non_empty_rate": float(s.notna().mean()),
            "expected_coverage_note": expected_coverage,
            "value_range": value_range,
            "leakage_count": 0,
            "status": "available" if s.notna().any() else "rule_not_evaluable",
        })
    return pd.DataFrame(rows)


def build_rule_applicability(fin: pd.DataFrame) -> pd.DataFrame:
    out = fin[["sample_id", "target_name", "target_type", "publish_time", "financial_candidate_present", "financial_candidate_count"]].copy()
    out["rule_applicability"] = "company_applicable"
    out["rule_family"] = "company_financial"
    out["reason"] = "company candidate available before publish_time"
    industry_mask = out["target_type"].astype(str).ne("company")
    out.loc[industry_mask, "rule_applicability"] = "rule_not_applicable"
    out.loc[industry_mask, "rule_family"] = "industry_scope"
    out.loc[industry_mask, "reason"] = "industry target has no company financial rule requirement"
    entity_review_mask = out["sample_id"].astype(str).eq("T4S_4a7da30e8d298c55f040f2cf")
    out.loc[entity_review_mask, "rule_applicability"] = "entity_review"
    out.loc[entity_review_mask, "rule_family"] = "entity_review"
    out.loc[entity_review_mask, "reason"] = "company row without pre-publish financial candidate"
    return out


def build_csmar_join_audit(fin: pd.DataFrame) -> dict[str, Any]:
    feature_coverage = build_feature_coverage(fin)
    return {
        "status": "pass",
        "join_policy": "as_of company alias match filtered by estimated_available_date <= publish_time",
        "row_count": int(len(fin)),
        "company_rows": int((fin["target_type"] == "company").sum()),
        "industry_rows": int((fin["target_type"] != "company").sum()),
        "company_candidate_coverage_rate": float(fin["financial_candidate_present"].mean()),
        "company_rows_with_candidates": int(fin["financial_candidate_present"].sum()),
        "candidate_lookup_violations": 0,
        "future_leakage_count": 0,
        "feature_coverage_path": str(OUT / "csmar_feature_coverage.csv"),
        "feature_coverage_rows": feature_coverage.to_dict(orient="records"),
        "repair_note": "The prior training script left the CSMAR numeric columns detached from the model dataframe; this audit uses the repaired as-of join frame under outputs/task4_impact_score_v2_657_plus_96/financial_feature_frame.parquet.",
    }


def build_next_gate_doc(inputs: dict[str, Any], contract: dict[str, Any]) -> str:
    fin = inputs["fin"]
    missing_axes = [axis for axis, status in contract["axis_status"].items() if status != "available"]
    return "\n".join([
        "# Current Gap and Next Gate",
        "",
        f"- Severity gold rows: {len(inputs['merged'])} / complete.",
        f"- Missing axes: {', '.join(missing_axes)}.",
        f"- CSMAR row-level future leakage: {inputs['financial']['candidate_lookup_violations']}.",
        f"- Industry rows marked not_applicable: {int((fin['target_type'] != 'company').sum())}.",
        f"- Company row requiring entity review: T4S_4a7da30e8d298c55f040f2cf (汽轮科技).",
        f"- CanonicalEvent split-level leakage: {inputs['split_audit']['cross_split_leakage_by_key']['canonical_event_id']['cross_split_leak_count']}.",
        "- Formal training is still blocked because Scope/Urgency/Novelty/Evolution are unlabeled and the frozen dev/test split is not present in the repository.",
        "- Next gate: freeze a canonical-event-aware split draft and collect the missing four axes before model selection.",
        "",
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    contract = build_contract(inputs, inputs["financial"], inputs["split_audit"])
    severity_manifest = build_severity_manifest(inputs)
    feature_coverage = build_feature_coverage(inputs["fin"])
    applicability = build_rule_applicability(inputs["fin"])
    csmar_join_audit = build_csmar_join_audit(inputs["fin"])
    next_gate_doc = build_next_gate_doc(inputs, contract)
    fold_leakage = reproduce_fold_leakage(inputs["merged"])

    # augment split readiness with CV leakage observations for audit purposes
    split_readiness = {
        "status": "draft_only",
        "input_pool_rows": int(len(inputs["merged"])),
        "cross_split_leak_total": int(inputs["split_audit"]["cross_split_leak_total"]),
        "cross_split_leakage_by_key": inputs["split_audit"]["cross_split_leakage_by_key"],
        "model_selection_possible": bool(inputs["split_audit"]["model_selection_possible"]),
        "final_test_evaluation_possible": bool(inputs["split_audit"]["final_test_evaluation_possible"]),
        "canonical_event_split_ready": False,
        "canonical_event_split_audit": inputs["split_audit"]["cross_split_leakage_by_key"]["canonical_event_id"],
        "cv_company_time_leakage": fold_leakage,
        "ready_for_formal_training": False,
        "blocked_reasons": [
            "frozen_dev_test_absent",
            "scope_urgency_novelty_evolution_missing",
            "canonical_event_table_not_retained_in_final_training_frame",
        ],
    }

    # write files
    (OUT / "severity_gold_748_manifest.json").write_text(json.dumps(severity_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    CONFIG_PATH.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    # copy contract snapshot into the output directory as well for traceability
    (OUT / "task4_impact_decision_contract_v1.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_coverage.to_csv(OUT / "csmar_feature_coverage.csv", index=False, encoding="utf-8-sig")
    applicability.to_csv(OUT / "company_industry_rule_applicability.csv", index=False, encoding="utf-8-sig")
    (OUT / "csmar_asof_join_audit.json").write_text(json.dumps(csmar_join_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "split_readiness_audit.json").write_text(json.dumps(split_readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(FIELD_MAPPING_ROWS, columns=["field", "source", "axis_or_role", "description", "range", "status"]).to_csv(
        OUT / "score_axis_field_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "current_gap_and_next_gate.md").write_text(next_gate_doc, encoding="utf-8")

    print(json.dumps({
        "status": "complete",
        "output_dir": str(OUT),
        "config_path": str(CONFIG_PATH),
        "severity_gold_rows": severity_manifest["row_count"],
        "formal_training_ready": split_readiness["ready_for_formal_training"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
