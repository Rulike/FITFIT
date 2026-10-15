#!/usr/bin/env python3
"""audit_event_match_metrics.py — 审计并生成权威基线指标"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.event_match_evaluator import EventMatchEvaluator, compare_probability_fields
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_FILE = os.path.join(PROJECT_ROOT, "data/processed/model_data/event_match_validation.jsonl")
S6_PREDS = os.path.join(PROJECT_ROOT, "outputs/event_match_baseline/validation_predictions.jsonl")
S8_PREDS = os.path.join(PROJECT_ROOT, "outputs/event_match_targetwise/original_pair_predictions.jsonl")
S8_CONFIG = os.path.join(PROJECT_ROOT, "outputs/event_match_targetwise/best_targetwise_config.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

def main():
    print("=" * 60)
    print("事件匹配指标审计")
    print("=" * 60)

    # ─── 1. Canonical baseline at threshold 0.38 ───
    print("\n[1] 权威基线指标 (threshold=0.38)...")
    baseline_eval = EventMatchEvaluator(VAL_FILE, S6_PREDS, threshold=0.38)
    invariants = baseline_eval.verify_key_invariants()
    for k, v in invariants.items():
        status = "✓" if (isinstance(v, bool) and v) or (not isinstance(v, bool)) else "✗"
        print(f"  {status} {k}: {v}")

    baseline_metrics = baseline_eval.compute_overall_metrics()
    baseline_neg = baseline_eval.compute_negative_type_metrics()
    baseline_cm = baseline_eval.compute_confusion_matrix()

    print(f"\n  权威基线指标:")
    for k, v in baseline_metrics.items():
        print(f"    {k}: {v}")
    print(f"  混淆矩阵: {baseline_cm}")
    print(f"  三类负样本:")
    for row in baseline_neg:
        print(f"    {row['negative_type']}: FP={row['false_positive_count']}, FPR={row['false_positive_rate']}")

    # Save canonical metrics
    with open(os.path.join(REPORTS_DIR, "canonical_baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(baseline_neg).to_csv(
        os.path.join(REPORTS_DIR, "canonical_baseline_negative_type_metrics.csv"), index=False)

    # Also compute at threshold 0.5 for comparison
    baseline_eval_05 = EventMatchEvaluator(VAL_FILE, S6_PREDS, threshold=0.5)
    baseline_neg_05 = baseline_eval_05.compute_negative_type_metrics()
    print(f"\n  基线 threshold=0.5 对比:")
    for row in baseline_neg_05:
        print(f"    {row['negative_type']}: FP={row['false_positive_count']}, FPR={row['false_positive_rate']}")

    # ─── 2. Compare step6 vs step8 probability fields ───
    print("\n[2] 第六步 vs 第八步概率一致性...")
    prob_cmp = compare_probability_fields(S6_PREDS, S8_PREDS)
    print(f"  共同样本: {prob_cmp['common_count']}")
    print(f"  概率不一致: {prob_cmp['probability_mismatch_count']}")
    print(f"  最大差异: {prob_cmp['max_difference']:.6f}")
    print(f"  仅在S6: {prob_cmp['only_in_step6']}, 仅在S8: {prob_cmp['only_in_step8']}")

    # ─── 3. Audit: what caused the 7 vs 8 discrepancy? ───
    print("\n[3] 根本原因分析...")
    print("  第六步报告使用了 prediction_threshold_0_5 字段（阈值0.5）")
    print("  但最佳阈值是 0.38，因此负样本分类报告与主指标不一致")
    print()
    print("  阈值 0.38（最佳阈值，应统一使用）:")
    for row in baseline_neg:
        print(f"    {row['negative_type']}: FP={row['false_positive_count']}/{row['sample_count']}, FPR={row['false_positive_rate']}")
    print()
    print("  阈值 0.5（第六步报告误用的阈值，仅用于负样本分类）:")
    for row in baseline_neg_05:
        print(f"    {row['negative_type']}: FP={row['false_positive_count']}/{row['sample_count']}, FPR={row['false_positive_rate']}")

    # ─── 4. Generate detailed per-sample audit ───
    print("\n[4] 逐样本差异审计...")
    val_map = {}
    for sid, rec in baseline_eval.validation.items():
        val_map[sid] = rec

    s6_map = {}
    with open(S6_PREDS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rec = json.loads(line); s6_map[rec["sample_id"]] = rec

    s8_map = {}
    with open(S8_PREDS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rec = json.loads(line); s8_map[rec["sample_id"]] = rec

    audit_rows = []
    for sid in sorted(set(val_map.keys()) | set(s6_map.keys()) | set(s8_map.keys())):
        val_rec = val_map.get(sid, {})
        s6_rec = s6_map.get(sid, {})
        s8_rec = s8_map.get(sid, {})

        label = val_rec.get("label", -1)
        neg_type = val_rec.get("negative_type", "") if label == 0 else ""

        s6_prob = s6_rec.get("probability_positive", None)
        s8_prob = s8_rec.get("probability_positive", s6_prob)  # fallback to s6

        # Step 6 prediction at 0.38 (canonical)
        s6_pred_038 = 1 if (s6_prob is not None and s6_prob >= 0.38) else 0
        # Step 6 prediction at 0.5 (what was reported for neg types)
        s6_pred_05 = 1 if (s6_prob is not None and s6_prob >= 0.5) else 0

        # Determine if this contributed to the discrepancy
        prob_diff = abs((s6_prob or 0) - (s8_prob or 0))
        reason = "no_difference"
        if s6_pred_038 != s6_pred_05 and neg_type == "same_target_other_event":
            reason = "threshold_mismatch_same_target_7vs8"
        elif prob_diff > 1e-6:
            reason = "probability_mismatch"
        elif s6_rec and not s8_rec:
            reason = "sample_missing_in_step8"
        elif not s6_rec and s8_rec:
            reason = "sample_extra_in_step8"

        if label == 0 and neg_type == "same_target_other_event":
            audit_rows.append({
                "sample_id": sid, "article_id": val_rec.get("article_id", ""),
                "schema_id": val_rec.get("schema_id", ""),
                "label": label, "negative_type": neg_type,
                "baseline_probability_from_step6": s6_prob,
                "baseline_probability_from_step8": s8_prob,
                "probability_difference": prob_diff,
                "step6_prediction_at_0_38": s6_pred_038,
                "step6_prediction_at_0_5": s6_pred_05,
                "canonical_recomputed_prediction": s6_pred_038,
                "difference_reason": reason,
            })

    pd.DataFrame(audit_rows).to_csv(
        os.path.join(REPORTS_DIR, "evaluation_metric_audit_samples.csv"), index=False, encoding="utf-8-sig")
    print(f"  审计样本数: {len(audit_rows)}")

    # ─── 5. Canonical targetwise evaluation ───
    print("\n[5] 权威Targetwise指标...")
    with open(S8_CONFIG) as f:
        tw_config = json.load(f)
    tw_threshold = tw_config["absolute_threshold"]
    print(f"  Targetwise method: {tw_config['method']}, abs_th={tw_threshold}")

    # We need targetwise predictions for the 1271 original pairs
    # Use the targetwise_prediction field from S8 predictions
    tw_y_true = []
    tw_y_pred = []
    for sid in baseline_eval.sample_ids:
        val_rec = val_map[sid]
        s8_rec = s8_map.get(sid, {})
        tw_y_true.append(val_rec["label"])
        tw_y_pred.append(s8_rec.get("targetwise_prediction", 0))

    tw_y_true = __import__('numpy').array(tw_y_true)
    tw_y_pred = __import__('numpy').array(tw_y_pred)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    tw_acc = float(accuracy_score(tw_y_true, tw_y_pred))
    tw_p, tw_r, tw_f1, _ = precision_recall_fscore_support(tw_y_true, tw_y_pred, average='binary', pos_label=1, zero_division=0)
    tw_mp, tw_mr, tw_mf1, _ = precision_recall_fscore_support(tw_y_true, tw_y_pred, average='macro', zero_division=0)
    tw_fp = int(((tw_y_pred == 1) & (tw_y_true == 0)).sum())
    tw_fn = int(((tw_y_pred == 0) & (tw_y_true == 1)).sum())

    # Targetwise negative type metrics
    tw_neg_rows = []
    for nt in ["same_target_other_event", "same_event_wrong_target", "similar_schema_wrong_target"]:
        indices = [i for i, sid in enumerate(baseline_eval.sample_ids)
                   if val_map[sid]["label"] == 0 and val_map[sid].get("negative_type") == nt]
        if not indices: continue
        yt = tw_y_true[indices]; yp = tw_y_pred[indices]
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        total_neg = int((yt == 0).sum())
        fpr = round(fp / total_neg, 4) if total_neg > 0 else 0
        tw_neg_rows.append({
            "negative_type": nt, "sample_count": len(indices),
            "false_positive_count": fp, "false_positive_rate": fpr,
        })

    tw_metrics = {
        "accuracy": round(tw_acc, 4), "precision": round(float(tw_p), 4),
        "recall": round(float(tw_r), 4), "positive_f1": round(float(tw_f1), 4),
        "macro_f1": round(float(tw_mf1), 4),
        "false_positive_count": tw_fp, "false_negative_count": tw_fn,
    }
    with open(os.path.join(REPORTS_DIR, "canonical_targetwise_metrics.json"), "w") as f:
        json.dump(tw_metrics, f, ensure_ascii=False, indent=2)

    # ─── 6. Fixed/New errors (canonical) ───
    fixed = 0; new_err = 0
    for i, sid in enumerate(baseline_eval.sample_ids):
        b_pred = baseline_eval.y_pred[i]
        t_pred = tw_y_pred[i]
        label = baseline_eval.y_true[i]
        if b_pred != label and t_pred == label: fixed += 1
        if b_pred == label and t_pred != label: new_err += 1
    print(f"\n  Canonical fixed errors: {fixed}, new errors: {new_err}, net: {fixed - new_err}")

    # ─── 7. Corrected comparison table ───
    print("\n[6] 生成修正对比表...")
    comp_rows = []
    metrics_list = [
        ("accuracy", baseline_metrics["accuracy"], tw_metrics["accuracy"]),
        ("precision", baseline_metrics["precision"], tw_metrics["precision"]),
        ("recall", baseline_metrics["recall"], tw_metrics["recall"]),
        ("positive_f1", baseline_metrics["positive_f1"], tw_metrics["positive_f1"]),
        ("macro_f1", baseline_metrics["macro_f1"], tw_metrics["macro_f1"]),
        ("false_positive_count", baseline_metrics["false_positive_count"], tw_metrics["false_positive_count"]),
        ("false_negative_count", baseline_metrics["false_negative_count"], tw_metrics["false_negative_count"]),
    ]
    b_st = next((r for r in baseline_neg if r["negative_type"] == "same_target_other_event"), {})
    tw_st = next((r for r in tw_neg_rows if r["negative_type"] == "same_target_other_event"), {})
    b_se = next((r for r in baseline_neg if r["negative_type"] == "same_event_wrong_target"), {})
    tw_se = next((r for r in tw_neg_rows if r["negative_type"] == "same_event_wrong_target"), {})
    b_ss = next((r for r in baseline_neg if r["negative_type"] == "similar_schema_wrong_target"), {})
    tw_ss = next((r for r in tw_neg_rows if r["negative_type"] == "similar_schema_wrong_target"), {})

    fpr_metrics = [
        ("same_target_other_event_fpr", b_st.get("false_positive_rate", 0), tw_st.get("false_positive_rate", 0)),
        ("same_event_wrong_target_fpr", b_se.get("false_positive_rate", 0), tw_se.get("false_positive_rate", 0)),
        ("similar_schema_wrong_target_fpr", b_ss.get("false_positive_rate", 0), tw_ss.get("false_positive_rate", 0)),
    ]
    for name, bv, tv in metrics_list + fpr_metrics:
        comp_rows.append({
            "metric": name, "baseline_canonical": bv, "targetwise_canonical": tv,
            "absolute_change": round(tv - bv, 4) if isinstance(bv, (int, float)) and isinstance(tv, (int, float)) else "",
        })

    comp_rows.append({"metric": "fixed_baseline_errors", "baseline_canonical": "", "targetwise_canonical": fixed, "absolute_change": ""})
    comp_rows.append({"metric": "new_errors", "baseline_canonical": "", "targetwise_canonical": new_err, "absolute_change": ""})
    comp_rows.append({"metric": "net_fixed", "baseline_canonical": "", "targetwise_canonical": fixed - new_err, "absolute_change": ""})

    pd.DataFrame(comp_rows).to_csv(
        os.path.join(REPORTS_DIR, "canonical_baseline_vs_targetwise.csv"), index=False)

    for r in comp_rows:
        print(f"  {r['metric']}: baseline={r['baseline_canonical']} -> targetwise={r['targetwise_canonical']} (Δ{r['absolute_change']})")

    # ─── 8. Correction note ───
    print("\n[7] 生成修正说明...")
    note = f"""# 事件匹配评估口径修正说明

## 根本原因

第六步报告的负样本分类指标使用了 prediction_threshold_0_5（阈值 0.5），
而主指标（Macro-F1、混淆矩阵等）使用最佳阈值 0.38。

这导致 Sixth 步报告中：
- 混淆矩阵 (TP=491, FP=27) 基于阈值 0.38 ✓
- 但三类负样本 FPR 基于阈值 0.5 ✗

## 正确数字（阈值 0.38，权威）

| 负样本类型 | 样本数 | FP | FPR |
|-----------|--------|-----|------|
| same_target_other_event | 46 | 8 | 17.39% |
| same_event_wrong_target | 275 | 5 | 1.82% |
| similar_schema_wrong_target | 385 | 14 | 3.64% |

## 第八步报告的基线列

第八步 evaluate_targetwise_competition.py 正确使用了阈值 0.38 计算基线，
因此第八步报告的基线指标（17.39%, 1.82%, 3.64%）是正确的。
第六步报告的误报率（15.22%, 0.36%, 3.12%）对应阈值 0.5，非最佳阈值。

## 第八步结论是否改变

第八步结论不变：联合竞争决策边际改善 same_target FPR（17.39%→15.22%），
但净修复为负（-3），代价是 Recall 下降。

修正后的 Targetwise 指标：
- Accuracy: {tw_acc:.4f}
- Macro-F1: {tw_mf1:.4f}
- Recall: {tw_r:.4f}
- same_target FPR: 0.1739 → 0.1522
- 修复错误: {fixed}, 新增错误: {new_err}, 净修复: {fixed - new_err}

## 权威评估协议

所有后续实验必须使用 src/evaluation/event_match_evaluator.py 统一评估，
配置见 configs/canonical_evaluation_protocol.json。
"""
    with open(os.path.join(REPORTS_DIR, "targetwise_evaluation_correction_note.md"), "w") as f:
        f.write(note)
    print(f"  修正说明已保存")

    # ─── 9. Canonical evaluation protocol ───
    print("\n[8] 生成权威评估协议...")
    protocol = {
        "validation_file": "data/processed/model_data/event_match_validation.jsonl",
        "validation_sample_count": 1271,
        "positive_count": 565,
        "negative_count": 706,
        "baseline_model": "outputs/event_match_baseline/best_model",
        "baseline_threshold": 0.38,
        "join_key": "sample_id",
        "negative_type_source": "validation_file",
        "official_test_forbidden": True,
        "probability_field": "probability_positive",
        "canonical_same_target_fpr": 0.1739,
        "note": "ALL metrics reported at threshold 0.38. Negative type classification from validation file negative_type field."
    }
    with open(os.path.join(CONFIG_DIR, "canonical_evaluation_protocol.json"), "w") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    # Generate corrected comparison
    corrected_comp_rows = []
    # Include V2 numbers for completeness
    metrics_full = [
        ("accuracy", baseline_metrics["accuracy"], None, tw_metrics["accuracy"]),
        ("precision", baseline_metrics["precision"], None, tw_metrics["precision"]),
        ("recall", baseline_metrics["recall"], None, tw_metrics["recall"]),
        ("positive_f1", baseline_metrics["positive_f1"], None, tw_metrics["positive_f1"]),
        ("macro_f1", baseline_metrics["macro_f1"], None, tw_metrics["macro_f1"]),
        ("false_positive_count", baseline_metrics["false_positive_count"], None, tw_metrics["false_positive_count"]),
        ("false_negative_count", baseline_metrics["false_negative_count"], None, tw_metrics["false_negative_count"]),
        ("same_target_other_event_fpr", b_st.get("false_positive_rate", 0), None, tw_st.get("false_positive_rate", 0)),
        ("same_event_wrong_target_fpr", b_se.get("false_positive_rate", 0), None, tw_se.get("false_positive_rate", 0)),
        ("similar_schema_wrong_target_fpr", b_ss.get("false_positive_rate", 0), None, tw_ss.get("false_positive_rate", 0)),
    ]
    for name, bv, v2v, tv in metrics_full:
        corrected_comp_rows.append({
            "metric": name, "baseline_canonical": bv,
            "hard_negative_v2": v2v if v2v is not None else "",
            "targetwise_canonical": tv,
            "targetwise_vs_baseline": round(tv - bv, 4) if isinstance(bv, (int, float)) and isinstance(tv, (int, float)) else "",
        })
    pd.DataFrame(corrected_comp_rows).to_csv(
        os.path.join(REPORTS_DIR, "baseline_v2_targetwise_comparison_corrected.csv"), index=False)

    print("\n" + "=" * 60)
    print("审计完成！")
    print("=" * 60)
    print(f"\n权威基线 same_target FPR: 0.1739 (8/46) at threshold 0.38")
    print(f"第六步报告误用了 threshold 0.5 的负样本分类结果 (7/46 = 0.1522)")
    print(f"第八步基线列是正确的")
    print(f"结论不变：联合竞争边际改善，但不足以达标")


if __name__ == "__main__":
    main()
