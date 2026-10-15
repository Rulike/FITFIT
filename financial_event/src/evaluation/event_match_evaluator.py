#!/usr/bin/env python3
"""event_match_evaluator.py — 统一事件匹配评估器，所有实验共用"""

import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class EventMatchEvaluator:
    """统一评估器。只接受验证集和预测概率，内部重新计算所有指标。"""

    def __init__(self, validation_file, prediction_file, threshold, join_key="sample_id"):
        # Load validation ground truth
        self.validation = {}
        with open(validation_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self.validation[rec[join_key]] = rec

        # Load predictions
        self.predictions = {}
        with open(prediction_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self.predictions[rec[join_key]] = rec

        self.threshold = threshold
        self.join_key = join_key

        # Validate join
        val_ids = set(self.validation.keys())
        pred_ids = set(self.predictions.keys())
        self.missing_in_pred = val_ids - pred_ids
        self.extra_in_pred = pred_ids - val_ids
        self.common_ids = val_ids & pred_ids

        if len(self.common_ids) != len(val_ids):
            print(f"WARNING: Validation has {len(val_ids)} records, but only {len(self.common_ids)} matched in predictions")
            print(f"  Missing: {len(self.missing_in_pred)}, Extra: {len(self.extra_in_pred)}")

        # Build aligned arrays
        self.y_true = []
        self.y_prob = []
        self.sample_ids = []
        for sid in sorted(self.common_ids):
            self.sample_ids.append(sid)
            self.y_true.append(self.validation[sid]["label"])
            self.y_prob.append(self.predictions[sid]["probability_positive"])

        self.y_true = np.array(self.y_true)
        self.y_prob = np.array(self.y_prob)

        # Recompute predictions at the given threshold
        self.y_pred = (self.y_prob >= self.threshold).astype(int)

    def compute_overall_metrics(self):
        acc = float(accuracy_score(self.y_true, self.y_pred))
        p, r, f1, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='binary', pos_label=1, zero_division=0)
        mp, mr, mf1, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='macro', zero_division=0)
        _, _, wf1, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='weighted', zero_division=0)
        fp = int(((self.y_pred == 1) & (self.y_true == 0)).sum())
        fn = int(((self.y_pred == 0) & (self.y_true == 1)).sum())
        return {
            "evaluation_file": "event_match_validation.jsonl",
            "sample_count": len(self.y_true),
            "positive_count": int((self.y_true == 1).sum()),
            "negative_count": int((self.y_true == 0).sum()),
            "threshold": self.threshold,
            "accuracy": round(acc, 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "positive_f1": round(float(f1), 4),
            "macro_f1": round(float(mf1), 4),
            "weighted_f1": round(float(wf1), 4),
            "false_positive_count": fp,
            "false_negative_count": fn,
        }

    def compute_negative_type_metrics(self):
        """三类负样本的FP和FPR。negative_type 来自验证集原始字段。"""
        rows = []
        for nt in ["same_target_other_event", "same_event_wrong_target", "similar_schema_wrong_target"]:
            items = []
            for i, sid in enumerate(self.sample_ids):
                val_rec = self.validation[sid]
                if val_rec["label"] == 0 and val_rec.get("negative_type") == nt:
                    items.append(i)
            if not items:
                continue
            yt = self.y_true[items]
            yp = self.y_pred[items]
            yprob = self.y_prob[items]
            fp = int(((yp == 1) & (yt == 0)).sum())
            tn = int(((yp == 0) & (yt == 0)).sum())
            total_neg = int((yt == 0).sum())
            fpr = round(fp / total_neg, 4) if total_neg > 0 else 0
            rows.append({
                "negative_type": nt,
                "sample_count": len(items),
                "true_negative_count": tn,
                "false_positive_count": fp,
                "false_positive_rate": fpr,
                "mean_positive_probability": round(float(yprob.mean()), 4),
                "threshold": self.threshold,
            })
        return rows

    def compute_confusion_matrix(self):
        tn = int(((self.y_pred == 0) & (self.y_true == 0)).sum())
        fp = int(((self.y_pred == 1) & (self.y_true == 0)).sum())
        fn = int(((self.y_pred == 0) & (self.y_true == 1)).sum())
        tp = int(((self.y_pred == 1) & (self.y_true == 1)).sum())
        return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

    def verify_key_invariants(self):
        """验证关键不变量。"""
        return {
            "sample_count": len(self.y_true),
            "positive_count": int((self.y_true == 1).sum()),
            "negative_count": int((self.y_true == 0).sum()),
            "expected_sample_count": 1271,
            "expected_positive_count": 565,
            "expected_negative_count": 706,
            "sample_count_ok": len(self.y_true) == 1271,
            "positive_count_ok": int((self.y_true == 1).sum()) == 565,
            "negative_count_ok": int((self.y_true == 0).sum()) == 706,
            "sample_ids_unique": len(self.sample_ids) == len(set(self.sample_ids)),
            "join_missing": len(self.missing_in_pred),
            "join_extra": len(self.extra_in_pred),
            "all_probs_in_range": bool(np.all((self.y_prob >= 0) & (self.y_prob <= 1))),
        }


def compare_probability_fields(step6_pred_file, step8_pred_file):
    """比较第六步和第八步预测文件中的基线概率是否一致。"""
    s6 = {}
    with open(step6_pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                s6[rec["sample_id"]] = rec

    s8 = {}
    with open(step8_pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                s8[rec["sample_id"]] = rec

    common = set(s6.keys()) & set(s8.keys())
    diffs = []
    for sid in common:
        p6 = s6[sid].get("probability_positive", 0)
        p8 = s8[sid].get("probability_positive", 0)
        if abs(p6 - p8) > 1e-6:
            diffs.append({
                "sample_id": sid,
                "step6_probability": p6,
                "step8_probability": p8,
                "absolute_difference": abs(p6 - p8),
            })
    diffs.sort(key=lambda x: -x["absolute_difference"])

    only_s6 = set(s6.keys()) - set(s8.keys())
    only_s8 = set(s8.keys()) - set(s6.keys())

    return {
        "common_count": len(common),
        "probability_mismatch_count": len(diffs),
        "max_difference": diffs[0]["absolute_difference"] if diffs else 0,
        "mean_difference": sum(d["absolute_difference"] for d in diffs) / len(diffs) if diffs else 0,
        "only_in_step6": len(only_s6),
        "only_in_step8": len(only_s8),
        "mismatched_samples": diffs,
    }
