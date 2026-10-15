#!/usr/bin/env python3
"""analyze_same_target_competition.py — 构造同实体完整候选集，基线打分，参数搜索"""

import argparse, json, os, sys, re
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_jsonl(fp):
    r = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line: r.append(json.loads(line))
    return r

def build_input_text(rec):
    return (f"[目标]\n类型：{rec.get('target_type','')}\n名称：{rec.get('target_name','')}\n代码：{rec.get('target_code','')}\n\n"
            f"[候选事件]\n事件名称：{rec.get('event_name','')}\n事件定义：{rec.get('event_description','')}\n\n"
            f"[新闻]\n标题：{rec.get('title','')}\n正文：{rec.get('content','')}")

def smart_truncate(text, tokenizer, max_length=512):
    marker = "\n[新闻]\n"
    if marker in text:
        schema_part, news_part = text.split(marker, 1); schema_part += marker
    else:
        schema_part = text[:int(len(text)*0.6)]; news_part = text[int(len(text)*0.6):]
    schema_tokens = tokenizer.encode(schema_part, add_special_tokens=False)
    available = max_length - 2 - len(schema_tokens)
    if available <= 0:
        desc_marker = "事件定义："
        if desc_marker in schema_part:
            idx = schema_part.index(desc_marker)
            prefix = schema_part[:idx+len(desc_marker)]
            desc = schema_part[idx+len(desc_marker):]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            desc_budget = max_length - 2 - len(prefix_tokens)
            if desc_budget > 0:
                desc_tokens = tokenizer.encode(desc, add_special_tokens=False)
                return prefix + tokenizer.decode(desc_tokens[:desc_budget], skip_special_tokens=True)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(tokens[:max_length-2], skip_special_tokens=True)
    news_tokens = tokenizer.encode(news_part, add_special_tokens=False)
    if len(news_tokens) <= available: return text
    title_marker = "标题："; body_marker = "\n正文："
    title_text = ""; body_text = news_part
    if title_marker in news_part and body_marker in news_part:
        t_idx = news_part.index(title_marker); b_idx = news_part.index(body_marker)
        title_text = news_part[t_idx:b_idx+len(body_marker)]; body_text = news_part[b_idx+len(body_marker):]
    title_tokens = tokenizer.encode(title_text, add_special_tokens=False)
    body_budget = available - len(title_tokens)
    if body_budget <= 0: kept_news = title_text
    else:
        body_tokens = tokenizer.encode(body_text, add_special_tokens=False)
        head_budget = int(body_budget * 0.7); tail_budget = body_budget - head_budget - 10
        if tail_budget <= 0:
            kept_body = tokenizer.decode(body_tokens[:body_budget], skip_special_tokens=True)
        else:
            kept_body = (tokenizer.decode(body_tokens[:head_budget], skip_special_tokens=True) +
                        "\n[正文中间省略]\n" +
                        tokenizer.decode(body_tokens[-tail_budget:], skip_special_tokens=True))
        kept_news = title_text + kept_body
    return schema_part + kept_news


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument("--schema_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    targetwise_dir = os.path.join(PROJECT_ROOT, "data", "processed", "targetwise_validation")
    os.makedirs(targetwise_dir, exist_ok=True)
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Load schemas
    schemas = load_jsonl(args.schema_file)
    schema_map = {s["schema_id"]: s for s in schemas}
    schemas_by_target = defaultdict(list)
    for s in schemas:
        schemas_by_target[(s["target_type"], str(s["target_code"]))].append(s)
    print(f"Schemas: {len(schemas)}, targets: {len(schemas_by_target)}")

    # Load validation data
    val_records = load_jsonl(args.validation_file)
    print(f"Validation records: {len(val_records)}")

    # Load baseline threshold
    bt_path = os.path.join(PROJECT_ROOT, "outputs", "event_match_baseline", "best_threshold.json")
    with open(bt_path) as f:
        baseline_threshold = json.load(f)["best_threshold"]
    print(f"Baseline threshold: {baseline_threshold}")

    # Group by (article_id, target_type, target_code)
    groups = defaultdict(list)
    article_meta = {}
    for r in val_records:
        gk = (r["article_id"], r["target_type"], str(r["target_code"]))
        groups[gk].append(r)
        if r["article_id"] not in article_meta:
            article_meta[r["article_id"]] = {
                "article_id": r["article_id"], "title": r.get("title", ""),
                "content": r.get("content", ""), "canonical_group_id": r.get("canonical_group_id", r["article_id"]),
            }

    print(f"Validation groups: {len(groups)}")
    multi_schema_groups = sum(1 for gk, items in groups.items() if len(items) > 1)
    single_schema_groups = sum(1 for gk, items in groups.items() if len(items) == 1)
    # Multi-positive groups
    multi_pos_groups = []
    for gk, items in groups.items():
        pos_sids = set(r["schema_id"] for r in items if r["label"] == 1)
        if len(pos_sids) > 1:
            multi_pos_groups.append((gk, pos_sids))
    print(f"Multi-schema groups: {multi_schema_groups}")
    print(f"Single-schema groups: {single_schema_groups}")
    print(f"Multi-positive groups: {len(multi_pos_groups)}")

    # Generate full candidate set: for each group, add ALL schemas for that target
    print("\nGenerating full candidate set...")
    all_candidates = []
    group_pos_sids = {}  # group_key -> set of positive schema_ids

    for gk, existing_items in groups.items():
        aid, tt, tc = gk
        pos_sids = set(r["schema_id"] for r in existing_items if r["label"] == 1)
        group_pos_sids[gk] = pos_sids
        existing_sids = set(r["schema_id"] for r in existing_items)

        target_schemas = schemas_by_target.get((tt, tc), [])
        meta = article_meta.get(aid, {})
        if not meta:
            continue

        for s in target_schemas:
            sid = s["schema_id"]
            label = 1 if sid in pos_sids else 0
            candidate_source = "existing_validation_pair" if sid in existing_sids else "schema_completed_candidate"

            all_candidates.append({
                "article_id": aid, "target_type": tt, "target_code": tc,
                "target_name": s["target_name"], "canonical_group_id": meta.get("canonical_group_id", aid),
                "schema_id": sid, "event_name": s["event_name"],
                "event_description": s.get("raw_description", ""),
                "title": meta["title"], "content": meta["content"],
                "label": label, "candidate_source": candidate_source,
            })

    print(f"Full candidates: {len(all_candidates)}")
    # Save candidate set
    cand_path = os.path.join(targetwise_dir, "all_target_candidates_validation.jsonl")
    with open(cand_path, "w", encoding="utf-8") as f:
        for c in all_candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved: {cand_path}")

    # Score all candidates with baseline model
    print("\nScoring with baseline model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    all_scores = []
    all_logits = []
    bs = 32
    for i in range(0, len(all_candidates), bs):
        batch = all_candidates[i:i+bs]
        texts = [smart_truncate(build_input_text(r), tokenizer, 512) for r in batch]
        enc = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        all_logits.extend(logits)

    for i, c in enumerate(all_candidates):
        logit_neg = float(all_logits[i][0]); logit_pos = float(all_logits[i][1])
        c["logit_negative"] = logit_neg; c["logit_positive"] = logit_pos
        c["score"] = logit_pos - logit_neg
        probs = np.exp(all_logits[i]) / np.exp(all_logits[i]).sum()
        c["probability_positive"] = float(probs[1])
        c["probability_negative"] = float(probs[0])

    # Save scored candidates
    with open(cand_path, "w", encoding="utf-8") as f:
        for c in all_candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Group scored candidates
    scored_groups = defaultdict(list)
    for c in all_candidates:
        gk = (c["article_id"], c["target_type"], c["target_code"])
        scored_groups[gk].append(c)

    # ─── Diagnosis ───
    print("\n=== Competition Diagnosis ===")
    diag_rows = []
    # Find same_target_other_event negatives from original pairs
    orig_preds_path = os.path.join(PROJECT_ROOT, "outputs", "event_match_baseline", "validation_predictions.jsonl")
    orig_preds = load_jsonl(orig_preds_path)

    same_target_neg = [p for p in orig_preds if p.get("negative_type") == "same_target_other_event"]
    baseline_fp = [p for p in same_target_neg if p["probability_positive"] >= baseline_threshold]
    print(f"Same-target negatives: {len(same_target_neg)}, baseline FP: {len(baseline_fp)}")

    fixable = 0; rep_error = 0; no_pos = 0
    for p in baseline_fp:
        aid = p["article_id"]; tt = p["target_type"]; tc = str(p["target_code"])
        gk = (aid, tt, tc)
        group_cands = scored_groups.get(gk, [])
        pos_cands = [c for c in group_cands if c["label"] == 1]
        neg_cand = next((c for c in group_cands if c["schema_id"] == p["schema_id"]), None)

        if not pos_cands:
            no_pos += 1; diag_type = "no_positive_candidate"
            best_pos_score = -999; best_pos_sid = ""; best_pos_name = ""
            best_pos_rank = -1; best_pos_prob = 0
        else:
            best_pos = max(pos_cands, key=lambda c: c["score"])
            best_pos_score = best_pos["score"]; best_pos_sid = best_pos["schema_id"]
            best_pos_name = best_pos["event_name"]; best_pos_prob = best_pos["probability_positive"]

        neg_score = neg_cand["score"] if neg_cand else 0
        neg_prob = neg_cand["probability_positive"] if neg_cand else p["probability_positive"]

        # Calculate ranks
        sorted_cands = sorted(group_cands, key=lambda c: -c["score"])
        neg_rank = next((i+1 for i, c in enumerate(sorted_cands) if c["schema_id"] == p["schema_id"]), -1)
        best_pos_rank_calc = next((i+1 for i, c in enumerate(sorted_cands) if c["label"] == 1), -1)

        if pos_cands and best_pos_score > neg_score:
            diag_type = "competition_fixable"; fixable += 1
        elif pos_cands:
            diag_type = "representation_error"; rep_error += 1
        else:
            diag_type = "no_positive_candidate"

        diag_rows.append({
            "article_id": aid, "canonical_group_id": p.get("canonical_group_id", ""),
            "target_type": tt, "target_code": tc, "target_name": p.get("target_name", ""),
            "negative_schema_id": p["schema_id"], "negative_event_name": p.get("event_name", ""),
            "negative_probability": neg_prob, "negative_rank": neg_rank,
            "best_positive_schema_id": best_pos_sid, "best_positive_event_name": best_pos_name,
            "best_positive_probability": best_pos_prob, "best_positive_rank": best_pos_rank_calc,
            "score_margin_positive_minus_negative": round(best_pos_score - neg_score, 4) if pos_cands else 0,
            "diagnosis_type": diag_type,
        })

    pd.DataFrame(diag_rows).to_csv(os.path.join(reports_dir, "same_target_competition_diagnosis.csv"), index=False, encoding="utf-8-sig")

    # Top-1/Top-2 stats
    top1_correct = 0; top2_correct = 0; total_groups = 0; rank_sum = 0
    for gk, cands in scored_groups.items():
        pos_cands = [c for c in cands if c["label"] == 1]
        if not pos_cands: continue
        total_groups += 1
        sorted_cands = sorted(cands, key=lambda c: -c["score"])
        top1 = sorted_cands[0]
        if top1["label"] == 1: top1_correct += 1
        if any(c["label"] == 1 for c in sorted_cands[:2]): top2_correct += 1
        ranks = [i+1 for i, c in enumerate(sorted_cands) if c["label"] == 1]
        rank_sum += min(ranks)

    diag_summary = {
        "same_target_negative_count": len(same_target_neg),
        "baseline_false_positive_count": len(baseline_fp),
        "competition_fixable_count": fixable,
        "representation_error_count": rep_error,
        "no_positive_candidate_count": no_pos,
        "competition_fixable_ratio": round(fixable / len(baseline_fp), 4) if baseline_fp else 0,
        "true_schema_top1_accuracy": round(top1_correct / total_groups, 4) if total_groups else 0,
        "true_schema_top2_recall": round(top2_correct / total_groups, 4) if total_groups else 0,
        "mean_best_positive_rank": round(rank_sum / total_groups, 4) if total_groups else 0,
    }
    with open(os.path.join(reports_dir, "same_target_competition_diagnosis_summary.json"), "w") as f:
        json.dump(diag_summary, f, ensure_ascii=False, indent=2)
    print(f"Diagnosis: {json.dumps(diag_summary, indent=2)}")

    # ─── Parameter Search ───
    print("\n=== Parameter Search ===")

    # Method B: Top-Margin
    abs_thresholds = np.arange(0.25, 0.60, 0.05)
    margins = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]

    # Method C: Relative Softmax
    temperatures = [0.50, 0.75, 1.00, 1.50, 2.00]
    relative_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    # First, build a mapping from original pair key to prediction
    # For evaluating on original 1271 pairs
    orig_pair_map = {}
    for p in orig_preds:
        key = (p["article_id"], p["schema_id"])
        orig_pair_map[key] = p

    def evaluate_method(method, **params):
        """Evaluate on original validation pairs."""
        abs_th = params["absolute_threshold"]
        y_true = []; y_pred = []
        for p in orig_preds:
            aid = p["article_id"]; sid = p["schema_id"]
            tt = p["target_type"]; tc = str(p["target_code"])
            gk = (aid, tt, tc)
            group_cands = scored_groups.get(gk, [])

            if not group_cands or len(group_cands) <= 1:
                # Single schema target: use absolute threshold only
                pred = 1 if p["probability_positive"] >= abs_th else 0
            else:
                if method == "top_margin":
                    max_score = max(c["score"] for c in group_cands)
                    margin = params.get("margin", 0)
                    cand = next((c for c in group_cands if c["schema_id"] == sid), None)
                    if cand:
                        score_ok = cand["score"] >= max_score - margin
                        pred = 1 if (cand["probability_positive"] >= abs_th and score_ok) else 0
                    else:
                        pred = 1 if p["probability_positive"] >= abs_th else 0
                elif method == "relative_softmax":
                    temp = params.get("temperature", 1.0)
                    rel_th = params.get("relative_threshold", 0.5)
                    scores = np.array([c["score"] for c in group_cands])
                    rel_probs = np.exp(scores / temp) / np.exp(scores / temp).sum()
                    cand_idx = next((i for i, c in enumerate(group_cands) if c["schema_id"] == sid), None)
                    if cand_idx is not None:
                        rel_ok = rel_probs[cand_idx] >= rel_th
                        cand = group_cands[cand_idx]
                        pred = 1 if (cand["probability_positive"] >= abs_th and rel_ok) else 0
                    else:
                        pred = 1 if p["probability_positive"] >= abs_th else 0
                else:
                    pred = 1 if p["probability_positive"] >= abs_th else 0

            y_true.append(p["label"]); y_pred.append(pred)

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        yt = np.array(y_true); yp = np.array(y_pred)
        acc = accuracy_score(yt, yp)
        p_s, r_s, f1_s, _ = precision_recall_fscore_support(yt, yp, average='binary', pos_label=1, zero_division=0)
        mp_s, mr_s, mf1_s, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)

        # same_target FPR
        st_indices = [i for i, pr in enumerate(orig_preds) if pr.get("negative_type") == "same_target_other_event"]
        st_yt = yt[st_indices]; st_yp = yp[st_indices]
        st_fp = int(((st_yp == 1) & (st_yt == 0)).sum())
        st_total = int((st_yt == 0).sum())
        st_fpr = st_fp / st_total if st_total > 0 else 0

        fp_count = int(((yp == 1) & (yt == 0)).sum())
        fn_count = int(((yp == 0) & (yt == 1)).sum())
        constraints_met = mf1_s >= 0.9158 and r_s >= 0.85

        return {
            "accuracy": acc, "precision": float(p_s), "recall": float(r_s),
            "positive_f1": float(f1_s), "macro_f1": float(mf1_s),
            "false_positive_count": fp_count, "false_negative_count": fn_count,
            "same_target_false_positive_count": st_fp,
            "same_target_false_positive_rate": round(st_fpr, 4),
            "constraints_met": constraints_met,
        }

    # Search method B
    b_results = []
    for abs_th in abs_thresholds:
        for margin in margins:
            metrics = evaluate_method("top_margin", absolute_threshold=float(abs_th), margin=margin)
            b_results.append({
                "method": "top_margin", "absolute_threshold": round(float(abs_th), 2),
                "margin": margin, "temperature": None, "relative_threshold": None,
                **metrics,
            })

    # Search method C
    c_results = []
    for abs_th in abs_thresholds:
        for temp in temperatures:
            for rel_th in relative_thresholds:
                metrics = evaluate_method("relative_softmax", absolute_threshold=float(abs_th),
                                         temperature=temp, relative_threshold=rel_th)
                c_results.append({
                    "method": "relative_softmax", "absolute_threshold": round(float(abs_th), 2),
                    "margin": None, "temperature": temp, "relative_threshold": rel_th,
                    **metrics,
                })

    pd.DataFrame(b_results).to_csv(os.path.join(args.output_dir, "top_margin_search.csv"), index=False)
    pd.DataFrame(c_results).to_csv(os.path.join(args.output_dir, "relative_softmax_search.csv"), index=False)
    print(f"Top-margin: {len(b_results)} configs, Relative softmax: {len(c_results)} configs")

    # Select best configuration
    all_configs = b_results + c_results
    # Filter by constraints
    valid_configs = [c for c in all_configs if c["constraints_met"]]
    if valid_configs:
        # Sort: min same_target_fpr, then max macro_f1, then max recall, then fewer FP
        valid_configs.sort(key=lambda c: (c["same_target_false_positive_rate"], -c["macro_f1"], -c["recall"], c["false_positive_count"]))
        best_config = valid_configs[0]
        constraints_met = True
    else:
        # No config meets constraints, pick best macro_f1
        all_configs.sort(key=lambda c: (-c["macro_f1"], c["same_target_false_positive_rate"]))
        best_config = all_configs[0]
        constraints_met = False

    print(f"\nBest config: {json.dumps(best_config, indent=2)}")
    print(f"Constraints met: {constraints_met}")

    best_config_out = {
        "method": best_config["method"],
        "absolute_threshold": best_config["absolute_threshold"],
        "margin": best_config.get("margin"),
        "temperature": best_config.get("temperature"),
        "relative_threshold": best_config.get("relative_threshold"),
        "selection_rule": "minimum_same_target_fpr_under_constraints" if constraints_met else "best_macro_f1_no_constraints",
        "acceptance_constraints_met": constraints_met,
        "validation_macro_f1": best_config["macro_f1"],
        "validation_recall": best_config["recall"],
        "same_target_false_positive_rate": best_config["same_target_false_positive_rate"],
    }
    with open(os.path.join(args.output_dir, "best_targetwise_config.json"), "w") as f:
        json.dump(best_config_out, f, ensure_ascii=False, indent=2)

    # ─── Apply best config to original pairs ───
    print("\n=== Applying best config to original pairs ===")
    best_method = best_config["method"]
    best_params = {k: v for k, v in best_config.items() if v is not None}

    orig_pair_preds = []
    for p in orig_preds:
        aid = p["article_id"]; sid = p["schema_id"]
        tt = p["target_type"]; tc = str(p["target_code"])
        gk = (aid, tt, tc)
        group_cands = scored_groups.get(gk, [])
        abs_th = best_config["absolute_threshold"]

        if len(group_cands) <= 1:
            tw_pred = 1 if p["probability_positive"] >= abs_th else 0
        elif best_method == "top_margin":
            max_score = max(c["score"] for c in group_cands)
            margin = best_config.get("margin", 0)
            cand = next((c for c in group_cands if c["schema_id"] == sid), None)
            if cand:
                tw_pred = 1 if (cand["probability_positive"] >= abs_th and cand["score"] >= max_score - margin) else 0
            else:
                tw_pred = 1 if p["probability_positive"] >= abs_th else 0
        elif best_method == "relative_softmax":
            temp = best_config.get("temperature", 1.0)
            rel_th = best_config.get("relative_threshold", 0.5)
            scores = np.array([c["score"] for c in group_cands])
            rel_probs = np.exp(scores / temp) / np.exp(scores / temp).sum()
            cand_idx = next((i for i, c in enumerate(group_cands) if c["schema_id"] == sid), None)
            if cand_idx is not None:
                cand = group_cands[cand_idx]
                tw_pred = 1 if (cand["probability_positive"] >= abs_th and rel_probs[cand_idx] >= rel_th) else 0
            else:
                tw_pred = 1 if p["probability_positive"] >= abs_th else 0
        else:
            tw_pred = 1 if p["probability_positive"] >= abs_th else 0

        p_out = dict(p)
        p_out["targetwise_prediction"] = tw_pred
        p_out["targetwise_method"] = best_method
        orig_pair_preds.append(p_out)

    ori_path = os.path.join(args.output_dir, "original_pair_predictions.jsonl")
    with open(ori_path, "w", encoding="utf-8") as f:
        for p in orig_pair_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    pd.DataFrame(orig_pair_preds).to_excel(
        os.path.join(args.output_dir, "original_pair_predictions.xlsx"), index=False, engine="openpyxl")
    print(f"Saved: {ori_path}")

    # Apply best config to ALL candidates
    all_preds = []
    for gk, cands in scored_groups.items():
        abs_th = best_config["absolute_threshold"]
        if len(cands) <= 1:
            for c in cands:
                c_out = dict(c)
                c_out["targetwise_prediction"] = 1 if c["probability_positive"] >= abs_th else 0
                all_preds.append(c_out)
        elif best_method == "top_margin":
            max_score = max(c["score"] for c in cands)
            margin = best_config.get("margin", 0)
            for c in cands:
                c_out = dict(c)
                c_out["targetwise_prediction"] = 1 if (c["probability_positive"] >= abs_th and c["score"] >= max_score - margin) else 0
                c_out["max_group_score"] = max_score
                c_out["candidate_rank"] = sum(1 for cc in cands if cc["score"] > c["score"]) + 1
                all_preds.append(c_out)
        elif best_method == "relative_softmax":
            temp = best_config.get("temperature", 1.0)
            rel_th = best_config.get("relative_threshold", 0.5)
            scores = np.array([c["score"] for c in cands])
            rel_probs = np.exp(scores / temp) / np.exp(scores / temp).sum()
            for i, c in enumerate(cands):
                c_out = dict(c)
                c_out["targetwise_prediction"] = 1 if (c["probability_positive"] >= abs_th and rel_probs[i] >= rel_th) else 0
                c_out["relative_probability"] = float(rel_probs[i])
                c_out["max_group_score"] = float(scores.max())
                c_out["candidate_rank"] = sum(1 for cc in cands if cc["score"] > c["score"]) + 1
                all_preds.append(c_out)

    all_path = os.path.join(args.output_dir, "all_candidate_predictions.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for p in all_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    pd.DataFrame(all_preds).to_excel(
        os.path.join(args.output_dir, "all_candidate_predictions.xlsx"), index=False, engine="openpyxl")
    print(f"Saved: {all_path}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
