#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / 'outputs/task3_kg_v2/frozen_test_package_agent_reviewed.xlsx'
FROZEN = ROOT / 'outputs/task3_kg_v2/step4_revision2_acceptance/frozen_test_package.xlsx'
ENRICHED = ROOT / 'outputs/task3_kg_v2/step4_revision3/frozen_test_package_enriched.xlsx'
DEV = ROOT / 'outputs/task3_kg_v2/step4_revision/annotation_package_agent_reviewed.xlsx'
PAIR_PRED = ROOT / 'outputs/task3_kg_v2/step4_revision3/pair_predictions.parquet'
CANON_MAP = ROOT / 'outputs/task3_kg_v2/step4_revision3/event_mention_canonical_mapping.jsonl'
OUT_JSON = ROOT / 'reports/task3_kg_v2_step4_revision3_clean_acceptance.json'
OUT_MD = ROOT / 'docs/task3_kg_v2_step4_revision3_clean_acceptance.md'
OUT_XLSX = ROOT / 'outputs/task3_kg_v2/step4_revision3/leakage_risk_pairs.xlsx'


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_safe(obj: Any) -> Any:
    if obj is pd.NA or (isinstance(obj, float) and math.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), ensure_ascii=False, indent=2), encoding='utf-8')


def ece_standard(y_true: np.ndarray, p_same: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    pred = (p_same >= 0.5).astype(int)
    edges = np.linspace(0, 1, bins + 1)
    total = len(p_same)
    rows = []
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p_same >= lo) & (p_same < hi if i < bins - 1 else p_same <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({'bin': i, 'lower': float(lo), 'upper': float(hi), 'count': 0, 'avg_p_same_event': None, 'empirical_positive_rate': None, 'accuracy': None, 'gap': None})
            continue
        avg_p = float(p_same[mask].mean())
        pos_rate = float(y_true[mask].mean())
        acc = float((pred[mask] == y_true[mask]).mean())
        gap = abs(avg_p - pos_rate)
        ece += (n / total) * gap
        rows.append({'bin': i, 'lower': float(lo), 'upper': float(hi), 'count': n, 'avg_p_same_event': avg_p, 'empirical_positive_rate': pos_rate, 'accuracy': acc, 'gap': gap})
    return float(ece), rows


def compute_metrics(df: pd.DataFrame, include_calibration: bool = True) -> dict[str, Any]:
    y_true = df['gold_same'].to_numpy()
    pred_same = (df['pred_label'] == 'same_event').astype(int).to_numpy()
    pred_diff = (df['pred_label'] == 'different_event').astype(int).to_numpy()
    gold_diff = 1 - y_true
    p_same, r_same, f1_same, _ = precision_recall_fscore_support(y_true, pred_same, average='binary', zero_division=0)
    p_diff, r_diff, f1_diff, _ = precision_recall_fscore_support(gold_diff, pred_diff, average='binary', zero_division=0)
    out = {
        'rows': int(len(df)),
        'same_event_precision': float(p_same),
        'same_event_recall': float(r_same),
        'same_event_f1': float(f1_same),
        'different_event_precision': float(p_diff),
        'different_event_recall': float(r_diff),
        'different_event_f1': float(f1_diff),
        'macro_f1': float((f1_same + f1_diff) / 2),
        'automatic_coverage': float((df['pred_label'] != 'uncertain_review').mean()),
        'confusion_matrix': {
            'gold_yes_pred_same_event': int(((df['gold_same'] == 1) & (df['pred_label'] == 'same_event')).sum()),
            'gold_yes_pred_different_event': int(((df['gold_same'] == 1) & (df['pred_label'] == 'different_event')).sum()),
            'gold_yes_pred_uncertain_review': int(((df['gold_same'] == 1) & (df['pred_label'] == 'uncertain_review')).sum()),
            'gold_no_pred_same_event': int(((df['gold_same'] == 0) & (df['pred_label'] == 'same_event')).sum()),
            'gold_no_pred_different_event': int(((df['gold_same'] == 0) & (df['pred_label'] == 'different_event')).sum()),
            'gold_no_pred_uncertain_review': int(((df['gold_same'] == 0) & (df['pred_label'] == 'uncertain_review')).sum()),
        },
    }
    strata = {}
    for stratum, sdf in df.groupby('sample_stratum'):
        yt = sdf['gold_same'].to_numpy()
        ps = (sdf['pred_label'] == 'same_event').astype(int).to_numpy()
        pdiff = (sdf['pred_label'] == 'different_event').astype(int).to_numpy()
        gd = 1 - yt
        sp, sr, sf1, _ = precision_recall_fscore_support(yt, ps, average='binary', zero_division=0)
        dp, dr, df1, _ = precision_recall_fscore_support(gd, pdiff, average='binary', zero_division=0)
        strata[str(stratum)] = {
            'rows': int(len(sdf)),
            'same_event_precision': float(sp),
            'same_event_recall': float(sr),
            'same_event_f1': float(sf1),
            'different_event_precision': float(dp),
            'different_event_recall': float(dr),
            'different_event_f1': float(df1),
            'macro_f1': float((sf1 + df1) / 2),
            'automatic_coverage': float((sdf['pred_label'] != 'uncertain_review').mean()),
        }
    out['by_sample_stratum'] = strata
    if include_calibration:
        p_same = df['same_event_score_r3'].to_numpy() if 'same_event_score_r3' in df.columns and df['same_event_score_r3'].notna().all() else None
        if p_same is None:
            out['calibration'] = {'probability_field': None, 'status': 'not_available', 'brier_score': None, 'ece': None, 'confidence_bins': []}
        else:
            brier = float(np.mean((p_same - y_true) ** 2))
            ece, bins = ece_standard(y_true, p_same, bins=10)
            out['calibration'] = {'probability_field': 'same_event_score_r3', 'status': 'available', 'brier_score': brier, 'ece': ece, 'confidence_bins': bins}
    return out


def main() -> dict[str, Any]:
    started = time.time()
    ann = pd.read_excel(ANN)
    frozen = pd.read_excel(FROZEN)
    audit = pd.read_excel(ENRICHED, sheet_name='audit_metadata')
    pred = pd.read_parquet(PAIR_PRED)
    dev = pd.read_excel(DEV)
    comp_map = {r['event_mention_id']: r['canonical_event_id'] for r in read_jsonl(CANON_MAP)}

    # exact restore by pair_id
    restore = audit[['pair_id', 'sample_stratum']].copy()
    if 'split_suggestion' in audit.columns:
        restore = restore[['pair_id', 'sample_stratum', 'split_suggestion']]
    else:
        restore['split_suggestion'] = pd.NA
    merged = ann.merge(restore, on='pair_id', how='left').merge(pred, on='pair_id', how='left', suffixes=('', '_pred'))
    merged['gold_same'] = merged['human_same_event'].map({'yes': 1, 'no': 0})
    merged['pred_label'] = merged['revision3_relation_label'].fillna('candidate_missing')

    dev_comps = {comp_map.get(x) for x in set(dev['event_mention_id_a']) | set(dev['event_mention_id_b']) if comp_map.get(x)}
    overlap_rows = []
    clean_mask = []
    for _, r in merged.iterrows():
        pair_comps = sorted({comp_map.get(r['event_mention_id_a']), comp_map.get(r['event_mention_id_b'])} - {None, ''})
        overlap = sorted(set(pair_comps) & dev_comps)
        has_overlap = len(overlap) > 0
        clean_mask.append(not has_overlap)
        if has_overlap:
            overlap_rows.append({
                'pair_id': r['pair_id'],
                'event_mention_id_a': r['event_mention_id_a'],
                'event_mention_id_b': r['event_mention_id_b'],
                'sample_stratum': r['sample_stratum'],
                'split_suggestion': None if pd.isna(r['split_suggestion']) else r['split_suggestion'],
                'overlap_components': '|'.join(overlap),
                'title_a': r.get('title_a', ''),
                'title_b': r.get('title_b', ''),
                'human_same_event': r['human_same_event'],
                'pred_label': r['pred_label'],
            })
    merged['component_overlap_with_dev'] = [not x for x in clean_mask]
    clean = merged[np.array(clean_mask)].copy()

    full_metrics = compute_metrics(merged, include_calibration=True)
    clean_metrics = compute_metrics(clean, include_calibration=True)

    under_merge = merged[(merged['human_same_event'] == 'yes') & (merged['pred_label'].isin(['different_event', 'uncertain_review', 'candidate_missing']))].copy()
    under_merge_cases = under_merge[['pair_id', 'sample_stratum', 'split_suggestion', 'title_a', 'title_b', 'human_same_event', 'pred_label', 'same_event_score_r3', 'different_event_score_r3', 'event_identity_key_a', 'event_identity_key_b', 'identity_fields_a', 'identity_fields_b', 'evidence_sentences_a', 'evidence_sentences_b']].to_dict('records')

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX) as writer:
        pd.DataFrame(overlap_rows).to_excel(writer, sheet_name='leakage_risk_pairs', index=False)
        clean[['pair_id', 'sample_stratum', 'split_suggestion', 'human_same_event', 'pred_label']].to_excel(writer, sheet_name='leakage_clean_pairs', index=False)
        under_merge.to_excel(writer, sheet_name='under_merge_cases', index=False)

    report = {
        'version': 'task3_kg_v2_step4_revision3_clean_acceptance_v0',
        'status': 'phase4_engineering_frozen_pending_human_gold',
        'no_model_change': True,
        'no_threshold_change': True,
        'no_canonical_event_rebuild': True,
        'no_phase5_entry': True,
        'restored_fields': {
            'sample_stratum_source': 'outputs/task3_kg_v2/step4_revision3/frozen_test_package_enriched.xlsx#audit_metadata via pair_id exact join',
            'split_suggestion_source': 'not_available_in_source_files',
        },
        'leakage_overlap': {
            'distinct_same_event_components': sorted({x for r in overlap_rows for x in str(r['overlap_components']).split('|') if x}),
            'distinct_component_count': len(sorted({x for r in overlap_rows for x in str(r['overlap_components']).split('|') if x})),
            'overlap_pair_count': int(len(overlap_rows)),
            'overlap_pairs': overlap_rows,
        },
        'candidate_pool': {
            'full_150_candidate_missing_count': int((merged['pred_label'] == 'candidate_missing').sum()),
            'clean_subset_candidate_missing_count': int((clean['pred_label'] == 'candidate_missing').sum()),
        },
        'full_150_metrics': full_metrics,
        'leakage_clean_metrics': clean_metrics,
        'under_merge_cases': under_merge_cases,
        'acceptance_recommendation': {
            'freeze_phase4_engineering_artifacts': True,
            'pending_human_gold_before_formal_acceptance': True,
            'new_dev_label_need_if_any': '对正文语义不足的欠合并样本补充开发标注，不根据冻结测试调参。',
        },
        'sha256': {
            'agent_reviewed_file': hashlib.sha256(ANN.read_bytes()).hexdigest(),
            'frozen_test_package': hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
            'pair_predictions': hashlib.sha256(PAIR_PRED.read_bytes()).hexdigest(),
            'canonical_mapping': hashlib.sha256(CANON_MAP.read_bytes()).hexdigest(),
        },
        'resource_usage': {
            'elapsed_seconds': round(time.time() - started, 3),
            'max_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        },
    }
    write_json(OUT_JSON, report)
    OUT_MD.write_text('# Task3 KG v2 Step4 Revision3 Clean Acceptance\n\n' + json.dumps({
        'status': report['status'],
        'leakage_overlap': {
            'distinct_component_count': report['leakage_overlap']['distinct_component_count'],
            'overlap_pair_count': report['leakage_overlap']['overlap_pair_count'],
        },
        'full_150_metrics': report['full_150_metrics'],
        'leakage_clean_metrics': report['leakage_clean_metrics'],
        'under_merge_cases': report['under_merge_cases'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    main()
