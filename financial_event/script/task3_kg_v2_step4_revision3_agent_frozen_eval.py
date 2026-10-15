#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / 'outputs/task3_kg_v2/frozen_test_package_agent_reviewed.xlsx'
REV3 = ROOT / 'outputs/task3_kg_v2/step4_revision3'
PAIR_PRED = REV3 / 'pair_predictions.parquet'
CANON_MAP = REV3 / 'event_mention_canonical_mapping.jsonl'
DEV = ROOT / 'outputs/task3_kg_v2/step4_revision/annotation_package_agent_reviewed.xlsx'
DEV2 = ROOT / 'outputs/task3_kg_v2/step4_revision/update_annotation_package.xlsx'
OUT_JSON = ROOT / 'reports/task3_kg_v2_step4_revision3_agent_frozen_evaluation.json'
OUT_MD = ROOT / 'docs/task3_kg_v2_step4_revision3_agent_frozen_evaluation.md'
OUT_XLSX = REV3 / 'agent_frozen_error_analysis.xlsx'


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def ece_score(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, conf: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0, 1, bins + 1)
    total = len(conf)
    rows = []
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({'bin': i, 'lower': float(lo), 'upper': float(hi), 'count': 0, 'avg_confidence': None, 'accuracy': None, 'gap': None})
            continue
        avg_conf = float(conf[mask].mean())
        acc = float((y_true_binary[mask] == y_pred_binary[mask]).mean())
        gap = abs(avg_conf - acc)
        ece += n / total * gap
        rows.append({'bin': i, 'lower': float(lo), 'upper': float(hi), 'count': n, 'avg_confidence': avg_conf, 'accuracy': acc, 'gap': gap})
    return float(ece), rows


def infer_error_type(r: pd.Series) -> str:
    true_label = r['gold_label']
    pred = r['pred_label']
    if r['candidate_missing']:
        return '候选未召回'
    if bool(r.get('hard_conflict', False)) and pred == 'same_event':
        return '硬冲突违规'
    if true_label == 'no' and pred == 'same_event':
        if float(r.get('event_identity_conflict', 0.0)) > 0:
            return 'IdentityKey冲突'
        if str(r.get('event_identity_key_a', '')) == str(r.get('event_identity_key_b', '')) and str(r.get('event_identity_key_a', '')):
            return 'IdentityKey过粗'
        if abs(float(r.get('time_gap_hours', 0.0))) >= 24:
            return '时间批次混淆'
        if not str(r.get('identity_fields_a', '')).strip('[]') or not str(r.get('identity_fields_b', '')).strip('[]'):
            return '字段缺失'
        return '语义误判'
    if true_label == 'yes' and pred in {'different_event', 'uncertain_review'}:
        if abs(float(r.get('time_gap_hours', 0.0))) <= 1 and pred != 'same_event':
            return '正文语义不足'
        if not str(r.get('identity_fields_a', '')).strip('[]') or not str(r.get('identity_fields_b', '')).strip('[]'):
            return '核心对象缺失'
        if pred == 'uncertain_review':
            return '证据不足'
        return '语义误判'
    return ''


def main() -> dict[str, Any]:
    started = time.time()
    ann = pd.read_excel(ANN)
    assert ann['pair_id'].is_unique
    assert (ann['human_same_event'].fillna('').astype(str).str.strip() != '').all()
    assert (ann['review_status'] == 'agent_reviewed_pending_human_confirmation').all()

    pred = pd.read_parquet(PAIR_PRED)
    merged = ann.merge(pred, on='pair_id', how='left', suffixes=('', '_pred'))
    merged['candidate_missing'] = merged['revision3_relation_label'].isna()
    merged['gold_label'] = merged['human_same_event'].map({'yes': 'yes', 'no': 'no'})
    merged['pred_label'] = merged['revision3_relation_label'].fillna('candidate_missing')
    merged.loc[merged['pred_label'] == 'same_event', 'pred_binary'] = 'yes'
    merged.loc[merged['pred_label'].isin(['different_event', 'uncertain_review', 'candidate_missing']), 'pred_binary'] = 'no'

    y_true = merged['gold_label'].map({'yes': 1, 'no': 0}).to_numpy()
    y_pred_same = (merged['pred_label'] == 'same_event').astype(int).to_numpy()
    p_same, r_same, f1_same, _ = precision_recall_fscore_support(y_true, y_pred_same, average='binary', zero_division=0)

    y_true_no = (merged['gold_label'] == 'no').astype(int).to_numpy()
    y_pred_no = (merged['pred_label'] == 'different_event').astype(int).to_numpy()
    p_no, r_no, f1_no, _ = precision_recall_fscore_support(y_true_no, y_pred_no, average='binary', zero_division=0)
    macro_f1 = float((f1_same + f1_no) / 2)

    uncertain_rate = float((merged['pred_label'] == 'uncertain_review').mean())
    selective_mask = merged['pred_label'] != 'uncertain_review'
    selective_acc = float((merged.loc[selective_mask, 'pred_binary'] == merged.loc[selective_mask, 'gold_label']).mean()) if selective_mask.any() else 0.0
    automatic_coverage = float(selective_mask.mean())

    conf = merged['revision3_confidence'].fillna(0.0).to_numpy()
    pred_binary = (merged['pred_label'] == 'same_event').astype(int).to_numpy()
    ece, bins = ece_score(y_true, pred_binary, conf, bins=10)
    brier = float(np.mean((conf - y_true) ** 2))

    cm_labels = ['same_event', 'different_event', 'uncertain_review', 'candidate_missing']
    cm = confusion_matrix(merged['gold_label'].map({'yes': 'same_event', 'no': 'different_event'}), merged['pred_label'], labels=['same_event', 'different_event'])
    confusion = {
        'gold_yes_pred_same_event': int(((merged['gold_label'] == 'yes') & (merged['pred_label'] == 'same_event')).sum()),
        'gold_yes_pred_different_event': int(((merged['gold_label'] == 'yes') & (merged['pred_label'] == 'different_event')).sum()),
        'gold_yes_pred_uncertain_review': int(((merged['gold_label'] == 'yes') & (merged['pred_label'] == 'uncertain_review')).sum()),
        'gold_no_pred_same_event': int(((merged['gold_label'] == 'no') & (merged['pred_label'] == 'same_event')).sum()),
        'gold_no_pred_different_event': int(((merged['gold_label'] == 'no') & (merged['pred_label'] == 'different_event')).sum()),
        'gold_no_pred_uncertain_review': int(((merged['gold_label'] == 'no') & (merged['pred_label'] == 'uncertain_review')).sum()),
        'candidate_missing': int((merged['candidate_missing']).sum()),
    }

    if 'sample_stratum' not in merged.columns:
        merged['sample_stratum'] = merged['pair_id'].map(lambda _: 'unknown')
    strata_metrics = {}
    for stratum, sdf in merged.groupby('sample_stratum'):
        yt = sdf['gold_label'].map({'yes': 1, 'no': 0}).to_numpy()
        yp = (sdf['pred_label'] == 'same_event').astype(int).to_numpy()
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
        strata_metrics[str(stratum)] = {
            'rows': int(len(sdf)),
            'same_event_precision': float(p),
            'same_event_recall': float(r),
            'same_event_f1': float(f1),
            'uncertain_rate': float((sdf['pred_label'] == 'uncertain_review').mean()),
            'automatic_coverage': float((sdf['pred_label'] != 'uncertain_review').mean()),
            'label_distribution': sdf['gold_label'].value_counts().to_dict(),
        }

    dev = pd.read_excel(DEV)
    dev2 = pd.read_excel(DEV2)
    ann_docs = set(merged['document_id_a']) | set(merged['document_id_b'])
    dev_docs = set(dev['document_id_a']) | set(dev['document_id_b'])
    ann_eu = set(merged['evidence_unit_id_a']) | set(merged['evidence_unit_id_b'])
    dev_eu = set(dev['evidence_unit_id_a']) | set(dev['evidence_unit_id_b'])
    dev2_eu = set(dev2['same_evidence_unit_ids_a'].dropna().astype(str)) | set(dev2['same_evidence_unit_ids_b'].dropna().astype(str))
    comp_map = {r['event_mention_id']: r['canonical_event_id'] for r in read_jsonl(CANON_MAP)}
    ann_comps = {comp_map.get(x, '') for x in set(merged['event_mention_id_a']) | set(merged['event_mention_id_b']) if comp_map.get(x, '')}
    dev_comps = {comp_map.get(x, '') for x in set(dev['event_mention_id_a']) | set(dev['event_mention_id_b']) if comp_map.get(x, '')}

    false_merge = merged[(merged['gold_label'] == 'no') & (merged['pred_label'] == 'same_event')].copy()
    under_merge = merged[(merged['gold_label'] == 'yes') & (merged['pred_label'].isin(['different_event', 'uncertain_review', 'candidate_missing']))].copy()
    error_df = pd.concat([false_merge, under_merge], ignore_index=True)
    error_df['error_type'] = error_df.apply(infer_error_type, axis=1)
    hard_conflict_violations = int(((merged['gold_label'] == 'no') & (merged['pred_label'] == 'same_event') & (merged['hard_conflict'] == True)).sum())

    with pd.ExcelWriter(OUT_XLSX) as writer:
        false_merge.to_excel(writer, sheet_name='false_merge', index=False)
        under_merge.to_excel(writer, sheet_name='under_merge', index=False)
        error_df.to_excel(writer, sheet_name='error_summary', index=False)

    report = {
        'version': 'task3_kg_v2_step4_revision3_agent_frozen_pre_acceptance_v0',
        'metric_status': 'agent_reviewed_frozen_pre_acceptance',
        'formal_human_gold_test': False,
        'rows': int(len(merged)),
        'label_checks': {
            'pair_id_unique': bool(ann['pair_id'].is_unique),
            'human_same_event_non_empty': bool((ann['human_same_event'].fillna('').astype(str).str.strip() != '').all()),
            'label_distribution': ann['human_same_event'].value_counts().to_dict(),
            'review_status_all_pending_human_confirmation': bool((ann['review_status'] == 'agent_reviewed_pending_human_confirmation').all()),
            'leakage': {
                'dev_pair_overlap': int(len(set(ann['pair_id']) & set(dev['pair_id']))),
                'dev_document_overlap': int(len(ann_docs & dev_docs)),
                'dev_evidence_unit_overlap': int(len({x for x in ann_eu if isinstance(x, str) and x and x != 'nan'} & {x for x in dev_eu if isinstance(x, str) and x and x != 'nan'})),
                'dev_update_evidence_unit_overlap': int(len({x for x in ann_eu if isinstance(x, str) and x and x != 'nan'} & {x for x in dev2_eu if isinstance(x, str) and x and x != 'nan'})),
                'same_event_component_overlap': int(len(ann_comps & dev_comps)),
            },
        },
        'candidate_coverage': {
            'candidate_missing_count': int((merged['candidate_missing']).sum()),
            'candidate_present_count': int((~merged['candidate_missing']).sum()),
        },
        'metrics': {
            'same_event_precision': float(p_same),
            'same_event_recall': float(r_same),
            'same_event_f1': float(f1_same),
            'different_event_precision': float(p_no),
            'different_event_recall': float(r_no),
            'different_event_f1': float(f1_no),
            'macro_f1': macro_f1,
            'uncertain_rejection_rate': uncertain_rate,
            'selective_accuracy': selective_acc,
            'automatic_coverage': automatic_coverage,
            'ece': float(ece),
            'brier_score': brier,
        },
        'by_sample_stratum': strata_metrics,
        'confusion_matrix': confusion,
        'confidence_bins': bins,
        'audits': {
            'false_merge_count': int(len(false_merge)),
            'under_merge_count': int(len(under_merge)),
            'hard_conflict_violation_count': hard_conflict_violations,
            'error_type_distribution': error_df['error_type'].value_counts().to_dict(),
        },
        'acceptance_recommendation': {
            'freeze_phase4': bool(macro_f1 >= 0.75 and hard_conflict_violations == 0),
            'no_model_change_from_frozen_test': True,
            'if_not_passed': '只总结错误原因并新增开发标注需求，不根据冻结测试修改模型。',
            'enter_phase5': False,
        },
        'sha256': {
            'agent_reviewed_file': hashlib.sha256(ANN.read_bytes()).hexdigest(),
            'revision3_pair_predictions': hashlib.sha256(PAIR_PRED.read_bytes()).hexdigest(),
            'revision3_canonical_mapping': hashlib.sha256(CANON_MAP.read_bytes()).hexdigest(),
        },
        'resource_usage': {
            'elapsed_seconds': round(time.time() - started, 3),
            'max_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        },
    }
    write_json(OUT_JSON, report)
    OUT_MD.write_text('# Task3 KG v2 Step4 Revision3 Agent Frozen Evaluation\n\n' + json.dumps({
        'metric_status': report['metric_status'],
        'metrics': report['metrics'],
        'acceptance_recommendation': report['acceptance_recommendation'],
        'audits': report['audits'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    main()
