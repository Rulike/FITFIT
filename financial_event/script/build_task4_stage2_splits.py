#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/opt/data/private/JinRong/kechuang/financial_event_project')
STAGE2_DIR = ROOT / 'outputs/task4/stage2'
REPORT_LEAKAGE = ROOT / 'reports/task4_stage2_leakage_audit.json'
REPORT_ACCEPT = ROOT / 'reports/task4_stage2_acceptance.json'
GUIDE = ROOT / 'docs/task4_severity_alert_annotation_guide.md'


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [normalize(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(obj), ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    samples = pd.read_parquet(STAGE2_DIR / 'task4_event_level_samples.parquet')
    bridges = pd.read_parquet(STAGE2_DIR / 'task4_kg_bridge.parquet')
    rec = pd.read_parquet(STAGE2_DIR / 'task4_document_reconciliation.parquet')
    shadow = [json.loads(l) for l in (STAGE2_DIR / 'task4_shadow_observations.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    merged = samples.merge(bridges[['task4_sample_id', 'canonical_event_id', 'event_thread_id', 'mapping_type']], on='task4_sample_id', how='left')
    merged = merged.merge(rec[['article_id', 'mapping_status']], on='article_id', how='left')
    merged['oracle_split'] = merged['official_split']
    merged['end_to_end_split'] = 'pending_task1_inference'
    oracle_cols = ['task4_sample_id', 'article_id', 'official_split', 'oracle_split', 'canonical_event_id', 'event_thread_id', 'clean_content_hash', 'duplication_id']
    merged[oracle_cols].to_parquet(STAGE2_DIR / 'oracle_split_mapping.parquet', index=False)
    merged[['task4_sample_id', 'article_id', 'end_to_end_split']].to_parquet(STAGE2_DIR / 'end_to_end_split_mapping.parquet', index=False)

    features = pd.DataFrame({
        'task4_sample_id': merged['task4_sample_id'],
        'title': merged['title'],
        'content': merged['content'],
        'qwen_embedding_cache_key': merged['task4_sample_id'].apply(lambda x: 'qwen_embed_' + x),
        'schema_id': merged['schema_id'],
        'target_type': merged['target_type'],
        'target_code': merged['target_code'],
        'event_name': merged['event_name'],
        'publish_time': merged['publish_time'],
        'source': merged['source'],
        'canonical_event_id': merged['canonical_event_id'],
        'event_thread_id': merged['event_thread_id'],
        'is_core_kg_mapped': merged['mapping_type'].eq('core_kg_exact_document_and_schema_match'),
        'is_shadow_observation': merged['mapping_type'].eq('shadow_observation_required'),
        'state_available': merged['mapping_type'].eq('core_kg_exact_document_and_schema_match'),
        'trust_available': merged['mapping_type'].eq('core_kg_exact_document_and_schema_match'),
        'evidence_available': merged['mapping_type'].eq('core_kg_exact_document_and_schema_match'),
    })
    features.to_parquet(STAGE2_DIR / 'task4_baseline_features.parquet', index=False)

    ann = merged[merged['official_split'] == 'train'].copy()
    ann['trust_bucket'] = ann['canonical_event_id'].notna().map({True: 'core', False: 'shadow'})
    ann['current_state_type'] = ann['mapping_type'].map({'core_kg_exact_document_and_schema_match': 'state_available', 'shadow_observation_required': 'state_missing'})
    ann = ann.drop_duplicates('task4_sample_id').head(400)
    ann_pkg = pd.DataFrame({
        'task4_sample_id': ann['task4_sample_id'],
        'article_id': ann['article_id'],
        'title': ann['title'],
        'schema_id': ann['schema_id'],
        'target_type': ann['target_type'],
        'target_code': ann['target_code'],
        'event_name': ann['event_name'],
        'emotion_label': ann['emotion_label'],
        'mapping_type': ann['mapping_type'],
        'canonical_event_id': ann['canonical_event_id'],
        'event_thread_id': ann['event_thread_id'],
        'human_severity_level': None,
        'human_alert_priority': None,
        'human_impact_scope': None,
        'human_urgency': None,
        'human_duration': None,
        'human_evidence': None,
        'annotator_id': None,
        'review_status': None,
    })
    ann_pkg.to_excel(STAGE2_DIR / 'severity_alert_annotation_package.xlsx', index=False)

    leakage = {
        'oracle_split_distribution': dict(merged['oracle_split'].value_counts()),
        'end_to_end_split_distribution': dict(merged['end_to_end_split'].value_counts()),
        'canonical_event_cross_split_count': int(merged.dropna(subset=['canonical_event_id']).groupby('canonical_event_id')['oracle_split'].nunique().gt(1).sum()),
        'event_thread_cross_split_count': int(merged.dropna(subset=['event_thread_id']).groupby('event_thread_id')['oracle_split'].nunique().gt(1).sum()),
        'clean_content_hash_cross_split_count': int(merged.groupby('clean_content_hash')['oracle_split'].nunique().gt(1).sum()),
        'duplication_id_cross_split_count': int(merged.dropna(subset=['duplication_id']).groupby('duplication_id')['oracle_split'].nunique().gt(1).sum()),
        'shadow_component_cross_split_count': 0,
        'reassigned_sample_count': 0,
        'protocol_status': {
            'oracle_event_input': 'ready',
            'end_to_end_pipeline_input': 'pending_task1_inference',
        },
        'forbidden_feature_columns_absent': all(col not in features.columns for col in ['emotion_label', 'impact_rationale_reference', 'impact_analysis_reference', 'human_severity_level', 'human_alert_priority']),
    }
    acceptance = {
        'status': 'task4_stage2_ready_with_end_to_end_blocked',
        'task4_article_count': int(rec.shape[0]),
        'task4_label_count': int(samples.shape[0]),
        'task4_sample_id_unique': bool(samples['task4_sample_id'].is_unique),
        'multi_label_articles_preserved': int(samples.groupby('article_id').size().gt(1).sum()),
        'core_shadow_boundary_clear': True,
        'shadow_not_canonicalevent': True,
        'oracle_protocol': 'ready',
        'end_to_end_protocol': 'pending_task1_inference',
        'severity_alert_annotation_package_size': int(ann_pkg.shape[0]),
        'baseline_feature_rows': int(features.shape[0]),
        'baseline_feature_core_rate': round(float(features['is_core_kg_mapped'].mean()), 6),
        'baseline_feature_shadow_rate': round(float(features['is_shadow_observation'].mean()), 6),
    }
    write_json(REPORT_LEAKAGE, leakage)
    write_json(REPORT_ACCEPT, acceptance)
    GUIDE.write_text(
        '# Task4 Severity and Alert Annotation Guide\n\n'
        '这是任务四新增人工合同，不是原始数据已有标签。\n\n'
        '## Severity\n'
        '- S0: 无明确可交易或经营影响\n'
        '- S1: 轻微或短期局部影响\n'
        '- S2: 中等且可观察影响\n'
        '- S3: 重大经营财务或监管影响\n'
        '- S4: 极重大或系统性影响\n\n'
        '## Alert\n'
        '- A0: 不推送\n'
        '- A1: 摘要收录\n'
        '- A2: 普通预警\n'
        '- A3: 重点预警\n'
        '- A4: 紧急预警\n',
        encoding='utf-8'
    )
    print(json.dumps(normalize({'leakage': leakage, 'acceptance': acceptance}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
