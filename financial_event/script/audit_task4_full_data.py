#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/opt/data/private/JinRong/kechuang/financial_event_project')
RELEASE_DIR = ROOT / 'outputs/task3_kg_v3_release'
FINAL_INTERFACE = ROOT / 'outputs/task3_kg_v3_final/task4_event_interface.jsonl'
OUT_AUDIT = ROOT / 'outputs/task4/data_audit'
REPORT_DATA = ROOT / 'reports/task4_stage1_data_audit.json'
REPORT_INTERFACE = ROOT / 'reports/task4_stage1_interface_audit.json'
REPORT_SPLIT = ROOT / 'reports/task4_stage1_split_audit.json'
DOC_DATA = ROOT / 'docs/task4_stage1_data_and_protocol.md'
DOC_ARCH = ROOT / 'docs/task4_trace_alert_architecture.md'


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_hash(text: str) -> str:
    return hashlib.sha256(' '.join((text or '').split()).encode('utf-8')).hexdigest()


def audit_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_rows = [lab for row in rows for lab in row.get('labels', [])]
    return {
        'row_count': len(rows),
        'label_count': len(label_rows),
        'non_null_rate': {
            'article_id': round(sum(1 for r in rows if r.get('article_id')) / len(rows), 6),
            'title': round(sum(1 for r in rows if r.get('title')) / len(rows), 6),
            'content': round(sum(1 for r in rows if r.get('content')) / len(rows), 6),
            'publish_time': round(sum(1 for r in rows if r.get('publish_time')) / len(rows), 6),
            'duplication_id': round(sum(1 for r in rows if r.get('duplication_id')) / len(rows), 6),
            'labels': round(sum(1 for r in rows if r.get('labels')) / len(rows), 6),
            'labels.emotion': round(sum(1 for lab in label_rows if lab.get('emotion')) / len(label_rows), 6) if label_rows else 0.0,
            'labels.impact_analysis': round(sum(1 for lab in label_rows if lab.get('impact_analysis')) / len(label_rows), 6) if label_rows else 0.0,
        },
        'unique_values': {
            'source_count': len({r.get('source') for r in rows}),
            'duplication_id_count': len({r.get('duplication_id') for r in rows if r.get('duplication_id')}),
            'emotion_values': sorted({lab.get('emotion') for lab in label_rows}),
            'target_type_values': sorted({lab.get('target_type') for lab in label_rows}),
            'event_name_count': len({lab.get('event_name') for lab in label_rows}),
            'schema_id_count': len({lab.get('schema_id') for lab in label_rows}),
        },
        'distributions': {
            'label_per_article': dict(Counter(len(r.get('labels', [])) for r in rows)),
            'emotion': dict(Counter(lab.get('emotion') for lab in label_rows)),
            'target_type': dict(Counter(lab.get('target_type') for lab in label_rows)),
            'top_event_name': dict(Counter(lab.get('event_name') for lab in label_rows).most_common(30)),
        },
        'impact_analysis_type': dict(Counter(type(lab.get('impact_analysis')).__name__ for lab in label_rows)),
        'impact_analysis_semantics': 'free_text_rationale',
        'supervision_assessment': {
            'impact_direction': 'usable_from_labels.emotion',
            'impact_target': 'usable_from_labels.target_type_target_code_target_name',
            'impact_severity': 'unresolved_no_verified_ordinal_label',
            'alert_priority': 'not_available',
        },
    }


def audit_task3_interface() -> dict[str, Any]:
    iface_rows = read_jsonl(RELEASE_DIR / 'task4_event_interface.jsonl')
    release_nodes = read_jsonl(RELEASE_DIR / 'graph_nodes.jsonl')
    release_edges = read_jsonl(RELEASE_DIR / 'graph_edges.jsonl')
    node_types = Counter(row['node_type'] for row in release_nodes)
    edge_types = Counter(row['relation_type'] for row in release_edges)
    return {
        'task4_interface_rows': len(iface_rows),
        'unique_canonical_event_ids': len({r['canonical_event_id'] for r in iface_rows}),
        'unique_event_thread_ids': len({r['event_thread_id'] for r in iface_rows}),
        'current_state_distribution': dict(Counter(r['current_state_type'] for r in iface_rows)),
        'trust_score_non_null_rate': round(sum(1 for r in iface_rows if r.get('trust_score') is not None) / len(iface_rows), 6),
        'canonical_event_node_count': node_types['CanonicalEvent'],
        'eventmention_node_count': node_types['EventMention'],
        'eventthread_node_count': node_types['EventThread'],
        'evidenceunit_node_count': node_types['EvidenceUnit'],
        'stateobservation_node_count': node_types['StateObservation'],
        'claim_node_count': node_types['Claim'],
        'evidencesentence_node_count': node_types['EvidenceSentence'],
        'company_node_count': node_types['Company'],
        'industry_node_count': node_types['Industry'],
        'document_node_count': node_types['Document'],
        'release_direct_transition_counts': {
            'ANNOUNCEMENT_TO_PROGRESS': edge_types['ANNOUNCEMENT_TO_PROGRESS'],
            'PROGRESS_TO_COMPLETION': edge_types['PROGRESS_TO_COMPLETION'],
            'CLARIFIES_OR_REFUTES': edge_types['CLARIFIES_OR_REFUTES'],
            'RUMOR_TO_CONFIRMATION': edge_types['RUMOR_TO_CONFIRMATION'],
        },
        'review_or_uncertain_in_release_layer': False,
        'real_propagation_relation_types': [
            rel for rel in ['TARGETS_COMPANY', 'TARGETS_INDUSTRY', 'MEMBER_OF_THREAD', 'ANNOUNCEMENT_TO_PROGRESS', 'PROGRESS_TO_COMPLETION', 'CLARIFIES_OR_REFUTES', 'RUMOR_TO_CONFIRMATION'] if edge_types[rel] > 0
        ],
        'nonexistent_propagation_relation_types': [
            rel for rel in ['SUPPLY_CHAIN', 'UPSTREAM_OF', 'DOWNSTREAM_OF', 'OWNS_EQUITY_IN', 'SHAREHOLDING_LINK'] if edge_types[rel] == 0
        ],
        'interface_hashes': {
            'task3_release_task4_interface': sha256_file(RELEASE_DIR / 'task4_event_interface.jsonl'),
            'task3_final_task4_interface': sha256_file(FINAL_INTERFACE),
            'task3_release_graph_nodes': sha256_file(RELEASE_DIR / 'graph_nodes.jsonl'),
            'task3_release_graph_edges': sha256_file(RELEASE_DIR / 'graph_edges.jsonl'),
        },
    }


def build_split_audit(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    release_nodes = read_jsonl(RELEASE_DIR / 'graph_nodes.jsonl')
    release_edges = read_jsonl(RELEASE_DIR / 'graph_edges.jsonl')
    article_to_doc = {r.get('article_id'): (r.get('stable_id') or r.get('document_id')) for r in release_nodes if r['node_type'] == 'Document' and r.get('article_id')}
    doc_to_mentions = defaultdict(set)
    mention_to_ce: dict[str, str] = {}
    doc_to_eu = defaultdict(set)
    ce_to_thread: dict[str, str] = {}
    for edge in release_edges:
        if edge['relation_type'] == 'CONTAINS_MENTION':
            doc_to_mentions[edge['source_id']].add(edge['target_id'])
        elif edge['relation_type'] == 'INSTANCE_OF_EVENT':
            mention_to_ce[edge['source_id']] = edge['target_id']
        elif edge['relation_type'] == 'BELONGS_TO_EVIDENCE_UNIT':
            doc_to_eu[edge['source_id']].add(edge['target_id'])
        elif edge['relation_type'] == 'MEMBER_OF_THREAD':
            ce_to_thread[edge['source_id']] = edge['target_id']
    records = []
    for split_name, rows in [('train', train_rows), ('test', test_rows)]:
        for row in rows:
            doc_id = article_to_doc.get(row['article_id'])
            mentions = sorted(doc_to_mentions.get(doc_id, set()))
            ce_ids = sorted({mention_to_ce[m] for m in mentions if m in mention_to_ce})
            thread_ids = sorted({ce_to_thread[c] for c in ce_ids if c in ce_to_thread})
            evidence_unit_ids = sorted(doc_to_eu.get(doc_id, set()))
            records.append({
                'article_id': row['article_id'],
                'official_split': split_name,
                'publish_time': row.get('publish_time'),
                'duplication_id': row.get('duplication_id'),
                'clean_content_hash': clean_hash(row.get('content', '')),
                'document_node_id': doc_id,
                'canonical_event_ids': ce_ids,
                'event_thread_ids': thread_ids,
                'evidence_unit_ids': evidence_unit_ids,
                'safe_split_violation': not bool(doc_id),
            })
    df = pd.DataFrame(records)
    ce_cross = sum(len(set(v)) > 1 for v in defaultdict(set, {k: set(g['official_split']) for k, g in df.explode('canonical_event_ids').dropna(subset=['canonical_event_ids']).groupby('canonical_event_ids')}).values())
    thread_cross = sum(len(set(v)) > 1 for v in defaultdict(set, {k: set(g['official_split']) for k, g in df.explode('event_thread_ids').dropna(subset=['event_thread_ids']).groupby('event_thread_ids')}).values())
    eu_cross = sum(len(set(v)) > 1 for v in defaultdict(set, {k: set(g['official_split']) for k, g in df.explode('evidence_unit_ids').dropna(subset=['evidence_unit_ids']).groupby('evidence_unit_ids')}).values())
    hash_cross = df.groupby('clean_content_hash')['official_split'].nunique().gt(1).sum()
    dup_cross = df.dropna(subset=['duplication_id']).groupby('duplication_id')['official_split'].nunique().gt(1).sum()
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs/task4/splits').mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / 'outputs/task4/splits/group_time_split_suggestion.parquet', index=False)
    return {
        'record_count': len(records),
        'document_node_mapped_count': int(df['document_node_id'].notna().sum()),
        'document_node_mapping_rate': round(float(df['document_node_id'].notna().mean()), 6),
        'canonical_event_cross_split_count': int(ce_cross),
        'event_thread_cross_split_count': int(thread_cross),
        'evidence_unit_cross_split_count': int(eu_cross),
        'clean_content_hash_cross_split_count': int(hash_cross),
        'duplication_id_cross_split_count': int(dup_cross),
        'safe_split_violation_count': int(df['safe_split_violation'].sum()),
        'official_split_counts': dict(df['official_split'].value_counts()),
        'time_order_is_train_before_test': bool(df[df['official_split'] == 'train']['publish_time'].max() <= df[df['official_split'] == 'test']['publish_time'].max()),
    }


def main() -> None:
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(ROOT / 'data/processed/labeled/news_train.jsonl')
    test_rows = read_jsonl(ROOT / 'data/processed/labeled/news_test.jsonl')
    data_audit = {
        'train': audit_labels(train_rows),
        'test': audit_labels(test_rows),
        'global': {
            'total_rows': len(train_rows) + len(test_rows),
            'formal_human_gold': 'not_available',
            'impact_analysis_global_type': 'free_text_rationale',
            'resolved_supervised_targets': ['impact_direction', 'impact_target'],
            'unresolved_targets': ['impact_severity', 'alert_priority'],
        },
    }
    interface_audit = audit_task3_interface()
    split_audit = build_split_audit(train_rows, test_rows)
    write_json(REPORT_DATA, data_audit)
    write_json(REPORT_INTERFACE, interface_audit)
    write_json(REPORT_SPLIT, split_audit)
    DOC_DATA.write_text('# Task4 Stage1 Data and Protocol\n\nSee machine-readable reports in reports/.\n', encoding='utf-8')
    DOC_ARCH.write_text('# TRACE-Alert Architecture\n\nStage1 only audits frozen Task3 interfaces and builds null-scored EventImpactCard contracts.\n', encoding='utf-8')
    print(json.dumps(normalize({'data_audit': data_audit, 'interface_audit': interface_audit, 'split_audit': split_audit}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
