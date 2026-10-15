#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/opt/data/private/JinRong/kechuang/financial_event_project')
RELEASE_DIR = ROOT / 'outputs/task3_kg_v3_release'
STAGE1_DIR = ROOT / 'outputs/task4'
OUT_DIR = ROOT / 'outputs/task4/stage2'
REPORT_RECON = ROOT / 'reports/task4_stage2_reconciliation.json'
REPORT_BRIDGE = ROOT / 'reports/task4_stage2_bridge.json'
REPORT_RATIONALE = ROOT / 'reports/task4_stage2_impact_rationale_audit.json'
DOC_BRIDGE = ROOT / 'docs/task4_stage2_bridge.md'


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


def clean_hash(text: str) -> str:
    return hashlib.sha256(' '.join((text or '').split()).encode('utf-8')).hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def sample_id(article_id: str, schema_id: str, target_type: str, target_code: str, event_name: str) -> str:
    return 'T4S_' + hashlib.sha1('||'.join([article_id, schema_id, target_type, target_code, event_name]).encode('utf-8')).hexdigest()[:24]


def token_count(text: str) -> int:
    return len(re.findall(r'\w+|[\u4e00-\u9fff]', text or ''))


def load_articles() -> list[dict[str, Any]]:
    rows = []
    for split, rel in [('train', 'data/processed/labeled/news_train.jsonl'), ('test', 'data/processed/labeled/news_test.jsonl')]:
        for line in (ROOT / rel).open(encoding='utf-8'):
            if not line.strip():
                continue
            row = json.loads(line)
            row['official_split'] = split
            rows.append(row)
    return rows


def build_indices() -> dict[str, Any]:
    fulltext_ids = set(pd.read_parquet(ROOT / 'data/processed/event_credibility/fulltext_article_index.parquet', columns=['article_id'])['article_id'].tolist())
    metadata_rows = read_jsonl(ROOT / 'data/processed/event_clusters/article_metadata_index.jsonl')
    metadata_ids = {row['article_id'] for row in metadata_rows}
    final_event_rows = read_jsonl(ROOT / 'data/processed/event_clusters/all_final_events_with_sources.jsonl')
    final_event_by_article = defaultdict(list)
    for row in final_event_rows:
        final_event_by_article[row['article_id']].append(row)
    norm_mentions = read_jsonl(ROOT / 'outputs/task3_fulltext_v2/normalized_event_mentions.jsonl')
    norm_mentions_by_article = defaultdict(list)
    for row in norm_mentions:
        norm_mentions_by_article[row['article_id']].append(row)
    release_nodes = read_jsonl(RELEASE_DIR / 'graph_nodes.jsonl')
    release_edges = read_jsonl(RELEASE_DIR / 'graph_edges.jsonl')
    article_to_doc = {}
    release_doc_row_by_article = {}
    node_by_id = {}
    for row in release_nodes:
        key = row.get('stable_id') or row.get('canonical_event_id') or row.get('event_thread_id') or row.get('state_observation_id')
        if key:
            node_by_id[key] = row
        if row.get('node_type') == 'Document' and str(row.get('article_id')) != 'nan':
            article_to_doc[row['article_id']] = row['stable_id']
            release_doc_row_by_article[row['article_id']] = row
    edges_by_source = defaultdict(list)
    edges_by_target = defaultdict(list)
    doc_to_mentions = defaultdict(list)
    mention_to_canonical = {}
    ce_to_thread = {}
    ce_to_state = defaultdict(list)
    ce_to_claims = defaultdict(list)
    claim_to_sentences = defaultdict(list)
    ce_to_evidence_units = defaultdict(set)
    mention_to_targets = defaultdict(list)
    for edge in release_edges:
        edges_by_source[edge['source_id']].append(edge)
        edges_by_target[edge['target_id']].append(edge)
        rel = edge['relation_type']
        if rel == 'CONTAINS_MENTION':
            doc_to_mentions[edge['source_id']].append(edge['target_id'])
        elif rel == 'INSTANCE_OF_EVENT':
            mention_to_canonical[edge['source_id']] = edge['target_id']
        elif rel == 'MEMBER_OF_THREAD':
            ce_to_thread[edge['source_id']] = edge['target_id']
        elif rel == 'HAS_STATE_OBSERVATION':
            ce_to_state[edge['source_id']].append(edge['target_id'])
        elif rel == 'SUPPORTS':
            ce_to_claims[edge['target_id']].append(edge['source_id'])
        elif rel == 'SUPPORTED_BY_SENTENCE':
            claim_to_sentences[edge['source_id']].append(edge['target_id'])
        elif rel == 'EVIDENCE_FOR':
            ce_to_evidence_units[edge['target_id']].add(edge['source_id'])
        elif rel in {'TARGETS_COMPANY', 'TARGETS_INDUSTRY'}:
            mention_to_targets[edge['source_id']].append({'relation_type': rel, 'target_id': edge['target_id']})
    return {
        'fulltext_ids': fulltext_ids,
        'metadata_ids': metadata_ids,
        'metadata_rows': metadata_rows,
        'final_event_by_article': final_event_by_article,
        'norm_mentions_by_article': norm_mentions_by_article,
        'release_nodes': release_nodes,
        'release_edges': release_edges,
        'release_doc_row_by_article': release_doc_row_by_article,
        'article_to_doc': article_to_doc,
        'edges_by_source': edges_by_source,
        'edges_by_target': edges_by_target,
        'doc_to_mentions': doc_to_mentions,
        'mention_to_canonical': mention_to_canonical,
        'ce_to_thread': ce_to_thread,
        'ce_to_state': ce_to_state,
        'ce_to_claims': ce_to_claims,
        'claim_to_sentences': claim_to_sentences,
        'ce_to_evidence_units': ce_to_evidence_units,
        'mention_to_targets': mention_to_targets,
        'node_by_id': node_by_id,
    }


def reconcile_articles(rows: list[dict[str, Any]], idx: dict[str, Any]) -> pd.DataFrame:
    reconciliation_rows = []
    for row in rows:
        aid = row['article_id']
        in_fulltext = aid in idx['fulltext_ids']
        in_metadata = aid in idx['metadata_ids']
        has_task1_mention = aid in idx['norm_mentions_by_article']
        in_task1_final = aid in idx['final_event_by_article']
        task3_doc_id = idx['article_to_doc'].get(aid)
        exact_match = task3_doc_id is not None
        if exact_match:
            status = 'exact_task3_document_match'
        elif in_task1_final:
            status = 'task1_event_but_not_materialized'
        elif in_metadata:
            status = 'official_document_without_task1_event'
        elif in_fulltext:
            status = 'fulltext_only_auxiliary'
        else:
            status = 'unresolved'
        reconciliation_rows.append({
            'article_id': aid,
            'title': row['title'],
            'source': row['source'],
            'publish_time': row['publish_time'],
            'label_count': len(row['labels']),
            'in_fulltext_universe': in_fulltext,
            'in_official_metadata_universe': in_metadata,
            'has_task1_eventmention': has_task1_mention,
            'has_task1_final_event': in_task1_final,
            'in_task3_document_node': exact_match,
            'article_id_exact_match': exact_match,
            'task3_document_id': task3_doc_id,
            'id_reconciliation_required': False,
            'mapping_status': status,
            'field_provenance': {
                'fulltext_universe': 'data/processed/event_credibility/fulltext_article_index.parquet',
                'official_metadata_universe': 'data/processed/event_clusters/article_metadata_index.jsonl',
                'task1_eventmention': 'outputs/task3_fulltext_v2/normalized_event_mentions.jsonl',
                'task3_document': 'outputs/task3_kg_v3_release/graph_nodes.jsonl',
            },
        })
    return pd.DataFrame(reconciliation_rows)


def explode_samples(rows: list[dict[str, Any]]) -> pd.DataFrame:
    sample_rows = []
    for row in rows:
        for label_index, label in enumerate(row['labels']):
            sample_rows.append({
                'task4_sample_id': sample_id(row['article_id'], label['schema_id'], label['target_type'], label['target_code'], label['event_name']),
                'article_id': row['article_id'],
                'title': row['title'],
                'content': row['content'],
                'publish_time': row['publish_time'],
                'source': row['source'],
                'schema_id': label['schema_id'],
                'target_type': label['target_type'],
                'target_code': label['target_code'],
                'target_name': label['target_name'],
                'event_name': label['event_name'],
                'emotion_label': label['emotion'],
                'impact_rationale_reference': label.get('impact_analysis'),
                'impact_analysis_reference': label.get('impact_analysis'),
                'label_index': label_index,
                'official_split': row['official_split'],
                'duplication_id': row.get('duplication_id'),
                'clean_content_hash': clean_hash(row.get('content', '')),
                'field_provenance': {
                    'article_fields': f"data/processed/labeled/news_{row['official_split']}.jsonl",
                    'label_fields': 'labels[*]',
                },
            })
    return pd.DataFrame(sample_rows)


def build_bridges(samples: pd.DataFrame, reconciliation: pd.DataFrame, idx: dict[str, Any]) -> pd.DataFrame:
    rec_map = {row['article_id']: row for row in reconciliation.to_dict('records')}
    bridge_rows = []
    for sample in samples.to_dict('records'):
        rec = rec_map[sample['article_id']]
        doc_id = rec['task3_document_id']
        mentions = idx['doc_to_mentions'].get(doc_id, []) if doc_id else []
        matched_mentions = []
        for mention_id in mentions:
            mention_node = idx['node_by_id'].get(mention_id, {})
            mention_schema = mention_node.get('provisional_event_type') or mention_node.get('schema_id')
            if mention_schema == sample['schema_id']:
                matched_mentions.append(mention_id)
        if matched_mentions:
            mention_id = matched_mentions[0]
            ce_id = idx['mention_to_canonical'].get(mention_id)
            thread_id = idx['ce_to_thread'].get(ce_id)
            state_ids = idx['ce_to_state'].get(ce_id, [])
            claim_ids = idx['ce_to_claims'].get(ce_id, [])
            sentence_ids = sorted({sid for cid in claim_ids for sid in idx['claim_to_sentences'].get(cid, [])})
            eu_ids = sorted(idx['ce_to_evidence_units'].get(ce_id, set()))
            bridge_rows.append({
                'task4_sample_id': sample['task4_sample_id'],
                'article_id': sample['article_id'],
                'task4_sample_type': 'Task4Sample',
                'document_id': doc_id,
                'mapping_status': rec['mapping_status'],
                'canonical_event_id': ce_id,
                'event_mention_id': mention_id,
                'event_thread_id': thread_id,
                'state_observation_ids': state_ids,
                'evidence_unit_ids': eu_ids,
                'evidence_sentence_ids': sentence_ids,
                'task2_assessment_ids': [e['target_id'] for e in idx['edges_by_source'].get(doc_id, []) if e['relation_type'] == 'ASSESSED_BY'],
                'mapping_type': 'core_kg_exact_document_and_schema_match',
                'mapping_confidence': 1.0,
                'verification_status': 'observed',
                'evidence_path': ['Task4Sample', 'Document', 'EventMention', 'CanonicalEvent', 'EventThread', 'StateObservation', 'EvidenceUnit', 'EvidenceSentence', 'Task2Assessment'],
                'graph_feature_availability': {
                    'document': True, 'event_mention': True, 'canonical_event': ce_id is not None,
                    'event_thread': thread_id is not None, 'state_observation': bool(state_ids),
                    'evidence_unit': bool(eu_ids), 'evidence_sentence': bool(sentence_ids), 'task2_assessment': True,
                },
            })
        else:
            bridge_rows.append({
                'task4_sample_id': sample['task4_sample_id'],
                'article_id': sample['article_id'],
                'task4_sample_type': 'Task4Sample',
                'document_id': doc_id,
                'mapping_status': rec['mapping_status'],
                'canonical_event_id': None,
                'event_mention_id': None,
                'event_thread_id': None,
                'state_observation_ids': [],
                'evidence_unit_ids': [],
                'evidence_sentence_ids': [],
                'task2_assessment_ids': [],
                'mapping_type': 'shadow_observation_required',
                'mapping_confidence': 0.0,
                'verification_status': 'provisional_bridge_candidate',
                'evidence_path': ['Task4Sample', 'Document'],
                'graph_feature_availability': {
                    'document': doc_id is not None, 'event_mention': False, 'canonical_event': False,
                    'event_thread': False, 'state_observation': False,
                    'evidence_unit': False, 'evidence_sentence': False, 'task2_assessment': False,
                },
            })
    return pd.DataFrame(bridge_rows)


def build_rationale_audit(samples: pd.DataFrame) -> dict[str, Any]:
    rationale_rows = [r for r in samples.to_dict('records') if isinstance(r.get('impact_rationale_reference'), str) and r.get('impact_rationale_reference', '').strip()]
    lengths = [len(r['impact_rationale_reference']) for r in rationale_rows]
    token_lengths = [token_count(r['impact_rationale_reference']) for r in rationale_rows]
    contains_target = sum(any(tok in (r['impact_rationale_reference'] or '') for tok in [str(r['target_name']), str(r['target_code'])]) for r in rationale_rows)
    contains_direction = sum(any(tok in (r['impact_rationale_reference'] or '') for tok in ['积极', '正面', '负面', '中性', '利好', '利空']) for r in rationale_rows)
    contains_causal = sum(any(tok in (r['impact_rationale_reference'] or '') for tok in ['因为', '由于', '因此', '带动', '导致', '有助于']) for r in rationale_rows)
    text_copy_like = sum(((r['impact_rationale_reference'] or '') in (r['content'] or '')) or ((r['content'] or '')[:60] in (r['impact_rationale_reference'] or '')) for r in rationale_rows)
    by_split = defaultdict(set)
    by_split_emotion = defaultdict(list)
    for r in rationale_rows:
        by_split[r['official_split']].add(clean_hash(r['impact_rationale_reference']))
        by_split_emotion[r['emotion_label']].append(r['impact_rationale_reference'])
    train_test_repeat = len(by_split['train'] & by_split['test'])
    return {
        'non_empty_rationale_count': len(rationale_rows),
        'empty_rationale_count': int(samples['impact_rationale_reference'].isna().sum() + (samples['impact_rationale_reference'].fillna('').eq('')).sum() - samples['impact_rationale_reference'].isna().sum()),
        'character_length_summary': {
            'min': min(lengths) if lengths else 0,
            'median': float(pd.Series(lengths).median()) if lengths else 0.0,
            'p95': float(pd.Series(lengths).quantile(0.95)) if lengths else 0.0,
            'max': max(lengths) if lengths else 0,
        },
        'token_length_summary': {
            'min': min(token_lengths) if token_lengths else 0,
            'median': float(pd.Series(token_lengths).median()) if token_lengths else 0.0,
            'p95': float(pd.Series(token_lengths).quantile(0.95)) if token_lengths else 0.0,
            'max': max(token_lengths) if token_lengths else 0,
        },
        'contains_target_rate': round(contains_target / len(rationale_rows), 6) if rationale_rows else 0.0,
        'contains_direction_rate': round(contains_direction / len(rationale_rows), 6) if rationale_rows else 0.0,
        'contains_causal_rate': round(contains_causal / len(rationale_rows), 6) if rationale_rows else 0.0,
        'copy_like_rate': round(text_copy_like / len(rationale_rows), 6) if rationale_rows else 0.0,
        'train_test_template_overlap_count': train_test_repeat,
        'emotion_language_distribution': {k: len(v) for k, v in by_split_emotion.items()},
        'leakage_policy': 'impact_rationale_reference is audit-only and must not be used as model input for emotion prediction',
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_articles()
    idx = build_indices()
    reconciliation = reconcile_articles(rows, idx)
    samples = explode_samples(rows)
    bridges = build_bridges(samples, reconciliation, idx)
    rationale_audit = build_rationale_audit(samples)

    reconciliation.to_parquet(OUT_DIR / 'task4_document_reconciliation.parquet', index=False)
    samples.to_parquet(OUT_DIR / 'task4_event_level_samples.parquet', index=False)
    bridges.to_parquet(OUT_DIR / 'task4_kg_bridge.parquet', index=False)
    unresolved = bridges[bridges['mapping_type'] == 'shadow_observation_required'][['task4_sample_id', 'article_id', 'mapping_status']].to_dict('records')
    with (OUT_DIR / 'unresolved_mapping_queue.jsonl').open('w', encoding='utf-8') as f:
        for row in unresolved:
            f.write(json.dumps(normalize(row), ensure_ascii=False) + '\n')

    recon_report = {
        'task4_article_count': int(len(reconciliation)),
        'task4_label_count': int(len(samples)),
        'mapping_status_counts': dict(reconciliation['mapping_status'].value_counts()),
        'fulltext_coverage_count': int(reconciliation['in_fulltext_universe'].sum()),
        'metadata_coverage_count': int(reconciliation['in_official_metadata_universe'].sum()),
        'task1_eventmention_coverage_count': int(reconciliation['has_task1_eventmention'].sum()),
        'task1_final_event_coverage_count': int(reconciliation['has_task1_final_event'].sum()),
        'task3_document_coverage_count': int(reconciliation['in_task3_document_node'].sum()),
        'low_mapping_root_cause': '3075篇中仅265篇进入Task3发布图，主要原因是Task3只物化事件文档，且当前任务四文章大多不在307888 official metadata universe，也未进入冻结Task1事件输出。',
    }
    bridge_report = {
        'task4_sample_count': int(len(samples)),
        'core_kg_bridge_count': int((bridges['mapping_type'] == 'core_kg_exact_document_and_schema_match').sum()),
        'shadow_required_count': int((bridges['mapping_type'] == 'shadow_observation_required').sum()),
        'core_bridge_article_count': int(bridges[bridges['mapping_type'] == 'core_kg_exact_document_and_schema_match']['article_id'].nunique()),
        'shadow_article_count': int(bridges[bridges['mapping_type'] == 'shadow_observation_required']['article_id'].nunique()),
        'mapping_confidence_summary': {
            'core_mean': float(bridges[bridges['mapping_type'] == 'core_kg_exact_document_and_schema_match']['mapping_confidence'].mean()) if (bridges['mapping_type'] == 'core_kg_exact_document_and_schema_match').any() else 0.0,
            'shadow_mean': float(bridges[bridges['mapping_type'] == 'shadow_observation_required']['mapping_confidence'].mean()) if (bridges['mapping_type'] == 'shadow_observation_required').any() else 0.0,
        },
    }
    write_json(REPORT_RECON, recon_report)
    write_json(REPORT_BRIDGE, bridge_report)
    write_json(REPORT_RATIONALE, rationale_audit)
    DOC_BRIDGE.write_text('# Task4 Stage2 Bridge\n\nSee machine-readable artifacts in outputs/task4/stage2 and reports/.\n', encoding='utf-8')
    print(json.dumps(normalize({'reconciliation': recon_report, 'bridge': bridge_report, 'rationale': rationale_audit}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
