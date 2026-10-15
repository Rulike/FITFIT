#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/opt/data/private/JinRong/kechuang/financial_event_project')
RELEASE_DIR = ROOT / 'outputs/task3_kg_v3_release'
STAGE2_DIR = ROOT / 'outputs/task4/stage2'
OUT_SHADOW = STAGE2_DIR / 'task4_shadow_observations.jsonl'
OUT_EDGES = STAGE2_DIR / 'task4_shadow_edges.jsonl'


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    samples = pd.read_parquet(STAGE2_DIR / 'task4_event_level_samples.parquet')
    bridges = pd.read_parquet(STAGE2_DIR / 'task4_kg_bridge.parquet')
    release_nodes = read_jsonl(RELEASE_DIR / 'graph_nodes.jsonl')
    release_edges = read_jsonl(RELEASE_DIR / 'graph_edges.jsonl')
    company_lookup = {}
    industry_lookup = {}
    document_lookup = {}
    source_lookup = {}
    for node in release_nodes:
        if node.get('node_type') == 'Company':
            company_lookup[str(node.get('company_code') or node.get('target_code'))] = node['stable_id']
        elif node.get('node_type') == 'Industry':
            industry_lookup[str(node.get('industry_code') or node.get('target_code'))] = node['stable_id']
        elif node.get('node_type') == 'Document' and str(node.get('article_id')) != 'nan':
            document_lookup[node['article_id']] = node['stable_id']
        elif node.get('node_type') == 'Source':
            source_lookup[str(node.get('source_name'))] = node['stable_id']

    same_content_doc = defaultdict(list)
    exact_duplicate_doc = defaultdict(list)
    for edge in release_edges:
        if edge['relation_type'] == 'BELONGS_TO_EVIDENCE_UNIT':
            same_content_doc[edge['target_id']].append(edge['source_id'])
        elif edge['relation_type'] == 'EXACT_CONTENT_MATCH':
            exact_duplicate_doc[edge['source_id']].append(edge['target_id'])

    merged = samples.merge(bridges[['task4_sample_id', 'mapping_type', 'canonical_event_id']], on='task4_sample_id', how='left')
    shadow_rows = []
    shadow_edges = []
    for row in merged.to_dict('records'):
        if row['mapping_type'] != 'shadow_observation_required':
            continue
        obs_id = 'T4OBS_' + row['task4_sample_id'].split('_', 1)[1]
        shadow_rows.append({
            'task4_observation_id': obs_id,
            'node_type': 'Task4Observation',
            'article_id': row['article_id'],
            'task4_sample_id': row['task4_sample_id'],
            'title': row['title'],
            'publish_time': row['publish_time'],
            'source': row['source'],
            'schema_id': row['schema_id'],
            'target_type': row['target_type'],
            'target_code': row['target_code'],
            'target_name': row['target_name'],
            'event_name': row['event_name'],
            'verification_status': 'provisional_bridge_candidate',
            'confidence': 0.0,
            'evidence_scope': 'task4_labeled_article',
            'field_provenance': {'sample': 'outputs/task4/stage2/task4_event_level_samples.parquet'},
        })
        if row['article_id'] in document_lookup:
            shadow_edges.append({
                'source_id': obs_id,
                'target_id': document_lookup[row['article_id']],
                'relation_type': 'ALIGNED_WITH_DOCUMENT',
                'verification_status': 'provisional_bridge_candidate',
                'confidence': 1.0,
                'evidence_path': ['Task4Observation', 'Document'],
            })
        if row['target_type'] == 'company' and row['target_code'] in company_lookup:
            shadow_edges.append({
                'source_id': obs_id,
                'target_id': company_lookup[row['target_code']],
                'relation_type': 'TARGETS_COMPANY_CANDIDATE',
                'verification_status': 'provisional_bridge_candidate',
                'confidence': 1.0,
                'evidence_path': ['Task4Observation', 'Company'],
            })
        if row['target_type'] == 'industry' and row['target_code'] in industry_lookup:
            shadow_edges.append({
                'source_id': obs_id,
                'target_id': industry_lookup[row['target_code']],
                'relation_type': 'TARGETS_INDUSTRY_CANDIDATE',
                'verification_status': 'provisional_bridge_candidate',
                'confidence': 1.0,
                'evidence_path': ['Task4Observation', 'Industry'],
            })
        if row['source'] in source_lookup:
            shadow_edges.append({
                'source_id': obs_id,
                'target_id': source_lookup[row['source']],
                'relation_type': 'PUBLISHED_BY_SOURCE_CANDIDATE',
                'verification_status': 'provisional_bridge_candidate',
                'confidence': 1.0,
                'evidence_path': ['Task4Observation', 'Source'],
            })
    write_jsonl(OUT_SHADOW, shadow_rows)
    write_jsonl(OUT_EDGES, shadow_edges)
    print(json.dumps({'shadow_observations': len(shadow_rows), 'shadow_edges': len(shadow_edges)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
