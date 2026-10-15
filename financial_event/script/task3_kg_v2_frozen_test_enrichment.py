#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'outputs/task3_kg_v2/step4_revision2_acceptance/frozen_test_package.xlsx'
OUT_XLSX = ROOT / 'outputs/task3_kg_v2/step4_revision3/frozen_test_package_enriched.xlsx'
OUT_REPORT = ROOT / 'reports/task3_kg_v2_frozen_test_enrichment_report.json'
OUT_UNRESOLVED = ROOT / 'outputs/task3_kg_v2/step4_revision3/frozen_test_unresolved_queue.jsonl'
CARDS = ROOT / 'outputs/task3_kg_v2/step4_revision3/event_cards_revision3.jsonl'
IDENTITY = ROOT / 'outputs/task3_kg_v2/step4_revision3/event_identity_keys.jsonl'
IDENTITY_FIELDS = ROOT / 'outputs/task3_kg_v2/step4_revision3/identity_field_evidence.jsonl'
MENTIONS = ROOT / 'outputs/task3_fulltext_v2/normalized_event_mentions.jsonl'
EVIDENCE = ROOT / 'outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl'
DOCS = ROOT / 'outputs/task3_fulltext_v2/normalized_documents.jsonl'
EU_MAP = ROOT / 'outputs/task3_kg_v2/step3/document_evidence_unit_mapping.jsonl'
HUMAN_FIELDS = ['human_same_event', 'human_update_relation', 'human_confidence', 'human_evidence', 'human_note', 'annotator_id']


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_blank(x: Any) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and pd.isna(x):
        return ''
    return str(x)


def topk_sentences(rows: list[dict[str, Any]], k: int = 3) -> list[str]:
    ordered = sorted(rows, key=lambda r: (int(r.get('rank', 9999)), -float(r.get('score', 0.0))))
    out = []
    for r in ordered[:k]:
        s = normalize_blank(r.get('sentence')).strip()
        if s:
            out.append(s)
    return out


def ensure_at_least_one_evidence(title: str, evidence: list[str], local_text: str) -> list[str]:
    if evidence:
        return evidence
    if local_text.strip():
        return [local_text.strip()]
    if title.strip():
        return [title.strip()]
    return []


def build_indexes() -> dict[str, Any]:
    cards = {(r['event_mention_id'], r['document_id']): r for r in read_jsonl(CARDS)}
    identity = {(r['event_mention_id']): r for r in read_jsonl(IDENTITY)}
    mentions = {(r['event_mention_id'], r['document_id']): r for r in read_jsonl(MENTIONS)}
    docs = {(r['document_id']): r for r in read_jsonl(DOCS)}
    ev_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in read_jsonl(EVIDENCE):
        ev_map[(r['event_mention_id'], r['document_id'])].append(r)
    id_fields: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in read_jsonl(IDENTITY_FIELDS):
        id_fields[r['event_mention_id']].append(r)
    eu_map: dict[str, list[str]] = defaultdict(list)
    for r in read_jsonl(EU_MAP):
        doc_id = normalize_blank(r.get('document_id'))
        eu = normalize_blank(r.get('evidence_unit_id'))
        if doc_id and eu:
            eu_map[doc_id].append(eu)
    return {
        'cards': cards,
        'identity': identity,
        'mentions': mentions,
        'docs': docs,
        'ev_map': ev_map,
        'id_fields': id_fields,
        'eu_map': {k: sorted(dict.fromkeys(v)) for k, v in eu_map.items()},
    }


def enrich_side(row: pd.Series, side: str, indexes: dict[str, Any], unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    emid = normalize_blank(row[f'event_mention_id_{side}'])
    docid = normalize_blank(row[f'document_id_{side}'])
    key = (emid, docid)
    card = indexes['cards'].get(key)
    mention = indexes['mentions'].get(key)
    doc = indexes['docs'].get(docid)
    identity = indexes['identity'].get(emid)
    field_rows = indexes['id_fields'].get(emid, [])
    ev_rows = indexes['ev_map'].get(key, [])
    eu_ids = indexes['eu_map'].get(docid, [])
    if not card or not mention or not doc:
        unresolved.append({'pair_id': normalize_blank(row['pair_id']), 'side': side, 'event_mention_id': emid, 'document_id': docid, 'missing_card': card is None, 'missing_mention': mention is None, 'missing_document': doc is None})
    title = normalize_blank((doc or {}).get('title'))
    event_local_text = normalize_blank((mention or card or {}).get('event_local_text'))
    evidence = topk_sentences(ev_rows, 3)
    evidence = ensure_at_least_one_evidence(title, evidence, event_local_text)
    identity_fields = [normalize_blank(x.get('field_value')) for x in field_rows if normalize_blank(x.get('field_value'))]
    identity_field_evidence = [
        {
            'field_name': normalize_blank(x.get('field_name')),
            'field_value': normalize_blank(x.get('field_value')),
            'evidence_text': normalize_blank(x.get('evidence_text')),
            'source': normalize_blank(x.get('source')),
            'confidence': x.get('confidence'),
        }
        for x in field_rows
        if normalize_blank(x.get('field_value'))
    ]
    return {
        f'title_{side}': title,
        f'event_local_text_{side}': event_local_text,
        f'evidence_sentences_{side}': ' | '.join(evidence),
        f'provisional_event_type_{side}': normalize_blank((mention or card or {}).get('provisional_event_type')),
        f'target_type_{side}': normalize_blank((mention or card or {}).get('target_type')),
        f'target_code_{side}': normalize_blank((mention or card or {}).get('target_code')),
        f'publish_time_{side}': normalize_blank((mention or card or {}).get('publish_time')),
        f'source_name_{side}': normalize_blank((mention or {}).get('source_name')),
        f'event_identity_key_{side}': normalize_blank((identity or card or {}).get('event_identity_key')),
        f'identity_fields_{side}': json.dumps(identity_fields, ensure_ascii=False),
        f'identity_field_evidence_{side}': json.dumps(identity_field_evidence, ensure_ascii=False),
        f'evidence_unit_id_{side}': '|'.join(eu_ids),
    }


def build_outputs() -> dict[str, Any]:
    started = time.time()
    before_sha = sha256_file(FROZEN)
    frozen = pd.read_excel(FROZEN)
    original_pair_ids = frozen['pair_id'].tolist()
    original_order = list(original_pair_ids)
    original_strata = frozen['sample_stratum'].tolist()
    indexes = build_indexes()
    unresolved: list[dict[str, Any]] = []

    enriched_rows = []
    for _, row in frozen.iterrows():
        record = row.to_dict()
        record.update(enrich_side(row, 'a', indexes, unresolved))
        record.update(enrich_side(row, 'b', indexes, unresolved))
        for hf in HUMAN_FIELDS:
            record[hf] = ''
        record['review_status'] = ''
        enriched_rows.append(record)
    enriched = pd.DataFrame(enriched_rows)

    assert enriched['pair_id'].tolist() == original_order
    assert set(enriched['pair_id']) == set(original_pair_ids)
    assert enriched['sample_stratum'].tolist() == original_strata
    assert all((enriched[hf].fillna('').astype(str).str.strip() == '').all() for hf in HUMAN_FIELDS)
    assert (enriched['review_status'].fillna('').astype(str).str.strip() == '').all()

    blind_cols = [
        'pair_id',
        'event_mention_id_a', 'document_id_a', 'title_a', 'event_local_text_a', 'evidence_sentences_a',
        'provisional_event_type_a', 'target_type_a', 'target_code_a', 'publish_time_a', 'source_name_a',
        'event_identity_key_a', 'identity_fields_a', 'identity_field_evidence_a', 'evidence_unit_id_a',
        'event_mention_id_b', 'document_id_b', 'title_b', 'event_local_text_b', 'evidence_sentences_b',
        'provisional_event_type_b', 'target_type_b', 'target_code_b', 'publish_time_b', 'source_name_b',
        'event_identity_key_b', 'identity_fields_b', 'identity_field_evidence_b', 'evidence_unit_id_b',
        'human_same_event', 'human_update_relation', 'human_confidence', 'human_evidence', 'human_note', 'annotator_id', 'review_status'
    ]
    blind = enriched[blind_cols].copy()

    forbidden_blind_cols = {'same_event_score', 'different_event_score', 'same_component', 'candidate_reason', 'sample_stratum', 'semantic_score', 'structure_score', 'graph_score', 'fusion_score_before_rerank'}
    assert not (forbidden_blind_cols & set(blind.columns))

    for side in ['a', 'b']:
        assert (blind[f'title_{side}'].fillna('').astype(str).str.strip() != '').all()
        assert (blind[f'evidence_sentences_{side}'].fillna('').astype(str).str.strip() != '').all()

    audit = enriched.copy()
    audit['pair_order'] = range(len(audit))

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX) as writer:
        blind.to_excel(writer, sheet_name='blind_annotation', index=False)
        audit.to_excel(writer, sheet_name='audit_metadata', index=False)

    after_sha = sha256_file(OUT_XLSX)
    unresolved_unique = len({(r['pair_id'], r['side']) for r in unresolved})
    write_jsonl(OUT_UNRESOLVED, unresolved)
    report = {
        'version': 'task3_kg_v2_frozen_test_enrichment_v0',
        'no_phase5_entry': True,
        'read_only_revision3_outputs': True,
        'frozen_input': str(FROZEN.relative_to(ROOT)),
        'frozen_output': str(OUT_XLSX.relative_to(ROOT)),
        'rows': int(len(enriched)),
        'pair_id_set_unchanged': enriched['pair_id'].tolist() == original_order,
        'sample_stratum_unchanged': enriched['sample_stratum'].tolist() == original_strata,
        'human_fields_empty': bool(all((enriched[hf].fillna('').astype(str).str.strip() == '').all() for hf in HUMAN_FIELDS)),
        'blind_annotation_no_model_fields': True,
        'min_title_and_evidence_per_side': True,
        'unresolved_pair_side_count': int(unresolved_unique),
        'unresolved_queue': str(OUT_UNRESOLVED.relative_to(ROOT)),
        'sha256': {
            'before_input_file': before_sha,
            'after_output_file': after_sha,
            'input_pair_id_hash': hashlib.sha256('\n'.join(original_order).encode('utf-8')).hexdigest(),
            'output_pair_id_hash': hashlib.sha256('\n'.join(enriched['pair_id'].tolist()).encode('utf-8')).hexdigest(),
        },
        'resource_usage': {
            'elapsed_seconds': round(time.time() - started, 3),
            'max_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        },
    }
    write_json(OUT_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    build_outputs()
