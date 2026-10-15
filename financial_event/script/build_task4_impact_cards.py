#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/opt/data/private/JinRong/kechuang/financial_event_project')
RELEASE_DIR = ROOT / 'outputs/task3_kg_v3_release'
OUT_CARD = ROOT / 'outputs/task4/impact_cards/event_impact_cards.parquet'
OUT_UNRESOLVED = ROOT / 'outputs/task4/impact_cards/unresolved_queue.jsonl'
OUT_DEMO = ROOT / 'outputs/task4/demo/stage1_vertical_slice.json'
REPORT_ACCEPT = ROOT / 'reports/task4_stage1_acceptance.json'


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    iface_rows = read_jsonl(RELEASE_DIR / 'task4_event_interface.jsonl')
    nodes = read_jsonl(RELEASE_DIR / 'graph_nodes.jsonl')
    edges = read_jsonl(RELEASE_DIR / 'graph_edges.jsonl')
    node_by_id: dict[str, dict[str, Any]] = {}
    for row in nodes:
        key = row.get('stable_id') or row.get('canonical_event_id') or row.get('event_thread_id') or row.get('state_observation_id')
        if key:
            node_by_id[key] = row

    ce_to_mentions = defaultdict(set)
    ce_to_state = defaultdict(set)
    ce_to_claims = defaultdict(set)
    ce_to_threads = {}
    ce_to_direct_targets = defaultdict(list)
    ce_to_sentences = defaultdict(set)
    ce_to_paths = defaultdict(list)
    ce_to_companies = defaultdict(set)
    ce_to_industries = defaultdict(set)
    ce_to_claim_supports = defaultdict(set)
    mention_to_companies = defaultdict(set)
    mention_to_industries = defaultdict(set)
    claim_to_sentences = defaultdict(set)

    for edge in edges:
        rel = edge['relation_type']
        if rel == 'INSTANCE_OF_EVENT':
            ce_to_mentions[edge['target_id']].add(edge['source_id'])
        elif rel == 'HAS_STATE_OBSERVATION':
            ce_to_state[edge['source_id']].add(edge['target_id'])
        elif rel == 'SUPPORTS':
            ce_to_claims[edge['target_id']].add(edge['source_id'])
            ce_to_claim_supports[edge['target_id']].add(edge['source_id'])
        elif rel == 'MEMBER_OF_THREAD':
            ce_to_threads[edge['source_id']] = edge['target_id']
        elif rel in {'ANNOUNCEMENT_TO_PROGRESS', 'PROGRESS_TO_COMPLETION', 'CLARIFIES_OR_REFUTES', 'RUMOR_TO_CONFIRMATION'}:
            ce_to_direct_targets[edge['source_id']].append({'target_id': edge['target_id'], 'relation_type': rel, 'confidence': edge.get('confidence')})
        elif rel == 'TARGETS_COMPANY':
            mention_to_companies[edge['source_id']].add(edge['target_id'])
        elif rel == 'TARGETS_INDUSTRY':
            mention_to_industries[edge['source_id']].add(edge['target_id'])
        elif rel == 'SUPPORTED_BY_SENTENCE':
            claim_to_sentences[edge['source_id']].add(edge['target_id'])

    cards = []
    unresolved = []
    for row in iface_rows:
        ce_id = row['canonical_event_id']
        mention_ids = sorted(ce_to_mentions.get(ce_id, set()))
        claim_ids = sorted(ce_to_claims.get(ce_id, set()))
        sentence_ids = sorted({sid for cid in claim_ids for sid in claim_to_sentences.get(cid, set())})
        company_ids = sorted({cid for mid in mention_ids for cid in mention_to_companies.get(mid, set())})
        industry_ids = sorted({iid for mid in mention_ids for iid in mention_to_industries.get(mid, set())})
        direct_targets = ce_to_direct_targets.get(ce_id, [])
        card = {
            'canonical_event_id': ce_id,
            'event_thread_id': row.get('event_thread_id'),
            'event_summary_input': {
                'current_state_type': row.get('current_state_type'),
                'mention_count': len(mention_ids),
                'claim_count': len(claim_ids),
                'sentence_count': len(sentence_ids),
            },
            'event_type': (node_by_id.get(ce_id) or {}).get('event_identity_key'),
            'current_state': row.get('current_state_type'),
            'direct_target_ids': company_ids + industry_ids,
            'propagated_target_candidates': None,
            'impact_direction_label': None,
            'severity_label': None,
            'credibility_score': row.get('trust_score'),
            'credibility_scope': 'derived_graph_score',
            'independent_evidence_count': len(sentence_ids),
            'official_evidence_count': 0,
            'contradiction_count': 0,
            'novelty_features': {
                'direct_transition_out_degree': len(direct_targets),
                'claim_count': len(claim_ids),
            },
            'urgency_features': {
                'current_state': row.get('current_state_type'),
                'has_direct_evolution_edge': bool(direct_targets),
            },
            'exposure_features': {
                'company_target_count': len(company_ids),
                'industry_target_count': len(industry_ids),
            },
            'evidence_sentence_ids': sentence_ids,
            'evidence_paths': [
                ['CanonicalEvent', 'Claim', 'EvidenceSentence'],
                ['CanonicalEvent', 'StateObservation'],
            ] if sentence_ids else [['CanonicalEvent', 'StateObservation']],
            'label_status': 'unknown_null_stage1',
            'split': None,
            'field_provenance': {
                'credibility_score': 'outputs/task3_kg_v3_release/task4_event_interface.jsonl',
                'targets': 'outputs/task3_kg_v3_release/graph_edges.jsonl',
                'evidence': 'outputs/task3_kg_v3_release/graph_edges.jsonl',
            },
        }
        cards.append(card)
        if not sentence_ids:
            unresolved.append({'canonical_event_id': ce_id, 'reason': 'missing_claim_sentence_trace'})

    df = pd.DataFrame(cards)
    OUT_CARD.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_CARD, index=False)
    write_jsonl(OUT_UNRESOLVED, unresolved)

    demo = next(card for card in cards if card['evidence_sentence_ids'])
    demo_payload = {
        'canonical_event_id': demo['canonical_event_id'],
        'event_thread_id': demo['event_thread_id'],
        'task3_interface_read': True,
        'event_impact_card': demo,
        'real_one_hop_targets': demo['direct_target_ids'],
        'evidence_sentence_trace': demo['evidence_sentence_ids'][:5],
        'alert_candidate': {
            'canonical_event_id': demo['canonical_event_id'],
            'alert_priority_score': None,
            'impact_direction_score': None,
            'severity_score': None,
            'status': 'unscored_stage1_contract_validation',
        },
    }
    write_json(OUT_DEMO, demo_payload)
    acceptance = {
        'status': 'task4_stage1_ready',
        'impact_card_count': len(cards),
        'unresolved_count': len(unresolved),
        'null_prediction_fields_preserved': True,
        'task3_release_read_only': True,
    }
    write_json(REPORT_ACCEPT, acceptance)
    print(json.dumps({'acceptance': acceptance, 'demo': demo_payload}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
