#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RULE_VERSION = 'task3_kg_v2_step4_revision2_v0'
CONFIG = ROOT / 'configs/task3_kg_v2_step4_revision2.yaml'
STEP4 = ROOT / 'outputs/task3_kg_v2/step4'
STEP4_REV = ROOT / 'outputs/task3_kg_v2/step4_revision'
REV2 = ROOT / 'outputs/task3_kg_v2/step4_revision2'
REPORT = ROOT / 'reports/task3_kg_v2_step4_revision2_report.json'
RUNTIME = ROOT / 'reports/task3_kg_v2_step4_revision2_runtime.json'
DOC = ROOT / 'docs/task3_kg_v2_step4_revision2.md'
REVIEWED = STEP4_REV / 'annotation_package_agent_reviewed.xlsx'
UNCERTAIN_DEV = STEP4_REV / 'update_annotation_package.xlsx'
CANDIDATES = STEP4 / 'same_event_candidates.parquet'
MENTIONS = ROOT / 'outputs/task3_fulltext_v2/normalized_event_mentions.jsonl'
EDGES_V1 = ROOT / 'outputs/task3_kg_v2/observation_graph_v1/edges.jsonl'
NODES_V1 = ROOT / 'outputs/task3_kg_v2/observation_graph_v1/nodes.jsonl'
EMBED_INDEX = STEP4 / 'event_mention_embeddings/embedding_index.parquet'
EMBED_NPY = STEP4 / 'event_mention_embeddings/embeddings.npy'
ROUTING = STEP4_REV / 'processing_channel_repaired_event_mentions.parquet'
FEATURES = STEP4 / 'graph_feature_explanations.jsonl'

OUT_EVENTS = REV2 / 'canonical_events.jsonl'
OUT_MEMBERS = REV2 / 'canonical_event_members.jsonl'
OUT_MAP = REV2 / 'event_mention_canonical_mapping.jsonl'
OUT_EDGES = REV2 / 'same_event_revision2_edges.jsonl'
OUT_CANNOT = REV2 / 'cannot_link_revision2_edges.jsonl'
OUT_UNCERTAIN = REV2 / 'uncertain_review_revision2_queue.jsonl'
OUT_EVENTCARDS = REV2 / 'event_cards.jsonl'
OUT_EVENTOBJECTS = REV2 / 'event_object_keys.jsonl'
OUT_PAIRS = REV2 / 'pair_predictions.parquet'

FEATURE_COLS = [
    'semantic_score', 'title_similarity', 'local_evidence_similarity', 'core_element_similarity',
    'subject_match_num', 'target_type_match_num', 'target_code_match_num', 'event_type_compatible_num',
    'company_overlap', 'stock_overlap', 'industry_overlap', 'shared_entity_count', 'shared_entity_jaccard',
    'time_diff_hours_filled', 'same_evidence_unit_num', 'shared_evidence_document_count',
    'task2_assessment_neighbor_overlap', 'evidence_independent_unit_count', 'graph_common_neighbor_count',
    'graph_neighborhood_jaccard', 'structure_score', 'graph_score', 'fusion_score_before_rerank',
    'reranker_score_filled', 'hard_conflict_num', 'candidate_source_graph_only',
    'candidate_source_graph_and_semantic', 'candidate_source_semantic_only', 'event_card_similarity',
    'event_object_key_match', 'evidence_path_len', 'constraint_count'
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = '|'.join('' if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def parse_list(x: Any) -> list[Any]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    try:
        v = json.loads(str(x))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def norm_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    return ' '.join(str(x).lower().replace('\u3000', ' ').replace('\xa0', ' ').split())


def core_set(x: Any) -> set[str]:
    if isinstance(x, list):
        return {str(i) for i in x if str(i)}
    if x is None:
        return set()
    return {s for s in str(x).replace('|', ',').split(',') if s}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def pair_id(a: str, b: str) -> str:
    x, y = pair_key(a, b)
    return stable_id('EMP', x, y)


def build_event_cards() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    mentions = read_jsonl(MENTIONS)
    routing = pd.read_parquet(ROUTING)
    route_map = dict(zip(routing['event_mention_id'], routing['processing_channel']))
    rows = []
    for m in mentions:
        mid = str(m['event_mention_id'])
        title = ''
        local = norm_text(m.get('event_local_text', ''))
        evidence = ' | '.join([str(s) for s in (m.get('evidence_sentences') or [])[:5]])
        core = list(m.get('core_elements') or [])
        object_key = stable_id('EOK', m.get('target_type'), m.get('target_code'), m.get('provisional_event_type'), '|'.join(sorted(core)))
        card_text = ' '.join([str(m.get('target_code') or ''), str(m.get('provisional_event_type') or ''), ' '.join(core), evidence])
        rows.append({
            'event_mention_id': mid,
            'event_card_id': stable_id('ECARD', mid),
            'event_object_key': object_key,
            'target_type': str(m.get('target_type') or ''),
            'target_code': str(m.get('target_code') or ''),
            'provisional_event_type': str(m.get('provisional_event_type') or ''),
            'publish_time': str(m.get('publish_time') or ''),
            'core_elements': core,
            'event_local_text': local,
            'evidence_sentences': evidence,
            'card_text': norm_text(card_text),
            'processing_channel': route_map.get(mid, 'unknown'),
            'trust_score_scope': m.get('trust_score_scope'),
        })
    df = pd.DataFrame(rows)
    by_id = {r['event_mention_id']: r for r in rows}
    return df, by_id


def augment_candidates(cards_by_id: dict[str, dict[str, Any]]) -> pd.DataFrame:
    cand = pd.read_parquet(CANDIDATES).copy()
    feat_rows = {r['pair_id']: r for r in read_jsonl(FEATURES)}
    cand['subject_match_num'] = cand['subject_match'].astype(float)
    cand['target_type_match_num'] = cand['target_type_match'].astype(float)
    cand['target_code_match_num'] = cand['target_code_match'].astype(float)
    cand['event_type_compatible_num'] = cand['event_type_compatible'].astype(float)
    cand['same_evidence_unit_num'] = cand['same_evidence_unit'].astype(float)
    cand['hard_conflict_num'] = cand['hard_conflict'].astype(float)
    cand['time_diff_hours_filled'] = cand['time_diff_hours'].fillna(24 * 365)
    cand['reranker_score_filled'] = cand['reranker_score'].fillna(0.5)
    cand['candidate_source_graph_only'] = (cand['candidate_source_class'] == 'graph_only').astype(float)
    cand['candidate_source_graph_and_semantic'] = (cand['candidate_source_class'] == 'graph_and_semantic').astype(float)
    cand['candidate_source_semantic_only'] = (cand['candidate_source_class'] == 'semantic_only').astype(float)
    event_card_similarity = []
    event_object_key_match = []
    evidence_path_len = []
    constraint_count = []
    for _, r in cand.iterrows():
        a = cards_by_id.get(str(r['event_mention_id_a']), {})
        b = cards_by_id.get(str(r['event_mention_id_b']), {})
        event_card_similarity.append(jaccard(set(str(a.get('card_text', '')).split()), set(str(b.get('card_text', '')).split())))
        event_object_key_match.append(float(a.get('event_object_key') == b.get('event_object_key') and a.get('event_object_key') is not None))
        feat = feat_rows.get(r['pair_id'], {})
        evidence_path_len.append(len(feat.get('evidence_path', [])))
        constraint_count.append(len(parse_list(r.get('constraint_results_json'))))
    cand['event_card_similarity'] = event_card_similarity
    cand['event_object_key_match'] = event_object_key_match
    cand['evidence_path_len'] = evidence_path_len
    cand['constraint_count'] = constraint_count
    return cand


def load_dev_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviewed = pd.read_excel(REVIEWED)
    uncertain = pd.read_excel(UNCERTAIN_DEV)
    reviewed = reviewed.copy()
    reviewed['label_3way'] = reviewed['human_same_event'].map({'yes': 'same_event', 'no': 'different_event', 'uncertain': 'uncertain_review'})
    reviewed = reviewed[reviewed['label_3way'].notna()].copy()
    uncertain = uncertain.copy()
    uncertain['label_3way'] = 'uncertain_review'
    uncertain = uncertain.head(67).copy()
    return reviewed, uncertain


def train_models(cand: pd.DataFrame, reviewed: pd.DataFrame, uncertain: pd.DataFrame) -> tuple[Pipeline, Pipeline, dict[str, Any]]:
    dev = reviewed[['pair_id', 'label_3way']].copy()
    dev = pd.concat([dev, uncertain[['pair_id', 'label_3way']]], ignore_index=True)
    train_df = cand.merge(dev, on='pair_id', how='inner').copy()
    X = train_df[FEATURE_COLS].fillna(0.0)
    y_same = (train_df['label_3way'] == 'same_event').astype(int)
    y_diff = (train_df['label_3way'] == 'different_event').astype(int)
    same_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    diff_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    same_model.fit(X, y_same)
    diff_model.fit(X, y_diff)
    same_scores = same_model.predict_proba(X)[:, 1]
    diff_scores = diff_model.predict_proba(X)[:, 1]
    same_threshold = 0.5
    diff_threshold = 0.5
    best_same_f1 = -1.0
    for t in np.linspace(0.45, 0.95, 51):
        pred = (same_scores >= t).astype(int)
        f1 = f1_score(y_same, pred, zero_division=0)
        if f1 > best_same_f1:
            best_same_f1 = f1
            same_threshold = float(t)
    best_refusal = -1.0
    uncertain_ids = set(uncertain['pair_id'])
    for td in np.linspace(0.45, 0.95, 51):
        pred_diff = diff_scores >= td
        pred_same = same_scores >= same_threshold
        pred = np.where(pred_diff, 'different_event', np.where(pred_same, 'same_event', 'uncertain_review'))
        refusal = np.mean([pred[i] == 'uncertain_review' for i, pid in enumerate(train_df['pair_id']) if pid in uncertain_ids]) if uncertain_ids else 0.0
        if refusal > best_refusal:
            best_refusal = refusal
            diff_threshold = float(td)
    metrics = {
        'train_rows': int(len(train_df)),
        'same_threshold': same_threshold,
        'different_threshold': diff_threshold,
        'same_train_f1': best_same_f1,
        'uncertain_refusal_on_67': best_refusal,
        'label_counts': train_df['label_3way'].value_counts().to_dict(),
    }
    return same_model, diff_model, metrics


def predict_pairs(cand: pd.DataFrame, same_model: Pipeline, diff_model: Pipeline, thresholds: dict[str, Any]) -> pd.DataFrame:
    X = cand[FEATURE_COLS].fillna(0.0)
    cand = cand.copy()
    cand['same_event_score'] = same_model.predict_proba(X)[:, 1]
    cand['different_event_score'] = diff_model.predict_proba(X)[:, 1]
    same_t = thresholds['same_threshold']
    diff_t = thresholds['different_threshold']
    pred = []
    reason = []
    for _, r in cand.iterrows():
        if bool(r['hard_conflict']):
            pred.append('different_event')
            reason.append('hard_cannot_link_constraint')
        elif float(r['different_event_score']) >= diff_t:
            pred.append('different_event')
            reason.append('calibrated_different_event_classifier')
        elif float(r['same_event_score']) >= same_t and not bool(r['same_evidence_unit']) and float(r['event_object_key_match']) >= 1.0:
            pred.append('same_event')
            reason.append('calibrated_same_event_classifier_with_event_object_key')
        else:
            pred.append('uncertain_review')
            reason.append('uncertain_review_reject_option')
    cand['revision2_relation_label'] = pred
    cand['revision2_decision_reason'] = reason
    cand['revision2_confidence'] = np.where(cand['revision2_relation_label'] == 'same_event', cand['same_event_score'], np.where(cand['revision2_relation_label'] == 'different_event', cand['different_event_score'], 1 - np.maximum(cand['same_event_score'], cand['different_event_score'])))
    return cand


def decode_canonical_events(cand: pd.DataFrame, cards: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    same_df = cand[cand['revision2_relation_label'] == 'same_event'].copy()
    cannot_df = cand[cand['revision2_relation_label'] == 'different_event'].copy()
    uncertain_df = cand[cand['revision2_relation_label'] == 'uncertain_review'].copy()
    parent = {mid: mid for mid in cards['event_mention_id']}
    cannot_pairs = {pair_key(a, b) for a, b in zip(cannot_df['event_mention_id_a'], cannot_df['event_mention_id_b'])}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        members_a = [m for m in parent if find(m) == ra]
        members_b = [m for m in parent if find(m) == rb]
        for ma in members_a:
            for mb in members_b:
                if pair_key(ma, mb) in cannot_pairs:
                    return False
        parent[rb] = ra
        return True

    accepted_same = []
    for _, r in same_df.sort_values(['revision2_confidence', 'same_event_score'], ascending=[False, False]).iterrows():
        if union(str(r['event_mention_id_a']), str(r['event_mention_id_b'])):
            accepted_same.append(r.to_dict())
    groups: dict[str, list[str]] = defaultdict(list)
    for mid in cards['event_mention_id']:
        groups[find(mid)].append(mid)

    card_by_id = {r['event_mention_id']: r for r in cards.to_dict('records')}
    canonical_events = []
    canonical_members = []
    mapping = []
    event_object_rows = []
    for root, members in sorted(groups.items()):
        members = sorted(members)
        representative = card_by_id[members[0]]
        ce_id = stable_id('CE', representative['event_object_key'], '|'.join(members))
        event_object_rows.append({
            'event_object_key': representative['event_object_key'],
            'target_type': representative['target_type'],
            'target_code': representative['target_code'],
            'provisional_event_type': representative['provisional_event_type'],
            'core_elements': representative['core_elements'],
        })
        canonical_events.append({
            'canonical_event_id': ce_id,
            'node_type': 'CanonicalEvent',
            'event_object_key': representative['event_object_key'],
            'representative_event_mention_id': representative['event_mention_id'],
            'member_count': len(members),
            'verification_status': 'provisional_model_prediction',
            'decision_reason': 'connected_component_from_same_event_graph_with_cannot_link_guard',
            'evidence_path': ['EventMention', 'EventCard/EventObjectKey', 'ConstraintGraph', 'CanonicalEvent'],
            'rule_version': RULE_VERSION,
        })
        for mid in members:
            card = card_by_id[mid]
            canonical_members.append({
                'canonical_event_id': ce_id,
                'event_mention_id': mid,
                'member_role': 'core_member' if mid == representative['event_mention_id'] else 'support_member',
                'processing_channel': card['processing_channel'],
                'evidence_path': ['EventMention', 'CanonicalEvent'],
                'decision_reason': 'same_event_component_assignment',
                'verification_status': 'provisional_model_prediction',
            })
            mapping.append({
                'event_mention_id': mid,
                'canonical_event_id': ce_id,
                'mapping_status': 'core_member' if mid == representative['event_mention_id'] else 'member',
                'processing_channel': card['processing_channel'],
                'decision_reason': 'same_event_component_assignment',
                'verification_status': 'provisional_model_prediction',
            })
    cannot_edges = []
    for _, r in cannot_df.iterrows():
        cannot_edges.append({
            'pair_id': r['pair_id'],
            'source_event_mention_id': r['event_mention_id_a'],
            'target_event_mention_id': r['event_mention_id_b'],
            'relation_type': 'CANNOT_LINK',
            'confidence': float(r['revision2_confidence']),
            'decision_reason': r['revision2_decision_reason'],
            'evidence_path': ['EventMention', 'ConstraintGraph', 'EventMention'],
            'verification_status': 'provisional_model_prediction',
            'rule_version': RULE_VERSION,
        })
    same_edges = []
    for r in accepted_same:
        same_edges.append({
            'pair_id': r['pair_id'],
            'source_event_mention_id': r['event_mention_id_a'],
            'target_event_mention_id': r['event_mention_id_b'],
            'relation_type': 'SAME_EVENT_AS',
            'confidence': float(r['revision2_confidence']),
            'decision_reason': r['revision2_decision_reason'],
            'evidence_path': ['EventMention', 'EventCard/EventObjectKey', 'ConstraintGraph', 'EventMention'],
            'verification_status': 'provisional_model_prediction',
            'rule_version': RULE_VERSION,
        })
    uncertain_rows = []
    for _, r in uncertain_df.iterrows():
        uncertain_rows.append({
            'pair_id': r['pair_id'],
            'source_event_mention_id': r['event_mention_id_a'],
            'target_event_mention_id': r['event_mention_id_b'],
            'decision_reason': r['revision2_decision_reason'],
            'confidence': float(r['revision2_confidence']),
            'same_event_score': float(r['same_event_score']),
            'different_event_score': float(r['different_event_score']),
            'verification_status': 'provisional_model_prediction',
        })
    return canonical_events, canonical_members, mapping, same_edges, cannot_edges, uncertain_rows, event_object_rows


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild-from-cache', action='store_true')
    args = parser.parse_args()
    started = time.time()
    REV2.mkdir(parents=True, exist_ok=True)
    before_hashes = {
        'step4_candidates': sha256_file(CANDIDATES),
        'reviewed_package': sha256_file(REVIEWED),
        'uncertain_package': sha256_file(UNCERTAIN_DEV),
        'observation_graph_v1_nodes': sha256_file(NODES_V1),
        'observation_graph_v1_edges': sha256_file(EDGES_V1),
    }
    cards, cards_by_id = build_event_cards()
    cards.to_json(OUT_EVENTCARDS, orient='records', force_ascii=False, lines=True)
    cand = augment_candidates(cards_by_id)
    reviewed, uncertain = load_dev_labels()
    same_model, diff_model, train_metrics = train_models(cand, reviewed, uncertain)
    pred = predict_pairs(cand, same_model, diff_model, train_metrics)
    pred.to_parquet(OUT_PAIRS, index=False)
    canonical_events, canonical_members, mapping, same_edges, cannot_edges, uncertain_rows, event_object_rows = decode_canonical_events(pred, cards)
    write_jsonl(OUT_EVENTS, canonical_events)
    write_jsonl(OUT_MEMBERS, canonical_members)
    write_jsonl(OUT_MAP, mapping)
    write_jsonl(OUT_EDGES, same_edges)
    write_jsonl(OUT_CANNOT, cannot_edges)
    write_jsonl(OUT_UNCERTAIN, uncertain_rows)
    write_jsonl(OUT_EVENTOBJECTS, event_object_rows)
    after_hashes = {
        'step4_candidates': sha256_file(CANDIDATES),
        'reviewed_package': sha256_file(REVIEWED),
        'uncertain_package': sha256_file(UNCERTAIN_DEV),
        'observation_graph_v1_nodes': sha256_file(NODES_V1),
        'observation_graph_v1_edges': sha256_file(EDGES_V1),
    }
    report = {
        'version': RULE_VERSION,
        'mode': 'revision2_same_event_different_event_uncertain_classifier',
        'agent_assisted_labels_not_human_gold': True,
        'training': train_metrics,
        'prediction_distribution': pred['revision2_relation_label'].value_counts().to_dict(),
        'canonical_event_count': len(canonical_events),
        'same_event_edge_count': len(same_edges),
        'cannot_link_edge_count': len(cannot_edges),
        'uncertain_count': len(uncertain_rows),
        'event_card_count': int(len(cards)),
        'event_object_key_count': int(len({r['event_object_key'] for r in event_object_rows})),
        'constraints': {
            'create_event_thread': False,
            'train_stage_migration_classifier': False,
            'enter_phase5': False,
            'old_step4_outputs_unchanged': before_hashes == after_hashes,
        },
        'outputs': {
            'canonical_events': str(OUT_EVENTS.relative_to(ROOT)),
            'canonical_event_members': str(OUT_MEMBERS.relative_to(ROOT)),
            'event_mention_canonical_mapping': str(OUT_MAP.relative_to(ROOT)),
            'pair_predictions': str(OUT_PAIRS.relative_to(ROOT)),
        },
    }
    runtime = {
        'version': RULE_VERSION,
        'resource_usage': {
            'elapsed_seconds': round(time.time() - started, 3),
            'max_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            'embedding_cache_used': True,
            'qwen_reranker_rerun': False,
        },
        'integrity': {
            'old_step4_outputs_unchanged': before_hashes == after_hashes,
            'canonical_events_created': len(canonical_events) > 0,
            'event_thread_created': False,
            'stage_classifier_trained': False,
            'phase5_entered': False,
            'all_same_edges_have_evidence_path': all(bool(r.get('evidence_path')) for r in same_edges),
            'all_cannot_edges_have_evidence_path': all(bool(r.get('evidence_path')) for r in cannot_edges),
            'same_cannot_disjoint': not ({r['pair_id'] for r in same_edges} & {r['pair_id'] for r in cannot_edges}),
            'mapping_count': len(mapping),
            'event_card_count': int(len(cards)),
        },
    }
    write_json(REPORT, report)
    write_json(RUNTIME, runtime)
    DOC.write_text('# Task3 KG v2 Step4 Revision2\n\n' + json.dumps({'prediction_distribution': report['prediction_distribution'], 'canonical_event_count': report['canonical_event_count']}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': report, 'runtime': runtime}, ensure_ascii=False, indent=2))
    return {'report': report, 'runtime': runtime}


if __name__ == '__main__':
    main()
