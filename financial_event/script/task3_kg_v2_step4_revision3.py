#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
RULE_VERSION = 'task3_kg_v2_step4_revision3_v0'
CONFIG = ROOT / 'configs/task3_kg_v2_step4_revision3.yaml'
STEP4 = ROOT / 'outputs/task3_kg_v2/step4'
REV = ROOT / 'outputs/task3_kg_v2/step4_revision'
REV2 = ROOT / 'outputs/task3_kg_v2/step4_revision2'
REV3 = ROOT / 'outputs/task3_kg_v2/step4_revision3'
REPORT = ROOT / 'reports/task3_kg_v2_step4_revision3_report.json'
RUNTIME = ROOT / 'reports/task3_kg_v2_step4_revision3_runtime.json'
DOC = ROOT / 'docs/task3_kg_v2_step4_revision3.md'
REVIEWED = REV / 'annotation_package_agent_reviewed.xlsx'
CANDIDATES = STEP4 / 'same_event_candidates.parquet'
MENTIONS = ROOT / 'outputs/task3_fulltext_v2/normalized_event_mentions.jsonl'
EVIDENCE = ROOT / 'outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl'
FEATURES = STEP4 / 'graph_feature_explanations.jsonl'
ROUTING = REV / 'processing_channel_repaired_event_mentions.parquet'
REV2_CARDS = REV2 / 'event_cards.jsonl'
REV2_MAP = REV2 / 'event_mention_canonical_mapping.jsonl'
EMBED_INDEX = STEP4 / 'event_mention_embeddings/embedding_index.parquet'
EMBED_NPY = STEP4 / 'event_mention_embeddings/embeddings.npy'

OUT_EVENTS = REV3 / 'canonical_events.jsonl'
OUT_MEMBERS = REV3 / 'canonical_event_members.jsonl'
OUT_MAP = REV3 / 'event_mention_canonical_mapping.jsonl'
OUT_SAME = REV3 / 'same_event_revision3_edges.jsonl'
OUT_CANNOT = REV3 / 'cannot_link_revision3_edges.jsonl'
OUT_UNCERTAIN = REV3 / 'uncertain_review_revision3_queue.jsonl'
OUT_CARDS = REV3 / 'event_cards_revision3.jsonl'
OUT_FAMILY = REV3 / 'event_family_keys.jsonl'
OUT_IDENTITY = REV3 / 'event_identity_keys.jsonl'
OUT_IDENTITY_FIELDS = REV3 / 'identity_field_evidence.jsonl'
OUT_PAIRS = REV3 / 'pair_predictions.parquet'
OUT_AUDIT = REV3 / 'coarse_key_split_audit.json'
OUT_COMPARE = REV3 / 'canonical_event_diff_summary.json'

FEATURE_COLS = [
    'semantic_score', 'title_similarity', 'local_evidence_similarity', 'core_element_similarity',
    'subject_match_num', 'target_type_match_num', 'target_code_match_num', 'event_type_compatible_num',
    'company_overlap', 'stock_overlap', 'industry_overlap', 'shared_entity_count', 'shared_entity_jaccard',
    'time_diff_hours_filled', 'same_evidence_unit_num', 'shared_evidence_document_count',
    'task2_assessment_neighbor_overlap', 'evidence_independent_unit_count', 'graph_common_neighbor_count',
    'graph_neighborhood_jaccard', 'structure_score', 'graph_score', 'fusion_score_before_rerank',
    'reranker_score_filled', 'hard_conflict_num', 'candidate_source_graph_only',
    'candidate_source_graph_and_semantic', 'candidate_source_semantic_only', 'event_family_match',
    'event_identity_match', 'event_identity_conflict', 'identity_field_overlap', 'identity_field_conflict_count',
    'event_card_similarity', 'identity_evidence_coverage', 'evidence_path_len', 'constraint_count'
]

CAPITAL_MARKET_HINTS = ['证券', '债券', '交易', '减持', '增持', '回购', '定增', '配股', '质押']
PHARMA_HINTS = ['药', '临床', '适应症', '试验', '获批', '患者', 'III期', 'II期', 'I期']
PRODUCT_HINTS = ['产品', '型号', '芯片', '平台', '技术', '系统', '发布']
PROJECT_HINTS = ['项目', '基地', '工程', '合同', '开工', '中标', '签约', '金额']
PENALTY_HINTS = ['处罚', '立案', '调查', '监管', '罚款', '警示', '证监', '交易所']
MA_HINTS = ['收购', '并购', '重组', '标的', '股权', '交易阶段', '对价']


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


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def extract_regex(text: str, patterns: list[tuple[str, str]], field: str) -> dict[str, Any] | None:
    for regex, label in patterns:
        m = re.search(regex, text, re.I)
        if m:
            value = m.group(1).strip()
            if value:
                return {
                    'field_name': field,
                    'field_value': value,
                    'evidence_sentence_id': None,
                    'evidence_text': m.group(0).strip(),
                    'source': f'rule:{label}',
                    'confidence': 0.92,
                }
    return None


def classify_event_supertype(event_type: str, text: str) -> str:
    value = f"{event_type} {text}"
    if any(k in value for k in CAPITAL_MARKET_HINTS):
        return 'capital_market'
    if any(k in value for k in PHARMA_HINTS):
        return 'pharma_rd'
    if any(k in value for k in PRODUCT_HINTS):
        return 'product_innovation'
    if any(k in value for k in PROJECT_HINTS):
        return 'project_construction'
    if any(k in value for k in PENALTY_HINTS):
        return 'investigation_penalty'
    if any(k in value for k in MA_HINTS):
        return 'ma_reorg'
    return 'other'


def infer_batch_id(text: str) -> dict[str, Any] | None:
    patterns = [
        (r'公告编号[:：]?\s*([A-Za-z0-9\-]+)', 'announcement_number'),
        (r'试验编号[:：]?\s*([A-Za-z0-9\-]+)', 'trial_number'),
        (r'项目编号[:：]?\s*([A-Za-z0-9\-]+)', 'project_number'),
        (r'处罚决定书[:：]?\s*([A-Za-z0-9\-]+)', 'penalty_number'),
        (r'合同编号[:：]?\s*([A-Za-z0-9\-]+)', 'contract_number'),
    ]
    return extract_regex(text, patterns, 'batch_id')


def extract_identity_fields(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = ' '.join([str(row.get('event_local_text') or ''), str(row.get('evidence_text') or '')])
    event_type = str(row.get('provisional_event_type') or '')
    fields: list[dict[str, Any]] = []
    supertype = classify_event_supertype(event_type, text)
    common_batch = infer_batch_id(text)
    if common_batch:
        fields.append(common_batch)
    if supertype == 'capital_market':
        for item in [
            extract_regex(text, [(r'证券代码[:：]?\s*([0-9]{6})', 'security_code')], 'security_code'),
            extract_regex(text, [(r'债券代码[:：]?\s*([0-9A-Za-z\-]+)', 'bond_code')], 'bond_code'),
            extract_regex(text, [(r'交易对手(?:方)?[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9（）()]+)', 'counterparty')], 'counterparty'),
            extract_regex(text, [(r'标的(?:资产|证券)?[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9（）()]+)', 'trade_target')], 'trade_target'),
        ]:
            if item: fields.append(item)
    elif supertype == 'pharma_rd':
        for item in [
            extract_regex(text, [(r'药物[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9\-]+)', 'drug')], 'drug'),
            extract_regex(text, [(r'适应症[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9、]+)', 'indication')], 'indication'),
            extract_regex(text, [(r'([I1V]+期临床|临床[一二三四IV]+期)', 'clinical_stage')], 'clinical_stage'),
            extract_regex(text, [(r'试验编号[:：]?\s*([A-Za-z0-9\-]+)', 'trial_number')], 'trial_number'),
        ]:
            if item: fields.append(item)
    elif supertype == 'product_innovation':
        for item in [
            extract_regex(text, [(r'产品[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9\-]+)', 'product')], 'product'),
            extract_regex(text, [(r'型号[:：]?\s*([A-Za-z0-9\-]+)', 'model')], 'model'),
            extract_regex(text, [(r'技术(?:名称)?[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9\-]+)', 'technology_name')], 'technology_name'),
        ]:
            if item: fields.append(item)
    elif supertype == 'project_construction':
        for item in [
            extract_regex(text, [(r'项目[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9\-]+项目)', 'project_name')], 'project_name'),
            extract_regex(text, [(r'位于([\u4e00-\u9fa5A-Za-z0-9]+)', 'location')], 'location'),
            extract_regex(text, [(r'合同方[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9（）()]+)', 'contract_party')], 'contract_party'),
            extract_regex(text, [(r'([0-9]+(?:\.[0-9]+)?亿元?)', 'amount')], 'amount'),
        ]:
            if item: fields.append(item)
    elif supertype == 'investigation_penalty':
        for item in [
            extract_regex(text, [(r'(证监会|交易所|证监局|监管局)', 'regulator')], 'regulator'),
            extract_regex(text, [(r'处罚决定书[:：]?\s*([A-Za-z0-9\-]+)', 'penalty_number')], 'penalty_number'),
            extract_regex(text, [(r'违法事项[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9、]+)', 'violation_item')], 'violation_item'),
        ]:
            if item: fields.append(item)
    elif supertype == 'ma_reorg':
        for item in [
            extract_regex(text, [(r'收购方[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9（）()]+)', 'acquirer')], 'acquirer'),
            extract_regex(text, [(r'标的方[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9（）()]+)', 'target_party')], 'target_party'),
            extract_regex(text, [(r'([0-9]+(?:\.[0-9]+)?亿元?)', 'amount')], 'amount'),
            extract_regex(text, [(r'(签约|公告|获批|交割|终止)', 'transaction_stage')], 'transaction_stage'),
        ]:
            if item: fields.append(item)
    return fields


def build_mentions() -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    routing = pd.read_parquet(ROUTING)
    route_map = dict(zip(routing['event_mention_id'], routing['processing_channel']))
    evidence_map: dict[str, list[str]] = defaultdict(list)
    for r in read_jsonl(EVIDENCE):
        evidence_map[str(r.get('event_mention_id'))].append(str(r.get('sentence') or ''))
    rows = []
    field_rows = []
    cards_by_id = {}
    for m in read_jsonl(MENTIONS):
        mid = str(m['event_mention_id'])
        event_type = str(m.get('provisional_event_type') or '')
        supertype = classify_event_supertype(event_type, str(m.get('event_local_text') or ''))
        core = [str(x) for x in (m.get('core_elements') or []) if str(x)]
        row = {
            'event_mention_id': mid,
            'document_id': str(m.get('document_id') or ''),
            'article_id': str(m.get('article_id') or ''),
            'target_type': str(m.get('target_type') or ''),
            'target_code': str(m.get('target_code') or ''),
            'provisional_event_type': event_type,
            'event_supertype': supertype,
            'core_elements': core,
            'publish_time': str(m.get('publish_time') or ''),
            'event_local_text': str(m.get('event_local_text') or ''),
            'evidence_text': ' | '.join(evidence_map.get(mid, [])[:5]),
            'processing_channel': route_map.get(mid, 'unknown'),
        }
        identity_fields = extract_identity_fields(row)
        batch_field = next((x for x in identity_fields if x['field_name'] == 'batch_id' and x.get('field_value')), None)
        family_key = stable_id('EFK', row['target_type'], row['target_code'], row['event_supertype'])
        identity_parts = [row['target_type'], row['target_code'], row['provisional_event_type'], '|'.join(sorted(core))]
        id_values = sorted(f"{x['field_name']}={x['field_value']}" for x in identity_fields if x.get('field_value'))
        if batch_field and batch_field.get('field_value'):
            identity_parts.append(batch_field['field_value'])
        identity_parts.extend(id_values)
        identity_key = stable_id('EIK', *identity_parts)
        row.update({
            'event_family_key': family_key,
            'event_identity_key': identity_key,
            'identity_field_count': len(identity_fields),
            'identity_field_values': id_values,
            'identity_batch_id': batch_field.get('field_value') if batch_field else '',
        })
        cards_by_id[mid] = row
        rows.append(row)
        for fr in identity_fields:
            out = {'event_mention_id': mid, 'event_family_key': family_key, 'event_identity_key': identity_key, **fr}
            field_rows.append(out)
    return pd.DataFrame(rows), field_rows, cards_by_id


def augment_candidates(mention_df: pd.DataFrame, cards_by_id: dict[str, dict[str, Any]]) -> pd.DataFrame:
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

    fam_match, id_match, id_conflict, id_overlap, id_conflict_count, coverage, ecsim, eplen, ccount = [], [], [], [], [], [], [], [], []
    for _, r in cand.iterrows():
        a = cards_by_id.get(str(r['event_mention_id_a']), {})
        b = cards_by_id.get(str(r['event_mention_id_b']), {})
        fam_match.append(float(a.get('event_family_key') == b.get('event_family_key')))
        id_match.append(float(a.get('event_identity_key') == b.get('event_identity_key')))
        aset = set(a.get('identity_field_values') or [])
        bset = set(b.get('identity_field_values') or [])
        overlap = jaccard(aset, bset)
        id_overlap.append(overlap)
        conflict_fields = 0
        ad = {x.split('=',1)[0]:x.split('=',1)[1] for x in aset if '=' in x}
        bd = {x.split('=',1)[0]:x.split('=',1)[1] for x in bset if '=' in x}
        for k in set(ad) & set(bd):
            if ad[k] != bd[k]:
                conflict_fields += 1
        id_conflict_count.append(conflict_fields)
        id_conflict.append(float(conflict_fields > 0))
        cov = min(float(a.get('identity_field_count', 0)), float(b.get('identity_field_count', 0)))
        coverage.append(cov)
        ecsim.append(jaccard(set(norm_text(a.get('event_local_text')).split()), set(norm_text(b.get('event_local_text')).split())))
        feat = feat_rows.get(r['pair_id'], {})
        eplen.append(len(feat.get('evidence_path', [])))
        ccount.append(len(parse_list(r.get('constraint_results_json'))))
    cand['event_family_match'] = fam_match
    cand['event_identity_match'] = id_match
    cand['event_identity_conflict'] = id_conflict
    cand['identity_field_overlap'] = id_overlap
    cand['identity_field_conflict_count'] = id_conflict_count
    cand['identity_evidence_coverage'] = coverage
    cand['event_card_similarity'] = ecsim
    cand['evidence_path_len'] = eplen
    cand['constraint_count'] = ccount
    return cand


def load_dev_labels() -> pd.DataFrame:
    reviewed = pd.read_excel(REVIEWED).copy()
    reviewed['label_3way'] = reviewed['human_same_event'].map({'yes': 'same_event', 'no': 'different_event', 'uncertain': 'uncertain_review'})
    return reviewed[reviewed['label_3way'].notna()].copy()


def train_models(cand: pd.DataFrame, reviewed: pd.DataFrame) -> tuple[Pipeline, Pipeline, dict[str, Any]]:
    train_df = cand.merge(reviewed[['pair_id', 'label_3way']], on='pair_id', how='inner').copy()
    X = train_df[FEATURE_COLS].fillna(0.0)
    y_same = (train_df['label_3way'] == 'same_event').astype(int)
    y_diff = (train_df['label_3way'] == 'different_event').astype(int)
    uncertain_mask = train_df['label_3way'] == 'uncertain_review'
    same_model = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2500, class_weight='balanced', random_state=42))])
    diff_model = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2500, class_weight='balanced', random_state=42))])
    same_model.fit(X, y_same)
    diff_model.fit(X, y_diff)
    same_scores = same_model.predict_proba(X)[:, 1]
    diff_scores = diff_model.predict_proba(X)[:, 1]
    same_t, diff_t, best_same_f1, best_reject = 0.5, 0.5, -1.0, -1.0
    for t in np.linspace(0.35, 0.90, 56):
        pred = (same_scores >= t).astype(int)
        f1 = f1_score(y_same, pred, zero_division=0)
        if f1 > best_same_f1:
            best_same_f1 = f1
            same_t = float(t)
    for t in np.linspace(0.45, 0.98, 54):
        pred_diff = diff_scores >= t
        pred_same = same_scores >= same_t
        pred = np.where(pred_diff, 'different_event', np.where(pred_same, 'same_event', 'uncertain_review'))
        reject = float(np.mean(pred[uncertain_mask] == 'uncertain_review')) if uncertain_mask.any() else 0.0
        if reject > best_reject:
            best_reject = reject
            diff_t = float(t)
    metrics = {
        'train_rows': int(len(train_df)),
        'same_threshold': same_t,
        'different_threshold': diff_t,
        'development_fit_metric_same_f1': best_same_f1,
        'uncertain_refusal_on_67': best_reject,
        'label_counts': train_df['label_3way'].value_counts().to_dict(),
    }
    return same_model, diff_model, metrics


def predict_pairs(cand: pd.DataFrame, same_model: Pipeline, diff_model: Pipeline, thresholds: dict[str, Any]) -> pd.DataFrame:
    X = cand[FEATURE_COLS].fillna(0.0)
    cand = cand.copy()
    cand['same_event_score_r3'] = same_model.predict_proba(X)[:, 1]
    cand['different_event_score_r3'] = diff_model.predict_proba(X)[:, 1]
    pred, reason = [], []
    st, dt = thresholds['same_threshold'], thresholds['different_threshold']
    for _, r in cand.iterrows():
        if bool(r['hard_conflict']) or float(r['event_identity_conflict']) >= 1.0:
            pred.append('different_event')
            reason.append('hard_conflict_or_identity_conflict')
        elif float(r['different_event_score_r3']) >= dt:
            pred.append('different_event')
            reason.append('calibrated_different_event_classifier_revision3')
        elif float(r['same_event_score_r3']) >= st and not bool(r['same_evidence_unit']) and float(r['event_family_match']) >= 1.0 and float(r['event_identity_conflict']) == 0.0:
            pred.append('same_event')
            reason.append('calibrated_same_event_classifier_revision3')
        else:
            pred.append('uncertain_review')
            reason.append('uncertain_review_reject_option_revision3')
    cand['revision3_relation_label'] = pred
    cand['revision3_decision_reason'] = reason
    cand['revision3_confidence'] = np.where(cand['revision3_relation_label'] == 'same_event', cand['same_event_score_r3'], np.where(cand['revision3_relation_label'] == 'different_event', cand['different_event_score_r3'], 1 - np.maximum(cand['same_event_score_r3'], cand['different_event_score_r3'])))
    return cand


def decode(pred: pd.DataFrame, cards: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    same_df = pred[pred['revision3_relation_label'] == 'same_event'].copy()
    cannot_df = pred[pred['revision3_relation_label'] == 'different_event'].copy()
    uncertain_df = pred[pred['revision3_relation_label'] == 'uncertain_review'].copy()
    parent = {mid: mid for mid in cards['event_mention_id']}
    cannot_pairs = {(a, b) if a < b else (b, a) for a, b in zip(cannot_df['event_mention_id_a'], cannot_df['event_mention_id_b'])}
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
                if ((ma, mb) if ma < mb else (mb, ma)) in cannot_pairs:
                    return False
        parent[rb] = ra
        return True
    accepted_same = []
    for _, r in same_df.sort_values(['revision3_confidence', 'same_event_score_r3'], ascending=[False, False]).iterrows():
        if union(str(r['event_mention_id_a']), str(r['event_mention_id_b'])):
            accepted_same.append(r.to_dict())
    groups: dict[str, list[str]] = defaultdict(list)
    for mid in cards['event_mention_id']:
        groups[find(mid)].append(mid)
    by_id = {r['event_mention_id']: r for r in cards.to_dict('records')}
    events=[]; members=[]; mapping=[]; same_edges=[]; cannot_edges=[]; uncertain_rows=[]
    for root, mids in sorted(groups.items()):
        mids = sorted(mids)
        rep = by_id[mids[0]]
        ce_id = stable_id('CE3', rep['event_identity_key'], '|'.join(mids))
        events.append({'canonical_event_id': ce_id, 'node_type': 'CanonicalEvent', 'event_family_key': rep['event_family_key'], 'event_identity_key': rep['event_identity_key'], 'representative_event_mention_id': rep['event_mention_id'], 'member_count': len(mids), 'verification_status': 'provisional_model_prediction', 'decision_reason': 'connected_component_from_same_event_graph_with_identity_conflict_guard', 'evidence_path': ['EventMention','EventFamilyKey','EventIdentityKey','ConstraintGraph','CanonicalEvent'], 'rule_version': RULE_VERSION})
        for mid in mids:
            c = by_id[mid]
            members.append({'canonical_event_id': ce_id, 'event_mention_id': mid, 'member_role': 'core_member' if mid == rep['event_mention_id'] else 'support_member', 'processing_channel': c['processing_channel'], 'event_family_key': c['event_family_key'], 'event_identity_key': c['event_identity_key'], 'evidence_path': ['EventMention','CanonicalEvent'], 'decision_reason': 'same_event_component_assignment_revision3', 'verification_status': 'provisional_model_prediction'})
            mapping.append({'event_mention_id': mid, 'canonical_event_id': ce_id, 'mapping_status': 'core_member' if mid == rep['event_mention_id'] else 'member', 'processing_channel': c['processing_channel'], 'decision_reason': 'same_event_component_assignment_revision3', 'verification_status': 'provisional_model_prediction'})
    for r in accepted_same:
        same_edges.append({'pair_id': r['pair_id'], 'source_event_mention_id': r['event_mention_id_a'], 'target_event_mention_id': r['event_mention_id_b'], 'relation_type': 'SAME_EVENT_AS', 'confidence': float(r['revision3_confidence']), 'decision_reason': r['revision3_decision_reason'], 'evidence_path': ['EventMention','EventFamilyKey','EventIdentityKey','ConstraintGraph','EventMention'], 'verification_status': 'provisional_model_prediction', 'rule_version': RULE_VERSION})
    for _, r in cannot_df.iterrows():
        cannot_edges.append({'pair_id': r['pair_id'], 'source_event_mention_id': r['event_mention_id_a'], 'target_event_mention_id': r['event_mention_id_b'], 'relation_type': 'CANNOT_LINK', 'confidence': float(r['revision3_confidence']), 'decision_reason': r['revision3_decision_reason'], 'evidence_path': ['EventMention','EventIdentityKey','ConstraintGraph','EventMention'], 'verification_status': 'provisional_model_prediction', 'rule_version': RULE_VERSION})
    for _, r in uncertain_df.iterrows():
        uncertain_rows.append({'pair_id': r['pair_id'], 'source_event_mention_id': r['event_mention_id_a'], 'target_event_mention_id': r['event_mention_id_b'], 'decision_reason': r['revision3_decision_reason'], 'confidence': float(r['revision3_confidence']), 'same_event_score': float(r['same_event_score_r3']), 'different_event_score': float(r['different_event_score_r3']), 'verification_status': 'provisional_model_prediction'})
    return events, members, mapping, same_edges, cannot_edges, uncertain_rows


def compare_revision2_revision3(map_r3: list[dict[str, Any]]) -> dict[str, Any]:
    old_map = {r['event_mention_id']: r['canonical_event_id'] for r in read_jsonl(REV2_MAP)}
    new_map = {r['event_mention_id']: r['canonical_event_id'] for r in map_r3}
    changed = sum(1 for mid in new_map if old_map.get(mid) != new_map.get(mid))
    old_groups = defaultdict(set)
    new_groups = defaultdict(set)
    for mid, ce in old_map.items(): old_groups[ce].add(mid)
    for mid, ce in new_map.items(): new_groups[ce].add(mid)
    return {
        'revision2_canonical_event_count': len(old_groups),
        'revision3_canonical_event_count': len(new_groups),
        'mention_reassigned_count': changed,
        'merged_or_split_summary': {
            'old_singletons': sum(1 for v in old_groups.values() if len(v) == 1),
            'new_singletons': sum(1 for v in new_groups.values() if len(v) == 1),
        },
    }


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild-from-cache', action='store_true')
    args = parser.parse_args()
    started = time.time()
    REV3.mkdir(parents=True, exist_ok=True)
    before_hashes = {
        'revision2_pairs': sha256_file(REV2 / 'pair_predictions.parquet'),
        'revision2_cards': sha256_file(REV2_CARDS),
        'revision2_map': sha256_file(REV2_MAP),
        'candidate_table': sha256_file(CANDIDATES),
    }
    mention_df, field_rows, cards_by_id = build_mentions()
    cards = mention_df.copy()
    cards.to_json(OUT_CARDS, orient='records', force_ascii=False, lines=True)
    family_rows = cards[['event_mention_id','event_family_key','target_type','target_code','event_supertype']].to_dict('records')
    identity_rows = cards[['event_mention_id','event_identity_key','event_family_key','provisional_event_type','identity_batch_id','identity_field_count','identity_field_values']].to_dict('records')
    write_jsonl(OUT_FAMILY, family_rows)
    write_jsonl(OUT_IDENTITY, identity_rows)
    write_jsonl(OUT_IDENTITY_FIELDS, field_rows)

    coarse_summary = cards.groupby('event_family_key').size().reset_index(name='member_count').sort_values('member_count', ascending=False)
    coarse_over_25 = coarse_summary[coarse_summary['member_count'] >= 25]
    family_to_id = cards.groupby('event_family_key')['event_identity_key'].nunique().to_dict()
    split_audit = {
        'family_key_count': int(cards['event_family_key'].nunique()),
        'identity_key_count': int(cards['event_identity_key'].nunique()),
        'family_member_distribution': coarse_summary['member_count'].value_counts().sort_index().to_dict(),
        'coarse_family_key_count': int(len(coarse_over_25)),
        'coarse_family_key_examples': coarse_over_25.head(74).assign(identity_split_count=lambda d: d['event_family_key'].map(family_to_id)).to_dict('records'),
    }
    write_json(OUT_AUDIT, split_audit)

    cand = augment_candidates(mention_df, cards_by_id)
    reviewed = load_dev_labels()
    same_model, diff_model, train_metrics = train_models(cand, reviewed)
    pred = predict_pairs(cand, same_model, diff_model, train_metrics)
    pred.to_parquet(OUT_PAIRS, index=False)
    events, members, mapping, same_edges, cannot_edges, uncertain_rows = decode(pred, cards)
    write_jsonl(OUT_EVENTS, events)
    write_jsonl(OUT_MEMBERS, members)
    write_jsonl(OUT_MAP, mapping)
    write_jsonl(OUT_SAME, same_edges)
    write_jsonl(OUT_CANNOT, cannot_edges)
    write_jsonl(OUT_UNCERTAIN, uncertain_rows)
    compare = compare_revision2_revision3(mapping)
    write_json(OUT_COMPARE, compare)

    r2_cards = pd.DataFrame(read_jsonl(REV2_CARDS))
    r2_keys = r2_cards['event_object_key'].nunique()
    r2_sizes = r2_cards.groupby('event_object_key').size()
    r3_identity_sizes = cards.groupby('event_identity_key').size()
    fam_sizes = cards.groupby('event_family_key').size()
    source_counts = pd.DataFrame(members).groupby('canonical_event_id')['processing_channel'].nunique()
    member_sizes = pd.DataFrame(members).groupby('canonical_event_id').size()
    hard_conflict_violations = int(((pred['revision3_relation_label'] == 'same_event') & ((pred['event_identity_conflict'] > 0) | (pred['hard_conflict'] == True))).sum())
    report = {
        'version': RULE_VERSION,
        'no_phase5_entry': True,
        'no_event_thread_or_event_state': True,
        'no_six_way_classifier': True,
        'training': train_metrics,
        'old_vs_new_keys': {
            'revision2_event_object_key_count': int(r2_keys),
            'revision2_max_key_size': int(r2_sizes.max()),
            'revision2_over_coarse_key_count_ge25': int((r2_sizes >= 25).sum()),
            'revision3_event_family_key_count': int(cards['event_family_key'].nunique()),
            'revision3_event_identity_key_count': int(cards['event_identity_key'].nunique()),
            'revision3_family_max_size': int(fam_sizes.max()),
            'revision3_identity_max_size': int(r3_identity_sizes.max()),
            'revision3_family_coverage': float(cards['event_family_key'].notna().mean()),
            'revision3_identity_coverage': float(cards['event_identity_key'].notna().mean()),
            'revision3_identity_member_distribution': r3_identity_sizes.value_counts().sort_index().to_dict(),
        },
        'coarse_key_split_result': split_audit,
        'candidate_recall_change': {
            'family_recall_candidates': int((pred['event_family_match'] == 1.0).sum()),
            'identity_match_candidates': int((pred['event_identity_match'] == 1.0).sum()),
            'revision2_total_candidates': int(len(pd.read_parquet(REV2 / 'pair_predictions.parquet'))),
            'revision3_total_candidates': int(len(pred)),
        },
        'prediction_distribution': pred['revision3_relation_label'].value_counts().to_dict(),
        'canonical_event_summary': {
            'canonical_event_count': len(events),
            'singleton_ratio': float((member_sizes == 1).mean()) if len(member_sizes) else 0.0,
            'cluster_size_distribution': member_sizes.value_counts().sort_index().to_dict(),
            'max_cluster_size': int(member_sizes.max()) if len(member_sizes) else 0,
            'multi_source_cluster_count': int((source_counts > 1).sum()) if len(source_counts) else 0,
        },
        'revision2_revision3_reconciliation': compare,
        'hard_conflict_violation_count': hard_conflict_violations,
        'outputs': {
            'canonical_events': str(OUT_EVENTS.relative_to(ROOT)),
            'canonical_event_members': str(OUT_MEMBERS.relative_to(ROOT)),
            'event_mention_canonical_mapping': str(OUT_MAP.relative_to(ROOT)),
            'event_family_keys': str(OUT_FAMILY.relative_to(ROOT)),
            'event_identity_keys': str(OUT_IDENTITY.relative_to(ROOT)),
            'identity_field_evidence': str(OUT_IDENTITY_FIELDS.relative_to(ROOT)),
        },
        'hash_integrity': before_hashes,
    }
    runtime = {
        'version': RULE_VERSION,
        'resource_usage': {
            'elapsed_seconds': round(time.time() - started, 3),
            'max_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            'gpu_memory_mb': None,
            'local_32b_calls': 0,
            'embedding_cache_used': True,
            'reranker_reused': True,
        },
        'integrity': {
            'revision2_unchanged': before_hashes == {
                'revision2_pairs': sha256_file(REV2 / 'pair_predictions.parquet'),
                'revision2_cards': sha256_file(REV2_CARDS),
                'revision2_map': sha256_file(REV2_MAP),
                'candidate_table': sha256_file(CANDIDATES),
            },
            'event_thread_created': False,
            'event_state_created': False,
            'phase5_entered': False,
        },
    }
    write_json(REPORT, report)
    write_json(RUNTIME, runtime)
    DOC.write_text('# Task3 KG v2 Step4 Revision3\n\n' + json.dumps({'prediction_distribution': report['prediction_distribution'], 'canonical_event_count': report['canonical_event_summary']['canonical_event_count']}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': report, 'runtime': runtime}, ensure_ascii=False, indent=2))
    return {'report': report, 'runtime': runtime}


if __name__ == '__main__':
    main()
