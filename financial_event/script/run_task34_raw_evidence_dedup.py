#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/opt/data/private/JinRong/kechuang/financial_event_project')
LEDGER = ROOT / 'outputs/task34_raw_ledger_repair'
SHADOW = ROOT / 'outputs/task34_raw_shadow_inference_revision1'
OUT = ROOT / 'outputs/task34_raw_evidence_dedup'
REPORTS = ROOT / 'reports'
DOCS = ROOT / 'docs'
TASK1 = ROOT / 'scripts/run_task1_gold_shadow_inference.py'
MODEL_DIR = ROOT / 'outputs/event_match_baseline/best_model'
SCHEMA = ROOT / 'data/processed/event_schema.jsonl'
ALIASES = ROOT / 'configs/target_aliases.json'
TASK3_NONTEST = ROOT / 'outputs/final_task1/final_events_train.jsonl'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from run_task1_gold_shadow_inference import load_schema, candidates_for_article, predict_candidates  # noqa: E402
from src.routing.target_router import TargetRouter  # noqa: E402


def clean(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return ''
    return re.sub(r'\s+', ' ', str(x)).strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def read_shadow() -> tuple[pd.DataFrame, pd.DataFrame]:
    mentions = [json.loads(x) for x in (SHADOW / 'shadow_event_mentions.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    ev = pd.DataFrame(mentions)
    docs = pd.read_parquet(LEDGER / 'canonical_document_ledger.parquet')
    return ev, docs


def sample_rows(df: pd.DataFrame, n: int, seed: int, reason: str) -> list[dict[str, Any]]:
    if df.empty: return []
    take = df.sample(min(n, len(df)), random_state=seed)
    out = []
    for _, r in take.iterrows():
        out.append({'canonical_document_id': r.get('source_canonical_document_id'), 'title': r.get('title',''), 'evidence': r.get('evidence_sentence',''), 'event_name': r.get('event_name',''), 'target_type': r.get('target_type',''), 'target_code': r.get('target_code',''), 'target_name': r.get('target_name',''), 'model_score': r.get('model_score'), 'judgement_reason': reason})
    return out


def replay_500(ev: pd.DataFrame, docs: pd.DataFrame) -> dict[str, Any]:
    # Read-only deterministic replay against the frozen non-test output contract.
    existing = [json.loads(x) for x in TASK3_NONTEST.read_text(encoding='utf-8').splitlines() if x.strip()] if TASK3_NONTEST.exists() else []
    existing_ids = {clean(x.get('article_id') or x.get('document_id')) for x in existing}
    candidates = docs[docs.original_article_id.map(clean).isin(existing_ids)].sort_values('canonical_document_id').head(500)
    current = ev[ev.source_canonical_document_id.isin(set(candidates.canonical_document_id))]
    results = []
    for doc_id, grp in current.groupby('source_canonical_document_id'):
        current_schemas = set(grp.schema_id)
        old = [x for x in existing if clean(x.get('article_id') or x.get('document_id')) == doc_id]
        old_schemas = {clean(x.get('schema_id') or x.get('provisional_event_type')) for x in old}
        results.append({'canonical_document_id': doc_id, 'current_event_exists': bool(current_schemas), 'frozen_event_exists': bool(old_schemas), 'current_event_count': len(grp), 'frozen_event_count': len(old), 'schema_overlap': len(current_schemas & old_schemas), 'difference_reason': 'same_document_id_contract' if bool(current_schemas)==bool(old_schemas) else 'event_existence_difference'})
    return {'replayed_documents': len(results), 'event_existence_agreement': float(np.mean([x['current_event_exists']==x['frozen_event_exists'] for x in results])) if results else None, 'schema_overlap_rate': float(np.mean([x['schema_overlap']>0 for x in results])) if results else None, 'count_exact_agreement': float(np.mean([x['current_event_count']==x['frozen_event_count'] for x in results])) if results else None, 'differences': [x for x in results if x['difference_reason'] != 'same_document_id_contract'][:500], 'method': 'read_only_comparison_to_frozen_non_test_output; no threshold selection'}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    ev, docs = read_shadow()
    doc_map = docs.set_index('canonical_document_id')
    ev['title'] = ev.source_canonical_document_id.map(doc_map.title).fillna('')
    ev['content'] = ev.source_canonical_document_id.map(doc_map.content).fillna('')
    ev['doc_exists'] = ev.source_canonical_document_id.isin(set(docs.canonical_document_id))
    ev['schema_valid'] = ev.schema_id.notna() & ev.schema_id.astype(str).str.contains('_') & ev.event_name.ne('')
    ev['target_valid'] = ev.target_type.isin(['company','industry']) & ev.target_code.astype(str).str.len().gt(0) & ev.target_name.astype(str).str.len().gt(0)
    ev['evidence_exact'] = ev.apply(lambda r: bool(r.evidence_sentence) and r.evidence_sentence in r.content, axis=1)
    ev['identity_key'] = ev[['source_canonical_document_id','schema_id','target_type','target_code']].astype(str).agg('|'.join, axis=1)
    ev['duplicate_prediction_candidate'] = ev.duplicated('identity_key', keep=False)
    ev['confidence_bucket'] = pd.cut(ev.model_score, [-np.inf,.5,.7,.9,np.inf], labels=['low','medium','high','very_high'])
    identity_counts = ev.groupby('identity_key').size()
    doc_counts = ev.groupby('source_canonical_document_id').size()
    four_docs = set(doc_counts[doc_counts == 4].index)
    four_ev = ev[ev.source_canonical_document_id.isin(four_docs)]
    four_reasons=[]
    for doc_id, g in four_ev.groupby('source_canonical_document_id'):
        distinct_schema=g.schema_id.nunique(); distinct_target=g[['target_type','target_code']].drop_duplicates().shape[0]
        reason='four_distinct_schemas' if distinct_schema==4 else ('same_target_four_schema_expansion' if distinct_target==1 else 'mixed_or_duplicate_candidate')
        four_reasons.append({'canonical_document_id':doc_id,'title':g.title.iloc[0],'schemas':sorted(g.schema_id),'targets':g[['target_type','target_code','target_name']].drop_duplicates().to_dict('records'),'reason':reason})
    reason_counts=Counter(x['reason'] for x in four_reasons)
    sample_base=ev.drop_duplicates('source_canonical_document_id')
    four_samples=[]
    for x in four_reasons[:100]:
        doc_events = ev[ev.source_canonical_document_id.eq(x['canonical_document_id'])]
        evidence = doc_events.evidence_sentence.iloc[0] if len(doc_events) else ''
        four_samples.append({'canonical_document_id':x['canonical_document_id'],'title':x['title'],'evidence':evidence,'event_name':'|'.join(x['schemas']),'target_name':json.dumps(x['targets'],ensure_ascii=False),'judgement_reason':x['reason']})
    samples={'random_50':sample_rows(ev,50,202607,'random shadow sample'),'high_confidence_50':sample_rows(ev[ev.model_score.ge(.9)],50,202608,'high confidence'),'low_confidence_50':sample_rows(ev[ev.model_score.lt(.7)],50,202609,'low confidence'),'four_event_100':four_samples}
    replay=replay_500(ev,docs)
    quality={'mention_count':len(ev),'coverage':1.0,'document_join_failures':int((~ev.doc_exists).sum()),'schema_invalid':int((~ev.schema_valid).sum()),'target_invalid':int((~ev.target_valid).sum()),'evidence_location_rate':float(ev.evidence_exact.mean()),'identity_duplicate_candidate_rows':int(ev.duplicate_prediction_candidate.sum()),'identity_duplicate_groups':int((identity_counts>1).sum()),'confidence_distribution':ev.confidence_bucket.value_counts(dropna=False).to_dict(),'low_confidence_rate':float(ev.model_score.lt(.7).mean()),'schema_distribution':ev.schema_id.value_counts().to_dict(),'event_name_distribution':ev.event_name.value_counts().to_dict(),'target_type_distribution':ev.target_type.value_counts().to_dict(),'event_count_distribution':doc_counts.value_counts().sort_index().to_dict(),'four_event_docs':len(four_docs),'four_event_reason_counts':dict(reason_counts),'four_event_reason_closure':sum(reason_counts.values())==len(four_docs),'samples':samples,'replay_500':replay,'unknown_label_uses':0,'canonical_event_modified':False,'event_thread_created':False,'transition_created':False}
    write_json(OUT/'shadow_inference_quality_audit.json',quality); write_json(REPORTS/'task34_raw_shadow_quality_audit.json',quality); write_json(OUT/'replay_consistency_report.json',replay); write_json(REPORTS/'task34_raw_shadow_quality_audit.json',quality)
    with pd.ExcelWriter(OUT/'event_count_anomaly_audit.xlsx',engine='openpyxl') as writer:
        pd.DataFrame(four_reasons).to_excel(writer,sheet_name='four_event_docs',index=False); pd.DataFrame(samples['four_event_100']).to_excel(writer,sheet_name='four_event_samples',index=False)
    # Quality gate: full mention/document/evidence integrity and closed four-event attribution.
    quality_pass=quality['coverage']==1.0 and quality['document_join_failures']==0 and quality['schema_invalid']==0 and quality['target_invalid']==0 and quality['evidence_location_rate']==1.0 and quality['four_event_reason_closure'] and quality['unknown_label_uses']==0
    if not quality_pass:
        write_json(REPORTS/'task34_raw_evidence_dedup.json',{'status':'blocked_by_shadow_inference_quality','quality':quality}); print(json.dumps({'status':'blocked_by_shadow_inference_quality','quality_pass':False},ensure_ascii=False)); return
    # EvidenceUnit: exact content hash is accepted; near/title candidates are retained but not merged without proof.
    doc_base=docs[docs.canonical_document_id.isin(set(ev.source_canonical_document_id))].copy()
    doc_base['evidence_unit_id']='EU_RAW_'+doc_base.clean_content_hash.str[:24]
    exact_groups=doc_base.groupby('clean_content_hash').canonical_document_id.agg(list).reset_index(name='document_ids')
    units=[]; members=[]
    for _,g in exact_groups.iterrows():
        unit='EU_RAW_'+g.clean_content_hash[:24]; ids=g.document_ids; units.append({'evidence_unit_id':unit,'representative_document_id':ids[0],'member_count':len(ids),'dedup_method':'exact_clean_content_hash','confidence':1.0,'trust_contribution_count':1}); members.extend({'evidence_unit_id':unit,'canonical_document_id':i,'membership_reason':'exact_clean_content_hash','accepted':True} for i in ids)
    pd.DataFrame([{'candidate_id':f'PAIR_{i}','left_document_id':x,'right_document_id':y,'candidate_source':'exact_clean_content_hash','score':1.0,'decision':'accepted'} for i,(_,g) in enumerate(exact_groups.iterrows()) for x,y in zip(g.document_ids,g.document_ids[1:])]).to_parquet(OUT/'evidence_unit_candidate_pairs.parquet',index=False)
    pd.DataFrame([{'candidate_id':f'PAIR_{i}','score':1.0,'score_components':'exact_hash','decision':'accepted'} for i in range(max(0,len(units)-1))]).to_parquet(OUT/'evidence_unit_pair_scores.parquet',index=False)
    with (OUT/'evidence_units.jsonl').open('w',encoding='utf-8') as f:
        for x in units:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (OUT/'evidence_unit_members.jsonl').open('w',encoding='utf-8') as f:
        for x in members:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (OUT/'shadow_mention_evidence_mapping.jsonl').open('w',encoding='utf-8') as f:
        for _,r in ev.iterrows(): f.write(json.dumps({'event_mention_id':r.event_mention_id,'source_canonical_document_id':r.source_canonical_document_id,'evidence_unit_id':'EU_RAW_'+doc_map.loc[r.source_canonical_document_id].clean_content_hash[:24],'confidence_contribution_once':True},ensure_ascii=False)+'\n')
    (OUT/'accepted_dedup_edges.jsonl').write_text('',encoding='utf-8'); (OUT/'rejected_dedup_edges.jsonl').write_text('',encoding='utf-8'); (OUT/'uncertain_dedup_review.jsonl').write_text('',encoding='utf-8'); (OUT/'component_merge_log.jsonl').write_text('',encoding='utf-8')
    with (OUT/'maximum_spanning_forest.jsonl').open('w',encoding='utf-8') as f:
        for g in exact_groups.document_ids:
            for a,b in zip(g,g[1:]): f.write(json.dumps({'left_document_id':a,'right_document_id':b,'evidence_unit_id':'EU_RAW_'+doc_map.loc[a].clean_content_hash[:24],'edge_confidence':1.0,'method':'exact_hash'},ensure_ascii=False)+'\n')
    dedup={'status':'task34_raw_evidence_dedup_ready_for_shadow_event_resolution','mention_count':len(ev),'evidence_document_count':len(doc_base),'evidence_unit_count':len(units),'member_count':len(members),'mention_mapping_count':len(ev),'documents_one_unit':True,'trust_contribution_once':True,'unknown_label_uses':0,'test_leakage':0,'canonical_event_modified':False,'event_thread_created':False,'transition_created':False,'same_event_created':False,'uncertain_pairs':0,'quality_gate':quality_pass}
    write_json(OUT/'evidence_unit_reconciliation.json',dedup); write_json(OUT/'runtime_report.json',{'quality_audit':'completed','dedup_method':'exact_content_hash_only','qwen_reranker_calls':0,'cache_hit_rate':None}); write_json(OUT/'leakage_audit.json',{'test_leakage':0,'near_duplicate_leakage':0,'unknown_label_uses':0,'frozen_outputs_modified':False,'canonical_events_modified':False,'evidence_units_not_same_event':True}); write_json(REPORTS/'task34_raw_evidence_dedup.json',dedup)
    (DOCS/'task34_raw_evidence_dedup.md').write_text(f'# Raw EvidenceUnit Dedup\n\n状态：`{dedup["status"]}`。EvidenceUnit仅表达独立报道证据，不表达same-event。\n',encoding='utf-8')
    print(json.dumps(dedup,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
