#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/opt/data/private/JinRong/kechuang/financial_event_project')
OUT = ROOT / 'outputs/task34_raw_shadow_inference'
REPORTS = ROOT / 'reports'
LEDGER = ROOT / 'outputs/task34_raw_ledger_repair'
TASK4 = ROOT / 'outputs/task4/stage2/task4_event_level_samples.parquet'
TASK3_DOCS = ROOT / 'outputs/task3/normalized_documents.jsonl'
TEST = ROOT / 'data/processed/labeled/news_test.jsonl'
MODEL_DIR = ROOT / 'outputs/event_match_baseline/best_model'
SCHEMA = ROOT / 'data/processed/event_schema.jsonl'
ALIASES = ROOT / 'configs/target_aliases.json'
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SCRIPTS))
from run_task1_gold_shadow_inference import load_schema, candidates_for_article, predict_candidates, build_event_mentions_from_predictions, confidence_level  # noqa: E402
from src.routing.target_router import TargetRouter  # noqa: E402


def clean(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return ''
    return re.sub(r'\s+', ' ', str(x)).strip()


def h(x: str) -> str:
    return hashlib.sha256(x.encode('utf-8')).hexdigest() if x else ''


def read_jsonl(path: Path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def text_hash(title: Any, content: Any) -> str:
    return h(clean(content))


def sentence(content: str) -> str:
    parts = re.split(r'(?<=[。！？])', content)
    return next((p.strip() for p in parts if p.strip()), content[:500])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    canon = pd.read_parquet(LEDGER / 'canonical_document_ledger.parquet')
    mapping = pd.read_parquet(LEDGER / 'raw_to_document_mapping.parquet')
    raw = pd.read_parquet(LEDGER / 'raw_record_ledger.parquet', columns=['raw_record_id','canonical_document_id','original_article_id','file_path','sheet_name','source_row_number','test_overlap'])
    test4 = pd.read_parquet(TASK4, columns=['article_id','title','content','clean_content_hash','official_split'])
    test_ids = set(test4.loc[test4.official_split.eq('test'), 'article_id'].map(clean))
    test_hashes = set(test4.loc[test4.official_split.eq('test'), 'clean_content_hash'].dropna().map(clean))
    test_hashes.discard('')
    raw['test_exact_id'] = raw.original_article_id.map(clean).isin(test_ids)
    raw['raw_content_hash'] = raw.canonical_document_id.map(canon.set_index('canonical_document_id').clean_content_hash)
    raw['test_exact_hash'] = raw.raw_content_hash.isin(test_hashes)
    near = pd.read_parquet(LEDGER / 'near_duplicate_candidates.parquet')
    near_raw = set()
    for x in near.raw_record_ids:
        near_raw.update(list(x))
    near_test_raw = set(raw.loc[raw.raw_record_id.isin(near_raw) & raw.test_exact_id, 'raw_record_id'])
    near_test_titles = set(near.loc[near.raw_record_ids.map(lambda x: any(i in near_test_raw for i in x)), 'title']) if len(near) else set()
    canon_to_raw = raw.groupby('canonical_document_id').agg(raw_rows=('raw_record_id','size'), original_ids=('original_article_id',lambda x: sorted(set(map(clean,x)))), source_files=('file_path',lambda x: sorted(set(x))), titles=('raw_record_id','size')).reset_index()
    canon = canon.merge(canon_to_raw, on='canonical_document_id', how='left')
    canon['test_exact_id'] = canon.original_ids.map(lambda xs: bool(set(xs) & test_ids) if isinstance(xs, list) else False)
    canon['test_exact_hash'] = canon.clean_content_hash.isin(test_hashes)
    canon['test_near_duplicate'] = canon.title.map(clean).isin(near_test_titles)
    existing3_ids = set()
    for row in read_jsonl(TASK3_DOCS): existing3_ids.add(clean(row.get('article_id') or row.get('document_id')))
    canon['existing_task3'] = canon.original_ids.map(lambda xs: bool(set(xs) & existing3_ids) if isinstance(xs, list) else False)
    canon['existing_frozen_task1'] = canon.existing_task3
    canon['eligibility_status'] = 'eligible_new_document'
    canon.loc[canon.content.map(clean).eq(''), 'eligibility_status'] = 'insufficient_text'
    canon.loc[canon.test_exact_id | canon.test_exact_hash, 'eligibility_status'] = 'excluded_task4_test_exact'
    canon.loc[canon.test_near_duplicate & ~canon.test_exact_id & ~canon.test_exact_hash, 'eligibility_status'] = 'excluded_task4_test_near_duplicate'
    canon.loc[canon.existing_frozen_task1 & canon.eligibility_status.eq('eligible_new_document'), 'eligibility_status'] = 'existing_frozen_task1_document'
    canon.loc[canon.existing_task3 & canon.eligibility_status.eq('eligible_new_document'), 'eligibility_status'] = 'existing_task3_document'
    # Quarantine is keyed by canonical only when a surviving canonical has a conflict; all conflict rows are excluded from inference.
    quarantine_ids = set(mapping.loc[mapping.mapping_status.eq('quarantined_row'), 'canonical_document_id'].dropna())
    canon.loc[canon.canonical_document_id.isin(quarantine_ids), 'eligibility_status'] = 'quarantined_conflict'
    assert canon.eligibility_status.notna().all() and canon.canonical_document_id.is_unique
    canon[['canonical_document_id','eligibility_status','clean_content_hash','title','content','publish_time','source','original_ids','raw_rows']].to_parquet(OUT / 'inference_eligibility_ledger.parquet', index=False)
    eligible = canon[canon.eligibility_status.eq('eligible_new_document')].copy()
    eligible[['canonical_document_id','clean_content_hash','title','content','publish_time','source','original_ids']].to_parquet(OUT / 'eligible_document_manifest.parquet', index=False)
    overlap = {'task4_test_unique_article_ids': len(test_ids), 'test_exact_raw_rows': int(raw.test_exact_id.sum()), 'test_exact_canonical_documents': int(canon.test_exact_id.sum()), 'test_clean_hash_canonical_documents': int(canon.test_exact_hash.sum()), 'test_near_duplicate_canonical_documents': int(canon.test_near_duplicate.sum()), 'test_exact_or_near_excluded_from_eligible': int((canon.eligibility_status.isin(['excluded_task4_test_exact','excluded_task4_test_near_duplicate'])).sum()), 'eligible_test_overlap': int(eligible.canonical_document_id.isin(set(canon.loc[canon.test_exact_id | canon.test_exact_hash | canon.test_near_duplicate,'canonical_document_id'])).sum())}
    (OUT / 'task4_test_overlap_reconciliation.json').write_text(json.dumps(overlap, ensure_ascii=False, indent=2), encoding='utf-8')

    # Frozen Task1 model contract; no raw labels are loaded into article records.
    schema_by_id, schemas_by_target = load_schema(); router = TargetRouter(schema_file=str(SCHEMA), alias_file=str(ALIASES))
    started = time.perf_counter(); records=[]; mentions=[]; unresolved=[]; model_batches=0; candidate_total=0; reranker_calls=0
    model_error = None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR)); model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device).eval()
    except Exception as exc:
        model = None
        model_error = f'{type(exc).__name__}: {exc}'
    for _, row in eligible.iterrows():
        doc_id=row.canonical_document_id; article={'article_id':doc_id,'title':row.title,'content':row.content,'publish_time':str(row.publish_time),'source':row.source}
        try:
            if model is None:
                raise RuntimeError(f'frozen Task1 runtime unavailable: {model_error}')
            routes=router.route(article['title'],article['content'])
            candidates=candidates_for_article(article,router,schemas_by_target)
            candidate_total += len(candidates)
            # Frozen Task1 contract: model forward over all routed schema candidates.
            predicted, calls=predict_candidates(candidates,torch,tokenizer,model,device,512,32,0.38); model_batches += calls
            events=[]
            for ev in predicted:
                if int(ev.get('prediction',0)):
                    events.append({'candidate_id':ev['candidate_id'],'schema_id':ev['schema_id'],'event_name':schema_by_id.get(str(ev['schema_id']),{}).get('event_name',''),'target_type':ev.get('target_type'),'target_code':ev.get('target_code'),'target_name':ev.get('target_name'),'evidence_sentence':sentence(article['content']),'model_score':ev.get('probability_positive'),'model_score_negative':ev.get('probability_negative'),'confidence_level':confidence_level(float(ev.get('probability_positive',0))),'inference_provenance':'frozen_task1_inference','source_canonical_document_id':doc_id,'node_type':'ShadowEventMention','verification_status':'provisional_model_prediction','label_source':'frozen_task1_inference','same_event_created':False,'transition_created':False,'event_thread_created':False})
            status='success_with_event' if events else 'success_no_event'
            records.append({'canonical_document_id':doc_id,'processing_status':status,'candidate_count':len(candidates),'predicted_event_count':len(events),'route_count':len(routes),'reranker_called':False,'raw_unknown_label_uses':0,'runtime_seconds':None})
            mentions.extend(events)
        except Exception as exc:
            records.append({'canonical_document_id':doc_id,'processing_status':'failed','candidate_count':None,'predicted_event_count':0,'route_count':None,'reranker_called':False,'raw_unknown_label_uses':0,'runtime_seconds':None})
            unresolved.append({'canonical_document_id':doc_id,'status':'unresolved','reason':f'{type(exc).__name__}: {exc}'})
    elapsed=time.perf_counter()-started
    # Stable deterministic mention IDs and output ordering.
    for i, ev in enumerate(sorted(mentions,key=lambda x:(x['source_canonical_document_id'],x['schema_id'])),1):
        ev['event_mention_id']='SHADOW_EM_' + hashlib.sha256(f"{ev['source_canonical_document_id']}|{ev['schema_id']}|{ev['target_type']}|{ev['target_code']}".encode()).hexdigest()[:24]
    records_df=pd.DataFrame(records).sort_values('canonical_document_id')
    records_df.to_parquet(OUT/'document_inference_ledger.parquet',index=False)
    with (OUT/'shadow_event_mentions.jsonl').open('w',encoding='utf-8') as f:
        for ev in sorted(mentions,key=lambda x:x['event_mention_id']): f.write(json.dumps(ev,ensure_ascii=False)+'\n')
    with (OUT/'no_event_documents.jsonl').open('w',encoding='utf-8') as f:
        for r in records_df[records_df.processing_status.eq('success_no_event')].to_dict('records'): f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')
    with (OUT/'unresolved_queue.jsonl').open('w',encoding='utf-8') as f:
        for r in unresolved: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    statuses=records_df.processing_status.value_counts().to_dict(); counts=canon.eligibility_status.value_counts().to_dict()
    provenance={'pipeline':'frozen_task1_inference','model_dir':str(MODEL_DIR.relative_to(ROOT)),'threshold':0.38,'raw_unknown_label_uses':0,'reranker_calls':reranker_calls,'model_forward_batches':model_batches,'candidate_count':candidate_total}
    (OUT/'inference_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    runtime={'eligible_documents':len(eligible),'elapsed_seconds':elapsed,'p50_seconds_per_document':float(records_df.runtime_seconds.dropna().median()) if records_df.runtime_seconds.notna().any() else None,'p95_seconds_per_document':None,'reranker_call_rate':0.0,'cache_hit_rate':None,'cpu_gpu_peak':'not instrumented','cold_start_seconds':None,'model_forward_batches':model_batches,'model_runtime_error':model_error}
    (OUT/'runtime_report.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding='utf-8')
    leakage={'test_overlap_in_eligible':overlap['eligible_test_overlap'],'test_exact_or_near_excluded':True,'raw_unknown_label_uses':0,'same_event_created':False,'transition_created':False,'event_thread_created':False,'canonical_events_modified':False,'evidence_units_created':False}
    (OUT/'leakage_audit.json').write_text(json.dumps(leakage,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'inputs':['canonical_document_ledger.parquet','raw_to_document_mapping.parquet','near_duplicate_candidates.parquet','quarantine_queue.jsonl'],'eligible_count':len(eligible),'formal_outputs_modified':False,'previous_175441_shadow_adopted':False}
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    eligible_coverage=len(records_df)/len(eligible) if len(eligible) else 1
    status='task34_raw_shadow_inference_ready_for_evidence_dedup' if all([canon.canonical_document_id.is_unique,overlap['eligible_test_overlap']==0,eligible_coverage==1,not records_df.processing_status.eq('failed').any(),len(mentions)!=len(eligible),provenance['raw_unknown_label_uses']==0]) else 'task34_raw_shadow_inference_blocked'
    summary={'status':status,'eligibility_counts':counts,'inference_status_counts':statuses,'eligible_documents':len(eligible),'shadow_event_mentions':len(mentions),'event_count_distribution':Counter(records_df.predicted_event_count).copy(),'company_industry_distribution':Counter(x.get('target_type') for x in mentions),'test_overlap':overlap,'eligible_coverage':eligible_coverage,'no_evidence_unit_created':True,'previous_mechanical_shadow_adopted':False,'runtime':runtime}
    for fn,obj in [('task34_raw_inference_eligibility.json',{'status':status,'eligibility_counts':counts,'test_overlap':overlap}),('task34_raw_shadow_inference.json',summary)]: (REPORTS/fn).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=dict),encoding='utf-8')
    (ROOT/'docs/task34_raw_shadow_inference.md').write_text(f'# Raw Shadow Inference\n\n状态：`{status}`。本轮仅对eligible文档运行冻结Task1，未创建EvidenceUnit、same-event、transition或EventThread。\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=dict))

if __name__=='__main__': main()
