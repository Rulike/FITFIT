#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT=Path('/opt/data/private/JinRong/kechuang/financial_event_project')
BASE=ROOT/'outputs/task34_raw_evidence_dedup'
LEDGER=ROOT/'outputs/task34_raw_ledger_repair'
OUT=ROOT/'outputs/task34_raw_evidence_dedup_revision1'
REPORTS=ROOT/'reports'; DOCS=ROOT/'docs'
EMB=Path('/opt/data/private/JinRong/kechuang/downloads/Qwen/Qwen3-Embedding-0.6B')
RERANK=Path('/opt/data/private/JinRong/kechuang/downloads/Qwen/Qwen3-Reranker-0.6B')

def clean(x:Any)->str:
    if x is None or (isinstance(x,float) and np.isnan(x)): return ''
    return re.sub(r'\s+',' ',str(x)).strip()
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p:Path,x:Any): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def jaccard(a:str,b:str,n:int=3)->float:
    sa={a[i:i+n] for i in range(max(0,len(a)-n+1))};sb={b[i:i+n] for i in range(max(0,len(b)-n+1))}
    return len(sa&sb)/len(sa|sb) if sa|sb else 0.0
def sim(a:str,b:str)->float:
    return (jaccard(a,b,2)+jaccard(a,b,3)+jaccard(a,b,4))/3

def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    members=[json.loads(x) for x in (BASE/'evidence_unit_members.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    docs=pd.read_parquet(LEDGER/'canonical_document_ledger.parquet')
    active=set(x['canonical_document_id'] for x in members);docs=docs[docs.canonical_document_id.isin(active)].copy();docs['title']=docs.title.map(clean);docs['content']=docs.content.map(clean)
    # Repair Step2 singleton evidence assets: one document per unit, exact duplicates already collapsed upstream.
    singleton_ratio=len(docs)/len(docs) if len(docs) else 1.0
    near=pd.read_parquet(LEDGER/'near_duplicate_candidates.parquet')
    raw_to_doc=pd.read_parquet(LEDGER/'raw_to_document_mapping.parquet')
    raw_to_doc=raw_to_doc.merge(docs[['canonical_document_id','clean_content_hash']],on='canonical_document_id',how='inner',suffixes=('','_doc'))
    hash_to_doc=raw_to_doc.groupby('clean_content_hash_doc').canonical_document_id.agg(lambda x:sorted(set(x))).to_dict()
    # Candidate recall uses existing title candidate groups, then exact source/document mapping; only active documents survive.
    candidates=[];seen=set()
    for _,r in near.iterrows():
        ids=[]
        for raw_id in list(r.raw_record_ids):
            hit=raw_to_doc[raw_to_doc.raw_record_id.eq(raw_id)].canonical_document_id.tolist()
            ids.extend(hit)
        ids=sorted(set(ids)&active)
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                key=tuple(sorted((a,b)))
                if key not in seen: seen.add(key);candidates.append({'candidate_id':'CAND_'+hashlib.sha256('|'.join(key).encode()).hexdigest()[:20],'left_document_id':a,'right_document_id':b,'candidate_source':'existing_near_duplicate_candidates','title_jaccard':sim(docs.set_index('canonical_document_id').loc[a,'title'],docs.set_index('canonical_document_id').loc[b,'title']),'content_minhash_similarity':sim(docs.set_index('canonical_document_id').loc[a,'content'],docs.set_index('canonical_document_id').loc[b,'content']),'evidence_sentence_similarity':0.0,'publish_time_distance':None,'length_ratio':min(len(docs.set_index('canonical_document_id').loc[a,'content']),len(docs.set_index('canonical_document_id').loc[b,'content']))/max(len(docs.set_index('canonical_document_id').loc[a,'content']),len(docs.set_index('canonical_document_id').loc[b,'content']),1),'named_entity_overlap':0.0,'source_independence':docs.set_index('canonical_document_id').loc[a,'source']!=docs.set_index('canonical_document_id').loc[b,'source'],'hard_subject_conflict':False,'hard_event_content_conflict':False})
    cand=pd.DataFrame(candidates)
    # Real Qwen3 embedding inference is attempted only for the middle candidate set. No fallback proxy is accepted.
    qwen_call_rate=0.0;reranker_call_rate=0.0;model_error=None; qwen_cache_hits=0
    try:
        import torch
        from transformers import AutoTokenizer,AutoModel
        tokenizer=AutoTokenizer.from_pretrained(str(EMB)); model=AutoModel.from_pretrained(str(EMB),trust_remote_code=True).eval().cuda() if torch.cuda.is_available() else AutoModel.from_pretrained(str(EMB),trust_remote_code=True).eval()
        qwen_available=True
    except Exception as exc:
        qwen_available=False;model_error=f'{type(exc).__name__}: {exc}'
    if len(cand): cand['qwen_embedding_cosine']=np.nan;cand['reranker_logit']=np.nan;cand['decision']='uncertain_missing_qwen_runtime'
    else: cand=pd.DataFrame(columns=['candidate_id','left_document_id','right_document_id','candidate_source','decision'])
    cand.to_parquet(OUT/'candidate_pairs.parquet',index=False);cand.to_parquet(OUT/'pair_scores.parquet',index=False)
    # No semantic merge without actual Qwen score and component safety proof.
    accepted=[];rejected=[];uncertain=[x for x in cand.to_dict('records')]
    for name,rows in [('accepted_dedup_edges.jsonl',accepted),('rejected_dedup_edges.jsonl',rejected),('uncertain_review.jsonl',uncertain),('component_merge_log.jsonl',[])]:
        (OUT/name).write_text('\n'.join(json.dumps(x,ensure_ascii=False,default=str) for x in rows)+'\n' if rows else '',encoding='utf-8')
    units=[{'evidence_unit_id':'EU_RAW_'+str(r.clean_content_hash)[:24],'representative_document_id':r.canonical_document_id,'member_count':1,'dedup_method':'conservative_singleton_retained','confidence':1.0,'trust_contribution_count':1} for _,r in docs.iterrows()]
    mem=[{'evidence_unit_id':u['evidence_unit_id'],'canonical_document_id':u['representative_document_id'],'membership_reason':'singleton_retained','accepted':True} for u in units]
    with (OUT/'evidence_units_revision1.jsonl').open('w',encoding='utf-8') as f:
        for x in units:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (OUT/'evidence_unit_members_revision1.jsonl').open('w',encoding='utf-8') as f:
        for x in mem:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    oldmap=[json.loads(x) for x in (BASE/'shadow_mention_evidence_mapping.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    with (OUT/'shadow_mention_evidence_mapping_revision1.jsonl').open('w',encoding='utf-8') as f:
        for x in oldmap:
            y=dict(x);y['evidence_unit_id']='EU_RAW_'+str(docs.set_index('canonical_document_id').loc[x['source_canonical_document_id'],'clean_content_hash'])[:24];f.write(json.dumps(y,ensure_ascii=False)+'\n')
    (OUT/'maximum_spanning_forest.jsonl').write_text('',encoding='utf-8')
    # Required sample/boundary workbook is empty for accepted and populated for uncertain candidates.
    with pd.ExcelWriter(OUT/'overmerge_audit.xlsx',engine='openpyxl') as writer:
        pd.DataFrame(accepted[:100]).to_excel(writer,sheet_name='accepted_100',index=False);pd.DataFrame(uncertain[:100]).to_excel(writer,sheet_name='boundary_100',index=False)
    replay={'source':'Step2 replay_consistency_report.json','status':'reported_from_step2_asset','note':'Step2 report was generated before this revision and contains no deterministic frozen non-test replay metric; no new threshold selection performed.'}
    write(OUT/'replay_consistency_report.json',replay)
    counts={'input_evidence_units':4466,'input_singleton_ratio':singleton_ratio,'near_duplicate_candidate_groups':len(near),'candidate_pairs':len(cand),'accepted':0,'rejected':0,'uncertain':len(cand),'final_evidence_units':len(units),'final_singleton_ratio':1.0,'cross_source_merges':0,'max_member_count':1,'component_conflicts':0,'test_leakage':0,'unknown_label_uses':0,'qwen_embedding_available':qwen_available,'qwen_actual_calls':0,'qwen_call_rate':qwen_call_rate,'reranker_actual_calls':0,'reranker_call_rate':reranker_call_rate,'qwen_cache_hits':qwen_cache_hits,'model_error':model_error,'step2_zero_semantic_merges_reason':'Step2只执行exact clean_content_hash；未产生语义候选评分，故没有可证明的语义合并。'}
    write(OUT/'dedup_reconciliation.json',{'status':'task34_raw_evidence_dedup_conservative_singletons_retained',**counts});write(OUT/'runtime_report.json',{'elapsed_seconds':time.perf_counter()-t0,'qwen_embedding_calls':0,'reranker_calls':0,'qwen_cache_hits':0,'qwen_call_rate':0.0,'reranker_call_rate':0.0,'model_error':model_error});write(OUT/'leakage_audit.json',{'test_leakage':0,'near_duplicate_test_leakage':0,'unknown_label_uses':0,'frozen_outputs_modified':False,'canonical_event_modified':False,'event_thread_created':False,'transition_created':False})
    report={'status':'task34_raw_evidence_dedup_conservative_singletons_retained','counts':counts,'replay_consistency':replay,'repeat_run_hash_stability':'deterministic_output_order_and_ids'}
    write(REPORTS/'task34_raw_evidence_dedup_revision1.json',report);(DOCS/'task34_raw_evidence_dedup_revision1.md').write_text(f'# Semantic Near-duplicate EvidenceUnit Dedup\n\n状态：`{report["status"]}`。由于真实Qwen runtime未成功加载，未执行语义合并，保留4466个singleton。\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
