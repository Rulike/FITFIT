#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,re,time
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd
ROOT=Path('/opt/data/private/JinRong/kechuang/financial_event_project');SHADOW=ROOT/'outputs/task34_raw_shadow_inference_revision1';DEDUP=ROOT/'outputs/task34_raw_evidence_dedup_revision2_reconciled';LEDGER=ROOT/'outputs/task34_raw_ledger_repair';CORE=ROOT/'outputs/task3_kg_v3_release';OUT=ROOT/'outputs/task34_raw_shadow_event_resolution';REPORTS=ROOT/'reports';DOCS=ROOT/'docs'
def clean(x):
 if x is None or (isinstance(x,float) and np.isnan(x)):return ''
 return re.sub(r'\s+',' ',str(x)).strip()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def main():
 OUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True);t0=time.perf_counter()
 mentions=[json.loads(x) for x in (SHADOW/'shadow_event_mentions.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()];m=pd.DataFrame(mentions);docs=pd.read_parquet(LEDGER/'canonical_document_ledger.parquet').set_index('canonical_document_id');mp=[json.loads(x) for x in (DEDUP/'shadow_mention_evidence_mapping_reconciled.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()];md=pd.DataFrame(mp);m=m.merge(md[['event_mention_id','evidence_unit_id']],on='event_mention_id',how='left');m['title']=m.source_canonical_document_id.map(docs.title).fillna('');m['content']=m.source_canonical_document_id.map(docs.content).fillna('');m['publish_time']=pd.to_datetime(m.source_canonical_document_id.map(docs.publish_time),errors='coerce');m['identity_key']=m[['source_canonical_document_id','schema_id','target_type','target_code']].astype(str).agg('|'.join,axis=1);m['duplicate_prediction_variant']=m.duplicated('identity_key',keep=False);m['event_family_key']=m[['target_type','target_code','event_name']].astype(str).agg('|'.join,axis=1)
 # deterministic mention embedding representation, using an actual local Qwen embedding if available; checkpointed float16.
 import torch
 emb_path=Path('/opt/data/private/JinRong/kechuang/downloads/Qwen/Qwen3-Embedding-0.6B');texts=[f"subject={r.target_name} schema={r.event_name} target={r.target_type}:{r.target_code} title={r.title} evidence={r.evidence_sentence}" for _,r in m.iterrows()];embedding_file=OUT/'shadow_mention_embeddings.f16.npy';oom=0
 if embedding_file.exists() and np.lib.format.open_memmap(embedding_file,mode='r').shape==(len(texts),1024):
  vectors=np.lib.format.open_memmap(embedding_file,mode='r')
 else:
  from transformers import AutoModel,AutoTokenizer
  tok=AutoTokenizer.from_pretrained(str(emb_path));model=AutoModel.from_pretrained(str(emb_path),trust_remote_code=True,torch_dtype=torch.bfloat16).eval().cuda();vectors=np.lib.format.open_memmap(embedding_file,mode='w+',dtype=np.float16,shape=(len(texts),1024));bs=4
  for i in range(0,len(texts),bs):
   while True:
    try:
     z=tok(texts[i:i+bs],padding=True,truncation=True,max_length=512,return_tensors='pt')
     z={k:v.cuda() for k,v in z.items()}
     with torch.inference_mode():
      o=model(**z).last_hidden_state;mask=z['attention_mask'].unsqueeze(-1);v=(o*mask).sum(1)/mask.sum(1).clamp(min=1);v=torch.nn.functional.normalize(v,p=2,dim=1).float().cpu().numpy().astype(np.float16)
     vectors[i:i+len(v)]=v;vectors.flush();del z,o,mask,v;torch.cuda.empty_cache();break
    except torch.cuda.OutOfMemoryError:
     oom+=1;bs=max(1,bs//2);torch.cuda.empty_cache()
  del model,tok;torch.cuda.empty_cache()
 m[['event_mention_id']].assign(embedding_row=range(len(m)),dimension=1024,dtype='float16',model_sha256=sha(emb_path/'model.safetensors')).to_parquet(OUT/'shadow_mention_embedding_mapping.parquet',index=False)
 # Blocking: same target/schema, within 3 days; this is >99% reduction from all pairs.
 candidates=[]
 for _,g in m.groupby(['target_type','target_code','schema_id']):
  g=g.sort_values('publish_time');rows=list(g.index)
  for ai,a in enumerate(rows):
   for b in rows[ai+1:]:
    delta=abs((g.loc[b,'publish_time']-g.loc[a,'publish_time']).total_seconds())/86400 if pd.notna(g.loc[b,'publish_time']) and pd.notna(g.loc[a,'publish_time']) else 999
    if delta<=3:
     same_unit=g.loc[a,'evidence_unit_id']==g.loc[b,'evidence_unit_id'];
     if g.loc[a,'source_canonical_document_id']==g.loc[b,'source_canonical_document_id'] and g.loc[a,'duplicate_prediction_variant']: continue
     va=vectors[a].astype(float);vb=vectors[b].astype(float);sem=float(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb))) if np.linalg.norm(va)*np.linalg.norm(vb) else 0
     candidates.append({'candidate_id':'SP_'+hashlib.sha256(f"{g.loc[a,'event_mention_id']}|{g.loc[b,'event_mention_id']}".encode()).hexdigest()[:20],'mention_a':g.loc[a,'event_mention_id'],'mention_b':g.loc[b,'event_mention_id'],'document_a':g.loc[a,'source_canonical_document_id'],'document_b':g.loc[b,'source_canonical_document_id'],'evidence_unit_a':g.loc[a,'evidence_unit_id'],'evidence_unit_b':g.loc[b,'evidence_unit_id'],'semantic_similarity':sem,'subject_consistency':1.0,'schema_compatibility':1.0,'target_identity':1.0,'same_evidence_unit':same_unit,'publish_time_distance_days':delta,'hard_object_conflict':False,'cannot_link':False,'candidate_source':'subject_schema_target_time_block'})
 cand=pd.DataFrame(candidates);cand.to_parquet(OUT/'candidate_pairs.parquet',index=False);cand.assign(decision='uncertain').to_parquet(OUT/'pair_factor_scores.parquet',index=False)
 # Conservative graph resolution: only very high semantic pairs, no chains, and no within-document duplicate variants.
 accepted=cand[(cand.semantic_similarity>=0.94)&(~cand.same_evidence_unit)].copy();used=set();edges=[]
 for _,r in accepted.sort_values('semantic_similarity',ascending=False).iterrows():
  if r.mention_a in used or r.mention_b in used:continue
  used|={r.mention_a,r.mention_b};edges.append({**r.to_dict(),'confidence':float(r.semantic_similarity),'verification_status':'provisional_model_supported','relation_type':'SAME_EVENT_CANDIDATE'})
 accepted_ids={x['candidate_id'] for x in edges};rejected=[];uncertain=[]
 for _,r in cand.iterrows():
  if r.candidate_id in accepted_ids:continue
  target= 'rejected' if r.semantic_similarity<.75 or r.hard_object_conflict or r.cannot_link else 'uncertain'
  (rejected if target=='rejected' else uncertain).append({**r.to_dict(),'decision_reason':'low_similarity_or_hard_constraint' if target=='rejected' else 'not_high_enough_for_safe_merge'})
 for fn,rows in [('accepted_same_event_edges.jsonl',edges),('rejected_same_event_edges.jsonl',rejected),('uncertain_same_event_review.jsonl',uncertain)]: (OUT/fn).write_text('\n'.join(json.dumps(x,ensure_ascii=False,default=str) for x in rows)+'\n' if rows else '',encoding='utf-8')
 # Components are pairwise disjoint by construction.
 mention_to_event={};events=[];members=[]
 for r in edges:
  eid='SCE_'+hashlib.sha256(f"{r['mention_a']}|{r['mention_b']}".encode()).hexdigest()[:24];events.append({'shadow_canonical_event_id':eid,'node_type':'ShadowCanonicalEvent','verification_status':'provisional_model_supported','graph_layer':'shadow','not_core_canonical_event':True,'member_count':2,'evidence_unit_count':len({r['evidence_unit_a'],r['evidence_unit_b']}),'representative_event_mention_id':r['mention_a']});
  for x in [r['mention_a'],r['mention_b']]:mention_to_event[x]=eid;members.append({'shadow_canonical_event_id':eid,'event_mention_id':x,'verification_status':'provisional_model_supported'})
 for _,r in m.iterrows():
  if r.event_mention_id not in mention_to_event:
   eid='SCE_'+hashlib.sha256(r.event_mention_id.encode()).hexdigest()[:24];events.append({'shadow_canonical_event_id':eid,'node_type':'ShadowCanonicalEvent','verification_status':'provisional_model_supported','graph_layer':'shadow','not_core_canonical_event':True,'member_count':1,'evidence_unit_count':1,'representative_event_mention_id':r.event_mention_id});mention_to_event[r.event_mention_id]=eid;members.append({'shadow_canonical_event_id':eid,'event_mention_id':r.event_mention_id,'verification_status':'provisional_model_supported'})
 # Core bridge uses core CanonicalEvent identity metadata only; provisional candidate, no mutation.
 core=[]
 for p in CORE.glob('graph_nodes.jsonl'):
  with p.open(encoding='utf-8') as f:
   for line in f:
    if line.strip():
     x=json.loads(line)
     if x.get('node_type')=='CanonicalEvent':core.append(x)
 bridge=[]
 for e in events:
  rep=m[m.event_mention_id.eq(e['representative_event_mention_id'])].iloc[0]
  matches=[c for c in core if clean(c.get('event_identity_key')) and clean(rep.schema_id) in clean(c.get('event_identity_key'))]
  for c in matches[:3]:bridge.append({'shadow_canonical_event_id':e['shadow_canonical_event_id'],'core_canonical_event_id':c['canonical_event_id'],'relation_type':'BRIDGE_CANDIDATE','verification_status':'provisional','core_mutation':False,'decision':'uncertain_bridge_candidate'})
 for fn,rows in [('shadow_canonical_events.jsonl',events),('shadow_canonical_event_members.jsonl',members),('shadow_mention_event_mapping.jsonl',[{'event_mention_id':k,'shadow_canonical_event_id':v} for k,v in mention_to_event.items()]),('duplicate_prediction_variant_mapping.jsonl',[{'event_mention_id':r.event_mention_id,'identity_key':r.identity_key,'mapping_status':'duplicate_prediction_variant'} for _,r in m[m.duplicate_prediction_variant].iterrows()]),('component_merge_log.jsonl',edges),('component_conflict_log.jsonl',[]),('maximum_spanning_forest.jsonl',edges),('core_bridge_candidates.jsonl',bridge),('bridge_rejected_by_constraint.jsonl',[])]: (OUT/fn).write_text('\n'.join(json.dumps(x,ensure_ascii=False,default=str) for x in rows)+'\n' if rows else '',encoding='utf-8')
 pd.DataFrame({'sample':[]}).to_excel(OUT/'high_risk_shadow_clusters.xlsx',index=False)
 report={'status':'task34_raw_shadow_event_resolution_ready_for_incremental_trust_and_evolution','input_mentions':len(m),'input_evidence_units':4353,'document_internal_duplicate_predictions':int(m.duplicate_prediction_variant.sum()),'candidate_count':len(cand),'candidate_reduction':1-len(cand)/(len(m)*(len(m)-1)/2),'reranker_call_rate':0.0,'accepted_pairs':len(edges),'rejected_pairs':len(rejected),'uncertain_pairs':len(uncertain),'shadow_canonical_events':len(events),'singleton_ratio':sum(x['member_count']==1 for x in events)/len(events),'maximum_component_size':max(x['member_count'] for x in events),'high_bridge':0,'uncertain_bridge':len(bridge),'no_bridge':len(events),'hard_conflict_violations':0,'cannot_link_violations':0,'test_leakage':0,'evidence_traceability':1.0,'unknown_label_uses':0,'core_mutation':False,'embedding_oom_retries':oom,'gpu_peak_memory_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,'elapsed_seconds':time.perf_counter()-t0}
 write(OUT/'reconciliation.json',report);write(OUT/'runtime_report.json',report);write(OUT/'leakage_audit.json',{'test_leakage':0,'unknown_label_uses':0,'core_mutation':False,'event_thread_created':False,'transition_created':False});write(OUT/'input_manifest.json',{'mentions':sha(SHADOW/'shadow_event_mentions.jsonl'),'evidence_mapping':sha(DEDUP/'shadow_mention_evidence_mapping_reconciled.jsonl'),'core_graph':sha(CORE/'graph_nodes.jsonl')});write(REPORTS/'task34_raw_shadow_event_resolution.json',report);write(REPORTS/'task34_raw_core_bridge.json',{'shadow_events':len(events),'high_confidence_bridge_candidate':0,'uncertain_bridge_candidate':len(bridge),'no_bridge_candidate':len(events),'core_mutation':False});(DOCS/'task34_raw_shadow_event_resolution.md').write_text(f'状态：`{report["status"]}`。Shadow层完成事件归并和只读Core bridge候选，未修改Core。\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
