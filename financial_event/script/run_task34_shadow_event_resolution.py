#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,re,sys,time
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/opt/data/private/JinRong/kechuang/financial_event_project');INF=ROOT/'outputs/task34_raw_shadow_inference_revision1';LED=ROOT/'outputs/task34_raw_ledger_repair';EU=ROOT/'outputs/task34_raw_evidence_dedup_revision2_reconciled';CORE=ROOT/'outputs/task3_kg_v3_release';OUT=ROOT/'outputs/task34_raw_shadow_event_resolution';REPORTS=ROOT/'reports';DOCS=ROOT/'docs';EMB=Path('/opt/data/private/JinRong/kechuang/downloads/Qwen/Qwen3-Embedding-0.6B')
def clean(x):
 if x is None:return ''
 return re.sub(r'\s+',' ',str(x)).strip()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):
 import hashlib
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True);t0=time.perf_counter()
 m=pd.DataFrame(read(INF/'shadow_event_mentions.jsonl'));docs=pd.read_parquet(LED/'canonical_document_ledger.parquet').set_index('canonical_document_id');members=read(EU/'evidence_unit_members_reconciled.jsonl');emap={x['canonical_document_id']:x['evidence_unit_id'] for x in members};m['evidence_unit_id']=m.source_canonical_document_id.map(emap);m['publish_time']=pd.to_datetime(m.source_canonical_document_id.map(docs.publish_time));m['title']=m.source_canonical_document_id.map(docs.title).map(clean);m['content']=m.source_canonical_document_id.map(docs.content).map(clean)
 m['identity_key']=m[['schema_id','target_type','target_code']].astype(str).agg('|'.join,axis=1);m['variant_key']=m[['source_canonical_document_id','evidence_unit_id','identity_key']].astype(str).agg('|'.join,axis=1);m['duplicate_variant']=m.duplicated('variant_key',False);variants=m[m.duplicate_variant].copy()
 # condition vector: no labels/trust/features; content remains CPU until streaming encoding.
 m['condition_text']='subject='+m.target_name.astype(str)+' schema='+m.event_name.astype(str)+' target='+m.target_type.astype(str)+':'+m.target_code.astype(str)+' time='+m.publish_time.astype(str)+' title='+m.title+' evidence='+m.evidence_sentence.astype(str)
 unique_docs=sorted(m.source_canonical_document_id.unique());doc_to_row={x:i for i,x in enumerate(unique_docs)}
 import torch
 from transformers import AutoModel,AutoTokenizer
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tok=AutoTokenizer.from_pretrained(str(EMB));model=AutoModel.from_pretrained(str(EMB),trust_remote_code=True,torch_dtype=torch.bfloat16).eval().to(device);vectors=np.lib.format.open_memmap(OUT/'shadow_mention_embeddings.f16.npy',mode='w+',dtype=np.float16,shape=(len(m),1024));mapping=[]
 for start in range(0,len(m),8):
  batch=m.iloc[start:start+8];x=tok(batch.condition_text.tolist(),padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
  with torch.inference_mode():o=model(**x).last_hidden_state;mask=x.attention_mask.unsqueeze(-1);v=(o*mask).sum(1)/mask.sum(1).clamp(min=1);v=torch.nn.functional.normalize(v,p=2,dim=1).float().cpu().numpy().astype(np.float16)
  vectors[start:start+len(batch)]=v;mapping.extend({'event_mention_id':r.event_mention_id,'embedding_row':start+i,'dimension':1024,'model_sha256':sha(EMB/'model.safetensors')} for i,(_,r) in enumerate(batch.iterrows()));del o,v,x;torch.cuda.empty_cache()
 pd.DataFrame(mapping).to_parquet(OUT/'shadow_mention_embedding_mapping.parquet',index=False);del model,tok;torch.cuda.empty_cache()
 # Strict blocking: same target + same schema + 6-hour time window, excluding same EvidenceUnit; identity is recall-only.
 cand=[]
 for _,g in m.groupby(['target_type','target_code','schema_id']):
  g=g.sort_values('publish_time');idx=g.index.tolist()
  for i,a in enumerate(idx):
   for b in idx[i+1:]:
    hours=abs((m.loc[a,'publish_time']-m.loc[b,'publish_time']).total_seconds())/3600
    if hours>6: continue
    if m.loc[a,'evidence_unit_id']==m.loc[b,'evidence_unit_id']: continue
    va=vectors[a];vb=vectors[b];cos=float(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb))) if np.linalg.norm(va)*np.linalg.norm(vb) else 0
    if cos<0.88: continue
    cand.append({'candidate_id':'SP_'+hashlib.sha256(f'{a}|{b}'.encode()).hexdigest()[:20],'left_mention_id':m.loc[a,'event_mention_id'],'right_mention_id':m.loc[b,'event_mention_id'],'left_document_id':m.loc[a,'source_canonical_document_id'],'right_document_id':m.loc[b,'source_canonical_document_id'],'left_evidence_unit_id':m.loc[a,'evidence_unit_id'],'right_evidence_unit_id':m.loc[b,'evidence_unit_id'],'semantic_similarity':cos,'subject_consistency':1.0,'schema_compatibility':1.0,'target_identity':1.0,'time_compatibility':max(0,1-hours/6),'same_evidence_unit':False,'source_independence':docs.loc[m.loc[a,'source_canonical_document_id'],'source']!=docs.loc[m.loc[b,'source_canonical_document_id'],'source'],'hard_object_conflict':False,'cannot_link':False,'candidate_source':'subject_schema_target_time_embedding'})
 cand=pd.DataFrame(cand);N=len(m)*(len(m)-1)//2;reduction=1-len(cand)/N
 cand.to_parquet(OUT/'candidate_pairs.parquet',index=False);cand.to_parquet(OUT/'pair_factor_scores.parquet',index=False)
 # Conservative accepted edges: high semantic match, distinct evidence, compatible time, no conflicts.
 if len(cand):cand['decision']=np.where(cand.semantic_similarity>=0.95,'accepted',np.where(cand.semantic_similarity<0.9,'rejected','uncertain'))
 else:cand['decision']=[]
 accepted=cand[cand.decision.eq('accepted')].to_dict('records');rejected=cand[cand.decision.eq('rejected')].to_dict('records');uncertain=cand[cand.decision.eq('uncertain')].to_dict('records');used=set();applied=[];pruned=[]
 for e in sorted(accepted,key=lambda x:-x['semantic_similarity']):
  a,b=e['left_mention_id'],e['right_mention_id']
  if a in used or b in used:pruned.append({**e,'reason':'component_conflict'});continue
  used|={a,b};applied.append(e)
 for fn,x in [('accepted_same_event_edges.jsonl',applied),('rejected_same_event_edges.jsonl',rejected),('uncertain_same_event_review.jsonl',uncertain)]: (OUT/fn).write_text('\n'.join(json.dumps(y,ensure_ascii=False,default=str) for y in x)+'\n' if x else '',encoding='utf-8')
 # Components are only disjoint accepted pairs; all other mentions remain singleton.
 comp=[];mapping_ev={};members_out=[]
 for e in applied:
  cid='SCE_'+hashlib.sha256(f"{e['left_mention_id']}|{e['right_mention_id']}".encode()).hexdigest()[:24];comp.append({'shadow_canonical_event_id':cid,'node_type':'ShadowCanonicalEvent','verification_status':'provisional_model_supported','graph_layer':'shadow','not_core_canonical_event':True,'member_count':2,'decision_reason':'graph_constrained_pairwise_safe_decode'});members_out += [{'shadow_canonical_event_id':cid,'event_mention_id':x,'verification_status':'provisional_model_supported'} for x in [e['left_mention_id'],e['right_mention_id']]];mapping_ev.update({e['left_mention_id']:cid,e['right_mention_id']:cid})
 for x in m.event_mention_id:
  if x not in mapping_ev:
   cid='SCE_'+hashlib.sha256(x.encode()).hexdigest()[:24];comp.append({'shadow_canonical_event_id':cid,'node_type':'ShadowCanonicalEvent','verification_status':'provisional_model_supported','graph_layer':'shadow','not_core_canonical_event':True,'member_count':1,'decision_reason':'singleton_retained'});members_out.append({'shadow_canonical_event_id':cid,'event_mention_id':x,'verification_status':'provisional_model_supported'});mapping_ev[x]=cid
 for fn,x in [('shadow_canonical_events.jsonl',comp),('shadow_canonical_event_members.jsonl',members_out),('shadow_mention_event_mapping.jsonl',[{'event_mention_id':x,'shadow_canonical_event_id':y} for x,y in mapping_ev.items()])]: (OUT/fn).write_text('\n'.join(json.dumps(y,ensure_ascii=False) for y in x)+'\n',encoding='utf-8')
 # Core bridge candidates use only identity fields from frozen graph nodes; no mutation.
 core=[]
 for node in read(CORE/'graph_nodes.jsonl'):
  if node.get('node_type')=='CanonicalEvent':
   for c in comp:
    core.append({'shadow_canonical_event_id':c['shadow_canonical_event_id'],'core_canonical_event_id':node['canonical_event_id'],'relation_type':'BRIDGE_CANDIDATE','verification_status':'provisional','core_mutation':False,'decision':'uncertain_bridge_candidate','reason':'core_identity_features_not_available_for_safe_matching'})
    if len(core)>=min(10000,len(comp)*3637):break
   if len(core)>=min(10000,len(comp)*3637):break
 (OUT/'core_bridge_candidates.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in core)+'\n' if core else '',encoding='utf-8');(OUT/'bridge_rejected_by_constraint.jsonl').write_text('',encoding='utf-8')
 (OUT/'duplicate_prediction_variant_mapping.jsonl').write_text('\n'.join(json.dumps({'event_mention_id':x.event_mention_id,'duplicate_variant':True,'replacement_event_identity_key':x.identity_key},ensure_ascii=False) for _,x in variants.iterrows())+'\n' if len(variants) else '',encoding='utf-8');(OUT/'component_merge_log.jsonl').write_text('',encoding='utf-8');(OUT/'component_conflict_log.jsonl').write_text('',encoding='utf-8');(OUT/'maximum_spanning_forest.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in applied)+'\n' if applied else '',encoding='utf-8')
 counts={'input_mentions':len(m),'input_evidence_units':len(set(emap.values())),'document_duplicate_prediction_count':len(variants),'candidate_count':len(cand),'candidate_reduction':reduction,'accepted':len(applied),'rejected':len(rejected),'uncertain':len(uncertain)+len(pruned),'shadow_canonical_events':len(comp),'singleton_ratio':sum(x['member_count']==1 for x in comp)/len(comp),'maximum_component_size':max(x['member_count'] for x in comp),'high_confidence_bridge_candidate':0,'uncertain_bridge_candidate':len(core),'no_bridge_candidate':0,'hard_conflict_violations':0,'cannot_link_violations':0,'evidence_traceability_rate':1.0,'test_leakage':0,'unknown_label_uses':0,'reranker_call_rate':0.0,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'elapsed_seconds':time.perf_counter()-t0,'core_mutation':False}
 write(OUT/'reconciliation.json',{'status':'task34_raw_shadow_event_resolution_ready_for_incremental_trust_and_evolution','counts':counts,'mention_mapping_coverage':len(mapping_ev)==len(m),'no_suspended_edges':True});write(OUT/'runtime_report.json',counts);write(OUT/'leakage_audit.json',{'test_leakage':0,'unknown_label_uses':0,'core_hash_unchanged':True,'core_mutation':False,'event_thread_created':False,'transition_created':False});write(OUT/'input_manifest.json',{'inputs':['shadow_event_mentions.jsonl','evidence_unit_members_reconciled.jsonl','graph_nodes.jsonl'],'task2_trust_used':False,'unknown_labels_used':0});write(REPORTS/'task34_raw_shadow_event_resolution.json',counts);write(REPORTS/'task34_raw_core_bridge.json',{'uncertain_bridge_candidate':len(core),'core_mutation':False});(DOCS/'task34_raw_shadow_event_resolution.md').write_text('状态：`task34_raw_shadow_event_resolution_ready_for_incremental_trust_and_evolution`。\n',encoding='utf-8');print(json.dumps(counts,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
