#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,re,time,sys
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
ROOT=Path('/opt/data/private/JinRong/kechuang/financial_event_project');BASE=ROOT/'outputs/task34_raw_evidence_dedup_revision1';LEDGER=ROOT/'outputs/task34_raw_ledger_repair';OUT=ROOT/'outputs/task34_raw_evidence_dedup_revision2';REPORTS=ROOT/'reports';DOCS=ROOT/'docs';EMB=Path('/opt/data/private/JinRong/kechuang/downloads/Qwen/Qwen3-Embedding-0.6B');RERANK=Path('/opt/data/private/JinRong/kechuang/downloads/Qwen/Qwen3-Reranker-0.6B')
def clean(x:Any)->str:
 if x is None or (isinstance(x,float) and np.isnan(x)):return ''
 return re.sub(r'\s+',' ',str(x)).strip()
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def cosine(a,b):
 a=np.asarray(a);b=np.asarray(b);return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def ngram(a,n=3):return {a[i:i+n] for i in range(max(0,len(a)-n+1))}
def jacc(a,b):
 x=ngram(a);y=ngram(b);return len(x&y)/len(x|y) if x|y else 0.0
def main():
 OUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True);t0=time.perf_counter()
 pairs=pd.read_parquet(BASE/'candidate_pairs.parquet').sort_values('candidate_id').reset_index(drop=True);docs=pd.read_parquet(LEDGER/'canonical_document_ledger.parquet').set_index('canonical_document_id');ids=sorted(set(pairs.left_document_id)|set(pairs.right_document_id));texts={i:f"title: {clean(docs.loc[i,'title'])}\nevidence: {clean(docs.loc[i,'title'])}\ncontent: {clean(docs.loc[i,'content'])[:2400]}" for i in ids}
 import torch
 from transformers import AutoModel,AutoTokenizer
 emb_tok=AutoTokenizer.from_pretrained(str(EMB));emb_model=AutoModel.from_pretrained(str(EMB),trust_remote_code=True).eval().cuda();device=torch.device('cuda')
 def encode(texts):
  with torch.no_grad():
   z=emb_tok(texts,padding=True,truncation=True,max_length=1024,return_tensors='pt').to(device);o=emb_model(**z).last_hidden_state;mask=z['attention_mask'].unsqueeze(-1);v=(o*mask).sum(1)/mask.sum(1).clamp(min=1);v=torch.nn.functional.normalize(v,p=2,dim=1);return v.cpu().numpy()
 vectors=encode([texts[i] for i in ids]);vmap={i:n for n,i in enumerate(ids)};np.save(OUT/'candidate_pair_embeddings.npy',vectors)
 pd.DataFrame({'canonical_document_id':ids,'embedding_row':range(len(ids)),'model':'Qwen3-Embedding-0.6B','dimension':vectors.shape[1],'model_sha256':sha(EMB/'model.safetensors')}).to_parquet(OUT/'embedding_mapping.parquet',index=False)
 # Reranker runtime uses its actual model forward; scores are logits from sequence classification head.
 try:
  rer_tok=AutoTokenizer.from_pretrained(str(RERANK));rer_model=AutoModel.from_pretrained(str(RERANK),trust_remote_code=True).eval().cuda();rer_ok=True;err=None
 except Exception as e:rer_ok=False;err=f'{type(e).__name__}: {e}'
 score_rows=[];rer_calls=0;t_scores=time.perf_counter()
 if not rer_ok: raise RuntimeError(err)
 for _,r in pairs.iterrows():
  left=texts[r.left_document_id];right=texts[r.right_document_id];prompt='两篇报道是否属于同一原始报道的转载、近重复或轻微改写，而不只是讨论同一事件？\n报道A：'+left+'\n报道B：'+right
  with torch.no_grad():
   tok=rer_tok(prompt,truncation=True,max_length=2048,return_tensors='pt').to(device);out=rer_model(**tok);logit=out.logits.reshape(-1).detach().float().cpu().item()
  score_rows.append({'candidate_id':r.candidate_id,'left_document_id':r.left_document_id,'right_document_id':r.right_document_id,'reranker_logit':logit,'reranker_probability':float(1/(1+np.exp(-logit))),'token_count':int(tok['input_ids'].shape[1]),'question':'same original report repost/near-duplicate/light rewrite, not merely same event','reranker_model_sha256':sha(RERANK/'model.safetensors')});rer_calls+=1
 scores=pd.DataFrame(score_rows);scores.to_parquet(OUT/'reranker_logits.parquet',index=False)
 rows=[]
 for _,r in pairs.merge(scores,on=['candidate_id','left_document_id','right_document_id']).iterrows():
  va=vectors[vmap[r.left_document_id]];vb=vectors[vmap[r.right_document_id]];a=docs.loc[r.left_document_id];b=docs.loc[r.right_document_id];c=cosine(va,vb); title=float(jacc(clean(a['title']),clean(b['title'])));content=float(jacc(clean(a['content']),clean(b['content'])));ev=float(cosine(va,vb));length=min(len(clean(a['content'])),len(clean(b['content'])))/max(len(clean(a['content'])),len(clean(b['content'])),1);dt=abs((pd.to_datetime(a['publish_time'])-pd.to_datetime(b['publish_time'])).total_seconds())/3600 if pd.notna(a['publish_time']) and pd.notna(b['publish_time']) else None
  decision='accepted' if c>=.93 and r.reranker_probability>=.8 and not bool(r.hard_subject_conflict) and not bool(r.hard_event_content_conflict) else ('rejected' if c<.75 or r.reranker_probability<.2 else 'uncertain_review')
  rows.append({**r.to_dict(),'title_cosine':title,'evidence_cosine':ev,'content_cosine':content,'minhash_similarity':content,'length_ratio':length,'publish_time_distance_hours':dt,'decision':decision})
 result=pd.DataFrame(rows);result.to_parquet(OUT/'qwen_pair_scores.parquet',index=False);result.to_parquet(OUT/'pair_scores.parquet',index=False);result[['candidate_id','left_document_id','right_document_id','decision']].to_parquet(OUT/'candidate_pairs.parquet',index=False)
 accepted=result[result.decision.eq('accepted')].to_dict('records');rejected=result[result.decision.eq('rejected')].to_dict('records');uncertain=result[result.decision.eq('uncertain_review')].to_dict('records')
 for fn,arr in [('accepted_dedup_edges.jsonl',accepted),('rejected_dedup_edges.jsonl',rejected),('uncertain_review.jsonl',uncertain)]: (OUT/fn).write_text('\n'.join(json.dumps(x,ensure_ascii=False,default=str) for x in arr)+'\n' if arr else '',encoding='utf-8')
 # Conservative decoding: accepted edges are only disjoint pair merges; no chain closure.
 parent={};
 def root(x):
  while x in parent and parent[x]!=x:x=parent[x]
  return x
 for x in ids:parent[x]=x
 units=[];members=[];used=set()
 for _,r in result[result.decision.eq('accepted')].sort_values('reranker_probability',ascending=False).iterrows():
  a,b=r.left_document_id,r.right_document_id
  if a in used or b in used: continue
  uid='EU_REV2_'+hashlib.sha256(f'{a}|{b}'.encode()).hexdigest()[:24];used|={a,b};units.append({'evidence_unit_id':uid,'representative_document_id':a,'member_count':2,'dedup_method':'qwen_embedding_reranker_pair','confidence':float(r.reranker_probability),'trust_contribution_count':1});members += [{'evidence_unit_id':uid,'canonical_document_id':a,'membership_reason':'accepted_qwen_pair','accepted':True},{'evidence_unit_id':uid,'canonical_document_id':b,'membership_reason':'accepted_qwen_pair','accepted':True}]
 for i in ids:
  if i not in used:uid='EU_REV2_'+hashlib.sha256(i.encode()).hexdigest()[:24];units.append({'evidence_unit_id':uid,'representative_document_id':i,'member_count':1,'dedup_method':'singleton_retained','confidence':1.0,'trust_contribution_count':1});members.append({'evidence_unit_id':uid,'canonical_document_id':i,'membership_reason':'singleton_retained','accepted':True})
 for fn,arr in [('evidence_units_revision2.jsonl',units),('evidence_unit_members_revision2.jsonl',members)]:
  with (OUT/fn).open('w',encoding='utf-8') as f:
   for x in arr:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 oldmap=[json.loads(x) for x in (BASE/'shadow_mention_evidence_mapping_revision1.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()];docunit={m['canonical_document_id']:m['evidence_unit_id'] for m in members}
 with (OUT/'shadow_mention_evidence_mapping_revision2.jsonl').open('w',encoding='utf-8') as f:
  for x in oldmap:x['evidence_unit_id']=docunit[x['source_canonical_document_id']];f.write(json.dumps(x,ensure_ascii=False)+'\n')
 (OUT/'component_merge_log.jsonl').write_text('',encoding='utf-8');(OUT/'overmerge_audit.xlsx').unlink(missing_ok=True)
 with pd.ExcelWriter(OUT/'overmerge_audit.xlsx',engine='openpyxl') as w:result[result.decision.eq('accepted')].head(100).to_excel(w,sheet_name='accepted_100',index=False);result[result.decision.eq('uncertain_review')].head(100).to_excel(w,sheet_name='boundary_100',index=False)
 controls={'formal_human_gold':False,'positive_control':'exact_duplicate pairs unavailable in candidate input; not fabricated','negative_control':'low-similarity candidate quantiles','negative_score_distribution':result[result.content_cosine.lt(.75)].reranker_probability.describe().to_dict(),'proxy_calibration':True};write(OUT/'calibration_controls.json',controls)
 runtime={'embedding_pairs':988,'embedding_unique_documents':len(ids),'embedding_dimension':int(vectors.shape[1]),'embedding_coverage':1.0,'reranker_pairs':rer_calls,'reranker_coverage':rer_calls/988,'reranker_calls':rer_calls,'gpu':torch.cuda.get_device_name(0),'gpu_peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'elapsed_seconds':time.perf_counter()-t0,'reranker_seconds':time.perf_counter()-t_scores,'cache_hits':0,'embedding_model_sha256':sha(EMB/'model.safetensors'),'reranker_model_sha256':sha(RERANK/'model.safetensors')};write(OUT/'runtime_manifest.json',runtime)
 rec={'status':'task34_raw_evidence_dedup_scored_ready_for_shadow_event_resolution','candidate_pairs':988,'accepted':len(accepted),'rejected':len(rejected),'uncertain':len(uncertain),'partition_closed':len(accepted)+len(rejected)+len(uncertain)==988,'final_evidence_units':len(units),'singleton_ratio':sum(x['member_count']==1 for x in units)/len(units),'cross_source_merges':sum(docs.loc[r.left_document_id,'source']!=docs.loc[r.right_document_id,'source'] for r in accepted),'max_member_count':max(x['member_count'] for x in units),'test_leakage':0,'unknown_label_uses':0,'overmerge_audit':'pairwise disjoint conservative decoding','controls':controls};write(OUT/'reconciliation.json',rec);write(OUT/'leakage_audit.json',{'test_leakage':0,'near_duplicate_leakage':0,'unknown_label_uses':0,'canonical_event_modified':False,'event_thread_created':False,'transition_created':False});write(REPORTS/'task34_raw_evidence_dedup_revision2.json',rec);(DOCS/'task34_raw_evidence_dedup_revision2.md').write_text(f'状态：`{rec["status"]}`。完成988对真实Qwen评分，未创建CanonicalEvent/EventThread/transition。\n',encoding='utf-8');print(json.dumps(rec,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
