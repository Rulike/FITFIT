#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from typing import Any
import pandas as pd
ROOT=Path('/opt/data/private/JinRong/kechuang/financial_event_project');OUT=ROOT/'outputs/task4_target_exposure_v1';REPORTS=ROOT/'reports';DOCS=ROOT/'docs';SOURCE=ROOT/'outputs/task4/stage2/task4_event_level_samples.parquet'
CHAIN_MARKERS=['→','->','下降','提升','增加','减少','改善','恶化','上升','下降','导致','带来','转化','影响','成本','收入','利润','份额','交付','融资','限制']
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def clean(x):return '' if x is None else re.sub(r'\s+',' ',str(x)).strip()
def contains_any(x,ws):return any(w in x for w in ws)
def main():
 frames=pd.read_parquet(OUT/'target_exposure_frames_3127.parquet');src=pd.read_parquet(SOURCE).drop_duplicates('task4_sample_id');df=frames.merge(src[['task4_sample_id','title','content','source','field_provenance']],on='task4_sample_id',how='left',suffixes=('','_src'))
 audit=df.sample(n=100,random_state=2026).sort_values('task4_sample_id').copy()
 findings=[]
 for _,r in audit.iterrows():
  ev=clean(r.get('event_name')); evidence=clean((r.get('evidence_reference') or {}).get('evidence_sentence') if isinstance(r.get('evidence_reference'),dict) else '')
  title=clean(r.get('title')); content=clean(r.get('content')); searchable=' '.join([ev,title,evidence,content[:1200]])
  path=clean(r.get('exposure_path')); exp=clean(r.get('exposure_type')); dim=clean(r.get('affected_dimension'))
  evidence_present=bool(evidence); evidence_mentions_target=bool(clean(r.get('target_name')) and clean(r.get('target_name')) in searchable); path_chain=bool(('→' in path or '->' in path) and len(path.split())>=3) or contains_any(path,CHAIN_MARKERS)
  direction_tokens=['正面','中性','负面','positive','neutral','negative','risk','opportunity']
  direction_leak=contains_any(' '.join([exp,dim,path]),direction_tokens)
  if exp=='unresolved': unresolved_reason='information_insufficient' if not evidence_present else ('target_unclear' if not evidence_mentions_target else 'exposure_uncertain')
  else: unresolved_reason='not_unresolved'
  automated_status='needs_human_review'
  if not evidence_present: error='missing_evidence_sentence'
  elif not path_chain: error='non_transmission_path'
  elif direction_leak: error='direction_leakage_token'
  elif not evidence_mentions_target: error='target_exposure_not_evidenced'
  else: error='candidate_pass_pending_human'
  findings.append({'exposure_frame_id':r.exposure_frame_id,'task4_sample_id':r.task4_sample_id,'event_name':ev,'target_name':clean(r.target_name),'exposure_type':exp,'affected_dimension':dim,'exposure_path':path,'evidence_sentence':evidence,'automated_evidence_present':evidence_present,'automated_target_mentioned_in_available_text':evidence_mentions_target,'automated_transmission_chain':path_chain,'automated_direction_leakage':direction_leak,'unresolved_reason':unresolved_reason,'automated_error_taxonomy':error,'audit_status':automated_status,'human_exposure_ok':'','human_evidence_support_ok':'','human_path_ok':'','human_error_taxonomy':'','human_notes':''})
 out=pd.DataFrame(findings);out.to_parquet(OUT/'exposure_audit_findings_100.parquet',index=False);out.to_json(OUT/'exposure_audit_findings_100.jsonl',orient='records',lines=True,force_ascii=False);out.to_excel(OUT/'exposure_audit_findings_100.xlsx',index=False)
 unresolved=frames[frames.exposure_type.eq('unresolved')].copy();ur=[]
 for _,r in unresolved.iterrows():
  evidence=clean((r.evidence_reference or {}).get('evidence_sentence') if isinstance(r.evidence_reference,dict) else '')
  reason='information_insufficient' if not evidence else ('target_unclear' if not clean(r.target_name) else 'exposure_uncertain')
  ur.append({'task4_sample_id':r.task4_sample_id,'exposure_type':r.exposure_type,'unresolved_reason':reason,'evidence_present':bool(evidence),'target_name':clean(r.target_name),'review_required':True})
 pd.DataFrame(ur).to_parquet(OUT/'unresolved_36_taxonomy.parquet',index=False);pd.DataFrame(ur).to_json(OUT/'unresolved_36_taxonomy.jsonl',orient='records',lines=True,force_ascii=False)
 counts=out.automated_error_taxonomy.value_counts().to_dict();report={'status':'task4_v2_phase1_5_exposure_audit_complete_pending_human_review','audit_set_count':len(out),'human_audit_completed':False,'human_audit_agreement':None,'automated_pre_audit_only':True,'findings_by_taxonomy':counts,'transmission_chain_candidate_pass_rate':float(out.automated_transmission_chain.mean()),'evidence_present_rate':float(out.automated_evidence_present.mean()),'direction_leakage_count':int(out.automated_direction_leakage.sum()),'target_mentioned_rate':float(out.automated_target_mentioned_in_available_text.mean()),'unresolved_36_count':len(ur),'unresolved_36_taxonomy':pd.DataFrame(ur).unresolved_reason.value_counts().to_dict() if ur else {},'schema_revision_recommended':['evidence_sentence must be populated from a true provenance source; empty evidence cannot pass audit','exposure_path must be a four-part transmission chain: event→exposure→economic variable→impact','exposure_type and affected_dimension must not encode direction','human audit fields remain blank until manual review'],'no_training':True,'no_prediction':True,'task3_connected':False,'historical_context_used':False,'core_mutation':False}
 write(OUT/'phase1_5_audit_report.json',report);write(REPORTS/'task4_v2_phase1_5_exposure_audit.json',report);(DOCS/'task4_v2_phase1_5_exposure_audit.md').write_text('# Task4 v2 Phase1.5 Exposure Audit\n\n当前为自动预审结果，100条人工字段仍为空，不能冒充人工 agreement。重点阻断：evidence provenance 为空、exposure_path 不是传导链。\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
