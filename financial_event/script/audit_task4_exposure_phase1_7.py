#!/usr/bin/env python3
from __future__ import annotations
import json,re
from pathlib import Path
import pandas as pd
ROOT=Path('/opt/data/private/JinRong/kechuang/financial_event_project');OUT=ROOT/'outputs/task4_target_exposure_v1';REPORTS=ROOT/'reports';DOCS=ROOT/'docs'
DIRECTION={'positive','negative','neutral','bullish','bearish','正面','负面','中性','risk','opportunity'}
def clean(x): return '' if x is None else re.sub(r'\s+',' ',str(x)).strip()
def main():
 df=pd.read_parquet(OUT/'target_exposure_frames_3127_repaired.parquet');audit=df.sample(100,random_state=2026).sort_values('task4_sample_id').copy();rows=[]
 for _,r in audit.iterrows():
  ref=r.evidence_reference if isinstance(r.evidence_reference,dict) else {}
  evidence=clean(ref.get('evidence_sentence'));target=clean(r.target_name);event=clean(r.event_name);path=clean(r.exposure_path);etype=clean(r.exposure_type);dim=clean(r.affected_dimension);text=' '.join([event,target,evidence])
  direction_leak=any(x in ' '.join([etype,dim,path]).lower() for x in DIRECTION)
  evidence_present=bool(evidence);target_supported=bool(target and target in evidence);event_supported=bool(event and (event in evidence or any(t in evidence for t in re.split(r'[（）()、，, ]',event) if len(t)>=2)))
  path_valid=('→' in path or '->' in path) and len([x for x in re.split(r'→|->',path) if clean(x)])>=3
  if direction_leak: status='reject';err='E5_direction_leakage'
  elif not evidence_present: status='reject';err='E1_evidence_unsupported'
  elif not path_valid: status='partial';err='E4_path_missing'
  elif not target_supported and not event_supported: status='reject';err='E2_target_exposure_wrong'
  else: status='partial';err='E3_mechanism_too_vague'
  rows.append({'exposure_frame_id':r.exposure_frame_id,'task4_sample_id':r.task4_sample_id,'article_id':r.article_id,'event_name':event,'target_name':target,'exposure_type':etype,'affected_dimension':dim,'exposure_path':path,'evidence_id':ref.get('evidence_id'),'evidence_sentence':evidence,'source_document_id':ref.get('source_document_id'),'sentence_position':ref.get('sentence_position'),'provenance_source':ref.get('provenance_source'),'automated_review_status':status,'automated_error_type':err,'automated_evidence_present':evidence_present,'automated_target_supported':target_supported,'automated_event_supported':event_supported,'automated_path_valid':path_valid,'automated_direction_leakage':direction_leak,'human_review_status':'','human_error_type':'','human_exposure_ok':'','human_evidence_support_ok':'','human_path_ok':'','human_temporal_ok':'','human_notes':''})
 out=pd.DataFrame(rows);out.to_parquet(OUT/'phase1_7_audit_set_100.parquet',index=False);out.to_json(OUT/'phase1_7_audit_set_100.jsonl',orient='records',lines=True,force_ascii=False);out.to_excel(OUT/'phase1_7_audit_set_100.xlsx',index=False)
 report={'status':'task4_v2_phase1_7_audit_set_ready_pending_manual_review','reviewed_count':100,'manual_review_completed':False,'automated_pre_audit_only':True,'automated_status_counts':out.automated_review_status.value_counts().to_dict(),'automated_error_counts':out.automated_error_type.value_counts().to_dict(),'exposure_chain_validity_rate_automated':float((out.automated_evidence_present & out.automated_target_supported & out.automated_path_valid & ~out.automated_direction_leakage).mean()),'human_exposure_chain_validity_rate':None,'error_taxonomy':['E1_evidence_unsupported','E2_target_exposure_wrong','E3_mechanism_too_vague','E4_path_missing','E5_direction_leakage','E6_temporal_violation'],'unresolved_36_retained':36,'m1_m2_allowed':False,'task3_connected':False,'historical_context_used':False,'core_mutation':False,'note':'Human fields are blank. Automated pre-audit is not human agreement and cannot be used as final acceptance.'}
 (OUT/'phase1_7_audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');(REPORTS/'task4_v2_phase1_7_exposure_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');(DOCS/'task4_v2_phase1_7_exposure_audit.md').write_text('# Task4 v2 Phase 1.7 — Human Exposure Audit\n\n审核集已冻结。当前输出是自动预审，人工字段为空，M1/M2 未开放。\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
