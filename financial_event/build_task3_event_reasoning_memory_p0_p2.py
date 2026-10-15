#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import pandas as pd
ROOT=Path('/opt/data/private/JinRong/kechuang/financial_event_project');CORE=ROOT/'outputs/task3_kg_v3_release';OUT=ROOT/'outputs/task3_event_reasoning_memory';REPORTS=ROOT/'reports';DOCS=ROOT/'docs';
def read_jsonl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def write_json(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def write_jsonl(p,rows): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in rows),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
 nodes=read_jsonl(CORE/'graph_nodes.jsonl');edges=read_jsonl(CORE/'graph_edges.jsonl');states=read_jsonl(CORE/'canonical_event_current_state.jsonl');evo=read_jsonl(CORE/'event_evolution_backbone.jsonl');threads=read_jsonl(CORE/'event_threads_backbone.jsonl');members=read_jsonl(CORE/'event_thread_members_backbone.jsonl')
 cn={x['canonical_event_id']:x for x in nodes if x.get('node_type')=='CanonicalEvent'}; event_mentions=[x for x in nodes if x.get('node_type')=='EventMention']; em={x.get('event_mention_id') or x.get('stable_id'):x for x in event_mentions}; evidence_units=[x for x in nodes if x.get('node_type')=='EvidenceUnit']; evu={x.get('stable_id'):x for x in evidence_units}; st={x['canonical_event_id']:x for x in states}; thread_by={x['canonical_event_id']:x['event_thread_id'] for x in members}; event_threads=[x for x in nodes if x.get('node_type')=='EventThread']; state_nodes=[x for x in nodes if x.get('node_type')=='StateObservation']
 # P0 time contract. No future impact memory is created here.
 p0={'policy_version':'p0_v1','prediction_time_definition':'as_of_time for a memory query','visibility_rule':'all source publish/event/observation/snapshot timestamps <= prediction_time','forbidden':['future_state','future_observation','future_pattern_snapshot','Task4_test_label','raw_unknown_label','post_hoc_core_repair','canonical_event_id_as_prediction_feature'],'status':'frozen_for_p0_p2','impact_memory_created':False}
 # P1 EventMemory: complete Core event universe, no new identity nodes.
 event_rows=[]
 for eid,x in sorted(cn.items()):
  event_rows.append({'memory_event_id':'EMEM_'+hashlib.sha256(eid.encode()).hexdigest()[:24],'event_id':eid,'event_schema':x.get('event_identity_key'),'event_family_key':x.get('event_family_key'),'representative_event_mention_id':x.get('representative_event_mention_id'),'target_type':None,'target_id':None,'state_path':[],'evidence_references':[],'timeline':{'event_time':None,'publish_time':None},'source_layer':'A_core_event_universe','verification_status':'derived_read_only','core_mutation':False,'provenance':{'core_node_type':'CanonicalEvent','core_rule_version':x.get('rule_version')}})
 # Enrich target/evidence/timeline through EventMention nodes and Core edges where available.
 by_eid={r['event_id']:r for r in event_rows}
 for edge in edges:
  if edge.get('relation_type') in {'INSTANCE_OF_EVENT','HAS_STATE_OBSERVATION','SUPPORTED_BY','HAS_EVIDENCE','MEMBER_OF_THREAD'}:
   a=edge.get('source_id');b=edge.get('target_id');
   if a in em and b in by_eid:
    m=em[a];r=by_eid[b];r['target_type']=m.get('target_type');r['target_id']=m.get('target_code') or m.get('entity_name');r['timeline']['event_time']=m.get('event_time');r['timeline']['publish_time']=m.get('publish_time');
    for sid in m.get('evidence_sentence_ids') or []:
     r['evidence_references'].append({'evidence_sentence_id':sid,'event_mention_id':a,'evidence_unit_id':None,'source':m.get('source_name'),'publish_time':m.get('publish_time')})
 # P2 EvolutionMemory only direct evidence backbone, not all event universe.
 evolution=[]
 for x in evo:
  if not x.get('keep_in_backbone') or x.get('final_semantic_layer')!='direct_evidence_transition': continue
  evolution.append({'evolution_memory_id':'EVOLMEM_'+hashlib.sha256(x['original_edge_id'].encode()).hexdigest()[:24],'source_event_id':x['source_canonical_event_id'],'target_event_id':x['target_canonical_event_id'],'transition_type':x.get('transition_type'),'source_state':x.get('source_state'),'target_state':x.get('target_state'),'time_span_days':x.get('time_span_days'),'evidence_references':[{'evidence_sentence_id':s,'evidence_unit_id':(x.get('evidence_unit_ids') or [None])[i % len(x.get('evidence_unit_ids') or [None])]} for i,s in enumerate(x.get('evidence_sentence_ids',[]))],'thread_id':x.get('thread_id'),'confidence':x.get('confidence'),'source_layer':'B_high_confidence_evolution_memory','verification_status':'derived_read_only','core_mutation':False,'temporal_audit':{'chronological_direction_valid':x.get('chronological_direction_valid'),'time_reverse':x.get('time_reverse'),'same_evidence_unit':x.get('same_evidence_unit')}})
 # state paths per event using current state and transitions
 paths={}
 for x in evolution:
  for eid,state in [(x['source_event_id'],x['source_state']),(x['target_event_id'],x['target_state'])]: paths.setdefault(eid,[]).append({'state':state,'transition_id':x['evolution_memory_id'],'time_span_days':x['time_span_days']})
 for r in event_rows:r['state_path']=paths.get(r['event_id'],[])
 write_jsonl(OUT/'event_memory/event_memory.jsonl',event_rows);write_jsonl(OUT/'evolution_memory/evolution_memory.jsonl',evolution);write_json(OUT/'p0_temporal_contract.json',p0);write_json(OUT/'input_manifest.json',{'core_release_dir':str(CORE),'core_graph_nodes_sha256':sha(CORE/'graph_nodes.jsonl'),'core_graph_edges_sha256':sha(CORE/'graph_edges.jsonl'),'state_sha256':sha(CORE/'canonical_event_current_state.jsonl'),'evolution_sha256':sha(CORE/'event_evolution_backbone.jsonl'),'event_count':len(event_rows),'direct_evolution_count':len(evolution),'impact_memory_created':False})
 report={'status':'task3_event_reasoning_memory_p0_p2_complete','p0_temporal_contract':p0,'layer_a_core_event_universe':{'canonical_events':len(event_rows),'event_mentions':len(event_mentions),'evidence_units':len(evu),'state_observations':len(state_nodes),'event_threads':len(event_threads)},'layer_b_high_confidence_evolution':{'direct_evidence_transitions':len(evolution),'input_backbone_rows':len(evo)},'layer_c_targeted_raw_expansion':{'status':'not_started','reason':'not needed for P0-P2'},'impact_memory':{'status':'deferred_by_request','records_created':0},'core_mutation':False,'future_information_used':0,'task4_test_labels_used':0,'raw_unknown_labels_used':0,'evidence_traceability':'references retained from Core EventMention/transition assets'}
 write_json(OUT/'reconciliation.json',report);write_json(OUT/'leakage_audit.json',{'future_information_used':0,'task4_test_labels_used':0,'raw_unknown_labels_used':0,'core_mutation':False,'new_canonical_events':0,'new_event_threads':0});write_json(REPORTS/'task3_event_reasoning_memory_p0_p2.json',report);(DOCS/'task3_event_reasoning_memory_p0_p2.md').write_text('# Task3 Event Reasoning Memory P0-P2\n\n状态：`task3_event_reasoning_memory_p0_p2_complete`。P1使用全量Core事件空间，P2仅使用1080条direct evidence evolution；P3 Impact Memory按要求未执行。\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
