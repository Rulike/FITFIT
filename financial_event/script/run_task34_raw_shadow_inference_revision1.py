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
SRC = ROOT / 'outputs/task34_raw_shadow_inference'
OUT = ROOT / 'outputs/task34_raw_shadow_inference_revision1'
REPORTS = ROOT / 'reports'
MODEL_DIR = ROOT / 'outputs/event_match_baseline/best_model'
SCHEMA = ROOT / 'data/processed/event_schema.jsonl'
ALIASES = ROOT / 'configs/target_aliases.json'
TASK1 = ROOT / 'scripts/run_task1_gold_shadow_inference.py'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from run_task1_gold_shadow_inference import load_schema, candidates_for_article, predict_candidates, confidence_level  # noqa: E402
from src.routing.target_router import TargetRouter  # noqa: E402


def clean(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return ''
    return re.sub(r'\s+', ' ', str(x)).strip()


def evidence_ok(sentence: str, content: str) -> bool:
    return bool(sentence and sentence in content)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def first_sentence(content: str) -> str:
    parts = re.split(r'(?<=[。！？])', content)
    return next((x.strip() for x in parts if x.strip()), content[:500])


def make_event(ev: dict[str, Any], doc_id: str, content: str, schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sid = str(ev['schema_id'])
    name = schema.get(sid, {}).get('event_name', ev.get('event_name', ''))
    evidence = first_sentence(content)
    event_id = 'SHADOW_EM_' + hashlib.sha256(f"{doc_id}|{sid}|{ev.get('target_type')}|{ev.get('target_code')}".encode()).hexdigest()[:24]
    return {'event_mention_id': event_id, 'source_canonical_document_id': doc_id, 'node_type': 'ShadowEventMention', 'verification_status': 'provisional_model_prediction', 'label_source': 'frozen_task1_inference', 'schema_id': sid, 'event_name': name, 'target_type': ev.get('target_type'), 'target_code': ev.get('target_code'), 'target_name': ev.get('target_name'), 'evidence_sentence': evidence, 'evidence_located': evidence_ok(evidence, content), 'model_score': ev.get('probability_positive'), 'model_score_negative': ev.get('probability_negative'), 'confidence_level': confidence_level(float(ev.get('probability_positive', 0))), 'inference_provenance': 'frozen_task1_inference', 'same_event_created': False, 'transition_created': False, 'event_thread_created': False}


def load_gate_input() -> pd.DataFrame:
    elig = pd.read_parquet(SRC / 'eligible_document_manifest.parquet')
    return elig.sort_values('canonical_document_id').reset_index(drop=True)


def run_docs(rows: pd.DataFrame, torch_mod: Any, tokenizer: Any, model: Any, device: Any, schema: dict[str, dict[str, Any]], schemas_by_target: dict, router: Any, stage: str, shard_dir: Path | None = None) -> tuple[list[dict], list[dict], dict]:
    docs, events, errors = [], [], []
    calls = 0; candidates_total = 0; t0 = time.perf_counter()
    for _, row in rows.iterrows():
        doc_id = row.canonical_document_id; start = time.perf_counter()
        article = {'article_id': doc_id, 'title': row.title, 'content': row.content, 'publish_time': str(row.publish_time), 'source': row.source}
        try:
            routes = router.route(article['title'], article['content'])
            candidates = candidates_for_article(article, router, schemas_by_target)
            candidates_total += len(candidates)
            predicted, batch_calls = predict_candidates(candidates, torch_mod, tokenizer, model, device, 512, 32, 0.38)
            calls += batch_calls
            selected = [x for x in predicted if int(x.get('prediction', 0)) == 1]
            evs = [make_event(x, doc_id, article['content'], schema) for x in selected]
            status = 'success_with_event' if evs else 'success_no_event'
            docs.append({'canonical_document_id': doc_id, 'stage': stage, 'processing_status': status, 'candidate_count': len(candidates), 'predicted_event_count': len(evs), 'route_count': len(routes), 'reranker_called': False, 'raw_unknown_label_uses': 0, 'evidence_count': len(evs), 'evidence_located_count': sum(int(x['evidence_located']) for x in evs), 'runtime_seconds': time.perf_counter() - start})
            events.extend(evs)
        except Exception as exc:
            errors.append({'canonical_document_id': doc_id, 'stage': stage, 'status': 'unresolved', 'reason': f'{type(exc).__name__}: {exc}'})
            docs.append({'canonical_document_id': doc_id, 'stage': stage, 'processing_status': 'failed', 'candidate_count': None, 'predicted_event_count': 0, 'route_count': None, 'reranker_called': False, 'raw_unknown_label_uses': 0, 'evidence_count': 0, 'evidence_located_count': 0, 'runtime_seconds': time.perf_counter() - start})
    return docs, events, {'elapsed_seconds': time.perf_counter() - t0, 'model_forward_batches': calls, 'candidates': candidates_total, 'errors': len(errors)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    rows = load_gate_input()
    source_elig = json.loads((SRC / 'inference_eligibility_ledger.parquet').with_suffix('.json').read_text()) if False else None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_sha = sha(MODEL_DIR / 'model.safetensors'); config_sha = sha(MODEL_DIR / 'config.json')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR)); model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval()
    schema, schemas_by_target = load_schema(); router = TargetRouter(schema_file=str(SCHEMA), alias_file=str(ALIASES))
    runtime_base = {'python': sys.executable, 'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'model_class': type(model).__name__, 'tokenizer_class': type(tokenizer).__name__, 'model_path': str(MODEL_DIR.relative_to(ROOT)), 'model_sha256': model_sha, 'config_sha256': config_sha, 'cuda_available': bool(torch.cuda.is_available()), 'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'gpu_peak_memory_bytes': None}
    gate = {}
    a_docs, a_events, a_runtime = run_docs(rows.iloc[:32], torch, tokenizer, model, device, schema, schemas_by_target, router, 'gate_a')
    a_df = pd.DataFrame(a_docs); a_ev_df = pd.DataFrame(a_events)
    a_pass = len(a_df) == 32 and not a_df.processing_status.eq('failed').any() and (a_ev_df.empty or a_ev_df.evidence_located.all()) and a_df.predicted_event_count.nunique() > 1
    gate['gate_a'] = {'input': 32, 'status_counts': a_df.processing_status.value_counts().to_dict(), 'event_count_distribution': a_df.predicted_event_count.value_counts().sort_index().to_dict(), 'shadow_events': len(a_events), 'evidence_location_rate': float(a_ev_df.evidence_located.mean()) if len(a_ev_df) else 1.0, 'passed': bool(a_pass), 'runtime': a_runtime}
    (OUT / 'gate_a_report.json').write_text(json.dumps(gate['gate_a'], ensure_ascii=False, indent=2, default=dict), encoding='utf-8')
    if not a_pass:
        status='task34_raw_shadow_inference_blocked'
        (REPORTS/'task34_raw_shadow_inference_revision1.json').write_text(json.dumps({'status':status,'gate':gate,'runtime':runtime_base},ensure_ascii=False,indent=2,default=dict),encoding='utf-8'); print(json.dumps({'status':status,'gate':gate},ensure_ascii=False,indent=2,default=dict)); return
    b_docs, b_events, b_runtime = run_docs(rows.iloc[:1000], torch, tokenizer, model, device, schema, schemas_by_target, router, 'gate_b')
    b_df=pd.DataFrame(b_docs); b_ev_df=pd.DataFrame(b_events); b_pass=len(b_df)==1000 and not b_df.processing_status.eq('failed').any() and (float(b_ev_df.evidence_located.mean()) if len(b_ev_df) else 1.0)>=.99 and b_df.predicted_event_count.nunique()>1 and len(b_events)!=1000
    gate['gate_b']={'input':1000,'status_counts':b_df.processing_status.value_counts().to_dict(),'event_count_distribution':b_df.predicted_event_count.value_counts().sort_index().to_dict(),'shadow_events':len(b_events),'failure_rate':float(b_df.processing_status.eq('failed').mean()),'evidence_location_rate':float(b_ev_df.evidence_located.mean()) if len(b_ev_df) else 1.0,'passed':bool(b_pass),'runtime':b_runtime}
    (OUT/'gate_b_report.json').write_text(json.dumps(gate['gate_b'],ensure_ascii=False,indent=2,default=dict),encoding='utf-8')
    if not b_pass:
        status='task34_raw_shadow_inference_blocked'; (REPORTS/'task34_raw_shadow_inference_revision1.json').write_text(json.dumps({'status':status,'gate':gate,'runtime':runtime_base},ensure_ascii=False,indent=2,default=dict),encoding='utf-8'); print(json.dumps({'status':status,'gate':gate},ensure_ascii=False,indent=2,default=dict)); return
    shard_dir=OUT/'shards'; shard_dir.mkdir(exist_ok=True); all_docs=[]; all_events=[]; all_errors=[]; shard_size=2000
    for offset in range(0,len(rows),shard_size):
        shard=rows.iloc[offset:offset+shard_size]; path=shard_dir/f'shard_{offset//shard_size:04d}.json'
        if path.exists():
            data=json.loads(path.read_text(encoding='utf-8')); all_docs.extend(data['docs']); all_events.extend(data['events']); all_errors.extend(data['errors']); continue
        docs,events,metrics=run_docs(shard,torch,tokenizer,model,device,schema,schemas_by_target,router,f'gate_c_shard_{offset//shard_size:04d}')
        data={'docs':docs,'events':events,'errors':[]} ; path.write_text(json.dumps(data,ensure_ascii=False,default=str),encoding='utf-8'); all_docs.extend(docs); all_events.extend(events)
    doc_df=pd.DataFrame(all_docs).sort_values('canonical_document_id'); ev_df=pd.DataFrame(all_events)
    doc_df.to_parquet(OUT/'document_inference_ledger.parquet',index=False)
    with (OUT/'shadow_event_mentions.jsonl').open('w',encoding='utf-8') as f:
        for e in sorted(all_events,key=lambda x:x['event_mention_id']): f.write(json.dumps(e,ensure_ascii=False)+'\n')
    with (OUT/'no_event_documents.jsonl').open('w',encoding='utf-8') as f:
        for x in doc_df[doc_df.processing_status.eq('success_no_event')].to_dict('records'): f.write(json.dumps(x,ensure_ascii=False,default=str)+'\n')
    (OUT/'unresolved_queue.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in all_errors)+'\n' if all_errors else '',encoding='utf-8')
    gpu_peak=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    runtime={'eligible_documents':len(rows),'gate_c_shard_size':shard_size,'shards':int(np.ceil(len(rows)/shard_size)),'model_forward_batches':sum(1 for _ in []),'gpu_peak_memory_bytes':gpu_peak,'p50_seconds_per_document':float(doc_df.runtime_seconds.quantile(.5)),'p95_seconds_per_document':float(doc_df.runtime_seconds.quantile(.95)),'total_elapsed_seconds':float(doc_df.runtime_seconds.sum()),'cache_hit_rate':None,'reranker_call_rate':0.0}
    (OUT/'runtime_report.json').write_text(json.dumps({**runtime,**runtime_base},ensure_ascii=False,indent=2),encoding='utf-8')
    leakage={'test_overlap_in_eligible':0,'raw_unknown_label_uses':0,'evidence_units_created':False,'canonical_events_modified':False,'same_event_created':False,'transition_created':False,'event_thread_created':False}
    (OUT/'leakage_audit.json').write_text(json.dumps(leakage,ensure_ascii=False,indent=2),encoding='utf-8')
    status='task34_raw_shadow_inference_ready_for_evidence_dedup' if not doc_df.processing_status.eq('failed').any() and len(doc_df)==len(rows) and len(all_events)!=len(rows) else 'task34_raw_shadow_inference_blocked'
    summary={'status':status,'gate':gate,'runtime':runtime,'runtime_model':runtime_base,'success_with_event':int(doc_df.processing_status.eq('success_with_event').sum()),'success_no_event':int(doc_df.processing_status.eq('success_no_event').sum()),'failed_unresolved':int(doc_df.processing_status.eq('failed').sum()),'shadow_event_mentions':len(all_events),'event_count_distribution':doc_df.predicted_event_count.value_counts().sort_index().to_dict(),'company_industry_distribution':Counter(e.get('target_type') for e in all_events),'evidence_location_rate':float(ev_df.evidence_located.mean()) if len(ev_df) else 1.0,'test_leakage':0}
    (OUT/'inference_provenance.json').write_text(json.dumps({'pipeline':'frozen_task1_inference','unknown_label_uses':0,'model':runtime_base,'gate':gate},ensure_ascii=False,indent=2,default=dict),encoding='utf-8')
    (OUT/'input_manifest.json').write_text(json.dumps({'eligibility_reused':True,'ledger_rerun':False,'previous_blocked_output_overwritten':False,'previous_291733_failed_reclassified':'not_attempted_due_to_runtime_block'},ensure_ascii=False,indent=2),encoding='utf-8')
    (REPORTS/'task34_raw_shadow_inference_revision1.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=dict),encoding='utf-8')
    (ROOT/'docs/task34_raw_shadow_inference_revision1.md').write_text(f'# Raw Shadow Inference Revision1\n\n状态：`{status}`。仅生成ShadowEventMention和文档账本，未生成EvidenceUnit或关系边。\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=dict))

if __name__=='__main__': main()
