import argparse
import json

import pandas as pd

from _event_common import ROOT, read_jsonl, write_jsonl


FIELDS=('event_subject','event_action','event_object','event_status','location','impact_direction','key_values','evidence_sentences')


def read_table(path):
    path=ROOT/path
    return pd.read_parquet(path).to_dict('records') if path.suffix=='.parquet' else read_jsonl(path)


def merge_values(records, field):
    values=[]
    for row in records:
        value=row.get(field)
        if isinstance(value,list): values.extend(value)
        elif value not in (None,''): values.append(value)
    return list(dict.fromkeys(str(x) for x in values))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--events',dest='events'); ap.add_argument('--events_with_sources',dest='events_alias'); ap.add_argument('--clusters',required=True); ap.add_argument('--members',required=True); ap.add_argument('--output_dir',required=True); ap.add_argument('--rebuild_sources_only',action='store_true'); args=ap.parse_args()
    event_path=args.events or args.events_alias
    if not event_path: raise SystemExit('--events is required')
    events=read_table(event_path); clusters=read_jsonl(ROOT/args.clusters); members=read_table(args.members); lookup={(str(r.get('dataset_split')),str(r.get('article_id'))):r for r in events}; by_cluster={}
    for member in members: by_cluster.setdefault(str(member['cluster_id']),[]).append(member)
    output=[]
    for cluster in clusters:
        records=[]
        for member in by_cluster.get(str(cluster['cluster_id']),[]):
            record=lookup.get((str(member.get('dataset_split')),str(member.get('article_id'))))
            if record is None: raise RuntimeError(f'missing event for member {member}')
            records.append(record)
        if not records: raise RuntimeError(f'cluster {cluster["cluster_id"]} has no members')
        sources=sorted({str(r.get('source_normalized',r.get('source','unknown'))) for r in records}); sources=[s for s in sources if s not in ('','unknown')]; unknown=sum(str(r.get('source_normalized',r.get('source','unknown'))) in ('','unknown') for r in records)
        provenance={field:{'value':merge_values(records,field),'article_ids':[str(r.get('article_id')) for r in records if r.get(field) not in (None,'',[])],'sources':[str(r.get('source_normalized','unknown')) for r in records if r.get(field) not in (None,'',[])]} for field in FIELDS}
        item={'canonical_event_id':f'CAN-{cluster["cluster_id"]}','cluster_id':cluster['cluster_id'],'target_type':cluster.get('target_type',''),'target_code':cluster.get('target_code',''),'target_name':cluster.get('target_name',''),'schema_id':cluster.get('schema_id',''),'event_name':cluster.get('event_name',''),'canonical_event_summary':cluster.get('event_name',''),'event_subject':provenance['event_subject']['value'][0] if len(provenance['event_subject']['value'])==1 else provenance['event_subject']['value'],'event_action':provenance['event_action']['value'][0] if len(provenance['event_action']['value'])==1 else provenance['event_action']['value'],'event_object':provenance['event_object']['value'][0] if len(provenance['event_object']['value'])==1 else provenance['event_object']['value'],'event_status':provenance['event_status']['value'][0] if len(provenance['event_status']['value'])==1 else provenance['event_status']['value'],'location':provenance['location']['value'][0] if len(provenance['location']['value'])==1 else provenance['location']['value'],'impact_objects':merge_values(records,'impact_objects'),'impact_direction':provenance['impact_direction']['value'][0] if len(provenance['impact_direction']['value'])==1 else provenance['impact_direction']['value'],'key_values':merge_values(records,'key_values'),'supporting_article_count':len(records),'supporting_source_count':len(sources),'supporting_articles':[str(r.get('article_id')) for r in records],'supporting_event_records':[str(r.get('event_record_id')) for r in records],'supporting_sources':sources,'unknown_source_count':unknown,'source_normalized_values':sources,'evidence_sentences':merge_values(records,'evidence_sentences'),'field_provenance':provenance,'conflicts':[],'has_conflict':False,'cluster_cohesion_score':cluster.get('cluster_cohesion_score',1.0),'completion_confidence':cluster.get('cluster_cohesion_score',1.0),'requires_manual_review':False,'model_name_or_path':'rule_merge','hallucinated_completion':[]}; output.append(item)
    if sum(x['supporting_article_count'] for x in output)!=len(events): raise RuntimeError('canonical output does not cover all input events')
    out=ROOT/args.output_dir; out.mkdir(parents=True,exist_ok=True); write_jsonl(out/'canonical_multisource_events.jsonl',output); pd.DataFrame(output).to_parquet(out/'canonical_multisource_events.parquet',index=False); pd.DataFrame(output).to_excel(out/'canonical_multisource_events.xlsx',index=False)
    if not all((out/name).exists() and (out/name).stat().st_size>0 for name in ('canonical_multisource_events.jsonl','canonical_multisource_events.parquet','canonical_multisource_events.xlsx')): raise RuntimeError('canonical outputs were not fully written')
    print(json.dumps({'input_event_count':len(events),'canonical_event_count':len(output),'rebuild_sources_only':args.rebuild_sources_only},ensure_ascii=False))

if __name__=='__main__': main()
