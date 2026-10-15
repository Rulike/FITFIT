from collections import Counter
def merge_values(records, field):
    vals=[]
    for r in records:
        v=r.get(field)
        if isinstance(v,list): vals.extend(v)
        elif v not in (None,'',[]): vals.append(v)
    return list(dict.fromkeys(map(str,vals)))
def merge_cluster(records):
    first=records[0]; out={}
    for f in ('target_code','target_name','schema_id','event_name','event_subject','event_action','event_object','event_status','location','impact_direction'):
        vals=merge_values(records,f); out[f]=vals[0] if len(vals)==1 else (vals if vals else '')
    out['sources']=list(dict.fromkeys(str(r.get('source_name',r.get('source',''))) for r in records if r.get('source_name',r.get('source',''))))
    out['article_ids']=[r['article_id'] for r in records]
    out['evidence_sentences']=merge_values(records,'evidence_sentences')
    out['impact_objects']=merge_values(records,'impact_objects')
    out['key_values']=merge_values(records,'key_values')
    return out
