import hashlib, re
from .blocking import event_date, jaccard, token_set

def _text(r): return ' '.join(str(r.get(k,'') or '') for k in ('event_name','title','normalized_title','evidence_sentences','content','event_action','event_object'))
def _overlap(a,b):
    x,y=token_set(a),token_set(b); return len(x&y)/max(1,len(x)) if x else 0.0
def _numbers(s): return set(re.findall(r'\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:亿|万|万元|亿元)', str(s or '')))

def build_pair_features(a,b):
    da,db=event_date(a),event_date(b)
    pd = abs((da-db).days) if da and db else 9999
    ta,tb = a.get('title',''),b.get('title','')
    ea,eb = a.get('evidence_sentences',''),b.get('evidence_sentences','')
    full_a,full_b=_text(a),_text(b)
    nums_a,nums_b=_numbers(full_a),_numbers(full_b)
    title_cos=jaccard(ta,tb); evidence_cos=jaccard(ea,eb); full=jaccard(full_a,full_b)
    same_source=str(a.get('source_name',a.get('source',''))) == str(b.get('source_name',b.get('source','')))
    na=re.sub(r'\W','',str(ta).lower()); nb=re.sub(r'\W','',str(tb).lower())
    return {'event_record_id_a':a['event_record_id'],'event_record_id_b':b['event_record_id'],'target_code':a.get('target_code',''),'schema_id':a.get('schema_id',''),'publish_day_difference':pd,'event_time_difference':pd,'same_day':pd==0,'within_3_days':pd<=3,'within_7_days':pd<=7,'within_30_days':pd<=30,'title_embedding_cosine':title_cos,'evidence_embedding_cosine':evidence_cos,'full_event_embedding_cosine':full,'title_token_jaccard':title_cos,'evidence_token_jaccard':evidence_cos,'subject_match':a.get('event_subject')==b.get('event_subject') and bool(a.get('event_subject')),'location_match':_overlap(a.get('location'),b.get('location'))>0,'action_similarity':jaccard(a.get('event_action'),b.get('event_action')),'object_similarity':jaccard(a.get('event_object'),b.get('event_object')),'product_name_overlap':jaccard(a.get('target_name'),b.get('target_name')),'project_name_overlap':jaccard(a.get('event_object'),b.get('event_object')),'organization_overlap':0.0,'person_overlap':0.0,'key_value_name_overlap':jaccard(a.get('key_values'),b.get('key_values')),'key_value_exact_match':bool(nums_a&nums_b),'key_value_conflict':bool(nums_a and nums_b and not nums_a&nums_b),'amount_conflict':False,'percentage_conflict':False,'date_conflict':False,'same_source':same_source,'exact_title_match':ta==tb and bool(ta),'normalized_title_match':na==nb and bool(na),'content_hash_match':hashlib.md5(str(a.get('content','')).encode()).hexdigest()==hashlib.md5(str(b.get('content','')).encode()).hexdigest() and bool(a.get('content')),'high_text_overlap':full>=.7}

def score_pair(f):
    time = 1.0 if f['within_3_days'] else (.8 if f['within_7_days'] else (.5 if f['within_30_days'] else 0))
    ent=max(f['product_name_overlap'],f['location_match']*.8,f['subject_match']*1.0)
    kv=1.0 if f['key_value_exact_match'] else (0.0 if f['key_value_conflict'] else .5)
    score=.35*f['full_event_embedding_cosine']+.15*f['evidence_embedding_cosine']+.10*f['title_embedding_cosine']+.10*time+.10*(.5*f['action_similarity']+.5*f['object_similarity'])+.10*ent+.10*kv
    return max(0.0,min(1.0,score))
