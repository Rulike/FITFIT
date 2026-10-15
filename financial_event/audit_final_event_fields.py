import csv,json
from collections import Counter
from _event_common import ROOT,load_events
def main():
 rows=load_events(); fields=['article_id','title','content','publish_time','source','target_type','target_code','target_name','schema_id','event_name','event_subject','event_time','event_action','event_object','event_status','location','impact_objects','impact_direction','key_values','evidence_sentences','final_confidence_score']; miss=[]
 for r in rows:
  for f in fields:
   if r.get(f) in (None,'',[]): miss.append({'dataset_split':r['dataset_split'],'article_id':r.get('article_id'),'field':f})
 report={'train_event_count':sum(r['dataset_split']=='train' for r in rows),'test_event_count':sum(r['dataset_split']=='test' for r in rows),'total_event_count':len(rows),'missing_counts':dict(Counter(x['field'] for x in miss)),'duplicate_article_id_count':len(rows)-len({r.get('article_id') for r in rows}),'duplicate_article_id_schema_id_count':len(rows)-len({(r.get('article_id'),r.get('schema_id')) for r in rows})}
 d=ROOT/'reports/event_clustering'; d.mkdir(parents=True,exist_ok=True); (d/'final_event_field_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 with open(d/'final_event_missing_fields.csv','w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=['dataset_split','article_id','field']); w.writeheader(); w.writerows(miss)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
