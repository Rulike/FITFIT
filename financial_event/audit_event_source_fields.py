import argparse
import csv
import json
from collections import Counter

from _event_common import ROOT, load_events, read_jsonl
from src.event_clustering.source_normalizer import FORBIDDEN, normalize_source


def audit_rows(rows):
    invalid=[]
    for row in rows:
        raw=row.get('source_raw', row.get('source', ''))
        if str(raw).strip().lower() in FORBIDDEN or normalize_source(raw) == 'unknown':
            invalid.append({'dataset_split':row.get('dataset_split',''), 'article_id':row.get('article_id',''), 'source_raw':raw, 'reason':'forbidden_or_missing'})
    return invalid


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--events',default=''); args=ap.parse_args()
    rows=read_jsonl(ROOT/args.events) if args.events else load_events(); bad=audit_rows(rows)
    report={'input_event_count':len(rows),'invalid_source_value_count':len(bad),'invalid_values':dict(Counter(str(x['source_raw']) for x in bad)),'quality_status_used_as_source':sum(str(r.get('source','')).strip().lower() in FORBIDDEN for r in rows),'outputs_final_task1_modified':False}
    d=ROOT/'reports/event_clustering'; d.mkdir(parents=True,exist_ok=True); (d/'source_field_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (d/'invalid_source_values.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['dataset_split','article_id','source_raw','reason']); w.writeheader(); w.writerows(bad)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
