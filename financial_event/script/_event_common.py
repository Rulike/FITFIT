import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def read_jsonl(p):
    with open(p,encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def write_jsonl(p,rows):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')
def load_events():
    rows=[]
    for split in ('train','test'):
        for r in read_jsonl(ROOT/f'outputs/final_task1/final_events_{split}.jsonl'):
            r['dataset_split']=split; r['event_record_id']=f"{split}:{r.get('event_id',r.get('article_id'))}"
            r['normalized_title']=''.join(str(r.get('title','')).lower().split()); r['normalized_evidence']=str(r.get('evidence_sentences',''))
            r['normalized_event_time']=str(r.get('event_time','')); r['publish_date']=str(r.get('publish_time',''))[:10]; r['source_name']=r.get('source','')
            rows.append(r)
    return rows

def load_event_file(path):
    rows=[]
    for r in read_jsonl(path):
        r=dict(r)
        split=str(r.get('dataset_split') or ('test' if str(path).lower().endswith('_test.jsonl') else 'train'))
        r['dataset_split']=split
        event_id=r.get('event_id') or r.get('article_id')
        r['event_record_id']=str(r.get('event_record_id') or f'{split}:{event_id}')
        r['source_name']=r.get('source_normalized',r.get('source_name',r.get('source','unknown')))
        rows.append(r)
    return rows

def load_event_files(paths):
    rows=[]
    for path in paths: rows.extend(load_event_file(path))
    return rows
