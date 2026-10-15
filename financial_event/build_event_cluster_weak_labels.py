import argparse
import json
import random

from _event_common import ROOT, read_jsonl, write_jsonl


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_file',default='data/processed/event_clusters/event_pair_features.parquet'); ap.add_argument('--output_file',default='data/processed/event_clusters/weak_labeled_event_pairs.jsonl'); args=ap.parse_args()
    import pandas as pd
    path=ROOT/args.input_file; rows=pd.read_parquet(path).to_dict('records') if path.suffix=='.parquet' else read_jsonl(path)
    if not rows: raise RuntimeError('cannot build weak labels from an empty feature file')
    out=[]
    for row in rows:
        positive=bool(row.get('normalized_title_match') or row.get('exact_title_match') or (float(row.get('full_event_embedding_cosine',0))>=.65 and not row.get('key_value_conflict',False)))
        negative=bool(row.get('key_value_conflict',False) and float(row.get('full_event_embedding_cosine',0))<.45)
        if positive or negative: out.append({**row,'label':1 if positive else 0,'label_source':'weak_rule'})
    if not out: raise RuntimeError('weak-label rules produced no labeled pairs')
    random.Random(2026).shuffle(out); n=len(out)
    for i,row in enumerate(out): row['split']='train' if i<n*.7 else ('validation' if i<n*.85 else 'test')
    write_jsonl(ROOT/args.output_file,out); d=ROOT/'reports/event_clustering'; d.mkdir(parents=True,exist_ok=True); (d/'weak_label_summary.json').write_text(json.dumps({'feature_rows':len(rows),'labeled_rows':n,'positive':sum(r['label']==1 for r in out),'negative':sum(r['label']==0 for r in out),'splits':{s:sum(r['split']==s for r in out) for s in ('train','validation','test')}},ensure_ascii=False,indent=2),encoding='utf-8'); print(n)

if __name__=='__main__': main()
