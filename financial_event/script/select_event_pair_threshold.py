import argparse
import json

from _event_common import ROOT, read_jsonl


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_file',default='data/processed/event_clusters/weak_labeled_event_pairs.jsonl'); ap.add_argument('--output_file',default='configs/event_clustering/event_pair_threshold.json'); args=ap.parse_args()
    rows=[r for r in read_jsonl(ROOT/args.input_file) if r.get('split')=='validation']
    if not rows: raise RuntimeError('validation weak-label set is empty; refusing to invent a threshold')
    best=None
    for i in range(0,101):
        threshold=i/100; tp=sum(float(r['same_event_score'])>=threshold and r['label']==1 for r in rows); fp=sum(float(r['same_event_score'])>=threshold and r['label']==0 for r in rows); fn=sum(float(r['same_event_score'])<threshold and r['label']==1 for r in rows); precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0; candidate=(f1,precision,recall,threshold)
        if best is None or candidate>best: best=candidate
    f1,precision,recall,threshold=best; result={'threshold':threshold,'validation_precision':precision,'validation_recall':recall,'validation_f1':f1,'validation_pair_count':len(rows)}; output=ROOT/args.output_file; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
