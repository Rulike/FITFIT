import argparse
import json
from collections import defaultdict
from pathlib import Path

from _event_common import ROOT, load_event_files, write_jsonl
from src.event_clustering.blocking import blocking_key, event_date, time_window_days, jaccard


def build_candidates(rows):
    if len({r['event_record_id'] for r in rows}) != len(rows):
        raise ValueError('event_record_id must be unique')
    groups=defaultdict(list)
    for row in rows: groups[blocking_key(row)].append(row)
    out=[]
    for values in groups.values():
        for i, left in enumerate(values):
            for right in values[i+1:]:
                dl,dr=event_date(left),event_date(right); diff=abs((dl-dr).days) if dl and dr else 0
                if diff>max(time_window_days(left),time_window_days(right)): continue
                reasons=[]
                if jaccard(left.get('title'),right.get('title'))>=.15: reasons.append('title_jaccard')
                if jaccard(left.get('evidence_sentences'),right.get('evidence_sentences'))>=.15: reasons.append('evidence_jaccard')
                if jaccard(left.get('title'),right.get('title'))>=.55 or jaccard(left.get('event_name'),right.get('event_name'))>=.55: reasons.append('semantic_overlap')
                if left.get('target_name') and left.get('target_name')==right.get('target_name'): reasons.append('entity_overlap')
                if left.get('event_object') and left.get('event_object')==right.get('event_object'): reasons.append('object_overlap')
                if reasons:
                    out.append({'event_record_id_a':left['event_record_id'],'event_record_id_b':right['event_record_id'],'target_code':left.get('target_code',''),'schema_id':left.get('schema_id',''),'publish_day_difference':diff,'event_time_difference':diff,'blocking_reasons':reasons})
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_events',nargs='+',required=True); ap.add_argument('--output_file',required=True); args=ap.parse_args()
    rows=load_event_files([ROOT/p for p in args.input_events]); candidates=build_candidates(rows); output=ROOT/args.output_file; output.parent.mkdir(parents=True,exist_ok=True); write_jsonl(output,candidates)
    report_dir=ROOT/'reports/event_clustering'; report_dir.mkdir(parents=True,exist_ok=True)
    summary={'input_event_count':len(rows),'candidate_pair_count':len(candidates),'output_file':str(output.relative_to(ROOT)),'non_empty':bool(candidates)}
    (report_dir/'blocking_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if not output.exists() or output.stat().st_size==0: raise RuntimeError('candidate file was not written or is empty')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
