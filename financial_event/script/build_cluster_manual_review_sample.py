import argparse
import random

import pandas as pd

from _event_common import ROOT, read_jsonl


def sample_rows(clusters, low_pairs=None, under_pairs=None, seed=2026):
    rng = random.Random(seed); selected = {}; by_size={2:[],3:[],6:[],11:[]}
    for c in clusters:
        n=int(c.get('member_count',0) or 0)
        if n == 2: by_size[2].append(c)
        elif n == 3: by_size[3].append(c)
        elif 6 <= n <= 10: by_size[6].append(c)
        elif n > 10: by_size[11].append(c)
    for bucket, rows in by_size.items():
        chosen = rows if bucket == 11 else rng.sample(rows, min(100, len(rows)))
        for row in chosen: selected[str(row.get('cluster_id'))] = dict(row)
    for row in sorted(clusters, key=lambda x: int(x.get('member_count',0) or 0), reverse=True)[:50]: selected[str(row.get('cluster_id'))] = dict(row)
    for category, pairs in [('low_margin', low_pairs or []), ('possible_undermerge', under_pairs or [])]:
        for pair in rng.sample(pairs, min(100, len(pairs))):
            key=f"{category}:{pair.get('event_record_id_a', pair.get('article_id_a'))}:{pair.get('event_record_id_b', pair.get('article_id_b'))}"
            selected[key] = {'review_item_type': category, **pair}
    return [{**row, 'manual_same_event':'', 'manual_cluster_purity':'', 'manual_should_split':'', 'manual_should_merge':'', 'manual_source_correct':'', 'manual_comment':''} for row in selected.values()]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clusters',default='outputs/event_clustering/event_clusters.jsonl'); ap.add_argument('--low_margin',default='reports/event_clustering/low_margin_cluster_pairs.csv'); ap.add_argument('--undermerged',default='reports/event_clustering/possible_undermerged_pairs.csv'); ap.add_argument('--output',default='reports/event_clustering/cluster_manual_review_sample.xlsx'); args=ap.parse_args()
    clusters=read_jsonl(ROOT/args.clusters)
    def read_csv(p):
        path=ROOT/p
        return pd.read_csv(path).to_dict('records') if path.exists() else []
    rows=sample_rows(clusters,read_csv(args.low_margin),read_csv(args.undermerged)); out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_excel(out,index=False); print(len(rows))

if __name__=='__main__': main()
