import argparse

import pandas as pd

from _event_common import ROOT, read_jsonl, write_jsonl
from src.event_clustering.source_normalizer import load_aliases, normalize_source


QUALITY_VALUES = {'auto_accept', 'llm_accepted', 'joint_adjudication', 'manual_review', 'reject', 'rejected'}


def backfill_rows(rows, split, grouped, aliases):
    result = []
    for row in rows:
        row = dict(row)
        matches = grouped.get((split, str(row.get('article_id', ''))), [])
        metadata = matches[0] if len(matches) == 1 else None
        old_source = row.get('source', '')
        if old_source and (normalize_source(old_source, aliases) == 'unknown' or str(old_source).strip().lower() in QUALITY_VALUES):
            row['legacy_source'] = old_source
        raw = metadata.get('article_source', '') if metadata else ''
        row.update({'dataset_split': split, 'source_raw': raw, 'source_normalized': normalize_source(raw, aliases),
                    'source': normalize_source(raw, aliases), 'source_name': normalize_source(raw, aliases)})
        row['source_lookup_status'] = ('duplicate_article_metadata' if len(matches) > 1 else
                                       'matched' if metadata is not None and raw else
                                       'missing_source' if metadata is not None else 'missing_article')
        result.append(row)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_events', required=True)
    ap.add_argument('--test_events', required=True)
    ap.add_argument('--metadata_index', required=True)
    ap.add_argument('--output_dir', default='data/processed/event_clusters')
    args = ap.parse_args()
    metadata = pd.read_parquet(args.metadata_index).to_dict('records')
    grouped = {}
    for row in metadata:
        grouped.setdefault((str(row['dataset_split']), str(row['article_id'])), []).append(row)
    aliases = load_aliases(ROOT / 'configs/event_clustering/source_aliases.json')
    train = backfill_rows(read_jsonl(args.train_events), 'train', grouped, aliases)
    test = backfill_rows(read_jsonl(args.test_events), 'test', grouped, aliases)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'final_events_with_sources_train.jsonl', train)
    write_jsonl(out / 'final_events_with_sources_test.jsonl', test)
    write_jsonl(out / 'all_final_events_with_sources.jsonl', train + test)
    pd.DataFrame(train + test).to_parquet(out / 'final_events_with_sources.parquet', index=False)
    print(len(train) + len(test))


if __name__ == '__main__':
    main()
