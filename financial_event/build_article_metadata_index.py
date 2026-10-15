import argparse
import json
from pathlib import Path

import pandas as pd

from _event_common import ROOT
from src.event_clustering.article_metadata import read_article_metadata
from src.event_clustering.source_normalizer import load_aliases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_file', required=True)
    ap.add_argument('--test_file', required=True)
    ap.add_argument('--output_dir', default='data/processed/event_clusters')
    args = ap.parse_args()
    aliases = load_aliases(ROOT / 'configs/event_clustering/source_aliases.json')
    rows = read_article_metadata(args.train_file, 'train', aliases) + read_article_metadata(args.test_file, 'test', aliases)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / 'article_metadata_index.parquet', index=False)
    (out / 'article_metadata_index.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8'
    )
    keys = [(r['dataset_split'], r['article_id']) for r in rows]
    report = {'records': len(rows), 'train': sum(r['dataset_split'] == 'train' for r in rows),
              'test': sum(r['dataset_split'] == 'test' for r in rows), 'unique_keys': len(set(keys)),
              'duplicate_keys': len(keys) - len(set(keys)),
              'unknown_sources': sum(r['source_normalized'] == 'unknown' for r in rows)}
    report_dir = ROOT / 'reports/event_clustering'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'article_metadata_index_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
