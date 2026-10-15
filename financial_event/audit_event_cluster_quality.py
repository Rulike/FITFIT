import argparse
import json
from pathlib import Path

import pandas as pd

from _event_common import ROOT, read_jsonl

LOW_MARGIN_GAP = 0.05
LOW_MARGIN_UPPER = 0.60
DEFAULT_THRESHOLD = 0.55

LARGEST_CLUSTER_COLUMNS = [
    'cluster_id',
    'target_code',
    'target_name',
    'schema_id',
    'event_name',
    'member_count',
    'supporting_source_count',
    'earliest_publish_time',
    'latest_publish_time',
    'time_span_days',
    'cluster_cohesion_score',
    'titles',
    'locations',
    'project_names',
    'product_names',
    'key_values',
    'computed_time_span_days',
]

OVERMERGED_CLUSTER_COLUMNS = LARGEST_CLUSTER_COLUMNS + ['reasons']
PAIR_COLUMNS = [
    'event_record_id_a',
    'event_record_id_b',
    'article_id_a',
    'article_id_b',
    'cluster_id_a',
    'cluster_id_b',
    'target_code_a',
    'target_code_b',
    'target_code',
    'target_name',
    'schema_id_a',
    'schema_id_b',
    'schema_id',
    'same_event_score',
]
UNDERMERGED_COLUMNS = PAIR_COLUMNS
LOW_MARGIN_COLUMNS = PAIR_COLUMNS


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_table(path):
    path = resolve_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        if path.suffix == '.parquet':
            return pd.read_parquet(path)
        return pd.DataFrame(read_jsonl(path))
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


def values(rows, fields):
    result = set()
    for row in rows:
        for field in fields:
            value = row.get(field)
            if isinstance(value, list):
                result.update(str(item) for item in value if str(item).strip())
            elif value is not None and str(value).strip():
                result.add(str(value).strip())
    return result


def get_value(row, *names):
    for name in names:
        if name in row:
            value = row.get(name)
            if value is not None and str(value).strip():
                return value
    return None


def pair_common_row(pair):
    target_code_a = get_value(pair, 'target_code_a', 'target_code')
    target_code_b = get_value(pair, 'target_code_b', 'target_code')
    schema_id_a = get_value(pair, 'schema_id_a', 'schema_id')
    schema_id_b = get_value(pair, 'schema_id_b', 'schema_id')

    target_code = pair.get('target_code')
    if target_code is None and target_code_a == target_code_b:
        target_code = target_code_a
    if target_code is None:
        target_code = target_code_a if target_code_a is not None else target_code_b

    schema_id = pair.get('schema_id')
    if schema_id is None and schema_id_a == schema_id_b:
        schema_id = schema_id_a
    if schema_id is None:
        schema_id = schema_id_a if schema_id_a is not None else schema_id_b

    target_name = get_value(pair, 'target_name', 'target_name_a', 'target_name_b')

    return {
        'event_record_id_a': get_value(pair, 'event_record_id_a'),
        'event_record_id_b': get_value(pair, 'event_record_id_b'),
        'article_id_a': get_value(pair, 'article_id_a'),
        'article_id_b': get_value(pair, 'article_id_b'),
        'cluster_id_a': get_value(pair, 'cluster_id_a'),
        'cluster_id_b': get_value(pair, 'cluster_id_b'),
        'target_code_a': target_code_a,
        'target_code_b': target_code_b,
        'target_code': target_code,
        'target_name': target_name,
        'schema_id_a': schema_id_a,
        'schema_id_b': schema_id_b,
        'schema_id': schema_id,
        'same_event_score': pair.get('same_event_score'),
    }


def resolve_cluster_id(pair, member_cluster_by_event, side):
    cluster_id = get_value(pair, f'cluster_id_{side}')
    if cluster_id is not None:
        return str(cluster_id)

    for key in (f'event_record_id_{side}', f'article_id_{side}'):
        event_id = get_value(pair, key)
        if event_id is None:
            continue
        resolved = member_cluster_by_event.get(str(event_id))
        if resolved is not None and str(resolved).strip():
            return str(resolved)
    return None


def audit_clusters(clusters, events, pairs, threshold=DEFAULT_THRESHOLD, time_windows=None, member_cluster_by_event=None):
    by_article = {}
    for row in events:
        by_article.setdefault(str(row.get('article_id')), []).append(row)

    largest = []
    overmerged = []
    for cluster in clusters:
        ids = [str(x) for x in cluster.get('member_article_ids', [])]
        rows = [by_article[x][0] for x in ids if x in by_article]
        member_count = int(cluster.get('member_count', len(ids)) or 0)
        cohesion = float(cluster.get('cluster_cohesion_score', 1) or 0)
        dates = pd.to_datetime([r.get('publish_time', r.get('article_publish_time')) for r in rows], errors='coerce').dropna()
        span = float((dates.max() - dates.min()).total_seconds() / 86400) if len(dates) else float(cluster.get('time_span_days', 0) or 0)
        target_type = str(cluster.get('target_type', 'company'))
        limit = (time_windows or {}).get(target_type, (time_windows or {}).get('default_company_days', 30))
        row = {k: cluster.get(k) for k in ('cluster_id', 'target_code', 'target_name', 'schema_id', 'event_name', 'member_count', 'supporting_source_count', 'earliest_publish_time', 'latest_publish_time', 'time_span_days', 'cluster_cohesion_score')}
        row.update({
            'titles': ' || '.join(str(r.get('title', '')) for r in rows[:50]),
            'locations': ' || '.join(sorted(values(rows, ['location', 'locations']))),
            'project_names': ' || '.join(sorted(values(rows, ['project_name', 'project_names']))),
            'product_names': ' || '.join(sorted(values(rows, ['product_name', 'product_names']))),
            'key_values': ' || '.join(sorted(values(rows, ['key_values', 'amount', 'amounts']))),
            'computed_time_span_days': span,
        })
        largest.append(row)

        reasons = []
        if member_count > 20:
            reasons.append('member_count_gt_20')
        if span > float(limit):
            reasons.append('time_span_exceeds_schema_window')
        if cohesion < 0.68:
            reasons.append('cohesion_below_0.68')
        if len(values(rows, ['project_name', 'project_names'])) > 1:
            reasons.append('multiple_project_names')
        if len(values(rows, ['product_name', 'product_names'])) > 1:
            reasons.append('multiple_product_names')
        if len(values(rows, ['location', 'locations'])) > 1:
            reasons.append('location_conflict')
        if len(values(rows, ['amount', 'amounts'])) > 1:
            reasons.append('amount_conflict')
        if reasons:
            overmerged.append({**row, 'reasons': '|'.join(reasons)})

    low_margin = []
    undermerged = []
    undermerged_lower = max(0.0, float(threshold) - LOW_MARGIN_GAP)
    undermerged_upper = float(threshold)
    low_margin_upper = min(1.0, LOW_MARGIN_UPPER)

    for pair in pairs:
        score = float(pair.get('same_event_score', 0) or 0)
        common = pair_common_row(pair)
        same_target = common['target_code_a'] is not None and common['target_code_a'] == common['target_code_b']
        same_schema = common['schema_id_a'] is not None and common['schema_id_a'] == common['schema_id_b']

        if float(threshold) <= score < low_margin_upper:
            low_margin.append(common)

        if (
            undermerged_lower <= score < undermerged_upper
            and same_target
            and same_schema
        ):
            cluster_a = resolve_cluster_id(pair, member_cluster_by_event or {}, 'a')
            cluster_b = resolve_cluster_id(pair, member_cluster_by_event or {}, 'b')
            if cluster_a is not None and cluster_b is not None and cluster_a != cluster_b:
                undermerged.append(common)

    return largest, overmerged, low_margin, undermerged, undermerged_lower, undermerged_upper, low_margin_upper


def ensure_csv(path, rows, columns):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        for column in columns:
            if column not in frame.columns:
                frame[column] = ''
        frame = frame[columns + [column for column in frame.columns if column not in columns]]
    frame.to_csv(path, index=False)


def load_threshold(config_path):
    config_path = resolve_path(config_path)
    if not config_path.exists() or config_path.stat().st_size == 0:
        return DEFAULT_THRESHOLD
    return float(json.loads(config_path.read_text(encoding='utf-8')).get('threshold', DEFAULT_THRESHOLD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clusters', default='outputs/event_clustering/event_clusters.jsonl')
    ap.add_argument('--members', default='outputs/event_clustering/event_cluster_members.parquet')
    ap.add_argument('--events', default='data/processed/event_clusters/final_events_with_sources.parquet')
    ap.add_argument('--pair_features', default='data/processed/event_clusters/event_pair_features.parquet')
    ap.add_argument('--threshold_config', default='configs/event_clustering/event_pair_threshold.json')
    args = ap.parse_args()

    clusters = read_jsonl(resolve_path(args.clusters))
    events = read_table(args.events).to_dict('records')
    pairs = read_table(args.pair_features).to_dict('records')

    member_cluster_by_event = {}
    members = read_table(args.members)
    if not members.empty:
        for row in members.to_dict('records'):
            cluster_id = row.get('cluster_id')
            if cluster_id is None or not str(cluster_id).strip():
                continue
            for key in ('event_record_id', 'article_id'):
                value = row.get(key)
                if value is not None and str(value).strip():
                    member_cluster_by_event[str(value)] = str(cluster_id)

    threshold = load_threshold(args.threshold_config)
    windows_path = resolve_path('configs/event_clustering/schema_time_windows.json')
    windows = json.loads(windows_path.read_text(encoding='utf-8')) if windows_path.exists() and windows_path.stat().st_size else {'default_company_days': 30}

    largest, over, low, under, undermerged_lower, undermerged_upper, low_margin_upper = audit_clusters(
        clusters,
        events,
        pairs,
        threshold,
        windows,
        member_cluster_by_event,
    )

    report_dir = ROOT / 'reports/event_clustering'
    report_dir.mkdir(parents=True, exist_ok=True)

    ensure_csv(report_dir / 'largest_clusters.csv', sorted(largest, key=lambda x: x.get('member_count') or 0, reverse=True)[:50], LARGEST_CLUSTER_COLUMNS)
    ensure_csv(report_dir / 'possible_overmerged_clusters.csv', over, OVERMERGED_CLUSTER_COLUMNS)
    ensure_csv(report_dir / 'low_margin_cluster_pairs.csv', low, LOW_MARGIN_COLUMNS)
    ensure_csv(report_dir / 'possible_undermerged_pairs.csv', under, UNDERMERGED_COLUMNS)

    report = {
        'cluster_count': len(clusters),
        'largest_cluster_size': max([int(c.get('member_count', 0) or 0) for c in clusters], default=0),
        'low_margin_pair_count': len(low),
        'possible_undermerged_pair_count': len(under),
        'possible_overmerged_cluster_count': len(over),
        'threshold': threshold,
        'low_margin_range': [float(threshold), low_margin_upper],
        'undermerged_range': [undermerged_lower, undermerged_upper],
    }
    (report_dir / 'cluster_quality_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
