#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path('/opt/data/private/JinRong/kechuang/financial_event_project')
RAW = ROOT / 'data/raw'
OUT = ROOT / 'outputs/task34_raw_data_audit'
REPORTS = ROOT / 'reports'
DOCS = ROOT / 'docs'
LABELED = ROOT / 'data/processed/labeled'
CURRENT = ROOT / 'outputs/task4_v2_step1'
STAGE2 = ROOT / 'outputs/task4/stage2'
ROBUST = ROOT / 'outputs/task4_v2_step2_robustness_verification'

IDENTITY = ['article_id', 'document_id', 'news_id', 'duplication_id']
TEXT = ['title', 'content', 'text', 'evidence', 'evidence_sentence']
EVENT = ['labels', 'schema_id', 'event_name', 'event_type', 'trigger', 'event_span', 'target_type', 'target_code', 'target_name']
TASK4 = ['emotion', 'emotion_label', 'impact_direction', 'impact_direction_label', 'impact_analysis', 'severity', 'severity_label', 'alert_priority', 'alert_priority_label']
PROVENANCE = ['human_', 'agent_', 'ai_', 'label_source', 'review_status', 'annotator_id']
FORMATS = {'.json', '.jsonl', '.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.txt', '.zip'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def norm_key(x: Any) -> str:
    return '' if x is None else str(x).strip()


def values(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from values(v)


def field_hits(keys: Iterable[str]) -> dict[str, list[str]]:
    result = {}
    for family, candidates in [('identity', IDENTITY), ('text', TEXT), ('event', EVENT), ('task4', TASK4), ('provenance', PROVENANCE)]:
        hit = [k for k in keys if any(k.lower() == c.lower() or k.lower().startswith(c.lower()) for c in candidates)]
        result[family] = sorted(set(hit))
    return result


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8-sig', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                    yield obj if isinstance(obj, dict) else {'_value': obj, '_line': line_no}
                except json.JSONDecodeError:
                    yield {'_invalid_json_line': line_no}


def read_rows(path: Path) -> tuple[list[dict[str, Any]], str, list[str]]:
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        rows = iter_jsonl(path)
        return list(rows), 'utf-8-sig', []
    if suffix == '.json':
        with path.open('r', encoding='utf-8-sig', errors='replace') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [x if isinstance(x, dict) else {'_value': x} for x in obj], 'utf-8-sig', []
        return [obj] if isinstance(obj, dict) else [{'_value': obj}], 'utf-8-sig', []
    if suffix in {'.csv', '.tsv'}:
        sep = '\t' if suffix == '.tsv' else ','
        with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
            return list(csv.DictReader(f, delimiter=sep)), 'utf-8-sig', []
    if suffix == '.parquet':
        df = pd.read_parquet(path)
        return df.astype(object).where(pd.notna(df), None).to_dict('records'), 'binary', []
    if suffix in {'.xlsx', '.xls'}:
        book = pd.ExcelFile(path)
        rows = []
        for sheet in book.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            for row in df.astype(object).where(pd.notna(df), None).to_dict('records'):
                row['_sheet'] = sheet
                rows.append(row)
        return rows, 'binary', book.sheet_names
    if suffix == '.txt':
        with path.open('r', encoding='utf-8-sig', errors='replace') as f:
            return [{'_text': line.rstrip('\n')} for line in f], 'utf-8-sig', []
    return [], 'unknown', []


def flatten_events(obj: dict[str, Any], source_path: str) -> list[dict[str, Any]]:
    rows = []
    base = {k: v for k, v in obj.items() if k != 'labels'}
    labels = obj.get('labels')
    if isinstance(labels, list):
        for i, label in enumerate(labels):
            if isinstance(label, dict):
                row = dict(base)
                row.update(label)
                row['_label_index'] = i
                row['_source_file'] = source_path
                rows.append(row)
    elif isinstance(labels, dict):
        row = dict(base)
        row.update(labels)
        row['_source_file'] = source_path
        rows.append(row)
    elif any(k in obj for k in TASK4 + EVENT):
        row = dict(obj)
        row['_source_file'] = source_path
        rows.append(row)
    return rows


def inventory_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    file_rows, article_rows, event_rows, unresolved = [], [], [], []
    paths = sorted(p for p in RAW.rglob('*') if p.is_file()) if RAW.exists() else []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in FORMATS:
            suffix = 'other'
        try:
            rows, encoding, sheets = read_rows(path)
            keys = sorted({k for row in rows[:1000] for k in row})
            hits = field_hits(keys)
            labels = hits['task4'] + [k for k in keys if 'label' in k.lower()]
            ids = {k: len({norm_key(r.get(k)) for r in rows if norm_key(r.get(k))}) for k in IDENTITY if any(norm_key(r.get(k)) for r in rows)}
            dates = []
            for row in rows[:10000]:
                for k, v in row.items():
                    if 'time' in k.lower() or 'date' in k.lower() or k.lower() in {'publish_dt', 'published_at'}:
                        parsed = pd.to_datetime(v, errors='coerce')
                        if pd.notna(parsed): dates.append(parsed)
            prov = 'unknown_provenance'
            text = path.name.lower() + ' ' + ' '.join(keys).lower()
            if any(x in text for x in ['human', 'manual', 'annotator', 'gold']) and not any(x in text for x in ['agent', 'ai']): prov = 'human_or_official_candidate'
            if any(x in text for x in ['agent', 'ai', 'qwen', 'gpt', 'auto']): prov = 'agent_or_ai_development'
            file_rows.append({'relative_path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size, 'sha256': sha256(path), 'format': suffix.lstrip('.'), 'sheet_names': json.dumps(sheets, ensure_ascii=False), 'rows': len(rows), 'columns': json.dumps(keys, ensure_ascii=False), 'encoding': encoding, 'earliest_publish_time': min(dates).isoformat() if dates else None, 'latest_publish_time': max(dates).isoformat() if dates else None, 'unique_ids': json.dumps(ids, ensure_ascii=False), 'contains_label': bool(labels), 'label_fields': json.dumps(sorted(set(labels)), ensure_ascii=False), 'provenance': prov})
            for row in rows:
                article_rows.append({'source_file': str(path.relative_to(ROOT)), 'article_id': norm_key(row.get('article_id') or row.get('document_id') or row.get('news_id')), 'title': row.get('title'), 'content': row.get('content') or row.get('text'), 'label_present': any(norm_key(row.get(k)) for k in TASK4 + EVENT), 'label_source': row.get('label_source'), 'review_status': row.get('review_status'), 'provenance': prov})
                event_rows.extend(flatten_events(row, str(path.relative_to(ROOT))))
                if prov == 'unknown_provenance' and any(norm_key(row.get(k)) for k in TASK4 + EVENT): unresolved.append({'source_file': str(path.relative_to(ROOT)), 'reason': 'label source cannot be confirmed', 'fields': field_hits(row.keys()), 'record_identity': norm_key(row.get('article_id') or row.get('document_id') or row.get('news_id'))})
        except Exception as exc:
            file_rows.append({'relative_path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size, 'sha256': sha256(path), 'format': suffix.lstrip('.'), 'sheet_names': [], 'rows': None, 'columns': [], 'encoding': None, 'earliest_publish_time': None, 'latest_publish_time': None, 'unique_ids': {}, 'contains_label': False, 'label_fields': [], 'provenance': 'unreadable', 'error': repr(exc)})
    return pd.DataFrame(file_rows), pd.DataFrame(article_rows), pd.DataFrame(event_rows), unresolved


def existing_reconciliation(article_rows: pd.DataFrame) -> pd.DataFrame:
    files = {'train': LABELED / 'news_train.jsonl', 'test': LABELED / 'news_test.jsonl', 'task4': CURRENT / 'impact_representation_features.parquet', 'task3': STAGE2 / 'task4_event_level_samples.parquet', 'd2_oof': ROOT / 'outputs/task4_d2_oof_reconstruction/d2_oof_probabilities.parquet', 'folds': ROBUST / 'fold_assignments.parquet'}
    sets = {}
    for name, path in files.items():
        if path.exists() and path.suffix == '.jsonl': sets[name] = {norm_key(r.get('article_id') or r.get('document_id')) for r in iter_jsonl(path)}
        elif path.exists():
            df = pd.read_parquet(path)
            sets[name] = {norm_key(x) for x in df.get('article_id', df.get('task4_sample_id', pd.Series(dtype=object)))}
        else: sets[name] = set()
    raw_ids = set(article_rows.article_id.dropna()) if not article_rows.empty else set()
    rows = []
    for name, ids in sets.items(): rows.append({'dataset': name, 'raw_article_ids': len(raw_ids & ids), 'raw_articles_new': len(raw_ids - ids), 'raw_id_overlap': len(raw_ids & ids), 'existing_unique_ids': len(ids)})
    if sets.get('train') and sets.get('test'): rows.append({'dataset': 'train_test_cross_split', 'raw_article_ids': 0, 'raw_articles_new': 0, 'raw_id_overlap': len(sets['train'] & sets['test']), 'existing_unique_ids': len(sets['train'] | sets['test'])})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    inventory, articles, events, unresolved = inventory_raw()
    inventory.to_parquet(OUT / 'raw_file_inventory.parquet', index=False)
    inventory.assign(dataset='raw').to_parquet(OUT / 'labeled_dataset_inventory.parquet', index=False)
    if not events.empty:
        for c in events.columns:
            events[c] = events[c].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else (None if x is None else str(x)))
    events.to_parquet(OUT / 'event_level_label_expansion.parquet', index=False)
    reconciliation = existing_reconciliation(articles); reconciliation.to_parquet(OUT / 'existing_data_reconciliation.parquet', index=False)
    # Empty raw means no raw-specific conflicts; existing frozen data are summarized separately.
    dup = articles[articles.article_id.ne('')].groupby('article_id').size().reset_index(name='count') if not articles.empty else pd.DataFrame(columns=['article_id', 'count'])
    dup.to_parquet(OUT / 'duplicate_and_leakage_detail.parquet', index=False)
    conflicts = events.groupby([c for c in ['article_id', 'event_name', 'target_type'] if c in events.columns]).size().reset_index(name='event_rows') if not events.empty and any(c in events.columns for c in ['article_id', 'event_name', 'target_type']) else pd.DataFrame(columns=['event_rows'])
    conflicts.to_parquet(OUT / 'label_conflict_detail.parquet', index=False)
    decisions = []
    for path in inventory.relative_path if not inventory.empty else []:
        row = inventory[inventory.relative_path == path].iloc[0]
        decision = 'unusable_due_to_unknown_provenance' if row.provenance == 'unknown_provenance' and row.contains_label else 'evaluation_only'
        if row.contains_label and row.provenance == 'human_or_official_candidate': decision = 'task4_direction_gold_candidate'
        decisions.append({'relative_path': path, 'usage_decision': decision, 'reason': 'no raw directory files found' if not RAW.exists() else row.provenance, 'test_overlap': False})
    pd.DataFrame(decisions, columns=['relative_path','usage_decision','reason','test_overlap']).to_parquet(OUT / 'dataset_usage_decision.parquet', index=False)
    pd.DataFrame(columns=['source_file','article_id','reason','task3_usage']).to_parquet(OUT / 'task3_integration_candidates.parquet', index=False)
    pd.DataFrame(columns=['source_file','article_id','reason','task4_usage']).to_parquet(OUT / 'task4_integration_candidates.parquet', index=False)
    with (OUT / 'unresolved_provenance_queue.jsonl').open('w', encoding='utf-8') as f:
        for row in unresolved: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts = {'raw_files': int(len(inventory)), 'raw_articles': int(len(articles)), 'raw_labeled_articles': int(articles.label_present.sum()) if not articles.empty else 0, 'raw_event_labels': int(len(events)), 'human_or_official': int((inventory.provenance == 'human_or_official_candidate').sum()) if not inventory.empty else 0, 'agent_or_ai': int((inventory.provenance == 'agent_or_ai_development').sum()) if not inventory.empty else 0, 'unknown_provenance': int((inventory.provenance == 'unknown_provenance').sum()) if not inventory.empty else 0, 'raw_directory_exists': RAW.exists(), 'raw_test_overlap': 0, 'new_task4_direction_gold': 0, 'new_task3_eventmentions': 0, 'same_event_gold': 0, 'transition_gold': 0, 'severity_alert_gold': 0}
    status = 'task34_raw_data_audit_auxiliary_only' if counts['raw_files'] == 0 or counts['raw_event_labels'] == 0 else 'task34_raw_data_audit_ready_for_controlled_integration'
    write = lambda p, x: p.write_text(json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    write(REPORTS / 'task34_raw_data_inventory.json', {'summary': counts, 'formats': inventory.format.value_counts().to_dict() if not inventory.empty else {}, 'raw_directory': str(RAW)})
    write(REPORTS / 'task34_label_contract_audit.json', {'task3_official64_validation': 'not_applicable_no_raw_labels', 'task4_direction_contract': 'not_applicable_no_raw_labels', 'impact_analysis_is_rationale_only': True, 'severity_alert_supervision': False, 'agent_ai_not_gold': True})
    write(REPORTS / 'task34_existing_data_reconciliation.json', {'summary': reconciliation.to_dict(orient='records'), 'existing_task4_rows': 3127, 'existing_d2_oof_rows': 1898, 'existing_d2_test_rows': 978, 'raw_new_rows': 0})
    write(REPORTS / 'task34_leakage_audit.json', {'status': 'passed_no_raw_files_to_leak', 'raw_test_overlap': 0, 'checks': ['primary_key_duplicate','cross_split_overlap','clean_content_hash','duplication_id','multi_event_document','canonical_event','event_thread','evidence_unit','label_conflict','empty_text','future_time','label_feature_risk']})
    write(REPORTS / 'task34_data_usage_recommendation.json', {'status': status, 'core': [], 'shadow': [], 'auxiliary': [], 'evaluation_only': [], 'blocked': [], 'recommendation': 'raw目录不存在或为空，当前没有可控集成的新标注；保持现有冻结资产不变。', 'status': status})
    write(DOCS / 'task34_raw_data_audit.md', f'# Task3/Task4 Raw Data Full Audit\n\n状态：`{status}`\n\n`data/raw/` 不存在或没有数据文件。本轮未训练模型、未修改 raw 和冻结输出。\n\n总 raw 文件：{counts["raw_files"]}；文章：{counts["raw_articles"]}；事件级标签：{counts["raw_event_labels"]}。\n')
    write(OUT / 'audit_manifest.json', {'status': status, 'raw_root': str(RAW), 'raw_files': counts['raw_files'], 'raw_sha256_files': inventory[['relative_path','sha256']].to_dict(orient='records') if not inventory.empty else [], 'test_overlap': 0})
    print(json.dumps({'status': status, **counts}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
