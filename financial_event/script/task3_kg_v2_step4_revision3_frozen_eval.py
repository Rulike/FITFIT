#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'outputs/task3_kg_v2/step4_revision2_acceptance/frozen_test_package.xlsx'
REV3 = ROOT / 'outputs/task3_kg_v2/step4_revision3'
CONFIG = ROOT / 'configs/task3_kg_v2_step4_revision3.yaml'
REPORT_DIR = ROOT / 'reports'
DOCS_DIR = ROOT / 'docs'
OUT_JSON = REPORT_DIR / 'task3_kg_v2_step4_revision3_frozen_test_evaluation.json'
OUT_MD = DOCS_DIR / 'task3_kg_v2_step4_revision3_frozen_test_metrics.md'
OUT_XLSX = ROOT / 'outputs/task3_kg_v2/step4_revision3/error_analysis.xlsx'
HUMAN_FIELDS = ['human_same_event', 'human_update_relation', 'human_confidence', 'human_evidence', 'human_note', 'annotator_id', 'review_status']


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> dict[str, Any]:
    started = time.time()
    df = pd.read_excel(FROZEN)
    sha = {
        'revision3_config': sha256_file(CONFIG),
        'revision3_pair_predictions': sha256_file(REV3 / 'pair_predictions.parquet'),
        'revision3_canonical_events': sha256_file(REV3 / 'canonical_events.jsonl'),
        'revision3_canonical_event_members': sha256_file(REV3 / 'canonical_event_members.jsonl'),
    }
    human_non_empty = {c: int((df[c].fillna('').astype(str).str.strip() != '').sum()) for c in HUMAN_FIELDS if c in df.columns}
    human_complete = all(v > 0 for v in human_non_empty.values())
    report = {
        'version': 'task3_kg_v2_step4_revision3_frozen_test_eval_v0',
        'dataset': 'frozen_test',
        'status': 'pending_human_labels' if not human_complete else 'ready_for_read_only_evaluation',
        'human_field_non_empty_counts': human_non_empty,
        'rows': int(len(df)),
        'sha256_before_eval': sha,
        'formal_metrics': None if not human_complete else 'not_computed_in_this_branch',
        'constraints': {
            'read_only_revision3': True,
            'no_retraining': True,
            'no_threshold_tuning_from_frozen_test': True,
            'no_phase5_entry': True,
            'no_event_state_or_event_thread': True,
            'no_32b_field_backfill': True,
        },
        'resource_usage': {
            'elapsed_seconds': round(time.time() - started, 3),
            'max_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        },
    }
    write_json(OUT_JSON, report)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        '# Task3 KG v2 Step4 Revision3 Frozen Test Evaluation\n\n'
        + json.dumps(
            {
                'dataset': 'frozen_test',
                'status': report['status'],
                'rows': report['rows'],
                'human_field_non_empty_counts': human_non_empty,
                'sha256_before_eval': sha,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    if not human_complete:
        pd.DataFrame(columns=['pair_id', 'error_type', 'notes']).to_excel(OUT_XLSX, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    main()
