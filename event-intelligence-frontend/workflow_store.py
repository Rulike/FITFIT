"""Persistent UI workflow state, separate from frozen model outputs."""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


DEFAULT_RULES = dict(high=75, critical=90, credibility=0, channel='平台消息')


class WorkflowStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=15)
        try:
            with connection:
                connection.execute('CREATE TABLE IF NOT EXISTS acknowledgements (event_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL)')
                connection.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
                connection.execute('CREATE TABLE IF NOT EXISTS watches (id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)')
                yield connection
        finally:
            connection.close()

    def acknowledgements(self):
        with self.connect() as connection:
            return dict(connection.execute('SELECT event_id, updated_at FROM acknowledgements'))

    def acknowledge(self, event_id):
        with self.connect() as connection:
            connection.execute('INSERT OR IGNORE INTO acknowledgements VALUES (?, ?)', (event_id, datetime.now().isoformat(timespec='seconds')))

    def rules(self):
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM settings WHERE key = 'alert_rules'").fetchone()
            return json.loads(row[0]) if row else dict(DEFAULT_RULES)

    def save_rules(self, payload):
        rules = dict(DEFAULT_RULES)
        try:
            for key in ('high', 'critical', 'credibility'):
                if isinstance(payload.get(key), bool):
                    raise ValueError()
                value = float(payload[key])
                if not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError()
                rules[key] = value
        except (KeyError, ValueError, TypeError):
            raise ValueError('阈值必须为 0 至 100 的有效数值')
        if rules['high'] >= rules['critical']:
            raise ValueError('重大预警阈值必须高于高影响预警阈值')
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO settings VALUES ('alert_rules', ?)", (json.dumps(rules),))
        return rules

    def watches(self):
        with self.connect() as connection:
            return [json.loads(row[0]) for row in connection.execute('SELECT payload FROM watches ORDER BY created_at DESC, id')]

    def create_watch(self, payload):
        target = str(payload.get('target') or '').strip()
        if not target:
            raise ValueError('请输入关注对象')
        watch = {key: str(payload.get(key) or '').strip()[:1000] for key in ('scope', 'level', 'keywords', 'note')}
        watch.update(id='WATCH-' + uuid.uuid4().hex, target=target[:200], createdAt=datetime.now().isoformat(timespec='seconds'))
        with self.connect() as connection:
            connection.execute('INSERT INTO watches VALUES (?, ?, ?)', (watch['id'], json.dumps(watch, ensure_ascii=False), watch['createdAt']))
        return watch
