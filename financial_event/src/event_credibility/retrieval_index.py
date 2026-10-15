from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import shutil
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.event_credibility import fulltext


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache/event_credibility/dea_retrieval_index"
INDEX_VERSION = "task2_retrieval_index_v1"
SQLITE_FILENAME = "retrieval_index.sqlite"
METADATA_FILENAME = "retrieval_index_metadata.json"
DEFAULT_CACHE_MAX_ENTRIES = 4096
DEFAULT_BUILD_BATCH_SIZE = 2000


def normalize_text(value: Any) -> str:
    return fulltext.normalize_text(value)


def tokenize_keywords(value: Any) -> list[str]:
    text = normalize_text(value)
    pieces = re.findall(r"[A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}", text)
    out: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        if piece in seen:
            continue
        if piece in {"关于", "事件", "主张", "相关", "公司", "股份", "有限", "集团"}:
            continue
        seen.add(piece)
        out.append(piece)
    return out


def target_aliases(row: dict[str, Any]) -> list[str]:
    target = normalize_text(row.get("target_name"))
    code = normalize_text(row.get("target_code"))
    schema_id = normalize_text(row.get("schema_id"))
    aliases: list[str] = []
    if target:
        aliases.append(target)
        short = target
        for suffix in ("股份有限公司", "有限责任公司", "有限公司", "集团股份", "控股集团", "股份", "集团"):
            if short.endswith(suffix) and len(short) > len(suffix) + 1:
                aliases.append(short[: -len(suffix)])
        aliases.append(re.sub(r"(股份有限|有限责任|有限|股份|集团|控股|公司)$", "", target))
    if code:
        aliases.extend(re.findall(r"\b\d{6}\b", code))
    aliases.extend(re.findall(r"\b\d{6}\b", schema_id))
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        value = normalize_text(alias).strip(" -_（）()")
        if len(value) < 2 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out[:5]


def claim_keywords(row: dict[str, Any], reason: str = "") -> list[str]:
    keys: list[str] = []
    for field in ("target_name", "event_name", "schema_id", "event_claim", "claim_text", "title_samples"):
        keys.extend(tokenize_keywords(row.get(field, "")))
    keys.extend(re.findall(r"\d+(?:\.\d+)?%?|\d+(?:\.\d+)?[亿元万股吨片台]+", normalize_text(row.get("event_claim", ""))))
    if reason:
        keys.extend(tokenize_keywords(reason))
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        value = key.strip("_ -")
        if len(value) < 2 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out[:18]


def parse_dt(value: Any):
    text = normalize_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            from datetime import datetime

            return datetime.strptime(text, fmt)
        except Exception:
            continue
    m = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", text)
    if not m:
        return None
    try:
        from datetime import datetime

        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def fts_phrase(term: str) -> str:
    term = normalize_text(term).strip()
    if not term:
        return ""
    term = term.replace('"', '""')
    return f'"{term}"'


def build_fts_query(terms: list[str], column: str) -> str:
    phrases = [fts_phrase(term) for term in terms if normalize_text(term)]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return f"{column}:{phrases[0]}"
    return f"{column}:(" + " OR ".join(phrases) + ")"


def source_fingerprint(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    stat = path.stat()
    parquet = pq.ParquetFile(path)
    schema = [f"{field.name}:{field.type}" for field in parquet.schema_arrow]
    return {
        "source_path": str(path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": hasher.hexdigest(),
        "source_rows": parquet.metadata.num_rows if parquet.metadata else 0,
        "source_schema": schema,
        "index_version": INDEX_VERSION,
    }


def metadata_path(index_dir: Path) -> Path:
    return index_dir / METADATA_FILENAME


def sqlite_path(index_dir: Path) -> Path:
    return index_dir / SQLITE_FILENAME


def load_index_metadata(index_dir: Path) -> dict[str, Any]:
    path = metadata_path(index_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_index_valid(index_dir: Path, source_fp: dict[str, Any]) -> bool:
    db_path = sqlite_path(index_dir)
    meta_path = metadata_path(index_dir)
    if not db_path.exists() or not meta_path.exists():
        return False
    meta = load_index_metadata(index_dir)
    if not meta:
        return False
    keys = ("source_path", "source_size", "source_mtime_ns", "source_sha256", "source_rows", "source_schema", "index_version")
    for key in keys:
        if meta.get(key) != source_fp.get(key):
            return False
    if meta.get("sqlite_filename") != SQLITE_FILENAME:
        return False
    return True


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_sqlite_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma query_only=ON")
    return conn


@dataclass
class SearchReport:
    query: str
    normalized_terms: list[str]
    matching_mode: str
    recall_stage: str
    raw_document_count: int
    hit_document_count: int
    subject_filter_count: int
    time_filter_count: int
    relevance_filter_count: int
    dedupe_count: int
    candidate_evidence_count: int
    excluded_reason_counts: dict[str, int]
    top_document_ids: list[str]


class LRUCache:
    def __init__(self, max_entries: int = DEFAULT_CACHE_MAX_ENTRIES):
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._data: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.sets = 0

    def get(self, key: str) -> list[dict[str, Any]] | None:
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            self.hits += 1
            value = self._data.pop(key)
            self._data[key] = value
            return [dict(item) for item in value]

    def set(self, key: str, value: list[dict[str, Any]]) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = [dict(item) for item in value]
            self.sets += 1
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "size": len(self._data),
                "max_entries": self.max_entries,
                "hit_rate": round(self.hits / total, 6) if total else 0.0,
            }


class SQLiteRetrievalIndex:
    def __init__(self, source_path: Path, index_dir: Path | None = None, cache_max_entries: int = DEFAULT_CACHE_MAX_ENTRIES):
        self.source_path = source_path.resolve()
        self.index_dir = (index_dir or CACHE_DIR).resolve()
        self.db_path = sqlite_path(self.index_dir)
        self.meta_path = metadata_path(self.index_dir)
        self.cache = LRUCache(cache_max_entries)
        self.available = False
        self.error = ""
        self.fingerprint: dict[str, Any] = {}
        self._local = threading.local()
        self._load_lock = threading.Lock()
        self._build_lock_path = self.index_dir / ".build.lock"

    def load(self, auto_build: bool = True) -> None:
        self.fingerprint = source_fingerprint(self.source_path)
        if not is_index_valid(self.index_dir, self.fingerprint):
            if not auto_build:
                self.available = False
                self.error = "index metadata mismatch or missing"
                return
            try:
                build_task2_retrieval_index(self.source_path, self.index_dir)
            except Exception as exc:
                self.available = False
                self.error = f"{type(exc).__name__}: {exc}"[:800]
                return
        try:
            _ = self.connection()
            self.available = True
            self.error = ""
        except Exception as exc:
            self.available = False
            self.error = f"{type(exc).__name__}: {exc}"[:800]

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(self.db_path)
            conn = _open_sqlite_ro(self.db_path)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def build_if_needed(self) -> None:
        self.fingerprint = source_fingerprint(self.source_path)
        if is_index_valid(self.index_dir, self.fingerprint):
            return
        build_task2_retrieval_index(self.source_path, self.index_dir)

    def _cache_key(self, column: str, terms: list[str], limit: int, stage: str, context_key: str) -> str:
        payload = {
            "index_version": INDEX_VERSION,
            "source_sha256": self.fingerprint.get("source_sha256", ""),
            "source_size": self.fingerprint.get("source_size", 0),
            "source_mtime_ns": self.fingerprint.get("source_mtime_ns", 0),
            "limit": int(limit),
            "stage": stage,
            "column": column,
            "terms": [normalize_text(term) for term in terms if normalize_text(term)],
            "context": context_key,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _fetch_documents(self, column: str, terms: list[str], limit: int, context_key: str = "") -> list[dict[str, Any]]:
        query = build_fts_query(terms, column)
        if not query:
            return []
        key = self._cache_key(column, terms, limit, column, context_key)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        conn = self.connection()
        sql = """
            SELECT
                d.docid,
                d.article_id,
                d.article_title,
                d.article_source,
                d.article_publish_time,
                d.content,
                d.source_type,
                d.source_reliability
            FROM docs_fts
            JOIN docs d ON d.docid = docs_fts.rowid
            WHERE docs_fts MATCH ?
            ORDER BY d.docid
            LIMIT ?
        """
        rows = [dict(row) for row in conn.execute(sql, (query, int(limit))).fetchall()]
        self.cache.set(key, rows)
        return [dict(row) for row in rows]

    def query_candidates(self, row: dict[str, Any], query: str, limit: int = 400) -> tuple[pd.DataFrame, dict[str, Any]]:
        aliases = target_aliases(row)
        event_keys = [term for term in claim_keywords(row) if term not in aliases][:10]
        terms = [term for term in tokenize_keywords(query) if term]
        broad_terms = list(dict.fromkeys(aliases + terms + event_keys[:4]))
        context_key = json.dumps(
            {
                "query": normalize_text(query),
                "subject": "|".join(aliases),
                "time_range": normalize_text(row.get("event_time") or row.get("earliest_publish_time") or row.get("latest_publish_time")),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if not broad_terms:
            return pd.DataFrame(), {
                "query": query,
                "normalized_terms": [],
                "matching_mode": "v2_broad_recall_ranked",
                "matching_backend": "sqlite_fts5_trigram",
                "recall_stage": "empty",
                "raw_document_count": 0,
                "hit_document_count": 0,
                "subject_filter_count": 0,
                "time_filter_count": 0,
                "relevance_filter_count": 0,
                "dedupe_count": 0,
                "candidate_evidence_count": 0,
                "excluded_reason_counts": {},
                "top_document_ids": [],
            }

        subject_terms = aliases[:1] if aliases else []
        if len(aliases) > 1:
            subject_terms.extend(aliases[1:])
        subject_rows = self._fetch_documents("subject_text", subject_terms or broad_terms, limit, context_key=context_key)
        recall_stage = "subject_alias_title_source"
        if not subject_rows:
            subject_rows = self._fetch_documents("search_text", broad_terms, limit, context_key=context_key)
            recall_stage = "body_fallback"
        df = pd.DataFrame(subject_rows)
        if df.empty:
            report = {
                "query": query,
                "normalized_terms": broad_terms[:12],
                "matching_mode": "v2_broad_recall_ranked",
                "matching_backend": "sqlite_fts5_trigram",
                "recall_stage": recall_stage,
                "raw_document_count": 0,
                "hit_document_count": 0,
                "subject_filter_count": 0,
                "time_filter_count": 0,
                "relevance_filter_count": 0,
                "dedupe_count": 0,
                "candidate_evidence_count": 0,
                "excluded_reason_counts": {},
                "top_document_ids": [],
            }
            return df, report

        report = {
            "query": query,
            "normalized_terms": broad_terms[:12],
            "matching_mode": "v2_broad_recall_ranked",
            "matching_backend": "sqlite_fts5_trigram",
            "recall_stage": recall_stage,
            "raw_document_count": int(len(df)),
            "hit_document_count": int(len(df)),
            "subject_filter_count": int(len(df)),
            "time_filter_count": int(len(df)),
            "relevance_filter_count": int(len(df)),
            "dedupe_count": int(len(df)),
            "candidate_evidence_count": 0,
            "excluded_reason_counts": {},
            "top_document_ids": [normalize_text(item) for item in df["article_id"].astype(str).head(5).tolist()],
        }
        return df, report

    def cache_stats(self) -> dict[str, Any]:
        return self.cache.stats()

    def fetch_articles_by_ids(self, article_ids: list[str]) -> pd.DataFrame:
        ids = [normalize_text(item) for item in article_ids if normalize_text(item)]
        if not ids:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in ids)
        sql = f"""
            SELECT
                article_id,
                article_title,
                article_source,
                article_publish_time,
                content,
                source_type,
                source_reliability
            FROM docs
            WHERE article_id IN ({placeholders})
        """
        conn = self.connection()
        rows = [dict(row) for row in conn.execute(sql, ids).fetchall()]
        return pd.DataFrame(rows)


def build_task2_retrieval_index(source_path: Path, index_dir: Path | None = None) -> dict[str, Any]:
    source_path = source_path.resolve()
    index_dir = (index_dir or CACHE_DIR).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    lock_path = index_dir / ".build.lock"
    start = time.perf_counter()
    deadline = time.perf_counter() + 1800
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            if time.perf_counter() > deadline:
                raise TimeoutError(f"timed out waiting for retrieval index build lock: {lock_path}")
            time.sleep(0.5)
    temp_dir = Path(tempfile.mkdtemp(prefix="retrieval_index_", dir=str(index_dir)))
    temp_db = temp_dir / SQLITE_FILENAME
    temp_meta = temp_dir / METADATA_FILENAME
    conn: sqlite3.Connection | None = None
    try:
        fingerprint = source_fingerprint(source_path)
        conn = sqlite3.connect(temp_db)
        conn.execute("pragma journal_mode=OFF")
        conn.execute("pragma synchronous=OFF")
        conn.execute("pragma temp_store=MEMORY")
        conn.execute("pragma cache_size=-200000")
        conn.execute("pragma foreign_keys=OFF")
        conn.execute(
            """
            create table docs (
                docid integer primary key,
                article_id text not null,
                article_title text,
                article_source text,
                article_publish_time text,
                content text,
                source_type text,
                source_reliability real
            )
            """
        )
        conn.execute("create unique index docs_article_id_idx on docs(article_id)")
        conn.execute("create index docs_publish_time_idx on docs(article_publish_time)")
        conn.execute("create index docs_source_type_idx on docs(source_type)")
        conn.execute(
            """
            create virtual table docs_fts using fts5(
                article_id unindexed,
                subject_text,
                search_text,
                tokenize='trigram'
            )
            """
        )
        parquet = pq.ParquetFile(source_path)
        columns = ["article_id", "article_title", "article_source", "article_publish_time", "content", "source_type", "source_reliability"]
        inserted = 0
        batch_size = DEFAULT_BUILD_BATCH_SIZE
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            df = batch.to_pandas().fillna("")
            rows = []
            fts_rows = []
            for rec in df.to_dict("records"):
                docid = inserted + len(rows) + 1
                article_id = normalize_text(rec.get("article_id"))
                title = normalize_text(rec.get("article_title"))
                source = normalize_text(rec.get("article_source"))
                content = normalize_text(rec.get("content"))
                subject_text = f"{title} {source}".strip()
                search_text = f"{title} {source} {content}".strip()
                rows.append(
                    (
                        docid,
                        article_id,
                        title,
                        source,
                        normalize_text(rec.get("article_publish_time")),
                        content,
                        normalize_text(rec.get("source_type")),
                        float(rec.get("source_reliability") or 0.0),
                    )
                )
                fts_rows.append((docid, article_id, subject_text, search_text))
            conn.executemany(
                "insert into docs(docid, article_id, article_title, article_source, article_publish_time, content, source_type, source_reliability) values (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.executemany(
                "insert into docs_fts(rowid, article_id, subject_text, search_text) values (?, ?, ?, ?)",
                fts_rows,
            )
            inserted += len(rows)
        conn.commit()
        meta = {
            **fingerprint,
            "sqlite_filename": SQLITE_FILENAME,
            "index_dir": str(index_dir),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_seconds": round(time.perf_counter() - start, 3),
            "inserted_rows": inserted,
            "sqlite_version": sqlite3.sqlite_version,
            "fts_tokenizer": "trigram",
        }
        temp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        conn.close()
        conn = None
        final_db = sqlite_path(index_dir)
        final_meta = metadata_path(index_dir)
        os.replace(temp_db, final_db)
        os.replace(temp_meta, final_meta)
        return meta
    except Exception:
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        try:
            if temp_db.exists():
                temp_db.unlink()
        except Exception:
            pass
        try:
            if temp_meta.exists():
                temp_meta.unlink()
        except Exception:
            pass
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            lock_path.rmdir()
        except Exception:
            pass


_SINGLETONS: dict[tuple[str, str, int], SQLiteRetrievalIndex] = {}
_SINGLETON_LOCK = threading.Lock()


def load_task2_retrieval_index(source_path: Path, index_dir: Path | None = None, auto_build: bool = True, cache_max_entries: int = DEFAULT_CACHE_MAX_ENTRIES) -> SQLiteRetrievalIndex:
    source = source_path.resolve()
    index_root = (index_dir or CACHE_DIR).resolve()
    key = (str(source), str(index_root), int(cache_max_entries))
    with _SINGLETON_LOCK:
        existing = _SINGLETONS.get(key)
        if existing and existing.available:
            return existing
        index = SQLiteRetrievalIndex(source, index_dir=index_root, cache_max_entries=cache_max_entries)
        index.load(auto_build=auto_build)
        _SINGLETONS[key] = index
        return index
