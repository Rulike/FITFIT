import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path, allow_missing: bool = False) -> List[Dict[str, Any]]:
    path = resolve_path(path)
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_csv_rows(path: str | Path, allow_missing: bool = False) -> List[Dict[str, Any]]:
    path = resolve_path(path)
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Missing CSV file: {path}")
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, float):
        try:
            if value != value:  # NaN
                return None
        except Exception:
            return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = text.replace("T", " ").replace("Z", "")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text[:19] if fmt.endswith("%S") else text, fmt)
        except Exception:
            continue
    m = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})(?:日)?(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?", text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hour = int(m.group(4) or 0)
        minute = int(m.group(5) or 0)
        second = int(m.group(6) or 0)
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(clean_text(item) for item in value if clean_text(item))
    if isinstance(value, dict):
        return " ".join(clean_text(item) for item in value.values() if clean_text(item))
    return re.sub(r"\s+", " ", str(value)).strip()


def first_nonempty(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return value
    return ""


def format_datetime(value: Any) -> str:
    dt = parse_datetime(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
    return text


def tokenize(value: Any) -> List[str]:
    text = "" if value is None else str(value).lower()
    return re.findall(r"[0-9a-zA-Z]+|[\u4e00-\u9fff]", text)


def token_jaccard(a: Any, b: Any) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def sequence_ratio(a: Any, b: Any) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def lexical_similarity(a: Any, b: Any) -> float:
    ratio = sequence_ratio(a, b)
    jaccard = token_jaccard(a, b)
    return round(0.7 * ratio + 0.3 * jaccard, 6)


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def canonical_list(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = "" if value is None else str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def join_list(values: Iterable[Any], sep: str = "|") -> str:
    return sep.join(canonical_list(values))


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def unique_by_key(rows: Sequence[Dict[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def group_records(rows: Sequence[Dict[str, Any]], key_fn) -> Dict[Any, List[Dict[str, Any]]]:
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def ensure_parent(path: str | Path) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_task1_events(paths: list[str | Path] | None = None) -> list[dict[str, Any]]:
    paths = paths or [
        ROOT / 'outputs/final_task1/final_events_train.jsonl',
        ROOT / 'outputs/final_task1/final_events_test.jsonl',
    ]
    metadata_index: dict[str, dict[str, Any]] = {}
    metadata_path = ROOT / 'data/processed/event_clusters/article_metadata_index.jsonl'
    if metadata_path.exists():
        for row in read_jsonl(metadata_path):
            article_id = clean_text(row.get('article_id'))
            if article_id:
                metadata_index[article_id] = row
    rows: list[dict[str, Any]] = []
    for path_value in paths:
        path = resolve_path(path_value)
        split = 'test' if path.stem.endswith('_test') else 'train'
        for row in read_jsonl(path):
            item = dict(row)
            article_id = clean_text(item.get('article_id'))
            meta = metadata_index.get(article_id, {})
            item['dataset_split'] = item.get('dataset_split') or split or meta.get('dataset_split') or ''
            item['article_id'] = article_id
            item['title'] = clean_text(first_nonempty(item, 'title', meta.get('article_title')))
            item['source_name'] = clean_text(first_nonempty(item, 'source_name', 'source', meta.get('article_source'), meta.get('source_normalized')))
            item['publish_time'] = clean_text(first_nonempty(item, 'publish_time', meta.get('article_publish_time')))
            item['publish_dt'] = parse_datetime(item.get('publish_time'))
            item['body_for_matching'] = clean_text(first_nonempty(item, 'content', 'evidence_sentences', item.get('title'), meta.get('article_title')))
            rows.append(item)
    if not rows:
        raise ValueError('任务一输入为空，无法构建任务三数据')
    return rows


def load_cluster_members(path: str | Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    by_event: dict[str, str] = {}
    articles_by_cluster: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(resolve_path(path)):
        cluster_id = clean_text(row.get('cluster_id'))
        article_id = clean_text(row.get('article_id'))
        event_record_id = clean_text(row.get('event_record_id'))
        if cluster_id and article_id:
            articles_by_cluster[cluster_id].add(article_id)
        if cluster_id and event_record_id:
            by_event[event_record_id] = cluster_id
        if cluster_id and article_id:
            by_event.setdefault(article_id, cluster_id)
    return by_event, dict(articles_by_cluster)


def build_article_snapshots(rows: list[dict[str, Any]], member_by_event: dict[str, str], metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    metadata = metadata or {}
    for row in rows:
        snapshot = dict(row)
        article_id = clean_text(snapshot.get('article_id'))
        event_id = clean_text(snapshot.get('event_id'))
        cluster_id = ''
        for key in (event_id, f"{snapshot.get('dataset_split', '')}:{event_id}", article_id):
            if key in member_by_event:
                cluster_id = member_by_event[key]
                break
        snapshot['cluster_id'] = cluster_id
        snapshot['article_metadata'] = metadata.get(article_id, {}) if isinstance(metadata, dict) else {}
        snapshots.append(snapshot)
    return snapshots


def build_cluster_indices() -> tuple[dict[str, str], dict[str, set[str]]]:
    return load_cluster_members(ROOT / 'outputs/event_clustering/event_cluster_members.jsonl')


def build_cluster_metadata() -> dict[str, Any]:
    path = ROOT / 'data/processed/event_clusters/article_metadata_index.jsonl'
    if not path.exists():
        return {}
    index: dict[str, Any] = {}
    for row in read_jsonl(path):
        article_id = clean_text(row.get('article_id'))
        if article_id:
            index[article_id] = row
    return index


def get_publish_time(row: dict[str, Any]) -> datetime | None:
    for key in ('publish_time', 'article_publish_time', 'publish_dt', 'earliest_publish_time', 'latest_publish_time'):
        value = row.get(key)
        dt = value if isinstance(value, datetime) else parse_datetime(value)
        if dt is not None:
            return dt
    return None


def load_task1_event_rows() -> list[dict[str, Any]]:
    return load_task1_events()


def load_cluster_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        clean_text(row.get('cluster_id')): row
        for row in read_jsonl(resolve_path(path))
        if clean_text(row.get('cluster_id'))
    }


def make_article_rows(events: list[dict[str, Any]], member_by_event: dict[str, str], scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        article_id = clean_text(row.get('article_id'))
        if article_id:
            grouped[article_id].append(row)
    result: list[dict[str, Any]] = []
    for article_id, article_events in grouped.items():
        first = next((row for row in article_events if clean_text(row.get('title')) or clean_text(row.get('publish_time'))), article_events[0])
        cluster_ids = set()
        for event in article_events:
            event_id = clean_text(event.get('event_id'))
            for key in (event_id, f"{event.get('dataset_split', '')}:{event_id}", article_id):
                if key in member_by_event:
                    cluster_ids.add(member_by_event[key])
        score_rows = [scores[cid] for cid in sorted(cluster_ids) if cid in scores]
        score = max(score_rows, key=lambda item: float(item.get('credibility_score', -1) or -1)) if score_rows else {}
        article = {
            'article_id': article_id,
            'title': clean_text(first.get('title')),
            'content': clean_text(first.get('content')),
            'source_name': clean_text(first.get('source_name') or first.get('source')),
            'publish_time': clean_text(first.get('publish_time')),
            'publish_dt': first.get('publish_dt'),
            'dataset_split': clean_text(first.get('dataset_split')),
            'event_records': article_events,
            'cluster_ids': sorted(cluster_ids),
            'trust_score': score.get('credibility_score', ''),
            'verdict': score.get('verification_status', ''),
            'source_level': score.get('source_level', ''),
            'conflicting_reports': score.get('risk_reasons', ''),
            'verification_hints': score.get('supporting_reasons', ''),
        }
        result.append(article)
    return result
