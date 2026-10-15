#!/usr/bin/env python3
"""Build the task 2 DEA SQLite FTS5 retrieval index."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.retrieval_index import (  # noqa: E402
    CACHE_DIR,
    build_task2_retrieval_index,
    is_index_valid,
    load_index_metadata,
    metadata_path,
    source_fingerprint,
    sqlite_path,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/processed/event_credibility/fulltext_article_index.parquet")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR.relative_to(ROOT)))
    parser.add_argument("--force", action="store_true")
    return parser


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    source = resolve(args.source)
    cache_dir = resolve(args.cache_dir)
    fp = source_fingerprint(source)
    valid = is_index_valid(cache_dir, fp)
    if valid and not args.force:
        meta = load_index_metadata(cache_dir)
        payload = {
            "status": "valid_existing",
            "source": str(source),
            "cache_dir": str(cache_dir),
            "sqlite_path": str(sqlite_path(cache_dir)),
            "metadata_path": str(metadata_path(cache_dir)),
            "sqlite_size_bytes": sqlite_path(cache_dir).stat().st_size,
            "metadata": meta,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    if args.force and cache_dir.exists():
        # The builder itself writes a temp DB and replaces atomically; this only
        # removes stale temp directories from previous interrupted builds.
        for path in cache_dir.glob("retrieval_index_*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    meta = build_task2_retrieval_index(source, cache_dir)
    payload = {
        "status": "built",
        "source": str(source),
        "cache_dir": str(cache_dir),
        "sqlite_path": str(sqlite_path(cache_dir)),
        "metadata_path": str(metadata_path(cache_dir)),
        "sqlite_size_bytes": sqlite_path(cache_dir).stat().st_size,
        "metadata": meta,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
