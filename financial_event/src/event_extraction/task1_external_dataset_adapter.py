#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapters and audits for DocFEE/ChFinAnn external transfer data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.event_extraction.task1_official_event_contract import (
    default_external_event_mappings,
    source_role_mapping,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").replace("&nbsp;", " ").replace("<br>", "\n"))


def text_sha(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_docfee_split(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        event_type = str(row.get("event_type") or "")
        content = str(row.get("content") or "")
        events = row.get("events") or []
        out.append(
            {
                "source_dataset": "docfee",
                "source_split": path.stem,
                "document_id": str(row.get("doc_id") or ""),
                "content": content,
                "title": "",
                "source_event_type": event_type,
                "source_events": events,
                "content_sha256": text_sha(content),
            }
        )
    return out


def load_chfinann_split(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, list) or len(item) != 2:
            continue
        doc_id, doc = item
        sentences = doc.get("sentences") or []
        content = "\n".join(str(s) for s in sentences)
        events = doc.get("recguid_eventname_eventdict_list") or []
        event_types = sorted({str(ev[1]) for ev in events if isinstance(ev, list) and len(ev) >= 2})
        out.append(
            {
                "source_dataset": "chfinann",
                "source_split": path.stem,
                "document_id": str(doc_id),
                "content": content,
                "title": str(sentences[1] if len(sentences) > 1 else sentences[0] if sentences else ""),
                "source_event_type": "|".join(event_types),
                "source_events": events,
                "content_sha256": text_sha(content),
            }
        )
    return out


def audit_docfee(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*"))
    splits: dict[str, Any] = {}
    role_counter: Counter[str] = Counter()
    event_counter: Counter[str] = Counter()
    for split in ["train", "test"]:
        path = root / f"{split}.jsonl"
        rows = load_docfee_split(path)
        for row in rows:
            event_counter[row["source_event_type"]] += 1
            for ev in row["source_events"]:
                if isinstance(ev, dict):
                    for key in ev.keys():
                        role_counter[str(key)] += 1
        splits[split] = {
            "path": str(path),
            "documents": len(rows),
            "unique_document_id": len({r["document_id"] for r in rows}),
            "unique_content_sha256": len({r["content_sha256"] for r in rows}),
            "empty_content": sum(1 for r in rows if not r["content"]),
        }
    return {
        "dataset": "docfee",
        "files": [{"path": str(p), "size": p.stat().st_size, "sha256": sha256_file(p)} for p in files if p.is_file()],
        "splits": splits,
        "event_type_counts": dict(event_counter),
        "role_counts": dict(role_counter),
        "license": "See Figshare article and third_party/DocFEE repository; no project-side license rewriting performed.",
    }


def audit_chfinann(root: Path, repo: Path) -> dict[str, Any]:
    files = sorted(root.glob("Data/*.json"))
    splits: dict[str, Any] = {}
    event_counter: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    for path in files:
        rows = load_chfinann_split(path)
        for row in rows:
            for ev in row["source_events"]:
                if isinstance(ev, list) and len(ev) >= 3:
                    event_counter[str(ev[1])] += 1
                    event_dict = ev[2] if isinstance(ev[2], dict) else {}
                    for key, value in event_dict.items():
                        if value:
                            role_counter[str(key)] += 1
        splits[path.stem] = {
            "path": str(path),
            "documents": len(rows),
            "unique_document_id": len({r["document_id"] for r in rows}),
            "unique_content_sha256": len({r["content_sha256"] for r in rows}),
            "empty_content": sum(1 for r in rows if not r["content"]),
        }
    license_path = repo / "LICENSE"
    return {
        "dataset": "chfinann",
        "files": [{"path": str(p), "size": p.stat().st_size, "sha256": sha256_file(p)} for p in [*files, repo / "Data.zip", license_path] if p.exists()],
        "splits": splits,
        "event_type_counts": dict(event_counter),
        "role_counts": dict(role_counter),
        "license": license_path.read_text(encoding="utf-8", errors="ignore")[:1000] if license_path.exists() else "LICENSE not found",
    }


def mapping_audit() -> dict[str, Any]:
    mappings = default_external_event_mappings()
    role_map = source_role_mapping()
    mapped = [m for m in mappings if m.official_event_name]
    return {
        "event_type_mappings": [m.__dict__ for m in mappings],
        "mapped_event_type_count": len(mapped),
        "total_event_type_count": len(mappings),
        "mapping_coverage": len(mapped) / len(mappings) if mappings else 0.0,
        "role_mapping": role_map,
        "schema_compatible_for_official_64_event_name_transfer": len(mapped) > 0,
    }


def overlap_audit(external_rows: list[dict[str, Any]], protected_rows: list[dict[str, str]]) -> dict[str, Any]:
    external_ids = {r["document_id"] for r in external_rows if r.get("document_id")}
    external_hashes = {r["content_sha256"] for r in external_rows if r.get("content_sha256")}
    protected_ids = {r.get("article_id", "") for r in protected_rows if r.get("article_id")}
    protected_hashes = {r.get("content_sha256", "") for r in protected_rows if r.get("content_sha256")}
    return {
        "external_document_count": len(external_rows),
        "external_unique_document_id": len(external_ids),
        "external_unique_content_hash": len(external_hashes),
        "protected_article_id_count": len(protected_ids),
        "protected_content_hash_count": len(protected_hashes),
        "article_id_overlap_count": len(external_ids & protected_ids),
        "content_sha256_overlap_count": len(external_hashes & protected_hashes),
    }
