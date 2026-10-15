#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "task3_kg_v4_official1799"

OFFICIAL_1799 = (
    ROOT
    / "outputs"
    / "task1_step5_8hf_retrospective_test"
    / "official_test_predictions_frozen.jsonl"
)
CURRENT_1707 = (
    ROOT / "outputs" / "task1_severity_event_expansion_v1" / "task1_event_expansion_result.jsonl"
)
CURRENT_615_FEATURES = (
    ROOT
    / "outputs"
    / "task3_canonical_event_v2_actual_repair"
    / "resolver_v2_run1"
    / "task4_feature_table.parquet"
)
CURRENT_615_CONTRACT = (
    ROOT
    / "outputs"
    / "task3_canonical_event_v2_actual_repair"
    / "task1_615_grounding_pass_recovered.jsonl"
)
CURRENT_QUARANTINE = (
    ROOT
    / "outputs"
    / "task3_canonical_event_v2_actual_repair"
    / "task1_contract_quarantine.jsonl"
)
KG_V4_SHADOW = ROOT / "outputs" / "task3_kg_v4_shadow"
FULLTEXT_INDEX = ROOT / "data" / "processed" / "event_credibility" / "fulltext_article_index.parquet"
TARGETABLE_CORPUS = ROOT / "outputs" / "task4_targetable_raw_v1" / "targetable_corpus.xlsx"
NON_TARGETABLE_CORPUS = ROOT / "outputs" / "task4_targetable_raw_v1" / "non_targetable_corpus.xlsx"

OUTPUT_GRAPH_FILES = [
    "graph_nodes.jsonl",
    "graph_edges.jsonl",
    "task4_event_interface.jsonl",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    text = clean_text(value)
    return sha256_text(text) if text else ""


def record_hash(title: Any, content: Any, publish_time: Any, source: Any) -> str:
    key = "\n".join(
        [
            clean_text(title),
            clean_text(content),
            clean_text(publish_time),
            clean_text(source),
        ]
    )
    return sha256_text(key) if clean_text(content) else ""


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)
        f.write("\n")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stat_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "size_bytes": None}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "sha256": file_sha256(path),
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
    }


def infer_article_id(row: dict[str, Any]) -> str:
    for key in ("article_id", "article_file_id", "doc_id", "source_id"):
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def infer_title(row: dict[str, Any]) -> str:
    for key in ("article_title", "title"):
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def infer_source(row: dict[str, Any]) -> str:
    for key in ("article_source", "source", "media"):
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def infer_publish_time(row: dict[str, Any]) -> str:
    for key in ("article_publish_time", "publish_time", "pub_time"):
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def infer_event_name(row: dict[str, Any]) -> str:
    for key in ("event_name", "event_type", "event_family"):
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def infer_target_key(row: dict[str, Any]) -> str:
    for key in ("target_key", "target_code", "target_name", "secu_abbr", "trading_code"):
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def add_content_keys(row: dict[str, Any]) -> dict[str, Any]:
    article_id = infer_article_id(row)
    title = infer_title(row)
    source = infer_source(row)
    publish_time = infer_publish_time(row)
    content = clean_text(row.get("content"))
    enriched = dict(row)
    enriched["_article_id"] = article_id
    enriched["_title"] = title
    enriched["_source"] = source
    enriched["_publish_time"] = publish_time
    enriched["_content_hash"] = content_hash(content)
    enriched["_record_hash"] = record_hash(title, content, publish_time, source)
    return enriched


def build_indexes(rows: list[dict[str, Any]]) -> dict[str, defaultdict[str, list[dict[str, Any]]]]:
    indexes: dict[str, defaultdict[str, list[dict[str, Any]]]] = {
        "article_id": defaultdict(list),
        "content_hash": defaultdict(list),
        "record_hash": defaultdict(list),
    }
    for row in rows:
        if row.get("_article_id"):
            indexes["article_id"][row["_article_id"]].append(row)
        if row.get("_content_hash"):
            indexes["content_hash"][row["_content_hash"]].append(row)
        if row.get("_record_hash"):
            indexes["record_hash"][row["_record_hash"]].append(row)
    return indexes


def load_fulltext_for_article_ids(article_ids: set[str]) -> dict[str, str]:
    if not article_ids or not FULLTEXT_INDEX.exists():
        return {}
    df = pd.read_parquet(FULLTEXT_INDEX, columns=["article_id", "content"])
    df["article_id"] = df["article_id"].astype(str)
    df = df[df["article_id"].isin(article_ids)]
    return dict(zip(df["article_id"], df["content"].fillna("").astype(str)))


def read_corpus(path: Path, content_by_id: dict[str, str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_excel(path)
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict("records"):
        article_id = clean_text(rec.get("article_id"))
        content = content_by_id.get(article_id, "")
        row = dict(rec)
        row["_article_id"] = article_id
        row["_content_hash"] = content_hash(content)
        rows.append(row)
    return rows


def task4_status_reason(feature_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not feature_rows:
        return "", ""
    statuses = Counter(clean_text(r.get("task4_acceptance_status")) for r in feature_rows)
    reasons = Counter(clean_text(r.get("task4_reject_reason") or r.get("resolver_rejection_reason")) for r in feature_rows)
    status = statuses.most_common(1)[0][0] if statuses else ""
    reason = reasons.most_common(1)[0][0] if reasons else ""
    return status, reason


def collect_failure_reasons(rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for row in rows:
        for key in (
            "grounding_failure_reasons",
            "failure_reasons",
            "contract_failure_reasons",
            "quality_reasons",
        ):
            value = row.get(key)
            if isinstance(value, list):
                reasons.extend(clean_text(v) for v in value if clean_text(v))
            elif clean_text(value):
                reasons.append(clean_text(value))
    return sorted(set(reasons))


def count_duplicate_extra(counter: Counter[str]) -> int:
    return sum(count - 1 for key, count in counter.items() if key and count > 1)


def count_duplicate_groups(counter: Counter[str]) -> int:
    return sum(1 for key, count in counter.items() if key and count > 1)


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(infer_source(r) or "missing_source" for r in rows).most_common())


def compact_matches(rows: list[dict[str, Any]], max_items: int = 8) -> str:
    ids = [clean_text(r.get("_article_id") or r.get("article_id")) for r in rows]
    ids = [i for i in ids if i]
    return "|".join(ids[:max_items])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    official_rows_raw = read_jsonl(OFFICIAL_1799)
    pool1707_raw = read_jsonl(CURRENT_1707)
    contract615_raw = read_jsonl(CURRENT_615_CONTRACT)
    quarantine_raw = read_jsonl(CURRENT_QUARANTINE)

    official_rows = [add_content_keys(row) for row in official_rows_raw]
    pool1707_rows = [add_content_keys(row) for row in pool1707_raw]

    feature_df = pd.read_parquet(CURRENT_615_FEATURES)
    feature_rows = feature_df.fillna("").to_dict("records")

    article_ids_for_fulltext: set[str] = set()
    article_ids_for_fulltext.update(
        clean_text(r.get("article_id")) for r in feature_rows if clean_text(r.get("article_id"))
    )
    article_ids_for_fulltext.update(
        infer_article_id(r) for r in pool1707_raw if infer_article_id(r)
    )

    targetable_article_ids: set[str] = set()
    for corpus_path in (TARGETABLE_CORPUS, NON_TARGETABLE_CORPUS):
        if corpus_path.exists():
            try:
                df = pd.read_excel(corpus_path, usecols=["article_id"])
                targetable_article_ids.update(clean_text(v) for v in df["article_id"].tolist() if clean_text(v))
            except ValueError:
                df = pd.read_excel(corpus_path)
                if "article_id" in df.columns:
                    targetable_article_ids.update(
                        clean_text(v) for v in df["article_id"].tolist() if clean_text(v)
                    )
    article_ids_for_fulltext.update(targetable_article_ids)

    fulltext_by_id = load_fulltext_for_article_ids(article_ids_for_fulltext)

    feature_enriched: list[dict[str, Any]] = []
    for row in feature_rows:
        article_id = clean_text(row.get("article_id"))
        content = fulltext_by_id.get(article_id, "")
        enriched = dict(row)
        enriched["_article_id"] = article_id
        enriched["_content_hash"] = content_hash(content)
        provenance_hash = clean_text(row.get("provenance_document_hash"))
        if not enriched["_content_hash"] and provenance_hash:
            enriched["_content_hash"] = provenance_hash
        feature_enriched.append(enriched)

    accepted_features = [
        row for row in feature_enriched if clean_text(row.get("task4_acceptance_status")) == "pass"
    ]
    rejected_features = [
        row for row in feature_enriched if clean_text(row.get("task4_acceptance_status")) != "pass"
    ]

    targetable_rows = read_corpus(TARGETABLE_CORPUS, fulltext_by_id)
    non_targetable_rows = read_corpus(NON_TARGETABLE_CORPUS, fulltext_by_id)

    idx1707 = build_indexes(pool1707_rows)
    idx615 = build_indexes(feature_enriched)
    idx615_contract = build_indexes([add_content_keys(r) for r in contract615_raw])
    idx204 = build_indexes(accepted_features)
    idx_quarantine = build_indexes([add_content_keys(r) for r in quarantine_raw])
    idx_targetable = build_indexes(targetable_rows)
    idx_non_targetable = build_indexes(non_targetable_rows)

    official_article_counter = Counter(r["_article_id"] for r in official_rows if r["_article_id"])
    official_hash_counter = Counter(r["_content_hash"] for r in official_rows if r["_content_hash"])
    official_record_counter = Counter(r["_record_hash"] for r in official_rows if r["_record_hash"])
    official_article_seen: Counter[str] = Counter()
    official_hash_seen: Counter[str] = Counter()
    official_record_seen: Counter[str] = Counter()

    status_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    status_without_duplicate_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    missing_reason_counter: Counter[str] = Counter()

    for idx, row in enumerate(official_rows, start=1):
        article_id = row["_article_id"]
        chash = row["_content_hash"]
        rhash = row["_record_hash"]
        official_article_seen[article_id] += 1
        official_hash_seen[chash] += 1
        official_record_seen[rhash] += 1

        matches1707_id = idx1707["article_id"].get(article_id, [])
        matches1707_hash = idx1707["content_hash"].get(chash, [])
        matches1707_record = idx1707["record_hash"].get(rhash, [])
        matches615_id = idx615["article_id"].get(article_id, [])
        matches615_hash = idx615["content_hash"].get(chash, [])
        matches615_contract_id = idx615_contract["article_id"].get(article_id, [])
        matches615_contract_hash = idx615_contract["content_hash"].get(chash, [])
        matches204_id = idx204["article_id"].get(article_id, [])
        matches204_hash = idx204["content_hash"].get(chash, [])
        matches_quarantine_id = idx_quarantine["article_id"].get(article_id, [])
        matches_quarantine_hash = idx_quarantine["content_hash"].get(chash, [])
        targetable_hash = idx_targetable["content_hash"].get(chash, [])
        non_targetable_hash = idx_non_targetable["content_hash"].get(chash, [])

        in_current1707_any = bool(matches1707_id or matches1707_hash or matches1707_record)
        in_current615_any = bool(
            matches615_id or matches615_hash or matches615_contract_id or matches615_contract_hash
        )
        in_current204_any = bool(matches204_id or matches204_hash)
        in_quarantine = bool(matches_quarantine_id or matches_quarantine_hash)
        content_missing = not bool(clean_text(row.get("content")))
        target_missing = not bool(infer_target_key(row))
        event_missing = not bool(infer_event_name(row))

        duplicate_rank_article_id = official_article_seen[article_id] if article_id else 0
        duplicate_rank_content_hash = official_hash_seen[chash] if chash else 0
        duplicate_rank_normalized = official_record_seen[rhash] if rhash else 0
        is_duplicate = any(
            rank > 1
            for rank in (
                duplicate_rank_article_id,
                duplicate_rank_content_hash,
                duplicate_rank_normalized,
            )
        )

        task4_status, task4_reason = task4_status_reason(matches615_id + matches615_hash)
        grounding_reasons = collect_failure_reasons(matches1707_id + matches1707_hash + matches1707_record)
        quarantine_reasons = collect_failure_reasons(matches_quarantine_id + matches_quarantine_hash)

        if content_missing:
            status_without_duplicate = "missing_content"
            reason = "official_row_missing_content"
        elif in_current204_any:
            status_without_duplicate = "accepted_to_main_graph"
            reason = "matched_current_kg_v4_shadow_accepted_event_by_article_or_content_hash"
        elif in_quarantine:
            status_without_duplicate = "quarantine"
            reason = ";".join(quarantine_reasons) or "matched_existing_contract_quarantine"
        elif in_current615_any:
            status_without_duplicate = "review"
            reason = task4_reason or "matched_grounding_pass_pool_but_not_task4_accepted"
        elif in_current1707_any:
            status_without_duplicate = "grounding_failed"
            reason = ";".join(grounding_reasons) or "matched_1707_pool_but_not_grounding_pass_615"
        elif target_missing:
            status_without_duplicate = "no_target"
            reason = "official_prediction_has_no_target_key"
        elif event_missing:
            status_without_duplicate = "no_event"
            reason = "official_prediction_has_no_event_name"
        elif non_targetable_hash:
            status_without_duplicate = "no_target"
            reason = "matched_non_targetable_corpus_by_content_hash"
        else:
            status_without_duplicate = "review"
            reason = "not_in_current1707_pool_by_article_id_content_hash_or_normalized_key"

        processing_status = (
            "missing_content" if content_missing else "duplicate" if is_duplicate else status_without_duplicate
        )

        status_counter[processing_status] += 1
        status_without_duplicate_counter[status_without_duplicate] += 1

        not_entered = not in_current1707_any
        missing_reason = ""
        if not_entered:
            if content_missing:
                missing_reason = "missing_content"
            elif is_duplicate:
                missing_reason = "duplicate_official1799_row"
            elif target_missing:
                missing_reason = "no_target_prediction_missing"
            elif event_missing:
                missing_reason = "no_event_prediction_missing"
            elif non_targetable_hash:
                missing_reason = "matched_non_targetable_corpus_by_content_hash"
            else:
                missing_reason = "source_cohort_mismatch_not_in_current1707_by_id_hash_or_normalized_key"
            missing_reason_counter[missing_reason] += 1

        status_row = {
            "official_row_id": f"OFF1799_{idx:04d}",
            "official_row_index": idx,
            "frozen_eval_partition": True,
            "article_id": article_id,
            "title": row["_title"],
            "publish_time": row["_publish_time"],
            "source": row["_source"],
            "content_hash": chash,
            "official_content_sha256_field": clean_text(row.get("content_sha256")),
            "normalized_record_hash": rhash,
            "prediction_status": clean_text(row.get("prediction_status")),
            "target_key": infer_target_key(row),
            "target_name": clean_text(row.get("target_name")),
            "target_type": clean_text(row.get("target_type")),
            "event_name": infer_event_name(row),
            "event_family": clean_text(row.get("event_group") or row.get("event_family")),
            "content_missing": content_missing,
            "target_missing": target_missing,
            "event_missing": event_missing,
            "article_id_occurrence_count": official_article_counter.get(article_id, 0),
            "content_hash_occurrence_count": official_hash_counter.get(chash, 0),
            "normalized_record_occurrence_count": official_record_counter.get(rhash, 0),
            "duplicate_rank_article_id": duplicate_rank_article_id,
            "duplicate_rank_content_hash": duplicate_rank_content_hash,
            "duplicate_rank_normalized": duplicate_rank_normalized,
            "is_duplicate_row": is_duplicate,
            "in_current1707_by_article_id": bool(matches1707_id),
            "in_current1707_by_content_hash": bool(matches1707_hash),
            "in_current1707_by_normalized_key": bool(matches1707_record),
            "in_current1707_any": in_current1707_any,
            "matched_1707_article_ids": compact_matches(
                matches1707_id + matches1707_hash + matches1707_record
            ),
            "in_current615_grounding_pass": in_current615_any,
            "in_current204_shadow_by_article_or_hash": in_current204_any,
            "matched_204_article_ids": compact_matches(matches204_id + matches204_hash),
            "matched_targetable_corpus_by_hash": bool(targetable_hash),
            "matched_non_targetable_corpus_by_hash": bool(non_targetable_hash),
            "contract_status": "pass" if in_current615_any else "not_available",
            "task4_acceptance_status": task4_status,
            "status_without_duplicate": status_without_duplicate,
            "processing_status": processing_status,
            "status_reason": reason,
            "not_entered_1707_pool": not_entered,
            "missing_reason": missing_reason,
        }
        status_rows.append(status_row)
        if not_entered:
            missing_rows.append(status_row)

    status_df = pd.DataFrame(status_rows)
    status_df.to_parquet(OUT_DIR / "official1799_processing_status.parquet", index=False)
    pd.DataFrame(missing_rows).to_csv(OUT_DIR / "official1799_missing92.csv", index=False)

    for filename in OUTPUT_GRAPH_FILES:
        src = KG_V4_SHADOW / filename
        dst = OUT_DIR / filename
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)

    graph_stats_path = KG_V4_SHADOW / "graph_statistics.json"
    graph_stats = json.loads(graph_stats_path.read_text(encoding="utf-8")) if graph_stats_path.exists() else {}

    edge_endpoint_missing = None
    graph_nodes_path = OUT_DIR / "graph_nodes.jsonl"
    graph_edges_path = OUT_DIR / "graph_edges.jsonl"
    if graph_nodes_path.exists() and graph_edges_path.exists():
        nodes = read_jsonl(graph_nodes_path)
        node_ids = {clean_text(n.get("stable_id") or n.get("id") or n.get("node_id")) for n in nodes}
        edges = read_jsonl(graph_edges_path)
        edge_endpoint_missing = 0
        for edge in edges:
            src = clean_text(edge.get("source_id") or edge.get("source") or edge.get("src"))
            dst = clean_text(edge.get("target_id") or edge.get("target") or edge.get("dst"))
            if src not in node_ids or dst not in node_ids:
                edge_endpoint_missing += 1

    official_unique_article_ids = len({r["_article_id"] for r in official_rows if r["_article_id"]})
    official_unique_content_hashes = len({r["_content_hash"] for r in official_rows if r["_content_hash"]})
    official_unique_record_hashes = len({r["_record_hash"] for r in official_rows if r["_record_hash"]})
    current1707_unique_article_ids = len({r["_article_id"] for r in pool1707_rows if r["_article_id"]})
    current1707_unique_content_hashes = len({r["_content_hash"] for r in pool1707_rows if r["_content_hash"]})

    entered_by_id_rows = sum(1 for r in status_rows if r["in_current1707_by_article_id"])
    entered_by_hash_rows = sum(1 for r in status_rows if r["in_current1707_by_content_hash"])
    entered_by_record_rows = sum(1 for r in status_rows if r["in_current1707_by_normalized_key"])
    entered_any_rows = sum(1 for r in status_rows if r["in_current1707_any"])

    funnel = {
        "run_status": "official1799_alignment_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "formal_delivery_articles": "official_1799",
            "full_310k_production_inference_started": False,
            "current_204_shadow_graph_declared_complete_graph": False,
            "note": "Current 204 trusted events are reused as the accepted shadow subset; all other official rows are statused, not force-ingested.",
        },
        "official1799": {
            "rows": len(official_rows),
            "unique_article_ids": official_unique_article_ids,
            "unique_content_hashes": official_unique_content_hashes,
            "unique_normalized_record_hashes": official_unique_record_hashes,
            "duplicate_article_id_groups": count_duplicate_groups(official_article_counter),
            "duplicate_article_id_extra_rows": count_duplicate_extra(official_article_counter),
            "duplicate_content_hash_groups": count_duplicate_groups(official_hash_counter),
            "duplicate_content_hash_extra_rows": count_duplicate_extra(official_hash_counter),
            "missing_content_rows": sum(1 for r in status_rows if r["content_missing"]),
            "missing_target_rows": sum(1 for r in status_rows if r["target_missing"]),
            "missing_event_rows": sum(1 for r in status_rows if r["event_missing"]),
            "source_counts": source_counts(official_rows_raw),
        },
        "current1707_pool": {
            "rows": len(pool1707_rows),
            "unique_article_ids": current1707_unique_article_ids,
            "unique_content_hashes": current1707_unique_content_hashes,
        },
        "alignment_1799_to_1707": {
            "entered_by_article_id_rows": entered_by_id_rows,
            "entered_by_content_hash_rows": entered_by_hash_rows,
            "entered_by_normalized_record_rows": entered_by_record_rows,
            "entered_any_rows": entered_any_rows,
            "not_entered_actual_rows": len(missing_rows),
            "expected_missing_92_reproduced": len(missing_rows) == 92,
            "note": "Actual local artifacts do not show 1707 as official1799-minus-92. Article IDs have zero overlap; only content hashes provide limited overlap.",
            "missing_reason_counts": dict(missing_reason_counter.most_common()),
        },
        "processing_status_counts": dict(status_counter.most_common()),
        "status_without_duplicate_counts": dict(status_without_duplicate_counter.most_common()),
        "trusted_shadow_reuse": {
            "task4_accepted_rows": len(accepted_features),
            "task4_feature_table_rows": len(feature_rows),
            "official_overlap_accepted_by_article_id_rows": sum(
                1 for r in status_rows if r["in_current204_shadow_by_article_or_hash"] and r["in_current1707_by_article_id"]
            ),
            "official_overlap_accepted_by_article_or_hash_rows": sum(
                1 for r in status_rows if r["in_current204_shadow_by_article_or_hash"]
            ),
            "source_graph_statistics": graph_stats,
            "edge_endpoint_missing_after_copy": edge_endpoint_missing,
        },
        "constraints": {
            "official_test_labels_read_this_run": False,
            "task34_benchmark_read": False,
            "severity_labels_read": False,
            "resolver_retrained_or_tuned": False,
            "task1_model_rerun": False,
            "old_task3_core_modified": False,
            "old_graph_overwritten": False,
        },
    }
    write_json(OUT_DIR / "official1799_selection_funnel.json", funnel)

    input_paths = {
        "official1799_predictions": stat_path(OFFICIAL_1799),
        "current1707_pool": stat_path(CURRENT_1707),
        "current615_feature_table": stat_path(CURRENT_615_FEATURES),
        "current615_contract": stat_path(CURRENT_615_CONTRACT),
        "current_quarantine": stat_path(CURRENT_QUARANTINE),
        "kg_v4_shadow_graph_nodes": stat_path(KG_V4_SHADOW / "graph_nodes.jsonl"),
        "kg_v4_shadow_graph_edges": stat_path(KG_V4_SHADOW / "graph_edges.jsonl"),
        "kg_v4_shadow_task4_interface": stat_path(KG_V4_SHADOW / "task4_event_interface.jsonl"),
        "fulltext_article_index": stat_path(FULLTEXT_INDEX),
        "targetable_corpus": stat_path(TARGETABLE_CORPUS),
        "non_targetable_corpus": stat_path(NON_TARGETABLE_CORPUS),
    }
    output_paths = {
        "official1799_processing_status": stat_path(OUT_DIR / "official1799_processing_status.parquet"),
        "official1799_selection_funnel": stat_path(OUT_DIR / "official1799_selection_funnel.json"),
        "official1799_missing92": stat_path(OUT_DIR / "official1799_missing92.csv"),
        "graph_nodes": stat_path(OUT_DIR / "graph_nodes.jsonl"),
        "graph_edges": stat_path(OUT_DIR / "graph_edges.jsonl"),
        "task4_event_interface": stat_path(OUT_DIR / "task4_event_interface.jsonl"),
    }
    manifest = {
        "run_status": "official1799_delivery_status_ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repro_command": "python scripts/build_task3_kg_v4_official1799.py",
        "input_paths": input_paths,
        "output_paths": output_paths,
        "row_counts": {
            "official1799_processing_status": len(status_rows),
            "official1799_missing_rows_actual": len(missing_rows),
            "task4_event_interface_rows": sum(1 for _ in (OUT_DIR / "task4_event_interface.jsonl").open("r", encoding="utf-8")),
            "graph_nodes_rows": sum(1 for _ in (OUT_DIR / "graph_nodes.jsonl").open("r", encoding="utf-8")),
            "graph_edges_rows": sum(1 for _ in (OUT_DIR / "graph_edges.jsonl").open("r", encoding="utf-8")),
        },
        "alignment_summary": funnel["alignment_1799_to_1707"],
        "processing_status_counts": funnel["processing_status_counts"],
        "status_without_duplicate_counts": funnel["status_without_duplicate_counts"],
        "graph_integrity_summary": funnel["trusted_shadow_reuse"],
        "constraints": funnel["constraints"],
    }
    write_json(OUT_DIR / "submission_manifest.json", manifest)

    print(json.dumps({
        "output_dir": str(OUT_DIR),
        "official_rows": len(status_rows),
        "entered_any_rows": entered_any_rows,
        "missing_actual_rows": len(missing_rows),
        "processing_status_counts": dict(status_counter.most_common()),
        "status_without_duplicate_counts": dict(status_without_duplicate_counter.most_common()),
        "task4_interface_rows": manifest["row_counts"]["task4_event_interface_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
