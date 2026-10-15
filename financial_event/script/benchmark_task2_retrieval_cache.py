#!/usr/bin/env python3
"""Benchmark the DEA retrieval cache against the recorded v2 scan profile."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility import fulltext  # noqa: E402
from src.event_credibility import retrieval_index as retrieval_mod  # noqa: E402
from src.event_credibility.dea import (  # noqa: E402
    FulltextArticleIndex,
    acquire_additional_evidence,
    normalize_text,
)
from src.event_credibility.retrieval_index import (  # noqa: E402
    CACHE_DIR,
    build_task2_retrieval_index,
    is_index_valid,
    load_index_metadata,
    source_fingerprint,
    sqlite_path,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rows", default="outputs/event_credibility/dea_v2_gray/sample_rows.jsonl")
    parser.add_argument("--v2-results", default="outputs/event_credibility/dea_v2_gray/dea_results.jsonl")
    parser.add_argument("--source", default="data/processed/event_credibility/fulltext_article_index.parquet")
    parser.add_argument("--members", default="outputs/event_clustering/event_cluster_members.parquet")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR.relative_to(ROOT)))
    parser.add_argument("--profile-before", default="reports/task2_dea_retrieval_profile_before.json")
    parser.add_argument("--output-json", default="reports/task2_dea_retrieval_cache_benchmark.json")
    parser.add_argument("--output-md", default="reports/task2_dea_retrieval_cache_benchmark.md")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return fulltext.read_jsonl_records(path)


def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round(ordered[lo], 6)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 6)


def evidence_key(item: dict[str, Any]) -> str:
    article_id = normalize_text(item.get("trace_id") or item.get("article_id"))
    text = normalize_text(item.get("sentence") or item.get("content") or item.get("title"))
    return f"{article_id}|{text[:160]}"


def evidence_keys(items: list[dict[str, Any]], source: str = "") -> set[str]:
    out: set[str] = set()
    for item in items:
        if source and item.get("evidence_source") != source:
            continue
        key = evidence_key(item)
        if key.strip("|"):
            out.add(key)
    return out


def overlap_rate(reference: dict[str, set[str]], current: dict[str, set[str]]) -> float:
    ref_total = sum(len(items) for items in reference.values())
    if ref_total == 0:
        return 1.0
    common = 0
    for cid, ref_items in reference.items():
        common += len(ref_items & current.get(cid, set()))
    return round(common / ref_total, 6)


def load_trigger_cases(sample_rows_path: Path, v2_results_path: Path) -> list[dict[str, Any]]:
    samples = {normalize_text(row.get("cluster_id")): row for row in read_jsonl(sample_rows_path)}
    results = [row for row in read_jsonl(v2_results_path) if row.get("dea_triggered")]
    cases: list[dict[str, Any]] = []
    for result in results:
        cid = normalize_text(result.get("cluster_id"))
        sample = samples.get(cid)
        if not sample:
            continue
        cases.append(
            {
                "cluster_id": cid,
                "row": sample,
                "reference_result": result,
                "initial_evidence": result.get("initial_evidence", []),
                "round1": result.get("round1", {}),
            }
        )
    cases.sort(key=lambda item: item["cluster_id"])
    return cases


def clear_index_singletons() -> None:
    with retrieval_mod._SINGLETON_LOCK:  # noqa: SLF001 - benchmark needs a clean process-level cache.
        retrieval_mod._SINGLETONS.clear()  # noqa: SLF001


def load_article_index(source: Path, members: Path, cache_dir: Path) -> tuple[FulltextArticleIndex, float]:
    clear_index_singletons()
    article_index = FulltextArticleIndex(source, members_path=members, cache_dir=cache_dir)
    start = time.perf_counter()
    article_index.load()
    elapsed = time.perf_counter() - start
    if not article_index.available:
        raise RuntimeError(f"article index unavailable: {article_index.error}")
    return article_index, elapsed


def run_acquisition_case(article_index: FulltextArticleIndex, case: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    acquisition = acquire_additional_evidence(
        case["row"],
        case["initial_evidence"],
        case["round1"],
        article_index,
        additional_limit=4,
    )
    elapsed = time.perf_counter() - start
    return {
        "cluster_id": case["cluster_id"],
        "latency_seconds": round(elapsed, 6),
        "new_evidence": acquisition.get("new_evidence", []),
        "queries": acquisition.get("queries", []),
        "retrieval_cache_stats": acquisition.get("retrieval_cache_stats", {}),
        "global_raw_hit_document_count": acquisition.get("global_raw_hit_document_count", 0),
        "global_hit_document_count": acquisition.get("global_hit_document_count", 0),
        "global_selected_evidence_count": acquisition.get("global_selected_evidence_count", 0),
    }


def run_acquisitions(article_index: FulltextArticleIndex, cases: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [run_acquisition_case(article_index, case) for case in cases]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda item: run_acquisition_case(article_index, item), cases))


def summarize_run(
    name: str,
    results: list[dict[str, Any]],
    initial_by_id: dict[str, list[dict[str, Any]]],
    reference_final: dict[str, set[str]],
    reference_additional: dict[str, set[str]],
    reference_global: dict[str, set[str]],
    cache_stats: dict[str, Any],
) -> dict[str, Any]:
    latencies = [float(row.get("latency_seconds") or 0.0) for row in results]
    current_final = {
        row["cluster_id"]: evidence_keys(initial_by_id.get(row["cluster_id"], []) + row.get("new_evidence", []))
        for row in results
    }
    current_additional = {row["cluster_id"]: evidence_keys(row.get("new_evidence", [])) for row in results}
    current_global = {
        row["cluster_id"]: evidence_keys(row.get("new_evidence", []), source="global_fulltext_index")
        for row in results
    }
    return {
        "name": name,
        "sample_count": len(results),
        "average_retrieval_seconds": avg(latencies),
        "p50_retrieval_seconds": percentile(latencies, 50),
        "p95_retrieval_seconds": percentile(latencies, 95),
        "max_retrieval_seconds": round(max(latencies), 6) if latencies else 0.0,
        "queries_per_second": round(len(results) / sum(latencies), 6) if sum(latencies) > 0 else 0.0,
        "global_raw_hit_document_count": int(sum(int(row.get("global_raw_hit_document_count") or 0) for row in results)),
        "global_selected_evidence_count": int(sum(int(row.get("global_selected_evidence_count") or 0) for row in results)),
        "final_evidence_overlap_rate": overlap_rate(reference_final, current_final),
        "additional_evidence_overlap_rate": overlap_rate(reference_additional, current_additional),
        "global_evidence_overlap_rate": overlap_rate(reference_global, current_global),
        "cache_stats": cache_stats,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Task 2 DEA Retrieval Cache Benchmark",
        "",
        "本报告只评估检索缓存和轻量索引，不改变 DEA v2 触发规则、阈值或证据上限。",
        "",
        f"- 索引方案: {payload['index']['scheme']}",
        f"- 索引构建时间: {payload['index']['build_seconds']} 秒",
        f"- SQLite 大小: {payload['index']['sqlite_size_bytes']} bytes",
        f"- 触发样本数: {payload['sample_count']}",
        f"- 修改前平均检索耗时: {payload['legacy_scan_profile']['average_retrieval_seconds']} 秒",
        f"- 修改前 p95 检索耗时: {payload['legacy_scan_profile']['p95_retrieval_seconds']} 秒",
        "",
        "| run | avg | p50 | p95 | max | qps | cache hit rate | global selected | overlap(final/additional/global) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["runs"]:
        stats = item.get("cache_stats", {})
        lines.append(
            f"| {item['name']} | {item['average_retrieval_seconds']} | {item['p50_retrieval_seconds']} | "
            f"{item['p95_retrieval_seconds']} | {item['max_retrieval_seconds']} | {item['queries_per_second']} | "
            f"{stats.get('hit_rate', 0.0)} | {item['global_selected_evidence_count']} | "
            f"{item['final_evidence_overlap_rate']}/{item['additional_evidence_overlap_rate']}/{item['global_evidence_overlap_rate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    source = resolve(args.source)
    members = resolve(args.members)
    cache_dir = resolve(args.cache_dir)
    profile_before = read_json(resolve(args.profile_before))
    cases = load_trigger_cases(resolve(args.sample_rows), resolve(args.v2_results))
    if not cases:
        raise RuntimeError("no triggered DEA v2 cases found")

    source_fp = source_fingerprint(source)
    if args.rebuild_index and cache_dir.exists():
        for path in cache_dir.glob("retrieval_index_*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    build_started = time.perf_counter()
    if not is_index_valid(cache_dir, source_fp):
        metadata = build_task2_retrieval_index(source, cache_dir)
    else:
        metadata = load_index_metadata(cache_dir)
    build_wall_seconds = round(time.perf_counter() - build_started, 6)

    reference_additional = {
        case["cluster_id"]: evidence_keys(case["reference_result"].get("additional_evidence", []))
        for case in cases
    }
    reference_final = {
        case["cluster_id"]: evidence_keys(case["reference_result"].get("final_evidence", []))
        for case in cases
    }
    initial_by_id = {case["cluster_id"]: case["initial_evidence"] for case in cases}
    reference_global = {
        case["cluster_id"]: evidence_keys(case["reference_result"].get("additional_evidence", []), source="global_fulltext_index")
        for case in cases
    }

    article_index, cold_load_seconds = load_article_index(source, members, cache_dir)
    first_results = run_acquisitions(article_index, cases, workers=1)
    first_summary = summarize_run(
        "new_index_first_query_workers1",
        first_results,
        initial_by_id,
        reference_final,
        reference_additional,
        reference_global,
        article_index.backend.cache_stats() if getattr(article_index, "backend", None) else {},
    )

    hot_results = run_acquisitions(article_index, cases, workers=1)
    hot_summary = summarize_run(
        "new_index_hot_query_workers1",
        hot_results,
        initial_by_id,
        reference_final,
        reference_additional,
        reference_global,
        article_index.backend.cache_stats() if getattr(article_index, "backend", None) else {},
    )

    workers4_results = run_acquisitions(article_index, cases, workers=args.workers)
    workers4_summary = summarize_run(
        f"new_index_hot_query_workers{args.workers}",
        workers4_results,
        initial_by_id,
        reference_final,
        reference_additional,
        reference_global,
        article_index.backend.cache_stats() if getattr(article_index, "backend", None) else {},
    )

    before_total = profile_before.get("summary", {}).get("total_retrieval_seconds", {})
    payload = {
        "benchmark_version": "step4_2_retrieval_cache_v1",
        "sample_count": len(cases),
        "source": str(source),
        "cache_dir": str(cache_dir),
        "legacy_scan_profile": {
            "source": str(resolve(args.profile_before)),
            "average_retrieval_seconds": before_total.get("avg", 0.0),
            "p50_retrieval_seconds": before_total.get("p50", 0.0),
            "p95_retrieval_seconds": before_total.get("p95", 0.0),
            "max_retrieval_seconds": before_total.get("max", 0.0),
            "parquet_read_seconds": profile_before.get("parquet_read_seconds", 0.0),
            "dataframe_build_seconds": profile_before.get("dataframe_build_seconds", 0.0),
            "parquet_load_count": profile_before.get("parquet_load_count", 0),
            "same_normalized_query_repeat_count": profile_before.get("same_normalized_query_repeat_count", 0),
        },
        "index": {
            "scheme": "SQLite FTS5 trigram + B-tree metadata + in-process bounded LRU",
            "sqlite_version": metadata.get("sqlite_version", ""),
            "build_seconds": metadata.get("build_seconds", build_wall_seconds),
            "build_wall_seconds": build_wall_seconds,
            "sqlite_path": str(sqlite_path(cache_dir)),
            "sqlite_size_bytes": sqlite_path(cache_dir).stat().st_size if sqlite_path(cache_dir).exists() else 0,
            "metadata": metadata,
            "cold_load_seconds": round(cold_load_seconds, 6),
            "parquet_read_count_for_queries": 0,
        },
        "runs": [first_summary, hot_summary, workers4_summary],
        "final_selected_evidence_overlap_rate": first_summary["final_evidence_overlap_rate"],
        "additional_evidence_overlap_rate": first_summary["additional_evidence_overlap_rate"],
        "global_selected_evidence_overlap_rate": first_summary["global_evidence_overlap_rate"],
        "notes": [
            "The legacy full-table scan numbers come from reports/task2_dea_retrieval_profile_before.json generated before code changes.",
            "FTS5 is used only for recall; DEA v2 relevance scoring, authority, time weighting and dedupe remain downstream.",
        ],
    }
    output_json = resolve(args.output_json)
    output_md = resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_md, payload)
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
