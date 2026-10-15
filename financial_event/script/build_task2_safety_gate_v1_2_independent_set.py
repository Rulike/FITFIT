#!/usr/bin/env python3
"""Build the blind 40+10 independent validation set for Safety Gate v1.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility import dea as dea_mod  # noqa: E402


SEED = 20260808
PRIMARY_COUNT = 40
STRESS_COUNT = 10
TOTAL_COUNT = PRIMARY_COUNT + STRESS_COUNT
FORMAL_OUTPUTS = [
    Path("outputs/event_credibility/llm_review_results.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.jsonl"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.parquet"),
    Path("outputs/event_credibility/event_credibility_scores_fulltext.xlsx"),
]
V1_2_FILES = [
    Path("src/event_credibility/dea_safety_gate_v1_2.py"),
    Path("configs/task2_dea_safety_gate_v1_2.yaml"),
]
V1_FILES = [
    Path("src/event_credibility/dea_safety_gate.py"),
    Path("configs/task2_dea_safety_gate_v1.yaml"),
    Path("scripts/test_task2_dea_safety_gate.py"),
]
V1_1_FILES = [
    Path("src/event_credibility/dea_safety_gate_v1_1.py"),
    Path("configs/task2_dea_safety_gate_v1_1.yaml"),
    Path("scripts/test_task2_dea_safety_gate_v1_1.py"),
]
ADAPTER_FILES = [Path("src/event_credibility/dea_safety_gate_adapter.py")]

DEVELOPMENT_EXCLUSION_SOURCES = [
    ("step4_4_b_v1", Path("outputs/event_credibility/dea_safety_gate_v1/safety_gate_results.jsonl")),
    ("step4_4_d_adapter_v2", Path("outputs/event_credibility/dea_safety_gate_v1_adapter_v2/safety_gate_results.jsonl")),
    ("step4_4_e_v1_1", Path("outputs/event_credibility/dea_safety_gate_v1_1_dev/safety_gate_v1_1_results.jsonl")),
    ("step4_4_g_v1_2", Path("outputs/event_credibility/dea_safety_gate_v1_2_dev/safety_gate_v1_2_results.jsonl")),
]
HISTORICAL_GRAY_SOURCES = [
    ("historical_dea_v1_60", Path("outputs/event_credibility/dea_v1_gray/sample_rows.jsonl")),
    ("historical_dea_v2_60", Path("outputs/event_credibility/dea_v2_gray/sample_rows.jsonl")),
    ("historical_cached_dea_60", Path("outputs/event_credibility/dea_v2_cached_gray/sample_rows.jsonl")),
]
STRESS_TERMS = [
    "否认",
    "不属实",
    "澄清",
    "未发生",
    "不存在",
    "终止",
    "取消",
    "更正",
    "与事实不符",
    "尚未",
    "没有计划",
    "未参与",
    "传闻不实",
]
HUMAN_FIELD_PATTERNS = (
    "human_",
    "gold_",
    "audit_",
    "preferred_result",
    "claim_is_clear",
    "evidence_relevance",
    "evidence_stance",
    "evidence_independent",
)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(resolve(path)) for path in paths if resolve(path).exists()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(obj)
    return rows


def ids_from_source(path: Path) -> set[str]:
    return {norm(row.get("cluster_id")) for row in read_jsonl(resolve(path)) if norm(row.get("cluster_id"))}


def collect_exclusions() -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    excluded_by_source: dict[str, list[str]] = {}
    union: set[str] = set()
    for label, path in DEVELOPMENT_EXCLUSION_SOURCES + HISTORICAL_GRAY_SOURCES:
        ids = ids_from_source(path)
        excluded_by_source[label] = sorted(ids)
        union.update(ids)
        source_rows.append(
            {
                "source": label,
                "path": str(resolve(path)),
                "cluster_id_count": len(ids),
                "cluster_ids": sorted(ids),
            }
        )
    return {
        "sources": source_rows,
        "excluded_cluster_id_count": len(union),
        "excluded_cluster_ids": sorted(union),
        "excluded_cluster_ids_by_source": excluded_by_source,
    }


def strip_human_fields(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(pattern in lowered for pattern in HUMAN_FIELD_PATTERNS):
                continue
            output[str(key)] = strip_human_fields(item)
        return output
    if isinstance(value, list):
        return [strip_human_fields(item) for item in value]
    return value


def evidence_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("evidence_items", [])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def evidence_is_traceable(item: dict[str, Any]) -> bool:
    evidence_id = norm(item.get("evidence_id") or item.get("id"))
    source = norm(item.get("source") or item.get("article_source") or item.get("来源"))
    text = norm(item.get("sentence") or item.get("text") or item.get("content") or item.get("证据文本"))
    trace_id = norm(item.get("trace_id") or item.get("article_id") or item.get("document_id"))
    return bool(evidence_id and source and text and trace_id)


def candidate_is_valid(row: dict[str, Any]) -> tuple[bool, str]:
    cluster_id = norm(row.get("cluster_id"))
    subject = norm(row.get("target_name") or row.get("subject"))
    event_name = norm(row.get("event_name") or row.get("schema_id"))
    claim = norm(row.get("event_claim") or row.get("claim_text"))
    if not cluster_id:
        return False, "missing_cluster_id"
    if not subject:
        return False, "missing_subject"
    if not event_name:
        return False, "missing_event_name"
    if not claim:
        return False, "missing_claim"
    items = evidence_items(row)
    if not items:
        return False, "missing_evidence_items"
    ids = [norm(item.get("evidence_id") or item.get("id")) for item in items]
    if any(not evidence_id for evidence_id in ids):
        return False, "missing_evidence_id"
    if len(set(ids)) != len(ids):
        return False, "duplicate_evidence_id"
    if not any(evidence_is_traceable(item) for item in items):
        return False, "no_traceable_evidence"
    return True, ""


def build_candidate_pool(scores_path: Path, evidence_path: Path) -> list[dict[str, Any]]:
    pool = dea_mod.build_review_pool(scores_path, evidence_path)
    valid: list[dict[str, Any]] = []
    for row in pool:
        clean = strip_human_fields(row)
        ok, _ = candidate_is_valid(clean)
        if ok:
            valid.append(clean)
    return valid


def stress_match(row: dict[str, Any]) -> list[str]:
    text_parts = [
        norm(row.get("event_claim")),
        norm(row.get("event_name")),
        norm(row.get("title_samples")),
        norm(row.get("risk_reasons")),
    ]
    for item in evidence_items(row):
        text_parts.extend(
            [
                norm(item.get("title") or item.get("article_title")),
                norm(item.get("sentence") or item.get("text") or item.get("content")),
                norm(item.get("source") or item.get("article_source")),
            ]
        )
    text = " ".join(text_parts)
    return [term for term in STRESS_TERMS if term in text]


def has_official_source(row: dict[str, Any]) -> bool:
    official_terms = ("官方", "公告", "政府", "交易所", "监管", "证监会", "巨潮", "人民政府")
    return any(
        any(term in norm(item.get(field)) for term in official_terms)
        for item in evidence_items(row)
        for field in ("source", "article_source", "source_type", "title", "article_title")
    )


def stratum_key(row: dict[str, Any]) -> str:
    event_name = norm(row.get("event_name") or row.get("schema_id")) or "unknown_event"
    items = evidence_items(row)
    evidence_bucket = "1" if len(items) == 1 else ("2_3" if len(items) <= 3 else "4_plus")
    sources = {
        norm(item.get("source") or item.get("article_source"))
        for item in items
        if norm(item.get("source") or item.get("article_source"))
    }
    source_bucket = "1" if len(sources) <= 1 else ("2_3" if len(sources) <= 3 else "4_plus")
    official = "official" if has_official_source(row) else "non_official"
    time_present = "time" if norm(row.get("event_time") or row.get("earliest_publish_time") or row.get("latest_publish_time")) else "no_time"
    return "|".join((event_name, evidence_bucket, source_bucket, official, time_present))


def stratified_select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"not enough candidates: need {count}, got {len(rows)}")
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum_key(row)].append(row)
    group_keys = sorted(groups)
    for key in group_keys:
        rng.shuffle(groups[key])
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available = [key for key in group_keys if groups[key]]
        if not available:
            break
        available.sort(key=lambda key: (-len(groups[key]), key))
        for key in available:
            if len(selected) >= count:
                break
            selected.append(groups[key].pop())
    return selected


def build_samples(
    candidates: list[dict[str, Any]],
    excluded_ids: set[str],
    primary_count: int = PRIMARY_COUNT,
    stress_count: int = STRESS_COUNT,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in candidates if norm(row.get("cluster_id")) not in excluded_ids]
    stress_pool = [row for row in eligible if stress_match(row)]
    if len(stress_pool) < stress_count:
        raise ValueError(f"not enough contradiction stress candidates: need {stress_count}, got {len(stress_pool)}")
    stress_rng = random.Random(seed + 1)
    stress_selected = stress_rng.sample(sorted(stress_pool, key=lambda row: norm(row.get("cluster_id"))), stress_count)
    stress_ids = {norm(row.get("cluster_id")) for row in stress_selected}
    primary_pool = [row for row in eligible if norm(row.get("cluster_id")) not in stress_ids]
    primary_selected = stratified_select(primary_pool, primary_count, seed)

    output: list[dict[str, Any]] = []
    for index, row in enumerate(primary_selected):
        output.append(
            {
                "input_index": index,
                "sample_group": "primary",
                "sample_bucket": "stratified_natural_distribution",
                "stratum_key": stratum_key(row),
                "stress_match_terms": [],
                **strip_human_fields(row),
            }
        )
    for offset, row in enumerate(stress_selected, primary_count):
        output.append(
            {
                "input_index": offset,
                "sample_group": "contradiction_stress",
                "sample_bucket": "keyword_recall_only",
                "stratum_key": stratum_key(row),
                "stress_match_terms": stress_match(row),
                **strip_human_fields(row),
            }
        )
    ids = [norm(row.get("cluster_id")) for row in output]
    if len(output) != primary_count + stress_count or len(set(ids)) != len(ids):
        raise AssertionError("independent sample count or uniqueness check failed")
    summary = {
        "candidate_count_after_exclusion": len(eligible),
        "stress_candidate_count": len(stress_pool),
        "primary_count": primary_count,
        "stress_count": stress_count,
        "total_count": len(output),
        "primary_strata": dict(Counter(row["stratum_key"] for row in output if row["sample_group"] == "primary")),
        "stress_terms": dict(Counter(term for row in output for term in row["stress_match_terms"])),
    }
    return output, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(
    exclusions: dict[str, Any],
    candidate_count: int,
    invalid_counts: dict[str, int],
    sample_summary: dict[str, Any],
) -> dict[str, Any]:
    frozen = hash_paths(V1_FILES + V1_1_FILES + ADAPTER_FILES + V1_2_FILES)
    formal = hash_paths(FORMAL_OUTPUTS)
    return {
        "step": "4.4-H-A",
        "sample_seed": SEED,
        "sample_targets": {"primary": PRIMARY_COUNT, "contradiction_stress": STRESS_COUNT, "total": TOTAL_COUNT},
        "candidate_pool": {
            "raw_candidate_count": candidate_count,
            "invalid_candidate_counts": invalid_counts,
            "candidate_count_after_exclusion": sample_summary["candidate_count_after_exclusion"],
            "final_selected_count": sample_summary["total_count"],
        },
        "exclusion": exclusions,
        "sample_summary": sample_summary,
        "frozen_sha256": frozen,
        "formal_output_sha256_before": formal,
        "blind_inference": {
            "human_fields_used_for_selection": False,
            "human_fields_passed_to_gate": False,
            "model_calls": 0,
            "status": "samples_built_before_inference",
        },
    }


def build_and_write(
    scores_path: Path,
    evidence_path: Path,
    output_dir: Path,
    primary_count: int = PRIMARY_COUNT,
    stress_count: int = STRESS_COUNT,
    seed: int = SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [path for path in output_dir.iterdir() if path.is_file()]
    if existing_files and not overwrite:
        raise FileExistsError(f"{output_dir} is non-empty; use --overwrite to rebuild the independent set")
    exclusions = collect_exclusions()
    excluded_ids = set(exclusions["excluded_cluster_ids"])
    pool = dea_mod.build_review_pool(resolve(scores_path), resolve(evidence_path))
    invalid_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in pool:
        row = strip_human_fields(raw)
        cid = norm(row.get("cluster_id"))
        valid, reason = candidate_is_valid(row)
        if not valid:
            invalid_counts[reason] += 1
            continue
        if cid in seen:
            invalid_counts["duplicate_candidate_cluster_id"] += 1
            continue
        seen.add(cid)
        candidates.append(row)
    samples, sample_summary = build_samples(candidates, excluded_ids, primary_count, stress_count, seed)
    write_jsonl(output_dir / "independent_samples.jsonl", samples)
    write_jsonl(output_dir / "primary_samples.jsonl", [row for row in samples if row["sample_group"] == "primary"])
    write_jsonl(output_dir / "contradiction_stress_samples.jsonl", [row for row in samples if row["sample_group"] == "contradiction_stress"])
    manifest = build_manifest(exclusions, len(pool), dict(sorted(invalid_counts.items())), sample_summary)
    manifest["output_dir"] = str(output_dir)
    manifest["input_paths"] = {"scores": str(resolve(scores_path)), "evidence": str(resolve(evidence_path))}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "run_checkpoint.json").write_text(
        json.dumps(
            {
                "stage": "sample_built",
                "total": len(samples),
                "primary": primary_count,
                "contradiction_stress": stress_count,
                "completed": 0,
                "model_calls": 0,
                "resume_command": "python scripts/run_task2_safety_gate_v1_2_independent.py --resume",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_only(
    scores_path: Path,
    evidence_path: Path,
    primary_count: int = PRIMARY_COUNT,
    stress_count: int = STRESS_COUNT,
    seed: int = SEED,
) -> dict[str, Any]:
    exclusions = collect_exclusions()
    excluded_ids = set(exclusions["excluded_cluster_ids"])
    pool = dea_mod.build_review_pool(resolve(scores_path), resolve(evidence_path))
    invalid_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in pool:
        row = strip_human_fields(raw)
        cid = norm(row.get("cluster_id"))
        valid, reason = candidate_is_valid(row)
        if not valid:
            invalid_counts[reason] += 1
            continue
        if cid in seen:
            invalid_counts["duplicate_candidate_cluster_id"] += 1
            continue
        seen.add(cid)
        candidates.append(row)
    samples, sample_summary = build_samples(candidates, excluded_ids, primary_count, stress_count, seed)
    manifest = build_manifest(exclusions, len(pool), dict(sorted(invalid_counts.items())), sample_summary)
    manifest["input_paths"] = {"scores": str(resolve(scores_path)), "evidence": str(resolve(evidence_path))}
    manifest["candidate_rows"] = candidates
    manifest["samples"] = samples
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "outputs/event_credibility/event_credibility_scores.parquet")
    parser.add_argument("--evidence", type=Path, default=ROOT / "outputs/event_credibility/fulltext_evidence.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/event_credibility/dea_safety_gate_v1_2_independent")
    parser.add_argument("--primary-count", type=int, default=PRIMARY_COUNT)
    parser.add_argument("--stress-count", type=int, default=STRESS_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_and_write(
        args.scores,
        args.evidence,
        args.output_dir,
        primary_count=args.primary_count,
        stress_count=args.stress_count,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["sample_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
