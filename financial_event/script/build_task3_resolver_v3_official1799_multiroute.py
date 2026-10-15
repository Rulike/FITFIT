#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts import build_task3_resolver_v3_candidate as multiroute  # noqa: E402
from scripts import train_task3_resolver_v3_pair_model as v3model  # noqa: E402
from task3_canonical_event_v2_resolver_v2_actual import (  # noqa: E402
    candidate_channels as v2_candidate_channels,
    feature_dict as v2_feature_dict,
)


LEGACY_OFFICIAL_ENTRYPOINT = ROOT / "scripts/build_task3_kg_v4_official1799_inference_v1.py"
LEGACY_OFFICIAL_DIR = ROOT / "outputs/task3_kg_v4_official1799_inference_v1"
DEFAULT_OUT_DIR = ROOT / "outputs/task3_resolver_v3_official1799_multiroute"
RAW_TEST_XLSX = ROOT / "data/raw/news_with_tags_test.xlsx"
V3_PACKAGE_DIR = ROOT / "outputs/task3_resolver_v3_candidate_recall"
V3_SOURCE_DIR = ROOT / "outputs/task3_resolver_v3_candidate"
V3_MODEL_PATH = V3_PACKAGE_DIR / "selected_model/resolver_v3_pair_model.joblib"
V3_SELECTION_PATH = V3_PACKAGE_DIR / "selected_model/resolver_v3_pair_model_selection.json"
if not V3_SELECTION_PATH.exists():
    V3_SELECTION_PATH = V3_PACKAGE_DIR / "resolver_v3_model/resolver_v3_pair_model_selection.json"
if not V3_MODEL_PATH.exists():
    V3_MODEL_PATH = V3_PACKAGE_DIR / "resolver_v3_model/resolver_v3_pair_model.joblib"

DEV_EVENT_SPLIT = V3_PACKAGE_DIR / "event_split_mapping.parquet"
DEV_PAIR_SPLIT = V3_PACKAGE_DIR / "repair_pair_split_mapping.parquet"
DEV_MULTIROUTE_CANDIDATES = V3_PACKAGE_DIR / "r_channel_candidate_pairs_dev.parquet"
DEV_LEGACY_CANDIDATES = ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_candidate_pairs.parquet"

ROUTE_BITS = {"R1": 1, "R2": 2, "R3": 4, "R4": 8}
PRIMARY_LABELS = ["same_event", "different_event"]
ALL_LABELS = ["same_event", "different_event", "uncertain"]
RECOVERY_CLUSTER_SAME_THR = 0.995
RECOVERY_MAX_COMPONENT_SIZE = 3
RUNTIME_PYTHON = "/root/miniconda3/envs/task4-impact-v3/bin/python"
DEFAULT_R3_ENCODE_PYTHON = "/root/miniconda3/envs/dianjin-cu128/bin/python"
DEFAULT_R3_ANN_PYTHON = "/root/miniconda3/envs/marag_retriever/bin/python"
RAW_ALLOWED_COLUMNS = {
    "article_id": ["article_id", "article_file_id"],
    "title": ["title", "article_title"],
    "content": ["content"],
    "source": ["source", "article_source"],
    "publish_time": ["publish_time", "article_publish_time"],
}
FORBIDDEN_TEST_LABEL_COLUMNS = {
    "trading_code",
    "secu_abbr",
    "industry_code",
    "industry_name",
    "event_name",
    "event_emotion",
    "event_impact_analysis",
    "duplication_id",
    "target_type",
    "target_code",
    "target_name",
}


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    raw = str(value).strip()
    return "" if raw.lower() in {"none", "nan", "null", "nat"} else raw


def stable_hash(*parts: Any, n: int = 24) -> str:
    payload = "||".join(text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


def pair_key(a: Any, b: Any) -> tuple[str, str]:
    aa, bb = text(a), text(b)
    return tuple(sorted((aa, bb)))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parquet_ready_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    safe_rows: list[dict[str, Any]] = []
    for row in rows:
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple, set)):
                safe_row[key] = json.dumps(jsonable(value), ensure_ascii=False)
            else:
                safe_row[key] = value
        safe_rows.append(safe_row)
    return pd.DataFrame(safe_rows)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_stat(path: Path, hash_file: bool = True) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        item["sha256"] = None
        return item
    item["type"] = "dir" if path.is_dir() else "file"
    if path.is_file():
        st = path.stat()
        item.update(
            {
                "size_bytes": st.st_size,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(path) if hash_file else None,
            }
        )
    else:
        item["sha256"] = None
        item["note"] = "directory contents not hashed here"
    return item


def norm_compact(value: Any) -> str:
    return "".join(text(value).split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    compact = norm_compact(value)
    return sha256_text(compact) if compact else ""


def excel_headers(path: Path) -> dict[str, list[str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    out: dict[str, list[str]] = {}
    for sheet in workbook.sheetnames:
        ws = workbook[sheet]
        out[sheet] = [text(cell.value) for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    workbook.close()
    return out


def first_present(headers: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in headers:
            return candidate
    return ""


def read_label_blind_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    headers = excel_headers(RAW_TEST_XLSX)
    rows: list[dict[str, Any]] = []
    sheet_manifest: dict[str, Any] = {}
    global_idx = 0
    read_columns_by_sheet: dict[str, list[str]] = {}
    label_columns_present: dict[str, list[str]] = {}
    for sheet, sheet_headers in headers.items():
        source_cols = {key: first_present(sheet_headers, aliases) for key, aliases in RAW_ALLOWED_COLUMNS.items()}
        usecols = [col for col in source_cols.values() if col]
        missing_required = [key for key, col in source_cols.items() if not col and key != "article_id"]
        if missing_required:
            raise RuntimeError(f"missing allowed input columns in sheet={sheet}: {missing_required}")
        df = pd.read_excel(RAW_TEST_XLSX, sheet_name=sheet, usecols=usecols, dtype=object).fillna("")
        read_columns_by_sheet[sheet] = usecols
        label_columns_present[sheet] = [col for col in sheet_headers if col in FORBIDDEN_TEST_LABEL_COLUMNS]
        sheet_manifest[sheet] = {"rows": len(df), "source_columns": source_cols}
        for local_idx, record in enumerate(df.to_dict("records"), start=2):
            global_idx += 1
            official_row_id = f"OFF1799_{global_idx:04d}"
            article_id = text(record.get(source_cols.get("article_id", ""))) if source_cols.get("article_id") else ""
            if not article_id:
                article_id = official_row_id
            title = text(record.get(source_cols["title"]))
            content = text(record.get(source_cols["content"]))
            source = text(record.get(source_cols["source"]))
            publish_time = text(record.get(source_cols["publish_time"]))
            rows.append(
                {
                    "official_row_id": official_row_id,
                    "sheet_name": sheet,
                    "sheet_row_index": local_idx,
                    "article_id": article_id,
                    "article_file_id": article_id,
                    "title": title,
                    "article_title": title,
                    "content": content,
                    "source": source,
                    "article_source": source,
                    "publish_time": publish_time,
                    "article_publish_time": publish_time,
                    "content_hash": content_hash(content),
                    "title_hash": sha256_text(norm_compact(title)) if title else "",
                    "input_status": "missing_content" if not content else "ready",
                }
            )
    manifest = {
        "source_file": path_stat(RAW_TEST_XLSX),
        "allowed_logical_fields": list(RAW_ALLOWED_COLUMNS.keys()),
        "allowed_source_columns_by_sheet": read_columns_by_sheet,
        "sheet_manifest": sheet_manifest,
        "all_header_columns_inspected": headers,
        "forbidden_label_columns_present_but_not_read": label_columns_present,
        "forbidden_label_columns_read": [],
        "official_test_labels_read": False,
        "read_method": "pandas.read_excel(usecols=allowed_source_columns_only)",
        "row_count": len(rows),
        "unique_content_hashes_nonempty": len({r["content_hash"] for r in rows if r["content_hash"]}),
        "missing_content_rows": sum(1 for r in rows if not r["content"]),
    }
    return rows, manifest


def build_representatives(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    representative_by_hash: dict[str, str] = {}
    duplicate_of: dict[str, str] = {}
    duplicates_by_rep: dict[str, list[str]] = defaultdict(list)
    representatives: list[dict[str, Any]] = []
    row_by_id = {row["official_row_id"]: row for row in rows}
    for row in rows:
        key = row["content_hash"] or f"missing:{row['official_row_id']}"
        if key not in representative_by_hash:
            representative_by_hash[key] = row["official_row_id"]
            duplicate_of[row["official_row_id"]] = ""
            representatives.append(row)
        else:
            rep = representative_by_hash[key]
            duplicate_of[row["official_row_id"]] = rep
            duplicates_by_rep[rep].append(row["official_row_id"])
    for row in rows:
        rep = duplicate_of.get(row["official_row_id"], "")
        row["is_representative"] = not bool(rep)
        row["duplicate_of"] = rep
        row["representative_official_row_id"] = rep or row["official_row_id"]
    for rep, duplicate_ids in duplicates_by_rep.items():
        row_by_id[rep]["duplicate_row_ids"] = duplicate_ids
    return representatives, duplicate_of, duplicates_by_rep


def time_bucket(value: str) -> str:
    value = text(value)
    if not value or value == "未明确":
        return ""
    match = __import__("re").search(r"(20\d{2})[-年/](\d{1,2})", value)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}"
    match = __import__("re").search(r"(\d{1,2})月", value)
    if match:
        return f"month-{int(match.group(1)):02d}"
    return value[:12]


def token_list(value: str) -> list[str]:
    import re

    return [tok for tok in re.findall(r"[0-9A-Za-z\u4e00-\u9fff]{2,}", norm_compact(value)) if tok]


def add_group_pairs(indices: list[int], pair_set: set[tuple[int, int]], max_group: int = 260) -> None:
    if len(indices) < 2:
        return
    indices = sorted(set(indices))
    if len(indices) > max_group:
        indices = indices[:max_group]
    for offset, left in enumerate(indices):
        for right in indices[offset + 1 :]:
            pair_set.add((left, right))


def generate_legacy_resolver_candidates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, event in enumerate(events):
        target = event["target"]
        ev = event["event"]
        src = event["source"]
        evidence = event["evidence"]
        target_key = text(target.get("target_key"))
        family = text(ev.get("event_family"))
        event_name = text(ev.get("event_name"))
        event_time = text(ev.get("event_time"))
        source_name = text(src.get("source_name"))
        object_text = text(ev.get("object_text"))
        trigger_text = text(ev.get("trigger_text"))
        evidence_text = text(evidence.get("evidence_sentence"))
        if target_key and family:
            groups[("target_family", target_key, family)].append(idx)
        if target_key and event_name:
            groups[("target_event", target_key, event_name)].append(idx)
        if target_key and time_bucket(event_time):
            groups[("target_time", target_key, time_bucket(event_time))].append(idx)
        if target_key and source_name:
            groups[("target_source", target_key, source_name)].append(idx)
        for tok in token_list(object_text)[:5]:
            groups[("target_object", target_key, tok)].append(idx)
        for tok in token_list(trigger_text)[:3]:
            groups[("target_trigger", target_key, family, tok)].append(idx)
        for tok in token_list(evidence_text)[:4]:
            groups[("target_evidence_token", target_key, family, tok)].append(idx)
    pair_set: set[tuple[int, int]] = set()
    for key, idxs in groups.items():
        add_group_pairs(idxs, pair_set, max_group=360 if key[0] in {"target_family", "target_event"} else 220)
    rows: list[dict[str, Any]] = []
    for left, right in sorted(pair_set):
        a = events[left]
        b = events[right]
        feats = v2_feature_dict(a, b)
        channels = v2_candidate_channels(a, b, feats)
        if not channels:
            continue
        rows.append(
            {
                "pair_id": "OFFPAIR_" + stable_hash(a["source_row"]["sample_id"], b["source_row"]["sample_id"]),
                "sample_id_a": a["source_row"]["sample_id"],
                "sample_id_b": b["source_row"]["sample_id"],
                "article_id_a": a["article_id"],
                "article_id_b": b["article_id"],
                "official_row_id_a": a["official_row_id"],
                "official_row_id_b": b["official_row_id"],
                "target_key_a": a["target"]["target_key"],
                "target_key_b": b["target"]["target_key"],
                "event_family_a": a["event"]["event_family"],
                "event_family_b": b["event"]["event_family"],
                "event_name_a": a["event"]["event_name"],
                "event_name_b": b["event"]["event_name"],
                "channels": "|".join(channels),
                "feature_dict": feats,
            }
        )
    return rows


def first_jsonl_schema(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return list(json.loads(line).keys())
    return []


def parquet_schema(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(pd.read_parquet(path).columns)


def load_v3_selection() -> dict[str, Any]:
    selection = read_json(V3_SELECTION_PATH)
    if "thresholds" not in selection:
        raise RuntimeError(f"missing thresholds in {V3_SELECTION_PATH}")
    return selection


def protected_sha(include_outputs: bool, out_dir: Path) -> dict[str, Any]:
    protected = {
        "status": "protected_sha256_snapshot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official1799_filled_gold_read_count": 0,
        "final_holdout_read_count": 0,
        "files": {
            "official_raw_input": path_stat(RAW_TEST_XLSX),
            "legacy_official_entrypoint": path_stat(LEGACY_OFFICIAL_ENTRYPOINT),
            "legacy_official_structured_events": path_stat(LEGACY_OFFICIAL_DIR / "official1799_structured_events.jsonl"),
            "legacy_official_pair_decisions": path_stat(LEGACY_OFFICIAL_DIR / "resolver_v2_pair_decisions.jsonl"),
            "legacy_official_canonical_events": path_stat(LEGACY_OFFICIAL_DIR / "canonical_events.jsonl"),
            "multiroute_generator": path_stat(ROOT / "scripts/build_task3_resolver_v3_candidate.py"),
            "selected_v3_model": path_stat(V3_MODEL_PATH),
            "selected_v3_selection": path_stat(V3_SELECTION_PATH),
            "dev_multiroute_candidate_manifest": path_stat(V3_PACKAGE_DIR / "candidate_recall_diagnosis.json"),
            "dev_multiroute_candidate_pairs": path_stat(DEV_MULTIROUTE_CANDIDATES),
            "integration_entrypoint": path_stat(Path(__file__).resolve()),
        },
        "protected_directories_not_read": {
            "official1799_filled_gold_dir": {
                "path": str(ROOT / "outputs/task34_official1799_final_evaluation_v1"),
                "exists": (ROOT / "outputs/task34_official1799_final_evaluation_v1").exists(),
                "content_hash": None,
                "read_count": 0,
            }
        },
    }
    if include_outputs:
        names = [
            "integration_inventory.json",
            "official1799_multiroute_predictions.xlsx",
            "official1799_multiroute_predictions.parquet",
            "official1799_candidate_pairs.parquet",
            "resolver_v3_pair_decisions.jsonl",
            "canonical_events.jsonl",
            "graph_nodes.jsonl",
            "graph_edges.jsonl",
            "task4_event_interface.jsonl",
            "route_distribution.json",
            "cluster_statistics.json",
            "legacy_vs_multiroute_unlabeled_comparison.json",
            "inference_manifest.json",
            "dev_regression_legacy_vs_multiroute.json",
        ]
        protected["outputs"] = {name: path_stat(out_dir / name) for name in names}
    return protected


def build_integration_inventory(out_dir: Path) -> dict[str, Any]:
    selection = load_v3_selection()
    ablation = read_json(V3_PACKAGE_DIR / "candidate_recall_ablation.json") if (V3_PACKAGE_DIR / "candidate_recall_ablation.json").exists() else {}
    diagnosis = read_json(V3_PACKAGE_DIR / "candidate_recall_diagnosis.json") if (V3_PACKAGE_DIR / "candidate_recall_diagnosis.json").exists() else {}
    return {
        "status": "integration_inventory_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "multiroute_generator_path": rel(ROOT / "scripts/build_task3_resolver_v3_candidate.py"),
        "multiroute_config_path": [
            rel(V3_PACKAGE_DIR / "candidate_recall_diagnosis.json"),
            rel(V3_PACKAGE_DIR / "candidate_recall_ablation.json"),
            rel(V3_PACKAGE_DIR / "production_like_efficiency_pool/production_like_efficiency_report.json"),
        ],
        "multiroute_routes": {
            "R1": "target/strong deterministic keys with fact/document/sentence/project support",
            "R2": "target + event_family + lexical/fact/argument/time support with no hard conflict",
            "R3": "Qwen3-Embedding-0.6B + FAISS dynamic Top-K dense retriever, semantic_score_floor=0.58",
            "R4": "constrained legacy source/time/modality structured support",
        },
        "non_official_dev_multiroute_recall": diagnosis.get("r_channel_independent_metrics", {}).get("R1_R2_R3_R4_union"),
        "non_official_dev_ablation_selected": next(
            (row for row in ablation.get("experiments", []) if row.get("name") == "selected_multi_channel_union_enhanced_resolver_v3_lightgbm"),
            {},
        ),
        "selected_v3_model_path": rel(V3_MODEL_PATH),
        "selected_v3_model_sha256": sha256_file(V3_MODEL_PATH),
        "selected_v3_selection_path": rel(V3_SELECTION_PATH),
        "selected_v3_thresholds": selection.get("thresholds"),
        "selected_v3_model_name": selection.get("selected_model_name"),
        "official_inference_entrypoint": rel(LEGACY_OFFICIAL_ENTRYPOINT),
        "integrated_configurable_entrypoint": rel(Path(__file__).resolve()),
        "official_raw_input": rel(RAW_TEST_XLSX),
        "official_structured_input_reused_for_resolver": rel(LEGACY_OFFICIAL_DIR / "official1799_structured_events.jsonl"),
        "official_prediction_schema": {
            "legacy_pair_decision_jsonl_fields": first_jsonl_schema(LEGACY_OFFICIAL_DIR / "resolver_v2_pair_decisions.jsonl"),
            "legacy_canonical_event_jsonl_fields": first_jsonl_schema(LEGACY_OFFICIAL_DIR / "canonical_events.jsonl"),
            "legacy_processing_status_parquet_fields": parquet_schema(LEGACY_OFFICIAL_DIR / "official1799_processing_status.parquet"),
            "multiroute_pair_decision_additional_fields": [
                "candidate_mode",
                "route_names",
                "route_count",
                "route_mask",
                "route_strength",
                "route_reasons",
                "cluster_edge_policy",
            ],
        },
        "official_evaluation_scripts_found": [
            rel(ROOT / "scripts/build_task34_official1799_final_evaluation_v1.py"),
            rel(ROOT / "scripts/finalize_task34_official1799_final_evaluation_v1.py"),
            rel(ROOT / "scripts/build_task34_official1799_full_benchmark_v1.py"),
            rel(ROOT / "scripts/build_task34_official1799_downstream_benchmark_v1.py"),
        ],
        "current_formal_pipeline_uses_legacy_candidate": True,
        "current_formal_candidate_generator": "scripts/build_task3_kg_v4_official1799_inference_v1.py::generate_resolver_candidates imports resolver_v2 candidate_channels/feature_dict/threshold_predict",
        "current_legacy_candidate_count_official1799": count_jsonl(LEGACY_OFFICIAL_DIR / "resolver_v2_pair_decisions.jsonl"),
        "constraints": {
            "official1799_filled_gold_read_count": 0,
            "final_holdout_read_count": 0,
            "resolver_retrained": False,
            "thresholds_tuned_on_official1799": False,
            "frozen_legacy_outputs_overwritten": False,
        },
        "output_dir": str(out_dir),
    }


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def r3_encode_worker(args: argparse.Namespace) -> None:
    payload = read_json(Path(args.encode_input))
    texts = [text(item) for item in payload["texts"]]
    cache_path = Path(payload["cache_path"])
    report = multiroute.encode_qwen_current_python(texts, cache_path)
    write_json(Path(args.encode_output), report)


def run_r3_ann_integrated(
    events: dict[str, dict[str, Any]],
    out_dir: Path,
    ann_worker_python: str | None,
    encode_worker_python: str | None,
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    ann_input = out_dir / "r3_ann_dev_input.json"
    ann_output = out_dir / "r3_ann_dev_output.json"
    encode_input = out_dir / "r3_encode_input.json"
    encode_output = out_dir / "r3_encode_output.json"
    rows = []
    for sid, event in sorted(events.items()):
        rows.append(
            {
                "event_uid": sid,
                "text": multiroute.build_event_template(event),
                "target_key": multiroute.target_meta(event)["target_key"],
                "event_family": multiroute.event_meta(event)["event_family"],
                "publish_time": text(event.get("publish_time")),
                "event_time": multiroute.event_meta(event)["event_time"],
            }
        )
    cache_path = out_dir / "r3_qwen_dev_embeddings.npy"
    encode_report: dict[str, Any]
    encode_py = encode_worker_python or (DEFAULT_R3_ENCODE_PYTHON if Path(DEFAULT_R3_ENCODE_PYTHON).exists() else None)
    if encode_py:
        write_json(encode_input, {"texts": [row["text"] for row in rows], "cache_path": str(cache_path)})
        completed = subprocess.run(
            [
                encode_py,
                str(Path(__file__).resolve()),
                "--encode-worker",
                "--encode-input",
                str(encode_input),
                "--encode-output",
                str(encode_output),
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3600,
        )
        if completed.returncode == 0 and encode_output.exists():
            encode_report = read_json(encode_output)
            encode_report["encode_worker_python"] = encode_py
            encode_report["encode_worker_stdout_tail"] = completed.stdout[-2000:]
            encode_report["encode_worker_stderr_tail"] = completed.stderr[-4000:]
        else:
            encode_report = {
                "status": "blocked_encode_worker_failed",
                "encode_worker_python": encode_py,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-4000:],
            }
            write_json(encode_output, encode_report)
    else:
        encode_report = multiroute.encode_qwen_current_python([row["text"] for row in rows], cache_path)

    payload = {
        "events": rows,
        "model_path": str(multiroute.QWEN_EMBED),
        "cache_path": str(cache_path),
        "batch_size": 4,
        "max_length": 512,
        "precomputed_embeddings": encode_report.get("status") == "ok",
        "encode_report": encode_report,
    }
    write_json(ann_input, payload)
    ann_py = ann_worker_python or (DEFAULT_R3_ANN_PYTHON if Path(DEFAULT_R3_ANN_PYTHON).exists() else multiroute.worker_python(None))
    if not ann_py:
        report = {"status": "blocked_no_python_with_faiss", "event_count": len(rows), "pairs": [], "encode_report": encode_report}
        write_json(ann_output, report)
        return set(), report
    completed = subprocess.run(
        [
            ann_py,
            str(ROOT / "scripts/build_task3_resolver_v3_candidate.py"),
            "--ann-worker",
            "--ann-input",
            str(ann_input),
            "--ann-output",
            str(ann_output),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3600,
    )
    if completed.returncode != 0:
        report = {
            "status": "blocked_ann_worker_failed",
            "worker_python": ann_py,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-4000:],
            "pairs": [],
            "encode_report": encode_report,
        }
        write_json(ann_output, report)
        return set(), report
    report = read_json(ann_output)
    pairs = {pair_key(row.get("a"), row.get("b")) for row in report.get("pairs", []) if row.get("a") and row.get("b")}
    report["encode_report"] = encode_report
    report["worker_python"] = ann_py
    report["valid_r3_backend"] = report.get("status") == "ok" and text(report.get("backend")).startswith("faiss")
    write_json(ann_output, report)
    return pairs, report


def sample_id_from_event(event: dict[str, Any]) -> str:
    return text(event.get("source_row", {}).get("sample_id"))


def official_row_id_from_event(event: dict[str, Any]) -> str:
    return text(event.get("official_row_id") or event.get("source_row", {}).get("official_row_id"))


def flat_event_for_v3(event: dict[str, Any]) -> dict[str, Any]:
    tm = multiroute.target_meta(event)
    em = multiroute.event_meta(event)
    pm = multiroute.provenance_meta(event)
    return {
        "event_uid": sample_id_from_event(event),
        "event_id": text(event.get("source_row", {}).get("event_id")) or sample_id_from_event(event),
        "article_id": text(event.get("article_id")),
        "content_hash": multiroute.content_hash(event),
        "duplication_key": multiroute.duplication_key(event),
        "target_key": tm["target_key"],
        "target_name": tm["target_name"],
        "event_family": em["event_family"],
        "event_type": em["event_type"],
        "event_name": em["event_name"],
        "event_fact_signature": em["event_fact_signature"],
        "trigger_text": em["trigger_text"],
        "object_text": em["object_text"],
        "publish_time": text(event.get("publish_time")),
        "event_time": em["event_time"],
        "document_hash": pm["document_hash"],
        "sentence_id": pm["sentence_id"],
    }


def route_names_from_row(row: dict[str, Any]) -> list[str]:
    return [r for r in ["R1", "R2", "R3", "R4"] if bool(row.get(r))]


def route_mask(route_names: list[str]) -> int:
    return sum(ROUTE_BITS[name] for name in route_names)


def route_strength(row: dict[str, Any]) -> str:
    names = set(route_names_from_row(row))
    if "R1" in names or "R2" in names:
        return "strong"
    if "R4" in names and (
        bool(row.get("same_document_hash"))
        or bool(row.get("same_event_fact_signature"))
        or float(row.get("object_similarity") or 0.0) >= 0.60
        or float(row.get("trigger_similarity") or 0.0) >= 0.60
    ):
        return "strong"
    if names:
        return "recovery"
    return "weak"


def pair_row_from_multiroute(
    sid_a: str,
    sid_b: str,
    event_a: dict[str, Any],
    event_b: dict[str, Any],
    features: dict[str, Any],
    reasons: dict[str, list[str]],
    prefix: str,
) -> dict[str, Any]:
    key = pair_key(sid_a, sid_b)
    route_flags = {route: bool(reasons.get(route)) for route in ["R1", "R2", "R3", "R4"]}
    route_list = [route for route in ["R1", "R2", "R3", "R4"] if route_flags[route]]
    row = {
        "pair_id": f"{prefix}_{stable_hash(key[0], key[1])}",
        "sample_id_a": key[0],
        "sample_id_b": key[1],
        "article_id_a": text(event_a.get("article_id")) if sample_id_from_event(event_a) == key[0] else text(event_b.get("article_id")),
        "article_id_b": text(event_b.get("article_id")) if sample_id_from_event(event_b) == key[1] else text(event_a.get("article_id")),
        "official_row_id_a": official_row_id_from_event(event_a) if sample_id_from_event(event_a) == key[0] else official_row_id_from_event(event_b),
        "official_row_id_b": official_row_id_from_event(event_b) if sample_id_from_event(event_b) == key[1] else official_row_id_from_event(event_a),
        "target_key_a": multiroute.target_meta(event_a)["target_key"] if sample_id_from_event(event_a) == key[0] else multiroute.target_meta(event_b)["target_key"],
        "target_key_b": multiroute.target_meta(event_b)["target_key"] if sample_id_from_event(event_b) == key[1] else multiroute.target_meta(event_a)["target_key"],
        "event_family_a": multiroute.event_meta(event_a)["event_family"] if sample_id_from_event(event_a) == key[0] else multiroute.event_meta(event_b)["event_family"],
        "event_family_b": multiroute.event_meta(event_b)["event_family"] if sample_id_from_event(event_b) == key[1] else multiroute.event_meta(event_a)["event_family"],
        "event_name_a": multiroute.event_meta(event_a)["event_name"] if sample_id_from_event(event_a) == key[0] else multiroute.event_meta(event_b)["event_name"],
        "event_name_b": multiroute.event_meta(event_b)["event_name"] if sample_id_from_event(event_b) == key[1] else multiroute.event_meta(event_a)["event_name"],
        "R1": route_flags["R1"],
        "R2": route_flags["R2"],
        "R3": route_flags["R3"],
        "R4": route_flags["R4"],
        "r1_reasons": "|".join(reasons.get("R1", [])),
        "r2_reasons": "|".join(reasons.get("R2", [])),
        "r3_reasons": "|".join(reasons.get("R3", [])),
        "r4_reasons": "|".join(reasons.get("R4", [])),
        "same_target": bool(features["same_target"]),
        "same_family": bool(features["same_family"]),
        "same_sentence": bool(features["same_sentence"]),
        "same_document_hash": bool(features["same_document_hash"]),
        "same_event_fact_signature": bool(features["same_event_fact_signature"]),
        "fact_similarity": round(float(features["fact_similarity"]), 6),
        "object_similarity": round(float(features["object_similarity"]), 6),
        "trigger_similarity": round(float(features["trigger_similarity"]), 6),
        "evidence_similarity": round(float(features["evidence_similarity"]), 6),
        "time_relation": text(features["time_relation"]),
        "hard_conflicts": "|".join(multiroute.hard_conflicts(features)),
    }
    row["route_names"] = "|".join(route_list)
    row["route_count"] = len(route_list)
    row["route_mask"] = route_mask(route_list)
    row["route_strength"] = route_strength(row)
    row["channels"] = row["route_names"]
    row["route_reasons"] = {
        "R1": reasons.get("R1", []),
        "R2": reasons.get("R2", []),
        "R3": reasons.get("R3", []),
        "R4": reasons.get("R4", []),
    }
    return row


def generate_multiroute_candidates(
    events: list[dict[str, Any]],
    out_dir: Path,
    r3_worker_python: str | None = None,
    r3_encode_python: str | None = None,
    prefix: str = "OFFPAIRV3",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_sid = {sample_id_from_event(event): event for event in events if sample_id_from_event(event)}
    ids = sorted(by_sid)
    r3_pairs, r3_report = run_r3_ann_integrated(by_sid, out_dir, r3_worker_python, r3_encode_python)
    rows: list[dict[str, Any]] = []
    route_sets: dict[str, set[tuple[str, str]]] = {route: set() for route in ["R1", "R2", "R3", "R4"]}
    for sid_a, sid_b in combinations(ids, 2):
        event_a, event_b = by_sid[sid_a], by_sid[sid_b]
        features = multiroute.identity_features(event_a, event_b)
        conflicts = multiroute.hard_conflicts(features)
        key = pair_key(sid_a, sid_b)
        reasons = {
            "R1": multiroute.r1_reasons(event_a, event_b, features),
            "R2": multiroute.r2_reasons(event_a, event_b, features, conflicts),
            "R4": multiroute.r4_reasons(event_a, event_b, features, conflicts),
            "R3": ["qwen_faiss_dynamic_topk"] if key in r3_pairs else [],
        }
        if not any(reasons.values()):
            continue
        for route, route_reasons in reasons.items():
            if route_reasons:
                route_sets[route].add(key)
        rows.append(pair_row_from_multiroute(sid_a, sid_b, event_a, event_b, features, reasons, prefix=prefix))
    all_possible = len(ids) * (len(ids) - 1) // 2
    union = {pair_key(row["sample_id_a"], row["sample_id_b"]) for row in rows}
    report = {
        "candidate_mode": "multiroute",
        "event_count": len(ids),
        "all_possible_pairs": all_possible,
        "candidate_pairs": len(rows),
        "candidate_ratio": round(len(rows) / max(1, all_possible), 6),
        "candidate_reduction": round(1.0 - len(rows) / max(1, all_possible), 6),
        "route_pair_counts": {route: len(route_sets[route]) for route in ["R1", "R2", "R3", "R4"]},
        "route_union_pairs": len(union),
        "route_overlap_histogram": dict(Counter(int(row["route_count"]) for row in rows)),
        "route_strength_counts": dict(Counter(row["route_strength"] for row in rows)),
        "r3_report": {k: v for k, v in r3_report.items() if k != "pairs"},
        "r3_encode_python": r3_encode_python or DEFAULT_R3_ENCODE_PYTHON,
        "r3_ann_python": r3_worker_python or DEFAULT_R3_ANN_PYTHON,
        "route_bit_definition": ROUTE_BITS,
    }
    return rows, report


def legacy_route_flags_from_channels(channels: str) -> dict[str, bool]:
    channel_set = {part for part in text(channels).split("|") if part}
    return {
        "R1": bool(channel_set & {"target_fact", "target_time_fact", "same_document_hash", "same_sentence"}),
        "R2": bool(channel_set & {"target_object", "family_object", "family_trigger", "target_family_time", "target_fact"}),
        "R3": bool(channel_set & {"target_semantic"}),
        "R4": bool(channel_set & {"same_source_same_target", "same_modality", "target_time_fact"}),
    }


def enrich_legacy_official_candidates(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    legacy_rows = generate_legacy_resolver_candidates(events)
    event_by_sid = {sample_id_from_event(event): event for event in events}
    out: list[dict[str, Any]] = []
    for row in legacy_rows:
        sid_a, sid_b = text(row["sample_id_a"]), text(row["sample_id_b"])
        if sid_a not in event_by_sid or sid_b not in event_by_sid:
            continue
        features = multiroute.identity_features(event_by_sid[sid_a], event_by_sid[sid_b])
        flags = legacy_route_flags_from_channels(row.get("channels", ""))
        reasons = {route: [f"legacy_channel:{row.get('channels', '')}"] if flags[route] else [] for route in ["R1", "R2", "R3", "R4"]}
        merged = pair_row_from_multiroute(sid_a, sid_b, event_by_sid[sid_a], event_by_sid[sid_b], features, reasons, prefix="OFFPAIRLEG")
        merged.update({k: v for k, v in row.items() if k not in {"feature_dict"}})
        merged["legacy_channels"] = row.get("channels", "")
        merged["candidate_mode"] = "legacy"
        out.append(merged)
    all_possible = len(event_by_sid) * (len(event_by_sid) - 1) // 2
    return out, {
        "candidate_mode": "legacy",
        "event_count": len(event_by_sid),
        "all_possible_pairs": all_possible,
        "candidate_pairs": len(out),
        "candidate_ratio": round(len(out) / max(1, all_possible), 6),
        "candidate_reduction": round(1.0 - len(out) / max(1, all_possible), 6),
        "legacy_generator": "scripts/build_task3_kg_v4_official1799_inference_v1.py::generate_resolver_candidates",
        "legacy_channel_mapping_to_v3_routes": "audit-only feature mapping; selected v3 still scores every pair",
    }


def feature_row_for_candidate(flat_events: dict[str, dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    return v3model.feature_for_pair(
        flat_events,
        text(row["sample_id_a"]),
        text(row["sample_id_b"]),
        SimpleNamespace(**row),
    )


def v3_probabilities(model: Any, features: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not features:
        return []
    raw = model.predict_proba(features)
    classes = [text(label) for label in model.named_steps["clf"].classes_]
    rows: list[dict[str, float]] = []
    for values in raw:
        item = {label: 0.0 for label in ALL_LABELS}
        for idx, label in enumerate(classes):
            item[label] = float(values[idx])
        rows.append(item)
    return rows


def selected_v3_predict(probs: dict[str, float], thresholds: dict[str, float]) -> tuple[str, float]:
    return v3model.threshold_predict(
        probs,
        float(thresholds["same_thr"]),
        float(thresholds["diff_thr"]),
        float(thresholds.get("margin", 0.0)),
    )


class ResolverDSU:
    def __init__(self, members: Iterable[str]) -> None:
        self.parent = {member: member for member in members}
        self.members = {member: {member} for member in members}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def component(self, item: str) -> set[str]:
        return set(self.members[self.find(item)])

    def merge(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members[rb]

    def components(self) -> list[list[str]]:
        roots = sorted({self.find(item) for item in self.parent})
        return [sorted(self.members[root]) for root in roots]


def hard_conflict_between(flat_events: dict[str, dict[str, Any]], a: str, b: str, diff_pairs: set[tuple[str, str]]) -> str:
    key = pair_key(a, b)
    if key in diff_pairs:
        return "high_confidence_resolver_different"
    features = v3model.base_features(flat_events, key[0], key[1])
    if not features["same_target"] and (features["target_key_a"] or features["target_key_b"]):
        return "target_conflict"
    if features["time_relation"] == "time_conflict" and not (features["same_document_hash"] or features["same_event_fact_signature"]):
        return "time_conflict"
    if (
        features["same_target"]
        and features["same_family"]
        and features["object_similarity"] == 0
        and text(flat_events[key[0]].get("object_text"))
        and text(flat_events[key[1]].get("object_text"))
    ):
        return "object_conflict"
    return ""


def cluster_policy_for_edge(
    row: dict[str, Any],
    left_members: set[str],
    right_members: set[str],
    diff_pairs: set[tuple[str, str]],
    flat_events: dict[str, dict[str, Any]],
) -> tuple[bool, str, int]:
    implied_transitive_pairs = max(0, len(left_members) * len(right_members) - 1)
    for left in sorted(left_members):
        for right in sorted(right_members):
            conflict = hard_conflict_between(flat_events, left, right, diff_pairs)
            if conflict:
                return False, conflict, implied_transitive_pairs
    strength = text(row.get("route_strength"))
    confidence = float(row.get("confidence") or 0.0)
    if strength == "strong":
        return True, "strong_edge_pass", implied_transitive_pairs
    new_size = len(left_members | right_members)
    if confidence < RECOVERY_CLUSTER_SAME_THR:
        return False, f"recovery_confidence_below_{RECOVERY_CLUSTER_SAME_THR}", implied_transitive_pairs
    if new_size > RECOVERY_MAX_COMPONENT_SIZE:
        return False, f"recovery_component_size_gt_{RECOVERY_MAX_COMPONENT_SIZE}", implied_transitive_pairs
    return True, "recovery_high_confidence_pass", implied_transitive_pairs


def run_selected_v3_resolver(
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    candidate_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not events:
        return [], [], [], {"candidate_pairs": 0, "pair_decision_counts": {}, "event_decision_counts": {}}
    selection = load_v3_selection()
    thresholds = selection["thresholds"]
    model = joblib.load(V3_MODEL_PATH)
    event_by_sid = {sample_id_from_event(event): event for event in events}
    flat_events = {sid: flat_event_for_v3(event) for sid, event in event_by_sid.items()}
    feature_rows = [feature_row_for_candidate(flat_events, row) for row in candidates]
    probs_rows = v3_probabilities(model, feature_rows)
    resolved_pairs: list[dict[str, Any]] = []
    pair_decisions = Counter()
    diff_pairs: set[tuple[str, str]] = set()
    uncertain_sids: set[str] = set()
    same_edges: list[dict[str, Any]] = []
    for row, features, probs in zip(candidates, feature_rows, probs_rows):
        pred, conf = selected_v3_predict(probs, thresholds)
        pair_decisions[pred] += 1
        pair = pair_key(row["sample_id_a"], row["sample_id_b"])
        resolved = {
            **{k: v for k, v in row.items() if k not in {"feature_dict"}},
            "candidate_mode": candidate_mode,
            "prediction": pred,
            "confidence": round(float(conf), 6),
            "probabilities": probs,
            "prob_same_event": round(float(probs.get("same_event", 0.0)), 6),
            "prob_different_event": round(float(probs.get("different_event", 0.0)), 6),
            "prob_uncertain": round(float(probs.get("uncertain", 0.0)), 6),
            "hard_conflict_count": int(features.get("hard_conflict_count", 0) or 0),
            "features": features,
            "decision_reason": (
                f"selected_v3_thresholds_same={thresholds['same_thr']}"
                f"_diff={thresholds['diff_thr']}_margin={thresholds.get('margin', 0.0)}"
            ),
        }
        if pred == "different_event" and conf >= max(0.75, float(thresholds["diff_thr"])):
            diff_pairs.add(pair)
        elif pred == "same_event" and conf >= float(thresholds["same_thr"]):
            same_edges.append(resolved)
        elif pred == "uncertain":
            uncertain_sids.update(pair)
        resolved_pairs.append(resolved)

    same_edges.sort(key=lambda item: (-float(item["confidence"]), item["pair_id"]))
    dsu = ResolverDSU(sorted(event_by_sid))
    cluster_edge_audit: list[dict[str, Any]] = []
    transitive_merge_count = 0
    weak_transitive_merge_count = 0
    used_same_edges: list[dict[str, Any]] = []
    for row in same_edges:
        a, b = text(row["sample_id_a"]), text(row["sample_id_b"])
        left_members = dsu.component(a)
        right_members = dsu.component(b)
        if left_members == right_members:
            cluster_edge_audit.append({**edge_audit_base(row), "merge_action": "skip_same_component", "policy_reason": "", "implied_transitive_pairs": 0})
            continue
        allowed, reason, implied = cluster_policy_for_edge(row, left_members, right_members, diff_pairs, flat_events)
        if not allowed:
            cluster_edge_audit.append({**edge_audit_base(row), "merge_action": "veto", "policy_reason": reason, "implied_transitive_pairs": implied})
            continue
        dsu.merge(a, b)
        used_same_edges.append(row)
        transitive_merge_count += implied
        if text(row.get("route_strength")) != "strong":
            weak_transitive_merge_count += implied
        cluster_edge_audit.append({**edge_audit_base(row), "merge_action": "merge", "policy_reason": reason, "implied_transitive_pairs": implied})

    cluster_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for members in dsu.components():
        component_uncertain = any(sid in uncertain_sids for sid in members)
        component_edges = [
            row for row in used_same_edges if row["sample_id_a"] in members and row["sample_id_b"] in members
        ]
        cluster_confidence = round(
            sum(float(row["confidence"]) for row in component_edges) / len(component_edges)
            if component_edges
            else 0.5,
            6,
        )
        layer = "review_uncertain_graph" if component_uncertain else "trusted_main_graph"
        resolver_decision = "uncertain" if component_uncertain else "accepted"
        prefix = "OFFCEV3REV" if component_uncertain else "OFFCEV3"
        canonical_event_id = f"{prefix}_{stable_hash(*members, n=20)}"
        event_thread_id = "OFFETF3_" + stable_hash(canonical_event_id, n=20)
        representative = event_by_sid[members[0]]
        cluster_rows.append(
            {
                "canonical_event_id": canonical_event_id,
                "event_thread_id": event_thread_id,
                "graph_layer": layer,
                "resolver_decision": resolver_decision,
                "resolver_confidence": cluster_confidence,
                "member_count": len(members),
                "representative_sample_id": members[0],
                "representative_article_id": representative["article_id"],
                "representative_official_row_id": representative["official_row_id"],
                "target_key": representative["target"]["target_key"],
                "event_type": representative["event"]["event_type"],
                "event_family": representative["event"]["event_family"],
                "event_name": representative["event"]["event_name"],
                "event_fact_signature": representative["event"]["event_fact_signature"],
                "member_sample_ids": members,
                "member_official_row_ids": [event_by_sid[sid]["official_row_id"] for sid in members],
                "candidate_mode": candidate_mode,
                "resolver_model": "selected_v3_lightgbm_structured",
                "cluster_edge_count": len(component_edges),
                "cluster_route_strength_counts": dict(Counter(text(row.get("route_strength")) for row in component_edges)),
            }
        )
        for rank, sid in enumerate(members):
            event = event_by_sid[sid]
            mapping_rows.append(
                {
                    "canonical_event_id": canonical_event_id,
                    "event_thread_id": event_thread_id,
                    "graph_layer": layer,
                    "official_row_id": event["official_row_id"],
                    "article_id": event["article_id"],
                    "sample_id": sid,
                    "event_mention_id": event["source_row"]["event_id"],
                    "evidence_unit_id": event["source_row"]["evidence_unit_id"],
                    "target_key": event["target"]["target_key"],
                    "target_name": event["target"]["target_name"],
                    "target_type": event["target"]["target_type"],
                    "event_type": event["event"]["event_type"],
                    "event_family": event["event"]["event_family"],
                    "event_name": event["event"]["event_name"],
                    "trigger_text": event["event"]["trigger_text"],
                    "event_time": event["event"]["event_time"],
                    "polarity_or_state": event["event"]["polarity_or_state"],
                    "support_type": event["evidence"]["support_type"],
                    "event_fact_signature": event["event"]["event_fact_signature"],
                    "mention_span": event["target"]["mention_span"],
                    "trigger_span": event["event"]["trigger_span"],
                    "evidence_span": event["evidence"]["evidence_span"],
                    "provenance": event["provenance"],
                    "contract_status": event["quality"]["contract_status"],
                    "grounding_pass": event["quality"]["grounding_pass"],
                    "resolver_decision": resolver_decision,
                    "resolver_confidence": cluster_confidence,
                    "cluster_member_count": len(members),
                    "cluster_rank": rank,
                    "trust_score": round(min(0.99, max(0.0, 0.55 + 0.4 * cluster_confidence)), 6),
                    "acceptance_status": "pass" if layer == "trusted_main_graph" else "review",
                    "reject_reason": "" if layer == "trusted_main_graph" else "resolver_uncertain_pair_boundary",
                }
            )
    cluster_sizes = [int(row["member_count"]) for row in cluster_rows]
    stats = {
        "candidate_mode": candidate_mode,
        "candidate_pairs": len(candidates),
        "pair_decision_counts": dict(pair_decisions),
        "same_edges_predicted": len(same_edges),
        "same_edges_used_for_clustering": len(used_same_edges),
        "same_edges_vetoed_or_skipped": len(same_edges) - len(used_same_edges),
        "cannot_link_pairs": len(diff_pairs),
        "uncertain_pair_count": pair_decisions.get("uncertain", 0),
        "event_decision_counts": dict(Counter(row["resolver_decision"] for row in mapping_rows)),
        "trusted_event_mentions": sum(1 for row in mapping_rows if row["graph_layer"] == "trusted_main_graph"),
        "review_event_mentions": sum(1 for row in mapping_rows if row["graph_layer"] == "review_uncertain_graph"),
        "trusted_canonical_events": sum(1 for row in cluster_rows if row["graph_layer"] == "trusted_main_graph"),
        "review_canonical_events": sum(1 for row in cluster_rows if row["graph_layer"] == "review_uncertain_graph"),
        "cluster_count": len(cluster_rows),
        "multi_mention_clusters": sum(1 for row in cluster_rows if int(row["member_count"]) > 1),
        "max_cluster_size": max(cluster_sizes, default=0),
        "cluster_size_histogram": {str(k): int(v) for k, v in sorted(Counter(cluster_sizes).items())},
        "transitive_merge_count": transitive_merge_count,
        "weak_edge_transitive_merge_count": weak_transitive_merge_count,
        "cluster_edge_policy_counts": dict(Counter(row["policy_reason"] for row in cluster_edge_audit if row["policy_reason"])),
        "thresholds": thresholds,
        "recovery_cluster_same_thr": RECOVERY_CLUSTER_SAME_THR,
        "recovery_max_component_size": RECOVERY_MAX_COMPONENT_SIZE,
        "resolver_model_sha256": sha256_file(V3_MODEL_PATH),
        "resolver_selection_sha256": sha256_file(V3_SELECTION_PATH),
        "cluster_edge_policy": "strong selected-v3 same edges may merge; recovery edges require 0.995 confidence and cannot grow components beyond size 3; all merges still respect target/time/object/diff vetoes",
    }
    return mapping_rows, cluster_rows, resolved_pairs, {**stats, "cluster_edge_audit": cluster_edge_audit}


def status_for_representative(
    structured: dict[str, Any],
    mapping_by_row: dict[str, dict[str, Any]],
) -> tuple[str, str, str, str]:
    quality = structured["quality"]
    oid = structured["official_row_id"]
    if not text(structured.get("source_row", {}).get("content_hash")):
        return "quarantine", "missing_content", "", ""
    if not quality.get("target_success"):
        return "no_target", "task1_target_missing", "", ""
    if not quality.get("event_success"):
        return "no_event", "task1_event_missing", "", ""
    if quality.get("contract_status") == "quarantine":
        return "quarantine", ";".join(quality.get("contract_reasons") or []) or "contract_quarantine", "", ""
    if not quality.get("grounding_pass"):
        return "grounding_failed", ";".join(quality.get("grounding_reasons") or quality.get("contract_reasons") or []) or "grounding_failed", "", ""
    mapping = mapping_by_row.get(oid)
    if not mapping:
        return "review", "resolver_not_mapped", "", "not_mapped"
    if mapping["graph_layer"] == "trusted_main_graph":
        return "accepted_to_main_graph", "", mapping["canonical_event_id"], mapping["resolver_decision"]
    return "review", mapping.get("reject_reason") or "resolver_uncertain", mapping["canonical_event_id"], mapping["resolver_decision"]


def build_processing_status(
    rows: list[dict[str, Any]],
    structured_all: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    structured_by_row = {row["official_row_id"]: row for row in structured_all}
    mapping_by_row = {row["official_row_id"]: row for row in mapping_rows}
    rep_status_cache: dict[str, tuple[str, str, str, str]] = {}
    status_rows: list[dict[str, Any]] = []
    for row in rows:
        structured = structured_by_row[row["official_row_id"]]
        rep_id = row["representative_official_row_id"]
        if row["duplicate_of"]:
            if rep_id not in rep_status_cache:
                rep_status_cache[rep_id] = status_for_representative(structured_by_row[rep_id], mapping_by_row)
            rep_status, rep_reason, rep_cid, rep_resolver = rep_status_cache[rep_id]
            status = "duplicate_of"
            reason = f"duplicate_of:{rep_id};representative_status:{rep_status};representative_reason:{rep_reason}"
            canonical_event_id = rep_cid
            resolver_decision = rep_resolver
            graph_layer = mapping_by_row.get(rep_id, {}).get("graph_layer", "")
        else:
            status, reason, canonical_event_id, resolver_decision = status_for_representative(structured, mapping_by_row)
            graph_layer = mapping_by_row.get(row["official_row_id"], {}).get("graph_layer", "")
            rep_status_cache[row["official_row_id"]] = (status, reason, canonical_event_id, resolver_decision)
        quality = structured["quality"]
        status_rows.append(
            {
                "official_row_id": row["official_row_id"],
                "article_id": row["article_id"],
                "sheet_name": row["sheet_name"],
                "sheet_row_index": row["sheet_row_index"],
                "content_hash": row["content_hash"],
                "is_representative": row["is_representative"],
                "duplicate_of": row["duplicate_of"],
                "representative_official_row_id": rep_id,
                "processing_status": status,
                "status_reason": reason,
                "task1_target_success": bool(quality.get("target_success")),
                "task1_event_success": bool(quality.get("event_success")),
                "task1_structure_success": bool(quality.get("structure_success")),
                "contract_status": text(quality.get("contract_status")),
                "grounding_pass": bool(quality.get("grounding_pass")),
                "target_grounded": bool(quality.get("target_grounded")),
                "event_grounded": bool(quality.get("event_grounded")),
                "evidence_grounded": bool(quality.get("evidence_grounded")),
                "provenance_valid": bool(quality.get("provenance_valid")),
                "contract_reasons": json.dumps(quality.get("contract_reasons") or [], ensure_ascii=False),
                "grounding_reasons": json.dumps(quality.get("grounding_reasons") or [], ensure_ascii=False),
                "target_span_status": text(quality.get("target_span_status")),
                "task1_target_key": structured["target"]["target_key"],
                "task1_target_name": structured["target"]["target_name"],
                "task1_target_type": structured["target"]["target_type"],
                "task1_event_name": structured["event"]["event_name"],
                "task1_event_family": structured["event"]["event_family"],
                "trigger_text": structured["event"]["trigger_text"],
                "event_time": structured["event"]["event_time"],
                "event_fact_signature": structured["event"]["event_fact_signature"],
                "event_mention_id": structured["source_row"]["event_id"],
                "canonical_event_id": canonical_event_id,
                "resolver_decision": resolver_decision,
                "graph_layer": graph_layer,
            }
        )
    return pd.DataFrame(status_rows)


def node(stable_id: str, node_type: str, layer: str, **props: Any) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "node_type": node_type,
        "graph_layer": layer,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **props,
    }


def edge(source_id: str, target_id: str, relation_type: str, layer: str, **props: Any) -> dict[str, Any]:
    return {
        "stable_id": "OFFED_" + stable_hash(source_id, target_id, relation_type, layer),
        "source_id": source_id,
        "target_id": target_id,
        "relation_type": relation_type,
        "graph_layer": layer,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **props,
    }


def build_graph(
    mapping_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    structured_by_row: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    canonical_events: list[dict[str, Any]] = []
    task4_rows: list[dict[str, Any]] = []
    for cluster in cluster_rows:
        layer = cluster["graph_layer"]
        cid = cluster["canonical_event_id"]
        cluster_props = {k: v for k, v in cluster.items() if k != "graph_layer"}
        nodes[cid] = node(cid, "CanonicalEvent", layer, **cluster_props)
        canonical_events.append(cluster)
    for mapping in mapping_rows:
        structured = structured_by_row[mapping["official_row_id"]]
        layer = mapping["graph_layer"]
        cid = mapping["canonical_event_id"]
        mid = mapping["event_mention_id"]
        eid = mapping["evidence_unit_id"]
        doc_id = "OFFDOC_" + text(structured["provenance"]["document_hash"])[:24]
        source_id = "OFFSRC_" + stable_hash(structured["source"]["source_name"])
        target_type = text(structured["target"]["target_type"])
        target_node_type = "Company" if target_type == "company" else "Industry"
        target_id = "OFFTGT_" + stable_hash(structured["target"]["target_key"])
        nodes[mid] = node(
            mid,
            "EventMention",
            layer,
            official_row_id=mapping["official_row_id"],
            article_id=mapping["article_id"],
            canonical_event_id=cid,
            event_mention_id=mid,
            target_key=mapping["target_key"],
            event_type=mapping["event_type"],
            event_family=mapping["event_family"],
            event_name=mapping["event_name"],
            event_fact_signature=mapping["event_fact_signature"],
            event_time=mapping["event_time"],
            contract_status=mapping["contract_status"],
            resolver_decision=mapping["resolver_decision"],
            resolver_confidence=mapping["resolver_confidence"],
        )
        nodes[eid] = node(
            eid,
            "EvidenceUnit",
            layer,
            evidence_unit_id=eid,
            article_id=mapping["article_id"],
            evidence_span=mapping["evidence_span"],
            provenance=mapping["provenance"],
            support_type=mapping["support_type"],
        )
        nodes[doc_id] = node(
            doc_id,
            "Document",
            layer,
            article_id=mapping["article_id"],
            official_row_id=mapping["official_row_id"],
            document_hash=structured["provenance"]["document_hash"],
            title=structured.get("title", ""),
            publish_time=structured["publish_time"],
        )
        nodes[source_id] = node(source_id, "Source", layer, source_name=structured["source"]["source_name"])
        nodes[target_id] = node(
            target_id,
            target_node_type,
            layer,
            target_key=structured["target"]["target_key"],
            target_name=structured["target"]["target_name"],
            target_type=structured["target"]["target_type"],
            target_code=structured["target"]["target_code"],
        )
        for graph_edge in [
            edge(mid, cid, "INSTANCE_OF_EVENT", layer, confidence=mapping["resolver_confidence"]),
            edge(mid, eid, "BELONGS_TO_EVIDENCE_UNIT", layer),
            edge(eid, cid, "EVIDENCE_FOR", layer),
            edge(mid, doc_id, "EXTRACTED_FROM", layer),
            edge(doc_id, source_id, "PUBLISHED_BY", layer),
            edge(cid, target_id, "TARGETS_COMPANY" if target_type == "company" else "TARGETS_INDUSTRY", layer),
        ]:
            edges[graph_edge["stable_id"]] = graph_edge
        if layer == "trusted_main_graph":
            task4_rows.append(
                {
                    "canonical_event_id": cid,
                    "event_thread_id": mapping["event_thread_id"],
                    "event_mention_id": mid,
                    "official_row_id": mapping["official_row_id"],
                    "article_id": mapping["article_id"],
                    "schema_version": structured["schema_version"],
                    "target_key": mapping["target_key"],
                    "target_name": mapping["target_name"],
                    "target_type": mapping["target_type"],
                    "event_type": mapping["event_type"],
                    "event_family": mapping["event_family"],
                    "event_name": mapping["event_name"],
                    "trigger_text": mapping["trigger_text"],
                    "event_time": mapping["event_time"],
                    "polarity_or_state": mapping["polarity_or_state"],
                    "support_type": mapping["support_type"],
                    "event_fact_signature": mapping["event_fact_signature"],
                    "evidence_span": mapping["evidence_span"],
                    "provenance": mapping["provenance"],
                    "contract_status": mapping["contract_status"],
                    "resolver_decision": mapping["resolver_decision"],
                    "resolver_confidence": mapping["resolver_confidence"],
                    "trust_score": mapping["trust_score"],
                    "graph_layer": layer,
                    "acceptance_status": "pass",
                }
            )
    return list(nodes.values()), list(edges.values()), canonical_events, task4_rows


def build_review_sidecar(status_df: pd.DataFrame, structured_by_row: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in status_df.to_dict("records"):
        if rec["processing_status"] == "accepted_to_main_graph":
            continue
        structured = structured_by_row.get(rec["official_row_id"])
        rows.append(
            {
                "official_row_id": rec["official_row_id"],
                "article_id": rec["article_id"],
                "processing_status": rec["processing_status"],
                "status_reason": rec["status_reason"],
                "duplicate_of": rec["duplicate_of"],
                "representative_official_row_id": rec["representative_official_row_id"],
                "canonical_event_id": rec["canonical_event_id"],
                "resolver_decision": rec["resolver_decision"],
                "graph_layer": rec["graph_layer"],
                "target_key": rec["task1_target_key"],
                "event_name": rec["task1_event_name"],
                "event_family": rec["task1_event_family"],
                "trigger_text": rec["trigger_text"],
                "event_time": rec["event_time"],
                "quality": structured.get("quality") if structured else {},
                "evidence": structured.get("evidence") if structured else {},
                "provenance": structured.get("provenance") if structured else {},
            }
        )
    return rows


def edge_audit_base(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": row.get("pair_id", ""),
        "sample_id_a": row["sample_id_a"],
        "sample_id_b": row["sample_id_b"],
        "confidence": row["confidence"],
        "prob_same_event": row["prob_same_event"],
        "prob_different_event": row["prob_different_event"],
        "prob_uncertain": row["prob_uncertain"],
        "route_names": row.get("route_names", ""),
        "route_count": row.get("route_count", 0),
        "route_mask": row.get("route_mask", 0),
        "route_strength": row.get("route_strength", ""),
    }


def metric_summary(rows: list[dict[str, Any]], labels: list[str] = PRIMARY_LABELS) -> dict[str, Any]:
    gold = [row["gold_relation"] for row in rows]
    pred = [row["prediction"] for row in rows]
    p, r, f, support = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    per_class = {
        label: {
            "precision": round(float(p[idx]), 6),
            "recall": round(float(r[idx]), 6),
            "f1": round(float(f[idx]), 6),
            "support": int(support[idx]),
        }
        for idx, label in enumerate(labels)
    }
    same = per_class.get("same_event", {})
    diff = per_class.get("different_event", {})
    return {
        "row_count": len(rows),
        "gold_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(pred)),
        "labels_parameter": labels,
        "per_class": per_class,
        "same_event_precision": same.get("precision", 0.0),
        "same_event_recall": same.get("recall", 0.0),
        "same_event_f1": same.get("f1", 0.0),
        "different_event_precision": diff.get("precision", 0.0),
        "different_event_recall": diff.get("recall", 0.0),
        "different_event_f1": diff.get("f1", 0.0),
        "pair_macro_f1": round(float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)), 6),
        "sklearn_macro_f1": round(float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)), 6),
        "accuracy": round(float(accuracy_score(gold, pred)), 6),
        "balanced_accuracy_same_different_only": round(
            float(balanced_accuracy_score(gold, [p if p in PRIMARY_LABELS else "uncertain" for p in pred]))
            if set(gold) <= set(PRIMARY_LABELS)
            else 0.0,
            6,
        ),
        "false_merge": sum(1 for row in rows if row["gold_relation"] == "different_event" and row["prediction"] == "same_event"),
        "split_error": sum(1 for row in rows if row["gold_relation"] == "same_event" and row["prediction"] != "same_event"),
        "uncertain_count": sum(1 for row in rows if row["prediction"] == "uncertain"),
        "uncertain_rate": round(sum(1 for row in rows if row["prediction"] == "uncertain") / max(1, len(rows)), 6),
    }


def score_dev_candidates(candidate_df: pd.DataFrame, flat_events: dict[str, dict[str, Any]], model: Any, thresholds: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = candidate_df.to_dict("records")
    features = [feature_row_for_candidate(flat_events, row) for row in rows]
    probs_rows = v3_probabilities(model, features)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row, features_row, probs in zip(rows, features, probs_rows):
        key = pair_key(row["sample_id_a"], row["sample_id_b"])
        pred, conf = selected_v3_predict(probs, thresholds)
        route_list = route_names_from_row(row)
        out[key] = {
            "sample_id_a": key[0],
            "sample_id_b": key[1],
            "prediction": pred,
            "confidence": round(float(conf), 6),
            "prob_same_event": round(float(probs.get("same_event", 0.0)), 6),
            "prob_different_event": round(float(probs.get("different_event", 0.0)), 6),
            "prob_uncertain": round(float(probs.get("uncertain", 0.0)), 6),
            "route_names": "|".join(route_list) if route_list else text(row.get("route_names")),
            "route_count": int(row.get("route_count") or len(route_list)),
            "route_mask": int(row.get("route_mask") or route_mask(route_list)),
            "route_strength": text(row.get("route_strength")) or route_strength(row),
            "candidate_hit": True,
            "features": features_row,
        }
    return out


def legacy_dev_candidates_as_v3(dev_ids: set[str]) -> pd.DataFrame:
    raw = pd.read_parquet(DEV_LEGACY_CANDIDATES)
    rows: list[dict[str, Any]] = []
    for row in raw.itertuples(index=False):
        sid_a, sid_b = text(row.sample_id_a), text(row.sample_id_b)
        if sid_a not in dev_ids or sid_b not in dev_ids:
            continue
        flags = legacy_route_flags_from_channels(text(getattr(row, "channels", "")))
        item = {
            "sample_id_a": pair_key(sid_a, sid_b)[0],
            "sample_id_b": pair_key(sid_a, sid_b)[1],
            "R1": flags["R1"],
            "R2": flags["R2"],
            "R3": flags["R3"],
            "R4": flags["R4"],
            "r1_reasons": "legacy_candidate" if flags["R1"] else "",
            "r2_reasons": "legacy_candidate" if flags["R2"] else "",
            "r3_reasons": "legacy_candidate" if flags["R3"] else "",
            "r4_reasons": "legacy_candidate" if flags["R4"] else "",
            "same_sentence": bool(getattr(row, "feature_trace", {}) and False),
            "same_document_hash": "same_document_hash" in text(getattr(row, "channels", "")),
            "same_event_fact_signature": "target_fact" in text(getattr(row, "channels", "")),
            "fact_similarity": float(getattr(row, "feature_trace", {})["fact_similarity"]) if isinstance(getattr(row, "feature_trace", {}), dict) and "fact_similarity" in getattr(row, "feature_trace", {}) else 0.0,
            "object_similarity": 0.0,
            "trigger_similarity": 0.0,
            "evidence_similarity": 0.0,
            "time_relation": text(getattr(row, "time_relation", "")),
            "hard_conflicts": text(getattr(row, "hard_conflicts", "")),
            "legacy_channels": text(getattr(row, "channels", "")),
        }
        routes = route_names_from_row(item)
        item["route_names"] = "|".join(routes)
        item["route_count"] = len(routes)
        item["route_mask"] = route_mask(routes)
        item["route_strength"] = route_strength(item)
        rows.append(item)
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = pair_key(row["sample_id_a"], row["sample_id_b"])
        old = dedup.get(key)
        if old:
            merged_routes = sorted(set(text(old.get("route_names")).split("|")) | set(text(row.get("route_names")).split("|")))
            merged_routes = [route for route in merged_routes if route]
            old["route_names"] = "|".join(merged_routes)
            old["route_count"] = len(merged_routes)
            old["route_mask"] = route_mask(merged_routes)
            for route in ["R1", "R2", "R3", "R4"]:
                old[route] = old[route] or row[route]
        else:
            dedup[key] = row
    return pd.DataFrame(list(dedup.values()))


def dev_regression(out_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    event_df = pd.read_parquet(DEV_EVENT_SPLIT)
    event_df = event_df[event_df["split"] == "dev"].copy()
    flat_events = {text(row.event_uid): row._asdict() for row in event_df.itertuples(index=False)}
    dev_ids = set(flat_events)
    pair_df = pd.read_parquet(DEV_PAIR_SPLIT)
    labeled = pair_df[
        (pair_df["pair_split"] == "dev")
        & (pair_df["gold_relation"].isin(PRIMARY_LABELS))
    ].copy()
    labels = {pair_key(row.sample_id_a, row.sample_id_b): text(row.gold_relation) for row in labeled.itertuples(index=False)}
    same_gold = {key for key, value in labels.items() if value == "same_event"}
    selection = load_v3_selection()
    thresholds = selection["thresholds"]
    model = joblib.load(V3_MODEL_PATH)

    multiroute_df = pd.read_parquet(DEV_MULTIROUTE_CANDIDATES)
    legacy_df = legacy_dev_candidates_as_v3(dev_ids)
    experiments = {}
    prediction_tables = []
    for name, candidate_df in [
        ("legacy_candidate_selected_v3", legacy_df),
        ("multiroute_candidate_selected_v3", multiroute_df),
    ]:
        scored = score_dev_candidates(candidate_df, flat_events, model, thresholds)
        rows: list[dict[str, Any]] = []
        for key, gold in sorted(labels.items()):
            base = scored.get(key)
            if base:
                row = {**base, "gold_relation": gold, "decision_source": "candidate_resolver"}
            else:
                row = {
                    "sample_id_a": key[0],
                    "sample_id_b": key[1],
                    "gold_relation": gold,
                    "prediction": "different_event",
                    "confidence": 1.0,
                    "prob_same_event": 0.0,
                    "prob_different_event": 1.0,
                    "prob_uncertain": 0.0,
                    "candidate_hit": False,
                    "decision_source": "default_non_candidate_different_for_labeled_dev_pair",
                    "route_names": "",
                    "route_count": 0,
                    "route_mask": 0,
                    "route_strength": "",
                }
            row["experiment"] = name
            rows.append(row)
        candidate_keys = set(scored)
        candidate_hit_same = len(candidate_keys & same_gold)
        stats = {
            "candidate_pairs": int(len(candidate_keys)),
            "dev_event_count": int(len(dev_ids)),
            "dev_all_possible_pairs": int(len(dev_ids) * (len(dev_ids) - 1) // 2),
            "candidate_ratio": round(len(candidate_keys) / max(1, len(dev_ids) * (len(dev_ids) - 1) // 2), 6),
            "candidate_reduction": round(1.0 - len(candidate_keys) / max(1, len(dev_ids) * (len(dev_ids) - 1) // 2), 6),
            "candidate_oracle_recall": round(candidate_hit_same / max(1, len(same_gold)), 6),
            "candidate_hit_same_event": candidate_hit_same,
            "same_event_denominator": len(same_gold),
            "formal_labeled_dev_metrics": metric_summary(rows),
            "route_distribution": dict(Counter(text(v.get("route_names")) for v in scored.values())),
            "route_strength_counts": dict(Counter(text(v.get("route_strength")) for v in scored.values())),
            "prediction_counts_on_candidates": dict(Counter(v["prediction"] for v in scored.values())),
        }
        experiments[name] = stats
        for row in rows:
            prediction_tables.append({k: v for k, v in row.items() if k != "features"})

    report = {
        "status": "dev_regression_complete_no_final_holdout_no_official_gold",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_v3_thresholds": thresholds,
        "selected_v3_model_sha256": sha256_file(V3_MODEL_PATH),
        "formal_labeled_dev_denominator": {
            "human_pair_count": int(len(labels)),
            "same_event": int(sum(1 for v in labels.values() if v == "same_event")),
            "different_event": int(sum(1 for v in labels.values() if v == "different_event")),
            "unlabeled_pairs_defaulted_as_different_for_formal_metrics": False,
        },
        "experiments": experiments,
        "comparison": {
            "candidate_pairs_delta": experiments["multiroute_candidate_selected_v3"]["candidate_pairs"] - experiments["legacy_candidate_selected_v3"]["candidate_pairs"],
            "candidate_oracle_recall_delta": round(
                experiments["multiroute_candidate_selected_v3"]["candidate_oracle_recall"]
                - experiments["legacy_candidate_selected_v3"]["candidate_oracle_recall"],
                6,
            ),
            "same_event_recall_delta": round(
                experiments["multiroute_candidate_selected_v3"]["formal_labeled_dev_metrics"]["same_event_recall"]
                - experiments["legacy_candidate_selected_v3"]["formal_labeled_dev_metrics"]["same_event_recall"],
                6,
            ),
            "same_event_precision_delta": round(
                experiments["multiroute_candidate_selected_v3"]["formal_labeled_dev_metrics"]["same_event_precision"]
                - experiments["legacy_candidate_selected_v3"]["formal_labeled_dev_metrics"]["same_event_precision"],
                6,
            ),
            "multiroute_improves_candidate_recall": (
                experiments["multiroute_candidate_selected_v3"]["candidate_oracle_recall"]
                > experiments["legacy_candidate_selected_v3"]["candidate_oracle_recall"]
            ),
            "precision_not_obviously_degraded_on_formal_labeled_dev": (
                experiments["multiroute_candidate_selected_v3"]["formal_labeled_dev_metrics"]["same_event_precision"]
                >= experiments["legacy_candidate_selected_v3"]["formal_labeled_dev_metrics"]["same_event_precision"] - 0.05
            ),
        },
        "constraints": {
            "official1799_filled_gold_read_count": 0,
            "final_holdout_read_count": 0,
            "resolver_retrained": False,
            "thresholds_tuned": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(out_dir / "dev_regression_legacy_vs_multiroute.json", report)
    parquet_ready_frame(prediction_tables).to_parquet(out_dir / "dev_regression_pair_predictions.parquet", index=False)
    return report


def load_official_structured_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    label_rows, _ = read_label_blind_inputs()
    representatives, _, _ = build_representatives(label_rows)
    all_structured = read_jsonl(LEGACY_OFFICIAL_DIR / "official1799_structured_events.jsonl")
    structured_by_row = {row["official_row_id"]: row for row in all_structured}
    rep_structured = [structured_by_row[row["official_row_id"]] for row in representatives]
    resolver_input_events = [
        row
        for row in rep_structured
        if row.get("quality", {}).get("grounding_pass")
        and row.get("quality", {}).get("contract_status") == "pass"
    ]
    return label_rows, representatives, all_structured, resolver_input_events


def write_prediction_tables(out_dir: Path, status_df: pd.DataFrame, mapping_rows: list[dict[str, Any]], cluster_rows: list[dict[str, Any]]) -> None:
    map_by_row = {row["official_row_id"]: row for row in mapping_rows}
    cluster_by_id = {row["canonical_event_id"]: row for row in cluster_rows}
    rows: list[dict[str, Any]] = []
    for item in status_df.to_dict("records"):
        mapping = map_by_row.get(text(item.get("official_row_id")), {})
        cluster = cluster_by_id.get(text(mapping.get("canonical_event_id")), {})
        rows.append(
            {
                **item,
                "canonical_event_id": text(mapping.get("canonical_event_id") or item.get("canonical_event_id")),
                "event_thread_id": text(mapping.get("event_thread_id")),
                "cluster_member_count": int(mapping.get("cluster_member_count") or cluster.get("member_count") or 0),
                "resolver_confidence": mapping.get("resolver_confidence"),
                "acceptance_status": text(mapping.get("acceptance_status")),
                "candidate_mode": "multiroute",
                "resolver_model": "selected_v3_lightgbm_structured",
            }
        )
    frame = pd.DataFrame(rows)
    parquet_ready_frame(rows).to_parquet(out_dir / "official1799_multiroute_predictions.parquet", index=False)
    with pd.ExcelWriter(out_dir / "official1799_multiroute_predictions.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="predictions", index=False)


def route_distribution(candidates: list[dict[str, Any]], resolved_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = Counter()
    route_combo_counts = Counter()
    route_strength_counts = Counter()
    for row in candidates:
        routes = route_names_from_row(row)
        route_combo_counts["|".join(routes)] += 1
        route_strength_counts[text(row.get("route_strength"))] += 1
        for route in routes:
            route_counts[route] += 1
    same_by_route = Counter()
    for row in resolved_pairs:
        if row.get("prediction") != "same_event":
            continue
        for route in route_names_from_row(row):
            same_by_route[route] += 1
    return {
        "candidate_pairs": len(candidates),
        "route_counts": dict(route_counts),
        "route_combo_counts": dict(route_combo_counts),
        "route_strength_counts": dict(route_strength_counts),
        "same_event_edges_by_route": dict(same_by_route),
        "route_bit_definition": ROUTE_BITS,
    }


def old_legacy_stats() -> dict[str, Any]:
    pair_rows = read_jsonl(LEGACY_OFFICIAL_DIR / "resolver_v2_pair_decisions.jsonl")
    canonical_rows = read_jsonl(LEGACY_OFFICIAL_DIR / "canonical_events.jsonl")
    sizes = [int(row.get("member_count") or 0) for row in canonical_rows]
    return {
        "source": rel(LEGACY_OFFICIAL_DIR),
        "candidate_pairs": len(pair_rows),
        "same_event_edges": sum(1 for row in pair_rows if row.get("prediction") == "same_event"),
        "different_event_edges": sum(1 for row in pair_rows if row.get("prediction") == "different_event"),
        "uncertain_edges": sum(1 for row in pair_rows if row.get("prediction") == "uncertain"),
        "cluster_count": len(canonical_rows),
        "max_cluster_size": max(sizes, default=0),
        "multi_mention_clusters": sum(1 for size in sizes if size > 1),
        "covered_mentions_in_clusters": len({sid for row in canonical_rows for sid in row.get("member_sample_ids", [])}),
    }


def official_inference(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "protected_sha256_before.json", protected_sha(False, out_dir))
    inventory = build_integration_inventory(out_dir)
    write_json(out_dir / "integration_inventory.json", inventory)
    dev_report = dev_regression(out_dir)
    if not (
        dev_report["comparison"]["multiroute_improves_candidate_recall"]
        and dev_report["comparison"]["precision_not_obviously_degraded_on_formal_labeled_dev"]
    ):
        raise RuntimeError("dev regression gate failed; not running official1799 inference")

    label_rows, representatives, all_structured, resolver_input_events = load_official_structured_inputs()
    if args.candidate_mode == "legacy":
        candidates, candidate_report = enrich_legacy_official_candidates(resolver_input_events)
    elif args.candidate_mode == "multiroute":
        candidates, candidate_report = generate_multiroute_candidates(
            resolver_input_events,
            out_dir,
            r3_worker_python=args.r3_worker_python,
            r3_encode_python=args.r3_encode_python,
            prefix="OFFPAIRV3",
        )
    else:
        raise ValueError(f"unsupported candidate mode: {args.candidate_mode}")

    mapping_rows, cluster_rows, resolver_pairs, resolver_stats_full = run_selected_v3_resolver(
        resolver_input_events,
        candidates,
        args.candidate_mode,
    )
    cluster_edge_audit = resolver_stats_full.pop("cluster_edge_audit")
    structured_by_row = {row["official_row_id"]: row for row in all_structured}
    status_df = build_processing_status(label_rows, all_structured, mapping_rows)
    status_df.to_parquet(out_dir / "official1799_processing_status.parquet", index=False)
    graph_nodes, graph_edges, canonical_events, task4_rows = build_graph(mapping_rows, cluster_rows, structured_by_row)
    review_sidecar = build_review_sidecar(status_df, structured_by_row)

    write_jsonl(out_dir / "resolver_v3_pair_decisions.jsonl", resolver_pairs)
    write_jsonl(out_dir / "canonical_events.jsonl", canonical_events)
    write_jsonl(out_dir / "graph_nodes.jsonl", graph_nodes)
    write_jsonl(out_dir / "graph_edges.jsonl", graph_edges)
    write_jsonl(out_dir / "review_sidecar.jsonl", review_sidecar)
    write_jsonl(out_dir / "task4_event_interface.jsonl", task4_rows)
    parquet_ready_frame(candidates).to_parquet(out_dir / "official1799_candidate_pairs.parquet", index=False)
    parquet_ready_frame([{k: v for k, v in row.items() if k != "features"} for row in resolver_pairs]).to_parquet(
        out_dir / "official1799_pair_decisions.parquet",
        index=False,
    )
    parquet_ready_frame(cluster_edge_audit).to_parquet(out_dir / "cluster_edge_audit.parquet", index=False)
    write_prediction_tables(out_dir, status_df, mapping_rows, cluster_rows)

    route_report = route_distribution(candidates, resolver_pairs)
    write_json(out_dir / "route_distribution.json", route_report)
    cluster_stats = {
        **resolver_stats_full,
        "canonical_event_rows": len(canonical_events),
        "graph_nodes": len(graph_nodes),
        "graph_edges": len(graph_edges),
        "task4_interface_rows": len(task4_rows),
        "review_sidecar_rows": len(review_sidecar),
    }
    write_json(out_dir / "cluster_statistics.json", cluster_stats)

    legacy_stats = old_legacy_stats()
    multiroute_stats = {
        "candidate_pairs": len(candidates),
        "same_event_edges": sum(1 for row in resolver_pairs if row.get("prediction") == "same_event"),
        "different_event_edges": sum(1 for row in resolver_pairs if row.get("prediction") == "different_event"),
        "uncertain_edges": sum(1 for row in resolver_pairs if row.get("prediction") == "uncertain"),
        "cluster_count": len(canonical_events),
        "max_cluster_size": max((int(row.get("member_count") or 0) for row in canonical_events), default=0),
        "multi_mention_clusters": sum(1 for row in canonical_events if int(row.get("member_count") or 0) > 1),
        "covered_mentions_in_clusters": len({sid for row in canonical_events for sid in row.get("member_sample_ids", [])}),
    }
    comparison = {
        "status": "unlabeled_official1799_comparison_not_gold_performance",
        "legacy": legacy_stats,
        "multiroute_selected_v3": multiroute_stats,
        "candidate_pairs_delta": multiroute_stats["candidate_pairs"] - legacy_stats["candidate_pairs"],
        "new_same_event_edges_delta": multiroute_stats["same_event_edges"] - legacy_stats["same_event_edges"],
        "new_recovered_mentions_delta": multiroute_stats["covered_mentions_in_clusters"] - legacy_stats["covered_mentions_in_clusters"],
        "cluster_count_delta": multiroute_stats["cluster_count"] - legacy_stats["cluster_count"],
        "max_cluster_size_delta": multiroute_stats["max_cluster_size"] - legacy_stats["max_cluster_size"],
        "weak_edge_transitive_merge_count": resolver_stats_full["weak_edge_transitive_merge_count"],
        "official_gold_labels_used": False,
    }
    write_json(out_dir / "legacy_vs_multiroute_unlabeled_comparison.json", comparison)

    manifest = {
        "status": "official1799_multiroute_prediction_frozen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "candidate_mode": args.candidate_mode,
        "repro_command": (
            f"{RUNTIME_PYTHON} "
            f"scripts/build_task3_resolver_v3_official1799_multiroute.py "
            f"--candidate-mode {args.candidate_mode} --out {rel(out_dir)} "
            f"--r3-encode-python {args.r3_encode_python or DEFAULT_R3_ENCODE_PYTHON} "
            f"--r3-worker-python {args.r3_worker_python or DEFAULT_R3_ANN_PYTHON}"
        ),
        "official_raw_input": path_stat(RAW_TEST_XLSX),
        "official_structured_input_reused": path_stat(LEGACY_OFFICIAL_DIR / "official1799_structured_events.jsonl"),
        "selected_v3_model": path_stat(V3_MODEL_PATH),
        "selected_v3_selection": path_stat(V3_SELECTION_PATH),
        "selected_v3_thresholds": load_v3_selection()["thresholds"],
        "candidate_generation": candidate_report,
        "dev_regression": dev_report,
        "route_distribution": route_report,
        "cluster_statistics": cluster_stats,
        "legacy_vs_multiroute_unlabeled_comparison": comparison,
        "constraints": {
            "official1799_filled_gold_read_count": 0,
            "final_holdout_read_count": 0,
            "official1799_gold_used_for_route_or_threshold_selection": False,
            "resolver_retrained": False,
            "thresholds_modified": False,
            "legacy_official_outputs_overwritten": False,
            "task1_core_modified": False,
            "structured_event_contract_modified": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(out_dir / "inference_manifest.json", manifest)
    write_readme(out_dir, manifest)
    write_repro_script(out_dir, args.candidate_mode)
    write_json(out_dir / "protected_sha256_after.json", protected_sha(True, out_dir))
    return manifest


def write_readme(out_dir: Path, manifest: dict[str, Any]) -> None:
    readme = f"""# Task3 Resolver v3 Official1799 Multiroute

This directory contains label-blind official1799 Task3 predictions generated with the existing Resolver v3 multiroute candidate recall and selected v3 LightGBM resolver.

No official1799 filled Gold or final_holdout labels were read. Resolver v2 frozen outputs were not overwritten.

Reproduce:

```bash
{manifest['repro_command']}
```

Key files:

- `integration_inventory.json`: paths, selected model, thresholds, routes, and legacy pipeline audit.
- `official1799_candidate_pairs.parquet`: multiroute candidate union with route names/count/mask.
- `resolver_v3_pair_decisions.jsonl`: selected v3 pair scores and decisions.
- `official1799_multiroute_predictions.xlsx/parquet`: row-level official1799 prediction table.
- `legacy_vs_multiroute_unlabeled_comparison.json`: unlabeled comparison against the frozen legacy official run; not Gold performance.
- `inference_manifest.json`: frozen command, SHA, constraints, route and cluster statistics.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def write_repro_script(out_dir: Path, candidate_mode: str) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd /opt/data/private/JinRong/kechuang/financial_event_project
{RUNTIME_PYTHON} scripts/build_task3_resolver_v3_official1799_multiroute.py --candidate-mode {candidate_mode} --out {rel(out_dir)} --r3-encode-python {DEFAULT_R3_ENCODE_PYTHON} --r3-worker-python {DEFAULT_R3_ANN_PYTHON}
"""
    path = out_dir / "reproduce_official1799_multiroute.sh"
    path.write_text(script, encoding="utf-8")
    os.chmod(path, 0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-mode", choices=["legacy", "multiroute"], default="multiroute")
    parser.add_argument("--out", default=rel(DEFAULT_OUT_DIR))
    parser.add_argument("--r3-worker-python", default=None)
    parser.add_argument("--r3-encode-python", default=None)
    parser.add_argument("--encode-worker", action="store_true")
    parser.add_argument("--encode-input", default="")
    parser.add_argument("--encode-output", default="")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--dev-regression-only", action="store_true")
    args = parser.parse_args()
    if args.encode_worker and (not args.encode_input or not args.encode_output):
        parser.error("--encode-worker requires --encode-input and --encode-output")
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.encode_worker:
        r3_encode_worker(args)
        return
    if args.inventory_only:
        write_json(out_dir / "protected_sha256_before.json", protected_sha(False, out_dir))
        write_json(out_dir / "integration_inventory.json", build_integration_inventory(out_dir))
        write_json(out_dir / "protected_sha256_after.json", protected_sha(True, out_dir))
        print(json.dumps({"status": "inventory_complete", "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))
        return
    if args.dev_regression_only:
        write_json(out_dir / "dev_regression_legacy_vs_multiroute.json", dev_regression(out_dir))
        print(json.dumps({"status": "dev_regression_complete", "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))
        return
    manifest = official_inference(args)
    print(json.dumps(jsonable({
        "status": manifest["status"],
        "output_dir": manifest["output_dir"],
        "candidate_mode": manifest["candidate_mode"],
        "candidate_pairs": manifest["candidate_generation"]["candidate_pairs"],
        "candidate_reduction": manifest["candidate_generation"]["candidate_reduction"],
        "same_event_edges": manifest["legacy_vs_multiroute_unlabeled_comparison"]["multiroute_selected_v3"]["same_event_edges"],
        "cluster_count": manifest["cluster_statistics"]["cluster_count"],
        "max_cluster_size": manifest["cluster_statistics"]["max_cluster_size"],
        "official1799_filled_gold_read_count": 0,
        "final_holdout_read_count": 0,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
