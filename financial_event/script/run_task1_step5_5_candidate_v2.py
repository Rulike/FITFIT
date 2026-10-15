#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5.5: candidate generator v2 and unified Task1 metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.task1_task_metrics import (
    candidate_recall_at_k,
    derive_per_class_thresholds,
    evaluate_event_predictions,
    sweep_global_thresholds,
)
from src.routing.task1_candidate_v2 import Task1CandidateGeneratorV2, percentile, read_jsonl


FORBIDDEN_GOLD_TEST_LABEL_PATHS = {
    "data/processed/labeled/news_test.jsonl",
    "data/processed/labeled/event_pairs_test.jsonl",
    "data/processed/model_data/event_match_official_test.jsonl",
}


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def log(message: str) -> None:
    print(f"[{now()}] {message}", file=sys.stderr, flush=True)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dict(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def xlsx_cell(value: Any, row_idx: int, col_idx: int) -> str:
    ref = f"{xlsx_col_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool) and (not isinstance(value, float) or math.isfinite(value)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = xml_escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def worksheet_xml(rows: list[dict[str, Any]], columns: list[str]) -> str:
    sheet_rows = ['<row r="1">' + "".join(xlsx_cell(col, 1, idx + 1) for idx, col in enumerate(columns)) + "</row>"]
    for row_idx, row in enumerate(rows, 2):
        sheet_rows.append(f'<row r="{row_idx}">' + "".join(xlsx_cell(row.get(col, ""), row_idx, idx + 1) for idx, col in enumerate(columns)) + "</row>")
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(sheet_rows) + "</sheetData></worksheet>"


def write_xlsx(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(sheets.items())
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for idx, _ in enumerate(items, 1):
        content_types.append(f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content_types.append("</Types>")
    workbook_sheets = []
    workbook_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for idx, (sheet, _) in enumerate(items, 1):
        workbook_sheets.append(f'<sheet name="{xml_escape(sheet[:31])}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>')
    workbook_rels.append("</Relationships>")
    workbook_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(workbook_sheets) + "</sheets></workbook>"
    root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for idx, (_, rows) in enumerate(items, 1):
            columns = sorted({key for row in rows for key in row.keys()})
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows, columns))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_forbidden_paths(paths: Iterable[Path]) -> None:
    forbidden = {str((ROOT / path).resolve()) for path in FORBIDDEN_GOLD_TEST_LABEL_PATHS}
    for path in paths:
        if str(path.resolve()) in forbidden:
            raise RuntimeError(f"forbidden gold test label path requested: {path}")


def threshold_grid(config: dict[str, Any]) -> list[float]:
    grid = config["threshold_grid"]
    start = float(grid["start"])
    end = float(grid["end"])
    step = float(grid["step"])
    values = []
    value = start
    while value <= end + 1e-9:
        values.append(round(value, 6))
        value += step
    return values


def protected_file_list() -> list[Path]:
    files: list[Path] = []
    roots = [
        ROOT / "outputs/final_task1",
        ROOT / "outputs/event_clustering/source_fixed",
        ROOT / "outputs/event_credibility/final_cached_dea_v1_260",
        ROOT / "outputs/event_match_baseline/best_model",
    ]
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    event_cluster_dir = ROOT / "data/processed/event_clusters"
    if event_cluster_dir.exists():
        for pattern in ["*event_cluster*", "final_events*", "all_final_events*"]:
            files.extend(path for path in event_cluster_dir.glob(pattern) if path.is_file())
    credibility_dir = ROOT / "outputs/event_credibility"
    if credibility_dir.exists():
        for pattern in ["event_credibility_scores*", "event_timeline_for_task3*"]:
            files.extend(path for path in credibility_dir.glob(pattern) if path.is_file())
    return sorted(set(files), key=lambda item: rel(item))


def write_protected_snapshot(output_dir: Path, before_path: Path | None = None) -> tuple[Path, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = protected_file_list()
    file_list_path = output_dir / "protected_files.txt"
    if not file_list_path.exists():
        file_list_path.write_text("\n".join(rel(path) for path in files) + "\n", encoding="utf-8")
    target = before_path or (output_dir / "protected_sha256_before_full.txt")
    if target.exists():
        return target, load_sha_file(target)
    snapshot = {rel(path): sha256_file(path) for path in files}
    target.write_text("".join(f"{digest}  {path}\n" for path, digest in snapshot.items()), encoding="utf-8")
    return target, snapshot


def load_sha_file(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, file_path = line.split("  ", 1)
            rows[file_path] = digest
    return rows


def write_after_snapshot(output_dir: Path, before: dict[str, str]) -> tuple[Path, dict[str, str], bool]:
    after = {}
    for path_text in sorted(before):
        path = ROOT / path_text
        after[path_text] = sha256_file(path) if path.exists() else "missing"
    after_path = output_dir / "protected_sha256_after_full.txt"
    after_path.write_text("".join(f"{digest}  {path}\n" for path, digest in after.items()), encoding="utf-8")
    return after_path, after, before == after


def recovery_check(output_dir: Path) -> dict[str, Any]:
    expected = [
        ROOT / "src/routing/task1_candidate_v2.py",
        ROOT / "src/evaluation/task1_task_metrics.py",
        ROOT / "configs/task1_candidate_v2.yaml",
        ROOT / "scripts/run_task1_step5_5_candidate_v2.py",
        ROOT / "scripts/evaluate_task1_step5_5_candidate_v2.py",
        ROOT / "scripts/test_task1_step5_5_candidate_v2.py",
        ROOT / "reports/task1_step5_5_candidate_v2_report.json",
        ROOT / "reports/task1_step5_5_candidate_v2_report.md",
        ROOT / "reports/task1_step5_5_candidate_v2_details.xlsx",
    ]
    existing_expected = [
        {"path": rel(path), "size_bytes": path.stat().st_size, "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat()}
        for path in expected
        if path.exists()
    ]
    output_files = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                output_files.append(
                    {
                        "path": rel(path),
                        "size_bytes": path.stat().st_size,
                        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    }
                )
    processes = []
    try:
        result = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], check=False, text=True, capture_output=True)
        process_rows = []
        ppid_by_pid = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split(maxsplit=2)
            if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
                continue
            pid = int(parts[0])
            ppid = int(parts[1])
            args = parts[2]
            process_rows.append((pid, ppid, args))
            ppid_by_pid[pid] = ppid
        skip_pids = {os.getpid()}
        parent = ppid_by_pid.get(os.getpid())
        while parent and parent not in skip_pids:
            skip_pids.add(parent)
            parent = ppid_by_pid.get(parent)
        for pid, _ppid, args in process_rows:
            if "task1_step5_5_candidate_v2" in args or "run_task1_step5_5" in args:
                if pid not in skip_pids:
                    processes.append(f"{pid} {args}")
    except OSError:
        processes = []
    completed_markers = []
    for marker in ["candidate_recall_summary.json", "pair_classifier_all64_predictions.jsonl", "new_classifier_training_summary.json", "manifest.json"]:
        if (output_dir / marker).exists():
            completed_markers.append(marker)
    return {
        "checked_at": now(),
        "output_dir": rel(output_dir),
        "existing_step5_5_expected_files": existing_expected,
        "existing_output_files": output_files,
        "completed_markers": completed_markers,
        "unfinished_parts": [
            name
            for name in [
                "candidate_generation",
                "pair_classifier_scoring",
                "new_classifier_training",
                "reporting",
            ]
            if {
                "candidate_generation": "candidate_recall_summary.json",
                "pair_classifier_scoring": "pair_classifier_all64_predictions.jsonl",
                "new_classifier_training": "new_classifier_training_summary.json",
                "reporting": "manifest.json",
            }[name]
            not in completed_markers
        ],
        "running_processes": processes,
        "reused_partial_outputs": bool(completed_markers),
    }


def load_frozen_split(config: dict[str, Any]) -> dict[str, Any]:
    step5_4_dir = ROOT / config["step5_4_dir"]
    train_id_path = step5_4_dir / "development_train_ids.jsonl"
    dev_id_path = step5_4_dir / "development_validation_ids.jsonl"
    news_path = ROOT / config["news_train_path"]
    pair_path = ROOT / config["event_pairs_train_path"]
    assert_no_forbidden_paths([train_id_path, dev_id_path, news_path, pair_path])
    train_ids = [str(row["article_id"]) for row in read_jsonl(train_id_path)]
    dev_ids = [str(row["article_id"]) for row in read_jsonl(dev_id_path)]
    articles = read_jsonl(news_path)
    pairs = read_jsonl(pair_path)
    article_by_id = {str(row["article_id"]): row for row in articles}
    train_set = set(train_ids)
    dev_set = set(dev_ids)
    train_articles = [article_by_id[aid] for aid in train_ids]
    dev_articles = [article_by_id[aid] for aid in dev_ids]
    train_pairs = [row for row in pairs if str(row["article_id"]) in train_set]
    dev_pairs = [row for row in pairs if str(row["article_id"]) in dev_set]
    checks = {
        "train_article_count": len(train_ids),
        "dev_article_count": len(dev_ids),
        "train_event_pair_count": len(train_pairs),
        "dev_event_pair_count": len(dev_pairs),
        "article_id_overlap": len(train_set & dev_set),
    }
    expected = {
        "train_article_count": 1637,
        "dev_article_count": 409,
        "train_event_pair_count": 1665,
        "dev_event_pair_count": 416,
        "article_id_overlap": 0,
    }
    checks["matches_step5_4_expected"] = all(checks[key] == value for key, value in expected.items())
    if not checks["matches_step5_4_expected"]:
        raise RuntimeError(f"frozen split changed or unexpected: {checks}")
    return {
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "article_by_id": article_by_id,
        "train_articles": train_articles,
        "dev_articles": dev_articles,
        "train_pairs": train_pairs,
        "dev_pairs": dev_pairs,
        "checks": checks,
    }


def slim_candidate(row: dict[str, Any], keep_probability: bool = False) -> dict[str, Any]:
    out = {
        "article_id": row.get("article_id", ""),
        "title": row.get("title", ""),
        "event_name": row.get("event_name", ""),
        "schema_id": row.get("schema_id", ""),
        "target_type": row.get("target_type", ""),
        "target_code": row.get("target_code", ""),
        "target_name": row.get("target_name", ""),
        "candidate_rank": row.get("candidate_rank", ""),
        "candidate_score": round(float(row.get("candidate_score", 0.0) or 0.0), 8),
        "candidate_source": row.get("candidate_source", ""),
        "source_scores": row.get("source_scores", {}),
        "train_support": row.get("train_support", 0),
        "generation_time_ms": row.get("generation_time_ms", 0.0),
    }
    if keep_probability:
        out["probability_positive"] = row.get("probability_positive", "")
    return out


def candidate_rows_for_top_k(rows: list[dict[str, Any]], top_k: int | None) -> list[dict[str, Any]]:
    if top_k is None:
        return list(rows)
    return [row for row in rows if int(row.get("candidate_rank", 10**9) or 10**9) <= top_k]


def source_contribution(
    article_ids: list[str],
    gold_pairs: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    source_order: list[str],
) -> list[dict[str, Any]]:
    gold_by_article: dict[str, set[str]] = defaultdict(set)
    for row in gold_pairs:
        gold_by_article[str(row["article_id"])].add(str(row["event_name"]))
    total = sum(len(gold_by_article.get(aid, set())) for aid in article_ids)
    by_source: dict[str, dict[str, set[str]]] = {source: defaultdict(set) for source in source_order}
    for row in candidate_rows:
        aid = str(row.get("article_id", ""))
        event_name = str(row.get("event_name", ""))
        scores = row.get("source_scores", {}) or {}
        for source in source_order:
            if float(scores.get(source, 0.0) or 0.0) > 0.0:
                by_source[source][aid].add(event_name)
    covered: dict[str, set[str]] = defaultdict(set)
    rows = []
    for source in source_order:
        independent_hits = 0
        new_hits = 0
        candidate_count = sum(len(events) for events in by_source[source].values())
        for aid in article_ids:
            gold_set = gold_by_article.get(aid, set())
            source_hits = gold_set & by_source[source].get(aid, set())
            independent_hits += len(source_hits)
            newly = source_hits - covered[aid]
            new_hits += len(newly)
            covered[aid].update(source_hits)
        rows.append(
            {
                "source": source,
                "candidate_count": candidate_count,
                "independent_hit_count": independent_hits,
                "independent_recall": round(independent_hits / total, 6) if total else 0.0,
                "new_hit_count_in_source_order": new_hits,
                "new_recall_in_source_order": round(new_hits / total, 6) if total else 0.0,
                "cumulative_hit_count": sum(len(gold_by_article.get(aid, set()) & covered.get(aid, set())) for aid in article_ids),
                "cumulative_recall": round(
                    sum(len(gold_by_article.get(aid, set()) & covered.get(aid, set())) for aid in article_ids) / total,
                    6,
                )
                if total
                else 0.0,
            }
        )
    return rows


def timing_summary(timing_rows: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    counts = [float(row["candidate_count"]) for row in timing_rows]
    times = [float(row["generation_time_ms"]) for row in timing_rows]
    summary = {
        "raw_all64_candidate_count_avg": float(np.mean(counts)) if counts else 0.0,
        "raw_all64_candidate_count_p95": percentile(counts, 0.95),
        "generation_time_ms_avg": float(np.mean(times)) if times else 0.0,
        "generation_time_ms_p95": percentile(times, 0.95),
    }
    for top_k in top_ks:
        summary[f"production_top_{top_k}_candidate_count_avg"] = float(top_k)
        summary[f"production_top_{top_k}_candidate_count_p95"] = float(top_k)
    return summary


def long_tail_candidate_recall(class_candidate_rows: list[dict[str, Any]], generator: Task1CandidateGeneratorV2) -> dict[str, Any]:
    threshold = int(generator.config.get("long_tail_support_threshold", 15))
    tail_names = {name for name in generator.event_names if int(generator.train_support.get(name, 0)) <= threshold}
    rows = [row for row in class_candidate_rows if row.get("event_name") in tail_names]
    gold = sum(int(row.get("gold_support", 0) or 0) for row in rows)
    out = {
        "long_tail_train_support_threshold": threshold,
        "long_tail_event_name_count": len(tail_names),
        "long_tail_dev_gold_support": gold,
    }
    for key in ["candidate_hits_at_10", "candidate_hits_at_20", "candidate_hits_at_32", "candidate_hits_at_64"]:
        hits = sum(int(row.get(key, 0) or 0) for row in rows)
        out[key.replace("hits", "recall")] = hits / gold if gold else 0.0
        out[key] = hits
    return out


def load_or_generate_candidates(
    output_dir: Path,
    config: dict[str, Any],
    split: dict[str, Any],
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Task1CandidateGeneratorV2]:
    schema_rows = read_jsonl(ROOT / config["schema_path"])
    alias_path = ROOT / config["target_alias_path"]
    generator = Task1CandidateGeneratorV2.from_paths(
        schema_path=ROOT / config["schema_path"],
        train_articles=split["train_articles"],
        train_gold_pairs=split["train_pairs"],
        alias_path=alias_path,
        config=config.get("candidate_generator", {}),
    )
    candidate_path = output_dir / "candidate_v2_dev_all64.jsonl"
    timing_path = output_dir / "candidate_generation_timing.csv"
    profile_path = output_dir / "event_schema_profiles_v2.csv"
    if resume and candidate_path.exists() and timing_path.exists():
        return read_jsonl(candidate_path), read_csv_dict(timing_path), generator
    candidate_rows, timing_rows = generator.generate_many(split["dev_articles"], include_all64=True)
    write_jsonl(candidate_path, [slim_candidate(row) for row in candidate_rows])
    write_csv(
        timing_path,
        timing_rows,
        ["article_id", "candidate_count", "generation_time_ms"],
    )
    write_csv(profile_path, generator.schema_profile_rows())
    return [slim_candidate(row) for row in candidate_rows], timing_rows, generator


def score_pair_classifier(
    candidate_rows: list[dict[str, Any]],
    article_by_id: dict[str, dict[str, Any]],
    model_dir: Path,
    output_path: Path,
    batch_size: int,
    max_length: int,
    resume: bool,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    if resume and output_path.exists():
        return read_jsonl(output_path), {"status": "reused", "path": rel(output_path)}
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.data.event_match_text import build_input_text
    except Exception as exc:  # pragma: no cover - exercised in base env if ML deps are absent
        return None, {"status": "dependency_missing", "error": repr(exc)}

    class PairDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = dict(self.records[idx])
            article = article_by_id[str(row["article_id"])]
            row["content"] = article.get("content", "")
            row["title"] = article.get("title", row.get("title", ""))
            return row

    log(f"loading frozen pair tokenizer: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    log(f"loading frozen pair model: {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    log(f"scoring frozen pair classifier records={len(candidate_rows)} device={device}")

    def collate(batch: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        texts = []
        for row in batch:
            content = str(row.get("content", ""))
            if len(content) > 2600:
                content = content[:1900] + "\n[正文中间省略]\n" + content[-600:]
            item = dict(row)
            item["content"] = content
            texts.append(build_input_text(item))
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        return enc, batch

    scored: list[dict[str, Any]] = []
    start = time.time()
    loader = DataLoader(PairDataset(candidate_rows), batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for enc, batch in loader:
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits
            if logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits).reshape(-1)
            for row, prob in zip(batch, probs.detach().cpu().tolist()):
                item = slim_candidate(row, keep_probability=False)
                item["probability_positive"] = float(prob)
                scored.append(item)
            if len(scored) % max(batch_size * 20, 1) == 0:
                log(f"frozen pair scored {len(scored)}/{len(candidate_rows)}")
    write_jsonl(output_path, scored)
    log(f"frozen pair predictions written: {output_path}")
    return scored, {
        "status": "completed",
        "path": rel(output_path),
        "record_count": len(scored),
        "runtime_seconds": round(time.time() - start, 3),
        "device": str(device),
    }


def event_rows_from_multilabel_scores(
    article_ids: list[str],
    event_names: list[str],
    scores: np.ndarray,
    candidate_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    meta_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if candidate_rows is not None:
        for row in candidate_rows:
            meta_by_key[(str(row["article_id"]), str(row["event_name"]))] = row
    rows = []
    for i, aid in enumerate(article_ids):
        ranked_events = sorted(
            [(event_name, float(scores[i, j])) for j, event_name in enumerate(event_names)],
            key=lambda item: (-item[1], item[0]),
        )
        score_rank = {event_name: rank for rank, (event_name, _) in enumerate(ranked_events, 1)}
        for j, event_name in enumerate(event_names):
            meta = meta_by_key.get((aid, event_name), {})
            rows.append(
                {
                    "article_id": aid,
                    "event_name": event_name,
                    "schema_id": meta.get("schema_id", f"event::{event_name}"),
                    "target_type": meta.get("target_type", ""),
                    "target_code": meta.get("target_code", ""),
                    "target_name": meta.get("target_name", ""),
                    "candidate_rank": meta.get("candidate_rank", score_rank[event_name]),
                    "candidate_score": meta.get("candidate_score", float(scores[i, j])),
                    "candidate_source": meta.get("candidate_source", "all64_model_output"),
                    "probability_positive": float(scores[i, j]),
                }
            )
    return rows


def make_article_text(article: dict[str, Any]) -> str:
    return f"[新闻]\n标题：{article.get('title', '')}\n正文：{article.get('content', '')}"


def train_new_multilabel_classifier(
    output_dir: Path,
    config: dict[str, Any],
    split: dict[str, Any],
    event_names: list[str],
    candidate_rows: list[dict[str, Any]],
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    summary_path = output_dir / "new_classifier_training_summary.json"
    score_path = output_dir / "new_classifier_all64_predictions.jsonl"
    if resume and summary_path.exists() and score_path.exists():
        return read_json(summary_path), read_jsonl(score_path)
    train_cfg = config.get("training", {})
    if not train_cfg.get("enabled", True):
        summary = {"status": "disabled_by_config"}
        write_json(summary_path, summary)
        return summary, None
    try:
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        summary = {"status": "dependency_missing", "error": repr(exc)}
        write_json(summary_path, summary)
        return summary, None

    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    event_to_idx = {name: idx for idx, name in enumerate(event_names)}
    train_labels_by_article: dict[str, set[str]] = defaultdict(set)
    dev_labels_by_article: dict[str, set[str]] = defaultdict(set)
    for row in split["train_pairs"]:
        train_labels_by_article[str(row["article_id"])].add(str(row["event_name"]))
    for row in split["dev_pairs"]:
        dev_labels_by_article[str(row["article_id"])].add(str(row["event_name"]))
    train_support = Counter(str(row["event_name"]) for row in split["train_pairs"])
    train_label_matrix = []
    for article in split["train_articles"]:
        values = np.zeros(len(event_names), dtype=np.float32)
        for event_name in train_labels_by_article[str(article["article_id"])]:
            values[event_to_idx[event_name]] = 1.0
        train_label_matrix.append(values)
    dev_label_matrix = []
    for article in split["dev_articles"]:
        values = np.zeros(len(event_names), dtype=np.float32)
        for event_name in dev_labels_by_article[str(article["article_id"])]:
            values[event_to_idx[event_name]] = 1.0
        dev_label_matrix.append(values)
    train_label_matrix_np = np.stack(train_label_matrix)
    pos_counts = train_label_matrix_np.sum(axis=0)
    neg_counts = len(train_label_matrix_np) - pos_counts
    pos_weight_np = np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 20.0).astype(np.float32)

    model_path = ROOT / train_cfg["model_name_or_path"]
    max_length = int(train_cfg.get("max_length", 512))
    batch_size = int(train_cfg.get("batch_size", 8))
    epochs = int(train_cfg.get("epochs", 2))
    lr = float(train_cfg.get("learning_rate", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grid = threshold_grid(config)

    class ArticleDataset(Dataset):
        def __init__(self, articles: list[dict[str, Any]], labels: list[np.ndarray], tokenizer: Any) -> None:
            self.articles = articles
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self) -> int:
            return len(self.articles)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return {"text": make_article_text(self.articles[idx]), "labels": self.labels[idx]}

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        enc = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(np.stack([row["labels"] for row in batch]), dtype=torch.float32)
        return enc

    def predict(model: Any, tokenizer: Any, articles: list[dict[str, Any]]) -> np.ndarray:
        model.eval()
        dummy = [np.zeros(len(event_names), dtype=np.float32) for _ in articles]
        loader = DataLoader(ArticleDataset(articles, dummy, tokenizer), batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
        chunks = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                del labels
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(**batch).logits
                chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(event_names)), dtype=np.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    best_global: dict[str, Any] | None = None
    best_scores: np.ndarray | None = None
    model_root = output_dir / "new_classifier_models"
    loss_names = list(train_cfg.get("losses", ["bce"]))
    start_all = time.time()
    for loss_name in loss_names:
        log(f"training new multilabel classifier loss={loss_name}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=len(event_names),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        model.config.id2label = {idx: name for idx, name in enumerate(event_names)}
        model.config.label2id = {name: idx for idx, name in enumerate(event_names)}
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
        train_loader = DataLoader(
            ArticleDataset(split["train_articles"], train_label_matrix, tokenizer),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        for epoch in range(1, epochs + 1):
            log(f"train epoch start loss={loss_name} epoch={epoch}")
            model.train()
            losses = []
            for batch in train_loader:
                labels = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                logits = model(**batch).logits
                if loss_name == "class_balanced_bce":
                    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                elif loss_name == "focal":
                    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                    prob = torch.sigmoid(logits)
                    p_t = prob * labels + (1 - prob) * (1 - labels)
                    alpha_t = 0.25 * labels + 0.75 * (1 - labels)
                    loss = (alpha_t * (1 - p_t).pow(2.0) * bce).mean()
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scores = predict(model, tokenizer, split["dev_articles"])
            log(f"dev prediction complete loss={loss_name} epoch={epoch}")
            pred_rows = event_rows_from_multilabel_scores(split["dev_ids"], event_names, scores, candidate_rows)
            sweep_rows, best = sweep_global_thresholds(
                split["dev_ids"],
                split["dev_pairs"],
                pred_rows,
                event_names,
                grid,
                top_k=None,
                primary=train_cfg.get("selection_primary_metric", "macro_f1_fixed_64"),
                secondary=train_cfg.get("selection_tie_metric", "micro_f1"),
            )
            history_row = {
                "loss_name": loss_name,
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)), 6) if losses else 0.0,
                "selection_threshold": best.get("threshold"),
                "macro_f1_fixed_64": best.get("macro_f1_fixed_64", 0.0),
                "micro_f1": best.get("micro_f1", 0.0),
                "micro_precision": best.get("micro_precision", 0.0),
                "micro_recall": best.get("micro_recall", 0.0),
                "article_exact_accuracy": best.get("article_exact_accuracy", 0.0),
            }
            results.append(history_row)
            key = (
                float(best.get(train_cfg.get("selection_primary_metric", "macro_f1_fixed_64"), 0.0)),
                float(best.get(train_cfg.get("selection_tie_metric", "micro_f1"), 0.0)),
                float(best.get("micro_recall", 0.0)),
            )
            if best_global is None or key > best_global["selection_key"]:
                candidate_model_dir = model_root / f"{loss_name}_epoch_{epoch}"
                best_global = {
                    "selection_key": key,
                    "loss_name": loss_name,
                    "epoch": epoch,
                    "best_model_dir": rel(candidate_model_dir),
                    "global_threshold": float(best.get("threshold", config.get("default_threshold", 0.5))),
                    "global_threshold_metrics": best,
                    "sweep_rows": sweep_rows,
                }
                best_scores = scores.copy()
                if not candidate_model_dir.exists():
                    model.save_pretrained(str(candidate_model_dir))
                    tokenizer.save_pretrained(str(candidate_model_dir))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_scores is None or best_global is None:
        summary = {"status": "no_model_trained"}
        write_json(summary_path, summary)
        return summary, None

    best_pred_rows = event_rows_from_multilabel_scores(split["dev_ids"], event_names, best_scores, candidate_rows)
    class_thresholds = derive_per_class_thresholds(
        split["dev_ids"],
        split["dev_pairs"],
        best_pred_rows,
        event_names,
        grid,
        dict(train_support),
        int(train_cfg.get("class_threshold_min_support", 5)),
        float(best_global["global_threshold"]),
    )
    per_class_metrics = evaluate_event_predictions(
        split["dev_ids"],
        split["dev_pairs"],
        best_pred_rows,
        event_names,
        threshold=class_thresholds,
        top_k=None,
    )
    threshold_config = {
        "source": "dev_calibration_only",
        "selection_metric": {
            "primary": train_cfg.get("selection_primary_metric", "macro_f1_fixed_64"),
            "tie": train_cfg.get("selection_tie_metric", "micro_f1"),
        },
        "global_threshold": best_global["global_threshold"],
        "per_class_thresholds_min_support": int(train_cfg.get("class_threshold_min_support", 5)),
        "per_class_thresholds": class_thresholds,
        "per_class_threshold_metrics": {
            key: per_class_metrics[key]
            for key in [
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "macro_f1_fixed_64",
                "macro_f1_observed",
                "weighted_f1",
                "article_exact_accuracy",
                "at_least_one_correct_article_ratio",
            ]
        },
    }
    write_json(output_dir / "new_classifier_threshold_config.json", threshold_config)
    write_jsonl(score_path, best_pred_rows)
    write_csv(output_dir / "new_classifier_training_history.csv", results)
    write_csv(output_dir / "new_classifier_threshold_sweep.csv", best_global["sweep_rows"])
    summary = {
        "status": "completed",
        "runtime_seconds": round(time.time() - start_all, 3),
        "device": str(device),
        "losses_compared": loss_names,
        "selection_rule": "primary fixed-64 macro-F1, tie micro-F1, tie recall",
        "best_loss": best_global["loss_name"],
        "best_epoch": best_global["epoch"],
        "best_model_dir": best_global["best_model_dir"],
        "global_threshold": best_global["global_threshold"],
        "global_threshold_metrics": best_global["global_threshold_metrics"],
        "per_class_threshold_metrics": threshold_config["per_class_threshold_metrics"],
        "train_support": dict(train_support),
    }
    write_json(summary_path, summary)
    log(f"new classifier training summary written: {summary_path}")
    return summary, best_pred_rows


def method_eval_row(
    method_name: str,
    classifier: str,
    candidate_mode: str,
    threshold_strategy: str,
    metrics: dict[str, Any],
    threshold: Any,
    candidate_recall: dict[str, Any] | None,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "method": method_name,
        "classifier": classifier,
        "candidate_mode": candidate_mode,
        "threshold_strategy": threshold_strategy,
        "threshold": json.dumps(threshold, ensure_ascii=False) if isinstance(threshold, dict) else threshold,
        "micro_precision": round(metrics.get("micro_precision", 0.0), 6),
        "micro_recall": round(metrics.get("micro_recall", 0.0), 6),
        "micro_f1": round(metrics.get("micro_f1", 0.0), 6),
        "macro_f1_fixed_64": round(metrics.get("macro_f1_fixed_64", 0.0), 6),
        "macro_f1_observed": round(metrics.get("macro_f1_observed", 0.0), 6),
        "weighted_f1": round(metrics.get("weighted_f1", 0.0), 6),
        "article_exact_accuracy": round(metrics.get("article_exact_accuracy", 0.0), 6),
        "at_least_one_correct_article_ratio": round(metrics.get("at_least_one_correct_article_ratio", 0.0), 6),
        "tp": metrics.get("tp", ""),
        "fp": metrics.get("fp", ""),
        "fn": metrics.get("fn", ""),
        "candidate_recall_at_20": round(candidate_recall.get("candidate_recall_at_20", 0.0), 6) if candidate_recall else "",
        "candidate_recall_at_32": round(candidate_recall.get("candidate_recall_at_32", 0.0), 6) if candidate_recall else "",
        "runtime_seconds": runtime_seconds if runtime_seconds is not None else "",
    }


def evaluate_prediction_methods(
    config: dict[str, Any],
    split: dict[str, Any],
    event_names: list[str],
    candidate_rows: list[dict[str, Any]],
    pair_scores: list[dict[str, Any]] | None,
    new_scores: list[dict[str, Any]] | None,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grid = threshold_grid(config)
    default_threshold = float(config.get("default_threshold", 0.38))
    candidate_recall_all = candidate_recall_at_k(
        split["dev_ids"],
        split["dev_pairs"],
        candidate_rows,
        event_names,
        config["candidate_recall_ks"],
    )
    candidate_by_topk = {
        top_k: candidate_recall_at_k(
            split["dev_ids"],
            split["dev_pairs"],
            candidate_rows_for_top_k(candidate_rows, top_k),
            event_names,
            config["candidate_recall_ks"],
        )
        for top_k in config["candidate_top_ks"]
    }
    method_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    step54_report = read_json(ROOT / config["step5_4_dir"] / "step5_4_report.json")
    step54_metrics = step54_report["development_validation_metrics_at_current_threshold"]
    method_rows.append(
        {
            "method": "Step 5.4 original baseline",
            "classifier": "step5_4_pair_classifier",
            "candidate_mode": "original_target_router",
            "threshold_strategy": "reported_current_threshold",
            "threshold": step54_metrics["threshold"],
            "micro_precision": round(step54_metrics["micro_precision"], 6),
            "micro_recall": round(step54_metrics["micro_recall"], 6),
            "micro_f1": round(step54_metrics["micro_f1"], 6),
            "macro_f1_fixed_64": round(step54_metrics["macro_f1_fixed_64"], 6),
            "macro_f1_observed": round(step54_metrics["macro_f1_observed"], 6),
            "weighted_f1": round(step54_metrics["weighted_f1"], 6),
            "article_exact_accuracy": round(step54_metrics["article_exact_accuracy"], 6),
            "at_least_one_correct_article_ratio": "",
            "tp": step54_metrics["tp"],
            "fp": step54_metrics["fp"],
            "fn": step54_metrics["fn"],
            "candidate_recall_at_20": "",
            "candidate_recall_at_32": "",
            "runtime_seconds": "",
        }
    )

    if pair_scores is not None:
        for mode_name, top_k in [("all-64", None), ("candidate-v2 Top-10", 10), ("candidate-v2 Top-20", 20), ("candidate-v2 Top-32", 32)]:
            rows = candidate_rows_for_top_k(pair_scores, top_k)
            sweep, best = sweep_global_thresholds(split["dev_ids"], split["dev_pairs"], rows, event_names, grid, top_k=None)
            metrics_default = evaluate_event_predictions(split["dev_ids"], split["dev_pairs"], rows, event_names, default_threshold)
            details[f"frozen_pair::{mode_name}"] = {"best_global": best, "default_threshold": metrics_default}
            rec = candidate_recall_all if top_k is None else candidate_by_topk[top_k]
            method_rows.append(method_eval_row(f"Original classifier + {mode_name}", "step5_4_pair_classifier", mode_name, "global_best_fixed64_on_dev", evaluate_event_predictions(split["dev_ids"], split["dev_pairs"], rows, event_names, float(best["threshold"])), best["threshold"], rec))
            if mode_name == "all-64":
                details["all64_existing_classifier_true_upper"] = best
    if new_scores is not None:
        training_summary = read_json(output_dir / "new_classifier_training_summary.json")
        global_threshold = float(training_summary.get("global_threshold", default_threshold))
        for mode_name, top_k in [("all-64", None), ("candidate-v2 Top-20", 20)]:
            rows = candidate_rows_for_top_k(new_scores, top_k)
            metrics = evaluate_event_predictions(split["dev_ids"], split["dev_pairs"], rows, event_names, global_threshold)
            rec = candidate_recall_all if top_k is None else candidate_by_topk[top_k]
            method_rows.append(method_eval_row(f"New classifier + {mode_name}", "new_multilabel_macbert", mode_name, "best_global_threshold_from_dev", metrics, global_threshold, rec))
        threshold_config_path = output_dir / "new_classifier_threshold_config.json"
        if threshold_config_path.exists():
            threshold_config = read_json(threshold_config_path)
            per_class = threshold_config.get("per_class_thresholds", {})
            rows = candidate_rows_for_top_k(new_scores, 20)
            metrics = evaluate_event_predictions(split["dev_ids"], split["dev_pairs"], rows, event_names, per_class)
            method_rows.append(method_eval_row("New classifier + candidate-v2 Top-20", "new_multilabel_macbert", "candidate-v2 Top-20", "per_class_min_support_thresholds", metrics, per_class, candidate_by_topk[20]))
            details["new_classifier_per_class_threshold_config"] = threshold_config

    fn_rows = []
    best_method = max(method_rows, key=lambda row: (float(row.get("macro_f1_fixed_64") or 0.0), float(row.get("micro_f1") or 0.0)))
    if new_scores is not None and "New classifier" in best_method["method"]:
        source_rows = candidate_rows_for_top_k(new_scores, 20 if "Top-20" in best_method["candidate_mode"] else None)
        threshold: float | dict[str, float] = float(best_method["threshold"]) if best_method["threshold_strategy"] != "per_class_min_support_thresholds" else read_json(output_dir / "new_classifier_threshold_config.json")["per_class_thresholds"]
    elif pair_scores is not None:
        source_rows = candidate_rows_for_top_k(pair_scores, 20)
        threshold = float(best_method["threshold"]) if str(best_method["threshold"]).replace(".", "", 1).isdigit() else default_threshold
    else:
        source_rows = candidate_rows_for_top_k(candidate_rows, 20)
        threshold = default_threshold
    by_article_event = {(str(row["article_id"]), str(row["event_name"])): row for row in source_rows}
    for gold in split["dev_pairs"]:
        key = (str(gold["article_id"]), str(gold["event_name"]))
        row = by_article_event.get(key)
        if row is None:
            reason = "candidate_missing"
            prob = ""
            rank = ""
        else:
            prob_val = float(row.get("probability_positive", row.get("candidate_score", 0.0)) or 0.0)
            th = float(threshold.get(key[1], threshold.get("__default__", default_threshold))) if isinstance(threshold, dict) else float(threshold)
            reason = "true_positive" if prob_val >= th else "candidate_present_classifier_below_threshold"
            prob = round(prob_val, 6)
            rank = row.get("candidate_rank", "")
        fn_rows.append(
            {
                "article_id": key[0],
                "event_name": key[1],
                "candidate_rank": rank,
                "probability_positive": prob,
                "fn_reason": reason,
            }
        )
    return method_rows, details, candidate_recall_all["class_rows"], fn_rows


def write_reports(
    config: dict[str, Any],
    output_dir: Path,
    recovery: dict[str, Any],
    split: dict[str, Any],
    generator: Task1CandidateGeneratorV2,
    candidate_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    candidate_metrics: dict[str, Any],
    source_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    method_details: dict[str, Any],
    class_candidate_rows: list[dict[str, Any]],
    fn_rows: list[dict[str, Any]],
    pair_inference_summary: dict[str, Any],
    training_summary: dict[str, Any],
    protected_before: dict[str, str],
    protected_after: dict[str, str],
    protected_unchanged: bool,
) -> dict[str, Any]:
    timing = timing_summary(timing_rows, config["candidate_top_ks"])
    long_tail_recall = long_tail_candidate_recall(class_candidate_rows, generator)
    acceptance_cfg = config["candidate_acceptance"]
    candidate_gate = {
        "candidate_recall_at_20": candidate_metrics["candidate_recall_at_20"],
        "candidate_recall_at_32": candidate_metrics["candidate_recall_at_32"],
        "candidate_recall_at_64": candidate_metrics["candidate_recall_at_64"],
        "candidate_missing_at_32": int(round((1.0 - candidate_metrics["candidate_recall_at_32"]) * candidate_metrics["gold_event_count"])),
    }
    candidate_gate["passed"] = (
        candidate_gate["candidate_recall_at_20"] >= acceptance_cfg["candidate_recall_at_20"]
        and candidate_gate["candidate_recall_at_32"] >= acceptance_cfg["candidate_recall_at_32"]
        and candidate_gate["candidate_recall_at_64"] >= acceptance_cfg["all64_recall"]
        and candidate_gate["candidate_missing_at_32"] <= acceptance_cfg["candidate_missing_at_32_max"]
    )
    best_dev = max(method_rows, key=lambda row: (float(row.get("macro_f1_fixed_64") or 0.0), float(row.get("micro_f1") or 0.0)))
    final_gate = {
        "micro_recall_ge_0_65": float(best_dev.get("micro_recall") or 0.0) >= 0.65,
        "micro_f1_ge_0_62": float(best_dev.get("micro_f1") or 0.0) >= 0.62,
        "fixed_64_macro_f1_ge_0_45": float(best_dev.get("macro_f1_fixed_64") or 0.0) >= 0.45,
        "article_exact_accuracy_ge_0_40": float(best_dev.get("article_exact_accuracy") or 0.0) >= 0.40,
        "candidate_recall_at_20_ge_0_90": candidate_gate["candidate_recall_at_20"] >= 0.90,
        "micro_f1_improvement_ge_0_10": float(best_dev.get("micro_f1") or 0.0) - 0.493506 >= 0.10,
        "precision_ge_0_60_recommended": float(best_dev.get("micro_precision") or 0.0) >= 0.60,
    }
    final_gate["passed_for_1029_blind_test"] = all(final_gate.values())
    report = {
        "generated_at": now(),
        "step": "Task1 Step 5.5 candidate_v2_dev",
        "output_dir": rel(output_dir),
        "recovery_check": recovery,
        "gold_test_usage": {
            "read_1029_gold_test_labels": False,
            "rerun_1029_gold_test": False,
            "forbidden_paths": sorted(FORBIDDEN_GOLD_TEST_LABEL_PATHS),
        },
        "llm_calls": {"dianjin_r1_32b_calls": 0, "external_llm_calls": 0},
        "split_checks": split["checks"],
        "metric_unification": {
            "trainer_eval_macro_f1_definition": "candidate-pair binary macro-F1 = average(F1_negative_class, F1_positive_class) over routed candidate rows",
            "task_fixed_64_macro_f1_definition": "article-level multi-label event_name macro-F1 averaged over all 64 fixed event classes",
            "checkpoint_849_selection": "checkpoint-849 was selected by Trainer metric_for_best_model=macro_f1, i.e. pair-level binary macro-F1, not article-level fixed-64 macro-F1",
            "future_selection_rule": "primary fixed-64 macro-F1, tie micro-F1 on the frozen 409-article dev split",
        },
        "candidate_v2": {
            "event_name_count": len(generator.event_names),
            "candidate_recall": {key: value for key, value in candidate_metrics.items() if key.startswith("candidate_recall_at_") or key == "gold_event_count"},
            "candidate_missing_remaining_at_32": candidate_gate["candidate_missing_at_32"],
            "candidate_gate": candidate_gate,
            "source_contribution": source_rows,
            "long_tail_candidate_recall": long_tail_recall,
            "timing_summary": timing,
        },
        "pair_inference_summary": pair_inference_summary,
        "training_summary": training_summary,
        "method_summary": method_rows,
        "method_details": method_details,
        "best_dev_method": best_dev,
        "final_dev_gate": final_gate,
        "formal_file_sha256_before": protected_before,
        "formal_file_sha256_after": protected_after,
        "formal_file_sha256_unchanged": protected_unchanged,
        "protected_file_count": len(protected_before),
    }
    write_json(output_dir / "candidate_recall_summary.json", candidate_metrics)
    write_json(output_dir / "step5_5_report.json", report)
    write_json(ROOT / config["report_json"], report)
    class_final_rows = []
    best_score_path = output_dir / "new_classifier_all64_predictions.jsonl"
    if best_score_path.exists():
        try:
            threshold = float(best_dev["threshold"]) if best_dev["threshold_strategy"] != "per_class_min_support_thresholds" else read_json(output_dir / "new_classifier_threshold_config.json")["per_class_thresholds"]
            metrics = evaluate_event_predictions(split["dev_ids"], split["dev_pairs"], read_jsonl(best_score_path), generator.event_names, threshold, top_k=None)
            class_final_rows = metrics["class_rows"]
        except Exception:
            class_final_rows = []
    confusion_rows = []
    if best_score_path.exists() and class_final_rows:
        metrics = evaluate_event_predictions(split["dev_ids"], split["dev_pairs"], read_jsonl(best_score_path), generator.event_names, float(report["training_summary"].get("global_threshold", config.get("default_threshold", 0.38))), top_k=None)
        labels = sorted(generator.event_names) + ["__NO_PRED__"]
        for gold_name in sorted(generator.event_names):
            item = {"gold_event_name": gold_name}
            for label in labels:
                item[label] = int(metrics["confusion"].get(gold_name, {}).get(label, 0))
            confusion_rows.append(item)
    threshold_rows = []
    threshold_config_path = output_dir / "new_classifier_threshold_config.json"
    if threshold_config_path.exists():
        threshold_config = read_json(threshold_config_path)
        for event_name, threshold in threshold_config.get("per_class_thresholds", {}).items():
            threshold_rows.append({"event_name": event_name, "threshold": threshold, "source": threshold_config.get("source", "")})
    xlsx_path = ROOT / config["report_xlsx"]
    write_xlsx(
        xlsx_path,
        {
            "方法总表": method_rows,
            "候选来源贡献": source_rows,
            "64类候选召回": class_candidate_rows,
            "64类最终指标": class_final_rows,
            "混淆矩阵": confusion_rows,
            "FN原因": fn_rows,
            "多标签文章结果": [row for row in evaluate_event_predictions(split["dev_ids"], split["dev_pairs"], [], generator.event_names, 1.0)["article_rows"] if row["gold_count"] > 1],
            "阈值配置": threshold_rows,
            "耗时统计": timing_rows,
        },
    )
    shutil.copyfile(xlsx_path, output_dir / "task1_step5_5_candidate_v2_details.xlsx")
    md = [
        "# Task1 Step 5.5 Candidate V2 Dev Report",
        "",
        f"- 恢复检查：复用半成品={recovery['reused_partial_outputs']}；现有输出文件={len(recovery['existing_output_files'])}；运行进程={len(recovery['running_processes'])}。",
        "- Trainer `eval_macro_f1` 是候选对二分类 0/1 macro-F1；fixed-64 macro-F1 是文章级 64 事件名集合指标，因此差异巨大。",
        "- `checkpoint-849` 按 pair-level `macro_f1` 选择，不是按文章级 fixed-64 macro-F1 选择。",
        "- 原候选器低召回主因：只依赖目标路由后的官方映射，路由/映射缺失时没有 schema 语义召回和 all-64 回退。",
        f"- candidate-v2 recall@20/@32：{candidate_gate['candidate_recall_at_20']:.6f}/{candidate_gate['candidate_recall_at_32']:.6f}；candidate_missing@32={candidate_gate['candidate_missing_at_32']}。",
        f"- all-64 candidate recall：{candidate_gate['candidate_recall_at_64']:.6f}。",
        f"- 长尾候选 recall@20/@32：{long_tail_recall['candidate_recall_at_20']:.6f}/{long_tail_recall['candidate_recall_at_32']:.6f}。",
        f"- 候选门槛通过：{candidate_gate['passed']}；重新训练状态：{training_summary.get('status', 'not_run')}。",
        f"- 最优 dev 方法：{best_dev['method']}；micro P/R/F1={best_dev['micro_precision']}/{best_dev['micro_recall']}/{best_dev['micro_f1']}；fixed-64 macro-F1={best_dev['macro_f1_fixed_64']}；article exact={best_dev['article_exact_accuracy']}。",
        f"- 进入 1029 篇 gold test 冻结盲测门槛：{final_gate['passed_for_1029_blind_test']}。",
        f"- 正式保护文件 SHA 不变：{protected_unchanged}；32B/外部 LLM 调用：0/0。",
    ]
    if method_details.get("all64_existing_classifier_true_upper"):
        upper = method_details["all64_existing_classifier_true_upper"]
        md.append(f"- 原分类器 all-64 条件下 dev 上限：threshold={upper['threshold']} micro-F1={upper['micro_f1']} fixed-64 macro-F1={upper['macro_f1_fixed_64']}。")
    md.append("")
    md.append("## 候选来源贡献")
    for row in source_rows:
        md.append(f"- {row['source']}: independent_recall={row['independent_recall']}, new_hits={row['new_hit_count_in_source_order']}")
    (ROOT / config["report_md"]).write_text("\n".join(md) + "\n", encoding="utf-8")
    (output_dir / "step5_5_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def write_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS.txt"}:
            rows.append({"path": path.relative_to(output_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    with (output_dir / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['sha256']}  {row['path']}\n")
    rows.append({"path": "SHA256SUMS.txt", "size_bytes": (output_dir / "SHA256SUMS.txt").stat().st_size, "sha256": sha256_file(output_dir / "SHA256SUMS.txt")})
    write_json(
        output_dir / "manifest.json",
        {
            "manifest_version": "task1_step5_5_candidate_v2_dev",
            "generated_at": now(),
            "output_dir": rel(output_dir),
            "file_count": len(rows),
            "files": rows,
        },
    )


def run(config_path: Path, resume: bool = False, skip_pair_inference: bool = False, skip_training: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log("Step 5.5 run started")
    recovery = recovery_check(output_dir)
    write_json(output_dir / "recovery_check.json", recovery)
    log("recovery check written")
    before_path, protected_before = write_protected_snapshot(output_dir, ROOT / config["protected_before_path"])
    log(f"protected SHA before loaded: {before_path}")
    split = load_frozen_split(config)
    log(f"frozen split loaded train={len(split['train_ids'])} dev={len(split['dev_ids'])}")
    for path_key in ["company_mapping_path", "industry_mapping_path", "schema_path", "target_alias_path"]:
        path = ROOT / config[path_key]
        if not path.exists():
            raise FileNotFoundError(path)
    candidate_rows, timing_rows, generator = load_or_generate_candidates(output_dir, config, split, resume=resume)
    log(f"candidate rows ready: {len(candidate_rows)}")
    candidate_metrics = candidate_recall_at_k(split["dev_ids"], split["dev_pairs"], candidate_rows, generator.event_names, config["candidate_recall_ks"])
    source_rows = source_contribution(split["dev_ids"], split["dev_pairs"], candidate_rows, generator.source_order)
    log("candidate metrics/source contribution computed")
    write_csv(output_dir / "candidate_source_contribution.csv", source_rows)
    write_csv(output_dir / "candidate_recall_64_classes.csv", candidate_metrics["class_rows"])
    write_json(output_dir / "official_mapping_manifest.json", {
        "schema_path": config["schema_path"],
        "company_mapping_path": config["company_mapping_path"],
        "industry_mapping_path": config["industry_mapping_path"],
        "target_alias_path": config["target_alias_path"],
        "schema_sha256": sha256_file(ROOT / config["schema_path"]),
        "company_mapping_sha256": sha256_file(ROOT / config["company_mapping_path"]),
        "industry_mapping_sha256": sha256_file(ROOT / config["industry_mapping_path"]),
        "target_alias_sha256": sha256_file(ROOT / config["target_alias_path"]),
        "event_name_count": len(generator.event_names),
    })
    pair_scores = None
    pair_summary: dict[str, Any] = {"status": "skipped_by_arg"} if skip_pair_inference else {}
    if not skip_pair_inference:
        log("starting frozen pair classifier inference")
        pair_scores, pair_summary = score_pair_classifier(
            candidate_rows,
            split["article_by_id"],
            ROOT / config["frozen_pair_model_dir"],
            output_dir / "pair_classifier_all64_predictions.jsonl",
            int(config["pair_inference"].get("batch_size", 64)),
            int(config["pair_inference"].get("max_length", 512)),
            resume=resume,
        )
        write_json(output_dir / "pair_classifier_inference_summary.json", pair_summary)
        log(f"frozen pair classifier inference status={pair_summary.get('status')}")
    gate_cfg = config["candidate_acceptance"]
    candidate_missing_at_32 = int(round((1.0 - candidate_metrics["candidate_recall_at_32"]) * candidate_metrics["gold_event_count"]))
    candidate_passed = (
        candidate_metrics["candidate_recall_at_20"] >= gate_cfg["candidate_recall_at_20"]
        and candidate_metrics["candidate_recall_at_32"] >= gate_cfg["candidate_recall_at_32"]
        and candidate_metrics["candidate_recall_at_64"] >= gate_cfg["all64_recall"]
        and candidate_missing_at_32 <= gate_cfg["candidate_missing_at_32_max"]
    )
    training_summary: dict[str, Any] = {"status": "not_run_candidate_gate_failed"}
    new_scores = None
    if candidate_passed:
        log("candidate gate passed")
        if skip_training:
            training_summary = {"status": "skipped_by_arg_after_candidate_gate_passed"}
        else:
            training_summary, new_scores = train_new_multilabel_classifier(output_dir, config, split, generator.event_names, candidate_rows, resume=resume)
            log(f"training status={training_summary.get('status')}")
    write_json(output_dir / "candidate_gate_summary.json", {
        "passed": candidate_passed,
        "candidate_recall_at_20": candidate_metrics["candidate_recall_at_20"],
        "candidate_recall_at_32": candidate_metrics["candidate_recall_at_32"],
        "candidate_recall_at_64": candidate_metrics["candidate_recall_at_64"],
        "candidate_missing_at_32": candidate_missing_at_32,
    })
    if new_scores is None and (output_dir / "new_classifier_all64_predictions.jsonl").exists():
        new_scores = read_jsonl(output_dir / "new_classifier_all64_predictions.jsonl")
    method_rows, method_details, class_candidate_rows, fn_rows = evaluate_prediction_methods(
        config,
        split,
        generator.event_names,
        candidate_rows,
        pair_scores,
        new_scores,
        output_dir,
    )
    after_path, protected_after, protected_unchanged = write_after_snapshot(output_dir, protected_before)
    log(f"protected SHA after written: {after_path}")
    report = write_reports(
        config,
        output_dir,
        recovery,
        split,
        generator,
        candidate_rows,
        timing_rows,
        candidate_metrics,
        source_rows,
        method_rows,
        method_details,
        class_candidate_rows,
        fn_rows,
        pair_summary,
        training_summary,
        protected_before,
        protected_after,
        protected_unchanged,
    )
    write_manifest(output_dir)
    log("Step 5.5 run completed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task1_candidate_v2.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-pair-inference", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    report = run(
        args.config.resolve(),
        resume=args.resume,
        skip_pair_inference=args.skip_pair_inference,
        skip_training=args.skip_training,
    )
    best = report["best_dev_method"]
    print(
        json.dumps(
            {
                "output_dir": report["output_dir"],
                "candidate_recall_at_20": report["candidate_v2"]["candidate_gate"]["candidate_recall_at_20"],
                "candidate_recall_at_32": report["candidate_v2"]["candidate_gate"]["candidate_recall_at_32"],
                "candidate_gate_passed": report["candidate_v2"]["candidate_gate"]["passed"],
                "training_status": report["training_summary"].get("status", ""),
                "best_method": best["method"],
                "best_micro_f1": best["micro_f1"],
                "best_fixed_64_macro_f1": best["macro_f1_fixed_64"],
                "formal_sha_unchanged": report["formal_file_sha256_unchanged"],
                "dianjin_r1_32b_calls": report["llm_calls"]["dianjin_r1_32b_calls"],
                "external_llm_calls": report["llm_calls"]["external_llm_calls"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
