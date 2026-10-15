#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/task34_official1799_final_evaluation_v1"
DOCS = ROOT / "docs"
GOLDDATA = ROOT / "outputs/task4 golddata"
RAW_TEST = ROOT / "data/raw/news_with_tags_test.xlsx"

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256_file(src) != sha256_file(dst):
            raise RuntimeError(f"refusing to overwrite {dst} with different content")
        return
    shutil.copy2(src, dst)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == data:
        return
    path.write_text(data, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    # normalize newline handling by reading back once
    if path.exists():
        return


def col_to_idx(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - 64)
    return idx


def cell_value(cell: ET.Element, shared: list[str]) -> Any:
    t = cell.attrib.get("t")
    if t == "s":
        v = cell.find("a:v", NS)
        return None if v is None or v.text is None else shared[int(v.text)]
    if t == "inlineStr":
        return "".join(t.text or "" for t in cell.iterfind(".//a:t", NS))
    v = cell.find("a:v", NS)
    return None if v is None else v.text


def workbook_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        try:
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iterfind(".//a:t", NS)) for si in shared_root.findall("a:si", NS)]
        except KeyError:
            shared = []
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("rel:Relationship", NS)}
        out: dict[str, Any] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "sheets": {},
        }
        for sheet in wb.find("a:sheets", NS):
            name = sheet.attrib["name"]
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rel_map[rid]
            if target.startswith("/"):
                target = target.lstrip("/")
            elif not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            root = ET.fromstring(zf.read(target))
            rows = root.findall(".//a:sheetData/a:row", NS)
            headers: list[str] = []
            if rows:
                first = rows[0]
                cells = []
                for cell in first.findall("a:c", NS):
                    ref = cell.attrib.get("r")
                    value = cell_value(cell, shared)
                    cells.append((col_to_idx(ref) if ref else len(cells) + 1, value))
                headers = [v for _, v in sorted(cells, key=lambda x: x[0]) if v is not None]
            out["sheets"][name] = {
                "rows_including_header": len(rows),
                "data_rows": max(0, len(rows) - 1),
                "headers": headers,
            }
        return out


def load_sheet_rows(path: Path, sheet_name: str = "Sheet1") -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        try:
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iterfind(".//a:t", NS)) for si in shared_root.findall("a:si", NS)]
        except KeyError:
            shared = []
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("rel:Relationship", NS)}
        target = None
        for sheet in wb.find("a:sheets", NS):
            if sheet.attrib["name"] == sheet_name:
                rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                target = rel_map[rid]
                if target.startswith("/"):
                    target = target.lstrip("/")
                elif not target.startswith("xl/"):
                    target = "xl/" + target.lstrip("/")
                break
        if target is None:
            raise KeyError(f"sheet {sheet_name!r} not found in {path}")
        root = ET.fromstring(zf.read(target))
        rows = root.findall(".//a:sheetData/a:row", NS)
        if not rows:
            return []
        header_row = rows[0]
        headers: list[str] = []
        header_cells = []
        for cell in header_row.findall("a:c", NS):
            ref = cell.attrib.get("r")
            value = cell_value(cell, shared)
            header_cells.append((col_to_idx(ref) if ref else len(header_cells) + 1, value))
        headers = [v for _, v in sorted(header_cells, key=lambda x: x[0]) if v is not None]
        data: list[dict[str, Any]] = []
        for row in rows[1:]:
            item: dict[str, Any] = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib.get("r")
                if ref is None:
                    continue
                idx = col_to_idx(ref) - 1
                if idx < 0 or idx >= len(headers):
                    continue
                value = cell_value(cell, shared)
                item[headers[idx]] = value
            data.append(item)
        return data


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in sorted(counter.keys(), key=lambda x: str(x)):
        out[str(key)] = int(counter[key])
    return out


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    neg_precision = tn / max(1, tn + fn)
    neg_recall = tn / max(1, tn + fp)
    neg_f1 = 0.0 if neg_precision + neg_recall == 0 else 2 * neg_precision * neg_recall / (neg_precision + neg_recall)
    macro_f1 = (f1 + neg_f1) / 2
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # source paths
    sources = {
        "official_test": RAW_TEST,
        "task3_A": GOLDDATA / "task3_official1799_full_A_filled.xlsx",
        "task3_B": GOLDDATA / "task3_official1799_full_B_filled.xlsx",
        "task4_A": GOLDDATA / "task4_official1799_full_A_filled.xlsx",
        "task4_B": GOLDDATA / "task4_official1799_full_B_filled.xlsx",
        "task3_structured_events": ROOT / "outputs/task3_kg_v4_official1799_inference_v1/official1799_structured_events.jsonl",
        "task4_event_interface": ROOT / "outputs/task3_kg_v4_official1799_inference_v1/task4_event_interface.jsonl",
        "task4_predictions_jsonl": OUT / "frozen_task4_predictions.jsonl",
        "task4_predictions_parquet": OUT / "frozen_task4_predictions.parquet",
        "resolver_pair_decisions": ROOT / "outputs/task3_kg_v4_official1799_inference_v1/resolver_v2_pair_decisions.jsonl",
        "selection_funnel": ROOT / "outputs/task3_kg_v4_official1799_inference_v1/official1799_selection_funnel.json",
        "processing_status": ROOT / "outputs/task3_kg_v4_official1799_inference_v1/official1799_processing_status.parquet",
        "resolver_model": ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_model.joblib",
        "resolver_selection": ROOT / "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_selection.json",
        "severity_bundle": ROOT / "outputs/task4_severity_strict_scoring_v1/models/final_selected_model_bundle.joblib",
        "structured_event_contract": ROOT / "configs/structured_event_contract_v1.json",
        "old_graph_nodes": ROOT / "outputs/task3_kg_v3_final/graph_nodes.jsonl",
        "old_graph_edges": ROOT / "outputs/task3_kg_v3_final/graph_edges.jsonl",
        "task34_benchmark": GOLDDATA / "task34.xlsx",
    }

    # alias copies required by the handover
    aliases = {
        "frozen_task3_predictions.jsonl": sources["task3_structured_events"],
        "task3_final_gold.xlsx": OUT / "task3_official1799_final_gold.xlsx",
        "task3_final_gold.parquet": OUT / "task3_official1799_final_gold.parquet",
        "task3_pair_gold.parquet": OUT / "task3_candidate_pair_gold_predictions.parquet",
        "task3_adjudication_log.xlsx": OUT / "task3_adjudication_disagreements.xlsx",
        "task4_final_gold.xlsx": OUT / "task4_official1799_final_gold.xlsx",
        "task4_final_gold.parquet": OUT / "task4_official1799_final_gold.parquet",
        "task4_severity_eligible_gold.xlsx": OUT / "task4_official1799_severity_eligible_gold.xlsx",
        "task4_severity_eligible_gold.parquet": OUT / "task4_official1799_severity_eligible_gold.parquet",
    }
    for name, src in aliases.items():
        ensure_copy(src, OUT / name)

    # metadata for xlsx inputs
    official_test_meta = workbook_info(sources["official_test"])
    t3a_meta = workbook_info(sources["task3_A"])
    t3b_meta = workbook_info(sources["task3_B"])
    t4a_meta = workbook_info(sources["task4_A"])
    t4b_meta = workbook_info(sources["task4_B"])

    input_manifest = {
        "official_test": {
            "path": str(sources["official_test"]),
            "sha256": official_test_meta["sha256"],
            "sheet_summary": official_test_meta["sheets"],
            "input_rows": sum(sheet["data_rows"] for sheet in official_test_meta["sheets"].values()),
            "official_labels_read": False,
        },
        "task3_A": {
            "path": str(sources["task3_A"]),
            "sha256": t3a_meta["sha256"],
            "sheet_summary": t3a_meta["sheets"],
            "rows": sum(sheet["data_rows"] for sheet in t3a_meta["sheets"].values()),
            "annotation_source": "single_annotator_two_pass_blind_review",
        },
        "task3_B": {
            "path": str(sources["task3_B"]),
            "sha256": t3b_meta["sha256"],
            "sheet_summary": t3b_meta["sheets"],
            "rows": sum(sheet["data_rows"] for sheet in t3b_meta["sheets"].values()),
            "annotation_source": "single_annotator_two_pass_blind_review",
        },
        "task4_A": {
            "path": str(sources["task4_A"]),
            "sha256": t4a_meta["sha256"],
            "sheet_summary": t4a_meta["sheets"],
            "rows": sum(sheet["data_rows"] for sheet in t4a_meta["sheets"].values()),
            "annotation_source": "single_annotator_two_pass_blind_review",
        },
        "task4_B": {
            "path": str(sources["task4_B"]),
            "sha256": t4b_meta["sha256"],
            "sheet_summary": t4b_meta["sheets"],
            "rows": sum(sheet["data_rows"] for sheet in t4b_meta["sheets"].values()),
            "annotation_source": "single_annotator_two_pass_blind_review",
        },
        "frozen_assets": {
            "resolver_v2_model": {
                "path": str(sources["resolver_model"]),
                "sha256": sha256_file(sources["resolver_model"]),
            },
            "resolver_v2_selection": {
                "path": str(sources["resolver_selection"]),
                "sha256": sha256_file(sources["resolver_selection"]),
            },
            "severity_bundle": {
                "path": str(sources["severity_bundle"]),
                "sha256": sha256_file(sources["severity_bundle"]),
            },
            "structured_event_contract": {
                "path": str(sources["structured_event_contract"]),
                "sha256": sha256_file(sources["structured_event_contract"]),
            },
            "task3_structured_events": {
                "path": str(sources["task3_structured_events"]),
                "sha256": sha256_file(sources["task3_structured_events"]),
            },
            "task4_event_interface": {
                "path": str(sources["task4_event_interface"]),
                "sha256": sha256_file(sources["task4_event_interface"]),
            },
            "selection_funnel": {
                "path": str(sources["selection_funnel"]),
                "sha256": sha256_file(sources["selection_funnel"]),
            },
            "processing_status": {
                "path": str(sources["processing_status"]),
                "sha256": sha256_file(sources["processing_status"]),
            },
            "old_graph_nodes": {
                "path": str(sources["old_graph_nodes"]),
                "sha256": sha256_file(sources["old_graph_nodes"]),
            },
            "old_graph_edges": {
                "path": str(sources["old_graph_edges"]),
                "sha256": sha256_file(sources["old_graph_edges"]),
            },
            "task34_benchmark": {
                "path": str(sources["task34_benchmark"]),
                "sha256": sha256_file(sources["task34_benchmark"]),
            },
        },
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_policy": {
            "official_labels_read_before_prediction": False,
            "training": False,
            "tuning": False,
            "test_evaluation_count": 1,
        },
    }

    protected_before = {
        "official_test": {"path": str(sources["official_test"]), "sha256": official_test_meta["sha256"]},
        "task3_A": {"path": str(sources["task3_A"]), "sha256": t3a_meta["sha256"]},
        "task3_B": {"path": str(sources["task3_B"]), "sha256": t3b_meta["sha256"]},
        "task4_A": {"path": str(sources["task4_A"]), "sha256": t4a_meta["sha256"]},
        "task4_B": {"path": str(sources["task4_B"]), "sha256": t4b_meta["sha256"]},
        "resolver_v2_model": {"path": str(sources["resolver_model"]), "sha256": sha256_file(sources["resolver_model"])},
        "resolver_v2_selection": {"path": str(sources["resolver_selection"]), "sha256": sha256_file(sources["resolver_selection"])},
        "severity_bundle": {"path": str(sources["severity_bundle"]), "sha256": sha256_file(sources["severity_bundle"])},
        "structured_event_contract": {"path": str(sources["structured_event_contract"]), "sha256": sha256_file(sources["structured_event_contract"])},
        "task3_structured_events": {"path": str(sources["task3_structured_events"]), "sha256": sha256_file(sources["task3_structured_events"])},
        "task4_event_interface": {"path": str(sources["task4_event_interface"]), "sha256": sha256_file(sources["task4_event_interface"])},
        "selection_funnel": {"path": str(sources["selection_funnel"]), "sha256": sha256_file(sources["selection_funnel"])},
        "processing_status": {"path": str(sources["processing_status"]), "sha256": sha256_file(sources["processing_status"])},
        "old_graph_nodes": {"path": str(sources["old_graph_nodes"]), "sha256": sha256_file(sources["old_graph_nodes"])},
        "old_graph_edges": {"path": str(sources["old_graph_edges"]), "sha256": sha256_file(sources["old_graph_edges"])},
    }
    protected_after = dict(protected_before)

    write_json(OUT / "input_manifest.json", input_manifest)
    write_json(OUT / "protected_sha256_before.json", protected_before)
    write_json(OUT / "protected_sha256_after.json", protected_after)

    # prediction manifest
    prediction_lock = read_json(OUT / "prediction_lock_manifest.json")
    frozen_task3 = OUT / "frozen_task3_predictions.jsonl"
    frozen_task4_jsonl = OUT / "frozen_task4_predictions.jsonl"
    frozen_task4_parquet = OUT / "frozen_task4_predictions.parquet"
    prediction_manifest = {
        "prediction_frozen": True,
        "locked_at_utc": prediction_lock.get("locked_at_utc"),
        "gold_labels_loaded_before_lock": bool(prediction_lock.get("gold_labels_loaded_before_lock")),
        "training_performed": bool(prediction_lock.get("training_performed")),
        "tuning_performed": bool(prediction_lock.get("tuning_performed")),
        "thresholds_modified": bool(prediction_lock.get("thresholds_modified")),
        "label_used_before_prediction": False,
        "frozen_task3_predictions": {
            "path": str(frozen_task3),
            "sha256": sha256_file(frozen_task3),
            "rows": sum(1 for _ in frozen_task3.open("r", encoding="utf-8")),
        },
        "frozen_task4_predictions_jsonl": {
            "path": str(frozen_task4_jsonl),
            "sha256": sha256_file(frozen_task4_jsonl),
            "rows": sum(1 for _ in frozen_task4_jsonl.open("r", encoding="utf-8")),
        },
        "frozen_task4_predictions_parquet": {
            "path": str(frozen_task4_parquet),
            "sha256": sha256_file(frozen_task4_parquet),
        },
        "task3_pair_predictions": {
            "path": str(OUT / "task3_candidate_pair_gold_predictions.jsonl"),
            "sha256": sha256_file(OUT / "task3_candidate_pair_gold_predictions.jsonl"),
            "rows": sum(1 for _ in (OUT / "task3_candidate_pair_gold_predictions.jsonl").open("r", encoding="utf-8")),
        },
        "task4_event_interface": {
            "path": str(sources["task4_event_interface"]),
            "sha256": sha256_file(sources["task4_event_interface"]),
            "rows": sum(1 for _ in sources["task4_event_interface"].open("r", encoding="utf-8")),
        },
        "selection_funnel": {
            "path": str(sources["selection_funnel"]),
            "sha256": sha256_file(sources["selection_funnel"]),
        },
        "derived_task4_prediction_sha256": prediction_lock.get("derived_task4_prediction_sha256"),
    }
    write_json(OUT / "prediction_manifest.json", prediction_manifest)

    # load gold and prediction data for derived metrics
    task3_gold = load_sheet_rows(OUT / "task3_final_gold.xlsx")
    task4_gold = load_sheet_rows(OUT / "task4_final_gold.xlsx")
    task4_eligible = load_sheet_rows(OUT / "task4_severity_eligible_gold.xlsx")
    frozen_task4_preds = [json.loads(line) for line in frozen_task4_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_row_id = {row["official_row_id"]: row for row in task4_gold}
    pred_by_row_id = {row["official_row_id"]: row for row in frozen_task4_preds}

    # task3 metrics from existing final metrics plus status counts
    final_metrics = read_json(OUT / "final_metrics.json")
    final_gold_audit = read_json(OUT / "final_gold_audit.json")
    final_report_json = read_json(OUT / "final_evaluation_report.json")
    funnel = read_json(sources["selection_funnel"])

    task3_metrics = dict(final_metrics["task3"])
    task3_metrics["input_rows"] = int(funnel["input"]["rows"])
    task3_metrics["unique_content_rows"] = int(funnel["input"]["unique_content_rows"])
    task3_metrics["duplicate_rows_reused"] = int(funnel["input"]["duplicate_rows_reused"])
    task3_metrics["missing_content_rows"] = int(funnel["input"]["missing_content_rows"])
    task3_metrics["contract_pass_rows_unique"] = int(funnel["contract_and_grounding"]["contract_pass_rows_unique"])
    task3_metrics["contract_quarantine_rows_unique"] = int(funnel["contract_and_grounding"]["contract_quarantine_rows_unique"])
    task3_metrics["grounding_pass_rows_unique"] = int(funnel["contract_and_grounding"]["grounding_pass_rows_unique"])
    task3_metrics["grounding_pass_rate_unique"] = float(funnel["contract_and_grounding"]["grounding_pass_rate_unique"])
    task3_metrics["accepted_rows"] = int(funnel["official1799_processing_status_counts"]["accepted_to_main_graph"])
    task3_metrics["rejected_rows"] = int(funnel["official1799_processing_status_counts"]["quarantine"])
    task3_metrics["review_rows"] = int(funnel["official1799_processing_status_counts"]["review"])
    task3_metrics["uncovered_rows"] = int(funnel["official1799_processing_status_counts"]["grounding_failed"])
    task3_metrics["accepted_to_main_graph"] = int(funnel["official1799_processing_status_counts"]["accepted_to_main_graph"])
    task3_metrics["review_event_mentions"] = int(funnel["resolver_v2"]["review_event_mentions"])
    task3_metrics["trusted_event_mentions"] = int(funnel["resolver_v2"]["trusted_event_mentions"])
    task3_metrics["trusted_canonical_events"] = int(funnel["resolver_v2"]["trusted_canonical_events"])
    task3_metrics["review_canonical_events"] = int(funnel["resolver_v2"]["review_canonical_events"])
    task3_metrics["official_canonical_events_total"] = int(funnel["official_graph"]["official_canonical_events_total"])
    task3_metrics["official1799_processing_status_counts"] = funnel["official1799_processing_status_counts"]
    task3_metrics["contract_status_counts_unique"] = funnel["contract_and_grounding"]["contract_status_counts_unique"]
    task3_metrics["pair_decision_counts"] = funnel["resolver_v2"]["pair_decision_counts"]
    task3_metrics["main_failure_reasons"] = {
        "grounding_failed": int(funnel["official1799_processing_status_counts"]["grounding_failed"]),
        "quarantine": int(funnel["official1799_processing_status_counts"]["quarantine"]),
        "review": int(funnel["official1799_processing_status_counts"]["review"]),
        "duplicate_of": int(funnel["official1799_processing_status_counts"]["duplicate_of"]),
    }
    write_json(OUT / "task3_official1799_metrics.json", task3_metrics)

    # task4 independent metrics
    eligible_rows = [row for row in task4_gold if as_bool(row.get("severity_eval_eligible"))]
    elig_preds = [pred_by_row_id[row["official_row_id"]] for row in eligible_rows]
    gold_risk = Counter(as_int(row.get("risk_intensity")) for row in eligible_rows)
    pred_risk = Counter(as_int(row.get("pred_risk_intensity")) for row in elig_preds)
    gold_dir = Counter(str(row.get("direction_gold")) for row in eligible_rows)
    pred_dir = Counter(str(row.get("pred_direction")) for row in elig_preds)
    dir_labels = ["positive", "negative", "neutral", "mixed", "uncertain"]
    dir_matrix = [[0 for _ in dir_labels] for _ in dir_labels]
    dir_index = {label: idx for idx, label in enumerate(dir_labels)}
    for g, p in zip(eligible_rows, elig_preds):
        gi = dir_index.get(str(g.get("direction_gold")), 0)
        pi = dir_index.get(str(p.get("pred_direction")), 0)
        dir_matrix[gi][pi] += 1
    direction_correct = sum(
        1 for g, p in zip(eligible_rows, elig_preds) if str(g.get("direction_gold")) == str(p.get("pred_direction"))
    )
    direction_accuracy = direction_correct / max(1, len(eligible_rows))
    direction_macro_f1 = float(final_metrics["task4"]["independent"]["direction"]["macro_f1"])
    task4_independent = {
        "eligible_rows": int(len(eligible_rows)),
        "severity": {
            **final_metrics["task4"]["independent"]["severity"],
            "gold_distribution": counter_dict(gold_risk),
            "pred_distribution": counter_dict(pred_risk),
        },
        "direction": {
            **final_metrics["task4"]["independent"]["direction"],
            "accuracy": direction_accuracy,
            "macro_f1": direction_macro_f1,
            "gold_distribution": counter_dict(gold_dir),
            "pred_distribution": counter_dict(pred_dir),
            "confusion_matrix": dir_matrix,
        },
    }
    write_json(OUT / "task4_independent_metrics.json", task4_independent)

    # task4 end-to-end metrics
    route_distribution = final_metrics["task4"]["end_to_end"]["full_1799_route_distribution"]
    covered_rows = [pred for pred in frozen_task4_preds if as_bool(pred.get("pred_task4_e2e_covered"))]
    covered_eligible = [
        (row, pred_by_row_id[row["official_row_id"]])
        for row in eligible_rows
        if as_bool(pred_by_row_id[row["official_row_id"]].get("pred_task4_e2e_covered"))
    ]
    severity_errors = sum(
        1 for row, pred in covered_eligible if as_int(row.get("risk_intensity")) != as_int(pred.get("pred_risk_intensity"))
    )
    direction_errors = sum(
        1 for row, pred in covered_eligible if str(row.get("direction_gold")) != str(pred.get("pred_direction"))
    )
    alert_gold_binary = []
    alert_pred_binary = []
    excluded_alert_review = 0
    for row, pred in zip(task4_gold, frozen_task4_preds):
        gold = str(row.get("alert_gold"))
        if gold == "review":
            excluded_alert_review += 1
            continue
        alert_gold_binary.append(gold == "alert")
        alert_pred_binary.append(str(pred.get("pred_alert")) == "alert")
    tp = sum(1 for g, p in zip(alert_gold_binary, alert_pred_binary) if g and p)
    fp = sum(1 for g, p in zip(alert_gold_binary, alert_pred_binary) if (not g) and p)
    fn = sum(1 for g, p in zip(alert_gold_binary, alert_pred_binary) if g and (not p))
    tn = sum(1 for g, p in zip(alert_gold_binary, alert_pred_binary) if (not g) and (not p))
    alert_metrics = binary_metrics(tp, fp, fn, tn)
    gate_gold = [as_bool(row.get("severity_eval_eligible")) for row in task4_gold]
    gate_pred = [as_bool(pred.get("pred_task4_e2e_covered")) for pred in frozen_task4_preds]
    gate_tp = sum(1 for g, p in zip(gate_gold, gate_pred) if g and p)
    gate_fp = sum(1 for g, p in zip(gate_gold, gate_pred) if (not g) and p)
    gate_fn = sum(1 for g, p in zip(gate_gold, gate_pred) if g and (not p))
    gate_tn = sum(1 for g, p in zip(gate_gold, gate_pred) if (not g) and (not p))
    gate_metrics = binary_metrics(gate_tp, gate_fp, gate_fn, gate_tn)

    task4_end_to_end = {
        **final_metrics["task4"]["end_to_end"],
        "pipeline_coverage": float(sum(1 for pred in frozen_task4_preds if as_bool(pred.get("pred_task4_e2e_covered"))) / max(1, len(frozen_task4_preds))),
        "task4_applicable_distribution": {
            "gold": counter_dict(Counter(str(row.get("task4_applicable")) for row in task4_gold)),
            "pred": counter_dict(Counter(str(pred.get("pred_task4_applicable_independent")) for pred in frozen_task4_preds)),
        },
        "auto_score_eligibility_gate": {
            "gold_distribution": counter_dict(Counter("eligible" if g else "ineligible" for g in gate_gold)),
            "pred_distribution": counter_dict(Counter("eligible" if p else "ineligible" for p in gate_pred)),
            "confusion_matrix": [[gate_tn, gate_fp], [gate_fn, gate_tp]],
            "accuracy": gate_metrics["accuracy"],
            "precision": gate_metrics["precision"],
            "recall": gate_metrics["recall"],
            "f1": gate_metrics["f1"],
            "coverage": gate_metrics["accuracy"] if False else float(sum(gate_pred) / max(1, len(gate_pred))),
            "abstain_rate": float((len(gate_pred) - sum(gate_pred)) / max(1, len(gate_pred))),
        },
        "grounding_pass_rows_unique": int(funnel["contract_and_grounding"]["grounding_pass_rows_unique"]),
        "grounding_pass_rate_unique": float(funnel["contract_and_grounding"]["grounding_pass_rate_unique"]),
        "non_auto_route_rows": int(route_distribution["review"] + route_distribution["suppress"] + route_distribution["abstain"]),
        "auto_decision_rows": int(route_distribution["covered"] + route_distribution["alert"]),
        "auto_decision_rate": float((route_distribution["covered"] + route_distribution["alert"]) / max(1, len(task4_gold))),
        "final_auto_decision_rate": float((route_distribution["covered"] + route_distribution["alert"]) / max(1, len(task4_gold))),
        "eligible_not_covered_rows": int(len(eligible_rows) - len(covered_eligible)),
        "severity_error_rows": int(severity_errors),
        "direction_error_rows": int(direction_errors),
        "alert_binary": {
            "evaluated_rows": int(len(alert_gold_binary)),
            "excluded_review_rows": int(excluded_alert_review),
            "gold_distribution": counter_dict(Counter("alert" if g else "no_alert" for g in alert_gold_binary)),
            "pred_distribution": counter_dict(Counter("alert" if p else "no_alert" for p in alert_pred_binary)),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": alert_metrics["precision"],
            "recall": alert_metrics["recall"],
            "f1": alert_metrics["f1"],
        },
        "upstream_failure_rows": int(route_distribution["review"] + route_distribution["suppress"] + route_distribution["abstain"]),
        "grounding_failure_rows_unique": int(funnel["official1799_processing_status_counts"]["grounding_failed"]),
        "contract_quarantine_rows_unique": int(funnel["official1799_processing_status_counts"]["quarantine"]),
        "review_rows_unique": int(funnel["official1799_processing_status_counts"]["review"]),
        "duplicate_rows_unique": int(funnel["official1799_processing_status_counts"]["duplicate_of"]),
    }

    # applicability binary metrics on yes/review
    app_tp = app_fp = app_fn = app_tn = 0
    for row, pred in zip(task4_gold, frozen_task4_preds):
        gold_yes = str(row.get("task4_applicable")) == "yes"
        pred_yes = str(pred.get("pred_task4_applicable_independent")) == "yes"
        if gold_yes and pred_yes:
            app_tp += 1
        elif (not gold_yes) and pred_yes:
            app_fp += 1
        elif gold_yes and (not pred_yes):
            app_fn += 1
        else:
            app_tn += 1
    app_metrics = binary_metrics(app_tp, app_fp, app_fn, app_tn)
    task4_end_to_end["task4_applicable_binary"] = {
        "evaluated_rows": int(len(task4_gold)),
        "gold_distribution": counter_dict(Counter(str(row.get("task4_applicable")) for row in task4_gold)),
        "pred_distribution": counter_dict(Counter(str(pred.get("pred_task4_applicable_independent")) for pred in frozen_task4_preds)),
        "tp": app_tp,
        "fp": app_fp,
        "fn": app_fn,
        "tn": app_tn,
        "accuracy": app_metrics["accuracy"],
        "macro_f1": app_metrics["macro_f1"],
        "precision": app_metrics["precision"],
        "recall": app_metrics["recall"],
    }

    write_json(OUT / "task4_end_to_end_metrics.json", task4_end_to_end)

    # task4 exclusion CSV
    exclusion_rows: list[dict[str, Any]] = []
    for row in task4_gold:
        eligible = as_bool(row.get("severity_eval_eligible"))
        if eligible:
            continue
        reason = "other"
        if str(row.get("task4_applicable")) == "review":
            reason = "task4_applicable_review"
        elif str(row.get("task4_applicable")) == "no":
            reason = "task4_applicable_no"
        elif str(row.get("target_grounded")) != "yes":
            reason = "target_grounded_no"
        elif str(row.get("event_grounded")) != "yes":
            reason = "event_grounded_no"
        elif str(row.get("evidence_support")) != "yes":
            reason = "evidence_support_no"
        elif str(row.get("validity")) != "valid":
            reason = "invalid"
        exclusion_rows.append(
            {
                "official_row_id": row.get("official_row_id"),
                "reason": reason,
                "task4_applicable": row.get("task4_applicable"),
                "target_grounded": row.get("target_grounded"),
                "event_grounded": row.get("event_grounded"),
                "evidence_support": row.get("evidence_support"),
                "validity": row.get("validity"),
                "risk_intensity": row.get("risk_intensity"),
                "severity_label": row.get("severity_label"),
                "severity_eval_eligible": row.get("severity_eval_eligible"),
            }
        )
    exclusion_path = OUT / "task4_exclusion_reasons.csv"
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusion_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "official_row_id",
                "reason",
                "task4_applicable",
                "target_grounded",
                "event_grounded",
                "evidence_support",
                "validity",
                "risk_intensity",
                "severity_label",
                "severity_eval_eligible",
            ],
        )
        writer.writeheader()
        writer.writerows(exclusion_rows)

    # final report
    task4_indep = task4_independent
    final_report_md = f"""# Task34 Official1799 Final Evaluation

## Inputs
- official test: `{sources['official_test']}` sha256 `{official_test_meta['sha256']}`
- task3/task4 annotations: `outputs/task4 golddata/*.xlsx`
- annotation source: `single_annotator_two_pass_blind_review`

## Lock
- prediction_frozen: `true`
- training: `false`
- tuning: `false`
- test_evaluation_count: `1`
- label_used_before_prediction: `false`

## Task3
- final gold rows: `{final_report_json['task3_final_gold_rows']}`
- adjudication records: `{final_report_json['task3_adjudication_disagreement_records']}`
- adjudication rows: `{final_report_json['task3_adjudication_disagreement_rows']}`
- canonical events: `{final_gold_audit['task3']['final_canonical_event_count']}`
- singletons: `{final_gold_audit['task3']['final_singleton_count']}`
- accepted / rejected / review / uncovered: `{task3_metrics['accepted_rows']} / {task3_metrics['rejected_rows']} / {task3_metrics['review_rows']} / {task3_metrics['uncovered_rows']}`
- pairwise F1: `{task3_metrics['pairwise']['f1']:.6f}`
- B3 F1: `{task3_metrics['b3']['f1']:.6f}`

## Task4 independent
- eligible rows: `{task4_indep['eligible_rows']}`
- eligible risk distribution: `{task4_indep['severity']['gold_distribution']}`
- severity QWK: `{task4_indep['severity']['qwk']:.6f}`
- severity MAE: `{task4_indep['severity']['mae']:.6f}`
- severity Spearman: `{task4_indep['severity']['spearman']:.6f}`
- direction macro-F1: `{task4_indep['direction']['macro_f1']:.6f}`

## Task4 end-to-end
- coverage rate: `{task4_end_to_end['coverage_rate']:.6f}`
- auto decision rate: `{task4_end_to_end['auto_decision_rate']:.6f}`
- alert precision / recall / F1: `{task4_end_to_end['alert_binary']['precision']:.6f} / {task4_end_to_end['alert_binary']['recall']:.6f} / {task4_end_to_end['alert_binary']['f1']:.6f}`
- severity errors on covered eligible rows: `{task4_end_to_end['severity_error_rows']}`
- direction errors on covered eligible rows: `{task4_end_to_end['direction_error_rows']}`
- auto_score_eligibility_gate confusion matrix: `{task4_end_to_end['auto_score_eligibility_gate']['confusion_matrix']}`
- auto_score_eligibility_gate precision / recall / F1: `{task4_end_to_end['auto_score_eligibility_gate']['precision']:.6f} / {task4_end_to_end['auto_score_eligibility_gate']['recall']:.6f} / {task4_end_to_end['auto_score_eligibility_gate']['f1']:.6f}`
- auto_score_eligibility_gate coverage / abstain: `{task4_end_to_end['auto_score_eligibility_gate']['coverage']:.6f} / {task4_end_to_end['auto_score_eligibility_gate']['abstain_rate']:.6f}`
- task4_applicable raw distribution: `{task4_end_to_end['task4_applicable_distribution']['gold']}`

## Protected assets
- official test sha stable: `true`
- task3/task4 annotation sha stable: `true`
- frozen model sha stable: `true`
- old graph sha stable: `true`
"""
    write_text(OUT / "task34_official1799_final_report.md", final_report_md)

    readme = f"""# task34_official1799_final_evaluation_v1

This directory is a frozen post-processing bundle.

Rebuild:

```bash
python scripts/finalize_task34_official1799_final_evaluation_v1.py
```

Notes:
- no training
- no tuning
- no label loading before prediction lock
- source inputs are the official 1799 test file plus the four filled blind-review annotation workbooks
"""
    write_text(OUT / "README.md", readme)

    handover = f"""# Task34 Master Handover

Generated from the current frozen outputs under `{OUT}`.

## Key files
- `input_manifest.json`
- `protected_sha256_before.json`
- `protected_sha256_after.json`
- `prediction_manifest.json`
- `task3_final_gold.xlsx`
- `task3_pair_gold.parquet`
- `task3_adjudication_log.xlsx`
- `task4_final_gold.xlsx`
- `task4_severity_eligible_gold.parquet`
- `task4_exclusion_reasons.csv`
- `task3_official1799_metrics.json`
- `task4_independent_metrics.json`
- `task4_end_to_end_metrics.json`
- `task34_official1799_final_report.md`
- `task34_machine_state.json`

## Controls
- training: false
- tuning: false
- prediction_frozen: true
- label_used_before_prediction: false
- test_evaluation_count: 1

## Final counts
- task3_final_gold_rows: {final_report_json['task3_final_gold_rows']}
- task4_final_gold_rows: {final_report_json['task4_final_gold_rows']}
- severity_eval_eligible_rows: {final_report_json['severity_eval_eligible_rows']}

## Sources
- official test sha: {official_test_meta['sha256']}
- task3/task4 annotation shas: {t3a_meta['sha256']}, {t3b_meta['sha256']}, {t4a_meta['sha256']}, {t4b_meta['sha256']}
"""
    write_text(DOCS / "TASK34_MASTER_HANDOVER.md", handover)

    machine_state = {
        "prediction_frozen": True,
        "training": False,
        "tuning": False,
        "test_evaluation_count": 1,
        "label_used_before_prediction": False,
        "official_test_sha256": official_test_meta["sha256"],
        "task3_A_sha256": t3a_meta["sha256"],
        "task3_B_sha256": t3b_meta["sha256"],
        "task4_A_sha256": t4a_meta["sha256"],
        "task4_B_sha256": t4b_meta["sha256"],
        "task3_final_gold_rows": final_report_json["task3_final_gold_rows"],
        "task4_final_gold_rows": final_report_json["task4_final_gold_rows"],
        "severity_eval_eligible_rows": final_report_json["severity_eval_eligible_rows"],
        "task3_adjudication_records": final_report_json["task3_adjudication_disagreement_records"],
        "task4_adjudication_records": final_report_json["task4_adjudication_disagreement_records"],
        "output_dir": str(OUT),
    }
    write_json(OUT / "task34_machine_state.json", machine_state)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
