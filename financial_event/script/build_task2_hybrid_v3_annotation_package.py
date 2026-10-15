#!/usr/bin/env python3
"""Build the Task2 hybrid-v3 1500-row human annotation package.

The package is derived only from unlabeled candidates that are not in the
locked-500 set and not part of the manual gold inventory. It does not write or
touch any formal Task2 outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_INVENTORY = ROOT / "outputs/event_credibility/task2_hybrid_v3/data_inventory.json"
DEFAULT_LOCKED = ROOT / "configs/task2_independent500_locked_ids.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/event_credibility/task2_hybrid_v3"
DEFAULT_WORKBOOK = ROOT / "reports/task2_hybrid_v3_human_annotation_1500.xlsx"
DEFAULT_GUIDE = ROOT / "reports/task2_hybrid_v3_annotation_guide.md"
DEFAULT_JSONL = DEFAULT_OUTPUT_DIR / "annotation_candidates.jsonl"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "annotation_manifest.json"
DEFAULT_RUN_STATE = DEFAULT_OUTPUT_DIR / "run_state.json"

VISIBLE_HEADERS = [
    "sample_id",
    "target",
    "event_claim",
    "article_title",
    "article_content",
    "publish_time",
    "evidence列表",
    "evidence来源和时间",
    "模型间分歧信息",
    "human_verdict",
    "human_confidence",
    "evidence_sufficient",
    "annotation_reason",
    "adjudication_required",
]

HIDDEN_HEADERS = [
    "sample_id",
    "cluster_id",
    "variant_index",
    "split",
    "priority_score",
    "queue_flag",
    "verification_status",
    "fulltext_evidence_status",
    "credibility_level",
    "credibility_score",
    "fulltext_support_count",
    "fulltext_contradict_count",
    "fulltext_official_support_count",
    "retrieved_article_count",
    "evidence_item_count",
    "selected_evidence_id",
    "selected_article_id",
    "selected_article_title",
    "selected_article_source",
    "selected_article_publish_time",
    "selection_reason",
    "model_disagreement_info",
]

BLANK_HUMAN_FIELDS = {"human_verdict", "human_confidence", "evidence_sufficient", "annotation_reason", "adjudication_required"}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def norm(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def evidence_score(item: dict[str, Any]) -> float:
    try:
        return float(item.get("sentence_score") or item.get("score") or 0.0)
    except Exception:
        return 0.0


def format_evidence_item(item: dict[str, Any]) -> str:
    return " | ".join(
        part
        for part in [
            norm(item.get("evidence_id")) or norm(item.get("article_id")),
            norm(item.get("source")),
            norm(item.get("publish_time")),
            norm(item.get("title")),
        ]
        if part
    )


def format_source_time(item: dict[str, Any]) -> str:
    return " | ".join(
        part for part in [norm(item.get("source")), norm(item.get("publish_time"))] if part
    )


def load_excluded_cluster_ids(inventory_path: Path, locked_path: Path) -> set[str]:
    inventory = read_json(inventory_path)
    excluded: set[str] = set()
    for source in inventory.get("sources", []):
        excluded.update(norm(cid) for cid in source.get("cluster_ids") or [] if norm(cid))
    locked = read_json(locked_path)
    excluded.update(norm(cid) for cid in locked.get("cluster_ids") or [] if norm(cid))
    return excluded


def load_candidate_frame(excluded_ids: set[str], inventory_path: Path, locked_path: Path) -> tuple[pd.DataFrame, dict[str, Any], set[str]]:
    score_path = ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet"
    evidence_path = ROOT / "outputs/event_credibility/fulltext_evidence.parquet"
    queue_path = ROOT / "data/processed/event_credibility/llm_review_queue.jsonl"
    article_path = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"

    scores = pd.read_parquet(
        score_path,
        columns=[
            "cluster_id",
            "target_code",
            "target_name",
            "schema_id",
            "event_name",
            "split",
            "verification_status",
            "fulltext_evidence_status",
            "credibility_level",
            "credibility_score",
            "retrieved_article_count",
            "evidence_sentence_count",
            "fulltext_support_count",
            "fulltext_contradict_count",
            "fulltext_official_support_count",
            "cluster_consistency_score",
            "contradiction_score",
            "rule_credibility_score",
            "rumor_signal_count",
            "reversal_signal_count",
            "clarification_signal_count",
            "hype_signal_count",
            "official_title_signal_count",
            "supporting_reasons",
            "risk_reasons",
            "evidence_items",
        ],
    )
    scores["cluster_id"] = scores["cluster_id"].astype(str)
    scores = scores[~scores["cluster_id"].isin(excluded_ids)].copy()

    queue_ids = {row["cluster_id"] for row in read_jsonl(queue_path)}
    scores["queue_flag"] = scores["cluster_id"].isin(queue_ids)

    # Priority is higher for lower credibility, manual-review status, stronger contradiction signal,
    # and queue membership. This is a deterministic heuristic only.
    verification_weight = {
        "reversed_or_contradicted": 8.0,
        "insufficient_evidence": 6.0,
        "needs_manual_review": 5.0,
        "multi_source_supported": 3.0,
        "official_supported": 2.0,
    }
    level_weight = {
        "疑似虚假或已反转": 8.0,
        "疑似低可信": 5.0,
        "较可信": 2.0,
        "高可信": 0.0,
    }

    def _priority(row: pd.Series) -> float:
        score = 0.0
        score += verification_weight.get(norm(row.get("verification_status")), 0.0)
        score += level_weight.get(norm(row.get("credibility_level")), 0.0)
        try:
            cred = float(row.get("credibility_score") or 0.0)
            score += max(0.0, (45.0 - cred) / 10.0)
        except Exception:
            pass
        score += 2.0 if bool(row.get("queue_flag")) else 0.0
        for field, weight in (
            ("fulltext_contradict_count", 1.0),
            ("reversal_signal_count", 1.5),
            ("clarification_signal_count", 0.6),
            ("hype_signal_count", 0.3),
            ("official_title_signal_count", 0.2),
            ("retrieved_article_count", 0.05),
        ):
            try:
                score += float(row.get(field) or 0.0) * weight
            except Exception:
                continue
        try:
            score += 0.15 * float(row.get("cluster_consistency_score") or 0.0)
        except Exception:
            pass
        return round(score, 6)

    scores["priority_score"] = scores.apply(_priority, axis=1)
    scores = scores.sort_values(["priority_score", "credibility_score", "cluster_id"], ascending=[False, True, True]).reset_index(drop=True)

    articles = pd.read_parquet(article_path, columns=["article_id", "article_title", "article_source", "article_publish_time", "content"])
    article_map = {
        str(row.article_id): {
            "article_title": norm(row.article_title),
            "article_source": norm(row.article_source),
            "article_publish_time": norm(row.article_publish_time),
            "content": norm(row.content),
        }
        for row in articles.itertuples(index=False)
    }

    queue_meta = {row["cluster_id"]: row for row in read_jsonl(queue_path)}
    return scores, article_map, queue_meta


def choose_evidence_items(items: list[dict[str, Any]], variant_index: int = 0) -> list[dict[str, Any]]:
    if not items:
        return []
    sorted_items = sorted(items, key=lambda item: (-evidence_score(item), norm(item.get("publish_time")), norm(item.get("article_id"))))
    if variant_index <= 0:
        return sorted_items
    if variant_index < len(sorted_items):
        rotated = sorted_items[variant_index:] + sorted_items[:variant_index]
        return rotated
    return sorted_items


def build_row(record: pd.Series, article_map: dict[str, dict[str, Any]], variant_index: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    items = choose_evidence_items(parse_evidence_items(record.get("evidence_items")), variant_index=variant_index)
    primary = items[0] if items else {}
    top_items = items[:3] if items else []
    selected_article_id = norm(primary.get("article_id"))
    selected_article = article_map.get(selected_article_id, {})
    target = " / ".join(
        part
        for part in [norm(record.get("target_name")), norm(record.get("target_code"))]
        if part
    )
    event_claim = f"关于{norm(record.get('target_name')) or target}的{norm(record.get('event_name')) or norm(record.get('schema_id'))}相关主张"
    evidence_list = "；".join(format_evidence_item(item) for item in top_items)
    evidence_source_time = "；".join(format_source_time(item) for item in top_items)
    model_info = "；".join(
        part
        for part in [
            f"verification={norm(record.get('verification_status')) or 'unknown'}",
            f"fulltext={norm(record.get('fulltext_evidence_status')) or 'unknown'}",
            f"level={norm(record.get('credibility_level')) or 'unknown'}",
            f"queue={'yes' if bool(record.get('queue_flag')) else 'no'}",
            f"risk={record.get('priority_score', 0):.3f}",
        ]
        if part
    )
    visible = {
        "sample_id": f"{norm(record.get('cluster_id'))}{'__v2' if variant_index else ''}",
        "target": target,
        "event_claim": event_claim,
        "article_title": selected_article.get("article_title") or norm(primary.get("title")),
        "article_content": selected_article.get("content") or norm(primary.get("sentence")),
        "publish_time": selected_article.get("article_publish_time") or norm(primary.get("publish_time")),
        "evidence列表": evidence_list,
        "evidence来源和时间": evidence_source_time,
        "模型间分歧信息": model_info,
        "human_verdict": "",
        "human_confidence": "",
        "evidence_sufficient": "",
        "annotation_reason": "",
        "adjudication_required": "",
    }
    hidden = {
        "sample_id": visible["sample_id"],
        "cluster_id": norm(record.get("cluster_id")),
        "variant_index": variant_index,
        "split": norm(record.get("split")),
        "priority_score": record.get("priority_score", 0.0),
        "queue_flag": bool(record.get("queue_flag")),
        "verification_status": norm(record.get("verification_status")),
        "fulltext_evidence_status": norm(record.get("fulltext_evidence_status")),
        "credibility_level": norm(record.get("credibility_level")),
        "credibility_score": record.get("credibility_score"),
        "fulltext_support_count": record.get("fulltext_support_count"),
        "fulltext_contradict_count": record.get("fulltext_contradict_count"),
        "fulltext_official_support_count": record.get("fulltext_official_support_count"),
        "retrieved_article_count": record.get("retrieved_article_count"),
        "evidence_item_count": len(items),
        "selected_evidence_id": norm(primary.get("evidence_id")),
        "selected_article_id": selected_article_id,
        "selected_article_title": selected_article.get("article_title") or norm(primary.get("title")),
        "selected_article_source": selected_article.get("article_source") or norm(primary.get("source")),
        "selected_article_publish_time": selected_article.get("article_publish_time") or norm(primary.get("publish_time")),
        "selection_reason": "queue_boost" if bool(record.get("queue_flag")) else "risk_ranked",
        "model_disagreement_info": model_info,
    }
    return visible, hidden


def write_sheet(wb: Workbook, title: str, rows: list[dict[str, Any]], hidden: bool = False) -> None:
    ws = wb.create_sheet(title=title[:31])
    if hidden:
        ws.sheet_state = "hidden"
    headers = list(rows[0].keys()) if rows else []
    ws.append(headers)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    auto_fill = PatternFill("solid", fgColor="F2F2F2")
    human_fill = PatternFill("solid", fgColor="FCE4D6")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in rows:
        ws.append([row.get(header, "") for header in headers])
    human_columns = BLANK_HUMAN_FIELDS if not hidden else set()
    for col_idx, header in enumerate(headers, 1):
        fill = human_fill if header in human_columns else auto_fill
        for row_idx in range(2, ws.max_row + 1):
            cell = ws.cell(row_idx, col_idx)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for idx, header in enumerate(headers, 1):
        values = [row.get(header, "") for row in rows[:200]]
        max_len = max([len(str(header))] + [min(len(str(value)), 120) for value in values])
        width = min(max(max_len + 2, 12), 90)
        if header in {"article_content", "evidence列表", "evidence来源和时间", "模型间分歧信息"}:
            width = 70
        ws.column_dimensions[get_column_letter(idx)].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    if not hidden:
        for idx, header in enumerate(headers, 1):
            if header in BLANK_HUMAN_FIELDS:
                dv = DataValidation(type="list", formula1='"support,contradict,insufficient,unclear"', allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{get_column_letter(idx)}2:{get_column_letter(idx)}{max(ws.max_row, 2)}")


def render_guide(summary: dict[str, Any]) -> str:
    lines = [
        "# Task2 Hybrid v3 1500-row Annotation Package",
        "",
        "This workbook is an annotation-only package. The visible review sheet contains blank human fields.",
        "",
        "## Selection",
        "",
        f"- Unique unlabeled cluster pool: {summary['unique_unlabeled_clusters']} rows.",
        f"- Base rows: {summary['base_rows']} unique clusters.",
        f"- Variant rows: {summary['variant_rows']} evidence-slice variants used to reach 1500 review rows.",
        f"- Locked 500 excluded: {summary['locked500_excluded']} clusters.",
        f"- Human-reviewed gold excluded: {summary['human_reviewed_excluded']} clusters.",
        "",
        "## What to fill",
        "",
        "- Fill only the blank human fields in the visible sheet.",
        "- Do not use the hidden audit sheet for annotation.",
        "- Do not infer answers from the model-disagreement summary.",
        "",
        "## Human fields",
        "",
        "| field | allowed values |",
        "|---|---|",
        "| human_verdict | support, contradict, insufficient, unclear |",
        "| human_confidence | high, medium, low |",
        "| evidence_sufficient | yes, no, uncertain |",
        "| annotation_reason | free text |",
        "| adjudication_required | yes, no |",
        "",
        "## Notes",
        "",
        "- The locked 500 independent evaluation set is excluded by cluster id.",
        "- Existing manual gold and diagnostic review sets are excluded by cluster id.",
        "- The visible model-disagreement field is a summary only; exact audit metadata is stored in the hidden sheet.",
        "",
    ]
    return "\n".join(lines) + "\n"


def sha256_manifest(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths if path.exists()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--locked-config", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-state", type=Path, default=DEFAULT_RUN_STATE)
    args = parser.parse_args(argv)

    inventory = read_json(args.inventory)
    locked = read_json(args.locked_config)
    excluded_ids = load_excluded_cluster_ids(args.inventory, args.locked_config)
    scores, article_map, queue_meta = load_candidate_frame(excluded_ids, args.inventory, args.locked_config)

    # We expect 1464 unique clusters in the unlabeled pool. Build 1500 rows by adding
    # 36 evidence-slice variants from the highest-risk clusters with multiple evidence items.
    base_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for record in scores.itertuples(index=False):
        row = record._asdict()
        candidate_rows.append(row)

    for row in candidate_rows:
        visible, hidden = build_row(pd.Series(row), article_map, variant_index=0)
        base_rows.append(visible)
        hidden_rows.append(hidden)

    variant_rows: list[dict[str, Any]] = []
    variant_hidden_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        if len(variant_rows) >= 36:
            break
        items = choose_evidence_items(parse_evidence_items(row.get("evidence_items")), variant_index=1)
        if len(items) < 2:
            continue
        visible, hidden = build_row(pd.Series(row), article_map, variant_index=1)
        variant_rows.append(visible)
        variant_hidden_rows.append(hidden)

    if len(variant_rows) < 36:
        for row in candidate_rows:
            if len(variant_rows) >= 36:
                break
            visible, hidden = build_row(pd.Series(row), article_map, variant_index=2)
            if visible["sample_id"] in {item["sample_id"] for item in base_rows} | {item["sample_id"] for item in variant_rows}:
                continue
            variant_rows.append(visible)
            variant_hidden_rows.append(hidden)

    # Final package rows.
    all_visible = base_rows + variant_rows
    all_hidden = hidden_rows + variant_hidden_rows
    if len(all_visible) != 1500:
        raise RuntimeError(f"expected 1500 rows, got {len(all_visible)}")

    unique_cluster_ids = {row["cluster_id"] for row in all_hidden}
    selected_cluster_ids = [row["cluster_id"] for row in all_hidden]
    split_counts = Counter(row["split"] or "unknown" for row in all_hidden)

    output_rows = []
    for visible, hidden in zip(all_visible, all_hidden):
        row = {**visible}
        row["cluster_id"] = hidden["cluster_id"]
        row["variant_index"] = hidden["variant_index"]
        row["split"] = hidden["split"]
        row["priority_score"] = hidden["priority_score"]
        row["queue_flag"] = hidden["queue_flag"]
        row["verification_status"] = hidden["verification_status"]
        row["fulltext_evidence_status"] = hidden["fulltext_evidence_status"]
        row["credibility_level"] = hidden["credibility_level"]
        row["credibility_score"] = hidden["credibility_score"]
        row["fulltext_support_count"] = hidden["fulltext_support_count"]
        row["fulltext_contradict_count"] = hidden["fulltext_contradict_count"]
        row["selected_article_id"] = hidden["selected_article_id"]
        row["selected_evidence_id"] = hidden["selected_evidence_id"]
        row["selection_reason"] = hidden["selection_reason"]
        output_rows.append(row)

    # Write outputs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workbook.parent.mkdir(parents=True, exist_ok=True)
    args.guide.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)
    intro_rows = [
        {"item": "package", "value": "Task2 Hybrid v3 1500-row annotation package"},
        {"item": "generated_at", "value": now_iso()},
        {"item": "seed", "value": 2026},
        {"item": "base_rows", "value": len(base_rows)},
        {"item": "variant_rows", "value": len(variant_rows)},
        {"item": "unique_cluster_ids", "value": len(unique_cluster_ids)},
        {"item": "human_fields_blank", "value": "yes"},
        {"item": "locked500_excluded", "value": len(locked.get("cluster_ids") or [])},
    ]
    write_sheet(wb, "填写说明", intro_rows, hidden=False)
    write_sheet(wb, "待标注样本", output_rows, hidden=False)
    write_sheet(wb, "模型审计", all_hidden, hidden=True)
    wb.save(args.workbook)

    args.jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": now_iso(),
        "seed": 2026,
        "inventory_source": str(args.inventory.relative_to(ROOT)),
        "locked_config": str(args.locked_config.relative_to(ROOT)),
        "candidate_pool_rows": int(len(scores)),
        "candidate_pool_unique_clusters": int(len(scores["cluster_id"].unique())),
        "unique_unlabeled_clusters": int(len(candidate_rows)),
        "base_rows": len(base_rows),
        "variant_rows": len(variant_rows),
        "total_rows": len(output_rows),
        "unique_sample_ids": len({row["sample_id"] for row in output_rows}),
        "unique_cluster_ids": len(unique_cluster_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "queue_overlap_rows": int(sum(1 for row in all_hidden if row["queue_flag"])),
        "locked500_excluded": len(locked.get("cluster_ids") or []),
        "human_reviewed_excluded": len(set(item["cluster_id"] for source in inventory["sources"] for item in [{"cluster_id": cid} for cid in source.get("cluster_ids") or []])),
        "source_sha256": {
            "inventory": sha256(args.inventory),
            "locked_config": sha256(args.locked_config),
            "score_table": sha256(ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet"),
            "article_index": sha256(ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"),
        },
        "selection_policy": "risk_ranked using verification_status, credibility_level, queue membership, and contradiction signals; 36 evidence-slice variants added to reach 1500 rows",
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    guide_text = render_guide(manifest)
    args.guide.write_text(guide_text, encoding="utf-8")

    run_state = {
        "status": "blocked",
        "blocked_stage": "training_gate_not_met",
        "blocker_reason": "Only 97 unique human-reviewed gold clusters are available outside the locked 500, below the 1200-label gate",
        "completed_stages": ["asset_audit", "locked_id_config", "annotation_package"],
        "pending_command": "human annotation of reports/task2_hybrid_v3_human_annotation_1500.xlsx",
        "no_fake_metrics": True,
    }
    args.run_state.write_text(json.dumps(run_state, ensure_ascii=False, indent=2), encoding="utf-8")

    sha_paths = [
        args.inventory,
        args.locked_config,
        args.jsonl,
        args.manifest,
        args.run_state,
        args.guide,
        args.workbook,
    ]
    sha_txt = "\n".join(f"{sha256(path)}  {path}" for path in sha_paths if path.exists()) + "\n"
    (args.output_dir / "SHA256SUMS.txt").write_text(sha_txt, encoding="utf-8")

    print(json.dumps({"ok": True, "manifest": str(args.manifest), "workbook": str(args.workbook), "jsonl": str(args.jsonl), "guide": str(args.guide), "run_state": str(args.run_state)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
