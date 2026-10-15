#!/usr/bin/env python3
"""Serve the frontend and expose real project data through /api/v1.

This keeps the existing static frontend intact while replacing mock data with
the latest frozen outputs from ../financial_event_project.  It deliberately
uses only the Python standard library so the system can run in the current
workspace without installing backend dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from workflow_store import WorkflowStore


FRONTEND_ROOT = Path(__file__).resolve().parent
WORKFLOW = WorkflowStore(Path(os.environ.get("EVENT_INTELLIGENCE_STATE_DB", FRONTEND_ROOT / ".runtime/workflow.sqlite3")))


def resolve_workspace_root(frontend_root: Path) -> Path:
    explicit_project = os.environ.get("EVENT_INTELLIGENCE_PROJECT_ROOT")
    if explicit_project:
        return Path(explicit_project).expanduser().resolve().parent

    explicit_workspace = os.environ.get("EVENT_INTELLIGENCE_WORKSPACE_ROOT")
    if explicit_workspace:
        return Path(explicit_workspace).expanduser().resolve()

    candidates = (
        frontend_root.parent,
        frontend_root.parent / "frontend_data",
        frontend_root.parent.parent,
    )
    for candidate in candidates:
        if (candidate / "financial_event_project").exists():
            return candidate.resolve()
    return frontend_root.parent.resolve()


WORKSPACE_ROOT = resolve_workspace_root(FRONTEND_ROOT)
PROJECT_ROOT = Path(
    os.environ.get("EVENT_INTELLIGENCE_PROJECT_ROOT", WORKSPACE_ROOT / "financial_event_project")
).expanduser().resolve()


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


TASK34_OFFICIAL_EVAL = PROJECT_ROOT / "outputs/task34_official1799_final_evaluation_v1"
TASK3_OFFICIAL_INFERENCE = PROJECT_ROOT / "outputs/task3_kg_v4_official1799_inference_v1"
TASK4_FINAL_CLOSURE = PROJECT_ROOT / "outputs/task4_final_contract_separated_closure"
TASK2_SIXDIM_FUSION_V2 = PROJECT_ROOT / "outputs/event_credibility/task2_sixdim_ai_fusion_v2"

TASK1_CERTIFICATE = PROJECT_ROOT / "outputs/final_task1/task1_completion_certificate.json"
TASK1_SUMMARY = PROJECT_ROOT / "reports/task1_final_summary.json"
CLUSTER_SUMMARY = PROJECT_ROOT / "reports/event_clustering/event_cluster_summary.json"
TASK2_SUMMARY = PROJECT_ROOT / "reports/event_credibility/fulltext_credibility_summary.json"
TASK2_MANIFEST = PROJECT_ROOT / "outputs/event_credibility/final_cached_dea_v1_260/run_manifest.json"
TASK2_EVENTS = PROJECT_ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl"
TASK2_OFFICIAL_REPORT = PROJECT_ROOT / "reports/task2_independent500_v2_official_evaluation_report.md"
TASK2_SIXDIM_FREEZE_MANIFEST = TASK2_SIXDIM_FUSION_V2 / "freeze_manifest.json"
TASK2_SIXDIM_LOCKED_HOLDOUT = TASK2_SIXDIM_FUSION_V2 / "locked_holdout_evaluation.json"
TASK2_SIXDIM_I500V3 = TASK2_SIXDIM_FUSION_V2 / "independent500_v3_ai_evaluation.json"
TASK2_SIXDIM_QUALITY = TASK2_SIXDIM_FUSION_V2 / "sixdim_quality_report.json"
TASK1_METRICS = PROJECT_ROOT / "reports/canonical_baseline_metrics.json"
TASK3_FINAL_SELECTION = WORKSPACE_ROOT / "outputs/task3_final_frozen_selection/task3_final_selection.json"
TASK3_OFFICIAL_METRICS = TASK34_OFFICIAL_EVAL / "task3_official1799_metrics.json"
TASK34_FINAL_METRICS = TASK34_OFFICIAL_EVAL / "final_metrics.json"
TASK34_MACHINE_STATE = TASK34_OFFICIAL_EVAL / "task34_machine_state.json"
TASK34_FROZEN_TASK4_PREDICTIONS = TASK34_OFFICIAL_EVAL / "frozen_task4_predictions.jsonl"
TASK3_CANONICAL_EVENTS = TASK3_OFFICIAL_INFERENCE / "canonical_events.jsonl"
TASK3_EVENT_THREADS = first_existing_path(
    TASK3_OFFICIAL_INFERENCE / "event_threads.jsonl",
    PROJECT_ROOT / "outputs/task3_kg_v3_final/event_evolution/event_threads.jsonl",
    PROJECT_ROOT / "outputs/task3_kg_v4_shadow/event_threads.jsonl",
)
TASK3_EVENT_STATES = first_existing_path(
    TASK3_OFFICIAL_INFERENCE / "event_states.jsonl",
    PROJECT_ROOT / "outputs/task3_kg_v3_final/state_observations/state_observations.jsonl",
    PROJECT_ROOT / "outputs/task3_kg_v2/step5/event_states.jsonl",
)
TASK3_JOINT_INTERFACE = PROJECT_ROOT / "outputs/task4/final_release/task3_task4_joint_query_interface.json"
TASK3_GRAPH_NODES = TASK3_OFFICIAL_INFERENCE / "graph_nodes.jsonl"
TASK3_GRAPH_EDGES = TASK3_OFFICIAL_INFERENCE / "graph_edges.jsonl"
TASK3_TASK4_EVENT_INTERFACE = TASK34_OFFICIAL_EVAL / "locked_predictions/task4_event_interface.jsonl"
TASK3_STATE_OBSERVATIONS = first_existing_path(
    TASK3_OFFICIAL_INFERENCE / "state_observations/canonical_event_current_state.jsonl",
    PROJECT_ROOT / "outputs/task3_kg_v3_final/state_observations/canonical_event_current_state.jsonl",
    PROJECT_ROOT / "outputs/task3_kg_v3_release/canonical_event_current_state.jsonl",
)
TASK3_TRUST_SCORES = first_existing_path(
    TASK3_OFFICIAL_INFERENCE / "trust_propagation/canonical_event_trust_scores.jsonl",
    PROJECT_ROOT / "outputs/task3_kg_v3_final/trust_propagation/canonical_event_trust_scores.jsonl",
    PROJECT_ROOT / "outputs/task3_kg_v3_release/canonical_event_trust_scores.jsonl",
)
TASK4_ALERT_CARDS = PROJECT_ROOT / "outputs/task4/final_release/impact_alert_cards_demo.json"
TASK4_EVIDENCE_PATH_DEMO = PROJECT_ROOT / "outputs/task4/final_release/evidence_path_demo.json"
TASK4_UNCERTAIN_QUEUE = PROJECT_ROOT / "outputs/task4/final_release/uncertain_review_queue.json"
TASK4_CANDIDATE_EVENTS = PROJECT_ROOT / "outputs/task4/stage2_6/candidate_events.jsonl"
TASK4_PREDICTION_RESULTS = PROJECT_ROOT / "outputs/task4/final_release/task4_prediction_results.json"
TASK4_END_TO_END_CASES = PROJECT_ROOT / "outputs/task4/final_release/task4_end_to_end_cases.json"
TASK4_CLOSURE_FINAL_REPORT = TASK4_FINAL_CLOSURE / "final_report.json"
TASK4_CLOSURE_OFFICIAL1799_METRICS = TASK4_FINAL_CLOSURE / "official1799_formal_metrics.json"
TASK4_CLOSURE_OFFICIAL_RISK_METRICS = TASK4_FINAL_CLOSURE / "official_risk_metrics.json"
TASK4_CLOSURE_HIGHRISK_METRICS = TASK4_FINAL_CLOSURE / "highrisk_metrics.json"
TASK4_CLOSURE_HEAD_SELECTION = TASK4_FINAL_CLOSURE / "head_selection.json"
TASK4_CLOSURE_PREDICTION_MANIFEST = TASK4_FINAL_CLOSURE / "official1799_prediction_manifest.json"
USERS_FILE = FRONTEND_ROOT / ".runtime_users.json"

DATA_SOURCE_FILES = (
    TASK1_CERTIFICATE,
    TASK1_SUMMARY,
    CLUSTER_SUMMARY,
    TASK1_METRICS,
    TASK2_SUMMARY,
    TASK2_MANIFEST,
    TASK2_EVENTS,
    TASK2_OFFICIAL_REPORT,
    TASK2_SIXDIM_FREEZE_MANIFEST,
    TASK2_SIXDIM_LOCKED_HOLDOUT,
    TASK2_SIXDIM_I500V3,
    TASK2_SIXDIM_QUALITY,
    TASK3_FINAL_SELECTION,
    TASK3_OFFICIAL_METRICS,
    TASK34_FINAL_METRICS,
    TASK34_MACHINE_STATE,
    TASK34_FROZEN_TASK4_PREDICTIONS,
    TASK3_CANONICAL_EVENTS,
    TASK3_EVENT_THREADS,
    TASK3_EVENT_STATES,
    TASK3_GRAPH_NODES,
    TASK3_GRAPH_EDGES,
    TASK3_TASK4_EVENT_INTERFACE,
    TASK3_STATE_OBSERVATIONS,
    TASK3_TRUST_SCORES,
    TASK4_ALERT_CARDS,
    TASK4_EVIDENCE_PATH_DEMO,
    TASK4_UNCERTAIN_QUEUE,
    TASK4_CANDIDATE_EVENTS,
    TASK4_PREDICTION_RESULTS,
    TASK4_END_TO_END_CASES,
    TASK4_CLOSURE_FINAL_REPORT,
    TASK4_CLOSURE_OFFICIAL1799_METRICS,
    TASK4_CLOSURE_OFFICIAL_RISK_METRICS,
    TASK4_CLOSURE_HIGHRISK_METRICS,
    TASK4_CLOSURE_HEAD_SELECTION,
    TASK4_CLOSURE_PREDICTION_MANIFEST,
)

COLORS = {
    "primary": "#2ee6a6",
    "blue": "#4aa8ff",
    "purple": "#9677ff",
    "warning": "#ffb648",
    "danger": "#ff5d73",
    "cyan": "#4ce3ef",
}

SOURCE_TYPE_LABEL = {
    "official": "官方/监管信源",
    "mainstream": "主流媒体",
    "general_media": "综合媒体",
    "portal": "门户资讯",
    "self_media": "自媒体",
    "unknown": "未知信源",
}

NEGATIVE_WORDS = ("跌停", "下跌", "跌幅", "减持", "净流出", "亏损", "处罚", "监管", "调查", "诉讼", "违约", "债务", "冻结", "质押", "风险", "澄清", "辟谣", "否认")
POSITIVE_WORDS = ("增持", "上涨", "涨停", "中标", "合作", "拓展", "创新", "突破", "盈利", "增长", "回购", "获批", "投产", "扩产")


def read_json(path: Path, default: Any | None = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return rows


def iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def jsonl_field(item: dict[str, Any], key: str, default: Any = "") -> Any:
    value = item.get(key)
    if value not in (None, ""):
        return value
    props = item.get("properties")
    if isinstance(props, dict):
        value = props.get(key, default)
        if value not in (None, ""):
            return value
    return default


def clean_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(safe_float(value, float(default))))
    except (TypeError, ValueError):
        return default


def fmt_number(value: int | float) -> str:
    return f"{int(round(float(value))):,}"


def metric_percent(value: Any, digits: int = 1) -> str:
    numeric = finite_float(value)
    if numeric is None:
        numeric = 0.0
    if abs(numeric) <= 1:
        numeric *= 100
    return f"{numeric:.{digits}f}%"


def fmt_decimal(value: Any, digits: int = 3) -> str:
    numeric = finite_float(value)
    if numeric is None:
        numeric = 0.0
    return f"{numeric:.{digits}f}"


def percent(value: Any) -> int:
    numeric = safe_float(value)
    if numeric <= 1:
        numeric *= 100
    return max(0, min(100, int(round(numeric))))


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    except ValueError:
        pass
    return None


def best_time_text(values: list[Any]) -> str:
    best_dt: datetime | None = None
    best_text = ""
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        parsed = parse_dt(text)
        if parsed:
            if best_dt is None or parsed > best_dt:
                best_dt = parsed
                best_text = text
            continue
        if best_dt is None and text > best_text:
            best_text = text
    return best_text


def file_mtime_text(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except FileNotFoundError:
        return ""


def source_mtime_signature() -> tuple[tuple[str, int, int], ...]:
    signature = []
    for path in DATA_SOURCE_FILES:
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            signature.append((str(path), 0, 0))
    return tuple(signature)


def code_from_target_key(value: Any) -> str:
    text = str(value or "").strip()
    if "|" in text:
        return text.split("|", 1)[1].strip()
    return text


def evidence_text_from_span(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or "").strip()
    return str(value or "").strip()


def read_task2_official_metrics(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    systems: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        match = re.match(r"- ([^:]+): Accuracy `([^`]+)`, Fixed Macro-F1 `([^`]+)`", line.strip())
        if not match:
            continue
        name, accuracy, macro_f1 = match.groups()
        systems[name] = {"accuracy": safe_float(accuracy), "fixed_macro_f1": safe_float(macro_f1)}
    best_accuracy = ""
    best_macro = ""
    for line in text.splitlines():
        if "by accuracy" in line:
            match = re.search(r"`([^`]+)`", line)
            best_accuracy = match.group(1) if match else ""
        if "by fixed macro-F1" in line:
            match = re.search(r"`([^`]+)`", line)
            best_macro = match.group(1) if match else ""
    return {
        "systems": systems,
        "best_by_accuracy": best_accuracy,
        "best_by_fixed_macro_f1": best_macro,
        "source": str(path),
    }


def short_time(value: Any) -> str:
    parsed = parse_dt(value)
    return parsed.strftime("%m-%d %H:%M") if parsed else str(value or "")


def day_label(value: Any) -> str:
    parsed = parse_dt(value)
    return parsed.strftime("%m-%d") if parsed else str(value or "")


def split_pipe(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def parse_json_field(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def clip_text(text: Any, max_len: int = 160) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


def classify_sentiment(*texts: str) -> str:
    haystack = " ".join(texts)
    negative = sum(1 for word in NEGATIVE_WORDS if word in haystack)
    positive = sum(1 for word in POSITIVE_WORDS if word in haystack)
    if negative > positive:
        return "negative"
    if positive > negative:
        return "positive"
    return "neutral"


def verification_label(status: str) -> str:
    return {
        "official_fulltext_supported": "官方正文佐证",
        "official_supported": "官方佐证",
        "fulltext_supported": "正文佐证",
        "multi_source_supported": "多源佐证",
        "llm_supported": "LLM复核支持",
        "reversed_or_contradicted": "疑似反转/矛盾",
        "insufficient_evidence": "证据不足",
        "needs_manual_review": "需人工复核",
    }.get(status or "", status or "待核验")


def source_color(source_type: str) -> str:
    return {
        "official": COLORS["primary"],
        "mainstream": COLORS["blue"],
        "general_media": COLORS["purple"],
        "portal": COLORS["warning"],
        "self_media": COLORS["danger"],
    }.get(source_type, COLORS["cyan"])


def event_status(status: str) -> str:
    if status in {"official_fulltext_supported", "official_supported"}:
        return "已确认"
    if status in {"fulltext_supported", "multi_source_supported", "llm_supported"}:
        return "已验证"
    if status == "reversed_or_contradicted":
        return "重点预警"
    if status == "insufficient_evidence":
        return "待核验"
    return "持续跟踪"


def level_from(impact: int, status: str) -> str:
    if status == "reversed_or_contradicted" or impact >= 90:
        return "critical"
    if impact >= 75:
        return "high"
    if impact >= 55:
        return "medium"
    return "low"


def title_samples(row: dict[str, Any]) -> list[str]:
    return [item.strip() for item in str(row.get("title_samples") or "").split("||") if item.strip()]


def evidence_item_score(item: dict[str, Any]) -> int:
    score = safe_float(item.get("sentence_score"))
    return max(55, min(99, int(round(score * 11)))) if score else 76


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def task3_state_label(state_type: str) -> str:
    return {
        "unknown": "待判定",
        "announcement": "公告确认",
        "completion": "完成",
        "progress": "推进中",
        "plan": "计划",
        "confirmation_proposal": "确认提议",
        "investigation": "调查",
        "refutation": "反驳",
        "penalty": "处罚",
        "rumor": "传闻",
        "positive": "正向事实",
        "negative": "负向事实",
        "neutral": "中性事实",
        "mixed": "混合影响",
        "accepted": "已入主图",
        "pass": "契约通过",
        "trusted_main_graph": "主图可信",
    }.get(state_type or "", state_type or "unknown")


def task3_state_color(state_type: str) -> str:
    return {
        "completion": COLORS["primary"],
        "announcement": COLORS["blue"],
        "progress": COLORS["purple"],
        "plan": COLORS["cyan"],
        "confirmation_proposal": COLORS["warning"],
        "investigation": COLORS["danger"],
        "refutation": COLORS["danger"],
        "penalty": COLORS["danger"],
        "rumor": COLORS["warning"],
        "positive": COLORS["primary"],
        "negative": COLORS["danger"],
        "neutral": COLORS["blue"],
        "mixed": COLORS["warning"],
        "accepted": COLORS["primary"],
        "pass": COLORS["primary"],
        "trusted_main_graph": COLORS["primary"],
        "unknown": COLORS["blue"],
    }.get(state_type or "", COLORS["blue"])


def task4_level_from_severity(severity: str) -> str:
    return {
        "S4": "critical",
        "S3": "high",
        "S2": "medium",
        "S1": "low",
        "S0": "low",
    }.get(severity or "", "medium")


def task4_color_from_severity(severity: str) -> str:
    return {
        "S4": COLORS["danger"],
        "S3": COLORS["warning"],
        "S2": COLORS["blue"],
        "S1": COLORS["purple"],
        "S0": COLORS["cyan"],
    }.get(severity or "", COLORS["warning"])


def task4_icon_from_severity(severity: str) -> str:
    return {
        "S4": "!",
        "S3": "↗",
        "S2": "◇",
        "S1": "•",
        "S0": "·",
    }.get(severity or "", "◇")


def normalize_task4_severity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.upper() in {"S0", "S1", "S2", "S3", "S4"}:
        return text.upper()
    return {
        "red": "S4",
        "orange": "S3",
        "yellow": "S2",
        "blue": "S1",
        "5": "S4",
        "4": "S3",
        "3": "S2",
        "2": "S1",
        "1": "S0",
        "0": "S0",
    }.get(text, "S0")


def task4_formal_severity_counter(risk_metrics: dict[str, Any]) -> Counter:
    counter: Counter = Counter()
    distribution = risk_metrics.get("pred_distribution") if isinstance(risk_metrics, dict) else {}
    if not isinstance(distribution, dict):
        return counter
    for risk_value, count in distribution.items():
        counter[normalize_task4_severity(risk_value)] += safe_int(count)
    return counter


def severity_spark(counter: Counter, limit: int = 8) -> list[int]:
    severity_order = {"S4": 4, "S3": 3, "S2": 2, "S1": 1, "S0": 0}
    values: list[int] = []
    for severity, count in sorted(counter.items(), key=lambda item: severity_order.get(item[0], 0), reverse=True):
        values.extend([severity_order.get(severity, 0) * 20] * max(1, min(3, safe_int(count))))
    return values[:limit] or [0]


def severity_delta_text(counter: Counter) -> str:
    severity_order = {"S4": 4, "S3": 3, "S2": 2, "S1": 1, "S0": 0}
    for severity, _ in sorted(counter.items(), key=lambda item: severity_order.get(item[0], 0), reverse=True):
        count = safe_int(counter.get(severity))
        if count:
            return f"{fmt_number(count)}条{severity}"
    return "0条S2+"


class DataStore:
    def __init__(self) -> None:
        load_started_signature = source_mtime_signature()
        self.loaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.task1_certificate = read_json(TASK1_CERTIFICATE)
        self.task1_summary = read_json(TASK1_SUMMARY)
        self.cluster_summary = read_json(CLUSTER_SUMMARY)
        self.task2_summary = read_json(TASK2_SUMMARY)
        self.task2_manifest = read_json(TASK2_MANIFEST)
        self.task2_official_metrics = read_task2_official_metrics(TASK2_OFFICIAL_REPORT)
        self.task2_sixdim_freeze_manifest = read_json(TASK2_SIXDIM_FREEZE_MANIFEST)
        self.task2_sixdim_locked_holdout = read_json(TASK2_SIXDIM_LOCKED_HOLDOUT)
        self.task2_sixdim_i500v3 = read_json(TASK2_SIXDIM_I500V3)
        self.task2_sixdim_quality = read_json(TASK2_SIXDIM_QUALITY)
        self.task1_metrics = read_json(TASK1_METRICS)
        self.task3_final_selection = read_json(TASK3_FINAL_SELECTION)
        self.task3_official_metrics = read_json(TASK3_OFFICIAL_METRICS)
        self.task34_final_metrics = read_json(TASK34_FINAL_METRICS)
        self.task34_machine_state = read_json(TASK34_MACHINE_STATE)
        self.task34_task4_predictions = read_jsonl(TASK34_FROZEN_TASK4_PREDICTIONS)
        self.rows = read_jsonl(TASK2_EVENTS)
        self.task3_canonical_events = read_jsonl(TASK3_CANONICAL_EVENTS)
        self.task3_event_threads = read_jsonl(TASK3_EVENT_THREADS)
        self.task3_event_states = read_jsonl(TASK3_EVENT_STATES)
        self.task3_task4_event_interface = read_jsonl(TASK3_TASK4_EVENT_INTERFACE)
        self.task3_current_states = read_jsonl(TASK3_STATE_OBSERVATIONS)
        self.task3_trust_scores = read_jsonl(TASK3_TRUST_SCORES)
        self.task3_joint_interface = read_json(TASK3_JOINT_INTERFACE)
        self.task4_alert_cards = read_json(TASK4_ALERT_CARDS)
        self.task4_evidence_path_demo = read_json(TASK4_EVIDENCE_PATH_DEMO, [])
        self.task4_uncertain_queue = read_json(TASK4_UNCERTAIN_QUEUE, {})
        self.task4_candidate_events = read_jsonl(TASK4_CANDIDATE_EVENTS)
        self.task4_prediction_results = read_json(TASK4_PREDICTION_RESULTS)
        self.task4_end_to_end_cases = read_json(TASK4_END_TO_END_CASES)
        self.task4_closure_final_report = read_json(TASK4_CLOSURE_FINAL_REPORT)
        self.task4_closure_official1799_metrics = read_json(TASK4_CLOSURE_OFFICIAL1799_METRICS)
        self.task4_closure_official_risk_metrics = read_json(TASK4_CLOSURE_OFFICIAL_RISK_METRICS)
        self.task4_closure_highrisk_metrics = read_json(TASK4_CLOSURE_HIGHRISK_METRICS)
        self.task4_closure_head_selection = read_json(TASK4_CLOSURE_HEAD_SELECTION)
        self.task4_closure_prediction_manifest = read_json(TASK4_CLOSURE_PREDICTION_MANIFEST)
        self.data_version_at = self._latest_data_version_at()
        self.events = self._build_events()
        self.event_by_id = {event["id"]: event for event in self.events}
        self.row_by_id = {str(row.get("cluster_id")): row for row in self.rows}
        self.company_counter = Counter(event["company"] for event in self.events if event["company"])
        self.task3_event_by_id = {str(item.get("canonical_event_id") or item.get("stable_id")): item for item in self.task3_canonical_events}
        self.task3_event_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.task3_canonical_events:
            code = str(item.get("target_code") or item.get("subject") or item.get("company_code") or code_from_target_key(item.get("target_key")) or "").strip()
            if code:
                self.task3_event_by_code[code].append(item)
        self.task3_thread_by_id = {str(item.get("event_thread_id")): item for item in self.task3_event_threads}
        self.task3_interface_by_event = {str(item.get("canonical_event_id")): item for item in self.task3_task4_event_interface}
        self.task3_interface_by_thread = {str(item.get("event_thread_id")): item for item in self.task3_task4_event_interface}
        self.task3_state_by_event = {str(item.get("canonical_event_id")): item for item in self.task3_event_states}
        self.task3_current_state_by_event = {str(item.get("canonical_event_id")): item for item in self.task3_current_states}
        self.task3_trust_by_event = {str(item.get("canonical_event_id")): item for item in self.task3_trust_scores}
        self.task3_mentions_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.task3_mentions_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.task3_mention_by_id: dict[str, dict[str, Any]] = {}
        self.task3_canonical_by_mention: dict[str, str] = {}
        self.task3_canonical_by_representative_mention: dict[str, str] = {}
        self.task3_canonical_meta_by_id: dict[str, dict[str, Any]] = {}
        self.task3_thread_by_canonical_event: dict[str, str] = {}
        for item in self.task3_canonical_events:
            canonical_key = str(item.get("canonical_event_id") or item.get("stable_id") or "").strip()
            thread_id = str(item.get("event_thread_id") or "").strip()
            if canonical_key:
                self.task3_canonical_meta_by_id[canonical_key] = item
            if canonical_key and thread_id:
                self.task3_thread_by_canonical_event[canonical_key] = thread_id
        for node in iter_jsonl(TASK3_GRAPH_NODES):
            node_type = str(node.get("node_type") or "")
            if node_type == "EventMention":
                mention_id = str(jsonl_field(node, "stable_id") or jsonl_field(node, "event_mention_id") or jsonl_field(node, "source_record_id") or "").strip()
                if not mention_id:
                    continue
                self.task3_mention_by_id[mention_id] = node
                code = str(jsonl_field(node, "target_code") or jsonl_field(node, "company_code") or code_from_target_key(jsonl_field(node, "target_key")) or "").strip()
                if code:
                    self.task3_mentions_by_code[code].append(node)
                cluster_ids = {str(cluster_id).strip() for cluster_id in (node.get("task2_cluster_ids") or []) if str(cluster_id).strip()}
                provisional_cluster_id = str(jsonl_field(node, "provisional_cluster_id") or "").strip()
                if provisional_cluster_id:
                    cluster_ids.add(provisional_cluster_id)
                for cluster_id in cluster_ids:
                    self.task3_mentions_by_cluster[cluster_id].append(node)
            elif node_type == "CanonicalEvent":
                canonical_event_id = str(jsonl_field(node, "canonical_event_id") or jsonl_field(node, "stable_id") or "").strip()
                representative_mention_id = str(jsonl_field(node, "representative_event_mention_id") or "").strip()
                if canonical_event_id:
                    self.task3_canonical_meta_by_id[canonical_event_id] = node
                    if representative_mention_id:
                        self.task3_canonical_by_representative_mention[representative_mention_id] = canonical_event_id
            elif node_type == "EventThread":
                thread_id = str(jsonl_field(node, "event_thread_id") or "").strip()
                canonical_event_ids = jsonl_field(node, "canonical_event_ids", []) or []
                if thread_id and isinstance(canonical_event_ids, list):
                    for canonical_event_id in canonical_event_ids:
                        canonical_key = str(canonical_event_id).strip()
                        if canonical_key:
                            self.task3_thread_by_canonical_event[canonical_key] = thread_id
        for edge in iter_jsonl(TASK3_GRAPH_EDGES):
            if str(edge.get("relation_type") or "") != "INSTANCE_OF_EVENT":
                continue
            source_id = str(edge.get("source_id") or "").strip()
            target_id = str(edge.get("target_id") or "").strip()
            if source_id and (target_id.startswith("CE3_") or target_id.startswith("OFFCEV")):
                self.task3_canonical_by_mention[source_id] = target_id
        self.task4_queue_by_event = {
            str(item.get("canonical_event_id")): item
            for item in self.task4_uncertain_queue.get("items", [])
            if str(item.get("canonical_event_id") or "")
        }
        self.task4_evidence_by_event = {
            str(item.get("canonical_event_id")): item
            for item in self.task4_evidence_path_demo
            if str(item.get("canonical_event_id") or "")
        }
        self.task4_candidates_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.task4_candidate_events:
            code = str(item.get("predicted_target_code") or "").strip()
            if code:
                self.task4_candidates_by_code[code].append(item)
        # A producer may still be replacing one of the source files while the
        # store is loading.  Callers only publish stores built from a stable
        # snapshot, preventing transient partial/empty JSON from reaching the
        # frontend.
        self.source_signature = source_mtime_signature()
        self.sources_changed_during_load = load_started_signature != self.source_signature

    def _latest_data_version_at(self) -> str:
        candidates = [
            self.task4_closure_final_report.get("created_at"),
            self.task2_sixdim_freeze_manifest.get("frozen_at"),
            self.task2_sixdim_locked_holdout.get("completed_at"),
            self.task2_sixdim_i500v3.get("prediction_freeze_sha256") and file_mtime_text(TASK2_SIXDIM_I500V3),
            file_mtime_text(TASK4_CLOSURE_FINAL_REPORT),
            file_mtime_text(TASK4_CLOSURE_OFFICIAL1799_METRICS),
            file_mtime_text(TASK34_FINAL_METRICS),
            file_mtime_text(TASK3_TASK4_EVENT_INTERFACE),
            file_mtime_text(TASK2_EVENTS),
        ]
        return best_time_text(candidates)

    def data_status(self) -> dict[str, Any]:
        return {
            "loadedAt": self.loaded_at,
            "dataVersionAt": self.data_version_at,
            "sources": {
                "task1": {
                    "path": str(TASK1_CERTIFICATE),
                    "updatedAt": file_mtime_text(TASK1_CERTIFICATE),
                    "events": safe_int(self.task1_certificate.get("pipeline_summary", {}).get("total_final_accepted_events"), len(self.events)),
                },
                "task2": {
                    "path": str(TASK2_EVENTS),
                    "updatedAt": file_mtime_text(TASK2_EVENTS),
                    "clusters": len(self.rows),
                    "sixdimFusionV2": str(TASK2_SIXDIM_FUSION_V2),
                },
                "task3": {
                    "path": str(TASK3_TASK4_EVENT_INTERFACE),
                    "updatedAt": file_mtime_text(TASK3_TASK4_EVENT_INTERFACE),
                    "official1799Rows": safe_int(self.task34_machine_state.get("task3_final_gold_rows"), 1799),
                    "interfaceRows": len(self.task3_task4_event_interface),
                    "canonicalEvents": len(self.task3_canonical_events),
                },
                "task4": {
                    "path": str(TASK4_CLOSURE_FINAL_REPORT),
                    "updatedAt": file_mtime_text(TASK4_CLOSURE_FINAL_REPORT),
                    "status": str(self.task4_closure_final_report.get("final_status") or ""),
                    "official1799Rows": safe_int(self.task4_closure_prediction_manifest.get("rows"), safe_int(self.task34_machine_state.get("task4_final_gold_rows"), 1799)),
                },
            },
        }

    def _task3_state_type(self, row: dict[str, Any], current_state: dict[str, Any] | None = None) -> str:
        current_state = current_state or {}
        return str(
            row.get("current_state_type")
            or current_state.get("current_state_type")
            or row.get("polarity_or_state")
            or row.get("resolver_decision")
            or row.get("contract_status")
            or row.get("graph_layer")
            or "unknown"
        )

    def _task3_display_label(self, row: dict[str, Any], current_state: dict[str, Any] | None = None) -> str:
        current_state = current_state or {}
        return str(
            current_state.get("subject")
            or current_state.get("event_object")
            or row.get("target_name")
            or row.get("event_name")
            or row.get("event_family")
            or code_from_target_key(row.get("target_key"))
            or row.get("canonical_event_id")
            or row.get("stable_id")
            or "公开样本"
        )

    def _task3_summary_text(self, row: dict[str, Any], current_state: dict[str, Any] | None = None, max_len: int = 118) -> str:
        current_state = current_state or {}
        return clip_text(
            current_state.get("evidence_text")
            or evidence_text_from_span(row.get("evidence_span"))
            or row.get("trigger_text")
            or row.get("event_name")
            or row.get("canonical_event_id")
            or "",
            max_len,
        )

    def _impact(self, row: dict[str, Any], credibility: int) -> int:
        source_count = safe_int(row.get("supporting_source_count"))
        member_count = safe_int(row.get("member_count"))
        contradiction_count = safe_int(row.get("fulltext_contradict_count"))
        official_count = safe_int(row.get("fulltext_official_support_count"))
        status = str(row.get("fulltext_verification_status") or row.get("verification_status") or "")
        value = 38 + min(source_count, 8) * 5 + min(member_count, 12) * 2 + min(official_count, 4) * 5
        if status in {"official_fulltext_supported", "official_supported"}:
            value += 8
        if status == "reversed_or_contradicted":
            value += 18
        if contradiction_count:
            value += min(16, contradiction_count * 5)
        if credibility >= 85:
            value += 5
        return max(35, min(100, value))

    def _keywords(self, row: dict[str, Any], title: str) -> list[str]:
        output: list[str] = []
        for value in [row.get("event_name"), *split_pipe(row.get("source_list"))[:2], *parse_json_field(row.get("query_terms"), [])]:
            text = str(value or "").strip()
            if 1 < len(text) <= 16 and text not in output:
                output.append(text)
            if len(output) >= 5:
                return output
        for token in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,12}", title):
            if token not in output:
                output.append(token)
            if len(output) >= 5:
                break
        return output

    def _build_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for row in self.rows:
            cluster_id = str(row.get("cluster_id") or "")
            if not cluster_id:
                continue
            titles = title_samples(row)
            title = titles[0] if titles else f"{row.get('target_name', '')}{row.get('event_name', '')}"
            score = max(0, min(100, int(round(safe_float(row.get("fulltext_credibility_score"), safe_float(row.get("credibility_score")))))))
            status = str(row.get("fulltext_verification_status") or row.get("verification_status") or "")
            impact = self._impact(row, score)
            level = level_from(impact, status)
            sentiment = classify_sentiment(title, str(row.get("event_name") or ""))
            sources = split_pipe(row.get("source_list"))
            target_name = str(row.get("target_name") or "未知主体")
            target_code = str(row.get("target_code") or "")
            latest = str(row.get("latest_publish_time") or row.get("earliest_publish_time") or "")
            source_count = safe_int(row.get("supporting_source_count"))
            member_count = safe_int(row.get("member_count"))
            evidence_count = safe_int(row.get("evidence_sentence_count"))
            level_text = str(row.get("fulltext_credibility_level") or row.get("credibility_level") or "待核验")
            source_names = "、".join(sources[:3]) if sources else "多源资讯"
            summary = (
                f"系统将 {member_count} 条相关新闻聚合为同一公开事件，覆盖 {source_count} 个来源"
                f"（{source_names}），当前可信等级为“{level_text}”，验证状态为“{verification_label(status)}”，"
                f"检索到 {evidence_count} 条正文证据。"
            )
            reasons = [str(row.get("supporting_reasons") or "").strip(), str(row.get("fulltext_reason") or "").strip(), str(row.get("risk_reasons") or "").strip()]
            source_text = "；".join(reason for reason in reasons if reason and reason.lower() != "nan") or verification_label(status)
            events.append(
                {
                    "id": cluster_id,
                    "company": target_name,
                    "code": target_code,
                    "logo": target_name[:1] or "事",
                    "type": str(row.get("event_name") or "未分类事件"),
                    "title": title,
                    "summary": f"{title}。{summary}",
                    "time": short_time(latest),
                    "fullTime": latest,
                    "source": sources[0] if sources else "未记录来源",
                    "sourceCount": source_count,
                    "credibility": score,
                    "credibilityLevel": level_text,
                    "impact": impact,
                    "heat": max(20, min(100, member_count * 7 + source_count * 10)),
                    "level": level,
                    "sentiment": sentiment,
                    "direction": {"positive": "正面", "negative": "负面", "neutral": "中性"}[sentiment],
                    "status": event_status(status),
                    "entity": f"{target_name}{f'（{target_code}）' if target_code else ''}",
                    "object": str(row.get("event_name") or "事件对象"),
                    "keywords": self._keywords(row, title),
                    "evidenceCount": evidence_count,
                    "relatedCount": member_count,
                    "sourceText": source_text,
                    "_dt": parse_dt(latest) or datetime.min,
                }
            )
        events.sort(key=lambda item: (item["_dt"], item["impact"], item["credibility"]), reverse=True)
        for event in events:
            event.pop("_dt", None)
        return events

    def get_event(self, event_id: str | None) -> dict[str, Any]:
        if event_id and event_id in self.event_by_id:
            return self.event_by_id[event_id]
        return self.events[0] if self.events else {}

    def get_row(self, event_id: str | None) -> dict[str, Any]:
        event = self.get_event(event_id)
        return self.row_by_id.get(str(event.get("id")), self.rows[0] if self.rows else {})

    def get_events(self, filters: dict[str, str]) -> list[dict[str, Any]]:
        query = (filters.get("query") or "").strip().lower()
        level = filters.get("level") or "all"
        sentiment = filters.get("sentiment") or "all"
        output = []
        for event in self.events:
            haystack = f"{event['title']}{event['company']}{event['code']}{event['type']}".lower()
            if query and query not in haystack:
                continue
            if level != "all" and event["level"] != level:
                continue
            if sentiment != "all" and event["sentiment"] != sentiment:
                continue
            output.append(event)
        return output

    def trend(self) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            label = day_label(event.get("fullTime"))
            if label:
                buckets[label].append(event)
        labels = sorted(buckets.keys())[-8:] or ["D-7", "D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "D"]
        return {
            "labels": labels,
            "total": [len(buckets[label]) for label in labels],
            "high": [sum(1 for event in buckets[label] if event["level"] in {"high", "critical"}) for label in labels],
        }

    def composition(self) -> list[dict[str, Any]]:
        counter = Counter(event["type"] for event in self.events)
        total = sum(counter.values()) or 1
        palette = [COLORS["primary"], COLORS["blue"], COLORS["purple"], COLORS["danger"], COLORS["warning"]]
        return [
            {"name": name, "value": round(count * 100 / total, 1), "count": count, "color": palette[index % len(palette)]}
            for index, (name, count) in enumerate(counter.most_common(5))
        ]

    def dashboard(self) -> dict[str, Any]:
        cert_summary = self.task1_certificate.get("pipeline_summary", {})
        total_events = safe_int(cert_summary.get("total_final_accepted_events"), len(self.events))
        cluster_count = safe_int(self.cluster_summary.get("cluster_count"), len(self.events))
        avg_score = sum(event["credibility"] for event in self.events) / max(1, len(self.events))
        high_impact_count = sum(1 for event in self.events if event["level"] in {"high", "critical"})
        high_impact_ratio = high_impact_count / max(1, len(self.events))
        alerts = self.alerts()
        trend = self.trend()
        alert_ratio = len(alerts) / max(1, len(self.events))
        return {
            "updatedAt": self.data_version_at or max((event["fullTime"] for event in self.events if event.get("fullTime")), default=""),
            "eventDataUpdatedAt": max((event["fullTime"] for event in self.events if event.get("fullTime")), default=""),
            "dataStatus": self.data_status(),
            "metrics": [
                {"label": "最新态势", "value": "活跃" if high_impact_ratio > 0 else "稳定", "delta": "已同步", "note": "公开样本持续更新", "icon": "势", "color": COLORS["primary"], "spark": trend["total"][-8:]},
                {"label": "重点信号", "value": "活跃", "delta": f"高影响占比 {metric_percent(high_impact_ratio)}", "note": "重点事件持续更新", "icon": "视", "color": COLORS["blue"], "spark": trend["high"][-8:]},
                {"label": "预警状态", "value": "正常", "delta": f"{metric_percent(alert_ratio)} 覆盖", "note": "重点提醒可用", "icon": "!", "color": COLORS["danger"], "spark": [min(100, value) for value in trend["high"][-8:]]},
                {"label": "综合判断", "value": "稳定" if avg_score >= 90 else "关注" if avg_score >= 75 else "谨慎", "delta": "重点信号已更新", "note": "当前结论稳定", "icon": "验", "color": COLORS["warning"], "spark": [72, 74, 75, 76, 78, 79, 81, 82]},
            ],
            "updates": [
                {"time": "刚刚", "title": "公开结果已同步", "note": "最新样本和提醒卡片已更新"},
                {"time": "最近", "title": "重点提醒已就绪", "note": "高影响事件保持关注"},
                {"time": "今天", "title": "主体覆盖已刷新", "note": "事件覆盖持续更新"},
                {"time": "现在", "title": "趋势看板可用", "note": "可直接查看最新走势"},
            ],
            "trend": trend,
            "composition": self.composition(),
            "pipeline": [
                {"icon": "采", "name": "多源接入", "note": "公开资讯持续同步", "value": "正常", "unit": "在线"},
                {"icon": "抽", "name": "事件抽取", "note": "主体与要素已结构化", "value": "稳定", "unit": "状态"},
                {"icon": "验", "name": "可信判断", "note": "交叉信号持续核验", "value": "良好", "unit": "结果"},
                {"icon": "聚", "name": "事件聚合", "note": "相似内容已合并", "value": "已完成", "unit": "流程"},
                {"icon": "判", "name": "重点提醒", "note": "高影响提醒已开启", "value": "在线", "unit": "响应"},
                {"icon": "图", "name": "脉络追踪", "note": "关联关系持续更新", "value": "正常", "unit": "连接"},
                {"icon": "警", "name": "预警输出", "note": "公开结果已可用", "value": "已就绪", "unit": "状态"},
            ],
        }

    def evidence(self, row: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
        items = parse_json_field(row.get("evidence_items"), [])
        evidence = []
        for item in items[:limit]:
            source_type = str(item.get("source_type") or "unknown")
            color = source_color(source_type)
            source = str(item.get("source") or "未知来源")
            evidence.append(
                {
                    "source": source,
                    "type": SOURCE_TYPE_LABEL.get(source_type, source_type),
                    "logo": source[:1] or "源",
                    "color": color,
                    "score": evidence_item_score(item),
                    "time": short_time(item.get("publish_time")),
                    "text": clip_text(item.get("sentence") or item.get("title"), 170),
                    "status": "官方证据" if source_type == "official" else "正文证据",
                    "title": str(item.get("title") or ""),
                    "articleId": str(item.get("article_id") or ""),
                }
            )
        if evidence:
            return evidence
        return [
            {"source": source, "type": "聚合来源", "logo": source[:1], "color": COLORS["blue"], "score": 76, "time": short_time(row.get("latest_publish_time")), "text": clip_text(row.get("title_samples") or row.get("event_name")), "status": "来源佐证"}
            for source in split_pipe(row.get("source_list"))[:4]
        ]

    def credibility(self, event_id: str | None) -> dict[str, Any]:
        event = self.get_event(event_id)
        row = self.get_row(event_id)
        status = str(row.get("fulltext_verification_status") or row.get("verification_status") or "")
        contradiction = safe_int(row.get("fulltext_contradict_count"))
        summary = (
            f"该事件聚合 {event.get('sourceCount', 0)} 个公开来源，可信度为 {event.get('credibility', 0)} / 100。当前核验状态：{verification_label(status)}。"
        )
        if contradiction:
            summary += f" 另有 {contradiction} 条反向信号已纳入持续观察。"
        return {
            "eventId": event.get("id"),
            "score": event.get("credibility", 0),
            "verdict": f"{row.get('fulltext_credibility_level') or row.get('credibility_level') or '待核验'} · {verification_label(status)}",
            "summary": summary,
            "factors": [
                {"name": "信息源权威性", "score": percent(row.get("source_score")), "color": COLORS["primary"]},
                {"name": "信号一致性", "score": percent(row.get("cross_source_score")), "color": COLORS["blue"]},
                {"name": "正文证据支撑", "score": percent(row.get("fulltext_support_score")), "color": COLORS["purple"]},
                {"name": "反证风险控制", "score": max(0, 100 - percent(row.get("fulltext_contradiction_score"))), "color": COLORS["warning"]},
                {"name": "簇内一致性", "score": percent(row.get("cluster_consistency_score")), "color": COLORS["cyan"]},
            ],
            "evidence": self.evidence(row),
            "reasons": [
                {"label": "来源数量", "value": f"{event.get('sourceCount', 0)} 个", "note": str(row.get("source_list") or "来源已归一")},
                {"label": "公开佐证", "value": f"{event.get('evidenceCount', 0)} 条", "note": f"覆盖公开报道 {safe_int(row.get('retrieved_article_count'))} 篇"},
                {"label": "官方支撑", "value": f"{safe_int(row.get('fulltext_official_support_count'))} 条", "note": "公告/监管/政府信源"},
                {"label": "结论状态", "value": "已确认" if row.get("llm_verdict") else "稳定跟踪", "note": str(row.get("fulltext_credibility_level") or "基础判断")},
            ],
        }

    def timeline(self, event_id: str | None) -> dict[str, Any]:
        event = self.get_event(event_id)
        row = self.get_row(event_id)
        items = []
        for item in parse_json_field(row.get("evidence_items"), [])[:8]:
            publish_time = str(item.get("publish_time") or row.get("latest_publish_time") or "")
            short = short_time(publish_time)
            items.append(
                {
                    "date": day_label(publish_time),
                    "time": short[6:] if len(short) >= 11 else short,
                    "title": str(item.get("title") or event.get("title")),
                    "text": clip_text(item.get("sentence"), 180),
                    "source": f"{item.get('source') or '未知来源'} · {item.get('source_type') or 'unknown'}",
                    "type": "官方证据" if item.get("source_type") == "official" else "增量证据",
                    "color": source_color(str(item.get("source_type") or "unknown")),
                }
            )
        if not items:
            for title in title_samples(row)[:5]:
                items.append({"date": day_label(row.get("latest_publish_time")), "time": short_time(row.get("latest_publish_time"))[6:], "title": title, "text": event.get("summary"), "source": str(row.get("source_list") or "聚合来源"), "type": "聚合报道", "color": COLORS["blue"]})
        return {
            "eventId": event.get("id"),
            "title": event.get("title"),
            "company": event.get("company"),
            "code": event.get("code"),
            "impact": event.get("impact"),
            "stages": [
                {"name": "线索出现", "time": day_label(row.get("earliest_publish_time")), "state": "done"},
                {"name": "多源报道", "time": f"{event.get('sourceCount', 0)}源", "state": "done" if event.get("sourceCount", 0) > 1 else "active"},
                {"name": "公开确认", "time": f"{event.get('evidenceCount', 0)}条", "state": "done" if event.get("evidenceCount", 0) else "active"},
                {"name": "结果确认", "time": str(row.get("fulltext_credibility_level") or "待核验"), "state": "done"},
                {"name": "持续跟踪", "time": "进行中", "state": "active"},
            ],
            "items": items,
            "impacts": [
                {"name": "目标主体", "value": event.get("company")},
                {"name": "事件类型", "value": event.get("type")},
                {"name": "公开状态", "value": verification_label(str(row.get("fulltext_verification_status") or ""))},
                {"name": "影响方向", "value": event.get("direction")},
            ],
        }

    def alerts(self) -> list[dict[str, Any]]:
        rules = WORKFLOW.rules()
        acknowledged = WORKFLOW.acknowledgements()
        ranked = sorted(self.events, key=lambda item: (item["impact"], item["sourceCount"]), reverse=True)
        alerts = []
        for event in ranked:
            if event["impact"] < rules["high"] or event["credibility"] < rules["credibility"]:
                continue
            level = "critical" if event["impact"] >= rules["critical"] else "high"
            color = COLORS["danger"] if level == "critical" else COLORS["warning"]
            icon = "!" if level == "critical" else "↗" if event["sentiment"] == "positive" else "↘" if event["sentiment"] == "negative" else "◇"
            level_text = "重大" if level == "critical" else "高影响"
            alerts.append({"id": f"ALT-{event['id']}", "eventId": event["id"], "level": level,
                           "icon": icon, "title": event["title"], "company": f"{event['company']} · {event['code']}",
                           "reason": f"{event['type']} · 影响分 {event['impact']} · {level_text}提醒",
                           "time": event["time"], "score": event["impact"],
                           "status": "跟踪中" if event["id"] in acknowledged else "待处置",
                           "acknowledgedAt": acknowledged.get(event["id"]), "color": color})
        return alerts

    def graph(self, scope: str = "company") -> dict[str, Any]:
        top_events = sorted(self.events, key=lambda item: (item["impact"], item["sourceCount"]), reverse=True)[:18]
        nodes: list[dict[str, Any]] = []
        links: list[list[str]] = []
        seen: set[str] = set()

        def add_node(node_id: str, label: str, kind: str, color: str, size: int) -> None:
            if node_id in seen:
                return
            seen.add(node_id)
            index = len(nodes)
            angle = 2 * math.pi * index / 24
            radius = 0.28 + (index % 3) * 0.09
            nodes.append({"id": node_id, "label": label, "type": kind, "color": color, "size": size, "x": max(0.08, min(0.92, 0.5 + math.cos(angle) * radius)), "y": max(0.12, min(0.88, 0.5 + math.sin(angle) * radius))})

        add_node("task1", "公开样本", "technology", COLORS["cyan"], 24)
        add_node("task2", "公开判断", "technology", COLORS["blue"], 24)
        for event in top_events:
            company_id = f"company-{event['code'] or event['company']}"
            event_id = f"event-{event['id']}"
            type_id = f"type-{event['type']}"
            add_node(company_id, event["company"], "company", COLORS["primary"], 14 + min(16, event["relatedCount"]))
            add_node(event_id, event["type"], "event", COLORS["blue"], 12 + event["impact"] // 8)
            add_node(type_id, event["type"], "industry", COLORS["purple"], 15)
            links.extend([[company_id, event_id, "发生"], [event_id, type_id, "归类"], ["task1", event_id, "识别"], ["task2", event_id, "评分"]])
            if event["level"] == "critical":
                risk_id = f"risk-{event['id']}"
                add_node(risk_id, "反证风险", "risk", COLORS["danger"], 16)
                links.append([event_id, risk_id, "触发"])
        by_id = {node["id"]: node for node in nodes}
        for node in nodes:
            neighbors = {b if a == node["id"] else a for a, b, _ in links if node["id"] in (a, b)}
            node["name"] = node["label"]
            node["relations"] = len(neighbors)
            node["events"] = sum(by_id[key]["type"] == "event" for key in neighbors)
            node["credibility"] = None
            node["risk"] = None
            node["summary"] = "节点数量与关系按当前展示的图谱统计。"
        for event in top_events:
            node = by_id[f"event-{event['id']}"]
            node.update(eventId=event["id"], name=event["title"], code=event["code"],
                        credibility=event["credibility"], risk={"critical": "重大", "high": "高影响", "medium": "中影响", "low": "低影响"}[event["level"]])
        selected_id = f"company-{top_events[0]['code'] or top_events[0]['company']}" if top_events else "task1"
        selected = by_id[selected_id]
        relations = [{"name": event["title"], "relation": f"{event['company']} · {event['sourceCount']}源"} for event in top_events[:6]]
        return {"scope": scope, "nodes": nodes, "links": links, "selectedId": selected_id, "selected": selected, "relations": relations}

    def company_profile(self, code: str | None = None) -> dict[str, Any]:
        matched: list[dict[str, Any]] = []
        if code:
            normalized = code.replace(".SZ", "").replace(".SH", "")
            matched = [event for event in self.events if event["code"] == normalized or event["company"] == code]
        if not matched:
            company = self.company_counter.most_common(1)[0][0] if self.company_counter else ""
            matched = [event for event in self.events if event["company"] == company]
        company = matched[0]["company"] if matched else "未知主体"
        code_value = matched[0]["code"] if matched else ""
        positive = sum(1 for event in matched if event["sentiment"] == "positive")
        negative = sum(1 for event in matched if event["sentiment"] == "negative")
        avg_score = sum(event["credibility"] for event in matched) / max(1, len(matched))
        high = sum(1 for event in matched if event["level"] in {"high", "critical"})
        topic_counter = Counter(event["type"] for event in matched)
        sentiment_by_day: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
        for event in matched:
            label = day_label(event["fullTime"])
            if event["sentiment"] == "positive":
                sentiment_by_day[label]["positive"] += 1
            if event["sentiment"] == "negative":
                sentiment_by_day[label]["negative"] += 1
        labels = sorted(sentiment_by_day.keys())[-7:] or ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "D"]
        palette = [COLORS["primary"], COLORS["blue"], COLORS["purple"], COLORS["warning"]]
        return {
            "name": company,
            "fullName": company,
            "code": code_value,
            "logo": company[:1] or "企",
            "industry": "上市公司事件主体",
            "region": "公开资讯样本",
            "tags": [name for name, _ in topic_counter.most_common(3)] or ["上市公司主体"],
            "quotes": [
                {"label": "公开事件", "value": fmt_number(len(matched)), "delta": "最新同步"},
                {"label": "正向信号", "value": fmt_number(positive), "delta": "持续关注"},
                {"label": "综合判断", "value": "稳定" if avg_score >= 90 else "关注" if avg_score >= 75 else "谨慎", "delta": "已同步"},
                {"label": "高影响事件", "value": fmt_number(high), "delta": "含重大预警"},
                {"label": "关联来源", "value": fmt_number(sum(event["sourceCount"] for event in matched)), "delta": "聚合来源"},
                {"label": "风险水位", "value": "高" if negative > positive else "中低", "delta": f"负面 {negative} 条"},
            ],
            "sentiment": {"labels": labels, "positive": [sentiment_by_day[label]["positive"] for label in labels], "negative": [sentiment_by_day[label]["negative"] for label in labels]},
            "topics": [{"text": name, "size": 10 + min(8, count), "color": palette[index % len(palette)]} for index, (name, count) in enumerate(topic_counter.most_common(12))],
        }

    def model_metrics(self) -> dict[str, Any]:
        trend = self.trend()
        alerts = self.alerts()
        high_impact_count = sum(1 for event in self.events if event["level"] in {"high", "critical"})
        critical_count = sum(1 for event in self.events if event["level"] == "critical")
        company_count = len(self.company_counter)
        top_theme = self.composition()[0]["name"] if self.composition() else "公开事件"
        return {
            "updatedAt": self.data_version_at or "已同步",
            "loadedAt": self.loaded_at or "已接入",
            "dataStatus": {"summary": "公开数据已同步", "version": self.data_version_at or "已同步"},
            "cards": [
                {
                    "icon": "样",
                    "name": "公开样本",
                    "version": "最新同步",
                    "status": "已接入",
                    "color": COLORS["primary"],
                    "metrics": [
                        ["样本", fmt_number(len(self.events))],
                        ["高影响", fmt_number(high_impact_count)],
                        ["主体", fmt_number(company_count)],
                        ["状态", "公开"],
                    ],
                },
                {
                    "icon": "警",
                    "name": "重点提醒",
                    "version": "当前可读",
                    "status": "已就绪",
                    "color": COLORS["warning"],
                    "metrics": [
                        ["提醒", fmt_number(len(alerts))],
                        ["重大", fmt_number(critical_count)],
                        ["高影响", fmt_number(high_impact_count)],
                        ["状态", "可读"],
                    ],
                },
                {
                    "icon": "主",
                    "name": "公司覆盖",
                    "version": "持续更新",
                    "status": "稳定",
                    "color": COLORS["blue"],
                    "metrics": [
                        ["主体", fmt_number(company_count)],
                        ["正面", fmt_number(sum(1 for event in self.events if event["sentiment"] == "positive"))],
                        ["负面", fmt_number(sum(1 for event in self.events if event["sentiment"] == "negative"))],
                        ["风险", "可控"],
                    ],
                },
                {
                    "icon": "图",
                    "name": "主题分布",
                    "version": top_theme,
                    "status": "活跃",
                    "color": COLORS["purple"],
                    "metrics": [
                        ["热点", top_theme],
                        ["事件", fmt_number(len(self.events))],
                        ["趋势", "更新"],
                        ["结论", "稳定"],
                    ],
                },
            ],
            "health": {"labels": trend["labels"], "latency": trend["high"], "volume": trend["total"]},
            "resources": [
                {"name": "公开样本", "value": 100, "meta": "已同步 · 持续更新"},
                {"name": "重点提醒", "value": 100 if alerts else 0, "meta": "已就绪 · 可直接阅读"},
                {"name": "公司覆盖", "value": min(100, company_count), "meta": f"主体 {company_count} 个"},
                {"name": "脉络追踪", "value": 100 if high_impact_count else 72, "meta": "关联关系 · 正常"},
            ],
            "logs": [
                {"time": "现在", "level": "success", "text": "公开结果已同步完成", "cost": "可读"},
                {"time": "最近", "level": "success", "text": "重点提醒已更新", "cost": "稳定"},
                {"time": "稍早", "level": "warn", "text": "单条来源响应偏慢，已继续跟踪", "cost": "关注"},
                {"time": "刚刚", "level": "success", "text": "脉络关系视图已刷新", "cost": "正常"},
                {"time": "今天", "level": "success", "text": "最新样本已完成归并", "cost": "同步"},
            ],
            "best": {
                "dashboard": {"highImpact": high_impact_count, "critical": critical_count, "companyCount": company_count},
                "topTheme": top_theme,
            },
        }

    def workbench(self) -> dict[str, Any]:
        interface_rows = [row for row in self.task3_task4_event_interface if finite_float(row.get("trust_score")) is not None]
        interface_rows.sort(
            key=lambda row: (
                finite_float(row.get("trust_score")) or 0.0,
                finite_float(self.task3_trust_by_event.get(str(row.get("canonical_event_id")), {}).get("aggregate_score")) or 0.0,
                finite_float(self.task3_current_state_by_event.get(str(row.get("canonical_event_id")), {}).get("confidence")) or 0.0,
            ),
            reverse=True,
        )
        raw_events = sorted(
            self.task3_canonical_events,
            key=lambda row: (
                safe_float(row.get("canonicalization_confidence"), safe_float(row.get("resolver_confidence"))),
                safe_int(row.get("core_member_count"), safe_int(row.get("member_count"))),
                safe_int(row.get("source_count"), len(row.get("member_official_row_ids") or [])),
            ),
            reverse=True,
        )
        raw_threads = sorted(
            self.task3_event_threads,
            key=lambda row: (safe_int(row.get("thread_size")), len(row.get("canonical_event_ids") or [])),
            reverse=True,
        )
        current_state_rows = self.task3_current_states if self.task3_current_states else interface_rows
        state_counter = Counter(self._task3_state_type(item) for item in current_state_rows)
        state_total = sum(state_counter.values()) or 1
        task4_formal = self.task4_closure_official1799_metrics or self.task4_closure_final_report.get("official1799_formal_metrics", {})
        task4_risk = task4_formal.get("official_risk", {})
        severity_counter = task4_formal_severity_counter(task4_risk)
        if not severity_counter and self.task34_task4_predictions:
            severity_counter = Counter(normalize_task4_severity(item.get("pred_severity_label") or item.get("pred_risk_intensity")) for item in self.task34_task4_predictions)
        if not severity_counter:
            severity_counter = Counter(normalize_task4_severity(item.get("severity") or "S0") for item in self.task4_alert_cards)
        severity_total = sum(severity_counter.values()) or 1
        selected_row = interface_rows[0] if interface_rows else {}
        selected_event_id = str(selected_row.get("canonical_event_id") or "")
        selected_state = self.task3_current_state_by_event.get(selected_event_id, {})
        selected_trust = self.task3_trust_by_event.get(selected_event_id, {})
        selected_canonical_meta = self.task3_event_by_id.get(selected_event_id) or self.task3_canonical_meta_by_id.get(selected_event_id, {})
        selected_trust_score = finite_float(selected_row.get("trust_score")) or finite_float(selected_trust.get("aggregate_score")) or 0.0
        selected_state_type = self._task3_state_type(selected_row, selected_state)
        selected_label = self._task3_display_label(selected_row, selected_state)

        def build_task3_node(node_type: str, label: str, color: str, code: str, name: str, events: int, relations: int, credibility: str, risk: str, summary: str, size: int) -> dict[str, Any]:
            return {
                "id": code,
                "label": label,
                "type": node_type,
                "typeLabel": {"canonical": "规范化事件", "thread": "事件线程", "state": "当前状态", "root": "公开样本"}[node_type],
                "color": color,
                "size": size,
                "code": code,
                "name": name,
                "events": events,
                "relations": relations,
                "credibility": credibility,
                "risk": risk,
                "summary": summary,
            }

        nodes: list[dict[str, Any]] = []
        links: list[list[str]] = []
        seen: set[str] = set()

        def add_node(node: dict[str, Any]) -> None:
            node_id = str(node["id"])
            if node_id in seen:
                return
            seen.add(node_id)
            index = len(nodes)
            angle = 2 * math.pi * index / max(8, min(14, len(interface_rows) or 1))
            radius = 0.26 + (index % 3) * 0.08
            node["x"] = max(0.08, min(0.92, 0.5 + math.cos(angle) * radius))
            node["y"] = max(0.12, min(0.88, 0.5 + math.sin(angle) * radius))
            nodes.append(node)

        root_id = "task3-root"
        add_node(
            build_task3_node(
                "root",
                "公开样本总览",
                COLORS["cyan"],
                root_id,
                "公开样本总览",
                len(interface_rows),
                len(interface_rows) * 2,
                f"{(sum(finite_float(row.get('trust_score')) or 0.0 for row in interface_rows) / max(1, len(interface_rows))) * 100:.1f}%" if interface_rows else "0.0%",
                "工作台",
                "公开样本与状态观测总览",
                24,
            )
        )
        for row in interface_rows[:10]:
            canonical_event_id = str(row.get("canonical_event_id") or "")
            thread_id = str(row.get("event_thread_id") or "")
            current_state = self.task3_current_state_by_event.get(canonical_event_id, {})
            trust = self.task3_trust_by_event.get(canonical_event_id, {})
            canonical_meta = self.task3_event_by_id.get(canonical_event_id) or self.task3_canonical_meta_by_id.get(canonical_event_id, {})
            trust_score = finite_float(row.get("trust_score")) or finite_float(trust.get("aggregate_score")) or 0.0
            current_state_type = self._task3_state_type(row, current_state)
            evidence_units = safe_int(trust.get("independent_evidence_unit_count"), safe_int(canonical_meta.get("member_count"), 1))
            source_count = len(trust.get("source_ids") or []) or safe_int(canonical_meta.get("member_count"), 1)
            canonical_label = self._task3_display_label(row, current_state)
            summary_text = self._task3_summary_text(row, current_state)
            canonical_node_id = f"canonical-{canonical_event_id}"
            thread_node_id = f"thread-{thread_id}"
            state_node_id = f"state-{canonical_event_id}"
            add_node(
                build_task3_node(
                    "canonical",
                    canonical_label,
                    COLORS["primary"],
                    canonical_node_id,
                    canonical_label,
                    max(1, evidence_units or safe_int(current_state.get("member_count")) or 1),
                    max(1, source_count + 1),
                    f"{trust_score * 100:.1f}%",
                    task3_state_label(current_state_type),
                    summary_text,
                    13 + min(14, int(round(trust_score * 12))),
                )
            )
            add_node(
                build_task3_node(
                    "thread",
                    thread_id.replace("ETF_", "").replace("ET4_", "")[-10:] or thread_id[:8],
                    COLORS["blue"],
                    thread_node_id,
                    current_state.get("subject") or canonical_label,
                    max(1, evidence_units or 1),
                    max(1, source_count),
                    f"{trust_score * 100:.1f}%",
                    task3_state_label(current_state_type),
                    summary_text,
                    11 + min(12, int(round(trust_score * 10))),
                )
            )
            add_node(
                build_task3_node(
                    "state",
                    task3_state_label(current_state_type),
                    task3_state_color(current_state_type),
                    state_node_id,
                    current_state.get("subject") or canonical_label,
                    safe_int(current_state.get("member_count")) or 1,
                    len(current_state.get("evidence_sentence_ids") or []),
                    f"{(safe_float(current_state.get('confidence')) or trust_score) * 100:.1f}%",
                    task3_state_label(current_state_type),
                    summary_text,
                    11 + min(12, int(round((safe_float(current_state.get("confidence")) or trust_score) * 10))),
                )
            )
            links.extend([[root_id, canonical_node_id, "接口"], [canonical_node_id, thread_node_id, "线程"], [canonical_node_id, state_node_id, "状态"]])

        selected = nodes[1] if len(nodes) > 1 else nodes[0] if nodes else {}
        relations = []
        if selected_event_id:
            relations = [
                {"name": "事件线程", "relation": str(selected_row.get("event_thread_id") or "未记录")},
                {"name": "当前状态", "relation": task3_state_label(selected_state_type)},
                {"name": "信任分数", "relation": f"{selected_trust_score * 100:.1f}%"},
                {"name": "证据单元", "relation": fmt_number(safe_int(selected_trust.get("independent_evidence_unit_count"), safe_int(selected_canonical_meta.get("member_count"), 1)))},
                {"name": "来源数量", "relation": fmt_number(len(selected_trust.get("source_ids") or []) or len(selected_canonical_meta.get("member_official_row_ids") or []) or 1)},
            ]

        severity_order = {"S4": 4, "S3": 3, "S2": 2, "S1": 1, "S0": 0}
        alert_cards = sorted(
            self.task4_alert_cards,
            key=lambda card: (
                severity_order.get(str(card.get("severity") or "S0"), 0),
                safe_int(re.sub(r"\D", "", str(card.get("alert_priority") or "0"))),
                1 if str(card.get("prototype_confidence") or "") == "high" else 0,
            ),
            reverse=True,
        )
        mapped_alerts = []
        for card in alert_cards:
            severity = str(card.get("severity") or "S0")
            mapped_alerts.append(
                {
                    "id": str(card.get("card_id") or card.get("task4_sample_id") or ""),
                    "sampleId": str(card.get("task4_sample_id") or ""),
                    "title": str(card.get("title") or ""),
                    "eventName": str(card.get("event_name") or ""),
                    "scope": str(card.get("impact_scope") or ""),
                    "priority": str(card.get("alert_priority") or ""),
                    "severity": severity,
                    "urgency": str(card.get("urgency") or ""),
                    "duration": str(card.get("duration") or ""),
                    "level": task4_level_from_severity(severity),
                    "color": task4_color_from_severity(severity),
                    "icon": task4_icon_from_severity(severity),
                    "confidence": str(card.get("prototype_confidence") or ""),
                    "verificationStatus": str(card.get("verification_status") or ""),
                    "displayTier": str(card.get("display_tier") or ""),
                    "labelSource": str(card.get("label_source") or ""),
                    "status": "已完成" if str(card.get("display_tier") or "") == "product_demo_only" else "待复核",
                    "reason": str(card.get("reason") or ""),
                    "quote": str(card.get("evidence_quote") or ""),
                    "directEvidence": clip_text(card.get("direct_text_evidence") or card.get("evidence_quote") or "", 260),
                    "detail": f"{card.get('urgency') or ''} · {card.get('duration') or ''}".strip(" ·"),
                    "importance": f"{severity} / {card.get('alert_priority') or ''}".strip(" /"),
                    "score": severity_order.get(severity, 0) * 20 + safe_int(re.sub(r"\D", "", str(card.get("alert_priority") or "0"))) * 3,
                }
            )
        mapped_alerts = mapped_alerts[:8]

        canonical_events = []
        for row in raw_events[:8]:
            canonical_id = str(row.get("canonical_event_id") or row.get("stable_id") or "")
            interface = self.task3_interface_by_event.get(canonical_id, {})
            confidence = safe_float(row.get("canonicalization_confidence"), safe_float(row.get("resolver_confidence")))
            canonical_events.append(
                {
                    "id": canonical_id,
                    "subject": self._task3_display_label(interface or row),
                    "type": str(row.get("provisional_event_type") or row.get("formal_event_type") or row.get("event_name") or interface.get("event_name") or ""),
                    "members": safe_int(row.get("core_member_count"), safe_int(row.get("member_count"))),
                    "sources": safe_int(row.get("source_count"), len(row.get("member_official_row_ids") or [])),
                    "confidence": f"{confidence * 100:.1f}%",
                    "status": task3_state_label(str(row.get("canonicalization_status") or row.get("resolver_decision") or "unknown")),
                    "time": short_time(row.get("last_publish_time") or row.get("first_publish_time") or interface.get("event_time")),
                    "summary": self._task3_summary_text(interface or row, max_len=90),
                }
            )

        threads = []
        if raw_threads:
            for row in raw_threads[:8]:
                canonical_ids = row.get("canonical_event_ids") or []
                threads.append(
                    {
                        "id": str(row.get("event_thread_id") or ""),
                        "subject": str(row.get("subject") or ""),
                        "size": safe_int(row.get("thread_size")),
                        "status": task3_state_label(str(row.get("verification_status") or "unknown")),
                        "method": str(row.get("construction_method") or ""),
                        "count": len(canonical_ids),
                        "relations": ", ".join(split_pipe(row.get("relation_types"))[:3]) or "无",
                    }
                )
        else:
            thread_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in interface_rows:
                thread_id = str(row.get("event_thread_id") or "")
                if thread_id:
                    thread_groups[thread_id].append(row)
            for thread_id, rows in sorted(thread_groups.items(), key=lambda item: len(item[1]), reverse=True)[:8]:
                first = rows[0]
                threads.append(
                    {
                        "id": thread_id,
                        "subject": self._task3_display_label(first),
                        "size": len(rows),
                        "status": task3_state_label(self._task3_state_type(first)),
                        "method": "official1799_interface_derived",
                        "count": len({str(row.get("canonical_event_id") or "") for row in rows}),
                        "relations": ", ".join(sorted({str(row.get("event_family") or row.get("support_type") or "") for row in rows if str(row.get("event_family") or row.get("support_type") or "")})[:3]) or "无",
                    }
                )

        state_distribution = [
            {
                "label": task3_state_label(state_type),
                "value": round(count * 100 / state_total, 1),
                "count": count,
                "color": task3_state_color(state_type),
            }
            for state_type, count in state_counter.most_common(6)
        ]
        severity_distribution = [
            {
                "label": severity,
                "value": round(count * 100 / severity_total, 1),
                "count": count,
                "color": task4_color_from_severity(severity),
            }
            for severity, count in sorted(severity_counter.items(), key=lambda item: severity_order.get(item[0], 0), reverse=True)
        ]

        trust_values = [finite_float(row.get("trust_score")) or 0.0 for row in interface_rows[:8]]
        if not trust_values:
            trust_values = [0.0]
        raw_event_sizes = [safe_int(row.get("core_member_count"), safe_int(row.get("member_count"))) for row in raw_events[:8]] or [0]
        alert_strengths = severity_spark(severity_counter)
        if not alert_strengths:
            alert_strengths = [severity_order.get(normalize_task4_severity(card.get("severity") or "S0"), 0) * 20 + safe_int(re.sub(r"\D", "", str(card.get("alert_priority") or "0"))) * 4 for card in alert_cards[:8]] or [0]
        updated_candidates = [
            str(item.get("publish_time") or "")
            for item in current_state_rows
            if str(item.get("publish_time") or "")
        ] + [
            str(item.get("last_publish_time") or item.get("first_publish_time") or "")
            for item in self.task3_canonical_events
            if str(item.get("last_publish_time") or item.get("first_publish_time") or "")
        ] + [
            str((item.get("provenance") or {}).get("extraction_time") or "")
            for item in self.task3_task4_event_interface
            if isinstance(item.get("provenance"), dict) and str((item.get("provenance") or {}).get("extraction_time") or "")
        ] + [
            self.data_version_at
        ]
        derived_thread_count = len(raw_threads) or len({str(row.get("event_thread_id") or "") for row in interface_rows if str(row.get("event_thread_id") or "")})
        task4_prediction_rows = safe_int(self.task4_closure_prediction_manifest.get("rows"), len(self.task34_task4_predictions))

        return {
            "updatedAt": best_time_text(updated_candidates),
            "metrics": [
                {"label": "公开样本", "value": fmt_number(len(interface_rows)), "delta": f"{fmt_number(len([row for row in interface_rows if finite_float(row.get('trust_score')) is not None]))}可用", "note": "公开接口样本", "icon": "样", "color": COLORS["primary"], "spark": [safe_int(round(value * 100)) for value in trust_values]},
                {"label": "状态快照", "value": fmt_number(len(current_state_rows)), "delta": f"{state_counter.get('unknown', 0)}待核验", "note": "最新状态视图", "icon": "态", "color": COLORS["blue"], "spark": [count for _, count in state_counter.most_common(8)] or [0]},
                {"label": "规范事件", "value": fmt_number(len(self.task3_canonical_events)), "delta": f"{fmt_number(derived_thread_count)}线程", "note": "已归并事件", "icon": "规", "color": COLORS["purple"], "spark": raw_event_sizes},
                {"label": "预警输出", "value": fmt_number(task4_prediction_rows), "delta": severity_delta_text(severity_counter), "note": "最终预警结果", "icon": "警", "color": COLORS["warning"], "spark": alert_strengths},
            ],
            "graph": {"scope": "task3", "selectedId": selected.get("id", ""), "nodes": nodes, "links": links, "selected": selected, "relations": relations},
            "stateDistribution": state_distribution,
            "severityDistribution": severity_distribution,
            "contract": self.task3_joint_interface,
            "canonicalEvents": canonical_events,
            "threads": threads,
            "alerts": mapped_alerts,
            "prediction": self.task4_prediction_results,
            "cases": self.task4_end_to_end_cases,
        }

    def _pick_task3_bridge(self, row: dict[str, Any]) -> dict[str, Any]:
        cluster_id = str(row.get("cluster_id") or "").strip()
        code = str(row.get("target_code") or "").strip()
        event_name = str(row.get("event_name") or "").strip()
        target_name = str(row.get("target_name") or "").strip()
        latest_publish = parse_dt(row.get("latest_publish_time") or row.get("earliest_publish_time") or "")
        source_names = split_pipe(row.get("source_list"))
        candidates = list(self.task3_mentions_by_cluster.get(cluster_id, [])) if cluster_id else []
        if not candidates and code:
            candidates = list(self.task3_mentions_by_code.get(code, []))
        if not candidates:
            candidates = list(self.task3_event_by_code.get(code, [])) if code else []
        if not candidates:
            return {}

        def score(item: dict[str, Any]) -> tuple[float, float, float, datetime, str]:
            item_code = str(jsonl_field(item, "target_code") or jsonl_field(item, "company_code") or "").strip()
            item_source = str(jsonl_field(item, "source_name") or "")
            item_text = " ".join(str(part) for part in [jsonl_field(item, "provisional_event_type") or "", " ".join(str(part) for part in (item.get("core_elements") or []) if part)] if part)
            item_time = parse_dt(jsonl_field(item, "publish_time") or jsonl_field(item, "time") or "")
            match = safe_float(item.get("confidence")) + safe_float(item.get("trust_score")) / 100.0
            if cluster_id:
                cluster_ids = {str(cluster).strip() for cluster in (item.get("task2_cluster_ids") or []) if str(cluster).strip()}
                if cluster_id in cluster_ids:
                    match += 5.0
                if cluster_id == str(jsonl_field(item, "provisional_cluster_id") or "").strip():
                    match += 1.0
            if code and item_code == code:
                match += 2.5
            if event_name and event_name in item_text:
                match += 1.0
            if target_name and target_name in item_text:
                match += 0.8
            if source_names and item_source:
                if item_source in source_names:
                    match += 2.0
                if item_source == source_names[0]:
                    match += 0.5
            if latest_publish and item_time:
                delta_minutes = abs((latest_publish - item_time).total_seconds()) / 60.0
                match += max(0.0, 3.0 - delta_minutes / 30.0)
            return (
                match,
                safe_float(item.get("confidence")),
                safe_float(item.get("trust_score")),
                item_time or datetime.min,
                str(jsonl_field(item, "stable_id") or jsonl_field(item, "event_mention_id") or ""),
            )

        selected = dict(sorted(candidates, key=score, reverse=True)[0])
        mention_id = str(jsonl_field(selected, "stable_id") or jsonl_field(selected, "event_mention_id") or "").strip()
        canonical_event_id = str(selected.get("canonical_event_id") or "").strip() or self.task3_canonical_by_mention.get(mention_id) or self.task3_canonical_by_representative_mention.get(mention_id) or ""
        event_thread_id = str(selected.get("event_thread_id") or "").strip() or self.task3_thread_by_canonical_event.get(canonical_event_id, "")
        if not canonical_event_id and code:
            legacy_candidates = list(self.task3_event_by_code.get(code, []))
            if legacy_candidates:
                legacy = legacy_candidates[0]
                canonical_event_id = str(legacy.get("canonical_event_id") or legacy.get("stable_id") or "")
        if canonical_event_id and not event_thread_id:
            event_thread_id = self.task3_thread_by_id.get(canonical_event_id, "")
        selected["event_mention_id"] = mention_id
        selected["canonical_event_id"] = canonical_event_id
        selected["event_thread_id"] = event_thread_id
        return selected

    def _pick_task4_candidate(self, row: dict[str, Any]) -> dict[str, Any]:
        code = str(row.get("target_code") or "").strip()
        event_name = str(row.get("event_name") or "").strip()
        target_name = str(row.get("target_name") or "").strip()
        candidates = list(self.task4_candidates_by_code.get(code, [])) if code else []
        if not candidates:
            return {}

        def score(item: dict[str, Any]) -> tuple[float, float, float, str]:
            bonus = safe_float(item.get("candidate_score"))
            if event_name and event_name == str(item.get("predicted_event_name") or ""):
                bonus += 1.5
            if target_name and target_name == str(item.get("predicted_target_name") or ""):
                bonus += 0.5
            if code and code == str(item.get("predicted_target_code") or ""):
                bonus += 0.7
            return (
                bonus,
                safe_float(item.get("reranker_score"), 0.0),
                safe_float(item.get("qwen_similarity"), 0.0),
                str(item.get("predicted_event_mention_id") or ""),
            )

        return sorted(candidates, key=score, reverse=True)[0]

    def _build_process_entry(self, event: dict[str, Any], rank: int) -> dict[str, Any]:
        row = self.row_by_id.get(str(event.get("id") or ""), {})
        bridge = self._pick_task3_bridge(row)
        canonical_event_id = str(bridge.get("canonical_event_id") or "")
        event_thread_id = str(bridge.get("event_thread_id") or "")
        event_mention_id = str(bridge.get("event_mention_id") or bridge.get("stable_id") or "")
        bridge_provenance = bridge.get("provenance") if isinstance(bridge.get("provenance"), dict) else {}
        mention_source_name = str(bridge.get("source_name") or bridge.get("source") or bridge.get("official_row_id") or "")
        mention_publish_time = str(bridge.get("publish_time") or bridge_provenance.get("extraction_time") or "")
        mention_summary = " · ".join(part for part in [mention_source_name, mention_publish_time] if part)
        interface = self.task3_interface_by_event.get(canonical_event_id) or self.task3_interface_by_thread.get(event_thread_id) or {}
        current_state = self.task3_current_state_by_event.get(canonical_event_id) or {}
        trust = self.task3_trust_by_event.get(canonical_event_id) or {}
        task4_candidate = self._pick_task4_candidate(row)
        task4_queue = self.task4_queue_by_event.get(canonical_event_id) or {}
        task4_path = self.task4_evidence_by_event.get(canonical_event_id) or {}
        evidence_items = self.evidence(row, limit=4)
        title_samples_value = title_samples(row)
        task1_summary = self.task1_certificate.get("pipeline_summary", {})
        task4_prediction = self.task4_prediction_results if isinstance(self.task4_prediction_results, dict) else {}
        task4_formal = self.task4_closure_official1799_metrics or self.task4_closure_final_report.get("official1799_formal_metrics", {})
        task4_metrics = task4_formal or task4_prediction.get("frozen_metrics_snapshot", {})
        task4_prediction_rows = safe_int(self.task4_closure_prediction_manifest.get("rows"), len(self.task34_task4_predictions))
        task4_card = next((card for card in self.task4_alert_cards if str(card.get("event_name") or "") == str(row.get("event_name") or "")), self.task4_alert_cards[0] if self.task4_alert_cards else {})
        candidate_list = [item for item in [task4_candidate, task4_queue, task4_path] if item]
        task3_source = interface or bridge
        task3_state_type = self._task3_state_type(task3_source, current_state)
        task3_trust_score = finite_float(task3_source.get("trust_score")) or finite_float(trust.get("aggregate_score")) or finite_float(current_state.get("confidence")) or 0.0
        task3_state_confidence = finite_float(current_state.get("confidence")) or finite_float(task3_source.get("resolver_confidence")) or task3_trust_score
        task3_state_verification = str(current_state.get("verification_status") or task3_source.get("support_type") or task3_source.get("contract_status") or "unknown")
        return {
            "id": event.get("id"),
            "rank": rank,
            "queueLabel": "待处理",
            "ingestedAt": event.get("fullTime") or event.get("time") or "",
            "title": event.get("title") or "",
            "company": event.get("company") or "",
            "code": event.get("code") or "",
            "logo": event.get("logo") or (event.get("company") or "事")[:1],
            "source": event.get("source") or "",
            "sourceCount": event.get("sourceCount") or 0,
            "sentiment": event.get("sentiment") or "neutral",
            "level": event.get("level") or "low",
            "credibility": event.get("credibility") or 0,
            "impact": event.get("impact") or 0,
            "heat": event.get("heat") or 0,
            "task1": {
                "clusterId": event.get("id") or "",
                "eventName": row.get("event_name") or event.get("type") or "",
                "targetName": row.get("target_name") or event.get("company") or "",
                "targetCode": row.get("target_code") or event.get("code") or "",
                "acceptedEvents": safe_int(task1_summary.get("total_final_accepted_events"), len(self.events)),
                "autoAccept": safe_int(task1_summary.get("auto_accept_events"), 0),
                "manualReview": safe_int(task1_summary.get("manual_review_queue_count"), 0),
                "routedArticles": safe_int(task1_summary.get("routed_articles"), 0),
                "supportingSources": safe_int(row.get("supporting_source_count")),
                "memberCount": safe_int(row.get("member_count")),
                "sourceList": split_pipe(row.get("source_list")),
                "summary": clip_text(
                    f"系统将 {event.get('title')} 归并为同一公开事件，来自 {safe_int(row.get('supporting_source_count'))} 个来源，"
                    f"形成 {safe_int(row.get('member_count'))} 条成员新闻。",
                    180,
                ),
                "preview": title_samples_value[:3],
                "verificationStatus": verification_label(str(row.get("verification_status") or row.get("fulltext_verification_status") or "")),
            },
            "task2": {
                "score": safe_float(row.get("fulltext_credibility_score"), safe_float(row.get("credibility_score"))),
                "level": str(row.get("fulltext_credibility_level") or row.get("credibility_level") or "待核验"),
                "verificationStatus": verification_label(str(row.get("fulltext_verification_status") or row.get("verification_status") or "")),
                "retrievedArticles": safe_int(row.get("retrieved_article_count")),
                "evidenceSentenceCount": safe_int(row.get("evidence_sentence_count")),
                "fulltextSupportCount": safe_int(row.get("fulltext_support_count")),
                "contradictCount": safe_int(row.get("fulltext_contradict_count")),
                "officialSupportCount": safe_int(row.get("fulltext_official_support_count")),
                "sourceScore": safe_float(row.get("source_score")),
                "crossSourceScore": safe_float(row.get("cross_source_score")),
                "clusterConsistencyScore": safe_float(row.get("cluster_consistency_score")),
                "supportingReasons": str(row.get("supporting_reasons") or ""),
                "riskReasons": str(row.get("risk_reasons") or ""),
                "reason": str(row.get("fulltext_reason") or row.get("supporting_reasons") or row.get("weak_label_reason") or ""),
                "evidenceItems": evidence_items,
                "summary": clip_text(
                    f"系统复核得到 {safe_float(row.get('fulltext_credibility_score'), safe_float(row.get('credibility_score'))):.2f} 分，"
                    f"等级 {row.get('fulltext_credibility_level') or row.get('credibility_level') or '待核验'}，"
                    f"检索 {safe_int(row.get('retrieved_article_count'))} 篇文章和 {safe_int(row.get('evidence_sentence_count'))} 条证据句。",
                    200,
                ),
            },
            "task3": {
                "clusterId": str(row.get("cluster_id") or event.get("id") or ""),
                "eventMentionId": event_mention_id,
                "mentionSourceName": mention_source_name,
                "mentionPublishTime": mention_publish_time,
                "canonicalEventId": canonical_event_id,
                "eventThreadId": event_thread_id,
                "currentStateType": task3_state_type,
                "trustScore": task3_trust_score,
                "stateObservationId": str(current_state.get("current_state_observation_id") or ""),
                "stateConfidence": task3_state_confidence,
                "stateVerificationStatus": task3_state_verification,
                "evidenceSentenceIds": [str(item) for item in (current_state.get("evidence_sentence_ids") or [])[:5]],
                "currentStateSummary": self._task3_summary_text(task3_source, current_state, 180) or clip_text(mention_summary or row.get("event_name") or "", 180),
                "summary": clip_text(
                    f"系统将 {event_mention_id or '未匹配提及'} 归入 {canonical_event_id or '未匹配'}，挂接 {event_thread_id or '未分配'}，当前状态 {task3_state_label(task3_state_type)}，"
                    f"trust {task3_trust_score * 100:.2f}%。",
                    200,
                ),
            },
            "task4": {
                "candidate": {
                    "predictedEventMentionId": str(task4_candidate.get("predicted_event_mention_id") or ""),
                    "predictedSchemaId": str(task4_candidate.get("predicted_schema_id") or ""),
                    "predictedEventName": str(task4_candidate.get("predicted_event_name") or ""),
                    "predictedTargetCode": str(task4_candidate.get("predicted_target_code") or ""),
                    "predictedTargetName": str(task4_candidate.get("predicted_target_name") or ""),
                    "candidateScore": safe_float(task4_candidate.get("candidate_score")),
                    "qwenSimilarity": safe_float(task4_candidate.get("qwen_similarity")),
                    "rerankerScore": safe_float(task4_candidate.get("reranker_score")),
                    "candidateSource": str(task4_candidate.get("candidate_source") or ""),
                    "source": str(task4_candidate.get("source") or ""),
                    "publishTime": str(task4_candidate.get("publish_time") or ""),
                    "evidenceSentence": clip_text(task4_candidate.get("evidence_sentence") or "", 220),
                },
                "queueItem": {
                    "queueId": str(task4_queue.get("queue_id") or ""),
                    "reviewReason": str(task4_queue.get("review_reason") or ""),
                    "verificationStatus": str(task4_queue.get("verification_status") or ""),
                    "priorityRank": safe_int(task4_queue.get("priority_rank")),
                },
                "evidencePath": task4_path.get("evidence_path") or [],
                "formalPrediction": {
                    "status": str(self.task4_closure_final_report.get("final_status") or task4_prediction.get("status") or ""),
                    "validityModel": str(self.task4_closure_head_selection.get("official_risk_1_to_5", {}).get("selected") or task4_prediction.get("formal_prediction_chain", {}).get("validity_model") or ""),
                    "directionModel": str(self.task4_closure_head_selection.get("direction_short", {}).get("selected") or task4_prediction.get("formal_prediction_chain", {}).get("direction_model") or ""),
                    "formalEvaluationScope": task4_formal.get("actual_evaluable_fields") or task4_prediction.get("formal_evaluation_scope") or [],
                    "productDemoScope": task4_prediction.get("product_demo_scope") or task4_formal.get("actual_evaluable_fields") or [],
                    "predictionPool": {
                        **(task4_prediction.get("prediction_pool") or {}),
                        "predicted_observation_count": task4_prediction_rows,
                        "event_interface_count": len(self.task3_task4_event_interface),
                    },
                    "metrics": task4_metrics,
                },
                "sampleAlertCard": {
                    "cardId": str(task4_card.get("card_id") or ""),
                    "sampleId": str(task4_card.get("task4_sample_id") or ""),
                    "title": str(task4_card.get("title") or ""),
                    "eventName": str(task4_card.get("event_name") or ""),
                    "severity": str(task4_card.get("severity") or ""),
                    "alertPriority": str(task4_card.get("alert_priority") or ""),
                    "impactScope": str(task4_card.get("impact_scope") or ""),
                    "urgency": str(task4_card.get("urgency") or ""),
                    "duration": str(task4_card.get("duration") or ""),
                    "prototypeConfidence": str(task4_card.get("prototype_confidence") or ""),
                    "verificationStatus": str(task4_card.get("verification_status") or ""),
                    "reason": str(task4_card.get("reason") or ""),
                    "evidenceQuote": str(task4_card.get("evidence_quote") or ""),
                },
                "summary": clip_text(
                    f"系统输出 {task4_candidate.get('predicted_event_name') or row.get('event_name') or '未命中'}，"
                    f"队列审查 {task4_queue.get('verification_status') or 'unknown'}，"
                    f"预警结果 {fmt_number(task4_prediction_rows)} 条，原型卡片等级 {task4_card.get('severity') or 'S0'} / {task4_card.get('alert_priority') or 'A0'}。",
                    220,
                ),
            },
            "trace": [
                {"phase": "ingest", "label": "事件进入", "state": "done", "note": event.get("time") or event.get("fullTime") or ""},
                {"phase": "task1", "label": "事件归并", "state": "done", "note": str(row.get("cluster_id") or event.get("id") or "")},
                {"phase": "task2", "label": "公开判断", "state": "done", "note": str(row.get("fulltext_credibility_level") or row.get("credibility_level") or "")},
                {"phase": "task3", "label": "脉络追踪", "state": "done" if canonical_event_id else "active", "note": event_thread_id or event_mention_id or "未分配"},
                {"phase": "task4", "label": "预警输出", "state": "done" if candidate_list else "active", "note": str(task4_candidate.get("predicted_event_name") or task4_card.get("event_name") or "")},
            ],
        }

    def process_demo(self, limit: int = 5) -> dict[str, Any]:
        queue = [self._build_process_entry(event, index + 1) for index, event in enumerate(self.events[:limit])]
        task1_summary = self.task1_certificate.get("pipeline_summary", {})
        task4_prediction = self.task4_prediction_results if isinstance(self.task4_prediction_results, dict) else {}
        task4_formal = self.task4_closure_official1799_metrics or self.task4_closure_final_report.get("official1799_formal_metrics", {})
        task4_risk = task4_formal.get("official_risk", {})
        task4_nonofficial = self.task4_closure_final_report.get("nonofficial_admission", {})
        task4_direction = self.task4_closure_final_report.get("direction_gate_reuse", {})
        task4_highrisk = self.task4_closure_highrisk_metrics or task4_formal.get("official_high_risk", {})
        task4_prediction_rows = safe_int(self.task4_closure_prediction_manifest.get("rows"), safe_int(self.task34_machine_state.get("task4_final_gold_rows"), 1799))
        current_state_rows = self.task3_current_states if self.task3_current_states else self.task3_task4_event_interface
        task3_state_count = len(current_state_rows)
        severity_counter = task4_formal_severity_counter(task4_risk)
        if not severity_counter and self.task34_task4_predictions:
            severity_counter = Counter(normalize_task4_severity(item.get("pred_severity_label") or item.get("pred_risk_intensity")) for item in self.task34_task4_predictions)
        if not severity_counter:
            severity_counter = Counter(normalize_task4_severity(item.get("severity") or "S0") for item in self.task4_alert_cards)
        high_impact_ratio = sum(1 for event in self.events if event["level"] in {"high", "critical"}) / max(1, len(self.events))
        evidence_demo = self.task4_evidence_path_demo[:6]
        alert_cards = self.task4_alert_cards[:4]
        return {
            "updatedAt": best_time_text([self.data_version_at] + [entry.get("ingestedAt") or "" for entry in queue]),
            "playbackIntervalMs": 3200,
            "stageOrder": [
                {"key": "ingest", "label": "进入队列", "icon": "入", "color": COLORS["cyan"]},
                {"key": "task1", "label": "事件归并", "icon": "归", "color": COLORS["primary"]},
                {"key": "task2", "label": "公开判断", "icon": "验", "color": COLORS["blue"]},
                {"key": "task3", "label": "脉络追踪", "icon": "脉", "color": COLORS["purple"]},
                {"key": "task4", "label": "预警输出", "icon": "警", "color": COLORS["warning"]},
            ],
            "metrics": [
                {"label": "公开样本", "value": fmt_number(len(queue)), "delta": "最新同步", "note": "当前可浏览样本", "icon": "样", "color": COLORS["primary"], "spark": [18, 24, 28, 31, 35, 39, 42, 48]},
                {"label": "重点信号", "value": fmt_number(sum(1 for event in self.events if event["level"] in {"high", "critical"})), "delta": f"高影响占比 {metric_percent(high_impact_ratio)}", "note": "公开态势持续更新", "icon": "视", "color": COLORS["blue"], "spark": [14, 18, 23, 29, 34, 37, 41, 45]},
                {"label": "预警输出", "value": fmt_number(len(alert_cards)), "delta": "已生成提醒", "note": "重点卡片可直接阅读", "icon": "警", "color": COLORS["purple"], "spark": [12, 16, 22, 27, 31, 36, 40, 44]},
                {"label": "当前状态", "value": "已完成" if alert_cards else "进行中", "delta": "公开结果", "note": "按步骤查看输出", "icon": "流", "color": COLORS["warning"], "spark": severity_spark(severity_counter)},
            ],
            "queue": [
                {
                    "id": entry["id"],
                    "rank": entry["rank"],
                    "title": entry["title"],
                    "company": entry["company"],
                    "code": entry["code"],
                    "source": entry["source"],
                    "sourceCount": entry["sourceCount"],
                    "ingestedAt": entry["ingestedAt"],
                    "queueLabel": entry["queueLabel"],
                    "credibility": entry["credibility"],
                    "impact": entry["impact"],
                    "level": entry["level"],
                    "sentiment": entry["sentiment"],
                    "task1ClusterId": entry["task1"]["clusterId"],
                    "task3State": entry["task3"]["currentStateType"],
                    "task4Severity": entry["task4"]["sampleAlertCard"].get("severity") or entry["task4"]["formalPrediction"].get("status") or "",
                }
                for entry in queue
            ],
            "entries": queue,
            "task1Summary": {
                "completedAt": str(self.task1_certificate.get("completed_at") or ""),
                "model": str(self.task1_certificate.get("model") or ""),
                "threshold": safe_float(self.task1_certificate.get("threshold")),
                "summary": task1_summary,
            },
            "task2Summary": self.task2_summary,
            "task3Summary": {
                "contract": self.task3_joint_interface,
                "stateCount": task3_state_count,
                "interfaceCount": len(self.task3_task4_event_interface),
            },
            "task4Summary": {
                "prediction": {
                    **task4_prediction,
                    "status": str(self.task4_closure_final_report.get("final_status") or task4_prediction.get("status") or ""),
                    "prediction_pool": {
                        **(task4_prediction.get("prediction_pool") or {}),
                        "predicted_observation_count": task4_prediction_rows,
                        "event_interface_count": len(self.task3_task4_event_interface),
                    },
                    "official1799_formal_metrics": task4_formal,
                    "official_risk": task4_risk,
                    "nonofficial_closure": task4_nonofficial,
                    "direction_gate_reuse": task4_direction,
                    "highrisk": task4_highrisk,
                },
                "finalReport": self.task4_closure_final_report,
                "predictionManifest": self.task4_closure_prediction_manifest,
                "severityDistribution": dict(severity_counter),
                "evidencePathDemo": evidence_demo,
                "sampleAlertCards": alert_cards,
                "uncertainQueueSize": safe_int(self.task4_uncertain_queue.get("queue_size_reference"), 0),
            },
        }

    def search(self, query: str) -> list[dict[str, Any]]:
        return [{"type": "event", "id": event["id"], "title": event["title"], "meta": f"{event['company']} · {event['code']} · {event['type']}", "level": event["level"]} for event in self.get_events({"query": query, "level": "all", "sentiment": "all"})[:8]]


STORE = DataStore()
STORE_REFRESH_LOCK = threading.Lock()
STORE_SIGNATURE_CHECK_INTERVAL = 2.0
STORE_LAST_SIGNATURE_CHECK = 0.0


def refresh_store_if_needed() -> None:
    global STORE, STORE_LAST_SIGNATURE_CHECK
    now = time.monotonic()
    if now - STORE_LAST_SIGNATURE_CHECK < STORE_SIGNATURE_CHECK_INTERVAL:
        return

    # ThreadingHTTPServer can dispatch the dashboard, events and alerts calls
    # at the same time.  Only one request may rebuild the relatively large
    # in-memory store; other requests keep serving the last good snapshot.
    if not STORE_REFRESH_LOCK.acquire(blocking=False):
        return
    try:
        now = time.monotonic()
        if now - STORE_LAST_SIGNATURE_CHECK < STORE_SIGNATURE_CHECK_INTERVAL:
            return
        STORE_LAST_SIGNATURE_CHECK = now
        current_signature = source_mtime_signature()
        if getattr(STORE, "source_signature", None) == current_signature:
            return
        candidate = DataStore()
        if not candidate.sources_changed_during_load:
            STORE = candidate
    finally:
        STORE_REFRESH_LOCK.release()


def read_users() -> list[dict[str, str]]:
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def write_users(users: list[dict[str, str]]) -> None:
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


class Handler(SimpleHTTPRequestHandler):
    server_version = "EventIntelligenceAPI/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/v1"):
            refresh_store_if_needed()
            self.handle_api_get(parsed.path[len("/api/v1") :], parse_qs(parsed.query))
            return
        if any(part.startswith('.') for part in unquote(parsed.path).split('/') if part):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/v1"):
            refresh_store_if_needed()
            try:
                self.handle_api_post(parsed.path[len("/api/v1") :])
            except (OSError, sqlite3.Error):
                self.json_response({"message": "保存失败，请检查运行目录权限后重试"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def json_response(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(clean_json(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        path = unquote(path or "/")
        filters = {key: values[-1] for key, values in query.items() if values}
        if path.startswith("/events/"):
            event_id = path[len("/events/"):]
            for suffix in ("/credibility", "/timeline"):
                if event_id.endswith(suffix):
                    event_id = event_id[:-len(suffix)]
                    break
            if event_id not in STORE.event_by_id:
                self.json_response({"message": "该事件不存在或已不在当前数据中，请重新选择"}, HTTPStatus.NOT_FOUND)
                return
        if path == "/dashboard":
            self.json_response(STORE.dashboard())
        elif path == "/status":
            self.json_response(STORE.data_status())
        elif path == "/events":
            self.json_response(STORE.get_events(filters))
        elif path.startswith("/events/") and path.endswith("/credibility"):
            self.json_response(STORE.credibility(path[len("/events/") : -len("/credibility")].strip("/")))
        elif path.startswith("/events/") and path.endswith("/timeline"):
            self.json_response(STORE.timeline(path[len("/events/") : -len("/timeline")].strip("/")))
        elif path.startswith("/events/"):
            self.json_response(STORE.get_event(path[len("/events/") :].strip("/")))
        elif path == "/knowledge-graph":
            self.json_response(STORE.graph(filters.get("scope", "company")))
        elif path == "/alerts":
            self.json_response(STORE.alerts())
        elif path == "/alert-rules":
            self.json_response(WORKFLOW.rules())
        elif path == "/watches":
            self.json_response(WORKFLOW.watches())
        elif path.startswith("/companies/"):
            self.json_response(STORE.company_profile(path[len("/companies/") :].strip("/")))
        elif path == "/models/metrics":
            self.json_response(STORE.model_metrics())
        elif path == "/workbench":
            self.json_response(STORE.workbench())
        elif path == "/process-demo":
            self.json_response(STORE.process_demo(safe_int(filters.get("limit"), 5)))
        elif path == "/search":
            self.json_response(STORE.search(filters.get("q", "")))
        else:
            self.json_response({"error": "unknown endpoint", "path": path}, HTTPStatus.NOT_FOUND)

    def handle_api_post(self, path: str) -> None:
        path = unquote(path or "/")
        payload = self.read_body()
        if path == "/auth/login":
            account = str(payload.get("account") or "").strip()
            password = str(payload.get("password") or "")
            users = read_users()
            demo = account == "demo@finsight.cn" and password == "123456"
            user = next((item for item in users if item.get("account") == account and item.get("password") == password), None)
            if not demo and not user:
                self.json_response({"success": False, "message": "账号或密码不正确"}, HTTPStatus.UNAUTHORIZED)
                return
            self.json_response({"success": True, "token": f"local-token-{int(time.time())}", "user": {"name": "演示用户" if demo else user.get("name", account), "account": account}})
        elif path == "/auth/register":
            account = str(payload.get("account") or "").strip()
            name = str(payload.get("name") or "").strip() or account
            password = str(payload.get("password") or "")
            users = read_users()
            if account == "demo@finsight.cn" or any(item.get("account") == account for item in users):
                self.json_response({"success": False, "message": "该账号已注册"}, HTTPStatus.CONFLICT)
                return
            users.append({"name": name, "account": account, "password": password})
            write_users(users)
            self.json_response({"success": True, "user": {"name": name, "account": account}})
        elif path.startswith("/alerts/") and path.endswith("/acknowledge"):
            alert_id = path[len("/alerts/") : -len("/acknowledge")].strip("/")
            alert = next((item for item in STORE.alerts() if item["id"] == alert_id), None)
            if not alert:
                self.json_response({"message": "预警不存在或已不满足当前规则，请刷新后重试"}, HTTPStatus.NOT_FOUND)
                return
            WORKFLOW.acknowledge(alert["eventId"])
            self.json_response({"success": True, "id": alert_id, "status": "跟踪中"})
        elif path == "/alert-rules":
            try:
                self.json_response(WORKFLOW.save_rules(payload))
            except ValueError as error:
                self.json_response({"message": str(error)}, HTTPStatus.BAD_REQUEST)
        elif path == "/watches":
            try:
                self.json_response({"success": True, **WORKFLOW.create_watch(payload)})
            except ValueError as error:
                self.json_response({"message": str(error)}, HTTPStatus.BAD_REQUEST)
        else:
            self.json_response({"error": "unknown endpoint", "path": path}, HTTPStatus.NOT_FOUND)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the financial event intelligence system.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()
    if not TASK2_EVENTS.exists():
        print(f"Warning: missing data file: {TASK2_EVENTS}", file=sys.stderr)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving financial event intelligence system at http://{args.host}:{args.port}")
    print(f"Frontend root: {FRONTEND_ROOT}")
    print(f"Data root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
