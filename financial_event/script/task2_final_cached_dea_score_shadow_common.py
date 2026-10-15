from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.event_credibility import fulltext


ROOT = Path(__file__).resolve().parents[1]
FORMAL_LLM_RESULTS = ROOT / "outputs/event_credibility/llm_review_results.jsonl"
FORMAL_BASE_SCORES = ROOT / "outputs/event_credibility/event_credibility_scores.parquet"
FORMAL_FULLTEXT_JSONL = ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.jsonl"
FORMAL_FULLTEXT_PARQUET = ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.parquet"
FORMAL_FULLTEXT_XLSX = ROOT / "outputs/event_credibility/event_credibility_scores_fulltext.xlsx"
FULLTEXT_EVIDENCE = ROOT / "outputs/event_credibility/fulltext_evidence.parquet"
FINAL_RESULTS_260 = ROOT / "outputs/event_credibility/final_cached_dea_v1_260/final_results.jsonl"
SHADOW_OUTPUT_DIR = ROOT / "outputs/event_credibility/final_cached_dea_v1_score_shadow"
SHADOW_LLM_RESULTS = SHADOW_OUTPUT_DIR / "llm_review_results_shadow.jsonl"
SHADOW_FULLTEXT_JSONL = SHADOW_OUTPUT_DIR / "event_credibility_scores_fulltext_shadow.jsonl"
SHADOW_FULLTEXT_PARQUET = SHADOW_OUTPUT_DIR / "event_credibility_scores_fulltext_shadow.parquet"
SHADOW_FULLTEXT_XLSX = SHADOW_OUTPUT_DIR / "event_credibility_scores_fulltext_shadow.xlsx"
FUSION_MANIFEST = SHADOW_OUTPUT_DIR / "fusion_manifest.json"
FAILED_RECORDS = SHADOW_OUTPUT_DIR / "failed_records.jsonl"
REPORT_JSON = ROOT / "reports/task2_final_cached_dea_score_shadow_report.json"
REPORT_MD = ROOT / "reports/task2_final_cached_dea_score_shadow_report.md"
REPORT_XLSX = ROOT / "reports/task2_final_cached_dea_score_shadow_changes.xlsx"
VALID_LLM_VERDICTS = {"support", "contradict", "insufficient"}
PIPELINE_VERSION = "final_cached_dea_v1"
MODEL_NAME = "DianJin-R1-32B"
FORMAL_OUTPUTS = [FORMAL_LLM_RESULTS, FORMAL_FULLTEXT_JSONL, FORMAL_FULLTEXT_PARQUET, FORMAL_FULLTEXT_XLSX]
FREEZE_FILES = [
    ROOT / "src/event_credibility/fulltext.py",
    ROOT / "src/event_credibility/dea.py",
    ROOT / "src/event_credibility/retrieval_index.py",
    ROOT / "src/event_credibility/dea_safety_gate_v1_1.py",
    ROOT / "src/event_credibility/dea_safety_gate_v1_2.py",
    ROOT / "src/event_credibility/claim_contract.py",
    ROOT / "src/event_credibility/isr.py",
    ROOT / "configs/task2_final_cached_dea_v1.yaml",
    ROOT / "configs/task2_dea_safety_gate_v1_1.yaml",
    ROOT / "configs/task2_dea_safety_gate_v1_2.yaml",
    ROOT / "configs/task2_claim_contract_adapter_v1.yaml",
    ROOT / "configs/task2_isr_diagnostic_v1.yaml",
]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def norm(value: Any) -> str:
    return fulltext.normalize_text(value)


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
        for path in paths
        if path.exists()
    }


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            rows.append(obj)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def llm_review_event_id(row: dict[str, Any]) -> str:
    return norm(row.get("cluster_id") or row.get("event_id") or row.get("event_cluster_id"))


def final_event_id(row: dict[str, Any]) -> str:
    return norm(row.get("cluster_id"))


def load_formal_llm_rows(path: Path = FORMAL_LLM_RESULTS) -> list[dict[str, Any]]:
    return read_jsonl_rows(path)


def load_final_rows(path: Path = FINAL_RESULTS_260) -> list[dict[str, Any]]:
    return read_jsonl_rows(path)


def load_formal_scores(path: Path = FORMAL_FULLTEXT_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_base_scores(path: Path = FORMAL_BASE_SCORES) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_fulltext_evidence(path: Path = FULLTEXT_EVIDENCE) -> pd.DataFrame:
    return pd.read_parquet(path)


def _normalised_evidence_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ids = [norm(item) for item in value if norm(item)]
    return ids


def build_shadow_llm_rows(
    formal_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formal_ids = [llm_review_event_id(row) for row in formal_rows]
    final_ids = [final_event_id(row) for row in final_rows]
    if len(formal_ids) != len(set(formal_ids)):
        duplicates = sorted(cid for cid, count in Counter(formal_ids).items() if count > 1)
        raise ValueError(f"formal llm review has duplicate cluster_id: {duplicates[:10]}")
    if len(final_ids) != len(set(final_ids)):
        duplicates = sorted(cid for cid, count in Counter(final_ids).items() if count > 1)
        raise ValueError(f"final 260 results have duplicate cluster_id: {duplicates[:10]}")
    formal_set = set(formal_ids)
    final_set = set(final_ids)
    if formal_set != final_set:
        raise ValueError(
            "formal llm review clusters do not match final 260 results: "
            f"missing={sorted(formal_set - final_set)[:10]}, extra={sorted(final_set - formal_set)[:10]}"
        )

    final_map = {final_event_id(row): row for row in final_rows}
    shadow_rows: list[dict[str, Any]] = []
    verdict_changes = 0
    confidence_abs_diffs: list[float] = []
    evidence_changes = 0
    same_rows = 0
    for formal_row in formal_rows:
        cid = llm_review_event_id(formal_row)
        final_row = final_map[cid]
        final_result = final_row.get("final_result") or {}
        verdict = norm(final_result.get("verdict"))
        confidence = float(final_result.get("confidence") or 0.0)
        reason = norm(final_result.get("reason"))
        evidence_ids = _normalised_evidence_ids(final_result.get("evidence_ids") or [])
        if verdict == "support":
            supporting = evidence_ids
            contradicting: list[str] = []
        elif verdict == "contradict":
            supporting = []
            contradicting = evidence_ids
        else:
            supporting = []
            contradicting = []
        canonical = {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "evidence_ids": evidence_ids,
        }
        shadow_row = dict(formal_row)
        shadow_row.update(
            {
                "cluster_id": cid,
                "model": MODEL_NAME,
                "model_name": MODEL_NAME,
                "pipeline_version": PIPELINE_VERSION,
                "input_index": int(final_row.get("input_index") or 0),
                "status": "ok",
                "attempts": int(final_row.get("logical_request_count") or formal_row.get("attempts") or 1),
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
                "evidence_ids": evidence_ids,
                "supporting_evidence_ids": supporting,
                "contradicting_evidence_ids": contradicting,
                "raw_response": json.dumps(canonical, ensure_ascii=False),
                "selected_result_source": norm(final_row.get("selected_result_source")),
                "dea_triggered": bool(final_row.get("dea_triggered")),
                "dea_trigger_reasons": [norm(x) for x in final_row.get("dea_trigger_reasons") or [] if norm(x)],
            }
        )
        formal_verdict = norm(formal_row.get("verdict"))
        formal_confidence = formal_row.get("confidence")
        formal_evidence = _normalised_evidence_ids(formal_row.get("evidence_ids") or formal_row.get("supporting_evidence_ids") or formal_row.get("contradicting_evidence_ids") or [])
        if formal_verdict != verdict:
            verdict_changes += 1
        try:
            if formal_confidence is not None and math.isfinite(float(formal_confidence)):
                confidence_abs_diffs.append(abs(float(formal_confidence) - confidence))
        except Exception:
            pass
        if formal_evidence != evidence_ids:
            evidence_changes += 1
        if formal_verdict == verdict and formal_evidence == evidence_ids and (
            formal_confidence is not None and math.isfinite(float(formal_confidence)) and float(formal_confidence) == confidence
        ):
            same_rows += 1
        shadow_rows.append(shadow_row)
    summary = {
        "formal_count": len(formal_rows),
        "final_count": len(final_rows),
        "replaced_count": len(shadow_rows),
        "same_cluster_count": len(formal_set & final_set),
        "missing_cluster_ids": sorted(formal_set - final_set),
        "extra_cluster_ids": sorted(final_set - formal_set),
        "verdict_change_count": verdict_changes,
        "evidence_ids_change_count": evidence_changes,
        "same_row_count": same_rows,
        "confidence_mean_abs_diff": round(sum(confidence_abs_diffs) / max(1, len(confidence_abs_diffs)), 6),
        "confidence_max_abs_diff": round(max(confidence_abs_diffs), 6) if confidence_abs_diffs else 0.0,
        "verdict_distribution": dict(Counter(row["verdict"] for row in shadow_rows)),
        "selected_result_source_distribution": dict(Counter(norm(row.get("selected_result_source")) for row in final_rows)),
        "dea_trigger_count": sum(1 for row in final_rows if row.get("dea_triggered")),
    }
    return shadow_rows, summary


def llm_adjustment(verdict: str, confidence: Any) -> float:
    try:
        value = float(confidence)
    except Exception:
        value = 0.0
    if verdict == "support":
        return 12.0 * value
    if verdict == "contradict":
        return -35.0 * value
    if verdict == "insufficient":
        return -5.0 * value
    return 0.0


def group_level(level: Any) -> str:
    text = norm(level)
    if text == "高可信":
        return "高可信"
    if text in {"较可信", "待核验"}:
        return "中可信"
    return "低可信"


def pctl(values: list[float], percentile: float) -> float:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return 0.0
    position = (len(clean) - 1) * percentile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return round(clean[low], 6)
    return round(clean[low] + (clean[high] - clean[low]) * (position - low), 6)


def mean(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return round(sum(clean) / max(1, len(clean)), 6)


def exact_row_equal(left: dict[str, Any], right: dict[str, Any], columns: list[str]) -> bool:
    def missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, dict, tuple, set)):
            return False
        try:
            result = pd.isna(value)
            return bool(result) if not hasattr(result, "__len__") else False
        except Exception:
            return False

    for column in columns:
        a = left.get(column)
        b = right.get(column)
        if missing(a) and missing(b):
            continue
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            try:
                if float(a) != float(b):
                    return False
                continue
            except Exception:
                pass
        if a != b:
            return False
    return True


def formal_schema_columns() -> list[str]:
    return load_formal_scores().columns.tolist()
