#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import hashlib
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path("/opt/data/private/JinRong/kechuang/financial_event_project")
ROUND1_OUT = ROOT / "outputs/task4_trace_refine_v3_round1"
OUT = ROOT / "outputs/task4_trace_refine_v3_1_t3_recovery"
MODULE_PATH = ROOT / "scripts/build_task4_trace_refine_v3_round1.py"
MODEL_BUNDLE = ROUND1_OUT / "models/task4_trace_refine_v3_teachers.joblib"
DIANJIN_BASE_URL = "http://127.0.0.1:8000/v1"
DIANJIN_MODEL_ID = "DianJin-R1-32B"
DIANJIN_TIMEOUT = 300
DIANJIN_MAX_TOKENS = 3072


def disable_local_proxy() -> None:
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"


def load_round1_module():
    spec = importlib.util.spec_from_file_location("task4_trace_refine_v3_round1_source", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_existing_predictions(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def dianjin_request_recovery(prompt: str, timeout: int = DIANJIN_TIMEOUT) -> str:
    payload = {
        "model": DIANJIN_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": DIANJIN_MAX_TOKENS,
    }
    req = urllib.request.Request(
        f"{DIANJIN_BASE_URL}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_json_recovery(text: str) -> dict[str, Any]:
    """Prefer the model answer block, then decode the first valid JSON object."""
    text = str(text or "")
    answer = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    candidates = [answer.group(1)] if answer else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.I | re.S)
        scan = fenced.group(1) if fenced else candidate
        for idx, char in enumerate(scan):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(scan[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return {}


def service_diagnostic() -> dict[str, Any]:
    report: dict[str, Any] = {
        "base_url": DIANJIN_BASE_URL,
        "model_id_requested": DIANJIN_MODEL_ID,
        "timeout_seconds": DIANJIN_TIMEOUT,
        "max_tokens": DIANJIN_MAX_TOKENS,
        "proxy_disabled": True,
        "endpoints": {},
    }
    models_url = f"{DIANJIN_BASE_URL}/models"
    try:
        req = urllib.request.Request(models_url, headers={"Authorization": "Bearer dummy"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body)
            report["endpoints"]["models"] = {
                "url": models_url,
                "http_status": int(resp.status),
                "model_ids": [str(item.get("id", "")) for item in payload.get("data", [])],
            }
    except Exception as exc:
        report["endpoints"]["models"] = {
            "url": models_url,
            "error_type": type(exc).__name__,
            "error_summary": str(exc)[:500],
        }
    return report


def aggregate_t3_reports(reports: list[dict[str, Any]], total_requested: int) -> dict[str, Any]:
    agg = {
        "enabled": True,
        "requested_max_samples": int(total_requested),
        "completed": 0,
        "parsed": 0,
        "unparsed": 0,
        "errors": 0,
        "raw_responses_persisted": False,
        "chunks": [],
    }
    for report in reports:
        agg["completed"] += int(report.get("completed", 0) or 0)
        agg["parsed"] += int(report.get("parsed", 0) or 0)
        agg["unparsed"] += int(report.get("unparsed", 0) or 0)
        agg["errors"] += int(report.get("errors", 0) or 0)
        agg["chunks"].append(report)
    agg["status"] = "complete_with_errors" if agg["errors"] else "complete"
    return agg


def run_t3_incremental(
    module,
    samples: pd.DataFrame,
    total_limit: int,
    smoke_samples: int,
    chunk_size: int,
    resume: bool,
    pred_path: Path,
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    samples = samples.drop_duplicates("sample_id").head(total_limit).reset_index(drop=True)
    existing = load_existing_predictions(pred_path) if resume else pd.DataFrame()
    processed_ids = set(existing["sample_id"].astype(str).tolist()) if not existing.empty and "sample_id" in existing.columns else set()
    pending = samples[~samples["sample_id"].astype(str).isin(processed_ids)].reset_index(drop=True)
    reports: list[dict[str, Any]] = []
    rows = [existing] if not existing.empty else []

    if not pending.empty and not resume:
        smoke_take = min(smoke_samples, len(pending))
        smoke_df = pending.head(smoke_take).copy()
        smoke_pred, smoke_report = module.run_t3_teacher(smoke_df, smoke_take, True)
        if int(smoke_report.get("errors", 0) or 0) or int(smoke_report.get("unparsed", 0) or 0) != 0:
            checkpoint = {
                "phase": "smoke_failed",
                "smoke_samples": int(smoke_take),
                "smoke_report": smoke_report,
                "pred_path": str(pred_path),
            }
            write_json(checkpoint_path, checkpoint)
            raise RuntimeError(f"T3 smoke test failed: {smoke_report}")
        rows.append(smoke_pred)
        reports.append({**smoke_report, "phase": "smoke"})
        processed_ids.update(smoke_pred["sample_id"].astype(str).tolist())
        pending = pending.iloc[smoke_take:].reset_index(drop=True)
        combined = pd.concat(rows, ignore_index=True).drop_duplicates("sample_id")
        combined.to_parquet(pred_path, index=False)
        write_json(
            checkpoint_path,
            {
                "phase": "smoke_passed",
                "processed_count": int(len(combined)),
                "processed_sample_ids_sha256": sha256_text("\n".join(sorted(combined["sample_id"].astype(str).tolist()))),
                "pred_path": str(pred_path),
                "smoke_samples": int(smoke_take),
            },
        )

    while not pending.empty:
        chunk = pending.head(chunk_size).copy()
        pending = pending.iloc[len(chunk) :].reset_index(drop=True)
        chunk_pred, chunk_report = module.run_t3_teacher(chunk, len(chunk), True)
        rows.append(chunk_pred)
        reports.append({**chunk_report, "phase": "chunk"})
        combined = pd.concat(rows, ignore_index=True).drop_duplicates("sample_id")
        combined = combined.sort_values("sample_id").reset_index(drop=True)
        combined.to_parquet(pred_path, index=False)
        processed_ids.update(chunk_pred["sample_id"].astype(str).tolist())
        write_json(
            checkpoint_path,
            {
                "phase": "running" if not pending.empty else "complete",
                "processed_count": int(len(combined)),
                "remaining_count": int(len(pending)),
                "processed_sample_ids_sha256": sha256_text("\n".join(sorted(combined["sample_id"].astype(str).tolist()))),
                "pred_path": str(pred_path),
                "chunk_size": int(chunk_size),
            },
        )

    final = pd.concat(rows, ignore_index=True).drop_duplicates("sample_id").sort_values("sample_id").reset_index(drop=True) if rows else pd.DataFrame()
    return final, aggregate_t3_reports(reports, total_limit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-t3-samples", type=int, default=120)
    parser.add_argument("--smoke-samples", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()

    disable_local_proxy()

    if OUT.exists() and any(OUT.iterdir()) and not args.allow_existing and not args.resume:
        raise RuntimeError(f"output directory exists and is not empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)

    module = load_round1_module()
    sys.modules["task4_trace_refine_v3_round1"] = module
    sys.modules["build_task4_trace_refine_v3_round1"] = module
    module.dianjin_request = dianjin_request_recovery
    module.extract_json = extract_json_recovery
    module.OUT = OUT
    module.MODELS = OUT / "models"
    module.CREATED_AT = datetime.now(timezone(timedelta(hours=8))).isoformat()

    service_report = service_diagnostic()
    write_json(OUT / "t3_service_diagnostic.json", service_report)
    model_ids = service_report.get("endpoints", {}).get("models", {}).get("model_ids", [])
    if DIANJIN_MODEL_ID not in model_ids:
        raise RuntimeError(f"requested local model id is unavailable: {service_report}")

    # Make the round1-pickled bundle loadable in this process.
    import __main__

    __main__.TextTeacher = module.TextTeacher

    before = module.protected_sha256()
    module.write_json(OUT / "protected_sha256_before.json", before)

    sources = module.load_annotation_sources()
    paths_read = [str(path) for path in module.PROTECTED_FILES.values()]
    ctx = module.canonical_context(sources)
    supervision, inventory, mapping_audit = module.build_field_supervision(sources, ctx)
    masked_ids, split_manifest = module.make_grouped_split(supervision)

    bundle = joblib.load(MODEL_BUNDLE)
    models = bundle["models"]

    teacher_metrics_path = ROUND1_OUT / "teacher_metrics.json"
    teacher_metrics = json.loads(teacher_metrics_path.read_text(encoding="utf-8"))

    t3_samples = ctx[ctx["sample_id"].isin(masked_ids)].copy()
    readiness = sources["readiness"]
    silver_b = readiness[readiness["label_tier"].isin(["Silver-B", "excluded_or_review"])].drop_duplicates("sample_id")
    t3_samples = pd.concat([t3_samples, silver_b], ignore_index=True).drop_duplicates("sample_id")

    pred_path = OUT / "t3_dianjin_teacher_predictions.parquet"
    checkpoint_path = OUT / "t3_recovery_checkpoint.json"
    t3_pred, t3_report = run_t3_incremental(
        module=module,
        samples=t3_samples,
        total_limit=args.max_t3_samples,
        smoke_samples=args.smoke_samples,
        chunk_size=args.chunk_size,
        resume=args.resume,
        pred_path=pred_path,
        checkpoint_path=checkpoint_path,
    )

    t3_metrics: dict[str, Any] = {"status": t3_report.get("status"), "fields": {}}
    if not t3_pred.empty:
        for field in module.ALL_FIELDS:
            if field not in t3_pred.columns:
                continue
            gold = supervision[(supervision["sample_id"].isin(masked_ids)) & supervision["field"].eq(field)].sort_values("weight", ascending=False).drop_duplicates("sample_id")
            merged = gold[["sample_id", "label"]].merge(t3_pred[["sample_id", field]].dropna(), on="sample_id", how="inner")
            merged = merged[merged[field].map(module.norm).ne("")]
            if len(merged):
                t3_metrics["fields"][field] = module.metric_for_field(field, merged["label"], merged[field].to_numpy())
    teacher_metrics["t3_dianjin_r1_32b"] = {
        **t3_report,
        "service": service_report,
        "metrics": t3_metrics,
        "checkpoint": {
            "checkpoint_file": str(checkpoint_path),
            "pred_path": str(pred_path),
            "resume_enabled": bool(args.resume),
            "smoke_samples": int(args.smoke_samples),
            "chunk_size": int(args.chunk_size),
        },
    }

    refined, pseudo_report = module.pseudo_label_rounds(supervision, sources, ctx, models, teacher_metrics, t3_pred, masked_ids)
    sam_report = module.sam_ablation(supervision, masked_ids)

    refined.to_parquet(OUT / "refined_training_pool.parquet", index=False)
    module.make_priority_review(refined, ctx, pseudo_report)
    joblib.dump({"models": models, "created_at": module.CREATED_AT, "supervision_inventory": inventory}, OUT / "models/task4_trace_refine_v3_teachers.joblib")
    module.write_inference_pipeline()

    (OUT / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /opt/data/private/JinRong/kechuang/financial_event_project\nPYTHONPATH=/root/miniconda3/envs/task4-impact-v3/lib/python3.11/site-packages:$PYTHONPATH NO_PROXY=127.0.0.1,localhost HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= /root/miniconda3/envs/dianjin-cu128/bin/python scripts/run_task4_trace_refine_v3_1_t3_recovery.py --allow-existing --max-t3-samples 120 --smoke-samples 3 --chunk-size 10\n",
        encoding="utf-8",
    )

    dependency_report = {
        "python": sys.executable,
        "lightgbm_available": getattr(module, "lgb", None) is not None,
        "setfit_available": False,
        "sentence_transformers_available": False,
        "torch_available": getattr(module, "torch", None) is not None,
        "environment_policy": "Used local DianJin-R1-32B at 127.0.0.1:8000 with proxies disabled; no official1799/final_holdout read.",
    }

    module.write_json(OUT / "supervision_inventory.json", inventory)
    module.write_json(OUT / "label_mapping_audit.json", mapping_audit)
    module.write_json(OUT / "grouped_split_manifest.json", split_manifest)
    module.write_json(OUT / "teacher_metrics.json", teacher_metrics)
    module.write_json(OUT / "pseudo_label_report.json", pseudo_report)
    module.write_json(OUT / "student_ablation_report.json", sam_report)
    module.write_json(OUT / "leakage_report.json", module.leakage_report(paths_read, dependency_report))

    readme = f"""# Task4 TRACE-Refine v3.1 T3 Recovery

Generated at: {module.CREATED_AT}

## Notes
- Reused round1 supervision and trained teacher bundle.
- Recovered T3 only against the local `DianJin-R1-32B` service on `http://127.0.0.1:8000/v1`.
- Proxies were disabled for localhost requests.
- No official1799 filled gold or final holdout was read.

## Outputs
- T3 predictions: `t3_dianjin_teacher_predictions.parquet`
- Final refined pool: `refined_training_pool.parquet`
- Teacher metrics: `teacher_metrics.json`
- Pseudo-label report: `pseudo_label_report.json`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    after = module.protected_sha256()
    module.write_json(OUT / "protected_sha256_after.json", after)
    module.write_json(OUT / "output_sha256.json", module.output_sha256())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
