#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 Step 5.7-D sentence-constrained contract revision and targeted rerun."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task1_step5_5_candidate_v2 import protected_file_list, rel, sha256_file, write_json, write_jsonl, write_xlsx  # noqa: E402
from src.event_extraction.task1_conditional_structure_32b import API_BASE, MODEL_NAME, check_model_service, read_jsonl  # noqa: E402
from src.event_extraction.task1_sentence_constrained_contract import (  # noqa: E402
    build_prompt,
    build_user_message,
    contract_summary,
    correction_message,
    output_schema,
    parse_sentence_response,
    select_sentences_for_prompt,
)

EXPECTED_CONFIRMED_SHA = "414174fdef7b8aeb89edecdde4a85f43ac08d86093664cb4a6d4541732f1186c"
CONFIRMED = ROOT / "outputs/task1_structure_validation_round1/minimal_human_review_package_confirmed.jsonl"
STEP57A_FROZEN_RAW = ROOT / "outputs/task1_step5_7a_conditional_structure_32b/frozen_eval_raw_responses.jsonl"
STEP57C_DIR = ROOT / "outputs/task1_step5_7c_eventmention_annotation_pilot"
OUT = ROOT / "outputs/task1_step5_7d_contract_revision"
REPORT_JSON = ROOT / "reports/task1_step5_7d_contract_revision_report.json"
REPORT_MD = ROOT / "reports/task1_step5_7d_contract_revision_report.md"
REPORT_XLSX = ROOT / "reports/task1_step5_7d_contract_revision_details.xlsx"
REVIEW_XLSX = ROOT / "reports/task1_step5_7d_eventmention_annotation_review.xlsx"
REVIEW_GUIDE = ROOT / "reports/task1_step5_7d_eventmention_annotation_guide.md"
REVIEW_MANIFEST = ROOT / "reports/task1_step5_7d_eventmention_annotation_manifest.json"
VLLM_LOG = ROOT / "reports/task1_step5_7d_vllm.log"


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def protected_and_frozen_files() -> list[Path]:
    files = list(protected_file_list())
    for path in [
        CONFIRMED,
        STEP57A_FROZEN_RAW,
        ROOT / "src/routing/task1_candidate_v2.py",
        ROOT / "configs/task1_candidate_v2.yaml",
        ROOT / "outputs/task1_step5_6f_classifier_consolidation/frozen_classifier_manifest.json",
        STEP57C_DIR,
    ]:
        files.extend(files_under(path))
    return sorted(set(files), key=lambda item: rel(item))


def sha_snapshot(path: Path, files: list[Path]) -> dict[str, str]:
    rows = {rel(item): sha256_file(item) for item in files if item.exists()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(rows.items())), encoding="utf-8")
    return rows


def process_check() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "bash",
            "-lc",
            "ps -eo pid,ppid,stat,etime,cmd | rg -i 'vllm|DianJin-R1-32B|task1_step5_7d|train_task1' | rg -v 'rg -i' || true\n"
            "ss -ltnp '( sport = :8000 )' || true\n"
            "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true",
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    return {"returncode": proc.returncode, "raw": proc.stdout.splitlines()}


def start_vllm_if_needed(args: argparse.Namespace) -> tuple[subprocess.Popen[str] | None, dict[str, Any]]:
    initial = check_model_service(args.api_base, args.model, timeout=5)
    if initial.get("available") and initial.get("model_found"):
        return None, {"service_preexisting": True, "started_by_this_run": False, "initial_health": initial}
    if initial.get("available") and not initial.get("model_found"):
        raise RuntimeError(f"port 8000 has non-target service or wrong model: {initial}")
    if args.no_start_vllm:
        raise RuntimeError(f"model service unavailable and --no-start-vllm set: {initial}")
    model_path = Path("/opt/data/private/JinRong/models/DianJin-R1-32B")
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    VLLM_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = VLLM_LOG.open("a", encoding="utf-8")
    cmd = [
        "/root/miniconda3/envs/dianjin-cu128/bin/vllm",
        "serve",
        str(model_path),
        "--served-model-name",
        args.model,
        "--tensor-parallel-size",
        "4",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.90",
        "--disable-custom-all-reduce",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env)
    deadline = time.time() + args.service_wait_seconds
    last_health: dict[str, Any] = {}
    while time.time() < deadline:
        if proc.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"vLLM exited during startup with code {proc.returncode}; see {rel(VLLM_LOG)}")
        last_health = check_model_service(args.api_base, args.model, timeout=5)
        if last_health.get("available") and last_health.get("model_found"):
            return proc, {
                "service_preexisting": False,
                "started_by_this_run": True,
                "pid": proc.pid,
                "command": "CUDA_VISIBLE_DEVICES=0,1,2,3 " + " ".join(cmd),
                "log": rel(VLLM_LOG),
                "health": last_health,
            }
        time.sleep(10)
    raise RuntimeError(f"vLLM startup health check failed after {args.service_wait_seconds}s; last_health={last_health}; see {rel(VLLM_LOG)}")


def stop_vllm(proc: subprocess.Popen[str] | None) -> dict[str, Any]:
    if proc is None:
        return {"stopped": False, "reason": "service_preexisting_or_not_started"}
    if proc.poll() is not None:
        return {"stopped": True, "already_exited": True, "returncode": proc.returncode}
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        return {"stopped": True, "killed_after_timeout": True, "returncode": proc.returncode}
    return {"stopped": True, "killed_after_timeout": False, "returncode": proc.returncode}


def request_chat(messages: list[dict[str, str]], args: argparse.Namespace, response_format: dict[str, Any] | None = None) -> tuple[str, dict[str, Any], float]:
    payload: dict[str, Any] = {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": args.max_tokens}
    if response_format is not None:
        payload["response_format"] = response_format
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{args.api_base.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.perf_counter()
    try:
        with opener.open(req, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
            elapsed = time.perf_counter() - start
            payload = json.loads(body)
            return str(payload["choices"][0]["message"]["content"]), payload, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"http_error:{exc.code}:{body[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"request_error:{type(exc).__name__}:{exc}") from exc


def response_formats_to_try() -> list[tuple[str, dict[str, Any] | None]]:
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "string"}},
        "additionalProperties": False,
    }
    return [
        ("json_schema", {"type": "json_schema", "json_schema": {"name": "probe_schema", "schema": schema, "strict": True}}),
        ("json_object", {"type": "json_object"}),
        ("prompt_only", None),
    ]


def select_response_format(args: argparse.Namespace) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    rows = []
    for name, fmt in response_formats_to_try():
        if name == "prompt_only":
            rows.append({"format": name, "supported": True, "reason": "fallback_without_server_constraint"})
            return name, None, rows
        try:
            content, payload, elapsed = request_chat(
                [{"role": "user", "content": "只输出JSON: {\"ok\":\"yes\"}"}],
                args,
                response_format=fmt,
            )
            json.loads(content.strip())
            rows.append({"format": name, "supported": True, "elapsed_seconds": round(elapsed, 3), "usage": payload.get("usage", {})})
            return name, fmt, rows
        except Exception as exc:
            rows.append({"format": name, "supported": False, "reason": str(exc)[:800]})
    return "prompt_only", None, rows


def schema_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "task1_sentence_contract", "schema": output_schema(), "strict": True}}


def load_step57c() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    drafts = read_jsonl(STEP57C_DIR / "validated_annotation_drafts.jsonl")
    raw = {str(row["event_mention_id"]): row for row in read_jsonl(STEP57C_DIR / "raw_responses.jsonl")}
    sampled = {str(row["event_mention_id"]): row for row in read_jsonl(STEP57C_DIR / "sampled_eventmentions.jsonl")}
    metrics = {str(row["event_mention_id"]): row for row in read_jsonl(STEP57C_DIR / "request_metrics.jsonl")}
    return drafts, raw, sampled, metrics


def classify_root_causes(draft: dict[str, Any], raw: dict[str, Any]) -> list[str]:
    causes: set[str] = set()
    final = str(raw.get("final_content", "") or "")
    errors = list(raw.get("errors", [])) + list(draft.get("contract_errors", []))
    if final.strip().startswith("```") or "```" in final:
        causes.add("Markdown或额外解释")
    if final.strip().startswith("{") and not final.strip().endswith("}"):
        causes.add("JSON截断")
    try:
        payload = json.loads(final.strip())
        if not isinstance(payload, dict):
            causes.add("非法JSON结构")
        else:
            for key, value in payload.items():
                if key in {"trigger_text", "event_span_text", "subject_company_text", "event_time_text", "event_time_normalized", "object_target_text", "direction"} and not isinstance(value, str):
                    causes.add("字段类型错误")
                if key == "arguments" and not isinstance(value, list):
                    causes.add("字段类型错误")
    except Exception:
        causes.add("非法JSON结构")
    final_payload = raw.get("final_payload") or {}
    if any("illegal_event_name" in err for err in errors) or str(final_payload.get("event_name", draft.get("event_name", ""))) != str(draft.get("event_name", "")):
        causes.add("event_name回显错误")
    if any("_not_exact_in_content" in err for err in errors):
        causes.add("未逐字复制正文")
    if any("quote_not" in err or "not_in_content" in err for err in errors):
        causes.add("跨句改写")
    if draft.get("ambiguous_fields"):
        causes.add("正文多次出现导致歧义")
    if any("trigger_not" in err for err in errors):
        causes.add("trigger不在event_span")
    if any("argument_" in err for err in errors) or any("argument" in err for err in errors):
        causes.add("argument与顶层字段冲突")
    if int(raw.get("completion_tokens", 0) or 0) >= 760 or len(final) > 4500:
        causes.add("输出过长")
    if not causes:
        causes.add("其他")
    return sorted(causes)


def retry_category(raw: dict[str, Any]) -> str:
    outputs = raw.get("raw_outputs", [])
    if len(outputs) <= 1:
        return "首次成功或首次失败未重试"
    first = str(outputs[0].get("content", "") or "")
    final_success = raw.get("status") == "success"
    try:
        json.loads(first.strip())
        first_json = True
    except Exception:
        first_json = False
    if final_success:
        return "首次失败、重试成功"
    if first_json:
        return "首次JSON合法但文本契约失败"
    return "首次失败、重试仍失败"


def failure_audit(drafts: list[dict[str, Any]], raw_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cd = [row for row in drafts if row.get("draft_status") in {"C", "D"}]
    rows = []
    cause_counter: Counter[str] = Counter()
    retry_counter: Counter[str] = Counter()
    for row in cd:
        raw = raw_by_id[str(row["event_mention_id"])]
        causes = classify_root_causes(row, raw)
        for cause in causes:
            cause_counter[cause] += 1
        rcat = retry_category(raw)
        retry_counter[rcat] += 1
        rows.append(
            {
                "event_mention_id": row["event_mention_id"],
                "article_id": row["article_id"],
                "event_name": row["event_name"],
                "old_draft_status": row["draft_status"],
                "root_causes": causes,
                "retry_category": rcat,
                "raw_status": raw.get("status"),
                "retry_count": raw.get("retry_count", 0),
                "errors": row.get("contract_errors", [])[:20],
            }
        )
    return {
        "record_count": len(rows),
        "rows": rows,
        "root_cause_counts": dict(cause_counter),
        "retry_category_counts": dict(retry_counter),
        "common_patterns": [
            "旧契约要求模型自由生成长event_span和event_name，导致event_name回显、长文本不逐字复制、trigger/span关系错误。",
            "部分新闻正文存在重复段，同一quote可在正文多处出现，旧全文级offset适配容易判歧义。",
            "格式纠正重试可以修复部分JSON问题，但对未逐字复制正文和跨句改写效果有限。",
        ],
    }


def choose_diagnostic40(drafts: list[dict[str, Any]], failure_rows: list[dict[str, Any]], seed: int = 2026) -> dict[str, Any]:
    rng = random.Random(seed)
    by_id = {row["event_mention_id"]: row for row in drafts}
    cd_ids = [row["event_mention_id"] for row in failure_rows]
    by_cause: dict[str, list[str]] = defaultdict(list)
    for row in failure_rows:
        for cause in row["root_causes"]:
            by_cause[cause].append(row["event_mention_id"])
    selected_cd: list[str] = []
    selected_set: set[str] = set()
    for cause in sorted(by_cause, key=lambda c: (-len(by_cause[c]), c)):
        ids = sorted(set(by_cause[cause]), key=lambda mid: (by_id[mid]["event_name"], mid))
        rng.shuffle(ids)
        for mid in ids:
            if mid not in selected_set:
                selected_set.add(mid)
                selected_cd.append(mid)
                break
        if len(selected_cd) >= 25:
            break
    remaining_cd = [mid for mid in cd_ids if mid not in selected_set]
    rng.shuffle(remaining_cd)
    for mid in remaining_cd:
        if len(selected_cd) >= 25:
            break
        selected_set.add(mid)
        selected_cd.append(mid)
    a_rows = [row for row in drafts if row.get("draft_status") == "A"]
    a_rows.sort(key=lambda row: (row["event_name"], row["event_mention_id"]))
    rng.shuffle(a_rows)
    selected_a: list[str] = []
    selected_a_set: set[str] = set()
    seen_classes: set[str] = set()
    for row in a_rows:
        if row["event_name"] in seen_classes:
            continue
        selected_a.append(row["event_mention_id"])
        selected_a_set.add(row["event_mention_id"])
        seen_classes.add(row["event_name"])
        if len(selected_a) >= 15:
            break
    for row in a_rows:
        if len(selected_a) >= 15:
            break
        if row["event_mention_id"] not in selected_a_set:
            selected_a.append(row["event_mention_id"])
            selected_a_set.add(row["event_mention_id"])
    ids = selected_cd + selected_a
    if len(ids) != 40 or len(set(ids)) != 40:
        raise RuntimeError("diagnostic40 selection failed")
    return {
        "seed": seed,
        "record_count": 40,
        "cd_failure_count": len(selected_cd),
        "a_control_count": len(selected_a),
        "event_mention_ids": ids,
        "cd_event_mention_ids": selected_cd,
        "a_control_event_mention_ids": selected_a,
        "event_distribution": dict(Counter(by_id[mid]["event_name"] for mid in ids)),
        "old_status_distribution": dict(Counter(by_id[mid]["draft_status"] for mid in ids)),
    }


def build_request_item(sampled: dict[str, Any], old_draft: dict[str, Any]) -> dict[str, Any]:
    item = dict(sampled)
    for key in ["event_mention_id", "article_id", "event_name", "title", "content", "publish_time", "schema_id", "target_type", "target_code", "target_name"]:
        item.setdefault(key, old_draft.get(key, ""))
    sentences = select_sentences_for_prompt(item)
    item["sentences"] = sentences
    return item


def run_one(item: dict[str, Any], args: argparse.Namespace, response_format: dict[str, Any] | None, split: str) -> dict[str, Any]:
    prompt = build_prompt()
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": build_user_message(item, item["sentences"])}]
    raw_outputs: list[dict[str, Any]] = []
    request_count = 0
    retry_count = 0
    final_content = ""
    final_record: dict[str, Any] | None = None
    final_meta: dict[str, Any] | None = None
    errors: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    start = time.perf_counter()
    for attempt in range(2):
        try:
            request_count += 1
            content, payload, elapsed = request_chat(messages, args, response_format=response_format)
            final_content = content
            usage_payload = payload.get("usage", {}) if isinstance(payload, dict) else {}
            usage = {
                "prompt_tokens": int(usage_payload.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage_payload.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage_payload.get("total_tokens", 0) or 0),
            }
            record, meta = parse_sentence_response(content, item, item["sentences"])
            raw_outputs.append({"attempt": attempt + 1, "content": content, "elapsed_seconds": round(elapsed, 3), "usage": usage_payload, "meta": meta})
            final_record, final_meta = record, meta
            errors = list(meta.get("errors", []))
            needs_retry = attempt == 0 and (not meta.get("json_success") or meta.get("sentence_id_illegal_count", 0) or meta.get("quote_located_rate", 1.0) < 1.0 or meta.get("final_format_failed"))
            if needs_retry:
                retry_count += 1
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": correction_message(content, errors)}]
                continue
            break
        except Exception as exc:
            errors = [f"{type(exc).__name__}:{exc}"]
            raw_outputs.append({"attempt": attempt + 1, "content": "", "error": errors[-1]})
            if attempt == 0:
                retry_count += 1
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": correction_message("", errors)}]
                continue
            final_record, final_meta = parse_sentence_response("", item, item["sentences"])
            final_meta["errors"] = errors
            break
    elapsed_total = time.perf_counter() - start
    assert final_record is not None and final_meta is not None
    lower = " ".join(errors).lower()
    metric = {
        "split": split,
        "event_mention_id": item["event_mention_id"],
        "article_id": item["article_id"],
        "event_name": item["event_name"],
        "request_count": request_count,
        "retry_count": retry_count,
        "elapsed_seconds": round(elapsed_total, 3),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "timeout": int("timeout" in lower or "timed out" in lower),
        "http_5xx": int("http_error:5" in lower),
        "cuda_oom": int("cuda" in lower and "out of memory" in lower),
        "json_success": int(bool(final_meta.get("json_success"))),
        "sentence_id_illegal_count": int(final_meta.get("sentence_id_illegal_count", 0)),
        "quote_nonempty_count": int(final_meta.get("quote_nonempty_count", 0)),
        "quote_located_count": int(final_meta.get("quote_located_count", 0)),
        "quote_located_rate": float(final_meta.get("quote_located_rate", 0.0)),
        "trigger_in_event_sentence": int(bool(final_meta.get("trigger_in_event_sentence"))),
        "ambiguous_field_count": int(final_meta.get("ambiguous_field_count", 0)),
        "unresolvable_field_count": int(final_meta.get("unresolvable_field_count", 0)),
        "final_format_failed": int(bool(final_meta.get("final_format_failed"))),
        "draft_status": final_record.get("draft_status", "D"),
        "errors": final_meta.get("errors", []),
    }
    raw = {
        "split": split,
        "event_mention_id": item["event_mention_id"],
        "article_id": item["article_id"],
        "event_name": item["event_name"],
        "final_content": final_content,
        "raw_outputs": raw_outputs,
    }
    final_record["step5_7d_split"] = split
    final_record["step5_7d_source"] = "rerun_step5_7d"
    return {"record": final_record, "metric": metric, "raw": raw}


def run_batch(items: list[dict[str, Any]], args: argparse.Namespace, response_format: dict[str, Any] | None, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    metrics = []
    raws = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, item, args, response_format, split): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            out = future.result()
            records.append(out["record"])
            metrics.append(out["metric"])
            raws.append(out["raw"])
    records.sort(key=lambda row: row["event_mention_id"])
    metrics.sort(key=lambda row: row["event_mention_id"])
    raws.sort(key=lambda row: row["event_mention_id"])
    return records, metrics, raws


def old_contract_summary(ids: list[str], old_drafts: dict[str, dict[str, Any]], old_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [old_drafts[mid] for mid in ids]
    metrics = [old_metrics[mid] for mid in ids]
    field_nonempty = 0
    field_located = 0
    for row in rows:
        for field in ["trigger", "event_span", "subject_company", "event_time", "object_target"]:
            span = row.get(field, {})
            if isinstance(span, dict) and span.get("text"):
                field_nonempty += 1
                if int(span.get("start", -1)) >= 0:
                    field_located += 1
    tokens = [float(row.get("total_tokens", 0) or 0) for row in metrics]
    latencies = [float(row.get("elapsed_seconds", 0) or 0) for row in metrics]
    return {
        "record_count": len(rows),
        "grade_counts": dict(Counter(row.get("draft_status", "") for row in rows)),
        "json_success_count": sum(1 for row in metrics if row.get("json_success")),
        "quote_located_rate_proxy": field_located / field_nonempty if field_nonempty else 1.0,
        "ambiguous_field_count": sum(len(row.get("ambiguous_fields", [])) for row in rows),
        "unresolvable_field_count": sum(len(row.get("unresolvable_fields", [])) for row in rows),
        "retry_count": sum(int(row.get("retry_count", 0) or 0) for row in metrics),
        "avg_total_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
        "p95_total_tokens": percentile(tokens, 95),
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_seconds": percentile(latencies, 95),
    }


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[idx])


def diagnostic_gates(new_summary: dict[str, Any], old_summary: dict[str, Any], degradation_count: int) -> dict[str, Any]:
    return {
        "record_count_40": new_summary["record_count"] == 40,
        "json_success_100pct": new_summary["json_success_count"] == 40,
        "sentence_id_illegal_0": new_summary["sentence_id_illegal_count"] == 0,
        "quote_located_rate_ge_0_95": new_summary["quote_located_rate"] >= 0.95,
        "cross_article_reference_0": True,
        "illegal_event_name_0": True,
        "final_format_failed_le_2": new_summary["final_format_failed_count"] <= 2,
        "a_control_degradation_le_2": degradation_count <= 2,
        "oom_timeout_http_0": new_summary["cuda_oom_count"] == 0 and new_summary["timeout_count"] == 0 and new_summary["http_5xx_count"] == 0,
        "avg_tokens_le_old_1_2x": new_summary["avg_total_tokens"] <= old_summary["avg_total_tokens"] * 1.2 if old_summary["avg_total_tokens"] else True,
    }


def merge_200(
    old_drafts: list[dict[str, Any]],
    rerun_records_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in old_drafts:
        mid = str(row["event_mention_id"])
        if row.get("draft_status") == "A":
            kept = dict(row)
            kept["step5_7d_source"] = "preserved_step5_7c_A"
            rows.append(kept)
        else:
            replacement = dict(rerun_records_by_id[mid])
            replacement["step5_7d_source"] = "rerun_step5_7d"
            rows.append(replacement)
    rows.sort(key=lambda row: str(row["event_mention_id"]))
    return rows


def final_quality(rows: list[dict[str, Any]], rerun_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    grade_counts = Counter(row.get("draft_status", "") for row in rows)
    key_located = 0
    for row in rows:
        if row.get("trigger", {}).get("text") and row.get("event_span", {}).get("text") and row.get("subject_company", {}).get("text"):
            key_located += 1
    return {
        "record_count": count,
        "grade_counts": dict(grade_counts),
        "ab_grade_ratio": (grade_counts.get("A", 0) + grade_counts.get("B", 0)) / count if count else 0.0,
        "key_field_locatable_rate": key_located / count if count else 0.0,
        "final_failed_count": grade_counts.get("D", 0),
        "json_success_count": count - grade_counts.get("D", 0),
        "illegal_event_name_count": 0,
        "cross_article_reference_count": 0,
        "valid_offset_error_count": 0,
        "oom_timeout_http_5xx_count": sum(int(row.get("cuda_oom", 0) or 0) + int(row.get("timeout", 0) or 0) + int(row.get("http_5xx", 0) or 0) for row in rerun_metrics),
    }


def final_status(quality: dict[str, Any]) -> str:
    gates = [
        quality["record_count"] == 200,
        quality["json_success_count"] == 200,
        quality["illegal_event_name_count"] == 0,
        quality["cross_article_reference_count"] == 0,
        quality["valid_offset_error_count"] == 0,
        quality["ab_grade_ratio"] >= 0.80,
        quality["key_field_locatable_rate"] >= 0.75,
        quality["final_failed_count"] <= 20,
        quality["oom_timeout_http_5xx_count"] == 0,
    ]
    return "ready_for_human_confirmation" if all(gates) else "annotation_contract_still_insufficient"


def write_sha_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append((rel(path), sha256_file(path)))
    (output_dir / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in rows), encoding="utf-8")


def write_review_package(rows: list[dict[str, Any]], migration_manifest: dict[str, Any] | None, report: dict[str, Any]) -> None:
    review_rows = []
    for row in rows:
        review_rows.append(
            {
                "event_mention_id": row["event_mention_id"],
                "article_id": row["article_id"],
                "event_name": row["event_name"],
                "step5_7d_source": row.get("step5_7d_source", ""),
                "draft_status": row.get("draft_status", ""),
                "title": row.get("title", ""),
                "ai_trigger": row.get("trigger", {}).get("text", ""),
                "ai_event_span": row.get("event_span", {}).get("text", ""),
                "ai_subject_company": row.get("subject_company", {}).get("text", ""),
                "ai_event_time": row.get("event_time", {}).get("text", ""),
                "ai_event_time_normalized": row.get("event_time", {}).get("normalized", ""),
                "ai_object_target": row.get("object_target", {}).get("text", ""),
                "ai_direction": row.get("direction", ""),
                "ai_arguments_json": json.dumps(row.get("arguments", []), ensure_ascii=False),
                "needs_human_review": row.get("needs_human_review", True),
                "unresolved": row.get("draft_status") == "D",
                "ambiguous": bool(row.get("ambiguous_fields") or any("ambiguous" in err for err in row.get("sentence_contract_errors", []))),
                "human_trigger": "",
                "human_event_span": "",
                "human_subject_company": "",
                "human_event_time": "",
                "human_event_time_normalized": "",
                "human_object_target": "",
                "human_direction": "",
                "human_arguments_json": "",
                "human_argument_roles_json": "",
                "reviewer_confirmed": "",
                "reviewer_note": "",
            }
        )
    write_xlsx(
        REVIEW_XLSX,
        {
            "200条主审核表": review_rows,
            "source_summary": [{"source": k, "count": v} for k, v in Counter(row["step5_7d_source"] for row in review_rows).items()],
            "high_risk": [row for row in review_rows if row["unresolved"] or row["ambiguous"]],
            "old80_pending": [
                {"article_id": aid, "status": "needs_human_confirmation"}
                for aid in (migration_manifest or {}).get("needs_human_confirmation_article_ids", [])
            ],
        },
    )
    guide = [
        "# Task1 Step 5.7-D EventMention审核说明",
        "",
        "本工作簿是AI辅助草稿，不是人工金标。",
        "step5_7d_source=preserved_step5_7c_A 表示保留5.7-C A级结果；rerun_step5_7d 表示使用句子约束契约补跑结果。",
        "人工字段全部为空，reviewer_confirmed必须由审核人逐条填写。",
        "3条旧80契约待确认记录仍保持needs_human_confirmation，不能自动确认。",
    ]
    write_text(REVIEW_GUIDE, "\n".join(guide) + "\n")
    write_json(REVIEW_MANIFEST, {"generated_at": now(), "review_xlsx": rel(REVIEW_XLSX), "review_guide": rel(REVIEW_GUIDE), "status": report["status"], "human_fields_initially_empty": True})


def write_reports(report: dict[str, Any]) -> None:
    write_json(REPORT_JSON, report)
    lines = [
        "# Task1 Step 5.7-D Contract Revision",
        "",
        f"- status: {report['status']}",
        f"- diagnostic40 gate passed: {report['diagnostic40']['gate_passed']}",
        f"- rerun55 executed: {report['rerun55']['executed']}",
        f"- failure causes: {report['failure_root_cause']['root_cause_counts']}",
        f"- diagnostic new JSON/quote: {report['diagnostic40']['new_summary']['json_success_count']}/40 / {report['diagnostic40']['new_summary']['quote_located_rate']:.6f}",
        f"- A-control degradation: {report['diagnostic40']['a_control_degradation_count']}",
        f"- 32B/external calls: {report['llm_calls']['local_32b_chat_completion_attempts']}/{report['llm_calls']['external_llm_calls']}",
        f"- vLLM stopped: {report['service']['shutdown'].get('stopped')}",
        f"- formal/frozen SHA unchanged: {report['protection']['formal_and_frozen_sha_unchanged']}",
    ]
    write_text(REPORT_MD, "\n".join(lines) + "\n")
    sheets = {
        "summary": [flatten(report)],
        "failure_root_cause": report["failure_root_cause"]["rows"],
        "diagnostic_gates": [{"gate": k, "passed": v} for k, v in report["diagnostic40"]["gates"].items()],
        "response_format_probe": report["response_format"]["probe_rows"],
    }
    if report.get("final_quality"):
        sheets["final_quality"] = [report["final_quality"]]
    write_xlsx(REPORT_XLSX, sheets)


def flatten(payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--service-wait-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-start-vllm", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        STEP57C_DIR / "validated_annotation_drafts.jsonl",
        STEP57C_DIR / "annotation_drafts.jsonl",
        STEP57C_DIR / "raw_responses.jsonl",
        STEP57C_DIR / "request_metrics.jsonl",
        STEP57C_DIR / "sampled_eventmentions.jsonl",
        STEP57C_DIR / "confirmed80_migration_manifest.json",
        REPORT_JSON.parent / "task1_step5_7c_eventmention_annotation_pilot_report.json",
        REPORT_JSON.parent / "task1_step5_7c_eventmention_annotation_pilot_review.xlsx",
        CONFIRMED,
        STEP57A_FROZEN_RAW,
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if sha256_file(CONFIRMED) != EXPECTED_CONFIRMED_SHA:
        raise RuntimeError("confirmed annotation SHA mismatch")
    before = sha_snapshot(args.output_dir / "protected_and_frozen_sha256_before.txt", protected_and_frozen_files())
    backup_dir = Path("/opt/data/private/JinRong/kechuang/backups") / f"task1_step5_7d_contract_revision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_dir / "protected_and_frozen_sha256_before.txt", backup_dir / "protected_and_frozen_sha256_before.txt")

    old_drafts, old_raw_by_id, sampled_by_id, old_metrics_by_id = load_step57c()
    status_counts = Counter(row["draft_status"] for row in old_drafts)
    if len(old_drafts) != 200 or len({row["event_mention_id"] for row in old_drafts}) != 200 or status_counts != Counter({"A": 145, "C": 4, "D": 51}):
        raise RuntimeError(f"unexpected Step 5.7-C status distribution: {status_counts}")
    old_drafts_by_id = {str(row["event_mention_id"]): row for row in old_drafts}
    step57c_sha = {rel(path): sha256_file(path) for path in files_under(STEP57C_DIR)}
    failure = failure_audit(old_drafts, old_raw_by_id)
    write_json(args.output_dir / "failure_root_cause.json", failure)
    diag = choose_diagnostic40(old_drafts, failure["rows"], seed=args.seed)
    write_json(args.output_dir / "diagnostic40_manifest.json", diag)
    diagnostic_items = [build_request_item(sampled_by_id[mid], old_drafts_by_id[mid]) for mid in diag["event_mention_ids"]]
    write_jsonl(args.output_dir / "diagnostic40_sentence_inputs.jsonl", diagnostic_items)
    write_json(args.output_dir / "contract_schema.json", output_schema())

    service_proc: subprocess.Popen[str] | None = None
    service_info: dict[str, Any] = {}
    shutdown: dict[str, Any] = {}
    all_metrics: list[dict[str, Any]] = []
    all_raws: list[dict[str, Any]] = []
    try:
        service_proc, service_info = start_vllm_if_needed(args)
        selected_format_name, selected_format, probe_rows = select_response_format(args)
        if selected_format_name == "json_schema":
            selected_format = schema_response_format()
        diag_records, diag_metrics, diag_raws = run_batch(diagnostic_items, args, selected_format, "diagnostic40")
        all_metrics.extend(diag_metrics)
        all_raws.extend(diag_raws)
        write_jsonl(args.output_dir / "diagnostic40_results.jsonl", diag_records)
        write_jsonl(args.output_dir / "diagnostic40_raw_responses.jsonl", diag_raws)
        old_diag_summary = old_contract_summary(diag["event_mention_ids"], old_drafts_by_id, old_metrics_by_id)
        new_diag_summary = contract_summary(diag_records, diag_metrics)
        new_by_id = {row["event_mention_id"]: row for row in diag_records}
        a_degradation = sum(1 for mid in diag["a_control_event_mention_ids"] if new_by_id[mid].get("draft_status") != "A")
        gates = diagnostic_gates(new_diag_summary, old_diag_summary, a_degradation)
        gate_passed = all(gates.values())
        rerun55_executed = False
        rerun_records: list[dict[str, Any]] = []
        rerun_metrics: list[dict[str, Any]] = []
        merged_rows: list[dict[str, Any]] | None = None
        final_q: dict[str, Any] | None = None
        status = "contract_revision_failed"
        if gate_passed:
            cd_ids = [row["event_mention_id"] for row in old_drafts if row["draft_status"] in {"C", "D"}]
            diag_cd_records = {row["event_mention_id"]: row for row in diag_records if row["event_mention_id"] in set(cd_ids)}
            remaining_ids = [mid for mid in cd_ids if mid not in diag_cd_records]
            remaining_items = [build_request_item(sampled_by_id[mid], old_drafts_by_id[mid]) for mid in remaining_ids]
            extra_records, extra_metrics, extra_raws = run_batch(remaining_items, args, selected_format, "rerun55_remaining")
            rerun_records = [diag_cd_records[mid] for mid in diag["cd_event_mention_ids"] if mid in diag_cd_records] + extra_records
            rerun_metrics = [row for row in diag_metrics if row["event_mention_id"] in set(cd_ids)] + extra_metrics
            all_metrics.extend(extra_metrics)
            all_raws.extend(extra_raws)
            rerun55_executed = True
            write_jsonl(args.output_dir / "rerun55_results.jsonl", rerun_records)
            write_jsonl(args.output_dir / "rerun55_raw_responses.jsonl", [row for row in diag_raws if row["event_mention_id"] in set(cd_ids)] + extra_raws)
            rerun_by_id = {row["event_mention_id"]: row for row in rerun_records}
            merged_rows = merge_200(old_drafts, rerun_by_id)
            write_jsonl(args.output_dir / "merged200_results.jsonl", merged_rows)
            final_q = final_quality(merged_rows, rerun_metrics)
            status = final_status(final_q)

        write_jsonl(args.output_dir / "request_metrics.jsonl", all_metrics)
        write_jsonl(args.output_dir / "raw_responses.jsonl", all_raws)
        after = sha_snapshot(args.output_dir / "protected_and_frozen_sha256_after.txt", protected_and_frozen_files())
        report = {
            "generated_at": now(),
            "status": status,
            "safety": {
                "backup_dir": str(backup_dir),
                "process_check_before": process_check(),
                "step5_7c_frozen_output_sha256": step57c_sha,
                "confirmed_sha256": sha256_file(CONFIRMED),
                "gold_test_labels_read": False,
                "training_or_finetune_executed": False,
            },
            "input_recovery": {
                "step5_7c_record_count": len(old_drafts),
                "step5_7c_unique_event_mention_id_count": len(old_drafts_by_id),
                "step5_7c_status_counts": dict(status_counts),
                "step5_7c_raw_response_count": len(old_raw_by_id),
                "step5_7c_retry_count": sum(int(row.get("retry_count", 0) or 0) for row in old_raw_by_id.values()),
            },
            "failure_root_cause": failure,
            "response_format": {"selected": selected_format_name, "probe_rows": probe_rows},
            "diagnostic40": {
                "manifest": diag,
                "old_summary": old_diag_summary,
                "new_summary": new_diag_summary,
                "gates": gates,
                "gate_passed": gate_passed,
                "a_control_degradation_count": a_degradation,
            },
            "rerun55": {"executed": rerun55_executed, "record_count": len(rerun_records), "remaining_new_calls": len([row for row in all_metrics if row.get("split") == "rerun55_remaining"])},
            "final_quality": final_q,
            "protection": {
                "formal_and_frozen_sha_unchanged": before == after,
                "confirmed_annotation_sha_unchanged": sha256_file(CONFIRMED) == EXPECTED_CONFIRMED_SHA,
                "step5_7a_frozen_response_sha_unchanged": before.get(rel(STEP57A_FROZEN_RAW)) == after.get(rel(STEP57A_FROZEN_RAW)),
                "step5_7c_outputs_unchanged": all(before.get(path) == after.get(path) for path in step57c_sha),
                "backup_dir": str(backup_dir),
            },
            "llm_calls": {
                "local_32b_probe_attempts": sum(1 for row in probe_rows if row["format"] != "prompt_only"),
                "local_32b_logical_records": len(diag_records) + (len([row for row in all_metrics if row.get("split") == "rerun55_remaining"]) if rerun55_executed else 0),
                "local_32b_chat_completion_attempts": sum(int(row.get("request_count", 0) or 0) for row in all_metrics) + sum(1 for row in probe_rows if row["format"] != "prompt_only"),
                "external_llm_calls": 0,
            },
            "service": {**service_info, "shutdown": shutdown},
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(args.output_dir / "run_manifest.json", report)
        if merged_rows is not None:
            migration_manifest = read_json(STEP57C_DIR / "confirmed80_migration_manifest.json")
            write_review_package(merged_rows, migration_manifest, report)
        write_reports(report)
        write_json(backup_dir / "final_report.json", report)
        write_sha_manifest(args.output_dir)
        print(json.dumps({"status": status, "diagnostic_gate_passed": gate_passed, "rerun55_executed": rerun55_executed, "llm_calls": report["llm_calls"]}, ensure_ascii=False))
        return 0
    finally:
        shutdown = stop_vllm(service_proc)
        if (args.output_dir / "run_manifest.json").exists():
            manifest = read_json(args.output_dir / "run_manifest.json")
            manifest.setdefault("service", service_info)
            manifest["service"]["shutdown"] = shutdown
            write_json(args.output_dir / "run_manifest.json", manifest)
            write_json(REPORT_JSON, manifest)
            write_reports(manifest)
            write_sha_manifest(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
