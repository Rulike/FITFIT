"""Offline iterative self-reflection (ISR) utilities for diagnostic experiments.

This module is deliberately independent from the production DEA and Safety Gate
paths. It accepts only a claim and evidence payload, and returns sanitized JSON
objects without storing hidden model reasoning.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable


VALID_VERDICTS = {"support", "contradict", "insufficient", "unclear"}
MAX_REASON_CHARS = 600
MAX_REVISION_CHARS = 600
MAX_RESPONSE_EXCERPT_CHARS = 4000


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strip_hidden_reasoning(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<reasoning\b[^>]*>.*?</reasoning>", "", text, flags=re.S | re.I)
    return text.strip()


def response_excerpt(value: Any) -> str:
    return strip_hidden_reasoning(value)[:MAX_RESPONSE_EXCERPT_CHARS]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _find_json_object(text: str) -> Any | None:
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_json_object(value: Any) -> dict[str, Any] | None:
    text = strip_hidden_reasoning(value).strip()
    if not text:
        return None
    candidates = [text]
    tag = re.search(r"<answer\b[^>]*>(.*?)</answer>", text, flags=re.S | re.I)
    if tag:
        candidates.append(tag.group(1).strip())
    block = re.search(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.S)
    if block:
        candidates.append(block.group(1).strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate, parse_constant=_reject_json_constant)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = _find_json_object(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def _valid_ids(value: Any, allowed_ids: set[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [], ["evidence_ids must be a string list"]
    ids = [item.strip() for item in value]
    if any(not item for item in ids):
        errors.append("evidence_ids contains an empty value")
    if len(set(ids)) != len(ids):
        errors.append("evidence_ids contains duplicates")
    unknown = sorted(set(ids) - allowed_ids)
    if unknown:
        errors.append(f"unknown evidence_ids: {unknown}")
    return ids, errors


def validate_final_object(value: Any, allowed_ids: Iterable[str]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["response is not a JSON object"]
    verdict = norm(value.get("verdict"))
    if verdict not in VALID_VERDICTS:
        errors.append(f"invalid verdict: {verdict!r}")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append("confidence must be numeric")
    else:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            errors.append("confidence must be finite and within [0, 1]")
    reason = value.get("reason")
    if not isinstance(reason, str) or not norm(reason):
        errors.append("reason must be a non-empty string")
    elif len(norm(reason)) > MAX_REASON_CHARS:
        errors.append(f"reason exceeds {MAX_REASON_CHARS} characters")
    ids, id_errors = _valid_ids(value.get("evidence_ids"), set(norm(x) for x in allowed_ids))
    errors.extend(id_errors)
    if verdict in {"support", "contradict"} and not ids:
        errors.append(f"{verdict} requires at least one evidence_id")
    if errors:
        return None, errors
    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "evidence_ids": ids,
        "reason": norm(reason),
    }, []


def validate_critique_object(value: Any, allowed_ids: Iterable[str]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["response is not a JSON object"]
    problems = value.get("problems")
    if not isinstance(problems, list) or not all(isinstance(item, str) for item in problems):
        errors.append("problems must be a string list")
        problems = []
    reassessment = value.get("evidence_reassessment")
    if not isinstance(reassessment, list):
        errors.append("evidence_reassessment must be a list")
        reassessment = []
    verdict = norm(value.get("recommended_verdict"))
    if verdict not in VALID_VERDICTS:
        errors.append(f"invalid recommended_verdict: {verdict!r}")
    ids, id_errors = _valid_ids(value.get("recommended_evidence_ids"), set(norm(x) for x in allowed_ids))
    errors.extend(id_errors)
    if not isinstance(value.get("needs_revision"), bool):
        errors.append("needs_revision must be boolean")
    if errors:
        return None, errors
    return {
        "problems": [norm(item) for item in problems if norm(item)],
        "evidence_reassessment": reassessment[:30],
        "recommended_verdict": verdict,
        "recommended_evidence_ids": ids,
        "needs_revision": bool(value["needs_revision"]),
    }, []


def validate_final_revision_object(value: Any, allowed_ids: Iterable[str]) -> tuple[dict[str, Any] | None, list[str]]:
    base, errors = validate_final_object(value, allowed_ids)
    if base is None:
        return None, errors
    revision_summary = value.get("revision_summary") if isinstance(value, dict) else None
    if not isinstance(revision_summary, str) or not norm(revision_summary):
        return None, ["revision_summary must be a non-empty string"]
    if len(norm(revision_summary)) > MAX_REVISION_CHARS:
        return None, [f"revision_summary exceeds {MAX_REVISION_CHARS} characters"]
    isr_rounds = value.get("isr_rounds") if isinstance(value, dict) else None
    if isinstance(isr_rounds, bool) or not isinstance(isr_rounds, int) or isr_rounds != 3:
        return None, ["isr_rounds must equal 3"]
    return {
        **base,
        "revision_summary": norm(revision_summary),
        "isr_rounds": 3,
    }, []


@dataclass(frozen=True)
class ISRConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "DianJin-R1-32B"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 180.0
    format_retries: int = 1


def endpoint_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _usage_fields(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    def integer(name: str) -> int:
        try:
            return max(0, int(usage.get(name) or 0))
        except (TypeError, ValueError):
            return 0
    return {
        "prompt_tokens": integer("prompt_tokens"),
        "completion_tokens": integer("completion_tokens"),
        "total_tokens": integer("total_tokens"),
    }


def chat_completion(
    config: ISRConfig,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    request = urllib.request.Request(
        endpoint_url(config.base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            status_code = int(getattr(response, "status", 200) or 200)
        decoded = json.loads(body)
        choices = decoded.get("choices") if isinstance(decoded, dict) else []
        message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return {
            "ok": True,
            "status_code": status_code,
            "content": str(content or ""),
            "usage": _usage_fields(decoded),
            "latency_seconds": round(time.perf_counter() - started, 6),
            "response_id": norm(decoded.get("id")) if isinstance(decoded, dict) else "",
            "error_type": "",
            "error_message": "",
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        return {
            "ok": False,
            "status_code": int(exc.code),
            "content": "",
            "usage": {},
            "latency_seconds": round(time.perf_counter() - started, 6),
            "response_id": "",
            "error_type": "http_5xx" if int(exc.code) >= 500 else "http_error",
            "error_message": f"HTTP {exc.code}: {body}",
        }
    except TimeoutError as exc:
        return {
            "ok": False,
            "status_code": 0,
            "content": "",
            "usage": {},
            "latency_seconds": round(time.perf_counter() - started, 6),
            "response_id": "",
            "error_type": "timeout",
            "error_message": str(exc) or "request timeout",
        }
    except (urllib.error.URLError, OSError) as exc:
        return {
            "ok": False,
            "status_code": 0,
            "content": "",
            "usage": {},
            "latency_seconds": round(time.perf_counter() - started, 6),
            "response_id": "",
            "error_type": "connection_error",
            "error_message": str(exc),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return {
            "ok": False,
            "status_code": 0,
            "content": "",
            "usage": {},
            "latency_seconds": round(time.perf_counter() - started, 6),
            "response_id": "",
            "error_type": "response_decode_error",
            "error_message": str(exc),
        }


def invoke_json_round(
    *,
    config: ISRConfig,
    cluster_id: str,
    phase: str,
    messages: list[dict[str, str]],
    allowed_ids: Iterable[str],
    validator: Callable[[Any, Iterable[str]], tuple[dict[str, Any] | None, list[str]]],
    metrics_sink: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    current_messages = messages
    allowed = list(allowed_ids)
    attempts: list[dict[str, Any]] = []
    max_attempts = 1 + max(0, int(config.format_retries))
    last_errors: list[str] = []
    for attempt_index in range(max_attempts):
        result = chat_completion(config, current_messages)
        content = result.get("content", "")
        parsed = parse_json_object(content) if result.get("ok") else None
        normalized, errors = validator(parsed, allowed) if result.get("ok") else (None, [result.get("error_message", "request failed")])
        metric = {
            "cluster_id": cluster_id,
            "phase": phase,
            "attempt": attempt_index + 1,
            "format_retry": attempt_index > 0,
            "status_code": result.get("status_code", 0),
            "latency_seconds": result.get("latency_seconds", 0.0),
            "prompt_sha256": sha256_text(json.dumps(current_messages, ensure_ascii=False)),
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0),
            "response_id": result.get("response_id", ""),
            "error_type": result.get("error_type", ""),
            "error_message": result.get("error_message", ""),
            "parse_errors": list(errors),
            "response_excerpt": response_excerpt(content),
        }
        metrics_sink(metric)
        attempts.append(metric)
        if normalized is not None:
            return {
                "status": "ok",
                "attempt_count": len(attempts),
                "format_retry_count": max(0, len(attempts) - 1),
                "parsed": normalized,
                "error_type": "",
                "error_message": "",
                "attempts": attempts,
            }
        last_errors = list(errors)
        if attempt_index + 1 < max_attempts:
            correction = (
                "上一次响应未通过严格JSON校验。请不要解释，不要输出思维过程，"
                "只按原要求输出一个JSON对象。格式校验错误："
                + json.dumps(last_errors, ensure_ascii=False)
            )
            current_messages = list(messages) + [{"role": "user", "content": correction}]
    return {
        "status": "failed",
        "attempt_count": len(attempts),
        "format_retry_count": max(0, len(attempts) - 1),
        "parsed": None,
        "error_type": "parse_failure" if any(last_errors) else "request_failed",
        "error_message": "; ".join(last_errors),
        "attempts": attempts,
    }
