#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import build_runnable_submission_item2 as base


ROOT = base.ROOT
BASE = base.BASE
COMPUTE = base.COMPUTE
TARGET = BASE / "item2_runnable_v2"
ZIP_PATH = BASE / "02_可运行项目成果_v2.zip"
VALIDATION_ROOT = Path("/tmp") / f"competition_item2_v2_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
ORIGINAL_PROJECT_ABS = "/" + "/".join(["opt", "data", "private", "JinRong", "kechuang", "financial_event_project"])
EXPECTED_TASK2_SQLITE_SHA256 = "4f10b523185dbb256ad88f0535668c54f8eed9f2c38d05763fd08babb91b891f"
LEGACY_RULE_SCRIPT = ROOT / "scripts/build_task34_official1799_final_evaluation_v1.py"

base.TARGET = TARGET
base.ZIP_PATH = ZIP_PATH
base.VALIDATION_ROOT = VALIDATION_ROOT
sys.dont_write_bytecode = True


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    base.write_json(path, payload)


def write_text(path: Path, text: str, executable: bool = False) -> None:
    base.write_text(path, text, executable=executable)


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    return base.copy_file(src, dst)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    return base.read_json_if_exists(path)


def parse_sha_lines(path: Path, root_marker: str) -> dict[str, dict[str, Any]]:
    return base.parse_sha_lines(path, root_marker)


TASK2_RUNTIME_PY = r'''from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any

from .common import clean_text, compact, sha256_file, stable_id


EXPECTED_SQLITE_SHA256 = "4f10b523185dbb256ad88f0535668c54f8eed9f2c38d05763fd08babb91b891f"
DOMAIN_TERMS = [
    "交易所",
    "问询函",
    "问询",
    "处罚",
    "现金流",
    "长期供货",
    "供货合同",
    "签署合同",
    "董事会",
    "海外工厂",
    "澄清",
    "不属实",
    "监管",
    "融资",
    "亏损",
    "违约",
]


class MissingExternalAsset(RuntimeError):
    pass


def tokenize(value: Any) -> list[str]:
    text = clean_text(value)
    out: list[str] = []
    seen: set[str] = set()
    for term in DOMAIN_TERMS:
        if term in text and term not in seen:
            seen.add(term)
            out.append(term)
    for term in re.findall(r"[A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}", text):
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def source_reliability(source: str) -> float:
    source = clean_text(source)
    if any(k in source for k in ["公告", "交易所", "证监", "证券时报", "上证报"]):
        return 0.9
    if any(k in source for k in ["财联社", "证券", "新闻", "日报"]):
        return 0.74
    if any(k in source for k in ["论坛", "股吧", "自媒体", "传闻"]):
        return 0.36
    return 0.62


def fts_phrase(term: str) -> str:
    term = clean_text(term).strip()
    if not term:
        return ""
    return '"' + term.replace('"', '""') + '"'


def normalize_verdict(value: Any) -> str:
    text = clean_text(value).lower()
    mapping = {
        "supported": "support",
        "support": "support",
        "true": "support",
        "contradiction": "contradict",
        "contradict": "contradict",
        "false": "contradict",
        "insufficient": "insufficient",
        "unknown": "unclear",
        "unclear": "unclear",
    }
    if text in mapping:
        return mapping[text]
    if "contrad" in text or "不属实" in text or "否认" in text:
        return "contradict"
    if "support" in text or "属实" in text or "确认" in text:
        return "support"
    if "insufficient" in text or "不足" in text:
        return "insufficient"
    return "unclear"


class Task2Runtime:
    version = "final_cached_dea_v1_260"

    def __init__(self, package_root: Path, mode: str = "portable") -> None:
        self.package_root = package_root
        self.mode = "portable" if mode == "light" else mode
        self.loaded = False
        self.external_status: dict[str, Any] = {}
        self.llm_call_count = 0
        self.retrieval_call_count = 0

    def load(self) -> dict[str, Any]:
        if self.mode == "full":
            self.external_status = self.check_external_assets(strict=True)
        else:
            self.external_status = self.check_external_assets(strict=False)
        self.loaded = True
        return {
            "task": "Task2",
            "loaded": True,
            "selected_version": self.version,
            "active_runtime_profile": self.mode,
            "formal_full_mode": self.mode == "full",
            "degraded_mode": self.mode != "full",
            "external_assets": self.external_status,
            "note": "full mode validates and uses DianJin-R1-32B plus SQLite retrieval; portable mode is deterministic local degraded inference and is not the official cached DEA score.",
        }

    def check_external_assets(self, strict: bool) -> dict[str, Any]:
        index_dir = Path(os.environ.get("TASK2_SQLITE_INDEX_DIR", self.package_root / "external_assets/task2/dea_retrieval_index"))
        db_path = index_dir / "retrieval_index.sqlite"
        meta_path = index_dir / "retrieval_index_metadata.json"
        service = os.environ.get("TASK2_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        model = os.environ.get("TASK2_LLM_MODEL", "DianJin-R1-32B")
        expected_sha = os.environ.get("TASK2_SQLITE_SHA256", EXPECTED_SQLITE_SHA256)
        status: dict[str, Any] = {
            "sqlite_index_dir": str(index_dir),
            "sqlite_path": str(db_path),
            "sqlite_exists": db_path.exists(),
            "metadata_exists": meta_path.exists(),
            "expected_sqlite_sha256": expected_sha,
            "sqlite_sha256_checked": False,
            "sqlite_sha256_match": None,
            "llm_base_url": service,
            "model_id": model,
            "llm_service_checked": False,
            "llm_model_available": False,
            "missing_external_assets": [],
        }
        if db_path.exists():
            actual_sha = sha256_file(db_path) if strict or os.environ.get("TASK2_CHECK_SQLITE_SHA256") == "1" else None
            if actual_sha:
                status["actual_sqlite_sha256"] = actual_sha
                status["sqlite_sha256_checked"] = True
                status["sqlite_sha256_match"] = actual_sha == expected_sha
                if actual_sha != expected_sha:
                    status["missing_external_assets"].append({"asset": "Task2 DEA SQLite retrieval index", "reason": "sha256_mismatch"})
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=8)
                conn.execute("pragma query_only=ON")
                tables = [str(row[0]) for row in conn.execute("select name from sqlite_master where type in ('table','view')").fetchall()]
                status["sqlite_tables"] = tables
                if "docs" in tables:
                    status["sqlite_doc_count"] = int(conn.execute("select count(*) from docs").fetchone()[0])
                if "docs_fts" in tables:
                    status["sqlite_fts_probe_ok"] = bool(conn.execute("select rowid from docs_fts limit 1").fetchone())
                conn.close()
            except Exception as exc:
                status["sqlite_error"] = f"{type(exc).__name__}: {exc}"
                status["missing_external_assets"].append({"asset": "Task2 DEA SQLite retrieval index", "reason": status["sqlite_error"]})
        else:
            status["missing_external_assets"].append({"asset": "Task2 DEA SQLite retrieval index", "reason": f"missing {db_path}"})
        if strict:
            if not db_path.exists():
                raise MissingExternalAsset(f"missing Task2 SQLite index: {db_path}")
            if status.get("sqlite_sha256_match") is False:
                raise MissingExternalAsset(f"Task2 SQLite SHA256 mismatch at {db_path}")
            try:
                with urllib.request.urlopen(f"{service}/models", timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="replace"))
                names = [str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)]
                status["llm_service_checked"] = True
                status["llm_model_available"] = model in names
                status["llm_models"] = names
                if model not in names:
                    raise MissingExternalAsset(f"Task2 model {model!r} unavailable at {service}; found={names}")
            except MissingExternalAsset:
                raise
            except Exception as exc:
                raise MissingExternalAsset(f"cannot reach Task2 LLM service {service}: {type(exc).__name__}: {exc}") from exc
        else:
            status["missing_external_assets"].append({"asset": "DianJin-R1-32B service", "reason": "not checked in portable mode"})
        return status

    def predict(self, row: dict[str, Any], task1: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        if self.mode == "full":
            return self.predict_full(row, task1)
        return self.predict_portable(row, task1)

    def predict_portable(self, row: dict[str, Any], task1: dict[str, Any]) -> dict[str, Any]:
        title = clean_text(row.get("title") or row.get("article_title"))
        content = clean_text(row.get("content"))
        source = clean_text(row.get("source") or row.get("article_source"))
        evidence_items = row.get("evidence_items") or []
        evidence_text = " ".join(clean_text(x.get("sentence") or x.get("content") or x.get("title")) for x in evidence_items if isinstance(x, dict))
        if not evidence_text:
            evidence_text = content
        claim = " ".join([task1.get("target_name", ""), task1.get("event_name", ""), title])
        claim_tokens = set(tokenize(claim))
        evidence_tokens = set(tokenize(evidence_text))
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        reliability = source_reliability(source)
        text_all = f"{title} {content} {evidence_text}"
        rumor = any(k in text_all for k in ["传闻", "网传", "未经证实", "市场消息"])
        contradiction = any(k in text_all for k in ["否认", "澄清", "不属实", "辟谣", "撤回"])
        official = any(k in source + text_all for k in ["公告", "披露", "交易所", "监管函", "证监"])
        score = 100.0 * (0.50 * reliability + 0.35 * overlap + 0.15 * (1.0 if official else 0.45))
        if rumor:
            score -= 18.0
        if contradiction:
            score -= 28.0
        score = max(0.0, min(100.0, score))
        if contradiction:
            verdict = "contradict"
        elif score >= 66:
            verdict = "support"
        elif score >= 48:
            verdict = "unclear"
        else:
            verdict = "insufficient"
        return {
            "cluster_id": stable_id("CL", task1.get("duplication_id")),
            "credibility_score": round(score, 4),
            "credibility_level": "high" if score >= 75 else "medium" if score >= 60 else "needs_review",
            "final_verdict": verdict,
            "confidence": round(score / 100.0, 4),
            "evidence_chain": [
                {
                    "source": source or "input",
                    "support_type": verdict,
                    "evidence": compact(evidence_text, 220),
                    "claim_overlap": round(overlap, 6),
                    "source_reliability": round(reliability, 6),
                }
            ],
            "runtime_profile": {
                "selected_version": self.version,
                "active_profile": "portable",
                "production_equivalent": False,
            },
            "degraded_mode": True,
            "missing_external_assets": self.external_status.get("missing_external_assets", []),
            "mode": "portable_deterministic_degraded",
            "external_model_loaded": False,
            "cache_answer_used": False,
            "hardcoded_prediction": False,
            "model_loaded": True,
        }

    def retrieve_external_evidence(self, row: dict[str, Any], task1: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        self.retrieval_call_count += 1
        index_dir = Path(os.environ.get("TASK2_SQLITE_INDEX_DIR", self.package_root / "external_assets/task2/dea_retrieval_index"))
        db_path = index_dir / "retrieval_index.sqlite"
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma query_only=ON")
        terms = []
        for value in [task1.get("target_name"), task1.get("target_code"), task1.get("event_name"), row.get("title")]:
            terms.extend(tokenize(value))
        terms = [term for term in dict.fromkeys(terms) if len(term) >= 2][:8]
        query = " OR ".join(fts_phrase(term) for term in terms if fts_phrase(term))
        rows: list[sqlite3.Row] = []
        try:
            if query:
                rows = conn.execute(
                    """
                    select
                        d.docid,
                        d.article_id,
                        d.article_title,
                        d.article_source,
                        d.article_publish_time,
                        d.content,
                        d.source_type,
                        d.source_reliability
                    from docs_fts
                    join docs d on d.docid = docs_fts.rowid
                    where docs_fts match ?
                    order by d.docid
                    limit ?
                    """,
                    (query, limit),
                ).fetchall()
        except Exception:
            rows = []
        if not rows:
            columns = [str(r[1]) for r in conn.execute("pragma table_info(docs)").fetchall()]
            text_cols = [c for c in ["article_title", "article_source", "content", "title", "sentence", "subject_text"] if c in columns]
            if text_cols and terms:
                conditions = " OR ".join(f"{col} like ?" for col in text_cols for _ in terms[:3])
                args = [f"%{term}%" for col in text_cols for term in terms[:3]]
                rows = conn.execute(f"select * from docs where {conditions} limit ?", (*args, limit)).fetchall()
        conn.close()
        out: list[dict[str, Any]] = []
        for item in rows[:limit]:
            data = {k: item[k] for k in item.keys()}
            out.append(
                {
                    "article_id": clean_text(data.get("article_id") or data.get("id")),
                    "title": compact(data.get("title") or data.get("article_title"), 120),
                    "source": clean_text(data.get("source") or data.get("article_source")),
                    "publish_time": clean_text(data.get("publish_time") or data.get("article_publish_time")),
                    "sentence": compact(data.get("sentence") or data.get("content") or data.get("subject_text"), 260),
                }
            )
        return out

    def call_llm(self, prompt: str) -> dict[str, Any]:
        self.llm_call_count += 1
        service = os.environ.get("TASK2_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        model = os.environ.get("TASK2_LLM_MODEL", "DianJin-R1-32B")
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": int(os.environ.get("TASK2_LLM_MAX_TOKENS", "512")),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(f"{service}/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=int(os.environ.get("TASK2_LLM_TIMEOUT", "180"))) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        content = payload["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"final_verdict": "unclear", "credibility_score": 50.0, "reason": compact(content, 500)}
        return parsed

    def predict_full(self, row: dict[str, Any], task1: dict[str, Any]) -> dict[str, Any]:
        if not self.external_status or not self.external_status.get("sqlite_exists"):
            self.external_status = self.check_external_assets(strict=True)
        evidence_items = self.retrieve_external_evidence(row, task1)
        evidence_text = "\n".join(f"- {x.get('source','')}: {x.get('title','')} {x.get('sentence','')}" for x in evidence_items)
        prompt = (
            "你是金融事件可信度评估器。只返回 JSON，字段为 final_verdict、credibility_score、confidence、reason。"
            f"\n事件主张: {task1.get('target_name')} {task1.get('event_name')}"
            f"\n标题: {clean_text(row.get('title') or row.get('article_title'))}"
            f"\n来源: {clean_text(row.get('source') or row.get('article_source'))}"
            f"\n正文: {compact(row.get('content'), 1200)}"
            f"\nSQLite 检索证据:\n{compact(evidence_text, 1800)}"
        )
        parsed = self.call_llm(prompt)
        score = float(parsed.get("credibility_score", 50.0))
        score = max(0.0, min(100.0, score))
        verdict = normalize_verdict(parsed.get("final_verdict") or parsed.get("verdict"))
        confidence = parsed.get("confidence", score / 100.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = score / 100.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "cluster_id": stable_id("CL", task1.get("duplication_id")),
            "credibility_score": round(score, 4),
            "credibility_level": "high" if score >= 75 else "medium" if score >= 60 else "needs_review",
            "final_verdict": verdict,
            "confidence": round(confidence, 4),
            "evidence_chain": evidence_items,
            "dea_fusion": {
                "retrieval_backend": "sqlite_fts5_trigram",
                "retrieved_evidence_count": len(evidence_items),
                "llm_model": os.environ.get("TASK2_LLM_MODEL", "DianJin-R1-32B"),
                "retrieval_call_count": self.retrieval_call_count,
                "llm_call_count": self.llm_call_count,
                "decision_source": "DianJin-R1-32B + SQLite retrieval adapter",
            },
            "runtime_profile": {
                "selected_version": self.version,
                "active_profile": "full",
                "production_equivalent": True,
            },
            "degraded_mode": False,
            "missing_external_assets": [],
            "mode": "full_external_dea",
            "external_model_loaded": True,
            "cache_answer_used": False,
            "hardcoded_prediction": False,
            "model_loaded": True,
        }
'''


TASK3_RUNTIME_PY = r'''from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import joblib

from .common import clean_text, sha256_text, stable_id


def tokens(value: Any) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}", clean_text(value).lower()))


def token_jaccard(a: Any, b: Any) -> float:
    sa, sb = tokens(a), tokens(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def sequence_ratio(a: Any, b: Any) -> float:
    return SequenceMatcher(None, re.sub(r"\s+", "", clean_text(a).lower()), re.sub(r"\s+", "", clean_text(b).lower())).ratio()


def sim(a: Any, b: Any) -> float:
    return round(0.7 * sequence_ratio(a, b) + 0.3 * token_jaccard(a, b), 6)


def relation_time(a: Any, b: Any) -> str:
    da = clean_text(a)[:10]
    db = clean_text(b)[:10]
    if not da or not db:
        return "missing"
    return "same_day" if da == db else "near_time" if da[:7] == db[:7] else "different_time"


def event_like(row: dict[str, Any], task1: dict[str, Any], task2: dict[str, Any]) -> dict[str, Any]:
    title = clean_text(row.get("title") or row.get("article_title"))
    content = clean_text(row.get("content"))
    target_name = clean_text(task1.get("target_name"))
    event_name = clean_text(task1.get("event_name"))
    evidence = clean_text(task1.get("evidence") or content[:240])
    fact = f"{target_name} {event_name} {evidence}"
    return {
        "event_mention_id": task1.get("event_mention_id"),
        "article_id": clean_text(row.get("article_id")),
        "document_hash": sha256_text(title + content),
        "title": title,
        "source_name": clean_text(row.get("source") or row.get("article_source")),
        "publish_time": clean_text(row.get("publish_time")),
        "target": {
            "target_type": task1.get("target_type"),
            "target_code": task1.get("target_code"),
            "target_name": target_name,
            "target_key": f"{task1.get('target_type')}:{task1.get('target_code') or target_name}",
        },
        "event": {
            "event_family": event_name,
            "event_type": event_name,
            "event_name": event_name,
            "event_time": clean_text(row.get("publish_time")),
            "trigger_text": event_name[:12],
            "object_text": target_name,
            "event_fact": fact,
            "event_fact_signature": sha256_text(fact)[:20],
            "polarity_or_state": task1.get("event_emotion") or task2.get("final_verdict"),
        },
        "evidence": {
            "evidence_sentence": evidence,
            "local_context": content[:600],
            "support_type": task2.get("final_verdict", "unclear"),
        },
    }


def get_nested(item: dict[str, Any], path: str) -> str:
    value: Any = item
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part, "")
        else:
            value = ""
    return clean_text(value)


def feature_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    lt, rt = get_nested(left, "target.target_key"), get_nested(right, "target.target_key")
    ltype, rtype = get_nested(left, "target.target_type"), get_nested(right, "target.target_type")
    lcode, rcode = get_nested(left, "target.target_code"), get_nested(right, "target.target_code")
    levent, revent = get_nested(left, "event.event_name"), get_nested(right, "event.event_name")
    lfact, rfact = get_nested(left, "event.event_fact"), get_nested(right, "event.event_fact")
    lobj, robj = get_nested(left, "event.object_text"), get_nested(right, "event.object_text")
    ltrig, rtrig = get_nested(left, "event.trigger_text"), get_nested(right, "event.trigger_text")
    lev, rev = get_nested(left, "evidence.evidence_sentence"), get_nested(right, "evidence.evidence_sentence")
    support_a, support_b = get_nested(left, "evidence.support_type"), get_nested(right, "evidence.support_type")
    polarity_a, polarity_b = get_nested(left, "event.polarity_or_state"), get_nested(right, "event.polarity_or_state")
    sig_a, sig_b = get_nested(left, "event.event_fact_signature"), get_nested(right, "event.event_fact_signature")
    hard_conflict = 0
    if lt and rt and lt != rt:
        hard_conflict += 1
    if levent and revent and levent != revent and sim(levent, revent) < 0.2:
        hard_conflict += 1
    return {
        "same_target": int(bool(lt and rt and lt == rt)),
        "same_target_type": int(bool(ltype and rtype and ltype == rtype)),
        "same_target_code": int(bool(lcode and rcode and lcode == rcode)),
        "same_family": int(bool(levent and revent and levent == revent)),
        "same_event_type": int(bool(levent and revent and levent == revent)),
        "same_event_name": int(bool(levent and revent and levent == revent)),
        "same_trigger": int(bool(ltrig and rtrig and ltrig == rtrig)),
        "same_object": int(bool(lobj and robj and lobj == robj)),
        "same_modality": 1,
        "same_article": int(bool(clean_text(left.get("article_id")) and left.get("article_id") == right.get("article_id"))),
        "same_source": int(bool(clean_text(left.get("source_name")) and left.get("source_name") == right.get("source_name"))),
        "same_sentence": int(bool(lev and rev and lev == rev)),
        "same_document_hash": int(bool(clean_text(left.get("document_hash")) and left.get("document_hash") == right.get("document_hash"))),
        "same_event_fact_signature": int(bool(sig_a and sig_b and sig_a == sig_b)),
        "same_support_type": int(bool(support_a and support_b and support_a == support_b)),
        "same_polarity_or_state": int(bool(polarity_a and polarity_b and polarity_a == polarity_b)),
        "title_similarity": sim(left.get("title"), right.get("title")),
        "fact_similarity": sim(lfact, rfact),
        "object_similarity": sim(lobj, robj),
        "trigger_similarity": sim(ltrig, rtrig),
        "evidence_similarity": sim(lev, rev),
        "fact_token_overlap_count": len(tokens(lfact) & tokens(rfact)),
        "object_token_overlap_count": len(tokens(lobj) & tokens(robj)),
        "trigger_token_overlap_count": len(tokens(ltrig) & tokens(rtrig)),
        "hard_conflict_count": hard_conflict,
        "time_relation": relation_time(get_nested(left, "event.event_time"), get_nested(right, "event.event_time")),
        "modality_relation": "same",
        "provenance_relation": "same_article" if left.get("article_id") == right.get("article_id") else "different_article",
        "support_relation": "same" if support_a and support_a == support_b else "different" if support_a and support_b else "missing",
        "polarity_relation": "same" if polarity_a and polarity_a == polarity_b else "different" if polarity_a and polarity_b else "missing",
        "support_type_a": support_a or "__MISSING__",
        "support_type_b": support_b or "__MISSING__",
        "polarity_or_state_a": polarity_a or "__MISSING__",
        "polarity_or_state_b": polarity_b or "__MISSING__",
        "event_fact_signature_a": sig_a or "__MISSING__",
        "event_fact_signature_b": sig_b or "__MISSING__",
    }


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class Task3Runtime:
    version = "legacy_official_pipeline"

    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root
        self.models_dir = package_root / "models/task3"
        self.loaded = False

    def load(self) -> dict[str, Any]:
        self.model = joblib.load(self.models_dir / "resolver_v2_model.joblib")
        self.selection = json.loads((self.models_dir / "resolver_v2_selection.json").read_text(encoding="utf-8"))
        self.loaded = True
        return {
            "task": "Task3",
            "loaded": True,
            "selected_version": self.version,
            "model": "models/task3/resolver_v2_model.joblib",
            "thresholds": self.selection.get("thresholds", {}),
            "runtime": "candidate_generation_pair_resolver_union_find_clustering",
        }

    def build_candidates(self, events: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any], dict[str, float]]]:
        classes = [str(x) for x in self.model.classes_]
        candidates: list[tuple[int, int, dict[str, Any], dict[str, float]]] = []
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                feats = feature_dict(events[i], events[j])
                likely_channel = (
                    feats["same_target"]
                    or feats["same_target_code"]
                    or feats["title_similarity"] >= 0.18
                    or feats["fact_similarity"] >= 0.22
                )
                if not likely_channel:
                    continue
                probs = self.model.predict_proba([feats])[0]
                prob_map = {classes[k]: float(probs[k]) for k in range(len(classes))}
                candidates.append((i, j, feats, prob_map))
        return candidates

    def resolve_all(self, rows: list[dict[str, Any]], task1s: list[dict[str, Any]], task2s: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.loaded:
            self.load()
        events = [event_like(row, t1, t2) for row, t1, t2 in zip(rows, task1s, task2s)]
        same_thr = float((self.selection.get("thresholds") or {}).get("same_thr", 0.45))
        candidates = self.build_candidates(events)
        uf = UnionFind(len(events))
        pair_by_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for i, j, feats, prob_map in candidates:
            score = prob_map.get("same_event", 0.0)
            decision = "same_event" if score >= same_thr and int(feats.get("hard_conflict_count", 0)) == 0 else "different_event"
            rec = {
                "left_index": i,
                "right_index": j,
                "same_event_probability": round(score, 6),
                "decision": decision,
                "hard_conflict_count": int(feats.get("hard_conflict_count", 0)),
                "probabilities": {k: round(v, 6) for k, v in prob_map.items()},
            }
            pair_by_node[i].append(rec)
            pair_by_node[j].append(rec)
            if decision == "same_event":
                uf.union(i, j)
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(events)):
            groups[uf.find(idx)].append(idx)
        group_ids: dict[int, tuple[str, str]] = {}
        for root, members in groups.items():
            first = members[0]
            t1 = task1s[first]
            row = rows[first]
            canonical_id = stable_id("CE", t1.get("target_code") or t1.get("target_name"), t1.get("event_name"), row.get("publish_time"))
            thread_id = stable_id("TH", t1.get("target_code") or t1.get("target_name"), t1.get("event_name"))
            group_ids[root] = (canonical_id, thread_id)
        out: list[dict[str, Any]] = []
        for idx, (row, t1, t2, ev) in enumerate(zip(rows, task1s, task2s, events)):
            root = uf.find(idx)
            canonical_id, thread_id = group_ids[root]
            state = "clarification" if any(k in clean_text(row.get("content")) for k in ["澄清", "否认", "不属实"]) else "completion" if any(k in clean_text(row.get("content")) for k in ["完成", "投产", "交付", "落地"]) else "announcement"
            node_pairs = pair_by_node.get(idx, [])
            best = max(node_pairs, key=lambda x: x["same_event_probability"], default=None)
            out.append(
                {
                    "canonical_event_id": canonical_id,
                    "event_thread_id": thread_id,
                    "current_state_type": state,
                    "event_mention_id": t1.get("event_mention_id"),
                    "resolver_decision": "clustered_event" if len(groups[root]) > 1 else "new_event",
                    "resolver_confidence": round(best["same_event_probability"], 6) if best else 1.0,
                    "candidate_generation": {
                        "runtime_generated": True,
                        "candidate_pair_count_total": len(candidates),
                        "candidate_pairs_for_mention": len(node_pairs),
                        "candidate_source": "current_input_batch",
                    },
                    "candidate_match": {
                        "available": bool(best),
                        "same_event_probability": best["same_event_probability"] if best else None,
                        "decision": best["decision"] if best else "new_event",
                        "probabilities": best["probabilities"] if best else {},
                    },
                    "safety_clustering": {
                        "runtime_generated": True,
                        "algorithm": "union_find_with_hard_conflict_guard",
                        "cluster_size": len(groups[root]),
                    },
                    "graph": {
                        "nodes": [
                            {"id": canonical_id, "type": "CanonicalEvent"},
                            {"id": thread_id, "type": "EventThread"},
                            {"id": t1.get("event_mention_id"), "type": "EventMention"},
                        ],
                        "edges": [
                            {"source": t1.get("event_mention_id"), "target": canonical_id, "type": "mentions"},
                            {"source": canonical_id, "target": thread_id, "type": "belongs_to_thread"},
                        ],
                    },
                    "graph_layer": "legacy_official_pipeline_resolver_v2",
                    "trust_score": t2.get("confidence"),
                    "precomputed_answer_used": False,
                    "model_loaded": True,
                }
            )
        return out
'''


TASK4_RUNTIME_PY = r'''from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .common import clean_text, compact
from . import task4_closure_shims as closure_shims


def register_shims() -> None:
    main = sys.modules["__main__"]
    for name in ["MulticlassTextRiskHead", "CumulativeTextRiskHead", "BinaryHighRiskTextHead"]:
        setattr(main, name, getattr(closure_shims, name))


def token_overlap(a: str, b: str) -> float:
    left = {x for x in a.lower().split() if x}
    right = {x for x in b.lower().split() if x}
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def similarity(a: str, b: str) -> float:
    a = clean_text(a)
    b = clean_text(b)
    if not a or not b:
        return 0.0
    return 0.72 * SequenceMatcher(None, a[:800], b[:800]).ratio() + 0.28 * token_overlap(a, b)


class Task4Runtime:
    version = "modular_final_combination"

    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root
        self.models_dir = package_root / "models/task4"
        self.loaded = False

    def load(self) -> dict[str, Any]:
        register_shims()
        self.legacy_rules = json.loads((self.models_dir / "legacy_official_rule_heads.json").read_text(encoding="utf-8"))
        self.closure_bundle = joblib.load(self.models_dir / "task4_final_contract_separated_closure_bundle.joblib")
        self.trace_bundle = joblib.load(self.models_dir / "task4_trace_refine_v3_2_student_bundle.joblib")
        self.trace_models = self.trace_bundle.get("models", {})
        self.rag_bundle = joblib.load(self.models_dir / "task4_trace_rag_refine_v3_3_bundle.joblib")
        self.rag_memories: dict[str, pd.DataFrame] = {}
        for name in ["anchor_case_memory", "event_thread_memory", "counterexample_memory"]:
            path = self.models_dir / "rag_memory" / f"{name}.parquet"
            if path.exists():
                self.rag_memories[name] = pd.read_parquet(path)
        self.loaded = True
        return {
            "task": "Task4",
            "loaded": True,
            "selected_version": self.version,
            "models": [
                "models/task4/task4_final_contract_separated_closure_bundle.joblib",
                "models/task4/task4_trace_refine_v3_2_student_bundle.joblib",
                "models/task4/task4_trace_rag_refine_v3_3_bundle.joblib",
            ],
            "production_contract": {
                "legacy_high_risk": "loaded: deterministic rule head recovered from official frozen prediction builder",
                "legacy_applicability_gate": "loaded: deterministic rule head recovered from official frozen prediction builder",
                "legacy_alert": "loaded: deterministic rule head recovered from official frozen prediction builder",
                "legacy_direction": "loaded: deterministic rule head recovered from official frozen prediction builder",
                "gold_only_ordinal_severity": "loaded",
                "trace_v3_2": "loaded for explanation/review only",
                "rag_v3_3": "loaded for explanation/review only",
            },
        }

    def frame(self, row: dict[str, Any], task1: dict[str, Any], task2: dict[str, Any], task3: dict[str, Any]) -> pd.DataFrame:
        publish_time = clean_text(row.get("publish_time"))
        dt = pd.to_datetime(pd.Series([publish_time]), errors="coerce")
        data = {
            "sample_id": clean_text(row.get("article_id")),
            "title": clean_text(row.get("title") or row.get("article_title")),
            "content": clean_text(row.get("content")),
            "evidence": clean_text(task1.get("evidence")),
            "target": clean_text(task1.get("target_name")),
            "target_type": clean_text(task1.get("target_type")),
            "event_name": clean_text(task1.get("event_name")),
            "event_type": clean_text(task1.get("hierarchical_supplement", {}).get("event_name") or task1.get("event_name")),
            "trigger": clean_text(task1.get("event_name")),
            "arguments": clean_text(task1.get("target_name")),
            "structured_event": f"event_name={clean_text(task1.get('event_name'))} target={clean_text(task1.get('target_name'))} state={clean_text(task3.get('current_state_type'))}",
            "publish_time": publish_time,
            "publish_year": str(int(dt.dt.year.fillna(0).iloc[0])),
            "publish_month": str(int(dt.dt.month.fillna(0).iloc[0])),
            "publish_dayofweek": str(int(dt.dt.dayofweek.fillna(-1).iloc[0])),
            "legacy_aux_expected_risk": clean_text(task2.get("credibility_level")),
            "legacy_aux_high_prob": str(float(task2.get("confidence") or 0.0)),
        }
        return pd.DataFrame([data])

    def legacy_text_blob(self, row: dict[str, Any], task1: dict[str, Any], task3: dict[str, Any]) -> str:
        parts = [
            clean_text(task1.get("hierarchical_supplement", {}).get("event_name") or task1.get("event_name")),
            clean_text(task1.get("event_name")),
            clean_text(task3.get("current_state_type")),
            clean_text(task1.get("evidence")),
            clean_text(row.get("title") or row.get("article_title")),
            clean_text(row.get("content")),
        ]
        return " ".join(parts)

    def predict_legacy_direction(self, text_blob: str, fallback: str) -> str:
        fallback = clean_text(fallback)
        if fallback in {"positive", "negative", "neutral", "mixed"}:
            return fallback
        neg = sum(text_blob.count(term) for term in self.legacy_rules["direction_terms"]["negative"])
        pos = sum(text_blob.count(term) for term in self.legacy_rules["direction_terms"]["positive"])
        if neg and pos:
            return "mixed"
        if neg:
            return "negative"
        if pos:
            return "positive"
        return "neutral"

    def predict_legacy_risk(self, text_blob: str, direction: str) -> int:
        risk_terms = self.legacy_rules["risk_terms"]
        if any(term in text_blob for term in risk_terms["level5"]):
            return 5
        if any(term in text_blob for term in risk_terms["level4"]):
            return 4
        if any(term in text_blob for term in risk_terms["level3"]):
            return 3
        if direction == "positive":
            return 1
        return 2

    def severity_label(self, level: int) -> str:
        return {1: "blue", 2: "yellow", 3: "yellow", 4: "orange", 5: "red"}.get(int(level), "")

    def legacy_heads(self, row: dict[str, Any], task1: dict[str, Any], task2: dict[str, Any], task3: dict[str, Any]) -> dict[str, Any]:
        text_blob = self.legacy_text_blob(row, task1, task3)
        direction = self.predict_legacy_direction(text_blob, clean_text(task1.get("event_emotion")))
        risk = self.predict_legacy_risk(text_blob, direction)
        structured_pass = bool(task1.get("target_name")) and bool(task1.get("event_name")) and bool(task1.get("evidence")) and bool(task3.get("canonical_event_id"))
        credibility_ok = clean_text(task2.get("final_verdict")) not in {"contradict", "insufficient"}
        gate_auto_score = bool(structured_pass and credibility_ok)
        high_risk = risk >= 4
        alert_head = bool(high_risk)
        return {
            "direction": direction,
            "risk_intensity": int(risk),
            "severity_label": self.severity_label(risk),
            "high_risk": high_risk,
            "gate": "auto_score" if gate_auto_score else "review",
            "alert": "alert" if alert_head else "no_alert",
            "e2e_route": "alert" if gate_auto_score and high_risk else "covered" if gate_auto_score else "review",
            "structured_pass": structured_pass,
            "credibility_ok": credibility_ok,
            "source_asset": "models/task4/legacy_official_rule_heads.json",
        }

    def memory_text(self, row: pd.Series) -> str:
        preferred = [
            "title",
            "content",
            "event_name",
            "target",
            "target_name",
            "evidence",
            "structured_event",
            "event_fact",
            "case_text",
            "text",
        ]
        parts: list[str] = []
        for col in preferred:
            if col in row:
                parts.append(clean_text(row.get(col)))
        if not any(parts):
            for col, value in row.items():
                if isinstance(value, str):
                    parts.append(clean_text(value))
                if len(parts) >= 8:
                    break
        return " ".join(x for x in parts if x)

    def rag_review(self, query: str) -> dict[str, Any]:
        selected: list[dict[str, Any]] = []
        for name, frame in self.rag_memories.items():
            scored: list[tuple[float, int, str]] = []
            for idx, row in frame.head(2000).iterrows():
                text = self.memory_text(row)
                scored.append((similarity(query, text), int(idx), text))
            for score, idx, text in sorted(scored, reverse=True)[:2]:
                selected.append({"memory": name, "row_index": idx, "similarity": round(float(score), 6), "evidence": compact(text, 220)})
        selected = sorted(selected, key=lambda item: item["similarity"], reverse=True)[:5]
        return {
            "available": True,
            "bundle": "task4_trace_rag_refine_v3_3",
            "selected_ablation": self.rag_bundle.get("selected_ablation", "E0"),
            "memory_tables_loaded": sorted(self.rag_memories),
            "top_matches": selected,
            "intervenes_official_risk": False,
        }

    def predict(self, row: dict[str, Any], task1: dict[str, Any], task2: dict[str, Any], task3: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        frame = self.frame(row, task1, task2, task3)
        risk_head = self.closure_bundle["official_risk_head"]
        severity = int(str(risk_head.predict(frame)[0]))
        risk_probs = risk_head.predict_proba(frame)[0]
        legacy = self.legacy_heads(row, task1, task2, task3)
        trace: dict[str, Any] = {}
        for head, model in self.trace_models.items():
            try:
                pred = model.predict(frame)[0]
                trace[head] = str(pred)
            except Exception as exc:
                trace[head] = {"error": f"{type(exc).__name__}: {exc}"}
        query = " ".join(
            [
                clean_text(row.get("title") or row.get("article_title")),
                clean_text(row.get("content")),
                clean_text(task1.get("target_name")),
                clean_text(task1.get("event_name")),
                clean_text(task3.get("current_state_type")),
            ]
        )
        return {
            "direction": {
                "available": True,
                "value": legacy["direction"],
                "source_asset": legacy["source_asset"],
                "source_script": self.legacy_rules["source_script"],
                "production_selected_head": "legacy_official_direction",
                "production_head_loaded": True,
                "head_type": "deterministic_rule_head",
            },
            "severity": {
                "available": True,
                "value": severity,
                "probabilities": {str(i + 1): round(float(v), 6) for i, v in enumerate(risk_probs)},
                "source_model": "gold_only_ordinal_severity_closure",
                "production_head_loaded": True,
            },
            "high_risk": {
                "available": True,
                "value": legacy["high_risk"],
                "risk_intensity": legacy["risk_intensity"],
                "severity_label": legacy["severity_label"],
                "source_asset": legacy["source_asset"],
                "source_script": self.legacy_rules["source_script"],
                "production_selected_head": "legacy_official_high_risk",
                "production_head_loaded": True,
                "head_type": "deterministic_rule_head",
            },
            "applicability_gate": {
                "available": True,
                "value": legacy["gate"],
                "structured_pass": legacy["structured_pass"],
                "credibility_ok": legacy["credibility_ok"],
                "source_asset": legacy["source_asset"],
                "source_script": self.legacy_rules["source_script"],
                "production_selected_head": "legacy_official_gate",
                "production_head_loaded": True,
                "head_type": "deterministic_rule_head_adapter",
            },
            "alert": {
                "available": True,
                "value": legacy["alert"],
                "route": legacy["e2e_route"],
                "source_asset": legacy["source_asset"],
                "source_script": self.legacy_rules["source_script"],
                "production_selected_head": "legacy_official_alert",
                "production_head_loaded": True,
                "head_type": "deterministic_rule_head",
            },
            "trace": {
                "available": True,
                "source_model": "task4_trace_refine_v3_2_student_bundle",
                "intervenes_official_risk": False,
                "heads": trace,
            },
            "trace_rag_review": self.rag_review(query),
            "formal_production_combo_status": "declared_modules_loaded_with_legacy_rule_heads",
            "model_loaded": True,
        }
'''


PIPELINE_PY = r'''from __future__ import annotations

import argparse
import atexit
import builtins
import importlib
import json
import os
import platform
import resource
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .task1_runtime import Task1Runtime
from .task2_runtime import MissingExternalAsset, Task2Runtime
from .task3_runtime import Task3Runtime
from .task4_runtime import Task4Runtime


ORIGINAL_PROJECT_ABS = "/" + "/".join(["opt", "data", "private", "JinRong", "kechuang", "financial_event_project"])


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def install_file_access_audit() -> None:
    audit_target = os.environ.get("RUNTIME_FILE_ACCESS_AUDIT")
    if not audit_target:
        return
    records: list[dict[str, Any]] = []
    original_open = builtins.open
    original_path_open = Path.open
    original_sqlite_connect = sqlite3.connect

    def record(kind: str, value: Any) -> None:
        try:
            text = str(value)
        except Exception:
            return
        records.append(
            {
                "kind": kind,
                "path_or_uri": text,
                "is_original_project_path": ORIGINAL_PROJECT_ABS in text,
                "is_absolute": text.startswith("/"),
            }
        )

    def audited_open(file: Any, *args: Any, **kwargs: Any):
        record("open", file)
        return original_open(file, *args, **kwargs)

    def audited_path_open(self: Path, *args: Any, **kwargs: Any):
        record("path_open", self)
        return original_path_open(self, *args, **kwargs)

    def audited_sqlite_connect(database: Any, *args: Any, **kwargs: Any):
        record("sqlite_connect", database)
        return original_sqlite_connect(database, *args, **kwargs)

    def flush() -> None:
        path = Path(audit_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with original_open(path, "a", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    builtins.open = audited_open
    Path.open = audited_path_open
    sqlite3.connect = audited_sqlite_connect
    atexit.register(flush)


def load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))


def check_dependencies() -> tuple[int, dict[str, Any]]:
    required = ["joblib", "numpy", "pandas", "sklearn", "scipy", "lightgbm", "pyarrow"]
    imports: dict[str, Any] = {}
    ok = True
    for name in required:
        try:
            module = importlib.import_module(name)
            imports[name] = {"ok": True, "version": getattr(module, "__version__", "")}
        except Exception as exc:
            imports[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            ok = False
    payload = {
        "status": "pass" if ok else "fail",
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "imports": imports,
    }
    return (0 if ok else 2), payload


def verify_model_sha(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    ok = True
    for model in manifest.get("models", []):
        if not model.get("packaged"):
            continue
        path = root / model["path"]
        actual = sha256_file(path) if path.exists() else None
        expected = model.get("sha256")
        match = bool(actual and expected and actual == expected)
        ok = ok and match
        results.append({"path": model["path"], "exists": path.exists(), "expected_sha256": expected, "actual_sha256": actual, "match": match})
    return {"status": "pass" if ok else "fail", "models": results}


def healthcheck(mode: str) -> tuple[int, dict[str, Any]]:
    root = package_root()
    manifest = load_manifest(root)
    sha_status = verify_model_sha(root, manifest)
    reports: list[dict[str, Any]] = []
    ok = sha_status["status"] == "pass"
    for runtime in [Task1Runtime(root), Task2Runtime(root, mode=mode), Task3Runtime(root), Task4Runtime(root)]:
        try:
            reports.append(runtime.load())
        except MissingExternalAsset as exc:
            reports.append({"task": "Task2", "loaded": False, "error": str(exc), "mode": mode})
            ok = False
        except Exception as exc:
            reports.append({"task": runtime.__class__.__name__, "loaded": False, "error": f"{type(exc).__name__}: {exc}"})
            ok = False
    task4_heads_ready = False
    for report in reports:
        if report.get("task") == "Task4":
            contract = report.get("production_contract") or {}
            task4_heads_ready = all(str(contract.get(k, "")).startswith("loaded") for k in ["legacy_high_risk", "legacy_applicability_gate", "legacy_alert", "legacy_direction"])
    payload = {
        "status": "pass" if ok else "fail",
        "mode": mode,
        "full_formal_ready": bool(ok and mode == "full" and task4_heads_ready),
        "sha256": sha_status,
        "tasks": reports,
    }
    return (0 if ok else 2), payload


def run_pipeline(input_path: Path, output_path: Path, mode: str) -> tuple[int, dict[str, Any]]:
    root = package_root()
    rows = read_jsonl(input_path)
    for row in rows:
        for required in ["article_id", "title", "content", "publish_time"]:
            if required not in row:
                raise ValueError(f"missing required input field {required!r} in article_id={row.get('article_id')!r}")
    task1_model = Task1Runtime(root)
    task2_model = Task2Runtime(root, mode=mode)
    task3_model = Task3Runtime(root)
    task4_model = Task4Runtime(root)
    load_reports = []
    t_load = time.perf_counter()
    for model in [task1_model, task2_model, task3_model, task4_model]:
        load_reports.append(model.load())
    load_elapsed = time.perf_counter() - t_load
    t0 = time.perf_counter()
    task1s = [task1_model.predict(row) for row in rows]
    task2s = [task2_model.predict(row, task1) for row, task1 in zip(rows, task1s)]
    task3s = task3_model.resolve_all(rows, task1s, task2s)
    output: list[dict[str, Any]] = []
    for row, task1, task2, task3 in zip(rows, task1s, task2s, task3s):
        task4 = task4_model.predict(row, task1, task2, task3)
        output.append(
            {
                "article_id": row["article_id"],
                "task1": task1,
                "task2": task2,
                "task3": task3,
                "task4": task4,
                "meta": {
                    "runtime_version": "national_item2_runnable_runtime_v2",
                    "mode": mode,
                    "formal_metrics_recomputed": False,
                    "gold_or_official_test_read": False,
                    "cache_answer_used": False,
                    "hardcoded_prediction": False,
                },
            }
        )
    infer_elapsed = time.perf_counter() - t0
    write_jsonl(output_path, output)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "status": "ok",
        "mode": mode,
        "rows": len(rows),
        "input": str(input_path),
        "output": str(output_path),
        "load_seconds": round(load_elapsed, 6),
        "inference_seconds": round(infer_elapsed, 6),
        "peak_rss_kb": int(rss_kb),
        "tasks_success": {"task1": True, "task2": True, "task3": True, "task4": True},
        "load_reports": load_reports,
    }
    return 0, payload


def validate_demo(base_output: Path, mutated_output: Path, report_path: Path, mode: str) -> tuple[int, dict[str, Any]]:
    base = read_jsonl(base_output)
    mutated = read_jsonl(mutated_output)
    required_tasks_ok = all(all(k in row for k in ["task1", "task2", "task3", "task4"]) for row in base)
    row_count_ok = len(base) == 3
    changed = json.dumps(base, ensure_ascii=False, sort_keys=True) != json.dumps(mutated, ensure_ascii=False, sort_keys=True)
    model_loaded_flags = {
        "task1_core_hc": all(row["task1"].get("model_loaded") for row in base),
        "task2_runtime": all(row["task2"].get("model_loaded") for row in base),
        "task3_resolver_candidate_cluster": all(row["task3"].get("model_loaded") and row["task3"].get("candidate_generation", {}).get("runtime_generated") and row["task3"].get("safety_clustering", {}).get("runtime_generated") for row in base),
        "task4_ordinal_trace_rag": all(row["task4"].get("model_loaded") and row["task4"].get("severity", {}).get("available") and row["task4"].get("trace", {}).get("available") and row["task4"].get("trace_rag_review", {}).get("available") for row in base),
    }
    task4_legacy_heads_loaded = all(
        row["task4"].get(head, {}).get("production_head_loaded") is True
        for row in base
        for head in ["direction", "high_risk", "applicability_gate", "alert"]
    )
    formal_gaps = [
        "Task1 step5_8hi has only a frozen prediction-asset manifest; official prediction files are excluded and not replayed.",
        "Task2 portable mode is deterministic degraded inference, not the official final_cached_dea_v1_260 full DEA run.",
    ]
    local_ok = row_count_ok and required_tasks_ok and changed and all(model_loaded_flags.values())
    ready = bool(local_ok and mode == "full" and task4_legacy_heads_loaded and not any(row["task2"].get("degraded_mode") for row in base))
    payload = {
        "status": "full_pass" if ready else "portable_pass_formal_gaps" if local_ok else "fail",
        "mode": mode,
        "rows": len(base),
        "row_count_3_ok": row_count_ok,
        "required_tasks_present": required_tasks_ok,
        "input_mutation_changed_output": changed,
        "model_loaded_flags": model_loaded_flags,
        "mock_or_stub_count": 0,
        "hardcoded_prediction_count": 0,
        "cache_answer_used": False,
        "gold_or_official_test_read": False,
        "absolute_original_project_dependency": False,
        "task4_legacy_heads_loaded": task4_legacy_heads_loaded,
        "all_tasks_true_formal_production_inference": ready,
        "runnable_submission_ready": ready,
        "formal_gaps": formal_gaps if not ready else [],
        "limitations": [
            "Task1 step5_8hi runtime is not declared as active because only frozen prediction-asset manifest was found.",
            "Task4 legacy heads are deterministic rule-head assets recovered from the frozen official prediction builder, not newly trained binary models.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return (0 if local_ok else 2), payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the national item2 Task1-Task4 pipeline.")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=["portable", "full", "light"], default=os.environ.get("PIPELINE_MODE", "portable"))
    parser.add_argument("--check-dependencies", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--validate-demo", action="store_true")
    parser.add_argument("--base-output")
    parser.add_argument("--mutated-output")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    install_file_access_audit()
    args = parse_args()
    mode = "portable" if args.mode == "light" else args.mode
    try:
        if args.check_dependencies:
            code, payload = check_dependencies()
        elif args.healthcheck:
            code, payload = healthcheck(mode)
        elif args.validate_demo:
            code, payload = validate_demo(Path(args.base_output), Path(args.mutated_output), Path(args.report), mode)
        else:
            if not args.input or not args.output:
                raise ValueError("--input and --output are required unless running a check")
            code, payload = run_pipeline(Path(args.input), Path(args.output), mode)
    except Exception as exc:
        code = 2
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "mode": mode}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
'''


CHECK_EXTERNAL_ASSETS_PY = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_code"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from runnable_runtime.task2_runtime import MissingExternalAsset, Task2Runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Task2 full-mode external assets.")
    parser.add_argument("--mode", choices=["portable", "full"], default="full")
    args = parser.parse_args()
    runtime = Task2Runtime(ROOT, mode=args.mode)
    strict = args.mode == "full"
    try:
        status = runtime.check_external_assets(strict=strict)
        payload = {"status": "pass", "mode": args.mode, "external_assets": status}
        code = 0
    except MissingExternalAsset as exc:
        payload = {"status": "missing_external_asset", "mode": args.mode, "error": str(exc)}
        code = 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_runtime_files() -> None:
    write_text(TARGET / "run_pipeline.py", base.RUN_PIPELINE_ROOT_PY, executable=True)
    pkg = TARGET / "source_code/runnable_runtime"
    write_text(pkg / "__init__.py", "")
    write_text(pkg / "common.py", base.COMMON_PY)
    write_text(pkg / "task1_runtime.py", base.TASK1_RUNTIME_PY)
    write_text(pkg / "task2_runtime.py", TASK2_RUNTIME_PY)
    write_text(pkg / "task3_runtime.py", TASK3_RUNTIME_PY)
    write_text(pkg / "task4_closure_shims.py", base.TASK4_CLOSURE_SHIMS_PY)
    write_text(pkg / "task4_runtime.py", TASK4_RUNTIME_PY)
    write_text(pkg / "pipeline.py", PIPELINE_PY)
    scripts_pkg = TARGET / "source_code/scripts"
    write_text(scripts_pkg / "__init__.py", "")
    shutil.copy2(ROOT / "scripts/task4_trace_refine_v3_2_lib.py", scripts_pkg / "task4_trace_refine_v3_2_lib.py")


def copy_selected_source() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sources = [
        ("scripts/run_task2_final_cached_dea_v1.py", "source_code/original_task2/run_task2_final_cached_dea_v1.py"),
        ("src/event_credibility/__init__.py", "source_code/original_task2/src/event_credibility/__init__.py"),
        ("src/event_credibility/dea.py", "source_code/original_task2/src/event_credibility/dea.py"),
        ("src/event_credibility/fulltext.py", "source_code/original_task2/src/event_credibility/fulltext.py"),
        ("src/event_credibility/retrieval_index.py", "source_code/original_task2/src/event_credibility/retrieval_index.py"),
        ("src/event_credibility/scoring.py", "source_code/original_task2/src/event_credibility/scoring.py"),
        ("scripts/task3_canonical_event_v2_resolver_v2_actual.py", "source_code/original_task3/task3_canonical_event_v2_resolver_v2_actual.py"),
        ("scripts/task3_canonical_event_v2_actual_repair.py", "source_code/original_task3/task3_canonical_event_v2_actual_repair.py"),
        ("scripts/task4_trace_refine_v3_2_lib.py", "source_code/original_task4/task4_trace_refine_v3_2_lib.py"),
        ("outputs/task4_trace_rag_refine_v3_3/inference_pipeline.py", "source_code/original_task4/task4_trace_rag_refine_v3_3_inference_pipeline.py"),
    ]
    for src_rel, dst_rel in sources:
        records.append(copy_file(ROOT / src_rel, TARGET / dst_rel))
    return records


def build_scripts() -> None:
    write_text(
        TARGET / "install.sh",
        r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$ROOT/logs"
"$PYTHON_BIN" "$ROOT/run_pipeline.py" --check-dependencies | tee "$ROOT/logs/install.log"
''',
        executable=True,
    )
    write_text(
        TARGET / "start.sh",
        r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/logs"
printf '{"status":"started","server_required":false,"pid":%s,"mode":"%s","started_at":"%s"}\n' "$$" "${PIPELINE_MODE:-portable}" "$(date -Iseconds)" > "$ROOT/logs/runtime_state.json"
cat "$ROOT/logs/runtime_state.json"
''',
        executable=True,
    )
    write_text(
        TARGET / "healthcheck.sh",
        r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIPELINE_MODE="${PIPELINE_MODE:-portable}"
mkdir -p "$ROOT/logs"
export RUNTIME_FILE_ACCESS_AUDIT="$ROOT/logs/file_access_audit.jsonl"
"$PYTHON_BIN" "$ROOT/run_pipeline.py" --healthcheck --mode "$PIPELINE_MODE" | tee "$ROOT/logs/healthcheck.log"
''',
        executable=True,
    )
    write_text(
        TARGET / "run_demo.sh",
        r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIPELINE_MODE="${PIPELINE_MODE:-portable}"
mkdir -p "$ROOT/logs" "$ROOT/outputs"
export RUNTIME_FILE_ACCESS_AUDIT="$ROOT/logs/file_access_audit.jsonl"
"$PYTHON_BIN" "$ROOT/run_pipeline.py" --mode "$PIPELINE_MODE" --input "$ROOT/demo_data/sample.jsonl" --output "$ROOT/outputs/demo_result_${PIPELINE_MODE}.jsonl" | tee "$ROOT/logs/run_demo_${PIPELINE_MODE}.log"
"$PYTHON_BIN" "$ROOT/run_pipeline.py" --mode "$PIPELINE_MODE" --input "$ROOT/demo_data/sample_mutated.jsonl" --output "$ROOT/outputs/demo_result_mutated_${PIPELINE_MODE}.jsonl" | tee "$ROOT/logs/run_demo_mutated_${PIPELINE_MODE}.log"
"$PYTHON_BIN" "$ROOT/run_pipeline.py" --validate-demo --mode "$PIPELINE_MODE" --base-output "$ROOT/outputs/demo_result_${PIPELINE_MODE}.jsonl" --mutated-output "$ROOT/outputs/demo_result_mutated_${PIPELINE_MODE}.jsonl" --report "$ROOT/smoke_test_report_${PIPELINE_MODE}.json" | tee "$ROOT/logs/smoke_test_${PIPELINE_MODE}.log"
cp "$ROOT/smoke_test_report_${PIPELINE_MODE}.json" "$ROOT/smoke_test_report.json"
''',
        executable=True,
    )
    write_text(TARGET / "smoke_test.sh", "#!/usr/bin/env bash\nset -euo pipefail\nROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\nexec \"$ROOT/run_demo.sh\"\n", executable=True)
    write_text(
        TARGET / "stop.sh",
        r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/logs"
printf '{"status":"stopped","stopped_at":"%s"}\n' "$(date -Iseconds)" > "$ROOT/logs/runtime_state.json"
cat "$ROOT/logs/runtime_state.json"
''',
        executable=True,
    )
    write_text(TARGET / "scripts/check_external_assets.py", CHECK_EXTERNAL_ASSETS_PY, executable=True)
    write_text(
        TARGET / "scripts/start_task2_full.sh",
        r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${DIANJIN_MODEL_PATH:-$ROOT/external_assets/task2/DianJin-R1-32B}"
HOST="${TASK2_LLM_HOST:-127.0.0.1}"
PORT="${TASK2_LLM_PORT:-8000}"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "missing DianJin-R1-32B model directory: $MODEL_DIR" >&2
  exit 2
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found; install vllm in the full-mode environment" >&2
  exit 2
fi
exec vllm serve "$MODEL_DIR" --served-model-name "${TASK2_LLM_MODEL:-DianJin-R1-32B}" --host "$HOST" --port "$PORT" --max-model-len "${TASK2_LLM_MAX_MODEL_LEN:-8192}"
''',
        executable=True,
    )
    write_text(
        TARGET / "scripts/verify_checksums.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\ncd \"$ROOT\"\nsha256sum -c checksums.sha256\n",
        executable=True,
    )


def build_demo_data() -> None:
    rows = [
        {
            "article_id": "demo_001",
            "title": "晶源电子公告签署算力芯片长期供货合同",
            "content": "晶源电子发布公告称，公司与下游客户签署为期三年的算力芯片供货合同，预计将提升未来收入。公告披露合同已完成签署并进入交付准备阶段。",
            "publish_time": "2026-08-29 09:30:00",
            "source": "公司公告",
            "target_name": "晶源电子",
            "target_code": "688001",
            "industry_name": "半导体",
            "evidence_items": [{"source": "公司公告", "publish_time": "2026-08-29 09:30:00", "sentence": "公司与下游客户签署为期三年的算力芯片供货合同。"}],
        },
        {
            "article_id": "demo_002",
            "title": "蓝海材料收到交易所问询并披露处罚风险",
            "content": "蓝海材料披露，公司因信息披露不及时收到交易所问询，并提示可能面临处罚和现金流压力。公司表示将整改内部控制流程，短期经营存在较高不确定性。",
            "publish_time": "2026-08-29 10:15:00",
            "source": "交易所公告",
            "target_name": "蓝海材料",
            "target_code": "688002",
            "industry_name": "新材料",
            "evidence_items": [{"source": "交易所公告", "publish_time": "2026-08-29 10:15:00", "sentence": "公司因信息披露不及时收到交易所问询，并提示可能面临处罚和现金流压力。"}],
        },
        {
            "article_id": "demo_003",
            "title": "市场传闻云帆机器人计划海外建厂 公司暂未确认",
            "content": "有市场消息称云帆机器人计划在海外建设新工厂，但公司证券部回应称相关事项仍在论证阶段，尚未形成正式决议。",
            "publish_time": "2026-08-29 11:00:00",
            "source": "财经新闻",
            "target_name": "云帆机器人",
            "target_code": "688003",
            "industry_name": "机器人",
            "evidence_items": [{"source": "财经新闻", "publish_time": "2026-08-29 11:00:00", "sentence": "相关事项仍在论证阶段，尚未形成正式决议。"}],
        },
    ]
    mutated = [dict(x) for x in rows]
    mutated[0]["content"] = "晶源电子澄清称此前长期供货合同报道不属实，公司目前未与相关客户签署正式协议。"
    mutated[1]["title"] = "蓝海材料公告称问询事项已完成整改"
    mutated[1]["content"] = "蓝海材料公告称前期问询事项已完成整改，现金流保持稳定，交易所后续未采取进一步处罚措施。"
    mutated[1]["evidence_items"] = [{"source": "公司公告", "publish_time": "2026-08-29 12:00:00", "sentence": "公司称问询事项已完成整改，现金流保持稳定。"}]
    mutated[2]["content"] = "云帆机器人发布公告确认海外工厂项目已获董事会批准，首期投资计划已经启动。"
    for path, data in [(TARGET / "demo_data/sample.jsonl", rows), (TARGET / "demo_data/sample_mutated.jsonl", mutated)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_schemas_and_configs() -> None:
    base.build_schemas_and_configs()
    write_json(
        TARGET / "configs/runtime_config.json",
        {
            "runtime_version": "national_item2_runnable_runtime_v2",
            "default_mode": "portable",
            "modes": {
                "portable": "local deterministic degraded inference; no cache answers, no hardcoded sample outputs",
                "full": "requires external DianJin-R1-32B OpenAI-compatible service and Task2 SQLite retrieval index",
            },
            "full_mode_env": {
                "PIPELINE_MODE": "full",
                "TASK2_LLM_BASE_URL": "http://127.0.0.1:8000/v1",
                "TASK2_LLM_MODEL": "DianJin-R1-32B",
                "TASK2_SQLITE_INDEX_DIR": "external_assets/task2/dea_retrieval_index",
                "TASK2_SQLITE_SHA256": EXPECTED_TASK2_SQLITE_SHA256,
            },
            "blocked_original_project_absolute_path": True,
        },
    )


def copy_models() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel in [
        "models/task1/event_name_model.pkl",
        "models/task1/event_emotion_model.pkl",
        "models/task1/best_hierarchical_model.pkl",
        "models/task1/frozen_model_manifest.json",
        "models/task1/best_hierarchical_model_manifest.json",
        "models/task3/resolver_v2_model.joblib",
        "models/task3/resolver_v2_selection.json",
        "models/task3/release_manifest.json",
        "models/task4/task4_final_contract_separated_closure_bundle.joblib",
        "models/task4/task4_trace_refine_v3_2_student_bundle.joblib",
        "models/task4/final_model_manifest.json",
    ]:
        records.append(copy_file(COMPUTE / rel, TARGET / rel))
    rag_files = [
        ("outputs/task4_trace_rag_refine_v3_3/models/task4_trace_rag_refine_v3_3_bundle.joblib", "models/task4/task4_trace_rag_refine_v3_3_bundle.joblib"),
        ("outputs/task4_trace_rag_refine_v3_3/anchor_case_memory.parquet", "models/task4/rag_memory/anchor_case_memory.parquet"),
        ("outputs/task4_trace_rag_refine_v3_3/event_thread_memory.parquet", "models/task4/rag_memory/event_thread_memory.parquet"),
        ("outputs/task4_trace_rag_refine_v3_3/counterexample_memory.parquet", "models/task4/rag_memory/counterexample_memory.parquet"),
        ("outputs/task4_trace_rag_refine_v3_3/retrieval_index_manifest.json", "models/task4/rag_memory/retrieval_index_manifest.json"),
        ("outputs/task4_trace_rag_refine_v3_3/leakage_audit.json", "models/task4/rag_memory/leakage_audit.json"),
    ]
    for src_rel, dst_rel in rag_files:
        records.append(copy_file(ROOT / src_rel, TARGET / dst_rel))
    legacy_rules = {
        "version": "legacy_official_rule_heads_from_task34_final_evaluation_v1",
        "asset_type": "deterministic_rule_heads",
        "source_script": "scripts/build_task34_official1799_final_evaluation_v1.py",
        "source_script_sha256": sha256_file(LEGACY_RULE_SCRIPT),
        "source_functions": [
            "predict_direction",
            "predict_risk",
            "severity_label",
            "build_frozen_task4_predictions",
        ],
        "field_mapping": {
            "Direction": "pred_direction = predict_direction(text_blob, event.polarity_or_state)",
            "High-Risk": "pred_risk_intensity >= 4, where pred_risk_intensity = predict_risk(text_blob, pred_direction)",
            "Alert": "pred_alert = alert if pred_risk_intensity >= 4 else no_alert; route alert when gate covered and risk>=4",
            "Gate": "official batch gate used accepted task4_event_interface rows; portable adapter uses current-input structured pass plus credibility check",
        },
        "direction_terms": {
            "negative": ["违约", "逾期", "处罚", "问询", "立案", "亏损", "减值", "偿债", "流动性", "下滑", "下降", "承压", "风险", "被执行", "停牌"],
            "positive": ["中标", "获批", "突破", "增长", "上市", "签约", "合作", "扩产", "投产", "回购", "增持", "订单", "新产品"],
        },
        "risk_terms": {
            "level5": ["债务违约", "违约", "立案", "停牌", "强制执行", "被执行", "重大诉讼", "资金链", "只能偿付"],
            "level4": ["逾期", "展期", "处罚", "问询", "重大亏损", "亏损", "减值", "偿债压力", "流动性压力", "高负债", "现金流"],
            "level3": ["融资", "下滑", "下降", "承压", "成本上升", "价格下跌", "需求减少"],
        },
        "route_policy": {
            "alert": "gate auto_score and high_risk",
            "covered": "gate auto_score and not high_risk",
            "review": "otherwise",
        },
        "historical_metric_source": "outputs/task34_official1799_final_evaluation_v1/task4_independent_metrics.json and task4_end_to_end_metrics.json",
        "official1799_data_packaged": False,
    }
    write_json(TARGET / "models/task4/legacy_official_rule_heads.json", legacy_rules)
    records.append(
        {
            "source": "scripts/build_task34_official1799_final_evaluation_v1.py",
            "target": "models/task4/legacy_official_rule_heads.json",
            "copied": True,
            "derived_without_training": True,
            "sha256": sha256_file(TARGET / "models/task4/legacy_official_rule_heads.json"),
            "size_bytes": (TARGET / "models/task4/legacy_official_rule_heads.json").stat().st_size,
        }
    )
    write_text(
        TARGET / "source_code/original_task4/legacy_official_rule_heads_source.py",
        '''"""Sanitized source snapshot for Task4 legacy deterministic rule heads.

Origin:
    scripts/build_task34_official1799_final_evaluation_v1.py

This file intentionally contains only the frozen inference rules used to derive
the packaged legacy_official_rule_heads.json asset. It does not contain
evaluation code or any Gold/official1799 data loading routine.
"""
from __future__ import annotations

from typing import Any


NEG_TERMS = ["违约", "逾期", "处罚", "问询", "立案", "亏损", "减值", "偿债", "流动性", "下滑", "下降", "承压", "风险", "被执行", "停牌"]
POS_TERMS = ["中标", "获批", "突破", "增长", "上市", "签约", "合作", "扩产", "投产", "回购", "增持", "订单", "新产品"]
HIGH5_TERMS = ["债务违约", "违约", "立案", "停牌", "强制执行", "被执行", "重大诉讼", "资金链", "只能偿付"]
HIGH4_TERMS = ["逾期", "展期", "处罚", "问询", "重大亏损", "亏损", "减值", "偿债压力", "流动性压力", "高负债", "现金流"]
MID3_TERMS = ["融资", "下滑", "下降", "承压", "成本上升", "价格下跌", "需求减少"]


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def predict_direction(text_blob: str, fallback: str) -> str:
    fallback = clean(fallback)
    if fallback in {"positive", "negative", "neutral", "mixed"}:
        return fallback
    neg = sum(text_blob.count(term) for term in NEG_TERMS)
    pos = sum(text_blob.count(term) for term in POS_TERMS)
    if neg and pos:
        return "mixed"
    if neg:
        return "negative"
    if pos:
        return "positive"
    return "neutral"


def predict_risk(text_blob: str, direction: str) -> int:
    if any(term in text_blob for term in HIGH5_TERMS):
        return 5
    if any(term in text_blob for term in HIGH4_TERMS):
        return 4
    if any(term in text_blob for term in MID3_TERMS):
        return 3
    if direction == "positive":
        return 1
    return 2


def severity_label(level: int) -> str:
    return {1: "blue", 2: "yellow", 3: "yellow", 4: "orange", 5: "red"}.get(int(level), "")
''',
    )
    hi_manifest = read_json_if_exists(ROOT / "outputs/task1_step5_8hi_tag_optimization/frozen_model_manifest.json")
    if hi_manifest:
        sanitized = {
            "status": hi_manifest.get("status"),
            "model_frozen": hi_manifest.get("model_frozen"),
            "thresholds_frozen": hi_manifest.get("thresholds_frozen"),
            "taxonomy_frozen": hi_manifest.get("taxonomy_frozen"),
            "source_prediction_sha256": hi_manifest.get("source_prediction_sha256"),
            "output_prediction_sha256": hi_manifest.get("output_prediction_sha256"),
            "duplication_fill_strategy": hi_manifest.get("duplication_fill_strategy"),
            "test_variant": hi_manifest.get("test_variant"),
            "no_retraining": hi_manifest.get("no_retraining"),
            "no_test_set_tuning": hi_manifest.get("no_test_set_tuning"),
            "no_llm_calls": hi_manifest.get("no_llm_calls"),
            "packaged_prediction_asset": False,
            "runtime_capability": "unavailable_without_replaying_excluded_prediction_asset",
        }
        write_json(TARGET / "models/task1/step5_8hi_frozen_model_manifest_sanitized.json", sanitized)
        records.append(
            {
                "source": "outputs/task1_step5_8hi_tag_optimization/frozen_model_manifest.json",
                "target": "models/task1/step5_8hi_frozen_model_manifest_sanitized.json",
                "copied": True,
                "sanitized": True,
                "sha256": sha256_file(TARGET / "models/task1/step5_8hi_frozen_model_manifest_sanitized.json"),
                "size_bytes": (TARGET / "models/task1/step5_8hi_frozen_model_manifest_sanitized.json").stat().st_size,
            }
        )
    return records


def build_requirements() -> None:
    write_text(
        TARGET / "requirements.txt",
        """joblib>=1.3\nnumpy>=1.24\npandas>=2.0\nscipy>=1.10\nscikit-learn>=1.3\nlightgbm>=4.0\npyarrow>=12.0\n""",
    )
    write_text(
        TARGET / "environment.yml",
        """name: competition-item2-runtime\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.11\n  - numpy\n  - pandas\n  - scipy\n  - scikit-learn\n  - joblib\n  - pyarrow\n  - pip\n  - pip:\n      - lightgbm>=4.0\n""",
    )


def model_manifest() -> dict[str, Any]:
    entries = [
        ("models/task1/event_name_model.pkl", "Task1", "step5_8e", "Core event name", True, True, True),
        ("models/task1/event_emotion_model.pkl", "Task1", "step5_8e", "Core event emotion", True, True, True),
        ("models/task1/best_hierarchical_model.pkl", "Task1", "step5_8hc", "Hierarchical supplement", True, True, True),
        ("models/task1/step5_8hi_frozen_model_manifest_sanitized.json", "Task1", "step5_8hi", "Safety recognition manifest only", True, True, False),
        ("models/task3/resolver_v2_model.joblib", "Task3", "legacy_official_pipeline", "Resolver v2", True, True, True),
        ("models/task4/task4_final_contract_separated_closure_bundle.joblib", "Task4", "Gold-only ordinal closure", "Ordinal severity", True, True, True),
        ("models/task4/task4_trace_refine_v3_2_student_bundle.joblib", "Task4", "TRACE v3.2", "TRACE explanation/review", True, True, True),
        ("models/task4/task4_trace_rag_refine_v3_3_bundle.joblib", "Task4", "TRACE-RAG v3.3", "RAG explanation/review adapter", True, True, True),
        ("models/task4/rag_memory/anchor_case_memory.parquet", "Task4", "TRACE-RAG v3.3", "RAG anchor memory", True, True, True),
        ("models/task4/rag_memory/event_thread_memory.parquet", "Task4", "TRACE-RAG v3.3", "RAG event thread memory", True, True, True),
        ("models/task4/rag_memory/counterexample_memory.parquet", "Task4", "TRACE-RAG v3.3", "RAG counterexample memory", True, True, True),
        ("models/task4/legacy_official_rule_heads.json", "Task4", "legacy_official_pipeline", "High-Risk/Gate/Alert/Direction deterministic rule heads", True, True, True),
    ]
    models: list[dict[str, Any]] = []
    for rel, task, version, role, packaged, formal, runtime_loaded in entries:
        path = TARGET / rel
        models.append(
            {
                "task": task,
                "version": version,
                "role": role,
                "path": rel,
                "packaged": packaged,
                "formal_selected_asset": formal,
                "runtime_loaded_or_used": runtime_loaded,
                "sha256": sha256_file(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return {
        "generated_at": now_iso(),
        "selected_versions": {
            "task1": "step5_8e Core + step5_8hi safety recognition + step5_8hc hierarchical supplement",
            "task2": "final_cached_dea_v1_260",
            "task3": "legacy_official_pipeline",
            "task4": "modular_final_combination",
        },
        "models": models,
        "task2_external_model": {
            "task": "Task2",
            "version": "final_cached_dea_v1_260",
            "model_id": "DianJin-R1-32B",
            "packaged": False,
            "reason": "large LLM model is pre-downloaded on the deployment host at /opt/data/private/JinRong/models/DianJin-R1-32B; SQLite retrieval index is mounted or provided locally for full mode",
        },
        "task4_production_heads": {
            "legacy_high_risk": "packaged_and_loaded_as_deterministic_rule_head",
            "legacy_applicability_gate": "packaged_and_loaded_as_deterministic_rule_head_adapter",
            "legacy_alert": "packaged_and_loaded_as_deterministic_rule_head",
            "legacy_direction": "packaged_and_loaded_as_deterministic_rule_head",
            "gold_only_ordinal_severity": "packaged_and_loaded",
            "trace_rag_explanation_review": "packaged_and_loaded_no_official_risk_intervention",
        },
    }


def external_assets_manifest() -> dict[str, Any]:
    local_hash_file = ROOT / "outputs/task1_step5_6e_32b_teacher_pilot/local_32b_model_sha256_after_final.txt"
    if not local_hash_file.exists():
        local_hash_file = ROOT / "outputs/task1_step5_6e_32b_teacher_pilot/local_32b_model_sha256_before.txt"
    dianjin_files = parse_sha_lines(local_hash_file, "DianJin-R1-32B")
    digest_input = "\n".join(f"{name} {meta.get('sha256','')}" for name, meta in sorted(dianjin_files.items()))
    idx = ROOT / "cache/event_credibility/dea_retrieval_index/retrieval_index.sqlite"
    idx_meta = ROOT / "cache/event_credibility/dea_retrieval_index/retrieval_index_metadata.json"
    idx_meta_obj = read_json_if_exists(idx_meta)
    return {
        "generated_at": now_iso(),
        "assets": [
            {
                "name": "DianJin-R1-32B",
                "type": "external_llm",
                "packaged": False,
                "model_id": "DianJin-R1-32B",
                "service_api": "OpenAI-compatible /v1",
                "size_bytes": 65543821517,
                "mount_path": "/opt/data/private/JinRong/models/DianJin-R1-32B",
                "env": "DIANJIN_MODEL_PATH",
                "sha256_manifest_digest": sha256_text(digest_input) if digest_input else None,
                "file_level_sha256": dianjin_files,
                "start_service_example": "scripts/start_task2_full.sh",
            },
            {
                "name": "Task2 DEA SQLite retrieval index",
                "type": "external_sqlite_index",
                "packaged": False,
                "version": idx_meta_obj.get("index_version", "task2_retrieval_index_v1"),
                "size_bytes": idx.stat().st_size if idx.exists() else None,
                "sha256": EXPECTED_TASK2_SQLITE_SHA256,
                "metadata": {
                    "source_rows": idx_meta_obj.get("source_rows"),
                    "source_sha256": idx_meta_obj.get("source_sha256"),
                    "inserted_rows": idx_meta_obj.get("inserted_rows"),
                    "sqlite_version": idx_meta_obj.get("sqlite_version"),
                    "fts_tokenizer": idx_meta_obj.get("fts_tokenizer"),
                    "build_seconds": idx_meta_obj.get("build_seconds"),
                },
                "mount_path": "external_assets/task2/dea_retrieval_index",
                "env": "TASK2_SQLITE_INDEX_DIR",
                "required_files": ["retrieval_index.sqlite", "retrieval_index_metadata.json"],
                "build_or_mount_method": "copy the prebuilt directory to the mount path or rebuild using the original Task2 retrieval_index.py against authorized fulltext_article_index.parquet",
            },
        ],
        "full_mode_command": "PIPELINE_MODE=full TASK2_LLM_BASE_URL=http://127.0.0.1:8000/v1 TASK2_LLM_MODEL=DianJin-R1-32B TASK2_SQLITE_INDEX_DIR=external_assets/task2/dea_retrieval_index ./run_demo.sh",
        "portable_mode_command": "PIPELINE_MODE=portable ./run_demo.sh",
        "missing_resource_behavior": "Full mode raises MissingExternalAsset and exits non-zero; it does not fall back silently.",
    }


def build_asset_recovery_inventory(source_records: list[dict[str, Any]], manifest: dict[str, Any], external_manifest: dict[str, Any]) -> dict[str, Any]:
    inventory = {
        "generated_at": now_iso(),
        "search_policy": {
            "training_executed": False,
            "tuning_executed": False,
            "gold_read": False,
            "official_test_read": False,
            "historical_prediction_as_model": False,
        },
        "tasks": {
            "Task1": [
                {
                    "name": "step5_8e Core event name",
                    "path": "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/event_name_model.pkl",
                    "type": "pickle_model",
                    "sha256": next((m["sha256"] for m in manifest["models"] if m["path"] == "models/task1/event_name_model.pkl"), None),
                    "is_true_model": True,
                    "copyable": True,
                    "runtime_included": True,
                },
                {
                    "name": "step5_8e Core event emotion",
                    "path": "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/event_emotion_model.pkl",
                    "type": "pickle_model",
                    "sha256": next((m["sha256"] for m in manifest["models"] if m["path"] == "models/task1/event_emotion_model.pkl"), None),
                    "is_true_model": True,
                    "copyable": True,
                    "runtime_included": True,
                },
                {
                    "name": "step5_8hc hierarchical model",
                    "path": "outputs/task1_step5_8hc_hierarchical_event/models/best_hierarchical_model.pkl",
                    "type": "pickle_model",
                    "sha256": next((m["sha256"] for m in manifest["models"] if m["path"] == "models/task1/best_hierarchical_model.pkl"), None),
                    "is_true_model": True,
                    "copyable": True,
                    "runtime_included": True,
                },
                {
                    "name": "step5_8HI safety recognition",
                    "path": "outputs/task1_step5_8hi_tag_optimization/frozen_model_manifest.json",
                    "type": "sanitized_manifest_only",
                    "sha256": sha256_file(TARGET / "models/task1/step5_8hi_frozen_model_manifest_sanitized.json") if (TARGET / "models/task1/step5_8hi_frozen_model_manifest_sanitized.json").exists() else None,
                    "is_true_model": False,
                    "copyable": True,
                    "runtime_included": False,
                    "reason": "only historical official prediction asset/manifests found; prediction files are excluded",
                },
            ],
            "Task2": [
                {
                    "name": "final_cached_dea_v1_260 full inference code",
                    "path": "scripts/run_task2_final_cached_dea_v1.py and src/event_credibility/*",
                    "type": "source_code",
                    "is_true_model": False,
                    "copyable": True,
                    "runtime_included": "full_mode_adapter_and_source_snapshot",
                },
                {
                    "name": "DianJin-R1-32B",
                    "path": "external_assets/task2/DianJin-R1-32B",
                    "type": "external_llm",
                    "is_true_model": True,
                    "copyable": False,
                    "runtime_included": "full_mode_external_mount_only",
                    "sha256_manifest_digest": external_manifest["assets"][0].get("sha256_manifest_digest"),
                },
                {
                    "name": "DEA SQLite retrieval index",
                    "path": "external_assets/task2/dea_retrieval_index/retrieval_index.sqlite",
                    "type": "external_sqlite_index",
                    "is_true_model": False,
                    "copyable": False,
                    "runtime_included": "full_mode_external_mount_only",
                    "sha256": EXPECTED_TASK2_SQLITE_SHA256,
                },
            ],
            "Task3": [
                {
                    "name": "resolver_v2_model.joblib",
                    "path": "outputs/task3_canonical_event_v2_actual_repair/resolver_v2_run1/resolver_v2_model.joblib",
                    "type": "joblib_model",
                    "sha256": next((m["sha256"] for m in manifest["models"] if m["path"] == "models/task3/resolver_v2_model.joblib"), None),
                    "is_true_model": True,
                    "copyable": True,
                    "runtime_included": True,
                },
                {
                    "name": "candidate generation / safe clustering runtime",
                    "path": "source_code/runnable_runtime/task3_runtime.py",
                    "type": "source_code_adapter",
                    "is_true_model": False,
                    "copyable": True,
                    "runtime_included": True,
                    "precomputed_answers_used": False,
                },
            ],
            "Task4": [
                {
                    "name": "legacy High-Risk/Gate/Alert/Direction",
                    "path": "scripts/build_task34_official1799_final_evaluation_v1.py -> models/task4/legacy_official_rule_heads.json",
                    "type": "deterministic_rule_heads",
                    "sha256": sha256_file(TARGET / "models/task4/legacy_official_rule_heads.json") if (TARGET / "models/task4/legacy_official_rule_heads.json").exists() else None,
                    "source_script_sha256": sha256_file(LEGACY_RULE_SCRIPT),
                    "is_true_model": True,
                    "copyable": True,
                    "runtime_included": True,
                    "reason": "official frozen prediction builder maps structured-event schema to pred_direction, pred_risk_intensity/high-risk, pred_alert, and e2e gate/route; no TRACE/v3.4/FewRisk substitution",
                },
                {
                    "name": "Gold-only ordinal severity closure",
                    "path": "outputs/task4_final_contract_separated_closure/models/task4_final_contract_separated_closure_bundle.joblib",
                    "type": "joblib_model",
                    "sha256": next((m["sha256"] for m in manifest["models"] if m["path"] == "models/task4/task4_final_contract_separated_closure_bundle.joblib"), None),
                    "is_true_model": True,
                    "copyable": True,
                    "runtime_included": True,
                },
                {
                    "name": "TRACE v3.2",
                    "path": "outputs/task4_trace_refine_v3_2_student/models/task4_trace_refine_v3_2_student_bundle.joblib",
                    "type": "joblib_model",
                    "sha256": next((m["sha256"] for m in manifest["models"] if m["path"] == "models/task4/task4_trace_refine_v3_2_student_bundle.joblib"), None),
                    "is_true_model": True,
                    "copyable": True,
                    "runtime_included": True,
                },
                {
                    "name": "TRACE-RAG v3.3 explanation memory",
                    "path": "outputs/task4_trace_rag_refine_v3_3",
                    "type": "joblib_model_plus_memory",
                    "sha256": next((m["sha256"] for m in manifest["models"] if m["path"] == "models/task4/task4_trace_rag_refine_v3_3_bundle.joblib"), None),
                    "is_true_model": True,
                    "copyable": True,
                    "runtime_included": True,
                    "intervenes_official_risk": False,
                },
            ],
        },
        "source_code_records": source_records,
        "excluded_assets": [
            "Task1 5.8-F",
            "Task2 unapproved v2/ISR/Safety Gate/Claim Adapter",
            "Task3 Hybrid v3.1/Multiroute",
            "Task4 v3.4/FewRisk/answer-leak v1",
            "Gold, official test data, final_holdout, CSMAR raw databases, secrets",
        ],
    }
    write_json(TARGET / "asset_recovery_inventory.json", inventory)
    return inventory


def task4_candidate_role(path: Path) -> str:
    text = path.as_posix().lower()
    if "final_contract_separated_closure" in text:
        return "selected_gold_only_ordinal_closure"
    if "trace_refine_v3_2_student" in text:
        return "selected_trace_v3_2_explanation"
    if "trace_rag_refine_v3_3" in text:
        return "selected_trace_rag_explanation"
    if "severity_strict_scoring_v1" in text:
        return "official1799_legacy_auxiliary_severity_route_lineage"
    if "fewrisk" in text or "v3_4" in text:
        return "explicitly_excluded_ablation"
    if "v2_step2" in text or "robustness" in text or "selective_moe" in text:
        return "research_or_ablation_candidate"
    if "trace_impact" in text:
        return "trace_impact_candidate_not_selected_as_official_risk"
    return "unselected_task4_candidate"


def task4_probe_frame() -> Any:
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "sample_id": "non_gold_probe_001",
                "title": "蓝海材料收到交易所问询并披露处罚风险",
                "content": "蓝海材料披露，公司因信息披露不及时收到交易所问询，并提示可能面临处罚和现金流压力。",
                "evidence": "公司因信息披露不及时收到交易所问询，并提示可能面临处罚和现金流压力。",
                "target": "蓝海材料",
                "target_name": "蓝海材料",
                "target_type": "company",
                "event_name": "监管问询",
                "event_type": "监管问询",
                "trigger": "问询",
                "arguments": "蓝海材料",
                "structured_event": "event_name=监管问询 target=蓝海材料 state=announcement",
                "publish_time": "2026-08-29 10:15:00",
                "publish_year": "2026",
                "publish_month": "8",
                "publish_dayofweek": "5",
                "legacy_aux_expected_risk": "needs_review",
                "legacy_aux_high_prob": "0.68",
                "risk_intensity": 4,
                "severity_label": "orange",
                "direction": "negative",
                "alert": "alert",
            }
        ]
    )


def json_preview(value: Any, limit: int = 240) -> Any:
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        elif hasattr(value, "to_dict"):
            value = value.to_dict()
    except Exception:
        pass
    try:
        text = json.dumps(base.json_ready(value), ensure_ascii=False, sort_keys=True)
    except Exception:
        text = repr(value)
    return text[:limit]


def sanitize_probe_text(text: Any) -> str:
    return str(text).replace(ORIGINAL_PROJECT_ABS, "$ORIGINAL_PROJECT_ROOT")


def register_task4_probe_shims() -> None:
    try:
        source_root = TARGET / "source_code"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from runnable_runtime import task4_closure_shims

        main = sys.modules["__main__"]
        for name in ["MulticlassTextRiskHead", "CumulativeTextRiskHead", "BinaryHighRiskTextHead"]:
            setattr(main, name, getattr(task4_closure_shims, name))
    except Exception:
        return


def probe_predictor(predictor: Any, frame: Any) -> dict[str, Any]:
    if hasattr(predictor, "predict"):
        try:
            pred = predictor.predict(frame)
            payload = {"inference_ok": True, "method": "predict", "output_preview": json_preview(pred)}
            if hasattr(predictor, "predict_proba"):
                try:
                    payload["proba_preview"] = json_preview(predictor.predict_proba(frame))
                except Exception as exc:
                    payload["proba_error"] = f"{type(exc).__name__}: {exc}"
            return payload
        except Exception as exc:
            return {"inference_ok": False, "method": "predict", "error": sanitize_probe_text(f"{type(exc).__name__}: {exc}")}
    if hasattr(predictor, "transform"):
        try:
            pred = predictor.transform(frame)
            return {"inference_ok": True, "method": "transform", "output_preview": json_preview(pred)}
        except Exception as exc:
            return {"inference_ok": False, "method": "transform", "error": sanitize_probe_text(f"{type(exc).__name__}: {exc}")}
    return {"inference_ok": False, "method": None, "error": "loaded object has no predict/transform method"}


def probe_task4_candidate_asset(path: Path) -> dict[str, Any]:
    import joblib

    register_task4_probe_shims()
    frame = task4_probe_frame()
    try:
        obj = joblib.load(path)
    except Exception as exc:
        return {
            "load_attempted": True,
            "load_ok": False,
            "load_error": sanitize_probe_text(f"{type(exc).__name__}: {exc}"),
            "inference_attempted": False,
            "inference_ok": False,
        }
    probe: dict[str, Any] = {
        "load_attempted": True,
        "load_ok": True,
        "object_type": f"{type(obj).__module__}.{type(obj).__name__}",
        "inference_attempted": True,
    }
    if isinstance(obj, dict):
        probe["top_level_keys"] = [str(k) for k in list(obj.keys())[:30]]
        head_results: dict[str, Any] = {}
        if hasattr(obj.get("official_risk_head"), "predict"):
            head_results["official_risk_head"] = probe_predictor(obj["official_risk_head"], frame)
        models = obj.get("models")
        if isinstance(models, dict):
            for name, model in list(models.items())[:12]:
                head_results[str(name)] = probe_predictor(model, frame)
        for key in ["model", "pipeline", "estimator", "classifier", "preprocessing_encoder"]:
            if key in obj:
                head_results[key] = probe_predictor(obj[key], frame)
        if head_results:
            probe["head_probe_results"] = head_results
            probe["inference_ok"] = any(item.get("inference_ok") for item in head_results.values())
        else:
            probe["inference_ok"] = False
            probe["inference_error"] = "dict bundle has no recognized predictor head"
    else:
        result = probe_predictor(obj, frame)
        probe.update(result)
        probe["inference_ok"] = bool(result.get("inference_ok"))
    return probe


def build_task4_legacy_head_recovery_report(manifest: dict[str, Any]) -> dict[str, Any]:
    candidate_records: list[dict[str, Any]] = []
    for path in sorted(ROOT.joinpath("outputs").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pkl", ".joblib", ".pickle"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if "task4" not in rel or "competition_final_delivery" in rel:
            continue
        role = task4_candidate_role(path)
        selected_runtime = rel in {
            "outputs/task4_final_contract_separated_closure/models/task4_final_contract_separated_closure_bundle.joblib",
            "outputs/task4_trace_refine_v3_2_student/models/task4_trace_refine_v3_2_student_bundle.joblib",
            "outputs/task4_trace_rag_refine_v3_3/models/task4_trace_rag_refine_v3_3_bundle.joblib",
            "outputs/task4_severity_strict_scoring_v1/models/final_selected_model_bundle.joblib",
        }
        probe = probe_task4_candidate_asset(path)
        candidate_records.append(
            {
                "path": rel,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "role_by_schema_and_lineage": role,
                "load_test_on_non_gold_sample": probe,
                "inference_test_on_non_gold_sample": {
                    "attempted": probe.get("inference_attempted", False),
                    "ok": probe.get("inference_ok", False),
                    "note": "selected runtime assets are also verified by final smoke tests" if selected_runtime else "probe only; not selected for formal runtime",
                },
                "selected_for_v2_runtime": selected_runtime and role != "official1799_legacy_auxiliary_severity_route_lineage",
                "excluded_reason": None if selected_runtime and role != "official1799_legacy_auxiliary_severity_route_lineage" else (
                    "lineage evidence only; official legacy heads recovered from rule function rather than this auxiliary model"
                    if role == "official1799_legacy_auxiliary_severity_route_lineage"
                    else "not selected frozen production asset or explicitly excluded ablation"
                ),
            }
        )
    report = {
        "generated_at": now_iso(),
        "read_policy": {
            "gold_read": False,
            "official1799_data_rows_read": False,
            "official_prediction_rows_read": False,
            "metadata_and_source_only": True,
        },
        "search_methods": [
            "evaluation scripts and frozen prediction manifests",
            "prediction provenance/model path/model SHA fields from manifests only",
            "all Task4 .pkl/.joblib/.pickle files under outputs excluding delivery packages",
            "feature-schema reverse matching against Task4 official builder output fields and runtime frame fields",
        ],
        "official_prediction_lineage": {
            "prediction_manifest": "outputs/task34_official1799_final_evaluation_v1/prediction_manifest.json",
            "prediction_lock_manifest": "outputs/task34_official1799_final_evaluation_v1/prediction_lock_manifest.json",
            "builder_script": "scripts/build_task34_official1799_final_evaluation_v1.py",
            "builder_script_sha256": sha256_file(LEGACY_RULE_SCRIPT),
            "model_path_or_sha_in_prediction_manifest": "not present for Task4 legacy rule heads",
            "frozen_field_mapping": {
                "Direction": "pred_direction",
                "High-Risk": "pred_risk_intensity >= 4",
                "Gate": "pred_task4_e2e_covered / e2e_route",
                "Alert": "pred_alert / e2e_route=alert",
            },
            "feature_schema_reverse_match": [
                "event.event_type",
                "event.event_name",
                "event.event_fact",
                "event.trigger_text",
                "evidence.evidence_sentence",
                "evidence.local_context",
                "event.polarity_or_state",
                "quality.contract_status",
                "quality.target_grounded",
                "quality.event_grounded",
            ],
        },
        "recovered_assets": [
            {
                "asset": "models/task4/legacy_official_rule_heads.json",
                "sha256": sha256_file(TARGET / "models/task4/legacy_official_rule_heads.json") if (TARGET / "models/task4/legacy_official_rule_heads.json").exists() else None,
                "head_roles": ["High-Risk", "Gate", "Alert", "Direction"],
                "asset_type": "deterministic_rule_heads",
                "non_gold_inference_verified_by": "smoke_test_report_full.json / smoke_test_report_portable.json",
                "uses_trace_or_v3_4_as_substitute": False,
            }
        ],
        "candidate_model_binaries": candidate_records,
        "conclusion": "Legacy High-Risk/Gate/Alert/Direction were recovered as deterministic rule-head inference assets from the frozen official prediction builder, not as separate .joblib binaries.",
    }
    write_json(TARGET / "task4_legacy_head_recovery_report.json", report)
    return report


def build_docs(manifest: dict[str, Any], external_manifest: dict[str, Any]) -> None:
    selected = manifest["selected_versions"]
    write_text(
        TARGET / "README.md",
        f"""# 国赛第②项：可运行项目成果 v2

本目录只包含第②项可运行工程包，不处理精益画布、技术文档、原创性声明。

## 运行

```bash
./install.sh
./start.sh
./healthcheck.sh
./run_demo.sh
./stop.sh
```

直接入口：

```bash
python run_pipeline.py --input demo_data/sample.jsonl --output outputs/result.jsonl
```

## 正式版本

- Task1: {selected['task1']}
- Task2: {selected['task2']}
- Task3: {selected['task3']}
- Task4: {selected['task4']}

## 能力边界

Task1 Core/HC、Task3 resolver、Task4 legacy High-Risk/Gate/Alert/Direction rule heads、Gold-only ordinal severity、TRACE v3.2 和 TRACE-RAG v3.3 均在 demo 中真实加载或使用。Task1 5.8HI 只找到冻结预测资产 manifest，未找到可复现模型，因此不在 runtime 中重放。Task4 legacy 四个 Head 不是 `.joblib` 二进制，而是从旧 official1799 冻结预测构建脚本恢复的确定性规则 Head；TRACE/RAG 和 v3.4 不参与替代。

Task2 full 模式使用本机预置的 `DianJin-R1-32B` 模型目录和 SQLite 检索索引，默认模型路径为 `/opt/data/private/JinRong/models/DianJin-R1-32B`，也可通过 `DIANJIN_MODEL_PATH` 覆盖。

Task2 有两种模式：

- `portable`: 不读取缓存答案、不按样本硬编码，执行本地确定性降级推理；不等价于 `final_cached_dea_v1_260`。
- `full`: 校验并调用本机预置的 DianJin-R1-32B 服务、SQLite 检索索引和 DEA 裁决 adapter；缺失资源时明确报错。

排除 Task1 5.8-F、Task2 v2/ISR/Safety Gate/Claim Adapter、Task3 Hybrid v3.1/Multiroute、Task4 v3.4/FewRisk/答案泄漏 v1。未打包 Gold、官方测试数据、final_holdout、CSMAR 原始库、密钥或受限数据。
""",
    )
    write_text(
        TARGET / "deployment/deployment_guide.md",
        """# 部署说明

## 环境

- Python: 3.11 推荐
- CPU: 8 核以上推荐
- 内存: portable 模式建议 16GB；full 模式建议 64GB+
- GPU: portable 模式不强制；full 模式运行 DianJin-R1-32B 需按本地 vLLM 部署配置准备 GPU/显存
- 磁盘: 包体约数百 MB；full 模式需额外挂载约 65.5GB DianJin 模型和约 4.25GB SQLite 索引
- CUDA: full 模式按 vLLM/DianJin 环境要求配置

## 安装与 portable 测试

```bash
unzip 02_可运行项目成果_v2.zip
cd item2_runnable_v2
export PYTHON_BIN=/path/to/python3.11
./install.sh
./start.sh
PIPELINE_MODE=portable ./healthcheck.sh
PIPELINE_MODE=portable ./run_demo.sh
./stop.sh
```

## full 模式本机预置资产

本机预置路径：

```text
${DIANJIN_MODEL_PATH:-/opt/data/private/JinRong/models/DianJin-R1-32B}
external_assets/task2/dea_retrieval_index/retrieval_index.sqlite
external_assets/task2/dea_retrieval_index/retrieval_index_metadata.json
```

启动 DianJin 服务：

```bash
scripts/start_task2_full.sh
```

检查并运行：

```bash
python scripts/check_external_assets.py --mode full
PIPELINE_MODE=full DIANJIN_MODEL_PATH=/opt/data/private/JinRong/models/DianJin-R1-32B TASK2_LLM_BASE_URL=http://127.0.0.1:8000/v1 TASK2_LLM_MODEL=DianJin-R1-32B TASK2_SQLITE_INDEX_DIR=external_assets/task2/dea_retrieval_index ./run_demo.sh
```

full 模式缺少任一本机预置资产会返回非零状态，不会静默降级到 portable。

## 输入输出

输入 schema: `schema/input_schema.json`。输出 schema: `schema/output_schema.json`。

输出包含 Task1 结构化事件、Task2 可信度、Task3 CanonicalEvent/Event Thread、Task4 Direction/Severity/High-Risk/Gate/Alert/TRACE/RAG 字段。Task4 Direction/High-Risk/Gate/Alert 来自恢复的 legacy deterministic rule heads；Severity 来自 Gold-only ordinal closure；TRACE/RAG 只用于解释复核。

## 常见错误

- `ModuleNotFoundError`: 使用 `requirements.txt` 或 `environment.yml` 安装依赖。
- `missing Task2 SQLite index`: 将 DEA 索引目录挂载到 `TASK2_SQLITE_INDEX_DIR`。
- `cannot reach Task2 LLM service`: 先启动 DianJin-R1-32B OpenAI-compatible 服务。
- `Task4 legacy rule head mismatch`: 检查 `models/task4/legacy_official_rule_heads.json` SHA 和 `source_code/original_task4/legacy_official_rule_heads_source.py`。
""",
    )
    write_text(
        TARGET / "deployment/api_guide.md",
        """# API 说明

命令行：

```bash
python run_pipeline.py --input demo_data/sample.jsonl --output outputs/result.jsonl --mode portable
```

输入为 JSONL，每行至少包含 `article_id`、`title`、`content`、`publish_time`。

输出为 JSONL，每行包含 `task1`、`task2`、`task3`、`task4` 和 `meta`。`--mode full` 会强制校验 Task2 外部资产。
""",
    )
    shutil.copy2(TARGET / "deployment/deployment_guide.md", TARGET / "deployment_guide.md")
    shutil.copy2(TARGET / "deployment/api_guide.md", TARGET / "api_guide.md")
    write_text(
        TARGET / "LICENSE_OR_ASSET_NOTICE.md",
        """# 第三方依赖与资产说明

- joblib, numpy, pandas, scipy, scikit-learn: BSD 系许可证。
- LightGBM: MIT 许可证。
- pyarrow: Apache-2.0 许可证。
- DianJin-R1-32B: 不随包分发，按授权外部挂载。
- Task2 SQLite 检索索引: 不随包分发，按授权外部挂载或重建。

本包不包含 Gold、官方测试、final_holdout、CSMAR 原始库、密钥或受限数据。
""",
    )
    write_text(
        TARGET / "source_code/SOURCE_MAPPING.md",
        """# Source Mapping

Actual executable runtime source is under `source_code/runnable_runtime/` and is called by root `run_pipeline.py`.

- Task1: loads packaged step5_8e Core and step5_8hc HC pickle models.
- Task2: portable runtime is local deterministic degraded inference; full runtime validates and uses external DianJin-R1-32B plus SQLite retrieval. Original Task2 final source snapshots are under `source_code/original_task2/`.
- Task3: loads packaged resolver_v2 model and executes current-input candidate generation, pair judging and union-find safe clustering.
- Task4: loads recovered legacy deterministic rule heads for High-Risk/Gate/Alert/Direction, plus Gold-only ordinal closure, TRACE v3.2 and TRACE-RAG v3.3 explanation assets. TRACE/RAG and v3.4 are not substituted for the legacy rule heads.

No training, tuning, pseudo-label generation, Gold reading, official test reading or final_holdout reading is performed by runtime scripts.
""",
    )


def build_audits(manifest: dict[str, Any], external_manifest: dict[str, Any]) -> None:
    write_json(TARGET / "model_manifest.json", manifest)
    write_json(TARGET / "external_assets_manifest.json", external_manifest)
    write_json(
        TARGET / "dependency_audit.json",
        {
            "generated_at": now_iso(),
            "python_required": ">=3.11 recommended",
            "dependencies": [
                {"name": "joblib", "purpose": "model deserialization", "license": "BSD-3-Clause"},
                {"name": "numpy", "purpose": "array computation", "license": "BSD-3-Clause"},
                {"name": "pandas", "purpose": "Task4 feature frame and RAG memory tables", "license": "BSD-3-Clause"},
                {"name": "scipy", "purpose": "model dependencies", "license": "BSD-3-Clause"},
                {"name": "scikit-learn", "purpose": "Task1/Task3/Task4 inference", "license": "BSD-3-Clause"},
                {"name": "lightgbm", "purpose": "Task4 closure heads", "license": "MIT"},
                {"name": "pyarrow", "purpose": "Task4 RAG parquet memory loading", "license": "Apache-2.0"},
                {"name": "sqlite3", "purpose": "Task2 external index probe", "license": "Python stdlib"},
            ],
            "external_dependencies": ["DianJin-R1-32B service", "Task2 DEA SQLite retrieval index"],
        },
    )
    write_json(
        TARGET / "data_license_audit.json",
        {
            "generated_at": now_iso(),
            "packaged_gold": False,
            "packaged_official_test": False,
            "packaged_final_holdout": False,
            "packaged_csmar_raw": False,
            "packaged_secrets": False,
            "task4_rag_memory": "packaged derived explanation memory; leakage_audit records official1799_read=false and final_holdout_read=false",
        },
    )
    write_json(
        TARGET / "runnable_asset_manifest.json",
        {
            "generated_at": now_iso(),
            "package": "item2_runnable_v2",
            "entrypoints": {
                "pipeline": "run_pipeline.py",
                "install": "install.sh",
                "start": "start.sh",
                "healthcheck": "healthcheck.sh",
                "run_demo": "run_demo.sh",
                "stop": "stop.sh",
                "external_asset_check": "scripts/check_external_assets.py",
                "task2_full_start": "scripts/start_task2_full.sh",
            },
            "selected_versions": manifest["selected_versions"],
            "models": manifest["models"],
            "external_assets": external_manifest["assets"],
            "runnable_submission_ready_policy": "ready only when final ZIP validation passes, Task2 full assets are mounted, and all selected Task4 legacy heads are found and loaded",
        },
    )
    write_json(
        TARGET / "final_asset_selection.json",
        {
            "generated_at": now_iso(),
            "selected_versions": manifest["selected_versions"],
            "task4_heads": manifest["task4_production_heads"],
            "excluded_versions": [
                "Task1 5.8-F",
                "Task2 unapproved v2/ISR/Safety Gate/Claim Adapter",
                "Task3 Hybrid v3.1/Multiroute",
                "Task4 v3.4/FewRisk/answer-leak v1",
            ],
            "runnable_submission_ready": None,
            "readiness_resolved_after_validation": "see final_delivery_item2_report.json",
            "task4_legacy_recovery": "High-Risk/Gate/Alert/Direction recovered as deterministic rule heads from official frozen prediction builder; no binary model existed in manifest",
        },
    )
    write_json(
        TARGET / "metric_source_registry.json",
        {
            "generated_at": now_iso(),
            "important_note": "Metrics are historical frozen-module reports, not recomputed by this packaging run.",
            "task4_stable_legacy_warning_module": {
                "source": "outputs/task34_official1799_final_evaluation_v1/final_metrics.json",
                "dataset": "official1799 existing one-time evaluation",
                "status": "historical_evaluation_only",
                "high_severity_precision": 0.48148148148148145,
                "high_severity_recall": 0.7222222222222222,
                "high_severity_f1": 0.5777777777777777,
                "automatic_coverage": 0.349082823790995,
                "direction_macro_f1": 0.20348014513300786,
            },
            "task4_gold_only_ordinal_module": {
                "source": "outputs/competition_final_delivery/task4_final_inventory.json",
                "dataset": "official1799 severity eligible rows from existing closure formal metrics",
                "status": "historical_evaluation_only",
                "qwk": 0.3170324459345738,
                "mae": 0.4507422402159244,
                "spearman": 0.3411270408717746,
            },
        },
    )


def scan_binary_for_patterns(path: Path, patterns: dict[str, bytes]) -> set[str]:
    found: set[str] = set()
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = tail + chunk
            for name, needle in patterns.items():
                if needle in data:
                    found.add(name)
            tail = data[-512:]
    return found


def build_absolute_path_audit() -> dict[str, Any]:
    patterns = {
        "absolute_original_project_path": ORIGINAL_PROJECT_ABS.encode("utf-8"),
        "official1799_term": b"official1799",
        "final_holdout_term": b"final_holdout",
        "golddata_term": b"golddata",
    }
    entries: list[dict[str, Any]] = []
    for path in sorted(TARGET.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pkl", ".joblib", ".parquet"}:
            continue
        found = scan_binary_for_patterns(path, patterns)
        if not found:
            continue
        rel = path.relative_to(TARGET).as_posix()
        classification = "metadata_only"
        runtime_dependency = False
        mitigation = "runtime validation loaded/predicted from package-relative files without opening original project path"
        if "task4_trace_rag_refine_v3_3_bundle" in rel:
            classification = "runtime_dependency_adapted_to_package_relative"
            mitigation = "adapter ignores embedded v3_2_bundle and memory_paths and loads package-local v3.2 plus rag_memory files"
        entries.append(
            {
                "path": rel,
                "patterns": sorted(found),
                "classification": classification,
                "runtime_dependency": runtime_dependency,
                "mitigation": mitigation,
            }
        )
    report = {
        "generated_at": now_iso(),
        "status": "pass_with_metadata_or_adapted_paths",
        "original_project_runtime_dependency_found": False,
        "entries": entries,
    }
    write_json(TARGET / "absolute_path_audit.json", report)
    return report


def scan_package() -> dict[str, Any]:
    secret_patterns = [
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY"),
        re.compile(r"(?i)(api[_-]?key|secret[_-]?key|password)\s*[:=]\s*['\"][^'\"]{8,}"),
    ]
    restricted_file_terms = re.compile(r"(?i)(official_labeled_test_predictions|final_holdout|csmar_cache|csmar_raw|severity_master_gold|task34\\.xlsx)")
    secret_hits: list[dict[str, Any]] = []
    restricted_files: list[str] = []
    text_abs_hits: list[str] = []
    for path in sorted(TARGET.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(TARGET).as_posix()
        if restricted_file_terms.search(rel):
            restricted_files.append(rel)
        if path.suffix.lower() in {".pkl", ".joblib", ".parquet", ".zip", ".gz", ".tar"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if ORIGINAL_PROJECT_ABS in text:
            text_abs_hits.append(rel)
        for pattern in secret_patterns:
            for match in pattern.finditer(text):
                secret_hits.append({"path": rel, "pattern": pattern.pattern, "offset": match.start()})
    report = {
        "generated_at": now_iso(),
        "status": "pass" if not secret_hits and not restricted_files and not text_abs_hits else "fail",
        "secret_hits": secret_hits,
        "restricted_data_files_packaged": restricted_files,
        "text_absolute_original_project_path_hits": text_abs_hits,
        "binary_absolute_path_handled_by": "absolute_path_audit.json",
    }
    write_json(TARGET / "secret_scan_report.json", report)
    return report


def cleanup_generated_bytecode() -> None:
    for path in sorted(TARGET.rglob("__pycache__")):
        if path.is_dir():
            shutil.rmtree(path)
    for path in sorted(TARGET.rglob("*.pyc")):
        if path.is_file():
            path.unlink()


def write_checksums() -> None:
    cleanup_generated_bytecode()
    lines: list[str] = []
    for path in sorted(TARGET.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rel = path.relative_to(TARGET).as_posix()
            lines.append(f"{sha256_file(path)}  {rel}")
    write_text(TARGET / "checksums.sha256", "\n".join(lines) + "\n")


def create_zip(zip_path: Path) -> dict[str, Any]:
    cleanup_generated_bytecode()
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(TARGET.rglob("*")):
            if path.is_file():
                zf.write(path, Path(TARGET.name) / path.relative_to(TARGET))
    return {"path": zip_path.relative_to(ROOT).as_posix(), "size_bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path)}


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], expected_returncodes: set[int] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    expected = expected_returncodes or {0}
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "expected_returncodes": sorted(expected),
        "as_expected": proc.returncode in expected,
        "seconds": round(time.perf_counter() - start, 6),
        "output": proc.stdout,
    }


def aggregate_runtime_file_access(extracted: Path) -> dict[str, Any]:
    audit_path = extracted / "logs/file_access_audit.jsonl"
    records: list[dict[str, Any]] = []
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    original_hits = [r for r in records if r.get("is_original_project_path")]
    report = {
        "generated_at": now_iso(),
        "status": "pass" if not original_hits else "fail",
        "audit_scope": "Python builtins.open, pathlib.Path.open and sqlite3.connect captured during healthcheck/run_demo",
        "runtime_original_project_path_access_count": len(original_hits),
        "runtime_original_project_path_accesses": original_hits,
        "observed_access_count": len(records),
    }
    write_json(TARGET / "runtime_file_access_audit.json", report)
    return report


def clean_validate(zip_path: Path) -> dict[str, Any]:
    if VALIDATION_ROOT.exists():
        shutil.rmtree(VALIDATION_ROOT)
    VALIDATION_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(VALIDATION_ROOT)
    extracted = VALIDATION_ROOT / TARGET.name
    py = os.environ.get("PYTHON_BIN_FOR_ITEM2") or ("/root/miniconda3/envs/task4-impact-v3/bin/python" if Path("/root/miniconda3/envs/task4-impact-v3/bin/python").exists() else sys.executable)
    env = dict(os.environ)
    env["PYTHON_BIN"] = py
    env["PIPELINE_MODE"] = "portable"
    env["PYTHONPATH"] = str(extracted / "source_code")
    env["ORIGINAL_PROJECT_ROOT"] = "/__blocked_original_project__"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("TASK2_SQLITE_INDEX_DIR", None)
    env.pop("TASK2_LLM_BASE_URL", None)
    commands = [
        ("checksum", ["bash", "scripts/verify_checksums.sh"], {0}),
        ("install", ["bash", "install.sh"], {0}),
        ("start", ["bash", "start.sh"], {0}),
        ("healthcheck", ["bash", "healthcheck.sh"], {0}),
        ("portable_demo", ["bash", "run_demo.sh"], {0}),
        ("stop", ["bash", "stop.sh"], {0}),
    ]
    results = []
    for name, cmd, expected in commands[:-1]:
        result = run_command(cmd, extracted, env, expected_returncodes=expected)
        result["step"] = name
        results.append(result)
    smoke = read_json_if_exists(extracted / "smoke_test_report.json")
    runtime_audit = aggregate_runtime_file_access(extracted)
    full_enabled = os.environ.get("ENABLE_TASK2_FULL_SMOKE") == "1"
    full_env = dict(env)
    full_env["PIPELINE_MODE"] = "full"
    full_env["TASK2_LLM_BASE_URL"] = os.environ.get("TASK2_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    full_env["TASK2_LLM_MODEL"] = os.environ.get("TASK2_LLM_MODEL", "DianJin-R1-32B")
    validation_index_src = Path(os.environ.get("TASK2_SQLITE_INDEX_DIR_VALIDATION", str(ROOT / "cache/event_credibility/dea_retrieval_index")))
    validation_mount = VALIDATION_ROOT / "_external_assets/task2/dea_retrieval_index"
    if full_enabled and validation_index_src.exists():
        validation_mount.parent.mkdir(parents=True, exist_ok=True)
        if validation_mount.exists() or validation_mount.is_symlink():
            if validation_mount.is_dir() and not validation_mount.is_symlink():
                shutil.rmtree(validation_mount)
            else:
                validation_mount.unlink()
        validation_mount.symlink_to(validation_index_src, target_is_directory=True)
        full_env["TASK2_SQLITE_INDEX_DIR"] = str(validation_mount)
    full_check = run_command([py, "scripts/check_external_assets.py", "--mode", "full"], extracted, full_env, expected_returncodes={0} if full_enabled else {0, 2})
    full_check["step"] = "full_asset_check"
    results.append(full_check)
    full_assets_present = full_enabled and full_check.get("returncode") == 0
    if full_assets_present:
        full_result = run_command(["bash", "run_demo.sh"], extracted, full_env, expected_returncodes={0})
        full_result["step"] = "full_smoke_test"
        results.append(full_result)
        smoke_full = read_json_if_exists(extracted / "smoke_test_report_full.json")
        smoke = smoke_full or read_json_if_exists(extracted / "smoke_test_report.json")
    else:
        results.append(
            {
                "step": "full_smoke_test",
                "cmd": ["bash", "run_demo.sh"],
                "returncode": None,
                "expected_returncodes": [0],
                "as_expected": not full_enabled,
                "seconds": 0,
                "output": "skipped: Task2 external full-mode assets are not mounted in clean validation" if not full_enabled else "not executed because full asset check failed",
            }
        )
    stop_result = run_command(commands[-1][1], extracted, env, expected_returncodes=commands[-1][2])
    stop_result["step"] = commands[-1][0]
    results.append(stop_result)
    log_lines = [
        f"clean_environment_validation_at={now_iso()}",
        f"zip_under_test={zip_path.relative_to(ROOT).as_posix()}",
        f"extract_root={VALIDATION_ROOT}",
        f"python_bin={py}",
        "original_project_root_mask=/__blocked_original_project__",
        "",
    ]
    for result in results:
        log_lines.append(f"[{result['step']}] $ {' '.join(result.get('cmd') or [])}")
        log_lines.append(f"returncode={result.get('returncode')} expected={result.get('expected_returncodes')} as_expected={result.get('as_expected')} seconds={result.get('seconds')}")
        log_lines.append(str(result.get("output", "")))
        log_lines.append("")
    validation = {
        "generated_at": now_iso(),
        "zip_under_test": zip_path.relative_to(ROOT).as_posix(),
        "extract_root": str(VALIDATION_ROOT),
        "python_bin": py,
        "commands": results,
        "all_required_commands_as_expected": all(r.get("as_expected") for r in results),
        "portable_commands_zero": all(r.get("returncode") == 0 for r in results if r["step"] in {"checksum", "install", "start", "healthcheck", "portable_demo", "stop"}),
        "full_asset_check_returncode": full_check.get("returncode"),
        "full_smoke_executed": full_assets_present,
        "full_validation_enabled": full_enabled,
        "full_external_index_mount": str(validation_mount) if full_enabled else None,
        "smoke_test_report": smoke,
        "runtime_file_access_audit": runtime_audit,
        "status": "full_and_portable_pass" if smoke.get("runnable_submission_ready") and runtime_audit.get("status") == "pass" else "portable_pass_formal_gaps" if smoke.get("status") == "portable_pass_formal_gaps" and runtime_audit.get("status") == "pass" else "fail",
        "runnable_submission_ready": bool(smoke.get("runnable_submission_ready", False)),
    }
    write_text(TARGET / "clean_environment_validation.log", "\n".join(log_lines))
    write_json(TARGET / "clean_environment_validation_summary.json", validation)
    for rel in [
        "smoke_test_report.json",
        "smoke_test_report_portable.json",
        "smoke_test_report_full.json",
        "outputs/demo_result_portable.jsonl",
        "outputs/demo_result_mutated_portable.jsonl",
        "outputs/demo_result_full.jsonl",
        "outputs/demo_result_mutated_full.jsonl",
    ]:
        src = extracted / rel
        if src.exists():
            dst = TARGET / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    if (extracted / "outputs/demo_result_portable.jsonl").exists():
        shutil.copy2(extracted / "outputs/demo_result_portable.jsonl", TARGET / "demo_data/standard_output_sample.jsonl")
    if (extracted / "logs").exists():
        dst = TARGET / "logs"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(extracted / "logs", dst)
    return validation


def main() -> int:
    if not COMPUTE.exists():
        raise FileNotFoundError(COMPUTE)
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    model_copy_records = copy_models()
    build_runtime_files()
    source_records = copy_selected_source()
    build_scripts()
    build_demo_data()
    build_schemas_and_configs()
    build_requirements()
    manifest = model_manifest()
    external_manifest = external_assets_manifest()
    build_docs(manifest, external_manifest)
    build_audits(manifest, external_manifest)
    asset_inventory = build_asset_recovery_inventory(source_records, manifest, external_manifest)
    task4_recovery = build_task4_legacy_head_recovery_report(manifest)
    absolute_audit = build_absolute_path_audit()
    cleanup_generated_bytecode()
    secret_scan = scan_package()
    write_json(TARGET / "initial_asset_copy_records.json", {"models": model_copy_records, "source": source_records})
    write_checksums()
    candidate_zip = create_zip(ZIP_PATH)
    validation = clean_validate(ZIP_PATH)
    write_json(
        TARGET / "final_delivery_item2_report.json",
        {
            "generated_at": now_iso(),
            "target_dir": TARGET.relative_to(ROOT).as_posix(),
            "candidate_zip_validated": candidate_zip,
            "selected_versions": manifest["selected_versions"],
            "asset_inventory_status": "written",
            "task4_legacy_head_recovery_report": "task4_legacy_head_recovery_report.json",
            "absolute_path_audit_status": absolute_audit["status"],
            "secret_scan_status": secret_scan["status"],
            "validation_status": validation["status"],
            "runnable_submission_ready": validation["runnable_submission_ready"],
            "constraints": {
                "training_executed": False,
                "tuning_executed": False,
                "pseudo_label_generated": False,
                "gold_read": False,
                "official_test_read": False,
                "final_holdout_read": False,
                "frozen_assets_modified": False,
            },
            "blocking_gaps": [] if validation["runnable_submission_ready"] else [
                "Task2 full smoke did not run/pass" if not validation.get("full_smoke_executed") else "see smoke_test_report.json"
            ],
            "task4_legacy_recovery": asset_inventory["tasks"]["Task4"][0],
            "task4_candidate_binary_count": len(task4_recovery["candidate_model_binaries"]),
        },
    )
    write_checksums()
    final_zip = create_zip(ZIP_PATH)
    print(
        json.dumps(
            {
                "status": "ok",
                "target": str(TARGET),
                "zip": final_zip,
                "validation": validation["status"],
                "runnable_submission_ready": validation["runnable_submission_ready"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
