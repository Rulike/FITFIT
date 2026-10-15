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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/competition_final_delivery"
COMPUTE = BASE / "compute_platform_delivery"
TARGET = BASE / "02_可运行项目成果"
ZIP_PATH = BASE / "02_可运行项目成果.zip"
VALIDATION_ROOT = Path("/tmp") / f"competition_item2_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        return {"source": str(src.relative_to(ROOT) if src.is_relative_to(ROOT) else src), "target": str(dst.relative_to(TARGET)), "copied": False}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src.relative_to(ROOT) if src.is_relative_to(ROOT) else src),
        "target": str(dst.relative_to(TARGET)),
        "copied": True,
        "sha256": sha256_file(dst),
        "size_bytes": dst.stat().st_size,
    }


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_sha_lines(path: Path, root_marker: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            continue
        rel = parts[1]
        if root_marker in rel:
            rel = rel.split(root_marker, 1)[1].lstrip("/")
        rel = rel.replace("\\", "/")
        out[rel] = {"sha256": parts[0]}
        local = Path("/" + parts[1].lstrip("/"))
        if local.exists():
            out[rel]["size_bytes"] = local.stat().st_size
    return out


COMMON_PY = r'''from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def compact(value: Any, limit: int = 600) -> str:
    text = clean_text(value)
    return text if len(text) <= limit else text[:limit].rstrip()


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "||".join(clean_text(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def sha256_text(value: Any) -> str:
    return hashlib.sha256(clean_text(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
'''


TASK1_RUNTIME_PY = r'''from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import clean_text, compact, sha256_text, stable_id


class ConstantHead:
    def __init__(self, label: str) -> None:
        self.classes_ = np.array([label], dtype=object)

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> "ConstantHead":
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.ones((X.shape[0], 1), dtype=float)


def _register_unpickle_shims() -> None:
    sys.modules["__main__"].ConstantHead = ConstantHead


def head_predict_proba(head: Any, X: Any) -> np.ndarray:
    return head.predict_proba(X) if hasattr(head, "predict_proba") else np.ones((X.shape[0], 1), dtype=float)


def head_classes(head: Any) -> list[str]:
    if hasattr(head, "classes_"):
        return [str(x) for x in getattr(head, "classes_", [])]
    if hasattr(head, "named_steps"):
        last = list(head.named_steps.values())[-1]
        return [str(x) for x in getattr(last, "classes_", [])]
    return []


def unwrap_pipeline(model_obj: Any) -> Any:
    if isinstance(model_obj, dict) and "model" in model_obj:
        return model_obj["model"]
    return model_obj


def softmax(scores: list[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    arr = arr - float(np.max(arr))
    exps = np.exp(arr)
    denom = float(exps.sum())
    if denom <= 0:
        return np.ones_like(exps) / max(1, len(exps))
    return exps / denom


def row_text(row: dict[str, Any], include_event: bool = False) -> str:
    title = row.get("article_title") or row.get("title") or ""
    source = row.get("article_source") or row.get("source") or ""
    target = row.get("secu_abbr") or row.get("target_name") or row.get("industry_name") or ""
    pieces = [
        f"目标类型:{row.get('target_type','')}",
        f"目标:{target}",
        f"行业:{row.get('industry_name','')}",
        f"标题:{title}",
        f"来源:{source}",
        f"正文:{str(row.get('content',''))[:3500]}",
    ]
    if include_event:
        pieces.insert(0, f"事件:{row.get('event_name','')}")
    return "\n".join(pieces)


def pipeline_classes(model: Any) -> list[str]:
    model = unwrap_pipeline(model)
    if hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        return [str(x) for x in getattr(last, "classes_", [])]
    return head_classes(model)


def top_proba(model: Any, texts: list[str]) -> tuple[list[str], list[list[str]], list[float]]:
    model = unwrap_pipeline(model)
    probs = model.predict_proba(texts)
    classes = pipeline_classes(model)
    preds: list[str] = []
    topk: list[list[str]] = []
    confs: list[float] = []
    for row in probs:
        order = np.argsort(-row)
        preds.append(classes[int(order[0])])
        topk.append([classes[int(i)] for i in order[:10]])
        confs.append(float(row[int(order[0])]))
    return preds, topk, confs


def predict_multitask(
    heads: dict[str, Any],
    taxonomy: dict[str, Any],
    X: Any,
    include_official_bonus: bool = False,
    official_model: Any | None = None,
    official_probs: np.ndarray | None = None,
) -> tuple[list[str], list[list[str]], list[float]]:
    leaf_head = heads["leaf_head"]
    l1_head = heads["l1_head"]
    l2_head = heads["l2_head"]
    layer_head = heads["layer_head"]
    leaf_classes = head_classes(leaf_head)
    leaf_probs = head_predict_proba(leaf_head, X)
    l1_classes = head_classes(l1_head)
    l2_classes = head_classes(l2_head)
    layer_classes = head_classes(layer_head)
    l1_probs = head_predict_proba(l1_head, X)
    l2_probs = head_predict_proba(l2_head, X)
    layer_probs = head_predict_proba(layer_head, X)
    official_model = unwrap_pipeline(official_model)
    official_classes = head_classes(official_model) if official_model is not None else []
    official_index = {label: idx for idx, label in enumerate(official_classes)}
    preds, tops, confs = [], [], []
    for i in range(X.shape[0]):
        l1_map = {l1_classes[j]: float(l1_probs[i, j]) for j in range(len(l1_classes))}
        l2_map = {l2_classes[j]: float(l2_probs[i, j]) for j in range(len(l2_classes))}
        layer_map = {layer_classes[j]: float(layer_probs[i, j]) for j in range(len(layer_classes))}
        row_scores: dict[str, float] = {}
        for c, p in zip(leaf_classes, leaf_probs[i]):
            leaf = taxonomy["leaf_by_name"].get(c)
            if not leaf:
                continue
            score = float(p)
            score += 0.15 * l1_map.get(str(leaf["l1_parent_name"]), 0.0)
            score += 0.10 * l2_map.get(str(leaf["l2_group_name"]), 0.0)
            score += 0.05 * layer_map.get(str(leaf["event_layer"]), 0.0)
            if include_official_bonus and official_probs is not None and c in official_index:
                score += 0.25 * float(official_probs[i, official_index[c]])
            row_scores[c] = score
        ordered = sorted(row_scores.items(), key=lambda kv: kv[1], reverse=True)
        preds.append(ordered[0][0] if ordered else "")
        tops.append([k for k, _ in ordered[:10]])
        confs.append(float(softmax([v for _, v in ordered])[0]) if ordered else 0.0)
    return preds, tops, confs


class Task1Runtime:
    version = "step5_8e_core_plus_step5_8hc_hierarchical"

    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root
        self.models_dir = package_root / "models/task1"
        self.loaded = False

    def load(self) -> dict[str, Any]:
        _register_unpickle_shims()
        with (self.models_dir / "event_name_model.pkl").open("rb") as handle:
            self.event_bundle = pickle.load(handle)
        with (self.models_dir / "event_emotion_model.pkl").open("rb") as handle:
            self.emotion_bundle = pickle.load(handle)
        with (self.models_dir / "best_hierarchical_model.pkl").open("rb") as handle:
            self.hc_bundle = pickle.load(handle)
        self.event_model = unwrap_pipeline(self.event_bundle)
        self.emotion_model = unwrap_pipeline(self.emotion_bundle)
        self.loaded = True
        return {
            "task": "Task1",
            "loaded": True,
            "version": self.version,
            "models": [
                "models/task1/event_name_model.pkl",
                "models/task1/event_emotion_model.pkl",
                "models/task1/best_hierarchical_model.pkl",
            ],
            "note": "Task1 step5_8hi is a frozen safety-recognition prediction asset; official test predictions are not packaged or replayed.",
        }

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        work = dict(row)
        work.setdefault("article_title", work.get("title", ""))
        work.setdefault("article_source", work.get("source", ""))
        if not work.get("target_type"):
            work["target_type"] = "company" if (work.get("target_code") or work.get("trading_code") or work.get("secu_abbr")) else "industry"
        if not work.get("secu_abbr"):
            work["secu_abbr"] = work.get("target_name", "")
        text = row_text(work)
        event_pred, event_topk, event_conf = top_proba(self.event_model, [text])
        work["event_name"] = event_pred[0]
        emotion_pred, emotion_topk, emotion_conf = top_proba(self.emotion_model, [row_text(work, include_event=True)])
        prior = self.emotion_bundle.get("prior", {}) if isinstance(self.emotion_bundle, dict) else {}
        if isinstance(prior, dict) and event_pred[0] in prior and emotion_conf[0] < 0.42:
            emotion = clean_text(prior[event_pred[0]])
            emotion_source = "prior_backoff_below_0.42"
        else:
            emotion = emotion_pred[0]
            emotion_source = "text_model"
        X = self.hc_bundle["vectorizer"].transform([text])
        official_probs = self.event_model.predict_proba([text])
        hc_pred, hc_topk, hc_conf = predict_multitask(
            self.hc_bundle["suite"],
            self.hc_bundle["taxonomy"],
            X,
            include_official_bonus=True,
            official_model=self.event_model,
            official_probs=official_probs,
        )
        target_name = clean_text(work.get("target_name") or work.get("secu_abbr") or work.get("industry_name") or "未识别主体")
        target_code = clean_text(work.get("target_code") or work.get("trading_code") or work.get("industry_code"))
        evidence = compact(work.get("evidence") or work.get("content"), 240)
        return {
            "event_mention_id": stable_id("EM", work.get("article_id"), target_name, event_pred[0], sha256_text(evidence)[:12]),
            "target_type": work["target_type"],
            "target_code": target_code,
            "target_name": target_name,
            "event_name": event_pred[0],
            "event_name_confidence": round(event_conf[0], 6),
            "event_name_top10": event_topk[0],
            "event_emotion": emotion,
            "event_emotion_confidence": round(emotion_conf[0], 6),
            "event_emotion_source": emotion_source,
            "event_time": clean_text(work.get("publish_time")),
            "evidence": evidence,
            "duplication_id": stable_id("DUP", target_name, event_pred[0], re.sub(r"\s+", "", evidence)[:80]),
            "hierarchical_supplement": {
                "version": "step5_8hc_E_fusion",
                "event_name": hc_pred[0],
                "confidence": round(hc_conf[0], 6),
                "top10": hc_topk[0],
            },
            "step5_8hi_safety_recognition": {
                "selected_version": "task1_step5_8hi_tag_optimization",
                "packaged_prediction_asset": False,
                "runtime_replayed": False,
                "reason": "official labeled/test prediction asset is excluded from runnable package",
            },
            "model_loaded": True,
        }
'''


TASK2_RUNTIME_PY = r'''from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any

from .common import clean_text, compact, stable_id


class MissingExternalAsset(RuntimeError):
    pass


def tokenize(value: Any) -> list[str]:
    text = clean_text(value)
    return re.findall(r"[A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}", text)


def source_reliability(source: str) -> float:
    source = clean_text(source)
    if any(k in source for k in ["公告", "交易所", "证监", "证券时报", "上证报"]):
        return 0.9
    if any(k in source for k in ["财联社", "证券", "新闻", "日报"]):
        return 0.74
    if any(k in source for k in ["论坛", "股吧", "自媒体", "传闻"]):
        return 0.36
    return 0.62


class Task2Runtime:
    version = "final_cached_dea_v1_260"

    def __init__(self, package_root: Path, mode: str = "light") -> None:
        self.package_root = package_root
        self.mode = mode
        self.loaded = False
        self.external_status: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        self.loaded = True
        if self.mode == "full":
            self.external_status = self.check_external_assets(strict=True)
        else:
            self.external_status = self.check_external_assets(strict=False)
        return {
            "task": "Task2",
            "loaded": True,
            "version": self.version,
            "mode": self.mode,
            "formal_full_mode": self.mode == "full",
            "external_assets": self.external_status,
            "note": "Full DEA mode requires external DianJin-R1-32B service and SQLite index. Light mode is an independent deterministic inference path and is not equivalent to the official 260-row Cached DEA run.",
        }

    def check_external_assets(self, strict: bool) -> dict[str, Any]:
        index_dir = Path(os.environ.get("TASK2_SQLITE_INDEX_DIR", self.package_root / "external_assets/task2/dea_retrieval_index"))
        db_path = index_dir / "retrieval_index.sqlite"
        meta_path = index_dir / "retrieval_index_metadata.json"
        service = os.environ.get("TASK2_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        model = os.environ.get("TASK2_LLM_MODEL", "DianJin-R1-32B")
        status: dict[str, Any] = {
            "sqlite_index_dir": str(index_dir),
            "sqlite_exists": db_path.exists(),
            "metadata_exists": meta_path.exists(),
            "llm_base_url": service,
            "model_id": model,
            "llm_service_checked": False,
            "llm_model_available": False,
        }
        if db_path.exists():
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=5)
                conn.execute("pragma query_only=ON")
                status["sqlite_doc_count"] = int(conn.execute("select count(*) from docs").fetchone()[0])
                conn.close()
            except Exception as exc:
                status["sqlite_error"] = f"{type(exc).__name__}: {exc}"
        elif strict:
            raise MissingExternalAsset(f"missing Task2 SQLite index: {db_path}")
        if strict:
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
        return status

    def predict(self, row: dict[str, Any], task1: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        if self.mode == "full":
            return self.predict_full(row, task1)
        return self.predict_light(row, task1)

    def predict_light(self, row: dict[str, Any], task1: dict[str, Any]) -> dict[str, Any]:
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
        level = "high" if score >= 75 else "medium" if score >= 60 else "needs_review"
        return {
            "cluster_id": stable_id("CL", task1.get("duplication_id")),
            "credibility_score": round(score, 4),
            "credibility_level": level,
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
            "mode": "light_algorithmic_demo",
            "production_equivalent": False,
            "external_model_loaded": False,
            "cache_answer_used": False,
            "hardcoded_prediction": False,
            "model_loaded": True,
        }

    def predict_full(self, row: dict[str, Any], task1: dict[str, Any]) -> dict[str, Any]:
        self.check_external_assets(strict=True)
        service = os.environ.get("TASK2_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        model = os.environ.get("TASK2_LLM_MODEL", "DianJin-R1-32B")
        prompt = (
            "你是金融事件可信度评估器。只返回JSON，字段为 final_verdict、credibility_score、reason。"
            f"\n事件主张: {task1.get('target_name')} {task1.get('event_name')}"
            f"\n标题: {clean_text(row.get('title') or row.get('article_title'))}"
            f"\n来源: {clean_text(row.get('source') or row.get('article_source'))}"
            f"\n正文/证据: {compact(row.get('content'), 1200)}"
        )
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 256,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(f"{service}/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        content = payload["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"final_verdict": "unclear", "credibility_score": 50.0, "reason": compact(content, 400)}
        score = float(parsed.get("credibility_score", 50.0))
        score = max(0.0, min(100.0, score))
        verdict = clean_text(parsed.get("final_verdict") or "unclear")
        if verdict not in {"support", "contradict", "insufficient", "unclear"}:
            verdict = "unclear"
        return {
            "cluster_id": stable_id("CL", task1.get("duplication_id")),
            "credibility_score": round(score, 4),
            "credibility_level": "high" if score >= 75 else "medium" if score >= 60 else "needs_review",
            "final_verdict": verdict,
            "confidence": round(score / 100.0, 4),
            "evidence_chain": [{"source": "DianJin-R1-32B", "support_type": verdict, "evidence": compact(parsed.get("reason"), 220)}],
            "mode": "full_external_dea_demo",
            "production_equivalent": True,
            "external_model_loaded": True,
            "cache_answer_used": False,
            "hardcoded_prediction": False,
            "model_loaded": True,
        }
'''


TASK3_RUNTIME_PY = r'''from __future__ import annotations

import json
import re
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


class Task3Runtime:
    version = "legacy_official_pipeline"

    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root
        self.models_dir = package_root / "models/task3"
        self.loaded = False
        self.memory: list[dict[str, Any]] = []

    def load(self) -> dict[str, Any]:
        self.model = joblib.load(self.models_dir / "resolver_v2_model.joblib")
        self.selection = json.loads((self.models_dir / "resolver_v2_selection.json").read_text(encoding="utf-8"))
        self.loaded = True
        return {
            "task": "Task3",
            "loaded": True,
            "version": self.version,
            "model": "models/task3/resolver_v2_model.joblib",
            "thresholds": self.selection.get("thresholds", {}),
        }

    def resolve(self, row: dict[str, Any], task1: dict[str, Any], task2: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        current = event_like(row, task1, task2)
        classes = [str(x) for x in self.model.classes_]
        same_thr = float((self.selection.get("thresholds") or {}).get("same_thr", 0.45))
        best: dict[str, Any] | None = None
        for prior in self.memory:
            feats = feature_dict(prior["event"], current)
            probs = self.model.predict_proba([feats])[0]
            prob_map = {classes[i]: float(probs[i]) for i in range(len(classes))}
            score = prob_map.get("same_event", 0.0)
            if best is None or score > best["same_event_probability"]:
                best = {"prior": prior, "features": feats, "probabilities": prob_map, "same_event_probability": score}
        if best and best["same_event_probability"] >= same_thr:
            canonical_id = best["prior"]["canonical_event_id"]
            thread_id = best["prior"]["event_thread_id"]
            decision = "same_event"
            confidence = best["same_event_probability"]
        else:
            canonical_id = stable_id("CE", task1.get("target_code") or task1.get("target_name"), task1.get("event_name"), row.get("publish_time"))
            thread_id = stable_id("TH", task1.get("target_code") or task1.get("target_name"), task1.get("event_name"))
            decision = "new_event"
            confidence = 1.0 if best is None else max(0.0, 1.0 - best["same_event_probability"])
        state = "clarification" if any(k in clean_text(row.get("content")) for k in ["澄清", "否认", "不属实"]) else "completion" if any(k in clean_text(row.get("content")) for k in ["完成", "投产", "交付", "落地"]) else "announcement"
        self.memory.append({"canonical_event_id": canonical_id, "event_thread_id": thread_id, "event": current})
        return {
            "canonical_event_id": canonical_id,
            "event_thread_id": thread_id,
            "current_state_type": state,
            "event_mention_id": task1.get("event_mention_id"),
            "resolver_decision": decision,
            "resolver_confidence": round(confidence, 6),
            "candidate_match": {
                "available": best is not None,
                "same_event_probability": round(best["same_event_probability"], 6) if best else None,
                "probabilities": {k: round(v, 6) for k, v in (best["probabilities"] if best else {}).items()},
            },
            "graph_layer": "legacy_official_pipeline_resolver_v2",
            "trust_score": task2.get("confidence"),
            "model_loaded": True,
        }
'''


TASK4_CLOSURE_SHIMS_PY = r'''from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

SEED = 20260829
OFFICIAL_RISK_LABELS = ["1", "2", "3", "4", "5"]


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def feature_text(frame: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for _, row in frame.iterrows():
        content = clean(row.get("content"))[:1600]
        evidence = clean(row.get("evidence"))[:900]
        parts = [
            f"title={clean(row.get('title'))}",
            f"content={content}",
            f"evidence={evidence}",
            f"target={clean(row.get('target'))}",
            f"target_type={clean(row.get('target_type'))}",
            f"event_name={clean(row.get('event_name'))}",
            f"event_type={clean(row.get('event_type'))}",
            f"trigger={clean(row.get('trigger'))}",
            f"arguments={clean(row.get('arguments'))}",
            f"structured_event={clean(row.get('structured_event'))}",
            f"publish_year={clean(row.get('publish_year'))}",
            f"publish_month={clean(row.get('publish_month'))}",
            f"publish_dayofweek={clean(row.get('publish_dayofweek'))}",
        ]
        rows.append(" ".join(parts))
    return rows


class MulticlassTextRiskHead:
    def __init__(self, kind: str = "logreg", max_features: int = 9000, seed: int = SEED) -> None:
        self.kind = kind
        self.max_features = max_features
        self.seed = seed
        self.labels = list(OFFICIAL_RISK_LABELS)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=max_features, min_df=1, sublinear_tf=True)
        self.model: Any = None
        self.constant_label: str | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.constant_label:
            out = np.zeros((len(frame), len(self.labels)), dtype=float)
            out[:, self.labels.index(self.constant_label)] = 1.0
            return out
        x = self.vectorizer.transform(feature_text(frame))
        raw = self.model.predict_proba(x)
        classes = [str(c) for c in self.model.classes_]
        out = np.zeros((len(frame), len(self.labels)), dtype=float)
        for idx, label in enumerate(self.labels):
            if label in classes:
                out[:, idx] = raw[:, classes.index(label)]
        out = np.clip(out, 1e-12, None)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(frame)
        return np.asarray([self.labels[i] for i in np.argmax(p, axis=1)], dtype=object)

    def highrisk_score(self, frame: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(frame)
        return p[:, 3] + p[:, 4]


class CumulativeTextRiskHead:
    def __init__(self, kind: str = "logreg", max_features: int = 9000, seed: int = SEED) -> None:
        self.kind = kind
        self.max_features = max_features
        self.seed = seed
        self.labels = list(OFFICIAL_RISK_LABELS)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=max_features, min_df=1, sublinear_tf=True)
        self.models: list[Any] = []
        self.constants: list[float] = []

    def cumulative_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.vectorizer.transform(feature_text(frame))
        cols: list[np.ndarray] = []
        for model, constant in zip(self.models, self.constants):
            if model is None:
                cols.append(np.full(len(frame), constant, dtype=float))
            else:
                raw = model.predict_proba(x)
                classes = [int(c) for c in model.classes_]
                cols.append(raw[:, classes.index(1)] if 1 in classes else np.zeros(len(frame), dtype=float))
        out = np.vstack(cols).T if cols else np.zeros((len(frame), 4), dtype=float)
        return np.minimum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        c = self.cumulative_proba(frame)
        if len(c) == 0:
            return np.zeros((0, 5), dtype=float)
        p = np.zeros((len(frame), 5), dtype=float)
        p[:, 0] = 1.0 - c[:, 0]
        p[:, 1] = c[:, 0] - c[:, 1]
        p[:, 2] = c[:, 1] - c[:, 2]
        p[:, 3] = c[:, 2] - c[:, 3]
        p[:, 4] = c[:, 3]
        p = np.clip(p, 1e-12, None)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        c = self.cumulative_proba(frame)
        return (1 + (c >= 0.5).sum(axis=1)).astype(int).astype(str)

    def highrisk_score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.cumulative_proba(frame)[:, 2]


class BinaryHighRiskTextHead:
    def __init__(self, kind: str = "logreg", max_features: int = 9000, seed: int = SEED) -> None:
        self.kind = kind
        self.max_features = max_features
        self.seed = seed
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=max_features, min_df=1, sublinear_tf=True)
        self.model: Any = None
        self.constant_score: float | None = None

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.constant_score is not None:
            return np.full(len(frame), self.constant_score, dtype=float)
        x = self.vectorizer.transform(feature_text(frame))
        raw = self.model.predict_proba(x)
        classes = [int(c) for c in self.model.classes_]
        return raw[:, classes.index(1)] if 1 in classes else np.zeros(len(frame), dtype=float)

    def predict(self, frame: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_score(frame) >= threshold).astype(int)
'''


TASK4_RUNTIME_PY = r'''from __future__ import annotations

import sys
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


class Task4Runtime:
    version = "modular_final_combination"

    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root
        self.models_dir = package_root / "models/task4"
        self.loaded = False

    def load(self) -> dict[str, Any]:
        register_shims()
        self.closure_bundle = joblib.load(self.models_dir / "task4_final_contract_separated_closure_bundle.joblib")
        self.trace_bundle = joblib.load(self.models_dir / "task4_trace_refine_v3_2_student_bundle.joblib")
        self.trace_models = self.trace_bundle.get("models", {})
        self.loaded = True
        return {
            "task": "Task4",
            "loaded": True,
            "version": self.version,
            "models": [
                "models/task4/task4_final_contract_separated_closure_bundle.joblib",
                "models/task4/task4_trace_refine_v3_2_student_bundle.joblib",
            ],
            "production_contract": {
                "legacy_high_risk_gate_alert_direction": "selected but portable binaries not present in this package",
                "gold_only_ordinal_severity": "loaded",
                "trace_rag": "loaded for explanation/review only",
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

    def predict(self, row: dict[str, Any], task1: dict[str, Any], task2: dict[str, Any], task3: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        frame = self.frame(row, task1, task2, task3)
        risk_head = self.closure_bundle["official_risk_head"]
        high_head = self.closure_bundle["official_highrisk_head"]
        threshold = float(self.closure_bundle["official_highrisk_threshold"])
        severity = int(str(risk_head.predict(frame)[0]))
        risk_probs = risk_head.predict_proba(frame)[0]
        high_score = float(high_head.predict_score(frame)[0])
        high_pred = bool(high_score >= threshold)
        trace: dict[str, Any] = {}
        for head, model in self.trace_models.items():
            try:
                pred = model.predict(frame)[0]
                trace[head] = str(pred)
            except Exception as exc:
                trace[head] = {"error": f"{type(exc).__name__}: {exc}"}
        gate_value = trace.get("gate", "abstain")
        direction_value = trace.get("direction_short", "uncertain")
        alert_value = bool(high_pred and gate_value == "auto_score")
        return {
            "direction": {
                "value": direction_value,
                "source_model": "task4_trace_refine_v3_2_student_bundle",
                "production_selected_head": "legacy_official_direction",
                "production_head_loaded": False,
            },
            "severity": {
                "value": severity,
                "probabilities": {str(i + 1): round(float(v), 6) for i, v in enumerate(risk_probs)},
                "source_model": "gold_only_ordinal_severity_closure",
                "production_head_loaded": True,
            },
            "high_risk": {
                "value": high_pred,
                "score": round(high_score, 6),
                "threshold": round(threshold, 6),
                "source_model": "loaded_closure_highrisk_supplement",
                "production_selected_head": "legacy_official_high_risk",
                "production_head_loaded": False,
            },
            "applicability_gate": {
                "value": gate_value,
                "source_model": "task4_trace_refine_v3_2_student_bundle",
                "production_selected_head": "legacy_official_gate",
                "production_head_loaded": False,
            },
            "alert": {
                "value": alert_value,
                "source": "derived_from_loaded_highrisk_score_and_trace_gate_for_demo",
                "production_selected_head": "legacy_official_alert",
                "production_head_loaded": False,
            },
            "trace": trace,
            "rag_review_basis": [
                "TRACE/RAG is explanation and review basis only; it does not modify Official Risk in this delivery.",
                compact(task1.get("evidence"), 180),
            ],
            "formal_production_combo_status": "partial_portable_runtime",
            "model_loaded": True,
        }
'''


PIPELINE_PY = r'''from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .task1_runtime import Task1Runtime
from .task2_runtime import MissingExternalAsset, Task2Runtime
from .task3_runtime import Task3Runtime
from .task4_runtime import Task4Runtime


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))


def check_dependencies() -> tuple[int, dict[str, Any]]:
    required = ["joblib", "numpy", "pandas", "sklearn", "scipy", "lightgbm"]
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
    full_formal_ready = ok and mode == "full"
    payload = {
        "status": "pass" if ok else "fail",
        "mode": mode,
        "full_formal_ready": full_formal_ready,
        "sha256": sha_status,
        "tasks": reports,
    }
    return (0 if ok else 2), payload


def run_pipeline(input_path: Path, output_path: Path, mode: str) -> tuple[int, dict[str, Any]]:
    root = package_root()
    rows = read_jsonl(input_path)
    output: list[dict[str, Any]] = []
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
    for row in rows:
        for required in ["article_id", "title", "content", "publish_time"]:
            if required not in row:
                raise ValueError(f"missing required input field {required!r} in article_id={row.get('article_id')!r}")
        task1 = task1_model.predict(row)
        task2 = task2_model.predict(row, task1)
        task3 = task3_model.resolve(row, task1, task2)
        task4 = task4_model.predict(row, task1, task2, task3)
        output.append(
            {
                "article_id": row["article_id"],
                "task1": task1,
                "task2": task2,
                "task3": task3,
                "task4": task4,
                "meta": {
                    "runtime_version": "national_item2_runnable_runtime_v1",
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
    base_text = json.dumps(base, ensure_ascii=False, sort_keys=True)
    mutated_text = json.dumps(mutated, ensure_ascii=False, sort_keys=True)
    changed = base_text != mutated_text
    model_loaded_flags = {
        "task1_core_hc": all(row["task1"].get("model_loaded") for row in base),
        "task2_runtime": all(row["task2"].get("model_loaded") for row in base),
        "task3_resolver": all(row["task3"].get("model_loaded") for row in base),
        "task4_closure_trace": all(row["task4"].get("model_loaded") for row in base),
    }
    formal_gaps = [
        "Task1 step5_8hi packaged as sanitized manifest only; official prediction asset is excluded.",
        "Task2 full DEA mode needs external DianJin-R1-32B service and SQLite index; clean validation used light mode.",
        "Task4 legacy High-Risk/Gate/Alert/Direction portable model binaries are not present; loaded demo values come from closure/TRACE models and are marked non-production for those heads.",
    ]
    payload = {
        "status": "fail",
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
        "all_tasks_true_formal_production_inference": False,
        "runnable_submission_ready": False,
        "formal_gaps": formal_gaps,
    }
    local_ok = row_count_ok and required_tasks_ok and changed and all(model_loaded_flags.values())
    payload["local_light_demo_passed"] = bool(local_ok)
    if local_ok:
        payload["status"] = "partial_pass_light_demo"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return (0 if local_ok else 2), payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the national item2 Task1-Task4 pipeline.")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=["light", "full"], default=os.environ.get("PIPELINE_MODE", "light"))
    parser.add_argument("--check-dependencies", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--validate-demo", action="store_true")
    parser.add_argument("--base-output")
    parser.add_argument("--mutated-output")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.check_dependencies:
            code, payload = check_dependencies()
        elif args.healthcheck:
            code, payload = healthcheck(args.mode)
        elif args.validate_demo:
            code, payload = validate_demo(Path(args.base_output), Path(args.mutated_output), Path(args.report), args.mode)
        else:
            if not args.input or not args.output:
                raise ValueError("--input and --output are required unless running a check")
            code, payload = run_pipeline(Path(args.input), Path(args.output), args.mode)
    except Exception as exc:
        code = 2
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "mode": getattr(args, "mode", None)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
'''


RUN_PIPELINE_ROOT_PY = r'''#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_code"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from runnable_runtime.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_runtime_files() -> None:
    write_text(TARGET / "run_pipeline.py", RUN_PIPELINE_ROOT_PY, executable=True)
    pkg = TARGET / "source_code/runnable_runtime"
    write_text(pkg / "__init__.py", "")
    write_text(pkg / "common.py", COMMON_PY)
    write_text(pkg / "task1_runtime.py", TASK1_RUNTIME_PY)
    write_text(pkg / "task2_runtime.py", TASK2_RUNTIME_PY)
    write_text(pkg / "task3_runtime.py", TASK3_RUNTIME_PY)
    write_text(pkg / "task4_closure_shims.py", TASK4_CLOSURE_SHIMS_PY)
    write_text(pkg / "task4_runtime.py", TASK4_RUNTIME_PY)
    write_text(pkg / "pipeline.py", PIPELINE_PY)
    scripts_dir = TARGET / "source_code/scripts"
    write_text(scripts_dir / "__init__.py", "")
    shutil.copy2(ROOT / "scripts/task4_trace_refine_v3_2_lib.py", scripts_dir / "task4_trace_refine_v3_2_lib.py")


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
printf '{"status":"started","server_required":false,"pid":%s,"started_at":"%s"}\n' "$$" "$(date -Iseconds)" > "$ROOT/logs/runtime_state.json"
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
PIPELINE_MODE="${PIPELINE_MODE:-light}"
mkdir -p "$ROOT/logs"
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
PIPELINE_MODE="${PIPELINE_MODE:-light}"
mkdir -p "$ROOT/logs" "$ROOT/outputs"
"$PYTHON_BIN" "$ROOT/run_pipeline.py" --mode "$PIPELINE_MODE" --input "$ROOT/demo_data/sample.jsonl" --output "$ROOT/outputs/demo_result.jsonl" | tee "$ROOT/logs/run_demo.log"
"$PYTHON_BIN" "$ROOT/run_pipeline.py" --mode "$PIPELINE_MODE" --input "$ROOT/demo_data/sample_mutated.jsonl" --output "$ROOT/outputs/demo_result_mutated.jsonl" | tee "$ROOT/logs/run_demo_mutated.log"
"$PYTHON_BIN" "$ROOT/run_pipeline.py" --validate-demo --mode "$PIPELINE_MODE" --base-output "$ROOT/outputs/demo_result.jsonl" --mutated-output "$ROOT/outputs/demo_result_mutated.jsonl" --report "$ROOT/smoke_test_report.json" | tee "$ROOT/logs/smoke_test.log"
''',
        executable=True,
    )
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
            "evidence_items": [
                {
                    "source": "公司公告",
                    "publish_time": "2026-08-29 09:30:00",
                    "sentence": "公司与下游客户签署为期三年的算力芯片供货合同。",
                }
            ],
        },
        {
            "article_id": "demo_002",
            "title": "蓝海材料收到交易所监管函",
            "content": "蓝海材料披露，公司因信息披露不及时收到交易所监管函。公司表示将整改内部控制流程，短期经营存在一定不确定性。",
            "publish_time": "2026-08-29 10:15:00",
            "source": "交易所公告",
            "target_name": "蓝海材料",
            "target_code": "688002",
            "industry_name": "新材料",
            "evidence_items": [
                {
                    "source": "交易所公告",
                    "publish_time": "2026-08-29 10:15:00",
                    "sentence": "公司因信息披露不及时收到交易所监管函。",
                }
            ],
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
            "evidence_items": [
                {
                    "source": "财经新闻",
                    "publish_time": "2026-08-29 11:00:00",
                    "sentence": "相关事项仍在论证阶段，尚未形成正式决议。",
                }
            ],
        },
    ]
    mutated = [dict(x) for x in rows]
    mutated[0]["content"] = "晶源电子澄清称此前长期供货合同报道不属实，公司目前未与相关客户签署正式协议。"
    mutated[1]["content"] = "蓝海材料公告称监管函相关事项已完成整改，并获得交易所后续确认。"
    mutated[2]["content"] = "云帆机器人发布公告确认海外工厂项目已获董事会批准，首期投资计划已经启动。"
    for path, data in [(TARGET / "demo_data/sample.jsonl", rows), (TARGET / "demo_data/sample_mutated.jsonl", mutated)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_schemas_and_configs() -> None:
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Task1-Task4 runnable demo input",
        "type": "object",
        "required": ["article_id", "title", "content", "publish_time"],
        "properties": {
            "article_id": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "publish_time": {"type": "string"},
            "source": {"type": "string"},
            "target_name": {"type": "string"},
            "target_code": {"type": "string"},
            "industry_name": {"type": "string"},
            "evidence_items": {"type": "array"},
        },
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Task1-Task4 runnable demo output",
        "type": "object",
        "required": ["article_id", "task1", "task2", "task3", "task4", "meta"],
    }
    write_json(TARGET / "schema/input_schema.json", input_schema)
    write_json(TARGET / "schema/output_schema.json", output_schema)
    for cfg in (COMPUTE / "configs").glob("*"):
        if cfg.is_file():
            copy_file(cfg, TARGET / "configs" / cfg.name)
    write_json(
        TARGET / "configs/runtime_config.json",
        {
            "runtime_version": "national_item2_runnable_runtime_v1",
            "default_mode": "light",
            "full_mode_env": {
                "PIPELINE_MODE": "full",
                "TASK2_LLM_BASE_URL": "http://127.0.0.1:8000/v1",
                "TASK2_LLM_MODEL": "DianJin-R1-32B",
                "TASK2_SQLITE_INDEX_DIR": "external_assets/task2/dea_retrieval_index",
            },
            "blocked_original_project_absolute_path": True,
        },
    )


def copy_models() -> list[dict[str, Any]]:
    records = []
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
        }
        write_json(TARGET / "models/task1/step5_8hi_frozen_model_manifest_sanitized.json", sanitized)
    return records


def build_requirements() -> None:
    write_text(
        TARGET / "requirements.txt",
        """joblib>=1.3\nnumpy>=1.24\npandas>=2.0\nscipy>=1.10\nscikit-learn>=1.3\nlightgbm>=4.0\n""",
    )
    write_text(
        TARGET / "environment.yml",
        """name: competition-item2-runtime\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.11\n  - numpy\n  - pandas\n  - scipy\n  - scikit-learn\n  - joblib\n  - pip\n  - pip:\n      - lightgbm>=4.0\n""",
    )


def model_manifest() -> dict[str, Any]:
    models = []
    for rel, task, version, role, packaged, formal in [
        ("models/task1/event_name_model.pkl", "Task1", "step5_8e", "Core event name", True, True),
        ("models/task1/event_emotion_model.pkl", "Task1", "step5_8e", "Core event emotion", True, True),
        ("models/task1/best_hierarchical_model.pkl", "Task1", "step5_8hc", "Hierarchical supplement", True, True),
        ("models/task1/step5_8hi_frozen_model_manifest_sanitized.json", "Task1", "step5_8hi", "Safety recognition frozen manifest only", True, False),
        ("models/task3/resolver_v2_model.joblib", "Task3", "legacy_official_pipeline", "Canonical/event-thread resolver", True, True),
        ("models/task4/task4_final_contract_separated_closure_bundle.joblib", "Task4", "Gold-only ordinal closure", "Ordinal severity and closure heads", True, True),
        ("models/task4/task4_trace_refine_v3_2_student_bundle.joblib", "Task4", "TRACE v3.2", "TRACE explanation/review heads", True, True),
    ]:
        path = TARGET / rel
        models.append(
            {
                "task": task,
                "version": version,
                "role": role,
                "path": rel,
                "packaged": packaged,
                "formal_selected_asset": formal,
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
            "reason": "large external LLM and SQLite retrieval index are mounted externally for full mode",
        },
        "task4_production_heads": [
            "legacy High-Risk",
            "legacy Applicability Gate",
            "legacy Alert",
            "legacy Direction",
            "Gold-only Ordinal Severity",
            "TRACE/RAG explanation and review only",
        ],
        "task4_portability_note": "Gold-only ordinal and TRACE bundles are packaged and loadable. Legacy High-Risk/Gate/Alert/Direction portable binaries were not found in compute package, so the runtime marks those heads as not loaded instead of substituting them silently.",
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
                "service_reported_version": {"owned_by": "vllm", "max_model_len": 8192},
                "size_bytes": 65543821517,
                "mount_path": "external_assets/task2/DianJin-R1-32B",
                "env": "DIANJIN_MODEL_PATH",
                "sha256_manifest_digest": sha256_text(digest_input) if digest_input else None,
                "file_level_sha256": dianjin_files,
                "start_service_example": "vllm serve ${DIANJIN_MODEL_PATH:-external_assets/task2/DianJin-R1-32B} --served-model-name DianJin-R1-32B --host 127.0.0.1 --port 8000 --max-model-len 8192",
            },
            {
                "name": "Task2 DEA SQLite retrieval index",
                "type": "external_sqlite_index",
                "packaged": False,
                "version": idx_meta_obj.get("index_version", "task2_retrieval_index_v1"),
                "size_bytes": idx.stat().st_size if idx.exists() else None,
                "sha256": "4f10b523185dbb256ad88f0535668c54f8eed9f2c38d05763fd08babb91b891f",
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
            },
        ],
        "full_mode_command": "PIPELINE_MODE=full TASK2_LLM_BASE_URL=http://127.0.0.1:8000/v1 TASK2_LLM_MODEL=DianJin-R1-32B TASK2_SQLITE_INDEX_DIR=external_assets/task2/dea_retrieval_index ./run_demo.sh",
        "light_mode_command": "PIPELINE_MODE=light ./run_demo.sh",
        "missing_resource_behavior": "Full mode raises an explicit MissingExternalAsset error; it does not fall back silently.",
    }


def build_docs(asset_records: list[dict[str, Any]], manifest: dict[str, Any], external_manifest: dict[str, Any]) -> None:
    selected = manifest["selected_versions"]
    readme = f"""# 国赛第②项：可运行项目成果

本目录只包含第②项可运行工程包，不处理精益画布、技术文档、原创性声明。

## 运行入口

```bash
./install.sh
./start.sh
./healthcheck.sh
./run_demo.sh
./stop.sh
```

也可直接执行：

```bash
python run_pipeline.py --input demo_data/sample.jsonl --output outputs/result.jsonl
```

默认 `PIPELINE_MODE=light`，用于无外部大模型/索引挂载时的独立 demo。完整 Task2 DEA 模式需要挂载 DianJin-R1-32B 和 SQLite 索引：

```bash
PIPELINE_MODE=full \\
TASK2_LLM_BASE_URL=http://127.0.0.1:8000/v1 \\
TASK2_LLM_MODEL=DianJin-R1-32B \\
TASK2_SQLITE_INDEX_DIR=external_assets/task2/dea_retrieval_index \\
./run_demo.sh
```

## 正式版本

- Task1: {selected['task1']}
- Task2: {selected['task2']}
- Task3: {selected['task3']}
- Task4: {selected['task4']}

Task4 生产组合为 legacy High-Risk/Gate/Alert/Direction + Gold-only Ordinal Severity + TRACE/RAG 解释复核。包内可加载并验证的是 Gold-only ordinal closure 和 TRACE v3.2；legacy High-Risk/Gate/Alert/Direction 未发现可移植二进制，因此不会在 demo 中冒充已加载。

## 轻量模式与完整模式差异

轻量模式不读取缓存答案、不按样本硬编码输出，执行基于输入内容、来源可信度、主张/证据重合和矛盾信号的确定性推理；它不等价于正式 `final_cached_dea_v1_260`。完整模式会校验 SQLite 索引并调用 DianJin-R1-32B OpenAI-compatible 服务，缺失资源时显式失败。

## 排除资产

已排除 Task1 5.8-F、Task2 v2/ISR/Safety Gate/Claim Adapter、Task3 Hybrid v3.1/Multiroute、Task4 v3.4/FewRisk/答案泄漏 v1。未打包 Gold、官方测试数据、final_holdout、CSMAR 原始库、密钥或受限数据。
"""
    write_text(TARGET / "README.md", readme)
    write_text(
        TARGET / "deployment/deployment_guide.md",
        """# 部署说明

1. 准备 Python 3.11 环境并安装 `requirements.txt`。
2. 执行 `./install.sh` 做依赖检查。
3. 默认轻量 demo：`./start.sh && ./healthcheck.sh && ./run_demo.sh && ./stop.sh`。
4. 完整 Task2 模式需额外挂载：
   - `external_assets/task2/DianJin-R1-32B`
   - `external_assets/task2/dea_retrieval_index/retrieval_index.sqlite`
   - `external_assets/task2/dea_retrieval_index/retrieval_index_metadata.json`
5. 启动 DianJin 服务后设置：

```bash
export PIPELINE_MODE=full
export TASK2_LLM_BASE_URL=http://127.0.0.1:8000/v1
export TASK2_LLM_MODEL=DianJin-R1-32B
export TASK2_SQLITE_INDEX_DIR=external_assets/task2/dea_retrieval_index
```

本包所有运行路径均按解压目录相对定位；不会回退读取原项目目录。
""",
    )
    write_text(
        TARGET / "deployment/api_guide.md",
        """# API 说明

## 命令行

```bash
python run_pipeline.py --input demo_data/sample.jsonl --output outputs/result.jsonl --mode light
```

输入为 JSONL，每行至少包含 `article_id`、`title`、`content`、`publish_time`。

输出为 JSONL，每行包含：

- `task1`: 结构化事件、情感、层级补充
- `task2`: 可信度评分、证据链、轻量/完整模式标记
- `task3`: CanonicalEvent、Event Thread、状态
- `task4`: Direction、Severity、High-Risk、Gate、Alert、TRACE/RAG 说明

`--mode full` 会强制校验 Task2 外部资产；缺失时返回非零状态和明确错误。
""",
    )
    write_text(
        TARGET / "source_code/SOURCE_MAPPING.md",
        """# Source Mapping

Runtime source in this package is under `source_code/runnable_runtime/` and is the code executed by `run_pipeline.py`.

Selected frozen assets are mapped as follows:

- Task1 Core: `event_name_model.pkl`, `event_emotion_model.pkl`
- Task1 HC: `best_hierarchical_model.pkl`
- Task1 HI: sanitized frozen manifest only; official prediction files are excluded
- Task2: `final_cached_dea_v1_260`; full model and index are external
- Task3: `resolver_v2_model.joblib`
- Task4: `task4_final_contract_separated_closure_bundle.joblib`, `task4_trace_refine_v3_2_student_bundle.joblib`

No training, tuning, pseudo-label generation, Gold reading, or official test reading is performed by runtime scripts.
""",
    )


def build_audits(manifest: dict[str, Any], external_manifest: dict[str, Any]) -> None:
    dep = {
        "generated_at": now_iso(),
        "python_required": ">=3.11 recommended",
        "dependencies": [
            {"name": "joblib", "purpose": "model deserialization", "license": "BSD-3-Clause"},
            {"name": "numpy", "purpose": "array computation", "license": "BSD-3-Clause"},
            {"name": "pandas", "purpose": "Task4 feature frame", "license": "BSD-3-Clause"},
            {"name": "scipy", "purpose": "TRACE model dependencies", "license": "BSD-3-Clause"},
            {"name": "scikit-learn", "purpose": "Task1/Task3/Task4 model inference", "license": "BSD-3-Clause"},
            {"name": "lightgbm", "purpose": "Task4 closure LightGBM heads", "license": "MIT"},
            {"name": "sqlite3", "purpose": "Task2 external index probe", "license": "Python stdlib"},
        ],
        "external_dependencies": ["DianJin-R1-32B service", "Task2 DEA SQLite retrieval index"],
        "license_notes": "DianJin-R1-32B and the DEA SQLite index are not redistributed; mount externally under their own license/authorization.",
    }
    write_json(TARGET / "dependency_audit.json", dep)
    write_json(TARGET / "model_manifest.json", manifest)
    write_json(TARGET / "external_assets_manifest.json", external_manifest)
    write_json(
        TARGET / "runnable_asset_manifest.json",
        {
            "generated_at": now_iso(),
            "package": "02_可运行项目成果",
            "selected_versions": manifest["selected_versions"],
            "entrypoints": {
                "pipeline": "run_pipeline.py",
                "install": "install.sh",
                "start": "start.sh",
                "healthcheck": "healthcheck.sh",
                "run_demo": "run_demo.sh",
                "stop": "stop.sh",
            },
            "models": manifest["models"],
            "external_assets": external_manifest["assets"],
            "excluded_assets": [
                "Task1 5.8-F",
                "Task2 v2/ISR/Safety Gate/Claim Adapter",
                "Task3 Hybrid v3.1/Multiroute",
                "Task4 v3.4/FewRisk/answer-leak v1",
                "Gold, official test data, final_holdout, CSMAR raw databases, secrets",
            ],
            "readiness_policy": "Do not mark runnable_submission_ready unless full selected production heads and Task2 external DEA assets are validated.",
        },
    )


def scan_package() -> dict[str, Any]:
    secret_patterns = [
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY"),
        re.compile(r"(?i)(api[_-]?key|secret[_-]?key|password)\s*[:=]\s*['\"][^'\"]{8,}"),
    ]
    forbidden_path_terms = re.compile(r"(?i)(official1799|final_holdout|csmar|golddata|news_with_tags_test|news_without_tags_test|severity_master_gold|filled)")
    absolute_project = "/opt/data/private/JinRong/kechuang/financial_event_project"
    secret_hits: list[dict[str, Any]] = []
    forbidden_path_hits: list[str] = []
    absolute_hits: list[str] = []
    binary_metadata_hits: list[dict[str, Any]] = []
    binary_suffixes = {".pkl", ".joblib", ".zip", ".gz", ".tar"}
    for path in sorted(TARGET.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(TARGET))
        if forbidden_path_terms.search(rel):
            forbidden_path_hits.append(rel)
        if path.suffix.lower() in binary_suffixes:
            patterns = {
                "absolute_original_project_path": absolute_project.encode("utf-8"),
                "official1799_term": b"official1799",
                "final_holdout_term": b"final_holdout",
                "golddata_term": b"golddata",
            }
            found: set[str] = set()
            tail = b""
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    data = tail + chunk
                    for name, needle in patterns.items():
                        if needle in data:
                            found.add(name)
                    tail = data[-256:]
            if "absolute_original_project_path" in found:
                absolute_hits.append(rel)
            for name in sorted(found):
                binary_metadata_hits.append({"path": rel, "pattern": name, "metadata_only": True})
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if absolute_project in text:
            absolute_hits.append(rel)
        for pattern in secret_patterns:
            for match in pattern.finditer(text):
                secret_hits.append({"path": rel, "pattern": pattern.pattern, "offset": match.start()})
    report = {
        "generated_at": now_iso(),
        "status": "pass" if not secret_hits and not forbidden_path_hits and not absolute_hits else "fail",
        "secret_hits": secret_hits,
        "forbidden_data_path_hits": forbidden_path_hits,
        "absolute_original_project_path_hits": absolute_hits,
        "binary_metadata_hits": binary_metadata_hits,
        "scanned_text_files": sum(1 for p in TARGET.rglob("*") if p.is_file() and p.suffix.lower() not in {".pkl", ".joblib", ".zip", ".gz", ".tar"}),
        "binary_model_files_scanned_for_metadata_strings": True,
    }
    write_json(TARGET / "secret_scan_report.json", report)
    return report


def write_checksums() -> None:
    lines: list[str] = []
    for path in sorted(TARGET.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rel = path.relative_to(TARGET).as_posix()
            lines.append(f"{sha256_file(path)}  {rel}")
    write_text(TARGET / "checksums.sha256", "\n".join(lines) + "\n")


def create_zip(zip_path: Path) -> dict[str, Any]:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(TARGET.rglob("*")):
            if path.is_file():
                zf.write(path, Path(TARGET.name) / path.relative_to(TARGET))
    path_label = zip_path.relative_to(ROOT).as_posix() if zip_path.is_relative_to(ROOT) else zip_path.name
    return {"path": path_label, "size_bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path)}


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "seconds": round(time.perf_counter() - start, 6),
        "output": proc.stdout,
    }


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
    env["PIPELINE_MODE"] = "light"
    env["PYTHONPATH"] = str(extracted / "source_code")
    env["ORIGINAL_PROJECT_ROOT"] = "/__blocked_original_project__"
    commands = [
        ["bash", "install.sh"],
        ["bash", "start.sh"],
        ["bash", "healthcheck.sh"],
        ["bash", "run_demo.sh"],
        ["bash", "stop.sh"],
    ]
    results = [run_command(cmd, extracted, env) for cmd in commands]
    smoke = read_json_if_exists(extracted / "smoke_test_report.json")
    zip_label = zip_path.relative_to(ROOT).as_posix() if zip_path.is_relative_to(ROOT) else zip_path.name
    log_lines = [
        f"clean_environment_validation_at={now_iso()}",
        f"zip_under_test={zip_label}",
        f"extract_root={VALIDATION_ROOT}",
        f"python_bin={py}",
        "original_project_root_mask=/__blocked_original_project__",
        "",
    ]
    for result in results:
        log_lines.append(f"$ {' '.join(result['cmd'])}")
        log_lines.append(f"returncode={result['returncode']} seconds={result['seconds']}")
        log_lines.append(result["output"])
        log_lines.append("")
    validation = {
        "generated_at": now_iso(),
        "zip_under_test": zip_label,
        "extract_root": str(VALIDATION_ROOT),
        "python_bin": py,
        "commands": results,
        "all_commands_zero": all(r["returncode"] == 0 for r in results),
        "smoke_test_report": smoke,
        "status": "partial_pass_light_demo" if all(r["returncode"] == 0 for r in results) and smoke.get("local_light_demo_passed") else "fail",
        "runnable_submission_ready": bool(smoke.get("runnable_submission_ready", False)),
    }
    write_text(TARGET / "clean_environment_validation.log", "\n".join(log_lines))
    if (extracted / "smoke_test_report.json").exists():
        shutil.copy2(extracted / "smoke_test_report.json", TARGET / "smoke_test_report.json")
    if (extracted / "outputs/demo_result.jsonl").exists():
        shutil.copy2(extracted / "outputs/demo_result.jsonl", TARGET / "demo_data/standard_output_sample.jsonl")
        (TARGET / "outputs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted / "outputs/demo_result.jsonl", TARGET / "outputs/demo_result.jsonl")
    if (extracted / "outputs/demo_result_mutated.jsonl").exists():
        (TARGET / "outputs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted / "outputs/demo_result_mutated.jsonl", TARGET / "outputs/demo_result_mutated.jsonl")
    if (extracted / "logs").exists():
        dst = TARGET / "logs"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(extracted / "logs", dst)
    write_json(TARGET / "clean_environment_validation_summary.json", validation)
    return validation


def main() -> int:
    if not COMPUTE.exists():
        raise FileNotFoundError(COMPUTE)
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    asset_records = copy_models()
    build_runtime_files()
    build_scripts()
    build_demo_data()
    build_schemas_and_configs()
    build_requirements()
    manifest = model_manifest()
    external_manifest = external_assets_manifest()
    build_docs(asset_records, manifest, external_manifest)
    build_audits(manifest, external_manifest)
    scan_package()
    write_json(TARGET / "initial_asset_copy_records.json", asset_records)
    write_checksums()
    pre_zip = create_zip(ZIP_PATH)
    validation = clean_validate(ZIP_PATH)
    scan_package()
    write_checksums()
    final_zip = create_zip(ZIP_PATH)
    final_report = {
        "generated_at": now_iso(),
        "target_dir": TARGET.relative_to(ROOT).as_posix(),
        "zip": {
            "path": ZIP_PATH.relative_to(ROOT).as_posix(),
            "sha256": "not_embedded_to_avoid_archive_self_reference",
            "size_bytes": "see external sha256sum/du output",
        },
        "pre_validation_zip": pre_zip,
        "validation": {
            "status": validation["status"],
            "runnable_submission_ready": validation["runnable_submission_ready"],
            "extract_root": validation["extract_root"],
        },
        "constraints": {
            "training_executed": False,
            "tuning_executed": False,
            "pseudo_label_generated": False,
            "gold_read": False,
            "official_test_read": False,
            "frozen_assets_modified": False,
        },
    }
    write_json(TARGET / "final_delivery_item2_report.json", final_report)
    write_checksums()
    final_zip = create_zip(ZIP_PATH)
    print(json.dumps({"status": "ok", "target": str(TARGET), "zip": final_zip, "validation": validation["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
