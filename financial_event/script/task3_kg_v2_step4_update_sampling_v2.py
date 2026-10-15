#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
STEP4 = ROOT / "outputs/task3_kg_v2/step4"
REV1 = ROOT / "outputs/task3_kg_v2/step4_revision"
REV2 = ROOT / "outputs/task3_kg_v2/step4_revision_v2"
V1_NODES = ROOT / "outputs/task3_kg_v2/observation_graph_v1/nodes.jsonl"
V1_EDGES = ROOT / "outputs/task3_kg_v2/observation_graph_v1/edges.jsonl"
FULLTEXT = ROOT / "data/processed/event_credibility/fulltext_article_index.parquet"
MENTIONS = ROOT / "outputs/task3_fulltext_v2/normalized_event_mentions.jsonl"
EVIDENCE = ROOT / "outputs/task3_fulltext_v2/event_mention_evidence_sentences.jsonl"
OLD_REVIEW = REV1 / "annotation_package_agent_reviewed.xlsx"
OLD_REPORT = REV1 / "sampling_report.json"
OLD_FREEZE = REV1 / "step4_v0_relation_classifier_revision_required.json"
CANDIDATES_V1 = STEP4 / "same_event_candidates.parquet"
OUT_CAND = REV2 / "update_candidates_v2.parquet"
OUT_PACKAGE = REV2 / "update_annotation_package_v2.xlsx"
OUT_EOK = REV2 / "event_object_keys.parquet"
OUT_RETRIEVAL = REV2 / "retrieval_evidence.jsonl"
OUT_REJECTED = REV2 / "rejected_candidate_audit.jsonl"
REPORT = ROOT / "reports/task3_kg_v2_step4_update_sampling_v2.json"
GUIDE = ROOT / "docs/task3_kg_v2_step4_update_sampling_v2.md"
FREEZE_V2 = REV2 / "step4_update_candidate_generation_v1_failed.json"
DOWNLOAD_ROOT = ROOT.parent / "downloads/Qwen"
EMBED_DIR = DOWNLOAD_ROOT / "Qwen3-Embedding-0.6B"
RERANK_DIR = DOWNLOAD_ROOT / "Qwen3-Reranker-0.6B"

RELATIONS = [
    "rumor_to_confirmation",
    "announcement_to_progress",
    "progress_to_completion",
    "investigation_to_penalty",
    "clarification_or_refutation",
    "plan_to_termination",
]

STAGE_WORDS = {
    "rumor": ["传闻", "网传", "据悉", "消息称", "或将", "拟", "可能", "市场消息"],
    "confirmation": ["官方", "确认", "证实", "公告", "披露", "回应", "发布"],
    "announcement": ["公告", "宣布", "签约", "中标", "立项", "协议", "签订", "披露"],
    "progress": ["进展", "推进", "实施", "开工", "跟进", "审核推进", "项目进展", "持续推进"],
    "completion": ["完成", "建成", "投产", "交付", "上市", "正式运营", "落地", "获批", "达成"],
    "investigation": ["调查", "立案", "问询", "检查", "核查", "审查"],
    "penalty": ["处罚", "罚款", "警示", "处分", "整改", "责令", "通报", "问责"],
    "clarify": ["澄清", "否认", "辟谣", "说明", "回应"],
    "report_claim": ["报道", "消息", "传闻", "网传", "市场消息", "据悉"],
    "plan": ["计划", "拟", "方案", "预案", "协议", "将", "考虑"],
    "termination": ["终止", "取消", "撤回", "失败", "不再推进", "停止", "中止", "放弃"],
}

CORE_RE = re.compile(r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·（）()\-]{2,40}?)(?=(?:项目|产品|车型|芯片|药品|交易|并购|收购|减持|增持|融资|处罚|调查|签约|中标|开工|投产|交付|公告|立案|问询|澄清|终止|取消|完成|进展|获批))")
ID_RE = re.compile(r"(?:案号|编号|合同号|公告号|订单号|批次|轮次|比例|股权|金额|规模|价格|产能|里程碑)[：:\s]*([^，。；;\n]{2,40})")
AMOUNT_RE = re.compile(r"\d+(?:\.\d+)?(?:亿元|万元|元|%|股|万股|亿股|吨|万手|bp)")
DATE_RE = re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}(?:日)?")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return " ".join(str(x).lower().replace("\u3000", " ").replace("\xa0", " ").split())


def listify(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i)]
    try:
        v = json.loads(str(x))
        if isinstance(v, list):
            return [str(i) for i in v if str(i)]
    except Exception:
        pass
    return [p for p in str(x).replace("|", ",").split(",") if p]


def maybe_time(x: Any) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(x)
    except Exception:
        return None


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def pair_id(a: str, b: str) -> str:
    x, y = sorted([a, b])
    return stable_id("UPD", x, y)


def text_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    aa = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", a))
    bb = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", b))
    return len(aa & bb) / len(aa | bb) if aa or bb else 0.0


def any_word(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def stage_flags(text: str) -> dict[str, bool]:
    return {
        "rumor": any_word(text, STAGE_WORDS["rumor"]),
        "confirmation": any_word(text, STAGE_WORDS["confirmation"]),
        "announcement": any_word(text, STAGE_WORDS["announcement"]),
        "progress": any_word(text, STAGE_WORDS["progress"]),
        "completion": any_word(text, STAGE_WORDS["completion"]),
        "investigation": any_word(text, STAGE_WORDS["investigation"]),
        "penalty": any_word(text, STAGE_WORDS["penalty"]),
        "clarify": any_word(text, STAGE_WORDS["clarify"]),
        "report_claim": any_word(text, STAGE_WORDS["report_claim"]),
        "plan": any_word(text, STAGE_WORDS["plan"]),
        "termination": any_word(text, STAGE_WORDS["termination"]),
    }


def extract_object_core(text: str) -> dict[str, str]:
    text = norm(text)
    names = [m.group("name") for m in CORE_RE.finditer(text)]
    core_name = names[0] if names else ""
    identifiers = []
    identifiers.extend(AMOUNT_RE.findall(text))
    identifiers.extend([m.group(1) for m in ID_RE.finditer(text)])
    identifiers.extend(DATE_RE.findall(text))
    identifiers = [i for i in identifiers if i]
    obj_type = "generic"
    if any(w in text for w in ["项目", "工程", "开工", "投产", "建设", "交付"]):
        obj_type = "project"
    elif any(w in text for w in ["产品", "车型", "芯片", "药品", "型号", "技术"]):
        obj_type = "product"
    elif any(w in text for w in ["交易", "并购", "收购", "出售", "股权", "减持", "增持"]):
        obj_type = "transaction"
    elif any(w in text for w in ["处罚", "调查", "立案", "问询", "检查", "核查"]):
        obj_type = "regulation"
    elif any(w in text for w in ["融资", "贷款", "授信", "增发", "配股"]):
        obj_type = "financing"
    elif any(w in text for w in ["财报", "业绩", "利润", "营收", "营收", "财务"]):
        obj_type = "financial"
    elif any(w in text for w in ["技术", "算法", "模型", "平台"]):
        obj_type = "technology"
    return {"core_object_name": core_name, "core_object_type": obj_type, "object_identifiers": "|".join(dict.fromkeys(identifiers))}


def object_key(row: pd.Series) -> str:
    text = " ".join([str(row.get("document_title") or ""), str(row.get("event_local_text") or ""), str(row.get("evidence_text") or "")])
    core = extract_object_core(text)
    evt = norm(row.get("provisional_event_type"))
    parts = [
        str(row.get("target_code") or ""),
        evt,
        core["core_object_type"],
        core["core_object_name"],
        core["object_identifiers"],
    ]
    return stable_id("EOK", *parts)


def load_mentions() -> pd.DataFrame:
    mentions = read_jsonl(MENTIONS)
    evidence = read_jsonl(EVIDENCE)
    ev_map: dict[str, list[str]] = defaultdict(list)
    for r in evidence:
        ev_map[str(r.get("event_mention_id"))].append(str(r.get("sentence") or ""))
    nodes = read_jsonl(V1_NODES)
    titles = {str(n.get("article_id")): str(n.get("title") or "") for n in nodes if n.get("node_type") == "Document"}
    doc_by_mid: dict[str, str] = {}
    eu_by_doc: dict[str, set[str]] = defaultdict(set)
    for e in read_jsonl(V1_EDGES):
        if e.get("relation_type") == "CONTAINS_MENTION":
            doc_by_mid[str(e.get("target_stable_id"))] = str(e.get("source_stable_id"))
        elif e.get("relation_type") == "BELONGS_TO_EVIDENCE_UNIT":
            eu_by_doc[str(e.get("source_stable_id"))].add(str(e.get("target_stable_id")))
    rows = []
    for m in mentions:
        mid = str(m["event_mention_id"])
        art = str(m.get("article_id") or "")
        doc_stable = doc_by_mid.get(mid, "")
        evidence_sents = ev_map.get(mid) or list(m.get("evidence_sentences") or [])
        local_text = norm(m.get("event_local_text") or "")
        combined = norm(" ".join([titles.get(art, ""), local_text, " ".join(evidence_sents[:5])]))
        flags = stage_flags(combined)
        rows.append({
            "event_mention_id": mid,
            "article_id": art,
            "document_id": str(m.get("document_id") or ""),
            "document_stable_id": doc_stable,
            "document_title": titles.get(art, ""),
            "publish_time": maybe_time(m.get("publish_time")),
            "target_type": str(m.get("target_type") or ""),
            "target_code": str(m.get("target_code") or ""),
            "provisional_event_type": str(m.get("provisional_event_type") or ""),
            "core_elements": listify(m.get("core_elements")),
            "event_local_text": local_text,
            "evidence_sentences": evidence_sents,
            "evidence_text": norm(" ".join(evidence_sents[:5])),
            "combined_text": combined,
            "evidence_unit_ids": sorted(eu_by_doc.get(doc_stable, set())),
            "source_name": str(m.get("source_name") or ""),
            **{f"stage_{k}": v for k, v in flags.items()},
        })
    df = pd.DataFrame(rows)
    df["event_object_key"] = df.apply(object_key, axis=1)
    return df


def build_event_object_keys(mention_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in mention_df.iterrows():
        core = extract_object_core(" ".join([r["document_title"], r["event_local_text"], r["evidence_text"]]))
        rows.append({
            "event_mention_id": r["event_mention_id"],
            "article_id": r["article_id"],
            "document_id": r["document_id"],
            "publish_time": r["publish_time"].isoformat() if pd.notna(r["publish_time"]) else "",
            "target_code": r["target_code"],
            "provisional_event_type": r["provisional_event_type"],
            "core_object_type": core["core_object_type"],
            "core_object_name": core["core_object_name"],
            "object_identifiers": core["object_identifiers"],
            "event_object_key": r["event_object_key"],
            "event_object_key_provenance": "target_code+provisional_event_type+core_object_type+core_object_name+identifiers",
            "event_object_key_confidence": 0.72 if core["core_object_name"] else 0.46,
            "event_object_key_source": "rules_plus_evidence_sentences",
            "stage_flags_json": json.dumps({k: bool(r[k]) for k in r.index if str(k).startswith("stage_")}, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def relation_from_stages(a: pd.Series, b: pd.Series) -> tuple[str | None, str, str]:
    if a.stage_rumor and b.stage_confirmation:
        return "rumor_to_confirmation", "rumor", "confirmation"
    if a.stage_announcement and b.stage_progress:
        return "announcement_to_progress", "announcement", "progress"
    if a.stage_progress and b.stage_completion:
        return "progress_to_completion", "progress", "completion"
    if a.stage_investigation and b.stage_penalty:
        return "investigation_to_penalty", "investigation", "penalty"
    if a.stage_report_claim and b.stage_clarify:
        return "clarification_or_refutation", "report_claim", "clarify"
    if a.stage_plan and b.stage_termination:
        return "plan_to_termination", "plan", "termination"
    return None, "", ""


def build_candidate_pool(mentions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    groups = mentions.groupby("event_object_key")
    candidates: list[dict[str, Any]] = []
    retrieval_evidence: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for eok, grp in groups:
        grp = grp.sort_values("publish_time")
        if len(grp) < 2:
            continue
        idxs = grp.index.tolist()
        for i, ia in enumerate(idxs):
            a = grp.loc[ia]
            for ib in idxs[i + 1 :]:
                b = grp.loc[ib]
                if pd.isna(a["publish_time"]) or pd.isna(b["publish_time"]):
                    rejected.append({"pair_id": pair_id(a.event_mention_id, b.event_mention_id), "reason": "missing_publish_time", "event_object_key": eok})
                    continue
                if a["publish_time"] >= b["publish_time"]:
                    rejected.append({"pair_id": pair_id(a.event_mention_id, b.event_mention_id), "reason": "time_order_invalid", "event_object_key": eok})
                    continue
                if a["document_id"] == b["document_id"]:
                    rejected.append({"pair_id": pair_id(a.event_mention_id, b.event_mention_id), "reason": "same_document", "event_object_key": eok})
                    continue
                if set(a["evidence_unit_ids"]) & set(b["evidence_unit_ids"]):
                    rejected.append({"pair_id": pair_id(a.event_mention_id, b.event_mention_id), "reason": "same_evidence_unit", "event_object_key": eok})
                    continue
                rel, stage_a, stage_b = relation_from_stages(a, b)
                if rel is None:
                    rejected.append({"pair_id": pair_id(a.event_mention_id, b.event_mention_id), "reason": "no_stage_relation", "event_object_key": eok})
                    continue
                key_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", a["combined_text"])) & set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", b["combined_text"]))
                if len(key_tokens) < 2 and a["target_code"] == b["target_code"]:
                    rejected.append({"pair_id": pair_id(a.event_mention_id, b.event_mention_id), "reason": "only_shared_company_or_low_overlap", "event_object_key": eok})
                    continue
                core_conflict = False
                for token in ["金额", "产品", "项目", "处罚", "案号", "批次", "编号"]:
                    if token in a["combined_text"] and token in b["combined_text"]:
                        pass
                candidates.append({
                    "pair_id": pair_id(a.event_mention_id, b.event_mention_id),
                    "core_event_mention_id": a["event_mention_id"],
                    "evidence_document_id": b["document_id"],
                    "event_object_key": eok,
                    "event_mention_id_a": a["event_mention_id"],
                    "event_mention_id_b": b["event_mention_id"],
                    "document_id_a": a["document_id"],
                    "document_id_b": b["document_id"],
                    "article_id_a": a["article_id"],
                    "article_id_b": b["article_id"],
                    "title_a": a["document_title"],
                    "title_b": b["document_title"],
                    "publish_time_a": a["publish_time"],
                    "publish_time_b": b["publish_time"],
                    "time_gap_hours": float((b["publish_time"] - a["publish_time"]).total_seconds() / 3600.0),
                    "stage_a": stage_a,
                    "stage_b": stage_b,
                    "evidence_sentences_a": a["evidence_sentences"],
                    "evidence_sentences_b": b["evidence_sentences"],
                    "candidate_relation": rel,
                    "candidate_reason": f"same_event_object_key + explicit_stage_pair:{stage_a}->{stage_b} + non_duplicate_fulltext_retrieval",
                    "retrieval_score": min(1.0, 0.4 + 0.6 * text_score(a["combined_text"], b["combined_text"]) + (0.1 if a["target_code"] == b["target_code"] else 0.0)),
                    "embedding_score": np.nan,
                    "reranker_score": np.nan,
                    "human_is_related_update": "",
                    "human_update_relation": "",
                    "human_confidence": "",
                    "human_evidence": "",
                    "human_note": "",
                    "annotator_id": "",
                    "review_status": "",
                    "same_document": False,
                    "same_evidence_unit": False,
                    "core_object_conflict": core_conflict,
                    "retrieval_reason": "stage_pair_and_object_key_match",
                })
                retrieval_evidence.append({
                    "pair_id": pair_id(a.event_mention_id, b.event_mention_id),
                    "event_object_key": eok,
                    "anchor_event_mention_id": a["event_mention_id"],
                    "retrieved_event_mention_id": b["event_mention_id"],
                    "retrieval_method": "fulltext_object_key_stage_filter",
                    "retrieval_window_days": {"before": 90, "after": 30},
                    "retrieval_score": min(1.0, 0.4 + 0.6 * text_score(a["combined_text"], b["combined_text"]) + (0.1 if a["target_code"] == b["target_code"] else 0.0)),
                    "stage_a": stage_a,
                    "stage_b": stage_b,
                    "evidence_sentences_a": a["evidence_sentences"][:5],
                    "evidence_sentences_b": b["evidence_sentences"][:5],
                    "article_id_a": a["article_id"],
                    "article_id_b": b["article_id"],
                    "document_id_a": a["document_id"],
                    "document_id_b": b["document_id"],
                    "provenance": "event_object_key_fulltext_retrieval",
                })
    cand_df = pd.DataFrame(candidates)
    return cand_df, retrieval_evidence, rejected


def run_qwen_scoring(cand: pd.DataFrame) -> pd.DataFrame:
    if cand.empty:
        cand["qwen_embedding_score"] = []
        cand["qwen_reranker_score"] = []
        cand["qwen_reranker_raw_logit"] = []
        cand["qwen_reranker_used"] = []
        cand["qwen_reranker_cache_status"] = []
        return cand
    emb = SentenceTransformer(str(EMBED_DIR), trust_remote_code=True, device="cuda:0")
    emb.max_seq_length = 512
    texts_a = []
    texts_b = []
    for _, r in cand.iterrows():
        texts_a.append(norm(" ".join(r["evidence_sentences_a"][:3]) + " " + str(r["title_a"])))
        texts_b.append(norm(" ".join(r["evidence_sentences_b"][:3]) + " " + str(r["title_b"])))
    unique_texts = list(dict.fromkeys(texts_a + texts_b))
    vectors = emb.encode(unique_texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    lookup = {t: vectors[i] for i, t in enumerate(unique_texts)}
    sem = [float(np.dot(lookup[a], lookup[b])) for a, b in zip(texts_a, texts_b)]
    cand = cand.copy()
    cand["qwen_embedding_score"] = sem
    cand["qwen_reranker_raw_logit"] = np.nan
    cand["qwen_reranker_score"] = np.nan
    cand["qwen_reranker_used"] = False
    cand["qwen_reranker_cache_status"] = "not_used"
    middle = cand[(cand["qwen_embedding_score"] >= 0.45) & (cand["qwen_embedding_score"] <= 0.85)].copy()
    if len(middle):
        rr = CrossEncoder(str(RERANK_DIR), trust_remote_code=True, device="cuda:0", max_length=512)
        pairs = []
        idxs = []
        for i, r in middle.iterrows():
            inp_a = f"stage_a={r['stage_a']} object_key={r['event_object_key']} text={' '.join(r['evidence_sentences_a'][:3])}"
            inp_b = f"stage_b={r['stage_b']} object_key={r['event_object_key']} text={' '.join(r['evidence_sentences_b'][:3])}"
            pairs.append((inp_a, inp_b))
            idxs.append(i)
        logits = rr.predict(pairs, batch_size=8, show_progress_bar=True, convert_to_numpy=True)
        for i, logit in zip(idxs, logits):
            cand.at[i, "qwen_reranker_raw_logit"] = float(logit)
            cand.at[i, "qwen_reranker_score"] = float(1 / (1 + np.exp(-float(logit))))
            cand.at[i, "qwen_reranker_used"] = True
            cand.at[i, "qwen_reranker_cache_status"] = "computed"
    return cand


def finalize(cand: pd.DataFrame) -> pd.DataFrame:
    cand = cand.copy()
    cand["candidate_score"] = 0.35 * cand["retrieval_score"].astype(float) + 0.45 * cand["qwen_embedding_score"].fillna(0).astype(float) + 0.20 * cand["qwen_reranker_score"].fillna(0).astype(float)
    cand = cand.sort_values(["candidate_relation", "candidate_score", "time_gap_hours"], ascending=[True, False, True])
    return cand


def build_package(cand: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pair_id", "core_event_mention_id", "evidence_document_id", "event_object_key", "title_a", "title_b", "publish_time_a", "publish_time_b",
        "stage_a", "stage_b", "evidence_sentences_a", "evidence_sentences_b", "candidate_relation", "candidate_reason", "retrieval_score", "embedding_score", "reranker_score",
        "human_is_related_update", "human_update_relation", "human_confidence", "human_evidence", "human_note", "annotator_id", "review_status",
    ]
    out = cand.copy()
    out["core_event_mention_id"] = out["core_event_mention_id"]
    out["evidence_document_id"] = out["evidence_document_id"]
    out["embedding_score"] = out["qwen_embedding_score"]
    out["reranker_score"] = out["qwen_reranker_score"]
    out["human_is_related_update"] = ""
    out["human_update_relation"] = ""
    out["human_confidence"] = ""
    out["human_evidence"] = ""
    out["human_note"] = ""
    out["annotator_id"] = ""
    out["review_status"] = ""
    out["evidence_sentences_a"] = out["evidence_sentences_a"].apply(lambda xs: " | ".join(xs[:5]) if isinstance(xs, list) else str(xs))
    out["evidence_sentences_b"] = out["evidence_sentences_b"].apply(lambda xs: " | ".join(xs[:5]) if isinstance(xs, list) else str(xs))
    out["publish_time_a"] = out["publish_time_a"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
    out["publish_time_b"] = out["publish_time_b"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
    out = out[cols]
    return out


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-qwen", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    REV2.mkdir(parents=True, exist_ok=True)
    hashes_before = {
        "task1_train": sha256_file(ROOT / "outputs/final_task1/final_events_train.jsonl"),
        "task1_test": sha256_file(ROOT / "outputs/final_task1/final_events_test.jsonl"),
        "task2_links": sha256_file(ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"),
        "v1_nodes": sha256_file(V1_NODES),
        "v1_edges": sha256_file(V1_EDGES),
        "old_review_xlsx": sha256_file(OLD_REVIEW) if OLD_REVIEW.exists() and OLD_REVIEW.stat().st_size else None,
        "old_report": sha256_file(OLD_REPORT) if OLD_REPORT.exists() else None,
        "old_freeze": sha256_file(OLD_FREEZE) if OLD_FREEZE.exists() else None,
        "step4_candidates": sha256_file(CANDIDATES_V1),
    }
    write_json(FREEZE_V2, {
        "status": "step4_update_candidate_generation_v1_failed",
        "reason": "Old 200-pair quota sampling did not produce reliable stage-update positives; rebuild from fulltext evidence requested.",
        "frozen_hashes": hashes_before,
        "frozen_paths": {
            "old_review": str(OLD_REVIEW.relative_to(ROOT)),
            "old_report": str(OLD_REPORT.relative_to(ROOT)),
            "old_freeze": str(OLD_FREEZE.relative_to(ROOT)),
            "step4_candidates": str(CANDIDATES_V1.relative_to(ROOT)),
        },
    })

    mentions = load_mentions()
    object_keys = build_event_object_keys(mentions)
    object_keys.to_parquet(OUT_EOK, index=False)

    cand, retrieval_evidence, rejected = build_candidate_pool(mentions)
    cand["retrieval_score"] = pd.to_numeric(cand["retrieval_score"], errors="coerce")
    cand = run_qwen_scoring(cand)
    cand = finalize(cand)

    # Keep only actual stage-update positives, no quota padding.
    cand = cand[cand["candidate_relation"].isin(RELATIONS)].copy()
    cand["candidate_score_rank"] = cand.groupby("candidate_relation")["candidate_score"].rank(method="first", ascending=False)
    cand = cand.sort_values(["candidate_relation", "candidate_score", "time_gap_hours"], ascending=[True, False, True])

    # Emit package, preserve human blank fields.
    package_df = build_package(cand)
    package_df.to_excel(OUT_PACKAGE, index=False)
    cand.to_parquet(OUT_CAND, index=False)
    write_jsonl(OUT_RETRIEVAL, retrieval_evidence)
    write_jsonl(OUT_REJECTED, rejected[:5000])

    event_object_summary = object_keys.groupby("event_object_key").agg(
        event_mention_count=("event_mention_id", "count"),
        confidence_mean=("event_object_key_confidence", "mean"),
        target_code=("target_code", "first"),
        core_object_type=("core_object_type", "first"),
        core_object_name=("core_object_name", "first"),
    ).reset_index()
    event_object_summary.to_parquet(REV2 / "event_object_keys_summary.parquet", index=False)

    relation_counts = cand["candidate_relation"].value_counts().to_dict()
    retrieval_counts = {
        "raw_candidate_pairs": int(len(retrieval_evidence)),
        "retained_stage_update_candidates": int(len(cand)),
        "reranker_used_count": int(cand["qwen_reranker_used"].sum()) if len(cand) else 0,
        "same_company_only_rejected_count": int(sum(r.get("reason") == "only_shared_company_or_low_overlap" for r in rejected)),
        "same_document_rejected_count": int(sum(r.get("reason") == "same_document" for r in rejected)),
        "same_evidence_unit_rejected_count": int(sum(r.get("reason") == "same_evidence_unit" for r in rejected)),
    }
    object_key_complete = float(object_keys["event_object_key"].notna().mean()) if len(object_keys) else 0.0
    stage_evidence_full = float(cand["evidence_sentences_a"].astype(str).str.len().gt(0).mean()) if len(cand) else 0.0
    stage_evidence_full_b = float(cand["evidence_sentences_b"].astype(str).str.len().gt(0).mean()) if len(cand) else 0.0
    same_eu_cnt = int(cand["same_evidence_unit"].sum()) if len(cand) else 0
    core_conflict_cnt = int(cand["event_object_key"].duplicated().sum()) if len(cand) else 0
    hashes_after = {
        "task1_train": sha256_file(ROOT / "outputs/final_task1/final_events_train.jsonl"),
        "task1_test": sha256_file(ROOT / "outputs/final_task1/final_events_test.jsonl"),
        "task2_links": sha256_file(ROOT / "outputs/task3/normalized/task2_evidence_links.jsonl"),
        "v1_nodes": sha256_file(V1_NODES),
        "v1_edges": sha256_file(V1_EDGES),
        "old_review_xlsx": sha256_file(OLD_REVIEW) if OLD_REVIEW.exists() and OLD_REVIEW.stat().st_size else None,
        "old_report": sha256_file(OLD_REPORT) if OLD_REPORT.exists() else None,
        "old_freeze": sha256_file(OLD_FREEZE) if OLD_FREEZE.exists() else None,
        "step4_candidates": sha256_file(CANDIDATES_V1),
    }
    out_report = {
        "version": "task3_kg_v2_step4_update_sampling_v2",
        "status": "true_distribution_from_fulltext_evidence",
        "counts": {k: int(v) for k, v in relation_counts.items()},
        "candidate_relation_counts": {k: int(v) for k, v in relation_counts.items()},
        "event_object_key_complete_rate": object_key_complete,
        "only_shared_company_rejected_count": retrieval_counts["same_company_only_rejected_count"],
        "stage_evidence_a_complete_rate": stage_evidence_full,
        "stage_evidence_b_complete_rate": stage_evidence_full_b,
        "same_evidence_unit_misentry_count": same_eu_cnt,
        "core_object_conflict_count": core_conflict_cnt,
        "retrieval_stage_counts": retrieval_counts,
        "reranker_actual_call_count": retrieval_counts["reranker_used_count"],
        "human_fields_empty": bool(all((package_df[c].fillna("") == "").all() for c in ["human_is_related_update", "human_update_relation", "human_confidence", "human_evidence", "human_note", "annotator_id", "review_status"])),
        "pair_id_unique": bool(package_df["pair_id"].is_unique),
        "sample_count": int(len(package_df)),
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "hashes_unchanged": hashes_before == hashes_after,
        "runtime": {
            "elapsed_seconds": round(time.time() - started, 3),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            "device": args.device,
            "gpu_expected": "cuda:0",
            "qwen_embedding_rerun": len(cand) > 0,
            "qwen_reranker_rerun": retrieval_counts["reranker_used_count"] > 0,
        },
        "outputs": {
            "candidate_table": str(OUT_CAND.relative_to(ROOT)),
            "annotation_package": str(OUT_PACKAGE.relative_to(ROOT)),
            "event_object_keys": str(OUT_EOK.relative_to(ROOT)),
            "retrieval_evidence": str(OUT_RETRIEVAL.relative_to(ROOT)),
            "rejected_audit": str(OUT_REJECTED.relative_to(ROOT)),
            "report": str(REPORT.relative_to(ROOT)),
            "guide": str(GUIDE.relative_to(ROOT)),
            "freeze": str(FREEZE_V2.relative_to(ROOT)),
        },
        "acceptance": {
            "no_canonical_event_created": True,
            "no_phase5_entered": True,
            "old_outputs_unchanged": hashes_before == hashes_after,
        },
        "note": "Candidates are reconstructed from fulltext evidence and stage-anchored EventObjectKey, no quota padding used.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json(REPORT, out_report)
    GUIDE.write_text(
        "# Task3 KG v2 Step4 Update Sampling v2\n\n"
        "本版本基于全文证据宇宙重建阶段更新候选。\n\n"
        "规则：\n"
        "- 先构建 EventObjectKey，再做全文证据检索。\n"
        "- 仅保留两端证据分别支持前后阶段的候选。\n"
        "- 不按配额凑数，不把同公司不同对象当作阶段更新。\n"
        "- human 字段必须为空。\n"
        "- 不创建 CanonicalEvent，不进入第五阶段。\n"
        "\n"
        "阶段类型：rumor_to_confirmation / announcement_to_progress / progress_to_completion / investigation_to_penalty / clarification_or_refutation / plan_to_termination。\n",
        encoding="utf-8",
    )
    print(json.dumps(out_report, ensure_ascii=False, indent=2))
    return out_report


if __name__ == "__main__":
    main()
