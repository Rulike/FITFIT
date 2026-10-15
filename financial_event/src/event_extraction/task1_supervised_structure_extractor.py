#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gold-free supervised structure extractor for Task1 EventMention rows."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib


DIR_LABELS = ["positive", "negative", "neutral", "unclear"]
ROLE_LABELS = ["subject_company", "event_time", "object_target", "trigger"]
SPAN_FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
ALLOWED_INFERENCE_FIELDS = ["content", "event_name", "title"]

ACTION_WORDS = [
    "发布",
    "推出",
    "上线",
    "启动",
    "落地",
    "签署",
    "签约",
    "合作",
    "中标",
    "获批",
    "获准",
    "增长",
    "下降",
    "上涨",
    "下跌",
    "增产",
    "减产",
    "扩产",
    "投产",
    "停产",
    "涨价",
    "降价",
    "突破",
    "创新",
    "并购",
    "收购",
    "投资",
    "融资",
    "上市",
    "开工",
    "竣工",
    "交付",
    "成立",
    "建设",
    "升级",
    "调整",
    "召回",
    "处罚",
    "亏损",
    "盈利",
    "扭亏",
    "预增",
    "预减",
    "供货",
    "采购",
    "恢复",
    "维持",
    "宣布",
    "完成",
    "提高",
    "减少",
    "增加",
    "扩大",
    "降低",
    "提升",
    "进入",
    "投建",
    "招标",
    "备案",
    "通过",
    "实施",
    "推进",
    "加快",
    "实现",
    "计划",
    "预计",
]

POSITIVE_WORDS = ["增长", "上涨", "突破", "获批", "中标", "盈利", "扭亏", "成功", "提升", "扩产", "投产", "签约", "交付", "增产", "创新"]
NEGATIVE_WORDS = ["下降", "下跌", "亏损", "处罚", "召回", "减产", "停产", "风险", "下滑", "取消", "终止", "降价"]
NEUTRAL_WORDS = ["发布", "宣布", "维持", "召开", "讨论", "计划", "预计", "拟", "称", "表示"]


@dataclass
class Candidate:
    field: str
    text: str
    start: int
    end: int
    source: str
    context: str
    sentence_index: int

    def as_span(self, publish_time: str = "") -> dict[str, Any]:
        if not self.text:
            if self.field == "event_time":
                return {"text": "", "normalized": "未明确", "start": -1, "end": -1}
            return {"text": "", "start": -1, "end": -1}
        span: dict[str, Any] = {"text": self.text, "start": self.start, "end": self.end}
        if self.field == "event_time":
            span["normalized"] = normalize_time(self.text, publish_time)
        return span


class ConstantBinaryModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, values: list[str]) -> list[list[float]]:
        p = self.probability
        return [[1.0 - p, p] for _ in values]


class ConstantDirectionModel:
    def __init__(self, label: str) -> None:
        self.label = label if label in DIR_LABELS else "unclear"

    def predict(self, values: list[str]) -> list[str]:
        return [self.label for _ in values]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_sha(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def span_empty(field: str = "") -> dict[str, Any]:
    if field == "event_time":
        return {"text": "", "normalized": "未明确", "start": -1, "end": -1}
    return {"text": "", "start": -1, "end": -1}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", "", text.lower())
    text = re.sub(r"[，。；：、,.!！?？（）()《》“”\"'`·\-—_<>【】\[\]{}]", "", text)
    for suffix in ["股份有限公司", "有限责任公司", "有限公司", "集团股份", "集团", "公司"]:
        text = text.replace(suffix, "")
    return text


def normalize_time(text: str, publish_time: str = "") -> str:
    value = str(text or "")
    if not value:
        return "未明确"
    year = ""
    m_pub = re.search(r"(20\d{2})", str(publish_time or ""))
    if m_pub:
        year = m_pub.group(1)
    m = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?", value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})月(\d{1,2})日", value)
    if m and year:
        return f"{int(year):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.search(r"(20\d{2})年(?:第一|第二|第三|第四|一|二|三|四)?季度", value)
    if m:
        qmap = {"第一": "01", "一": "01", "第二": "04", "二": "04", "第三": "07", "三": "07", "第四": "10", "四": "10"}
        q = re.search(r"(第一|第二|第三|第四|一|二|三|四)", value)
        return f"{m.group(1)}-{qmap.get(q.group(1) if q else '', '01')}-01"
    if any(word in value for word in ["近日", "日前", "当日", "当天", "今日", "近期", "今年", "未来", "上半年", "下半年"]):
        return value
    return value


def project_for_inference(content: str, event_name: str, title: str | None = None) -> dict[str, str]:
    return {"content": str(content or ""), "event_name": str(event_name or ""), "title": str(title or "")}


def split_sentences(content: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    start = 0
    for idx, ch in enumerate(content):
        if ch in "。！？!?；;\n":
            end = idx + 1
            segment = content[start:end]
            stripped = segment.strip()
            if stripped:
                lead = len(segment) - len(segment.lstrip())
                trail = len(segment[start + lead - start :]) - len(segment[start + lead - start :].rstrip())
                real_start = start + lead
                real_end = end - trail
                if real_start < real_end:
                    sentences.append({"index": len(sentences), "text": content[real_start:real_end], "start": real_start, "end": real_end})
            start = end
    if start < len(content):
        segment = content[start:]
        stripped = segment.strip()
        if stripped:
            lead = len(segment) - len(segment.lstrip())
            real_start = start + lead
            real_end = len(content) - (len(segment) - len(segment.rstrip()))
            if real_start < real_end:
                sentences.append({"index": len(sentences), "text": content[real_start:real_end], "start": real_start, "end": real_end})
    if not sentences and content:
        sentences.append({"index": 0, "text": content, "start": 0, "end": len(content)})
    return sentences


def sentence_for(sentences: list[dict[str, Any]], start: int, end: int) -> dict[str, Any] | None:
    for sent in sentences:
        if sent["start"] <= start < end <= sent["end"]:
            return sent
    return None


def context_slice(content: str, start: int, end: int, radius: int) -> str:
    return content[max(0, start - radius) : min(len(content), end + radius)]


def add_candidate(items: list[Candidate], seen: set[tuple[str, int, int]], field: str, content: str, start: int, end: int, source: str, sentence_index: int, context_radius: int) -> None:
    if not (0 <= start <= end <= len(content)):
        return
    text = content[start:end]
    if not text:
        return
    key = (field, start, end)
    if key in seen:
        return
    seen.add(key)
    items.append(Candidate(field, text, start, end, source, context_slice(content, start, end, context_radius), sentence_index))


def event_terms(event_name: str) -> list[str]:
    terms = [str(event_name or "")]
    for sep in ["与", "及", "和", "、", "/", "+", "（", "）", "(", ")", "，"]:
        for part in str(event_name or "").split(sep):
            part = part.strip()
            if 2 <= len(part) <= 8:
                terms.append(part)
    return list(dict.fromkeys(terms))


def generate_candidates(content: str, event_name: str, title: str = "", scheme: str = "A") -> dict[str, list[Candidate]]:
    content = str(content or "")
    sentences = split_sentences(content)
    radius = {"A": 48, "B": 96, "C": 72}.get(scheme, 64)
    output: dict[str, list[Candidate]] = {field: [] for field in SPAN_FIELDS}
    seen: set[tuple[str, int, int]] = set()

    for idx, sent in enumerate(sentences):
        add_candidate(output["event_span"], seen, "event_span", content, sent["start"], sent["end"], "single_sentence", idx, radius)
        if idx + 1 < len(sentences):
            nxt = sentences[idx + 1]
            if nxt["start"] - sent["end"] <= 3:
                add_candidate(output["event_span"], seen, "event_span", content, sent["start"], nxt["end"], "two_sentences", idx, radius)
        if scheme == "C" and idx + 2 < len(sentences):
            nxt2 = sentences[idx + 2]
            if nxt2["start"] - sent["end"] <= 6:
                add_candidate(output["event_span"], seen, "event_span", content, sent["start"], nxt2["end"], "three_sentences", idx, radius)

    trigger_words = list(dict.fromkeys(ACTION_WORDS + event_terms(event_name)))
    for word in trigger_words:
        if len(word) < 2:
            continue
        start = 0
        while True:
            hit = content.find(word, start)
            if hit < 0:
                break
            end = hit + len(word)
            sent = sentence_for(sentences, hit, end)
            sent_idx = int(sent["index"]) if sent else -1
            add_candidate(output["trigger"], seen, "trigger", content, hit, end, "action_word", sent_idx, radius)
            left_bound = max(content.rfind(ch, 0, hit) for ch in "。！？!?；;\n，,、")
            left = max(left_bound + 1, hit - 8, 0)
            right_candidates = [content.find(ch, end) for ch in "。！？!?；;\n，,、" if content.find(ch, end) >= 0]
            right_bound = min(right_candidates) if right_candidates else min(len(content), end + 32)
            right = min(right_bound, end + 32)
            if 2 <= right - left <= 80:
                add_candidate(output["trigger"], seen, "trigger", content, left, right, "action_phrase", sent_idx, radius)
            start = end

    subject_patterns = [
        r"[\u4e00-\u9fa5A-Za-z0-9·]{2,32}(?:股份有限公司|有限责任公司|有限公司|集团|公司|银行|证券|基金|能源|科技|材料|汽车|药业|电力|股份)",
        r"(?:OPEC\+?|欧佩克\+?|工信部|发改委|国家能源局|商务部|证监会|上交所|深交所|财政部|交通运输部|生态环境部)",
    ]
    for pattern in subject_patterns:
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            sent = sentence_for(sentences, match.start(), match.end())
            add_candidate(output["subject_company"], seen, "subject_company", content, match.start(), match.end(), "entity_pattern", int(sent["index"]) if sent else -1, radius)
    output["subject_company"].append(Candidate("subject_company", "", -1, -1, "empty_option", "", -1))

    time_patterns = [
        r"20\d{2}年(?:第一|第二|第三|第四|一|二|三|四)?季度",
        r"20\d{2}[年/-]\d{1,2}[月/-]\d{1,2}日?",
        r"\d{1,2}月\d{1,2}日",
        r"20\d{2}年\d{1,2}月",
        r"(?:近日|日前|当日|当天|今日|近期|今年|未来|上半年|下半年)",
    ]
    for pattern in time_patterns:
        for match in re.finditer(pattern, content):
            sent = sentence_for(sentences, match.start(), match.end())
            add_candidate(output["event_time"], seen, "event_time", content, match.start(), match.end(), "time_pattern", int(sent["index"]) if sent else -1, radius)
    output["event_time"].append(Candidate("event_time", "", -1, -1, "empty_option", "", -1))

    object_patterns = [
        r"[\u4e00-\u9fa5A-Za-z0-9《》“”\"'·\-]{3,45}(?:项目|产品|技术|基地|市场|政策|方案|白皮书|订单|产能|价格|业务|平台|车型|股份|股权|装置|系统|电池|组件|材料|服务|合同|协议)",
        r"[“\"《][^”\"》]{2,40}[”\"》]",
    ]
    for pattern in object_patterns:
        for match in re.finditer(pattern, content):
            sent = sentence_for(sentences, match.start(), match.end())
            add_candidate(output["object_target"], seen, "object_target", content, match.start(), match.end(), "object_pattern", int(sent["index"]) if sent else -1, radius)
    for word in ACTION_WORDS:
        start = 0
        while True:
            hit = content.find(word, start)
            if hit < 0:
                break
            end = hit + len(word)
            right_candidates = [content.find(ch, end) for ch in "。！？!?；;\n，,、" if content.find(ch, end) >= 0]
            right = min(right_candidates) if right_candidates else min(len(content), end + 40)
            if 3 <= right - end <= 60:
                sent = sentence_for(sentences, hit, right)
                add_candidate(output["object_target"], seen, "object_target", content, end, right, "post_action_phrase", int(sent["index"]) if sent else -1, radius)
            start = end
    output["object_target"].append(Candidate("object_target", "", -1, -1, "empty_option", "", -1))

    caps = {"event_span": 260, "trigger": 260, "subject_company": 180, "event_time": 100, "object_target": 220}
    for field, cap in caps.items():
        nonempty = [c for c in output[field] if c.text]
        empty = [c for c in output[field] if not c.text]
        nonempty.sort(key=lambda c: (0 if c.start < 0 else c.start, len(c.text)))
        output[field] = nonempty[:cap] + empty
    return output


def feature_text(row: dict[str, str], candidate: Candidate, scheme: str) -> str:
    title = row.get("title", "")
    event_name = row.get("event_name", "")
    length_bin = min(20, max(0, len(candidate.text) // 5))
    pos_bin = 0 if candidate.start < 0 else min(20, candidate.start // 250)
    query = {
        "trigger": "event trigger action",
        "event_span": "event sentence span",
        "subject_company": "event subject company organization",
        "event_time": "event time date",
        "object_target": "event object target product project market",
    }.get(candidate.field, candidate.field)
    prefix = f"scheme={scheme} field={candidate.field} query={query} event={event_name} source={candidate.source} len={length_bin} pos={pos_bin} title={title} "
    if scheme == "B":
        return prefix + f"question={event_name}{query} candidate={candidate.text} context={candidate.context}"
    if scheme == "C":
        return prefix + f"candidate={candidate.text} context={candidate.context} event_terms={' '.join(event_terms(event_name))}"
    return prefix + f"candidate={candidate.text} context={candidate.context}"


def candidate_overlap(candidate: Candidate, span: dict[str, Any]) -> float:
    if not candidate.text and not span.get("text"):
        return 1.0
    if not candidate.text or not span.get("text"):
        return 0.0
    c_set = set(range(candidate.start, candidate.end))
    g_set = set(range(int(span.get("start", -1)), int(span.get("end", -1))))
    if not c_set or not g_set:
        return 0.0
    overlap = len(c_set & g_set)
    p = overlap / len(c_set)
    r = overlap / len(g_set)
    return 2 * p * r / (p + r) if p + r else 0.0


def label_candidates(candidates: list[Candidate], span: dict[str, Any], field: str) -> list[int]:
    if not span.get("text"):
        return [1 if not cand.text else 0 for cand in candidates]
    labels = [1 if cand.text == span.get("text") and cand.start == span.get("start") and cand.end == span.get("end") else 0 for cand in candidates]
    if any(labels):
        return labels
    best_idx = -1
    best_score = 0.0
    for idx, cand in enumerate(candidates):
        score = candidate_overlap(cand, span)
        if score > best_score:
            best_idx = idx
            best_score = score
    threshold = 0.35 if field == "event_span" else 0.55
    if best_idx >= 0 and best_score >= threshold:
        labels[best_idx] = 1
    return labels


class CandidateRankerExtractor:
    def __init__(self, scheme: str, field_models: dict[str, Any], direction_model: Any, thresholds: dict[str, float]) -> None:
        self.scheme = scheme
        self.field_models = field_models
        self.direction_model = direction_model
        self.thresholds = thresholds

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "CandidateRankerExtractor":
        return joblib.load(path)

    def _score_candidates(self, field: str, row: dict[str, str], candidates: list[Candidate]) -> list[tuple[float, Candidate]]:
        if not candidates:
            return []
        values = [feature_text(row, cand, self.scheme) for cand in candidates]
        model = self.field_models[field]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(values)
            scores = [float(prob[1]) for prob in probs]
        else:
            scores = [0.0 for _ in values]
        return sorted(zip(scores, candidates), key=lambda item: (item[0], len(item[1].text), -max(item[1].start, 0)), reverse=True)

    def _choose_field(self, field: str, row: dict[str, str], candidates: list[Candidate]) -> dict[str, Any]:
        scored = self._score_candidates(field, row, candidates)
        if not scored:
            return span_empty(field)
        score, candidate = scored[0]
        if field in {"subject_company", "event_time", "object_target"} and (not candidate.text or score < self.thresholds.get(field, 0.0)):
            return span_empty(field)
        return candidate.as_span(row.get("publish_time", ""))

    def _predict_direction(self, row: dict[str, str], spans: dict[str, dict[str, Any]]) -> str:
        text = (
            f"event={row.get('event_name','')} title={row.get('title','')} "
            f"trigger={spans['trigger'].get('text','')} span={spans['event_span'].get('text','')} "
            f"head={row.get('content','')[:1200]} tail={row.get('content','')[-600:]}"
        )
        if hasattr(self.direction_model, "predict"):
            label = str(self.direction_model.predict([text])[0])
            if label in DIR_LABELS:
                return label
        joined = spans["event_span"].get("text", "") + spans["trigger"].get("text", "")
        if any(word in joined for word in POSITIVE_WORDS):
            return "positive"
        if any(word in joined for word in NEGATIVE_WORDS):
            return "negative"
        if any(word in joined for word in NEUTRAL_WORDS):
            return "neutral"
        return "unclear"

    def predict_structure(self, content: str, event_name: str, title: str | None = None) -> dict[str, Any]:
        row = project_for_inference(content, event_name, title)
        row["publish_time"] = ""
        candidates = generate_candidates(row["content"], row["event_name"], row["title"], self.scheme)
        spans = {field: self._choose_field(field, row, candidates[field]) for field in SPAN_FIELDS}
        spans = enforce_consistency(row["content"], spans, candidates.get("event_span", []), self.scheme)
        direction = self._predict_direction(row, spans)
        arguments = [
            {"text": spans[role]["text"], "role": role, "start": spans[role]["start"], "end": spans[role]["end"]}
            for role in ROLE_LABELS
            if spans[role].get("text")
        ]
        return {
            **spans,
            "direction": direction,
            "arguments": arguments,
            "argument_roles": list(arguments),
        }


def sentence_span_containing(content: str, start: int, end: int) -> dict[str, Any]:
    for sent in split_sentences(content):
        if sent["start"] <= start < end <= sent["end"]:
            return {"text": content[sent["start"] : sent["end"]], "start": sent["start"], "end": sent["end"]}
    return span_empty("event_span")


def enforce_consistency(content: str, spans: dict[str, dict[str, Any]], span_candidates: list[Candidate], scheme: str) -> dict[str, dict[str, Any]]:
    trigger = spans.get("trigger", span_empty())
    event_span = spans.get("event_span", span_empty("event_span"))
    if not trigger.get("text"):
        return spans
    if event_span.get("text") and event_span["start"] <= trigger["start"] < trigger["end"] <= event_span["end"]:
        return spans
    containing = [cand for cand in span_candidates if cand.text and cand.start <= trigger["start"] < trigger["end"] <= cand.end]
    if containing:
        chosen = min(containing, key=lambda cand: (abs(len(cand.text) - max(20, len(trigger["text"]) * 8)), len(cand.text)))
        spans["event_span"] = chosen.as_span()
        return spans
    sent_span = sentence_span_containing(content, trigger["start"], trigger["end"])
    if sent_span.get("text"):
        spans["event_span"] = sent_span
        return spans
    if event_span.get("text"):
        start = min(event_span["start"], trigger["start"])
        end = max(event_span["end"], trigger["end"])
        if 0 <= start <= end <= len(content):
            spans["event_span"] = {"text": content[start:end], "start": start, "end": end}
    return spans


def build_extractor(train_rows: list[dict[str, Any]], scheme: str, thresholds: dict[str, float]) -> tuple[CandidateRankerExtractor, dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    field_models: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for field in SPAN_FIELDS:
        values: list[str] = []
        labels: list[int] = []
        exact_hits = 0
        positive_rows = 0
        candidate_total = 0
        for row in train_rows:
            allowed = project_for_inference(row["content"], row["event_name"], row.get("title", ""))
            candidates = generate_candidates(allowed["content"], allowed["event_name"], allowed["title"], scheme)[field]
            candidate_total += len(candidates)
            row_labels = label_candidates(candidates, row[field], field)
            if any(label == 1 for label in row_labels):
                positive_rows += 1
            if row[field].get("text") and any(c.text == row[field].get("text") and c.start == row[field].get("start") and c.end == row[field].get("end") for c in candidates):
                exact_hits += 1
            for cand, label in zip(candidates, row_labels):
                values.append(feature_text(allowed, cand, scheme))
                labels.append(label)
        if len(set(labels)) < 2:
            field_models[field] = ConstantBinaryModel(sum(labels) / max(1, len(labels)))
        else:
            ngram = (1, 4) if scheme in {"B", "C"} else (2, 4)
            model = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=ngram, min_df=1, max_features=80000, sublinear_tf=True)),
                    ("clf", LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear", random_state=2026)),
                ]
            )
            model.fit(values, labels)
            field_models[field] = model
        coverage[field] = {
            "train_rows_with_supervised_positive": positive_rows,
            "train_exact_candidate_hits": exact_hits,
            "train_candidate_total": candidate_total,
        }

    direction_values = [
        f"event={row['event_name']} title={row.get('title','')} head={row['content'][:1200]} tail={row['content'][-600:]}"
        for row in train_rows
    ]
    direction_labels = [row["direction"] if row["direction"] in DIR_LABELS else "unclear" for row in train_rows]
    if len(set(direction_labels)) < 2:
        direction_model: Any = ConstantDirectionModel(direction_labels[0] if direction_labels else "unclear")
    else:
        direction_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1, max_features=100000, sublinear_tf=True)),
                ("clf", OneVsRestClassifier(LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear", random_state=2026))),
            ]
        )
        direction_model.fit(direction_values, direction_labels)
    return CandidateRankerExtractor(scheme, field_models, direction_model, thresholds), {"field_candidate_coverage": coverage}


def predict_batch(model: CandidateRankerExtractor, rows: list[dict[str, Any]], include_ids: bool = True) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        pred = model.predict_structure(row.get("content", ""), row.get("event_name", ""), row.get("title", ""))
        if include_ids:
            pred = {"event_mention_id": row.get("event_mention_id", ""), "article_id": row.get("article_id", ""), "scheme": model.scheme, **pred}
        predictions.append(pred)
    return predictions


def validate_offsets(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, int]:
    by_id = {row.get("event_mention_id"): row for row in rows}
    invalid = 0
    containment = 0
    bad_direction = 0
    for pred in predictions:
        row = by_id.get(pred.get("event_mention_id"), {})
        content = row.get("content", "")
        for field in SPAN_FIELDS:
            span = pred.get(field, {})
            text = str(span.get("text", "") or "")
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
            if text:
                if not (0 <= start <= end <= len(content)) or content[start:end] != text:
                    invalid += 1
            elif start != -1 or end != -1:
                invalid += 1
        trigger = pred.get("trigger", {})
        event_span = pred.get("event_span", {})
        if trigger.get("text") and event_span.get("text") and not (event_span["start"] <= trigger["start"] < trigger["end"] <= event_span["end"]):
            containment += 1
        if pred.get("direction") not in DIR_LABELS:
            bad_direction += 1
    return {"offset_illegal_count": invalid, "trigger_not_in_event_span_count": containment, "illegal_direction_count": bad_direction}
