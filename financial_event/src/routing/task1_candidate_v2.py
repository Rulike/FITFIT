#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Candidate generator v2 for Task1 64-class event-name recall."""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STOP_TERMS = {
    "事件",
    "定义",
    "示例",
    "公司",
    "企业",
    "行业",
    "市场",
    "发展",
    "变化",
    "影响",
    "进行",
    "通过",
    "关注",
    "重点",
    "包括",
    "相关",
    "重大",
    "具体",
    "动态",
    "例如",
    "直接",
    "间接",
    "业务",
    "产品",
    "技术",
    "政策",
    "调整",
    "创新",
    "提升",
    "推动",
    "风险",
    "能力",
    "情况",
    "领域",
    "方面",
    "形成",
    "实现",
    "监测",
    "用于",
    "以及",
    "需要",
    "主要",
    "核心",
    "可能",
    "持续",
    "新型",
    "进一步",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.lower()
    return re.sub(r"\s+", " ", value).strip()


def extract_terms(text: Any, stop_terms: set[str] | None = None) -> list[str]:
    stop = stop_terms or DEFAULT_STOP_TERMS
    value = normalize_text(text)
    terms: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fffA-Za-z0-9+]{2,}", value):
        if seg not in stop:
            terms.append(seg)
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", value):
        for n in (2, 3, 4):
            if len(run) < n:
                continue
            for idx in range(len(run) - n + 1):
                gram = run[idx : idx + n]
                if gram not in stop:
                    terms.append(gram)
    return terms


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


class Task1CandidateGeneratorV2:
    """Generate event-name candidates from official mappings plus train-only semantic profiles."""

    source_order = [
        "official_target_mapping",
        "official_industry_mapping",
        "schema_match",
        "title_semantic",
        "body_semantic",
        "long_tail_compensation",
        "all64_fallback",
    ]

    def __init__(
        self,
        schema_rows: list[dict[str, Any]],
        train_articles: list[dict[str, Any]],
        train_gold_pairs: list[dict[str, Any]],
        target_aliases: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.schema_rows = list(schema_rows)
        self.train_articles = {str(row["article_id"]): row for row in train_articles}
        self.train_gold_pairs = list(train_gold_pairs)
        self.target_aliases = target_aliases or {}
        self.config = config or {}
        self.stop_terms = set(DEFAULT_STOP_TERMS) | set(self.config.get("stop_terms", []))
        self.event_names = sorted({str(row["event_name"]) for row in self.schema_rows})
        self.schema_by_id = {str(row["schema_id"]): row for row in self.schema_rows}
        self.schemas_by_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.schemas_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.schema_rows:
            target_key = (str(row["target_type"]), str(row["target_code"]))
            self.schemas_by_target[target_key].append(row)
            self.schemas_by_event[str(row["event_name"])].append(row)
        for rows in self.schemas_by_target.values():
            rows.sort(key=lambda item: str(item["schema_id"]))
        for rows in self.schemas_by_event.values():
            rows.sort(key=lambda item: str(item["schema_id"]))

        self.train_support = Counter(
            str(row["event_name"])
            for row in self.train_gold_pairs
            if str(row.get("article_id", "")) in self.train_articles
        )
        self.event_aliases = self._build_event_aliases()
        self.target_lookup = self._build_target_lookup()
        self.event_profiles, self.event_keywords = self._build_event_profiles()
        self.idf = self._build_idf()
        self.event_vectors = self._build_event_vectors()

    @classmethod
    def from_paths(
        cls,
        schema_path: Path,
        train_articles: list[dict[str, Any]],
        train_gold_pairs: list[dict[str, Any]],
        alias_path: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> "Task1CandidateGeneratorV2":
        aliases = read_json(alias_path) if alias_path and alias_path.exists() else {}
        return cls(read_jsonl(schema_path), train_articles, train_gold_pairs, aliases, config)

    def _build_event_aliases(self) -> dict[str, set[str]]:
        configured = self.config.get("event_aliases", {})
        aliases: dict[str, set[str]] = defaultdict(set)
        for event_name in self.event_names:
            aliases[event_name].add(event_name)
            stripped = re.sub(r"[（(].*?[）)]", "", event_name).strip()
            if len(stripped) >= 2:
                aliases[event_name].add(stripped)
            for part in re.split(r"[（(、/，,及与和等\s]+", event_name):
                part = part.strip("）) ")
                if len(part) >= 2 and part not in self.stop_terms:
                    aliases[event_name].add(part)
            for alias in configured.get(event_name, []):
                if str(alias).strip():
                    aliases[event_name].add(str(alias).strip())
        return aliases

    def _build_target_lookup(self) -> dict[str, list[dict[str, Any]]]:
        targets: dict[tuple[str, str], str] = {}
        for row in self.schema_rows:
            targets[(str(row["target_type"]), str(row["target_code"]))] = str(row["target_name"])
        lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (target_type, target_code), target_name in targets.items():
            key = normalize_text(target_name)
            if len(key) >= 2:
                lookup[key].append(
                    {
                        "target_type": target_type,
                        "target_code": target_code,
                        "target_name": target_name,
                        "alias": target_name,
                        "is_formal": True,
                    }
                )
        for target_type in ("company", "industry"):
            for target_name, info in self.target_aliases.get(target_type, {}).items():
                code = str(info.get("code", ""))
                for alias in info.get("aliases", []):
                    key = normalize_text(alias)
                    if len(key) >= 2:
                        lookup[key].append(
                            {
                                "target_type": target_type,
                                "target_code": code,
                                "target_name": str(target_name),
                                "alias": str(alias),
                                "is_formal": False,
                            }
                        )
        return lookup

    def _build_event_profiles(self) -> tuple[dict[str, Counter[str]], dict[str, list[str]]]:
        profiles: dict[str, Counter[str]] = defaultdict(Counter)
        for row in self.schema_rows:
            event_name = str(row["event_name"])
            profile_text = f"{event_name} {row.get('raw_description', '')}"
            for term in extract_terms(profile_text, self.stop_terms):
                profiles[event_name][term] += 1
            for alias in self.event_aliases[event_name]:
                for term in extract_terms(alias, self.stop_terms):
                    profiles[event_name][term] += 4
        for row in self.train_gold_pairs:
            aid = str(row.get("article_id", ""))
            event_name = str(row.get("event_name", ""))
            article = self.train_articles.get(aid)
            if not article or event_name not in self.event_names:
                continue
            for term in extract_terms(article.get("title", ""), self.stop_terms):
                profiles[event_name][term] += 5
            body_text = str(article.get("content", ""))[: int(self.config.get("profile_body_chars", 3000))]
            for term in extract_terms(body_text, self.stop_terms):
                profiles[event_name][term] += 1
        max_terms = int(self.config.get("profile_max_terms", 300))
        top_profiles = {name: Counter(dict(counter.most_common(max_terms))) for name, counter in profiles.items()}
        keywords = {
            name: [term for term, _ in counter.most_common(int(self.config.get("event_keyword_count", 40)))]
            for name, counter in top_profiles.items()
        }
        return top_profiles, keywords

    def _build_idf(self) -> dict[str, float]:
        df = Counter()
        for counter in self.event_profiles.values():
            for term in counter:
                df[term] += 1
        total = len(self.event_names)
        return {term: math.log((1 + total) / (1 + count)) + 1.0 for term, count in df.items()}

    def _build_event_vectors(self) -> dict[str, dict[str, float]]:
        vectors: dict[str, dict[str, float]] = {}
        for event_name, counter in self.event_profiles.items():
            values = {term: weight * self.idf.get(term, 1.0) for term, weight in counter.items()}
            denom = math.sqrt(sum(value * value for value in values.values())) or 1.0
            vectors[event_name] = {term: value / denom for term, value in values.items()}
        return vectors

    def _article_vector(self, text: str) -> dict[str, float]:
        counter = Counter(extract_terms(text, self.stop_terms))
        values = {term: weight * self.idf.get(term, 0.5) for term, weight in counter.items()}
        denom = math.sqrt(sum(value * value for value in values.values())) or 1.0
        return {term: value / denom for term, value in values.items()}

    def _cosine(self, article_vector: dict[str, float], event_name: str) -> float:
        event_vector = self.event_vectors.get(event_name, {})
        if len(article_vector) < len(event_vector):
            return sum(value * event_vector.get(term, 0.0) for term, value in article_vector.items())
        return sum(article_vector.get(term, 0.0) * value for term, value in event_vector.items())

    def route_targets(self, article: dict[str, Any]) -> list[dict[str, Any]]:
        title = normalize_text(article.get("title", ""))
        content = normalize_text(str(article.get("content", ""))[: int(self.config.get("target_content_chars", 8000))])
        full = f"{title} {content}"
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for alias_key, matches in self.target_lookup.items():
            if len(alias_key) < 2:
                continue
            scope = ""
            if alias_key in title:
                scope = "title"
            elif alias_key in content:
                scope = "content"
            if not scope:
                continue
            for match in matches:
                target_type = str(match["target_type"])
                target_code = str(match["target_code"])
                key = (target_type, target_code, scope)
                if key in seen:
                    continue
                seen.add(key)
                if scope == "title":
                    confidence = 1.0 if match.get("is_formal") else 0.95
                elif target_type == "industry":
                    confidence = 0.82 if match.get("is_formal") else 0.76
                else:
                    confidence = 0.68 if match.get("is_formal") else 0.62
                rows.append(
                    {
                        "target_type": target_type,
                        "target_code": target_code,
                        "target_name": match["target_name"],
                        "matched_alias": match["alias"],
                        "matched_scope": scope,
                        "route_confidence": confidence,
                        "route_method": f"{scope}_{'formal' if match.get('is_formal') else 'alias'}_target_match",
                    }
                )
        code_pattern_text = full
        for (target_type, target_code), schemas in self.schemas_by_target.items():
            if target_type != "company" or len(target_code) != 6:
                continue
            if re.search(rf"(?<!\d){re.escape(target_code)}(?:\.(?:sh|sz|bj|hk))?(?!\d)", code_pattern_text):
                key = (target_type, target_code, "code")
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "target_type": target_type,
                        "target_code": target_code,
                        "target_name": str(schemas[0].get("target_name", "")),
                        "matched_alias": target_code,
                        "matched_scope": "title_or_content",
                        "route_confidence": 0.84,
                        "route_method": "stock_code_match",
                    }
                )
        rows.sort(key=lambda item: (-float(item["route_confidence"]), item["target_type"], item["target_code"]))
        return rows

    def _choose_schema(self, event_name: str, routed_targets: list[dict[str, Any]]) -> dict[str, Any]:
        best_schema = None
        best_score = -1.0
        for route in routed_targets:
            target_key = (str(route["target_type"]), str(route["target_code"]))
            for schema in self.schemas_by_target.get(target_key, []):
                if str(schema["event_name"]) == event_name and float(route["route_confidence"]) > best_score:
                    best_schema = schema
                    best_score = float(route["route_confidence"])
        if best_schema is not None:
            return best_schema
        return self.schemas_by_event[event_name][0]

    def _source_scores_for_event(
        self,
        article: dict[str, Any],
        event_name: str,
        title_vector: dict[str, float],
        body_vector: dict[str, float],
        routed_targets: list[dict[str, Any]],
    ) -> dict[str, float]:
        title = normalize_text(article.get("title", ""))
        body = normalize_text(str(article.get("content", ""))[: int(self.config.get("body_semantic_chars", 5000))])
        full = f"{title} {body}"
        scores = {source: 0.0 for source in self.source_order}

        for route in routed_targets:
            schemas = self.schemas_by_target.get((str(route["target_type"]), str(route["target_code"])), [])
            if any(str(schema["event_name"]) == event_name for schema in schemas):
                source = "official_target_mapping" if str(route["target_type"]) == "company" else "official_industry_mapping"
                scores[source] = max(scores[source], float(route["route_confidence"]))

        direct = 0.0
        for alias in self.event_aliases[event_name]:
            alias_norm = normalize_text(alias)
            if len(alias_norm) < 2:
                continue
            if alias_norm in title:
                direct = max(direct, 4.0 + 0.5 * len(alias_norm))
            elif alias_norm in full:
                direct = max(direct, 1.2 + 0.1 * len(alias_norm))
        keyword_score = 0.0
        for term in self.event_keywords.get(event_name, [])[:30]:
            if len(term) >= 2 and term in title:
                keyword_score += 0.08 * min(self.event_profiles[event_name].get(term, 1), 10)
            elif len(term) >= 3 and term in body:
                keyword_score += 0.01 * min(self.event_profiles[event_name].get(term, 1), 10)
        scores["schema_match"] = direct + keyword_score

        scores["title_semantic"] = 4.0 * self._cosine(title_vector, event_name)
        scores["body_semantic"] = 1.2 * self._cosine(body_vector, event_name)
        long_tail_threshold = int(self.config.get("long_tail_support_threshold", 15))
        if int(self.train_support.get(event_name, 0)) <= long_tail_threshold:
            rare_boost = 0.08 + 0.02 * max(0, long_tail_threshold - int(self.train_support.get(event_name, 0)))
            if scores["schema_match"] > 0 or scores["title_semantic"] > 0.08 or scores["body_semantic"] > 0.08:
                scores["long_tail_compensation"] = rare_boost
        scores["all64_fallback"] = 0.001
        return scores

    def generate(self, article: dict[str, Any], include_all64: bool = True) -> tuple[list[dict[str, Any]], float]:
        start = time.perf_counter()
        title = str(article.get("title", ""))
        body = str(article.get("content", ""))[: int(self.config.get("body_semantic_chars", 5000))]
        title_vector = self._article_vector(title)
        body_vector = self._article_vector(body)
        routed_targets = self.route_targets(article)
        prior_weight = float(self.config.get("support_prior_weight", 0.03))
        rows: list[dict[str, Any]] = []
        for event_name in self.event_names:
            source_scores = self._source_scores_for_event(article, event_name, title_vector, body_vector, routed_targets)
            active_sources = [source for source in self.source_order if source_scores.get(source, 0.0) > 0.0]
            if not include_all64 and not active_sources:
                continue
            schema = self._choose_schema(event_name, routed_targets)
            support_prior = prior_weight * math.log1p(int(self.train_support.get(event_name, 0)))
            total_score = support_prior + sum(source_scores.values())
            rows.append(
                {
                    "article_id": str(article.get("article_id", "")),
                    "title": title,
                    "publish_time": article.get("publish_time", ""),
                    "source": article.get("source", ""),
                    "event_name": event_name,
                    "schema_id": str(schema.get("schema_id", "")),
                    "target_type": str(schema.get("target_type", "")),
                    "target_code": str(schema.get("target_code", "")),
                    "target_name": str(schema.get("target_name", "")),
                    "event_description": str(schema.get("raw_description", "")),
                    "candidate_score": total_score,
                    "candidate_source": "|".join(active_sources),
                    "source_scores": source_scores,
                    "train_support": int(self.train_support.get(event_name, 0)),
                    "routed_targets": routed_targets[:5],
                }
            )
        rows.sort(key=lambda item: (-float(item["candidate_score"]), item["event_name"]))
        for idx, row in enumerate(rows, 1):
            row["candidate_rank"] = idx
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        for row in rows:
            row["generation_time_ms"] = elapsed_ms
        return rows, elapsed_ms

    def generate_many(self, articles: Iterable[dict[str, Any]], include_all64: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        for article in articles:
            rows, elapsed_ms = self.generate(article, include_all64=include_all64)
            all_rows.extend(rows)
            timing_rows.append(
                {
                    "article_id": str(article.get("article_id", "")),
                    "candidate_count": len(rows),
                    "generation_time_ms": round(elapsed_ms, 6),
                }
            )
        return all_rows, timing_rows

    def schema_profile_rows(self) -> list[dict[str, Any]]:
        rows = []
        for event_name in self.event_names:
            schemas = self.schemas_by_event[event_name]
            rows.append(
                {
                    "event_name": event_name,
                    "schema_count": len(schemas),
                    "train_support": int(self.train_support.get(event_name, 0)),
                    "aliases": "|".join(sorted(self.event_aliases[event_name])),
                    "keywords": "|".join(self.event_keywords.get(event_name, [])[:30]),
                    "representative_schema_id": str(schemas[0].get("schema_id", "")),
                }
            )
        return rows
