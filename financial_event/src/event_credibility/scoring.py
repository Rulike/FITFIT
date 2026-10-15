from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


OFFICIAL_PATTERNS = (
    "证监会",
    "交易所",
    "上交所",
    "深交所",
    "北交所",
    "国家",
    "国务院",
    "发改委",
    "工信部",
    "财政部",
    "商务部",
    "央行",
    "人民银行",
    "银保监",
    "金融监管",
    "药监局",
    "市场监管",
    "人民政府",
    "国资委",
    "官网",
    "公司公告",
    "巨潮资讯",
    "上证e互动",
    "互动易",
    "集团有限公司",
    "股份有限公司",
)

MAINSTREAM_PATTERNS = (
    "新华社",
    "人民日报",
    "央视",
    "经济日报",
    "证券时报",
    "中国证券报",
    "上海证券报",
    "证券日报",
    "第一财经",
    "财联社",
    "每日经济新闻",
    "21世纪经济报道",
    "界面新闻",
    "澎湃新闻",
    "北京商报",
    "财新",
    "华尔街见闻",
    "中国证券网",
    "中证网",
    "人民网",
    "人民财讯",
    "经济参考报",
    "中国基金报",
    "南方财经",
    "南方财经网",
    "贝壳财经",
    "e公司",
)

PORTAL_PATTERNS = (
    "东方财富",
    "新浪财经",
    "腾讯",
    "网易",
    "凤凰",
    "同花顺",
    "证券之星",
    "金融界",
    "和讯网",
    "格隆汇",
    "智通财经",
)

SELF_MEDIA_PATTERNS = (
    "微信公众号",
    "百家号",
    "雪球",
    "头条",
    "微博",
    "小红书",
    "抖音",
    "快手",
    "号",
)

RUMOR_PATTERNS = (
    "网传",
    "传闻",
    "据传",
    "疑似",
    "爆料",
    "小作文",
    "未经证实",
    "市场传言",
    "有消息称",
    "据悉",
    "或将",
    "或许",
    "坊间",
)

REVERSAL_PATTERNS = (
    "澄清",
    "辟谣",
    "否认",
    "不属实",
    "失实",
    "虚假",
    "造假",
    "假消息",
    "无未披露",
    "未披露潜在利空",
    "撤回",
    "更正",
)

CLARIFICATION_PATTERNS = (
    "声明",
    "回应",
    "说明",
)

HYPE_PATTERNS = (
    "惊爆",
    "重磅炸弹",
    "大利好",
    "大利空",
    "彻底爆了",
    "横空出世",
    "弯道超车",
    "暴涨",
    "暴跌",
    "封神",
    "狂飙",
)

OFFICIAL_TITLE_PATTERNS = (
    "公告",
    "批复",
    "通知",
    "决定",
    "监管函",
    "问询函",
    "回复函",
    "年度报告",
    "半年度报告",
    "招股说明书",
)

FEATURE_COLUMNS = [
    "member_count_log",
    "source_count_log",
    "source_entropy",
    "distinct_source_type_count",
    "mean_source_reliability",
    "max_source_reliability",
    "official_source_ratio",
    "mainstream_source_ratio",
    "portal_source_ratio",
    "self_media_source_ratio",
    "unknown_source_ratio",
    "hearsay_article_ratio",
    "dominant_source_ratio",
    "cluster_cohesion_score",
    "has_conflict_numeric",
    "overmerged_risk_numeric",
    "rumor_signal_count",
    "reversal_signal_count",
    "hype_signal_count",
    "official_title_signal_count",
    "source_score",
    "official_support_score",
    "cross_source_score",
    "cluster_consistency_score",
    "propagation_score",
    "content_quality_score",
    "contradiction_score",
    "rule_credibility_score",
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return low
    return max(low, min(high, float(value)))


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, list) else [value]
            except Exception:
                return [value]
        return [value]
    return [value]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def count_patterns(text: str, patterns: Iterable[str]) -> int:
    return sum(text.count(pattern) for pattern in patterns)


def source_profile(source: str) -> tuple[str, float]:
    value = str(source or "unknown").strip()
    if not value or value == "unknown":
        return "unknown", 0.25
    if contains_any(value, OFFICIAL_PATTERNS):
        return "official", 0.92
    if contains_any(value, MAINSTREAM_PATTERNS):
        return "mainstream", 0.82
    if contains_any(value, PORTAL_PATTERNS):
        return "portal", 0.62
    if contains_any(value, SELF_MEDIA_PATTERNS):
        return "self_media", 0.38
    return "general_media", 0.55


def text_signals(text: str) -> dict:
    value = str(text or "")
    return {
        "rumor_signal_count": count_patterns(value, RUMOR_PATTERNS),
        "reversal_signal_count": count_patterns(value, REVERSAL_PATTERNS),
        "clarification_signal_count": count_patterns(value, CLARIFICATION_PATTERNS),
        "hype_signal_count": count_patterns(value, HYPE_PATTERNS),
        "official_title_signal_count": count_patterns(value, OFFICIAL_TITLE_PATTERNS),
    }


def normalized_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(counter))


def stable_eval_split(key: str, ratio: float = 0.2) -> str:
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < ratio else "train"


def credibility_level(score_0_100: float, contradiction_score: float = 0.0) -> str:
    if contradiction_score >= 0.75:
        return "疑似虚假或已反转"
    if score_0_100 >= 85:
        return "高可信"
    if score_0_100 >= 70:
        return "较可信"
    if score_0_100 >= 50:
        return "待核验"
    if score_0_100 >= 30:
        return "疑似低可信"
    return "疑似低可信"


def risk_reasons(row: dict) -> list[str]:
    reasons = []
    if row["contradiction_score"] >= 0.55:
        reasons.append("存在澄清/否认/反转信号")
    if row["supporting_source_count"] <= 1:
        reasons.append("单一来源支撑")
    if row["mean_source_reliability"] < 0.45:
        reasons.append("平均信源可信度偏低")
    if row["rumor_signal_count"] > 0:
        reasons.append("标题或摘要含传闻表达")
    if row["hype_signal_count"] > 0:
        reasons.append("标题或摘要含夸张表达")
    if row["has_conflict_numeric"] > 0:
        reasons.append("事件簇存在要素冲突")
    if row["overmerged_risk_numeric"] > 0:
        reasons.append("聚类审计提示可能过度合并")
    if row["unknown_source_ratio"] > 0:
        reasons.append("部分来源缺失或未知")
    return reasons


def support_reasons(row: dict) -> list[str]:
    reasons = []
    if row["official_support_score"] >= 0.75:
        reasons.append("存在公告/监管/政府等权威佐证信号")
    if row["supporting_source_count"] >= 3:
        reasons.append("多独立来源交叉验证")
    if row["source_entropy"] >= 0.65:
        reasons.append("来源分布较均衡")
    if row["cluster_consistency_score"] >= 0.75:
        reasons.append("簇内一致性较高")
    if row["mean_source_reliability"] >= 0.70:
        reasons.append("来源整体可信度较高")
    return reasons


def build_cluster_features(
    clusters: list[dict],
    members: pd.DataFrame,
    events: pd.DataFrame,
    overmerged_clusters: set[str] | None = None,
) -> pd.DataFrame:
    overmerged_clusters = overmerged_clusters or set()
    events = events.copy()
    if "event_record_id" not in events.columns:
        events["event_record_id"] = events["dataset_split"].astype(str) + ":" + events["event_id"].astype(str)
    event_by_record = {str(row["event_record_id"]): row for _, row in events.iterrows()}
    event_by_article_split = {
        (str(row.get("dataset_split", "")), str(row.get("article_id", ""))): row
        for _, row in events.iterrows()
    }

    member_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in members.to_dict("records"):
        member_by_cluster[str(row.get("cluster_id"))].append(row)

    rows = []
    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id"))
        event_rows = []
        for member in member_by_cluster.get(cluster_id, []):
            event = event_by_record.get(str(member.get("event_record_id")))
            if event is None:
                event = event_by_article_split.get(
                    (str(member.get("dataset_split", "")), str(member.get("article_id", "")))
                )
            if event is not None:
                event_rows.append(dict(event))

        if not event_rows:
            event_rows = []

        sources = []
        source_types = []
        source_scores = []
        titles = []
        article_ids = []
        publish_times = []
        final_confidences = []
        for event in event_rows:
            source = (
                event.get("source_normalized")
                or event.get("source_name")
                or event.get("source")
                or "unknown"
            )
            source_type, source_score = source_profile(source)
            sources.append(str(source or "unknown"))
            source_types.append(source_type)
            source_scores.append(source_score)
            titles.append(str(event.get("title", "")))
            article_ids.append(str(event.get("article_id", "")))
            publish_times.append(event.get("publish_time"))
            final_confidences.append(float(event.get("final_confidence_score") or 0.0))

        source_counter = Counter(sources)
        source_type_counter = Counter(source_types)
        total_members = len(event_rows) or int(cluster.get("member_count", 0) or 0)
        source_count = len([s for s in source_counter if s and s != "unknown"])
        dominant_ratio = (
            max(source_counter.values()) / sum(source_counter.values())
            if source_counter
            else 1.0
        )
        text = "。".join(titles + [str(cluster.get("event_name", "")), str(cluster.get("target_name", ""))])
        signals = text_signals(text)
        hearsay_count = sum(1 for aid in article_ids if "HEARSAY" in aid.upper())
        unknown_source_count = source_type_counter.get("unknown", 0)
        official_count = source_type_counter.get("official", 0)
        mainstream_count = source_type_counter.get("mainstream", 0)
        portal_count = source_type_counter.get("portal", 0)
        self_media_count = source_type_counter.get("self_media", 0)

        official_title = signals["official_title_signal_count"]
        mean_source = float(np.mean(source_scores)) if source_scores else 0.25
        max_source = float(np.max(source_scores)) if source_scores else 0.25
        source_score = clamp(0.65 * mean_source + 0.35 * max_source)
        source_entropy = normalized_entropy(source_counter)
        cross_source_score = clamp(
            0.35 * min(math.log1p(source_count) / math.log(6), 1.0)
            + 0.25 * source_entropy
            + 0.20 * min(math.log1p(total_members) / math.log(10), 1.0)
            + 0.20 * min(len(source_type_counter) / 4, 1.0)
        )
        official_support_score = clamp(
            0.70 * min(official_count, 1)
            + 0.35 * min(official_title, 1)
            + 0.20 * min(mainstream_count / 2, 1.0)
            + 0.10 * min(source_count / 3, 1.0)
        )
        cohesion = float(cluster.get("cluster_cohesion_score", 0.68) or 0.0)
        has_conflict = bool(cluster.get("has_conflict", False))
        overmerged = cluster_id in overmerged_clusters
        cluster_consistency_score = clamp(
            cohesion
            + 0.08 * min(source_count / 3, 1.0)
            - 0.25 * int(has_conflict)
            - 0.20 * int(overmerged)
        )
        sorted_events = sorted(
            event_rows,
            key=lambda r: str(r.get("publish_time") or "9999-99-99"),
        )
        first_source_score = 0.25
        first_source_type = "unknown"
        if sorted_events:
            first_source = (
                sorted_events[0].get("source_normalized")
                or sorted_events[0].get("source_name")
                or sorted_events[0].get("source")
                or "unknown"
            )
            first_source_type, first_source_score = source_profile(first_source)
        has_later_official = official_count > 0 and first_source_type != "official"
        propagation_score = clamp(
            0.45 * first_source_score
            + 0.25 * cross_source_score
            + 0.20 * int(has_later_official)
            + 0.10 * (1.0 - dominant_ratio)
        )
        rumor_flag = min(signals["rumor_signal_count"], 3) / 3
        hype_flag = min(signals["hype_signal_count"], 3) / 3
        hearsay_ratio = hearsay_count / total_members if total_members else 0.0
        unknown_ratio = unknown_source_count / total_members if total_members else 0.0
        avg_final_confidence = float(np.mean(final_confidences)) if final_confidences else 0.0
        content_quality_score = clamp(
            1.0
            - 0.25 * rumor_flag
            - 0.20 * hype_flag
            - 0.20 * hearsay_ratio
            - 0.15 * unknown_ratio
            + 0.10 * avg_final_confidence
        )
        contradiction_score = clamp(
            0.45 * min(signals["reversal_signal_count"], 2)
            + 0.15 * min(signals["clarification_signal_count"], 1) * int(signals["rumor_signal_count"] > 0)
            + 0.25 * int(has_conflict)
            + 0.20 * int(overmerged)
            + 0.10 * rumor_flag
        )
        rule_score = clamp(
            0.25 * source_score
            + 0.25 * official_support_score
            + 0.20 * cross_source_score
            + 0.15 * cluster_consistency_score
            + 0.10 * propagation_score
            + 0.05 * content_quality_score
            - 0.30 * contradiction_score
        )

        row = {
            "cluster_id": cluster_id,
            "target_type": cluster.get("target_type", ""),
            "target_code": cluster.get("target_code", ""),
            "target_name": cluster.get("target_name", ""),
            "schema_id": cluster.get("schema_id", ""),
            "event_name": cluster.get("event_name", ""),
            "member_count": total_members,
            "supporting_source_count": source_count,
            "source_list": " | ".join(sorted(source_counter)),
            "source_type_distribution": json.dumps(source_type_counter, ensure_ascii=False),
            "earliest_publish_time": cluster.get("earliest_publish_time", ""),
            "latest_publish_time": cluster.get("latest_publish_time", ""),
            "time_span_days": cluster.get("time_span_days", 0),
            "title_samples": " || ".join(titles[:5]),
            "member_count_log": math.log1p(total_members),
            "source_count_log": math.log1p(source_count),
            "source_entropy": source_entropy,
            "distinct_source_type_count": len(source_type_counter),
            "mean_source_reliability": mean_source,
            "max_source_reliability": max_source,
            "official_source_ratio": official_count / total_members if total_members else 0.0,
            "mainstream_source_ratio": mainstream_count / total_members if total_members else 0.0,
            "portal_source_ratio": portal_count / total_members if total_members else 0.0,
            "self_media_source_ratio": self_media_count / total_members if total_members else 0.0,
            "unknown_source_ratio": unknown_ratio,
            "hearsay_article_ratio": hearsay_ratio,
            "dominant_source_ratio": dominant_ratio,
            "cluster_cohesion_score": cohesion,
            "has_conflict_numeric": int(has_conflict),
            "overmerged_risk_numeric": int(overmerged),
            "source_score": source_score,
            "official_support_score": official_support_score,
            "cross_source_score": cross_source_score,
            "cluster_consistency_score": cluster_consistency_score,
            "propagation_score": propagation_score,
            "content_quality_score": content_quality_score,
            "contradiction_score": contradiction_score,
            "rule_credibility_score": rule_score,
            **signals,
        }
        row["supporting_reasons"] = "；".join(support_reasons(row))
        row["risk_reasons"] = "；".join(risk_reasons(row))
        rows.append(row)

    return pd.DataFrame(rows)


def assign_weak_labels(features: pd.DataFrame) -> pd.DataFrame:
    rows = features.copy()
    labels = []
    weights = []
    reasons = []
    for row in rows.to_dict("records"):
        label = None
        weight = 0.0
        reason = "unlabeled_boundary_case"
        if row["contradiction_score"] >= 0.70:
            label, weight, reason = 0, 1.0, "strong_contradiction_or_reversal"
        elif (
            row["supporting_source_count"] <= 1
            and row["mean_source_reliability"] <= 0.45
            and (row["rumor_signal_count"] > 0 or row["hype_signal_count"] > 0 or row["content_quality_score"] < 0.55)
        ):
            label, weight, reason = 0, 0.85, "single_weak_source_with_low_quality_signal"
        elif (
            row["supporting_source_count"] <= 1
            and row["official_support_score"] < 0.35
            and row["cross_source_score"] < 0.45
            and row["source_score"] <= 0.62
        ):
            label, weight, reason = 0, 0.55, "single_source_insufficient_evidence"
        elif row["overmerged_risk_numeric"] > 0 and row["cluster_consistency_score"] < 0.55:
            label, weight, reason = 0, 0.60, "cluster_audit_overmerge_risk"
        elif row["official_support_score"] >= 0.75 and row["contradiction_score"] < 0.35:
            label, weight, reason = 1, 1.0, "official_or_announcement_supported"
        elif (
            row["cross_source_score"] >= 0.72
            and row["mean_source_reliability"] >= 0.58
            and row["cluster_consistency_score"] >= 0.62
            and row["contradiction_score"] < 0.35
        ):
            label, weight, reason = 1, 0.80, "multi_source_consistent"
        elif row["rule_credibility_score"] >= 0.82 and row["contradiction_score"] < 0.25:
            label, weight, reason = 1, 0.65, "high_rule_score"
        elif row["rule_credibility_score"] <= 0.35:
            label, weight, reason = 0, 0.65, "low_rule_score"
        labels.append(label)
        weights.append(weight)
        reasons.append(reason)
    rows["weak_label"] = labels
    rows["weak_label_weight"] = weights
    rows["weak_label_reason"] = reasons
    rows["split"] = rows["cluster_id"].map(stable_eval_split)
    return rows


@dataclass
class ModelResult:
    backend: str
    model_score: np.ndarray
    metrics: dict
    feature_importance: pd.DataFrame


class NumpyLogisticModel:
    def __init__(self, lr: float = 0.08, steps: int = 1200, l2: float = 0.02):
        self.lr = lr
        self.steps = steps
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        sample_weight = np.ones_like(y, dtype=float) if sample_weight is None else sample_weight.astype(float)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        z = self._transform(x)
        z = np.c_[np.ones(len(z)), z]
        w = np.zeros(z.shape[1])
        denom = max(sample_weight.sum(), 1e-9)
        for _ in range(self.steps):
            p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
            grad = (z.T @ ((p - y) * sample_weight)) / denom
            grad[1:] += self.l2 * w[1:]
            w -= self.lr * grad
        self.weights_ = w
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self._transform(x)
        z = np.c_[np.ones(len(z)), z]
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ self.weights_, -30, 30)))
        return np.c_[1.0 - p, p]


def binary_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {}
    pred = (score >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true)
    auc = roc_auc(y_true, score)
    return {
        "count": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def try_train_model(labeled: pd.DataFrame, all_features: pd.DataFrame) -> ModelResult:
    feature_frame = all_features[FEATURE_COLUMNS].fillna(0.0).astype(float)
    labeled = labeled.dropna(subset=["weak_label"])
    if len(labeled) < 30 or labeled["weak_label"].nunique() < 2:
        return ModelResult(
            backend="rule_only_insufficient_weak_labels",
            model_score=all_features["rule_credibility_score"].to_numpy(dtype=float),
            metrics={},
            feature_importance=pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": 0.0}),
        )

    train = labeled[labeled["split"] == "train"]
    eval_df = labeled[labeled["split"] == "eval"]
    if train["weak_label"].nunique() < 2:
        train = labeled
    x_train = train[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float)
    y_train = train["weak_label"].to_numpy(dtype=int)
    w_train = train["weak_label_weight"].fillna(1.0).to_numpy(dtype=float)

    backend = "numpy_logistic_weak_supervision"
    model = None
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=10,
            random_state=2026,
            verbose=-1,
        )
        model.fit(x_train, y_train, sample_weight=w_train)
        backend = "lightgbm_weak_supervision"
    except Exception:
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=2026,
                eval_metric="logloss",
            )
            model.fit(x_train, y_train, sample_weight=w_train)
            backend = "xgboost_weak_supervision"
        except Exception:
            try:
                from sklearn.ensemble import GradientBoostingClassifier

                model = GradientBoostingClassifier(random_state=2026)
                model.fit(x_train, y_train, sample_weight=w_train)
                backend = "sklearn_gradient_boosting_weak_supervision"
            except Exception:
                model = NumpyLogisticModel()
                model.fit(x_train, y_train, sample_weight=w_train)

    model_score = model.predict_proba(feature_frame.to_numpy(dtype=float))[:, 1]

    metrics = {"backend": backend}
    for split_name, split_df in (("train", train), ("eval", eval_df), ("all_labeled", labeled)):
        if split_df.empty:
            continue
        split_scores = model.predict_proba(split_df[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float))[:, 1]
        metrics[split_name] = binary_metrics(split_df["weak_label"].to_numpy(dtype=int), split_scores)

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif isinstance(model, NumpyLogisticModel):
        importance = np.abs(model.weights_[1:])
    else:
        importance = np.zeros(len(FEATURE_COLUMNS))
    feature_importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance})
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    return ModelResult(backend=backend, model_score=model_score, metrics=metrics, feature_importance=feature_importance)


def finalize_scores(features: pd.DataFrame, model_result: ModelResult) -> pd.DataFrame:
    output = features.copy()
    output["model_credibility_score"] = model_result.model_score
    use_model = not model_result.backend.startswith("rule_only")
    if use_model:
        output["credibility_score"] = (
            100.0
            * (
                0.60 * output["model_credibility_score"].astype(float)
                + 0.40 * output["rule_credibility_score"].astype(float)
            )
        )
    else:
        output["credibility_score"] = 100.0 * output["rule_credibility_score"].astype(float)
    output["credibility_score"] = output["credibility_score"].clip(0, 100).round(2)
    output["credibility_level"] = output.apply(
        lambda r: credibility_level(float(r["credibility_score"]), float(r["contradiction_score"])),
        axis=1,
    )
    output["verification_status"] = np.select(
        [
            output["contradiction_score"] >= 0.75,
            output["official_support_score"] >= 0.75,
            output["cross_source_score"] >= 0.72,
            output["credibility_score"] < 50,
        ],
        ["reversed_or_contradicted", "official_supported", "multi_source_supported", "needs_manual_review"],
        default="insufficient_evidence",
    )
    return output


def build_manual_review_sample(scores: pd.DataFrame, max_rows: int = 260) -> pd.DataFrame:
    buckets = []
    candidates = [
        ("boundary_band", scores.assign(_distance=(scores["credibility_score"] - 50).abs()).sort_values(["_distance", "credibility_score", "member_count"], ascending=[True, True, False]).head(110)),
        ("needs_manual_review", scores[scores["verification_status"] == "needs_manual_review"].sort_values(["credibility_score", "member_count"], ascending=[True, False]).head(90)),
        ("single_weak_source", scores[(scores["supporting_source_count"] <= 1) | (scores["mean_source_reliability"] < 0.55)].sort_values("credibility_score").head(80)),
        ("reversal_signal", scores[scores["verification_status"] == "reversed_or_contradicted"].sort_values("credibility_score").head(40)),
        ("large_cluster_risk", scores.sort_values(["overmerged_risk_numeric", "member_count", "credibility_score"], ascending=[False, False, True]).head(60)),
        ("high_confidence", scores.sort_values("credibility_score", ascending=False).head(40)),
    ]
    seen = set()
    for bucket, frame in candidates:
        for row in frame.to_dict("records"):
            if row["cluster_id"] in seen:
                continue
            seen.add(row["cluster_id"])
            item = dict(row)
            item["review_bucket"] = bucket
            buckets.append(item)
            if len(buckets) >= max_rows:
                break
        if len(buckets) >= max_rows:
            break
    if len(buckets) < max_rows:
        fallback = scores.sort_values(["credibility_score", "member_count"], ascending=[True, False])
        for row in fallback.to_dict("records"):
            cluster_id = row["cluster_id"]
            if cluster_id in seen:
                continue
            seen.add(cluster_id)
            item = dict(row)
            item["review_bucket"] = "fill"
            buckets.append(item)
            if len(buckets) >= max_rows:
                break
    review = pd.DataFrame(buckets)
    for col in (
        "manual_is_true_event",
        "manual_source_reliable",
        "manual_needs_official_check",
        "manual_is_reversed",
        "manual_comment",
    ):
        review[col] = ""
    keep = [
        "review_bucket",
        "cluster_id",
        "target_code",
        "target_name",
        "schema_id",
        "event_name",
        "credibility_score",
        "credibility_level",
        "verification_status",
        "risk_reasons",
        "supporting_reasons",
        "source_list",
        "title_samples",
        "manual_is_true_event",
        "manual_source_reliable",
        "manual_needs_official_check",
        "manual_is_reversed",
        "manual_comment",
    ]
    return review[[c for c in keep if c in review.columns]]


def build_timeline(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sort_df = scores.copy()
    sort_df["earliest_publish_time_sort"] = pd.to_datetime(sort_df["earliest_publish_time"], errors="coerce")
    sort_df = sort_df.sort_values(["target_code", "schema_id", "earliest_publish_time_sort", "cluster_id"])
    for (_, _), group in sort_df.groupby(["target_code", "schema_id"], dropna=False):
        previous = None
        for _, row in group.iterrows():
            stage = "initial_report"
            if row["contradiction_score"] >= 0.75:
                stage = "reversal_or_clarification"
            elif row["official_support_score"] >= 0.75:
                stage = "official_confirmation"
            elif row["cross_source_score"] >= 0.72:
                stage = "multi_source_confirmation"
            current_time = row["earliest_publish_time_sort"]
            prev_time = previous["earliest_publish_time_sort"] if previous is not None else pd.NaT
            days_since_prev = None
            if previous is not None and pd.notna(current_time) and pd.notna(prev_time):
                days_since_prev = (current_time - prev_time).total_seconds() / 86400
            rows.append(
                {
                    "target_code": row["target_code"],
                    "target_name": row["target_name"],
                    "schema_id": row["schema_id"],
                    "event_name": row["event_name"],
                    "cluster_id": row["cluster_id"],
                    "event_stage": stage,
                    "credibility_score": row["credibility_score"],
                    "credibility_level": row["credibility_level"],
                    "verification_status": row["verification_status"],
                    "earliest_publish_time": row["earliest_publish_time"],
                    "latest_publish_time": row["latest_publish_time"],
                    "previous_cluster_id": previous["cluster_id"] if previous is not None else "",
                    "days_since_previous": days_since_prev,
                    "timeline_reason": row["supporting_reasons"] or row["risk_reasons"],
                }
            )
            previous = row
    return pd.DataFrame(rows)


def source_reliability_report(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source_col = "source_normalized" if "source_normalized" in events.columns else "source"
    for source, group in events.groupby(source_col, dropna=False):
        source_type, score = source_profile(str(source or "unknown"))
        rows.append(
            {
                "source_normalized": source,
                "source_type": source_type,
                "prior_reliability_score": score,
                "accepted_event_count": len(group),
                "avg_task1_confidence": float(pd.to_numeric(group.get("final_confidence_score", 0), errors="coerce").fillna(0).mean()),
                "unknown_lookup_count": int((group.get("source_lookup_status", "") != "matched").sum())
                if "source_lookup_status" in group
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["prior_reliability_score", "accepted_event_count"], ascending=[False, False])


def summarize_outputs(scores: pd.DataFrame, labels: pd.DataFrame, model_result: ModelResult) -> dict:
    return {
        "cluster_count": int(len(scores)),
        "model_backend": model_result.backend,
        "weak_labeled_count": int(labels["weak_label"].notna().sum()),
        "weak_positive_count": int((labels["weak_label"] == 1).sum()),
        "weak_negative_count": int((labels["weak_label"] == 0).sum()),
        "credibility_level_distribution": scores["credibility_level"].value_counts().to_dict(),
        "verification_status_distribution": scores["verification_status"].value_counts().to_dict(),
        "score_mean": float(scores["credibility_score"].mean()),
        "score_min": float(scores["credibility_score"].min()),
        "score_max": float(scores["credibility_score"].max()),
        "reversal_candidate_count": int((scores["reversal_signal_count"] > 0).sum()),
        "official_supported_count": int((scores["official_support_score"] >= 0.75).sum()),
        "multi_source_supported_count": int((scores["cross_source_score"] >= 0.72).sum()),
        "metrics": model_result.metrics,
    }


def write_method_report(report_dir: Path, summary: dict) -> None:
    metrics = summary.get("metrics", {})
    eval_metrics = metrics.get("eval", {})
    lines = [
        "# 任务二事件可信评估与辨伪实验报告",
        "",
        "## 实验定位",
        "",
        "本实验以任务一输出的事件簇为评分对象，构建事件级可信评估模型，而不是单篇新闻真假二分类。",
        "模型综合信源可信度、多源交叉验证、公告/权威佐证、传播路径、文本低质信号、簇内一致性和反转/澄清信号，输出可解释可信分与人工复核队列。",
        "",
        "## 本次运行结果",
        "",
        f"- 事件簇数量：{summary['cluster_count']}",
        f"- 模型后端：{summary['model_backend']}",
        f"- 弱监督样本：{summary['weak_labeled_count']}，其中正样本 {summary['weak_positive_count']}，负样本 {summary['weak_negative_count']}",
        f"- 平均可信分：{summary['score_mean']:.2f}，最低 {summary['score_min']:.2f}，最高 {summary['score_max']:.2f}",
        f"- 公告/权威佐证事件簇：{summary['official_supported_count']}",
        f"- 多源交叉支撑事件簇：{summary['multi_source_supported_count']}",
        f"- 反转/澄清候选事件簇：{summary['reversal_candidate_count']}",
        "",
        "## 可信等级分布",
        "",
    ]
    for level, count in summary["credibility_level_distribution"].items():
        lines.append(f"- {level}: {count}")
    lines.extend(["", "## 验证状态分布", ""])
    for status, count in summary["verification_status_distribution"].items():
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## 弱监督评估说明",
            "",
            "当前指标是模型对弱监督规则标签的拟合/泛化自检，不等价于人工标注真实性准确率。",
            "后续人工复核样本回填后，应使用人工标签重新计算真实性评估指标。",
            "",
        ]
    )
    if eval_metrics:
        lines.extend(
            [
                f"- Eval count: {eval_metrics.get('count')}",
                f"- Eval precision: {eval_metrics.get('precision'):.4f}",
                f"- Eval recall: {eval_metrics.get('recall'):.4f}",
                f"- Eval F1: {eval_metrics.get('f1'):.4f}",
                f"- Eval ROC-AUC: {eval_metrics.get('roc_auc'):.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 主要产物",
            "",
            "- `data/processed/event_credibility/cluster_credibility_features.parquet`: 事件簇可信评估特征",
            "- `data/processed/event_credibility/cluster_credibility_weak_labels.parquet`: 弱监督标签",
            "- `outputs/event_credibility/event_credibility_scores.parquet/jsonl/xlsx`: 可信分主结果",
            "- `outputs/event_credibility/event_timeline_for_task3.parquet/jsonl/csv`: 面向任务三的事件时间线接口",
            "- `reports/event_credibility/credibility_manual_review_sample.xlsx`: 人工复核样本",
            "- `reports/event_credibility/reversal_candidates.csv`: 反转/澄清候选",
            "- `reports/event_credibility/source_reliability.csv`: 信源可信度先验表",
            "- `reports/event_credibility/feature_importance.csv`: 模型特征重要性",
            "- `reports/event_credibility/credibility_metrics.json`: 机器可读指标汇总",
            "",
            "## 当前限制",
            "",
            f"- 本轮运行后端为 `{summary['model_backend']}`，脚本支持 `scikit-learn` / `LightGBM` / `XGBoost` 自动回退。",
            "- 当前反转/澄清识别主要基于标题与事件簇文本信号，尚未读取 30 万篇原文正文做深度 NLI 证据判别。",
            "- 任务一聚类质量仍有待人工审计，可信评估会继承聚类阶段的过合并或漏合并风险。",
            "- 弱监督标签用于冷启动，不应替代人工复核标签。",
        ]
    )
    (report_dir / "task2_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict:
    clusters = read_jsonl(ROOT / args.clusters)
    members = pd.read_parquet(ROOT / args.members)
    events = pd.read_parquet(ROOT / args.events)
    overmerged_df = read_csv_safe(ROOT / args.overmerged_report)
    overmerged_clusters = set(overmerged_df.get("cluster_id", pd.Series(dtype=str)).dropna().astype(str))

    features = build_cluster_features(clusters, members, events, overmerged_clusters)
    labels = assign_weak_labels(features)
    labeled = labels[labels["weak_label"].notna()].copy()
    model_result = try_train_model(labeled, labels)
    scores = finalize_scores(labels, model_result)
    manual_review = build_manual_review_sample(scores, max_rows=args.manual_sample_size)
    timeline = build_timeline(scores)
    source_report = source_reliability_report(events)
    reversal = scores[scores["reversal_signal_count"] > 0].sort_values(
        ["contradiction_score", "credibility_score"], ascending=[False, True]
    )

    processed_dir = ROOT / args.processed_dir
    output_dir = ROOT / args.output_dir
    report_dir = ROOT / args.report_dir
    for directory in (processed_dir, output_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    features.to_parquet(processed_dir / "cluster_credibility_features.parquet", index=False)
    labels.to_parquet(processed_dir / "cluster_credibility_weak_labels.parquet", index=False)
    scores.to_parquet(output_dir / "event_credibility_scores.parquet", index=False)
    scores.to_excel(output_dir / "event_credibility_scores.xlsx", index=False)
    write_jsonl(output_dir / "event_credibility_scores.jsonl", scores.to_dict("records"))
    timeline.to_parquet(output_dir / "event_timeline_for_task3.parquet", index=False)
    timeline.to_csv(output_dir / "event_timeline_for_task3.csv", index=False, encoding="utf-8-sig")
    write_jsonl(output_dir / "event_timeline_for_task3.jsonl", timeline.to_dict("records"))

    manual_review.to_excel(report_dir / "credibility_manual_review_sample.xlsx", index=False)
    manual_review.to_csv(report_dir / "credibility_manual_review_sample.csv", index=False, encoding="utf-8-sig")
    source_report.to_csv(report_dir / "source_reliability.csv", index=False, encoding="utf-8-sig")
    reversal.to_csv(report_dir / "reversal_candidates.csv", index=False, encoding="utf-8-sig")
    model_result.feature_importance.to_csv(report_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    scores[FEATURE_COLUMNS].describe().T.to_csv(report_dir / "feature_summary.csv", encoding="utf-8-sig")

    summary = summarize_outputs(scores, labels, model_result)
    (report_dir / "credibility_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_method_report(report_dir, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="outputs/event_clustering/event_clusters.jsonl")
    parser.add_argument("--members", default="outputs/event_clustering/event_cluster_members.parquet")
    parser.add_argument("--events", default="data/processed/event_clusters/final_events_with_sources.parquet")
    parser.add_argument("--overmerged_report", default="reports/event_clustering/possible_overmerged_clusters.csv")
    parser.add_argument("--processed_dir", default="data/processed/event_credibility")
    parser.add_argument("--output_dir", default="outputs/event_credibility")
    parser.add_argument("--report_dir", default="reports/event_credibility")
    parser.add_argument("--manual_sample_size", type=int, default=260)
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    main()
