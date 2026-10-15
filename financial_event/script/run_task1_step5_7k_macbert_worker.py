#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MacBERT workers for Task1 Step 5.7-K structure extraction.

The predict path intentionally consumes only event_mention_id, article_id,
content, event_name, and title. Training labels are read only in train mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


DIR_LABELS = ["positive", "negative", "neutral", "unclear"]
SPAN_FIELDS = ["trigger", "event_span", "subject_company", "event_time", "object_target"]
ROLE_FIELDS = ["subject_company", "event_time", "object_target", "trigger"]
FIELD_QUERIES = {
    "trigger": "事件触发词",
    "event_span": "事件完整描述片段",
    "subject_company": "事件主体公司或机构",
    "event_time": "事件发生时间",
    "object_target": "事件客体目标",
}
OPTIONAL_FIELDS = {"subject_company", "event_time", "object_target"}
SCHEME_NAMES = {
    "A": "A_macbert_multitask_token",
    "B": "B_macbert_event_conditioned_qa",
    "C": "C_macbert_hybrid_span_consistency",
}
SCHEME_TYPES = {"A": "token", "B": "qa", "C": "span"}
MAX_LENGTH = 512
STRIDE = 128
SEED = 2026


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_time(text: str) -> str:
    value = str(text or "")
    if not value:
        return "未明确"
    import re

    m = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?", value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(20\d{2})年(\d{1,2})月", value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-01"
    m = re.search(r"(20\d{2})年(?:第一|第二|第三|第四|一|二|三|四)?季度", value)
    if m:
        qmap = {"第一": "01", "一": "01", "第二": "04", "二": "04", "第三": "07", "三": "07", "第四": "10", "四": "10"}
        q = re.search(r"(第一|第二|第三|第四|一|二|三|四)", value)
        return f"{m.group(1)}-{qmap.get(q.group(1) if q else '', '01')}-01"
    if value in {"近日", "日前", "当日", "当天", "今日", "近期", "今年", "未来", "上半年", "下半年"}:
        return value
    return value


def empty_span(field: str = "") -> dict[str, Any]:
    if field == "event_time":
        return {"text": "", "normalized": "未明确", "start": -1, "end": -1}
    return {"text": "", "start": -1, "end": -1}


def as_span(content: str, start: int, end: int, field: str = "") -> dict[str, Any]:
    if not (0 <= start < end <= len(content)):
        return empty_span(field)
    text = content[start:end]
    out: dict[str, Any] = {"text": text, "start": start, "end": end}
    if field == "event_time":
        out["normalized"] = normalize_time(text)
    return out


def split_sentences(content: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    start = 0
    for idx, ch in enumerate(content):
        if ch in "。！？!?；;\n":
            end = idx + 1
            segment = content[start:end]
            lead = len(segment) - len(segment.lstrip())
            trail = len(segment) - len(segment.rstrip())
            real_start = start + lead
            real_end = end - trail
            if real_start < real_end:
                sentences.append({"start": real_start, "end": real_end, "text": content[real_start:real_end]})
            start = end
    if start < len(content):
        segment = content[start:]
        lead = len(segment) - len(segment.lstrip())
        trail = len(segment) - len(segment.rstrip())
        real_start = start + lead
        real_end = len(content) - trail
        if real_start < real_end:
            sentences.append({"start": real_start, "end": real_end, "text": content[real_start:real_end]})
    if not sentences and content:
        sentences.append({"start": 0, "end": len(content), "text": content})
    return sentences


def sentence_containing(content: str, start: int, end: int) -> dict[str, Any]:
    for sent in split_sentences(content):
        if sent["start"] <= start < end <= sent["end"]:
            return {"text": content[sent["start"] : sent["end"]], "start": sent["start"], "end": sent["end"]}
    return empty_span("event_span")


def allowed_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_mention_id": row.get("event_mention_id", ""),
        "article_id": row.get("article_id", ""),
        "content": str(row.get("content", "") or ""),
        "event_name": str(row.get("event_name", "") or ""),
        "title": str(row.get("title", "") or ""),
    }


@dataclass
class WindowFeature:
    row_index: int
    field_index: int
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    offsets: list[tuple[int, int]]
    content_mask: list[int]
    labels: list[list[int]] | None = None
    start_labels: list[int] | None = None
    end_labels: list[int] | None = None
    exists_labels: list[float] | None = None
    direction_label: int = 0


def encode_windows(
    tokenizer: Any,
    row: dict[str, Any],
    scheme: str,
    field: str | None = None,
    with_labels: bool = False,
) -> list[WindowFeature]:
    content = str(row.get("content", "") or "")
    title = str(row.get("title", "") or "")
    event_name = str(row.get("event_name", "") or "")
    if scheme == "B" and field:
        prefix = f"{event_name}。问题：请抽取{FIELD_QUERIES[field]}。标题：{title}"
    elif scheme == "C":
        prefix = f"事件类型：{event_name}。标题：{title}"
    else:
        prefix = f"事件类型：{event_name}。标题：{title}"
    enc = tokenizer(
        prefix,
        content,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    features: list[WindowFeature] = []
    dir_label = DIR_LABELS.index(row.get("direction", "unclear") if row.get("direction") in DIR_LABELS else "unclear")
    for i, input_ids in enumerate(enc["input_ids"]):
        seq_ids = enc.sequence_ids(i)
        offsets = [tuple(pair) for pair in enc["offset_mapping"][i]]
        content_mask = [1 if seq == 1 and pair != (0, 0) else 0 for seq, pair in zip(seq_ids, offsets)]
        feat = WindowFeature(
            row_index=-1,
            field_index=SPAN_FIELDS.index(field) if field else -1,
            input_ids=list(input_ids),
            attention_mask=list(enc["attention_mask"][i]),
            token_type_ids=list(enc.get("token_type_ids", [[0] * len(input_ids)])[i]),
            offsets=offsets,
            content_mask=content_mask,
            direction_label=dir_label,
        )
        if with_labels:
            if scheme == "A":
                feat.labels = [bio_labels_for_field(offsets, content_mask, row.get(span_field, {})) for span_field in SPAN_FIELDS]
            else:
                fields = [field] if scheme == "B" and field else SPAN_FIELDS
                start_labels: list[int] = []
                end_labels: list[int] = []
                exists_labels: list[float] = []
                for span_field in fields:
                    start_idx, end_idx, exists = span_start_end(offsets, content_mask, row.get(span_field, {}))
                    start_labels.append(start_idx)
                    end_labels.append(end_idx)
                    exists_labels.append(float(exists))
                feat.start_labels = start_labels
                feat.end_labels = end_labels
                feat.exists_labels = exists_labels
        features.append(feat)
    return features


def bio_labels_for_field(offsets: list[tuple[int, int]], content_mask: list[int], span: dict[str, Any]) -> list[int]:
    labels = [-100 if not mask else 0 for mask in content_mask]
    if not span or not span.get("text"):
        return labels
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    first = True
    for idx, ((s, e), mask) in enumerate(zip(offsets, content_mask)):
        if not mask:
            continue
        if e <= start or s >= end:
            continue
        labels[idx] = 1 if first else 2
        first = False
    return labels


def span_start_end(offsets: list[tuple[int, int]], content_mask: list[int], span: dict[str, Any]) -> tuple[int, int, int]:
    if not span or not span.get("text"):
        return 0, 0, 0
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    covered = [idx for idx, ((s, e), mask) in enumerate(zip(offsets, content_mask)) if mask and not (e <= start or s >= end)]
    if not covered:
        return 0, 0, 0
    token_start = covered[0]
    token_end = covered[-1]
    if offsets[token_start][0] > start or offsets[token_end][1] < end:
        return 0, 0, 0
    return token_start, token_end, 1


def token_window_coverage(tokenizer: Any, rows: list[dict[str, Any]], scheme: str) -> dict[str, Any]:
    uncovered = 0
    checked = 0
    for row in rows:
        features = encode_windows(tokenizer, row, scheme, with_labels=True)
        for field in SPAN_FIELDS:
            span = row.get(field, {})
            if not span.get("text"):
                continue
            checked += 1
            ok = False
            for feat in features:
                s, e, exists = span_start_end(feat.offsets, feat.content_mask, span)
                if exists and s >= 0 and e >= s:
                    ok = True
                    break
            if not ok:
                uncovered += 1
    return {"checked_span_count": checked, "gold_span_uncovered_count": uncovered, "max_length": MAX_LENGTH, "stride": STRIDE}


class StructureDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, scheme: str, train: bool) -> None:
        self.rows = rows
        self.scheme = scheme
        self.features: list[WindowFeature] = []
        for row_index, row in enumerate(rows):
            if scheme == "B":
                row_features: list[WindowFeature] = []
                for field in SPAN_FIELDS:
                    row_features.extend(encode_windows(tokenizer, row, scheme, field=field, with_labels=train))
            else:
                row_features = encode_windows(tokenizer, row, scheme, with_labels=train)
            for feat in row_features:
                feat.row_index = row_index
            self.features.extend(row_features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> WindowFeature:
        return self.features[idx]


def collate(features: list[WindowFeature]) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        "token_type_ids": torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        "offsets": torch.tensor([f.offsets for f in features], dtype=torch.long),
        "content_mask": torch.tensor([f.content_mask for f in features], dtype=torch.bool),
        "row_index": torch.tensor([f.row_index for f in features], dtype=torch.long),
        "field_index": torch.tensor([f.field_index for f in features], dtype=torch.long),
        "direction_label": torch.tensor([f.direction_label for f in features], dtype=torch.long),
    }
    if features[0].labels is not None:
        batch["labels"] = torch.tensor([f.labels for f in features], dtype=torch.long)
    if features[0].start_labels is not None:
        batch["start_labels"] = torch.tensor([f.start_labels for f in features], dtype=torch.long)
        batch["end_labels"] = torch.tensor([f.end_labels for f in features], dtype=torch.long)
        batch["exists_labels"] = torch.tensor([f.exists_labels for f in features], dtype=torch.float32)
    return batch


class TokenStructureModel(nn.Module):
    def __init__(self, model_dir: str, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.span_heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in SPAN_FIELDS])
        self.direction_head = nn.Linear(hidden, len(DIR_LABELS))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        seq = self.dropout(outputs.last_hidden_state)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return {
            "span_logits": torch.stack([head(seq) for head in self.span_heads], dim=1),
            "direction_logits": self.direction_head(pooled),
        }


class SpanStructureModel(nn.Module):
    def __init__(self, model_dir: str, field_count: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.start_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(field_count)])
        self.end_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(field_count)])
        self.exists_head = nn.Linear(hidden, field_count)
        self.direction_head = nn.Linear(hidden, len(DIR_LABELS))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        seq = self.dropout(outputs.last_hidden_state)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        start = torch.stack([head(seq).squeeze(-1) for head in self.start_heads], dim=1)
        end = torch.stack([head(seq).squeeze(-1) for head in self.end_heads], dim=1)
        return {
            "start_logits": start,
            "end_logits": end,
            "exists_logits": self.exists_head(pooled),
            "direction_logits": self.direction_head(pooled),
        }


def freeze_encoder(model: nn.Module, train_last_layers: int) -> None:
    encoder = model.encoder
    for param in encoder.parameters():
        param.requires_grad = False
    layers = getattr(encoder, "encoder").layer
    for layer in layers[-train_last_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    if hasattr(encoder, "pooler") and encoder.pooler is not None:
        for param in encoder.pooler.parameters():
            param.requires_grad = True


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheme: str, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=torch.tensor([0.25, 4.0, 2.0], device=device))
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], device=device))
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        if scheme == "A":
            labels = batch["labels"]
            span_loss = sum(token_loss_fn(out["span_logits"][:, idx].reshape(-1, 3), labels[:, idx].reshape(-1)) for idx in range(len(SPAN_FIELDS)))
            loss = span_loss + 0.3 * ce_loss(out["direction_logits"], batch["direction_label"])
        else:
            start_labels = batch["start_labels"]
            end_labels = batch["end_labels"]
            exists_labels = batch["exists_labels"]
            field_count = start_labels.shape[1]
            span_loss = torch.tensor(0.0, device=device)
            active_fields = 0
            valid_mask = batch["content_mask"]
            masked_start = out["start_logits"].masked_fill(~valid_mask.unsqueeze(1), -1e4)
            masked_end = out["end_logits"].masked_fill(~valid_mask.unsqueeze(1), -1e4)
            for idx in range(field_count):
                positive = exists_labels[:, idx] > 0.5
                if positive.any():
                    span_loss = span_loss + ce_loss(masked_start[positive, idx], start_labels[positive, idx])
                    span_loss = span_loss + ce_loss(masked_end[positive, idx], end_labels[positive, idx])
                    active_fields += 1
            if active_fields:
                span_loss = span_loss / active_fields
            exists_loss = bce_loss(out["exists_logits"], exists_labels)
            loss = span_loss + exists_loss + 0.2 * ce_loss(out["direction_logits"], batch["direction_label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))


def choose_bio_span(probs: torch.Tensor, offsets: torch.Tensor, mask: torch.Tensor, content: str, field: str, mandatory: bool) -> tuple[dict[str, Any], float]:
    labels = torch.argmax(probs, dim=-1).tolist()
    b_scores = probs[:, 1].tolist()
    i_scores = probs[:, 2].tolist()
    best: tuple[int, int, float] | None = None
    i = 0
    while i < len(labels):
        if not mask[i]:
            i += 1
            continue
        if labels[i] == 1 or (mandatory and not best and labels[i] == 2):
            start_i = i
            scores = [b_scores[i]]
            i += 1
            while i < len(labels) and mask[i] and labels[i] == 2:
                scores.append(i_scores[i])
                i += 1
            score = sum(scores) / max(1, len(scores))
            end_i = i - 1
            if best is None or score > best[2]:
                best = (start_i, end_i, score)
            continue
        i += 1
    if best is None and mandatory:
        valid = [(idx, b_scores[idx]) for idx in range(len(labels)) if mask[idx]]
        if valid:
            idx, score = max(valid, key=lambda item: item[1])
            best = (idx, idx, score)
    if best is None:
        return empty_span(field), 0.0
    start = int(offsets[best[0]][0])
    end = int(offsets[best[1]][1])
    return as_span(content, start, end, field), best[2]


def choose_span_pair(start_logits: torch.Tensor, end_logits: torch.Tensor, exists_logit: torch.Tensor, offsets: torch.Tensor, mask: torch.Tensor, content: str, field: str, mandatory: bool) -> tuple[dict[str, Any], float]:
    valid_indices = [idx for idx, value in enumerate(mask.tolist()) if value]
    if not valid_indices:
        return empty_span(field), 0.0
    start_scores = torch.softmax(start_logits.masked_fill(~mask, -1e4), dim=-1)
    end_scores = torch.softmax(end_logits.masked_fill(~mask, -1e4), dim=-1)
    exists_score = float(torch.sigmoid(exists_logit).detach().cpu())
    best: tuple[int, int, float] | None = None
    max_tokens = 150 if field == "event_span" else 32
    for s in valid_indices:
        end_limit = min(valid_indices[-1], s + max_tokens)
        for e in valid_indices:
            if e < s or e > end_limit:
                continue
            score = float(start_scores[s] * end_scores[e]) * max(exists_score, 0.05)
            if best is None or score > best[2]:
                best = (s, e, score)
    if best is None or (not mandatory and exists_score < 0.45):
        return empty_span(field), exists_score
    start = int(offsets[best[0]][0])
    end = int(offsets[best[1]][1])
    return as_span(content, start, end, field), best[2]


def enforce_prediction(content: str, pred: dict[str, Any]) -> dict[str, Any]:
    trigger = pred["trigger"]
    event_span = pred["event_span"]
    if trigger.get("text") and not (
        event_span.get("text")
        and event_span["start"] <= trigger["start"] < trigger["end"] <= event_span["end"]
    ):
        sent = sentence_containing(content, trigger["start"], trigger["end"])
        if sent.get("text"):
            pred["event_span"] = sent
    for field in SPAN_FIELDS:
        span = pred[field]
        text = span.get("text", "")
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
        if text and not (0 <= start <= end <= len(content) and content[start:end] == text):
            pred[field] = empty_span(field)
    args = [
        {"text": pred[role]["text"], "role": role, "start": pred[role]["start"], "end": pred[role]["end"]}
        for role in ROLE_FIELDS
        if pred[role].get("text")
    ]
    pred["arguments"] = args
    pred["argument_roles"] = list(args)
    return pred


def build_model(model_dir: str, scheme: str) -> nn.Module:
    if scheme == "A":
        model: nn.Module = TokenStructureModel(model_dir)
        freeze_encoder(model, train_last_layers=2)
        return model
    field_count = 1 if scheme == "B" else len(SPAN_FIELDS)
    model = SpanStructureModel(model_dir, field_count=field_count)
    freeze_encoder(model, train_last_layers=2 if scheme == "C" else 1)
    return model


def save_checkpoint(path: Path, model: nn.Module, scheme: str, model_dir: str, epoch: int, train_log: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scheme": scheme,
            "scheme_name": SCHEME_NAMES[scheme],
            "scheme_type": SCHEME_TYPES[scheme],
            "model_dir": model_dir,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "max_length": MAX_LENGTH,
            "stride": STRIDE,
            "train_log": train_log,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=False)
    model = build_model(data["model_dir"], data["scheme"])
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    return model, data


def predict_rows(model: nn.Module, tokenizer: Any, rows: list[dict[str, Any]], scheme: str, device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    dataset = StructureDataset([allowed_projection(row) for row in rows], tokenizer, scheme, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    per_row: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        projected = allowed_projection(row)
        per_row[idx] = {
            "event_mention_id": projected["event_mention_id"],
            "article_id": projected["article_id"],
            "scheme": SCHEME_NAMES[scheme],
            "direction_scores": torch.zeros(len(DIR_LABELS)),
            "direction_count": 0,
            "field_candidates": {field: [] for field in SPAN_FIELDS},
            "content": projected["content"],
        }
    with torch.no_grad():
        for batch in loader:
            row_indices = batch["row_index"].tolist()
            field_indices = batch["field_index"].tolist()
            offsets = batch["offsets"]
            masks = batch["content_mask"]
            device_batch = {k: v.to(device) for k, v in batch.items() if k not in {"offsets", "content_mask", "row_index", "field_index"}}
            out = model(device_batch)
            dir_probs = torch.softmax(out["direction_logits"].detach().cpu(), dim=-1)
            if scheme == "A":
                span_probs = torch.softmax(out["span_logits"].detach().cpu(), dim=-1)
                for item_idx, row_idx in enumerate(row_indices):
                    row_state = per_row[row_idx]
                    row_state["direction_scores"] += dir_probs[item_idx]
                    row_state["direction_count"] += 1
                    for field_idx, field in enumerate(SPAN_FIELDS):
                        mandatory = field not in OPTIONAL_FIELDS
                        span, score = choose_bio_span(span_probs[item_idx, field_idx], offsets[item_idx], masks[item_idx], row_state["content"], field, mandatory)
                        row_state["field_candidates"][field].append((score, span))
            else:
                start_logits = out["start_logits"].detach().cpu()
                end_logits = out["end_logits"].detach().cpu()
                exists_logits = out["exists_logits"].detach().cpu()
                for item_idx, row_idx in enumerate(row_indices):
                    row_state = per_row[row_idx]
                    row_state["direction_scores"] += dir_probs[item_idx]
                    row_state["direction_count"] += 1
                    if scheme == "B":
                        field = SPAN_FIELDS[field_indices[item_idx]]
                        mandatory = field not in OPTIONAL_FIELDS
                        span, score = choose_span_pair(start_logits[item_idx, 0], end_logits[item_idx, 0], exists_logits[item_idx, 0], offsets[item_idx], masks[item_idx], row_state["content"], field, mandatory)
                        row_state["field_candidates"][field].append((score, span))
                    else:
                        for field_idx, field in enumerate(SPAN_FIELDS):
                            mandatory = field not in OPTIONAL_FIELDS
                            span, score = choose_span_pair(start_logits[item_idx, field_idx], end_logits[item_idx, field_idx], exists_logits[item_idx, field_idx], offsets[item_idx], masks[item_idx], row_state["content"], field, mandatory)
                            row_state["field_candidates"][field].append((score, span))
    predictions: list[dict[str, Any]] = []
    for row_idx in range(len(rows)):
        state = per_row[row_idx]
        pred: dict[str, Any] = {
            "event_mention_id": state["event_mention_id"],
            "article_id": state["article_id"],
            "scheme": state["scheme"],
        }
        for field in SPAN_FIELDS:
            candidates = state["field_candidates"][field]
            nonempty = [(score, span) for score, span in candidates if span.get("text")]
            if nonempty:
                _, chosen = max(nonempty, key=lambda item: (item[0], len(item[1].get("text", ""))))
                pred[field] = chosen
            else:
                pred[field] = empty_span(field)
        direction_scores = state["direction_scores"] / max(1, state["direction_count"])
        pred["direction"] = DIR_LABELS[int(torch.argmax(direction_scores).item())]
        pred = enforce_prediction(state["content"], pred)
        predictions.append(pred)
    return predictions


def train(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    train_rows = read_jsonl(Path(args.train))
    dev_rows = read_jsonl(Path(args.dev))
    train_dataset = StructureDataset(train_rows, tokenizer, args.scheme, train=True)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = build_model(args.model_dir, args.scheme).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    started = time.time()
    epoch_losses = []
    for epoch in range(args.epochs):
        epoch_losses.append(train_epoch(model, loader, optimizer, args.scheme, device))
    train_seconds = time.time() - started
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log = {
        "scheme": args.scheme,
        "scheme_name": SCHEME_NAMES[args.scheme],
        "scheme_type": SCHEME_TYPES[args.scheme],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_window_count": len(train_dataset),
        "dev_row_count": len(dev_rows),
        "epoch_losses": [round(value, 6) for value in epoch_losses],
        "train_seconds": round(train_seconds, 3),
        "device": str(device),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3) if device.type == "cuda" else 0.0,
        "token_window_coverage": token_window_coverage(tokenizer, train_rows, args.scheme),
    }
    checkpoint = out_dir / "model.pt"
    save_checkpoint(checkpoint, model, args.scheme, args.model_dir, args.epochs, train_log)
    preds = predict_rows(model, tokenizer, dev_rows, args.scheme, device, args.batch_size)
    write_jsonl(out_dir / "dev_predictions.jsonl", preds)
    train_log["checkpoint_sha256"] = sha256_file(checkpoint)
    write_json(out_dir / "worker_training_log.json", train_log)
    print(json.dumps({"scheme": args.scheme, "checkpoint": str(checkpoint), "sha256": train_log["checkpoint_sha256"], "train_seconds": train_log["train_seconds"]}, ensure_ascii=False))
    return 0


def predict(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    model, data = load_checkpoint(Path(args.checkpoint), device)
    rows = read_jsonl(Path(args.input))
    started = time.time()
    preds = predict_rows(model, tokenizer, rows, data["scheme"], device, args.batch_size)
    elapsed = time.time() - started
    write_jsonl(Path(args.output), preds)
    write_json(
        Path(args.output).with_suffix(".manifest.json"),
        {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
            "input": args.input,
            "output": args.output,
            "row_count": len(rows),
            "predict_seconds": round(elapsed, 3),
            "avg_inference_ms": round(elapsed * 1000 / max(1, len(rows)), 3),
            "device": str(device),
        },
    )
    print(json.dumps({"rows": len(rows), "output": args.output, "seconds": round(elapsed, 3)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--scheme", choices=["A", "B", "C"], required=True)
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--dev", required=True)
    train_parser.add_argument("--out-dir", required=True)
    train_parser.add_argument("--model-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=6)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--seed", type=int, default=SEED)
    train_parser.add_argument("--cpu", action="store_true")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--model-dir", required=True)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    predict_parser.add_argument("--seed", type=int, default=SEED)
    predict_parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.mode == "train":
        return train(args)
    return predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
