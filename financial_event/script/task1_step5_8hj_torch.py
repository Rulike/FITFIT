#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch/Transformers helpers for Task1 Step 5.8-HJ."""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.task1_step5_8hj_common import LOCAL_MACBERT, sha256_file, write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_bottom_layers(model: torch.nn.Module, keep_last_n: int = 4) -> int:
    frozen = 0
    if hasattr(model, "base_model"):
        base = model.base_model
    elif hasattr(model, "bert"):
        base = model.bert
    elif hasattr(model, "roberta"):
        base = model.roberta
    else:
        base = model
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if layers is None:
        return frozen
    total = len(layers)
    for idx, layer in enumerate(layers):
        trainable = idx >= max(0, total - keep_last_n)
        for param in layer.parameters():
            param.requires_grad = trainable
        if not trainable:
            frozen += 1
    return frozen


def class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_labels).astype(float)
    weights = np.zeros(num_labels, dtype=float)
    valid = counts > 0
    weights[valid] = len(labels) / (num_labels * counts[valid])
    weights[~valid] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        self.input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class TextLossTrainer(Trainer):
    def __init__(self, *args: Any, loss_type: str = "ce", class_weight_tensor: torch.Tensor | None = None, focal_gamma: float = 2.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weight_tensor = class_weight_tensor
        self.focal_gamma = focal_gamma

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: Any | None = None):  # type: ignore[override]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.loss_type == "focal":
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            idx = torch.arange(labels.shape[0], device=labels.device)
            pt = probs[idx, labels]
            logpt = log_probs[idx, labels]
            alpha = None
            if self.class_weight_tensor is not None and self.class_weight_tensor.numel() == logits.shape[-1]:
                alpha = self.class_weight_tensor.to(logits.device)[labels]
            else:
                alpha = torch.ones_like(pt)
            loss = -alpha * ((1.0 - pt) ** self.focal_gamma) * logpt
            loss = loss.mean()
        else:
            weight = self.class_weight_tensor.to(logits.device) if self.class_weight_tensor is not None else None
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def make_tokenizer(model_dir: Path | None = None) -> Any:
    return AutoTokenizer.from_pretrained(str(model_dir or LOCAL_MACBERT), local_files_only=True, use_fast=True)


def make_model(num_labels: int, model_dir: Path | None = None) -> Any:
    return AutoModelForSequenceClassification.from_pretrained(str(model_dir or LOCAL_MACBERT), num_labels=num_labels, local_files_only=True)


def eval_multiclass(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(cm), 6),
        "per_class": [
            {"label": labels[i], "precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "f1": round(float(f[i]), 6), "support": int(s[i])}
            for i in range(len(labels))
        ],
    }


def safe_training_args(
    output_dir: Path,
    seed: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_steps: int = -1,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        seed=seed,
        data_seed=seed,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        bf16=False,
        logging_steps=25,
        logging_strategy="steps",
        report_to=[],
        disable_tqdm=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        max_steps=max_steps,
    )


def save_bundle(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = payload.get("model")
    tokenizer = payload.get("tokenizer")
    if model is not None:
        model.save_pretrained(str(output_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))
    meta = {k: v for k, v in payload.items() if k not in {"model", "tokenizer"}}
    write_json(output_dir / "hj_training_meta.json", meta)
    model_file = output_dir / "model.safetensors"
    if not model_file.exists():
        model_file = output_dir / "pytorch_model.bin"
    if model_file.exists():
        meta["model_sha256"] = sha256_file(model_file)
    meta["model_dir"] = str(output_dir)
    return meta
