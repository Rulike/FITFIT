#!/usr/bin/env python3
"""event_match_text.py — 统一的文本构造和截断模块，保持与第六步完全一致"""

import re
import numpy as np


def build_input_text(rec):
    """构造模型输入文本。不含 label/negative_type。"""
    return (
        f"[目标]\n类型：{rec.get('target_type', '')}\n名称：{rec.get('target_name', '')}\n代码：{rec.get('target_code', '')}\n\n"
        f"[候选事件]\n事件名称：{rec.get('event_name', '')}\n事件定义：{rec.get('event_description', '')}\n\n"
        f"[新闻]\n标题：{rec.get('title', '')}\n正文：{rec.get('content', '')}"
    )


def smart_truncate(text, tokenizer, max_length=512):
    """智能截断：优先保留目标+事件+标题，正文头70%+尾30%。与第六步完全一致。"""
    marker = "\n[新闻]\n"
    if marker in text:
        schema_part, news_part = text.split(marker, 1)
        schema_part += marker
    else:
        schema_part = text[:int(len(text) * 0.6)]
        news_part = text[int(len(text) * 0.6):]

    schema_tokens = tokenizer.encode(schema_part, add_special_tokens=False)
    available = max_length - 2 - len(schema_tokens)

    if available <= 0:
        desc_marker = "事件定义："
        if desc_marker in schema_part:
            idx = schema_part.index(desc_marker)
            prefix = schema_part[:idx + len(desc_marker)]
            desc = schema_part[idx + len(desc_marker):]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            desc_budget = max_length - 2 - len(prefix_tokens)
            if desc_budget > 0:
                desc_tokens = tokenizer.encode(desc, add_special_tokens=False)
                return prefix + tokenizer.decode(desc_tokens[:desc_budget], skip_special_tokens=True)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(tokens[:max_length - 2], skip_special_tokens=True)

    news_tokens = tokenizer.encode(news_part, add_special_tokens=False)
    if len(news_tokens) <= available:
        return text

    title_marker = "标题："
    body_marker = "\n正文："
    title_text = ""
    body_text = news_part
    if title_marker in news_part and body_marker in news_part:
        t_idx = news_part.index(title_marker)
        b_idx = news_part.index(body_marker)
        title_text = news_part[t_idx:b_idx + len(body_marker)]
        body_text = news_part[b_idx + len(body_marker):]

    title_tokens = tokenizer.encode(title_text, add_special_tokens=False)
    body_budget = available - len(title_tokens)

    if body_budget <= 0:
        kept_news = title_text
    else:
        body_tokens = tokenizer.encode(body_text, add_special_tokens=False)
        head_budget = int(body_budget * 0.7)
        tail_budget = body_budget - head_budget - 10
        if tail_budget <= 0:
            kept_body = tokenizer.decode(body_tokens[:body_budget], skip_special_tokens=True)
        else:
            kept_body = (
                tokenizer.decode(body_tokens[:head_budget], skip_special_tokens=True) +
                "\n[正文中间省略]\n" +
                tokenizer.decode(body_tokens[-tail_budget:], skip_special_tokens=True)
            )
        kept_news = title_text + kept_body

    return schema_part + kept_news
