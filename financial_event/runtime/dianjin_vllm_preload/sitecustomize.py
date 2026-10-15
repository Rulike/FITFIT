"""Preload vLLM attention initialization before Qwen2 inspection."""
import importlib
importlib.import_module("vllm.attention.layer")
