#!/usr/bin/env bash
set -euo pipefail

MAX_LLM_REVIEWS="${MAX_LLM_REVIEWS:-260}"
LLM_SLEEP_SECONDS="${LLM_SLEEP_SECONDS:-1.0}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-4}"
LLM_RETRY_BACKOFF_SECONDS="${LLM_RETRY_BACKOFF_SECONDS:-1.5}"
MODEL="${TASK2_LLM_MODEL:-gpt-5.5}"

python scripts/run_task2_llm_review.py \
  --base_url "https://mdkj.lol" \
  --wire_api responses \
  --endpoint_style openai_v1 \
  --model "${MODEL}" \
  --api_key_getpass \
  --max_llm_reviews "${MAX_LLM_REVIEWS}" \
  --llm_sleep_seconds "${LLM_SLEEP_SECONDS}" \
  --max_retries "${LLM_MAX_RETRIES}" \
  --retry_backoff_seconds "${LLM_RETRY_BACKOFF_SECONDS}"
