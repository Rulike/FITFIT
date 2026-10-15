#!/usr/bin/env bash
set -euo pipefail

cd /opt/data/private/JinRong/kechuang/financial_event_project

python scripts/run_task2_fulltext_credibility_experiment.py "$@"

