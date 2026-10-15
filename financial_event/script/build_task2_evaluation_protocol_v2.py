#!/usr/bin/env python3
"""Write Task2 evaluation protocol v2 and retire the old locked500.

This script records governance metadata only. It does not train, tune, run
evaluation, generate predictions, or use old gold labels for sampling.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OLD_LOCKED = ROOT / "outputs/event_credibility/task2_independent500_v1/independent_samples.jsonl"
OLD_EVALUATION = ROOT / "outputs/event_credibility/task2_market_rule_enhanced_cached_dea_v1/locked500_evaluation.json"
DEV_CONFIG = ROOT / "configs/task2_human_gold_development_v1.json"
PROTOCOL_JSON = ROOT / "outputs/event_credibility/task2_hybrid_v3/evaluation_protocol_v2.json"
PROTOCOL_MD = ROOT / "reports/task2_evaluation_protocol_v2.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def jsonl_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render_protocol_markdown(protocol: dict[str, Any]) -> str:
    old = protocol["retired_independent_test"]
    lines = [
        "# Task2 Evaluation Protocol v2",
        "",
        "## 生效状态",
        "",
        f"- 生效时间：{protocol['effective_time']}",
        "- 本协议将旧 `task2_independent500_v1` 正式降级为 `human_gold_development_v1`。",
        "- 从本协议生效后，旧500条永远不得再被称为独立测试集。",
        "",
        "## 旧500退役原因",
        "",
        old["retired_reason"],
        "",
        "## 历史使用",
        "",
    ]
    for item in old["historical_usage"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 允许用途",
            "",
        ]
    )
    for item in old["allowed_uses"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 禁止用途", ""])
    for item in old["forbidden_uses"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 原始资产",
            "",
            f"- 路径：`{old['source_path']}`",
            f"- SHA256：`{old['source_sha256']}`",
            f"- 行数：{old['row_count']}",
            f"- 标签分布：`{json.dumps(old['label_distribution'], ensure_ascii=False)}`",
            "",
            "## 新独立测试要求",
            "",
            "- `independent500_v2` 的ID冻结后不得修改。",
            "- 标注工作簿不得包含模型预测、旧标签、银标、置信分或规则verdict。",
            "- 新500条gold填写完成前，不得运行模型评测或调参。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    if not OLD_LOCKED.exists():
        raise FileNotFoundError(rel(OLD_LOCKED))
    evaluation = read_json(OLD_EVALUATION)
    gold_meta = evaluation.get("gold") if isinstance(evaluation.get("gold"), dict) else {}
    label_distribution = gold_meta.get("distribution") if isinstance(gold_meta.get("distribution"), dict) else {}
    effective_time = now_iso()
    old_sha = sha256(OLD_LOCKED)
    row_count = jsonl_count(OLD_LOCKED)
    old_config = {
        "dataset_id": "task2_human_gold_development_v1",
        "previous_dataset_id": "task2_independent500_v1",
        "new_status": "retired_independent_test",
        "effective_time": effective_time,
        "source_path": rel(OLD_LOCKED),
        "source_sha256": old_sha,
        "row_count": row_count,
        "label_distribution": label_distribution,
        "retired_reason": (
            "The old locked500 gold has been read multiple times and used for retrospective error analysis, "
            "so it no longer satisfies the independence requirements for a blind test set."
        ),
        "historical_usage": [
            "Built as task2_independent500_v1 blind audit package.",
            "Auto-assisted human-confirmed gold was created after the blind package.",
            "Gold was read in later locked500 evaluations.",
            "Gold was read again for retrospective locked500 error diagnosis and market-contradictory audit.",
        ],
        "allowed_uses": [
            "training after explicit inclusion in a development protocol",
            "development-set error analysis",
            "rule debugging on development data",
            "documentation of previous system behavior",
        ],
        "forbidden_uses": [
            "official independent testing",
            "claims of blind or leakage-free independent performance",
            "threshold/model/weight selection reported as independent-test tuning",
            "comparison baselines presented as final independent results",
        ],
        "constraints": {
            "original_file_preserved": True,
            "original_sha256_preserved": old_sha == "80a3666317a23e5a3c79539a6c000a11293d08fa24efdf41af06dfffc2a5472c",
            "training_performed_by_this_script": False,
            "tuning_performed_by_this_script": False,
            "evaluation_run_by_this_script": False,
            "predictions_generated_by_this_script": False,
            "old_gold_label_values_used_for_new_sampling": False,
        },
    }
    protocol = {
        "protocol_id": "task2_evaluation_protocol_v2",
        "effective_time": effective_time,
        "project_root": str(ROOT),
        "retired_independent_test": old_config,
        "independent500_v2_policy": {
            "status": "to_be_frozen_by_configs/task2_independent500_v2_locked_ids.json",
            "gold_may_be_read_before_prediction_freeze": False,
            "model_predictions_may_be_generated_before_human_gold": False,
            "model_predictions_may_be_used_for_sampling": False,
            "sampling_seed": 20260826,
            "label_contract": ["support", "contradict", "insufficient", "unclear"],
        },
        "this_script_actions": {
            "training_performed": False,
            "tuning_performed": False,
            "locked500_evaluation_run": False,
            "new500_evaluation_run": False,
            "model_predictions_read_for_sampling": False,
        },
    }
    return old_config, protocol


def main() -> int:
    old_config, protocol = build_payload()
    write_json(DEV_CONFIG, old_config)
    write_json(PROTOCOL_JSON, protocol)
    PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_MD.write_text(render_protocol_markdown(protocol), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "old_locked500_demoted_to": old_config["dataset_id"],
                "source_sha256": old_config["source_sha256"],
                "label_distribution": old_config["label_distribution"],
                "outputs": {
                    "development_config": rel(DEV_CONFIG),
                    "protocol_json": rel(PROTOCOL_JSON),
                    "protocol_md": rel(PROTOCOL_MD),
                },
                "training_performed": False,
                "tuning_performed": False,
                "evaluation_run": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
