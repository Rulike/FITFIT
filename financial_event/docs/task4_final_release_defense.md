# Task4 Final Release Defense

## Final status
`task4_core_complete_alert_prototype_only`

## Freeze decisions
- Stage 3.2 is frozen as the formal supervised prediction main chain.
  - Validity: `V4`
  - Impact Direction: `D2-hierarchical`
- Stage 4.1 is frozen as a read-only knowledge-graph evidence audit and review-ranking layer.
  - It does **not** claim predictive gain.

## Strict evaluation boundary
### Formal evaluation
- Validity
- Impact Direction

### Product demo only
- Severity
- Alert Priority

Severity and Alert Priority are shown only through `ImpactAlertCard` prototype artifacts.
All such labels are marked:
- `label_source=agent_assisted_prototype`
- `verification_status=not_human_gold`

## Formal metrics snapshot
- Validity test precision: 0.400563
- Validity test recall: 0.873211
- Validity test F1: 0.549196
- Direction macro F1: 0.556105
- Direction balanced accuracy: 0.608176

## Why Stage 4.1 is read-only
The frozen Stage 4.1 summary explicitly records:
- `stage3_2_frozen_prediction_chain_stage4_read_only_audit`
- `claim_predictive_gain = false`
- no legal Pareto workpoint can be established without prohibited reruns

Therefore Stage 4.1 is retained only for:
- evidence audit
- review ordering
- uncertainty surfacing

## Released artifacts
- `outputs/task4/final_release/task4_prediction_results.json`
- `outputs/task4/final_release/impact_alert_cards_demo.json`
- `outputs/task4/final_release/evidence_path_demo.json`
- `outputs/task4/final_release/uncertain_review_queue.json`
- `outputs/task4/final_release/task3_task4_joint_query_interface.json`
- `outputs/task4/final_release/task4_end_to_end_cases.json`
- `outputs/task4/final_release/task4_formal_metrics_efficiency_ablation_report.json`

## Prohibited claims not made
- No agent severity/alert labels are called human gold.
- No Human Kappa is reported for severity/alert.
- No QWK is reported as formal evaluation.
- No additional severity/alert supervised model is trained.
- No waiting for remaining 370 labels.
