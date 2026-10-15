{
  "title": "Task4 v3 Step1 Final Frozen Experiment Contract",
  "status": "task4_v3_step1_contract_frozen",
  "experiments": [
    "R0",
    "R0.5",
    "R1",
    "R2",
    "R3"
  ],
  "sample_counts": {
    "oracle_total": 3127,
    "oracle_train": 2081,
    "oracle_test": 1046,
    "e2e_total": 2876,
    "e2e_train": 1898,
    "e2e_test": 978
  },
  "acceptance_gate": {
    "oof_stable_improvement": true,
    "paired_bootstrap_ci_lower_bound_gt_zero": true,
    "balanced_accuracy_not_degraded": true,
    "neutral_recall_not_degraded": true,
    "negative_recall_not_degraded": true,
    "test_improvement_direction_consistent": true,
    "no_test_leakage": true,
    "no_future_information": true,
    "no_unknown_agent_labels": true,
    "no_core_mutation": true
  }
}