# Task4 Final Metrics Table

## Metric boundary rules
- Candidate-level: measures candidate pool coverage and review burden.
- Conditional Direction: classifier-only evaluation after candidate alignment / conditional framing.
- End-to-End Joint: full chain from candidate pool to retained event plus direction correctness.
- Agent-assisted Prototype: demo-only Severity / Alert Priority, not human gold and not formal metrics.

## Validity metric reconciliation
| Metric tuple | Scope | Dataset / denominator | Threshold | Model | Semantics |
|---|---|---|---|---|---|
| 0.5297 / 0.8016 / 0.6624 | End-to-End Joint validity | candidate pool test: predicted positive=1480, gold positive=978, pool=10690 | V4@0.30 | Stage 3.2 frozen chain | Includes upstream candidate false positives |
| 0.4006 / 0.8732 / 0.5492 | Conditional Validity | classifier test: predicted positive=2132, gold positive=978, samples=10690 | V4@0.30 | Stage 3.2 validity test | Classifier-only, not full candidate-pool end-to-end |

## Candidate-level
| Metric | Value |
|---|---:|
| test_candidate_total | 10690 |
| test_gold_event_total | 978 |
| aligned_candidate_total | 978 |
| candidate_recall | 1.000000 |
| review_rate | 0.064920 |

## Conditional Direction
| Metric | Value |
|---|---:|
| Validity precision | 0.400563 |
| Validity recall | 0.873211 |
| Validity F1 | 0.549196 |
| Direction macro F1 | 0.556105 |
| Direction balanced accuracy | 0.608176 |
| Negative recall | 0.600000 |
| Neutral recall | 0.282051 |
| ECE | 0.119413 |

## End-to-End Joint
| Metric | Value |
|---|---:|
| validity_precision | 0.529730 |
| validity_recall | 0.801636 |
| validity_f1 | 0.662442 |
| joint_event_direction_precision | 0.597297 |
| joint_event_direction_recall | 0.903885 |
| joint_event_direction_f1 | 0.719284 |
| achieved_upstream_ceiling_ratio | 1.127551 |
| retained_false_positive | 696 |
| review_or_abstention_rate | 0.064920 |

## Efficiency
| Metric | Value |
|---|---:|
| embedding_total_seconds | 602.938939 |
| candidate_avg_seconds_per_row | 0.016945 |
| reranker_p50_latency_seconds | 0.030745 |
| reranker_p95_latency_seconds | 0.037799 |
| gpu_peak_mb_embedding | 5193.313 |
| gpu_peak_mb_reranker | 14250.976 |

## Agent-assisted Prototype
| Field | Value |
|---|---|
| ImpactAlertCard count | 22 |
| label_source | agent_assisted_prototype |
| verification_status | not_human_gold |
| formal metric allowed | no |
