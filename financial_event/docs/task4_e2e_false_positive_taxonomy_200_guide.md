# Task4 E2E False-Positive Taxonomy 200

Use the suggested taxonomy as a pre-audit label only.

Labels:
- `event_validity_failure`
- `target_grounding_failure`
- `evidence_failure`
- `event_type_failure`
- `rumor_as_fact`
- `forecast_as_fact`
- `duplicate_or_propagation_noise`
- `other`

Heuristic taxonomy counts on the full FP pool: {"target_grounding_failure": 196, "duplicate_or_propagation_noise": 121, "forecast_as_fact": 59, "event_type_failure": 54, "event_validity_failure": 51, "evidence_failure": 21, "rumor_as_fact": 11}
Main root cause: target_grounding_failure
