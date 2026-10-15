# Task4 Grounding Audit 200

Formal chain: `sample_id -> target -> event -> evidence_sentence -> provenance`.

Gate rules:
- evidence must be non-empty.
- target must be verifiable from the evidence sentence.
- provenance must be traceable.
- if any gate fails, the sample must not enter Direction / Severity supervision.

Eligible ratio on the current accepted-candidate pool: 0.655882
Failure counts: {"target_grounding_failure": 349, "provenance_failure": 2, "evidence_failure": 0}
