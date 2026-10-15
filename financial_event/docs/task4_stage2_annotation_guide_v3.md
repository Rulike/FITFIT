# Task4 Annotation Guide v3

This version is for human confirmation after agent-assisted adjudication.

## Core rule
- Do not copy agent labels blindly.
- Use title and direct evidence first.
- `impact_rationale_reference` is not available in this blind sheet and must not be treated as a label source.

## Severity boundaries
- S0: no material business / regulatory / market impact.
- S1: limited or short-lived impact; mostly informational.
- S2: clear but moderate impact; visible operational or financial effect.
- S3: major impact on business, finance, regulation, or reputation.
- S4: systemic or potentially existential impact.

## Alert priority boundaries
- A0: no alert.
- A1: archive / low-priority digest.
- A2: normal monitoring alert.
- A3: high-priority alert.
- A4: urgent alert.

## Practical tie-breaks
- If headline is dramatic but body only reports routine business progress, prefer lower severity/priority.
- If the body contains concrete loss, penalty, accident, regulatory action, or disruption, escalate severity/priority.
- If evidence is weak, choose conservative labels and add a precise `human_evidence` quote from title or body.

## Evidence rule
- `human_evidence` must quote or tightly paraphrase a span from the title or body text.
- Do not use agent final labels as evidence.
