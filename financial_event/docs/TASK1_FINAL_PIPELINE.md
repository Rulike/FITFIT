# Task1 Final Pipeline

Inputs are official-format article rows. Outputs exactly match the official Excel schema.

Pipeline: target fields preserved from the official candidate universe -> event_name classifier over official64 labels -> event_emotion classifier -> evidence-constrained impact template -> deterministic duplication clustering.

Forbidden final fields: trigger, event_span, arguments, argument_roles, external source labels, provenance/debug fields.

## Step 5.8-F Update

Status: `task1_duplication_validation_insufficient`. Event_name remains frozen from Step 5.8-E. Step 5.8-F only updates event_emotion and duplication_id candidate logic; no formal output is overwritten.
