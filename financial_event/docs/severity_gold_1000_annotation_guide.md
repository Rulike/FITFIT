# Severity Gold 1000 Annotation Guide

## Annotator view
Only use: sample_id, target_name, target_type, event_name, event_type, title, evidence_sentence, local_context, publish_time, financial_facts.

## Not shown to annotators
- legacy direction
- model predictions
- severity hints
- previous severity labels
- review status
- another annotator result

## Labels
- red:重大且紧急，可能造成严重经营/财务/合规后果。
- orange:明显重要，需要重点关注。
- yellow:一般经营变化，需要持续跟踪。
- blue:低风险、常规信息、轻微波动。
- If evidence is insufficient, set review_required=yes and do not force a level.

## Decision principle
Severity is about intensity and business importance, not polarity. Positive events may still be high severity; negative events are not automatically red.

## Fill fields
- severity_label: red/orange/yellow/blue
- risk_intensity: 1/2/3/4/5
- review_required: yes/no
- evidence_sufficient: yes/no
- annotation_confidence: high/medium/low
- annotation_basis: one-sentence rationale
