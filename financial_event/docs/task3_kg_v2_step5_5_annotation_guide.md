# Task3 KG v2 Step5.5 Annotation Guide

- `same_state_repeat` 必须回流到 EvidenceUnit/CanonicalEvent 审计，不得作为状态迁移正例。
- `actual_transition_positive` 仅表示 AI 草稿候选，不能替代人工标签。
- 标注顺序：先 `human_same_thread`，后 `human_transition_type`。
- 若对象锚点不稳定、时间方向不清或证据句不足，应优先标为 uncertain 或 no。
