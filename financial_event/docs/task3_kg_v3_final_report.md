# Task3 KG v3 Final Report

- status: `task3_kg_v3_release_ready`
- selected_canonical_event_version: `step5_6_repaired_v2`
- final_status: `task3_kg_v3_final_complete`
- acceptance_status: `task3_kg_v3_evolution_backbone_ready`

## Final method statement

通过 EvidenceUnit 消除转载证据的重复计数，以结构化事件身份和图硬约束控制错误归并，在身份冻结后执行可信度传播，并通过带证据的 StateObservation 提取稀疏事件演化骨架，最终构建可查询、可溯源、可增量更新的动态事件知识图谱。

## Backbone explanation

- direct_evidence_transition: direct_evidence_transition 仅保留同时满足状态机语义兼容、双端 EvidenceSentence 可追溯、独立 EvidenceUnit 支撑且非 review/proposal 端点的边，因此从 69447 条原始迁移中只保留 1080 条。
- progress_to_completion: 原始 PROGRESS_TO_COMPLETION 共 622 条，其中 522 条进入 direct_evidence_transition，100 条因端点含 review/proposal 标记进入 uncertain_review。
- announcement_to_progress: 原始 ANNOUNCEMENT_TO_PROGRESS 共 459 条，其中 331 条进入 direct_evidence_transition，128 条因端点含 review/proposal 标记进入 uncertain_review。
- count_difference: 622+459=1081 是两类原始状态机迁移的总数，不等于 direct 总数。direct=1080 还包含 222 条 CLARIFIES_OR_REFUTES 与 5 条 RUMOR_TO_CONFIRMATION，同时有 228 条 PROGRESS/ANNOUNCEMENT 边被分流到 uncertain_review；因此 1081 与 1080 不应直接相减解读为错误。

## Release freeze

- No new revision
- No Task4 transition
- No same-event universal gain claim
- No formal human gold overclaim

## Efficiency

- runtime_seconds_current_release: `18.128`
- avg_seconds_per_event_mention_current_release: `0.003645`
- max_rss_mb_current_release: `1659.43`
