# Task3 Step3 Same Event Matching

- evidence_scope: `title_structured_only`
- semantic model: `/opt/data/private/JinRong/kechuang/downloads/Qwen/Qwen3-Embedding-0.6B`
- reranker model: `/opt/data/private/JinRong/kechuang/downloads/Qwen/Qwen3-Reranker-0.6B`
- candidate_count: 81566
- pilot_candidate_recall: 1.000000 (50 agent-assisted pilot pairs only, not full-corpus recall)
- final Proposed: `structure_plus_qwen3_embedding`
- reranker retained in Proposed: False
- reranker role: ablation and optional review tool
- prediction_counts: {'different_event': 75336, 'related_update': 90, 'same_event': 135, 'uncertain_review': 6005}
- full refusal_rate: 0.073621

Metrics are reported in two separate views: event_related binary metrics merge same_event and related_update as the positive class; per-relation metrics report same_event, related_update, and different_event separately. Macro-F1 is computed over the three relation classes.

Threshold calibration uses the 50 agent-assisted pilot pairs from development/validation. Stratified 5-fold OOF metrics are reported for audit, but the final thresholds are selected on the full pilot set. Test split count for threshold calibration is 0.

Reranker audit: pilot_scored_count=5, full_scored_count=0, decision_used_count=0, cache_hit_count=5, cache_miss_count=0, cache_total_rows=2116.

Cold start benchmark: embedding=11.803779s, reranker=2.005659s, full=13.825146s, GPU peak=2411.271MB. Cache rerun=79.260872s.

The 50 pilot labels are agent-assisted pilot labels pending human audit, not a formal test set.
This step outputs pair relations only and does not perform event clustering.
