# Task4 并行候选端到端性能修复报告

## Dynamic Demo真实性审计
- 当前Dynamic Demo中的V4逐样本真实推理：False
- 当前Dynamic Demo中的D2-hierarchical逐样本真实推理：False
- 说明：当前Dynamic Demo中的约3ms级时延仅代表shadow规则+图更新，不代表完整模型端到端真实推理延迟。

## 并行候选版本结果
### groupwise_pointwise
- Candidate Recall@1/3/5: {'Recall@1': 0.5828220858895705, 'Recall@3': 0.8732106339468303, 'Recall@5': 0.9601226993865031}
- Retained Recall: 0.628834
- Validity P/R/F1/PR-AUC: {'TP': 615, 'FP': 412, 'FN': 363, 'precision': 0.5988315481986368, 'recall': 0.6288343558282209, 'f1': 0.6134663341645885, 'pr_auc': 0.5158154052598024}
- Count Accuracy/MAE/Macro-F1: {'accuracy': 0.9280855199222546, 'mae': 0.07191448007774538, 'macro_f1': 0.38303440496333896, 'per_class_recall': {'0': 0.109375, '1': 0.9957983193277311, '2+': 0.0}}
- Multi-event exact match: 0.651118
- Subgroup Recall: {'company_recall': 0.6292749658002736, 'industry_recall': 0.6275303643724697, 'multi_event_recall': 0.34615384615384615}
- Review Rate / Automatic Coverage: 0.104584 / 0.096071
- Joint P/R/F1: 0.536514 / 0.563395 / 0.549626
- ECE / Brier: 0.242457 / 0.108335
- P50 / P95 latency: 0.000507 / 0.000684

### pairwise_ranking
- Candidate Recall@1/3/5: {'Recall@1': 0.5828220858895705, 'Recall@3': 0.8732106339468303, 'Recall@5': 0.9601226993865031}
- Retained Recall: 0.336401
- Validity P/R/F1/PR-AUC: {'TP': 329, 'FP': 112, 'FN': 649, 'precision': 0.746031746031746, 'recall': 0.33640081799591004, 'f1': 0.46370683579985905, 'pr_auc': 0.6329005965774318}
- Count Accuracy/MAE/Macro-F1: {'accuracy': 0.9280855199222546, 'mae': 0.07191448007774538, 'macro_f1': 0.38303440496333896, 'per_class_recall': {'0': 0.109375, '1': 0.9957983193277311, '2+': 0.0}}
- Multi-event exact match: 0.380952
- Subgroup Recall: {'company_recall': 0.3337893296853625, 'industry_recall': 0.3441295546558704, 'multi_event_recall': 0.038461538461538464}
- Review Rate / Automatic Coverage: 0.049205 / 0.041254
- Joint P/R/F1: 0.657596 / 0.296524 / 0.408739
- ECE / Brier: 0.246259 / 0.114946
- P50 / P95 latency: 0.000449 / 0.000606

### listwise_ranking
- Candidate Recall@1/3/5: {'Recall@1': 0.5828220858895705, 'Recall@3': 0.8732106339468303, 'Recall@5': 0.9601226993865031}
- Retained Recall: 0.347648
- Validity P/R/F1/PR-AUC: {'TP': 340, 'FP': 131, 'FN': 638, 'precision': 0.721868365180467, 'recall': 0.3476482617586912, 'f1': 0.46928916494133877, 'pr_auc': 0.6095775065150711}
- Count Accuracy/MAE/Macro-F1: {'accuracy': 0.9280855199222546, 'mae': 0.07191448007774538, 'macro_f1': 0.38303440496333896, 'per_class_recall': {'0': 0.109375, '1': 0.9957983193277311, '2+': 0.0}}
- Multi-event exact match: 0.387755
- Subgroup Recall: {'company_recall': 0.36662106703146374, 'industry_recall': 0.291497975708502, 'multi_event_recall': 0.19230769230769232}
- Review Rate / Automatic Coverage: 0.040599 / 0.044060
- Joint P/R/F1: 0.639066 / 0.307771 / 0.415459
- ECE / Brier: 0.248729 / 0.114686
- P50 / P95 latency: 0.000435 / 0.000587

## 版本门控结论
- best_scheme: groupwise_pointwise
- replace_stage3_2: False
- decision: freeze_stage3_2_keep_parallel_candidate_only
- dynamic_demo_integration: keep_existing_demo_and_label_real_vs_cache_vs_rule