# Task4 Stage 4.1

## Conclusion

- Freeze `Stage 3.2` as the predictive main chain.
- Reposition `Stage 4` as a read-only evidence audit and manual-review ranking layer.
- Do not claim predictive gain from Stage 4.

## Candidate closure audit

- Stage 3.2 candidate count: 10690
- Stage 4 candidate count: 10690
- Join expansion factor: 1.0
- Candidate sets equal: True

## Overall-with-abstention

- Validity P/R/F1: 0.5164 / 0.4509 / 0.4814
- Validity retained FP: 413
- Review count: 1278
- Direction Macro-F1: 0.3095

## Selective metrics

- Validity over Stage 3.2 accept pool P/R/F1: 0.5164 / 0.5625 / 0.5385
- Direction over retained subset Macro-F1: 0.5122

## FP reconciliation

- Stage 3.2 FP retained: 696
- Old Stage 4 FP retained: 1126
- Stage 4.1 FP retained: 413
- Illegal promotion FP added: 713
- Baseline FP removed by audit: 283

## Pareto protocol status

- Legal Pareto possible: False
- Reason: No saved train/OOF scores for frozen Stage 4 router/reranker overlay were found, and rerunning Qwen/V4/D2/Graph Retriever is prohibited.
- Test-only counterfactual candidates exist: True
