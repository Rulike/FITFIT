# Task1 Hierarchical Model Pipeline

1. Build taxonomy from official 64 labels plus audited incremental labels.
2. Split official and incremental rows with article/dup/content/title leakage guards.
3. Train flat, cascade, multitask, masked and fusion classifiers over the leaf set.
4. Compare against the frozen Step 5.8-E official64 model.
5. Freeze the best safe model without touching formal Task1/Task2 outputs.
