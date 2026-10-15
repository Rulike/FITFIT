# Task1 Final Optimized Pipeline

1. Reuse frozen HF retrospective predictions.
2. Normalize empty duplication IDs to deterministic singleton IDs.
3. Score target, event, emotion, impact, and duplication tags on the labeled user test set.
4. Freeze the candidate outputs and write SHA manifests.
5. Do not retrain, retune, or modify formal Task1/Task2 outputs.
