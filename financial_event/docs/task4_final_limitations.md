# Task4 Final Limitations

## 1. Validity numbers must not be conflated
Two validity result triplets are both retained because they answer different questions:
- Conditional classifier quality on the validity test split.
- End-to-end chain quality over the frozen candidate pool.

The end-to-end result is the decision metric for deployment realism; the conditional result is not a substitute.

## 2. Stage 4.1 is not a predictive gain layer
Stage 4.1 is frozen as a read-only audit and review-ranking layer.
No legal test-time threshold sweep may be used for model selection.
No claim of predictive gain is accepted.

## 3. Severity / Alert Priority are prototype-only
Severity and Alert Priority are derived from `agent_adjudication` and exposed only through demo cards.
They are marked `label_source=agent_assisted_prototype` and `verification_status=not_human_gold`.
They are not supervised gold labels and are excluded from formal metric reporting.

## 4. Class imbalance remains severe
Direction evaluation is dominated by positive samples.
Negative and neutral recall remain materially lower than positive recall.
This limits the reliability of minority-direction interpretation.

## 5. Review burden is non-zero
The frozen chain still retains false positives and produces a non-zero review / abstention burden.
Any product claim must include the manual review requirement.

## 6. No further threshold optimization is allowed
No new threshold adjustment, no retraining, and no new stage are allowed under final acceptance.
All reported artifacts are acceptance-time audits over frozen outputs.
