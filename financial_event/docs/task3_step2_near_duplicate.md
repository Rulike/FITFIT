# Task3 Step2 Near Duplicate Detection

This step uses title and structured fields only. It keeps all source documents and maps duplicate groups to independent evidence units.

- evidence_scope: `title_structured_only`
- pilot labels: 50 near-duplicate rows, yes=35, no=15
- thresholds: T_low=0.45, T_high=0.8
- MinHash candidates: 10842, combined candidates: 10842
- prediction counts: {'duplicate': 1767, 'non_duplicate': 7821, 'uncertain_review': 1254}
- duplicate groups: 774, max group size: 8

Same Event labels are exported unchanged for step 3 and are not used in step 2.
