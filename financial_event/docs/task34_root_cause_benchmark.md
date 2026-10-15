# Task3/Task4 Root-Cause Benchmark

## 0. Frozen Artifact

- Benchmark artifact: `/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4 golddata/task34.xlsx`
- SHA256: `b63bc6c7e18d353003664480a1d08c1482517cbefea5813484864f3b184c4f29`
- Metrics JSON: `/opt/data/private/JinRong/kechuang/financial_event_project/reports/task34_root_cause_benchmark_metrics.json`
- Unified contract: `/opt/data/private/JinRong/kechuang/financial_event_project/configs/structured_event_contract_v1.json`

This benchmark is frozen as a read-only evaluation and root-cause artifact. It is not a training set.

Hard prohibitions:

- Do not modify `task34.xlsx`.
- Do not train or tune with any `human_*` audit label or `diagnostic_bucket_hint`.
- Do not add sample-level blacklists keyed by company, source, or `event_name`.
- Do not re-extract events inside Task4.
- Do not stop or modify the currently running Task1 process.

## 1. Benchmark Scope

`task34.xlsx` contains 1513 audited rows:

| Sheet | Rows | Purpose | Human label coverage |
|---|---:|---|---|
| `task3_event_resolution` | 500 | Task3 event identity audit | 500/500 merge/error labels |
| `task4_grounding` | 500 | Task4 target/event/evidence grounding audit | 500/500 status labels, 195 failure roots |
| `task4_e2e_fp_census` | 513 | Task4 false-positive census | 513/513 status labels, 423 failure roots |
| `sampling_meta` | 1513 | Hidden sampling and machine-decision metadata | no training labels |

## 2. Quantified Findings

### 2.1 Task3 Event Resolution

Task3 benchmark rows:

- 400 pair rows, 100 singleton rows.
- Machine decisions: 200 `same_event`, 200 `different_event`, 100 `singleton`.

Primary metrics:

| Metric | Definition | Count | Denominator | Rate |
|---|---|---:|---:|---:|
| False merge | `machine_decision=same_event` and human says `no_merge` | 106 | 200 | 53.0% |
| Merge precision | `machine_decision=same_event` and human says `merge` | 94 | 200 | 47.0% |
| Split error, strict | `machine_decision=different_event` and human says `merge` | 75 | 200 | 37.5% |
| Split risk, expanded | strict split error plus inconsistent split/duplicate error tags | 80 | 200 | 40.0% |
| Coverage gap | singleton row labeled `coverage_gap` | 100 | 100 | 100.0% |

High-risk Task3 buckets:

| Bucket | Main failure | Rate |
|---|---|---:|
| `same_event_cross_date` | false merge | 44/50 = 88.0% |
| `same_event_low_margin` | false merge | 27/50 = 54.0% |
| `split_uncertain` | strict split error | 29/40 = 72.5% |
| `split_same_family` | strict split error | 29/60 = 48.3% |
| `singleton_degree0` + `singleton_degree1_2` | coverage gap | 100/100 = 100.0% |

Conclusion: Task3 event identity is not reliable enough to be treated as a stable upstream truth for Task4. The dominant issue is not a missing post-hoc rule for 1513 rows; it is unstable event identity under target-local facts.

### 2.2 Task4 Grounding

Task4 grounding audit:

- 500 rows: 320 representative rows, 180 diagnostic rows.
- Unweighted status: 305 `no_issue`, 195 `failure`; failure rate 39.0%.
- Representative population-weighted estimate: failure weight 427.799 / 1020 = 41.94%.

Task4 grounding failure contributions:

| Layer/root cause | Unweighted count | Share of failures | Representative weighted share of failures | Representative weighted share of population |
|---|---:|---:|---:|---:|
| Target grounding | 91 | 46.7% | 45.6% | 19.1% |
| Event resolution | 75 | 38.5% | 36.5% | 15.3% |
| Evidence failure | 29 | 14.9% | 17.8% | 7.5% |
| Severity failure | 0 | 0.0% | 0.0% | 0.0% |

Conclusion: the grounding layer fails before Severity is reached. Target-local event/evidence validity is the gating problem.

### 2.3 Task4 E2E False-Positive Census

Task4 false-positive census:

- 513 rows.
- 423 rows are true failures; 90 rows are valid chains and should not be counted as model false positives.
- Failure rate: 423/513 = 82.46%.

Failure contribution by layer:

| Layer/root cause | Count | Share of 423 failures | Share of all 513 rows |
|---|---:|---:|---:|
| Event resolution | 214 | 50.6% | 41.7% |
| Target grounding | 161 | 38.1% | 31.4% |
| Evidence failure | 33 | 7.8% | 6.4% |
| Severity failure | 14 | 3.3% | 2.7% |
| Provenance failure | 1 | 0.2% | 0.2% |

Conclusion: Severity false alerts are real but small. The false-positive census is dominated by wrong input events and event identity errors, not by the final Severity scorer.

## 3. Root-Cause Separation

Required distinction:

- Symptom: Red/orange shortage and Severity false alerts. Red/orange shortage appears because few candidate events survive target/event/evidence gates; Severity false alerts are only 14/423 Task4 FP failures.
- Direct cause: Task4 input events are wrong or insufficiently grounded. Task4 is consuming target/event/evidence/provenance records that should have been rejected before direction or severity.
- Middle cause: Task3 event identity merge/split is unreliable. False merge is 53.0%, strict split error is 37.5%, and singleton coverage gap is 100.0% on the benchmark slices.
- Upstream root cause: Task1 does not provide a stable target-local structured event fact contract. Article-level labels and broad `event_name` values are insufficient for downstream identity and severity.
- Root-cause fix: adopt a unified Structured Event Contract, enforce layered quality gates, and abstain/reject when target/event/evidence/provenance is not grounded.
- Later training: only after upstream events pass the quality gates should new Severity Gold be added. Audit labels in this benchmark remain evaluation-only.

## 4. Responsibility Boundary

| Boundary | Owns | Evidence from benchmark | Required action |
|---|---|---|---|
| Task1 | Target-local structured event fact, evidence sentence, provenance | Task4 grounding target+evidence failures: 120/195; Task4 FP target+evidence+provenance failures: 195/423 | Emit only target-local event facts or reject/review |
| Task3 | Same-event identity, split/merge, coverage | False merge 106/200, strict split 75/200, coverage gap 100/100 | Resolve identity from structured facts; enforce cannot-link and review queues |
| Task4 validation layer | Reject invalid input event chains before direction/severity | Grounding failure rate 41.94% weighted on representative rows | Add hard gates and abstention, not event re-extraction |
| Task4 severity layer | Severity score only after valid target/event/evidence | Severity-only FP failures: 14/423 | Train Severity only on quality-gated upstream events |

## 5. Unified Structured Event Contract

Machine-readable contract: `configs/structured_event_contract_v1.json`.

Minimum record:

```json
{
  "schema_version": "structured_event_contract_v1",
  "article_id": "...",
  "source": "...",
  "publish_time": "...",
  "target": {
    "target_key": "...",
    "target_name": "...",
    "target_type": "company|industry|market|other",
    "target_code": "...",
    "mention_text": "...",
    "mention_span": [0, 0],
    "role": "affected|actor|issuer|counterparty|sector"
  },
  "event": {
    "event_type": "...",
    "event_name": "...",
    "event_fact": "...",
    "event_fact_signature": "...",
    "trigger_text": "...",
    "trigger_span": [0, 0],
    "event_time": "...",
    "polarity_or_state": "positive|negative|neutral|mixed|state_change"
  },
  "evidence": {
    "evidence_sentence": "...",
    "local_context": "...",
    "evidence_span": [0, 0],
    "support_type": "direct_fact",
    "certainty": "confirmed|quoted|uncertain"
  },
  "provenance": {
    "document_hash": "...",
    "sentence_id": "...",
    "extraction_method": "...",
    "extraction_time": "...",
    "upstream_model_or_rule_version": "..."
  },
  "quality": {
    "target_grounded": true,
    "event_grounded": true,
    "evidence_grounded": true,
    "provenance_valid": true,
    "contract_status": "pass|reject|review",
    "reject_reasons": []
  }
}
```

Identity rule:

- Same-event identity must use `target_key + event_fact_signature + event_type + event_time + polarity_or_state`.
- `event_name` alone is never an identity key.
- A record with missing target, event fact, evidence, or provenance cannot enter Task3 canonical identity or Task4 Severity.

## 6. Layer Contracts and Rejection

### 6.1 Task1

Input contract:

- Raw article text plus source, publish time, and stable `article_id`.

Output contract:

- One or more target-local `structured_event_record` objects.
- Emit no event when the article cannot support a target-local fact.

Reject/review conditions:

- Target is not locally grounded.
- Event trigger or event fact is missing.
- Evidence is only market price, boilerplate, rumor, or forecast.
- Event fact is article-global rather than target-local.
- Provenance cannot trace to document and sentence.

Acceptance gate:

- Target grounding >= 90%.
- Event grounding >= 65%.
- Evidence grounding >= 70%.
- Article-global noise <= 15%.

### 6.2 Task3

Input contract:

- Only Task1 records with `contract_status=pass` or explicit review status.

Output contract:

- Canonical event identity, members, same-event edges, cannot-link edges, confidence, and review/reject queues.

Reject/review conditions:

- Missing or unstable target key.
- Missing `event_fact_signature`.
- Evidence contradiction.
- Incompatible event time or mutually exclusive event state.
- Low-confidence identity edge.

Acceptance gate:

- False merge rate <= 10%.
- Split error rate <= 15%.
- Coverage gap rate <= 10%.
- Cannot-link violations = 0.

### 6.3 Task4

Input contract:

- Task3 canonical event plus Task1 evidence. Task4 must not re-extract events.

Output contract:

- Target/event/evidence validity, direction, severity score, red/orange/yellow/blue mapping, alert decision, and abstention reason.

Reject/review conditions:

- Input event contract status is not `pass`.
- Target grounding, event identity, evidence, or provenance fails.
- Severity evidence is insufficient.

Acceptance gate:

- Grounding failure rate <= 10%.
- E2E false-positive rate <= 20%.
- Severity-only share of FP failures <= 10%.
- Red/orange Severity Gold may be added only after upstream gates pass.

## 7. Enhanced Validation Repair Plan

The repair belongs in validation and contracts, not in sample-level blacklists:

1. Add a Task1 target-local event quality gate before Task3.
2. Use `event_fact_signature` and evidence spans as Task3 identity inputs.
3. Add Task3 cannot-link constraints for target conflict, fact conflict, incompatible time, and mutually exclusive state.
4. Add Task3 review queues for low-margin same-event and uncertain split buckets.
5. Add Task4 pre-severity validation: reject when target/event/evidence/provenance is not grounded.
6. Make Task4 abstain instead of repairing by re-extracting events.
7. Run Severity Gold expansion only after upstream pass; keep this benchmark evaluation-only.

## 8. Regression Test Design

Regression tests must verify:

- Frozen artifact hash matches `b63bc6c7e18d353003664480a1d08c1482517cbefea5813484864f3b184c4f29`.
- Sheet sizes remain 500/500/513/1513.
- Human label coverage remains complete for required audit columns.
- Task3 metrics remain exactly reproducible: false merge 106/200, strict split 75/200, coverage gap 100/100.
- Task4 grounding remains exactly reproducible: 195 failures, with roots 91 target, 75 event resolution, 29 evidence.
- Task4 FP census remains exactly reproducible: 423 failures, with roots 214 event resolution, 161 target, 33 evidence, 14 severity, 1 provenance.
- Contract prohibitions remain enabled: no benchmark mutation, no audit-label training, no sample blacklist, no Task4 event re-extraction.

Regression command:

```bash
python scripts/test_task34_root_cause_benchmark.py
```

## 9. Final Benchmark Decision

`task34.xlsx` is frozen as the Task3/Task4 Root-Cause Benchmark.

The benchmark should be used to evaluate root-cause repair, interface compliance, abstention behavior, and regression stability. It must not be used to fit rules for 1513 samples, train Severity, tune thresholds, create company/source/event-name blacklists, or move event extraction into Task4.
