{
  "title": "Task4 v3 Step1 Representation Build",
  "status": "task4_v3_step1_representation_build_complete",
  "outputs": [
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/embedding_manifest_oracle.parquet",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/e2e_exclusion_ledger.parquet",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/r1_event_conditioned_embedding_manifest.parquet",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/representation_build_report.json",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/task4_v3_step1_contract.json",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/embedding_manifest_event_conditioned.parquet",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/step1_audit_reconciliation.json",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/embeddings",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/step1_embedding_manifest.parquet",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/embedding_sanity_audit.json",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/embedding_reload_consistency.json",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/e2e_exclusion_ledger.json",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/r0_5_document_level_embedding_manifest.parquet",
    "/opt/data/private/JinRong/kechuang/financial_event_project/outputs/task4_v3_step1/input_manifest.json"
  ],
  "sanity": {
    "status": "task4_v3_step1_representation_build_complete",
    "oracle": {
      "count": 3127,
      "shape_ok": true,
      "nan_free": true,
      "sample_aligned": true,
      "train_test_isolated": true,
      "reload_consistent": true,
      "checksum_consistent": true,
      "model_config_hash": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"
    },
    "event_conditioned": {
      "count": 2876,
      "shape_ok": true,
      "nan_free": true,
      "sample_aligned": true,
      "train_test_isolated": true,
      "reload_consistent": true,
      "checksum_consistent": true,
      "model_config_hash": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"
    },
    "train_test_isolated_overall": true,
    "sample_alignment_closed": true,
    "failed_samples": [],
    "gpu_peak_mb": {
      "oracle": 14750.1318359375,
      "event_conditioned": 14846.42041015625
    },
    "elapsed_seconds": {
      "oracle": 178.994896,
      "event_conditioned": 164.884401
    },
    "e2e_exclusion_ledger": {
      "oracle_total": 3127,
      "e2e_direction_total": 2876,
      "diff_total": 251,
      "reason": {
        "no_matching_e2e_direction_sample": 251
      },
      "label_distribution_impact": {
        "oracle_total": 3127,
        "matched_total": 2876,
        "unmatched_total": 251,
        "oracle_label_distribution": {
          "正面": 2821,
          "中性": 198,
          "负面": 108
        },
        "matched_label_distribution": {
          "正面": 2596,
          "中性": 179,
          "负面": 101
        },
        "delta_by_label": {
          "正面": 225,
          "中性": 19,
          "负面": 7
        }
      }
    },
    "r0_5_input_reconciliation": {
      "oracle_train": 2081,
      "oracle_test": 1046,
      "oracle_total": 3127
    },
    "r1_input_reconciliation": {
      "e2e_train": 1898,
      "e2e_test": 978,
      "e2e_total": 2876
    }
  }
}