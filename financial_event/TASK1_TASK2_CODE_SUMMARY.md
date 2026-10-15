# 任务一与任务二完整代码总结

## 总体说明

本工程围绕华泰证券“面向上市公司的事件驱动智能识别、可信评估与脉络追踪”赛题实现了任务一和任务二的主要链路。

- 任务一：从原始事件体系和新闻 Excel 出发，完成事件 Schema 标准化、带标签样本构建、负样本构造、事件匹配模型训练、无标签新闻事件识别、LLM 联合裁决、多源事件聚类，最终输出结构化事件和事件簇。
- 任务二：以任务一输出的事件簇为评分对象，完成事件可信度特征构建、弱监督评分、全文证据检索、正文支撑/反证识别、在线大模型复核和最终可信分融合。

## 目录结构

```text
financial_event_project/
├── scripts/                    # 任务一、任务二的可执行脚本
├── src/                        # 可复用模块
│   ├── data/                   # 事件匹配输入文本构造
│   ├── evaluation/             # 事件匹配评估器
│   ├── routing/                # 无标签新闻主体路由
│   ├── event_clustering/       # 多源事件聚类
│   ├── event_completion/       # 事件簇字段合并、冲突检测、证据对齐
│   └── event_credibility/      # 任务二可信度评分与全文证据
├── configs/                    # 模型、评估、别名和阈值配置
├── reports/                    # 统计、评估和交付报告
├── outputs/final_task1/        # 任务一最终事件识别结果
├── outputs/event_clustering/   # 任务一事件簇结果
└── outputs/event_credibility/  # 任务二可信度结果
```

## 任务一代码链路

### 1. 事件 Schema 标准化

核心脚本：

- `scripts/build_event_schema.py`
- `scripts/test_event_schema.py`

功能：

- 读取 `事件类型_标的.json` 与 `事件类型_行业.json`。
- 统一生成 `schema_id、target_type、target_code、target_name、event_name、raw_description`。
- 校验字段完整性、唯一性、代码格式和描述完整性。

主要产物：

- `data/processed/event_schema.jsonl`
- `data/processed/event_schema.xlsx`
- `reports/event_schema_summary.json`

### 2. 带标签新闻标准化

核心脚本：

- `scripts/build_labeled_dataset.py`
- `scripts/test_labeled_dataset.py`

功能：

- 读取带标签新闻 Excel 的个股新闻、行业新闻、重复新闻工作表。
- 建立新闻级数据 `news_*.jsonl` 和事件对数据 `event_pairs_*.jsonl`。
- 通过目标代码、目标名称和事件名称匹配统一 Schema。
- 记录无法匹配的 Schema、发布时间异常和重复新闻关联异常。

主要产物：

- `data/processed/labeled/news_train.jsonl`
- `data/processed/labeled/news_test.jsonl`
- `data/processed/labeled/event_pairs_train.jsonl`
- `data/processed/labeled/event_pairs_test.jsonl`
- `reports/labeled_dataset_summary.json`

### 3. 重复新闻标签传播与正样本增强

核心脚本：

- `scripts/augment_duplicate_news.py`
- `scripts/test_duplicate_augmentation.py`

功能：

- 使用 `duplication_id -> canonical_article_id` 关系，把主新闻标签传播到重复新闻。
- 保证 train/test 不跨 split 传播。
- 输出原始样本和增强样本的合并版本。

主要产物：

- `data/processed/augmented/news_train_augmented.jsonl`
- `data/processed/augmented/news_test_augmented.jsonl`
- `data/processed/augmented/event_pairs_train_augmented.jsonl`
- `data/processed/augmented/event_pairs_test_augmented.jsonl`
- `reports/duplicate_augmentation_summary.json`

### 4. 负样本构造与训练集切分

核心脚本：

- `scripts/build_negative_pairs.py`
- `scripts/test_negative_pairs.py`
- `scripts/split_train_validation.py`
- `scripts/test_train_validation_split.py`

功能：

- 构造三类负样本：同主体不同事件、同事件错误主体、相似 Schema 错误主体。
- 删除疑似假负样本，避免与正样本冲突。
- 以 `canonical_group_id` 为单位划分内部训练集和验证集，防止重复新闻泄漏。

主要产物：

- `data/processed/training_pairs/train_pairs_balanced.jsonl`
- `data/processed/training_pairs/test_pairs_balanced.jsonl`
- `data/processed/model_data/event_match_train.jsonl`
- `data/processed/model_data/event_match_validation.jsonl`
- `data/processed/model_data/event_match_official_test.jsonl`

### 5. 事件匹配模型与同主体困难负样本

核心脚本：

- `scripts/train_event_match_baseline.py`
- `scripts/evaluate_event_match_baseline.py`
- `scripts/mine_same_target_hard_negatives.py`
- `scripts/train_event_match_hard_negative.py`
- `scripts/evaluate_event_match_hard_negative.py`
- `scripts/train_event_match_pairwise.py`
- `scripts/evaluate_event_match_pairwise.py`
- `src/data/event_match_text.py`
- `src/evaluation/event_match_evaluator.py`

功能：

- 使用结构化文本拼接新闻标题、正文、候选事件和事件定义。
- 训练 MacBERT 事件匹配模型。
- 挖掘同主体困难负样本，降低同一公司/行业下相似事件误判。
- 提供 baseline、hard negative、pairwise/ranknet 等实验配置。

主要配置：

- `configs/event_match_macbert_base.json`
- `configs/event_match_macbert_hard_negative.json`
- `configs/event_match_pairwise_ranknet.json`
- `configs/event_match_pairwise_control.json`
- `configs/canonical_evaluation_protocol.json`

### 6. 无标签新闻事件识别与联合裁决

核心脚本：

- `scripts/build_target_aliases.py`
- `scripts/build_unlabeled_event_candidates.py`
- `scripts/test_unlabeled_event_inference.py`
- `scripts/build_event_quality_gate.py`
- `scripts/run_event_joint_adjudication.py`
- `scripts/aggregate_final_task1_results.py`
- `scripts/test_final_task1_pipeline.py`
- `src/routing/target_router.py`

功能：

- 构建目标实体别名，用于无标签新闻路由。
- 为无标签新闻生成候选事件。
- 使用事件匹配模型筛选正例。
- 对中低置信、冲突或多候选样本进行 LLM 联合裁决。
- 汇总自动接受和裁决接受的事件，形成任务一最终输出。

任务一最终结果：

- 总无标签文章：307888
- 成功路由文章：9453
- MacBERT 正预测：12841
- 自动接受事件：4049
- LLM 联合裁决接受事件：924
- 最终接受事件：4973
- 公司事件：4801
- 行业事件：172

主要产物：

- `outputs/final_task1/final_events_train.jsonl`
- `outputs/final_task1/final_events_test.jsonl`
- `outputs/final_task1/final_events_train.xlsx`
- `outputs/final_task1/final_events_test.xlsx`
- `outputs/final_task1/task1_completion_certificate.json`
- `reports/task1_final_summary.json`

### 7. 多源事件聚类

核心脚本：

- `scripts/build_article_metadata_index.py`
- `scripts/build_event_cluster_weak_labels.py`
- `scripts/build_event_cluster_candidates.py`
- `scripts/compute_event_pair_features.py`
- `scripts/select_event_pair_threshold.py`
- `scripts/cluster_multisource_events.py`
- `scripts/complete_clustered_event_elements.py`
- `scripts/validate_clustered_events.py`
- `scripts/aggregate_multisource_events.py`
- `src/event_clustering/blocking.py`
- `src/event_clustering/pair_features.py`
- `src/event_clustering/graph_clustering.py`
- `src/event_clustering/source_normalizer.py`
- `src/event_completion/field_merger.py`
- `src/event_completion/conflict_detector.py`
- `src/event_completion/evidence_alignment.py`

功能：

- 为任务一事件建立文章元数据索引。
- 构造事件对候选和事件对特征。
- 通过阈值和图连通分量把同一事件的多源报道聚合成事件簇。
- 合并事件要素、统计来源、发现过度合并和欠合并风险。

聚类结果：

- 输入事件数：4973
- 事件簇数量：2061
- 单文章簇：1266
- 多源簇：795
- 聚类阈值：0.44

主要产物：

- `outputs/event_clustering/event_clusters.jsonl`
- `outputs/event_clustering/event_cluster_members.jsonl`
- `outputs/event_clustering/canonical_multisource_events.jsonl`
- `outputs/event_clustering/canonical_multisource_events.xlsx`
- `reports/event_clustering/event_cluster_summary.json`

## 任务二代码链路

### 1. 事件簇可信度基础评分

核心代码：

- `src/event_credibility/scoring.py`
- `scripts/run_task2_credibility_experiment.py`
- `scripts/run_task2_credibility_experiment.sh`
- `scripts/test_event_credibility_pipeline.py`

功能：

- 以任务一事件簇为评分对象。
- 构建来源可信度、官方来源、多源一致性、来源熵、HEARSAY 比例、谣言词、反转词、冲突风险等特征。
- 通过规则生成弱监督标签。
- 优先使用 LightGBM 建模，缺失时回退到 XGBoost、sklearn 梯度提升或 numpy 逻辑回归。
- 融合模型分和规则分，生成基础可信度分数、可信等级和人工复核样本。

主要产物：

- `data/processed/event_credibility/cluster_credibility_features.parquet`
- `data/processed/event_credibility/cluster_credibility_weak_labels.parquet`
- `outputs/event_credibility/event_credibility_scores.jsonl`
- `outputs/event_credibility/event_credibility_scores.xlsx`
- `reports/event_credibility/credibility_metrics.json`
- `reports/event_credibility/feature_importance.csv`
- `reports/event_credibility/task2_experiment_summary.md`

### 2. 全文索引与正文证据增强

核心代码：

- `src/event_credibility/fulltext.py`
- `scripts/run_task2_fulltext_credibility_experiment.py`
- `scripts/run_task2_fulltext_credibility_experiment.sh`
- `scripts/test_event_credibility_fulltext.py`

功能：

- 从四个原始新闻 Excel 建立全文索引。
- 对 2061 个事件簇检索正文证据句。
- 过滤公告免责声明和非事件反证语境。
- 将证据分为 support、contradict、insufficient。
- 在基础可信分上叠加全文支撑和全文反证调整。

全文增强公式：

```text
fulltext_credibility_score =
    base_credibility_score
    + 18 * fulltext_support_score
    - 42 * fulltext_contradiction_score
    + llm_adjust
```

主要结果：

- 全文索引文章：310088
- 评分事件簇：2061
- 有全文证据的事件簇：2061
- 有正文支撑信号：1903
- 有正文矛盾/反转信号：474

主要产物：

- `data/processed/event_credibility/fulltext_article_index.parquet`
- `outputs/event_credibility/fulltext_evidence.jsonl`
- `outputs/event_credibility/fulltext_evidence.parquet`
- `outputs/event_credibility/event_credibility_scores_fulltext.jsonl`
- `outputs/event_credibility/event_credibility_scores_fulltext.xlsx`
- `reports/event_credibility/fulltext_credibility_summary.json`
- `reports/event_credibility/task2_fulltext_experiment_summary.md`

### 3. LLM 在线复核与融合

核心代码：

- `src/event_credibility/fulltext.py`
- `scripts/run_task2_llm_review.py`
- `scripts/run_task2_llm_review_mdkj.sh`
- `scripts/retry_task2_failed_llm_reviews.py`

功能：

- 从事件簇中筛选风险导向的 260 条样本进入复核队列。
- 为每条样本提供主体、事件、当前可信分、风险原因、支撑原因和证据句。
- 调用兼容 Responses API 或 Chat Completions API 的在线模型。
- 要求模型返回 `support`、`contradict` 或 `insufficient`。
- 将 LLM 结论融合回最终可信分。

LLM 调整项：

```text
support      -> +12 * confidence
contradict   -> -35 * confidence
insufficient ->  -5 * confidence
```

复核结果：

- 模型：gpt-5.5
- 复核数量：260
- 失败数量：0
- 状态：completed

主要产物：

- `data/processed/event_credibility/llm_review_queue.jsonl`
- `outputs/event_credibility/llm_review_results.jsonl`
- `reports/event_credibility/task2_delivery_status.md`

### 4. 任务二最终结果

最终可信等级分布：

| 等级 | 数量 |
|---|---:|
| 疑似低可信 | 1315 |
| 高可信 | 477 |
| 疑似虚假或已反转 | 107 |
| 较可信 | 105 |
| 待核验 | 57 |

验证状态分布：

| 状态 | 数量 |
|---|---:|
| needs_manual_review | 1102 |
| insufficient_evidence | 294 |
| official_fulltext_supported | 249 |
| fulltext_supported | 189 |
| llm_supported | 120 |
| reversed_or_contradicted | 107 |

分数统计：

- 平均分：40.22
- 最低分：0
- 最高分：100

## 建议运行顺序

```bash
cd financial_event_project

# 任务一：Schema、标签、负样本、模型数据
python scripts/build_event_schema.py
python scripts/test_event_schema.py
python scripts/build_labeled_dataset.py
python scripts/test_labeled_dataset.py
python scripts/augment_duplicate_news.py
python scripts/test_duplicate_augmentation.py
python scripts/build_negative_pairs.py
python scripts/test_negative_pairs.py
python scripts/split_train_validation.py
python scripts/test_train_validation_split.py

# 任务一：模型训练、无标签识别和最终聚合
python scripts/train_event_match_baseline.py --config configs/event_match_macbert_base.json
python scripts/evaluate_event_match_baseline.py
python scripts/build_target_aliases.py
python scripts/build_unlabeled_event_candidates.py
python scripts/run_event_joint_adjudication.py
python scripts/aggregate_final_task1_results.py
python scripts/test_final_task1_pipeline.py

# 任务一：事件聚类
bash scripts/run_full_event_clustering_on_server.sh

# 任务二：基础可信度、全文增强、LLM复核
bash scripts/run_task2_credibility_experiment.sh
bash scripts/run_task2_fulltext_credibility_experiment.sh
bash scripts/run_task2_llm_review_mdkj.sh
```

## 打包说明

本次打包建议包含：

- `README.md`
- `TASK1_TASK2_CODE_SUMMARY.md`
- `scripts/`
- `src/`
- `configs/`
- `reports/`
- `outputs/final_task1/`
- `outputs/event_clustering/`
- `outputs/event_credibility/`
- `data/processed/event_schema.jsonl`
- `data/processed/event_schema.xlsx`

为控制体积，打包时建议排除：

- Python 缓存：`__pycache__/`、`*.pyc`
- 训练 checkpoint：`checkpoint-*`
- 大模型权重和优化器状态：`*.safetensors`、`*.pt`、`*.pth`
- 大体积原始 Excel 和中间 parquet，除非提交方要求复现实验全量数据

