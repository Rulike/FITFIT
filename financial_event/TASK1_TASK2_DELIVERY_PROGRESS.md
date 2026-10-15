# 任务一与任务二进度交付说明

## 赛题任务口径

根据赛题文档《12-知识图谱与智能推荐赛道-华泰证券-面向上市公司的事件驱动智能识别、可信评估与脉络追踪.docx》：

- 任务一：多源金融事件识别，完成事件主体、时间、影响对象、影响方向等事件要素抽取与标准化。
- 任务二：事件可信评估与辨伪，结合信息源可信度、多源交叉验证、公告佐证与传播路径分析量化事件真实性。

## 任务一当前进度

任务一已经形成端到端链路：事件 Schema 标准化、带标签样本构建、重复新闻标签传播、负样本构造、MacBERT 事件匹配、无标签新闻候选生成、质量门控、联合裁决队列、最终事件聚合、多源事件聚类。

当前关键结果：

- 无标签事件识别输出：`outputs/final_task1/final_events_train.jsonl` 3003 条，`outputs/final_task1/final_events_test.jsonl` 1970 条，合计 4973 条。
- 事件聚类输入事件数：4973。
- 事件簇数量：2061。
- 单文章簇：1266。
- 多源簇：795。
- 聚类阈值：0.44。

口径备注：

- `reports/task1_final_summary.json` 保留了一次质量门控汇总口径，其中高置信自动接受事件为 4049 条，人工复核队列为 1864 条。
- 下游聚类和任务二实际采用的是 `outputs/final_task1/final_events_train.jsonl` 与 `outputs/final_task1/final_events_test.jsonl` 合计 4973 条的最终事件文件。

任务一核心代码：

- Schema 与标签标准化：`scripts/build_event_schema.py`、`scripts/build_labeled_dataset.py`
- 样本增强与负样本：`scripts/augment_duplicate_news.py`、`scripts/build_negative_pairs.py`、`scripts/split_train_validation.py`
- 事件匹配训练评估：`scripts/train_event_match_baseline.py`、`scripts/train_event_match_hard_negative.py`、`scripts/train_event_match_pairwise.py`
- 无标签识别与最终汇总：`scripts/build_target_aliases.py`、`scripts/build_unlabeled_event_candidates.py`、`scripts/build_event_quality_gate.py`、`scripts/run_event_joint_adjudication.py`、`scripts/aggregate_final_task1_results.py`
- 多源聚类：`scripts/build_event_cluster_candidates.py`、`scripts/compute_event_pair_features.py`、`scripts/cluster_multisource_events.py`、`scripts/complete_clustered_event_elements.py`、`scripts/aggregate_multisource_events.py`
- 可复用模块：`src/data/`、`src/evaluation/`、`src/routing/`、`src/event_clustering/`、`src/event_completion/`

## 任务二当前进度

任务二已经形成增强版可信评估链路：事件簇特征构建、弱监督可信评分、全文索引、证据句检索、正文支撑/反证识别、在线大模型复核、最终可信分融合和报告输出。

当前关键结果：

- 评分事件簇：2061。
- 全文索引文章：310088。
- 有全文证据事件簇：2061。
- 有正文支撑信号事件簇：1903。
- 有正文矛盾/反转信号事件簇：474。
- LLM 复核队列：260 条。
- LLM 复核失败数：0。
- 最终可信分均值：40.22，最低 0，最高 100。

最终可信等级分布：

- 疑似低可信：1315。
- 高可信：477。
- 疑似虚假或已反转：107。
- 较可信：105。
- 待核验：57。

最终验证状态分布：

- needs_manual_review：1102。
- insufficient_evidence：294。
- official_fulltext_supported：249。
- fulltext_supported：189。
- llm_supported：120。
- reversed_or_contradicted：107。

任务二核心代码：

- 基础可信评分：`src/event_credibility/scoring.py`、`scripts/run_task2_credibility_experiment.py`
- 全文证据增强：`src/event_credibility/fulltext.py`、`scripts/run_task2_fulltext_credibility_experiment.py`
- LLM 在线复核：`scripts/run_task2_llm_review.py`、`scripts/run_task2_llm_review_mdkj.sh`、`scripts/retry_task2_failed_llm_reviews.py`
- 结果报告：`reports/event_credibility/task2_delivery_status.md`、`reports/event_credibility/fulltext_credibility_summary.json`

## 交付包内容

本次关键代码包包含：

- `README.md`
- `TASK1_TASK2_CODE_SUMMARY.md`
- `TASK1_TASK2_DELIVERY_PROGRESS.md`
- `scripts/`
- `src/`
- `configs/`
- `reports/`
- `outputs/final_task1/`
- `outputs/event_clustering/`
- `outputs/event_credibility/`

压缩包排除了 Python 缓存、训练 checkpoint、模型权重、优化器状态和中间 parquet，以控制体积并突出可审阅代码与最终结果。

## 2026-08-09最终验收状态

Step 4.5-D正式提升已完成，promotion transaction_status=committed。正式LLM结果260条，失败0；正式评分JSONL/Parquet/XLSX各2061条；正式四文件SHA256与`reports/task2_final_cached_dea_v1_promotion_manifest.json`一致。

Task2最终正式架构为：基础可信评分和全文证据抽取、DianJin-R1-32B单轮严格盲审、困难样本条件触发Cached DEA、SQLite FTS5 trigram全局证据召回、证据排序去重与可追溯校验、既定公式融合可信分。正式启用Cached DEA、SQLite缓存检索、strict validation、checkpoint/resume、workers=4；正式停用ISR、Safety Gate、Claim Contract Adapter。

最终验收文档：

- `reports/TASK1_TASK2_FINAL_ACCEPTANCE.md`
- `reports/TASK2_FINAL_TECHNICAL_REPORT.md`
- `docs/TASK2_FINAL_PIPELINE.md`
- `docs/TASK2_REPRODUCTION_GUIDE.md`
- `docs/TASK2_ROLLBACK_GUIDE.md`

最终交付包目标路径：`/opt/data/private/JinRong/kechuang/task1_task2_final_delivery_20260809.tar.gz`。
