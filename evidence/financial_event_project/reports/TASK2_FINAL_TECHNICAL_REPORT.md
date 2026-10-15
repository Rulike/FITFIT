# Task2最终技术报告

## 目标

Task2目标是对Task1产生的事件簇进行可信评估与辨伪，输出最终可信分、可信等级、证据链和解释。最终正式流程采用DianJin-R1-32B单轮复核、条件触发Cached DEA、SQLite FTS5检索和既定可信分融合公式；ISR、Safety Gate、Claim Contract Adapter不进入正式流程。

## 正式链路

1. 基础可信评分：由`src/event_credibility/scoring.py`生成规则、来源、传播和质量特征，形成基础可信分。
2. 全文证据抽取：由`src/event_credibility/fulltext.py`抽取支撑、反证和官方来源信号。
3. 单轮严格盲审：DianJin-R1-32B只看主张与证据，不看当前可信分、等级或规则结论。
4. 条件触发Cached DEA：Round1为insufficient等困难样本进入DEA；本轮260条中触发155条。
5. SQLite FTS5 trigram检索：通过`src/event_credibility/retrieval_index.py`和SQLite缓存索引做全局召回，DEA继续做排序、去重和可追溯校验。
6. 可信分融合：保留既定公式和阈值，不调整生产评分逻辑。

融合公式：

- support：`+12 * confidence`
- contradict：`-35 * confidence`
- insufficient：`-5 * confidence`
- failed：隔离，不参与融合

正式参数：Cached DEA启用，SQLite缓存检索启用，strict validation启用，checkpoint/resume启用，workers=4。

## 正式停用项

| 模块 | 状态 | 结论 |
|---|---|---|
| ISR | 停用 | 6条候选纠正0条，增加24次请求和29429 tokens。 |
| Safety Gate | 停用 | 独立50条零触发，输出与Cached DEA一致。 |
| Claim Contract Adapter | 停用 | 20条诊断中Adapter+DEA Accuracy 0.40，低于Cached 0.50，耗时倍率1.438988。 |

停用模块代码、配置和摘要报告保留为消融记录，不进入正式执行路径，不继续开发新版本。

## 评估结果

### 独立50条

来源：`reports/task2_safety_gate_v1_2_independent_gold_evaluation.json`。

| 方法 | Accuracy | fixed macro-F1 |
|---|---:|---:|
| Baseline | 68.00% | 0.6005 |
| Round1 | 72.00% | 0.3804 |
| Cached DEA | 74.00% | 0.6327 |

标签分布为support=25、insufficient=20、unclear=4、contradict=1。标签为AI辅助、用户确认。该集合样本量仅50条，contradict仅1条，不能用于统计显著性宣称；结论限定为Cached DEA在当前受控评估中表现最佳。

### 260条正式运行

来源：`reports/task2_final_cached_dea_v1_260_report.json`。

- completed_count=260。
- dea_trigger_count=155，dea_trigger_ratio=0.596154。
- round1_final_verdict_change_count=53。
- final_verdict_distribution：insufficient=102、support=155、contradict=3。
- new_evidence_total_count=620；new_independent_source_count_sum=359。
- total_avg=7.082188秒；total_p95=12.64665秒。
- http_attempt_count=415；total_tokens_total=445460。
- timeout_count、http_5xx_count、parse_failure_count、cuda_oom_count均为0。
- checkpoint_resume.resume_used=false。

### 正式评分

来源：`reports/task2_final_cached_dea_score_shadow_report.json`。

- 评分总量：2061条。
- 分数变化：247条，其中升高189条、降低58条。
- 平均绝对差0.47017，P50=0，P95=1.65，最大差26.07。
- 可信等级变化0。
- 方向异常0，工程告警0。
- 未覆盖1801条完全一致，非预期变化0。

## 检索性能记录

来源：`reports/task2_dea_retrieval_cache_benchmark.json`。

正式检索采用SQLite FTS5 trigram、B-tree metadata和进程内LRU缓存。35条样本基准中，旧全表扫描平均检索耗时7.079778秒；新索引workers=1首轮平均0.276991秒、热查询平均0.270128秒。正式260条运行中检索平均0.239342秒，P95为0.757409秒。

## 工程保护

来源：`reports/task2_final_cached_dea_v1_promotion_manifest.json`、`reports/task2_final_cached_dea_v1_promotion_audit.json`。

- promotion transaction_status=committed。
- rollback round-trip已通过；重新apply返回committed/already_committed口径。
- 正式输出SHA256与manifest一致。
- LLM结果260条，失败0，重复cluster_id 0。
- 正式评分三格式2061条，核心字段一致。
- 32B新增调用0。

## 限制

- 独立集只有50条，且contradict只有1条。
- 不宣称统计显著提升。
- 旧报告中的部分均值口径不同；最终评分变化以`task2_final_cached_dea_score_shadow_report.json`的全2061条口径为准。
- JSONL可选缺失字段保留`NaN`占位；必填评分字段和可信等级字段通过验收。

