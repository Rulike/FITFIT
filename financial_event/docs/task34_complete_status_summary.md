# Task3 + Task4 完整状态总结文档

> 生成时间：2026-06-16  
> 范围：覆盖 Task1（事件抽取+结构化）、Task2（事件可信度）、Task3（事件图谱与演化）、Task4（事件影响与 Severity）全链路  
> 特别说明：Task1 和 Task2 已冻结交付，本文档重点总结 Task3（KG v3 核心）与 Task4（Severity 补全）的当前进度、方法、问题、根因及后续计划。

---

## 一、项目背景与定位

本项目是一个**金融事件监控与风险预警系统**，从多源新闻中抽取事件，构建事件知识图谱，评估事件可信度，最终判断事件对目标（公司/行业）的**影响严重程度（Severity）**，输出红/橙/黄/蓝四级预警。

| 任务 | 定位 | 状态 |
|------|------|------|
| **Task1** | 事件抽取 + 结构化（PPUIE 方案） | **已冻结交付** |
| **Task2** | 事件可信度评分（DEA + Safety Gate） | **已冻结交付** |
| **Task3** | 事件图谱构建、事件归一化、演化追踪、信任传播 | **KG v3 核心已冻结** |
| **Task4** | 事件影响评估、Severity 分级（红/橙/黄/蓝）、预警输出 | **核心链路已冻结，Severity 补全进行中** |

---

## 二、Task3 当前状态

### 2.1 已实现模块

| 模块 | 方法 | 状态 |
|------|------|------|
| **事件归一化（Event Resolution）** | 140,760 候选对 → 11,912 高置信度 same-event 边 + 16,297 cannot-link 边。Reranker 调用仅 0.43% 候选对。总体压缩率 97.87%。 | **已冻结** |
| **事件归一化评价** | Proxy Macro-F1 = 0.481（Ablation: embedding-only 0.100, +identity 0.339, +graph 约束 0.339, +独立证据单元 0.384, +Task2 信任因子 0.473, 全量 0.481） | **已冻结** |
| **Canonical Event 总量** | V2: 3,637 个（单源率 83.6%），V3: 1,277 个（单源率 65.6%），多源事件从 595 → 439 | **已冻结** |
| **事件演化（Event Evolution）** | 528 条高置信度状态转移边，302 条同状态重复，最多线程长度 27（V2 仅 6），线程总数 871 | **已冻结** |
| **信任传播（Trust Propagation）** | 4,973 条 claim 级评分，1,277 个 canonical event 评分，全部标记为 derived_graph_score | **已冻结** |
| **约束合规** | cannot-link 违规 0，时间逆序违规 0，证据可追溯率 100% | **已冻结** |
| **Task3 → Task4 接口** | `task4_event_interface.jsonl` 输出 canonical event → 当前状态 → 信任分数，供 Task4 查询 | **已冻结** |

### 2.2 未完成/待解决

| 问题 | 说明 |
|------|------|
| **Human Gold 验证缺失** | 两次 human audit（v2 200 样本、v3 500 样本）均已创建 audit 包，但标注数据尚未收集（labeled_rows=0）。无法计算 merge_precision / false_merge_rate / split_error_rate。 |
| **Proxy 指标局限** | KG v3 的 Proxy Macro-F1 0.481 是 agent-assisted 指标，非正式 human gold 评测。 |
| **单源事件占比仍高** | 65.6% 的 canonical event 仍是单源，说明跨源事件归一化覆盖仍有提升空间。 |
| **不确定队列** | 115,648 条 uncertain review 未处理，无法自动纳入 KG。 |
| **演化线程验证** | 528 条高置信度转移边依赖规则约束，缺乏 human gold 验证 transition 质量。 |

---

## 三、Task4 当前状态

### 3.1 核心链路（已冻结交付）

```
Task1 事件抽取 → Task3 图谱 → Task4 候选生成 → 有效性分类 → 方向分类 → 预警输出
```

| 模块 | 方法 | 核心指标 | 状态 |
|------|------|----------|------|
| **候选生成** | Task1 事件 + Task3 canonical event 组装 | 候选池 10,690，gold 覆盖率 100% | **已冻结** |
| **有效性分类（V4）** | 分层模型，threshold=0.30 | End-to-End: P=0.530, R=0.802, F1=0.662 | **已冻结** |
| **方向分类（D2-Hierarchical）** | 分层方向模型 | P=0.401, R=0.873, F1=0.549（有效性条件测试） | **已冻结** |
| **预警策略** | 红→立即预警，橙→批量预警，黄→汇总，蓝→不预警，Review→人工复核 | 已定义 | **已冻结** |
| **Severity 形式化** | 严重度分为 Red/Orange/Yellow/Blue/Review 五级，明确数值规则 | 已定义但未全量执行 | **已冻结** |
| **Impact Scoring 模型** | 尝试 Multilabel / Ordinal / Hierarchical 方案 | 最佳 Macro-F1=0.457，Red Recall=0.053 | **已冻结** |

### 3.2 Severity 补全（当前进行中）

#### 3.2.1 历史尝试与关键发现

| 轮次 | 方法 | 结果 | 结论 |
|------|------|------|------|
| **Raw/Shadow Expansion v1** | 从 Raw 补充语料中召回 1,670 条，QC 保留 40 条 | Target grounding 63.65%, Event grounding 13.65%, Article-global noise 14.67% | 旧 article-level event label 与 target-local evidence 严重不匹配 |
| **Alignment Repair v1** | 修复失败样本 1,110 条，通过 target mention + ±1/2句窗口重判 | Target grounding 87.48%, Event grounding 10.99%, Noise 61.80% | 旧 event label 无法修复——标签本身错位 |
| **Target-Local Event Re-Extraction v1** | 放弃旧标签，从 target-local evidence 重新抽取 event fact → event type | TG 82.34%, EG 4.95%, Noise 95.05% | 旧 1,110 失败样本中 95% 的文章本身不适合做 target-local 事件抽取 |
| **Targetable Raw Corpus Selection v1** | 从 3,124 篇完整 Raw 中筛出 1,707 篇 targetable 文章 | TG 100%, EG 43.72%, 新候选 732 个 | 数据母池错误是根因，换母池后 EG 从 5% 提升到 43.7% |
| **Severity Gold 1000** | 从 1,000 条 Severity gold 标注包生成 | Positive 588, Neutral 285, Negative 127 | 严重偏斜，Red Recall 低 |
| **Severity HighRisk Mining v1** | 520 候选 → 300 最终候选（Red 106, Orange 78, Mixed 116） | 展期/资产减值/违约/停牌/立案 覆盖率最高 | 但后续 audit 发现 grounding 问题 |
| **Severity HighRisk Mining v2** | 520 候选 → 仅 2 条通过 grounding 检查 | 518 条因 no_local_evidence / no_target_grounding 被排除 | 旧候选池 grounding 严重不足 |
| **HighRisk v2 Reject Audit** | 200 条拒绝样本审计 | 50 条无局部证据，50 条无 target 定位，50 条无事件定位，仅 2 条通过 | 确认：不是缺候选，是候选质量不够 |

#### 3.2.2 关键根因链

```
问题：Event grounding（事件定位率）极低，Red/Orange 候选数量严重不足

根因分析（向下递进）：

1. 表层：Red/Orange 候选通过率低
   → Severity HighRisk Mining v2: 520 候选仅 2 条通过 grounding

2. 中层：Old article-level event label 与 target-local evidence 严重错配
   → Legacy event mismatch rate = 87.27%
   → Article-global noise = 95.05%（在旧失败池中）

3. 深层：Raw/shadow 补充语料中大部分文章不适合做 target-local 事件抽取
   → 3,124 篇完整 Raw 中仅 1,707 篇（54.6%）天然 targetable
   → 1,110 篇失败样本中 ~95% 是宏观快讯、多公司拼盘、行业晨报——无法形成 target-local 样本
   → 旧 Event Label 体系（140+ 细粒度事件类型）与高风险 Severity 目标不匹配

4. 根因：数据母池选择错误
   → 修复错误样本不如更换正确数据母池
   → Targetable Corpus Selection 已证明：换母池后 EG 从 5% → 43.7%
   → 但仍未达到验收线（EG ≥ 65%）
```

#### 3.2.3 当前最新指标

| 指标 | 起始值 | 最新值 | 验收线 | 备注 |
|------|--------|--------|--------|------|
| **Target Grounding** | 63.65% | **100%**（新母池） | ≥ 90% | ✅ 达标 |
| **Evidence Grounding** | 14.67% | **45.70%**（新母池） | ≥ 70% | ❌ |
| **Event Grounding** | 13.65% | **43.72%**（新母池） | ≥ 65% | ❌ |
| **Article-global Noise** | 14.67% | 待计算（新母池） | ≤ 15% | ❌ |
| **Legacy/New Event 一致率** | — | 12.73% | — | 旧标签与新标签严重不一致 |
| **Legacy Event 错配率** | — | 87.27% | — | 确认旧标签不可用 |
| **新 Red/Orange 候选** | 0 | **732 个** | 300-500 | ✅ 数量达标但质量未验证 |

### 3.3 当前数据资产清单

| 数据集 | 规模 | 是否 gold | 可用性 |
|--------|------|-----------|--------|
| **Task1 Structure Gold** | ~3,300 条 | 正式 gold | ✅ 已冻结 |
| **Task2 DEA Gold** | ~1,000 条 | 正式 gold | ✅ 已冻结 |
| **Task3 Canonical Events** | 1,277 个 | Agent 标注 | ✅ 已冻结 |
| **Task3 Trust Scores** | 1,277 个 canonical + 4,973 claim | Agent 标注 | ✅ 已冻结 |
| **Task4 Severity Gold 1000** | 1,000 条 | 正式 gold | ✅ 已生成 |
| **Task4 候选池（冻结测试集）** | 10,690 条 | 含 gold 978 条 | ✅ 已冻结 |
| **Task4 有效/方向模型** | V4 + D2-Hierarchical | 已训练 | ✅ 已冻结 |
| **Task4 Targetable Raw 母池** | 1,707 篇 | 非 gold | ✅ 已完成 |
| **Task4 新 Red/Orange 候选** | 732 个 | 候选 | ⚠️ 需 grounding 验证 |
| **Task4 旧失败池** | 1,110 条 | 非 gold | ❌ 已废弃 |

---

## 四、当前面临的核心问题

### 4.1 问题一：Targetable 母池内 Event Grounding 仍不足

- **现象**：新母池 1,707 篇 targetable 文章中，event grounding 仅 43.72%
- **原因**：target-local 窗口内虽能定位 target，但许多文章的目标信息是财务/经营公告，仍缺乏具体的事件触发词（违约/展期/亏损/减值/立案等）
- **影响**：可进入 Severity 的候选数量不足，且质量无法保证

### 4.2 问题二：旧 Event Label 体系与 Severity 目标不匹配

- **现象**：legacy event label 错配率 87.27%
- **原因**：旧体系有 140+ 细粒度事件类型（如"技术研发进展""新能源汽车出海"），但真正对 Severity 有预测力的事件类型仅 10 个左右（偿债/违约/经营恶化/合规等）
- **影响**：训练/标注时使用的 event 标签与 Severity 预测目标之间存在语义鸿沟

### 4.3 问题三：Gold 标注偏斜

- **现象**：Severity Gold 1000 中 Positive 58.8%，Neutral 28.5%，Negative 12.7%
- **原因**：候选池本身偏斜
- **影响**：Red Recall 极低（5.26%），Impact Scoring 模型难以学到 Red 模式

### 4.4 问题四：Human Audit 标注未执行

- **现象**：Task3 v2/v3 和 Task4 的 human audit 包已生成但未标注
- **原因**：标注流程未启动
- **影响**：无法获得正式 human gold 来验证代理指标

### 4.5 问题五：Severity 模型数据瓶颈

- **现象**：Impact Scoring 最佳 Macro-F1=0.457，官方结论是"data bottleneck"
- **原因**：真正 Red/Orange 样本太少，模型无法学到有效判别边界
- **影响**：仅靠现有 gold 无法支撑训练，需要更多高质量 Red/Orange 样本

---

## 五、当前采取的方法

### 5.1 数据层面

| 方法 | 说明 |
|------|------|
| **Targetable Corpus Selection** | 从 3,124 篇完整 Raw 中筛选天然适合 target-local 事件抽取的文章，排除宏观快讯、多公司拼盘、行业晨报 |
| **Target-Local Event Re-Extraction** | 放弃 legacy event label，从 target 所在句 ±1/2 句窗口重新抽取 event fact → event type |
| **新 Event Ontology** | 仅保留 10 个高风险事件族（偿债/违约/重组/经营恶化/减值/合规/持续经营/融资/系统性冲击/其他），放弃 140+ 细粒度 |
| **去重黑名单** | 收集所有现有 Severity gold/silver/candidate/audit/24 条裁决，按 article+target+event+fact signature 去重 |

### 5.2 模型层面

| 模块 | 方法 |
|------|------|
| **有效性分类 V4** | 分层模型，threshold=0.30，F1=0.662 |
| **方向分类 D2-Hierarchical** | 分层方向模型，F1=0.549 |
| **Impact Scoring** | 尝试 Multilabel / Ordinal / Hierarchical 方案，最佳 Macro-F1=0.457 |
| **Severity 形式化** | 五级制（Red/Orange/Yellow/Blue/Review），明确 abstention gate |

### 5.3 质量评估层面

| 方法 | 说明 |
|------|------|
| **Proxy 指标** | Agent-assisted 评估（非正式 human gold），用于迭代优化 |
| **Human Audit 包** | v2 200 样本 + v3 500 样本（Task3）+ v3 500 样本（Task4 grounding）+ v3 513 样本（Task4 FP 普查） |
| **Ablation 分析** | 从 embedding-only 到全量链路的逐步消融，验证每个模块增益 |

---

## 六、还需做什么

### 6.1 短期（立即优先）

| 优先级 | 事项 | 目标 | 预计耗时 |
|--------|------|------|----------|
| **P0** | 在 Targetable 母池（1,707 篇）内提升 Event Grounding 到 ≥ 65% | 让可进入 Severity 的候选达到 500+ | 1-2 轮 |
| **P0** | 调用 Human Gold 标注 Task3 的 event resolution audit（200+500 样本） | 获取真实 merge_precision / false_merge_rate | 1 周 |
| **P0** | 调用 Human Gold 标注 Task4 的 grounding audit（500 样本） | 验证 grounding 规则的实际准确率 | 1 周 |
| **P1** | 对 732 个新候选做 grounding 质量审计 | 确认真正可用的 Red/Orange 候选数量 | 1 轮 |

### 6.2 中期

| 优先级 | 事项 | 目标 |
|--------|------|------|
| **P1** | 在新候选上重新训练 Impact Scoring 模型 | 解决数据瓶颈，提升 Red Recall |
| **P1** | 合并旧 Severity Gold 1000 与新候选，做 balanced sampling | 解决 Positive 偏斜 |
| **P1** | 对 Task3 不确定队列（115,648 条）做批量人机协同处理 | 提升 canonical event 覆盖率 |
| **P2** | 完善 Event Evolution 的 human gold 验证 | 验证 528 条转移边的质量 |

### 6.3 长期

| 优先级 | 事项 | 目标 |
|--------|------|------|
| **P2** | 将新 targetable 母池纳入正式训练 pipeline | 替换旧 Raw/shadow 补充语料 |
| **P2** | 完成端到端 Severity 预警的正式 human gold 评测 | 获取正式 Red/Orange/Yellow/Blue 的 P/R/F1 |
| **P2** | 固化 10 个新 Event Ontology 到 Task1 抽取层 | 源头减少不相关事件类型 |
| **P3** | 落地 24 条裁决样本的持续更新流程 | 保证去重黑名单的实时性 |

---

## 七、核心数据流一览

### 7.1 当前数据流（已冻结核心链路）

```
新闻报道
  → Task1: PPUIE 事件抽取（结构化事件 + 角色）
  → Task2: DEA 可信度评分 + Safety Gate
  → Task3: 事件归一化 → 事件演化 → 信任传播 → 知识图谱（Neo4j）
  → Task4: 候选生成 → 有效性分类（V4）→ 方向分类（D2-Hierarchical）→ 预警输出
```

### 7.2 Severity 补全数据流（进行中）

```
完整 Raw Corpus（3,124 篇）
  → Targetability 分类（筛选 targetable 1,707 篇）
  → Target-Local Event Re-Extraction（新 10 类 Ontology）
  → Event Fact + Event Type 映射
  → Event Grounding 检查（≥ 65% 才进入 Severity）
  → Financial Consequence 提取
  → Red/Orange 候选生成
  → 去重（与现有所有 gold/silver/audit/裁决）
  → Severity 标注 + 模型训练
  → 预警输出
```

### 7.3 关键阈值

| 检查点 | 要求 | 当前（新母池） |
|--------|------|----------------|
| Target Grounding | ≥ 90% | 100% |
| Evidence Grounding | ≥ 70% | 45.70% |
| Event Grounding | ≥ 65% | 43.72% |
| Article-global Noise | ≤ 15% | 待计算 |
| 新 Red/Orange 候选数 | 300-500 | 732（未质检） |
| Red Recall（Impact Scoring） | ≥ 0.50 | 0.053 |

---

## 八、已废弃/停用方法

| 方法 | 废弃原因 | 替代方案 |
|------|----------|----------|
| **旧 140+ 细粒度 Event Type** | 与 Severity 预测目标不匹配，错配率 87.27% | 10 个高风险事件族 |
| **旧 Article-level Event Label** | 无 target-local 证据，导致 article-global noise 95% | Target-local event fact 重新抽取 |
| **旧 1,110 失败样本修复** | 95% 的文章本身不适合 target-local 抽取 | 换 Targetable 母池 |
| **旧 Raw/shadow 补充语料** | 54.6% 可 targetable，45.4% 为宏观快讯/拼盘/晨报 | 新 Targetable Corpus Selection |
| **旧 Event 标签指导 Severity** | legacy event 与 evidence 不匹配 | 从 event fact → event type → severity |

---

## 九、关键数字快照

| 指标 | 值 |
|------|-----|
| 完整 Raw 文章数 | 3,124 |
| Targetable 文章数 | 1,707（54.6%） |
| Single-target strong | 1,626 |
| Multi-target separable | 8 |
| Industry target valid | 73 |
| 新 Red/Orange 候选（未质检） | 732 |
| 新 Target Grounding | 100% |
| 新 Evidence Grounding | 45.70% |
| 新 Event Grounding | 43.72% |
| Legacy Event 错配率 | 87.27% |
| 旧 失败池 废弃样本数 | 1,110 |
| Severity Gold 1000 | 已生成 |
| Task3 Canonical Event 数 | 1,277 |
| Task3 高置信度转移边 | 528 |
| Task3 不确定队列 | 115,648 |
| Task4 冻结测试候选池 | 10,690 |
| Task4 端到端 Validity F1 | 0.662 |
| Task4 Impact Scoring Macro-F1 | 0.457 |
| Human Audit 待标注 | 1,213 样本（200+500+513） |

---

## 十、结论

1. **Task1-2-3-4 核心链路已冻结交付**，可运行端到端 demo。
2. **Severity 补全的瓶颈在数据而非模型**。旧 Raw/shadow 语料中 45.4% 的文章不适合做 target-local 事件抽取，这是所有 grounding 问题的根源。
3. **新 Targetable 母池策略已验证有效**：换母池后 Event Grounding 从 5% 提升到 43.7%，但离验收线（≥ 65%）还有差距，需继续在母池内优化 event fact 抽取。
4. **Human Gold 标注是当前最大缺失**：Task3 和 Task4 的 audit 包已就绪但未标注，无法获得正式指标。
5. **下一步唯一正确路径**：在 Targetable 母池内提升 Event Grounding → 质检新候选 → 调用 Human Gold 标注 → 验证达到验收线 → 恢复 Severity Mining。