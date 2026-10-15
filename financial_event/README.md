# 金融事件 Schema 项目

将华泰证券赛事提供的公司事件体系和行业事件体系转换为统一的事件 Schema 格式。

## 目录结构

```text
financial_event_project/
├── data/
│   ├── raw/                          # 原始输入文件
│   │   ├── 事件类型_标的.json          # 公司事件体系
│   │   └── 事件类型_行业.json          # 行业事件体系
│   └── processed/                    # 处理后输出文件
│       ├── event_schema.jsonl        # 统一事件 Schema（JSONL）
│       └── event_schema.xlsx         # 统一事件 Schema（Excel）
├── scripts/
│   ├── build_event_schema.py         # 构建脚本
│   └── test_event_schema.py          # 测试脚本
├── reports/
│   ├── event_schema_summary.json     # 统计汇总报告
│   └── event_name_frequency.csv      # 事件名称频次表
└── README.md
```

## 输入文件

| 文件 | 说明 |
|------|------|
| `data/raw/事件类型_标的.json` | 公司事件体系，包含多家上市公司的事件定义 |
| `data/raw/事件类型_行业.json` | 行业事件体系，包含多个行业的事件定义 |

### 输入结构

**公司事件** (`事件类型_标的.json`):

```json
[
  {
    "company": {
      "trading_code": "002594",
      "secu_abbr": "比亚迪"
    },
    "event_schema": [
      {
        "event_name": "新车出海",
        "description": "事件定义与事件示例"
      }
    ]
  }
]
```

**行业事件** (`事件类型_行业.json`):

```json
[
  {
    "industry": {
      "inducode": "630700",
      "induname": "电池"
    },
    "event_schema": [
      {
        "event_name": "行业调控政策发布",
        "description": "事件定义与事件示例"
      }
    ]
  }
]
```

## 输出文件

### 统一 Schema 格式

每条记录包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_id` | string | 唯一标识，格式：`{target_type}_{code}_{event_name}` |
| `target_type` | string | 目标类型：`company` 或 `industry` |
| `target_code` | string | 代码：公司用 `trading_code`，行业用 `inducode` |
| `target_name` | string | 名称：公司用 `secu_abbr`，行业用 `induname` |
| `event_name` | string | 事件名称 |
| `raw_description` | string | 完整的原始事件定义（已清理异常空白但未删减） |

### 示例

```json
{
  "schema_id": "company_002594_新车出海",
  "target_type": "company",
  "target_code": "002594",
  "target_name": "比亚迪",
  "event_name": "新车出海",
  "raw_description": "事件定义：比亚迪在海外市场推出新车型的行为..."
}
```

## 运行方法

### 1. 安装依赖

```bash
pip install pandas openpyxl
```

### 2. 构建 Schema

```bash
cd financial_event_project
python scripts/build_event_schema.py
```

### 3. 运行测试

```bash
python scripts/test_event_schema.py
```

## 数据校验规则

1. **必需字段检查**：`schema_id`, `target_type`, `target_code`, `target_name`, `event_name`, `raw_description` 必须存在
2. **非空检查**：以上字段均不能为空字符串或 null
3. **唯一性检查**：`schema_id` 必须唯一，出现重复时程序终止
4. **代码类型检查**：`target_code` 必须为字符串，不能是 float/int
5. **值域检查**：`target_type` 只能是 `company` 或 `industry`
6. **描述完整性**：事件描述不得删减或缩写，仅清理异常空格、制表符和连续空行

## 清理规则

- 制表符替换为空格
- `\r\n` 统一为 `\n`
- 每行首尾多余空格去除
- 连续 3 个及以上空行合并为 1 个空行
- 保留段落结构和完整语义

---

## 第二步：带标签数据标准化

### 输入 Excel

| 文件 | 说明 |
|------|------|
| `data/raw/news_with_tags_train.xlsx` | 训练集带标签新闻 |
| `data/raw/news_with_tags_test.xlsx` | 测试集带标签新闻 |

### 工作表结构

每个 Excel 包含 4 个工作表：

| 工作表 | 说明 | 主要字段 |
|--------|------|----------|
| `个股新闻` | 公司级新闻标签 | article_file_id, article_title, article_publish_time, article_source, content, trading_code, secu_abbr, industry_code, industry_name, event_name, event_emotion, event_impact_analysis, duplication_id |
| `行业新闻` | 行业级新闻标签 | article_file_id, article_title, article_publish_time, article_source, content, industry_code, industry_name, event_name, event_emotion, event_impact_analysis, duplication_id |
| `个股重复新闻` | 公司新闻重复关系 | article_file_id, article_title, article_source, content, article_publish_time, duplication_id |
| `行业重复新闻` | 行业新闻重复关系 | article_file_id, article_title, article_source, content, article_publish_time, duplication_id |

### 字段映射

| 标准字段 | Excel 原始字段 |
|----------|---------------|
| `article_id` | `article_file_id` |
| `title` | `article_title` |
| `publish_time` | `article_publish_time` |
| `source` | `article_source` |
| `content` | `content` |
| `target_code` | `trading_code`（个股）/ `industry_code`（行业） |
| `target_name` | `secu_abbr`（个股）/ `industry_name`（行业） |
| `event_name` | `event_name` |
| `emotion` | `event_emotion` |
| `impact_analysis` | `event_impact_analysis` |
| `duplication_id` | `duplication_id` |

### Schema 映射规则

映射顺序（精确匹配）：

1. `(target_type, target_code, event_name)` 精确匹配
2. `(target_type, target_name, event_name)` 精确匹配
3. 规范化文本后再次精确匹配
4. 仍无法匹配则记录为 unresolved

公司代码处理：int → 字符串 → 补齐 6 位（保留前导零）。
行业代码处理：int → 字符串 → 去除 `.0`。

### 新闻级数据 vs 事件对数据

**新闻级数据** (`news_*.jsonl`)：一篇新闻一条记录，所有事件标签放入 `labels` 数组。

```json
{
  "article_id": "NEWS_001",
  "title": "...",
  "publish_time": "2026-01-01 10:00:00",
  "source": "...",
  "content": "...",
  "duplication_id": null,
  "labels": [
    {
      "schema_id": "company_002594_海外产能进展",
      "target_type": "company",
      "target_code": "002594",
      "target_name": "比亚迪",
      "event_name": "海外产能进展",
      "emotion": "正面",
      "impact_analysis": "..."
    }
  ]
}
```

**事件对数据** (`event_pairs_*.jsonl`)：每个"新闻—事件标签"一条记录，用于事件匹配模型训练。

```json
{
  "sample_id": "train_NEWS_001_company_002594_海外产能进展",
  "split": "train",
  "article_id": "NEWS_001",
  "title": "...",
  "schema_id": "company_002594_海外产能进展",
  "target_type": "company",
  "target_code": "002594",
  "event_name": "海外产能进展",
  "event_description": "来自 event_schema.jsonl 的完整定义",
  "emotion": "正面",
  "impact_analysis": "...",
  "label": 1
}
```

### 重复新闻表处理方式

两个重复新闻表（`个股重复新闻`、`行业重复新闻`）的结构为：

- 字段：`article_file_id`, `article_title`, `article_source`, `content`, `article_publish_time`, `duplication_id`
- **不含事件标签**（无 event_name、emotion 等字段）
- 提供 `article_file_id → duplication_id` 的映射关系
- 重复文章编号为 `NEWS_HEARSAY-*`，指向 `NEWS_MULTISOURCE_D-*` 格式的原始文章

处理方式：

1. 从重复新闻表提取 `article_file_id → duplication_id` 映射
2. 用于回填主新闻表中缺失的 `duplication_id`
3. 不将重复新闻表作为独立标签数据追加
4. 后续数据集划分使用 `duplication_id` 防止数据泄漏

### 输出文件

```text
data/processed/labeled/
├── news_train.jsonl              # 训练集新闻级数据
├── news_test.jsonl               # 测试集新闻级数据
├── event_pairs_train.jsonl       # 训练集事件对数据
├── event_pairs_test.jsonl        # 测试集事件对数据
├── event_pairs_train.xlsx        # 训练集事件对 Excel
└── event_pairs_test.xlsx         # 测试集事件对 Excel

reports/
├── labeled_dataset_summary.json   # 统计汇总
├── labeled_event_frequency.csv   # 事件频率
├── excel_sheet_profile.json      # Excel 工作表概况
├── unresolved_schema_mappings.csv # 未匹配记录
├── invalid_publish_times.csv     # 时间解析异常
└── unresolved_duplication_relations.csv # 重复新闻关联异常
```

### 运行命令

```bash
cd financial_event_project
python scripts/build_labeled_dataset.py
python scripts/test_labeled_dataset.py
```

### 测试覆盖

1. 输出文件存在性
2. JSONL 逐行可解析
3. `sample_id` 唯一性
4. `schema_id` 有效性（存在于 event_schema.jsonl）
5. `event_name`/`target_code` 与 Schema 一致性
6. `target_type` 值域（company/industry）
7. 公司代码 6 位格式
8. `label` 全部为 1
9. 新闻级数据 labels 非空
10. 新闻标签总数 = 事件对记录数
11. JSONL 与 Excel 数量一致
12. train/test 完全隔离
13. 事件标签去重
14. 正文完整性
15. 必需字段非空
16. 统计报告一致性
17. Excel 列顺序

---

## 第三步：重复新闻标签传播与正样本增强

### 原理

`duplication_id` 是重复新闻表的核心关联字段：转载新闻的 `duplication_id` 指向对应主新闻的 `article_id`。由于转载新闻与主新闻内容高度重复（标题 Jaccard 中位数 1.0，正文 3-gram 中位数 ~0.93），转载新闻可以安全继承主新闻的事件标签，从而扩大正样本规模。

### 标签传播逻辑

```
转载新闻.duplication_id → 主新闻.article_id → 继承全部 labels
```

限制条件：
1. train/test 完全隔离，禁止跨 split 查找
2. 只有 `duplication_id` 精确匹配主新闻 `article_id` 时才传播
3. 主新闻为多标签时，转载新闻继承全部标签
4. 转载新闻的 `article_id` 不得与主新闻重复

### 原始数据 vs 增强数据

| 字段 | 说明 |
|------|------|
| `canonical_article_id` | 指向主新闻 ID（= duplication_id） |
| `is_duplicate_news` | `true` |
| `label_source` | `inherited_from_canonical_article` |

增强数据 = 原始数据 + 成功传播的重复新闻数据。

### 输出文件

```text
data/processed/augmented/
├── duplicate_news_train.jsonl              # 仅重复新闻（train）
├── duplicate_news_test.jsonl               # 仅重复新闻（test）
├── duplicate_event_pairs_train.jsonl       # 重复事件对（train）
├── duplicate_event_pairs_test.jsonl        # 重复事件对（test）
├── duplicate_event_pairs_train.xlsx
├── duplicate_event_pairs_test.xlsx
├── news_train_augmented.jsonl              # 原始+增强合并（train）
├── news_test_augmented.jsonl               # 原始+增强合并（test）
├── event_pairs_train_augmented.jsonl       # 原始+增强合并（train）
└── event_pairs_test_augmented.jsonl        # 原始+增强合并（test）

reports/
├── duplicate_augmentation_summary.json     # 增强统计
├── duplicate_augmented_event_frequency.csv # 增强事件频率
├── duplicate_text_similarity.csv           # 转载-主新闻文本相似度
├── unresolved_duplicate_label_propagation.csv # 传播失败记录
└── duplicate_article_conflicts.csv         # 冲突记录
```

### 运行命令

```bash
cd financial_event_project
python scripts/augment_duplicate_news.py
python scripts/test_duplicate_augmentation.py
```

### 测试覆盖

1. 所有输出文件存在
2. JSONL 可解析
3. article_id 唯一
4. canonical_article_id 存在且有效
5. labels 非空、标签继承完全一致
6. schema_id 有效、event_description 与 Schema 一致
7. is_duplicate_news=true, label_source=inherited_from_canonical_article
8. label=1
9. 数量一致性：epairs = labels 总数, augmented = orig + dup
10. train/test 完全隔离，无跨 split 传播
11. 原始数据未被修改
12. 事件标签去重
13. 未匹配记录全部进入报告

---

## 第四步：事件识别负样本构造

### 三种负样本类型

| 类型 | 名称 | 含义 | 置信度 |
|------|------|------|--------|
| A | `same_target_other_event` | 同公司/行业，不同事件Schema | medium |
| B | `same_event_wrong_target` | 同事件名，不同公司/行业 | high |
| C | `similar_schema_wrong_target` | 相似Schema，不同目标实体 | high |

### 为什么不能把所有其他Schema当负样本

1. 一篇新闻可能是多标签新闻
2. 一篇新闻可能同时影响多个公司或行业
3. 公司新闻可能触发行业事件
4. 转载新闻不应过度放大

### `canonical_group_id` 的作用

- 主新闻：`canonical_group_id = article_id`
- 转载新闻：`canonical_group_id = canonical_article_id`
- 平衡集中每个 canonical group 最多保留 1 主新闻 + 2 转载新闻
- 防止转载新闻过度放大某些事件类别

### 负样本字段

| 字段 | 说明 |
|------|------|
| `negative_type` | A/B/C 类型标识 |
| `negative_confidence` | medium / high |
| `schema_similarity` | Type C 的 Schema 相似度分数 |
| `source_positive_sample_id` | 来源正样本 ID |

### 假负样本风险控制

自动删除以下可疑负样本：
- 候选 target_name 出现在标题或正文
- 候选 event_name 完整出现在标题
- 与正样本存在 (article_id, schema_id) 冲突

最终正负标签冲突 = 0。

### 输出文件

```text
data/processed/training_pairs/
├── negative_pairs_train.jsonl       # 全量负样本（train）
├── negative_pairs_test.jsonl        # 全量负样本（test）
├── all_pairs_train.jsonl            # 全量正负合并（train）
├── all_pairs_test.jsonl             # 全量正负合并（test）
├── train_pairs_balanced.jsonl       # 平衡训练集（train）
├── test_pairs_balanced.jsonl        # 平衡评估集（test）
├── train_pairs_balanced.xlsx
└── test_pairs_balanced.xlsx

reports/
├── negative_sampling_summary.json
├── negative_type_frequency.csv
├── schema_negative_frequency.csv
├── articles_without_negative_candidates.csv
├── canonical_group_sampling.csv
└── suspicious_negative_pairs.csv
```

### 运行命令

```bash
cd financial_event_project
python scripts/build_negative_pairs.py
python scripts/test_negative_pairs.py
```

---

## 第五步：按事件组划分内部训练集与验证集

### 为什么不能随机按事件对切分

同一篇主新闻的转载新闻、正负事件对共享相同的 `canonical_group_id`。随机切分事件对会导致同一组的数据分散到训练集和验证集，造成严重的近重复数据泄漏。

### 划分原则

- 以 `canonical_group_id` 为最小划分单位
- 目标比例 ~80% 训练 / ~20% 验证（按组数）
- 优先将稀有 Schema 的组分配到验证集以确保覆盖
- 官方 test 完全封存，不参与划分
- 固定随机种子 2026

### 新增字段

| 字段 | 值 |
|------|-----|
| `internal_split` | `train` / `validation` / `official_test` |
| `split_seed` | 2026 |
| `model_input_text` | 结构化输入文本（7行固定格式） |

### `model_input_text` 格式

```
目标类型：{target_type}
目标名称：{target_name}
目标代码：{target_code}
候选事件：{event_name}
事件定义：{event_description}
新闻标题：{title}
新闻正文：{content}
```

### 三个数据集用途

| 数据集 | 用途 | 禁止 |
|--------|------|------|
| `event_match_train` | 模型训练 | — |
| `event_match_validation` | 超参调优、早停 | 参与训练 |
| `event_match_official_test` | 最终评估 | 调参、选模型 |

### 输出文件

```text
data/processed/model_data/
├── event_match_train.jsonl              (4,986 条)
├── event_match_validation.jsonl         (1,271 条)
├── event_match_official_test.jsonl      (3,041 条，原样复制)
├── event_match_train.xlsx
├── event_match_validation.xlsx
├── event_match_train_minimal.jsonl      (仅 sample_id, text, label)
└── event_match_validation_minimal.jsonl

reports/
├── train_validation_split_summary.json
├── train_validation_schema_distribution.csv
├── train_validation_negative_type_distribution.csv
├── schemas_missing_in_validation.csv
└── train_validation_leakage.csv
```

### 运行命令

```bash
cd financial_event_project
python scripts/split_train_validation.py
python scripts/test_train_validation_split.py
```

---

## 第六步：MacBERT 事件匹配基线

### 任务目标

二分类 Cross-Encoder：给定（目标实体 + 候选事件 + 事件定义 + 新闻标题 + 新闻正文），判断新闻是否触发该事件。

### 模型

- `hfl/chinese-macbert-base`（~102M 参数）
- `AutoModelForSequenceClassification(num_labels=2)`
- 4× RTX 3090 DDP 训练

### 文本输入格式

```
[目标]
类型：company
名称：比亚迪
代码：002594

[候选事件]
事件名称：海外产能进展
事件定义：……

[新闻]
标题：……
正文：……
```

不含 label、negative_type 等泄漏字段。

### 长文本截断

- max_length=512
- 优先保留目标信息 + 事件名称 + 事件定义 + 标题
- 正文采用 头70% + [正文中间省略] + 尾30%
- 截断比例：train 92.2%，val 95.3%

### 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 6（早停 patience=2） |
| 学习率 | 2e-5 |
| 有效批量 | 16 (8×2) |
| FP16 | ✅ |
| 最佳模型选择 | macro_f1 |

### 阈值搜索

- 搜索范围：0.05–0.95（步长 0.01）
- 选择指标：macro_f1

### 结果

| 指标 | Threshold 0.5 | Best Threshold 0.38 |
|------|---------------|---------------------|
| Accuracy | 91.66% | 92.05% |
| Precision | 95.99% | 94.79% |
| Recall | 84.78% | 86.90% |
| Positive F1 | 90.04% | 90.67% |
| Macro F1 | 91.43% | 91.88% |
| ROC-AUC | 96.54% | 96.54% |

混淆矩阵（best threshold）：
- TN: 679, FP: 27
- FN: 74, TP: 491

### 运行命令

```bash
# 下载模型
HF_ENDPOINT=https://hf-mirror.com python -c "from transformers import AutoModel; AutoModel.from_pretrained('hfl/chinese-macbert-base')"

# 训练
torchrun --nproc_per_node=4 scripts/train_event_match_baseline.py --config configs/event_match_macbert_base.json

# 评估
python scripts/evaluate_event_match_baseline.py --model_dir outputs/event_match_baseline/best_model --validation_file data/processed/model_data/event_match_validation.jsonl --output_dir outputs/event_match_baseline

# 测试
python scripts/test_event_match_training.py
```

---

## 第七步：同实体事件困难负样本挖掘

### 动机

基线模型最难区分的负样本类型是 `same_target_other_event`（误报率 15.22%），即同一家公司/行业的不同事件之间的混淆。本步骤通过基线模型打分挖掘困难负样本，构建增强训练集。

### 方法

1. 从训练集生成全部（article, 同目标其他Schema）候选
2. 基线模型推理打分
3. 按正类概率分级：very_hard(≥0.70)、hard(0.38-0.70)、medium(0.20-0.38)、easy(<0.20)
4. 采样约束：每文章≤2、每canonical group≤4、每Schema≤80、总计≤1000
5. 添加到训练集重新训练

### 结果（未达验收标准）

| 指标 | 基线 | V2 | 变化 |
|------|------|-----|------|
| Macro-F1 | 91.88% | 89.77% | **-2.10%** |
| Positive F1 | 90.67% | 88.23% | -2.44% |
| Recall | 86.90% | 84.25% | -2.65% |
| same_target FPR | 15.22% | 21.74% | **+6.52%** |

### 失败原因分析

1. 653条新增负样本中 90.8% 为 very_hard（基线高分误报）
2. 仅覆盖 16 个目标和 33 个 Schema，过于集中
3. 过度密集的负信号使模型对特定 Schema 过度保守
4. 行业 Schema 误报率全面上升

### 教训

- 困难负样本挖掘需要更严格的数量控制
- 应优先覆盖更多样化的 Schema 而非堆砌高置信误报
- 结论：基线模型作为当前最佳模型，继续用于后续步骤

---

## 第八步：同实体候选事件联合竞争决策

### 动机

基线独立二分类容易同时激活同一目标下的多个相似事件（如同为比亚迪的"新车出海""海外产能进展"）。联合竞争在同目标候选事件间进行相对排名，在不重新训练模型的情况下减少误报。

### 三种决策方法

| 方法 | 规则 | 说明 |
|------|------|------|
| A. Absolute Threshold | `prob >= th` | 基线独立决策 |
| B. Top-Margin | `prob >= th AND score >= max_score - margin` | 与组内最高分差距约束 |
| C. Relative Softmax | `prob >= th AND softmax(score/temp) >= rel_th` | 温度Softmax相对概率约束 |

### 参数搜索（仅验证集）

- Method B: abs_th 0.25-0.55 × margin 0.0-1.5 → 56 configs
- Method C: abs_th 0.25-0.55 × temp 0.5-2.0 × rel_th 0.2-0.8 → 245 configs
- 选择约束: macro_f1≥91.58%, recall≥85%, 然后 min same_target_fpr

### 结果

| 指标 | 基线 | Targetwise | Δ |
|------|------|-----------|-----|
| Macro-F1 | 91.88% | 91.62% | -0.26% |
| Recall | 86.90% | 85.84% | -1.06% |
| same_target FPR | 17.39% | 15.22% | **-2.17%** |
| FP Count | 27 | 24 | -3 |
| 修复基线错误 | — | 3 | — |
| 新增错误 | — | 6 | — |
| 净修复 | — | **-3** | — |

### 诊断

- 62.5% (5/8) 的基线 same_target 误报是 competition_fixable
- 37.5% (3/8) 是 representation_error（模型本身偏向错误Schema）
- 最佳配置: Top-Margin, abs_th=0.40, margin=1.5
- 大margin(1.5)说明需要非常宽松的竞争约束才能维持召回率

### 结论

联合竞争决策边际改善了same_target FPR（-2.17%），但代价是Recall下降（-1.06%）和净修复为负（-3）。同实体事件混淆的核心问题属于表示错误，需要模型层面的改进（如对比学习或Pairwise Ranking训练），而非纯后处理可解决。基线模型仍是当前最佳选择。

---

## 第九步：评估口径审计与统一

### 发现的不一致

第六步报告的负样本分类指标与主指标使用了不同阈值：
- 主指标（Macro-F1、混淆矩阵）：阈值 **0.38**（最佳）
- 负样本分类报告：阈值 **0.5**（`prediction_threshold_0_5` 字段）

### 根本原因

`evaluate_event_match_baseline.py` 在生成负样本分类报告时使用了 `prediction_threshold_0_5` 字段而非基于 `best_threshold` 重新计算。

### 权威数字（阈值 0.38）

| 负样本类型 | 样本数 | FP | FPR |
|-----------|--------|-----|------|
| same_target_other_event | 46 | **8** | **17.39%** |
| same_event_wrong_target | 275 | **5** | **1.82%** |
| similar_schema_wrong_target | 385 | **14** | **3.64%** |

### 第八步基线列正确

第八步报告的基线 FPR（17.39%, 1.82%, 3.64%）与权威数字一致，结论不变。

### 统一评估模块

创建 `src/evaluation/event_match_evaluator.py`，所有后续实验必须使用此模块。
配置：`configs/canonical_evaluation_protocol.json`

---

## 第十步：CE + Pairwise Ranking联合训练

### 动机

第七步证明直接加入困难负样本会损害性能。本步骤用RankNet Pairwise Ranking损失替代直接交叉熵，引导模型学习同实体事件间的相对排序。

### 实验设计

两个严格可比的实验（相同初始化、训练集、epoch、LR）：

| 实验 | 损失 | lambda_rank |
|------|------|-------------|
| Control (CE only) | L_ce | 0.0 |
| Pairwise (CE + RankNet) | L_ce + 0.2 × L_rank | 0.2 |

### Pairwise数据

- 只使用 `same_target_other_event` 作为排序负例
- Train: 186 pairs, Val: 46 pairs
- 覆盖 7 个训练目标实体，11 个 Schema 对

### 结果

| 指标 | 基线 | Control | Pairwise |
|------|------|---------|----------|
| Macro-F1 | 91.88% | **92.44%** | 91.61% |
| Recall | 86.90% | 87.43% | 85.66% |
| same_target FPR | 17.39% | 19.57% | **17.39%** |
| Pairwise Acc | 86.96% | 86.96% | **89.13%** |

### 结论

- **Control模型最优**：仅额外3 epoch CE训练即可提升macro_f1至92.44%
- Pairwise排序准确率提升（+2.17%），但未转化为same_target FPR改善
- same_target FPR 顽固在17.39%，需更根本的表示层面改进
- 修复16条基线错误，新增19条，净-3

### 输出文件

```text
data/processed/hard_negative/
├── all_same_target_candidates_train.jsonl
├── mined_same_target_negatives_train.jsonl
├── mined_same_target_negatives_train.xlsx
└── event_match_train_hard_negative.jsonl

outputs/event_match_hard_negative/
├── best_model/  (V2模型，指标低于基线)
├── validation_predictions.jsonl
├── threshold_search.csv
└── *.json
```
