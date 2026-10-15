# Task4 Final Metrics Repaired

## 集合定义
- `G`：冻结 test gold 事件，`|G| = 978`
- `C`：进入候选池的 test 候选，`|C| = 10690`
- `A`：Validity 接受候选，`|A| = 1020`
- `T`：`A` 中正确匹配 gold 的候选，`|T| = 507`
- `D`：`T` 中方向预测正确的候选，`|D| = 458`

## 集合约束
- `D ⊆ T ⊆ A`：通过
- `|D| ≤ |T| ≤ |G|`：通过
- `joint recall ≤ validity recall ≤ candidate recall`：通过
- `achieved_ceiling_ratio ∈ [0,1]`：通过

## Candidate-level
| Metric | Value |
|---|---:|
| Candidate Recall | 1.000000 |
| Review Rate | 0.270720 |
| Candidate Total | 10690 |
| Gold Total | 978 |

## Validity
| Metric | Value |
|---|---:|
| TP | 507 |
| FP | 513 |
| FN | 471 |
| TN | 9199 |
| Precision | 0.497059 |
| Recall | 0.518405 |
| F1 | 0.507508 |
| retained FP | 513 |

## Conditional Direction
| Metric | Value |
|---|---:|
| Macro-F1 | 0.541125 |
| 正面 Recall | 0.936441 |
| 中性 Recall | 0.176471 |
| 负面 Recall | 0.722222 |
| ECE | 0.115830 |
| Sample Count | 507 |

## Joint Event + Direction
| Metric | Value |
|---|---:|
| TP | 458 |
| FP | 562 |
| Precision | 0.449020 |
| Recall | 0.468303 |
| F1 | 0.458458 |
| achieved_ceiling_ratio | 0.468303 |

## 旧错误解释
- `884`：来自全量 978 条 gold-aligned direction 评测明细中的方向正确数，不是 `T` 集合。
- `784`：来自旧版 closure 的 retained true positive，属于另一套 validity 接受集合。
- `1.127551`：`884 / 784` 跨集合计算，导致 `|D| > |T|` 的不合法结论。

## 冻结边界
- 未训练模型
- 未改阈值
- 未修改 Stage3.2
- 未修改 Stage4.1
- 未修改 Final Release
