# Task4 v4 Impact Score Annotation Guideline

{
  "impact_score_range": [
    0,
    100
  ],
  "impact_direction": [
    "positive",
    "neutral",
    "negative"
  ],
  "impact_confidence_range": [
    0,
    1
  ],
  "bands": {
    "0-20": "无明显影响/弱相关",
    "21-40": "轻微影响",
    "41-60": "中等影响",
    "61-80": "明显影响",
    "81-100": "重大影响"
  },
  "rules": [
    "先判断证据是否明确描述目标受到事件影响，再判断影响强弱。",
    "没有量化结果或可比较经营结果时不得仅凭情绪词打高分。",
    "影响方向与影响强度分开标注。",
    "impact_confidence表示证据对分数的支持程度，不表示预测概率。"
  ],
  "risk_factors": [
    "多数样本有方向和理由，但强弱没有统一客观锚点。",
    "同一证据可能同时描述销量、利润、技术和品牌，综合分容易受标注者权重影响。",
    "新闻篇幅和措辞会造成严重性偏差。",
    "中性样本可能代表无影响，也可能代表影响尚未兑现，二者需区分。"
  ],
  "recommendation": "A"
}