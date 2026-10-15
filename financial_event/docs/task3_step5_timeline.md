# Task3 Step5 时间线、事实冲突、澄清、否认与反转

本步骤基于 title 和结构化字段构建可追溯候选结果。event_time 当前不可用时使用 publish_time 代理，不将关键词命中直接视为真实反转；冲突、澄清、否认和反转均保留人工复核标记。cluster 级可信度保持 trust_score_scope=cluster。
