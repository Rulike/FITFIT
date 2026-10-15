(function () {
  'use strict';

  const events = [
    {
      id: 'EVT-240816-001', company: '宁德时代', code: '300750', logo: '宁', type: '产能与供应链',
      title: '宁德时代发布新一代磷酸铁锂电池量产进展，多家车企确认合作',
      summary: '公司在技术发布会上披露新一代磷酸铁锂电池已进入量产阶段，能量密度和快充性能均有提升；三家主流车企随后确认将在新车型中搭载。',
      time: '08-16 10:42', fullTime: '2026-08-16 10:42:18', source: '公司公告', sourceCount: 8,
      credibility: 96, impact: 87, heat: 92, level: 'high', sentiment: 'positive', direction: '正面',
      status: '持续跟踪', entity: '宁德时代股份有限公司', object: '动力电池产业链',
      keywords: ['磷酸铁锂', '量产', '快充', '车企合作'], evidenceCount: 12, relatedCount: 7,
      sourceText: '公司官方发布、交易所公告及多家产业链媒体报道形成交叉验证，关键指标表述一致。'
    },
    {
      id: 'EVT-240816-002', company: '中芯国际', code: '688981', logo: '芯', type: '业绩预告',
      title: '中芯国际二季度产能利用率环比提升，先进制程收入占比增长',
      summary: '最新经营数据表明晶圆出货量与产能利用率环比改善，先进制程产品组合持续优化，市场关注后续资本开支节奏。',
      time: '08-16 09:18', fullTime: '2026-08-16 09:18:06', source: '定期报告', sourceCount: 6,
      credibility: 98, impact: 81, heat: 79, level: 'high', sentiment: 'positive', direction: '正面',
      status: '已确认', entity: '中芯国际集成电路制造有限公司', object: '半导体制造',
      keywords: ['产能利用率', '先进制程', '资本开支'], evidenceCount: 9, relatedCount: 4,
      sourceText: '定期报告为核心证据，券商研报与行业数据对趋势提供补充佐证。'
    },
    {
      id: 'EVT-240816-003', company: '药明康德', code: '603259', logo: '药', type: '政策与合规',
      title: '海外监管提案进入新审议阶段，公司回应业务运营暂未受到实质影响',
      summary: '相关提案进入委员会审议阶段，尚未形成生效条款。公司在投资者互动平台回应称当前订单履约与业务运营保持正常。',
      time: '08-16 08:55', fullTime: '2026-08-16 08:55:47', source: '多源资讯', sourceCount: 14,
      credibility: 89, impact: 94, heat: 96, level: 'critical', sentiment: 'negative', direction: '负面',
      status: '重点预警', entity: '无锡药明康德新药开发股份有限公司', object: 'CXO产业链',
      keywords: ['海外监管', '法案审议', '业务影响'], evidenceCount: 18, relatedCount: 11,
      sourceText: '海外立法网站与公司官方回应对事实部分形成验证，但最终条款和执行范围仍存在不确定性。'
    },
    {
      id: 'EVT-240816-004', company: '比亚迪', code: '002594', logo: '比', type: '新车出海',
      title: '比亚迪在东南亚市场发布两款新能源车型并扩建本地销售网络',
      summary: '两款新能源车型在区域市场同步发布，公司计划新增多家直营网点，进一步推进海外市场本地化经营。',
      time: '08-15 22:31', fullTime: '2026-08-15 22:31:52', source: '行业媒体', sourceCount: 11,
      credibility: 93, impact: 76, heat: 84, level: 'medium', sentiment: 'positive', direction: '正面',
      status: '持续跟踪', entity: '比亚迪股份有限公司', object: '新能源汽车产业链',
      keywords: ['新车出海', '东南亚', '渠道扩张'], evidenceCount: 15, relatedCount: 9,
      sourceText: '发布会资料、区域官网与当地经销商信息相互印证。'
    },
    {
      id: 'EVT-240816-005', company: '隆基绿能', code: '601012', logo: '隆', type: '价格与经营',
      title: '光伏主产业链价格继续承压，隆基绿能调整部分产品排产计划',
      summary: '行业报价显示硅片与组件价格仍处低位，公司结合库存与订单情况动态调整部分基地排产，短期盈利修复承压。',
      time: '08-15 20:06', fullTime: '2026-08-15 20:06:25', source: '产业数据', sourceCount: 9,
      credibility: 86, impact: 88, heat: 73, level: 'high', sentiment: 'negative', direction: '负面',
      status: '持续跟踪', entity: '隆基绿能科技股份有限公司', object: '光伏产业链',
      keywords: ['组件价格', '排产', '库存'], evidenceCount: 10, relatedCount: 8,
      sourceText: '产业报价具备多源一致性，排产信息仍需等待公司进一步确认。'
    },
    {
      id: 'EVT-240816-006', company: '海康威视', code: '002415', logo: '海', type: '技术突破',
      title: '海康威视发布端侧多模态模型，新品面向工业视觉场景',
      summary: '公司披露端侧多模态模型及配套硬件方案，强调低功耗部署和复杂环境识别能力，首批落地聚焦工业质检。',
      time: '08-15 18:40', fullTime: '2026-08-15 18:40:11', source: '公司官网', sourceCount: 5,
      credibility: 97, impact: 69, heat: 77, level: 'medium', sentiment: 'positive', direction: '正面',
      status: '已确认', entity: '杭州海康威视数字技术股份有限公司', object: '计算机视觉产业链',
      keywords: ['端侧模型', '多模态', '工业视觉'], evidenceCount: 7, relatedCount: 3,
      sourceText: '产品发布信息来自公司官网，商业订单与收入贡献尚待后续观察。'
    },
    {
      id: 'EVT-240816-007', company: '万科A', code: '000002', logo: '万', type: '债务与流动性',
      title: '万科完成一笔境内债券兑付，后续公开债到期节奏仍受关注',
      summary: '公司公告完成本期债券本息兑付，短期流动性压力有所缓释；未来两个季度仍有公开债务进入到期窗口。',
      time: '08-15 16:22', fullTime: '2026-08-15 16:22:09', source: '交易所公告', sourceCount: 12,
      credibility: 99, impact: 91, heat: 89, level: 'critical', sentiment: 'neutral', direction: '中性',
      status: '重点预警', entity: '万科企业股份有限公司', object: '房地产产业链',
      keywords: ['债券兑付', '流动性', '到期窗口'], evidenceCount: 16, relatedCount: 10,
      sourceText: '本次兑付由交易所公告确认；后续偿债压力判断来自公开债券日历和财务数据。'
    },
    {
      id: 'EVT-240816-008', company: '中国神华', code: '601088', logo: '神', type: '资产整合',
      title: '中国神华披露控股股东资产整合进展，相关审计评估工作推进',
      summary: '公司更新控股股东资产整合事项进度，目前处于审计评估与方案论证阶段，交易边界尚未最终确定。',
      time: '08-15 14:08', fullTime: '2026-08-15 14:08:34', source: '公司公告', sourceCount: 7,
      credibility: 95, impact: 83, heat: 71, level: 'high', sentiment: 'positive', direction: '正面',
      status: '持续跟踪', entity: '中国神华能源股份有限公司', object: '煤炭能源产业链',
      keywords: ['资产整合', '审计评估', '控股股东'], evidenceCount: 11, relatedCount: 6,
      sourceText: '事项进展来自公司公告，交易范围与估值需以后续正式方案为准。'
    }
  ];

  window.MockData = {
    dashboard: {
      updatedAt: '2026-08-16 11:06:32',
      metrics: [
        { label: '最新态势', value: '活跃', delta: '已同步', note: '公开样本持续更新', icon: '势', color: '#2ee6a6', spark: [18,23,20,31,28,39,42,48] },
        { label: '重点预警', value: '12', delta: '优先关注', note: '重大信号已单独标注', icon: '警', color: '#ffb648', spark: [22,18,24,21,31,29,36,39] },
        { label: '覆盖主体', value: '18', delta: '26 条样本', note: '不同公司主体已聚合', icon: '主', color: '#4aa8ff', spark: [8,12,9,17,13,19,22,25] },
        { label: '综合判断', value: '稳定', delta: '重点信号已更新', note: '当前结论稳定', icon: '验', color: '#4aa8ff', spark: [73,76,78,77,82,84,87,91] }
      ],
      updates: [
        { time: '刚刚', title: '公开结果已同步', note: '最新样本和提醒卡片已更新' },
        { time: '最近', title: '重点提醒已就绪', note: '高影响事件保持关注' },
        { time: '今天', title: '主体覆盖已刷新', note: '事件覆盖持续更新' },
        { time: '现在', title: '趋势看板可用', note: '可直接查看最新走势' }
      ],
      trend: {
        labels: ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00'],
        total: [42,30,35,114,178,226,193,158],
        high: [5,3,4,18,27,43,38,31]
      },
      composition: [
        { name: '公司经营', value: 36, count: 462, color: '#2ee6a6' },
        { name: '政策监管', value: 24, count: 308, color: '#4aa8ff' },
        { name: '产业链变化', value: 18, count: 231, color: '#9677ff' },
        { name: '风险舆情', value: 13, count: 167, color: '#ff5d73' },
        { name: '技术突破', value: 9, count: 116, color: '#ffb648' }
      ],
      pipeline: [
        { icon: '采', name: '公开样本接入', note: '资讯 / 公告 / 研报 / 政策', value: '38.6万', unit: '条/日' },
        { icon: '抽', name: '事件要素整理', note: '主体、时间、方向、对象', value: '96.4%', unit: '完整率' },
        { icon: '验', name: '公开判断', note: '来源、证据与公开状态', value: '已同步', unit: '结果' },
        { icon: '聚', name: '脉络更新', note: '相似内容与关联关系', value: '持续', unit: '更新' },
        { icon: '警', name: '重点提醒', note: '高影响样本已单独展示', value: '可用', unit: '状态' }
      ]
    },
    events,
      credibility: {
      verdict: '公开结论 · 已确认',
      summary: '核心事实由公司公告和权威媒体共同确认，关键信号一致，暂无明显冲突。',
      evidence: [
        { source: '公司公告', type: '一级信源', logo: '公', color: '#2ee6a6', score: 99, time: '10:18', text: '披露新一代磷酸铁锂电池量产节点、主要性能指标及首批客户导入计划。', status: '直接证据' },
        { source: '证券时报', type: '权威媒体', logo: '证', color: '#4aa8ff', score: 94, time: '10:31', text: '现场报道技术发布会，关键参数和量产进度与公司披露保持一致。', status: '独立确认' },
        { source: '产业链调研', type: '行业信源', logo: '产', color: '#9677ff', score: 86, time: '10:38', text: '上游材料供应和下游车型验证节奏与公司公布时间窗口基本匹配。', status: '侧面佐证' },
        { source: '车企官网', type: '合作方信源', logo: '车', color: '#ffb648', score: 96, time: '10:42', text: '三家合作车企在新品页面确认将在后续车型中搭载相关电池产品。', status: '交叉验证' }
      ],
      reasons: [
        { label: '来源数量', value: '4 条', note: '覆盖公告、媒体和合作方' },
        { label: '公开佐证', value: '4 / 4 一致', note: '核心事实保持一致' },
        { label: '异常传播', value: '低风险', note: '无集中异常传播迹象' },
        { label: '结论状态', value: '稳定', note: '公开信号持续一致' }
      ]
    },
    timeline: {
      title: '新一代磷酸铁锂电池量产事件', company: '宁德时代', code: '300750', impact: 87,
      stages: [
        { name: '线索出现', time: '08-12', state: 'done' },
        { name: '媒体报道', time: '08-14', state: 'done' },
        { name: '官方确认', time: '08-16', state: 'done' },
        { name: '持续关注', time: '进行中', state: 'active' },
        { name: '后续观察', time: '待观察', state: '' }
      ],
      items: [
        { date: '08-16', time: '10:42', title: '三家车企确认首批搭载计划', text: '合作方官网陆续更新车型信息，影响范围进一步扩展到整车产业链。', source: '车企官网 · 3条', type: '公开确认', color: '#2ee6a6' },
        { date: '08-16', time: '10:18', title: '公司正式披露量产进展', text: '公司发布技术与量产信息，明确关键性能与首批交付节奏。', source: '公司公告', type: '关键节点', color: '#4aa8ff' },
        { date: '08-14', time: '16:30', title: '产业媒体报道中试线转量产', text: '两家产业媒体同步披露量产线进展，后续等待公司进一步确认。', source: '产业媒体 · 2条', type: '增量更新', color: '#9677ff' },
        { date: '08-12', time: '09:05', title: '供应链出现扩产与备货线索', text: '上游材料企业排产变化与招聘信息形成早期线索，进入持续观察。', source: '产业数据 · 4条', type: '早期线索', color: '#ffb648' }
      ],
      impacts: [
        { name: '公司基本面', value: '显著正面' },
        { name: '动力电池板块', value: '正面' },
        { name: '上游材料', value: '温和正面' },
        { name: '竞争格局', value: '关注加剧' }
      ]
    },
    graph: {
      nodes: [
        { id: 'catl', label: '宁德时代', type: 'company', color: '#2ee6a6', size: 28, x: .48, y: .48 },
        { id: 'event', label: '电池量产', type: 'event', color: '#4aa8ff', size: 23, x: .67, y: .32 },
        { id: 'battery', label: '动力电池', type: 'industry', color: '#9677ff', size: 21, x: .29, y: .30 },
        { id: 'lithium', label: '锂电材料', type: 'industry', color: '#9677ff', size: 18, x: .22, y: .61 },
        { id: 'byd', label: '比亚迪', type: 'company', color: '#2ee6a6', size: 18, x: .74, y: .65 },
        { id: 'ev', label: '新能源汽车', type: 'industry', color: '#9677ff', size: 20, x: .52, y: .77 },
        { id: 'policy', label: '产业政策', type: 'policy', color: '#ffb648', size: 16, x: .47, y: .16 },
        { id: 'supplier', label: '上游供应商', type: 'company', color: '#2ee6a6', size: 16, x: .08, y: .43 },
        { id: 'tech', label: '快充技术', type: 'technology', color: '#4ce3ef', size: 17, x: .87, y: .29 },
        { id: 'risk', label: '原料波动', type: 'risk', color: '#ff5d73', size: 15, x: .32, y: .84 }
      ],
      links: [
        ['catl','event','发布'], ['catl','battery','主营'], ['event','byd','潜在供货'], ['event','tech','技术特征'],
        ['battery','lithium','上游'], ['battery','ev','应用'], ['policy','battery','支持'], ['supplier','lithium','供给'],
        ['byd','ev','布局'], ['catl','supplier','采购'], ['lithium','risk','暴露'], ['event','ev','影响']
      ],
      selected: { name: '宁德时代', code: '300750.SZ', type: '上市公司', events: 26, relations: 18, credibility: '94.2%', risk: '中低' },
      relations: [
        { name: '动力电池', relation: '主营业务' }, { name: '电池量产', relation: '事件主体' },
        { name: '锂电材料', relation: '上游产业' }, { name: '新能源汽车', relation: '下游产业' }
      ]
    },
    alerts: [
      { id: 'ALT-001', level: 'critical', icon: '!', title: '海外监管提案审议进度变化', company: '药明康德 · 603259', reason: '重大政策风险 · 重点关注', time: '11分钟前', score: 94, status: '待处置', color: '#ff5d73' },
      { id: 'ALT-002', level: 'critical', icon: '!', title: '后续公开债进入集中到期窗口', company: '万科A · 000002', reason: '流动性风险 · 持续跟踪', time: '27分钟前', score: 91, status: '跟踪中', color: '#ff5d73' },
      { id: 'ALT-003', level: 'high', icon: '↘', title: '光伏组件价格与排产同步下行', company: '隆基绿能 · 601012', reason: '经营压力 · 重点关注', time: '1小时前', score: 88, status: '待处置', color: '#ffb648' },
      { id: 'ALT-004', level: 'high', icon: '↗', title: '新一代电池确认量产并获客户验证', company: '宁德时代 · 300750', reason: '技术突破 · 正向关注', time: '1小时前', score: 87, status: '已推送', color: '#ffb648' },
      { id: 'ALT-005', level: 'high', icon: '◇', title: '控股股东资产整合进入审计评估', company: '中国神华 · 601088', reason: '资产重组 · 持续跟踪', time: '2小时前', score: 83, status: '跟踪中', color: '#ffb648' },
      { id: 'ALT-006', level: 'medium', icon: 'i', title: '先进制程收入占比环比提升', company: '中芯国际 · 688981', reason: '业绩变化 · 关注', time: '2小时前', score: 81, status: '已读', color: '#4aa8ff' }
    ],
    company: {
      name: '宁德时代', fullName: '宁德时代新能源科技股份有限公司', code: '300750.SZ', industry: '电力设备 · 电池', region: '福建省宁德市',
      tags: ['沪深300', '动力电池龙头', '产业链核心'],
      quotes: [
        { label: '近30日事件', value: '26', delta: '最新同步' },
        { label: '正向信号', value: '16', delta: '持续关注' },
        { label: '综合判断', value: '稳定', delta: '已同步' },
        { label: '高影响事件', value: '7', delta: '+2' },
        { label: '关联产业节点', value: '43', delta: '+5' },
        { label: '风险水位', value: '中低', delta: '稳定' }
      ],
      sentiment: { labels: ['07-18','07-23','07-28','08-02','08-07','08-12','08-16'], positive: [52,58,54,63,60,66,71], negative: [23,20,28,19,25,18,16] },
      topics: [
        { text: '动力电池', size: 15, color: '#2ee6a6' }, { text: '海外产能', size: 12, color: '#4aa8ff' },
        { text: '磷酸铁锂', size: 14, color: '#9677ff' }, { text: '快充技术', size: 13, color: '#4ce3ef' },
        { text: '车企合作', size: 11, color: '#ffb648' }, { text: '锂价波动', size: 10, color: '#ff5d73' },
        { text: '储能业务', size: 12, color: '#2ee6a6' }, { text: '供应链', size: 10, color: '#4aa8ff' }
      ]
    },
    models: {
      cards: [
        { icon: '样', name: '公开样本', version: '最新同步', color: '#2ee6a6', status: '已接入', metrics: [['样本','8'],['高影响','6'],['主体','6'],['状态','公开']] },
        { icon: '警', name: '重点提醒', version: '当前可读', color: '#4aa8ff', status: '已就绪', metrics: [['提醒','6'],['重大','2'],['高影响','4'],['状态','可读']] },
        { icon: '主', name: '公司覆盖', version: '持续更新', color: '#9677ff', status: '稳定', metrics: [['主体','6'],['正面','4'],['负面','2'],['风险','可控']] }
      ],
      health: { labels: ['00','03','06','09','12','15','18','21'], latency: [310,286,294,385,420,398,366,342], volume: [14,9,11,37,62,78,65,51] },
      resources: [
        { name: '公开样本', value: 72, meta: '已同步 · 持续更新' },
        { name: '重点提醒', value: 54, meta: '已就绪 · 可直接阅读' },
        { name: '公司覆盖', value: 63, meta: '主体 12 个 · 稳定' },
        { name: '脉络追踪', value: 31, meta: '关联关系 · 正常' }
      ],
      logs: [
        { time: '现在', level: 'success', text: '公开结果已同步完成', cost: '可读' },
        { time: '最近', level: 'success', text: '重点提醒已更新', cost: '稳定' },
        { time: '稍早', level: 'warn', text: '单条来源响应偏慢，已继续跟踪', cost: '关注' },
        { time: '刚刚', level: 'success', text: '脉络关系视图已刷新', cost: '正常' },
        { time: '今天', level: 'success', text: '最新样本已完成归并', cost: '同步' }
      ]
    }
  };
})();
